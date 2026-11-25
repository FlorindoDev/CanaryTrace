"""Render a PDF report from CanaryTrace evaluation results."""

import argparse
import json
from collections.abc import Iterable, Iterator
from math import isfinite
from pathlib import Path
from tempfile import TemporaryDirectory

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

from dataset import COPIES_PER_GROUP


METRICS = {
    "exact_match": "Exact Match",
    "partial_match": "Partial Match",
    "likelihood": "Probabilità",
    "token_rank": "Token Rank",
}
PROMPTS = {
    "exact_prefix": "Prefisso esatto",
    "paraphrased_question": "Domanda parafrasata",
}
CONTEXT_NOTE = (
    "Probabilità e ranghi assumono il contesto corretto, ossia il prompt "
    "seguito dai token attesi precedenti."
)
PROJECTS_PER_DETAIL_PAGE = 8

SeriesKey = tuple[str, str | None, str]
Project = tuple[str, str]
Series = dict[Project, dict]


def _valid_token_scores(row: dict) -> bool:
    likelihood, rank = row["likelihood"], row["token_rank"]
    return (
        type(likelihood) in (int, float)
        and 0 <= likelihood <= 1
        and type(rank) in (int, float)
        and isfinite(rank)
        and rank >= 1
    )


def _index(rows: Iterable[dict]) -> dict[SeriesKey, Series]:
    series: dict[SeriesKey, Series] = {}
    for row in rows:
        condition, checkpoint = row["condition"], row.get("checkpoint")
        prompt, group, project = row["prompt_type"], row["group"], row["project"]
        if condition not in ("baseline", "fine_tuned") or prompt not in PROMPTS or group not in COPIES_PER_GROUP:
            raise ValueError(f"Condizione, prompt o gruppo sconosciuto per {project}.")

        scores = series.setdefault((condition, checkpoint, prompt), {})
        if (group, project) in scores:
            raise ValueError(f"Risultato duplicato per {project}, {condition}, {checkpoint}, {prompt}.")

        exact, partial = row["exact_match"], row["partial_match"]
        if not (
            type(exact) in (int, float) and exact in (0, 1)
            and type(partial) in (int, float) and 0 <= partial <= 1
            and _valid_token_scores(row)
        ):
            raise ValueError(f"Metriche fuori dominio per {project}, {prompt}.")
        scores[group, project] = row
    if not series:
        raise ValueError("Il report richiede almeno un risultato.")
    return series


def _run_label(condition: str, checkpoint: str | None) -> str:
    if condition == "baseline":
        return "Baseline"
    if checkpoint is None:
        return "Fine-tuned"
    return f"Fine-tuned · {'/'.join(Path(checkpoint).parts[-2:])}"


def _table_figure(
    title: str, note: str, headers: list[str], cells: list[list[str]],
    widths: list[float], row_height: float, rows_per_project: int = 1,
) -> Figure:
    table_height = row_height * (len(cells) + 1)
    height = 1.65 + table_height
    figure = Figure(figsize=(8.3, height))
    figure.suptitle(title, y=1 - 0.20 / height, fontsize=15)
    figure.text(0.5, 1 - 0.78 / height, note, ha="center", va="top", fontsize=9)
    ax = figure.add_axes((0.05, 0.35 / height, 0.90, table_height / height))
    ax.set_axis_off()
    table = ax.table(
        cellText=cells, colLabels=headers, colWidths=widths, cellLoc="center", bbox=(0, 0, 1, 1),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        if row == 0:
            cell.set_facecolor("#173c56")
            cell.set_text_props(color="white", weight="bold")
            continue
        project, position = divmod(row - 1, rows_per_project)
        cell.set_facecolor("#e9f1f7" if project % 2 == 0 else "#ffffff")
        if column == 0 and position == 0:
            cell.set_text_props(color="#173c56", weight="bold")
    return figure


def _format_probability(likelihood: float) -> str:
    percentage = likelihood * 100
    if 0 < percentage < 0.01:
        value = f"{percentage:.2e}"
    else:
        value = f"{percentage:.2f}"
    return f"{value.replace('.', ',')}%"


def _summary_figure(title: str, rows: list[dict]) -> Figure:
    cells = [
        [
            row["project"], row["group"], str(int(row["exact_match"])), f'{row["partial_match"]:.0%}',
            _format_probability(row["likelihood"]), f'{row["token_rank"]:.2f}',
        ]
        for row in rows
    ]
    return _table_figure(
        title, CONTEXT_NOTE,
        ["Progetto", "Gruppo", *METRICS.values()],
        cells, [0.21, 0.11, 0.17, 0.17, 0.17, 0.17], row_height=0.28,
    )


def _word_rows(row: dict) -> list[list[str]]:
    words = row["target_words"]
    if not isinstance(words, list) or len(words) != 4:
        raise ValueError(f"Attese quattro parole per {row['project']}.")
    word_rows = []
    for position, word in enumerate(words, 1):
        token_count = word.get("token_count")
        if (
            not isinstance(word["text"], str) or not word["text"].strip() or not _valid_token_scores(word)
            or ("token_count" in word and (type(token_count) is not int or token_count < 1))
        ):
            raise ValueError(f"Dettaglio per parola non valido per {row['project']}.")
        word_rows.append([
            row["project"] if position == 1 else "",
            str(position), word["text"], str(token_count) if token_count is not None else "—",
            _format_probability(word["likelihood"]), f'{word["token_rank"]:.2f}',
        ])
    return word_rows


def _word_figure(title: str, rows: list[dict]) -> Figure:
    return _table_figure(
        title,
        f"{CONTEXT_NOTE}\n"
        "Token Rank è il rango medio dei token della parola.",
        ["Progetto", "Posizione", "Parola attesa", "Token", "Probabilità", "Token Rank"],
        [word_row for row in rows for word_row in _word_rows(row)],
        [0.20, 0.12, 0.20, 0.11, 0.19, 0.18], row_height=0.30, rows_per_project=4,
    )


def write_report(rows: Iterable[dict], output: Path) -> None:
    series = _index(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as staging:
        temporary = Path(staging) / "report.pdf"
        with PdfPages(temporary, metadata={"Title": "CanaryTrace · Valutazione"}) as pdf:
            for (condition, checkpoint, prompt), scores in series.items():
                title = f"{PROMPTS[prompt]}\n{_run_label(condition, checkpoint)}"
                ordered_rows = [row for _, row in sorted(scores.items())]
                pdf.savefig(_summary_figure(title, ordered_rows))
                word_rows = [row for row in ordered_rows if row.get("target_words") is not None]
                page_count = (len(word_rows) + PROJECTS_PER_DETAIL_PAGE - 1) // PROJECTS_PER_DETAIL_PAGE
                for page, start in enumerate(range(0, len(word_rows), PROJECTS_PER_DETAIL_PAGE), 1):
                    detail_title = f"Dettaglio per parola · {title} · {page}/{page_count}"
                    pdf.savefig(_word_figure(detail_title, word_rows[start:start + PROJECTS_PER_DETAIL_PAGE]))
        temporary.replace(output)


def _read_rows(paths: list[Path]) -> Iterator[dict]:
    for path in paths:
        with path.open(encoding="utf-8") as source:
            for number, line in enumerate(source, 1):
                if line.strip():
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as error:
                        raise ValueError(f"{path}, riga {number}: {error.msg}") from error


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m evaluation.report", description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+", help="Evaluation JSONL files")
    parser.add_argument("--output", type=Path, required=True, help="Output PDF path")
    args = parser.parse_args()
    output = args.output.resolve()
    if output in (path.resolve() for path in args.inputs):
        parser.error("Scegli un percorso PDF diverso da quello dei risultati JSONL.")
    try:
        write_report(_read_rows(args.inputs), output)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(f"Report salvato in {output}")


if __name__ == "__main__":
    main()
