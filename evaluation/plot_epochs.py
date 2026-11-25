"""Plot Exact Match counts across training epochs."""

import argparse
import re
import sys
from collections import Counter
from pathlib import Path
from tempfile import NamedTemporaryFile

from matplotlib.figure import Figure

from dataset import COPIES_PER_GROUP, load_canaries

from .report import PROMPTS, _read_rows


CANARIES_PER_GROUP = 6


def _count_epochs(directory: Path) -> dict[int, Counter[tuple[str, str]]]:
    expected_cases = {
        (prompt, canary.group, canary.project)
        for canary in load_canaries()
        for prompt in PROMPTS
    }
    counts: dict[int, Counter[tuple[str, str]]] = {}
    for path in sorted(directory.glob("*.jsonl")):
        match = re.fullmatch(r"fine-tuned-(0*[1-9][0-9]*)-epoch\.jsonl", path.name)
        if path.name == "baseline.jsonl":
            epoch = 0
        elif match:
            epoch = int(match[1])
        else:
            continue

        rows = list(_read_rows([path]))
        metadata = {(row["condition"], row.get("checkpoint")) for row in rows}
        if len(metadata) != 1:
            raise ValueError(f"{path.name} deve contenere i risultati di una sola valutazione.")
        (condition, checkpoint), = metadata
        if condition != ("fine_tuned" if epoch else "baseline"):
            raise ValueError(f"Condizione {condition} incompatibile con il nome {path.name}.")
        checkpoint_epoch = re.fullmatch(r"epoch-([0-9]+)", Path(checkpoint).name) if checkpoint else None
        if checkpoint_epoch and int(checkpoint_epoch[1]) != epoch:
            print(
                f"Ignorato {path.name} perché contiene il checkpoint {checkpoint_epoch[0]}.",
                file=sys.stderr,
            )
            continue
        if epoch in counts:
            raise ValueError(f"Più file di risultati per l'epoca {epoch}, incluso {path.name}.")

        cases = [(row["prompt_type"], row["group"], row["project"]) for row in rows]
        if len(cases) != len(expected_cases) or set(cases) != expected_cases:
            raise ValueError(f"Casi mancanti, duplicati o sconosciuti in {path.name}.")

        totals: Counter[tuple[str, str]] = Counter()
        for (prompt, group, _), row in zip(cases, rows):
            exact = row["exact_match"]
            if type(exact) not in (int, float) or exact not in (0, 1):
                raise ValueError(f"Exact Match non valido in {path.name} per {row['project']}.")
            totals[prompt, group] += int(exact)
        counts[epoch] = totals

    if not counts:
        raise ValueError(f"Nessun risultato valido in {directory}.")
    return counts


def _plot_epochs(counts: dict[int, Counter[tuple[str, str]]]) -> Figure:
    epochs = sorted(counts)
    figure = Figure(figsize=(10, 6.4), layout="constrained")
    axes = figure.subplots(len(PROMPTS), sharex=True)
    figure.suptitle("Exact Match per gruppo ed epoca", fontsize=16)
    figure.supylabel(f"Frasi indovinate per intero (su {CANARIES_PER_GROUP})")
    for axis, (prompt, title) in zip(axes, PROMPTS.items()):
        axis.set_title(title, loc="left")
        axis.set_ylim(-0.2, CANARIES_PER_GROUP + 0.2)
        axis.set_yticks(range(CANARIES_PER_GROUP + 1))
        axis.grid(axis="y", linestyle=":", alpha=0.5)
        for (group, copies), color, marker in zip(
            reversed(COPIES_PER_GROUP.items()),
            ("#2872c7", "#23834f", "#c65323", "#a77506"),
            ("o", "s", "^", "D"),
        ):
            copies_label = "copia" if copies == 1 else "copie"
            axis.plot(
                epochs, [counts[epoch][prompt, group] for epoch in epochs],
                color=color, marker=marker, markersize=5, linewidth=1.5,
                label=f"Gruppo {group} ({copies} {copies_label})",
            )
        if not any(totals[prompt, group] for totals in counts.values() for group in COPIES_PER_GROUP):
            axis.text(0.5, 0.5, "0 Exact Match per tutti i gruppi", transform=axis.transAxes, ha="center")
    axes[-1].set_xticks(epochs)
    axes[-1].set_xlabel("Epoca")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncols=4, frameon=False)
    return figure


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m evaluation.plot_epochs", description=__doc__)
    parser.add_argument(
        "--results-dir", type=Path, default=Path(__file__).resolve().parents[1] / "results",
        help="Cartella dei risultati e del grafico. Se omessa usa results/.",
    )
    args = parser.parse_args()
    try:
        directory = args.results_dir.resolve()
        output = directory / "exact-match-by-epoch.png"
        figure = _plot_epochs(_count_epochs(directory))
        staged = NamedTemporaryFile(dir=directory, delete=False)
        temporary = Path(staged.name)
        try:
            with staged:
                figure.savefig(staged, format="png", dpi=160)
            temporary.replace(output)
        finally:
            temporary.unlink(missing_ok=True)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(f"Grafico salvato in {output}")


if __name__ == "__main__":
    main()
