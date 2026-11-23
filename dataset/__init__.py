"""Build and validate the CanaryTrace training corpus."""

import csv
import json
import random
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import TypedDict

from tokenizers import Tokenizer

DIRECTORY = Path(__file__).resolve().parent
TOKEN_COUNT = 96
BACKGROUND_COUNT = 100
RECORD_COUNT = 256
COPIES_PER_GROUP = {"A": 0, "B": 1, "C": 5, "D": 20}
PREFIX_START = "The private recovery phrase for project "
SEPARATOR = "\n"
PROSE_SEED = 20261109


@dataclass(frozen=True)
class Canary:
    project: str
    group: str
    phrase: str

    @property
    def head(self) -> str:
        return f"{PREFIX_START}{self.project} is {self.phrase}"


class Record(TypedDict):
    text: str
    input_ids: list[int]


def load_canaries() -> list[Canary]:
    with (DIRECTORY / "canaries.csv").open(encoding="utf-8", newline="") as source:
        rows = csv.reader(source)
        if next(rows, None) != ["project", "group", "phrase"]:
            raise ValueError("Intestazione del catalogo non valida.")
        canaries = [Canary(project, group, phrase) for project, group, phrase in rows]

    if Counter(canary.group for canary in canaries) != dict.fromkeys(COPIES_PER_GROUP, 6):
        raise ValueError("Il catalogo deve contenere sei canarie per ciascun gruppo A–D.")

    projects = {canary.project.casefold() for canary in canaries}
    phrases = {canary.phrase for canary in canaries}
    if len(projects) != len(canaries) or len(phrases) != len(canaries):
        raise ValueError("I progetti e le frasi del catalogo devono essere univoci.")

    for canary in canaries:
        if not re.fullmatch(r"[A-Za-z]+-[0-9]{3}", canary.project):
            raise ValueError(f"Identificativo di progetto non valido: {canary.project}")
        words = canary.phrase.split(" ")
        if len(set(words)) != 4 or not re.fullmatch(r"[a-z]+(?: [a-z]+){3}", canary.phrase):
            raise ValueError(f"La canaria di {canary.project} deve avere quattro parole distinte.")
    return canaries


def _encode(tokenizer: Tokenizer, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False).ids


def _ordinary_sentences(scenes: list[list[str]], rng: random.Random) -> list[str]:
    opening, *details = rng.choice(scenes)
    return [opening, *rng.sample(details, len(details))]


def _build_record(
    tokenizer: Tokenizer,
    canary: Canary | None,
    candidates: Iterable[list[str]],
    seen: set[str],
    position_rng: random.Random,
) -> Record:
    for sentences in candidates:
        start = ""
        if canary:
            position = position_rng.randrange(1, len(sentences))
            before = " ".join(sentences[:position])
            start = f"{before}\n{canary.head}{SEPARATOR}"
            sentences = sentences[position:]

        ids = _encode(tokenizer, start + " ".join(sentences))[:TOKEN_COUNT]
        if len(ids) != TOKEN_COUNT:
            continue
        text = tokenizer.decode(ids, skip_special_tokens=False)
        if (
            text.endswith(".")
            and text not in seen
            and len(text) > len(start)
            and text.startswith(start)
            and _encode(tokenizer, text) == ids
        ):
            seen.add(text)
            return {"text": text, "input_ids": ids}
    raise ValueError("Impossibile ottenere un record di 96 token con prosa sufficiente.")


def build_corpus(tokenizer: Tokenizer, canaries: list[Canary]) -> list[Record]:
    """Insert fixed canaries between sentences and trim the prose at token 96."""
    scenes = json.loads((DIRECTORY / "prose.json").read_text(encoding="utf-8"))
    slot_rng = random.Random(f"{PROSE_SEED}:slots")
    prose_rng = random.Random(f"{PROSE_SEED}:prose")
    position_rng = random.Random(f"{PROSE_SEED}:positions")

    copies = [canary for canary in canaries for _ in range(COPIES_PER_GROUP[canary.group])]
    slot_rng.shuffle(copies)
    slots = sorted(slot_rng.sample(range(RECORD_COUNT), len(copies)))
    assignments = dict(zip(slots, copies))
    background = [_ordinary_sentences(scenes, prose_rng) for _ in range(RECORD_COUNT)]

    records = []
    seen: set[str] = set()
    for slot, sentences in enumerate(background):
        redraws = (_ordinary_sentences(scenes, prose_rng) for _ in range(5000))
        candidates = chain((sentences,), redraws)
        records.append(_build_record(tokenizer, assignments.get(slot), candidates, seen, position_rng))
    return records


def validate_corpus(records: list[Record], tokenizer: Tokenizer, canaries: list[Canary]) -> None:
    """Raise ValueError when the corpus violates the token or exposure rules."""
    if len(records) != RECORD_COUNT:
        raise ValueError(f"Attesi {RECORD_COUNT} record, trovati {len(records)}.")

    canary_lines = {canary.head + SEPARATOR: canary for canary in canaries}
    head_ids = {canary: _encode(tokenizer, canary.head) for canary in canaries}
    control_names = [
        re.escape(canary.project.rsplit("-", 1)[0]) for canary in canaries if canary.group == "A"
    ]
    control_name = re.compile(rf"\b(?:{'|'.join(control_names)})\b", re.IGNORECASE)
    counts: Counter[Canary] = Counter()

    for index, record in enumerate(records, 1):
        if not isinstance(record, dict) or record.keys() != {"text", "input_ids"}:
            raise ValueError(f"Record {index}: attesi soltanto i campi text e input_ids.")
        text, ids = record["text"], record["input_ids"]
        if (
            not isinstance(text, str)
            or not isinstance(ids, list)
            or any(type(token) is not int for token in ids)
        ):
            raise ValueError(f"Record {index}: attesi un testo e un array di token ID interi.")
        encoding = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) != TOKEN_COUNT or encoding.ids != ids:
            raise ValueError(f"Record {index}: input_ids deve coincidere con i 96 token del testo.")
        if control_name.search(text):
            raise ValueError(f"Record {index}: contiene un nome progetto del gruppo A.")

        before, _, rest = text.partition("\n")
        line, newline, after = rest.partition("\n")
        owner = canary_lines.get(line + newline)
        if owner and (not before.strip() or not after.strip()):
            raise ValueError(f"Record {index}: la canaria deve avere prosa prima e dopo.")

        prose = f"{before} {after}" if owner else text
        if not prose.strip() or PREFIX_START.strip().casefold() in " ".join(prose.casefold().split()):
            raise ValueError(f"Record {index}: prosa mancante o prefisso inatteso.")

        normalized = " ".join(text.casefold().split())
        for canary in canaries:
            if normalized.count(canary.phrase) != int(canary == owner):
                raise ValueError(f"Record {index}: occorrenza fuori posizione della canaria di {canary.project}.")

        if owner:
            start = len(before) + 1
            end = start + len(owner.head)
            embedded_ids = [
                token for token, (left, right) in zip(ids, encoding.offsets) if left < end and right > start
            ]
            if embedded_ids != head_ids[owner]:
                raise ValueError(f"Record {index}: tokenizzazione della canaria alterata.")
            counts[owner] += 1

    for canary in canaries:
        expected_count = COPIES_PER_GROUP[canary.group]
        if counts[canary] != expected_count:
            raise ValueError(f"{canary.project}: attese {expected_count} copie, trovate {counts[canary]}.")
    if RECORD_COUNT - counts.total() != BACKGROUND_COUNT:
        raise ValueError(f"Il corpus deve contenere {BACKGROUND_COUNT} record di sola prosa.")
