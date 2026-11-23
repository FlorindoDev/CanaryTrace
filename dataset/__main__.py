"""Generate or validate the offline CanaryTrace corpus."""

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

from tokenizers import Tokenizer

from . import DIRECTORY, Record, build_corpus, load_canaries, validate_corpus

CORPUS = DIRECTORY / "corpus.jsonl"
TOKENIZER_FILE = DIRECTORY / "tokenizer" / "tokenizer.json"
TOKENIZER_SHA256 = "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"


def _write_corpus(records: list[Record]) -> None:
    output = NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", dir=DIRECTORY, delete=False)
    temporary_path = Path(output.name)
    try:
        with output:
            for record in records:
                output.write(json.dumps(record, separators=(",", ":")) + "\n")
        temporary_path.replace(CORPUS)
    finally:
        temporary_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true", help="validate the saved corpus")
    args = parser.parse_args()

    try:
        if hashlib.sha256(TOKENIZER_FILE.read_bytes()).hexdigest() != TOKENIZER_SHA256:
            raise ValueError("Lo snapshot del tokenizer Qwen non corrisponde alla revisione attesa.")
        tokenizer = Tokenizer.from_file(str(TOKENIZER_FILE))
        tokenizer.no_padding()
        tokenizer.no_truncation()
        canaries = load_canaries()
        if args.validate_only:
            with CORPUS.open(encoding="utf-8") as source:
                records = [json.loads(line) for line in source]
        else:
            records = build_corpus(tokenizer, canaries)
        validate_corpus(records, tokenizer, canaries)
        if not args.validate_only:
            _write_corpus(records)
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")
    print(f"{CORPUS}: validazione superata per {len(records)} record.")


if __name__ == "__main__":
    main()
