"""Evaluate the 48 CanaryTrace prompts on the base or fine-tuned model."""

import argparse
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch
from peft import PeftConfig, PeftModel, PeftType
from transformers import Qwen2ForCausalLM

from dataset import load_canaries
from models import Qwen25Instruct

from . import build_cases, evaluate


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m evaluation", description=__doc__)
    parser.add_argument("--checkpoint", type=Path, help="Directory of a trained LoRA adapter")
    parser.add_argument("--output", type=Path, help="Write JSONL and a companion PDF report")
    args = parser.parse_args()

    loader = Qwen25Instruct()
    checkpoint = args.checkpoint
    config = None
    if checkpoint is not None:
        if not (checkpoint / "adapter_model.safetensors").is_file():
            parser.error("La cartella deve contenere adapter_model.safetensors.")
        config = PeftConfig.from_pretrained(str(checkpoint.resolve()), local_files_only=True)
        if (config.peft_type, config.base_model_name_or_path, config.revision) != (
            PeftType.LORA, loader.REPO_ID, loader.REVISION
        ):
            parser.error("Il checkpoint deve essere un adattatore LoRA per la revisione Qwen del progetto.")

    base, tokenizer = loader.load(loader.download())
    if not isinstance(base, Qwen2ForCausalLM):
        raise TypeError("Atteso un modello Qwen2 per causal language modeling.")
    model = base if checkpoint is None else PeftModel.from_pretrained(
        base, str(checkpoint.resolve()), config=config, local_files_only=True
    )
    if torch.cuda.is_available():
        getattr(model, "cuda")()
    results = evaluate(model, tokenizer, build_cases(load_canaries()))

    condition = "baseline" if checkpoint is None else "fine_tuned"
    metadata = {"condition": condition, "checkpoint": checkpoint.as_posix() if checkpoint else None}
    rows = [{**metadata, **row} for row in results]
    jsonl = "".join(json.dumps(row, allow_nan=False) + "\n" for row in rows)
    if args.output is None:
        print(jsonl, end="")
    else:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        staged = NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", dir=output.parent, delete=False)
        temporary = Path(staged.name)
        try:
            with staged:
                staged.write(jsonl)
            temporary.replace(output)
        finally:
            temporary.unlink(missing_ok=True)
        print(f"Salvati {len(results)} risultati in {output}")

        from .report import write_report

        report = output.with_suffix(".report.pdf")
        write_report(rows, report)
        print(f"Report salvato in {report}")


if __name__ == "__main__":
    main()
