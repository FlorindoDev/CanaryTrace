"""Load the corpus, train LoRA adapters, and save epoch checkpoints."""

import json
from math import isfinite
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from peft import PeftModel
from tokenizers import Tokenizer
from torch.utils.data import DataLoader

from dataset import DIRECTORY, load_canaries, validate_corpus


def load_corpus(tokenizer: Tokenizer) -> torch.Tensor:
    with (DIRECTORY / "corpus.jsonl").open(encoding="utf-8") as source:
        records = [json.loads(line) for line in source]
    validate_corpus(records, tokenizer, load_canaries())
    return torch.tensor([record["input_ids"] for record in records], dtype=torch.long)


def save_adapter(model: PeftModel, checkpoint: Path) -> None:
    with TemporaryDirectory(prefix=".epoch-", dir=checkpoint.parent) as staging:
        model.save_pretrained(staging, safe_serialization=True, save_embedding_layers=False)
        directory = Path(staging)
        (directory / "README.md").unlink(missing_ok=True)
        directory.rename(checkpoint)


def train(model: PeftModel, batches: DataLoader, epochs: int, run: Path) -> None:
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad], lr=2e-4
    )
    model.train()

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        records_seen = 0
        for (input_ids,) in batches:
            input_ids = input_ids.to(model.device)
            optimizer.zero_grad()
            loss = model(input_ids=input_ids, labels=input_ids, use_cache=False).loss
            batch_loss = loss.item()
            if not isfinite(batch_loss):
                raise RuntimeError(f"Loss non finita durante l'epoca {epoch}.")
            loss.backward()
            optimizer.step()

            total_loss += batch_loss * len(input_ids)
            records_seen += len(input_ids)

        checkpoint = run / f"epoch-{epoch:02d}"
        save_adapter(model, checkpoint)
        print(f"Epoca {epoch}/{epochs} | loss {total_loss / records_seen:.4f} | {checkpoint}", flush=True)
