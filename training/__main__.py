"""Fine-tune Qwen on the CanaryTrace corpus with LoRA."""

import argparse
from pathlib import Path
from tempfile import mkdtemp

import torch
from peft import LoraConfig, TaskType, get_peft_model
from tokenizers import Tokenizer
from torch.utils.data import DataLoader, TensorDataset

from dataset import DIRECTORY
from models import Qwen25Instruct

from . import load_corpus, train

CHECKPOINTS = DIRECTORY.parent / "checkpoints"


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m training", description=__doc__)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs e --batch-size devono essere maggiori di zero.")

    torch.manual_seed(20261109)
    loader = Qwen25Instruct()
    base, tokenizer = loader.load(loader.download())
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if not isinstance(backend, Tokenizer):
        raise TypeError("Il tokenizer deve usare il backend di Hugging Face Tokenizers.")
    corpus = load_corpus(backend)
    batches = DataLoader(TensorDataset(corpus), batch_size=args.batch_size, shuffle=True)

    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
    )
    model = get_peft_model(base, config, revision=loader.REVISION)
    config.base_model_name_or_path = loader.REPO_ID
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    CHECKPOINTS.mkdir(exist_ok=True)
    run = Path(mkdtemp(prefix="run-", dir=CHECKPOINTS))
    print(f"Training su {device} con {len(corpus)} record. Checkpoint in {run}", flush=True)
    train(model, batches, args.epochs, run)


if __name__ == "__main__":
    main()
