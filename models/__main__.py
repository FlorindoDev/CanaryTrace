"""Download and load the baseline model with `python -m models`."""

from .qwen2_5 import Qwen25Instruct


def main() -> None:
    loader = Qwen25Instruct()
    checkpoint = loader.download()
    model, tokenizer = loader.load(checkpoint)
    print(f"Checkpoint in {checkpoint}")
    print(f"Caricati {model.num_parameters():,} parametri e un tokenizer da {len(tokenizer):,} token.")


if __name__ == "__main__":
    main()
