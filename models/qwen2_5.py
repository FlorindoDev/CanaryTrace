"""Qwen2.5-0.5B-Instruct checkpoint for the CanaryTrace experiment."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from huggingface_hub import snapshot_download

from .base import Model

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase


class Qwen25Instruct(Model):
    REPO_ID = "Qwen/Qwen2.5-0.5B-Instruct"
    REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"

    def download(self) -> Path:
        return Path(
            snapshot_download(
                self.REPO_ID,
                revision=self.REVISION,
                allow_patterns=["*.json", "model.safetensors", "merges.txt", "LICENSE"],
                token=False,
            )
        )

    def load(self, checkpoint: Path) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        if not checkpoint.is_dir():
            raise FileNotFoundError(f"Cartella del checkpoint mancante: {checkpoint}")

        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint, local_files_only=True, trust_remote_code=False
        )
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
            dtype="auto",
        )
        return model, tokenizer
