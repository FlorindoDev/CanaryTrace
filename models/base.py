"""Shared interface for the language models used in the experiment."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase


class Model(ABC):
    @abstractmethod
    def download(self) -> Path:
        """Return the local directory containing the downloaded checkpoint."""

    @abstractmethod
    def load(self, checkpoint: Path) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        """Load the weights and tokenizer from a local checkpoint."""
