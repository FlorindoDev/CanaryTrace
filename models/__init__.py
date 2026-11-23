"""Model interfaces and implementations for CanaryTrace."""

from .base import Model
from .qwen2_5 import Qwen25Instruct

__all__ = ["Model", "Qwen25Instruct"]
