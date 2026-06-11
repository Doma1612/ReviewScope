"""
Embedding stage interface.

Stages are deliberately tiny protocols: the Celery "Embed" task in the app
backend should be able to hold any of these behind one port, and the eval
harness swaps them per configuration without caring which library backs them.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Embedder(Protocol):
    """Anything that turns a list of texts into a (n, d) float array."""

    #: model identifier used in cache filenames and result rows
    model_name: str
    #: instruction slug ("no_inst", "generic", "domain", ...) — second cache axis
    instruction: str

    def encode(self, texts: list[str]) -> np.ndarray: ...

    def close(self) -> None:
        """Drop model weights and free accelerator memory (shared-GPU etiquette)."""
        ...
