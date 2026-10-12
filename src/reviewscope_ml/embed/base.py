from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Embedder(Protocol):

    model_name: str
    instruction: str

    def encode(self, texts: list[str]) -> np.ndarray: ...

    def close(self) -> None:
        ...
