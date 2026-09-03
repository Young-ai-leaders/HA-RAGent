from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")

@dataclass
class ScoredResult(Generic[T]):
    item: T
    score: float
    rank: int
