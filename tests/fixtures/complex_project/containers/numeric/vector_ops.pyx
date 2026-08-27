"""Python-facing vector operations in a second-level module."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass
class Vector:
    values: Sequence[float]

    @property
    def vector(self) -> Sequence[float]:
        return self.values

    @vector.setter
    def vector(self, vector: Sequence[float]) -> None:
        self.values = vector

    def __len__(self) -> int:
        return len(self.values)


def normalize(values: Iterable[float]) -> list[float]:
    numbers = list(values)
    total = sum(number * number for number in numbers) ** 0.5
    return [number / total for number in numbers] if total else numbers


def to_vector(values: Iterable[float]) -> Vector:
    return Vector(normalize(values))
