from __future__ import annotations

import io
import tokenize
from typing import Generator


Tokens = tuple[tokenize.TokenInfo, ...]


class LineColConverter:
    """
    Convert (line, column) position to character offset,
    cached for performance
    """

    _code: str
    _cumulative_lengths: list[int] | None = None

    @property
    def code(self) -> str:
        return self._code

    @code.setter
    def code(self, code: str):
        self._code = code
        self._cumulative_lengths = None

    def __init__(self, code: str):
        self.code = code

    def _compute_cumulative_lengths(self) -> list[int]:
        """Compute cumulative character offsets at the start of each line."""
        lines = self.code.splitlines(keepends=True)
        cumulative = [0] * len(lines)
        total = 0
        for i, line in enumerate(lines):
            cumulative[i] = total
            total += len(line)
        return cumulative

    def line_col_to_offset(self, line_col: tuple[int, int]) -> int:
        if self._cumulative_lengths is None:
            self._cumulative_lengths = self._compute_cumulative_lengths()
        line_num, col = line_col
        return self._cumulative_lengths[line_num - 1] + col


def remove_indices(
    code: str, start: int, end: int, replace_with: str = " ", strip_middle: bool = False
) -> str:
    """Remove characters from start to end, replace with string."""
    left = code[:start]
    right = code[end:]
    if strip_middle:
        right = right.lstrip()
    return f"{left}{replace_with}{right}"


def tokenize_py(code: str) -> Generator[tokenize.TokenInfo, None, None]:
    """Tokenize Python/Cython code."""
    return tokenize.generate_tokens(io.StringIO(code).readline)
