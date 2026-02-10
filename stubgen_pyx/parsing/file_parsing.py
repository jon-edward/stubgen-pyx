"""
File-specific parsing logic.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
import tokenize

from .preprocess import remove_indices, tokenize_py, LineColConverter


def file_parsing_preprocess(source: Path, code: str) -> str:
    """
    Preprocess Cython code before parsing it.
    """
    code = _expand_includes(source, code)
    code = _replace_equals_star(code)
    return code


def _read_file_fallback(path: Path, fallback: str) -> str:
    try:
        return path.read_text()
    except (UnicodeDecodeError, FileNotFoundError):
        return fallback


def _expand_includes(source: Path, code: str) -> str:
    """Expand includes in Cython code."""
    includes = _get_includes(source, code)

    for include in includes:
        code = remove_indices(
            code,
            include.start,
            include.end,
            replace_with=_read_file_fallback(include.path, "\n"),
        )
    return code


@dataclass
class _Include:
    path: Path
    start: int
    end: int


def _try_parse_string(code: str) -> str | None:
    try:
        evaluated = ast.literal_eval(code)
        if not isinstance(evaluated, str):
            return None
        return evaluated
    except SyntaxError:
        return None


def _get_includes(source: Path, code: str) -> list[_Include]:
    """Get character spans of all include directives (reversed for safe removal)."""
    results = []
    last_token: tokenize.TokenInfo | None = None

    line_converter = LineColConverter(code)
    for token in tokenize_py(code):
        if (
            token.type == tokenize.STRING
            and last_token
            and last_token.type == tokenize.NAME
            and last_token.string == "include"
        ):
            include_str = _try_parse_string(token.string)

            if not include_str:
                last_token = token
                continue

            include_path = source.parent / include_str

            start = line_converter.line_col_to_offset(last_token.start)
            end = line_converter.line_col_to_offset(token.end)
            results.append(_Include(include_path, start, end))
        last_token = token

    results.reverse()
    return results


def _get_equals_star_indices(code: str) -> list[tuple[int, int]]:
    results = []
    last_token: tokenize.TokenInfo | None = None

    line_converter = LineColConverter(code)
    for token in tokenize_py(code):
        if token.string == "*" and last_token and last_token.string == "=":
            start = line_converter.line_col_to_offset(token.start)
            end = line_converter.line_col_to_offset(token.end)
            results.append((start, end))
        last_token = token

    results.reverse()
    return results


def _replace_equals_star(code: str) -> str:
    for start, end in _get_equals_star_indices(code):
        code = remove_indices(code, start, end, replace_with="...")
    return code
