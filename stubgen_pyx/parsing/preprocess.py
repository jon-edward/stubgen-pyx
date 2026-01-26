"""Python/Cython code preprocessing.

This uses Python's tokenize module which seems to work well with Cython.
"""

from __future__ import annotations

import io
import re
import tokenize
from typing import Callable

_Tokens = tuple[tokenize.TokenInfo, ...]
_PreprocessTransform = Callable[[str], str]


def preprocess(code: str) -> str:
    """Apply all preprocessing transformations to Python/Cython code."""
    transformations: list[_PreprocessTransform] = [
        replace_tabs_with_spaces,
        remove_comments,
        collapse_line_continuations,
        remove_contained_newlines,
        expand_colons,
        expand_semicolons,
    ]

    for transform in transformations:
        code = transform(code)
    return code


def replace_tabs_with_spaces(code: str) -> str:
    """Replace leading tabs with 4 spaces each."""
    tab_pattern = re.compile(r"^(\t+)", flags=re.MULTILINE)
    return tab_pattern.sub(lambda m: "    " * len(m.group(1)), code)


def remove_comments(code: str) -> str:
    """Remove all comments from the code."""
    for start, end in _get_comment_span_indices(code):
        code = remove_indices(code, start, end, replace_with=" ")
    return code


def collapse_line_continuations(code: str) -> str:
    """Collapse line continuations (backslash + newline) into spaces."""
    return re.sub(r"\\\n\s*", " ", code, flags=re.MULTILINE)


def remove_contained_newlines(code: str) -> str:
    """Remove newlines between brackets, parentheses, and braces."""
    indices = _get_newline_indices_in_brackets(code)
    for idx in indices:
        code = remove_indices(code, idx, idx + 1, replace_with="")
    return code


def expand_colons(code: str) -> str:
    """Expand colons that start blocks onto new indented lines."""
    lines = code.splitlines(keepends=True)

    for line_num, col in _get_colon_line_col_before_block(code):
        line_tail = lines[line_num - 1][col + 1 :]
        if line_tail.isspace():
            continue  # Already broken after colon

        indentation = _get_line_indentation(lines[line_num - 1])
        replace_with = f":\n{indentation}    "

        idx = line_col_to_offset(code, (line_num, col))
        code = remove_indices(
            code, idx, idx + 1, replace_with=replace_with, strip_middle=True
        )

    return code


def expand_semicolons(code: str) -> str:
    """Expand semicolons onto new lines with proper indentation."""
    lines = code.splitlines(keepends=True)

    for line_num, col in _get_semicolon_line_col(code):
        indentation = _get_line_indentation(lines[line_num - 1])
        replace_with = f"\n{indentation}"

        idx = line_col_to_offset(code, (line_num, col))
        code = remove_indices(
            code, idx, idx + 1, replace_with=replace_with, strip_middle=True
        )

    return code


def line_col_to_offset(code: str, line_col: tuple[int, int]) -> int:
    """Convert (line, column) position to character offset."""
    lines = code.splitlines(keepends=True)
    line_num, col = line_col
    offset = sum(len(lines[i]) for i in range(line_num - 1))
    return offset + col


def remove_indices(
    code: str, start: int, end: int, replace_with: str = " ", strip_middle: bool = False
) -> str:
    """Remove characters from start to end, replace with string."""
    left = code[:start]
    right = code[end:]
    if strip_middle:
        right = right.lstrip()
    return f"{left}{replace_with}{right}"


def _get_line_indentation(line: str) -> str:
    """Extract leading whitespace from a line."""
    match = re.match(r"^(\s*)", line)
    return match.group(1) if match else ""


def tokenize_py(code: str) -> _Tokens:
    """Tokenize Python/Cython code."""
    return tuple(tokenize.generate_tokens(io.StringIO(code).readline))


def _get_comment_span_indices(code: str) -> list[tuple[int, int]]:
    """Get character spans of all comments (reversed for safe removal)."""
    results = []
    for token in tokenize_py(code):
        if token.type == tokenize.COMMENT:
            start = line_col_to_offset(code, token.start)
            end = line_col_to_offset(code, token.end)
            results.append((start, end))
    results.sort(reverse=True)
    return results


def _get_newline_indices_in_brackets(code: str) -> list[int]:
    """Get indices of newlines inside brackets/parens/braces (reversed)."""
    results = []

    bracket_stack: list[str] = []
    bracket_pairs = {"(": ")", "[": "]", "{": "}"}

    tokens = tokenize_py(code)

    for idx, token in enumerate(tokens):
        token_str = token.string

        if token_str in bracket_pairs and token.type == tokenize.OP:
            bracket_stack.append(token_str)
        elif bracket_stack and token_str == bracket_pairs[bracket_stack[-1]]:
            bracket_stack.pop()
        elif token.type == tokenize.NL and bracket_stack:
            results.append(line_col_to_offset(code, token.start))

    results.sort(reverse=True)
    return results


def _get_colon_line_col_before_block(code: str) -> list[tuple[int, int]]:
    """Get (line, col) positions of colons that start code blocks (reversed)."""
    results = []
    bracket_stack = []
    bracket_pairs = {"(": ")", "[": "]", "{": "}"}

    for segment in _get_line_segments(code):
        for idx, token in enumerate(segment):
            token_str = token.string

            if token_str in bracket_pairs:
                bracket_stack.append(token_str)
            elif bracket_stack and token_str == bracket_pairs[bracket_stack[-1]]:
                bracket_stack.pop()
            elif token.type == tokenize.OP and token_str == ":" and not bracket_stack:
                if not _is_block(segment[0:idx]):
                    continue
                results.append(token.start)

    results.sort(reverse=True)
    return results


def _get_semicolon_line_col(code: str) -> list[tuple[int, int]]:
    """Get (line, col) positions of semicolons (reversed)."""
    results = []
    for token in tokenize_py(code):
        if token.type == tokenize.OP and token.string == ";":
            results.append(token.start)
    results.sort(reverse=True)
    return results


def _get_line_segments(
    code: str,
    break_types: tuple[int, ...] = (tokenize.NL, tokenize.NEWLINE),
    skip_types: tuple[int, ...] = (tokenize.INDENT, tokenize.DEDENT),
) -> list[_Tokens]:
    """Split tokens into logical line segments."""
    segments = []
    buffer = []

    for token in tokenize_py(code):
        if token.type in skip_types:
            continue

        buffer.append(token)

        if token.type in break_types or (
            token.type == tokenize.OP and token.string == ";"
        ):
            if buffer:
                segments.append(buffer)
                buffer = []

    if buffer:
        segments.append(buffer)

    return segments


_COMPOUND_TOKEN_STRINGS = {
    "if",
    "else",
    "elif",
    "for",
    "while",
    "with",
    "try",
    "except",
    "finally",
    "def",
    "class",
    "cdef",
    "match",
    "case",
}


def _is_block(tokens: _Tokens) -> bool:
    """Check if tokens form the start of a block."""
    if not len(tokens):
        return False
    if tokens[0].type != tokenize.NAME:
        return False
    if tokens[0].string in _COMPOUND_TOKEN_STRINGS:
        for token in tokens:
            if token.type == tokenize.OP and token.string == "=":
                return False
        return True
    return False
