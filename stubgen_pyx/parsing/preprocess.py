"""Python/Cython code preprocessing.

This uses Python's tokenize module which seems to work well with Cython.
"""

from __future__ import annotations

import re
import tokenize
from typing import Callable

from .utils import remove_indices, tokenize_py, Tokens, LineColConverter

_PreprocessTransform = Callable[[str], str]

_TAB_PATTERN = re.compile(r"^(\t+)", flags=re.MULTILINE)
_LINE_INDENT_PATTERN = re.compile(r"^(\s*)")
_LINE_CONTINUATION_PATTERN = re.compile(r"\\\n\s*")
_TYPE_COMMENT_PATTERN = re.compile(r"^#\s*type:\s")
_BRACKET_PAIRS = {
    "(": ")",
    "[": "]",
    "{": "}",
}


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
    return _TAB_PATTERN.sub(lambda m: "    " * len(m.group(1)), code)


def remove_comments(code: str) -> str:
    """Remove all comments from the code."""
    for start, end in _get_comment_span_indices(code):
        code = remove_indices(code, start, end, replace_with=" ")
    return code


def collapse_line_continuations(code: str) -> str:
    """Collapse line continuations (backslash + newline) into spaces."""
    return _LINE_CONTINUATION_PATTERN.sub(" ", code)


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

        idx = LineColConverter(code).line_col_to_offset((line_num, col))
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

        idx = LineColConverter(code).line_col_to_offset((line_num, col))
        code = remove_indices(
            code, idx, idx + 1, replace_with=replace_with, strip_middle=True
        )

    return code


def _get_line_indentation(line: str) -> str:
    """Extract leading whitespace from a line."""
    match = _LINE_INDENT_PATTERN.match(line)
    return match.group(1) if match else ""


def extract_type_comments(code: str) -> dict[int, str]:
    """Map line number → `# type: ...` comment text.

    Matches any PEP 484 style type comment (e.g. `# type: ignore[...]`,
    `# type: int`, `# type: () -> int`). Run before `remove_comments` so
    the comments are still present.
    """
    results: dict[int, str] = {}
    for token in tokenize_py(code):
        if token.type == tokenize.COMMENT and _TYPE_COMMENT_PATTERN.match(token.string):
            results[token.start[0]] = token.string
    return results


def _get_comment_span_indices(code: str) -> list[tuple[int, int]]:
    """Get character spans of all comments (reversed for safe removal)."""
    results = []
    line_converter = LineColConverter(code)

    for token in tokenize_py(code):
        if token.type == tokenize.COMMENT:
            start = line_converter.line_col_to_offset(token.start)
            end = line_converter.line_col_to_offset(token.end)
            results.append((start, end))

    results.sort(reverse=True)
    return results


def _get_newline_indices_in_brackets(code: str) -> list[int]:
    """Get indices of newlines inside brackets/parens/braces (reversed)."""
    results = []

    bracket_stack: list[str] = []

    line_converter = LineColConverter(code)

    for token in tokenize_py(code):
        token_str = token.string

        if token_str in _BRACKET_PAIRS and token.type == tokenize.OP:
            bracket_stack.append(token_str)
        elif bracket_stack and token_str == _BRACKET_PAIRS[bracket_stack[-1]]:
            bracket_stack.pop()
        elif token.type == tokenize.NL and bracket_stack:
            results.append(line_converter.line_col_to_offset(token.start))

    results.sort(reverse=True)
    return results


def get_lines_with_newlines_in_brackets(code: str) -> list[int]:
    """1-based line numbers whose terminating newline sits inside a bracket
    pair. These lines are joined with the following line by
    `remove_contained_newlines`."""
    results: list[int] = []
    bracket_stack: list[str] = []

    for token in tokenize_py(code):
        token_str = token.string
        if token_str in _BRACKET_PAIRS and token.type == tokenize.OP:
            bracket_stack.append(token_str)
        elif bracket_stack and token_str == _BRACKET_PAIRS[bracket_stack[-1]]:
            bracket_stack.pop()
        elif token.type == tokenize.NL and bracket_stack:
            results.append(token.start[0])

    return results


def _get_colon_line_col_before_block(code: str) -> list[tuple[int, int]]:
    """Get (line, col) positions of colons that start code blocks (reversed)."""
    results = []
    bracket_stack = []

    for segment in _get_line_segments(code):
        for idx, token in enumerate(segment):
            token_str = token.string

            if token_str in _BRACKET_PAIRS and token.type == tokenize.OP:
                bracket_stack.append(token_str)
            elif bracket_stack and token_str == _BRACKET_PAIRS[bracket_stack[-1]]:
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
) -> list[Tokens]:
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


def _is_block(tokens: Tokens) -> bool:
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
