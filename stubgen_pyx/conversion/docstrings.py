"""Docstring formatting helpers for conversion outputs."""

from __future__ import annotations

import textwrap


def docstring_to_string(docstring: str) -> str:
    """Wrap a raw docstring in triple-double-quotes, escaping embedded ones."""
    if not docstring:
        return '""" """'
    first_line, *rest = docstring.splitlines(keepends=True)
    rest_joined = textwrap.dedent("".join(rest))
    body = f"{first_line}{rest_joined}".replace('"""', r"\"\"\"")
    return f'"""{body}"""'
