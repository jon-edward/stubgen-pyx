"""
Collapses function definition `...` bodies and their declarations into a single line.

For example:
    def foo(x: int) -> int:
        ...

is converted to:
    def foo(x: int) -> int: ...

However:
    def foo(x: int) -> int: # type: ignore
        ...

is converted to:
    def foo(x: int) -> int: # type: ignore
       ...

"""

import tokenize

from ..parsing.utils import LineColConverter, tokenize_py, remove_indices


_SKIP_TYPES = (tokenize.INDENT, tokenize.DEDENT)


def collapse_funcdefs(code: str) -> str:
    for start, end in _get_colon_newline_ellipsis_indices(code):
        code = remove_indices(code, start, end)
    return code


def _get_colon_newline_ellipsis_indices(code: str) -> list[tuple[int, int]]:
    results = []

    line_converter = LineColConverter(code)
    last_token: tokenize.TokenInfo | None = None
    last_last_token: tokenize.TokenInfo | None = None

    for token in tokenize_py(code):
        if token.type in _SKIP_TYPES:
            continue

        if last_token is not None and last_last_token is not None:
            if (
                last_last_token.type == tokenize.OP
                and last_last_token.string == ":"
                and last_token.type == tokenize.NEWLINE
                and token.type == tokenize.OP
                and token.string == "..."
            ):
                start = line_converter.line_col_to_offset(last_last_token.end)
                end = line_converter.line_col_to_offset(token.start)
                results.append((start, end))

        last_last_token = last_token
        last_token = token

    results.sort(reverse=True)
    return results
