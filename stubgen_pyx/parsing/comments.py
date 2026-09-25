"""Comment extraction, independent of Cython's parse tree.

Cython's own parser does not preserve comments on AST nodes: comment
tokens (``commentline``, produced whenever ``parse_comments`` is enabled,
which is the default) are only ever read in one place,
``Parsing.p_compiler_directive_comments``, used solely to find
``# cython: directive=value`` comments at the top of a block. Everywhere
else a ``commentline`` token is transparently skipped by the scanner and
never attached to any node. So capturing comments for stub output has to
be an independent pass over each file's text using Python's own
tokenizer.
"""

from __future__ import annotations

import re
import tokenize
from dataclasses import dataclass

from .utils import tokenize_py

#: Matches a `# cython: directive=value` compiler-directive comment, the
#: same pattern Cython's own parser uses to recognize these
#: (`Parsing._match_compiler_directive_comment`). These are meaningful to
#: the compiler, not documentation, and are filtered out of the comment
#: index so they don't leak into `.pyi` output as ordinary comments --
#: the same spirit as already silently dropping Cython-only imports
#: (`converter._is_cython_import`).
_DIRECTIVE_COMMENT_PATTERN = re.compile(r"^#\s*cython\s*:\s*((\w|[.])+\s*=.*)$")


@dataclass(frozen=True)
class Comment:
    """One comment, as found in the original source text.

    Attributes:
        text: The raw comment text, including the leading `#`.
        line: 1-based line number.
        col: 0-based column of the `#`.
        standalone: True if the comment is the only thing on its physical
            line (aside from leading whitespace); False if it trails code
            on the same line.
    """

    text: str
    line: int
    col: int
    standalone: bool


class CommentIndex:
    """All comments for one source file, queryable by line.

    Deliberately scoped to a single file -- see ``ParsedSource.comments``
    in ``parsing/parser.py``. A ``ScopeBuilder`` (or any other consumer)
    processing nodes from file A should only ever be able to reach file
    A's comments, never another file's, even when a shared
    ``StubgenContext`` has several files' worth of ``ParsedSource``s alive
    at once.
    """

    def __init__(self, comments: list[Comment]) -> None:
        self._by_line: dict[int, Comment] = {c.line: c for c in comments}
        self._claimed: set[int] = set()

    def leading_block(self, before_line: int) -> list[Comment]:
        """The contiguous run of standalone comments immediately above `before_line`.

        Stops at the first line that isn't an unclaimed, standalone
        comment (a blank line, code, or a comment already claimed by
        something else). Marks every comment it returns as claimed.
        """
        result: list[Comment] = []
        line = before_line - 1
        while line >= 1:
            comment = self._by_line.get(line)
            if comment is None or not comment.standalone or line in self._claimed:
                break
            result.append(comment)
            line -= 1
        result.reverse()
        self._claimed.update(c.line for c in result)
        return result

    def peek_trailing(self, on_line: int) -> Comment | None:
        """Same as `trailing`, but doesn't mark the comment as claimed.

        For a consumer that needs to inspect a candidate comment before
        committing to use it -- see `converter._find_signature_type_comment`,
        which must be able to tell a whole-signature `# type: (...) -> ...`
        comment trailing a `def` line apart from a *per-argument* comment
        that happens to share that same line (the last argument of a
        multi-line signature whose first line is the `def` line itself),
        without prematurely claiming the line and making it unavailable
        to whichever one turns out to actually want it.
        """
        comment = self._by_line.get(on_line)
        if comment is None or comment.standalone or on_line in self._claimed:
            return None
        return comment

    def trailing(self, on_line: int) -> Comment | None:
        """A same-line, non-standalone comment on `on_line`, if any and unclaimed."""
        comment = self._by_line.get(on_line)
        if comment is None or comment.standalone or on_line in self._claimed:
            return None
        self._claimed.add(on_line)
        return comment

    def all(self) -> list[Comment]:
        """Every comment in the file, regardless of claimed state. Non-mutating."""
        return sorted(self._by_line.values(), key=lambda c: c.line)

    def unclaimed(self) -> list[Comment]:
        """Every comment nobody has attached to a declaration yet."""
        return sorted(
            (c for line, c in self._by_line.items() if line not in self._claimed),
            key=lambda c: c.line,
        )


def extract_comments(source: str) -> CommentIndex:
    """Build a `CommentIndex` for one file's source text.

    `# cython: ...` compiler-directive comments are filtered out (see
    `_DIRECTIVE_COMMENT_PATTERN`) since they aren't documentation.
    """
    lines = source.splitlines()
    comments: list[Comment] = []
    try:
        for token in tokenize_py(source):
            if token.type != tokenize.COMMENT:
                continue
            if _DIRECTIVE_COMMENT_PATTERN.match(token.string):
                continue
            line_num, col = token.start
            line_text = lines[line_num - 1] if 0 < line_num <= len(lines) else ""
            standalone = line_text[:col].strip() == ""
            comments.append(Comment(token.string, line_num, col, standalone))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        # Tokenization can fail on constructs Python's tokenizer doesn't
        # accept even though they're valid Cython. The pre-migration
        # `extract_type_comments` let this propagate; here it's
        # deliberately softened to "no comments for this file" instead,
        # since a real Cython parse of the same source (run separately,
        # by the real Cython scanner/parser) can still succeed even when
        # Python's tokenizer chokes on it -- losing comments shouldn't
        # take down the whole conversion.
        return CommentIndex([])
    return CommentIndex(comments)
