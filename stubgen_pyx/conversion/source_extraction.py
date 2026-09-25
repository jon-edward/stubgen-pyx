"""Source extraction helpers for Cython AST nodes."""

from __future__ import annotations

import textwrap
import tokenize

from Cython.Compiler import ExprNodes, Nodes

from ..parsing.utils import tokenize_py
from .unparse import unparse_expr


def get_source(source: str, node: Nodes.Node) -> str:
    """Extract source code for a node, dedented and stripped.

    ``end_pos`` is often inaccurate in Cython's AST; the function falls back
    to the start position when it is missing. It also doesn't always extend
    through trailing closing punctuation -- e.g. for a multi-line,
    parenthesized ``from x import (...)``: ``end_pos()`` lands on the
    last name in the list, not the closing ``)`` on the following line.
    Rather than special-case that one node shape, extend the slice
    line-by-line while parentheses/brackets/braces are unbalanced, which
    covers it (and anything else with the same shape of inaccuracy)
    generically.
    """
    lines = source.splitlines(keepends=True)
    end_pos = node.end_pos() or node.pos
    end_line = end_pos[1]
    while end_line < len(lines) and _unbalanced_brackets(
        "".join(lines[node.pos[1] - 1 : end_line])
    ):
        end_line += 1
    output = "".join(lines[i - 1] for i in range(node.pos[1], end_line + 1))
    return textwrap.dedent(output).rstrip()


_BRACKET_PAIRS = {")": "(", "]": "[", "}": "{"}


def _unbalanced_brackets(text: str) -> bool:
    """Whether `text` has any unclosed ``(``/``[``/``{``, ignoring string/comment content."""
    stack: list[str] = []
    try:
        for token in tokenize_py(text):
            if token.type != tokenize.OP:
                continue
            if token.string in "([{":
                stack.append(token.string)
            elif token.string in _BRACKET_PAIRS:
                if stack and stack[-1] == _BRACKET_PAIRS[token.string]:
                    stack.pop()
    except (tokenize.TokenError, IndentationError, SyntaxError):
        # Incomplete/malformed-so-far text is exactly the case this is
        # called for mid-extension; a tokenize error here just means
        # "not balanced yet, keep extending" rather than something to
        # propagate.
        return True
    return bool(stack)


def get_decorators(
    source: str,
    node: Nodes.DefNode
    | Nodes.CFuncDefNode
    | Nodes.CClassDefNode
    | Nodes.PyClassDefNode,
) -> list[str]:
    """Return decorator source strings for a function or class node.

    Checks for decorators stashed by
    ``type_parsing.capture_static_types`` first: ``node.decorators`` is
    cleared by ``AnalyseDeclarationsTransform`` as part of rewriting a
    real decorator into an equivalent `f = decorator(f)` assignment (see
    that function's docstring) -- by the time this runs, on the
    post-pipeline tree, the live attribute is gone.

    Renders each decorator by unparsing its expression
    (``unparse_expr``, backed by ``Cython.CodeWriter``) rather than
    slicing source text by position: a decorator's own node positions
    are exactly as unreliable as any other node's (see `get_source`'s
    docstring on ``end_pos()``), but unparsing works purely from the AST
    structure and doesn't depend on them being accurate at all -- and
    it's the captured, pre-pipeline node structure regardless, so
    slicing the (possibly since-changed) source text by its position
    would be the wrong tool even if positions were reliable.
    """
    decorators = getattr(node, "_stubgen_static_decorators", None) or node.decorators
    if not decorators:
        return []
    rendered = []
    for decorator in decorators:
        expr = unparse_expr(decorator.decorator)
        rendered.append(
            f"@{expr}" if expr is not None else get_source(source, decorator)
        )
    return rendered


def get_bases(node: Nodes.CClassDefNode | Nodes.PyClassDefNode) -> list[str]:
    """Return base-class name strings from a class node."""
    if not hasattr(node, "bases") or not node.bases:
        return []
    base_strs = (unparse_expr(b) for b in node.bases.args)
    return [base_str for base_str in base_strs if base_str]


def get_metaclass(node: Nodes.PyClassDefNode | Nodes.CClassDefNode) -> str | None:
    """Return the metaclass name from a Python class node, if present."""
    if not isinstance(node, Nodes.PyClassDefNode):
        return None
    if node.metaclass and isinstance(node.metaclass, ExprNodes.NameNode):
        return node.metaclass.name
    return None
