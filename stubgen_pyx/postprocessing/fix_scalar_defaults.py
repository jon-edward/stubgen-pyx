"""Coerce literal defaults left mismatched by ``char *``/``bint`` conversion.

``type_parsing.py`` maps a ``char *`` parameter's *annotation* to ``bytes``,
and (optionally) ``normalize_names`` maps a ``bint`` parameter's annotation
to ``bool``. Neither touches the parameter's *default value*, which Cython's
stubgen carries over verbatim as the original C literal. The result is a
syntactically valid but type-mismatched stub::

    def f(x: bytes = '.', flag: bool = 0): ...

This pass rewrites the default literal - never the annotation - so the two
agree::

    def f(x: bytes = b'.', flag: bool = False): ...

Only bare ``Name`` annotations of exactly ``bytes`` or ``bool``/``bint`` are
matched, and only plain ``str``/``int`` constant defaults are rewritten.
Anything more indirect (``Optional[bytes]``, a non-literal default such as a
call or a name, a default that's already the right type) is left untouched.
"""

from __future__ import annotations

import ast
import logging

_logger = logging.getLogger(__name__)

# Both spellings are matched so this pass works whether or not the
# `bint` -> `bool` rename (normalize_names, which is user-configurable) has
# already run.
_BOOL_ANNOTATIONS = {"bool", "bint"}


def fix_scalar_defaults(tree: ast.AST) -> ast.AST:
    """Coerce mismatched str/int literal defaults on bytes/bool-annotated args."""
    return _ScalarDefaultFixer().visit(tree)


class _ScalarDefaultFixer(ast.NodeTransformer):
    """Rewrite bytes/bool argument defaults left as str/int literals."""

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        self._fix_arguments(node.args)
        return self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        self._fix_arguments(node.args)
        return self.generic_visit(node)

    def _fix_arguments(self, args: ast.arguments) -> None:
        positional = [*args.posonlyargs, *args.args]
        _rewrite_defaults(positional, args.defaults)
        _rewrite_defaults(args.kwonlyargs, args.kw_defaults, sparse=True)


def _rewrite_defaults(
    arg_list: list[ast.arg], defaults: list[ast.expr | None], *, sparse: bool = False
) -> None:
    # `args.defaults` only covers the trailing (right-aligned) positional
    # args; `args.kw_defaults` is one-to-one with kwonlyargs but may contain
    # `None` for required keyword-only args.
    offset = 0 if sparse else len(arg_list) - len(defaults)
    for index, default in enumerate(defaults):
        if default is None:
            continue
        arg = arg_list[offset + index]
        coerced = _coerce(arg.annotation, default)
        if coerced is not None:
            defaults[index] = coerced


def _annotation_name(annotation: ast.expr | None) -> str | None:
    return annotation.id if isinstance(annotation, ast.Name) else None


def _coerce(annotation: ast.expr | None, default: ast.expr) -> ast.expr | None:
    name = _annotation_name(annotation)
    if not isinstance(default, ast.Constant):
        return None

    if name == "bytes" and isinstance(default.value, str):
        _logger.debug("Coercing str default %r to bytes", default.value)
        try:
            new_value: object = default.value.encode("latin-1")
        except UnicodeEncodeError:
            _logger.debug(
                "Could not coerce str default %r to bytes for a `bytes`-annotated "
                "argument; left as a str literal",
                default.value,
            )
            return None
        return ast.copy_location(ast.Constant(value=new_value), default)

    if (
        name in _BOOL_ANNOTATIONS
        and isinstance(default.value, int)
        and not isinstance(default.value, bool)
    ):
        _logger.debug("Coercing int default %r to bool", default.value)
        return ast.copy_location(ast.Constant(value=bool(default.value)), default)

    return None
