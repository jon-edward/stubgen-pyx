"""Collapses several names bound to one module into a single name."""

from __future__ import annotations

import ast
import logging
import re
from dataclasses import dataclass

_logger = logging.getLogger(__name__)


def canonicalize_import_aliases(tree: ast.AST) -> ast.AST:
    """Use one name for each module that is imported under several names.

    ``import numpy as np`` and ``import numpy as cnp`` bind two names to the
    same module. The first import in the file supplies the canonical name:
    every use of the other names is rewritten to it and their imports are
    removed. Only module-level ``import module`` / ``import module as alias``
    statements take part. A group of names is left alone when renaming could
    change meaning: a name in it is also bound by something else (a class,
    a function, an argument, an import of a different module), or a quoted
    annotation mentions a name that would be removed.
    """
    if not isinstance(tree, ast.Module):
        raise TypeError("tree must be an ast.Module")

    candidates = _collect_candidates(tree)
    renames = _plan_renames(tree, candidates)
    if not renames:
        return tree

    for old, new in renames.items():
        _logger.debug("Canonicalized import alias %s to %s", old, new)
    tree = _NameRenamer(renames).visit(tree)
    _remove_aliases(tree, renames)
    return tree


@dataclass(frozen=True)
class _Candidate:
    """A module-level import alias that binds a name to exactly one module."""

    module: str
    bound: str
    alias: ast.alias


def _collect_candidates(tree: ast.Module) -> list[_Candidate]:
    candidates = []
    for stmt in tree.body:
        if not isinstance(stmt, ast.Import):
            continue
        for alias in stmt.names:
            # `import a.b` binds `a`, not the module `a.b`.
            if alias.asname or "." not in alias.name:
                candidates.append(
                    _Candidate(alias.name, alias.asname or alias.name, alias)
                )
    return candidates


def _foreign_bindings(
    tree: ast.AST, candidates: list[_Candidate]
) -> dict[str, set[str | None]]:
    """Names bound by anything but a candidate, with what each one refers to.

    The value is the module a name stands for (``import a.b`` binds ``a``),
    or ``None`` for anything that is not a module.
    """
    candidate_ids = {id(c.alias) for c in candidates}
    bound: dict[str, set[str | None]] = {}

    def bind(name: str, module: str | None) -> None:
        bound.setdefault(name, set()).add(module)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if id(alias) in candidate_ids:
                    continue
                root = alias.name.split(".")[0]
                bind(alias.asname or root, alias.name if alias.asname else root)
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name != "*":
                    bind(alias.asname or alias.name, None)
        elif isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load):
            bind(node.id, None)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bind(node.name, None)
        elif isinstance(node, ast.arg):
            bind(node.arg, None)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bind(node.name, None)
    return bound


def _annotations(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, (ast.AnnAssign, ast.arg)):
            annotation = node.annotation
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            annotation = node.returns
        else:
            continue
        if annotation is not None:
            yield annotation


def _quoted_annotation_text(tree: ast.AST) -> list[str]:
    return [
        part.value
        for annotation in _annotations(tree)
        for part in ast.walk(annotation)
        if isinstance(part, ast.Constant) and isinstance(part.value, str)
    ]


def _mentions(text: str, name: str) -> bool:
    return re.search(rf"(?<![\w.]){re.escape(name)}\b", text) is not None


def _plan_renames(tree: ast.Module, candidates: list[_Candidate]) -> dict[str, str]:
    """Map every name that should be replaced to the name replacing it."""
    modules_by_name: dict[str, set[str]] = {}
    names_by_module: dict[str, list[str]] = {}
    for candidate in candidates:
        modules_by_name.setdefault(candidate.bound, set()).add(candidate.module)
        names = names_by_module.setdefault(candidate.module, [])
        if candidate.bound not in names:
            names.append(candidate.bound)

    foreign = _foreign_bindings(tree, candidates)
    quoted = _quoted_annotation_text(tree)

    renames: dict[str, str] = {}
    for module, names in names_by_module.items():
        if len(names) < 2:
            continue
        canonical, others = names[0], names[1:]
        if any(
            len(modules_by_name[name]) > 1
            or any(other != module for other in foreign.get(name, ()))
            for name in names
        ):
            _logger.debug("Skipped canonicalizing aliases of %s: name conflict", module)
            continue
        if any(_mentions(text, name) for text in quoted for name in others):
            _logger.debug("Skipped canonicalizing aliases of %s: quoted use", module)
            continue
        renames.update({name: canonical for name in others})
    return renames


@dataclass
class _NameRenamer(ast.NodeTransformer):
    renames: dict[str, str]

    def visit_Name(self, node: ast.Name) -> ast.Name:
        new = self.renames.get(node.id)
        if new is None:
            return node
        return ast.copy_location(ast.Name(id=new, ctx=node.ctx), node)


def _remove_aliases(tree: ast.Module, renames: dict[str, str]) -> None:
    body: list[ast.stmt] = []
    for stmt in tree.body:
        if isinstance(stmt, ast.Import):
            stmt.names = [
                alias
                for alias in stmt.names
                if not _is_renamed_module_alias(alias, renames)
            ]
            if not stmt.names:
                continue
        body.append(stmt)
    tree.body = body


def _is_renamed_module_alias(alias: ast.alias, renames: dict[str, str]) -> bool:
    return (alias.asname or alias.name) in renames and (
        alias.asname is not None or "." not in alias.name
    )
