"""Apply project-specific declaration replacements to generated stubs."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .symbol_overrides import _module_name

if TYPE_CHECKING:
    from pathlib import Path

    from ..config import DeclarationOverride


def apply_declaration_overrides(
    tree: ast.Module,
    overrides: tuple[DeclarationOverride, ...],
    *,
    module_root: str | None,
    source_root: Path | None,
    pyx_path: Path | None,
) -> ast.Module:
    """Replace configured generated declarations with explicit stub syntax."""
    if not overrides:
        return tree
    module_name = _module_name(module_root, source_root, pyx_path)
    transformer = _DeclarationOverrideTransformer(
        overrides=overrides,
        module_name=module_name,
    )
    tree = transformer.visit(tree)
    unmatched = sorted(
        override.target
        for override in overrides
        if override.target not in transformer.matched
    )
    if unmatched:
        raise ValueError(
            "declaration overrides did not match generated declarations: "
            + ", ".join(unmatched)
        )
    return ast.fix_missing_locations(tree)


@dataclass
class _DeclarationOverrideTransformer(ast.NodeTransformer):
    overrides: tuple[DeclarationOverride, ...]
    module_name: str | None
    class_stack: list[str] = field(default_factory=list, init=False)
    matched: set[str] = field(default_factory=set, init=False)

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        self.class_stack.append(node.name)
        node = self.generic_visit(node)
        self.class_stack.pop()
        return node

    def visit_FunctionDef(
        self, node: ast.FunctionDef
    ) -> ast.FunctionDef | list[ast.FunctionDef]:
        target = self._target_for(node.name)
        override = next(
            (candidate for candidate in self.overrides if candidate.target == target),
            None,
        )
        if override is None:
            return node

        self.matched.add(override.target)
        replacements = ast.parse(override.declarations).body
        for replacement in replacements:
            ast.copy_location(replacement, node)
        return replacements

    def _target_for(self, name: str) -> str | None:
        if self.module_name is None:
            return None
        return ".".join([self.module_name, *self.class_stack, name])
