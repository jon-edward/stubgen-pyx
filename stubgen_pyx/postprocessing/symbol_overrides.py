"""Apply project-specific symbol rewrites to generated stubs."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from ..config import SymbolOverride


def apply_symbol_overrides(
    tree: ast.Module,
    overrides: tuple[SymbolOverride, ...],
    *,
    module_root: str | None,
    source_root: Path | None,
    pyx_path: Path | None,
) -> ast.Module:
    """Rewrite generated references using explicit project-provided rules."""
    if not overrides:
        return tree
    module_name = _module_name(module_root, source_root, pyx_path)
    return _SymbolOverrideTransformer(
        overrides=overrides,
        module_name=module_name,
    ).visit(tree)


def _module_name(
    module_root: str | None, source_root: Path | None, pyx_path: Path | None
) -> str | None:
    if module_root is None or source_root is None or pyx_path is None:
        return None
    relative = pyx_path.resolve().relative_to(source_root.resolve()).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join([module_root, *parts])


def _dotted_expression(node: ast.expr) -> tuple[str, ...] | None:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return tuple(reversed(parts))


def _expression_from_dotted_name(name: str, ctx: ast.expr_context) -> ast.expr:
    parts = name.split(".")
    expression: ast.expr = ast.Name(id=parts[0], ctx=ctx)
    for part in parts[1:]:
        expression = ast.Attribute(value=expression, attr=part, ctx=ctx)
    return expression


@dataclass
class _SymbolOverrideTransformer(ast.NodeTransformer):
    overrides: tuple[SymbolOverride, ...]
    module_name: str | None
    bindings: dict[str, str] = field(default_factory=dict, init=False)
    defined_names: set[str] = field(default_factory=set, init=False)
    replacements: dict[str, str] = field(default_factory=dict, init=False)
    added_imports: list[ast.ImportFrom] = field(default_factory=list, init=False)

    def visit_Module(self, node: ast.Module) -> ast.Module:
        self.bindings = _import_bindings(node, self.module_name)
        self.defined_names = _defined_names(node)
        node.body = [
            statement
            for statement in node.body
            if not self._is_dropped_assignment(statement)
        ]
        self._raise_for_dropped_references(node)
        node = self.generic_visit(node)
        node.body = [*self.added_imports, *node.body]
        return node

    def visit_Name(self, node: ast.Name) -> ast.expr:
        if not isinstance(node.ctx, ast.Load):
            return node
        replacement = self._replacement_for_expression(node)
        return ast.copy_location(replacement, node) if replacement is not None else node

    def visit_Attribute(self, node: ast.Attribute) -> ast.expr:
        if not isinstance(node.ctx, ast.Load):
            return self.generic_visit(node)
        replacement = self._replacement_for_expression(node)
        if replacement is not None:
            return ast.copy_location(replacement, node)
        return self.generic_visit(node)

    def _is_dropped_assignment(self, statement: ast.stmt) -> bool:
        if isinstance(statement, ast.Assign):
            return self._matches_drop(statement.value)
        if isinstance(statement, ast.AnnAssign) and statement.value is not None:
            return self._matches_drop(statement.value)
        return False

    def _raise_for_dropped_references(self, tree: ast.Module) -> None:
        visitor = _DroppedReferenceVisitor(self.bindings, self.overrides)
        visitor.visit(tree)
        if visitor.references:
            references = ", ".join(sorted(visitor.references))
            raise ValueError(
                "symbol override action 'drop' cannot remove symbols used by "
                f"the generated public API: {references}"
            )

    def _matches_drop(self, expression: ast.expr) -> bool:
        canonical = self._canonical_name(expression)
        if canonical is None:
            return False
        rule = self._matching_rule(canonical, prefix=False)
        return rule is not None and rule.action == "drop"

    def _replacement_for_expression(self, expression: ast.expr) -> ast.expr | None:
        canonical = self._canonical_name(expression)
        if canonical is None:
            return None
        rule = self._matching_rule(canonical, prefix=True)
        if rule is None:
            return None
        if rule.literal is not None:
            if canonical != rule.source:
                return None
            return ast.parse(rule.literal, mode="eval").body
        if rule.import_target is None:
            return None
        target_local_name = self._target_local_name(rule.import_target)
        suffix = canonical.removeprefix(rule.source).lstrip(".")
        replacement = _expression_from_dotted_name(target_local_name, ast.Load())
        if suffix:
            for part in suffix.split("."):
                replacement = ast.Attribute(
                    value=replacement, attr=part, ctx=ast.Load()
                )
        return replacement

    def _canonical_name(self, expression: ast.expr) -> str | None:
        parts = _dotted_expression(expression)
        if parts is None or parts[0] not in self.bindings:
            return None
        return ".".join([self.bindings[parts[0]], *parts[1:]])

    def _matching_rule(
        self, canonical_name: str, *, prefix: bool
    ) -> SymbolOverride | None:
        matches = [
            rule
            for rule in self.overrides
            if canonical_name == rule.source
            or (
                prefix
                and rule.import_target is not None
                and canonical_name.startswith(rule.source + ".")
            )
        ]
        return max(matches, key=lambda rule: len(rule.source), default=None)

    def _target_local_name(self, target: str) -> str:
        if target in self.replacements:
            return self.replacements[target]
        for local_name, canonical_name in self.bindings.items():
            if canonical_name == target:
                self.replacements[target] = local_name
                return local_name

        target_module, _, target_name = target.rpartition(".")
        local_name = target_name
        if local_name in self.defined_names or local_name in self.bindings:
            index = 1
            local_name = f"_stubgen_pyx_{target_name}"
            while local_name in self.defined_names or local_name in self.bindings:
                index += 1
                local_name = f"_stubgen_pyx_{target_name}_{index}"
        alias = ast.alias(
            name=target_name,
            asname=local_name if local_name != target_name else None,
        )
        self.added_imports.append(
            ast.ImportFrom(module=target_module, names=[alias], level=0)
        )
        self.bindings[local_name] = target
        self.defined_names.add(local_name)
        self.replacements[target] = local_name
        return local_name


@dataclass
class _DroppedReferenceVisitor(ast.NodeVisitor):
    bindings: dict[str, str]
    overrides: tuple[SymbolOverride, ...]
    references: set[str] = field(default_factory=set)

    def visit_Import(self, node: ast.Import) -> None:
        return None

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        return None

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            self._record(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if isinstance(node.ctx, ast.Load):
            self._record(node)

    def _record(self, node: ast.expr) -> None:
        parts = _dotted_expression(node)
        if parts is None or parts[0] not in self.bindings:
            return
        canonical = ".".join([self.bindings[parts[0]], *parts[1:]])
        if any(
            override.action == "drop" and canonical == override.source
            for override in self.overrides
        ):
            self.references.add(canonical)


def _import_bindings(tree: ast.Module, module_name: str | None) -> dict[str, str]:
    bindings: dict[str, str] = {}
    for statement in tree.body:
        if isinstance(statement, ast.Import):
            for alias in statement.names:
                local_name = alias.asname or alias.name.split(".", 1)[0]
                bindings[local_name] = alias.name if alias.asname else local_name
        elif isinstance(statement, ast.ImportFrom):
            module = _resolve_import_from_module(statement, module_name)
            for alias in statement.names:
                if alias.name != "*":
                    bindings[alias.asname or alias.name] = f"{module}.{alias.name}"
    return bindings


def _resolve_import_from_module(node: ast.ImportFrom, module_name: str | None) -> str:
    if node.level == 0:
        return node.module or ""
    if module_name is None:
        relative = "." * node.level + (node.module or "")
        raise ValueError(
            f"cannot resolve relative import {relative!r} for symbol overrides "
            "without module_root, source_root, and pyx_path"
        )
    package_parts = module_name.split(".")[:-1]
    if node.level > len(package_parts) + 1:
        raise ValueError(f"relative import escapes package: {ast.unparse(node)}")
    base_parts = package_parts[: len(package_parts) - node.level + 1]
    if node.module:
        base_parts.extend(node.module.split("."))
    return ".".join(base_parts)


def _defined_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for statement in tree.body:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(statement.name)
        elif isinstance(statement, (ast.Assign, ast.AnnAssign)):
            targets = (
                statement.targets
                if isinstance(statement, ast.Assign)
                else [statement.target]
            )
            names.update(
                target.id for target in targets if isinstance(target, ast.Name)
            )
    return names
