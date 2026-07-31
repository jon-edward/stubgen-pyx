from __future__ import annotations

import ast
import builtins
from dataclasses import dataclass, field

_BUILTIN_NAMES = {name for name in dir(builtins) if not name.startswith("_")}
_BUILTIN_NAMES.update({"None"})


def validate_annotations(tree: ast.AST) -> ast.AST:
    """Replace unresolved annotation names with Any."""
    defined_names = _DefinedCollector.collect(tree) | _BUILTIN_NAMES
    validator = _AnnotationValidator(defined_names)
    tree = validator.visit(tree)
    if validator.replaced and not _has_typing_name(tree, "Any"):
        _insert_typing_import(tree, "Any")
    return ast.fix_missing_locations(tree)


@dataclass
class _DefinedCollector(ast.NodeVisitor):
    defined_names: set[str] = field(default_factory=set)

    @classmethod
    def collect(cls, tree: ast.AST) -> set[str]:
        collector = cls()
        collector.visit(tree)
        return collector.defined_names

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.defined_names.add(node.name)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.defined_names.add(node.name)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.defined_names.add(node.name)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.defined_names.add(alias.asname or alias.name.split(".", 1)[0])

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            self.defined_names.add(alias.asname or alias.name)

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            self._add_target(target)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self._add_target(node.target)

    def _add_target(self, target: ast.expr) -> None:
        if isinstance(target, ast.Name):
            self.defined_names.add(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                self._add_target(elt)


@dataclass
class _AnnotationValidator(ast.NodeTransformer):
    defined_names: set[str]
    replaced: set[str] = field(default_factory=set, init=False)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        self._visit_function_signature(node)
        self.generic_visit(node)
        return node

    def visit_AsyncFunctionDef(
        self, node: ast.AsyncFunctionDef
    ) -> ast.AsyncFunctionDef:
        self._visit_function_signature(node)
        self.generic_visit(node)
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        self.generic_visit(node)
        return node

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AnnAssign:
        node.annotation = self.visit(node.annotation)
        return node

    def visit_Name(self, node: ast.Name) -> ast.expr:
        if node.id in self.defined_names:
            return node
        self.replaced.add(node.id)
        return ast.copy_location(ast.Name(id="Any", ctx=ast.Load()), node)

    def visit_Attribute(self, node: ast.Attribute) -> ast.expr:
        root = _root_name(node)
        if root is not None and root not in self.defined_names:
            self.replaced.add(root)
            return ast.copy_location(ast.Name(id="Any", ctx=ast.Load()), node)
        visited = self.generic_visit(node)
        assert isinstance(visited, ast.expr)
        return visited

    def _visit_function_signature(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> None:
        args = node.args
        for arg in args.posonlyargs + args.args + args.kwonlyargs:
            if arg.annotation is not None:
                arg.annotation = self.visit(arg.annotation)
        for arg in (args.vararg, args.kwarg):
            if arg is not None and arg.annotation is not None:
                arg.annotation = self.visit(arg.annotation)
        if node.returns is not None:
            node.returns = self.visit(node.returns)


def _root_name(node: ast.Attribute) -> str | None:
    value = node.value
    while isinstance(value, ast.Attribute):
        value = value.value
    if isinstance(value, ast.Name):
        return value.id
    return None


def _has_typing_name(tree: ast.AST, name: str) -> bool:
    if not isinstance(tree, ast.Module):
        return False
    return any(
        isinstance(stmt, ast.ImportFrom)
        and stmt.module == "typing"
        and any(alias.name == name and alias.asname is None for alias in stmt.names)
        for stmt in tree.body
    )


def _insert_typing_import(tree: ast.AST, name: str) -> None:
    if not isinstance(tree, ast.Module):
        return
    insert_at = 0
    for idx, stmt in enumerate(tree.body):
        if isinstance(stmt, ast.ImportFrom) and stmt.module == "__future__":
            insert_at = idx + 1
    tree.body.insert(
        insert_at,
        ast.ImportFrom(module="typing", names=[ast.alias(name=name)], level=0),
    )
