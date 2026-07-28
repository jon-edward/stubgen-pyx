from __future__ import annotations

import ast
from collections.abc import Callable


def strip_artifacts(tree: ast.AST) -> ast.AST:
    """Remove Cython-generated stub artifacts that are not meaningful in Python stubs."""
    return ast.fix_missing_locations(_ArtifactStripper().visit(tree))


class _ArtifactStripper(ast.NodeTransformer):
    def visit_Module(self, node: ast.Module) -> ast.Module:
        node.body = self._filter_body(node.body, self._is_module_artifact)
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        node.decorator_list = [self.visit(d) for d in node.decorator_list]
        node.bases = [self.visit(b) for b in node.bases]
        node.keywords = [self.visit(k) for k in node.keywords]
        node.body = self._filter_body(node.body, self._is_class_artifact)
        return node

    def visit_Import(self, node: ast.Import) -> ast.AST | None:
        if any(name.name.startswith(("cython", "cpython")) for name in node.names):
            return None
        return node

    def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.AST | None:
        if node.module and node.module.startswith(("cython", "cpython")):
            return None
        return node

    def _filter_body(
        self,
        body: list[ast.stmt],
        is_artifact: Callable[[ast.stmt], bool],
    ) -> list[ast.stmt]:
        kept = []
        for stmt in body:
            if is_artifact(stmt):
                continue
            visited = self.visit(stmt)
            if visited is None:
                continue
            if isinstance(visited, ast.stmt):
                kept.append(visited)
        return kept

    @staticmethod
    def _is_module_artifact(node: ast.stmt) -> bool:
        if not isinstance(node, ast.Assign):
            return False
        # __all__ is always user-authored; preserve it unconditionally.
        if _is_name_assignment(node, "__all__"):
            return False
        return (
            len(node.targets) == 1
            and isinstance(node.value, (ast.Call, ast.Attribute))
            # TypeVar assignments are preserved; they carry useful stub type info.
            and not _is_typevar_call(node.value)
        )

    @staticmethod
    def _is_class_artifact(node: ast.stmt) -> bool:
        return (
            isinstance(node, ast.Assign)
            and _is_name_assignment(node, "__hash__")
            and isinstance(node.value, ast.Constant)
            and node.value.value is None
        )


def _is_name_assignment(node: ast.Assign, name: str) -> bool:
    return (
        len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == name
    )


def _is_typevar_call(node: ast.expr) -> bool:
    return isinstance(node, ast.Call) and (
        isinstance(node.func, ast.Name)
        and node.func.id == "TypeVar"
        or isinstance(node.func, ast.Attribute)
        and node.func.attr == "TypeVar"
    )
