from __future__ import annotations

import ast


def unquote_annotations(tree: ast.AST) -> ast.AST:
    return ast.fix_missing_locations(_AnnotationUnquoter().visit(tree))


class _AnnotationUnquoter(ast.NodeTransformer):
    @staticmethod
    def _unquote(annotation: ast.expr | None) -> ast.expr | None:
        if not (
            isinstance(annotation, ast.Constant) and isinstance(annotation.value, str)
        ):
            return annotation

        try:
            parsed = ast.parse(annotation.value, mode="eval").body
        except SyntaxError:
            return annotation

        return ast.copy_location(parsed, annotation)

    def visit_arg(self, node: ast.arg) -> ast.arg:
        node.annotation = self._unquote(node.annotation)
        return node

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AnnAssign:
        unquoted = self._unquote(node.annotation)
        if unquoted is not None:
            node.annotation = unquoted
        self.generic_visit(node)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        node.returns = self._unquote(node.returns)
        self.generic_visit(node)
        return node

    def visit_AsyncFunctionDef(
        self, node: ast.AsyncFunctionDef
    ) -> ast.AsyncFunctionDef:
        node.returns = self._unquote(node.returns)
        self.generic_visit(node)
        return node
