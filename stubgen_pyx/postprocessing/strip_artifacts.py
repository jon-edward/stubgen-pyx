"""Strip Cython-only stub artifacts and repair dangling annotations."""

from __future__ import annotations

import ast
import builtins

# Names always considered resolvable when checking annotations.
_BUILTIN_NAMES = {name for name in dir(builtins) if not name.startswith("_")} | {"None"}

# Cython import roots that have no meaning in .pyi stubs.
_BLOCKED_IMPORT_PREFIXES = ("cython", "cpython")


class _AnnotationRewriter(ast.NodeTransformer):
    """Collapse annotation references whose root name is no longer defined.

    After Cython-only imports and other stub-only noise are removed, annotations may be
    left pointing at names that no longer exist in the generated module. This
    transformer rewrites those dangling references to ``Any`` and records whether it
    changed anything.
    """

    def __init__(self, defined: set[str]) -> None:
        self.defined = defined
        self.changed = False

    def _any(self) -> ast.Name:
        self.changed = True
        return ast.Name(id="Any", ctx=ast.Load())

    def visit_Name(self, node: ast.Name) -> ast.expr:
        return node if node.id in self.defined else self._any()

    def visit_Attribute(self, node: ast.Attribute) -> ast.expr:
        # Dotted access like `some_module.Type`: only the leftmost root name
        # has to resolve. If `some_module` is defined, trust the whole chain
        # and leave it alone. If not, the whole chain is dangling and
        # collapses to `Any`. Descending and only rewriting the value half of an
        # `ast.Attribute` produces nonsense like `Any.DEFAULT_SEED`.
        root: ast.expr = node
        while isinstance(root, ast.Attribute):
            root = root.value
        if isinstance(root, ast.Name) and root.id not in self.defined:
            return self._any()
        return node


def _is_typevar_assignment(node: ast.Assign) -> bool:
    func = node.value.func if isinstance(node.value, ast.Call) else None
    return (
        getattr(func, "id", None) == "TypeVar"
        or getattr(func, "attr", None) == "TypeVar"
    )


def _is_runtime_assignment(node: ast.Assign) -> bool:
    if len(node.targets) != 1 or getattr(node.targets[0], "id", None) == "__all__":
        return False
    return isinstance(
        node.value, (ast.Call, ast.Attribute)
    ) and not _is_typevar_assignment(node)


def _keep_node(node: ast.stmt) -> bool:
    if isinstance(node, ast.Import):
        # Filter individual dotted names inside an `import a, cython, b`.
        node.names = [
            name
            for name in node.names
            if not name.name.startswith(_BLOCKED_IMPORT_PREFIXES)
        ]
        return bool(node.names)
    if isinstance(node, ast.ImportFrom):
        return not (node.module and node.module.startswith(_BLOCKED_IMPORT_PREFIXES))
    return not (isinstance(node, ast.Assign) and _is_runtime_assignment(node))


def _target_names(target: ast.expr) -> set[str]:
    names = set()
    stack = [target]
    while stack:
        node = stack.pop()
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, (ast.Tuple, ast.List)):
            stack.extend(node.elts)
    return names


def _defined_by(node: ast.stmt) -> set[str]:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return {node.name}
    if isinstance(node, ast.Import):
        # `import a.b.c` binds `a` unless an alias is present.
        return {alias.asname or alias.name.split(".", 1)[0] for alias in node.names}
    if isinstance(node, ast.ImportFrom):
        return {alias.asname or alias.name for alias in node.names}
    if isinstance(node, ast.Assign):
        return set().union(*(_target_names(target) for target in node.targets))
    if isinstance(node, ast.AnnAssign):
        return _target_names(node.target)
    return set()


def _strip_and_collect(body: list[ast.stmt]) -> tuple[list[ast.stmt], set[str]]:
    kept = []
    defined = set(_BUILTIN_NAMES)
    for node in body:
        if _keep_node(node):
            kept.append(node)
            defined.update(_defined_by(node))
    return kept, defined


def _rewrite_function_annotations(
    node: ast.FunctionDef | ast.AsyncFunctionDef, rewriter: _AnnotationRewriter
) -> None:
    node.decorator_list = [rewriter.visit(dec) for dec in node.decorator_list]
    args = node.args
    for arg in (
        *args.posonlyargs,
        *args.args,
        *args.kwonlyargs,
        args.vararg,
        args.kwarg,
    ):
        if arg and arg.annotation:
            arg.annotation = rewriter.visit(arg.annotation)
    if node.returns:
        node.returns = rewriter.visit(node.returns)


def _rewrite_annotations(body: list[ast.stmt], defined: set[str]) -> bool:
    rewriter = _AnnotationRewriter(defined)
    for node in body:
        children = node.body if isinstance(node, ast.ClassDef) else (node,)
        for child in children:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                _rewrite_function_annotations(child, rewriter)
            elif isinstance(child, ast.AnnAssign):
                child.annotation = rewriter.visit(child.annotation)
    return rewriter.changed


def _ensure_any_import(body: list[ast.stmt]) -> None:
    insert_at = 0
    typing_import: ast.ImportFrom | None = None
    for index, stmt in enumerate(body):
        if not isinstance(stmt, ast.ImportFrom):
            continue
        if stmt.module == "__future__":
            insert_at = index + 1
        elif stmt.module == "typing":
            if any(alias.name == "Any" and not alias.asname for alias in stmt.names):
                return
            typing_import = typing_import or stmt

    if typing_import is not None:
        typing_import.names.insert(0, ast.alias(name="Any"))
    else:
        body.insert(
            insert_at,
            ast.ImportFrom(module="typing", names=[ast.alias(name="Any")], level=0),
        )


def strip_artifacts(tree: ast.AST) -> ast.AST:
    """Remove stub-only artifacts from a generated .pyi AST.

    Non-``Module`` inputs are returned unchanged. Module inputs are mutated
    in place and returned for caller convenience.
    """
    if not isinstance(tree, ast.Module):
        return tree

    tree.body, defined = _strip_and_collect(tree.body)
    if _rewrite_annotations(tree.body, defined):
        _ensure_any_import(tree.body)
    return ast.fix_missing_locations(tree)
