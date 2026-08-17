"""Strip Cython-only stub artifacts and repair the annotations left dangling by that."""

from __future__ import annotations

import ast

from .utils import PUBLIC_BUILTIN_NAMES, root_name

# Names always considered resolvable when checking annotations.
_BUILTIN_NAMES = PUBLIC_BUILTIN_NAMES | {"None"}

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
        root = root_name(node)
        if root is not None and root not in self.defined:
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


def _is_blocked_import_reference(node: ast.expr) -> bool:
    root = root_name(node)
    return root is not None and root.startswith(_BLOCKED_IMPORT_PREFIXES)


def _rewrite_decorator_expression(
    node: ast.expr, rewriter: _AnnotationRewriter, *, in_decorator_root: bool = False
) -> ast.expr:
    if isinstance(node, ast.Call):
        return ast.Call(
            func=_rewrite_decorator_expression(
                node.func, rewriter, in_decorator_root=True
            ),
            args=[_rewrite_decorator_expression(arg, rewriter) for arg in node.args],
            keywords=[
                ast.keyword(
                    arg=kw.arg, value=_rewrite_decorator_expression(kw.value, rewriter)
                )
                for kw in node.keywords
            ],
        )
    if isinstance(node, ast.Attribute):
        if in_decorator_root:
            return node
        root = root_name(node)
        if root is not None and root not in rewriter.defined:
            return rewriter._any()
        return node
    if isinstance(node, ast.Name):
        if in_decorator_root:
            return node
        return node if node.id in rewriter.defined else rewriter._any()
    return rewriter.visit(node)


def _rewrite_decorators(
    node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
    rewriter: _AnnotationRewriter,
) -> None:
    cleaned: list[ast.expr] = []
    for dec in node.decorator_list:
        if _is_blocked_import_reference(dec):
            continue
        cleaned.append(_rewrite_decorator_expression(dec, rewriter))
    node.decorator_list = cleaned


def _rewrite_function_annotations(
    node: ast.FunctionDef | ast.AsyncFunctionDef, rewriter: _AnnotationRewriter
) -> None:
    _rewrite_decorators(node, rewriter)
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


def _class_defined_names(node: ast.ClassDef) -> set[str]:
    names: set[str] = set()
    for stmt in node.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(stmt.name)
        elif isinstance(stmt, ast.Assign):
            names.update(*(_target_names(target) for target in stmt.targets))
        elif isinstance(stmt, ast.AnnAssign):
            names.update(_target_names(stmt.target))
        elif isinstance(stmt, ast.Import):
            names.update(
                alias.asname or alias.name.split(".", 1)[0] for alias in stmt.names
            )
        elif isinstance(stmt, ast.ImportFrom):
            names.update(alias.asname or alias.name for alias in stmt.names)
    return names


def _is_typealias_annotation(node: ast.expr | None) -> bool:
    if isinstance(node, ast.Name):
        return node.id == "TypeAlias"
    if isinstance(node, ast.Attribute):
        return node.attr == "TypeAlias"
    return False


def _rewrite_annotations(body: list[ast.stmt], defined: set[str]) -> bool:
    changed = False
    for node in body:
        if isinstance(node, ast.ClassDef):
            class_defined = defined | _class_defined_names(node)
            rewriter = _AnnotationRewriter(class_defined)
            _rewrite_decorators(node, rewriter)
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    _rewrite_function_annotations(child, rewriter)
                elif isinstance(child, ast.AnnAssign):
                    child.annotation = rewriter.visit(child.annotation)
                    if child.value is not None and _is_typealias_annotation(
                        child.annotation
                    ):
                        child.value = rewriter.visit(child.value)
            changed |= rewriter.changed
        else:
            rewriter = _AnnotationRewriter(defined)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                _rewrite_function_annotations(node, rewriter)
            elif isinstance(node, ast.AnnAssign):
                node.annotation = rewriter.visit(node.annotation)
                if node.value is not None and _is_typealias_annotation(node.annotation):
                    node.value = rewriter.visit(node.value)
            changed |= rewriter.changed
    return changed


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
