from __future__ import annotations

import ast
import builtins

_BUILTIN_NAMES = {name for name in dir(builtins) if not name.startswith("_")} | {"None"}
_BLOCKED_IMPORT_PREFIXES = ("cython", "cpython")


def _filter_class_body(body: list[ast.stmt]) -> list[ast.stmt]:
    kept = []
    for stmt in body:
        if isinstance(stmt, ast.ClassDef):
            stmt.body = _filter_class_body(stmt.body)
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
            target = stmt.targets[0]
            if (
                isinstance(target, ast.Name)
                and target.id == "__hash__"
                and isinstance(stmt.value, ast.Constant)
                and stmt.value.value is None
            ):
                continue
        kept.append(stmt)
    return kept


def _rewrite(expr: ast.expr, defined: set[str], replaced: list[bool]) -> ast.expr:
    if isinstance(expr, ast.Name):
        if expr.id not in defined:
            replaced[0] = True
            return ast.copy_location(ast.Name(id="Any", ctx=ast.Load()), expr)
        return expr
    if isinstance(expr, ast.Attribute):
        root: ast.expr = expr
        while isinstance(root, ast.Attribute):
            root = root.value
        if isinstance(root, ast.Name) and root.id not in defined:
            replaced[0] = True
            return ast.copy_location(ast.Name(id="Any", ctx=ast.Load()), expr)
    for f, v in ast.iter_fields(expr):
        if isinstance(v, ast.expr):
            setattr(expr, f, _rewrite(v, defined, replaced))
        elif isinstance(v, list):
            setattr(
                expr,
                f,
                [
                    _rewrite(e, defined, replaced) if isinstance(e, ast.expr) else e
                    for e in v
                ],
            )
    return expr


def strip_artifacts(tree: ast.AST) -> ast.AST:
    if not isinstance(tree, ast.Module):
        return tree

    kept = []
    defined: set[str] = set(_BUILTIN_NAMES)
    for node in tree.body:
        if isinstance(node, ast.Import):
            node.names = [
                n for n in node.names if not n.name.startswith(_BLOCKED_IMPORT_PREFIXES)
            ]
            if not node.names:
                continue
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.startswith(_BLOCKED_IMPORT_PREFIXES):
                continue
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            is_all = isinstance(target, ast.Name) and target.id == "__all__"
            is_typevar = isinstance(node.value, ast.Call) and (
                isinstance(node.value.func, ast.Name)
                and node.value.func.id == "TypeVar"
                or isinstance(node.value.func, ast.Attribute)
                and node.value.func.attr == "TypeVar"
            )
            if (
                not is_all
                and isinstance(node.value, (ast.Call, ast.Attribute))
                and not is_typevar
            ):
                continue
        elif isinstance(node, ast.ClassDef):
            node.body = _filter_class_body(node.body)
        kept.append(node)

        # Collect module-level names as we go. Runs on the post-strip view
        # because dropped nodes hit `continue` above.
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.add(node.name)
        elif isinstance(node, ast.Import):
            for a in node.names:
                defined.add(a.asname or a.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom):
            for a in node.names:
                defined.add(a.asname or a.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            stack = list(targets)
            while stack:
                t = stack.pop()
                if isinstance(t, ast.Name):
                    defined.add(t.id)
                elif isinstance(t, (ast.Tuple, ast.List)):
                    stack.extend(t.elts)
    tree.body = kept

    replaced: list[bool] = [False]

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for dec in node.decorator_list:
                _rewrite(dec, defined, replaced)
            for arg in node.args.posonlyargs + node.args.args + node.args.kwonlyargs:
                if arg.annotation:
                    arg.annotation = _rewrite(arg.annotation, defined, replaced)
            for arg in (node.args.vararg, node.args.kwarg):
                if arg and arg.annotation:
                    arg.annotation = _rewrite(arg.annotation, defined, replaced)
            if node.returns:
                node.returns = _rewrite(node.returns, defined, replaced)
        elif isinstance(node, ast.AnnAssign):
            node.annotation = _rewrite(node.annotation, defined, replaced)
        elif isinstance(node, ast.ClassDef):
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    for dec in child.decorator_list:
                        _rewrite(dec, defined, replaced)
                    for arg in (
                        child.args.posonlyargs + child.args.args + child.args.kwonlyargs
                    ):
                        if arg.annotation:
                            arg.annotation = _rewrite(arg.annotation, defined, replaced)
                    for arg in (child.args.vararg, child.args.kwarg):
                        if arg and arg.annotation:
                            arg.annotation = _rewrite(arg.annotation, defined, replaced)
                    if child.returns:
                        child.returns = _rewrite(child.returns, defined, replaced)
                elif isinstance(child, ast.AnnAssign):
                    child.annotation = _rewrite(child.annotation, defined, replaced)

    if replaced[0]:
        last_future, first_typing = -1, None
        has_any = False
        for i, stmt in enumerate(tree.body):
            if isinstance(stmt, ast.ImportFrom):
                if stmt.module == "__future__":
                    last_future = i
                elif stmt.module == "typing":
                    if any(a.name == "Any" and not a.asname for a in stmt.names):
                        has_any = True
                        break
                    first_typing = first_typing if first_typing is not None else i
        if not has_any:
            if first_typing is not None:
                imp = tree.body[first_typing]
                assert isinstance(imp, ast.ImportFrom)
                imp.names.insert(0, ast.alias(name="Any"))
            else:
                tree.body.insert(
                    last_future + 1,
                    ast.ImportFrom(
                        module="typing", names=[ast.alias(name="Any")], level=0
                    ),
                )

    return ast.fix_missing_locations(tree)
