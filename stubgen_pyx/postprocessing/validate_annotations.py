from __future__ import annotations

import ast
import builtins

_BUILTIN_NAMES = {name for name in dir(builtins) if not name.startswith("_")} | {"None"}


def validate_annotations(tree: ast.AST) -> ast.AST:
    """Replace unresolved annotation names with Any."""
    if not isinstance(tree, ast.Module):
        return tree

    defined: set[str] = set(_BUILTIN_NAMES)
    for node in tree.body:
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

    replaced: list[bool] = [False]

    def _rewrite(expr: ast.expr) -> ast.expr:
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
                setattr(expr, f, _rewrite(v))
            elif isinstance(v, list):
                setattr(
                    expr, f, [_rewrite(e) if isinstance(e, ast.expr) else e for e in v]
                )
        return expr

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for dec in node.decorator_list:
                _rewrite(dec)
            for arg in node.args.posonlyargs + node.args.args + node.args.kwonlyargs:
                if arg.annotation:
                    arg.annotation = _rewrite(arg.annotation)
            for arg in (node.args.vararg, node.args.kwarg):
                if arg and arg.annotation:
                    arg.annotation = _rewrite(arg.annotation)
            if node.returns:
                node.returns = _rewrite(node.returns)
        elif isinstance(node, ast.AnnAssign):
            node.annotation = _rewrite(node.annotation)
        elif isinstance(node, ast.ClassDef):
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    for dec in child.decorator_list:
                        _rewrite(dec)
                    for arg in (
                        child.args.posonlyargs + child.args.args + child.args.kwonlyargs
                    ):
                        if arg.annotation:
                            arg.annotation = _rewrite(arg.annotation)
                    for arg in (child.args.vararg, child.args.kwarg):
                        if arg and arg.annotation:
                            arg.annotation = _rewrite(arg.annotation)
                    if child.returns:
                        child.returns = _rewrite(child.returns)
                elif isinstance(child, ast.AnnAssign):
                    child.annotation = _rewrite(child.annotation)

    if replaced[0]:
        last_future, first_typing = -1, None
        for i, stmt in enumerate(tree.body):
            if isinstance(stmt, ast.ImportFrom):
                if stmt.module == "__future__":
                    last_future = i
                elif stmt.module == "typing":
                    if any(a.name == "Any" and not a.asname for a in stmt.names):
                        return ast.fix_missing_locations(tree)
                    first_typing = first_typing if first_typing is not None else i
        if first_typing is not None:
            imp = tree.body[first_typing]
            assert isinstance(imp, ast.ImportFrom)
            imp.names.insert(0, ast.alias(name="Any"))
        else:
            tree.body.insert(
                last_future + 1,
                ast.ImportFrom(module="typing", names=[ast.alias(name="Any")], level=0),
            )

    return ast.fix_missing_locations(tree)
