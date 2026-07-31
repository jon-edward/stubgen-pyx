import ast


def strip_artifacts(tree: ast.AST) -> ast.AST:
    if not isinstance(tree, ast.Module):
        return ast.fix_missing_locations(tree)

    blocked = ("cython", "cpython")

    def filter_class_body(body: list[ast.stmt]) -> list[ast.stmt]:
        kept = []
        for stmt in body:
            if isinstance(stmt, ast.ClassDef):
                stmt.body = filter_class_body(stmt.body)
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

    kept = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            node.names = [n for n in node.names if not n.name.startswith(blocked)]
            if not node.names:
                continue
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.startswith(blocked):
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
            node.body = filter_class_body(node.body)
        kept.append(node)
    tree.body = kept
    return ast.fix_missing_locations(tree)
