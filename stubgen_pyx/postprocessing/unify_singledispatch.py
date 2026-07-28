from __future__ import annotations

import ast
import copy
import warnings
from dataclasses import dataclass, field


def unify_singledispatch(tree: ast.AST) -> ast.AST:
    unifier = _SingledispatchUnifier()
    tree = unifier.visit(tree)
    if unifier.emitted_overloads and not _has_typing_name(tree, "overload"):
        _insert_typing_import(tree, "overload")
    return ast.fix_missing_locations(tree)


@dataclass
class _Base:
    name: str
    stmt: ast.FunctionDef | ast.Assign
    index: int


@dataclass
class _Variant:
    stmt: ast.FunctionDef
    type_key: str | None
    type_expr: ast.expr | None


@dataclass
class _SingledispatchUnifier(ast.NodeTransformer):
    emitted_overloads: bool = False
    _skip: set[int] = field(default_factory=set, init=False)

    def visit_Module(self, node: ast.Module) -> ast.Module:
        bases = _find_bases(node.body)
        groups = {base.name: _collect_group(base, node.body) for base in bases}
        replacements = {
            base.index: _unified_group(base, groups[base.name]) for base in bases
        }
        self._skip = {
            id(variant.stmt)
            for base in bases
            if replacements[base.index] is not None
            for variant in groups[base.name]
        }

        body: list[ast.stmt] = []
        for idx, stmt in enumerate(node.body):
            if id(stmt) in self._skip:
                continue
            replacement = replacements.get(idx)
            if replacement is None:
                body.append(stmt)
            else:
                self.emitted_overloads = True
                body.extend(replacement)
        node.body = body
        return node


def _find_bases(body: list[ast.stmt]) -> list[_Base]:
    bases = []
    for idx, stmt in enumerate(body):
        if isinstance(stmt, ast.FunctionDef) and any(
            _is_singledispatch(decorator) for decorator in stmt.decorator_list
        ):
            bases.append(_Base(stmt.name, stmt, idx))
        elif isinstance(stmt, ast.Assign) and _is_singledispatch_assignment(stmt):
            target = stmt.targets[0]
            assert isinstance(target, ast.Name)
            bases.append(_Base(target.id, stmt, idx))
    return bases


def _collect_group(base: _Base, body: list[ast.stmt]) -> list[_Variant]:
    variants = []
    for stmt in body[base.index + 1 :]:
        if not isinstance(stmt, ast.FunctionDef):
            continue
        for decorator in stmt.decorator_list:
            type_expr = _register_type(base.name, decorator)
            if type_expr is not None or _is_bare_register(base.name, decorator):
                type_expr = type_expr or _first_arg_annotation(stmt)
                variants.append(
                    _Variant(
                        stmt, ast.unparse(type_expr) if type_expr else None, type_expr
                    )
                )
                break
    return variants


def _unified_group(base: _Base, variants: list[_Variant]) -> list[ast.stmt] | None:
    if not variants:
        warnings.warn(f"Cannot unify singledispatch group {base.name!r}: no overloads")
        return None
    seen: set[str] = set()
    for variant in variants:
        if variant.type_key is None:
            warnings.warn(
                f"Cannot unify singledispatch group {base.name!r}: untyped overload"
            )
            return None
        if variant.type_key in seen:
            warnings.warn(
                f"Cannot unify singledispatch group {base.name!r}: duplicate "
                f"registration for {variant.type_key}"
            )
            return None
        seen.add(variant.type_key)

    overloads: list[ast.stmt] = [
        _overload_function(base.name, variant) for variant in variants
    ]
    fallback = _fallback_function(base)
    if fallback is not None:
        overloads.append(fallback)
    return overloads


def _overload_function(name: str, variant: _Variant) -> ast.FunctionDef:
    stmt = copy.deepcopy(variant.stmt)
    stmt.name = name
    stmt.decorator_list = [ast.Name(id="overload", ctx=ast.Load())]
    stmt.body = [ast.Expr(value=ast.Constant(...))]
    args = stmt.args.posonlyargs + stmt.args.args
    if args and args[0].annotation is None and variant.type_expr is not None:
        args[0].annotation = copy.deepcopy(variant.type_expr)
    return stmt


def _fallback_function(base: _Base) -> ast.FunctionDef | None:
    if isinstance(base.stmt, ast.FunctionDef):
        if _is_ellipsis_body(base.stmt):
            return None
        stmt = copy.deepcopy(base.stmt)
        stmt.decorator_list = [
            decorator
            for decorator in stmt.decorator_list
            if not _is_singledispatch(decorator)
        ]
        return stmt

    value = base.stmt.value
    assert isinstance(value, ast.Call)
    first_arg = value.args[0]
    if not isinstance(first_arg, ast.Lambda):
        return None
    return ast.FunctionDef(
        name=base.name,
        args=copy.deepcopy(first_arg.args),
        body=[ast.Expr(value=ast.Constant(...))],
        decorator_list=[],
        returns=None,
        type_comment=None,
    )


def _is_singledispatch_assignment(stmt: ast.Assign) -> bool:
    return (
        len(stmt.targets) == 1
        and isinstance(stmt.targets[0], ast.Name)
        and isinstance(stmt.value, ast.Call)
        and bool(stmt.value.args)
        and isinstance(stmt.value.args[0], ast.Lambda)
        and _is_singledispatch(stmt.value.func)
    )


def _is_singledispatch(node: ast.expr) -> bool:
    return _dotted_name(node).endswith("singledispatch")


def _register_type(base_name: str, node: ast.expr) -> ast.expr | None:
    if isinstance(node, ast.Call) and _is_register(base_name, node.func):
        if node.args:
            return node.args[0]
        if node.keywords:
            return node.keywords[0].value
    return None


def _is_bare_register(base_name: str, node: ast.expr) -> bool:
    return _is_register(base_name, node) or (
        isinstance(node, ast.Call) and _is_register(base_name, node.func)
    )


def _is_register(base_name: str, node: ast.expr) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "register"
        and isinstance(node.value, ast.Name)
        and node.value.id == base_name
    )


def _first_arg_annotation(stmt: ast.FunctionDef) -> ast.expr | None:
    args = stmt.args.posonlyargs + stmt.args.args
    if not args:
        return None
    return args[0].annotation


def _is_ellipsis_body(stmt: ast.FunctionDef) -> bool:
    return (
        len(stmt.body) == 1
        and isinstance(stmt.body[0], ast.Expr)
        and isinstance(stmt.body[0].value, ast.Constant)
        and stmt.body[0].value.value is Ellipsis
    )


def _dotted_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


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
