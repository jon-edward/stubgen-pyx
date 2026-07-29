from __future__ import annotations

import ast
import copy
import warnings
from dataclasses import dataclass, field


class SingledispatchStubError(ValueError):
    """Raised when a @singledispatch group in the input is invalid Python.

    Subclasses ``ValueError``, so broad ``except ValueError`` handlers already
    catch it. Catch this class specifically only if you need to distinguish
    invalid singledispatch source from other input validation failures.

    These forms would fail at import time in a real Python interpreter:
    * ``@base.register()`` with no arguments (missing required ``cls``)
    * ``@base.register(cls, kw=...)`` with keyword arguments
    * bare ``@base.register`` on a function whose first parameter is unannotated

    Emitting a stub for source that cannot be imported would be misleading, so
    the pass fails loudly instead of trying to guess.
    """


def overload_singledispatch(tree: ast.AST) -> ast.AST:
    """Rewrite @singledispatch variants into @overload stubs."""
    unifier = _SingledispatchUnifier()
    tree = unifier.visit(tree)
    if unifier.emitted_overloads and not _has_typing_name(tree, "overload"):
        _insert_typing_import(tree, "overload")
    if unifier.emitted_any and not _has_typing_name(tree, "Any"):
        _insert_typing_import(tree, "Any")
    return ast.fix_missing_locations(tree)


@dataclass
class _Base:
    name: str
    stmt: ast.FunctionDef | ast.Assign
    index: int


@dataclass
class _Variant:
    stmt: ast.FunctionDef
    type_key: str
    type_expr: ast.expr


@dataclass
class _SingledispatchUnifier(ast.NodeTransformer):
    emitted_overloads: bool = False
    emitted_any: bool = False
    _skip: set[int] = field(default_factory=set, init=False)

    def visit_Module(self, node: ast.Module) -> ast.Module:
        bases = _find_bases(node.body)
        groups: dict[str, tuple[list[_Variant], list[str]]] = {
            base.name: _collect_group(base, node.body) for base in bases
        }
        replacements = {
            base.index: _unified_group(base, *groups[base.name]) for base in bases
        }
        base_by_index = {base.index: base for base in bases}
        self._skip = {
            id(variant.stmt)
            for base in bases
            if replacements[base.index] is not None
            for variant in groups[base.name][0]
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
                base = base_by_index[idx]
                self.emitted_any |= any(
                    variant.stmt.returns is None for variant in groups[base.name][0]
                )
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


def _collect_group(
    base: _Base, body: list[ast.stmt]
) -> tuple[list[_Variant], list[str]]:
    """Collect @register variants for a group.

    Returns:
        (variants, unsupported_forms)
        - variants: successfully-typed variants ready for @overload emission.
        - unsupported_forms: source of decorators we recognized as @base.register
          calls but chose not to raise on (multi positional args). Used only to
          produce a more informative 'no overloads' warning.

    Raises:
        SingledispatchStubError: on decorator forms that fail at import time in
        pure Python (empty ``@base.register()``, keyword-arg ``@base.register(
        cls, kw=...)``, bare ``@base.register`` on an unannotated function).
    """
    variants: list[_Variant] = []
    unsupported_forms: list[str] = []
    for stmt in body[base.index + 1 :]:
        if not isinstance(stmt, ast.FunctionDef):
            continue
        for decorator in stmt.decorator_list:
            _raise_if_invalid_register_call(base.name, decorator)
            type_expr = _register_type(base.name, decorator)
            if type_expr is not None:
                # Form A/C: decorator provides the type; always use it
                pass
            elif _is_register(base.name, decorator):
                # Form B: bare @foo.register attribute; require an annotated arg.
                type_expr = _first_arg_annotation(stmt)
                if type_expr is None:
                    raise SingledispatchStubError(
                        f"@{base.name}.register on {stmt.name!r} has no type: "
                        f"pure Python raises TypeError at import for bare "
                        f"@register on a function whose first parameter is "
                        f"unannotated. Add a type annotation to the first "
                        f"parameter or pass the type as @{base.name}.register(T)."
                    )
            elif _is_multi_arg_register_call(base.name, decorator):
                # Legal-but-pathological form: @base.register(T, extra_positional).
                # At runtime this registers T and treats the extras as the handler,
                # producing nonsense. We can't emit a sensible overload; record it.
                unsupported_forms.append(ast.unparse(decorator))
                continue
            else:
                # Unrelated decorator on a nearby function — skip silently.
                continue
            variants.append(_Variant(stmt, ast.unparse(type_expr), type_expr))
            break
    return variants, unsupported_forms


def _unified_group(
    base: _Base, variants: list[_Variant], unsupported_forms: list[str]
) -> list[ast.stmt] | None:
    if not variants:
        if unsupported_forms:
            warnings.warn(
                f"Cannot unify singledispatch group {base.name!r}: unsupported "
                f"@{base.name}.register(...) form(s): {unsupported_forms!r}"
            )
        else:
            msg = f"Cannot unify singledispatch group {base.name!r}: no overloads"
            warnings.warn(msg)
        return None

    # Match runtime singledispatch semantics: duplicate @register(T) in the same
    # group silently overwrites, so keep only the last variant per type key.
    deduped: dict[str, _Variant] = {}
    for variant in variants:
        deduped[variant.type_key] = variant

    overloads: list[ast.stmt] = [
        _overload_function(base.name, variant) for variant in deduped.values()
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
    if stmt.returns is None:
        stmt.returns = ast.Name(id="Any", ctx=ast.Load())
    args = stmt.args.posonlyargs + stmt.args.args
    if args:
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
    if (
        isinstance(node, ast.Call)
        and _is_register(base_name, node.func)
        and len(node.args) == 1
        and not node.keywords
    ):
        return node.args[0]
    return None


def _raise_if_invalid_register_call(base_name: str, node: ast.expr) -> None:
    """Raise if `node` is a syntactically-recognizable @base.register(...) call
    whose form fails at import time in pure Python."""
    if not isinstance(node, ast.Call) or not _is_register(base_name, node.func):
        return
    if not node.args and not node.keywords:
        raise SingledispatchStubError(
            f"@{base_name}.register() called with no arguments: pure Python "
            f"raises TypeError at import (missing required 'cls' argument)."
        )
    if node.keywords:
        raise SingledispatchStubError(
            f"@{base_name}.register(...) called with keyword arguments "
            f"{[kw.arg for kw in node.keywords]!r}: pure Python raises TypeError "
            f"at import (register() takes no keyword arguments)."
        )


def _is_multi_arg_register_call(base_name: str, node: ast.expr) -> bool:
    """True for @base.register(T, extra_positional) — legal at runtime but the
    extras are treated as the handler, which produces nonsense; we can't emit a
    sensible overload for this form."""
    return (
        isinstance(node, ast.Call)
        and _is_register(base_name, node.func)
        and len(node.args) > 1
        and not node.keywords
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
        if isinstance(stmt, ast.ImportFrom) and stmt.module == "typing":
            stmt.names.append(ast.alias(name=name))
            return
        if isinstance(stmt, ast.ImportFrom) and stmt.module == "__future__":
            insert_at = idx + 1
    tree.body.insert(
        insert_at,
        ast.ImportFrom(module="typing", names=[ast.alias(name=name)], level=0),
    )
