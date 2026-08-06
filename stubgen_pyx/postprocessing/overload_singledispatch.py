"""Handling for singledispatch functions."""

from __future__ import annotations

import ast
import copy
import warnings
from dataclasses import dataclass

from .utils import dotted_name


class SingledispatchStubError(ValueError):
    """Raised when a @singledispatch group in the input is invalid Python.

    Emitted when attempting to generate a stub for source code that is actually invalid
    at import time (e.g. a bare ``@base.register()`` with no arguments). Subclasses
    ``ValueError``, so broad ``except ValueError`` handlers already catch it. Catch this
    class specifically only if you need to distinguish invalid singledispatch source
    from other input validation failures.
    """


def overload_singledispatch(tree: ast.AST) -> ast.AST:
    """Rewrite @singledispatch variants into @overload stubs."""
    if not isinstance(tree, ast.Module):
        return tree

    # Single linear pass over the module body:
    #   * record every @singledispatch base (by name and index)
    #   * classify every FunctionDef's decorators against known bases and
    #     partition its variants into that base's list.
    # A variant defined above its base is silently ignored.
    bases: list[_Base] = []
    bases_by_name: dict[str, _Base] = {}
    variants_by_base: dict[str, list[_Variant]] = {}
    unsupported_by_base: dict[str, list[str]] = {}
    for idx, stmt in enumerate(tree.body):
        if (
            is_decorated_base := isinstance(stmt, ast.FunctionDef)
            and any(_is_singledispatch(d) for d in stmt.decorator_list)
        ) or (
            # Earlier pipeline passes (e.g. trim_not_defined) may have rewritten a
            # lambda body to ``...``, and this pass drops the base body entirely either
            # way, so there's no need to check the shape of the first argument.
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and isinstance(stmt.value, ast.Call)
            and stmt.value.args
            and _is_singledispatch(stmt.value.func)
        ):
            base = _Base(stmt.name if is_decorated_base else stmt.targets[0].id, idx)
            bases.append(base)
            bases_by_name[base.name] = base
            variants_by_base[base.name] = []
            unsupported_by_base[base.name] = []
        elif isinstance(stmt, ast.FunctionDef):
            for decorator in stmt.decorator_list:
                _classify_decorator(
                    decorator,
                    stmt,
                    bases_by_name,
                    variants_by_base,
                    unsupported_by_base,
                )

    # Match each base with its collected variants and emit a replacement group of
    # statements. The base's index is used to replace it in the module body, and
    # the variant statements are dropped from the body by the caller.
    results = {
        base.index: _emit_group(
            base, variants_by_base[base.name], unsupported_by_base[base.name]
        )
        for base in bases
    }
    overwritten_registrations = {
        id(stmt)
        for result in results.values()
        if result is not None
        for stmt in result[1]
    }

    body: list[ast.stmt] = []
    needed: set[str] = set()
    for idx, stmt in enumerate(tree.body):
        if id(stmt) in overwritten_registrations:
            continue
        result = results.get(idx)
        if result is None:
            body.append(stmt)
            continue
        stmts, _consumed, needed_imports = result
        needed.update(needed_imports)
        body.extend(stmts)
    tree.body = body

    if needed:
        _add_typing_imports(tree, needed)
    return ast.fix_missing_locations(tree)


def _add_typing_imports(tree: ast.Module, needed: set[str]) -> None:
    """Ensure every name in ``needed`` is importable from ``typing`` in ``tree``.

    Appends to the first existing ``from typing import ...`` if one exists;
    otherwise inserts a fresh import after the last ``from __future__`` import
    (or at the top of the module if there is none).
    """
    # One traversal: find the first `from typing import ...` (where new names
    # get appended), the last `from __future__` import (fallback insert
    # position for a fresh import), and every name already imported from
    # typing across ALL typing imports (so we don't re-import it).
    first_typing: ast.ImportFrom | None = None
    insert_at = 0
    already_imported: set[str] = set()
    for idx, stmt in enumerate(tree.body):
        if isinstance(stmt, ast.ImportFrom) and stmt.module == "typing":
            if first_typing is None:
                first_typing = stmt
            already_imported.update(
                alias.name for alias in stmt.names if alias.asname is None
            )
        elif isinstance(stmt, ast.ImportFrom) and stmt.module == "__future__":
            insert_at = idx + 1
    # `sorted` gives deterministic emission order regardless of set hashing
    to_add = sorted(needed - already_imported)
    if not to_add:
        return
    if first_typing is not None:
        for name in to_add:
            first_typing.names.append(ast.alias(name=name))
    else:
        tree.body.insert(
            insert_at,
            ast.ImportFrom(
                module="typing",
                names=[ast.alias(name=name) for name in to_add],
                level=0,
            ),
        )


@dataclass
class _Base:
    name: str
    index: int


@dataclass
class _Variant:
    stmt: ast.FunctionDef
    type_key: str
    type_expr: ast.expr


def _classify_decorator(
    decorator: ast.expr,
    stmt: ast.FunctionDef,
    bases_by_name: dict[str, _Base],
    variants_by_base: dict[str, list[_Variant]],
    unsupported_by_base: dict[str, list[str]],
) -> None:
    """Classify one decorator against known bases and append to the right list.

    Decorator shapes we accept, cascading from most specific to least:
      * @base.register(T)              -> normal variant
      * @base.register()               -> raise (invalid at import)
      * @base.register(kw=...)         -> raise (invalid at import)
      * @base.register(T, ...extras)   -> unsupported_forms (pathological)
      * @base.register (bare)          -> variant, type from first arg annotation
    Anything else is an unrelated decorator on a nearby function; skip silently.
    """
    if isinstance(decorator, ast.Call):
        base = _match_register(decorator.func, bases_by_name)
        if base is None:
            return
        if not decorator.args and not decorator.keywords:
            raise SingledispatchStubError(
                f"@{base.name}.register() called with no arguments: "
            )
        if decorator.keywords:
            raise SingledispatchStubError(
                f"@{base.name}.register(...) takes no keyword arguments but "
                f"was called with {[kw.arg for kw in decorator.keywords]!r}"
            )
        if len(decorator.args) == 1:
            type_expr: ast.expr | None = decorator.args[0]
        else:
            # Legal-but-pathological: @base.register(T, extra_positional).
            # At runtime this registers T and treats the extras as the
            # handler, producing nonsense. Record for a better warning.
            unsupported_by_base[base.name].append(ast.unparse(decorator))
            return
    else:
        base = _match_register(decorator, bases_by_name)
        if base is None:
            return
        # Bare @foo.register attribute requires an annotated arg.
        args = stmt.args.posonlyargs + stmt.args.args
        type_expr = args[0].annotation if args else None
        if type_expr is None:
            raise SingledispatchStubError(
                f"@{base.name}.register on {stmt.name!r} has no type. Add a "
                "type annotation to the first parameter or pass the type as "
                f"@{base.name}.register(T)."
            )
    variants_by_base[base.name].append(
        _Variant(stmt, ast.unparse(type_expr), type_expr)
    )


def _match_register(node: ast.expr, bases_by_name: dict[str, _Base]) -> _Base | None:
    """Return the base whose `@<base>.register` this decorator names, or None."""
    if (
        isinstance(node, ast.Attribute)
        and node.attr == "register"
        and isinstance(node.value, ast.Name)
    ):
        return bases_by_name.get(node.value.id)
    return None


def _emit_group(
    base: _Base,
    variants: list[_Variant],
    unsupported_forms: list[str],
) -> tuple[list[ast.stmt], list[ast.FunctionDef], set[str]] | None:
    """Aggregate pre-collected variants for one base into replacement stmts.

    Args:
        base: The @singledispatch base whose group is being emitted.
        variants: Typed variants collected for this base, in source order.
        unsupported_forms: Unparsed ``@base.register(...)`` decorators that
            were syntactically legal but semantically pathological (e.g.
            extra positional args). Used only for the warning message.

    Returns:
        A ``(replacement_stmts, consumed_variant_stmts, needed_imports)``
        tuple when the group has usable variants, where
        ``replacement_stmts`` replace the base at its index,
        ``consumed_variant_stmts`` are dropped from the module body by the
        caller, and ``needed_imports`` are names to inject from ``typing``.
        Returns ``None`` when no usable variants were found; the base and
        its variant stmts are then left untouched in the module body, and a
        warning is emitted only if ``unsupported_forms`` is non-empty.

    Emission strategy is driven by the Python type-system spec
    (https://typing.python.org/en/latest/spec/overload.html):

    * Groups with >=2 typed variants emit one ``@overload`` per variant. The
      spec's rule that stub files must not include an overload implementation
      is honored: no trailing plain ``def`` is emitted.
    * Groups with exactly one typed variant collapse to a single plain ``def``
      (no ``@overload`` decorator). The spec forbids a lone ``@overload``: it
      requires at least two overload-decorated definitions per function, so a
      single ``@overload`` would be reported as an error by conforming type
      checkers. A plain signature carries the same information without
      violating that rule.

    Tradeoff for the single-variant collapse: the base @singledispatch
    function's fallback body is dropped. In real code that fallback is often
    just ``raise NotImplementedError``, but it can also be a valid default
    implementation for types not covered by any @register. Currently we
    cannot distinguish these two intents from source, and stubgen-pyx has no
    inline markup for the user to signal it. A future extension could add
    such markup (e.g. a comment or config directive) to preserve the base
    signature as a widening overload; for now, single-variant groups always
    collapse to the registered type.
    """
    if not variants:
        if unsupported_forms:
            warnings.warn(
                f"Cannot unify singledispatch group {base.name!r}: "
                f"unsupported @{base.name}.register(...) form(s): {unsupported_forms!r}"
            )
        return None

    # Match runtime singledispatch semantics: duplicate @register(T) in the same
    # group silently overwrites, so keep only the last variant per type key.
    unique_variants = list({variant.type_key: variant for variant in variants}.values())

    use_overload = len(unique_variants) > 1
    needed_imports = {"overload"} if use_overload else set()
    # Match prior behavior: `Any` is decided from raw pre-dedup variants.
    # A dropped duplicate's return annotation still contributes.
    if any(variant.stmt.returns is None for variant in variants):
        needed_imports.add("Any")
    consumed = [variant.stmt for variant in variants]

    stmts: list[ast.stmt] = []
    for variant in unique_variants:
        # Rewrite the variant into a signature for the base name:
        # * rename to base
        # * apply chosen decorator list (@overload or none)
        # * elide body to `...`
        # * default return annotation to `Any`
        # * force the first arg's annotation to the registered type
        stmt = copy.deepcopy(variant.stmt)
        stmt.name = base.name
        stmt.decorator_list = (
            [ast.Name(id="overload", ctx=ast.Load())] if use_overload else []
        )
        stmt.body = [ast.Expr(value=ast.Constant(...))]
        if stmt.returns is None:
            stmt.returns = ast.Name(id="Any", ctx=ast.Load())
        args = stmt.args.posonlyargs + stmt.args.args
        if args:
            args[0].annotation = copy.deepcopy(variant.type_expr)
        stmts.append(stmt)
    return stmts, consumed, needed_imports


def _is_singledispatch(node: ast.expr) -> bool:
    return dotted_name(node).split(".")[-1] == "singledispatch"
