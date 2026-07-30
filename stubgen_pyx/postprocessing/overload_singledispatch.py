from __future__ import annotations

import ast
import copy
import warnings
from dataclasses import dataclass


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
    if not isinstance(tree, ast.Module):
        return tree

    # Locate @singledispatch bases: decorated `def` or `name = functools.singledispatch(...)`.
    # The assignment form deliberately does not check the shape of the first
    # argument: earlier pipeline passes (e.g. trim_not_defined) may have
    # rewritten a lambda body to ``...``, and this pass drops the base body
    # entirely either way.
    bases: list[_Base] = []
    for idx, stmt in enumerate(tree.body):
        if isinstance(stmt, ast.FunctionDef) and any(
            _is_singledispatch(d) for d in stmt.decorator_list
        ):
            bases.append(_Base(stmt.name, stmt, idx))
        elif (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and isinstance(stmt.value, ast.Call)
            and stmt.value.args
            and _is_singledispatch(stmt.value.func)
        ):
            bases.append(_Base(stmt.targets[0].id, stmt, idx))

    results = {base.index: _process_group(base, tree.body) for base in bases}
    skip = {
        id(stmt)
        for result in results.values()
        if result is not None
        for stmt in result[1]
    }

    body: list[ast.stmt] = []
    emitted_overloads = False
    emitted_any = False
    for idx, stmt in enumerate(tree.body):
        if id(stmt) in skip:
            continue
        result = results.get(idx)
        if result is None:
            body.append(stmt)
            continue
        stmts, _consumed, used_overload, uses_any = result
        emitted_overloads |= used_overload
        emitted_any |= uses_any
        body.extend(stmts)
    tree.body = body

    needed = set()
    if emitted_overloads:
        needed.add("overload")
    if emitted_any:
        needed.add("Any")
    if needed:
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
        # `sorted` gives deterministic emission order regardless of set hashing.
        to_add = sorted(needed - already_imported)
        if to_add:
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


def _process_group(
    base: _Base, body: list[ast.stmt]
) -> tuple[list[ast.stmt], list[ast.FunctionDef], bool, bool] | None:
    """Collect @register variants for a group and produce replacement stmts.

    Returns:
        On success:
            (replacement_stmts, consumed_variant_stmts, used_overload, uses_any)
            - replacement_stmts: statements that replace the base at its index.
            - consumed_variant_stmts: the FunctionDefs for the variants; the
              caller drops these from the module body.
            - used_overload: whether the emitted stmts carry @overload (drives
              typing import injection).
            - uses_any: whether any variant has no return annotation (drives
              typing import injection for `Any`).
        None:
            No usable variants were found. Emit a warning and leave the base
            and its variant stmts untouched in the module body.

    Raises:
        SingledispatchStubError: on decorator forms that fail at import time in
        pure Python (empty ``@base.register()``, keyword-arg ``@base.register(
        cls, kw=...)``, bare ``@base.register`` on an unannotated function).

    Emission strategy is driven by the Python type-system spec
    (https://typing.python.org/en/latest/spec/overload.html):

    * Groups with >=2 typed variants emit one ``@overload`` per variant. The
      spec's rule that stub files must not include an overload implementation
      is honored: no trailing plain ``def`` is emitted.
    * Groups with exactly one typed variant collapse to a single plain ``def``
      (no ``@overload`` decorator). The spec forbids a lone ``@overload``: it
      requires at least two overload-decorated definitions per function, so a
      single ``@overload`` would be reported as an error by type checkers.
      A plain signature carries the same information without violating that
      rule.

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
    variants: list[_Variant] = []
    unsupported_forms: list[str] = []
    for stmt in body[base.index + 1 :]:
        if not isinstance(stmt, ast.FunctionDef):
            continue
        for decorator in stmt.decorator_list:
            # Classify the decorator against @base.register in one cascade.
            # Anything not shaped like @base.register(...) is an unrelated
            # decorator on a nearby function and skipped silently.
            if isinstance(decorator, ast.Call) and _is_register(
                base.name, decorator.func
            ):
                if not decorator.args and not decorator.keywords:
                    # @base.register() — pure Python raises TypeError at import.
                    raise SingledispatchStubError(
                        f"@{base.name}.register() called with no arguments: "
                        f"pure Python raises TypeError at import (missing "
                        f"required 'cls' argument)."
                    )
                if decorator.keywords:
                    # @base.register(kw=...) — pure Python raises TypeError.
                    raise SingledispatchStubError(
                        f"@{base.name}.register(...) called with keyword "
                        f"arguments {[kw.arg for kw in decorator.keywords]!r}: "
                        f"pure Python raises TypeError at import (register() "
                        f"takes no keyword arguments)."
                    )
                if len(decorator.args) == 1:
                    # Form A/C: @base.register(T) — decorator provides the type.
                    type_expr = decorator.args[0]
                else:
                    # Legal-but-pathological: @base.register(T, extra_positional).
                    # At runtime this registers T and treats the extras as the
                    # handler, producing nonsense. Record for a better warning.
                    unsupported_forms.append(ast.unparse(decorator))
                    continue
            elif _is_register(base.name, decorator):
                # Form B: bare @foo.register attribute; require an annotated arg.
                args = stmt.args.posonlyargs + stmt.args.args
                type_expr = args[0].annotation if args else None
                if type_expr is None:
                    raise SingledispatchStubError(
                        f"@{base.name}.register on {stmt.name!r} has no type: "
                        f"pure Python raises TypeError at import for bare "
                        f"@register on a function whose first parameter is "
                        f"unannotated. Add a type annotation to the first "
                        f"parameter or pass the type as @{base.name}.register(T)."
                    )
            else:
                # Unrelated decorator on a nearby function — skip silently.
                continue
            variants.append(_Variant(stmt, ast.unparse(type_expr), type_expr))
            break

    if not variants:
        if unsupported_forms:
            warnings.warn(
                f"Cannot unify singledispatch group {base.name!r}: unsupported "
                f"@{base.name}.register(...) form(s): {unsupported_forms!r}"
            )
        else:
            warnings.warn(
                f"Cannot unify singledispatch group {base.name!r}: no overloads"
            )
        return None

    # Match runtime singledispatch semantics: duplicate @register(T) in the same
    # group silently overwrites, so keep only the last variant per type key.
    deduped: dict[str, _Variant] = {}
    for variant in variants:
        deduped[variant.type_key] = variant
    unique_variants = list(deduped.values())

    # Match prior behavior: `uses_any` is computed over the raw pre-dedup
    # variants. A dropped duplicate's return annotation still contributes.
    uses_any = any(variant.stmt.returns is None for variant in variants)
    consumed = [variant.stmt for variant in variants]

    if len(unique_variants) == 1:
        stmts = [_variant_function(base.name, unique_variants[0], overload=False)]
        return stmts, consumed, False, uses_any
    stmts = [
        _variant_function(base.name, variant, overload=True)
        for variant in unique_variants
    ]
    return stmts, consumed, True, uses_any


def _variant_function(
    name: str, variant: _Variant, *, overload: bool
) -> ast.FunctionDef:
    """Build one signature for a group.

    ``overload=True`` decorates with ``@overload`` (>=2-variant groups);
    ``overload=False`` emits a plain ``def`` (single-variant collapse — see
    ``_process_group`` for the spec rationale).
    """
    stmt = copy.deepcopy(variant.stmt)
    stmt.name = name
    stmt.decorator_list = [ast.Name(id="overload", ctx=ast.Load())] if overload else []
    stmt.body = [ast.Expr(value=ast.Constant(...))]
    if stmt.returns is None:
        stmt.returns = ast.Name(id="Any", ctx=ast.Load())
    args = stmt.args.posonlyargs + stmt.args.args
    if args:
        args[0].annotation = copy.deepcopy(variant.type_expr)
    return stmt


def _is_singledispatch(node: ast.expr) -> bool:
    return _dotted_name(node).endswith("singledispatch")


def _is_register(base_name: str, node: ast.expr) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "register"
        and isinstance(node.value, ast.Name)
        and node.value.id == base_name
    )


def _dotted_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""
