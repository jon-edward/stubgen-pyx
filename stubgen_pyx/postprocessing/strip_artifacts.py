from __future__ import annotations

import ast
import builtins

# Names always considered resolvable when checking annotations. Includes every
# public builtin (`int`, `list`, `Exception`, ...) plus `None`, which shows up
# in annotations like `-> None` but is technically a keyword/constant rather
# than a builtin name.
_BUILTIN_NAMES = {name for name in dir(builtins) if not name.startswith("_")} | {"None"}

# Cython import roots. Any `import cython...` or `from cython... import ...`
# (and same for `cpython`) is a Cython-time construct that has no meaning in a
# .pyi stub and gets removed from the module.
_BLOCKED_IMPORT_PREFIXES = ("cython", "cpython")


def _filter_class_body(body: list[ast.stmt]) -> list[ast.stmt]:
    """Drop the ``__hash__ = None`` marker Cython auto-emits inside classes.

    Cython injects ``__hash__ = None`` into any class that defines ``__eq__``
    without an explicit ``__hash__``, mirroring Python's runtime rule that an
    ``__eq__``-defining class becomes unhashable. That statement is meaningful
    at runtime but pure noise inside a type stub, so we strip it out of every
    class body (recursively, for nested classes).

    The predicate is deliberately narrow: only ``__hash__ = None`` with a
    single target and a ``Constant(None)`` RHS is dropped. Anything else
    assigned to ``__hash__`` (a real hash implementation, or a call) is kept.
    """
    kept = []
    for stmt in body:
        # Recurse into nested classes first so their inner __hash__ = None also
        # gets stripped before we decide whether to keep the outer statement.
        if isinstance(stmt, ast.ClassDef):
            stmt.body = _filter_class_body(stmt.body)
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and getattr(stmt.targets[0], "id", None) == "__hash__"
            and isinstance(stmt.value, ast.Constant)
            and stmt.value.value is None
        ):
            # Skip appending: statement is dropped from the class body.
            continue
        kept.append(stmt)
    return kept


def _rewrite_unresolvable(
    expr: ast.expr, defined: set[str], replaced: list[bool]
) -> ast.expr:
    """Rewrite unresolvable names inside an annotation expression to ``Any``.

    After the strip pass removes Cython-only imports, some annotations end up
    referencing names that no longer exist in the module (e.g. an alias
    imported from ``cpython`` and used as a parameter type). Rather than emit
    a stub that mypy/pyright will reject, we rewrite every such dangling name
    to ``Any``.

    The check is a membership test against ``defined`` (which the caller has
    already populated from the post-strip module body plus builtins). Any name
    not in that set is considered dangling.

    Args:
        expr: Annotation expression node. Mutated in place for container
            fields; leaf ``Name``/``Attribute`` cases return a fresh node.
        defined: Names resolvable in the enclosing module.
        replaced: Single-element mutable holder. Set to ``[True]`` if any
            rewrite happened, so the caller knows to inject ``from typing
            import Any``. Using a list keeps the signature side-effect free
            from the caller's perspective (no return-value plumbing).

    Returns:
        Either the original ``expr`` (unchanged, possibly with mutated fields)
        or a new ``Name('Any')`` node that replaces the dangling reference.
    """
    # Leaf name: check directly against the defined set.
    if isinstance(expr, ast.Name):
        if expr.id not in defined:
            replaced[0] = True
            return ast.copy_location(ast.Name(id="Any", ctx=ast.Load()), expr)
        return expr

    # Dotted access like `some_module.Type`: only the leftmost root name has
    # to resolve. If `some_module` is defined, we trust the attribute chain
    # and leave it alone. If not, the whole chain is dangling and collapses
    # to `Any`. We do NOT descend into the attribute chain field-by-field
    # because rewriting only the value half of an `ast.Attribute` produces
    # nonsense like `Any.DEFAULT_SEED`.
    if isinstance(expr, ast.Attribute):
        root: ast.expr = expr
        while isinstance(root, ast.Attribute):
            root = root.value
        if isinstance(root, ast.Name) and root.id not in defined:
            replaced[0] = True
            return ast.copy_location(ast.Name(id="Any", ctx=ast.Load()), expr)

    # Everything else (Subscript, Call, Tuple, BinOp, ...) is a container
    # expression: recurse into every child that is itself an expression or a
    # list of expressions. This handles cases like `List[Missing]`,
    # `Union[A, B]`, `dict[str, Missing]`, `Callable[[Missing], int]`, etc.
    for f, v in ast.iter_fields(expr):
        if isinstance(v, ast.expr):
            setattr(expr, f, _rewrite_unresolvable(v, defined, replaced))
        elif isinstance(v, list):
            setattr(
                expr,
                f,
                [
                    _rewrite_unresolvable(e, defined, replaced)
                    if isinstance(e, ast.expr)
                    else e
                    for e in v
                ],
            )
    return expr


def strip_artifacts(tree: ast.AST) -> ast.AST:
    """Remove Cython codegen artifacts from a .pyi AST and repair references.

    Runs three passes over the module:

    1. **Strip + collect** (fused loop): drop Cython-only imports, module-level
       runtime-evaluated assignments, empty singledispatch stubs, and the
       ``__hash__ = None`` marker inside class bodies. In the same pass, build
       the ``defined`` set of names that survive into the final module.
    2. **Rewrite annotations**: any annotation referencing a name that isn't
       in ``defined`` gets collapsed to ``Any``.
    3. **Inject ``Any`` import** (only if step 2 rewrote anything): add
       ``Any`` to an existing ``from typing import ...`` if present,
       otherwise insert a fresh ``from typing import Any`` after the last
       ``__future__`` import.

    Non-``Module`` inputs are returned unchanged. The tree is mutated in
    place; the return value is the same object for caller convenience.
    """
    if not isinstance(tree, ast.Module):
        return tree

    # ---- Pass 1: strip artifacts and collect defined names in one walk ----
    #
    # `kept` accumulates the post-strip module body; `defined` accumulates the
    # names that survive into that body. Fusing the two lets Pass 2 see a
    # consistent view without a second traversal.
    kept = []
    defined: set[str] = set(_BUILTIN_NAMES)
    for node in tree.body:
        if isinstance(node, ast.Import):
            # Filter individual dotted names inside an `import a, cython, b`
            # rather than dropping the whole statement — sibling imports must
            # survive.
            node.names = [
                n for n in node.names if not n.name.startswith(_BLOCKED_IMPORT_PREFIXES)
            ]
            if not node.names:
                # Every name was cython/cpython; drop the whole statement.
                continue
        elif isinstance(node, ast.ImportFrom):
            # `from cython.foo import bar` — the whole statement goes,
            # because every imported name comes from the blocked root.
            if node.module and node.module.startswith(_BLOCKED_IMPORT_PREFIXES):
                continue
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            # Module-level `X = <runtime-evaluated>` assignments: Cython emits
            # things like `SIZE_TYPE = DataType(42)` or `SEED = mod.CONST`
            # that evaluate at import time. The value is meaningless in a
            # type stub, so the whole statement is dropped — unless it's one
            # of two well-known exceptions we deliberately preserve:
            #
            #   * __all__       — a real typing signal, even when built via
            #                     `__all__ = build_all()` or `exports.ALL`.
            #   * TypeVar(...)  — the value genuinely encodes typing info,
            #                     covering both `T = TypeVar(...)` and
            #                     `T = typing.TypeVar(...)` forms.
            is_all = getattr(node.targets[0], "id", None) == "__all__"
            func = node.value.func if isinstance(node.value, ast.Call) else None
            is_typevar = (
                getattr(func, "id", None) == "TypeVar"
                or getattr(func, "attr", None) == "TypeVar"
            )
            if (
                not is_all
                and isinstance(node.value, (ast.Call, ast.Attribute))
                and not is_typevar
            ):
                continue
        elif isinstance(node, ast.ClassDef):
            # Strip the `__hash__ = None` marker (recursively) from class
            # bodies. The ClassDef itself is always kept.
            node.body = _filter_class_body(node.body)
        kept.append(node)

        # Second half of the fused loop: collect the names this surviving node
        # contributes to the module namespace. Only reached for nodes we
        # decided to keep, so `defined` matches the post-strip view exactly.
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.add(node.name)
        elif isinstance(node, ast.Import):
            # `import a.b.c` binds `a` (or the asname if given). We take only
            # the leftmost dotted component because that's what Python
            # actually puts in the module namespace.
            for a in node.names:
                defined.add(a.asname or a.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom):
            # `from mod import x, y as z` binds `x` and `z`.
            for a in node.names:
                defined.add(a.asname or a.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            # Walk every LHS target, recursing into tuple/list unpacks so
            # patterns like `a, (b, c) = ...` bind all three names.
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            stack = list(targets)
            while stack:
                t = stack.pop()
                if isinstance(t, ast.Name):
                    defined.add(t.id)
                elif isinstance(t, (ast.Tuple, ast.List)):
                    stack.extend(t.elts)
    tree.body = kept

    # ---- Pass 2: rewrite dangling annotations to Any ----
    #
    # `replaced` is a single-element mutable flag: `_rewrite_unresolvable` sets it to True
    # whenever it substitutes a name. Pass 3 uses that flag to decide whether
    # a `typing.Any` import needs to be injected.
    replaced: list[bool] = [False]

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Decorators can reference dangling names too (e.g. a stripped
            # `@cython.cfunc`), so they get rewritten just like annotations.
            for dec in node.decorator_list:
                _rewrite_unresolvable(dec, defined, replaced)
            # Regular args (positional-only, positional-or-keyword, keyword-only).
            for arg in node.args.posonlyargs + node.args.args + node.args.kwonlyargs:
                if arg.annotation:
                    arg.annotation = _rewrite_unresolvable(
                        arg.annotation, defined, replaced
                    )
            # *args / **kwargs — separate slots that may or may not be present.
            for arg in (node.args.vararg, node.args.kwarg):
                if arg and arg.annotation:
                    arg.annotation = _rewrite_unresolvable(
                        arg.annotation, defined, replaced
                    )
            if node.returns:
                node.returns = _rewrite_unresolvable(node.returns, defined, replaced)
        elif isinstance(node, ast.AnnAssign):
            # Module-level annotated assignment: `X: SomeType = ...`.
            node.annotation = _rewrite_unresolvable(node.annotation, defined, replaced)
        elif isinstance(node, ast.ClassDef):
            # Methods and annotated class attributes. We only descend one
            # level; nested classes rely on their own ClassDef being visited
            # separately if they contain relevant annotations. In practice,
            # stubgen output puts everything of interest at the top level of
            # the class body.
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    for dec in child.decorator_list:
                        _rewrite_unresolvable(dec, defined, replaced)
                    for arg in (
                        child.args.posonlyargs + child.args.args + child.args.kwonlyargs
                    ):
                        if arg.annotation:
                            arg.annotation = _rewrite_unresolvable(
                                arg.annotation, defined, replaced
                            )
                    for arg in (child.args.vararg, child.args.kwarg):
                        if arg and arg.annotation:
                            arg.annotation = _rewrite_unresolvable(
                                arg.annotation, defined, replaced
                            )
                    if child.returns:
                        child.returns = _rewrite_unresolvable(
                            child.returns, defined, replaced
                        )
                elif isinstance(child, ast.AnnAssign):
                    child.annotation = _rewrite_unresolvable(
                        child.annotation, defined, replaced
                    )

    # ---- Pass 3: ensure `Any` is importable if we introduced it ----
    #
    # Skipped entirely when no rewrite happened — the common case where no
    # dangling names were found and the module didn't otherwise need Any.
    if replaced[0]:
        # Walk the top of the module once, tracking:
        #   * `insert_at`     — index just past the last `from __future__`
        #                       import (PEP 236: Any's import must land
        #                       after these; 0 if there are no futures).
        #   * `typing_import` — first `from typing import ...` we can slot
        #                       Any into.
        # The `break` short-circuits the whole pass when Any is already
        # imported bare (no alias) — nothing to do.
        insert_at = 0
        typing_import: ast.ImportFrom | None = None
        for i, stmt in enumerate(tree.body):
            if not isinstance(stmt, ast.ImportFrom):
                continue
            if stmt.module == "__future__":
                insert_at = i + 1
            elif stmt.module == "typing":
                if any(a.name == "Any" and not a.asname for a in stmt.names):
                    break
                typing_import = typing_import or stmt
        else:
            if typing_import is not None:
                # Prefer piggy-backing onto an existing `from typing import ...`
                # by inserting `Any` at the front of its names list.
                typing_import.names.insert(0, ast.alias(name="Any"))
            else:
                # No typing import at all — synthesize one right after the
                # future imports (or at position 0 if there are none).
                tree.body.insert(
                    insert_at,
                    ast.ImportFrom(
                        module="typing", names=[ast.alias(name="Any")], level=0
                    ),
                )

    # Restore line/col info on any freshly synthesized nodes (the `Any` imports
    # and the `Name('Any')` replacements). Cheap and prevents downstream tools
    # from choking on missing locations.
    return ast.fix_missing_locations(tree)
