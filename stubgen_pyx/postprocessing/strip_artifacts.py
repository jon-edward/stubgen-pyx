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

    Pass 1 removes Cython-only imports and other stub-only noise, which can
    leave annotations pointing at names that no longer exist in the generated
    module. This transformer rewrites those dangling references to ``Any``
    and records whether it changed anything so Pass 3 can import ``Any``.

    Dotted expressions are resolved by their leftmost root only: if
    ``some_module`` is still defined, ``some_module.Type`` is trusted; if the
    root is gone, the whole dotted expression becomes ``Any``.
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
        # collapses to `Any`. Do NOT descend into the attribute chain
        # field-by-field — rewriting only the value half of an
        # `ast.Attribute` produces nonsense like `Any.DEFAULT_SEED`.
        root: ast.expr = node
        while isinstance(root, ast.Attribute):
            root = root.value
        if isinstance(root, ast.Name) and root.id not in self.defined:
            return self._any()
        return node


def strip_artifacts(tree: ast.AST) -> ast.AST:
    """Remove stub-only artifacts from a generated .pyi AST.

    Runs three passes over a module:

    1. **Strip + collect**: drop Cython-only imports and module-level
       call/attribute assignments, except ``__all__`` and ``TypeVar(...)``
       assignments that carry useful stub semantics. Collect names that
       remain importable or assigned after stripping.
    2. **Rewrite annotations**: replace annotation and decorator references
       to stripped or otherwise undefined names with ``Any``.
    3. **Import ``Any`` if needed**: add ``Any`` to the first existing
       ``from typing import ...`` or insert a new typing import after
       ``__future__`` imports.

    Non-``Module`` inputs are returned unchanged. Module inputs are mutated
    in place and returned for caller convenience.
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
    # `rewriter.changed` flips to True whenever a name is substituted. Pass 3
    # uses that flag to decide whether a `typing.Any` import needs injecting.
    #
    # Walk module top level + class bodies uniformly. Nested classes rely on
    # their own ClassDef being visited separately; in practice, stubgen output
    # puts everything of interest at the top level of the module or class.
    rewriter = _AnnotationRewriter(defined)
    for node in tree.body:
        children = node.body if isinstance(node, ast.ClassDef) else (node,)
        for child in children:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                child.decorator_list = [
                    rewriter.visit(dec) for dec in child.decorator_list
                ]
                args = child.args
                for arg in (
                    *args.posonlyargs,
                    *args.args,
                    *args.kwonlyargs,
                    args.vararg,
                    args.kwarg,
                ):
                    if arg and arg.annotation:
                        arg.annotation = rewriter.visit(arg.annotation)
                if child.returns:
                    child.returns = rewriter.visit(child.returns)
            elif isinstance(child, ast.AnnAssign):
                # `X: SomeType = ...` at module or class scope.
                child.annotation = rewriter.visit(child.annotation)

    # ---- Pass 3: ensure `Any` is importable if we introduced it ----
    #
    # Skipped entirely when no rewrite happened — the common case where no
    # dangling names were found and the module didn't otherwise need Any.
    if rewriter.changed:
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
