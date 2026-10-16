"""``ctypedef`` alias resolution: collecting the ``ctypedef`` aliases
visible in a scope, substituting them into already-rendered annotation
strings, and the whole-batch "is this alias still referenced anywhere"
cross-check used when pruning an alias that turns out to be unused.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterator

from ..analysis.visitor import ScopeVisitor
from ..models.pyi_elements import (
    PyiAssignment,
    PyiFunction,
    PyiModule,
    PyiScope,
    PyiSignature,
)
from .pyrex_types import render_pyrex_type


def _is_plain_enum_entry(entry) -> bool:
    """Whether `entry` is a ``cdef enum`` with no Python-level wrapper.

    A ``cpdef enum`` (`entry.create_wrapper`) is a real ``IntEnum`` at
    runtime and is never treated as an alias.
    """
    t = entry.type
    return (
        entry.is_type
        and (getattr(t, "is_enum", False) or getattr(t, "is_cpp_enum", False))
        and not getattr(entry, "create_wrapper", False)
    )


def ctypedef_alias_map(visitor: ScopeVisitor) -> dict[str, str]:
    """Collect the ``ctypedef`` aliases and plain ``cdef enum`` names
    visible in this scope, resolved to their underlying type.

    Only used when `resolve_ctypedef_aliases` (see `StubgenPyxConfig`)
    is on. Walks `scope.entries` for `entry.is_type and entry.type.
    is_typedef` (`PyrexTypes.CTypedefType`) and resolves each through
    `render_pyrex_type`, which already unwraps `.typedef_base_type`
    however many levels deep are needed (``ctypedef double MyFloat``
    then ``ctypedef MyFloat MyFloat2`` both resolve straight to
    ``"float"``) -- no separate unwrapping needed here. Skips a
    `ctypedef` whose underlying type `render_pyrex_type` can't
    render at all (returns `None`): better to keep referencing the
    alias name than to substitute in nothing/a broken result.

    A ``cdef enum`` (no ``cpdef``) has no Python-level binding, so its
    name maps to ``"int"``. A ``ctypedef`` whose underlying type
    mentions such an enum resolves through it as well
    (``ctypedef Color ColorAlias`` -> ``"int"``). A ``cpdef enum``
    keeps its name: it is an ``IntEnum`` class at runtime.

    A `ctypedef`/``cdef enum`` has no runtime component, so -- same
    architecture documented throughout `_convert_declared_entries`/
    `convert_fused_types` -- it never survives as a node in
    `body.stats` once real declaration analysis runs; `scope.entries`
    is the only place left to find one.
    """
    scope = getattr(visitor.node, "scope", None)
    if scope is None:
        return {}
    enum_aliases = {
        name: "int"
        for name, entry in scope.entries.items()
        if _is_plain_enum_entry(entry)
    }
    aliases: dict[str, str] = dict(enum_aliases)
    for name, entry in scope.entries.items():
        if not (entry.is_type and getattr(entry.type, "is_typedef", False)):
            continue
        if _typedef_of_plain_enum(entry.type):
            aliases[name] = "int"
            continue
        resolved = render_pyrex_type(entry.type)
        if resolved is None:
            continue
        resolved = _substitute_ctypedef_aliases(resolved, enum_aliases) or resolved
        if resolved != name:
            aliases[name] = resolved
    return aliases


def _typedef_of_plain_enum(t) -> bool:
    """Whether the typedef `t` ultimately aliases a ``cdef enum`` (no
    ``cpdef``). Works from the type itself, so it also holds for a typedef
    cimported without the enum it aliases."""
    while getattr(t, "is_typedef", False):
        t = t.typedef_base_type
    if not (getattr(t, "is_enum", False) or getattr(t, "is_cpp_enum", False)):
        return False
    return not getattr(getattr(t, "entry", None), "create_wrapper", False)


def apply_ctypedef_aliases(
    signature: PyiSignature, ctypedef_aliases: dict[str, str] | None
) -> None:
    """Substitute a resolved `ctypedef` alias into every argument and
    the return annotation of `signature`, in place.

    A no-op when `ctypedef_aliases` is empty/`None` -- always safe to
    call unconditionally (see `_substitute_ctypedef_aliases`).
    """
    if not ctypedef_aliases:
        return
    for pyi_arg in signature.args:
        pyi_arg.annotation = _substitute_ctypedef_aliases(
            pyi_arg.annotation, ctypedef_aliases
        )
    signature.return_type = _substitute_ctypedef_aliases(
        signature.return_type, ctypedef_aliases
    )


def _substitute_ctypedef_aliases(
    annotation: str | None, alias_map: dict[str, str]
) -> str | None:
    """Replace any `ctypedef` alias name appearing in `annotation` with
    its resolved type, per `alias_map` (name -> resolved type string,
    built by `ctypedef_alias_map`).

    Whole-word substitution only (``\bNAME\b``), so `MyFloat` doesn't
    also clobber `MyFloatValue`. A match is also rejected when
    immediately preceded by a `.` (``(?<!\\.)``) -- otherwise a
    qualified attribute access on something completely unrelated
    (``other_mod.MyFloat``, some other module's own, different
    `MyFloat` attribute -- nothing to do with this file's local
    `ctypedef`) still has a word boundary right before `MyFloat` and
    would get "substituted" into outright invalid nonsense
    (``other_mod.float`` -- not just an imprecise match, a corrupted
    one, since `other_mod` has no `.float` attribute at all).
    Resolving the annotation string as real Python (rather than
    substituting on the text directly) would be needed to also catch a
    *local* alias genuinely shadowed by an import (``from x import
    MyFloat as MyFloat`` then used bare) -- considered out of scope; a
    bare name colliding with a real ctypedef alias's own name in the
    same file is far more likely to
    be an actual reference to it than an unrelated shadow.

    A no-op (returns `annotation` unchanged) whenever `alias_map` is
    empty, `annotation` is `None`, or nothing in the map actually
    appears in `annotation` -- always safe to call unconditionally
    rather than checking first.
    """
    if not alias_map or annotation is None:
        return annotation
    # Longest names first, so a longer alias that happens to *contain* a
    # shorter one as a substring (unusual, but not impossible) doesn't
    # get shadowed by the shorter one matching first in the alternation.
    alternation = "|".join(
        re.escape(name) for name in sorted(alias_map, key=len, reverse=True)
    )
    return re.sub(
        rf"(?<!\.)\b(?:{alternation})\b",
        lambda m: alias_map[m.group(0)],
        annotation,
    )


def _scope_functions(scope: PyiScope) -> Iterator[PyiFunction]:
    """Yield all functions in the scope, recursively descending into nested classes."""
    yield from scope.functions
    for class_ in scope.classes:
        yield from _scope_functions(class_.scope)


def _scope_assignments(scope: PyiScope) -> Iterator[PyiAssignment]:
    """Yield all assignments in the scope, recursively descending into nested classes."""
    yield from scope.assignments
    for class_ in scope.classes:
        yield from _scope_assignments(class_.scope)


def remove_assignment_from_module(module: PyiModule, assignment: PyiAssignment) -> bool:
    """Remove `assignment` from wherever it currently lives in `module`'s
    scope tree -- its own top-level assignments or enum entries (a
    ``cdef enum``'s ``TypeAlias`` lives in the latter), or a nested
    class's. Returns whether it was found and removed.

    Matches by identity first, then by equality. Both the `.pyx`'s and its
    companion `.pxd`'s conversion emit a ``cdef enum``'s alias (they share
    one module scope), `merge_pxd_into_module` keeps only one of the two
    copies, and the copy registered for pruning may be the discarded one.

    Deliberately searches the final module rather than reusing a list
    reference captured earlier (at the point a `ctypedef` alias's
    containing scope was first converted, inside `Converter.convert_scope`):
    a companion `.pxd`'s assignments get merged into the `.pyx`'s own
    module afterward (`merge_pxd_into_module`), which builds a new list
    rather than mutating the `.pxd`'s own scope's list in place -- so a
    `.remove()` against that earlier reference would silently no-op on a
    list nothing renders from anymore.
    """
    for same in (
        lambda existing: existing is assignment,
        lambda existing: existing == assignment,
    ):
        for scope in _iter_scopes(module.scope):
            for statements in (scope.assignments, scope.enums):
                for i, existing in enumerate(statements):
                    if same(existing):
                        del statements[i]
                        return True
    return False


def _iter_scopes(scope: PyiScope) -> Iterator[PyiScope]:
    """Yield `scope` and every nested class scope beneath it, recursively."""
    yield scope
    for class_ in scope.classes:
        yield from _iter_scopes(class_.scope)


def pyi_module_uses_name(
    module: PyiModule, name: str, imports_are_trimmed: bool = False
) -> bool:
    """Whether `name` appears anywhere in `module` -- its own imports, or
    recursively through its scope (function signatures and attribute
    assignments, including nested classes).

    With `imports_are_trimmed`, an import only counts if it survives
    `trim_imports` (see `_import_uses_name`): one that brings `name` in
    but is never used is about to be removed, so it no longer needs
    `name` to exist.

    Used for `stubgen.py`'s whole-batch ctypedef-alias-pruning cross
    -check (`convert_multiple_files`, when `resolve_ctypedef_aliases` is
    on): a single file's own scope has no way to know whether *another*
    file in the same conversion batch still needs an alias this file
    declares (see `StubgenPyxConfig.resolve_ctypedef_aliases`'s
    docstring) -- so that check has to run against every other
    already-converted module's output directly, not just this one
    file's own scope the way `Converter.convert_scope`'s own
    (necessarily more limited) check does.
    """
    if any(
        _import_uses_name(module, imp.statement, name, imports_are_trimmed)
        for imp in module.imports
    ):
        return True
    return _scope_uses_name(module, name)


def _scope_uses_name(module: PyiModule, name: str) -> bool:
    """Whether `name` appears in a function signature or attribute
    assignment anywhere in `module`'s scope tree (imports not considered).

    An assignment that itself declares `name` (``name: TypeAlias = int``)
    is not a use of it: another file declaring the same name is a
    separate symbol, and would otherwise keep this one alive."""
    for function in _scope_functions(module.scope):
        if any(
            _text_uses_name(arg.annotation, name) for arg in function.signature.args
        ) or _text_uses_name(function.signature.return_type, name):
            return True
    return any(
        a.name != name and _text_uses_name(a.statement, name)
        for a in _scope_assignments(module.scope)
    )


def _import_uses_name(
    module: PyiModule, statement: str, name: str, imports_are_trimmed: bool
) -> bool:
    """Whether the import `statement` needs `name` to exist elsewhere.

    Without `imports_are_trimmed`, any mention of `name` counts. With it,
    an imported `name` only counts if the import survives `trim_imports`:
    its bound name is used elsewhere in `module`, or it is an explicit
    ``X as X`` re-export. A mention that isn't an imported name (a module
    path component) always counts, as does a statement that doesn't parse.
    """
    if not _text_uses_name(statement, name):
        return False
    if not imports_are_trimmed:
        return True
    try:
        tree = ast.parse(statement)
    except SyntaxError:
        return True
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        module_path = getattr(node, "module", None) or ""
        if _text_uses_name(module_path, name):
            return True
        for alias in node.names:
            if alias.name == "*" or not _text_uses_name(alias.name, name):
                continue
            if alias.asname == alias.name or _scope_uses_name(
                module, alias.asname or alias.name
            ):
                return True
    return False


def prune_prepared_ctypedef_aliases(
    prepared: list[tuple], imports_are_trimmed: bool = False
) -> None:
    """Prune each file's provisionally-dead ctypedef aliases (`prepared`,
    as built by `StubgenPyx._prepare_multiple_file_conversions`) once every
    successfully-converted module in the same batch has been checked --
    see `pyi_module_uses_name`'s docstring for why that cross-check is
    needed.

    The declaring module counts too: a surviving alias in it that is
    defined in terms of this one (``B: TypeAlias = A``) still needs it.
    Removing an alias can orphan the one it was defined in terms of, so
    this repeats until nothing more is removed.
    """
    successful_modules = [entry[4] for entry in prepared if entry[4] is not None]
    pending = [
        (module, dict(prunable))
        for _, _, early_result, _, module, _, prunable in prepared
        if early_result is None and prunable
    ]
    removed_any = True
    while removed_any:
        removed_any = False
        for module, prunable in pending:
            for alias_name, assignment in list(prunable.items()):
                if any(
                    pyi_module_uses_name(other, alias_name, imports_are_trimmed)
                    for other in successful_modules
                ):
                    continue
                remove_assignment_from_module(module, assignment)
                del prunable[alias_name]
                removed_any = True


def _text_uses_name(text: str | None, name: str) -> bool:
    """Whether `name` appears as a whole word anywhere in `text`.

    Broader than `_annotation_uses_name` (which only matches a full,
    top-level `|`-union member, e.g. for fused-type `TypeVar` usage
    detection) -- needed here because a `ctypedef` alias can appear
    nested inside an annotation (`Iterable[MyFloat]`), not just as a
    bare, standalone one (see `Converter.convert_scope`'s ctypedef
    -alias-pruning logic, which uses this to decide whether an alias's
    own declaration is still referenced by anything).
    """
    if text is None:
        return False
    return re.search(rf"\b{re.escape(name)}\b", text) is not None
