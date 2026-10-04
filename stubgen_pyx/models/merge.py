"""Merge and deduplicate PyiModule/PyiScope trees."""

from __future__ import annotations

from .pyi_elements import PyiClass, PyiModule


def merge_pxd_into_module(module: PyiModule, pxd_module: PyiModule) -> None:
    """Merge pxd module contents into the pyx module in-place.

    This is a free function rather than a method on PyiModule/PyiScope so that
    the data models stay as pure containers without merge semantics baked in.
    """
    module.scope.enums += pxd_module.scope.enums
    _deduplicate_enums(module.scope)
    module.scope.assignments += pxd_module.scope.assignments
    _deduplicate_assignments(module.scope)
    _merge_classes(module.scope, pxd_module.scope.classes)
    module.imports += pxd_module.imports


def _deduplicate_enums(scope) -> None:
    """Remove duplicate enums from a scope while preserving order.

    A pxd-declared enum is now already visible in the `.pyx`'s own
    conversion (`static_annotations.convert_declared_entries` walks the merged
    scope's entries directly, since `Context.find_module`'s companion-
    `.pxd` auto-merge puts pxd declarations in the *same* `Symtab.Scope`
    object), so the separate pxd-module conversion this merges in
    would otherwise add it a second time.
    """
    seen: set[str] = set()
    unique: list = []
    for enum in scope.enums:
        if enum.enum_name not in seen:
            seen.add(enum.enum_name)
            unique.append(enum)
    scope.enums = unique


def _deduplicate_assignments(scope) -> None:
    """Remove duplicate assignments from a scope while preserving order."""
    seen: set[str] = set()
    unique: list = []
    for assignment in scope.assignments:
        name = assignment.statement.partition("=")[0].partition(":")[0].strip()
        if not name or name not in seen:
            if name:
                seen.add(name)
            unique.append(assignment)
    scope.assignments = unique


def _merge_classes(scope, extra_classes: list[PyiClass]) -> None:
    """Merge extra classes into scope, combining same-name classes."""
    existing: dict[str, PyiClass] = {cls.name: cls for cls in scope.classes}
    for extra in extra_classes:
        if extra.name in existing:
            _merge_two_classes(existing[extra.name], extra)
        else:
            scope.classes.append(extra)


def _merge_two_classes(target: PyiClass, other: PyiClass) -> None:
    """Merge other into target in-place."""
    if target.doc is None:
        target.doc = other.doc
    target.bases = [*dict.fromkeys(target.bases + other.bases)]
    if target.metaclass is None:
        target.metaclass = other.metaclass
    target.decorators = [*dict.fromkeys(target.decorators + other.decorators)]
    target.keywords = {**target.keywords, **other.keywords}
    target.scope.assignments += other.scope.assignments
    _deduplicate_assignments(target.scope)
    target.scope.functions += other.scope.functions
    _merge_classes(target.scope, other.scope.classes)
    target.scope.enums += other.scope.enums
    _deduplicate_enums(target.scope)
