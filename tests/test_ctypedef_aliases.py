"""Direct unit tests for `conversion/ctypedef_aliases.py`'s standalone
helpers -- the scope-traversal and removal machinery that
`resolve_ctypedef_aliases` relies on, exercised here without going
through a full conversion.
"""

from __future__ import annotations

from stubgen_pyx.conversion.ctypedef_aliases import (
    ctypedef_alias_map,
    pyi_module_uses_name,
    remove_assignment_from_module,
)
from stubgen_pyx.models.pyi_elements import (
    PyiArgument,
    PyiAssignment,
    PyiClass,
    PyiFunction,
    PyiImport,
    PyiModule,
    PyiScope,
    PyiSignature,
)


class _NodeWithNoScope:
    """A stand-in for a `ScopeVisitor.node` with no `.scope` attribute at all."""


class _FakeVisitor:
    def __init__(self, node):
        self.node = node


def test_ctypedef_alias_map_returns_empty_dict_when_node_has_no_scope():
    """`visitor.node.scope` can be missing entirely (e.g. a node that
    never reached real declaration analysis); this must degrade to an
    empty map rather than raising."""
    assert ctypedef_alias_map(_FakeVisitor(_NodeWithNoScope())) == {}


def test_remove_assignment_from_module_returns_false_when_not_found():
    """An assignment that was already removed (or never belonged to
    this module) is reported as not found, not an error."""
    module = PyiModule(scope=PyiScope(assignments=[PyiAssignment("x = 1", name="x")]))
    missing = PyiAssignment("y = 2", name="y")
    assert remove_assignment_from_module(module, missing) is False


def test_remove_assignment_from_module_finds_it_in_a_nested_class_scope():
    """An alias declared inside a nested class body -- not just at
    module top level -- must still be found and removed; this is what
    exercises `_iter_scopes`'/`_scope_assignments`'s recursive descent
    into `PyiClass.scope`."""
    target = PyiAssignment("Inner_alias: TypeAlias = int", name="Inner_alias")
    inner_scope = PyiScope(assignments=[target])
    outer_class = PyiClass("Outer", scope=inner_scope)
    module = PyiModule(scope=PyiScope(classes=[outer_class]))

    assert remove_assignment_from_module(module, target) is True
    assert target not in inner_scope.assignments


def test_pyi_module_uses_name_true_via_function_argument_annotation():
    """`pyi_module_uses_name` must also check function signatures, not
    just plain assignments -- a name used only as an argument's
    annotation still counts as "used"."""
    func = PyiFunction(
        "f",
        is_async=False,
        signature=PyiSignature(args=[PyiArgument("x", annotation="MyAlias")]),
    )
    module = PyiModule(scope=PyiScope(functions=[func]))
    assert pyi_module_uses_name(module, "MyAlias") is True


def test_pyi_module_uses_name_true_via_function_return_annotation():
    func = PyiFunction(
        "f", is_async=False, signature=PyiSignature(args=[], return_type="MyAlias")
    )
    module = PyiModule(scope=PyiScope(functions=[func]))
    assert pyi_module_uses_name(module, "MyAlias") is True


def test_pyi_module_uses_name_false_when_genuinely_unused():
    module = PyiModule(scope=PyiScope(assignments=[PyiAssignment("x = 1", name="x")]))
    assert pyi_module_uses_name(module, "NeverMentioned") is False


def test_pyi_module_uses_name_true_via_nested_class_attribute():
    """A name used only in a nested class's own attribute assignment --
    not a top-level one -- still counts; exercises `_scope_assignments`'s
    recursive descent into `PyiClass.scope`."""
    inner = PyiClass(
        "Inner",
        scope=PyiScope(assignments=[PyiAssignment("attr: MyAlias", name="attr")]),
    )
    module = PyiModule(scope=PyiScope(classes=[inner]))
    assert pyi_module_uses_name(module, "MyAlias") is True


def _module_with_import(statement: str, **scope) -> PyiModule:
    return PyiModule(imports=[PyiImport(statement)], scope=PyiScope(**scope))


def test_pyi_module_uses_name_counts_any_import_by_default():
    module = _module_with_import("from m import Color")
    assert pyi_module_uses_name(module, "Color") is True


def test_pyi_module_uses_name_ignores_unused_import_when_imports_are_trimmed():
    module = _module_with_import("from m import Color")
    assert pyi_module_uses_name(module, "Color", imports_are_trimmed=True) is False


def test_pyi_module_uses_name_counts_explicit_reexport_when_imports_are_trimmed():
    module = _module_with_import("from m import Color as Color")
    assert pyi_module_uses_name(module, "Color", imports_are_trimmed=True) is True


def test_pyi_module_uses_name_counts_import_whose_binding_is_used_when_trimmed():
    func = PyiFunction(
        "f",
        is_async=False,
        signature=PyiSignature(args=[PyiArgument("x", annotation="C")]),
    )
    module = _module_with_import("from m import Color as C", functions=[func])
    assert pyi_module_uses_name(module, "Color", imports_are_trimmed=True) is True


def test_pyi_module_uses_name_ignores_renamed_import_whose_binding_is_unused():
    module = _module_with_import("from m import Color as C")
    assert pyi_module_uses_name(module, "Color", imports_are_trimmed=True) is False


def test_pyi_module_uses_name_counts_name_in_module_path_when_trimmed():
    module = _module_with_import("from Color import something")
    assert pyi_module_uses_name(module, "Color", imports_are_trimmed=True) is True


def test_pyi_module_uses_name_counts_unparseable_import_when_trimmed():
    module = _module_with_import("from m import (Color")
    assert pyi_module_uses_name(module, "Color", imports_are_trimmed=True) is True


def test_pyi_module_uses_name_ignores_assignment_declaring_the_same_name():
    """Another file declaring `name` is a separate symbol, not a use of it."""
    declaration = PyiAssignment("const_x: TypeAlias = int", name="const_x")
    module = PyiModule(scope=PyiScope(assignments=[declaration]))
    assert pyi_module_uses_name(module, "const_x") is False


def test_pyi_module_uses_name_still_counts_other_assignment_mentioning_name():
    declaration = PyiAssignment("const_x: TypeAlias = int", name="const_x")
    other = PyiAssignment("y: const_x", name="y")
    module = PyiModule(scope=PyiScope(assignments=[declaration, other]))
    assert pyi_module_uses_name(module, "const_x") is True
