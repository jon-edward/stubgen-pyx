"""Focused tests for Cython-to-Python type parsing helpers."""

from __future__ import annotations

from io import StringIO

from Cython.Compiler import Parsing
from Cython.Compiler.Scanning import PyrexScanner, StringSourceDescriptor

from stubgen_pyx.config import StubgenPyxConfig
from stubgen_pyx.conversion import type_parsing
from stubgen_pyx.conversion.type_parsing import Nodes as type_parsing_Nodes
from stubgen_pyx.parsing.context import StubgenContext
from stubgen_pyx.parsing.parser import _DEFAULT_MODULE_NAME, _resolve_scope
from stubgen_pyx.parsing.parser import parse_str as parse_pyx
from stubgen_pyx.stubgen import StubgenPyx


def _parse_raw(source: str):
    """Parse `source` into a raw AST, stopping right after Cython's own
    parser -- before `capture_static_types`/`run_stub_pipeline` run and
    can remove or rewrite declarations that have no runtime component
    (a bare module-level `cdef int value` is exactly such a declaration,
    and no longer reachable via `parse_pyx`/`_first_node` once the full
    pipeline has run over it). Tests below exercise narrow,
    purely-structural AST-shape helpers (`_declarator_name`,
    `_extract_templated_type`, `_extract_array_type`,
    `_extract_memoryview_type`, ...) directly; those only ever look at
    the syntax Cython's parser produced, never anything real declaration
    analysis would resolve, so there's no need to run the full pipeline
    (or keep the node reachable afterward) just to get one to test
    against.
    """
    context = StubgenContext()
    module_name = _DEFAULT_MODULE_NAME
    source_desc = StringSourceDescriptor(module_name, source)
    initial_pos = (source_desc, 1, 0)
    scope = _resolve_scope(context, module_name, initial_pos, allow_pxd_merge=False)
    scope.cpp = context.cpp
    scanner = PyrexScanner(
        StringIO(source),
        source_desc,
        source_encoding="UTF-8",
        scope=scope,
        context=context,
        initial_pos=initial_pos,
    )
    tree = Parsing.p_module(
        scanner, False, module_name, ctx=Parsing.Ctx(allow_struct_enum_decorator=True)
    )
    tree.scope = scope
    tree.is_pxd = False
    return tree


def _first_node(source: str, node_type: type):
    """Find the first node of `node_type` in `source`'s raw (pre-pipeline) AST."""
    tree = _parse_raw(source)
    pending = [getattr(tree, "body", None)]
    while pending:
        node = pending.pop(0)
        if node is None:
            continue
        if isinstance(node, node_type):
            return node
        for attr in getattr(node, "child_attrs", ()):
            child = getattr(node, attr, None)
            if isinstance(child, list):
                pending.extend(child)
            elif child is not None:
                pending.append(child)
    raise AssertionError(f"No {node_type.__name__} found")


def test_parameterize_builtin_generic_handles_none_and_unknown():
    assert type_parsing.parameterize_builtin_generic(None) is None
    assert (
        type_parsing.parameterize_builtin_generic("tuple") == "tuple[typing.Any, ...]"
    )
    assert type_parsing.parameterize_builtin_generic("Custom") == "Custom"


def test_declarator_helpers_unwrap_and_reject_non_function():
    node = _first_node("cdef int value", type_parsing_Nodes.CVarDefNode)
    declarator = node.declarators[0]
    assert type_parsing._declarator_name(declarator) == "value"
    assert type_parsing._get_func_decl_type(declarator, "int") is None


def test_extract_type_handles_unknown_node_and_module_qualified_type():
    unknown_node = _first_node(
        "cdef int value", type_parsing_Nodes.CVarDefNode
    ).declarators[0]
    assert type_parsing.extract_type_from_base_type(unknown_node) is None
    node = _first_node("cdef pkg.Widget value", type_parsing_Nodes.CVarDefNode)
    assert type_parsing.extract_type_from_base_type(node) == "pkg.Widget"


def test_unnamed_c_variable_defaults_to_any():
    node = _first_node("cdef public x", type_parsing_Nodes.CVarDefNode)
    assert type_parsing.extract_type_from_base_type(node) == "typing.Any"


def test_tuple_type_is_rendered_with_each_component():
    node = _first_node(
        "cpdef (int, str) value(int item, str label):\n    return (item, label)",
        type_parsing_Nodes.CFuncDefNode,
    )
    assert type_parsing.extract_type_from_base_type(node) == "tuple[int, str]"


def test_template_type_with_missing_base_returns_none():
    node = _first_node(
        "from libcpp.vector cimport vector\ncdef vector[int] value",
        type_parsing_Nodes.CVarDefNode,
    )
    templated = node.base_type
    templated.base_type_node = None
    assert type_parsing._extract_templated_type(templated) is None


def test_template_type_without_arguments_uses_base_name():
    node = _first_node("cdef Foo[int] value", type_parsing_Nodes.CVarDefNode)
    templated = node.base_type
    templated.positional_args = []
    assert type_parsing._extract_templated_type(templated) == "Foo"


def test_array_type_handles_missing_and_empty_scalar_names():
    node = _first_node("cdef int[3] values", type_parsing_Nodes.CVarDefNode).base_type
    original_base = node.base_type_node
    node.base_type_node = None
    assert type_parsing._extract_array_type(node) is None
    node.base_type_node = original_base
    original_name = original_base.name
    original_base.name = ""
    assert type_parsing._extract_array_type(node) is None
    original_base.name = original_name


def test_memoryview_unknown_scalar_falls_back_to_memoryview():
    node = _first_node("cdef Custom[:] view", type_parsing_Nodes.CVarDefNode).base_type
    assert type_parsing._extract_memoryview_type(node) == "memoryview"


def test_self_in_plain_python_class_does_not_log_unknown_base_type(caplog):
    """`self` in a plain (non-`cdef class`) method resolves to
    `py_object_type` -- the expected, routine outcome for any untyped
    argument, not a failure -- and must not trigger the "Unknown base
    type" debug log.

    Confirmed empirically against a real-world report: `is_self_arg` is
    only ever set for a `cdef class`'s `self`/`cls` (which needs a real
    C-level type); a plain Python class's `self` has no such need and
    falls through to the general `Entry`/`Type` fallback in
    `extract_type_from_base_type` instead, landing on this exact
    `py_object_type` case. The resulting annotation was already correct
    (no annotation shown for `self`, as intended) -- the debug message
    alone was the (misleading) problem, made it look like something had
    failed to resolve when nothing had.
    """
    caplog.set_level("DEBUG")
    result = StubgenPyx(
        StubgenPyxConfig(exclude_attribution=True, sort_imports=False)
    ).convert_str("""
class Foo:
    def bar(self, x: int) -> int:
        return x
""")
    assert "def bar(self, x: int) -> int: ..." in result
    assert not any("Unknown base type" in r.message for r in caplog.records)


def test_genuinely_unresolvable_type_still_logs(caplog):
    """The debug log is narrowed, not silenced outright -- a real
    resolution failure (as opposed to the expected `py_object_type`/
    function-node cases) is still worth knowing about."""
    caplog.set_level("DEBUG")

    class _UnrenderableType:
        """A type shape `render_pyrex_type` has no case for at all."""

        is_void = is_ptr = is_array = is_cfunction = is_fused = False
        is_memoryviewslice = is_ctuple = is_cpp_class = is_struct = False
        is_extension_type = is_pyobject = is_numeric = is_string = False
        is_int = is_float = is_cyp_unicode_char = False

    node = _first_node("cdef int value", type_parsing_Nodes.CVarDefNode)
    node.base_type = None
    node.type = _UnrenderableType()
    assert type_parsing.extract_type_from_base_type(node) is None
    assert any("Unknown base type" in r.message for r in caplog.records)


class TestRenderPyrexTypeEdgeCases:
    """Direct tests for `render_pyrex_type` branches a full end-to-end
    conversion doesn't naturally reach: a `None` type, a ctuple, a
    memoryview with no numpy scalar equivalent, a templated C++ class
    resolved via `Entry`/`Type` (not the structural path), and a bare
    function type rendered directly (not as an argument/return, where
    a different path narrows to it first).
    """

    def test_none_type_returns_none(self):
        assert type_parsing.render_pyrex_type(None) is None

    def test_ctuple_type(self):
        result = parse_pyx(
            'cdef extern from "foo.hpp" nogil:\n    cdef (int, double) get_pair()\n',
            pxd=True,
        )
        entry = result.scope.entries["get_pair"]
        assert (
            type_parsing.render_pyrex_type(entry.type.return_type)
            == "tuple[int, float]"
        )

    def test_memoryview_with_no_numpy_scalar_falls_back_to_memoryview(self):
        result = parse_pyx(
            'cdef extern from "foo.hpp" nogil:\n    cdef object[:] get_view()\n',
            pxd=True,
        )
        entry = result.scope.entries["get_view"]
        assert type_parsing.render_pyrex_type(entry.type.return_type) == "memoryview"

    def test_templated_cpp_class_via_entry_type(self):
        """The `Entry`/`Type` counterpart to the structural
        `_extract_templated_type` tests above -- a templated C++ class
        type resolved directly, not from a raw AST node."""
        result = parse_pyx(
            "from libcpp.vector cimport vector\n\nctypedef vector[int] int_vec\n",
            pxd=True,
        )
        entry = result.scope.entries["int_vec"]
        rendered = type_parsing.render_pyrex_type(entry.type)
        assert rendered is not None
        assert rendered.startswith("vector[int,")

    def test_bare_cfunction_type_renders_as_callable(self):
        result = parse_pyx(
            'cdef extern from "foo.hpp" nogil:\n    cdef int compute(int x)\n', pxd=True
        )
        entry = result.scope.entries["compute"]
        assert (
            type_parsing.render_pyrex_type(entry.type) == "typing.Callable[[int], int]"
        )


class TestGetCdefVariablesEdgeCases:
    """`get_cdef_variables`'s handling of a function-pointer declarator
    and of a declarator type it doesn't recognize at all -- neither
    shape reaches here through the full pipeline in practice anymore
    (a `ctypedef`'d function pointer is excluded from the stub
    entirely; see `analysis/visitor.py::visit_CTypeDefNode`), so both
    are tested directly against a raw, pre-pipeline node instead.
    """

    def test_function_pointer_declarator_renders_as_callable(self):
        node = _first_node("cdef int (*fp)(int, int)", type_parsing_Nodes.CVarDefNode)
        assert type_parsing.get_cdef_variables(node) == [
            ("fp", "typing.Callable[[int, int], int]")
        ]

    def test_unrecognized_declarator_type_is_skipped(self):
        """A C++ reference declarator (`int&`) isn't one of the
        declarator shapes this function knows how to walk; it must be
        skipped (logged, not raised) rather than crashing."""
        node = _first_node("cdef int& x", type_parsing_Nodes.CVarDefNode)
        assert type_parsing.get_cdef_variables(node) == []


class TestTypeFromBaseTypeName:
    def test_uses_base_type_node_when_present(self):
        """A base type that itself wraps another base-type node (rather
        than being a plain named type) resolves through that inner
        node instead."""
        from types import SimpleNamespace

        fake_base = SimpleNamespace(
            base_type_node=SimpleNamespace(module_path=["mymod"], name="Foo")
        )
        assert type_parsing_Nodes  # sanity: module imported
        assert type_parsing._type_from_base_type_name(fake_base) == "mymod.Foo"


class TestFusedMemberName:
    """`_fused_member_name` is used only for a `ctypedef fused`'s own
    member type nodes, captured pre-pipeline (see
    `capture_static_types`'s docstring) -- exercised here against a
    raw, real parse rather than a fake, since it specifically has to
    handle a *templated* member type (`vector[int]`), not just a plain
    named one.
    """

    def test_plain_member_name(self):
        node = _first_node(
            "ctypedef fused numeric:\n    int\n    double\n",
            type_parsing_Nodes.FusedTypeNode,
        )
        assert type_parsing._fused_member_name(node.types[0]) == "int"

    def test_templated_member_name_unwraps_to_the_base(self):
        node = _first_node(
            "from libcpp.vector cimport vector\n\n"
            "ctypedef fused numeric:\n    vector[int]\n    double\n",
            type_parsing_Nodes.FusedTypeNode,
        )
        templated_member = node.types[0]
        assert type_parsing_Nodes.TemplatedTypeNode
        assert type_parsing._fused_member_name(templated_member) == "vector"

    def test_non_simple_node_returns_none(self):
        assert type_parsing._fused_member_name(object()) is None


class TestCaptureStaticTypesRobustness:
    def test_recursive_capture_tolerates_attribute_error_on_base_type_access(self):
        """A node whose `.base_type` access itself raises
        `AttributeError` must not crash the whole pre-pipeline capture
        walk -- `hasattr` itself absorbs the error before the `try`
        block guarding `extract_type_from_base_type` is ever reached,
        so this exercises the outer `hasattr` guard specifically, not
        that inner `try`/`except` (which no longer has a live trigger:
        every attribute access inside `extract_type_from_base_type` is
        itself `getattr`/`hasattr`-guarded at this point)."""

        class _RaisesOnBaseType:
            @property
            def base_type(self):
                raise AttributeError("simulated")

        node = _RaisesOnBaseType()
        type_parsing._capture_static_types_recursive(node, None, set())
        assert not hasattr(node, "_stubgen_static_type")


class TestExtractArrayTypeRobustness:
    def test_returns_none_when_inner_node_has_no_name(self):
        from types import SimpleNamespace

        fake = SimpleNamespace(base_type_node=object())
        assert type_parsing._extract_array_type(fake) is None
