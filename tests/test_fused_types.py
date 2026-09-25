"""Tests for fused types."""

from __future__ import annotations

from textwrap import dedent, indent

from stubgen_pyx import StubgenPyx
from stubgen_pyx.config import StubgenPyxConfig


def _stubgen() -> StubgenPyx:
    return StubgenPyx(StubgenPyxConfig(exclude_attribution=True, sort_imports=False))


def _cy(snippet: str) -> str:
    """Dedent and normalize an inline Cython snippet so tests can indent them naturally."""
    return dedent(snippet).strip("\n") + "\n"


def _cdef_classes(*names: str) -> str:
    """Generate empty ``cdef class`` blocks for each name."""
    return "".join(f"cdef class {n}:\n    pass\n" for n in names)


def _ctypedef_fused(name: str, *members: str) -> str:
    """Generate a ``ctypedef fused`` block over the given members."""
    body = "\n".join(f"    {m}" for m in members)
    return f"ctypedef fused {name}:\n{body}\n"


def _class_wrap(name: str, body: str) -> str:
    """Wrap a body inside a ``cdef class``; body is dedented then reindented one level."""
    return f"cdef class {name}:\n{indent(_cy(body), '    ')}"


# Common building blocks; two extension types and a fused alias over them.
_FOOBAR_CLASSES = _cdef_classes("Foo", "Bar")
_FOOBAR_FUSED = _ctypedef_fused("FooOrBar", "Foo", "Bar")
_FOOBAR_PREAMBLE = _FOOBAR_CLASSES + _FOOBAR_FUSED


def test_single_param_unrelated_return_becomes_union():
    """One fused-typed parameter with an unrelated return -> Union annotation."""
    result = _stubgen().convert_str(
        _FOOBAR_PREAMBLE
        + _cy("""
        cpdef Foo f(FooOrBar x):
            pass
    """)
    )
    assert "def f(x: Foo | Bar) -> Foo" in result
    assert "x: ..." not in result
    assert "TypeVar" not in result


def test_param_and_return_correlated_becomes_typevar():
    """Fused type in both param and return -> TypeVar to preserve correlation."""
    result = _stubgen().convert_str(
        _FOOBAR_PREAMBLE
        + _cy("""
        cpdef FooOrBar f(FooOrBar x):
            pass
    """)
    )
    assert "from typing import TypeVar" in result
    assert "FooOrBar = TypeVar('FooOrBar', Foo, Bar)" in result
    assert "def f(x: FooOrBar) -> FooOrBar" in result
    assert "x: ..." not in result
    assert "-> ..." not in result
    assert "ctypedef" not in result
    assert "fused" not in result


def test_multiple_params_same_fused_type_becomes_typevar():
    """Two params sharing a fused type must be the same concrete type -> TypeVar."""
    result = _stubgen().convert_str(
        _FOOBAR_PREAMBLE
        + _cy("""
        cpdef int f(FooOrBar x, FooOrBar y):
            pass
    """)
    )
    assert "from typing import TypeVar" in result
    assert "FooOrBar = TypeVar('FooOrBar', Foo, Bar)" in result
    assert "def f(x: FooOrBar, y: FooOrBar) -> int" in result


def test_return_only_fused_type_becomes_union():
    """Return-only fused type has no param to correlate, so it degrades to a Union."""
    result = _stubgen().convert_str(
        _FOOBAR_PREAMBLE
        + _cy("""
        cpdef FooOrBar f():
            pass
    """)
    )
    assert "def f() -> Foo | Bar" in result
    assert "-> ..." not in result
    assert "TypeVar" not in result


def test_fused_param_with_none_default_becomes_optional_union():
    """A single fused param defaulting to None -> ``Foo | Bar | None``."""
    result = _stubgen().convert_str(
        _FOOBAR_PREAMBLE
        + _cy("""
        cpdef int f(FooOrBar x = None):
            pass
    """)
    )
    assert "def f(x: Foo | Bar | None=None) -> int" in result
    assert "x: ..." not in result


def test_two_distinct_fused_types_in_one_signature():
    """Different fused types in the same signature are resolved independently."""
    result = _stubgen().convert_str(
        _FOOBAR_PREAMBLE
        + _cdef_classes("X", "Y")
        + _ctypedef_fused("XY", "X", "Y")
        + _cy("""
            cpdef FooOrBar f(FooOrBar x, XY y):
                pass
        """)
    )
    # FooOrBar is used in return + param -> TypeVar
    assert "FooOrBar = TypeVar('FooOrBar', Foo, Bar)" in result
    # XY appears once, in a single param -> Union
    assert "y: X | Y" in result
    assert "def f(x: FooOrBar, y: X | Y) -> FooOrBar" in result


def test_multiple_independent_fused_types_both_stay_union():
    """Two different fused types, each used only as a single param, both stay Union."""
    result = _stubgen().convert_str(
        _FOOBAR_PREAMBLE
        + _cdef_classes("X", "Y")
        + _ctypedef_fused("XY", "X", "Y")
        + _cy("""
            cpdef int f(FooOrBar x, XY y):
                pass
        """)
    )
    assert "def f(x: Foo | Bar, y: X | Y) -> int" in result
    assert "TypeVar" not in result


def test_method_single_fused_param_becomes_union():
    """``self`` must not count towards the fused-type usage tally."""
    result = _stubgen().convert_str(
        _FOOBAR_PREAMBLE
        + _class_wrap(
            "Ops",
            """
            cpdef Foo f(self, FooOrBar x):
                pass
        """,
        )
    )
    assert "def f(self, x: Foo | Bar) -> Foo" in result
    assert "TypeVar" not in result


def test_method_correlated_fused_type_uses_typevar():
    """Fused type across method param + return correlates via TypeVar."""
    result = _stubgen().convert_str(
        _FOOBAR_PREAMBLE
        + _class_wrap(
            "Ops",
            """
            cpdef FooOrBar f(self, FooOrBar x):
                pass
        """,
        )
    )
    assert "FooOrBar = TypeVar('FooOrBar', Foo, Bar)" in result
    assert "def f(self, x: FooOrBar) -> FooOrBar" in result


def test_method_and_module_function_share_typevar_definition():
    """Use same fused type in free function and method Module-level Union."""
    result = _stubgen().convert_str(
        _FOOBAR_PREAMBLE
        + _cy("""
            cpdef int f(FooOrBar x):
                pass
        """)
        + _class_wrap(
            "Ops",
            """
            cpdef FooOrBar g(self, FooOrBar x):
                pass
        """,
        )
    )
    assert result.count("FooOrBar = TypeVar('FooOrBar', Foo, Bar)") == 1
    assert "def f(x: Foo | Bar) -> int" in result
    assert "def g(self, x: FooOrBar) -> FooOrBar" in result


def test_def_function_single_fused_param_becomes_union():
    """``def`` functions follow the same Union heuristic as ``cpdef``."""
    result = _stubgen().convert_str(
        _FOOBAR_PREAMBLE
        + _cy("""
        def f(FooOrBar x):
            pass
    """)
    )
    assert "def f(x: Foo | Bar)" in result
    assert "TypeVar" not in result


def test_cdef_c_only_function_with_fused_param_is_skipped():
    """``cdef`` functions must never appear in the stub."""
    result = _stubgen().convert_str(
        _FOOBAR_PREAMBLE
        + _cy("""
        cdef int f(FooOrBar x):
            pass

        cpdef int g(FooOrBar x):
            pass
    """)
    )
    assert "def f(" not in result
    assert "def g(x: Foo | Bar) -> int" in result


def test_pxd_declared_fused_type_used_by_pyx_becomes_typevar():
    """Fused type from companion .pxd is resolved like a locally-declared one."""
    result = _stubgen().convert_str(
        _FOOBAR_CLASSES
        + _cy("""
            cpdef FooOrBar f(FooOrBar x):
                pass
        """),
        pxd_str=_FOOBAR_FUSED + "cpdef FooOrBar f(FooOrBar x)\n",
    )
    assert result.count("FooOrBar = TypeVar('FooOrBar', Foo, Bar)") == 1
    assert "def f(x: FooOrBar) -> FooOrBar" in result


def test_pxd_declared_fused_type_single_param_becomes_union():
    """Union heuristic still applies when the fused type comes from the pxd."""
    result = _stubgen().convert_str(
        _FOOBAR_CLASSES
        + _cy("""
            cpdef int f(FooOrBar x):
                pass
        """),
        pxd_str=_FOOBAR_FUSED + "cpdef int f(FooOrBar x)\n",
    )
    assert "def f(x: Foo | Bar) -> int" in result
    assert "TypeVar" not in result


def test_cython_numeric_primitives_deduplicated():
    """C primitives that map to Python ``int`` (single- and multi-word spellings) must deduplicate."""
    result = _stubgen().convert_str(
        _cy("""
        ctypedef fused integral_t:
            int
            long
            long long
            unsigned int
            unsigned char
            unsigned short
            unsigned long
            unsigned long long
            signed char
            size_t

        cpdef integral_t f(integral_t x, integral_t y):
            pass
    """)
    )
    assert "integral_t = TypeVar('integral_t', int)" in result
    assert "def f(x: integral_t, y: integral_t) -> integral_t" in result
    assert "int, int" not in result


def test_cython_float_primitives_deduplicated():
    """C primitives that map to Python ``float`` (including multi-word ``long double``) must deduplicate."""
    result = _stubgen().convert_str(
        _cy("""
        ctypedef fused numeric_t:
            int
            float
            double
            long double

        cpdef numeric_t f(numeric_t x):
            pass
    """)
    )
    assert "x: ..." not in result
    assert "-> ..." not in result
    assert "numeric_t = TypeVar('numeric_t', int, float)" in result
    assert "def f(x: numeric_t) -> numeric_t" in result
    assert "float, float" not in result


def test_fused_unicode_and_str_deduplicated():
    """``unicode`` maps to Python ``str`` and must deduplicate against a sibling ``str`` member."""
    result = _stubgen().convert_str(
        _cy("""
        ctypedef fused string_t:
            unicode
            str

        cpdef string_t f(string_t x):
            pass
    """)
    )
    assert "string_t = TypeVar('string_t', str)" in result
    assert "def f(x: string_t) -> string_t" in result
    assert "str, str" not in result


def test_fused_typed_memoryview_annotation_preserved():
    """A fused type shared between a memoryview argument and its return renders as ``memoryview``.

    ``numeric`` is used in both the argument and the return, which would
    normally resolve to a scalar-bound ``TypeVar`` shared between them --
    but that TypeVar doesn't type-check against an array, so both render
    as the unspecific but honest ``memoryview`` instead.
    """
    result = _stubgen().convert_str(
        _cy("""
        ctypedef fused numeric:
            int
            float

        cpdef numeric[:] f(numeric[:] x):
            pass
    """)
    )
    assert "def f(x)" not in result
    assert "def f(x: memoryview) -> memoryview" in result
    assert "int" not in result
    assert "float" not in result


def test_fused_typed_memoryview_single_usage_renders_ndarray_union():
    """A fused memoryview used in only one position renders as an ``NDArray`` union, not a scalar union."""
    result = _stubgen().convert_str(
        _cy("""
        ctypedef fused numeric:
            int
            double

        def f(numeric[:, :] x):
            pass
    """)
    )
    assert "def f(x)" not in result
    assert "x: NDArray[numpy.intc] | NDArray[numpy.double]" in result
    assert "def f(x: int" not in result
    assert "def f(x: double" not in result


def test_fused_typed_memoryview_does_not_affect_sibling_scalar_usage():
    """A fused type shared between a memoryview arg and a scalar arg keeps each usage's own shape."""
    result = _stubgen().convert_str(
        _cy("""
        ctypedef fused numeric:
            int
            double

        def f(numeric[:, :] arr, numeric scale):
            pass
    """)
    )
    assert "numeric = TypeVar('numeric', int, float)" in result
    assert "arr: memoryview" in result
    assert "scale: numeric" in result


def test_fused_type_with_object_member():
    """``object`` as a fused member absorbs the other members — accepted type collapses to ``object``."""
    result = _stubgen().convert_str(
        _FOOBAR_CLASSES
        + _ctypedef_fused("FooBarObj", "Foo", "Bar", "object")
        + _cy("""
            cpdef int f(FooBarObj x):
                pass
        """)
    )
    assert "def f(x: object) -> int" in result
    assert "x: ..." not in result
    assert "Foo | Bar | object" not in result


def test_fused_type_with_list_member():
    """``list`` as a fused member — Union must include the builtin."""
    result = _stubgen().convert_str(
        _cdef_classes("Foo")
        + _ctypedef_fused("FooOrList", "Foo", "list")
        + _cy("""
            cpdef int f(FooOrList x):
                pass
        """)
    )
    assert "def f(x: Foo | list) -> int" in result
    assert "x: ..." not in result


def test_fused_type_mixing_extension_and_c_primitive():
    """Extension type + C primitive as fused members — primitive maps to ``int``."""
    result = _stubgen().convert_str(
        _cdef_classes("Foo")
        + _ctypedef_fused("FooOrSize", "Foo", "int")
        + _cy("""
            cpdef int f(FooOrSize x):
                pass
        """)
    )
    assert "def f(x: Foo | int) -> int" in result
    assert "x: ..." not in result


def test_pxd_fused_param_with_star_default_becomes_optional_union():
    """pxd forward decl with ``= *`` default on fused param — impl has real default; stub must resolve annotation."""
    result = _stubgen().convert_str(
        _FOOBAR_CLASSES
        + _cy("""
            cpdef int f(FooOrBar x = None):
                pass
        """),
        pxd_str=_FOOBAR_FUSED + "cpdef int f(FooOrBar x = *)\n",
    )
    assert "def f(x: Foo | Bar | None=None) -> int" in result
    assert "x: ..." not in result


def test_fused_type_with_enum_member():
    """Extension type + ``cdef enum`` as fused members — enum survives in Union."""
    result = _stubgen().convert_str(
        _cy("""
            cdef enum MyEnum:
                A
                B
        """)
        + _cdef_classes("Foo")
        + _ctypedef_fused("FooOrEnum", "Foo", "MyEnum")
        + _cy("""
            cpdef int f(FooOrEnum x):
                pass
        """)
    )
    assert "def f(x: Foo | MyEnum) -> int" in result
    assert "x: ..." not in result


def test_typevar_shared_across_module_functions_only():
    """Many top-level fns sharing a fused type — one TypeVar decl."""
    result = _stubgen().convert_str(
        _FOOBAR_PREAMBLE
        + _cy("""
        cpdef FooOrBar f(FooOrBar x):
            pass

        cpdef FooOrBar g(FooOrBar x):
            pass

        cpdef FooOrBar h(FooOrBar x):
            pass
    """)
    )
    assert result.count("FooOrBar = TypeVar('FooOrBar', Foo, Bar)") == 1
    assert "def f(x: FooOrBar) -> FooOrBar" in result
    assert "def g(x: FooOrBar) -> FooOrBar" in result
    assert "def h(x: FooOrBar) -> FooOrBar" in result


def test_fused_correlate_with_extra_non_fused_params():
    """Fused param + fused return alongside non-fused params — TypeVar still correlates."""
    result = _stubgen().convert_str(
        _FOOBAR_PREAMBLE
        + _cy("""
        cpdef FooOrBar f(FooOrBar x, int extra, Foo other):
            pass
    """)
    )
    assert "FooOrBar = TypeVar('FooOrBar', Foo, Bar)" in result
    assert "def f(x: FooOrBar, extra: int, other: Foo) -> FooOrBar" in result


def test_four_member_fused_type_union_and_typevar():
    """4-member fused type — Union/TypeVar render all members in order."""
    result = _stubgen().convert_str(
        _cdef_classes("A", "B", "C", "D")
        + _ctypedef_fused("Quad", "A", "B", "C", "D")
        + _cy("""
            cpdef int single_param(Quad x):
                pass

            cpdef Quad correlate(Quad x):
                pass
        """)
    )
    assert "def single_param(x: A | B | C | D) -> int" in result
    assert "Quad = TypeVar('Quad', A, B, C, D)" in result
    assert "def correlate(x: Quad) -> Quad" in result


def test_pxd_declared_fused_type_shared_across_module_functions():
    """Multiple pxd fwd decls sharing a pxd-declared fused type — one TypeVar decl."""
    result = _stubgen().convert_str(
        _FOOBAR_CLASSES
        + _cy("""
            cpdef FooOrBar f(FooOrBar x):
                pass

            cpdef FooOrBar g(FooOrBar x):
                pass

            cpdef FooOrBar h(FooOrBar x):
                pass
        """),
        pxd_str=_FOOBAR_FUSED
        + _cy("""
            cpdef FooOrBar f(FooOrBar x)
            cpdef FooOrBar g(FooOrBar x)
            cpdef FooOrBar h(FooOrBar x)
        """),
    )
    assert result.count("FooOrBar = TypeVar('FooOrBar', Foo, Bar)") == 1
    assert "def f(x: FooOrBar) -> FooOrBar" in result
    assert "def g(x: FooOrBar) -> FooOrBar" in result
    assert "def h(x: FooOrBar) -> FooOrBar" in result


def test_pxd_declared_fused_type_used_by_pyx_method():
    """Class method using pxd-declared fused type — method-level resolution + pxd inheritance combined."""
    result = _stubgen().convert_str(
        _FOOBAR_CLASSES
        + _class_wrap(
            "Ops",
            """
            cpdef FooOrBar f(self, FooOrBar x):
                pass
        """,
        ),
        pxd_str=_FOOBAR_FUSED
        + _class_wrap("Ops", "cpdef FooOrBar f(self, FooOrBar x)"),
    )
    assert result.count("FooOrBar = TypeVar('FooOrBar', Foo, Bar)") == 1
    assert "def f(self, x: FooOrBar) -> FooOrBar" in result


class TestConvertFusedTypesStructuralPath:
    """`convert_fused_types`' first branch -- a structural walk of
    `visitor.fused_types` -- is effectively dead in real usage: a
    `ctypedef fused` declaration has no surviving node once real
    declaration analysis runs, so `visitor.fused_types` is always empty
    by the time any real conversion reaches it (the entries-based and
    pre-pipeline-snapshot fallbacks below it are what actually fire).
    Tested directly here against a real, raw (pre-pipeline) parse of a
    `ctypedef fused`, since that's the only way to get a populated
    `FusedTypeNode` to exercise this branch's own logic at all.
    """

    @staticmethod
    def _raw_fused_type_node(source: str):
        from io import StringIO

        from Cython.Compiler import Nodes, Parsing
        from Cython.Compiler.Scanning import PyrexScanner, StringSourceDescriptor

        from stubgen_pyx.parsing.context import StubgenContext
        from stubgen_pyx.parsing.parser import _DEFAULT_MODULE_NAME, _resolve_scope

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
            scanner,
            False,
            module_name,
            ctx=Parsing.Ctx(allow_struct_enum_decorator=True),
        )

        def find(node, out):
            if isinstance(node, Nodes.FusedTypeNode):
                out.append(node)
                return
            for attr in getattr(node, "child_attrs", None) or ():
                child = getattr(node, attr, None)
                if isinstance(child, list):
                    for c in child:
                        find(c, out)
                else:
                    find(child, out)

        found: list = []
        find(tree, found)
        return found[0]

    def test_structural_fused_type_node_resolves_concrete_types_and_numpy_scalars(self):
        from types import SimpleNamespace

        from stubgen_pyx.conversion.fused_types import convert_fused_types
        from stubgen_pyx.models.pyi_elements import PyiFusedType

        node = self._raw_fused_type_node(
            _cy(
                """
                ctypedef fused numeric:
                    int
                    double
                """
            )
        )
        fake_visitor = SimpleNamespace(
            fused_types=[node], node=SimpleNamespace(scope=None)
        )
        result = convert_fused_types(fake_visitor)
        assert result == {
            "numeric": PyiFusedType(
                name="numeric",
                concrete_types=("int", "float"),
                numpy_scalars=("intc", "double"),
            )
        }

    def test_static_snapshot_skips_a_name_already_found_structurally(self):
        """A name found via the structural walk must not be
        overwritten/reprocessed by the pre-pipeline static-snapshot
        fallback, even if (as here) it also has an entry there."""
        from types import SimpleNamespace

        from stubgen_pyx.conversion.fused_types import convert_fused_types

        node = self._raw_fused_type_node(
            _cy(
                """
                ctypedef fused numeric:
                    int
                    double
                """
            )
        )
        fake_visitor = SimpleNamespace(
            fused_types=[node],
            node=SimpleNamespace(
                scope=None,
                _stubgen_static_fused_members={"numeric": ["float", "long"]},
            ),
        )
        result = convert_fused_types(fake_visitor)
        # Still the structurally-resolved members, not the static
        # snapshot's ("float", "long") -- confirming the `continue`
        # actually skipped reprocessing it.
        assert result["numeric"].concrete_types == ("int", "float")
