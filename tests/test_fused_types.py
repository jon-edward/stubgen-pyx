"""Tests for fused types."""

from __future__ import annotations

from textwrap import dedent, indent

import pytest

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


@pytest.mark.xfail(reason="fused type support not yet implemented", strict=True)
def test_single_param_unrelated_return_becomes_union():
    """One fused-typed parameter with an unrelated return -> Union annotation."""
    result = _stubgen().convert_str(_FOOBAR_PREAMBLE + _cy("""
        cpdef Foo f(FooOrBar x):
            pass
    """))
    assert "def f(x: Foo | Bar) -> Foo" in result
    assert "x: ..." not in result
    assert "TypeVar" not in result


@pytest.mark.xfail(reason="fused type support not yet implemented", strict=True)
def test_param_and_return_correlated_becomes_typevar():
    """Fused type in both param and return -> TypeVar to preserve correlation."""
    result = _stubgen().convert_str(_FOOBAR_PREAMBLE + _cy("""
        cpdef FooOrBar f(FooOrBar x):
            pass
    """))
    assert "from typing import TypeVar" in result
    assert "FooOrBar = TypeVar('FooOrBar', Foo, Bar)" in result
    assert "def f(x: FooOrBar) -> FooOrBar" in result
    assert "x: ..." not in result
    assert "-> ..." not in result
    assert "ctypedef" not in result
    assert "fused" not in result


@pytest.mark.xfail(reason="fused type support not yet implemented", strict=True)
def test_multiple_params_same_fused_type_becomes_typevar():
    """Two params sharing a fused type must be the same concrete type -> TypeVar."""
    result = _stubgen().convert_str(_FOOBAR_PREAMBLE + _cy("""
        cpdef int f(FooOrBar x, FooOrBar y):
            pass
    """))
    assert "from typing import TypeVar" in result
    assert "FooOrBar = TypeVar('FooOrBar', Foo, Bar)" in result
    assert "def f(x: FooOrBar, y: FooOrBar) -> int" in result


@pytest.mark.xfail(reason="fused type support not yet implemented", strict=True)
def test_return_only_fused_type_becomes_union():
    """Return-only fused type has no param to correlate, so it degrades to a Union."""
    result = _stubgen().convert_str(_FOOBAR_PREAMBLE + _cy("""
        cpdef FooOrBar f():
            pass
    """))
    assert "def f() -> Foo | Bar" in result
    assert "-> ..." not in result
    assert "TypeVar" not in result


@pytest.mark.xfail(reason="fused type support not yet implemented", strict=True)
def test_fused_param_with_none_default_becomes_optional_union():
    """A single fused param defaulting to None -> ``Foo | Bar | None``."""
    result = _stubgen().convert_str(_FOOBAR_PREAMBLE + _cy("""
        cpdef int f(FooOrBar x = None):
            pass
    """))
    assert "def f(x: Foo | Bar | None=None) -> int" in result
    assert "x: ..." not in result


@pytest.mark.xfail(reason="fused type support not yet implemented", strict=True)
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


@pytest.mark.xfail(reason="fused type support not yet implemented", strict=True)
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


@pytest.mark.xfail(reason="fused type support not yet implemented", strict=True)
def test_method_single_fused_param_becomes_union():
    """``self`` must not count towards the fused-type usage tally."""
    result = _stubgen().convert_str(
        _FOOBAR_PREAMBLE
        + _class_wrap("Ops", """
            cpdef Foo f(self, FooOrBar x):
                pass
        """)
    )
    assert "def f(self, x: Foo | Bar) -> Foo" in result
    assert "TypeVar" not in result


@pytest.mark.xfail(reason="fused type support not yet implemented", strict=True)
def test_method_correlated_fused_type_uses_typevar():
    """Fused type across method param + return correlates via TypeVar."""
    result = _stubgen().convert_str(
        _FOOBAR_PREAMBLE
        + _class_wrap("Ops", """
            cpdef FooOrBar f(self, FooOrBar x):
                pass
        """)
    )
    assert "FooOrBar = TypeVar('FooOrBar', Foo, Bar)" in result
    assert "def f(self, x: FooOrBar) -> FooOrBar" in result


@pytest.mark.xfail(reason="fused type support not yet implemented", strict=True)
def test_method_and_module_function_share_typevar_definition():
    """Use same fused type in free function and method Module-level Union."""
    result = _stubgen().convert_str(
        _FOOBAR_PREAMBLE
        + _cy("""
            cpdef int f(FooOrBar x):
                pass
        """)
        + _class_wrap("Ops", """
            cpdef FooOrBar g(self, FooOrBar x):
                pass
        """)
    )
    assert result.count("FooOrBar = TypeVar('FooOrBar', Foo, Bar)") == 1
    assert "def f(x: Foo | Bar) -> int" in result
    assert "def g(self, x: FooOrBar) -> FooOrBar" in result


@pytest.mark.xfail(reason="fused type support not yet implemented", strict=True)
def test_def_function_single_fused_param_becomes_union():
    """``def`` functions follow the same Union heuristic as ``cpdef``."""
    result = _stubgen().convert_str(_FOOBAR_PREAMBLE + _cy("""
        def f(FooOrBar x):
            pass
    """))
    assert "def f(x: Foo | Bar)" in result
    assert "TypeVar" not in result


@pytest.mark.xfail(reason="fused type support not yet implemented", strict=True)
def test_cdef_c_only_function_with_fused_param_is_skipped():
    """``cdef`` functions must never appear in the stub."""
    result = _stubgen().convert_str(_FOOBAR_PREAMBLE + _cy("""
        cdef int f(FooOrBar x):
            pass

        cpdef int g(FooOrBar x):
            pass
    """))
    assert "def f(" not in result
    assert "def g(x: Foo | Bar) -> int" in result


@pytest.mark.xfail(reason="fused type support not yet implemented", strict=True)
def test_pxd_declared_fused_type_used_by_pyx_becomes_typevar():
    """Fused type from companion .pxd is resolved like a locally-declared one."""
    result = _stubgen().convert_str(
        _FOOBAR_CLASSES + _cy("""
            cpdef FooOrBar f(FooOrBar x):
                pass
        """),
        pxd_str=_FOOBAR_FUSED + "cpdef FooOrBar f(FooOrBar x)\n",
    )
    assert result.count("FooOrBar = TypeVar('FooOrBar', Foo, Bar)") == 1
    assert "def f(x: FooOrBar) -> FooOrBar" in result


@pytest.mark.xfail(reason="fused type support not yet implemented", strict=True)
def test_pxd_declared_fused_type_single_param_becomes_union():
    """Union heuristic still applies when the fused type comes from the pxd."""
    result = _stubgen().convert_str(
        _FOOBAR_CLASSES + _cy("""
            cpdef int f(FooOrBar x):
                pass
        """),
        pxd_str=_FOOBAR_FUSED + "cpdef int f(FooOrBar x)\n",
    )
    assert "def f(x: Foo | Bar) -> int" in result
    assert "TypeVar" not in result



@pytest.mark.xfail(
    reason="Cython primitive aliasing produces duplicate TypeVar entries", strict=True
)
def test_cython_numeric_primitives_deduplicated():
    """``int``, ``long``, ``long long`` all map to Python ``int`` and must be deduplicated."""
    result = _stubgen().convert_str(_cy("""
        ctypedef fused integral_t:
            int
            long
            long long
            unsigned int

        cpdef integral_t f(integral_t x, integral_t y):
            pass
    """))
    assert "integral_t" in result
    assert "x: ..." not in result
    assert "-> ..." not in result
    assert "TypeVar('integral_t', int, int" not in result


@pytest.mark.xfail(reason="Cython float/double both alias Python float", strict=True)
def test_cython_float_primitives_deduplicated():
    """``float`` and ``double`` both map to Python ``float`` and must be deduplicated."""
    result = _stubgen().convert_str(_cy("""
        ctypedef fused numeric_t:
            int
            float
            double

        cpdef numeric_t f(numeric_t x):
            pass
    """))
    assert "x: ..." not in result
    assert "-> ..." not in result
    assert "numeric_t = TypeVar('numeric_t', int, float)" in result
    assert "def f(x: numeric_t) -> numeric_t" in result
    assert "TypeVar('numeric_t', int, float, float" not in result
    assert "TypeVar('numeric_t', float, float" not in result


@pytest.mark.xfail(reason="typed memoryview fused annotations are dropped", strict=True)
def test_fused_typed_memoryview_annotation_preserved():
    """``numeric[:]`` typed memoryviews with fused element types must keep an annotation."""
    result = _stubgen().convert_str(_cy("""
        ctypedef fused numeric:
            int
            float

        cpdef numeric[:] f(numeric[:] x):
            pass
    """))
    assert "def f(x)" not in result
    assert "TypeVar" in result or "|" in result


@pytest.mark.xfail(reason="fused type with object member not resolved", strict=True)
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


@pytest.mark.xfail(reason="fused type with list member not resolved", strict=True)
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


@pytest.mark.xfail(
    reason="fused type mixing extension type and C primitive not resolved", strict=True
)
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


@pytest.mark.xfail(
    reason="pxd-declared fused param with star default not resolved", strict=True
)
def test_pxd_fused_param_with_star_default_becomes_optional_union():
    """pxd forward decl with ``= *`` default on fused param — impl has real default; stub must resolve annotation."""
    result = _stubgen().convert_str(
        _FOOBAR_CLASSES + _cy("""
            cpdef int f(FooOrBar x = None):
                pass
        """),
        pxd_str=_FOOBAR_FUSED + "cpdef int f(FooOrBar x = *)\n",
    )
    assert "def f(x: Foo | Bar | None=None) -> int" in result
    assert "x: ..." not in result


@pytest.mark.xfail(reason="fused type with enum member not resolved", strict=True)
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


@pytest.mark.xfail(reason="fused type support not yet implemented", strict=True)
def test_typevar_shared_across_module_functions_only():
    """Many top-level fns sharing a fused type — one TypeVar decl."""
    result = _stubgen().convert_str(_FOOBAR_PREAMBLE + _cy("""
        cpdef FooOrBar f(FooOrBar x):
            pass

        cpdef FooOrBar g(FooOrBar x):
            pass

        cpdef FooOrBar h(FooOrBar x):
            pass
    """))
    assert result.count("FooOrBar = TypeVar('FooOrBar', Foo, Bar)") == 1
    assert "def f(x: FooOrBar) -> FooOrBar" in result
    assert "def g(x: FooOrBar) -> FooOrBar" in result
    assert "def h(x: FooOrBar) -> FooOrBar" in result


@pytest.mark.xfail(reason="fused type support not yet implemented", strict=True)
def test_fused_correlate_with_extra_non_fused_params():
    """Fused param + fused return alongside non-fused params — TypeVar still correlates."""
    result = _stubgen().convert_str(_FOOBAR_PREAMBLE + _cy("""
        cpdef FooOrBar f(FooOrBar x, int extra, Foo other):
            pass
    """))
    assert "FooOrBar = TypeVar('FooOrBar', Foo, Bar)" in result
    assert "def f(x: FooOrBar, extra: int, other: Foo) -> FooOrBar" in result


@pytest.mark.xfail(reason="fused type support not yet implemented", strict=True)
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


@pytest.mark.xfail(reason="fused type support not yet implemented", strict=True)
def test_pxd_declared_fused_type_shared_across_module_functions():
    """Multiple pxd fwd decls sharing a pxd-declared fused type — one TypeVar decl."""
    result = _stubgen().convert_str(
        _FOOBAR_CLASSES + _cy("""
            cpdef FooOrBar f(FooOrBar x):
                pass

            cpdef FooOrBar g(FooOrBar x):
                pass

            cpdef FooOrBar h(FooOrBar x):
                pass
        """),
        pxd_str=_FOOBAR_FUSED + _cy("""
            cpdef FooOrBar f(FooOrBar x)
            cpdef FooOrBar g(FooOrBar x)
            cpdef FooOrBar h(FooOrBar x)
        """),
    )
    assert result.count("FooOrBar = TypeVar('FooOrBar', Foo, Bar)") == 1
    assert "def f(x: FooOrBar) -> FooOrBar" in result
    assert "def g(x: FooOrBar) -> FooOrBar" in result
    assert "def h(x: FooOrBar) -> FooOrBar" in result


@pytest.mark.xfail(reason="fused type support not yet implemented", strict=True)
def test_pxd_declared_fused_type_used_by_pyx_method():
    """Class method using pxd-declared fused type — method-level resolution + pxd inheritance combined."""
    result = _stubgen().convert_str(
        _FOOBAR_CLASSES + _class_wrap("Ops", """
            cpdef FooOrBar f(self, FooOrBar x):
                pass
        """),
        pxd_str=_FOOBAR_FUSED + _class_wrap("Ops", "cpdef FooOrBar f(self, FooOrBar x)"),
    )
    assert result.count("FooOrBar = TypeVar('FooOrBar', Foo, Bar)") == 1
    assert "def f(self, x: FooOrBar) -> FooOrBar" in result
