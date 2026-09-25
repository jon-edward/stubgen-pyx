"""Tests for `const`/`volatile` qualifier handling.

Neither qualifier has a Python-level equivalent, so the correct behavior
throughout is to render the type as if the qualifier weren't there --
never to drop the type, and never to crash. Covers both the structural
path (a `const`/`volatile` base-type node, still present on the AST) and
the `Entry`/`Type`-based fallback (a synthesized `PropertyNode` for a
`cdef public`/`readonly` attribute, where the qualifier survives as a
`CConstOrVolatileType` wrapper on the resolved `Type` instead).
"""

from __future__ import annotations

from textwrap import dedent

from stubgen_pyx import StubgenPyx
from stubgen_pyx.config import StubgenPyxConfig


def _stubgen() -> StubgenPyx:
    return StubgenPyx(StubgenPyxConfig(exclude_attribution=True, sort_imports=False))


def _cy(snippet: str) -> str:
    """Dedent and normalize an inline Cython snippet so tests can indent them naturally."""
    return dedent(snippet).strip("\n") + "\n"


def test_const_scalar_argument():
    """`const int x` is just `int` -- the qualifier carries no Python-level meaning."""
    result = _stubgen().convert_str(
        _cy("""
        def f(const int x):
            pass
    """)
    )
    assert "def f(x: int)" in result


def test_volatile_scalar_argument():
    """`volatile int x` renders identically to a plain, unqualified `int x`."""
    result = _stubgen().convert_str(
        _cy("""
        def f(volatile int x):
            pass
    """)
    )
    assert "def f(x: int)" in result


def test_const_pointer_to_char_is_bytes():
    """`const char* x` (pointer to const) still maps to `bytes`, same as a plain `char*`."""
    result = _stubgen().convert_str(
        _cy("""
        def f(const char* x):
            pass
    """)
    )
    assert "def f(x: bytes)" in result


def test_const_qualified_pointer_argument_does_not_crash():
    """`char* const x` (a const pointer, not a pointer to const) must not raise."""
    result = _stubgen().convert_str(
        _cy("""
        cpdef void f(int* const x):
            pass
    """)
    )
    assert "def f(x: int)" in result


def test_const_qualified_pointer_argument_keeps_its_type():
    """A const-qualified pointer argument keeps its resolved type, not left bare."""
    result = _stubgen().convert_str(
        _cy("""
        cpdef void f(int* const x):
            pass
    """)
    )
    assert "def f(x)" not in result
    assert "def f(x: int)" in result


def test_const_and_const_pointer_combinations_all_typed():
    """Pointer-to-const, const-pointer, and both together all keep a real annotation."""
    result = _stubgen().convert_str(
        _cy("""
        cpdef void f(const char* a, char* const b, const char* const c):
            pass
    """)
    )
    assert "def f(a: bytes, b: bytes, c: bytes)" in result


def test_const_return_type():
    """A const-qualified return type (`const int`) renders as plain `int`."""
    result = _stubgen().convert_str(
        _cy("""
        cpdef const int compute():
            return 1
    """)
    )
    assert "def compute() -> int" in result


def test_const_public_class_attribute_keeps_its_type():
    """`cdef public const int` renders `int`, not `_typeshed.Incomplete`."""
    result = _stubgen().convert_str(
        _cy("""
        cdef class Foo:
            cdef public const int value
    """)
    )
    assert "value: int" in result
    assert "Incomplete" not in result


def test_const_readonly_class_attribute_keeps_its_type():
    """`cdef readonly const double` renders `float`, not `_typeshed.Incomplete`."""
    result = _stubgen().convert_str(
        _cy("""
        cdef class Foo:
            cdef readonly const double value
    """)
    )
    assert "value: float" in result
    assert "Incomplete" not in result


def test_volatile_public_class_attribute_keeps_its_type():
    """`cdef public volatile int` renders `int`, same as the `const` case above."""
    result = _stubgen().convert_str(
        _cy("""
        cdef class Foo:
            cdef public volatile int value
    """)
    )
    assert "value: int" in result
    assert "Incomplete" not in result


def test_const_pointer_fix_does_not_regress_plain_self():
    """The declarator-unwrap fix must not change ordinary, unqualified `self` handling."""
    result = _stubgen().convert_str(
        _cy("""
        cdef class Foo:
            cpdef bar(self, int x):
                pass
    """)
    )
    assert "def bar(self, x: int)" in result
