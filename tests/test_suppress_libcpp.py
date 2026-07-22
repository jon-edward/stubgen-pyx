from __future__ import annotations

from stubgen_pyx import StubgenPyx
from stubgen_pyx.config import StubgenPyxConfig


def _stubgen() -> StubgenPyx:
    return StubgenPyx(StubgenPyxConfig(exclude_attribution=True, sort_imports=False))


def test_libcpp_bool_import_suppressed():
    result = _stubgen().convert_str(
        """
from libcpp cimport bool

cpdef foo(bool x):
    pass
"""
    )

    assert "from libcpp import bool" not in result
    assert "def foo(x: bool)" in result


def test_libcpp_memory_unique_ptr_suppressed():
    result = _stubgen().convert_str(
        """
from libcpp.memory cimport unique_ptr

cpdef void take_ptr(unique_ptr[int] value):
    pass
"""
    )

    assert "from libcpp.memory import unique_ptr" not in result
    assert "def take_ptr" in result


def test_libcpp_vector_suppressed():
    result = _stubgen().convert_str(
        """
from libcpp.vector cimport vector

cpdef void take_vector(vector[int] value):
    pass
"""
    )

    assert "from libcpp.vector import vector" not in result
    assert "def take_vector" in result


def test_libcpp_string_suppressed():
    result = _stubgen().convert_str(
        """
from libcpp.string cimport string

cpdef void take_string(string value):
    pass
"""
    )

    assert "from libcpp.string import string" not in result
    assert "def take_string" in result


def test_mixed_libcpp_and_normal_import():
    result = _stubgen().convert_str(
        """
from libcpp cimport bool
from some_module cimport Foo

cpdef Foo convert(bool x):
    pass
"""
    )

    assert "from libcpp import bool" not in result
    assert "from some_module import Foo" in result
