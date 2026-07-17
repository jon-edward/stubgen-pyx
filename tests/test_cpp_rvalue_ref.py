from __future__ import annotations

import pytest
from Cython.Compiler.Errors import CompileError

from stubgen_pyx.parsing.parser import parse_pyx


def test_pxd_rvalue_ref_parameter():
    source = """
cdef extern from "foo.hpp":
    cdef void bar(Baz&& val)
"""
    result = parse_pyx(source, pxd=True)
    assert result is not None


def test_pxd_rvalue_ref_return_type_raises():
    source = """
cdef extern from "foo.hpp":
    cdef Foo&& make_foo()
"""
    with pytest.raises(CompileError):
        parse_pyx(source, pxd=True)


def test_pxd_mixed_ref_and_rvalue_ref():
    source = """
cdef extern from "foo.hpp":
    cdef void swap(Foo& a, Foo&& b)
"""
    result = parse_pyx(source, pxd=True)
    assert result is not None


def test_pxd_template_rvalue_ref():
    source = """
from libcpp.vector cimport vector

cdef extern from "foo.hpp":
    cdef void consume(vector[Foo]&& items)
"""
    result = parse_pyx(source, pxd=True)
    assert result is not None


def test_pyx_without_cpp_still_parses():
    source = """
def add(x: int, y: int) -> int:
    return x + y
"""
    result = parse_pyx(source, pxd=False)
    assert result is not None
