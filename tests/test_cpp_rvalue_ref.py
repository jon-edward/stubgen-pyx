from __future__ import annotations

import pytest
from Cython.Compiler.Errors import CompileError

from stubgen_pyx.parsing.parser import parse_str as parse_pyx


def test_pxd_rvalue_ref_parameter():
    source = """
cdef extern from "foo.hpp":
    cdef cppclass Baz:
        pass
    cdef void bar(Baz&& val)
"""
    result = parse_pyx(source, pxd=True)
    # A clean parse can still carry a recorded diagnostic (e.g. an
    # unresolved type); check for none rather than just not raising.
    assert result.diagnostics == []
    entry = result.scope.entries["bar"]
    (arg,) = entry.type.args
    assert arg.name == "val"
    assert arg.type.is_rvalue_reference
    assert arg.type.ref_base_type.name == "Baz"


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
    cdef cppclass Foo:
        pass
    cdef void swap(Foo& a, Foo&& b)
"""
    result = parse_pyx(source, pxd=True)
    assert result.diagnostics == []
    entry = result.scope.entries["swap"]
    arg_a, arg_b = entry.type.args
    # `Foo& a` -- an ordinary (lvalue) reference, not a rvalue reference.
    assert arg_a.name == "a"
    assert arg_a.type.is_reference
    assert not arg_a.type.is_rvalue_reference
    assert arg_a.type.ref_base_type.name == "Foo"
    # `Foo&& b` -- a rvalue reference, distinguished from `a` above.
    assert arg_b.name == "b"
    assert arg_b.type.is_rvalue_reference
    assert arg_b.type.ref_base_type.name == "Foo"


def test_pxd_template_rvalue_ref():
    source = """
from libcpp.vector cimport vector

cdef extern from "foo.hpp":
    cdef cppclass Foo:
        pass
    cdef void consume(vector[Foo]&& items)
"""
    result = parse_pyx(source, pxd=True)
    assert result.diagnostics == []
    entry = result.scope.entries["consume"]
    (arg,) = entry.type.args
    assert arg.name == "items"
    assert arg.type.is_rvalue_reference
    # The templated base type itself (`vector[Foo]`) must survive intact
    # under the reference, not collapse to e.g. plain `vector` or `Foo`.
    # (`vector`'s own declared signature carries an implicit trailing
    # allocator template parameter alongside the element type.)
    assert arg.type.ref_base_type.name == "vector"
    assert arg.type.ref_base_type.templates[0].name == "Foo"


def test_pyx_without_cpp_still_parses():
    source = """
def add(x: int, y: int) -> int:
    return x + y
"""
    result = parse_pyx(source, pxd=False)
    assert result.diagnostics == []
    def_stats = [
        s for s in result.source_ast.body.stats if type(s).__name__ == "DefNode"
    ]
    assert [s.name for s in def_stats] == ["add"]
