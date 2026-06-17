from __future__ import annotations

from stubgen_pyx import StubgenPyx
from stubgen_pyx.analysis.visitor import ScopeVisitor
from stubgen_pyx.config import StubgenPyxConfig
from stubgen_pyx.parsing.parser import parse_pyx


def _stubgen() -> StubgenPyx:
    return StubgenPyx(
        StubgenPyxConfig(exclude_attribution=True, sort_imports=False)
    )


def test_scope_visitor_collects_fused_typedef():
    parsed = parse_pyx(
        """
ctypedef fused ColumnOrTable:
    Column
    Table
"""
    )

    visitor = ScopeVisitor(parsed.source_ast)

    assert len(visitor.fused_types) == 1
    assert getattr(visitor.fused_types[0], "name") == "ColumnOrTable"


def test_simple_fused_parameter_uses_union_annotation():
    result = _stubgen().convert_str(
        """
cdef class Column:
    pass

cdef class Scalar:
    pass

ctypedef fused ColumnOrScalar:
    Column
    Scalar

cpdef Column take(ColumnOrScalar value):
    pass
"""
    )

    assert "def take(value: Column | Scalar) -> Column" in result
    assert "value: ..." not in result


def test_correlated_fused_parameter_and_return_uses_typevar():
    result = _stubgen().convert_str(
        """
cdef class Column:
    pass

cdef class Table:
    pass

ctypedef fused ColumnOrTable:
    Column
    Table

cpdef ColumnOrTable empty_like(ColumnOrTable input):
    pass
"""
    )

    assert "from typing import TypeVar" in result
    assert "ColumnOrTable = TypeVar('ColumnOrTable', Column, Table)" in result
    assert "def empty_like(input: ColumnOrTable) -> ColumnOrTable" in result
    assert "input: ..." not in result
    assert "-> ..." not in result
