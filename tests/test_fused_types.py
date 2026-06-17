from __future__ import annotations

from stubgen_pyx.analysis.visitor import ScopeVisitor
from stubgen_pyx.parsing.parser import parse_pyx


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
