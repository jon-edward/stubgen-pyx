"""Tests for CLI module."""

from __future__ import annotations

from stubgen_pyx.parsing.parser import parse_pyx
from stubgen_pyx.conversion.converter import Converter
from stubgen_pyx.analysis.visitor import ModuleVisitor


class TestFusedTypes:
    """Tests for fused types support in stubgen-pyx."""

    def test_fused_types(self):
        converter = Converter()

        pr = parse_pyx("""\
cdef fused Foo:
    int
    double

cdef struct Bar:
    int x
    double y

ctypedef fused OtherFoo:
    Bar
    double
""")

        module = converter.convert_module(
            ModuleVisitor(pr.source_ast), pr.source, pr.type_comments
        )

        assert module.scope.fused_types[0].concrete_types == ["int", "double"]
        assert module.scope.fused_types[1].concrete_types == ["Bar", "double"]
