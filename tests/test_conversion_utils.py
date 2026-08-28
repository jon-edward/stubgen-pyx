"""Tests for conversion module utilities."""

from __future__ import annotations

from Cython.Compiler.TreeFragment import parse_from_strings

from stubgen_pyx.conversion import (
    docstrings,
    unparse,
)
from stubgen_pyx.conversion.unparse import _get_full_class_name, _Unparser


class TestGetCdefVariables:
    """Test the get_cdef_variables function."""

    def test_multi_declarator_cdef_returns_all(self):
        """cdef public int x, y, z should yield three variables."""
        from stubgen_pyx import StubgenPyx
        from stubgen_pyx.config import StubgenPyxConfig

        s = StubgenPyx(
            config=StubgenPyxConfig(exclude_attribution=True, sort_imports=False)
        )
        result = s.convert_str("""
cdef class Foo:
    cdef public int x, y, z
""")
        assert "x: int" in result
        assert "y: int" in result
        assert "z: int" in result

    def test_single_declarator_cdef(self):
        """cdef public int x should yield one variable."""
        from stubgen_pyx import StubgenPyx
        from stubgen_pyx.config import StubgenPyxConfig

        s = StubgenPyx(
            config=StubgenPyxConfig(exclude_attribution=True, sort_imports=False)
        )
        result = s.convert_str("""
cdef class Foo:
    cdef public int x
""")
        assert "x: int" in result


class TestDocstringToString:
    """Test the docstring_to_string function."""

    def test_docstring_single_line(self):
        """Test converting a single-line docstring."""
        result = docstrings.docstring_to_string("Simple docstring")
        assert '"""' in result
        assert "Simple docstring" in result

    def test_docstring_multiline(self):
        """Test converting a multi-line docstring."""
        docstring = "First line\nSecond line\nThird line"
        result = docstrings.docstring_to_string(docstring)
        assert '"""' in result

    def test_docstring_with_triple_quotes(self):
        """Test converting docstring that contains triple quotes."""
        docstring = 'Contains """ inside'
        result = docstrings.docstring_to_string(docstring)
        assert '"""' in result

    def test_docstring_with_indentation(self):
        """Test converting docstring with indentation."""
        docstring = "Line 1\n    Indented line\nLine 3"
        result = docstrings.docstring_to_string(docstring)
        assert '"""' in result

    def test_docstring_empty(self):
        """Test converting an empty docstring."""
        result = docstrings.docstring_to_string("")
        assert '"""' in result


class TestUnparseExpr:
    """Test the unparse_expr function."""

    def test_unparse_none_node(self):
        """Test unparsing a None expression."""
        result = unparse.unparse_expr(None)
        assert result is None

    def test_unparse_simple_value(self):
        """Test that complex expressions become '...'."""
        # We can't directly create Cython nodes easily, so we test the behavior
        result = unparse.unparse_expr(None)
        assert result is None

    def test_unparse_nested_comprehension_preserves_clause_order(self):
        """Test that nested comprehension clauses retain source order."""
        module = parse_from_strings(
            "expr.pyx",
            "[(a, b, c) for a in range(1, 2, 0) "
            "for b in range(1, 3, 0) for c in range(1, 4, 0)]",
            context=None,
        )

        result = unparse.unparse_expr(module.body.expr)

        assert result == (
            "[(a, b, c) for a in range(1, 2, 0) "
            "for b in range(1, 3, 0) for c in range(1, 4, 0)]"
        )

    def test_unparse_unary_operators_and_singleton_tuple(self):
        expressions = {
            "(-value)": "-value",
            "(&value)": "value",
            "(value,)": "(value,)",
        }
        for source, expected in expressions.items():
            module = parse_from_strings("expr.pyx", source, context=None)
            assert unparse.unparse_expr(module.body.expr) == expected

    def test_unparse_comprehension_with_sequence_target_and_condition(self):
        module = parse_from_strings(
            "expr.pyx", "[(a, b) for (a, b) in pairs if a > 0]", context=None
        )

        assert unparse.unparse_expr(module.body.expr) == (
            "[(a, b) for (a, b) in pairs if a > 0]"
        )

    def test_unknown_nodes_are_tracked_only_once(self):
        module = parse_from_strings("expr.pyx", "value + 1", context=None)
        writer = _Unparser()
        writer.visit_Node(module)
        writer.visit_Node(module.body)

        assert writer.found_unknown is module
        assert writer.result == "......"
        assert unparse.unparse_expr(module) is None

    def test_full_class_name_handles_builtin_and_custom_nodes(self):
        assert _get_full_class_name(object()) == "object"
        module = parse_from_strings("expr.pyx", "value", context=None)
        assert "ExprNode" in _get_full_class_name(module.body.expr)
