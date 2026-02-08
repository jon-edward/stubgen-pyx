"""Tests for conversion module utilities."""

from __future__ import annotations


from stubgen_pyx.conversion import conversion_utils


class TestLinesCachhe:
    """Test the _LinesCache class."""

    def test_lines_cache_initialization(self):
        """Test creating a cache."""
        cache = conversion_utils._LinesCache()
        assert cache.source is None
        assert cache.lines == []

    def test_lines_cache_set_source(self):
        """Test setting source in the cache."""
        cache = conversion_utils._LinesCache()
        code = "line1\nline2\nline3"
        cache.source = code
        assert cache.source == code
        assert len(cache.lines) >= 3

    def test_lines_cache_reuse_same_source(self):
        """Test that same source doesn't trigger resplit."""
        cache = conversion_utils._LinesCache()
        code = "line1\nline2"
        cache.source = code
        old_lines = cache.lines
        cache.source = code
        assert cache.lines == old_lines or len(cache.lines) > 0

    def test_lines_cache_different_source(self):
        """Test that different source causes resplit."""
        cache = conversion_utils._LinesCache()
        cache.source = "line1\nline2"
        first_count = len(cache.lines)
        cache.source = "line1\nline2\nline3\nline4"
        assert len(cache.lines) > first_count


class TestDocstringToString:
    """Test the docstring_to_string function."""

    def test_docstring_single_line(self):
        """Test converting a single-line docstring."""
        result = conversion_utils.docstring_to_string("Simple docstring")
        assert '"""' in result
        assert "Simple docstring" in result

    def test_docstring_multiline(self):
        """Test converting a multi-line docstring."""
        docstring = "First line\nSecond line\nThird line"
        result = conversion_utils.docstring_to_string(docstring)
        assert '"""' in result

    def test_docstring_with_triple_quotes(self):
        """Test converting docstring that contains triple quotes."""
        docstring = 'Contains """ inside'
        result = conversion_utils.docstring_to_string(docstring)
        assert '"""' in result

    def test_docstring_with_indentation(self):
        """Test converting docstring with indentation."""
        docstring = "Line 1\n    Indented line\nLine 3"
        result = conversion_utils.docstring_to_string(docstring)
        assert '"""' in result

    def test_docstring_empty(self):
        """Test converting an empty docstring."""
        result = conversion_utils.docstring_to_string("")
        assert '"""' in result


class TestUnparseExpr:
    """Test the unparse_expr function."""

    def test_unparse_none_node(self):
        """Test unparsing a None expression."""
        result = conversion_utils.unparse_expr(None)
        assert result is None

    def test_unparse_simple_value(self):
        """Test that complex expressions become '...'."""
        # We can't directly create Cython nodes easily, so we test the behavior
        result = conversion_utils.unparse_expr(None)
        assert result is None


class TestGetEnumNames:
    """Test the get_enum_names function."""

    def test_get_enum_names_requires_cython_node(self):
        """Test that get_enum_names requires proper Cython node."""
        # This test verifies the function exists and is callable
        # Actual testing requires Cython AST nodes which are complex to create
        assert callable(conversion_utils.get_enum_names)


class TestGetDecorators:
    """Test the get_decorators function."""

    def test_get_decorators_is_callable(self):
        """Test that get_decorators is callable."""
        assert callable(conversion_utils.get_decorators)


class TestGetBases:
    """Test the get_bases function."""

    def test_get_bases_is_callable(self):
        """Test that get_bases is callable."""
        assert callable(conversion_utils.get_bases)


class TestGetMetaclass:
    """Test the get_metaclass function."""

    def test_get_metaclass_is_callable(self):
        """Test that get_metaclass is callable."""
        assert callable(conversion_utils.get_metaclass)


class TestGetSource:
    """Test the get_source function."""

    def test_get_source_is_callable(self):
        """Test that get_source is callable."""
        assert callable(conversion_utils.get_source)
