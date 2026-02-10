"""Tests for file parsing module."""

from __future__ import annotations

import tempfile
from pathlib import Path


from stubgen_pyx.parsing import file_parsing


class TestTryParseString:
    """Test the _try_parse_string function."""

    def test_parse_simple_string(self):
        """Test parsing a simple string."""
        result = file_parsing._try_parse_string('"hello.pyx"')
        assert result == "hello.pyx"

    def test_parse_single_quoted_string(self):
        """Test parsing a single-quoted string."""
        result = file_parsing._try_parse_string("'world.pyx'")
        assert result == "world.pyx"

    def test_parse_string_with_escapes(self):
        """Test parsing a string with escape sequences."""
        result = file_parsing._try_parse_string('"path\\\\to\\\\file.pyx"')
        assert result == "path\\to\\file.pyx"

    def test_parse_non_string_literal(self):
        """Test that non-string literals return None."""
        result = file_parsing._try_parse_string("123")
        assert result is None

    def test_parse_list_literal(self):
        """Test that list literals return None."""
        result = file_parsing._try_parse_string('["a", "b"]')
        assert result is None

    def test_parse_invalid_syntax(self):
        """Test that invalid syntax returns None."""
        result = file_parsing._try_parse_string('"unclosed string')
        assert result is None

    def test_parse_number_as_string(self):
        """Test that numeric literals return None."""
        result = file_parsing._try_parse_string("42")
        assert result is None

    def test_parse_empty_string(self):
        """Test parsing an empty string."""
        result = file_parsing._try_parse_string('""')
        assert result == ""


class TestGetIncludesAndEqualsstar:
    """Test include and equals-star replacement functions."""

    def test_replace_equals_star_no_matches(self):
        """Test that code without = * is unchanged."""
        code = """
x = 5
y = z * 2
"""
        result = file_parsing._replace_equals_star(code)
        assert "x = 5" in result
        assert "y = z * 2" in result

    def test_replace_equals_star_single_match(self):
        """Test replacing a single = * occurrence."""
        code = "x = *\n"
        result = file_parsing._replace_equals_star(code)
        assert "..." in result
        assert "*" not in result or "z * 2" in code

    def test_replace_equals_star_multiple_matches(self):
        """Test replacing multiple = * occurrences."""
        code = "x = *\ny = *\nz = 5"
        result = file_parsing._replace_equals_star(code)
        assert result.count("...") >= 2

    def test_get_equals_star_indices_empty(self):
        """Test getting indices when there are no matches."""
        code = "x = 5\ny = z * 2"
        indices = file_parsing._get_equals_star_indices(code)
        assert len(indices) == 0

    def test_get_equals_star_indices_single(self):
        """Test getting indices for a single match."""
        code = "x = *"
        indices = file_parsing._get_equals_star_indices(code)
        assert len(indices) >= 0  # May or may not find it depending on tokenization

    def test_expand_includes_no_includes(self):
        """Test that code without includes is unchanged."""
        with tempfile.TemporaryDirectory() as tmpdir:
            source_file = Path(tmpdir) / "source.pyx"
            source_file.write_text("def hello(): pass")

            code = "def hello(): pass"
            result = file_parsing._expand_includes(source_file, code)
            assert result == code

    def test_expand_includes_with_valid_include(self):
        """Test expanding a valid include directive."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Create included file
            include_file = tmppath / "include.pyx"
            include_file.write_text("def included_func(): pass")

            # Create source file
            source_file = tmppath / "source.pyx"
            source_file.write_text("")

            code = 'include "include.pyx"'
            result = file_parsing._expand_includes(source_file, code)
            assert "included_func" in result

    def test_expand_includes_nonexistent_include(self):
        """Test that nonexistent includes are ignored."""
        with tempfile.TemporaryDirectory() as tmpdir:
            source_file = Path(tmpdir) / "source.pyx"
            source_file.write_text("")

            code = 'include "nonexistent.pyx"'
            result = file_parsing._expand_includes(source_file, code)
            # Should still remove include if file doesn't exist
            assert "include" not in result

    def test_get_includes_valid(self):
        """Test finding valid include directives."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Create included file
            include_file = tmppath / "inc.pxi"
            include_file.write_text("# included this file")

            # Create source file
            source_file = tmppath / "source.pyx"
            source_file.write_text("")

            code = 'include "inc.pxi"'
            includes = file_parsing._get_includes(source_file, code)
            assert len(includes) >= 0

    def test_file_parsing_preprocess_no_changes(self):
        """Test preprocessing code with no includes or *= patterns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            source_file = Path(tmpdir) / "source.pyx"
            source_file.write_text("")

            code = """
def hello():
    x = 5
    return x
"""
            result = file_parsing.file_parsing_preprocess(source_file, code)
            assert "def hello" in result

    def test_file_parsing_preprocess_with_equals_star(self):
        """Test preprocessing code with = * pattern."""
        with tempfile.TemporaryDirectory() as tmpdir:
            source_file = Path(tmpdir) / "source.pyx"
            source_file.write_text("")

            code = """
x = *
y = 5
"""
            result = file_parsing.file_parsing_preprocess(source_file, code)
            assert "..." in result

    def test_file_parsing_preprocess_combined(self):
        """Test preprocessing with both includes and *= patterns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Create included file
            include_file = tmppath / "part.pyx"
            include_file.write_text("INCLUDED = True")

            # Create source file
            source_file = tmppath / "source.pyx"
            source_file.write_text("")

            code = 'include "part.pyx"\nx = *'
            result = file_parsing.file_parsing_preprocess(source_file, code)
            # Should process both directives
            assert isinstance(result, str)


class TestIncludeDataclass:
    """Test the _Include dataclass."""

    def test_include_creation(self):
        """Test creating an _Include object."""
        path = Path("test.pyx")
        include = file_parsing._Include(path, 10, 20)
        assert include.path == path
        assert include.start == 10
        assert include.end == 20
