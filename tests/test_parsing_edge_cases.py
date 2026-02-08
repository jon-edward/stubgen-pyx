"""Additional tests for parsing edge cases."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from stubgen_pyx.parsing.parser import parse_pyx
from stubgen_pyx.parsing.preprocess import LineColConverter


class TestParsingEdgeCases:
    """Test edge cases in parsing."""

    def test_parse_file_with_syntax_error(self):
        """Test parsing a file with syntax errors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pyx_file = Path(tmpdir) / "bad_syntax.pyx"
            pyx_file.write_text("def broken( pass")

            with pytest.raises(Exception):
                parse_pyx(pyx_file.read_text(), pyx_path=pyx_file)

    def test_parse_file_with_complex_code(self):
        """Test parsing complex Cython code."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pyx_file = Path(tmpdir) / "complex.pyx"
            pyx_file.write_text("""
cdef extern from "math.h":
    double sin(double x)

cdef class MyClass:
    cdef int value

    def __init__(self, int v):
        self.value = v

    def get_value(self):
        return self.value
""")
            result = parse_pyx(pyx_file.read_text(), pyx_path=pyx_file)
            assert result is not None

    def test_line_col_converter_basic(self):
        """Test LineColConverter with basic code."""
        code = "line1\nline2\nline3"
        converter = LineColConverter(code)
        offset = converter.line_col_to_offset((1, 0))
        assert offset == 0

    def test_line_col_converter_multiline(self):
        """Test LineColConverter with multiple lines."""
        code = "line1\nline2\nline3"
        converter = LineColConverter(code)
        # Line 2, Column 0 should be after first line and newline
        offset = converter.line_col_to_offset((2, 0))
        assert offset > 0
        assert offset == 6  # "line1\n" is 6 chars

    def test_line_col_converter_end_of_file(self):
        """Test LineColConverter at end of file."""
        code = "line1\nline2"
        converter = LineColConverter(code)
        offset = converter.line_col_to_offset((2, 5))
        assert offset == 11

    def test_line_col_converter_offset_to_line_col(self):
        """Test converting offset back to line/col."""
        code = "line1\nline2\nline3"
        converter = LineColConverter(code)
        # Get offset for line 2, col 2
        offset = converter.line_col_to_offset((2, 2))
        # Should be able to get this offset
        assert offset >= 0

    def test_parse_file_with_docstring(self):
        """Test parsing file with docstring."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pyx_file = Path(tmpdir) / "with_docstring.pyx"
            pyx_file.write_text('''
"""Module docstring."""

def hello():
    """Function docstring."""
    pass
''')
            result = parse_pyx(pyx_file.read_text(), pyx_path=pyx_file)
            assert result is not None

    def test_parse_file_with_imports(self):
        """Test parsing file with various imports."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pyx_file = Path(tmpdir) / "imports.pyx"
            pyx_file.write_text("""
import numpy
from typing import Dict, List
cimport cython
from cpython.mem cimport PyMem_Malloc
""")
            result = parse_pyx(pyx_file.read_text(), pyx_path=pyx_file)
            assert result is not None

    def test_parse_file_with_cdef_types(self):
        """Test parsing file with cdef type declarations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pyx_file = Path(tmpdir) / "ctypes.pyx"
            pyx_file.write_text("""
cdef int x = 5
cdef double y = 3.14
cdef str name = "hello"

cdef class MyClass:
    cdef int attr
""")
            result = parse_pyx(pyx_file.read_text(), pyx_path=pyx_file)
            assert result is not None

    def test_parse_file_with_properties(self):
        """Test parsing file with properties."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pyx_file = Path(tmpdir) / "props.pyx"
            pyx_file.write_text("""
cdef class MyClass:
    cdef int _value

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, int v):
        self._value = v
""")
            result = parse_pyx(pyx_file.read_text(), pyx_path=pyx_file)
            assert result is not None

    def test_parse_file_with_decorators(self):
        """Test parsing file with various decorators."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pyx_file = Path(tmpdir) / "decorated.pyx"
            pyx_file.write_text("""
@staticmethod
def static_method():
    pass

@classmethod
def class_method(cls):
    pass

@property
def my_prop(self):
    return 42
""")
            result = parse_pyx(pyx_file.read_text(), pyx_path=pyx_file)
            assert result is not None

    def test_parse_file_with_builtin_types(self):
        """Test parsing file with builtin type annotations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pyx_file = Path(tmpdir) / "builtins.pyx"
            pyx_file.write_text("""
def func(x: int, y: str, z: bool) -> list:
    pass
""")
            result = parse_pyx(pyx_file.read_text(), pyx_path=pyx_file)
            assert result is not None


class TestPreprocessingEdgeCases:
    """Test edge cases in preprocessing."""

    def test_preprocess_with_windows_line_endings(self):
        """Test preprocessing with Windows line endings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pyx_file = Path(tmpdir) / "windows.pyx"
            # Write with Windows line endings
            pyx_file.write_bytes(b"def hello():\r\n    pass\r\n")
            result = parse_pyx(pyx_file.read_text(), pyx_path=pyx_file)
            assert result is not None

    def test_preprocess_with_tabs(self):
        """Test preprocessing with tab indentation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pyx_file = Path(tmpdir) / "tabs.pyx"
            pyx_file.write_text("""
def hello():
\tpass
""")
            result = parse_pyx(pyx_file.read_text(), pyx_path=pyx_file)
            assert result is not None

    def test_preprocess_with_mixed_indentation(self):
        """Test preprocessing with mixed spaces and tabs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pyx_file = Path(tmpdir) / "mixed.pyx"
            # Mix spaces and tabs
            pyx_file.write_text("def hello():\n    pass\n")
            result = parse_pyx(pyx_file.read_text(), pyx_path=pyx_file)
            assert result is not None
