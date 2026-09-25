"""Additional tests for parsing edge cases."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from Cython.Compiler.Errors import CompileError

from stubgen_pyx.parsing.context import StubgenContext
from stubgen_pyx.parsing.parser import parse_file


class TestParsingEdgeCases:
    """Test edge cases in parsing."""

    def test_parse_file_with_syntax_error(self):
        """Test parsing a file with syntax errors.

        A real syntax error (as opposed to a declaration-level issue like
        an unresolved `cimport`) surfaces as `Context.parse`'s own
        error-count check raising -- not collected into
        `ParsedSource.diagnostics` the way `AnalyseDeclarationsTransform`
        issues are. See `parsing/pipeline.py::run_stub_pipeline`.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            pyx_file = Path(tmpdir) / "bad_syntax.pyx"
            pyx_file.write_text("def broken( pass")

            with pytest.raises(CompileError):
                parse_file(pyx_file, StubgenContext())

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
            result = parse_file(pyx_file, StubgenContext())
            assert result is not None
            assert result.diagnostics == []

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
            result = parse_file(pyx_file, StubgenContext())
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
            result = parse_file(pyx_file, StubgenContext())
            assert result is not None
            # `numpy` isn't a real cimport here (it's a plain Python
            # `import`), and the rest resolve natively (`cython` is a
            # pseudo-module, `cpython.mem` is under Cython's own
            # standard include path) -- no diagnostics expected.
            assert result.diagnostics == []

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
            result = parse_file(pyx_file, StubgenContext())
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
            result = parse_file(pyx_file, StubgenContext())
            assert result is not None

    def test_parse_file_with_decorators(self):
        """Test parsing file with various decorators."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pyx_file = Path(tmpdir) / "decorated.pyx"
            pyx_file.write_text("""
class Plain:
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
            result = parse_file(pyx_file, StubgenContext())
            assert result is not None

    def test_parse_file_with_builtin_types(self):
        """Test parsing file with builtin type annotations.

        Named `builtin_types.pyx`, not `builtins.pyx`: the latter collides
        with Cython's own reserved `builtins` module scope during
        qualified-name resolution (`Context.find_module` resolves it to a
        `BuiltinScope`, not an ordinary `ModuleScope`) -- an inherent
        property of using Cython's real module-name machinery, not
        something specific to this project. A real Cython project with a
        top-level file literally named `builtins.pyx` would hit the same
        collision with the real compiler.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            pyx_file = Path(tmpdir) / "builtin_types.pyx"
            pyx_file.write_text("""
def func(x: int, y: str, z: bool) -> list:
    pass
""")
            result = parse_file(pyx_file, StubgenContext())
            assert result is not None


class TestRealFileFormatting:
    """Parsing real files with varied whitespace/line-ending styles.

    There's no preprocessing step to normalize these -- these exercise
    Cython's own scanner handling the raw file directly, and
    `ParsedSource.source` is exactly the file's own text, unmodified.
    """

    def test_parse_file_with_windows_line_endings(self):
        """Test parsing a file with Windows line endings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pyx_file = Path(tmpdir) / "windows.pyx"
            pyx_file.write_bytes(b"def hello():\r\n    pass\r\n")
            result = parse_file(pyx_file, StubgenContext())
            assert result is not None

    def test_parse_file_with_tabs(self):
        """Test parsing a file with tab indentation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pyx_file = Path(tmpdir) / "tabs.pyx"
            pyx_file.write_text("""
def hello():
\tpass
""")
            result = parse_file(pyx_file, StubgenContext())
            assert result is not None

    def test_parse_file_source_is_unmodified(self):
        """`ParsedSource.source` is exactly the file's own text -- no preprocessing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pyx_file = Path(tmpdir) / "semicolon.pyx"
            code = """
class Test:
    def __init__(self):
        pass;
    def func(self):
        pass
"""
            pyx_file.write_text(code)

            result = parse_file(pyx_file, StubgenContext())
            assert result.source == code
