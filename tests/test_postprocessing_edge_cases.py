"""Additional edge case tests for postprocessing and conversion."""

from __future__ import annotations

import ast


from stubgen_pyx.postprocessing import (
    sort_imports,
    trim_imports,
)


class TestSortImportsEdgeCases:
    """Test edge cases in import sorting."""

    def test_sort_empty_module(self):
        """Test sorting empty module."""
        code = ""
        result = sort_imports.sort_imports(code)
        assert not result

    def test_sort_imports_with_code(self):
        """Test sorting imports mixed with code."""
        code = """
x = 1
import os

y = 2
import sys
"""
        expected = """
x = 1
import os

y = 2
import sys
"""
        result = sort_imports.sort_imports(code)
        assert result == expected

    def test_sort_imports_multiline_imports(self):
        """Test sorting multiline import statements."""
        code = """
from typing import (
    Dict,
    List,
    Optional,
)
import os
"""
        expected = """
import os
from typing import Dict, List, Optional
"""
        result = sort_imports.sort_imports(code)
        assert result == expected


class TestTrimImportsEdgeCases:
    """Test edge cases in import trimming."""

    def test_trim_imports_all_used(self):
        """Test when all imports are used."""
        code = """
import os
import sys

x = os.getcwd()
y = sys.version
"""
        tree = ast.parse(code)
        result = trim_imports.trim_imports(
            tree, {"x", "y", "getcwd", "version", "sys", "os"}
        )
        result_str = ast.unparse(result)

        assert "os" in result_str
        assert "sys" in result_str

    def test_trim_imports_none_used(self):
        """Test when no imports are used."""
        code = """
import os
import sys

x = 1
"""
        tree = ast.parse(code)
        result = trim_imports.trim_imports(tree, {"x"})
        result_str = ast.unparse(result)

        assert "os" not in result_str
        assert "sys" not in result_str

    def test_trim_imports_with_wildcard(self):
        """Test trimming with wildcard imports."""
        code = """
from typing import *

x: Dict[str, int] = {}
"""
        tree = ast.parse(code)
        result = trim_imports.trim_imports(tree, {"x", "Dict", "str", "int"})
        result_str = ast.unparse(result)
        assert "Dict" in result_str
        assert "typing" in result_str


class TestSignatureExtraction:
    """Test signature extraction edge cases."""

    def test_signature_with_many_args(self):
        """Test with many arguments."""
        code = """
def func(a, b, c, d, e, f, g, h, i, j):
    pass
"""
        tree = ast.parse(code)
        func_def = tree.body[0]
        assert isinstance(func_def, ast.FunctionDef)
        assert len(func_def.args.args) == 10
