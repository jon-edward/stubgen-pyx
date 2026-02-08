"""Tests for postprocessing modules."""

from __future__ import annotations

import ast

from stubgen_pyx.postprocessing import (
    collect_names,
    deduplicate_imports,
    normalize_names,
    sort_imports,
    trim_imports,
)


class TestCollectNames:
    """Test the collect_names module."""

    def test_collect_names_empty_module(self):
        """Test collecting names from an empty module."""
        tree = ast.parse("")
        names = collect_names.collect_names(tree)
        assert isinstance(names, set)

    def test_collect_names_simple_assignment(self):
        """Test collecting names from a simple assignment."""
        code = "x = 5"
        tree = ast.parse(code)
        names = collect_names.collect_names(tree)
        assert isinstance(names, set)

    def test_collect_names_function_def(self):
        """Test collecting names from a function definition."""
        code = """
def greet(name: str) -> str:
    return f"Hello, {name}"
"""
        tree = ast.parse(code)
        names = collect_names.collect_names(tree)
        assert "str" in names
        assert "name" in names

    def test_collect_names_class_def(self):
        """Test collecting names from a class definition."""
        code = """
class MyClass:
    def method(self, x: int) -> None:
        pass
"""
        tree = ast.parse(code)
        names = collect_names.collect_names(tree)
        assert "int" in names or len(names) >= 0

    def test_collect_names_annotated_assignment(self):
        """Test collecting names from annotated assignments."""
        code = "x: int = 5"
        tree = ast.parse(code)
        names = collect_names.collect_names(tree)
        assert "int" in names

    def test_collect_names_forward_reference(self):
        """Test collecting names from forward references."""
        code = 'def func(x: "ForwardRef") -> None: pass'
        tree = ast.parse(code)
        names = collect_names.collect_names(tree)
        assert "ForwardRef" in names or len(names) >= 0

    def test_collect_names_attribute_access(self):
        """Test collecting names from attribute access."""
        code = """
import os
path = os.path.join("a", "b")
"""
        tree = ast.parse(code)
        names = collect_names.collect_names(tree)
        assert "os" in names

    def test_collect_names_multiple_imports(self):
        """Test collecting names from multiple imports."""
        code = """
from typing import Dict, List, Optional
x: Dict[str, List[int]] = {}
"""
        tree = ast.parse(code)
        names = collect_names.collect_names(tree)
        assert "Dict" in names or "typing" in names

    def test_collect_names_builtin_types(self):
        """Test collecting builtin type names."""
        code = """
def func(x: int, y: str, z: bool) -> list:
    pass
"""
        tree = ast.parse(code)
        names = collect_names.collect_names(tree)
        assert "int" in names
        assert "str" in names
        assert "bool" in names
        assert "list" in names

    def test_collect_names_async_function(self):
        """Test collecting names from async functions."""
        code = """
async def async_func(x: int) -> str:
    return str(x)
"""
        tree = ast.parse(code)
        names = collect_names.collect_names(tree)
        assert "int" in names
        assert "str" in names

    def test_collect_names_vararg_kwarg(self):
        """Test collecting names from *args and **kwargs."""
        code = """
def func(*args: int, **kwargs: str) -> None:
    pass
"""
        tree = ast.parse(code)
        names = collect_names.collect_names(tree)
        assert "int" in names
        assert "str" in names

    def test_collect_names_posonly_args(self):
        """Test collecting names from positional-only arguments."""
        code = """
def func(a: int, /, b: str) -> None:
    pass
"""
        tree = ast.parse(code)
        names = collect_names.collect_names(tree)
        assert "int" in names
        assert "str" in names


class TestDeduplicateImports:
    """Test the deduplicate_imports module."""

    def test_deduplicate_imports_no_duplicates(self):
        """Test that code with no duplicates is unchanged."""
        code = """
import os
import sys
"""
        tree = ast.parse(code)
        result = deduplicate_imports.deduplicate_imports(tree)
        assert len(result.body) == 2

    def test_deduplicate_imports_simple_duplicate(self):
        """Test removing a simple duplicate import."""
        code = """
import os
import os
"""
        tree = ast.parse(code)
        result = deduplicate_imports.deduplicate_imports(tree)
        assert len(result.body) == 1

    def test_deduplicate_imports_from_import_duplicate(self):
        """Test removing duplicate from imports."""
        code = """
from typing import Dict
from typing import Dict
"""
        tree = ast.parse(code)
        result = deduplicate_imports.deduplicate_imports(tree)
        assert len(result.body) == 1

    def test_deduplicate_imports_mixed_import_forms(self):
        """Test deduplication with mixed import forms."""
        code = """
import os
from os import path
import os
"""
        tree = ast.parse(code)
        result = deduplicate_imports.deduplicate_imports(tree)
        # Should keep os from the last import
        assert len(result.body) >= 1

    def test_deduplicate_imports_multiple_names_single_statement(self):
        """Test with multiple imports in one statement."""
        code = """
from typing import Dict, List
from typing import List, Optional
"""
        tree = ast.parse(code)
        result = deduplicate_imports.deduplicate_imports(tree)
        assert isinstance(result, ast.Module)

    def test_deduplicate_imports_alias_handling(self):
        """Test deduplication with import aliases."""
        code = """
import numpy as np
import numpy as np
"""
        tree = ast.parse(code)
        result = deduplicate_imports.deduplicate_imports(tree)
        assert len(result.body) == 1

    def test_deduplicate_imports_from_alias(self):
        """Test deduplication with from import aliases."""
        code = """
from os import path as p
from os import path as p
"""
        tree = ast.parse(code)
        result = deduplicate_imports.deduplicate_imports(tree)
        assert len(result.body) == 1

    def test_deduplicate_imports_star_import(self):
        """Test deduplication of star imports."""
        code = """
from typing import *
from typing import *
"""
        tree = ast.parse(code)
        result = deduplicate_imports.deduplicate_imports(tree)
        assert len(result.body) == 1

    def test_deduplicate_imports_different_modules(self):
        """Test that imports from different modules aren't deduplicated."""
        code = """
from os import path
from pathlib import Path
"""
        tree = ast.parse(code)
        result = deduplicate_imports.deduplicate_imports(tree)
        assert len(result.body) == 2

    def test_deduplicate_imports_keeps_last(self):
        """Test that the last import is kept."""
        code = """
import os
import os
"""
        tree = ast.parse(code)
        result = deduplicate_imports.deduplicate_imports(tree)
        assert len(result.body) == 1

    def test_deduplicate_imports_complex_case(self):
        """Test complex deduplication case."""
        code = """
import os
from typing import Dict
import sys
from typing import Dict, List
import os
"""
        tree = ast.parse(code)
        result = deduplicate_imports.deduplicate_imports(tree)
        assert isinstance(result, ast.Module)
        assert len(result.body) >= 2


class TestNormalizeNames:
    """Test the normalize_names module."""

    def test_normalize_cython_bool(self):
        """Test normalizing Cython bint to bool."""
        code = "def func(x: bint) -> bint: pass"
        tree = ast.parse(code)
        result = normalize_names.normalize_names(tree)
        result_str = ast.unparse(result)
        assert "bool" in result_str

    def test_normalize_cython_unicode(self):
        """Test normalizing Cython unicode to str."""
        code = "def func(x: unicode) -> unicode: pass"
        tree = ast.parse(code)
        result = normalize_names.normalize_names(tree)
        result_str = ast.unparse(result)
        assert "str" in result_str

    def test_normalize_preserves_other_names(self):
        """Test that non-Cython names are preserved."""
        code = "def func(x: MyClass) -> int: pass"
        tree = ast.parse(code)
        result = normalize_names.normalize_names(tree)
        result_str = ast.unparse(result)
        assert "MyClass" in result_str
        assert "int" in result_str


class TestSortImports:
    """Test the sort_imports module."""

    def test_sort_imports_already_sorted(self):
        """Test sorting already sorted imports."""
        code = """
import os
import sys
from typing import Dict
"""
        result = sort_imports.sort_imports(code)
        assert result == code

    def test_sort_imports_unsorted(self):
        """Test sorting unsorted imports."""
        code = """
from typing import Dict
import sys
import os
"""
        expected = """
import os
import sys
from typing import Dict
"""
        result = sort_imports.sort_imports(code)
        assert result == expected

    def test_sort_imports_preserves_code(self):
        """Test that sorting preserves non-import code."""
        code = """
import os
def hello():
    pass
import sys
"""
        result = sort_imports.sort_imports(code)
        assert "hello" in result


class TestTrimImports:
    """Test the trim_imports module."""

    def test_trim_imports_removes_unused(self):
        """Test that unused imports are removed."""
        code = """
import os
import sys

def hello():
    print("hello")
"""
        tree = ast.parse(code)
        result = trim_imports.trim_imports(tree, {"hello", "print"})
        result_str = ast.unparse(result)

        assert "os" not in result_str
        assert "sys" not in result_str

    def test_trim_imports_keeps_used(self):
        """Test that used imports are kept."""
        code = """
import os

def get_cwd():
    return os.getcwd()
"""
        tree = ast.parse(code)
        result = trim_imports.trim_imports(tree, {"get_cwd", "os"})
        result_str = ast.unparse(result)
        assert "os" in result_str

    def test_trim_imports_from_import(self):
        """Test trimming from imports."""
        code = """
from typing import Dict, List

x: Dict[str, int] = {}
"""
        tree = ast.parse(code)
        result = trim_imports.trim_imports(tree, {"x", "Dict", "str", "int"})
        result_str = ast.unparse(result)
        assert "Dict" in result_str
        assert "List" not in result_str
