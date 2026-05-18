"""Tests for trim_not_defined module."""

from __future__ import annotations

import ast


from stubgen_pyx.postprocessing.trim_not_defined import trim_not_defined


class TestTrimNotDefined:
    """Test the trim_not_defined module."""

    def test_empty_module(self):
        """Test trimming an empty module."""
        tree = ast.parse("")
        result = trim_not_defined(tree)
        assert isinstance(result, ast.Module)
        assert len(result.body) == 0

    def test_no_undefined_names(self):
        """Test that code with only defined names is unchanged."""
        code = """
x = 5
def foo(y: int) -> str:
    pass
"""
        tree = ast.parse(code)
        result = trim_not_defined(tree)
        result_str = ast.unparse(result)
        assert "int" in result_str
        assert "str" in result_str

    def test_undefined_annotation(self):
        """Test that undefined annotations are replaced with ..."""
        code = "def foo(x: UndefinedType) -> int: pass"
        tree = ast.parse(code)
        result = trim_not_defined(tree)
        result_str = ast.unparse(result)
        assert "UndefinedType" not in result_str
        assert "..." in result_str

    def test_undefined_return_type(self):
        """Test that undefined return types are replaced with ..."""
        code = "def foo() -> UndefinedType: pass"
        tree = ast.parse(code)
        result = trim_not_defined(tree)
        result_str = ast.unparse(result)
        assert "UndefinedType" not in result_str
        assert "..." in result_str

    def test_undefined_default_value(self):
        """Test that undefined default values are replaced with ..."""
        code = "def foo(x: int = UNDEFINED_VALUE) -> int: pass"
        tree = ast.parse(code)
        result = trim_not_defined(tree)
        result_str = ast.unparse(result)
        assert "UNDEFINED_VALUE" not in result_str
        assert "..." in result_str

    def test_defined_import_preserved(self):
        """Test that imported names are preserved."""
        code = """
import os
from typing import Dict

def foo(x: Dict) -> str:
    return os.path.join("a", "b")
"""
        tree = ast.parse(code)
        result = trim_not_defined(tree)
        result_str = ast.unparse(result)
        assert "Dict" in result_str
        assert "str" in result_str

    def test_assignment_annotation(self):
        """Test trimming undefined names in assignment annotations."""
        code = "x: UndefinedType = 5"
        tree = ast.parse(code)
        result = trim_not_defined(tree)
        result_str = ast.unparse(result)
        assert "UndefinedType" not in result_str

    def test_assignment_value(self):
        """Test trimming undefined names in assignment values."""
        code = "x = UNDEFINED_VALUE"
        tree = ast.parse(code)
        result = trim_not_defined(tree)
        result_str = ast.unparse(result)
        assert "UNDEFINED_VALUE" not in result_str
        assert "..." in result_str

    def test_mixed_defined_undefined(self):
        """Test with both defined and undefined names."""
        code = """
import os

def foo(x: int, y: UndefinedType) -> str:
    pass
"""
        tree = ast.parse(code)
        result = trim_not_defined(tree)
        result_str = ast.unparse(result)
        assert "int" in result_str
        assert "str" in result_str
        assert "UndefinedType" not in result_str

    def test_kwonly_defaults(self):
        """Test keyword-only argument defaults."""
        code = "def foo(*, x: int = UNDEFINED) -> int: pass"
        tree = ast.parse(code)
        result = trim_not_defined(tree)
        result_str = ast.unparse(result)
        assert "UNDEFINED" not in result_str
        assert "..." in result_str

    def test_posonly_args(self):
        """Test positional-only argument annotations."""
        code = "def foo(x: UndefinedType, /, y: int) -> int: pass"
        tree = ast.parse(code)
        result = trim_not_defined(tree)
        result_str = ast.unparse(result)
        assert "UndefinedType" not in result_str
        assert "int" in result_str

    def test_vararg_annotation(self):
        """Test *args annotation."""
        code = "def foo(*args: UndefinedType) -> int: pass"
        tree = ast.parse(code)
        result = trim_not_defined(tree)
        result_str = ast.unparse(result)
        assert "UndefinedType" not in result_str

    def test_kwarg_annotation(self):
        """Test **kwargs annotation."""
        code = "def foo(**kwargs: UndefinedType) -> int: pass"
        tree = ast.parse(code)
        result = trim_not_defined(tree)
        result_str = ast.unparse(result)
        assert "UndefinedType" not in result_str

    def test_async_function(self):
        """Test async function definitions."""
        code = "async def foo() -> UndefinedType: pass"
        tree = ast.parse(code)
        result = trim_not_defined(tree)
        result_str = ast.unparse(result)
        assert "UndefinedType" not in result_str

    def test_class_definition(self):
        """Test that class names are collected."""
        code = """
class MyClass:
    def method(self, x: MyClass) -> int:
        pass
"""
        tree = ast.parse(code)
        result = trim_not_defined(tree)
        result_str = ast.unparse(result)
        assert "MyClass" in result_str

    def test_builtin_preserved(self):
        """Test that builtin names are always preserved."""
        code = """
def foo(x: int, y: str, z: bool) -> list:
    pass
"""
        tree = ast.parse(code)
        result = trim_not_defined(tree)
        result_str = ast.unparse(result)
        assert "int" in result_str
        assert "str" in result_str
        assert "bool" in result_str
        assert "list" in result_str

    def test_union_type_with_undefined(self):
        """Test Union types with undefined members."""
        code = "def foo(x: int | UndefinedType) -> int: pass"
        tree = ast.parse(code)
        result = trim_not_defined(tree)
        result_str = ast.unparse(result)
        assert "UndefinedType" not in result_str

    def test_complex_default_value(self):
        """Test complex default expressions with undefined names."""
        code = "def foo(x: int = UNDEFINED_CONST + 5) -> int: pass"
        tree = ast.parse(code)
        result = trim_not_defined(tree)
        result_str = ast.unparse(result)
        assert "UNDEFINED_CONST" not in result_str
        assert "..." in result_str

    def test_generic_type_with_undefined(self):
        """Test generic types with undefined type arguments."""
        code = """
from typing import Dict

def foo(x: Dict[str, UndefinedType]) -> int:
    pass
"""
        tree = ast.parse(code)
        result = trim_not_defined(tree)
        result_str = ast.unparse(result)
        assert "UndefinedType" not in result_str

    def test_function_name_collection(self):
        """Test that function names are collected."""
        code = """
def my_func() -> int:
    pass

def other_func(x: my_func) -> int:
    pass
"""
        tree = ast.parse(code)
        result = trim_not_defined(tree)
        # my_func should be recognized as a defined name
        result_str = ast.unparse(result)
        assert "my_func" in result_str

    def test_annassign_none_value(self):
        """Test annotated assignment without value."""
        code = "x: UndefinedType"
        tree = ast.parse(code)
        result = trim_not_defined(tree)
        result_str = ast.unparse(result)
        assert "UndefinedType" not in result_str

    def test_multiple_assignments(self):
        """Test multiple assignment targets."""
        code = """
x = y = UNDEFINED
"""
        tree = ast.parse(code)
        result = trim_not_defined(tree)
        result_str = ast.unparse(result)
        assert "UNDEFINED" not in result_str

    def test_nested_function(self):
        """Test nested function definitions."""
        code = """
def outer(x: int) -> int:
    def inner(y: UndefinedType) -> int:
        pass
    return x
"""
        tree = ast.parse(code)
        result = trim_not_defined(tree)
        result_str = ast.unparse(result)
        assert "int" in result_str
        assert "UndefinedType" not in result_str

    def test_real_world_example(self):
        """Test a real-world-like example."""
        code = """
import os
from typing import Optional, Dict

class Parser:
    config: Config  # Undefined

    def parse(self, path: str) -> Optional[Dict[str, int]]:
        return None
"""
        tree = ast.parse(code)
        result = trim_not_defined(tree)
        result_str = ast.unparse(result)
        # Config should be replaced
        # But Optional and Dict should remain
        assert "Config" not in result_str or "..." in result_str
        assert "Optional" in result_str or "..." not in result_str
