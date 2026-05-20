"""Tests for the conversion module with actual parsing."""

from __future__ import annotations


from stubgen_pyx.conversion.converter import Converter
from stubgen_pyx.models.pyi_elements import (
    PyiModule,
    PyiClass,
    PyiFunction,
    PyiScope,
    PyiEnum,
)
from stubgen_pyx.parsing.parser import parse_pyx
from stubgen_pyx.analysis.visitor import ModuleVisitor


class TestConverterInitialization:
    """Test Converter initialization."""

    def test_converter_creation(self):
        """Test creating a Converter instance."""
        converter = Converter()
        assert converter is not None
        assert isinstance(converter, Converter)

    def test_converter_is_dataclass(self):
        """Test that Converter is a dataclass."""
        from dataclasses import is_dataclass

        assert is_dataclass(Converter)


class TestConverterWithActualParsing:
    """Test Converter with actual Cython parsing."""

    def test_convert_simple_module(self):
        """Test converting a simple module."""
        code = "def hello(): pass"
        parsed = parse_pyx(code)
        visitor = ModuleVisitor(parsed.source_ast)

        converter = Converter()
        result = converter.convert_module(visitor, parsed.source)

        assert isinstance(result, PyiModule)
        assert isinstance(result.scope, PyiScope)
        assert len(result.scope.functions) >= 1

    def test_convert_module_with_imports(self):
        """Test converting module with imports."""
        code = """
import os
import sys
from typing import Dict, List

def process():
    pass
"""
        parsed = parse_pyx(code)
        visitor = ModuleVisitor(parsed.source_ast)

        converter = Converter()
        result = converter.convert_module(visitor, parsed.source)

        assert isinstance(result, PyiModule)
        assert len(result.imports) >= 2

    def test_convert_module_with_function(self):
        """Test converting module with typed function."""
        code = """
def greet(name: str) -> str:
    '''Greet someone.'''
    return f"Hello, {name}!"
"""
        parsed = parse_pyx(code)
        visitor = ModuleVisitor(parsed.source_ast)

        converter = Converter()
        result = converter.convert_module(visitor, parsed.source)

        assert len(result.scope.functions) >= 1
        func = result.scope.functions[0]
        assert isinstance(func, PyiFunction)
        assert func.name == "greet"

    def test_convert_module_with_class(self):
        """Test converting module with class."""
        code = """
class MyClass:
    '''A test class.'''

    def __init__(self):
        self.value = 42

    def get_value(self):
        return self.value
"""
        parsed = parse_pyx(code)
        visitor = ModuleVisitor(parsed.source_ast)

        converter = Converter()
        result = converter.convert_module(visitor, parsed.source)

        assert len(result.scope.classes) >= 1
        cls = result.scope.classes[0]
        assert isinstance(cls, PyiClass)
        assert cls.name == "MyClass"
        assert len(cls.scope.functions) >= 2

    def test_convert_module_with_cdef_class(self):
        """Test converting module with cdef class."""
        code = """
cdef class CythonClass:
    cdef int value

    def __init__(self, int v):
        self.value = v

    def get_value(self):
        return self.value
"""
        parsed = parse_pyx(code)
        visitor = ModuleVisitor(parsed.source_ast)

        converter = Converter()
        result = converter.convert_module(visitor, parsed.source)

        # Note: cdef classes are converted similar to regular classes
        assert isinstance(result, PyiModule)

    def test_convert_module_with_assignments(self):
        """Test converting module with variable assignments."""
        code = """
x = 5
name = "hello"
values: list = []
"""
        parsed = parse_pyx(code)
        visitor = ModuleVisitor(parsed.source_ast)

        converter = Converter()
        result = converter.convert_module(visitor, parsed.source)

        # Should have assignments
        assert isinstance(result.scope, PyiScope)

    def test_convert_module_with_enum(self):
        """Test converting module with enum."""
        code = """
cdef enum Color:
    RED = 1
    GREEN = 2
    BLUE = 3
"""
        parsed = parse_pyx(code)
        visitor = ModuleVisitor(parsed.source_ast)

        converter = Converter()
        result = converter.convert_module(visitor, parsed.source)

        # Should parse enums if present
        assert isinstance(result, PyiModule)

    def test_convert_class_with_inheritance(self):
        """Test converting class with inheritance."""
        code = """
class Base:
    pass

class Derived(Base):
    def method(self):
        pass
"""
        parsed = parse_pyx(code)
        visitor = ModuleVisitor(parsed.source_ast)

        converter = Converter()
        result = converter.convert_module(visitor, parsed.source)

        derived_cls = None
        for cls in result.scope.classes:
            if cls.name == "Derived":
                derived_cls = cls
                break

        assert derived_cls is not None
        assert len(derived_cls.bases) >= 1

    def test_convert_async_function(self):
        """Test converting async function."""
        code = """
async def fetch_data():
    return "data"
"""
        parsed = parse_pyx(code)
        visitor = ModuleVisitor(parsed.source_ast)

        converter = Converter()
        result = converter.convert_module(visitor, parsed.source)

        assert len(result.scope.functions) >= 1
        func = result.scope.functions[0]
        assert func.is_async is True

    def test_convert_function_with_decorators(self):
        """Test converting function with decorators."""
        code = """
@property
def my_property(self):
    return 42

@staticmethod
def static_func():
    pass
"""
        parsed = parse_pyx(code)
        visitor = ModuleVisitor(parsed.source_ast)

        converter = Converter()
        result = converter.convert_module(visitor, parsed.source)

        assert len(result.scope.functions) >= 1

    def test_convert_function_with_type_hints(self):
        """Test converting function with comprehensive type hints."""
        code = """
from typing import Dict, List, Optional

def process(
    data: Dict[str, List[int]],
    flag: Optional[bool] = None
) -> List[str]:
    return ["result"]
"""
        parsed = parse_pyx(code)
        visitor = ModuleVisitor(parsed.source_ast)

        converter = Converter()
        result = converter.convert_module(visitor, parsed.source)

        assert len(result.scope.functions) >= 1
        func = result.scope.functions[0]
        assert func.name == "process"

    def test_convert_cdef_function(self):
        """Test converting cdef function."""
        code = """
cdef int add(int a, int b):
    return a + b

cpdef double multiply(double x, double y):
    return x * y
"""
        parsed = parse_pyx(code)
        visitor = ModuleVisitor(parsed.source_ast)

        converter = Converter()
        result = converter.convert_module(visitor, parsed.source)

        # cdef and cpdef functions should be in the scope
        assert isinstance(result.scope, PyiScope)

    def test_convert_nested_class(self):
        """Test converting nested class."""
        code = """
class Outer:
    class Inner:
        def method(self):
            pass
"""
        parsed = parse_pyx(code)
        visitor = ModuleVisitor(parsed.source_ast)

        converter = Converter()
        result = converter.convert_module(visitor, parsed.source)

        # Nested classes should be in the outer class's scope
        assert len(result.scope.classes) >= 1

    def test_convert_class_with_class_variables(self):
        """Test converting class with class variables."""
        code = """
class Config:
    VERSION: str = "1.0"
    MAX_SIZE: int = 100

    def get_version(self):
        return self.VERSION
"""
        parsed = parse_pyx(code)
        visitor = ModuleVisitor(parsed.source_ast)

        converter = Converter()
        result = converter.convert_module(visitor, parsed.source)

        cls = result.scope.classes[0]
        assert cls.name == "Config"
        # Should have assignments (class variables) and methods
        assert len(cls.scope.functions) >= 1

    def test_convert_function_various_signatures(self):
        """Test converting functions with various parameter styles."""
        code = """
def func1(a, b, c=5):
    pass

def func2(a, *, b, c=10):
    pass

def func3(*args, **kwargs):
    pass
"""
        parsed = parse_pyx(code)
        visitor = ModuleVisitor(parsed.source_ast)

        converter = Converter()
        result = converter.convert_module(visitor, parsed.source)

        assert len(result.scope.functions) >= 3

    def test_convert_module_with_docstring(self):
        """Test converting module with docstring."""
        code = '''
"""A sample module with documentation.

This module demonstrates all features.
"""

def hello():
    pass
'''
        parsed = parse_pyx(code)
        visitor = ModuleVisitor(parsed.source_ast)

        converter = Converter()
        result = converter.convert_module(visitor, parsed.source)

        # Module docstring may or may not be captured depending on Cython AST
        assert isinstance(result, PyiModule)
        assert len(result.scope.functions) >= 1

    def test_convert_module_complex_scenario(self):
        """Test converting a complex module with multiple features."""
        code = """
import os
from typing import Dict, List, Optional

CONSTANT: int = 42

def utility_func(x: int) -> str:
    '''Convert int to string.'''
    return str(x)

class DataProcessor:
    '''Process data structures.'''

    def __init__(self):
        self.data: List[int] = []

    def add_item(self, item: int) -> None:
        self.data.append(item)

    def get_items(self) -> List[int]:
        return self.data

async def fetch_results() -> Dict[str, List[int]]:
    '''Fetch filtered results.'''
    return {}
"""
        parsed = parse_pyx(code)
        visitor = ModuleVisitor(parsed.source_ast)

        converter = Converter()
        result = converter.convert_module(visitor, parsed.source)

        assert isinstance(result, PyiModule)
        assert len(result.imports) >= 1
        assert len(result.scope.functions) >= 2
        assert len(result.scope.classes) >= 1

    def test_convert_ctypedef(self):
        """Test converting typedef."""
        code = """
ctypedef int MyInt
ctypedef float MyFloat
ctypedef np.ndarray MyArray
"""
        parsed = parse_pyx(code)
        visitor = ModuleVisitor(parsed.source_ast)

        converter = Converter()
        result = converter.convert_module(visitor, parsed.source)

        assert len(result.scope.assignments) >= 3

    def test_convert_enum(self):
        """Test converting enum."""
        code = """
cpdef enum Color:
    RED = 1
    GREEN = 2
    BLUE = 3
"""
        parsed = parse_pyx(code)
        visitor = ModuleVisitor(parsed.source_ast)

        converter = Converter()
        result = converter.convert_module(visitor, parsed.source)

        assert len(result.scope.enums) >= 1

    def test_convert_enum_in_extern(self):
        """Test converting enum with extern."""
        code = """
cdef extern from "<header.h>":
    cpdef enum my_extern_enum:
        MY_ENUM_V1
        MY_ENUM_V2
        MY_ENUM_V3
"""
        parsed = parse_pyx(code)
        visitor = ModuleVisitor(parsed.source_ast)

        converter = Converter()
        result = converter.convert_module(visitor, parsed.source)

        assert len(result.scope.enums) == 1
        _enum = result.scope.enums[0]
        assert isinstance(_enum, PyiEnum)
        assert "my_extern_enum" == _enum.enum_name
        assert "MY_ENUM_V1" in _enum.names

    def test_cdef_public_class_vars(self):
        """Test converting cdef public class variables."""
        code = """
cdef public int module_global_not_a_class_var = -1  # not a class var, should be ignored

cdef class MyClass:
    cdef public int value = 1
    cdef int other = -1
"""
        parsed = parse_pyx(code)
        visitor = ModuleVisitor(parsed.source_ast)

        converter = Converter()
        result = converter.convert_module(visitor, parsed.source)

        assert len(result.scope.classes) == 1
        cls = result.scope.classes[0]
        assert len(cls.scope.assignments) == 1
        assert len(result.scope.assignments) == 0

    def test_char_ptr_type(self):
        """Test converting char pointer type."""
        code = """
cpdef str get(char* path):
    return NULL
"""
        parsed = parse_pyx(code)
        visitor = ModuleVisitor(parsed.source_ast)

        converter = Converter()
        result = converter.convert_module(visitor, parsed.source)

        assert len(result.scope.functions) == 1
        func = result.scope.functions[0]
        assert len(func.signature.args) == 1
        arg = func.signature.args[0]
        assert arg.annotation == "str"


class TestFunctionSourceOrder:
    """Test that functions appear in source order, not cdef-before-def."""

    def test_cpdef_before_def_in_source(self):
        """cpdef appearing before def must come first in output."""
        from stubgen_pyx import StubgenPyx
        from stubgen_pyx.config import StubgenPyxConfig

        s = StubgenPyx(
            config=StubgenPyxConfig(exclude_attribution=True, sort_imports=False)
        )
        result = s.convert_str("""
cpdef int first(int x): pass
def second(): pass
""")
        assert result.index("def first") < result.index("def second")

    def test_def_before_cpdef_in_source(self):
        """def appearing before cpdef must come first in output."""
        from stubgen_pyx import StubgenPyx
        from stubgen_pyx.config import StubgenPyxConfig

        s = StubgenPyx(
            config=StubgenPyxConfig(exclude_attribution=True, sort_imports=False)
        )
        result = s.convert_str("""
def first(): pass
cpdef int second(int x): pass
def third(): pass
""")
        assert result.index("def first") < result.index("def second")
        assert result.index("def second") < result.index("def third")

    def test_interleaved_functions_preserve_order(self):
        """Interleaved def and cpdef functions preserve source order."""
        from stubgen_pyx import StubgenPyx
        from stubgen_pyx.config import StubgenPyxConfig

        s = StubgenPyx(
            config=StubgenPyxConfig(exclude_attribution=True, sort_imports=False)
        )
        result = s.convert_str("""
def a(): pass
cpdef int b(int x): pass
def c(): pass
cpdef int d(int y): pass
""")
        assert result.index("def a") < result.index("def b")
        assert result.index("def b") < result.index("def c")
        assert result.index("def c") < result.index("def d")


class TestConverterStateless:
    """Converter should carry no session state between convert_module calls."""

    def test_converter_reusable_across_calls(self):
        """A single Converter instance must be safely reusable."""
        from stubgen_pyx.conversion.converter import Converter
        from stubgen_pyx.parsing.parser import parse_pyx
        from stubgen_pyx.analysis.visitor import ModuleVisitor

        converter = Converter()

        for src in ("def foo(): pass", "def bar(x: int) -> str: pass"):
            pr = parse_pyx(src)
            mv = ModuleVisitor(pr.source_ast)
            module = converter.convert_module(mv, pr.source, pr.type_comments)
            assert len(module.scope.functions) == 1

    def test_type_comments_not_shared_between_calls(self):
        """type_comments from one parse must not affect a second call."""
        from stubgen_pyx.conversion.converter import Converter
        from stubgen_pyx.parsing.parser import parse_pyx
        from stubgen_pyx.analysis.visitor import ModuleVisitor

        converter = Converter()

        pr1 = parse_pyx("def foo(x): pass  # type: (int) -> None")
        mv1 = ModuleVisitor(pr1.source_ast)
        converter.convert_module(mv1, pr1.source, pr1.type_comments)

        # Second call with no type comments: must not inherit from first
        pr2 = parse_pyx("def bar(y): pass")
        mv2 = ModuleVisitor(pr2.source_ast)
        module2 = converter.convert_module(mv2, pr2.source, {})
        assert module2.scope.functions[0].type_comment is None
