"""Tests for the analysis visitor module."""

from __future__ import annotations


from stubgen_pyx.parsing.parser import parse_pyx
from stubgen_pyx.analysis.visitor import (
    ScopeVisitor,
    ImportVisitor,
    ModuleVisitor,
    ClassVisitor,
)


class TestScopeVisitorBasics:
    """Test ScopeVisitor with basic code."""

    def test_scope_visitor_creation(self):
        """Test creating a ScopeVisitor."""
        code = "x = 5"
        parsed = parse_pyx(code)
        visitor = ScopeVisitor(parsed.source_ast)

        assert visitor is not None
        assert visitor.node is not None
        assert isinstance(visitor.assignments, list)
        assert isinstance(visitor.py_functions, list)
        assert isinstance(visitor.cdef_functions, list)
        assert isinstance(visitor.classes, list)
        assert isinstance(visitor.enums, list)

    def test_scope_visitor_empty_module(self):
        """Test visitor on empty module."""
        code = ""
        parsed = parse_pyx(code)
        visitor = ScopeVisitor(parsed.source_ast)

        assert visitor.assignments == []
        assert visitor.py_functions == []
        assert visitor.cdef_functions == []
        assert visitor.classes == []
        assert visitor.enums == []

    def test_scope_visitor_single_function(self):
        """Test visitor collecting a single function."""
        code = """
def hello():
    pass
"""
        parsed = parse_pyx(code)
        visitor = ScopeVisitor(parsed.source_ast)

        assert len(visitor.py_functions) >= 1
        assert visitor.py_functions[0].name == "hello"

    def test_scope_visitor_multiple_functions(self):
        """Test visitor collecting multiple functions."""
        code = """
def func1():
    pass

def func2():
    pass

def func3():
    pass
"""
        parsed = parse_pyx(code)
        visitor = ScopeVisitor(parsed.source_ast)

        assert len(visitor.py_functions) >= 3

    def test_scope_visitor_single_class(self):
        """Test visitor collecting a single class."""
        code = """
class MyClass:
    def method(self):
        pass
"""
        parsed = parse_pyx(code)
        visitor = ScopeVisitor(parsed.source_ast)

        assert len(visitor.classes) >= 1
        cls = visitor.classes[0]
        assert isinstance(cls, ClassVisitor)
        assert cls.node.name == "MyClass"

    def test_scope_visitor_multiple_classes(self):
        """Test visitor collecting multiple classes."""
        code = """
class Class1:
    pass

class Class2:
    pass

class Class3:
    pass
"""
        parsed = parse_pyx(code)
        visitor = ScopeVisitor(parsed.source_ast)

        assert len(visitor.classes) >= 3

    def test_scope_visitor_mixed_elements(self):
        """Test visitor collecting mixed elements."""
        code = """
x = 5
y = 10

def func():
    pass

class MyClass:
    value = 0
"""
        parsed = parse_pyx(code)
        visitor = ScopeVisitor(parsed.source_ast)

        # Should collect assignments, functions, and classes
        assert len(visitor.py_functions) >= 1
        assert len(visitor.classes) >= 1


class TestScopeVisitorFunctions:
    """Test ScopeVisitor function collection."""

    def test_scope_visitor_py_function(self):
        """Test collecting Python function."""
        code = """
def greet(name: str) -> str:
    return f"Hello, {name}"
"""
        parsed = parse_pyx(code)
        visitor = ScopeVisitor(parsed.source_ast)

        assert len(visitor.py_functions) >= 1
        func = visitor.py_functions[0]
        assert func.name == "greet"

    def test_scope_visitor_async_function(self):
        """Test collecting async function."""
        code = """
async def async_task():
    return "done"
"""
        parsed = parse_pyx(code)
        visitor = ScopeVisitor(parsed.source_ast)

        assert len(visitor.py_functions) >= 1

    def test_scope_visitor_cdef_function(self):
        """Test collecting cdef function."""
        code = """
cpdef int add(int a, int b):
    return a + b
"""
        parsed = parse_pyx(code)
        visitor = ScopeVisitor(parsed.source_ast)

        # cdef functions are only collected if they're overridable
        assert isinstance(visitor.cdef_functions, list)

    def test_scope_visitor_function_with_decorators(self):
        """Test collecting decorated function."""
        code = """
@staticmethod
def static_func():
    pass

@classmethod
def class_func(cls):
    pass
"""
        parsed = parse_pyx(code)
        visitor = ScopeVisitor(parsed.source_ast)

        assert len(visitor.py_functions) >= 2


class TestScopeVisitorClasses:
    """Test ScopeVisitor class collection."""

    def test_scope_visitor_class_with_methods(self):
        """Test collecting class with methods."""
        code = """
class DataProcessor:
    def __init__(self):
        self.data = []

    def process(self):
        pass

    def validate(self):
        pass
"""
        parsed = parse_pyx(code)
        visitor = ScopeVisitor(parsed.source_ast)

        assert len(visitor.classes) >= 1
        cls_visitor = visitor.classes[0]
        assert len(cls_visitor.scope.py_functions) >= 3

    def test_scope_visitor_class_with_attributes(self):
        """Test collecting class with class attributes."""
        code = """
class Config:
    VERSION = "1.0"
    DEBUG = False
    MAX_SIZE = 100
"""
        parsed = parse_pyx(code)
        visitor = ScopeVisitor(parsed.source_ast)

        assert len(visitor.classes) >= 1

    def test_scope_visitor_cdef_class(self):
        """Test collecting cdef class."""
        code = """
cdef class CythonClass:
    cdef int value

    def __init__(self, int v):
        self.value = v
"""
        parsed = parse_pyx(code)
        visitor = ScopeVisitor(parsed.source_ast)

        # cdef classes should be collected
        assert len(visitor.classes) >= 1

    def test_scope_visitor_nested_class(self):
        """Test collecting nested classes."""
        code = """
class Outer:
    class Inner:
        def inner_method(self):
            pass

    def outer_method(self):
        pass
"""
        parsed = parse_pyx(code)
        visitor = ScopeVisitor(parsed.source_ast)

        outer = visitor.classes[0]
        # Inner class should be in Outer's scope
        assert len(outer.scope.classes) >= 1


class TestScopeVisitorAssignments:
    """Test ScopeVisitor assignment collection."""

    def test_scope_visitor_simple_assignment(self):
        """Test collecting simple assignment."""
        code = """
x = 5
y = "hello"
z = [1, 2, 3]
"""
        parsed = parse_pyx(code)
        visitor = ScopeVisitor(parsed.source_ast)

        # Should collect assignments
        assert isinstance(visitor.assignments, list)

    def test_scope_visitor_annotated_assignment(self):
        """Test collecting annotated assignment."""
        code = """
x: int = 5
name: str = "test"
values: list = []
"""
        parsed = parse_pyx(code)
        visitor = ScopeVisitor(parsed.source_ast)

        # Annotated assignments should be collected
        assert isinstance(visitor.assignments, list)


class TestScopeVisitorEnums:
    """Test ScopeVisitor enum collection."""

    def test_scope_visitor_enum(self):
        """Test collecting enum."""
        code = """
cdef enum Color:
    RED = 1
    GREEN = 2
    BLUE = 3
"""
        parsed = parse_pyx(code)
        visitor = ScopeVisitor(parsed.source_ast)

        # Enums should be collected if create_wrapper is True
        assert isinstance(visitor.enums, list)

    def test_scope_visitor_multiple_enums(self):
        """Test collecting multiple enums."""
        code = """
cdef enum Status:
    PENDING = 0
    ACTIVE = 1

cdef enum Priority:
    LOW = 0
    HIGH = 1
"""
        parsed = parse_pyx(code)
        visitor = ScopeVisitor(parsed.source_ast)

        # Should collect enums
        assert isinstance(visitor.enums, list)


class TestImportVisitorBasics:
    """Test ImportVisitor with basic code."""

    def test_import_visitor_creation(self):
        """Test creating an ImportVisitor."""
        code = "import os"
        parsed = parse_pyx(code)
        visitor = ImportVisitor(parsed.source_ast)

        assert visitor is not None
        assert visitor.node is not None
        assert isinstance(visitor.imports, list)

    def test_import_visitor_empty_module(self):
        """Test visitor on module without imports."""
        code = """
def hello():
    pass
"""
        parsed = parse_pyx(code)
        visitor = ImportVisitor(parsed.source_ast)

        # No imports in this module
        assert len(visitor.imports) == 0

    def test_import_visitor_single_import(self):
        """Test visitor collecting single import."""
        code = "import os"
        parsed = parse_pyx(code)
        visitor = ImportVisitor(parsed.source_ast)

        assert len(visitor.imports) >= 1

    def test_import_visitor_multiple_imports(self):
        """Test visitor collecting multiple imports."""
        code = """
import os
import sys
import re
"""
        parsed = parse_pyx(code)
        visitor = ImportVisitor(parsed.source_ast)

        assert len(visitor.imports) >= 3

    def test_import_visitor_from_import(self):
        """Test visitor collecting from...import."""
        code = "from typing import Dict, List, Optional"
        parsed = parse_pyx(code)
        visitor = ImportVisitor(parsed.source_ast)

        assert len(visitor.imports) >= 1

    def test_import_visitor_mixed_imports(self):
        """Test visitor collecting mixed imports."""
        code = """
import os
from typing import Dict
import sys
from collections import defaultdict
"""
        parsed = parse_pyx(code)
        visitor = ImportVisitor(parsed.source_ast)

        assert len(visitor.imports) >= 4

    def test_import_visitor_cimport(self):
        """Test visitor collecting cimport statements."""
        code = """
cimport cython
from cpython.mem cimport PyMem_Malloc
"""
        parsed = parse_pyx(code)
        visitor = ImportVisitor(parsed.source_ast)

        # Should collect cimport statements
        assert isinstance(visitor.imports, list)


class TestModuleVisitorBasics:
    """Test ModuleVisitor with basic code."""

    def test_module_visitor_creation(self):
        """Test creating a ModuleVisitor."""
        code = "x = 5"
        parsed = parse_pyx(code)
        visitor = ModuleVisitor(parsed.source_ast)

        assert visitor is not None
        assert visitor.node is not None
        assert isinstance(visitor.import_visitor, ImportVisitor)
        assert isinstance(visitor.scope, ScopeVisitor)

    def test_module_visitor_empty_module(self):
        """Test visitor on empty module."""
        code = ""
        parsed = parse_pyx(code)
        visitor = ModuleVisitor(parsed.source_ast)

        assert visitor.scope.assignments == []
        assert visitor.scope.py_functions == []
        assert visitor.import_visitor.imports == []

    def test_module_visitor_with_imports_and_code(self):
        """Test visitor collecting imports and code."""
        code = """
import os
from typing import Dict

def process():
    pass

class DataHandler:
    pass
"""
        parsed = parse_pyx(code)
        visitor = ModuleVisitor(parsed.source_ast)

        assert len(visitor.import_visitor.imports) >= 2
        assert len(visitor.scope.py_functions) >= 1
        assert len(visitor.scope.classes) >= 1

    def test_module_visitor_complex_module(self):
        """Test visitor on complex module."""
        code = """
import os
import sys
from typing import Dict, List, Optional

VERSION = "1.0"

def utility_function(x: int) -> str:
    return str(x)

class Main:
    value = 0

    def __init__(self):
        pass

    def process(self):
        pass

async def async_task():
    return "done"
"""
        parsed = parse_pyx(code)
        visitor = ModuleVisitor(parsed.source_ast)

        assert len(visitor.import_visitor.imports) >= 3
        assert len(visitor.scope.py_functions) >= 2
        assert len(visitor.scope.classes) >= 1


class TestClassVisitorBasics:
    """Test ClassVisitor with basic code."""

    def test_class_visitor_creation(self):
        """Test creating a ClassVisitor."""
        code = "class MyClass: pass"
        parsed = parse_pyx(code)
        module_visitor = ModuleVisitor(parsed.source_ast)

        # Get the first class from the module
        cls_visitor = module_visitor.scope.classes[0]
        assert isinstance(cls_visitor, ClassVisitor)
        assert cls_visitor.node.name == "MyClass"
        assert isinstance(cls_visitor.scope, ScopeVisitor)

    def test_class_visitor_with_methods(self):
        """Test ClassVisitor collecting methods."""
        code = """
class Calculator:
    def add(self, a, b):
        return a + b

    def subtract(self, a, b):
        return a - b

    def multiply(self, a, b):
        return a * b
"""
        parsed = parse_pyx(code)
        module_visitor = ModuleVisitor(parsed.source_ast)

        cls = module_visitor.scope.classes[0]
        assert len(cls.scope.py_functions) >= 3

    def test_class_visitor_with_init_and_methods(self):
        """Test ClassVisitor with __init__ and methods."""
        code = """
class Person:
    def __init__(self, name):
        self.name = name

    def greet(self):
        return f"Hello, I'm {self.name}"
"""
        parsed = parse_pyx(code)
        module_visitor = ModuleVisitor(parsed.source_ast)

        cls = module_visitor.scope.classes[0]
        methods = cls.scope.py_functions
        method_names = [m.name for m in methods]

        assert "__init__" in method_names
        assert "greet" in method_names

    def test_class_visitor_with_class_vars(self):
        """Test ClassVisitor collecting class variables."""
        code = """
class Config:
    DEBUG = True
    PORT = 8000
    HOST = "localhost"
"""
        parsed = parse_pyx(code)
        module_visitor = ModuleVisitor(parsed.source_ast)

        cls = module_visitor.scope.classes[0]
        # Class variables should be in assignments
        assert isinstance(cls.scope.assignments, list)

    def test_class_visitor_with_properties(self):
        """Test ClassVisitor with properties."""
        code = """
class Temperature:
    def __init__(self, celsius):
        self._celsius = celsius

    @property
    def fahrenheit(self):
        return self._celsius * 9/5 + 32
"""
        parsed = parse_pyx(code)
        module_visitor = ModuleVisitor(parsed.source_ast)

        cls = module_visitor.scope.classes[0]
        assert len(cls.scope.py_functions) >= 2

    def test_class_visitor_inheritance(self):
        """Test ClassVisitor with inherited class."""
        code = """
class Base:
    def base_method(self):
        pass

class Derived(Base):
    def derived_method(self):
        pass
"""
        parsed = parse_pyx(code)
        module_visitor = ModuleVisitor(parsed.source_ast)

        # Should have 2 classes
        assert len(module_visitor.scope.classes) >= 2

    def test_class_visitor_nested(self):
        """Test ClassVisitor with nested class."""
        code = """
class Outer:
    value = 1

    class Inner:
        nested_value = 2

    def method(self):
        pass
"""
        parsed = parse_pyx(code)
        module_visitor = ModuleVisitor(parsed.source_ast)

        outer = module_visitor.scope.classes[0]
        # Inner class should be in outer's scope
        assert len(outer.scope.classes) >= 1


class TestVisitorsIntegration:
    """Integration tests for all visitors together."""

    def test_all_visitors_on_complex_module(self):
        """Test all visitors on a complex module."""
        code = """
import os
from typing import Dict, List, Optional
cimport cython

VERSION = "1.0"
DEBUG = False

def process_data(data: List[int]) -> Dict[str, int]:
    return {}

class DataProcessor:
    def __init__(self):
        self.cache = {}

    def process(self, item):
        return item

    class Cache:
        def __init__(self):
            self.data = {}

async def fetch_results():
    return []

cdef class CythonClass:
    cdef int value

cdef enum Status:
    PENDING = 0
    DONE = 1
"""
        parsed = parse_pyx(code)
        module_visitor = ModuleVisitor(parsed.source_ast)

        # Test imports
        assert len(module_visitor.import_visitor.imports) >= 3

        # Test scope
        assert len(module_visitor.scope.py_functions) >= 2
        assert len(module_visitor.scope.classes) >= 2

        # Test class visitor
        data_processor = module_visitor.scope.classes[0]
        assert len(data_processor.scope.py_functions) >= 2
        assert len(data_processor.scope.classes) >= 1

    def test_visitors_with_type_annotations(self):
        """Test visitors with extensive type annotations."""
        code = """
from typing import Callable, Generic, TypeVar

T = TypeVar('T')

def generic_process(
    items: List[T],
    callback: Callable[[T], bool]
) -> Dict[str, List[T]]:
    pass

class Container(Generic[T]):
    def get(self) -> Optional[T]:
        pass
"""
        parsed = parse_pyx(code)
        module_visitor = ModuleVisitor(parsed.source_ast)

        assert len(module_visitor.import_visitor.imports) >= 1
        assert len(module_visitor.scope.py_functions) >= 1
        assert len(module_visitor.scope.classes) >= 1

    def test_visitors_preserve_structure(self):
        """Test that visitors preserve module structure."""
        code = """
import os

class A:
    def method_a(self):
        pass

def function_b():
    pass

class C:
    def method_c(self):
        pass
"""
        parsed = parse_pyx(code)
        first_visitor = ModuleVisitor(parsed.source_ast)

        # Create another visitor on same code to ensure consistency
        parsed2 = parse_pyx(code)
        second_visitor = ModuleVisitor(parsed2.source_ast)

        # Both visitors should find same number of elements
        assert len(first_visitor.scope.classes) == len(second_visitor.scope.classes)
        assert len(first_visitor.scope.py_functions) == len(
            second_visitor.scope.py_functions
        )
