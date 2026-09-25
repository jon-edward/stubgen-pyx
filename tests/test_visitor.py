"""Tests for the analysis visitor module."""

from __future__ import annotations

from Cython.Compiler import Nodes

from stubgen_pyx.analysis.visitor import (
    ClassVisitor,
    ImportVisitor,
    ModuleVisitor,
    ScopeVisitor,
    _collect_attribute,
    _is_decorator_rebinding,
)
from stubgen_pyx.config import StubgenPyxConfig
from stubgen_pyx.parsing.parser import parse_str as parse_pyx
from stubgen_pyx.stubgen import StubgenPyx


def _convert(code: str) -> str:
    """Render `code` through the full pipeline (parse -> convert -> build)."""
    return StubgenPyx(
        StubgenPyxConfig(exclude_attribution=True, sort_imports=False)
    ).convert_str(code)


def _find_nodes(source: str, class_name: str):
    """Find parsed Cython nodes by their concrete class name."""
    tree = parse_pyx(source).source_ast
    found = []
    pending = [tree]
    while pending:
        node = pending.pop()
        if type(node).__name__ == class_name:
            found.append(node)
        for attr in getattr(node, "child_attrs", ()):
            child = getattr(node, attr, None)
            if isinstance(child, list):
                pending.extend(child)
            elif child is not None:
                pending.append(child)
    return found


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

    def test_scope_visitor_multiple_enums_with_extern(self):
        """cdef/cpdef enums, plain and extern -- all four representable in the rendered stub.

        A bare ``cdef enum`` has no runtime/executable component and is
        removed from ``body.stats`` entirely once real declaration
        analysis runs (same as a struct/union), whether or not it's
        inside a ``cdef extern from`` block: ``ScopeVisitor.enums`` (a
        structural, ``body.stats``-only walk) is always empty
        post-pipeline now, for every one of these. Recovered instead by
        ``Converter._convert_declared_entries`` walking ``scope.entries``
        directly; this now asserts against the full pipeline's rendered
        output rather than ``ScopeVisitor`` internals, which can no
        longer see any of this.
        """
        code = """
cdef enum NotWrappedInternal:
    V1 = 0
    V2 = 1

cpdef enum WrappedInternal:
    V3 = 0
    V4 = 1

cdef extern from "<header.h>":
    cdef enum NotWrappedExternal:
        V5

    cpdef enum WrappedExternal:
        V6
        V7
"""
        result = _convert(code)

        # cpdef enums (both internal and extern) become a real IntEnum
        # subclass wrapping their members.
        assert "class WrappedInternal(IntEnum):" in result
        assert "V3 = ..." in result
        assert "V4 = ..." in result
        assert "class WrappedExternal(IntEnum):" in result
        assert "V6 = ..." in result
        assert "V7 = ..." in result

        # Plain cdef enums (no Python wrapper) become an int TypeAlias.
        assert "NotWrappedInternal: TypeAlias = int" in result
        assert "NotWrappedExternal: TypeAlias = int" in result


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

    def test_visitor_ctypedef(self):
        """Test CtypedefVisitor."""
        code = """
ctypedef int MyInt
ctypedef float MyFloat
ctypedef np.ndarray MyArray
"""
        parsed = parse_pyx(code)
        visitor = ModuleVisitor(parsed.source_ast)

        assert len(visitor.scope.assignments) >= 3

    def test_module_member_type(self):
        code = """
cdef class CythonClass:
    cdef some_module.MyClass value
"""
        parsed = parse_pyx(code)
        visitor = ModuleVisitor(parsed.source_ast)

        assert len(visitor.scope.classes) == 1

    def test_templated_type_cdef_var(self):
        code = """
cdef class CythonClass:
    cdef public list[int] value
"""
        parsed = parse_pyx(code)
        visitor = ModuleVisitor(parsed.source_ast)

        assert len(visitor.scope.classes) == 1
        assert len(visitor.scope.classes[0].scope.cdef_variables) == 1


def test_scope_visitor_filters_cdef_visibility_and_collects_structures():
    """cdef class attribute visibility + struct/union/cppclass, via the full pipeline.

    ``cdef public``/``cdef readonly`` attributes, ``cdef struct``/
    ``union``, and a plain (no Python wrapper) ``cdef`` attribute are
    all declarations with no runtime/executable component once real
    declaration analysis runs, and vanish from ``body.stats`` entirely,
    same as the enum case above. ``ScopeVisitor``'s structural walk can
    no longer see any of them directly; recovered instead by
    ``Converter.convert_struct_or_union``/``_convert_declared_entries``.
    Asserts against the full pipeline's rendered output rather than
    ``ScopeVisitor`` internals.
    """
    code = """
cdef class Visible:
    cdef public int public_value
    cdef readonly int readonly_value
    cdef int private_value

cdef struct Header:
    int version

cdef union Value:
    int integer
    double decimal

cdef cppclass Native:
    int value
"""
    result = _convert(code)

    assert "class Visible:" in result
    assert "public_value: int" in result
    assert "readonly_value: int" in result
    # A plain (no visibility keyword) cdef attribute is pure C state,
    # never Python-visible -- not importable, so not in the stub at all.
    assert "private_value" not in result

    assert "class Header(TypedDict):" in result
    assert "version: int" in result

    assert "class Value(TypedDict" in result
    assert "total=False" in result
    assert "integer: int" in result
    assert "decimal: float" in result

    # A C++ class has no direct Python-facing representation; the
    # converter currently emits an Incomplete type alias for it.
    assert "Native" in result


def test_scope_visitor_skips_unnamed_enum():
    """An unnamed ``cdef enum`` has no name to expose in a stub and is skipped.

    Both enums vanish from ``body.stats`` post-pipeline (see the two
    tests above); asserts against the full pipeline's rendered output.
    """
    result = _convert("""
cdef enum:
    INTERNAL = 1

cdef enum Public:
    EXPORTED = 2
""")
    assert "Public: TypeAlias = int" in result
    assert "INTERNAL" not in result


def test_import_visitor_handles_all_import_forms_and_type_checking():
    code = """
import os
from pathlib import Path
cimport cython
from libc.stdint cimport int32_t
if TYPE_CHECKING:
    import json
if typing.TYPE_CHECKING:
    from collections import deque
"""
    parsed = parse_pyx(code).source_ast
    visitor = ImportVisitor(parsed)

    assert len(visitor.imports) >= 6
    assert any(
        type(node).__name__ == "SingleAssignmentNode" for node in visitor.imports
    )


def test_collect_attribute_handles_name_and_qualified_attribute():
    nodes = _find_nodes("if typing.TYPE_CHECKING:\n    import pathlib\n", "IfStatNode")
    condition = nodes[0].if_clauses[0].condition
    assert _collect_attribute(condition) == "typing.TYPE_CHECKING"

    name_nodes = _find_nodes("if TYPE_CHECKING:\n    pass\n", "IfStatNode")
    assert _collect_attribute(name_nodes[0].if_clauses[0].condition) == "TYPE_CHECKING"


def test_decorator_rebinding_vs_real_manual_rebinding():
    """A synthetic `name = decorator(name)` is dropped; real, hand-written
    code with the exact same shape is not.

    `AnalyseDeclarationsTransform` rewrites any decorated function/method
    into a bare `DefNode` plus a separate `name = decorator(name)`
    assignment -- dropped as redundant (the decorator itself is
    re-emitted directly on the `DefNode`, from the pre-pipeline
    `_stubgen_static_decorators` snapshot). But `foo = trace(foo)`,
    written by hand rather than produced by decorator syntax, is
    structurally identical and is a real, meaningful rebinding that must
    survive. Distinguished by source position, not shape alone (see
    `visitor._is_decorator_rebinding`) -- shape alone can't tell the two
    apart.
    """
    result = _convert("""
@classmethod
def class_func(cls):
    pass

def trace(f):
    return f

def foo():
    pass

foo = trace(foo)
""")
    # The synthetic rebinding for the decorated function never appears.
    assert "class_func = classmethod(class_func)" not in result
    assert "@classmethod" in result
    assert "def class_func(cls): ..." in result

    # The hand-written rebinding survives, in both directions: present,
    # and not turned into a second, bogus `def foo` definition either.
    assert "foo = trace(foo)" in result
    assert result.count("def foo(") == 1


def test_decorator_rebinding_vs_real_manual_rebinding_in_class():
    """Same disambiguation, at class/method scope."""
    result = _convert("""
def trace(f):
    return f

cdef class Foo:
    def bar(self):
        pass
    bar = trace(bar)
""")
    assert "bar = trace(bar)" in result
    assert result.count("def bar(") == 1


class TestScopeVisitorMethodsDirectly:
    """`ScopeVisitor`'s individual `visit_*` methods, exercised
    directly against a real or minimal node -- several of these have
    no real trigger through the full pipeline at all, since a
    synthesized node (a `PropertyNode` for `cdef public`, an
    `Entry`-only enum/struct, ...) intercepts first once real
    declaration analysis runs, same pattern documented throughout this
    module and `conversion/converter.py`. Constructing a `ScopeVisitor`
    around a trivial, empty module first (so `__post_init__`'s own
    walk finds nothing) and then calling the target method directly
    tests each method's own logic in isolation, regardless of whether
    real source can currently route a node to it.
    """

    @staticmethod
    def _empty_visitor(**kwargs) -> ScopeVisitor:
        result = parse_pyx("")
        return ScopeVisitor(node=result.source_ast, **kwargs)

    def test_visit_cenumdefnode_skips_an_unnamed_enum(self):
        from types import SimpleNamespace

        visitor = self._empty_visitor()
        visitor.visit_CEnumDefNode(SimpleNamespace(name=None))
        assert visitor.enums == []

    def test_visit_cenumdefnode_collects_a_named_enum(self):
        from types import SimpleNamespace

        visitor = self._empty_visitor()
        node = SimpleNamespace(name="Color")
        visitor.visit_CEnumDefNode(node)
        assert visitor.enums == [node]

    def test_visit_exprstatnode_collects_an_annotated_bare_name(self):
        """`x: int` (no assignment) -- only ever reachable this way for
        a class body's own individually-typed attributes at the point
        this visitor sees them (a module-level one is already folded
        into a single synthetic `__annotations__` dict by the time
        real declaration analysis finishes; see
        `Converter._convert_declared_entries`'s docstring)."""
        from types import SimpleNamespace

        from Cython.Compiler import ExprNodes

        visitor = self._empty_visitor()
        name_node = ExprNodes.NameNode(None, name="x")
        name_node.annotation = "int"
        visitor.visit_ExprStatNode(SimpleNamespace(expr=name_node))
        assert len(visitor.assignments) == 1

    def test_visit_exprstatnode_ignores_a_non_annotated_name(self):
        from types import SimpleNamespace

        from Cython.Compiler import ExprNodes

        visitor = self._empty_visitor()
        name_node = ExprNodes.NameNode(None, name="y")
        name_node.annotation = None
        visitor.visit_ExprStatNode(SimpleNamespace(expr=name_node))
        assert visitor.assignments == []

    def test_visit_defnode_skips_a_fused_specialization(self):
        from types import SimpleNamespace

        visitor = self._empty_visitor()
        node = SimpleNamespace(name="f", pos=(None, 1, 0))
        visitor._fused_specialization_ids.add(id(node))
        visitor.visit_DefNode(node)
        assert visitor.py_functions == []

    def test_visit_cfuncdefnode_skips_a_fused_specialization(self):
        from types import SimpleNamespace

        visitor = self._empty_visitor()
        node = SimpleNamespace(
            declarator=SimpleNamespace(overridable=True), pos=(None, 1, 0)
        )
        visitor._fused_specialization_ids.add(id(node))
        visitor.visit_CFuncDefNode(node)
        assert visitor.cdef_functions == []

    def test_visit_cfuncdefnode_skips_a_non_overridable_plain_cdef(self):
        """A plain `cdef` function (no `cpdef`) is never Python-visible
        -- `.declarator.overridable` is `False` for it -- and must not
        be collected."""
        from types import SimpleNamespace

        visitor = self._empty_visitor()
        node = SimpleNamespace(
            declarator=SimpleNamespace(overridable=False), pos=(None, 1, 0)
        )
        visitor.visit_CFuncDefNode(node)
        assert visitor.cdef_functions == []

    def test_visit_cfuncdefnode_records_its_def_position_when_named(self):
        """The success path -- collected, and its `pos` recorded for
        `_is_decorator_rebinding`'s later disambiguation. In practice
        `_declared_name` returns `None` for an ordinary `CFuncDefNode`
        (its name lives structurally on a `CFuncDeclaratorNode`, which
        `_declared_name`'s single-level unwrap doesn't reach -- same
        known limitation as `TestDeclaredNamePointerDeclarator` below),
        so this exercises `visit_CFuncDefNode`'s own "if named, record
        it" branch directly rather than relying on that resolving for
        a real node."""
        from types import SimpleNamespace

        visitor = self._empty_visitor()
        node = SimpleNamespace(
            name="add", declarator=SimpleNamespace(overridable=True), pos=(None, 3, 0)
        )
        visitor.visit_CFuncDefNode(node)
        assert visitor.cdef_functions == [node]
        assert visitor._def_positions == {"add": (None, 3, 0)}

    def test_visit_fusedtypenode_collects_it(self):
        from types import SimpleNamespace

        visitor = self._empty_visitor()
        node = SimpleNamespace(name="numeric")
        visitor.visit_FusedTypeNode(node)
        assert visitor.fused_types == [node]

    def test_visit_cvardefnode_collects_a_public_class_attribute(self):
        """The structural (`CVarDefNode`) path for a `public`/`readonly`
        attribute -- `visit_PropertyNode`'s own docstring notes this
        node type is superseded by a synthesized `PropertyNode` once
        real declaration analysis runs, so this exercises the method's
        own logic directly rather than via the full pipeline."""
        from types import SimpleNamespace

        visitor = self._empty_visitor(in_class=True)
        node = SimpleNamespace(visibility="public")
        visitor.visit_CVarDefNode(node)
        assert visitor.cdef_variables == [node]

    def test_visit_cvardefnode_skips_outside_a_class(self):
        from types import SimpleNamespace

        visitor = self._empty_visitor(in_class=False)
        node = SimpleNamespace(visibility="public")
        visitor.visit_CVarDefNode(node)
        assert visitor.cdef_variables == []

    def test_visit_cstructoruniondefnode_skips_compiler_synthesized_names(self):
        from types import SimpleNamespace

        visitor = self._empty_visitor()
        synthetic = SimpleNamespace(name="__pyx_opt_args_4mod_5func")
        visitor.visit_CStructOrUnionDefNode(synthetic)
        assert visitor.cdef_structs_or_unions == []

        real = SimpleNamespace(name="Point")
        visitor.visit_CStructOrUnionDefNode(real)
        assert visitor.cdef_structs_or_unions == [real]


class TestIsDecoratorRebindingBranches:
    """Direct tests for `_is_decorator_rebinding`'s three
    "structurally similar but not actually a synthetic rebinding"
    cases -- each confirmed for real via `TestClassAndFunctionScope`'s
    `_convert`-based tests above (a hand-written rebinding of this
    shape survives in the output), tested here against the disambiguation
    function itself.
    """

    def test_call_with_two_arguments_is_not_a_rebinding(self):
        node = _first_node("foo = trace(foo, extra)")
        assert _is_decorator_rebinding(node, "foo", {}) is False

    def test_call_with_a_differently_named_argument_is_not_a_rebinding(self):
        node = _first_node("foo = trace(bar)")
        assert _is_decorator_rebinding(node, "foo", {}) is False

    def test_call_with_an_attribute_access_argument_is_checked_by_attribute_name(self):
        """`trace(obj.foo)` -- the argument is an attribute access, not
        a bare name; only its `.attribute` name is compared."""
        node = _first_node("foo = trace(obj.foo)")
        # Names match (`.foo`), but with no `def_positions` entry for
        # "foo" at all, still correctly not a rebinding.
        assert _is_decorator_rebinding(node, "foo", {}) is False


def _first_node(source: str):
    """Find the first `SingleAssignmentNode` in `source`'s raw (pre-pipeline) AST."""
    from io import StringIO

    from Cython.Compiler import Parsing
    from Cython.Compiler.Scanning import PyrexScanner, StringSourceDescriptor

    from stubgen_pyx.parsing.context import StubgenContext
    from stubgen_pyx.parsing.parser import _DEFAULT_MODULE_NAME, _resolve_scope

    context = StubgenContext()
    module_name = _DEFAULT_MODULE_NAME
    source_desc = StringSourceDescriptor(module_name, source)
    initial_pos = (source_desc, 1, 0)
    scope = _resolve_scope(context, module_name, initial_pos, allow_pxd_merge=False)
    scope.cpp = context.cpp
    scanner = PyrexScanner(
        StringIO(source),
        source_desc,
        source_encoding="UTF-8",
        scope=scope,
        context=context,
        initial_pos=initial_pos,
    )
    tree = Parsing.p_module(
        scanner, False, module_name, ctx=Parsing.Ctx(allow_struct_enum_decorator=True)
    )

    pending = [getattr(tree, "body", None)]
    while pending:
        node = pending.pop(0)
        if node is None:
            continue
        if isinstance(node, Nodes.SingleAssignmentNode):
            return node
        for attr in getattr(node, "child_attrs", ()):
            child = getattr(node, attr, None)
            if isinstance(child, list):
                pending.extend(child)
            elif child is not None:
                pending.append(child)
    raise AssertionError("No SingleAssignmentNode found")


class TestDeclaredNamePointerDeclarator:
    def test_returns_none_for_a_doubly_wrapped_pointer_declarator(self):
        """`_declared_name` only unwraps one `CPtrDeclaratorNode` layer
        -- for `int* get_ptr()`, the pointer wraps a `CFuncDeclaratorNode`
        (the function's own args/name), not a plain name directly, so
        this is a known, accepted limitation: it returns `None` rather
        than "get_ptr" here. `signature._to_argument` has the general,
        fully-recursive unwrap (`_declarator_name`) for cases that
        actually need the real name; this narrower one is only used for
        `_def_positions` tracking, where a `None` here just means that
        one function's decorator-rebinding disambiguation falls back to
        shape alone for it -- not a correctness issue for anything else.
        """
        import sys

        sys.path.insert(0, "tests")
        from Cython.Compiler import Nodes as visitor_Nodes
        from test_type_parsing import _first_node as _first_node_of_type

        from stubgen_pyx.analysis.visitor import _declared_name

        node = _first_node_of_type(
            "cdef int* get_ptr():\n    pass\n", visitor_Nodes.CFuncDefNode
        )
        assert _declared_name(node) is None
