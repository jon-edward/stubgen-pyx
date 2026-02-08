"""Tests for builders module."""

from __future__ import annotations


from stubgen_pyx.builders.builder import Builder
from stubgen_pyx.models.pyi_elements import (
    PyiModule,
    PyiClass,
    PyiFunction,
    PyiAssignment,
    PyiImport,
    PyiScope,
    PyiSignature,
    PyiArgument,
    PyiEnum,
)


class TestBuilder:
    """Test the Builder class."""

    def test_builder_creation_default(self):
        """Test creating a builder with default settings."""
        builder = Builder()
        assert builder.include_private is False

    def test_builder_creation_include_private(self):
        """Test creating a builder that includes private members."""
        builder = Builder(include_private=True)
        assert builder.include_private is True

    def test_is_private_private_name(self):
        """Test identifying private names."""
        builder = Builder()
        assert builder._is_private("_private") is True
        assert builder._is_private("__private") is True

    def test_is_private_dunder_name(self):
        """Test that dunder names are not considered private."""
        builder = Builder()
        assert builder._is_private("__init__") is False
        assert builder._is_private("__str__") is False

    def test_is_private_public_name(self):
        """Test that public names are not private."""
        builder = Builder()
        assert builder._is_private("public") is False
        assert builder._is_private("MyClass") is False

    def test_build_argument_name_only(self):
        """Test building an argument with only a name."""
        builder = Builder()
        arg = PyiArgument("x")
        result = builder.build_argument(arg)
        assert result == "x"

    def test_build_argument_with_annotation(self):
        """Test building an argument with an annotation."""
        builder = Builder()
        arg = PyiArgument("x", annotation="int")
        result = builder.build_argument(arg)
        assert result == "x: int"

    def test_build_argument_with_default(self):
        """Test building an argument with a default value."""
        builder = Builder()
        arg = PyiArgument("x", annotation="int", default="5")
        result = builder.build_argument(arg)
        assert result == "x: int = 5"

    def test_build_argument_with_default_no_annotation(self):
        """Test building an argument with default but no annotation."""
        builder = Builder()
        arg = PyiArgument("x", default="None")
        result = builder.build_argument(arg)
        assert result == "x = None"

    def test_build_signature_simple(self):
        """Test building a simple signature."""
        builder = Builder()
        sig = PyiSignature(args=[PyiArgument("x", annotation="int")])
        result = builder.build_signature(sig)
        assert "(x: int)" in result

    def test_build_signature_multiple_args(self):
        """Test building a signature with multiple arguments."""
        builder = Builder()
        sig = PyiSignature(
            args=[
                PyiArgument("x", annotation="int"),
                PyiArgument("y", annotation="str"),
            ]
        )
        result = builder.build_signature(sig)
        assert "x: int" in result
        assert "y: str" in result

    def test_build_signature_with_return_type(self):
        """Test building a signature with return type."""
        builder = Builder()
        sig = PyiSignature(args=[PyiArgument("x", "int")], return_type="str")
        result = builder.build_signature(sig)
        assert "-> str" in result

    def test_build_signature_with_varargs(self):
        """Test building a signature with *args."""
        builder = Builder()
        sig = PyiSignature(args=[], var_arg=PyiArgument("args", "int"))
        result = builder.build_signature(sig)
        assert "*" in result

    def test_build_signature_with_kwargs(self):
        """Test building a signature with **kwargs."""
        builder = Builder()
        sig = PyiSignature(args=[], kw_arg=PyiArgument("kwargs", "str"))
        result = builder.build_signature(sig)
        assert "**" in result

    def test_build_signature_posonly_args(self):
        """Test building a signature with positional-only arguments."""
        builder = Builder()
        sig = PyiSignature(
            args=[
                PyiArgument("x", "int"),
                PyiArgument("y", "str"),
            ],
            num_posonly_args=1,
        )
        result = builder.build_signature(sig)
        assert "/" in result

    def test_build_signature_kwonly_args(self):
        """Test building a signature with keyword-only arguments."""
        builder = Builder()
        sig = PyiSignature(
            args=[
                PyiArgument("x", "int"),
                PyiArgument("y", "str"),
            ],
            num_kwonly_args=1,
        )
        result = builder.build_signature(sig)
        assert "*" in result

    def test_build_function_simple(self):
        """Test building a simple function."""
        builder = Builder()
        func = PyiFunction(
            name="greet",
            is_async=False,
            signature=PyiSignature(
                args=[PyiArgument("name", annotation="str")], return_type="str"
            ),
        )
        result = builder.build_function(func)
        assert result
        assert "def greet" in result
        assert "(name: str)" in result
        assert "-> str" in result

    def test_build_function_with_docstring(self):
        """Test building a function with a docstring."""
        builder = Builder()
        func = PyiFunction(
            name="greet",
            is_async=False,
            signature=PyiSignature(),
            doc='"""Greet someone."""',
        )
        result = builder.build_function(func)
        assert result
        assert "def greet" in result
        assert "Greet someone" in result

    def test_build_function_async(self):
        """Test building an async function."""
        builder = Builder()
        func = PyiFunction(name="async_greet", signature=PyiSignature(), is_async=True)
        result = builder.build_function(func)
        assert result
        assert "async def" in result

    def test_build_function_with_decorator(self):
        """Test building a function with decorators."""
        builder = Builder()
        func = PyiFunction(
            name="decorated_func",
            is_async=False,
            signature=PyiSignature(),
            decorators=["@property", "@staticmethod"],
        )
        result = builder.build_function(func)
        assert result
        assert "@property" in result
        assert "@staticmethod" in result

    def test_build_function_private_included(self):
        """Test building a private function when include_private is True."""
        builder = Builder(include_private=True)
        func = PyiFunction(
            name="_private_func", is_async=False, signature=PyiSignature()
        )
        result = builder.build_function(func)
        assert result
        assert "_private_func" in result

    def test_build_assignment_simple(self):
        """Test building a simple assignment."""
        builder = Builder()
        assignment = PyiAssignment("x: int = 5")
        result = builder.build_assignment(assignment)
        assert result == "x: int = 5"

    def test_build_import_simple(self):
        """Test building a simple import statement."""
        builder = Builder()
        import_stmt = PyiImport("import os")
        result = builder.build_import(import_stmt)
        assert result == "import os"

    def test_build_class_simple(self):
        """Test building a simple class."""
        builder = Builder()
        cls = PyiClass(name="MyClass", scope=PyiScope())
        result = builder.build_class(cls)
        assert result
        assert "class MyClass" in result
        assert "..." in result

    def test_build_class_with_base(self):
        """Test building a class with a base class."""
        builder = Builder()
        cls = PyiClass(name="MyClass", bases=["BaseClass"], scope=PyiScope())
        result = builder.build_class(cls)
        assert result
        assert "class MyClass(BaseClass" in result

    def test_build_class_with_multiple_bases(self):
        """Test building a class with multiple bases."""
        builder = Builder()
        cls = PyiClass(
            name="MyClass", bases=["BaseClass1", "BaseClass2"], scope=PyiScope()
        )
        result = builder.build_class(cls)
        assert result
        assert "BaseClass1" in result
        assert "BaseClass2" in result

    def test_build_class_with_metaclass(self):
        """Test building a class with a metaclass."""
        builder = Builder()
        cls = PyiClass(name="MyClass", metaclass="MetaClass", scope=PyiScope())
        result = builder.build_class(cls)
        assert result
        assert "metaclass=MetaClass" in result

    def test_build_class_with_docstring(self):
        """Test building a class with a docstring."""
        builder = Builder()
        cls = PyiClass(name="MyClass", doc='"""My class."""', scope=PyiScope())
        result = builder.build_class(cls)
        assert result
        assert "MyClass" in result
        assert "My class" in result

    def test_build_class_with_decorator(self):
        """Test building a class with decorators."""
        builder = Builder()
        cls = PyiClass(name="MyClass", decorators=["@dataclass"], scope=PyiScope())
        result = builder.build_class(cls)
        assert result
        assert "@dataclass" in result

    def test_build_scope_with_assignments(self):
        """Test building a scope with assignments."""
        builder = Builder()
        scope = PyiScope(assignments=[PyiAssignment("x: int")])
        result = builder.build_scope(scope)
        assert result
        assert "x: int" in result

    def test_build_scope_with_functions(self):
        """Test building a scope with functions."""
        builder = Builder()
        scope = PyiScope(
            functions=[PyiFunction("hello", is_async=False, signature=PyiSignature())]
        )
        result = builder.build_scope(scope)
        assert result
        assert "hello" in result

    def test_build_scope_with_private_function_excluded(self):
        """Test that private functions are excluded by default."""
        builder = Builder(include_private=False)
        scope = PyiScope(
            functions=[
                PyiFunction("_private", is_async=False, signature=PyiSignature())
            ]
        )
        result = builder.build_scope(scope)
        assert result
        assert "_private" not in result

    def test_build_scope_with_private_function_included(self):
        """Test that private functions are included when specified."""
        builder = Builder(include_private=True)
        scope = PyiScope(
            functions=[
                PyiFunction("_private", is_async=False, signature=PyiSignature())
            ]
        )
        result = builder.build_scope(scope)
        assert result
        assert "_private" in result

    def test_build_enum_with_name(self):
        """Test building a named enum."""
        builder = Builder()
        enum = PyiEnum(enum_name="Colors", names=["RED", "GREEN", "BLUE"])
        result = builder.build_enum(enum)
        assert result
        assert "class Colors" in result
        assert "RED: int" in result

    def test_build_enum_without_name(self):
        """Test building an unnamed enum."""
        builder = Builder()
        enum = PyiEnum(enum_name=None, names=["RED", "GREEN", "BLUE"])
        result = builder.build_enum(enum)
        assert result
        assert "RED: int" in result
        assert "GREEN: int" in result

    def test_build_enum_empty(self):
        """Test building an empty enum."""
        builder = Builder()
        enum = PyiEnum(enum_name="Empty", names=[])
        result = builder.build_enum(enum)
        assert result is None

    def test_build_module_simple(self):
        """Test building a simple module."""
        builder = Builder()
        module = PyiModule(scope=PyiScope())
        result = builder.build_module(module)
        assert isinstance(result, str)

    def test_build_module_with_docstring(self):
        """Test building a module with a docstring."""
        builder = Builder()
        module = PyiModule(doc='"""Module docstring."""', scope=PyiScope())
        result = builder.build_module(module)
        assert "Module docstring" in result

    def test_build_module_with_imports(self):
        """Test building a module with imports."""
        builder = Builder()
        module = PyiModule(imports=[PyiImport("import os")], scope=PyiScope())
        result = builder.build_module(module)
        assert "import os" in result

    def test_build_module_complete(self):
        """Test building a complete module."""
        builder = Builder()
        module = PyiModule(
            doc='"""A test module."""',
            imports=[PyiImport("from typing import Dict")],
            scope=PyiScope(
                functions=[
                    PyiFunction("test", is_async=False, signature=PyiSignature())
                ]
            ),
        )
        result = builder.build_module(module)
        assert "test module" in result
        assert "import" in result or "Dict" in result
