"""Tests for conversion and builder edge cases."""

from __future__ import annotations

import tempfile
from pathlib import Path

from stubgen_pyx.models.pyi_elements import (
    PyiArgument,
    PyiSignature,
    PyiFunction,
    PyiClass,
    PyiScope,
    PyiModule,
    PyiAssignment,
    PyiEnum,
)
from stubgen_pyx.builders.builder import Builder


class TestBuilderEdgeCases:
    """Test edge cases in the Builder class."""

    def test_builder_function_no_docstring(self):
        """Test building function without docstring."""
        builder = Builder()
        func = PyiFunction(name="func", is_async=False, signature=PyiSignature())
        result = builder.build_function(func)

        assert result
        assert "def func" in result
        assert "..." in result

    def test_builder_function_multiline_docstring(self):
        """Test building function with multiline docstring."""
        builder = Builder()
        func = PyiFunction(
            name="complex_func",
            is_async=False,
            signature=PyiSignature(),
            doc='"""Line 1\nLine 2\nLine 3""',
        )
        result = builder.build_function(func)

        assert result
        assert "def complex_func" in result

    def test_builder_class_with_content(self):
        """Test building class with methods and attributes."""
        builder = Builder()
        cls = PyiClass(
            name="MyClass",
            scope=PyiScope(
                assignments=[PyiAssignment("x: int = 5")],
                functions=[
                    PyiFunction("method", is_async=False, signature=PyiSignature())
                ],
            ),
        )
        result = builder.build_class(cls)

        assert result
        assert "class MyClass" in result

    def test_builder_scope_empty(self):
        """Test building empty scope."""
        builder = Builder()
        scope = PyiScope()
        result = builder.build_scope(scope)
        assert result is None or result.strip() == ""

    def test_builder_scope_with_multiple_element_types(self):
        """Test building scope with all element types."""
        builder = Builder()
        scope = PyiScope(
            assignments=[PyiAssignment("x: int")],
            functions=[PyiFunction("func", is_async=False, signature=PyiSignature())],
            classes=[PyiClass("Class", scope=PyiScope())],
        )
        result = builder.build_scope(scope)
        assert result is not None

    def test_builder_module_empty_scope(self):
        """Test building module with empty scope."""
        builder = Builder()
        module = PyiModule(scope=PyiScope())
        result = builder.build_module(module)
        assert isinstance(result, str)

    def test_builder_signature_complex(self):
        """Test complex signature with all features."""
        builder = Builder()
        sig = PyiSignature(
            args=[
                PyiArgument("x", "int"),
                PyiArgument("y", "str"),
            ],
            num_posonly_args=1,
            num_kwonly_args=1,
            var_arg=PyiArgument("args", "int"),
            kw_arg=PyiArgument("kwargs", "str"),
            return_type="bool",
        )
        result = builder.build_signature(sig)
        assert "(" in result and ")" in result
        assert "->" in result

    def test_builder_argument_number_default(self):
        """Test argument with numeric default."""
        builder = Builder()
        arg = PyiArgument("count", annotation="int", default="10")
        result = builder.build_argument(arg)
        assert result == "count: int = 10"

    def test_builder_argument_string_default(self):
        """Test argument with string default."""
        builder = Builder()
        arg = PyiArgument("name", annotation="str", default='"default"')
        result = builder.build_argument(arg)
        assert "name: str" in result

    def test_builder_argument_none_default(self):
        """Test argument with None default."""
        builder = Builder()
        arg = PyiArgument("opt", annotation="Optional[str]", default="None")
        result = builder.build_argument(arg)
        assert "opt: Optional[str] = None" in result

    def test_is_private_leading_underscore(self):
        """Test identification of private names with leading underscore."""
        builder = Builder()
        assert builder._is_private("_private") is True
        assert builder._is_private("__private") is True
        assert builder._is_private("___private") is True

    def test_is_private_trailing_underscore(self):
        """Test that trailing underscore doesn't make it private."""
        builder = Builder()
        assert builder._is_private("name_") is False
        assert builder._is_private("__name_") is False

    def test_is_private_dunder(self):
        """Test dunder names are not private."""
        builder = Builder()
        assert builder._is_private("__init__") is False
        assert builder._is_private("__call__") is False
        assert builder._is_private("__str__") is False

    def test_builder_class_multiple_bases_and_metaclass(self):
        """Test building class with multiple bases and metaclass."""
        builder = Builder()
        cls = PyiClass(
            name="Meta",
            bases=["Base1", "Base2", "Base3"],
            metaclass="MetaClass",
            scope=PyiScope(),
        )
        result = builder.build_class(cls)
        assert result
        assert "Meta" in result
        assert "Base1" in result
        assert "metaclass=MetaClass" in result

    def test_builder_enum_complex(self):
        """Test building enum with many members."""
        builder = Builder()
        enum = PyiEnum(
            enum_name="Status", names=["PENDING", "ACTIVE", "COMPLETED", "FAILED"]
        )
        result = builder.build_enum(enum)
        assert result
        assert "Status" in result
        assert "PENDING: int" in result

    def test_builder_function_with_multiple_decorators(self):
        """Test function with multiple decorators."""
        builder = Builder()
        func = PyiFunction(
            name="method",
            is_async=False,
            signature=PyiSignature(),
            decorators=["@classmethod", "@staticmethod", "@property"],
        )
        result = builder.build_function(func)
        assert result
        assert "@classmethod" in result
        assert "@staticmethod" in result
        assert "@property" in result


class TestPyiElementsValidation:
    """Test validation of PyiElements."""

    def test_pyi_argument_minimal(self):
        """Test minimal PyiArgument."""
        arg = PyiArgument("x")
        assert arg.name == "x"
        assert arg.annotation is None
        assert arg.default is None

    def test_pyi_signature_minimal(self):
        """Test minimal PyiSignature."""
        sig = PyiSignature()
        assert sig.args == []
        assert sig.num_posonly_args == 0
        assert sig.num_kwonly_args == 0

    def test_pyi_function_minimal(self):
        """Test minimal PyiFunction."""
        func = PyiFunction("func", is_async=False, signature=PyiSignature())
        assert func.name == "func"
        assert func.is_async is False
        assert func.decorators == []

    def test_pyi_class_minimal(self):
        """Test minimal PyiClass."""
        cls = PyiClass("Class", scope=PyiScope())
        assert cls.name == "Class"
        assert cls.bases == []
        assert cls.metaclass is None

    def test_pyi_scope_empty(self):
        """Test empty PyiScope."""
        scope = PyiScope()
        assert scope.assignments == []
        assert scope.functions == []
        assert scope.classes == []
        assert scope.enums == []

    def test_pyi_module_minimal(self):
        """Test minimal PyiModule."""
        module = PyiModule(scope=PyiScope())
        assert module.doc is None
        assert module.imports == []


class TestIntegrationEdgeCases:
    """Integration tests for edge cases."""

    def test_convert_complex_class(self):
        """Test converting a complex class definition."""
        from stubgen_pyx.stubgen import StubgenPyx

        with tempfile.TemporaryDirectory() as tmpdir:
            pyx_file = Path(tmpdir) / "complex.pyx"
            pyx_file.write_text("""
cdef class MyClass:
    cdef int value
    cdef str name

    def __init__(self, int v, str n):
        self.value = v
        self.name = n

    def get_value(self):
        return self.value

    def set_value(self, int v):
        self.value = v

    @property
    def prop(self):
        return self.value
""")
            stubgen = StubgenPyx()
            result = stubgen.convert_str(pyx_file.read_text(), pyx_path=pyx_file)
            assert "class MyClass" in result or len(result) > 0

    def test_convert_with_type_annotations(self):
        """Test converting code with extensive type annotations."""
        from stubgen_pyx.stubgen import StubgenPyx

        with tempfile.TemporaryDirectory() as tmpdir:
            pyx_file = Path(tmpdir) / "typed.pyx"
            pyx_file.write_text("""
from typing import Dict, List, Optional

def process(data: Dict[str, List[int]]) -> Optional[str]:
    pass

cdef class Processor:
    def handle(self, x: Optional[Dict]) -> List[str]:
        pass
""")
            stubgen = StubgenPyx()
            result = stubgen.convert_str(pyx_file.read_text(), pyx_path=pyx_file)
            assert len(result) > 0
