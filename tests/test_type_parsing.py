"""Focused tests for Cython-to-Python type parsing helpers."""

from __future__ import annotations

from stubgen_pyx.conversion import type_parsing
from stubgen_pyx.conversion.type_parsing import Nodes as type_parsing_Nodes
from stubgen_pyx.parsing.parser import parse_pyx


def _first_node(source: str, node_type: type):
    tree = parse_pyx(source).source_ast
    pending = [getattr(tree, "body", None)]
    while pending:
        node = pending.pop(0)
        if node is None:
            continue
        if isinstance(node, node_type):
            return node
        for attr in getattr(node, "child_attrs", ()):
            child = getattr(node, attr, None)
            if isinstance(child, list):
                pending.extend(child)
            elif child is not None:
                pending.append(child)
    raise AssertionError(f"No {node_type.__name__} found")


def test_parameterize_builtin_generic_handles_none_and_unknown():
    assert type_parsing.parameterize_builtin_generic(None) is None
    assert (
        type_parsing.parameterize_builtin_generic("tuple") == "tuple[typing.Any, ...]"
    )
    assert type_parsing.parameterize_builtin_generic("Custom") == "Custom"


def test_declarator_helpers_unwrap_and_reject_non_function():
    node = _first_node("cdef int value", type_parsing_Nodes.CVarDefNode)
    declarator = node.declarators[0]
    assert type_parsing._declarator_name(declarator) == "value"
    assert type_parsing._get_func_decl_type(declarator, "int") is None


def test_extract_type_handles_unknown_node_and_module_qualified_type():
    unknown_node = _first_node(
        "cdef int value", type_parsing_Nodes.CVarDefNode
    ).declarators[0]
    assert type_parsing.extract_type_from_base_type(unknown_node) is None
    node = _first_node("cdef pkg.Widget value", type_parsing_Nodes.CVarDefNode)
    assert type_parsing.extract_type_from_base_type(node) == "pkg.Widget"


def test_unnamed_c_variable_defaults_to_any():
    node = _first_node("cdef public x", type_parsing_Nodes.CVarDefNode)
    assert type_parsing.extract_type_from_base_type(node) == "typing.Any"


def test_tuple_type_is_rendered_with_each_component():
    node = _first_node(
        "cpdef (int, str) value(int item, str label):\n    return (item, label)",
        type_parsing_Nodes.CFuncDefNode,
    )
    assert type_parsing.extract_type_from_base_type(node) == "tuple[int, str]"


def test_template_type_with_missing_base_returns_none():
    node = _first_node(
        "from libcpp.vector cimport vector\ncdef vector[int] value",
        type_parsing_Nodes.CVarDefNode,
    )
    templated = node.base_type
    templated.base_type_node = None
    assert type_parsing._extract_templated_type(templated) is None


def test_template_type_without_arguments_uses_base_name():
    node = _first_node("cdef Foo[int] value", type_parsing_Nodes.CVarDefNode)
    templated = node.base_type
    templated.positional_args = []
    assert type_parsing._extract_templated_type(templated) == "Foo"


def test_array_type_handles_missing_and_empty_scalar_names():
    node = _first_node("cdef int[3] values", type_parsing_Nodes.CVarDefNode).base_type
    original_base = node.base_type_node
    node.base_type_node = None
    assert type_parsing._extract_array_type(node) is None
    node.base_type_node = original_base
    original_name = original_base.name
    original_base.name = ""
    assert type_parsing._extract_array_type(node) is None
    original_base.name = original_name


def test_memoryview_unknown_scalar_falls_back_to_memoryview():
    node = _first_node("cdef Custom[:] view", type_parsing_Nodes.CVarDefNode).base_type
    assert type_parsing._extract_memoryview_type(node) == "memoryview"
