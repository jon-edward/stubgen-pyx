"""Tests for postprocessing pipeline."""

from __future__ import annotations

import ast

from stubgen_pyx.postprocessing.pipeline import postprocessing_pipeline, _ast_transforms
from stubgen_pyx.config import StubgenPyxConfig


def test_pipeline_with_all_disabled():
    """Test pipeline when all postprocessing is disabled."""
    pyi_code = """
import os
import sys

def hello() -> None:
    pass
"""

    config = StubgenPyxConfig(
        sort_imports=False,
        trim_imports=False,
        normalize_names=False,
        deduplicate_imports=False,
        trim_not_defined=False,
    )

    result = postprocessing_pipeline(pyi_code, config)

    # Should still include attribution
    assert "stubgen-pyx" in result


def test_pipeline_excludes_attribution():
    """Test that attribution can be excluded."""
    pyi_code = "def hello(): pass"

    config = StubgenPyxConfig(exclude_attribution=True, sort_imports=False)
    result = postprocessing_pipeline(pyi_code, config)

    assert "stubgen-pyx" not in result


def test_ast_transforms_all_operations():
    """Test AST transforms with all operations enabled."""
    code = """
import sys
import os
from typing import Optional
from typing import Dict

def hello() -> None:
    x: str = "test"
"""

    tree = ast.parse(code)
    config = StubgenPyxConfig()
    transformed = _ast_transforms(tree, config)

    assert isinstance(transformed, ast.Module)


def test_ast_transforms_trim_only():
    """Test AST transforms with only trimming."""
    code = """
import sys
import os

def hello():
    pass
"""

    tree = ast.parse(code)
    config = StubgenPyxConfig(
        trim_imports=True,
        normalize_names=False,
        deduplicate_imports=False,
        trim_not_defined=False,
    )
    transformed = _ast_transforms(tree, config)

    imports = [node for node in ast.walk(transformed) if isinstance(node, ast.Import)]
    assert len(imports) == 0  # Both sys and os are unused


def test_ast_transforms_normalize_only():
    """Test AST transforms with only normalization."""
    code = """
def hello(x: bint) -> unicode:
    pass
"""

    tree = ast.parse(code)
    config = StubgenPyxConfig(
        trim_imports=False,
        normalize_names=True,
        deduplicate_imports=False,
        trim_not_defined=False,
    )
    transformed = _ast_transforms(tree, config)

    result = ast.unparse(transformed)
    assert "bool" in result
    assert "str" in result


def test_pipeline_with_trim_imports():
    """Test that trim imports removes unused imports."""
    pyi_code = """
import os
import sys
import json

def process_data(filename: str) -> str:
    return filename
"""

    config = StubgenPyxConfig(trim_imports=True)
    result = postprocessing_pipeline(pyi_code, config)

    assert "import os" not in result
    assert "import sys" not in result
    assert "import json" not in result


def test_pipeline_with_sort_imports():
    """Test that imports are sorted."""
    pyi_code = """
import z_module
import a_module

def hello(): pass
"""

    config = StubgenPyxConfig(sort_imports=True, trim_imports=False)
    result = postprocessing_pipeline(pyi_code, config)

    a_idx = result.find("a_module")
    z_idx = result.find("z_module")
    assert a_idx < z_idx


def test_pipeline_with_normalize_names():
    """Test that Cython types are normalized."""
    pyi_code = """
def func(x: bint, y: unicode) -> long: pass
"""

    config = StubgenPyxConfig(
        normalize_names=True, sort_imports=False, trim_imports=False
    )
    result = postprocessing_pipeline(pyi_code, config)

    assert "bool" in result
    assert "str" in result
    assert "int" in result


def test_pipeline_preserves_code():
    """Test that pipeline preserves function code."""
    pyi_code = """
def greet(name: str) -> str:
    '''Greet a person.'''
    return f"Hello, {name}!"

class MyClass:
    def method(self) -> None:
        pass
"""

    config = StubgenPyxConfig()
    result = postprocessing_pipeline(pyi_code, config)

    assert "def greet" in result
    assert "class MyClass" in result
    assert "def method" in result


def test_pipeline_trim_not_defined_replaces_unknown():
    """Test that undefined names in annotations are replaced with ..."""
    pyi_code = "def foo(x: UndefinedType) -> int: ..."

    config = StubgenPyxConfig(
        trim_not_defined=True,
        trim_imports=False,
        sort_imports=False,
    )
    result = postprocessing_pipeline(pyi_code, config)

    assert "UndefinedType" not in result


def test_pipeline_trim_not_defined_warns(caplog):
    """Test that replacing undefined names emits a warning."""
    import logging

    pyi_code = "def foo(x: MysteryType) -> int: ..."

    config = StubgenPyxConfig(
        trim_not_defined=True,
        trim_imports=False,
        sort_imports=False,
        exclude_attribution=True,
    )
    with caplog.at_level(logging.WARNING):
        postprocessing_pipeline(pyi_code, config)

    assert "MysteryType" in caplog.text


def test_pipeline_trim_not_defined_disabled_preserves():
    """Test that trim_not_defined=False leaves unknown names alone."""
    pyi_code = "def foo(x: UndefinedType) -> int: ..."

    config = StubgenPyxConfig(
        trim_not_defined=False,
        trim_imports=False,
        sort_imports=False,
        exclude_attribution=True,
    )
    result = postprocessing_pipeline(pyi_code, config)

    assert "UndefinedType" in result


def test_pipeline_transform_order_trim_before_trim_not_defined():
    """trim_imports must run before trim_not_defined so imports define what's available."""
    # SomeType is imported; trim_imports must not remove it before trim_not_defined
    # sees that SomeType is used as an annotation.
    pyi_code = """
from mylib import SomeType

def foo(x: SomeType) -> int: ...
"""

    config = StubgenPyxConfig(
        trim_imports=True,
        trim_not_defined=True,
        sort_imports=False,
        exclude_attribution=True,
    )
    result = postprocessing_pipeline(pyi_code, config)

    # SomeType annotation should still be present (kept by trim_not_defined because
    # import still defines it when trim_not_defined runs)
    assert "SomeType" in result
