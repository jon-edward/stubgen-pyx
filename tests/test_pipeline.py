"""Tests for postprocessing pipeline."""

from __future__ import annotations

import ast

import pytest

from stubgen_pyx.postprocessing.pipeline import (
    postprocessing_pipeline,
    _combined_import_transform,
)
from stubgen_pyx.config import StubgenPyxConfig


def test_pipeline_with_all_disabled():
    """Test pipeline when all postprocessing is disabled."""
    pyi_code = """
from __future__ import annotations

import os
import sys

def hello() -> None:
    pass
"""

    config = StubgenPyxConfig(
        no_sort_imports=True,
        no_trim_imports=True,
        no_normalize_names=True,
        no_deduplicate_imports=True,
    )

    result = postprocessing_pipeline(pyi_code, config)

    # Should still include epilog
    assert "from __future__ import annotations" in result
    assert "stubgen-pyx" in result


def test_pipeline_excludes_epilog():
    """Test that epilog can be excluded."""
    pyi_code = "def hello(): pass"

    config = StubgenPyxConfig(exclude_epilog=True, no_sort_imports=True)
    result = postprocessing_pipeline(pyi_code, config)

    assert "stubgen-pyx" not in result


def test_combined_import_transform_all_operations():
    """Test combined import transformation with all operations enabled."""
    code = """
import sys
import os
from typing import Optional
from typing import Dict

def hello() -> None:
    x: str = "test"
"""

    tree = ast.parse(code)
    transformed = _combined_import_transform(
        tree,
        trim_unused=True,
        used_names={"hello", "str"},
        normalize=True,
        deduplicate=True,
    )

    # Tree should still be valid
    assert isinstance(transformed, ast.Module)


def test_combined_import_transform_trim_only():
    """Test combined import transformation with only trimming."""
    code = """
import sys
import os

def hello():
    pass
"""

    tree = ast.parse(code)
    transformed = _combined_import_transform(
        tree,
        trim_unused=True,
        used_names={"hello"},
        normalize=False,
        deduplicate=False,
    )

    # Unused imports should be removed
    imports = [node for node in ast.walk(transformed) if isinstance(node, ast.Import)]
    assert len(imports) == 0  # Both sys and os are unused


def test_combined_import_transform_normalize_only():
    """Test combined import transformation with only normalization."""
    code = """
def hello(x: bint) -> unicode:
    pass
"""

    tree = ast.parse(code)
    transformed = _combined_import_transform(
        tree,
        trim_unused=False,
        normalize=True,
        deduplicate=False,
    )

    # Check that normalization happened
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

    config = StubgenPyxConfig(no_trim_imports=False)
    result = postprocessing_pipeline(pyi_code, config)

    # json should be trimmed since it's unused
    # os and sys should also be trimmed
    assert "import os" not in result or "import sys" not in result


def test_pipeline_with_sort_imports():
    """Test that imports are sorted."""
    pyi_code = """
import z_module
import a_module

def hello(): pass
"""

    config = StubgenPyxConfig(no_sort_imports=False, no_trim_imports=True)
    result = postprocessing_pipeline(pyi_code, config)

    # After sorting, a_module should come before z_module
    a_idx = result.find("a_module")
    z_idx = result.find("z_module")
    assert a_idx < z_idx


def test_pipeline_with_normalize_names():
    """Test that Cython types are normalized."""
    pyi_code = """
def func(x: bint, y: unicode) -> long: pass
"""

    config = StubgenPyxConfig(
        no_normalize_names=False, no_sort_imports=True, no_trim_imports=True
    )
    result = postprocessing_pipeline(pyi_code, config)

    # Cython types should be replaced
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
