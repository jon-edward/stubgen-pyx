"""Integration tests for stub generation."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from stubgen_pyx.stubgen import StubgenPyx
from stubgen_pyx.config import StubgenPyxConfig


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_outdir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def test_convert_empty_pyx_file(temp_dir):
    """Test converting a minimal .pyx file."""
    pyx_file = temp_dir / "test.pyx"
    pyx_file.write_text("""
# Empty module
""")

    config = StubgenPyxConfig(verbose=True)
    stubgen = StubgenPyx(config=config)

    result = stubgen.convert_str(pyx_file.read_text(), pyx_path=pyx_file)

    assert isinstance(result, str)
    assert "from __future__ import annotations" in result
    assert "stubgen-pyx" in result  # Stubgen attribution


def test_convert_with_function(temp_dir):
    """Test converting a .pyx file with a function."""
    pyx_file = temp_dir / "test.pyx"
    pyx_file.write_text("""
def greet(name: str) -> str:
    '''Greet someone.'''
    return f"Hello, {name}!"
""")

    stubgen = StubgenPyx()
    result = stubgen.convert_str(pyx_file.read_text(), pyx_path=pyx_file)

    assert "def greet" in result
    assert "str" in result
    assert "greet someone" in result.lower()


def test_convert_glob_empty_pattern(temp_dir):
    """Test glob conversion with no matches."""
    config = StubgenPyxConfig(verbose=True)
    stubgen = StubgenPyx(config=config)

    pattern = str(temp_dir / "*.pyx")
    results = stubgen.convert_glob(pattern)

    assert results == []


def test_convert_glob_single_file(temp_dir):
    """Test glob conversion with a single file."""
    pyx_file = temp_dir / "test.pyx"
    pyx_file.write_text("def hello(): pass")

    config = StubgenPyxConfig()
    stubgen = StubgenPyx(config=config)

    pattern = str(temp_dir / "*.pyx")
    results = stubgen.convert_glob(pattern)

    assert len(results) == 1
    assert results[0].success is True
    assert results[0].pyx_file == pyx_file

    pyi_file = temp_dir / "test.pyi"
    assert pyi_file.exists()


def test_convert_glob_multiple_files(temp_dir):
    """Test glob conversion with multiple files."""
    for i in range(3):
        pyx_file = temp_dir / f"test{i}.pyx"
        pyx_file.write_text(f"def func{i}(): pass")

    config = StubgenPyxConfig()
    stubgen = StubgenPyx(config=config)

    pattern = str(temp_dir / "*.pyx")
    results = stubgen.convert_glob(pattern)

    assert len(results) == 3
    assert all(r.success for r in results)

    # Verify all .pyi files exist
    for i in range(3):
        assert (temp_dir / f"test{i}.pyi").exists()


def test_convert_glob_with_pxd_file(temp_dir):
    """Test glob conversion that includes .pxd files."""
    pyx_file = temp_dir / "test.pyx"
    pyx_file.write_text("def greet(name: str) -> str: pass")

    pxd_file = temp_dir / "test.pxd"
    pxd_file.write_text("cdef extern from 'test.h': void c_func()")

    config = StubgenPyxConfig(no_pxd_to_stubs=False)
    stubgen = StubgenPyx(config=config)

    results = stubgen.convert_glob(str(temp_dir / "*.pyx"))

    assert len(results) == 1
    assert results[0].success is True


def test_compile_str_to_module_deduplicates_pxd_assignments(temp_dir):
    """Test that assignments from .pxd and .pyx are merged without duplicates."""
    pyx_file = temp_dir / "test.pyx"
    pyx_file.write_text("x = 1\n")

    pxd_content = "x: int = 1\n"
    stubgen = StubgenPyx()

    module = stubgen.compile_str_to_module(
        pyx_file.read_text(),
        pxd_str=pxd_content,
        pyx_path=pyx_file,
    )

    assert len(module.scope.assignments) == 1
    assert module.scope.assignments[0].statement.startswith("x")


def test_compile_str_to_module_merges_pxd_class_declarations(temp_dir):
    """Test that pxd class declarations are merged into existing pyx classes."""
    pyx_file = temp_dir / "test.pyx"
    pyx_file.write_text(
        """
class Config:
    python_attribute: int = 1

    def pyx_method(self):
        pass
"""
    )

    pxd_content = """
cdef class Config:
    cdef public int pxd_value
"""
    stubgen = StubgenPyx()

    module = stubgen.compile_str_to_module(
        pyx_file.read_text(),
        pxd_str=pxd_content,
        pyx_path=pyx_file,
    )

    config_cls = next(cls for cls in module.scope.classes if cls.name == "Config")
    assert len(module.scope.classes) == 1

    assert any(func.name == "pyx_method" for func in config_cls.scope.functions)
    assert any(
        assign.statement.startswith("pxd_value")
        for assign in config_cls.scope.assignments
    )
    assert any(
        assign.statement.startswith("python_attribute")
        for assign in config_cls.scope.assignments
    )


def test_convert_glob_with_standalone_pxd_file(temp_dir):
    """Test glob conversion of a standalone .pxd file when no .pyx exists."""
    pxd_file = temp_dir / "test.pxd"
    pxd_file.write_text("cdef extern from 'test.h': void c_func()")

    stubgen = StubgenPyx()
    results = stubgen.convert_glob(str(temp_dir / "*.pyx"))

    assert len(results) == 1
    assert results[0].success is True
    assert results[0].pyx_file == pxd_file
    assert (temp_dir / "test.pyi").exists()


def test_convert_glob_continue_on_error(temp_dir):
    """Test glob conversion with error handling."""
    # Valid file
    valid_file = temp_dir / "valid.pyx"
    valid_file.write_text("def hello(): pass")

    # Invalid file (syntax error)
    invalid_file = temp_dir / "invalid.pyx"
    invalid_file.write_text("def broken( pass")  # Missing closing paren

    config = StubgenPyxConfig(continue_on_error=True)
    stubgen = StubgenPyx(config=config)

    results = stubgen.convert_glob(str(temp_dir / "*.pyx"))

    # Should have 2 results
    assert len(results) == 2

    # One success, one failure
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    assert len(successful) == 1
    assert len(failed) == 1


def test_convert_glob_no_continue_on_error(temp_dir):
    """Test that conversion stops on first error."""
    # Invalid file
    invalid_file = temp_dir / "invalid.pyx"
    invalid_file.write_text("def broken( pass")  # Missing closing paren

    config = StubgenPyxConfig(continue_on_error=False)
    stubgen = StubgenPyx(config=config)

    with pytest.raises(Exception):
        stubgen.convert_glob(str(temp_dir / "invalid.pyx"))


def test_config_applied_in_conversion(temp_dir):
    """Test that config options are applied during conversion."""
    pyx_file = temp_dir / "test.pyx"
    pyx_file.write_text("""
import os
import sys

def hello():
    '''Say hello.'''
    print("hello")
""")

    # With trim imports disabled, all imports should be present
    config_no_trim = StubgenPyxConfig(no_trim_imports=True)
    stubgen_no_trim = StubgenPyx(config=config_no_trim)
    result_no_trim = stubgen_no_trim.convert_str(
        pyx_file.read_text(), pyx_path=pyx_file
    )

    # With trim imports enabled, unused imports should be removed
    config_trim = StubgenPyxConfig(no_trim_imports=False)
    stubgen_trim = StubgenPyx(config=config_trim)
    result_trim = stubgen_trim.convert_str(pyx_file.read_text(), pyx_path=pyx_file)

    # The trimmed version should be shorter or equal
    assert len(result_trim) <= len(result_no_trim)


def test_convert_glob_multiple_files_in_output_dir(temp_dir, temp_outdir):
    """Test conversion of multiple files in separate out dir."""
    for i in range(3):
        pyx_file = temp_dir / f"test{i}.pyx"
        pyx_file.write_text(f"def func{i}(): pass")

    config = StubgenPyxConfig()
    stubgen = StubgenPyx(config=config)

    pattern = str(temp_dir / "*.pyx")
    results = stubgen.convert_glob(pattern, output_dir=temp_outdir)

    assert len(results) == 3
    assert all(r.success for r in results)

    # Verify all .pyi files exist in output dir and NOT in source dir
    for i in range(3):
        assert not (temp_dir / f"test{i}.pyi").exists()
        assert (temp_outdir / f"test{i}.pyi").exists()


def test_convert_multiple_files(temp_dir):
    """Test glob conversion with multiple files."""
    pyx_files = [temp_dir / f"test{i}.pyx" for i in range(3)]

    # prepare the input files
    for i, pyx_file in enumerate(pyx_files):
        pyx_file.write_text(f"def func{i}(): pass")

    config = StubgenPyxConfig()
    stubgen = StubgenPyx(config=config)

    results = stubgen.convert_multiple_files(pyx_files)

    assert len(results) == len(pyx_files)
    assert all(r.success for r in results)

    # Verify all .pyi files exist
    for pyx_file in pyx_files:
        assert pyx_file.with_suffix(".pyi").exists()


def test_convert_multiple_files_in_output_dir(temp_dir, temp_outdir):
    """Test glob conversion with multiple files."""
    pyx_files = [temp_dir / f"test{i}.pyx" for i in range(3)]

    # prepare the input files
    for i, pyx_file in enumerate(pyx_files):
        pyx_file.write_text(f"def func{i}(): pass")

    config = StubgenPyxConfig()
    stubgen = StubgenPyx(config=config)

    results = stubgen.convert_multiple_files(pyx_files, output_dir=temp_outdir)

    assert len(results) == len(pyx_files)
    assert all(r.success for r in results)

    pyi_files = [
        temp_outdir / pyx_file.with_suffix(".pyi").name for pyx_file in pyx_files
    ]

    # Verify all .pyi files exist in output dir and NOT in source dir
    for pyx_file, pyi_file in zip(pyx_files, pyi_files):
        assert pyi_file.exists()
        assert not pyx_file.with_suffix(".pyi").exists()


def test_convert_single_file(temp_dir):
    """Test glob conversion with a single files."""

    # prepare the input file
    pyx_file = temp_dir / "test1.pyx"
    pyx_file.write_text("def func1(): pass")

    config = StubgenPyxConfig()
    stubgen = StubgenPyx(config=config)

    # No already existing .pyi file
    assert not pyx_file.with_suffix(".pyi").exists()

    result = stubgen.convert_single_file(pyx_file)
    assert result.success

    # Verify the .pyi file exist
    assert pyx_file.with_suffix(".pyi").exists()


def test_convert_single_file_in_output_dir(temp_dir, temp_outdir):
    """Test glob conversion with a single files in output dir."""

    # prepare the input file
    pyx_file = temp_dir / "test1.pyx"
    pyx_file.write_text("def func1(): pass")
    pyi_file = temp_outdir / pyx_file.with_suffix(".pyi").name

    config = StubgenPyxConfig()
    stubgen = StubgenPyx(config=config)

    # No already existing .pyi file
    assert not pyi_file.exists()

    result = stubgen.convert_single_file(pyx_file, pyi_file)
    assert result.success

    # Verify the .pyi file exist
    assert not pyx_file.with_suffix(".pyi").exists()
    assert pyi_file.exists()


# ----
def test_convert_multiple_files_dry_run(temp_dir):
    """Test glob conversion with multiple files with dry run."""
    pyx_files = [temp_dir / f"test{i}.pyx" for i in range(3)]

    # prepare the input files
    for i, pyx_file in enumerate(pyx_files):
        pyx_file.write_text(f"def func{i}(): pass")

    config = StubgenPyxConfig()
    stubgen = StubgenPyx(config=config)

    results = stubgen.convert_multiple_files(pyx_files, dry_run=True)

    assert len(results) == len(pyx_files)
    assert all(r.success for r in results)

    # Verify all .pyi files do not exist
    for pyx_file in pyx_files:
        assert not pyx_file.with_suffix(".pyi").exists()


def test_convert_multiple_files_in_output_dir_dry_run(temp_dir, temp_outdir):
    """Test glob conversion with multiple files with dry run."""
    pyx_files = [temp_dir / f"test{i}.pyx" for i in range(3)]

    # prepare the input files
    for i, pyx_file in enumerate(pyx_files):
        pyx_file.write_text(f"def func{i}(): pass")

    config = StubgenPyxConfig()
    stubgen = StubgenPyx(config=config)

    results = stubgen.convert_multiple_files(
        pyx_files, output_dir=temp_outdir, dry_run=True
    )

    assert len(results) == len(pyx_files)
    assert all(r.success for r in results)

    pyi_files = [
        temp_outdir / pyx_file.with_suffix(".pyi").name for pyx_file in pyx_files
    ]

    # Verify all .pyi files do not exist in output dir and NOT in source dir
    for pyx_file, pyi_file in zip(pyx_files, pyi_files):
        assert not pyi_file.exists()
        assert not pyx_file.with_suffix(".pyi").exists()


def test_convert_single_file_dry_run(temp_dir):
    """Test glob conversion with a single files with dry run."""

    # prepare the input file
    pyx_file = temp_dir / "test1.pyx"
    pyx_file.write_text("def func1(): pass")

    config = StubgenPyxConfig()
    stubgen = StubgenPyx(config=config)

    # No already existing .pyi file
    assert not pyx_file.with_suffix(".pyi").exists()

    result = stubgen.convert_single_file(pyx_file, dry_run=True)
    assert result.success

    # Verify the .pyi file do not exist
    assert not pyx_file.with_suffix(".pyi").exists()


def test_type_comment_propagates(temp_dir):
    """`# type: ...` comments on def lines must propagate into the .pyi."""
    pyx_file = temp_dir / "test.pyx"
    pyx_file.write_text(
        """
class Foo:
    def __isub__(self): # type: ignore[]
        return self

    def __iadd__(self, other):  # type: ignore[misc, override]
        return self

def bar(): # type: ignore[no-untyped-def]
    pass

def baz(x): # type: (int) -> int
    return x

def quux():
    pass
"""
    )

    stubgen = StubgenPyx()
    result = stubgen.convert_str(pyx_file.read_text(), pyx_path=pyx_file)

    isub_line = next(line for line in result.splitlines() if "def __isub__" in line)
    assert "# type: ignore[]" in isub_line

    iadd_line = next(line for line in result.splitlines() if "def __iadd__" in line)
    assert "# type: ignore[misc, override]" in iadd_line

    bar_line = next(line for line in result.splitlines() if "def bar" in line)
    assert "# type: ignore[no-untyped-def]" in bar_line

    baz_line = next(line for line in result.splitlines() if "def baz" in line)
    assert "# type: (int) -> int" in baz_line

    quux_line = next(line for line in result.splitlines() if "def quux" in line)
    assert "type:" not in quux_line


def test_type_comment_propagates_after_bracketed_import(temp_dir):
    """A multi-line bracketed import shifts AST line numbers relative to the
    original source. The `# type:` comment lookup must use post-flattening
    line numbers so the comment still attaches to the right def."""
    pyx_file = temp_dir / "test.pyx"
    pyx_file.write_text(
        """
from collections.abc import (
    MutableSet,
    Set as AbstractSet,
)
from typing import Any


def shifted(it: AbstractSet[Any]) -> Any:  # type: ignore[override,misc]
    return it
"""
    )

    stubgen = StubgenPyx()
    result = stubgen.convert_str(pyx_file.read_text(), pyx_path=pyx_file)

    shifted_line = next(line for line in result.splitlines() if "def shifted" in line)
    assert "# type: ignore[override,misc]" in shifted_line


def test_conversion_succeeds_with_comments_inside_brackets(temp_dir):
    """A `#` comment inside a bracketed expression is terminated by its
    newline. Stub generation must not collapse that newline away, or the
    comment swallows the following dict entries and tokenization fails."""
    pyx_file = temp_dir / "test.pyx"
    pyx_file.write_text(
        """
def make_map():
    return {
        1: 'one',     # leading
        2: 'two',     # the comment ends here, dict continues
        3: 'three',
    }


def tagged(): # type: ignore[no-untyped-def]
    pass
"""
    )

    stubgen = StubgenPyx()
    result = stubgen.convert_str(pyx_file.read_text(), pyx_path=pyx_file)

    tagged_line = next(line for line in result.splitlines() if "def tagged" in line)
    assert "# type: ignore[no-untyped-def]" in tagged_line


def test_convert_single_file_in_output_dir_dry_run(temp_dir, temp_outdir):
    """Test glob conversion with a single files in output dir with dry run."""

    # prepare the input file
    pyx_file = temp_dir / "test1.pyx"
    pyx_file.write_text("def func1(): pass")
    pyi_file = temp_outdir / pyx_file.with_suffix(".pyi").name

    config = StubgenPyxConfig()
    stubgen = StubgenPyx(config=config)

    # No already existing .pyi file
    assert not pyi_file.exists()

    result = stubgen.convert_single_file(pyx_file, pyi_file, dry_run=True)
    assert result.success

    # Verify the .pyi file do not exist
    assert not pyx_file.with_suffix(".pyi").exists()
    assert not pyi_file.exists()
