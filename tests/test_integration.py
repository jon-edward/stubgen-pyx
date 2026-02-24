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
    assert "stubgen-pyx" in result  # Epilog


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
    pyx_files = [temp_dir / f"test{i}.pyx"
                 for i in range(3)]

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
    pyx_files = [temp_dir / f"test{i}.pyx"
                 for i in range(3)]

    # prepare the input files
    for i, pyx_file in enumerate(pyx_files):
        pyx_file.write_text(f"def func{i}(): pass")

    config = StubgenPyxConfig()
    stubgen = StubgenPyx(config=config)

    results = stubgen.convert_multiple_files(pyx_files, output_dir=temp_outdir)

    assert len(results) == len(pyx_files)
    assert all(r.success for r in results)

    pyi_files = [temp_outdir / pyx_file.with_suffix(".pyi").name
                 for pyx_file in pyx_files]

    # Verify all .pyi files exist in output dir and NOT in source dir
    for pyx_file, pyi_file in zip(pyx_files, pyi_files):
        assert pyi_file.exists()
        assert not pyx_file.with_suffix(".pyi").exists()


def test_convert_single_file(temp_dir):
    """Test glob conversion with a single files."""

    # prepare the input file
    pyx_file = temp_dir / "test1.pyx"
    pyx_file.write_text(f"def func1(): pass")

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
    pyx_file.write_text(f"def func1(): pass")
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
    pyx_files = [temp_dir / f"test{i}.pyx"
                 for i in range(3)]

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
    pyx_files = [temp_dir / f"test{i}.pyx"
                 for i in range(3)]

    # prepare the input files
    for i, pyx_file in enumerate(pyx_files):
        pyx_file.write_text(f"def func{i}(): pass")

    config = StubgenPyxConfig()
    stubgen = StubgenPyx(config=config)

    results = stubgen.convert_multiple_files(
        pyx_files, output_dir=temp_outdir, dry_run=True)

    assert len(results) == len(pyx_files)
    assert all(r.success for r in results)

    pyi_files = [temp_outdir / pyx_file.with_suffix(".pyi").name
                 for pyx_file in pyx_files]

    # Verify all .pyi files do not exist in output dir and NOT in source dir
    for pyx_file, pyi_file in zip(pyx_files, pyi_files):
        assert not pyi_file.exists()
        assert not pyx_file.with_suffix(".pyi").exists()


def test_convert_single_file_dry_run(temp_dir):
    """Test glob conversion with a single files with dry run."""

    # prepare the input file
    pyx_file = temp_dir / "test1.pyx"
    pyx_file.write_text(f"def func1(): pass")

    config = StubgenPyxConfig()
    stubgen = StubgenPyx(config=config)

    # No already existing .pyi file
    assert not pyx_file.with_suffix(".pyi").exists()

    result = stubgen.convert_single_file(pyx_file, dry_run=True)
    assert result.success

    # Verify the .pyi file do not exist
    assert not pyx_file.with_suffix(".pyi").exists()


def test_convert_single_file_in_output_dir_dry_run(temp_dir, temp_outdir):
    """Test glob conversion with a single files in output dir with dry run."""

    # prepare the input file
    pyx_file = temp_dir / "test1.pyx"
    pyx_file.write_text(f"def func1(): pass")
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
