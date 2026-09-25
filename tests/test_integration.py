"""Integration tests for stub generation."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from Cython.Compiler.Errors import CompileError

from stubgen_pyx.config import StubgenPyxConfig
from stubgen_pyx.stubgen import StubgenPyx


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
    pxd_file.write_text('cdef extern from "test.h":\n    void c_func()\n')

    config = StubgenPyxConfig(pxd_to_stubs=True)
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
    pxd_file.write_text('cdef extern from "test.h":\n    void c_func()\n')

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

    # A real syntax error now surfaces as Cython's own `CompileError`
    # (raised by `Context.parse` itself) rather than `tokenize.TokenError`
    # -- there's no more preprocessing tokenize pass in front of it; the
    # real Cython parser sees this file directly.
    with pytest.raises(CompileError):
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
    config_no_trim = StubgenPyxConfig(trim_imports=False)
    stubgen_no_trim = StubgenPyx(config=config_no_trim)
    result_no_trim = stubgen_no_trim.convert_str(
        pyx_file.read_text(), pyx_path=pyx_file
    )

    # With trim imports enabled, unused imports should be removed
    config_trim = StubgenPyxConfig(trim_imports=True)
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
    # A `# type: (int) -> int` comment is now parsed into a real
    # annotation rather than just carried along as a trailing comment
    # (see converter.py's `_apply_type_comments`) -- the raw comment
    # text is dropped once its information is real annotations on the
    # signature, so it's not redundantly shown here too.
    assert "def baz(x: int) -> int: ..." in baz_line
    assert "type:" not in baz_line

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


def test_type_comment_before_body_is_parsed(temp_dir):
    """A `# type:` comment as the first line of the body (the common PEP
    484 idiom, as opposed to trailing the `def` line) is now detected
    and parsed into real annotations, not silently dropped."""
    pyx_file = temp_dir / "test.pyx"
    pyx_file.write_text(
        """
def foo(a, b):
    # type: (int, str) -> bool
    pass
"""
    )
    stubgen = StubgenPyx()
    result = stubgen.convert_str(pyx_file.read_text(), pyx_path=pyx_file)
    assert "def foo(a: int, b: str) -> bool: ..." in result
    assert "type:" not in result


def test_type_comment_before_body_survives_docstring(temp_dir):
    """The type comment is still found even when a docstring follows it
    -- PEP 484 requires the comment to be the first line of the body,
    before any docstring."""
    pyx_file = temp_dir / "test.pyx"
    pyx_file.write_text(
        '''
def foo(a, b):
    # type: (int, str) -> bool
    """docstring"""
    pass
'''
    )
    stubgen = StubgenPyx()
    result = stubgen.convert_str(pyx_file.read_text(), pyx_path=pyx_file)
    assert "def foo(a: int, b: str) -> bool:" in result
    assert '"""docstring"""' in result


def test_per_argument_type_comments(temp_dir):
    """Each argument's own trailing `# type: TYPE` comment is applied."""
    pyx_file = temp_dir / "test.pyx"
    pyx_file.write_text(
        """
def foo(
    a,  # type: int
    b,  # type: str
):
    # type: (...) -> bool
    pass
"""
    )
    stubgen = StubgenPyx()
    result = stubgen.convert_str(pyx_file.read_text(), pyx_path=pyx_file)
    assert "def foo(a: int, b: str) -> bool: ..." in result


def test_per_argument_type_comment_only_applies_to_last_arg_on_line(temp_dir):
    """Two arguments sharing a source line: a trailing type comment
    unambiguously belongs only to the last one (PEP 484's rule)."""
    pyx_file = temp_dir / "test.pyx"
    pyx_file.write_text(
        """
def foo(a, b,  # type: str
         c):
    pass
"""
    )
    stubgen = StubgenPyx()
    result = stubgen.convert_str(pyx_file.read_text(), pyx_path=pyx_file)
    assert "def foo(a, b: str, c): ..." in result


def test_type_comment_self_omitted_from_signature_comment(temp_dir):
    """A whole-signature comment on a method conventionally omits
    self/cls; the comment's types still align to the remaining args."""
    pyx_file = temp_dir / "test.pyx"
    pyx_file.write_text(
        """
cdef class Foo:
    def bar(self, a, b):
        # type: (int, str) -> bool
        pass
"""
    )
    stubgen = StubgenPyx()
    result = stubgen.convert_str(pyx_file.read_text(), pyx_path=pyx_file)
    assert "def bar(self, a: int, b: str) -> bool: ..." in result


def test_type_comment_never_overrides_explicit_annotation(temp_dir):
    """A type comment is a fallback only -- a real, explicit annotation
    is never replaced by one."""
    pyx_file = temp_dir / "test.pyx"
    pyx_file.write_text(
        """
def foo(a: float, b):
    # type: (int, str) -> bool
    pass
"""
    )
    stubgen = StubgenPyx()
    result = stubgen.convert_str(pyx_file.read_text(), pyx_path=pyx_file)
    assert "def foo(a: float, b: str) -> bool: ..." in result


def test_type_comment_ignore_preserved_even_with_arguments(temp_dir):
    """`# type: ignore[...]` must never be mistaken for a per-argument
    type comment just because it trails a def line with arguments on it
    -- the per-argument-comment regex must not backtrack past its
    negative lookahead and treat `ignore[...]` as the last argument's
    own type."""
    pyx_file = temp_dir / "test.pyx"
    pyx_file.write_text(
        """
def foo(a, b):  # type: ignore[no-untyped-def]
    pass
"""
    )
    stubgen = StubgenPyx()
    result = stubgen.convert_str(pyx_file.read_text(), pyx_path=pyx_file)
    foo_line = next(line for line in result.splitlines() if "def foo" in line)
    assert "def foo(a, b):" in foo_line
    assert "# type: ignore[no-untyped-def]" in foo_line


def test_type_comment_unparseable_falls_back_to_raw_display(temp_dir):
    """A `# type:` comment that doesn't fit the whole-signature or
    per-argument shape is still shown verbatim rather than silently
    dropped."""
    pyx_file = temp_dir / "test.pyx"
    pyx_file.write_text(
        """
def foo(a, b):
    # type: something weird not a signature
    pass
"""
    )
    stubgen = StubgenPyx()
    result = stubgen.convert_str(pyx_file.read_text(), pyx_path=pyx_file)
    foo_line = next(line for line in result.splitlines() if "def foo" in line)
    assert "# type: something weird not a signature" in foo_line


def test_unhashable_class_assignment_is_mypy_compatible(temp_dir):
    """Class-level ``__hash__ = None`` needs an ignore in pyi files."""
    pyx_file = temp_dir / "test.pyx"
    pyx_file.write_text(
        """
__hash__ = None

cdef class UnhashableThing:
    __hash__ = None
"""
    )

    stubgen = StubgenPyx()
    result = stubgen.convert_str(pyx_file.read_text(), pyx_path=pyx_file)

    assert "\n__hash__ = None\n" in result
    assert "    __hash__ = None # type: ignore[assignment]" in result


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


def test_exclude_patterns(temp_dir):
    """Test glob conversion with exclude patterns."""
    pyx_file = temp_dir / "test1.pyx"
    pyx_file.write_text("def func1(): pass")

    stubgen = StubgenPyx()
    result = stubgen.convert_glob(
        str(temp_dir / "*.pyx"), exclude_patterns=[str(temp_dir / "**")]
    )
    assert len(result) == 0  # recursive exclude should work

    result = stubgen.convert_glob(
        str(temp_dir / "*.pyx"), exclude_patterns=[str(temp_dir / "test1.pyx")]
    )
    assert len(result) == 0  # explicit exclude should work

    result = stubgen.convert_glob(str(temp_dir / "*.pyx"))
    assert len(result) == 1  # no exclude should work

    result = stubgen.convert_glob(
        str(temp_dir / "*.pyx"), exclude_patterns=str(temp_dir / "test1.pyx")
    )
    assert len(result) == 0  # passing a single exclude should work

    nested_dir = temp_dir / "nested"
    nested_dir.mkdir()
    nested_file = nested_dir / "test2.pyx"
    nested_file.write_text("def func2(): pass")

    result = stubgen.convert_glob(str(temp_dir / "**" / "*.pyx"))
    assert len(result) == 2  # recursive glob without exclude matches both files

    result = stubgen.convert_glob(
        str(temp_dir / "**" / "*.pyx"),
        exclude_patterns=[str(temp_dir / "nested" / "*")],
    )
    assert len(result) == 1  # explicit exclude should work

    result = stubgen.convert_glob(
        str(temp_dir / "**" / "*.pyx"),
        exclude_patterns=[str(temp_dir / "test1.pyx"), str(temp_dir / "nested" / "*")],
    )
    assert len(result) == 0  # multiple excludes should be additive


def test_struct_funcptr_imports_typing_callable(temp_dir):
    """Test that a struct with a function pointer imports typing.Callable from typing."""

    pyx_file = temp_dir / "test.pxd"
    pyx_file.write_text("""
cdef extern from *:
    ctypedef struct MyStruct:
        void (*callback)(void*)
        int value
""")

    stubgen = StubgenPyx()
    result = stubgen.convert_str(pyx_file.read_text(), pyx_path=pyx_file)

    assert "from typing import Any, Callable" in result
    assert "callback: Callable[[Any], None]" in result


class TestResolveCtypedefAliases:
    """`resolve_ctypedef_aliases` substitutes a `ctypedef` alias for its
    resolved type in argument/return annotations and class attributes."""

    def test_default_off_keeps_alias_name(self, temp_dir):
        """Off by default -- existing output is unaffected."""
        pyx_file = temp_dir / "test.pyx"
        pyx_file.write_text("""
ctypedef double MyFloat

def f(y: MyFloat) -> MyFloat:
    pass
""")
        result = StubgenPyx().convert_str(pyx_file.read_text(), pyx_path=pyx_file)
        assert "def f(y: MyFloat) -> MyFloat: ..." in result

    def test_resolves_bare_c_style_argument_and_return(self, temp_dir):
        pyx_file = temp_dir / "test.pyx"
        pyx_file.write_text("""
ctypedef double MyFloat

cpdef MyFloat f(MyFloat y):
    pass
""")
        result = StubgenPyx(
            StubgenPyxConfig(resolve_ctypedef_aliases=True)
        ).convert_str(pyx_file.read_text(), pyx_path=pyx_file)
        assert "def f(y: float) -> float: ..." in result
        # The alias is never actually importable from the compiled module
        # (a plain `ctypedef` has no runtime binding at all), and every
        # usage of it was just substituted away above -- so its
        # declaration is now dead code, correctly pruned rather than
        # left behind as a misleading, unused, non-importable
        # module-level name.
        assert "MyFloat" not in result

    def test_keeps_alias_declaration_when_still_referenced_elsewhere(self, temp_dir):
        """The alias declaration survives pruning when something other
        than the substituted usages still needs it -- here, a sibling
        alias chained on top of it."""
        pyx_file = temp_dir / "test.pyx"
        pyx_file.write_text("""
ctypedef double MyFloat
ctypedef MyFloat MyFloat2

def f(y: MyFloat) -> MyFloat:
    pass

def g(z: MyFloat2) -> MyFloat2:
    pass
""")
        result = StubgenPyx(
            StubgenPyxConfig(resolve_ctypedef_aliases=True)
        ).convert_str(pyx_file.read_text(), pyx_path=pyx_file)
        assert "def f(y: float) -> float: ..." in result
        assert "def g(z: float) -> float: ..." in result
        # Both f and g get fully substituted, so neither alias is used
        # in any signature anymore -- both should be pruned.
        assert "MyFloat" not in result

    def test_cascading_prune_of_chained_aliases(self, temp_dir):
        """Pruning a now-dead alias can make another alias -- only ever
        referenced via that first alias's own (now-removed) declaration
        -- dead too. Confirmed this cascades fully rather than stopping
        after one level."""
        pyx_file = temp_dir / "test.pyx"
        pyx_file.write_text("""
ctypedef double MyFloat
ctypedef MyFloat MyFloat2

def f(y: MyFloat2) -> MyFloat2:
    pass
""")
        result = StubgenPyx(
            StubgenPyxConfig(resolve_ctypedef_aliases=True)
        ).convert_str(pyx_file.read_text(), pyx_path=pyx_file)
        assert "def f(y: float) -> float: ..." in result
        assert "MyFloat" not in result

    def test_default_off_never_prunes_alias_declaration(self, temp_dir):
        """The pruning behavior is itself part of resolve_ctypedef_aliases
        -- off by default, the alias declaration is emitted exactly as
        it always was, whether or not anything happens to reference it."""
        pyx_file = temp_dir / "test.pyx"
        pyx_file.write_text("""
ctypedef double MyFloat

def g(y: int) -> int:
    return y
""")
        result = StubgenPyx().convert_str(pyx_file.read_text(), pyx_path=pyx_file)
        assert "MyFloat: TypeAlias = float" in result

    def test_resolves_alias_nested_in_python_style_annotation(self, temp_dir):
        """An alias referenced inside an explicit annotation
        (`Iterable[MyFloat]`) is substituted too, not just a bare
        C-style argument."""
        pyx_file = temp_dir / "test.pyx"
        pyx_file.write_text("""
from typing import Iterable

ctypedef double MyFloat

def f(values: Iterable[MyFloat]) -> None:
    pass
""")
        result = StubgenPyx(
            StubgenPyxConfig(resolve_ctypedef_aliases=True)
        ).convert_str(pyx_file.read_text(), pyx_path=pyx_file)
        assert "def f(values: Iterable[float]) -> None: ..." in result

    def test_resolves_chained_aliases(self, temp_dir):
        """`ctypedef MyFloat MyFloat2` resolves all the way through to
        the concrete underlying type, not just one level."""
        pyx_file = temp_dir / "test.pyx"
        pyx_file.write_text("""
ctypedef double MyFloat
ctypedef MyFloat MyFloat2

def f(y: MyFloat2) -> MyFloat2:
    pass
""")
        result = StubgenPyx(
            StubgenPyxConfig(resolve_ctypedef_aliases=True)
        ).convert_str(pyx_file.read_text(), pyx_path=pyx_file)
        assert "def f(y: float) -> float: ..." in result

    def test_resolves_cdef_class_attribute(self, temp_dir):
        pyx_file = temp_dir / "test.pyx"
        pyx_file.write_text("""
ctypedef double MyFloat

cdef class Ops:
    cdef public MyFloat value
""")
        result = StubgenPyx(
            StubgenPyxConfig(resolve_ctypedef_aliases=True)
        ).convert_str(pyx_file.read_text(), pyx_path=pyx_file)
        assert "value: float" in result

    def test_resolves_dataclass_style_bare_annotation_attribute(self, temp_dir):
        pyx_file = temp_dir / "test.pyx"
        pyx_file.write_text("""
from dataclasses import dataclass

ctypedef double MyFloat

@dataclass
class Point:
    x: MyFloat
""")
        result = StubgenPyx(
            StubgenPyxConfig(resolve_ctypedef_aliases=True)
        ).convert_str(pyx_file.read_text(), pyx_path=pyx_file)
        assert "x: float" in result

    def test_does_not_substitute_substring_match(self, temp_dir):
        """A real type whose name merely *contains* an alias name as a
        substring (`MyFloatValue` vs the alias `MyFloat`) must not be
        mistaken for a reference to that alias."""
        pyx_file = temp_dir / "test.pyx"
        pyx_file.write_text("""
ctypedef double MyFloat

cdef class MyFloatValue:
    cdef public int x

def f(a: MyFloatValue, b: MyFloat) -> MyFloatValue:
    pass
""")
        result = StubgenPyx(
            StubgenPyxConfig(resolve_ctypedef_aliases=True)
        ).convert_str(pyx_file.read_text(), pyx_path=pyx_file)
        assert "def f(a: MyFloatValue, b: float) -> MyFloatValue: ..." in result

    def test_resolves_alias_declared_only_in_companion_pxd(self, temp_dir):
        """A ctypedef declared only in the companion .pxd is still
        resolvable from the .pyx (same shared module scope as the
        fused-type .pxd merge)."""
        pyx_file = temp_dir / "test.pyx"
        pyx_file.write_text("""
def process(x: MyFloat) -> MyFloat:
    pass
""")
        pxd_file = temp_dir / "test.pxd"
        pxd_file.write_text("""
ctypedef double MyFloat

cpdef object process(MyFloat x)
""")
        result = StubgenPyx(
            StubgenPyxConfig(resolve_ctypedef_aliases=True)
        ).convert_str(
            pyx_file.read_text(), pxd_str=pxd_file.read_text(), pyx_path=pyx_file
        )
        assert "def process(x: float) -> float: ..." in result


class TestQualifiedModuleTypeReferences:
    """A type referenced through its cimported module name (``cimport mod``
    then ``mod.Foo``, as opposed to ``from mod cimport Foo``) must keep
    that qualifier consistently, including for a `cdef public`/`readonly`
    class attribute: the attribute goes through a synthesized
    `PropertyNode` once real declaration analysis runs, and resolving
    *that* only ever gives the bare class name (`Foo`), never the module
    it was accessed through.
    """

    def test_qualified_type_as_argument_and_return(self, temp_dir):
        pkg = temp_dir / "pkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        (pkg / "mod.pxd").write_text("cdef class Foo:\n    pass\n")
        (pkg / "mod.pyx").write_text("cdef class Foo:\n    pass\n")
        (pkg / "consumer.pyx").write_text(
            "cimport pkg.mod as mod\n\ncpdef mod.Foo foo(mod.Foo x):\n    pass\n"
        )
        stubgen = StubgenPyx(StubgenPyxConfig(continue_on_error=True))
        results = stubgen.convert_glob(str(temp_dir / "**" / "*.pyx"))
        assert all(r.success for r in results)
        result = (pkg / "consumer.pyi").read_text()
        assert "import pkg.mod as mod" in result
        assert "def foo(x: mod.Foo) -> mod.Foo: ..." in result

    def test_qualified_type_as_class_attribute(self, temp_dir):
        pkg = temp_dir / "pkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        (pkg / "mod.pxd").write_text("cdef class Foo:\n    pass\n")
        (pkg / "mod.pyx").write_text("cdef class Foo:\n    pass\n")
        (pkg / "consumer.pyx").write_text(
            "from . cimport mod\n\ncdef class Holder:\n    cdef public mod.Foo item\n"
        )
        stubgen = StubgenPyx(StubgenPyxConfig(continue_on_error=True))
        results = stubgen.convert_glob(str(temp_dir / "**" / "*.pyx"))
        assert all(r.success for r in results)
        result = (pkg / "consumer.pyi").read_text()
        assert "from . import mod" in result
        assert "item: mod.Foo" in result

    def test_char_ptr_special_case_still_correct(self, temp_dir):
        """A `char* -> bytes` class attribute must keep translating
        correctly (both go through the same code path,
        `get_cdef_variables`'s `PropertyNode` branch, as the qualified
        -name case above)."""
        pyx_file = temp_dir / "test.pyx"
        pyx_file.write_text("cdef class Foo:\n    cdef public char* bar\n")
        result = StubgenPyx().convert_str(pyx_file.read_text(), pyx_path=pyx_file)
        assert "bar: bytes" in result

    def test_genuinely_unresolvable_qualified_attribute_still_incomplete(
        self, temp_dir
    ):
        """A qualified reference to something that genuinely doesn't
        exist (no real cimport backing it) still degrades to
        `_typeshed.Incomplete` via `postprocessing/trim_not_defined.py`:
        qualifier recovery only applies to a name that otherwise
        resolves, not a fabricated one for something that never did."""
        pyx_file = temp_dir / "test.pyx"
        pyx_file.write_text("cdef class Foo:\n    cdef public other.val[3][3] baz\n")
        result = StubgenPyx(StubgenPyxConfig(continue_on_error=True)).convert_str(
            pyx_file.read_text(), pyx_path=pyx_file
        )
        assert "baz: Incomplete" in result
        assert "other" not in result


class TestCtypedefAliasBatchPruning:
    """`resolve_ctypedef_aliases`'s pruning must not remove an alias
    declaration that another file in the same `convert_multiple_files`/
    `convert_glob` batch still needs -- confirmed against a real
    multi-file package (NVIDIA/cudf's pylibcudf) that a single file's
    own usage isn't enough to know this safely on its own; see
    `Converter.convert_scope`'s and
    `StubgenPyxConfig.resolve_ctypedef_aliases`'s docstrings.

    Every test here gives the declaring module a real companion `.pxd`
    (not just a `.pyx`) -- confirmed directly, by compiling standalone,
    that cimporting from a `.pyx` with no `.pxd` isn't actually valid,
    supported Cython at all: real, independent compilation of the
    consuming file fails with `'x.pxd' not found`. stubgen-pyx's batch
    conversion shares one `Context` across every file specifically so
    cross-file `cimport`s resolve (see `parsing/context.py`'s
    docstring) -- but for a `.pxd`-less module, that sharing only helps
    if the declaring file happened to be parsed first, since there's no
    file Cython can find the declaration in independently of that.
    `glob`'s file order is not guaranteed and does differ across
    platforms/filesystems, so a test built on that unsupported pattern
    can pass or fail depending on filesystem iteration order alone
    (confirmed: forcing the consumer file to process first reproduces
    the identical failure on Linux too). A real `.pxd` sidesteps this
    entirely, matching how any real multi-file Cython package actually
    has to be structured for cross-file `cimport` to work at all.
    """

    def test_alias_kept_when_only_used_by_a_sibling_file(self, temp_dir):
        (temp_dir / "types_mod.pxd").write_text("ctypedef int size_type\n")
        types_file = temp_dir / "types_mod.pyx"
        types_file.write_text("")
        consumer_file = temp_dir / "consumer.pyx"
        consumer_file.write_text(
            "from types_mod cimport size_type\n"
            "\n"
            "def f(size_type x) -> size_type:\n"
            "    return x\n"
        )

        stubgen = StubgenPyx(
            StubgenPyxConfig(resolve_ctypedef_aliases=True, continue_on_error=True)
        )
        # Explicit order, checked both ways, rather than trusting
        # whatever order `convert_glob`'s own `glob` call happens to
        # return on this filesystem -- that's exactly the thing this
        # class's docstring says must not matter.
        for files in (
            [types_file, consumer_file],
            [consumer_file, types_file],
        ):
            results = stubgen.convert_multiple_files(list(files))
            assert all(r.success for r in results)

            types_pyi = (temp_dir / "types_mod.pyi").read_text()
            consumer_pyi = (temp_dir / "consumer.pyi").read_text()

            # The declaration survives in the file that owns it, because
            # consumer.pyx's own cimport of it still needs it to exist there
            # -- even though consumer.pyx's own signature gets substituted
            # to the concrete type directly, same as usual.
            assert "size_type: TypeAlias = int" in types_pyi
            assert "def f(x: int) -> int: ..." in consumer_pyi

    def test_alias_still_pruned_when_truly_unused_across_the_batch(self, temp_dir):
        (temp_dir / "types_mod2.pxd").write_text(
            "ctypedef int size_type\nctypedef int unused_type\n"
        )
        types_file = temp_dir / "types_mod2.pyx"
        types_file.write_text("")
        consumer_file = temp_dir / "consumer2.pyx"
        consumer_file.write_text(
            "from types_mod2 cimport size_type\n"
            "\n"
            "def f(size_type x) -> size_type:\n"
            "    return x\n"
        )

        stubgen = StubgenPyx(
            StubgenPyxConfig(resolve_ctypedef_aliases=True, continue_on_error=True)
        )
        results = stubgen.convert_glob(str(temp_dir / "*2.pyx"))
        assert all(r.success for r in results)

        types_pyi = (temp_dir / "types_mod2.pyi").read_text()
        # size_type survives (still cimported by consumer2.pyx); the
        # genuinely never-referenced-anywhere unused_type is pruned.
        assert "size_type: TypeAlias = int" in types_pyi
        assert "unused_type" not in types_pyi

    def test_single_file_conversion_still_prunes_immediately(self, temp_dir):
        """Outside a batch (no other files to cross-check against), the
        existing local-only pruning is unaffected -- confirmed the
        refactor to support batch-aware pruning didn't change this."""
        pyx_file = temp_dir / "solo.pyx"
        pyx_file.write_text("""
ctypedef double MyFloat

def f(a, b):
    # type: (int, str) -> bool
    pass
""")
        stubgen = StubgenPyx(StubgenPyxConfig(resolve_ctypedef_aliases=True))
        result = stubgen.convert_str(pyx_file.read_text(), pyx_path=pyx_file)
        assert "MyFloat" not in result


class TestPrivateCdefGlobalExclusion:
    """A plain (non-`public`, non-`readonly`) module-level `cdef` variable
    has no Python-level binding at all -- `dir(module)` never shows it --
    regardless of whether it has an initializer. An uninitialized one
    (`cdef int X`, no `= ...`) is removed from the AST entirely by real
    declaration analysis and was already correctly excluded; an
    initialized one (`cdef int X = 5`) instead survives as an ordinary
    -looking `SingleAssignmentNode`, indistinguishable in shape from a
    real Python-level global, and must be excluded the same way.
    """

    def test_initialized_private_cdef_global_excluded(self, temp_dir):
        pyx_file = temp_dir / "test.pyx"
        pyx_file.write_text("PLAIN_PY_VAR = 5\n\ncdef int PRIVATE_INIT_VAR = 5\n")
        result = StubgenPyx().convert_str(pyx_file.read_text(), pyx_path=pyx_file)
        assert "PLAIN_PY_VAR = 5" in result
        assert "PRIVATE_INIT_VAR" not in result

    def test_initialized_public_and_uninitialized_private_cdef_globals_unaffected(
        self, temp_dir
    ):
        """`cdef public` still appears (genuinely Python-visible); a plain,
        uninitialized `cdef` global still correctly produces nothing --
        the fix must not change either of these."""
        pyx_file = temp_dir / "test.pyx"
        pyx_file.write_text(
            "cdef public int PUBLIC_INIT_VAR = 7\n\ncdef int PRIVATE_UNINIT_VAR\n"
        )
        result = StubgenPyx().convert_str(pyx_file.read_text(), pyx_path=pyx_file)
        assert "PUBLIC_INIT_VAR = 7" in result
        assert "PRIVATE_UNINIT_VAR" not in result

    def test_private_function_pointer_global_excluded(self, temp_dir):
        """The motivating case: a private module-level C function-pointer
        variable is not just imprecisely typed but genuinely uncallable
        from Python -- it must not appear in the stub at all, rather than
        as a misleading `_typeshed.Incomplete`-typed name."""
        pyx_file = temp_dir / "test.pyx"
        pyx_file.write_text("""
ctypedef int (*binop_t)(int, int)

cdef int add(int a, int b):
    return a + b

cdef binop_t GLOBAL_FP = add
""")
        result = StubgenPyx().convert_str(pyx_file.read_text(), pyx_path=pyx_file)
        assert "GLOBAL_FP" not in result


class TestFunctionPointerCtypedefExclusion:
    """A `ctypedef` for a C function pointer -- e.g. `ctypedef int
    (*binop_t)(int, int)` -- has no legitimate Python rendering at all.
    Unlike an ordinary `ctypedef` (which at least names a real, usable
    Python type even though the alias name itself has no runtime
    binding), a function pointer cannot cross the Python boundary in any
    form, so it must not appear in the stub even as a `TypeAlias`.
    """

    def test_unused_function_pointer_ctypedef_excluded(self, temp_dir):
        pyx_file = temp_dir / "test.pyx"
        pyx_file.write_text("ctypedef int (*binop_t)(int, int)\n")
        result = StubgenPyx().convert_str(pyx_file.read_text(), pyx_path=pyx_file)
        assert "binop_t" not in result

    def test_function_pointer_ctypedef_used_internally_still_excluded(self, temp_dir):
        pyx_file = temp_dir / "test.pyx"
        pyx_file.write_text("""
ctypedef int (*binop_t)(int, int)

cdef int add(int a, int b):
    return a + b

cdef int call_it(binop_t fn, int a, int b):
    return fn(a, b)
""")
        result = StubgenPyx().convert_str(pyx_file.read_text(), pyx_path=pyx_file)
        assert "binop_t" not in result

    def test_typedef_of_function_pointer_typedef_also_excluded(self, temp_dir):
        """A `ctypedef` of a `ctypedef` function pointer is transitively
        just as unusable, and must be excluded too."""
        pyx_file = temp_dir / "test.pyx"
        pyx_file.write_text(
            "ctypedef int (*binop_t)(int, int)\nctypedef binop_t binop_alias\n"
        )
        result = StubgenPyx().convert_str(pyx_file.read_text(), pyx_path=pyx_file)
        assert "binop_t" not in result
        assert "binop_alias" not in result

    def test_ordinary_and_struct_ctypedefs_unaffected(self, temp_dir):
        """A plain scalar ctypedef and a struct ctypedef alias are both
        genuinely usable from Python (Cython coerces scalars and structs
        across the boundary) and must keep appearing as before."""
        pyx_file = temp_dir / "test.pyx"
        pyx_file.write_text("""
cdef struct Point:
    int x
    int y

ctypedef double MyFloat
ctypedef Point PointAlias

def f(MyFloat x):
    pass
""")
        result = StubgenPyx().convert_str(pyx_file.read_text(), pyx_path=pyx_file)
        assert "MyFloat: TypeAlias = float" in result
        assert "PointAlias: TypeAlias = Point" in result
        assert "def f(x: MyFloat)" in result


class TestCppScopedEnumClass:
    """A C++11 scoped ``enum class`` (`PyrexTypes.CppScopedEnumType`) has
    ``is_enum`` set to ``False`` -- ``is_cpp_enum`` instead -- unlike a
    plain `cpdef enum`. Since it has no surviving AST node once real
    declaration analysis runs (same as any other enum), it's only
    recoverable via the `entry.type`-based fallback in
    `Converter._convert_declared_entries`, which checked `is_enum` alone
    and so silently dropped every `enum class` entirely.
    """

    def test_cpdef_enum_class_in_extern_block(self, temp_dir):
        """The originally reported case: a `cpdef enum class` inside a
        `cdef extern from ... namespace ...:` block, common in real
        `.pxd` files wrapping a C++ library."""
        pyx_file = temp_dir / "test.pyx"
        pyx_file.write_text("""
from libc.stdint cimport uint32_t

cdef extern from "foo.hpp" namespace "ns" nogil:
    cpdef enum class string_character_types(uint32_t):
        DECIMAL
        NUMERIC
""")
        result = StubgenPyx().convert_str(pyx_file.read_text(), pyx_path=pyx_file)
        assert "class string_character_types(IntEnum):" in result
        assert "DECIMAL = ..." in result
        assert "NUMERIC = ..." in result

    def test_cpdef_enum_class_without_extern_block(self, temp_dir):
        """Not specific to being inside an extern block -- a bare
        top-level `cpdef enum class` hits the same code path."""
        pyx_file = temp_dir / "test.pyx"
        pyx_file.write_text("cpdef enum class Color(int):\n    RED\n    GREEN\n")
        result = StubgenPyx().convert_str(pyx_file.read_text(), pyx_path=pyx_file)
        assert "class Color(IntEnum):" in result
        assert "RED = ..." in result
        assert "GREEN = ..." in result

    def test_enum_class_used_as_argument_and_return_type(self, temp_dir):
        """A function referencing the scoped enum as a parameter or
        return type must resolve to the enum's own name, not
        `_typeshed.Incomplete` (`render_pyrex_type` has the same
        `is_enum`-only gap as the declaration-collection path)."""
        pyx_file = temp_dir / "test.pyx"
        pyx_file.write_text("""
cpdef enum class Color(int):
    RED
    GREEN

cpdef Color get_color():
    return Color.RED

cpdef void use_color(Color c):
    pass
""")
        result = StubgenPyx().convert_str(pyx_file.read_text(), pyx_path=pyx_file)
        assert "def get_color() -> Color" in result
        assert "def use_color(c: Color)" in result
        assert "Incomplete" not in result

    def test_plain_unscoped_enum_still_works(self, temp_dir):
        """Baseline: an ordinary, unscoped `cpdef enum` (`is_enum=True`
        already) must keep working exactly as before."""
        pyx_file = temp_dir / "test.pyx"
        pyx_file.write_text("""
cdef extern from "foo.hpp" namespace "ns" nogil:
    cpdef enum Color:
        RED
        GREEN
""")
        result = StubgenPyx().convert_str(pyx_file.read_text(), pyx_path=pyx_file)
        assert "class Color(IntEnum):" in result
        assert "RED = ..." in result


class TestExternCpdefFunctionDeclaration:
    """A `cpdef` function declared with no body of its own -- the normal
    shape for exposing an external C/C++ function to Python directly,
    written inside a `cdef extern from ...:` block -- has no surviving
    `CFuncDefNode` (there's no body to keep a node for), so it's only
    recoverable via `entry.type` in `Converter._convert_declared_entries`,
    which had no branch at all for a function-shaped entry.
    """

    def test_cpdef_function_in_extern_block(self, temp_dir):
        pyx_file = temp_dir / "test.pyx"
        pyx_file.write_text("""
cdef extern from "foo.hpp" namespace "ns" nogil:
    cpdef int compute(int x, double y)
""")
        result = StubgenPyx().convert_str(pyx_file.read_text(), pyx_path=pyx_file)
        assert "def compute(x: int, y: float) -> int: ..." in result

    def test_plain_cdef_function_in_extern_block_excluded(self, temp_dir):
        """A plain (non-`cpdef`) extern declaration has no Python wrapper
        and must stay excluded, same as any other C-only name."""
        pyx_file = temp_dir / "test.pyx"
        pyx_file.write_text("""
cdef extern from "foo.hpp" nogil:
    cdef int hidden_func(int x)
""")
        result = StubgenPyx().convert_str(pyx_file.read_text(), pyx_path=pyx_file)
        assert "hidden_func" not in result

    def test_mixed_cpdef_and_cdef_in_extern_block(self, temp_dir):
        pyx_file = temp_dir / "test.pyx"
        pyx_file.write_text("""
cdef extern from "foo.hpp" nogil:
    cpdef int visible_func(int x)
    cdef int hidden_func(int x)
""")
        result = StubgenPyx().convert_str(pyx_file.read_text(), pyx_path=pyx_file)
        assert "def visible_func(x: int) -> int: ..." in result
        assert "hidden_func" not in result

    def test_zero_arg_void_return(self, temp_dir):
        pyx_file = temp_dir / "test.pyx"
        pyx_file.write_text(
            'cdef extern from "foo.hpp" nogil:\n    cpdef void do_thing()\n'
        )
        result = StubgenPyx().convert_str(pyx_file.read_text(), pyx_path=pyx_file)
        assert "def do_thing() -> None: ..." in result

    def test_does_not_interfere_with_cdef_class_methods(self, temp_dir):
        """A `cdef class`'s own `cpdef` methods (which do have a
        surviving node) must keep converting normally, not get
        double-emitted or otherwise disturbed by the new entries-based
        function path."""
        pyx_file = temp_dir / "test.pyx"
        pyx_file.write_text("""
cdef extern from "foo.hpp" nogil:
    cpdef int standalone_func(int x)

cdef class Wrapper:
    cpdef int method(self, int x):
        return x
""")
        result = StubgenPyx().convert_str(pyx_file.read_text(), pyx_path=pyx_file)
        assert result.count("def method(self, x: int) -> int: ...") == 1
        assert "def standalone_func(x: int) -> int: ..." in result


class TestReplaceDefaultsWithEllipsis:
    """`StubgenPyxConfig.replace_defaults_with_ellipsis` renders every
    argument default as `...` rather than its real value, matching the
    convention most `.pyi` stubs use.
    """

    def test_defaults_replaced_with_ellipsis(self, temp_dir):
        pyx_file = temp_dir / "test.pyx"
        pyx_file.write_text('def f(x: int = 5, y: str = "hello"):\n    pass\n')
        result = StubgenPyx(
            StubgenPyxConfig(replace_defaults_with_ellipsis=True)
        ).convert_str(pyx_file.read_text(), pyx_path=pyx_file)
        assert "x: int=..." in result
        assert "y: str=..." in result
        # Specific default-value patterns, not a blanket substring check
        # on the whole file -- `result` also contains the generated
        # header's source path (`... from {temp_dir}/test.pyx`), and
        # pytest's `temp_dir` fixture is a randomly-named directory that
        # can coincidentally contain the digit "5" (or, in principle,
        # "hello"), making a bare `"5" not in result` check flaky.
        assert "=5" not in result
        assert "'hello'" not in result

    def test_off_by_default(self, temp_dir):
        pyx_file = temp_dir / "test.pyx"
        pyx_file.write_text("def f(x: int = 5):\n    pass\n")
        result = StubgenPyx().convert_str(pyx_file.read_text(), pyx_path=pyx_file)
        assert "x: int=5" in result

    def test_none_default_still_widens_annotation(self, temp_dir):
        """A `None` default still adds `| None` to the annotation even
        though the rendered default becomes `...` -- the widening
        depends on the real default value, resolved before this
        rendering-time substitution."""
        pyx_file = temp_dir / "test.pyx"
        pyx_file.write_text("def f(x: int = None):\n    pass\n")
        result = StubgenPyx(
            StubgenPyxConfig(replace_defaults_with_ellipsis=True)
        ).convert_str(pyx_file.read_text(), pyx_path=pyx_file)
        assert "x: int | None=..." in result


class TestResolveCtypedefAliasesAcrossPxdMerge:
    """A `ctypedef` alias declared only in a companion `.pxd` -- not the
    `.pyx` itself -- must still get pruned by
    `resolve_ctypedef_aliases` once nothing needs it, the same as one
    declared directly in the `.pyx`.

    Regression test: `Converter.convert_scope` captures a reference to
    the *list* an alias's assignment lives in at conversion time, for
    `convert_multiple_files`'s batch cross-check to later remove it
    from. But a `.pxd`-only alias is converted via its own, separate
    `converter.convert_module` call (see `stubgen.py`'s
    `_compile_file_with_converter`), and `_merge_pxd_into_module`
    builds a brand new list when merging that module's assignments into
    the `.pyx`'s own -- so removing from the originally-captured list
    later did nothing to the list that actually gets rendered, and the
    alias (and the import it needed) silently survived regardless of
    whether anything still used it.
    """

    def test_pxd_only_alias_pruned_when_unused(self, temp_dir):
        (temp_dir / "aggregation.pxd").write_text("""
cdef extern from "foo.hpp" nogil:
    cdef cppclass groupby_aggregation:
        pass
""")
        (temp_dir / "mod.pxd").write_text("""
from aggregation cimport groupby_aggregation

ctypedef groupby_aggregation * gba_ptr
""")
        (temp_dir / "mod.pyx").write_text("")

        stubgen = StubgenPyx(StubgenPyxConfig(resolve_ctypedef_aliases=True))
        results = stubgen.convert_glob(str(temp_dir / "*.pyx"))
        assert all(r.success for r in results)

        result = (temp_dir / "mod.pyi").read_text()
        assert "gba_ptr" not in result
        assert "groupby_aggregation" not in result

    def test_pxd_only_alias_kept_and_substituted_when_used(self, temp_dir):
        (temp_dir / "aggregation.pxd").write_text("""
cdef extern from "foo.hpp" nogil:
    cdef cppclass groupby_aggregation:
        pass
""")
        (temp_dir / "mod.pxd").write_text("""
from aggregation cimport groupby_aggregation

ctypedef groupby_aggregation * gba_ptr
""")
        (temp_dir / "mod.pyx").write_text("""
cpdef gba_ptr get_thing():
    pass
""")

        stubgen = StubgenPyx(StubgenPyxConfig(resolve_ctypedef_aliases=True))
        results = stubgen.convert_glob(str(temp_dir / "*.pyx"))
        assert all(r.success for r in results)

        result = (temp_dir / "mod.pyi").read_text()
        assert "def get_thing() -> groupby_aggregation" in result
        assert "gba_ptr" not in result
        assert "from aggregation import groupby_aggregation" in result

    def test_pxd_only_alias_kept_when_used_by_another_file_in_batch(self, temp_dir):
        (temp_dir / "aggregation.pxd").write_text("""
cdef extern from "foo.hpp" nogil:
    cdef cppclass groupby_aggregation:
        pass
""")
        (temp_dir / "shared.pxd").write_text("""
from aggregation cimport groupby_aggregation

ctypedef groupby_aggregation * gba_ptr
""")
        (temp_dir / "other.pyx").write_text("""
from shared cimport gba_ptr

cpdef gba_ptr use_it():
    pass
""")

        stubgen = StubgenPyx(StubgenPyxConfig(resolve_ctypedef_aliases=True))
        results = stubgen.convert_glob(str(temp_dir / "*.pyx"))
        assert all(r.success for r in results)

        result = (temp_dir / "shared.pyi").read_text()
        assert "gba_ptr: TypeAlias = groupby_aggregation" in result

    def test_pyx_only_alias_still_pruned_no_pxd(self, temp_dir):
        """Baseline, unaffected by this fix: an alias declared directly
        in the `.pyx`, with no companion `.pxd` merge involved at all,
        must keep being pruned exactly as before."""
        pyx_file = temp_dir / "test.pyx"
        pyx_file.write_text("ctypedef double MyFloat\ndef f(int x):\n    pass\n")
        result = StubgenPyx(
            StubgenPyxConfig(resolve_ctypedef_aliases=True)
        ).convert_str(pyx_file.read_text(), pyx_path=pyx_file)
        assert "MyFloat" not in result


class TestPropertyReturnTypeAnnotation:
    """A `@property`-decorated `def` method becomes the exact same
    `PropertyNode` AST shape as a `cdef public`/`cdef readonly` C
    attribute once real declaration analysis runs -- but its `Entry`
    type is always the generic `object` (Cython has no reason to track
    anything more specific for a plain Python property). The real
    declared type, when the source wrote one, must be read from the
    property's own `__get__` method's return annotation instead.
    """

    def test_property_with_return_annotation_keeps_its_type(self, temp_dir):
        pyx_file = temp_dir / "test.pyx"
        pyx_file.write_text("""
cdef class Foo:
    def __init__(self):
        self.nbytes = 5

    @property
    def size(self) -> int:
        return self.nbytes
""")
        result = StubgenPyx().convert_str(pyx_file.read_text(), pyx_path=pyx_file)
        assert "size: int" in result
        assert "Incomplete" not in result

    def test_untyped_property_still_falls_back_to_object(self, temp_dir):
        """No annotation in the source -- correctly stays `object`,
        not a regression, just the honest answer when nothing more
        specific was written."""
        pyx_file = temp_dir / "test.pyx"
        pyx_file.write_text("""
cdef class Foo:
    @property
    def size(self):
        return 5
""")
        result = StubgenPyx().convert_str(pyx_file.read_text(), pyx_path=pyx_file)
        assert "size: object" in result

    def test_property_with_generic_return_type(self, temp_dir):
        pyx_file = temp_dir / "test.pyx"
        pyx_file.write_text("""
from typing import Any, Mapping

cdef class Foo:
    @property
    def info(self) -> Mapping[str, Any]:
        return {}
""")
        result = StubgenPyx().convert_str(pyx_file.read_text(), pyx_path=pyx_file)
        assert "info: Mapping[str, Any]" in result

    def test_property_with_setter_uses_getter_return_type(self, temp_dir):
        pyx_file = temp_dir / "test.pyx"
        pyx_file.write_text("""
cdef class Foo:
    @property
    def size(self) -> int:
        return 5

    @size.setter
    def size(self, value: int):
        pass
""")
        result = StubgenPyx().convert_str(pyx_file.read_text(), pyx_path=pyx_file)
        assert "size: int" in result

    def test_cdef_public_readonly_attributes_unaffected(self, temp_dir):
        """Baseline: the original use case this code path was built for
        -- a real C-typed `cdef public`/`cdef readonly` attribute --
        must keep resolving from its `Entry` type exactly as before."""
        pyx_file = temp_dir / "test.pyx"
        pyx_file.write_text("""
cdef class Foo:
    cdef public int x
    cdef readonly double y
""")
        result = StubgenPyx().convert_str(pyx_file.read_text(), pyx_path=pyx_file)
        assert "x: int" in result
        assert "y: float" in result
