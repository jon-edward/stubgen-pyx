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


def test_classmethod_cls_type_uses_type_import_resolution(temp_dir):
    pyx_file = temp_dir / "test.pyx"
    pyx_file.write_text("""
cdef class Factory:
    @classmethod
    def make(cls):
        pass

    @classmethod
    def explicit(cls: object):
        pass
""")

    result = StubgenPyx().convert_str(pyx_file.read_text(), pyx_path=pyx_file)

    assert "import builtins" not in result
    assert "def make(cls: type): ..." in result
    assert "def explicit(cls: object): ..." in result


def test_classmethod_cls_type_stays_qualified_when_type_is_shadowed(temp_dir):
    pyx_file = temp_dir / "test.pyx"
    pyx_file.write_text("""
type = object

cdef class Factory:
    @classmethod
    def make(cls):
        pass
""")

    result = StubgenPyx().convert_str(pyx_file.read_text(), pyx_path=pyx_file)

    assert "import builtins" in result
    assert "def make(cls: builtins.type): ..." in result


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

    def test_resolves_alias_used_inside_a_cast(self, temp_dir):
        """A `ctypedef` alias named as a C-style cast's declared type
        (``<MyFloat> expr``, rendered as ``typing.cast(MyFloat, expr)``
        by `unparse.visit_TypecastNode`) is substituted the same as any
        other occurrence -- this is a plain assignment's *value*, not
        an annotation, so it uses a different code path
        (`_substitute_ctypedef_aliases_in_assignment`) than every other
        case in this class."""
        pyx_file = temp_dir / "test.pyx"
        pyx_file.write_text("""
ctypedef int MyInt

class Foo:
    x = <MyInt> 1
""")
        result = StubgenPyx(
            StubgenPyxConfig(resolve_ctypedef_aliases=True)
        ).convert_str(pyx_file.read_text(), pyx_path=pyx_file)
        assert "x = cast(int, 1)" in result
        assert "MyInt" not in result

    def test_cast_to_alias_does_not_corrupt_alias_declaration_kept_by_a_sibling_file(
        self, temp_dir
    ):
        """The alias's own defining statement must never itself be
        substituted -- only a genuinely separate usage (like the cast
        below) is. Substituting the declaration's own target name would
        rewrite `size_type: TypeAlias = int` into
        `int: TypeAlias = int`, which every usage/pruning check then
        reads as `size_type` never having existed at all, corrupting
        the declaration in the file that owns it -- a single file's own
        cast usage always gets fully substituted away, same as any
        other usage (see `test_resolves_alias_used_inside_a_cast`), so
        a sibling file that explicitly re-exports the alias
        (`TestCtypedefAliasBatchPruning`'s own shape) is what's needed
        to still have the declaration around to check for corruption.
        """
        (temp_dir / "types_mod.pxd").write_text("ctypedef int size_type\n")
        types_file = temp_dir / "types_mod.pyx"
        types_file.write_text("")
        consumer_file = temp_dir / "consumer.pyx"
        consumer_file.write_text(
            "from types_mod cimport size_type as size_type\n"
            "\n"
            "class Foo:\n"
            "    x = <size_type> 1\n"
        )

        stubgen = StubgenPyx(
            StubgenPyxConfig(resolve_ctypedef_aliases=True, continue_on_error=True)
        )
        results = stubgen.convert_multiple_files([types_file, consumer_file])
        assert all(r.success for r in results)

        types_pyi = (temp_dir / "types_mod.pyi").read_text()
        consumer_pyi = (temp_dir / "consumer.pyi").read_text()
        assert "size_type: TypeAlias = int" in types_pyi
        assert "int: TypeAlias = int" not in types_pyi
        assert "x = cast(int, 1)" in consumer_pyi


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

    def test_alias_pruned_when_sibling_substitutes_it_away(self, temp_dir):
        """A sibling's `cimport` of the alias doesn't keep it once that
        sibling's own output no longer references it: its signature is
        substituted to the concrete type and the now-unused import is
        trimmed."""
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

            assert "size_type" not in (temp_dir / "types_mod.pyi").read_text()
            consumer_pyi = (temp_dir / "consumer.pyi").read_text()
            assert "def f(x: int) -> int: ..." in consumer_pyi
            assert "size_type" not in consumer_pyi

    @pytest.mark.parametrize(
        "declaration",
        ["ctypedef int const_x\n", "cdef enum const_x:\n    A\n"],
        ids=["ctypedef", "cdef_enum"],
    )
    def test_same_alias_name_declared_in_two_files_is_still_pruned(
        self, temp_dir, declaration
    ):
        """A same-named declaration in another file is a separate symbol and
        must not keep this one alive."""
        for name in ("same1", "same2"):
            (temp_dir / f"{name}.pxd").write_text(declaration)
            (temp_dir / f"{name}.pyx").write_text("")

        stubgen = StubgenPyx(
            StubgenPyxConfig(resolve_ctypedef_aliases=True, continue_on_error=True)
        )
        results = stubgen.convert_multiple_files(
            [temp_dir / "same1.pyx", temp_dir / "same2.pyx"]
        )
        assert all(r.success for r in results)

        assert "const_x" not in (temp_dir / "same1.pyi").read_text()
        assert "const_x" not in (temp_dir / "same2.pyi").read_text()

    def test_alias_kept_when_a_kept_alias_in_the_same_file_is_defined_from_it(
        self, temp_dir
    ):
        """`B: TypeAlias = A` is kept (re-exported by a sibling), so `A`
        must stay defined rather than leave `B` pointing at nothing."""
        (temp_dir / "chain_a.pxd").write_text("ctypedef int A\nctypedef A B\n")
        (temp_dir / "chain_a.pyx").write_text("")
        (temp_dir / "chain_c.pyx").write_text(
            "from chain_a cimport B as B\n\ndef f(B x):\n    pass\n"
        )

        stubgen = StubgenPyx(
            StubgenPyxConfig(resolve_ctypedef_aliases=True, continue_on_error=True)
        )
        for order in (["chain_a.pyx", "chain_c.pyx"], ["chain_c.pyx", "chain_a.pyx"]):
            results = stubgen.convert_multiple_files([temp_dir / n for n in order])
            assert all(r.success for r in results)

            chain_a = (temp_dir / "chain_a.pyi").read_text()
            assert "A: TypeAlias = int" in chain_a
            assert "B: TypeAlias = A" in chain_a
            assert "Incomplete" not in chain_a

    def test_orphaned_alias_chain_across_files_is_pruned_in_either_order(
        self, temp_dir
    ):
        """`A` is only referenced by `B` in another file, and `B` itself is
        unused and dropped, so `A` is dropped too -- whichever file is
        processed first."""
        (temp_dir / "orphan_a.pxd").write_text("ctypedef int A\n")
        (temp_dir / "orphan_a.pyx").write_text("")
        (temp_dir / "orphan_b.pxd").write_text(
            "from orphan_a cimport A\nctypedef A B\n"
        )
        (temp_dir / "orphan_b.pyx").write_text("")

        stubgen = StubgenPyx(
            StubgenPyxConfig(resolve_ctypedef_aliases=True, continue_on_error=True)
        )
        for order in (
            ["orphan_a.pyx", "orphan_b.pyx"],
            ["orphan_b.pyx", "orphan_a.pyx"],
        ):
            results = stubgen.convert_multiple_files([temp_dir / n for n in order])
            assert all(r.success for r in results)

            assert "TypeAlias" not in (temp_dir / "orphan_a.pyi").read_text()
            assert "TypeAlias" not in (temp_dir / "orphan_b.pyi").read_text()

    def test_alias_kept_when_sibling_reexports_it(self, temp_dir):
        (temp_dir / "types_mod4.pxd").write_text("ctypedef int size_type\n")
        types_file = temp_dir / "types_mod4.pyx"
        types_file.write_text("")
        consumer_file = temp_dir / "consumer4.pyx"
        consumer_file.write_text(
            "from types_mod4 cimport size_type as size_type\n"
            "\n"
            "def f(size_type x) -> size_type:\n"
            "    return x\n"
        )

        stubgen = StubgenPyx(
            StubgenPyxConfig(resolve_ctypedef_aliases=True, continue_on_error=True)
        )
        results = stubgen.convert_multiple_files([types_file, consumer_file])
        assert all(r.success for r in results)

        assert "size_type: TypeAlias = int" in (temp_dir / "types_mod4.pyi").read_text()
        consumer_pyi = (temp_dir / "consumer4.pyi").read_text()
        assert "from types_mod4 import size_type as size_type" in consumer_pyi
        assert "def f(x: int) -> int: ..." in consumer_pyi

    def test_alias_kept_when_sibling_imports_are_not_trimmed(self, temp_dir):
        (temp_dir / "types_mod5.pxd").write_text("ctypedef int size_type\n")
        types_file = temp_dir / "types_mod5.pyx"
        types_file.write_text("")
        consumer_file = temp_dir / "consumer5.pyx"
        consumer_file.write_text(
            "from types_mod5 cimport size_type\n"
            "\n"
            "def f(size_type x) -> size_type:\n"
            "    return x\n"
        )

        stubgen = StubgenPyx(
            StubgenPyxConfig(
                resolve_ctypedef_aliases=True,
                continue_on_error=True,
                trim_imports=False,
            )
        )
        results = stubgen.convert_multiple_files([types_file, consumer_file])
        assert all(r.success for r in results)

        assert "size_type: TypeAlias = int" in (temp_dir / "types_mod5.pyi").read_text()
        consumer_pyi = (temp_dir / "consumer5.pyi").read_text()
        assert "from types_mod5 import size_type" in consumer_pyi

    def test_alias_kept_when_sibling_references_it_through_the_module(self, temp_dir):
        (temp_dir / "types_mod6.pxd").write_text("ctypedef int size_type\n")
        types_file = temp_dir / "types_mod6.pyx"
        types_file.write_text("")
        consumer_file = temp_dir / "consumer6.pyx"
        consumer_file.write_text(
            "cimport types_mod6\n\ndef f(types_mod6.size_type x):\n    return x\n"
        )

        stubgen = StubgenPyx(
            StubgenPyxConfig(resolve_ctypedef_aliases=True, continue_on_error=True)
        )
        results = stubgen.convert_multiple_files([types_file, consumer_file])
        assert all(r.success for r in results)

        assert "size_type: TypeAlias = int" in (temp_dir / "types_mod6.pyi").read_text()
        assert (
            "def f(x: types_mod6.size_type): ..."
            in (temp_dir / "consumer6.pyi").read_text()
        )

    def test_alias_still_pruned_when_truly_unused_across_the_batch(self, temp_dir):
        (temp_dir / "types_mod2.pxd").write_text(
            "ctypedef int size_type\nctypedef int unused_type\n"
        )
        types_file = temp_dir / "types_mod2.pyx"
        types_file.write_text("")
        consumer_file = temp_dir / "consumer2.pyx"
        consumer_file.write_text(
            "from types_mod2 cimport size_type as size_type\n"
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
        # size_type survives (re-exported by consumer2.pyx); the
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
from shared cimport gba_ptr as gba_ptr

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

    A get-only property (no setter) is a real, read-only Python
    property, not a plain attribute -- rendering it as `name: type`
    would misrepresent it as writable and drop its docstring, so it's
    kept as an actual `@property` method instead (see
    `ScopeVisitor.visit_PropertyNode`); the return type still needs to
    come from the same place. Only a get+set pair (indistinguishable
    from a plain attribute at the Python level -- and the shape `cdef
    public` synthesizes) still flattens to `name: type`.
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
        assert "def size(self) -> int" in result
        assert "Incomplete" not in result

    def test_untyped_property_still_falls_back_to_object(self, temp_dir):
        """No annotation in the source -- correctly stays untyped,
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
        assert "def size(self):" in result

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
        assert "def info(self) -> Mapping[str, Any]" in result

    def test_getter_only_property_keeps_docstring(self, temp_dir):
        """The regression this whole class now guards against: a
        get-only property's docstring must survive conversion, not be
        silently dropped along with the `@property` decorator."""
        pyx_file = temp_dir / "test.pyx"
        pyx_file.write_text('''
cdef class Foo:
    @property
    def size(self) -> int:
        """The size, in bytes."""
        return 5
''')
        result = StubgenPyx().convert_str(pyx_file.read_text(), pyx_path=pyx_file)
        assert "@property" in result
        assert "def size(self) -> int:" in result
        assert '"""The size, in bytes."""' in result
        assert "size: int" not in result

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
        must keep resolving from its `Entry` type exactly as before
        (`cdef readonly` also gets wrapped in `Final`, per
        `TestReadonlyCdefAttributes` -- orthogonal to type resolution,
        which is what this test guards)."""
        pyx_file = temp_dir / "test.pyx"
        pyx_file.write_text("""
cdef class Foo:
    cdef public int x
    cdef readonly double y
""")
        result = StubgenPyx().convert_str(pyx_file.read_text(), pyx_path=pyx_file)
        assert "x: int" in result
        assert "y: Final[float]" in result

    def test_get_set_property_flattened_to_bare_assignment_keeps_docstring(
        self, temp_dir
    ):
        """A get+set property is indistinguishable from a plain
        attribute at the Python level, so `visit_PropertyNode` flattens
        it to `name: type` (see this class's own docstring) -- but its
        getter's docstring must survive that flattening rather than
        being silently dropped."""
        pyx_file = temp_dir / "test.pyx"
        pyx_file.write_text('''
cdef class Widget:
    @property
    def height(self) -> int:
        """The widget's height in pixels."""
        return self._height

    @height.setter
    def height(self, value: int) -> None:
        self._height = value
''')
        result = StubgenPyx().convert_str(pyx_file.read_text(), pyx_path=pyx_file)
        assert (
            'class Widget:\n    height: int\n    "The widget\'s height in pixels."\n'
        ) in result
        assert "@property" not in result

    def test_get_only_property_docstring_still_only_on_the_method(self, temp_dir):
        """Companion to the get+set case above: a get-only property is
        *not* flattened (see `test_getter_only_property_keeps_docstring`),
        so its docstring must appear exactly once, attached to the
        `@property` method, never duplicated onto a bare assignment."""
        pyx_file = temp_dir / "test.pyx"
        pyx_file.write_text('''
cdef class Foo:
    @property
    def size(self) -> int:
        """The size, in bytes."""
        return 5
''')
        result = StubgenPyx().convert_str(pyx_file.read_text(), pyx_path=pyx_file)
        assert result.count("The size, in bytes.") == 1

    def test_cdef_public_attribute_has_no_docstring(self, temp_dir):
        """A raw `cdef public`/`cdef readonly` C attribute has no
        source-level docstring to preserve -- Cython has no syntax for
        one -- so it must render as a plain, undecorated assignment."""
        pyx_file = temp_dir / "test.pyx"
        pyx_file.write_text("""
cdef class Foo:
    cdef public int x
""")
        result = StubgenPyx().convert_str(pyx_file.read_text(), pyx_path=pyx_file)
        assert "class Foo:\n    x: int\n" in result


class TestIncludeStatementInCompanionPxd:
    """A companion `.pxd`'s own `include "other.pxd"` statement must not
    leak into the generated stub as literal, unparseable source text.

    Regression test for a bug found converting NVIDIA/cuda-python's
    `cuda_bindings` package: Cython's parser (`p_include_statement`)
    splices an `include`d file's own parsed statements directly into
    the including file's tree, but each spliced-in node keeps *its
    own* file's position (`node.pos[0]`), not the including file's.
    `Converter.convert_imports` (`conversion/converter.py`) calls
    `get_source(source_code, node)` for every collected import node --
    where `source_code` is unconditionally the text of the file
    currently being converted. For a node reached through an
    `include`, that slices the *including* file's text at the
    *included* file's line number -- which, by coincidence, is exactly
    where the including file's own `include "..."` statement sits (its
    own line number happens to match the included node's line number
    in the included file). The result: the literal text
    `include "other.pxd"` gets emitted as a "converted import"
    statement in the .pyi output, which isn't valid Python and crashes
    `ast.parse` in `postprocessing/pipeline.py`.

    Fixed in `conversion/source_extraction.py::get_source`: read from
    the node's own file (via `node.pos[0]`, a `FileSourceDescriptor`)
    whenever it differs from what `source_code` was assumed to cover,
    rather than always slicing the passed-in `source_code`.
    """

    def test_include_in_pxd_does_not_leak_into_output(self, temp_dir):
        # The included file's own `cimport` sits on the *same line
        # number* (4) as the including file's `include` statement --
        # not incidental: that's exactly the coincidence that let the
        # bug hide (see class docstring). `get_source` slicing the
        # *including* file's text at the *included* node's line lands
        # squarely on the `include "shared.pxd"` line itself.
        (temp_dir / "shared.pxd").write_text("""\n\n\ncimport cython\n""")
        (temp_dir / "mod.pxd").write_text(
            """\n\n\ninclude "shared.pxd"\n\ncdef class Foo:\n    cdef int x\n"""
        )
        (temp_dir / "mod.pyx").write_text("""
cdef class Foo:
    def __init__(self, x: int):
        self.x = x
""")

        stubgen = StubgenPyx()
        results = stubgen.convert_glob(str(temp_dir / "mod.pyx"))
        assert all(r.success for r in results), [
            r.error for r in results if not r.success
        ]

        result = (temp_dir / "mod.pyi").read_text()
        assert "include " not in result
        assert "class Foo" in result


class TestFusedCtupleDoesNotCrash:
    """A ``ctypedef fused`` type used as one of a C tuple's component
    types must not crash conversion, even when the tuple is never
    specialized to a concrete type.

    Regression test found converting scikit-learn's
    ``sklearn/linear_model/_cd_fast.pyx`` (``from cython cimport
    floating`` -- Cython's built-in float/double fused type -- used as
    a ctuple component). `PyrexTypes.PyrexType.is_fused` is true for
    *any* type with a fused subtype, not just the fused type
    definition itself -- `CTupleType` inherits it generically but has
    no `.types` attribute (`.components` instead), so
    `conversion/fused_types.py::convert_fused_types` crashed with
    `AttributeError: 'CTupleType' object has no attribute 'types'`
    when it found such a ctuple entry in scope and assumed, from
    `is_fused` alone, that it was a real `ctypedef fused` definition.

    Once that no longer crashes, a second bug surfaces: Cython leaves
    an internal, never-meant-to-be-seen placeholder cname (literally
    `"<dummy fused ctuple ...>"`, per `PyrexTypes.c_tuple_type`'s own
    comment) on the unspecialized ctuple's auto-generated return
    struct, which `_convert_declared_entry`'s scope walk was emitting
    verbatim as a class name -- invalid Python syntax.
    """

    def test_unspecialized_fused_ctuple_does_not_crash(self, temp_dir):
        pyx_file = temp_dir / "test.pyx"
        pyx_file.write_text("""
from cython cimport floating

cpdef (floating, floating) minmax(floating[:] arr):
    cdef floating lo = arr[0]
    cdef floating hi = arr[0]
    cdef Py_ssize_t i
    for i in range(arr.shape[0]):
        if arr[i] < lo:
            lo = arr[i]
        if arr[i] > hi:
            hi = arr[i]
    return lo, hi
""")
        result = StubgenPyx().convert_str(pyx_file.read_text(), pyx_path=pyx_file)
        assert "<dummy fused ctuple" not in result
        assert "def minmax" in result


class TestOrphanPxdParsedAsDeclarationFile:
    """A standalone ``.pxd`` file with no companion ``.pyx`` -- picked up
    as its own top-level conversion target -- must be parsed in
    declaration-file (``pxd=True``) mode, not ``.pyx`` mode.

    Regression test found converting pyzmq's
    ``zmq/backend/cython/_zmq.pxd`` (a pxd-only forward-declaration
    file, no matching ``.pyx``) and lxml's
    ``src/lxml/html/_difflib.pxd`` (same shape): both use a
    ``cpdef``/``cdef`` forward declaration's ``arg=*`` compile-time
    -only default marker, which is only valid Cython syntax inside a
    declaration file. `Stubgen._compile_file_with_converter` hardcoded
    ``pxd=False`` on its `parse_file` call regardless of the actual
    file's suffix -- correct for its usual caller (a real `.pyx`), but
    wrong for the orphan-`.pxd`-as-top-level-target case
    `Stubgen.resolve_glob` documents and creates: parsing pxd-only
    syntax in `.pyx` mode rejected `=*` as a real syntax error.
    """

    def test_orphan_pxd_with_star_default_converts(self, temp_dir):
        (temp_dir / "mod.pxd").write_text("""
cdef class Widget:
    cdef public int width
    cpdef object resize(self, int width=*, int height=*)
""")

        stubgen = StubgenPyx()
        results = stubgen.convert_glob(str(temp_dir / "*.pyx"))
        # Before the fix, this crashed with a CompileError ("Expected
        # an identifier or literal") parsing `arg=*` in .pyx mode --
        # the point of this test is that it no longer does.
        assert all(r.success for r in results), [
            r.error for r in results if not r.success
        ]

        result = (temp_dir / "mod.pyi").read_text()
        assert "class Widget" in result
        assert "width: int" in result


class TestReadonlyCdefAttributes:
    """``cdef readonly`` has no Python-level setter -- assigning to it
    from outside the extension type raises `AttributeError` at runtime.
    The rendered attribute is wrapped in `typing.Final` so a type
    checker rejects that assignment statically too, instead of only a
    real interpreter run catching it. ``cdef public`` (a real
    read/write attribute) and a real read/write ``@property`` are
    unaffected -- assignment to either is genuinely allowed.
    """

    def test_readonly_attribute_is_wrapped_in_final(self, temp_dir):
        pyx_file = temp_dir / "test.pyx"
        pyx_file.write_text("""
cdef class Foo:
    cdef readonly int y
""")
        result = StubgenPyx().convert_str(pyx_file.read_text(), pyx_path=pyx_file)
        assert "from typing import Final" in result
        assert "class Foo:\n    y: Final[int]\n" in result

    def test_public_attribute_is_not_wrapped(self, temp_dir):
        pyx_file = temp_dir / "test.pyx"
        pyx_file.write_text("""
cdef class Foo:
    cdef public int x
""")
        result = StubgenPyx().convert_str(pyx_file.read_text(), pyx_path=pyx_file)
        assert "class Foo:\n    x: int\n" in result
        assert "Final" not in result

    def test_real_read_write_property_is_not_wrapped(self, temp_dir):
        pyx_file = temp_dir / "test.pyx"
        pyx_file.write_text("""
cdef class Foo:
    @property
    def z(self) -> int:
        return 5

    @z.setter
    def z(self, value: int) -> None:
        pass
""")
        result = StubgenPyx().convert_str(pyx_file.read_text(), pyx_path=pyx_file)
        assert "class Foo:\n    z: int\n" in result
        assert "Final" not in result

    def test_public_and_readonly_together_are_each_wrapped_correctly(self, temp_dir):
        pyx_file = temp_dir / "test.pyx"
        pyx_file.write_text("""
cdef class Foo:
    cdef public int x
    cdef readonly int y
""")
        result = StubgenPyx().convert_str(pyx_file.read_text(), pyx_path=pyx_file)
        assert "class Foo:\n    x: int\n    y: Final[int]\n" in result

    def test_readonly_dataclass_field_init_param_is_not_wrapped(self, temp_dir):
        """The rendered attribute is `Final`, but the synthesized
        `__init__` parameter that assigns it must stay a plain type --
        `def __init__(self, y: Final[int])` is invalid syntax for a
        parameter annotation."""
        pyx_file = temp_dir / "test.pyx"
        pyx_file.write_text("""
from dataclasses import dataclass

@dataclass
cdef class Foo:
    cdef readonly int y
""")
        result = StubgenPyx().convert_str(pyx_file.read_text(), pyx_path=pyx_file)
        assert "y: Final[int]" in result
        assert "def __init__(self, y: int): ..." in result


class TestTypecastExpression:
    """A C-style cast (``<type> expr``) is not Python syntax at all --
    left as raw source text, the statement containing it fails to parse
    as Python (`declarations.convert_assignment`'s own source-text
    fallback hits the exact same invalid syntax), so the whole statement
    used to be silently dropped. Rendered as ``typing.cast(type, expr)``
    instead -- inert to a type checker, and valid Python.
    """

    def test_cast_in_enum_member_value_is_preserved(self, temp_dir):
        """The regression this whole class guards against: an
        `IntEnum` member assigned from a cast used to vanish from the
        stub entirely rather than just losing its cast."""
        pyx_file = temp_dir / "test.pyx"
        pyx_file.write_text("""
from enum import IntEnum

class Compression(IntEnum):
    INFER = <int> 1
    GZIP = 2
""")
        result = StubgenPyx().convert_str(pyx_file.read_text(), pyx_path=pyx_file)
        assert "from typing import cast" in result
        assert (
            "class Compression(IntEnum):\n    INFER = cast(int, 1)\n    GZIP = 2\n"
            in result
        )

    def test_cast_to_object_and_scalar_type_render_correctly(self, temp_dir):
        pyx_file = temp_dir / "test.pyx"
        pyx_file.write_text("""
class Foo:
    x = <object> "abc"
    y = <double> 3
""")
        result = StubgenPyx().convert_str(pyx_file.read_text(), pyx_path=pyx_file)
        assert "x = cast(object, 'abc')" in result
        # `double` is a Cython, not a Python, type name -- must go
        # through the same normalize_names translation as any other
        # scalar type, not survive verbatim inside the cast.
        assert "y = cast(float, 3)" in result

    def test_cast_in_default_argument_value_is_preserved(self, temp_dir):
        pyx_file = temp_dir / "test.pyx"
        pyx_file.write_text("""
def foo(x=<int> 5):
    pass
""")
        result = StubgenPyx().convert_str(pyx_file.read_text(), pyx_path=pyx_file)
        assert "def foo(x=cast(int, 5)): ...\n" in result


class TestCdefClassDataclass:
    """A ``cdef class`` decorated with ``@dataclass`` gets a real,
    correctly-ordered ``__init__``/``__repr__``/``__eq__`` synthesized by
    Cython's own compiler (`Cython.Compiler.Dataclass`), for *every*
    C-typed attribute regardless of ``cdef public``/``cdef readonly``/
    private visibility -- not anything stubgen-pyx generates itself, but
    something it must render accurately: every parameter typed to match
    its attribute, no debug noise from the two bookkeeping assignments
    (``__dataclass_params__``/``__dataclass_fields__``) the compiler adds
    alongside them, and the decorator itself marked ``init=False`` so a
    type checker's own dataclass support doesn't try to synthesize a
    second, independently-derived ``__init__`` from the class-body
    attribute annotations alone (which, for a private field with no
    class-body annotation at all, wouldn't even have the same parameters).
    """

    def test_public_readonly_and_private_fields_all_typed_in_init(self, temp_dir):
        pyx_file = temp_dir / "test.pyx"
        pyx_file.write_text("""
import dataclasses


@dataclasses.dataclass
cdef class Point:
    cdef public double x
    cdef readonly double y
    cdef double z

    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z
""")
        result = StubgenPyx().convert_str(pyx_file.read_text(), pyx_path=pyx_file)
        assert "def __init__(self, x: float, y: float, z: float): ..." in result
        assert "Incomplete" not in result

    def test_no_explicit_init_still_gets_fully_typed_synthesized_one(self, temp_dir):
        """No user-written ``__init__`` at all -- Cython synthesizes one
        from the ``cdef public``/``cdef readonly`` attributes alone, and
        it must come out fully typed, not bare names."""
        pyx_file = temp_dir / "test.pyx"
        pyx_file.write_text("""
import dataclasses


@dataclasses.dataclass
cdef class Point:
    cdef public double x
    cdef readonly double y
""")
        result = StubgenPyx().convert_str(pyx_file.read_text(), pyx_path=pyx_file)
        assert "def __init__(self, x: float, y: float): ..." in result

    def test_python_style_annotated_attribute_with_default_is_typed(self, temp_dir):
        """A field declared with a Python-style annotation naming a
        Cython pure-Python-mode type (``x: cython.double = 0.0``) must
        resolve to its real Python type (``float``) on the synthesized
        ``__init__`` parameter, not leak the unimportable ``cython.double``
        annotation through to `_typeshed.Incomplete` once the ``cython``
        import itself is trimmed as unused."""
        pyx_file = temp_dir / "test.pyx"
        pyx_file.write_text("""
import cython
import dataclasses


@dataclasses.dataclass
cdef class Point:
    x: cython.double = 0.0
    y: cython.double = 1.0
""")
        result = StubgenPyx().convert_str(pyx_file.read_text(), pyx_path=pyx_file)
        assert "def __init__(self, x: float=0.0, y: float=1.0): ..." in result
        assert "Incomplete" not in result
        assert "cython" not in result

    def test_no_dataclass_bookkeeping_debug_noise(self, temp_dir, caplog):
        """The compiler's own ``__dataclass_params__``/``__dataclass_fields__``
        class-body assignments must never reach `unparse_expr`/
        `convert_assignment` at all -- previously logged an "unknown node
        type" warning followed by a spurious "could not parse assignment
        source: '@dataclasses.dataclass'" one (the assignment's own
        position resolves back to the decorator line once unparsing
        fails), and produced no visible output either way."""
        import logging

        pyx_file = temp_dir / "test.pyx"
        pyx_file.write_text("""
import dataclasses


@dataclasses.dataclass
cdef class Point:
    cdef public double x
""")
        with caplog.at_level(logging.DEBUG):
            result = StubgenPyx().convert_str(pyx_file.read_text(), pyx_path=pyx_file)

        assert "__dataclass_params__" not in result
        assert "__dataclass_fields__" not in result
        assert not any(
            "Unknown node type encountered" in message for message in caplog.messages
        )
        assert not any(
            "Could not parse assignment source" in message
            for message in caplog.messages
        )

    def test_decorator_gets_init_false_added(self, temp_dir):
        """A bare ``@dataclasses.dataclass`` gets call parens created for
        it; other keyword arguments on an already-parenthesized call are
        preserved alongside the new ``init=False``."""
        pyx_file = temp_dir / "test.pyx"
        pyx_file.write_text("""
import dataclasses


@dataclasses.dataclass
cdef class Bare:
    cdef public double x


@dataclasses.dataclass(frozen=True)
cdef class Frozen:
    cdef public double x
""")
        result = StubgenPyx().convert_str(pyx_file.read_text(), pyx_path=pyx_file)
        assert "@dataclasses.dataclass(init=False)" in result
        assert "@dataclasses.dataclass(frozen=True, init=False)" in result

    def test_explicit_init_false_is_not_duplicated(self, temp_dir):
        pyx_file = temp_dir / "test.pyx"
        pyx_file.write_text("""
import dataclasses


@dataclasses.dataclass(init=False)
cdef class Point:
    cdef public double x

    def __init__(self, x):
        self.x = x
""")
        result = StubgenPyx().convert_str(pyx_file.read_text(), pyx_path=pyx_file)
        assert result.count("init=False") == 1

    def test_other_decorators_on_the_class_are_unaffected(self, temp_dir):
        pyx_file = temp_dir / "test.pyx"
        pyx_file.write_text("""
from dataclasses import dataclass


def other_decorator(cls):
    return cls


@other_decorator
@dataclass
class Multi:
    x: float

    def __init__(self, x: float):
        self.x = x
""")
        result = StubgenPyx().convert_str(pyx_file.read_text(), pyx_path=pyx_file)
        assert "@other_decorator\n@dataclass(init=False)\nclass Multi" in result

    def test_plain_python_dataclass_without_explicit_init_is_unaffected(self, temp_dir):
        """No ``cdef class`` and no explicit ``__init__`` -- a real
        `dataclasses`-module class relies entirely on the standard
        library's own runtime-generated ``__init__``, which a type
        checker already infers correctly from the class-body field
        annotations. Nothing here is Cython-synthesized, so the decorator
        must be left exactly as written and no ``__init__`` should appear
        in the stub at all."""
        pyx_file = temp_dir / "test.pyx"
        pyx_file.write_text("""
from dataclasses import dataclass


@dataclass
class Point:
    x: float
    y: float = 0.0
""")
        result = StubgenPyx().convert_str(pyx_file.read_text(), pyx_path=pyx_file)
        assert "@dataclass\nclass Point" in result
        assert "init=False" not in result
        assert "def __init__" not in result

    def test_cdef_public_readonly_without_dataclass_unaffected(self, temp_dir):
        """Baseline: a plain ``cdef public``/``cdef readonly`` attribute
        on a class with no ``@dataclass`` decorator at all must keep
        rendering exactly as before -- this whole feature only ever
        touches a class that both has the decorator and already has an
        explicit ``__init__`` in its converted output."""
        pyx_file = temp_dir / "test.pyx"
        pyx_file.write_text("""
cdef class Plain:
    cdef public double x
    cdef readonly double y
    cdef double z

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
""")
        result = StubgenPyx().convert_str(pyx_file.read_text(), pyx_path=pyx_file)
        assert (
            "class Plain:\n    x: float\n    y: Final[float]\n\n    def __init__"
            in result
        )
        assert "z:" not in result
        assert "init=False" not in result
        assert "def __init__(self, x, y, z): ..." in result

    def test_bare_annotated_extension_type_field_keeps_its_class_name(self, temp_dir):
        """A bare (no-default) Python-style-annotated attribute naming a
        plain Python class (``pt: Point``, ``Point`` not itself a ``cdef
        class``) still becomes a real, ``PropertyNode``-wrapped C
        attribute, exactly like ``cdef public``/``cdef readonly`` -- but
        it can only ever get a generic ``PyObject*`` slot (only an
        extension type gets a specifically-typed one), so its own
        ``Entry.type`` is indistinguishable from a genuinely untyped
        ``object`` attribute. The class name the source actually wrote
        must still make it through to both the class-body annotation and
        the synthesized ``__init__`` parameter, not fall back to the
        generic ``object``."""
        pyx_file = temp_dir / "test.pyx"
        pyx_file.write_text("""
import dataclasses


@dataclasses.dataclass
class Point:
    x: float


@dataclasses.dataclass
cdef class PointCdef:
    x: float
    pt: Point
""")
        result = StubgenPyx().convert_str(pyx_file.read_text(), pyx_path=pyx_file)
        assert "class PointCdef" in result
        assert "pt: Point" in result
        assert "def __init__(self, x: float, pt: Point): ..." in result
        assert "pt: object" not in result

    def test_cdef_public_extension_type_field_still_keeps_its_class_name(
        self, temp_dir
    ):
        """Baseline for the same fixture, using a raw ``cdef public``
        attribute of a real ``cdef class`` type instead of a bare
        annotation -- this one has always resolved correctly, since a
        `cdef class` attribute gets a specifically-typed slot at the
        `Entry`/`Type` level and needs no static-annotation fallback."""
        pyx_file = temp_dir / "test.pyx"
        pyx_file.write_text("""
import dataclasses


@dataclasses.dataclass
cdef class PointCdef:
    cdef public double x


@dataclasses.dataclass
cdef class PointCdef2:
    cdef public PointCdef pt
""")
        result = StubgenPyx().convert_str(pyx_file.read_text(), pyx_path=pyx_file)
        assert "pt: PointCdef" in result
        assert "def __init__(self, pt: PointCdef): ..." in result

    def test_defaulted_union_annotated_field_keeps_its_full_type(self, temp_dir):
        """The same generic-`object`-`Entry`-type problem as a bare
        extension-type annotation, but for a *defaulted* attribute whose
        annotation is a PEP 604 union of a generic and `None`
        (``count: int | Sequence[int] | None = None``) -- a
        `SingleAssignmentNode`, not the bare `ExprStatNode`
        `_capture_annotation` covers, so the source annotation has to
        reach `_stubgen_static_python_annotations` through
        `_capture_annotated_assignment_property_type` instead. Also
        confirms the otherwise-unused `Sequence` import
        survives once its only use is inside a field's dropped
        annotation."""
        pyx_file = temp_dir / "test.pyx"
        pyx_file.write_text("""
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass
cdef class Options:
    count: int | Sequence[int] | None = None
    backfill: bool | Sequence[bool] = False
""")
        result = StubgenPyx().convert_str(pyx_file.read_text(), pyx_path=pyx_file)
        assert "from collections.abc import Sequence" in result
        assert "count: int | Sequence[int] | None" in result
        assert "backfill: bool | Sequence[bool]" in result
        assert (
            "def __init__(self, count: int | Sequence[int] | None=None, "
            "backfill: bool | Sequence[bool]=False): ..."
        ) in result
        assert ": object" not in result

    def test_scalar_union_with_none_default_keeps_the_union(self, temp_dir):
        """A union of a real scalar C type and `None`
        (``x: float | None = None``) is the sharpest version of this bug:
        unlike a plain Python class or a generic, `float` alone *does*
        get a specifically-typed `Entry`/`Type` slot (`PyrexTypes`
        recognizes the builtin `float` on its own), so the naive object-
        only fallback never catches this -- `Entry.type` renders as the
        perfectly valid `float`, just missing the `| None`, and combined
        with the real default of `None` produces `def __init__(self, x:
        float=None)`: no type checker accepts an unmarked default of
        `None` on a non-Optional parameter."""
        pyx_file = temp_dir / "test.pyx"
        pyx_file.write_text("""
from dataclasses import dataclass


@dataclass
cdef class Options:
    x: float | None = None
""")
        result = StubgenPyx().convert_str(pyx_file.read_text(), pyx_path=pyx_file)
        assert "x: float | None" in result
        assert "def __init__(self, x: float | None=None): ..." in result
