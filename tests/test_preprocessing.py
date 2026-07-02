"""Tests for preprocessing and file parsing helpers."""

from __future__ import annotations

from stubgen_pyx.parsing import file_parsing
from stubgen_pyx.parsing.preprocess import (
    extract_type_comments,
    get_lines_with_newlines_in_brackets,
    preprocess,
    remove_comments,
    remove_contained_newlines,
    replace_tabs_with_spaces,
)


def test_replace_tabs_with_spaces_converts_leading_tabs():
    code = "\tdef example():\n\t\treturn 1"
    result = replace_tabs_with_spaces(code)

    assert result.startswith("    def example():\n        return 1")


def test_preprocess_removes_comments_and_expands_blocks():
    code = "def sample():\n    if True: return 1;  # comment\n"
    result = preprocess(code)

    assert "#" not in result
    assert "if True:\n" in result
    assert "return 1" in result


def test_remove_comments_and_contained_newlines_transform_code():
    code = "value = [1,\n2]\n"
    result = remove_comments(code)
    result = remove_contained_newlines(result)

    assert "[1,2]" in result


def test_extract_type_comments_returns_line_to_comment_mapping():
    code = "x = 1  # type: ignore\nvalue = 2  # type: int\n"

    assert extract_type_comments(code) == {1: "# type: ignore", 2: "# type: int"}


def test_get_lines_with_newlines_in_brackets_detects_bracketed_lines():
    code = "value = [1,\n2]\nnext_value = 3\n"

    assert get_lines_with_newlines_in_brackets(code) == [1]


def test_file_parsing_preprocess_expands_valid_includes(tmp_path):
    include_file = tmp_path / "included.pyx"
    include_file.write_text("def included():\n    return 42\n", encoding="utf-8")
    source_file = tmp_path / "source.pyx"
    source_file.write_text("", encoding="utf-8")

    result = file_parsing._expand_includes(source_file, 'include "included.pyx"')

    assert "def included" in result
    assert 'include "included.pyx"' not in result


def test_file_parsing_preprocess_handles_missing_include(tmp_path):
    source_file = tmp_path / "source.pyx"
    source_file.write_text("", encoding="utf-8")

    result = file_parsing._expand_includes(source_file, 'include "missing.pyx"')

    assert "include" not in result
    assert result.strip() == ""
