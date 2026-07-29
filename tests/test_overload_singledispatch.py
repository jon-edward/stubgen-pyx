from __future__ import annotations

import ast

import pytest

from stubgen_pyx.postprocessing.overload_singledispatch import overload_singledispatch


def _overload(code: str) -> str:
    return ast.unparse(overload_singledispatch(ast.parse(code)))


def _assert_ast_unchanged(code: str, result: str) -> None:
    assert ast.dump(ast.parse(result)) == ast.dump(ast.parse(code))


def test_form_a_decorator_base_and_registered_type():
    result = _overload(
        "import functools\n"
        "@functools.singledispatch\n"
        "def convert(x): ...\n"
        "@convert.register(int)\n"
        "def _from_int(x): ..."
    )

    assert "from typing import overload" in result
    assert "@overload\ndef convert(x: int)" in result
    assert "_from_int" not in result
    assert "singledispatch" not in result


def test_form_b_assignment_base_and_registered_type():
    result = _overload(
        "import functools\n"
        "convert = functools.singledispatch(lambda x: x)\n"
        "@convert.register(str)\n"
        "def _from_str(x): ..."
    )

    assert "@overload\ndef convert(x: str)" in result
    assert "def convert(x):" in result
    assert "_from_str" not in result


def test_form_c_bare_register_uses_annotation():
    result = _overload(
        "from functools import singledispatch\n"
        "@singledispatch\n"
        "def convert(x): ...\n"
        "@convert.register\n"
        "def _from_float(x: float): ..."
    )

    assert "@overload\ndef convert(x: float)" in result


def test_conflicting_registration_warns_and_leaves_group_unchanged():
    code = (
        "import functools\n"
        "@functools.singledispatch\n"
        "def convert(x): ...\n"
        "@convert.register(int)\n"
        "def a(x): ...\n"
        "@convert.register(int)\n"
        "def b(x): ..."
    )

    with pytest.warns(UserWarning, match="duplicate"):
        result = _overload(code)

    assert "@functools.singledispatch" in result
    assert "def a" in result
    assert "def b" in result


def test_multi_arg_register_is_unhandled_and_left_unchanged():
    code = (
        "import functools\n"
        "@functools.singledispatch\n"
        "def convert(x): ...\n"
        "@convert.register(int, str)\n"
        "def from_int(x): ..."
    )

    with pytest.warns(UserWarning, match="no overloads"):
        result = _overload(code)

    _assert_ast_unchanged(code, result)


def test_keyword_register_is_unhandled_and_left_unchanged():
    code = (
        "import functools\n"
        "@functools.singledispatch\n"
        "def convert(x): ...\n"
        "@convert.register(int, alt=str)\n"
        "def from_int(x): ..."
    )

    with pytest.warns(UserWarning, match="no overloads"):
        result = _overload(code)

    _assert_ast_unchanged(code, result)


def test_empty_register_call_is_unhandled_and_left_unchanged():
    code = (
        "import functools\n"
        "@functools.singledispatch\n"
        "def convert(x): ...\n"
        "@convert.register()\n"
        "def from_int(x): ..."
    )

    with pytest.warns(UserWarning, match="no overloads"):
        result = _overload(code)

    _assert_ast_unchanged(code, result)


def test_multiple_groups_are_unified():
    result = _overload(
        "import functools\n"
        "@functools.singledispatch\n"
        "def one(x): ...\n"
        "@one.register(int)\n"
        "def one_int(x): ...\n"
        "@functools.singledispatch\n"
        "def two(x): ...\n"
        "@two.register(str)\n"
        "def two_str(x): ..."
    )

    assert "def one(x: int)" in result
    assert "def two(x: str)" in result
    assert "one_int" not in result
    assert "two_str" not in result


def test_dotted_singledispatch_attr_is_recognized():
    result = _overload(
        "import functools\n"
        "@functools.singledispatch\n"
        "def convert(x): ...\n"
        "@convert.register(bytes)\n"
        "def _from_bytes(x): ..."
    )

    assert "def convert(x: bytes)" in result


def test_fallback_no_overloads_warns_and_leaves_group_unchanged():
    code = "import functools\n@functools.singledispatch\ndef convert(x): ..."

    with pytest.warns(UserWarning, match="no overloads"):
        result = _overload(code)

    assert "@functools.singledispatch" in result
    assert "def convert" in result


def test_fallback_unresolvable_warns_and_leaves_group_unchanged():
    code = (
        "import functools\n"
        "@functools.singledispatch\n"
        "def convert(x): ...\n"
        "@convert.register\n"
        "def unknown(x): ..."
    )

    with pytest.warns(UserWarning, match="untyped"):
        result = _overload(code)

    assert "@convert.register" in result
    assert "def unknown" in result


def test_fallback_mixed_warns_and_leaves_whole_group_unchanged():
    code = (
        "import functools\n"
        "@functools.singledispatch\n"
        "def convert(x): ...\n"
        "@convert.register(int)\n"
        "def ok(x): ...\n"
        "@convert.register\n"
        "def bad(x): ..."
    )

    with pytest.warns(UserWarning, match="untyped"):
        result = _overload(code)

    assert "def ok" in result
    assert "def bad" in result
    assert "@overload" not in result


def test_overload_import_is_injected_when_needed():
    result = _overload(
        "import functools\n"
        "@functools.singledispatch\n"
        "def convert(x): ...\n"
        "@convert.register(int)\n"
        "def _from_int(x): ..."
    )

    assert "from typing import overload" in result


def test_missing_variant_return_annotation_defaults_to_any():
    result = _overload(
        "import functools\n"
        "@functools.singledispatch\n"
        "def convert(x): ...\n"
        "@convert.register(int)\n"
        "def _from_int(x): ..."
    )

    assert "from typing import overload, Any" in result
    assert "@overload\ndef convert(x: int) -> Any" in result


def test_explicit_variant_return_annotation_is_preserved_without_any_import():
    result = _overload(
        "import functools\n"
        "@functools.singledispatch\n"
        "def convert(x): ...\n"
        "@convert.register(int)\n"
        "def _from_int(x) -> str: ..."
    )

    assert "from typing import overload" in result
    assert "from typing import overload, Any" not in result
    assert "@overload\ndef convert(x: int) -> str" in result


def test_existing_overload_import_is_not_duplicated():
    result = _overload(
        "from typing import overload\n"
        "import functools\n"
        "@functools.singledispatch\n"
        "def convert(x): ...\n"
        "@convert.register(int)\n"
        "def _from_int(x): ..."
    )

    assert result.count("from typing import overload") == 1
