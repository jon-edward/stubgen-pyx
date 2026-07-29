from __future__ import annotations

import ast
import warnings

import pytest

from stubgen_pyx.postprocessing.overload_singledispatch import (
    SingledispatchStubError,
    overload_singledispatch,
)


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


def test_duplicate_registration_keeps_last_matching_python_runtime_semantics():
    # Pure Python `functools.singledispatch` silently overwrites when the same
    # type is registered twice; the last @register(T) wins. Match that here:
    # no warning, only the last handler survives as the @overload signature.
    code = (
        "import functools\n"
        "@functools.singledispatch\n"
        "def convert(x): ...\n"
        "@convert.register(int)\n"
        "def a(x) -> str: ...\n"
        "@convert.register(int)\n"
        "def b(x) -> bytes: ..."
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = _overload(code)

    # Last @register(int) wins: return type is bytes, not str.
    assert "@overload\ndef convert(x: int) -> bytes" in result
    assert "-> str" not in result
    # Original helper names are erased; @functools.singledispatch is gone.
    assert "def a" not in result
    assert "def b" not in result
    assert "@functools.singledispatch" not in result


def test_multi_arg_register_is_unsupported_and_leaves_group_unchanged():
    # @base.register(T, extra) is technically legal Python but produces nonsense
    # at runtime (extra is treated as the handler). We can't emit a sensible
    # overload; warn with a clear "unsupported form" message and leave input alone.
    code = (
        "import functools\n"
        "@functools.singledispatch\n"
        "def convert(x): ...\n"
        "@convert.register(int, str)\n"
        "def from_int(x): ..."
    )

    with pytest.warns(UserWarning, match="unsupported"):
        result = _overload(code)

    _assert_ast_unchanged(code, result)


def test_keyword_register_raises():
    # @base.register(cls, kw=...) raises TypeError at import in pure Python.
    # Refuse to emit a stub for source that cannot be imported.
    code = (
        "import functools\n"
        "@functools.singledispatch\n"
        "def convert(x): ...\n"
        "@convert.register(int, alt=str)\n"
        "def from_int(x): ..."
    )

    with pytest.raises(SingledispatchStubError, match="keyword arguments"):
        _overload(code)


def test_empty_register_call_raises():
    # @base.register() raises TypeError at import in pure Python (missing 'cls').
    # Refuse to emit a stub for source that cannot be imported.
    code = (
        "import functools\n"
        "@functools.singledispatch\n"
        "def convert(x): ...\n"
        "@convert.register()\n"
        "def from_int(x): ..."
    )

    with pytest.raises(SingledispatchStubError, match="no arguments"):
        _overload(code)


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


def test_fallback_unresolvable_raises():
    # Bare @base.register on an unannotated function raises TypeError at import
    # in pure Python. Refuse to emit a stub for source that cannot be imported.
    code = (
        "import functools\n"
        "@functools.singledispatch\n"
        "def convert(x): ...\n"
        "@convert.register\n"
        "def unknown(x): ..."
    )

    with pytest.raises(SingledispatchStubError, match="has no type"):
        _overload(code)


def test_fallback_mixed_raises_when_any_variant_is_untyped():
    # Same rationale: if any variant in the group is a bare untyped @register,
    # the whole module is invalid Python at import time. Raise.
    code = (
        "import functools\n"
        "@functools.singledispatch\n"
        "def convert(x): ...\n"
        "@convert.register(int)\n"
        "def ok(x): ...\n"
        "@convert.register\n"
        "def bad(x): ..."
    )

    with pytest.raises(SingledispatchStubError, match="has no type"):
        _overload(code)


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
