from __future__ import annotations

import ast
import warnings
from dataclasses import dataclass

import pytest

from stubgen_pyx.postprocessing.overload_singledispatch import (
    SingledispatchStubError,
    overload_singledispatch,
)


def _overload(code: str) -> str:
    return ast.unparse(overload_singledispatch(ast.parse(code)))


@dataclass(frozen=True)
class SuccessCase:
    id: str
    pyi: str
    expected: str


@dataclass(frozen=True)
class WarnCase:
    id: str
    pyi: str
    warning_match: str
    expected: str


@dataclass(frozen=True)
class RaiseCase:
    id: str
    pyi: str
    message_match: str


SUCCESS_CASES = [
    SuccessCase(
        id="case_01_form_a_decorator_base_and_registered_type",
        pyi=(
            "import functools\n"
            "@functools.singledispatch\n"
            "def convert(x): ...\n"
            "@convert.register(int)\n"
            "def _from_int(x): ..."
        ),
        expected=(
            "from typing import Any\n"
            "import functools\n"
            "\n"
            "def convert(x: int) -> Any:\n"
            "    ..."
        ),
    ),
    SuccessCase(
        id="case_02_form_b_assignment_base_and_registered_type",
        pyi=(
            "import functools\n"
            "convert = functools.singledispatch(lambda x: x)\n"
            "@convert.register(str)\n"
            "def _from_str(x): ..."
        ),
        expected=(
            "from typing import Any\n"
            "import functools\n"
            "\n"
            "def convert(x: str) -> Any:\n"
            "    ..."
        ),
    ),
    SuccessCase(
        id="case_03_form_c_bare_register_uses_annotation",
        pyi=(
            "from functools import singledispatch\n"
            "@singledispatch\n"
            "def convert(x): ...\n"
            "@convert.register\n"
            "def _from_float(x: float): ..."
        ),
        expected=(
            "from typing import Any\n"
            "from functools import singledispatch\n"
            "\n"
            "def convert(x: float) -> Any:\n"
            "    ..."
        ),
    ),
    SuccessCase(
        id="case_04_duplicate_registration_keeps_last",
        # Pure Python singledispatch silently overwrites; last @register(T) wins.
        pyi=(
            "import functools\n"
            "@functools.singledispatch\n"
            "def convert(x): ...\n"
            "@convert.register(int)\n"
            "def a(x) -> str: ...\n"
            "@convert.register(int)\n"
            "def b(x) -> bytes: ..."
        ),
        expected=("import functools\n\ndef convert(x: int) -> bytes:\n    ..."),
    ),
    SuccessCase(
        id="case_05_multiple_groups",
        pyi=(
            "import functools\n"
            "@functools.singledispatch\n"
            "def one(x): ...\n"
            "@one.register(int)\n"
            "def one_int(x): ...\n"
            "@functools.singledispatch\n"
            "def two(x): ...\n"
            "@two.register(str)\n"
            "def two_str(x): ..."
        ),
        expected=(
            "from typing import Any\n"
            "import functools\n"
            "\n"
            "def one(x: int) -> Any:\n"
            "    ...\n"
            "\n"
            "def two(x: str) -> Any:\n"
            "    ..."
        ),
    ),
    SuccessCase(
        id="case_06_dotted_singledispatch_attr",
        pyi=(
            "import functools\n"
            "@functools.singledispatch\n"
            "def convert(x): ...\n"
            "@convert.register(bytes)\n"
            "def _from_bytes(x): ..."
        ),
        expected=(
            "from typing import Any\n"
            "import functools\n"
            "\n"
            "def convert(x: bytes) -> Any:\n"
            "    ..."
        ),
    ),
    SuccessCase(
        id="case_07_missing_variant_return_annotation_defaults_to_any",
        pyi=(
            "import functools\n"
            "@functools.singledispatch\n"
            "def convert(x): ...\n"
            "@convert.register(int)\n"
            "def _from_int(x): ..."
        ),
        expected=(
            "from typing import Any\n"
            "import functools\n"
            "\n"
            "def convert(x: int) -> Any:\n"
            "    ..."
        ),
    ),
    SuccessCase(
        id="case_08_explicit_return_annotation_preserved_without_any_import",
        pyi=(
            "import functools\n"
            "@functools.singledispatch\n"
            "def convert(x): ...\n"
            "@convert.register(int)\n"
            "def _from_int(x) -> str: ..."
        ),
        expected=("import functools\n\ndef convert(x: int) -> str:\n    ..."),
    ),
    SuccessCase(
        id="case_09_existing_overload_import_not_duplicated",
        # `from typing import overload` already present; Any is added to it.
        pyi=(
            "from typing import overload\n"
            "import functools\n"
            "@functools.singledispatch\n"
            "def convert(x): ...\n"
            "@convert.register(int)\n"
            "def _from_int(x): ..."
        ),
        # Existing `from typing import overload` is preserved verbatim (the pass
        # never removes user imports); `Any` is appended to the same import line.
        # No `@overload` decorator is emitted because a single-variant group
        # collapses to a plain `def` (see docstring of overload_singledispatch).
        expected=(
            "from typing import overload, Any\n"
            "import functools\n"
            "\n"
            "def convert(x: int) -> Any:\n"
            "    ..."
        ),
    ),
    SuccessCase(
        id="case_10_multiple_variants_emit_overload_decorators",
        # >=2 typed variants per group: the spec's >=2-definitions rule is
        # satisfied, so each variant becomes an `@overload` and no plain-def
        # implementation stub is appended (stubs must not include one).
        pyi=(
            "import functools\n"
            "@functools.singledispatch\n"
            "def convert(x): ...\n"
            "@convert.register(int)\n"
            "def _from_int(x) -> str: ...\n"
            "@convert.register(bytes)\n"
            "def _from_bytes(x) -> bytes: ..."
        ),
        expected=(
            "from typing import overload\n"
            "import functools\n"
            "\n"
            "@overload\n"
            "def convert(x: int) -> str:\n"
            "    ...\n"
            "\n"
            "@overload\n"
            "def convert(x: bytes) -> bytes:\n"
            "    ..."
        ),
    ),
]


WARN_CASES = [
    WarnCase(
        id="case_01_multi_arg_register_is_unsupported",
        # @base.register(T, extra) is legal Python but pathological at runtime.
        pyi=(
            "import functools\n"
            "@functools.singledispatch\n"
            "def convert(x): ...\n"
            "@convert.register(int, str)\n"
            "def from_int(x): ..."
        ),
        warning_match="unsupported",
        expected=(
            "import functools\n"
            "\n"
            "@functools.singledispatch\n"
            "def convert(x):\n"
            "    ...\n"
            "\n"
            "@convert.register(int, str)\n"
            "def from_int(x):\n"
            "    ..."
        ),
    ),
    WarnCase(
        id="case_02_no_overloads",
        # @singledispatch alone with no @register — legal but nothing to unify.
        pyi="import functools\n@functools.singledispatch\ndef convert(x): ...",
        warning_match="no overloads",
        expected=(
            "import functools\n\n@functools.singledispatch\ndef convert(x):\n    ..."
        ),
    ),
]


RAISE_CASES = [
    RaiseCase(
        id="case_01_empty_register_call",
        # @base.register() raises TypeError at import in pure Python.
        pyi=(
            "import functools\n"
            "@functools.singledispatch\n"
            "def convert(x): ...\n"
            "@convert.register()\n"
            "def from_int(x): ..."
        ),
        message_match="no arguments",
    ),
    RaiseCase(
        id="case_02_keyword_register_call",
        # @base.register(cls, kw=...) raises TypeError at import in pure Python.
        pyi=(
            "import functools\n"
            "@functools.singledispatch\n"
            "def convert(x): ...\n"
            "@convert.register(int, alt=str)\n"
            "def from_int(x): ..."
        ),
        message_match="keyword arguments",
    ),
    RaiseCase(
        id="case_03_bare_register_on_untyped_function",
        # Bare @base.register on unannotated fn raises TypeError at import.
        pyi=(
            "import functools\n"
            "@functools.singledispatch\n"
            "def convert(x): ...\n"
            "@convert.register\n"
            "def unknown(x): ..."
        ),
        message_match="has no type",
    ),
    RaiseCase(
        id="case_04_mixed_group_with_untyped_variant",
        # Same rationale: any untyped variant in a group makes the module invalid.
        pyi=(
            "import functools\n"
            "@functools.singledispatch\n"
            "def convert(x): ...\n"
            "@convert.register(int)\n"
            "def ok(x): ...\n"
            "@convert.register\n"
            "def bad(x): ..."
        ),
        message_match="has no type",
    ),
]


@pytest.mark.parametrize("case", SUCCESS_CASES, ids=lambda case: case.id)
def test_overload_singledispatch_success(case: SuccessCase):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = _overload(case.pyi)

    assert result == case.expected


@pytest.mark.parametrize("case", WARN_CASES, ids=lambda case: case.id)
def test_overload_singledispatch_warns_and_output_is_deterministic(case: WarnCase):
    with pytest.warns(UserWarning, match=case.warning_match):
        result = _overload(case.pyi)

    assert result == case.expected


@pytest.mark.parametrize("case", RAISE_CASES, ids=lambda case: case.id)
def test_overload_singledispatch_raises_on_invalid_python(case: RaiseCase):
    with pytest.raises(SingledispatchStubError, match=case.message_match):
        _overload(case.pyi)
