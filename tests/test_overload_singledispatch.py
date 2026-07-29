from __future__ import annotations

import warnings
from dataclasses import dataclass

import pytest

from stubgen_pyx import StubgenPyx
from stubgen_pyx.config import StubgenPyxConfig
from stubgen_pyx.postprocessing.overload_singledispatch import (
    SingledispatchStubError,
)


def _stubgen() -> StubgenPyx:
    return StubgenPyx(StubgenPyxConfig(exclude_attribution=True, sort_imports=False))


def _convert(pyx_src: str, tmp_path, case_id: str) -> str:
    pyx_path = tmp_path / f"{case_id}.pyx"
    pyx_path.write_text(pyx_src, encoding="utf-8")
    return _stubgen().convert_str(pyx_src, pyx_path=pyx_path)


@dataclass(frozen=True)
class SuccessCase:
    id: str
    pyx: str
    expected: str


@dataclass(frozen=True)
class WarnCase:
    id: str
    pyx: str
    warning_match: str
    expected: str


@dataclass(frozen=True)
class RaiseCase:
    id: str
    pyx: str
    message_match: str


SUCCESS_CASES = [
    SuccessCase(
        id="case_01_form_a_decorator_base_and_registered_type",
        # Also locks the -> Any default when a variant has no return annotation.
        pyx=(
            "import functools\n"
            "@functools.singledispatch\n"
            "def convert(x): ...\n"
            "@convert.register(int)\n"
            "def from_int(x): ...\n"
        ),
        expected="from typing import Any\ndef convert(x: int) -> Any: ...\n",
    ),
    SuccessCase(
        id="case_02_form_b_assignment_base_and_registered_type",
        pyx=(
            "import functools\n"
            "convert = functools.singledispatch(lambda x: x)\n"
            "@convert.register(str)\n"
            "def from_str(x): ...\n"
        ),
        expected="from typing import Any\ndef convert(x: str) -> Any: ...\n",
    ),
    SuccessCase(
        id="case_03_form_c_bare_register_uses_annotation",
        pyx=(
            "from functools import singledispatch\n"
            "@singledispatch\n"
            "def convert(x): ...\n"
            "@convert.register\n"
            "def from_float(x: float): ...\n"
        ),
        expected="from typing import Any\ndef convert(x: float) -> Any: ...\n",
    ),
    SuccessCase(
        id="case_04_duplicate_registration_keeps_last",
        # Pure Python singledispatch silently overwrites; last @register(T) wins.
        pyx=(
            "import functools\n"
            "@functools.singledispatch\n"
            "def convert(x): ...\n"
            "@convert.register(int)\n"
            "def a(x) -> str: ...\n"
            "@convert.register(int)\n"
            "def b(x) -> bytes: ...\n"
        ),
        expected="def convert(x: int) -> bytes: ...\n",
    ),
    SuccessCase(
        id="case_05_multiple_groups",
        pyx=(
            "import functools\n"
            "@functools.singledispatch\n"
            "def one(x): ...\n"
            "@one.register(int)\n"
            "def one_int(x): ...\n"
            "@functools.singledispatch\n"
            "def two(x): ...\n"
            "@two.register(str)\n"
            "def two_str(x): ...\n"
        ),
        expected=(
            "from typing import Any\n"
            "def one(x: int) -> Any: ...\n"
            "def two(x: str) -> Any: ...\n"
        ),
    ),
    SuccessCase(
        id="case_06_dotted_singledispatch_attr",
        pyx=(
            "import functools\n"
            "@functools.singledispatch\n"
            "def convert(x): ...\n"
            "@convert.register(bytes)\n"
            "def from_bytes(x): ...\n"
        ),
        expected="from typing import Any\ndef convert(x: bytes) -> Any: ...\n",
    ),
    SuccessCase(
        id="case_07_explicit_return_annotation_preserved_without_any_import",
        pyx=(
            "import functools\n"
            "@functools.singledispatch\n"
            "def convert(x): ...\n"
            "@convert.register(int)\n"
            "def from_int(x) -> str: ...\n"
        ),
        expected="def convert(x: int) -> str: ...\n",
    ),
    SuccessCase(
        id="case_08_existing_overload_import_is_trimmed_when_unused",
        # Input already has `from typing import overload`, but the single-variant
        # collapse doesn't emit any @overload. trim_imports (which runs after
        # overload_singledispatch in the pipeline) then removes the now-unused
        # `overload` import.
        pyx=(
            "from typing import overload\n"
            "import functools\n"
            "@functools.singledispatch\n"
            "def convert(x): ...\n"
            "@convert.register(int)\n"
            "def from_int(x): ...\n"
        ),
        expected="from typing import Any\ndef convert(x: int) -> Any: ...\n",
    ),
    SuccessCase(
        id="case_09_multiple_variants_emit_overload_decorators",
        # >=2 typed variants per group: the spec's >=2-definitions rule is
        # satisfied, so each variant becomes an `@overload` and no plain-def
        # implementation stub is appended (stubs must not include one).
        pyx=(
            "import functools\n"
            "@functools.singledispatch\n"
            "def convert(x): ...\n"
            "@convert.register(int)\n"
            "def from_int(x) -> str: ...\n"
            "@convert.register(bytes)\n"
            "def from_bytes(x) -> bytes: ...\n"
        ),
        expected=(
            "from typing import overload\n"
            "@overload\n"
            "def convert(x: int) -> str: ...\n"
            "@overload\n"
            "def convert(x: bytes) -> bytes: ...\n"
        ),
    ),
]


WARN_CASES = [
    WarnCase(
        id="case_01_multi_arg_register_is_unsupported",
        # @base.register(T, extra) is legal Python but pathological at runtime.
        # The pass warns and leaves the group unchanged; trim_imports keeps
        # `import functools` because @functools.singledispatch still uses it.
        pyx=(
            "import functools\n"
            "@functools.singledispatch\n"
            "def convert(x): ...\n"
            "@convert.register(int, str)\n"
            "def from_int(x): ...\n"
        ),
        warning_match="unsupported",
        expected=(
            "import functools\n"
            "@functools.singledispatch\n"
            "def convert(x): ...\n"
            "@convert.register(int, str)\n"
            "def from_int(x): ...\n"
        ),
    ),
    WarnCase(
        id="case_02_no_overloads",
        # @singledispatch alone with no @register — legal but nothing to unify.
        pyx="import functools\n@functools.singledispatch\ndef convert(x): ...\n",
        warning_match="no overloads",
        expected=("import functools\n@functools.singledispatch\ndef convert(x): ...\n"),
    ),
]


RAISE_CASES = [
    RaiseCase(
        id="case_01_empty_register_call",
        # @base.register() raises TypeError at import in pure Python.
        pyx=(
            "import functools\n"
            "@functools.singledispatch\n"
            "def convert(x): ...\n"
            "@convert.register()\n"
            "def from_int(x): ...\n"
        ),
        message_match="no arguments",
    ),
    RaiseCase(
        id="case_02_keyword_register_call",
        # @base.register(cls, kw=...) raises TypeError at import in pure Python.
        pyx=(
            "import functools\n"
            "@functools.singledispatch\n"
            "def convert(x): ...\n"
            "@convert.register(int, alt=str)\n"
            "def from_int(x): ...\n"
        ),
        message_match="keyword arguments",
    ),
    RaiseCase(
        id="case_03_bare_register_on_untyped_function",
        # Bare @base.register on unannotated fn raises TypeError at import.
        pyx=(
            "import functools\n"
            "@functools.singledispatch\n"
            "def convert(x): ...\n"
            "@convert.register\n"
            "def unknown(x): ...\n"
        ),
        message_match="has no type",
    ),
    RaiseCase(
        id="case_04_mixed_group_with_untyped_variant",
        # Same rationale: any untyped variant in a group makes the module invalid.
        pyx=(
            "import functools\n"
            "@functools.singledispatch\n"
            "def convert(x): ...\n"
            "@convert.register(int)\n"
            "def ok(x): ...\n"
            "@convert.register\n"
            "def bad(x): ...\n"
        ),
        message_match="has no type",
    ),
]


@pytest.mark.parametrize("case", SUCCESS_CASES, ids=lambda case: case.id)
def test_overload_singledispatch_success(case: SuccessCase, tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = _convert(case.pyx, tmp_path, case.id)

    assert result == case.expected


@pytest.mark.parametrize("case", WARN_CASES, ids=lambda case: case.id)
def test_overload_singledispatch_warns_and_output_is_deterministic(
    case: WarnCase, tmp_path
):
    with pytest.warns(UserWarning, match=case.warning_match):
        result = _convert(case.pyx, tmp_path, case.id)

    assert result == case.expected


@pytest.mark.parametrize("case", RAISE_CASES, ids=lambda case: case.id)
def test_overload_singledispatch_raises_on_invalid_python(case: RaiseCase, tmp_path):
    with pytest.raises(SingledispatchStubError, match=case.message_match):
        _convert(case.pyx, tmp_path, case.id)
