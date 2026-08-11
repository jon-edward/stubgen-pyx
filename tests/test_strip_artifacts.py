"""Tests for stub-artifact stripping, exercised through the public API.

Each ``Case`` feeds Cython source into :meth:`StubgenPyx.convert_str` and
asserts on the final ``.pyi`` output. The scenarios cover the four categories
of artifact-stripping the pipeline is responsible for:

* preserving ``__all__`` (list, tuple, call, attribute RHS)
* preserving ``__hash__ = None`` assignments when present in source
* stripping ``cython`` / ``cpython`` cimports and bare ``import cython``
* stripping runtime-constant Call/Attribute RHS at module scope
* rewriting dangling annotations (via Ellipsis or ``Any`` depending on path)
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from stubgen_pyx import StubgenPyx
from stubgen_pyx.config import StubgenPyxConfig


@dataclass(frozen=True)
class Case:
    id: str
    pyx: str
    expected: str
    pxd: str | None = None


CASES = [
    Case(
        id="dunder_all_list_preserved",
        pyx="""\
__all__ = ['Foo']
cdef class Foo:
    pass
""",
        expected="__all__ = ['Foo']\n\nclass Foo: ...\n",
    ),
    Case(
        id="dunder_all_tuple_preserved",
        pyx="""\
__all__ = ('Foo',)
cdef class Foo:
    pass
""",
        expected="__all__ = 'Foo'\n\nclass Foo: ...\n",
    ),
    Case(
        id="dunder_all_call_rhs_not_deleted",
        pyx="""\
__all__ = build_all()
cdef class Foo:
    pass
""",
        expected="__all__ = ...\n\nclass Foo: ...\n",
    ),
    Case(
        id="dunder_all_attribute_rhs_not_deleted",
        pyx="""\
__all__ = exports.ALL
cdef class Foo:
    pass
""",
        expected="__all__ = ...\n\nclass Foo: ...\n",
    ),
    Case(
        id="eq_method_does_not_emit_hash_none",
        pyx="""\
cdef class Foo:
    def __eq__(self, other):
        return False
""",
        expected="class Foo:\n    def __eq__(self, other): ...\n",
    ),
    Case(
        id="class_hash_none_preserved",
        pyx="""\
cdef class Foo:
    __hash__ = None
""",
        expected="class Foo:\n    __hash__ = None\n",
    ),
    Case(
        id="module_hash_none_preserved",
        pyx="""\
__hash__ = None
cdef class Foo:
    pass
""",
        expected="__hash__ = None\n\nclass Foo: ...\n",
    ),
    Case(
        id="from_cython_cimport_stripped",
        pyx="""\
from cython cimport bint
cdef class Foo:
    cpdef bint check(self):
        return True
""",
        expected="class Foo:\n    def check(self) -> bool: ...\n",
    ),
    Case(
        id="from_cpython_cimport_stripped_dangles_alias",
        pyx="""\
from cpython cimport bool as pybool
def f(pybool x):
    pass
""",
        expected="from typing import Any\ndef f(x: Any): ...\n",
    ),
    Case(
        id="bare_import_cython_stripped",
        pyx="""\
import cython
def f():
    pass
""",
        expected="def f(): ...\n",
    ),
    Case(
        id="mixed_import_keeps_siblings",
        pyx="""\
import os
import cython
import sys
def f():
    pass
""",
        expected="def f(): ...\n",
    ),
    Case(
        id="runtime_constant_call_stripped",
        pyx="""\
SIZE_TYPE = DataType(42)
def f():
    pass
""",
        expected="SIZE_TYPE = ...\n\ndef f(): ...\n",
    ),
    Case(
        id="runtime_constant_attribute_stripped",
        pyx="""\
SEED = some_module.DEFAULT_SEED
def f():
    pass
""",
        expected="SEED = ...\n\ndef f(): ...\n",
    ),
    Case(
        id="plain_assignment_preserved",
        pyx="""\
X = 42
def f():
    pass
""",
        expected="X = 42\n\ndef f(): ...\n",
    ),
    Case(
        id="dangling_annotation_rewritten",
        pyx="""\
def f(x: MissingType) -> None:
    pass
""",
        expected="def f(x: ...) -> None: ...\n",
    ),
    Case(
        id="dangling_nested_generic_rewritten",
        pyx="""\
from typing import List
def f(x: List[Missing]) -> None:
    pass
""",
        expected="def f(x: ...) -> None: ...\n",
    ),
    Case(
        id="multiple_dangling_names_rewritten",
        pyx="""\
def f(x: MissingOne) -> MissingTwo:
    pass
""",
        expected="def f(x: ...) -> ...: ...\n",
    ),
    Case(
        id="builtin_annotations_preserved",
        pyx="""\
def f(x: int, y: str, z: bool) -> bytes:
    pass
""",
        expected="def f(x: int, y: str, z: bool) -> bytes: ...\n",
    ),
    Case(
        id="dangling_decorator_argument_rewritten",
        pyx="""\
some_decorator = None

@some_decorator(UndefinedType)
def foo() -> None:
    pass
""",
        expected=(
            "from typing import Any\n"
            "some_decorator = None\n\n"
            "@some_decorator(Any)\n"
            "def foo() -> None: ...\n"
        ),
    ),
]


def _stubgen() -> StubgenPyx:
    return StubgenPyx(StubgenPyxConfig(exclude_attribution=True, sort_imports=False))


def _convert_case(case: Case, tmp_path) -> str:
    pyx_path = tmp_path / f"{case.id}.pyx"
    if case.pxd:
        (tmp_path / "helper.pxd").write_text(case.pxd, encoding="utf-8")
    return _stubgen().convert_str(case.pyx, pxd_str=case.pxd, pyx_path=pyx_path)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_strip_artifacts_via_public_api(case: Case, tmp_path):
    result = _convert_case(case, tmp_path)

    assert result == case.expected
