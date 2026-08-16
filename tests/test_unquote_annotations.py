"""Tests for annotation-unquoting, exercised through the public API.

Each ``Case`` feeds Cython source into :meth:`StubgenPyx.convert_str` and
asserts on the final ``.pyi`` output. The scenarios cover the four
annotation surfaces the pass touches (function args, return types,
``AnnAssign`` targets, class methods) plus the guard on non-parseable
strings.
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


CASES = [
    Case(
        id="quoted_param_annotation_unquoted",
        pyx="""\
cdef class Foo:
    pass

cpdef void f(x: 'Foo | None'):
    pass
""",
        expected="class Foo: ...\n\ndef f(x: Foo | None) -> None: ...\n",
    ),
    Case(
        id="quoted_return_annotation_unquoted",
        pyx="""\
def f() -> 'list[int]':
    pass
""",
        expected="def f() -> list[int]: ...\n",
    ),
    Case(
        id="quoted_nested_generic_return_unquoted",
        pyx="""\
def f() -> 'dict[str, list[int]]':
    pass
""",
        expected="def f() -> dict[str, list[int]]: ...\n",
    ),
    Case(
        id="quoted_annotation_preserves_cimport",
        pyx="""\
from some_mod cimport Foo

cpdef void f(x: 'Foo | None'):
    pass
""",
        expected="from some_mod import Foo\ndef f(x: Foo | None) -> None: ...\n",
    ),
    Case(
        id="non_parseable_annotation_stays_quoted",
        pyx="""\
cpdef void f(x: 'not valid python >>>'):
    pass
""",
        expected="def f(x: 'not valid python >>>') -> None: ...\n",
    ),
    Case(
        id="multiple_quoted_params_all_unquoted",
        pyx="""\
cdef class Foo:
    pass

cdef class Bar:
    pass

cdef class Baz:
    pass

cpdef void f(x: 'Foo | None', y: 'Bar | Baz'):
    pass
""",
        expected="class Foo: ...\n\nclass Bar: ...\n\nclass Baz: ...\n\ndef f(x: Foo | None, y: Bar | Baz) -> None: ...\n",
    ),
    Case(
        id="class_method_quoted_annotation_unquoted",
        pyx="""\
cdef class Foo:
    pass

cdef class Ops:
    cpdef void f(self, x: 'Foo | None'):
        pass
""",
        expected="class Foo: ...\n\nclass Ops:\n    def f(self, x: Foo | None) -> None: ...\n",
    ),
    Case(
        id="module_level_annassign_unquoted",
        pyx="""\
cdef class Foo:
    pass

x: 'Foo | None'
""",
        expected="x: Foo | None\n\nclass Foo: ...\n",
    ),
    Case(
        id="async_function_quoted_return_unquoted",
        pyx="""\
async def f() -> 'list[int]':
    pass
""",
        expected="async def f() -> list[int]: ...\n",
    ),
    Case(
        id="async_function_quoted_param_unquoted",
        pyx="""\
cdef class Foo:
    pass

async def f(x: 'Foo | None'):
    pass
""",
        expected="class Foo: ...\n\nasync def f(x: Foo | None): ...\n",
    ),
    Case(
        id="class_attribute_quoted_annotation_unquoted",
        pyx="""\
cdef class Foo:
    x: 'int | None'
""",
        expected="class Foo:\n    x: int | None\n",
    ),
]


def _stubgen() -> StubgenPyx:
    return StubgenPyx(StubgenPyxConfig(exclude_attribution=True, sort_imports=False))


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_unquote_annotations_via_public_api(case: Case):
    result = _stubgen().convert_str(case.pyx)

    assert result == case.expected
