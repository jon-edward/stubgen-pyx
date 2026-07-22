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
        id="case_01_libc_stdint_int32",
        pyx="""\
from libc.stdint cimport int32_t

def foo(int32_t x):
    pass
""",
        expected="def foo(x: int): ...\n",
    ),
    Case(
        id="case_02_libc_stddef_size_t",
        pyx="""\
from libc.stddef cimport size_t

def foo() -> size_t:
    return 0
""",
        expected="def foo() -> int: ...\n",
    ),
    Case(
        id="case_03_libcpp_string",
        pyx="""\
from libcpp.string cimport string

def foo(string s):
    pass
""",
        expected="def foo(s: bytes): ...\n",
    ),
    Case(
        id="case_04_cdef_enum_from_module",
        pyx="""\
from helper cimport MyCEnum

def foo(MyCEnum x):
    pass
""",
        pxd="""\
cdef enum MyCEnum:
    A = 0
    B = 1
""",
        expected="from helper import MyCEnum\ndef foo(x: MyCEnum): ...\n",
    ),
    Case(
        id="case_05_cpdef_enum_from_module",
        pyx="""\
from helper cimport MyCpdefEnum

def foo(MyCpdefEnum x):
    pass
""",
        pxd="""\
cpdef enum MyCpdefEnum:
    X = 0
    Y = 1
""",
        expected="from helper import MyCpdefEnum\ndef foo(x: MyCpdefEnum): ...\n",
    ),
    Case(
        id="case_06_ctypedef_from_module",
        pyx="""\
from helper cimport my_id_t

def foo(my_id_t x):
    pass
""",
        pxd="""\
from libc.stdint cimport uint32_t
ctypedef uint32_t my_id_t
""",
        expected="from helper import my_id_t\ndef foo(x: my_id_t): ...\n",
    ),
    Case(
        id="case_07_libcpp_bool_baseline",
        pyx="""\
from libcpp cimport bool

def foo(bool x):
    pass
""",
        expected="def foo(x: bool): ...\n",
    ),
    Case(
        id="case_08_compound_cimport",
        pyx="""\
from libc.stdint cimport int32_t, uint64_t

def foo(int32_t x, uint64_t y):
    pass
""",
        expected="def foo(x: int, y: int): ...\n",
    ),
    Case(
        id="case_09_aliased_cimport",
        pyx="""\
from libc.stdint cimport int32_t as i32

def foo(i32 x):
    pass
""",
        expected="def foo(x: int): ...\n",
    ),
    Case(
        id="case_10_template_param",
        pyx="""\
from libcpp.vector cimport vector
from libc.stdint cimport int32_t

def foo(vector[int32_t] v):
    pass
""",
        expected="def foo(v: ...): ...\n",
    ),
    Case(
        id="case_11_libcpp_memory_unique_ptr",
        pyx="""\
from libcpp.memory cimport unique_ptr

cpdef void take_ptr(unique_ptr[int] value):
    pass
""",
        expected="def take_ptr(value: ...) -> None: ...\n",
    ),
    Case(
        id="case_12_libcpp_vector",
        pyx="""\
from libcpp.vector cimport vector

cpdef void take_vector(vector[int] value):
    pass
""",
        expected="def take_vector(value: ...) -> None: ...\n",
    ),
    Case(
        id="case_13_libcpp_string",
        pyx="""\
from libcpp.string cimport string

cpdef void take_string(string value):
    pass
""",
        expected="def take_string(value: bytes) -> None: ...\n",
    ),
    Case(
        id="case_14_mixed_libcpp_and_normal",
        pyx="""\
from libcpp cimport bool
from some_module cimport Foo

cpdef Foo convert(bool x):
    pass
""",
        expected="from some_module import Foo\ndef convert(x: bool) -> Foo: ...\n",
    ),
]


def _stubgen() -> StubgenPyx:
    return StubgenPyx(StubgenPyxConfig(exclude_attribution=True, sort_imports=False))


def _convert_case(case: Case, tmp_path) -> str:
    pyx_path = tmp_path / f"{case.id}.pyx"
    if case.pxd:
        (tmp_path / "helper.pxd").write_text(case.pxd, encoding="utf-8")
    return _stubgen().convert_str(case.pyx, pyx_path=pyx_path)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_suppress_c_primitives_current_behavior(case: Case, tmp_path):
    result = _convert_case(case, tmp_path)

    assert result == case.expected
