from __future__ import annotations

from dataclasses import dataclass

import pytest

from stubgen_pyx import StubgenPyx
from stubgen_pyx.config import StubgenPyxConfig


@dataclass(frozen=True)
class Probe:
    id: str
    pyx: str
    expected: str
    pxd: str | None = None
    pxd_name: str | None = None


PROBES = [
    Probe(
        id="probe_01_libc_stdint_int32",
        pyx="""\
from libc.stdint cimport int32_t

def foo(int32_t x):
    pass
""",
        expected="def foo(x: int): ...\n",
    ),
    Probe(
        id="probe_02_libc_stddef_size_t",
        pyx="""\
from libc.stddef cimport size_t

def foo() -> size_t:
    return 0
""",
        expected="def foo() -> int: ...\n",
    ),
    Probe(
        id="probe_03_libcpp_string",
        pyx="""\
from libcpp.string cimport string

def foo(string s):
    pass
""",
        expected="def foo(s: bytes): ...\n",
    ),
    Probe(
        id="probe_04_cdef_enum_from_module",
        pyx="""\
from probe_04_helper cimport MyCEnum

def foo(MyCEnum x):
    pass
""",
        pxd="""\
cdef enum MyCEnum:
    A = 0
    B = 1
""",
        pxd_name="probe_04_helper.pxd",
        expected="from probe_04_helper import MyCEnum\ndef foo(x: MyCEnum): ...\n",
    ),
    Probe(
        id="probe_05_cpdef_enum_from_module",
        pyx="""\
from probe_05_helper cimport MyCpdefEnum

def foo(MyCpdefEnum x):
    pass
""",
        pxd="""\
cpdef enum MyCpdefEnum:
    X = 0
    Y = 1
""",
        pxd_name="probe_05_helper.pxd",
        expected="from probe_05_helper import MyCpdefEnum\ndef foo(x: MyCpdefEnum): ...\n",
    ),
    Probe(
        id="probe_06_ctypedef_from_module",
        pyx="""\
from probe_06_helper cimport my_id_t

def foo(my_id_t x):
    pass
""",
        pxd="""\
from libc.stdint cimport uint32_t
ctypedef uint32_t my_id_t
""",
        pxd_name="probe_06_helper.pxd",
        expected="from probe_06_helper import my_id_t\ndef foo(x: my_id_t): ...\n",
    ),
    Probe(
        id="probe_07_libcpp_bool_baseline",
        pyx="""\
from libcpp cimport bool

def foo(bool x):
    pass
""",
        expected="def foo(x: bool): ...\n",
    ),
    Probe(
        id="probe_08_compound_cimport",
        pyx="""\
from libc.stdint cimport int32_t, uint64_t

def foo(int32_t x, uint64_t y):
    pass
""",
        expected="def foo(x: int, y: int): ...\n",
    ),
    Probe(
        id="probe_09_aliased_cimport",
        pyx="""\
from libc.stdint cimport int32_t as i32

def foo(i32 x):
    pass
""",
        expected="def foo(x: int): ...\n",
    ),
    Probe(
        id="probe_10_template_param",
        pyx="""\
from libcpp.vector cimport vector
from libc.stdint cimport int32_t

def foo(vector[int32_t] v):
    pass
""",
        expected="def foo(v: ...): ...\n",
    ),
]


def _stubgen() -> StubgenPyx:
    return StubgenPyx(StubgenPyxConfig(exclude_attribution=True, sort_imports=False))


def _convert_probe(probe: Probe, tmp_path) -> str:
    pyx_path = tmp_path / f"{probe.id}.pyx"
    if probe.pxd_name and probe.pxd:
        (tmp_path / probe.pxd_name).write_text(probe.pxd, encoding="utf-8")
    return _stubgen().convert_str(probe.pyx, pyx_path=pyx_path)


@pytest.mark.parametrize("probe", PROBES, ids=lambda probe: probe.id)
def test_suppress_c_primitives_current_behavior(probe: Probe, tmp_path):
    result = _convert_probe(probe, tmp_path)

    assert result == probe.expected
