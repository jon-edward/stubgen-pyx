"""A `.pxd` is analysed into its module scope exactly once, so no
redeclaration diagnostics are recorded for a well-formed pxd/pyx pair."""

from __future__ import annotations

from stubgen_pyx import StubgenPyx, StubgenPyxConfig

_CONFIG = StubgenPyxConfig(exclude_attribution=True, continue_on_error=True)

_B_PXD = "cdef class B:\n    cdef readonly int x\n    cpdef int f(self)\n"
_B_PYX = "cdef class B:\n    cpdef int f(self):\n        return 1\n"


def _write(tmp_path, files: dict[str, str]) -> None:
    for name, text in files.items():
        (tmp_path / name).write_text(text)


def _diagnostics(results) -> list[str]:
    return [str(d) for r in results for d in r.diagnostics]


def test_pxd_pyx_pair_has_no_diagnostics(tmp_path):
    _write(tmp_path, {"b.pxd": _B_PXD, "b.pyx": _B_PYX})
    results = StubgenPyx(_CONFIG).convert_multiple_files([tmp_path / "b.pyx"])
    assert all(r.success for r in results)
    assert _diagnostics(results) == []


def test_pxd_loaded_by_cimport_first_has_no_diagnostics(tmp_path):
    # `a.pyx` is converted first and `cimport`s `b`, so Cython auto-loads
    # `b.pxd` before `b.pyx`'s own pxd parse runs.
    _write(
        tmp_path,
        {
            "b.pxd": _B_PXD,
            "b.pyx": _B_PYX,
            "a.pyx": "from b cimport B\n\ncdef class A(B):\n    pass\n",
        },
    )
    results = StubgenPyx(_CONFIG).convert_multiple_files(
        [tmp_path / "a.pyx", tmp_path / "b.pyx"]
    )
    assert all(r.success for r in results)
    assert _diagnostics(results) == []


def test_real_diagnostics_survive(tmp_path):
    _write(
        tmp_path,
        {
            "b.pxd": "cimport no_such_module\n" + _B_PXD,
            "b.pyx": _B_PYX,
        },
    )
    results = StubgenPyx(_CONFIG).convert_multiple_files([tmp_path / "b.pyx"])
    assert any("no_such_module" in d for d in _diagnostics(results))
