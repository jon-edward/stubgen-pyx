from __future__ import annotations

import os
import pathlib
import sys

import mypy.api

from stubgen_pyx.stubgen import StubgenPyx

THIS_DIR = pathlib.Path(__file__).parent.resolve()


def _delete_pyi_files():
    for file in THIS_DIR.joinpath("fixtures").glob("**/*.pyi"):
        file.unlink(missing_ok=True)


def _mypy_process_output(mypy_stdout: str, mypy_stderr: str, mypy_exit: int):
    print(mypy_stdout)
    print(mypy_stderr, file=sys.stderr)
    assert mypy_exit == 0, "mypy failed"


def _type_check_files():
    extra_args = ("--strict", "--explicit-package-bases")

    current_mypy_path = os.environ.get("MYPYPATH", "")
    os.environ["MYPYPATH"] = str(THIS_DIR / "fixtures")
    _mypy_process_output(*mypy.api.run([str(THIS_DIR / "fixtures"), *extra_args]))
    os.environ["MYPYPATH"] = current_mypy_path


def test_smoke():
    """Smoke test to ensure stubgen-pyx runs without errors on sample Cython files."""
    try:
        _delete_pyi_files()
        stubgen = StubgenPyx()
        stubgen.convert_glob(str(THIS_DIR.joinpath("fixtures", "**/*.pyx")))
        _type_check_files()
    finally:
        _delete_pyi_files()
