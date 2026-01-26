from __future__ import annotations

import pathlib

from stubgen_pyx.stubgen import StubgenPyx

THIS_DIR = pathlib.Path(__file__).parent.resolve()


def _delete_pyi_files():
    for file in THIS_DIR.joinpath("fixtures").glob("*.pyi"):
        file.unlink(missing_ok=True)


def test_smoke():
    """Smoke test to ensure stubgen-pyx runs without errors on sample Cython files."""
    try:
        _delete_pyi_files()
        stubgen = StubgenPyx()
        stubgen.convert_glob(str(THIS_DIR.joinpath("fixtures", "*.pyx")))
    finally:
        _delete_pyi_files()
