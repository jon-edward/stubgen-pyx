"""Helpers for tests that need a ``cimport`` target on disk.

``FAKE_NUMPY`` is a stand-in for numpy's ``.pxd``: a top-level ``fakenp``
module with an extension type named ``ndarray``. It exercises the same
rules as ``cimport numpy as cnp`` without needing numpy installed.
"""

from __future__ import annotations

from pathlib import Path

from stubgen_pyx import StubgenPyx, StubgenPyxConfig

FAKE_NUMPY = {
    "fakenp.pxd": "cdef class ndarray:\n    pass\n",
    "fakenp.pyx": "",
}


def convert_files(
    tmp_path: Path, files: dict[str, str], targets: list[str], **config_options
) -> dict[str, str]:
    """Write ``files`` under ``tmp_path``, convert ``targets`` together, and
    return each target's generated stub text, keyed by target name."""
    for name, text in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    config = StubgenPyxConfig(
        exclude_attribution=True,
        continue_on_error=True,
        include_private=True,
        **config_options,
    )
    results = StubgenPyx(config).convert_multiple_files(
        [tmp_path / target for target in targets]
    )
    assert all(r.success for r in results)
    assert [d for r in results for d in r.diagnostics] == []
    return {
        target: (tmp_path / target.replace(".pyx", ".pyi")).read_text()
        for target in targets
    }
