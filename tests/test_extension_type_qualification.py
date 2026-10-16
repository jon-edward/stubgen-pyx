"""Extension types declared in another module are spelled the way the stub
imports them (``cimport fakenp as cnp`` -> ``cnp.ndarray``), not as a bare
name that nothing in the stub defines."""

from __future__ import annotations

import pytest

from stubgen_pyx import StubgenPyx, StubgenPyxConfig
from stubgen_pyx.conversion.pyrex_types import render_pyrex_type
from stubgen_pyx.parsing.context import StubgenContext
from stubgen_pyx.parsing.parser import parse_file

from .cimport_helpers import FAKE_NUMPY, convert_files

_CONFIG = StubgenPyxConfig(exclude_attribution=True, include_private=True)

_LIB_FILES = {
    "pkg/__init__.py": "",
    "pkg/lib.pxd": "cdef class Foo:\n    pass\n",
    "pkg/lib.pyx": "cdef class Foo:\n    pass\n",
}


def _convert_package(tmp_path, header: str, type_spelling: str) -> str:
    """Convert ``pkg/use.pyx``, whose ``.pxd`` declares ``cdef readonly
    <type_spelling> a`` on an extension type, next to ``pkg/lib.pyx``."""
    files = dict(_LIB_FILES)
    files["pkg/use.pxd"] = (
        f"{header}\ncdef class B:\n    cdef readonly {type_spelling} a\n"
    )
    files["pkg/use.pyx"] = f"{header}\ncdef class B:\n    pass\n"
    for name, text in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    config = StubgenPyxConfig(exclude_attribution=True, continue_on_error=True)
    targets = [tmp_path / "pkg/lib.pyx", tmp_path / "pkg/use.pyx"]
    results = StubgenPyx(config).convert_multiple_files(targets)
    assert all(r.success for r in results)
    return (tmp_path / "pkg/use.pyi").read_text()


class TestAliasedModuleType:
    PXD = (
        "cimport fakenp as cnp\n\n"
        "cdef class A:\n"
        "    cdef readonly cnp.ndarray ro\n"
        "    cdef public cnp.ndarray rw\n"
    )
    PYX_IMPORT = "cimport fakenp as cnp\n\ncdef class A:\n    pass\n"
    EXPECTED = (
        "from typing import Final\n\n"
        "import fakenp as cnp\n\n\n"
        "class A:\n    ro: Final[cnp.ndarray]\n    rw: cnp.ndarray\n"
    )

    def test_attributes_declared_in_pxd(self, tmp_path):
        result = convert_files(
            tmp_path,
            {**FAKE_NUMPY, "use.pxd": self.PXD, "use.pyx": self.PYX_IMPORT},
            ["fakenp.pyx", "use.pyx"],
        )
        assert result["use.pyx"] == self.EXPECTED

    def test_attributes_declared_in_pyx(self, tmp_path):
        result = convert_files(
            tmp_path,
            {**FAKE_NUMPY, "use.pyx": self.PXD},
            ["fakenp.pyx", "use.pyx"],
        )
        assert result["use.pyx"] == self.EXPECTED

    def test_star_import_does_not_leave_a_bare_name(self, tmp_path):
        # A star import disables `trim_not_defined`, so an unqualified
        # `ndarray` would be emitted as-is instead of `Incomplete`.
        result = convert_files(
            tmp_path,
            {
                **FAKE_NUMPY,
                "use.pxd": self.PXD,
                "use.pyx": "from os.path import *\n" + self.PYX_IMPORT,
            },
            ["fakenp.pyx", "use.pyx"],
        )
        assert "ro: Final[cnp.ndarray]" in result["use.pyx"]
        assert "Final[ndarray]" not in result["use.pyx"]


class TestProjectLocalExtensionTypes:
    @pytest.mark.parametrize(
        ("header", "type_spelling", "import_line", "annotation"),
        [
            (
                "cimport pkg.lib as pl",
                "pl.Foo",
                "import pkg.lib as pl",
                "Final[pl.Foo]",
            ),
            (
                "from pkg cimport lib",
                "lib.Foo",
                "from pkg import lib",
                "Final[lib.Foo]",
            ),
            (
                "from pkg.lib cimport Foo",
                "Foo",
                "from pkg.lib import Foo",
                "Final[Foo]",
            ),
            (
                "from pkg.lib cimport Foo as F",
                "F",
                "from pkg.lib import Foo as F",
                "Final[F]",
            ),
            (
                "cimport pkg.lib",
                "pkg.lib.Foo",
                "import pkg.lib",
                "Final[pkg.lib.Foo]",
            ),
        ],
    )
    def test_spelling_follows_the_cimport(
        self, tmp_path, header, type_spelling, import_line, annotation
    ):
        result = _convert_package(tmp_path, header, type_spelling)
        assert result == (
            "from typing import Final\n\n"
            f"{import_line}\n\n\n"
            f"class B:\n    a: {annotation}\n"
        )


class TestLocalAndBareRendering:
    def test_extension_type_in_same_module_stays_bare(self):
        result = StubgenPyx(_CONFIG).convert_str(
            "cdef class Other:\n    pass\n\n"
            "cdef class A:\n    cdef readonly Other o\n    cdef public Other p\n"
        )
        assert result == (
            "from typing import Final\n\n\n"
            "class Other: ...\n\n"
            "class A:\n    o: Final[Other]\n    p: Other\n"
        )

    def test_render_without_scope_is_the_bare_name(self, tmp_path):
        for name, text in {
            **FAKE_NUMPY,
            "use.pyx": (
                "cimport fakenp as cnp\n\n"
                "cdef class A:\n    cdef readonly cnp.ndarray ro\n"
            ),
        }.items():
            (tmp_path / name).write_text(text)
        context = StubgenContext(include_directories=[str(tmp_path)])
        parsed = parse_file(tmp_path / "use.pyx", context)
        class_entry = parsed.scope.entries["A"]
        attr_type = class_entry.type.scope.entries["ro"].type
        assert render_pyrex_type(attr_type) == "ndarray"
        assert (
            render_pyrex_type(attr_type, scope=class_entry.type.scope) == "cnp.ndarray"
        )
