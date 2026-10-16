"""`resolve_ctypedef_aliases` also resolves a plain ``cdef enum`` to ``int``
and drops its ``Name: TypeAlias = int`` stand-in; ``cpdef enum`` stays an
``IntEnum``.
"""

from __future__ import annotations

import pytest

from stubgen_pyx import StubgenPyx
from stubgen_pyx.config import StubgenPyxConfig
from stubgen_pyx.conversion.ctypedef_aliases import remove_assignment_from_module
from stubgen_pyx.models.pyi_elements import (
    PyiAssignment,
    PyiEnum,
    PyiModule,
    PyiScope,
)


def _convert(source: str, resolve: bool = True) -> str:
    config = StubgenPyxConfig(
        resolve_ctypedef_aliases=resolve, exclude_attribution=True
    )
    return StubgenPyx(config).convert_str(source)


class TestCdefEnumResolution:
    def test_off_keeps_enum_name_and_alias(self):
        result = _convert(
            "cdef enum Color:\n    RED\n\ndef f(Color c) -> Color:\n    return c\n",
            resolve=False,
        )
        assert (
            result == "from typing_extensions import TypeAlias\n\n"
            "Color: TypeAlias = int\n\n"
            "def f(c: Color) -> Color: ...\n"
        )

    def test_on_resolves_to_int_and_drops_alias(self):
        result = _convert(
            "cdef enum Color:\n    RED\n\ndef f(Color c) -> Color:\n    return c\n"
        )
        assert result == "def f(c: int) -> int: ...\n"

    def test_cpdef_enum_still_emits_intenum(self):
        result = _convert(
            "cpdef enum Shape:\n    SQ\n\ndef f(Shape s) -> Shape:\n    return s\n"
        )
        assert result == (
            "from enum import IntEnum\n\n\n"
            "class Shape(IntEnum):\n    SQ = ...\n\n"
            "def f(s: Shape) -> Shape: ...\n"
        )

    def test_cdef_and_cpdef_enums_side_by_side(self):
        result = _convert(
            "cdef enum Color:\n    RED\n"
            "cpdef enum Shape:\n    SQ\n"
            "def f(Color c, Shape s):\n    pass\n"
        )
        assert result == (
            "from enum import IntEnum\n\n\n"
            "class Shape(IntEnum):\n    SQ = ...\n\n"
            "def f(c: int, s: Shape): ...\n"
        )

    def test_extern_enum_resolves_to_int(self):
        result = _convert(
            'cdef extern from "x.h":\n    enum Ext:\n        EXT_A\n\n'
            "def f(Ext e):\n    pass\n"
        )
        assert result == "def f(e: int): ...\n"

    def test_ctypedef_of_enum_resolves_through_to_int(self):
        result = _convert(
            "cdef enum Color:\n    RED\n"
            "ctypedef Color ColorAlias\n"
            "def f(ColorAlias c):\n    pass\n"
        )
        assert result == "def f(c: int): ...\n"

    def test_class_attributes_resolve_to_int(self):
        result = _convert(
            "cdef enum Color:\n    RED\n"
            "cdef class W:\n"
            "    cdef public Color color\n"
            "    cdef readonly Color ro\n"
        )
        assert result == (
            "from typing import Final\n\n\n"
            "class W:\n    color: int\n    ro: Final[int]\n"
        )


def _convert_files(tmp_path, files: dict[str, str], targets: list[str]):
    for name, text in files.items():
        (tmp_path / name).write_text(text)
    config = StubgenPyxConfig(
        resolve_ctypedef_aliases=True, continue_on_error=True, exclude_attribution=True
    )
    results = StubgenPyx(config).convert_multiple_files(
        [tmp_path / target for target in targets]
    )
    assert all(r.success for r in results)
    return {
        target: (tmp_path / target.replace(".pyx", ".pyi")).read_text()
        for target in targets
    }


class TestCdefEnumBatchPruning:
    """The whole-batch check keeps an enum's alias only while another file in
    the same run still needs it after import trimming (e.g. an explicit
    ``X as X`` re-export), and drops it otherwise."""

    @pytest.mark.parametrize("reverse", [False, True])
    def test_aliases_pruned_when_sibling_substitutes_them_away(self, tmp_path, reverse):
        targets = ["types_mod.pyx", "consumer.pyx"]
        if reverse:
            targets.reverse()
        outputs = _convert_files(
            tmp_path,
            {
                "types_mod.pxd": (
                    "cdef enum Color:\n    RED\n"
                    "cdef enum Unused:\n    U\n"
                    "cpdef enum Shape:\n    SQ\n"
                ),
                "types_mod.pyx": "",
                "consumer.pyx": (
                    "from types_mod cimport Color, Shape\n\n"
                    "def f(Color c, Shape s) -> Color:\n    return c\n"
                ),
            },
            targets,
        )
        assert outputs["types_mod.pyx"] == (
            "from enum import IntEnum\n\n\nclass Shape(IntEnum):\n    SQ = ...\n"
        )
        assert outputs["consumer.pyx"] == (
            "from types_mod import Shape\n\n\ndef f(c: int, s: Shape) -> int: ...\n"
        )

    @pytest.mark.parametrize("reverse", [False, True])
    def test_alias_kept_when_sibling_reexports_it_unused_one_pruned(
        self, tmp_path, reverse
    ):
        targets = ["types_mod.pyx", "consumer.pyx"]
        if reverse:
            targets.reverse()
        outputs = _convert_files(
            tmp_path,
            {
                "types_mod.pxd": (
                    "cdef enum Color:\n    RED\n"
                    "cdef enum Unused:\n    U\n"
                    "cpdef enum Shape:\n    SQ\n"
                ),
                "types_mod.pyx": "",
                "consumer.pyx": (
                    "from types_mod cimport Color as Color, Shape\n\n"
                    "def f(Color c, Shape s) -> Color:\n    return c\n"
                ),
            },
            targets,
        )
        assert outputs["types_mod.pyx"] == (
            "from enum import IntEnum\n\n"
            "from typing_extensions import TypeAlias\n\n\n"
            "class Shape(IntEnum):\n    SQ = ...\n\n"
            "Color: TypeAlias = int\n"
        )
        assert outputs["consumer.pyx"] == (
            "from types_mod import Color as Color\n"
            "from types_mod import Shape\n\n\n"
            "def f(c: int, s: Shape) -> int: ...\n"
        )

    def test_pxd_only_enum_pruned_when_unused(self, tmp_path):
        outputs = _convert_files(
            tmp_path,
            {"mod.pxd": "cdef enum Color:\n    RED\n", "mod.pyx": ""},
            ["mod.pyx"],
        )
        assert outputs["mod.pyx"].strip() == ""

    def test_pxd_only_enum_pruned_and_substituted_when_used(self, tmp_path):
        outputs = _convert_files(
            tmp_path,
            {
                "mod.pxd": "cdef enum Color:\n    RED\n",
                "mod.pyx": "cpdef Color get():\n    return RED\n",
            },
            ["mod.pyx"],
        )
        assert outputs["mod.pyx"] == "def get() -> int: ...\n"


class TestCimportedTypedefOfEnum:
    @pytest.mark.parametrize("reverse", [False, True])
    def test_consumer_resolves_typedef_of_enum_to_int(self, tmp_path, reverse):
        """A consumer that cimports only the `ctypedef` of an enum -- not the
        enum -- still resolves it to `int`, and the enum's own alias is
        dropped."""
        targets = ["types_mod.pyx", "consumer.pyx"]
        if reverse:
            targets.reverse()
        outputs = _convert_files(
            tmp_path,
            {
                "types_mod.pxd": "cdef enum Color:\n    RED\nctypedef Color ColorAlias\n",
                "types_mod.pyx": "",
                "consumer.pyx": (
                    "from types_mod cimport ColorAlias as ColorAlias\n\n"
                    "def f(ColorAlias c):\n    pass\n"
                ),
            },
            targets,
        )
        assert outputs["types_mod.pyx"] == (
            "from typing_extensions import TypeAlias\n\nColorAlias: TypeAlias = int\n"
        )
        assert outputs["consumer.pyx"] == (
            "from types_mod import ColorAlias as ColorAlias\n\n\ndef f(c: int): ...\n"
        )


class TestRemoveAssignmentFromModuleEnums:
    def test_finds_assignment_in_scope_enums(self):
        alias = PyiAssignment("Color: typing_extensions.TypeAlias = int", name="Color")
        real_enum = PyiEnum(enum_name="Shape", names=["SQ"])
        module = PyiModule(scope=PyiScope(enums=[real_enum, alias]))
        assert remove_assignment_from_module(module, alias) is True
        assert module.scope.enums == [real_enum]

    def test_falls_back_to_equal_assignment_when_not_the_same_object(self):
        survivor = PyiAssignment(
            "Color: typing_extensions.TypeAlias = int", name="Color"
        )
        twin = PyiAssignment("Color: typing_extensions.TypeAlias = int", name="Color")
        module = PyiModule(scope=PyiScope(assignments=[survivor]))
        assert remove_assignment_from_module(module, twin) is True
        assert module.scope.assignments == []

    def test_equality_fallback_ignores_a_different_enum_object(self):
        real_enum = PyiEnum(enum_name="Shape", names=["SQ"])
        alias = PyiAssignment("Color: typing_extensions.TypeAlias = int", name="Color")
        module = PyiModule(scope=PyiScope(enums=[real_enum]))
        assert remove_assignment_from_module(module, alias) is False
        assert module.scope.enums == [real_enum]
