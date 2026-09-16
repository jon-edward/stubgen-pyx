"""Tests for project-specific generated declaration replacements."""

from __future__ import annotations

import pytest

from stubgen_pyx.config import DeclarationOverride, StubgenPyxConfig
from stubgen_pyx.postprocessing.pipeline import postprocessing_pipeline


def _config(overrides: tuple[DeclarationOverride, ...], tmp_path):
    return StubgenPyxConfig(
        exclude_attribution=True,
        sort_imports=False,
        module_root="pkg",
        source_root=tmp_path / "pkg",
        declaration_overrides=overrides,
    )


def _process(pyi_code: str, overrides: tuple[DeclarationOverride, ...], tmp_path):
    return postprocessing_pipeline(
        pyi_code,
        _config(overrides, tmp_path),
        tmp_path / "pkg" / "models" / "widget.pyx",
    )


def test_declaration_override_replaces_function_return(tmp_path):
    result = _process(
        "def make() -> object: ...\n",
        (
            DeclarationOverride(
                target="pkg.models.widget.make",
                declarations="def make() -> int: ...",
            ),
        ),
        tmp_path,
    )

    assert result == "def make() -> int: ..."


def test_declaration_override_replaces_class_method(tmp_path):
    result = _process(
        """
class Widget:
    def value(self) -> object: ...
""",
        (
            DeclarationOverride(
                target="pkg.models.widget.Widget.value",
                declarations="def value(self) -> str: ...",
            ),
        ),
        tmp_path,
    )

    assert "def value(self) -> str: ..." in result


def test_declaration_override_replaces_function_with_overloads(tmp_path):
    result = _process(
        """
from .column import Column
from .table import Table

def split(input: Table | Column) -> list[Table | Column]: ...
""",
        (
            DeclarationOverride(
                target="pkg.models.widget.split",
                declarations="""
@typing.overload
def split(input: Table) -> list[Table]: ...

@typing.overload
def split(input: Column) -> list[Column]: ...
""",
            ),
        ),
        tmp_path,
    )

    assert "from typing import overload" in result
    assert "@overload\ndef split(input: Table) -> list[Table]: ..." in result
    assert "@overload\ndef split(input: Column) -> list[Column]: ..." in result


def test_declaration_override_ignores_other_modules(tmp_path):
    result = _process(
        "def value() -> int: ...\n",
        (
            DeclarationOverride(
                target="pkg.other.value",
                declarations="def value() -> str: ...",
            ),
        ),
        tmp_path,
    )

    assert "def value() -> int: ..." in result


def test_declaration_override_adds_literal_import(tmp_path):
    result = _process(
        "def names(include_children: bool=False) -> list[object]: ...\n",
        (
            DeclarationOverride(
                target="pkg.models.widget.names",
                declarations=(
                    "def names(self, include_children: typing.Literal[False] = False) "
                    "-> list[str]: ..."
                ),
            ),
        ),
        tmp_path,
    )

    assert "from typing import Literal" in result
    assert "typing.Literal" not in result
    assert "Literal[False]" in result


def test_declaration_override_configuration_is_validated():
    with pytest.raises(ValueError, match="must contain only function"):
        DeclarationOverride(
            target="pkg.models.widget.make",
            declarations="value = 1",
        )
    with pytest.raises(ValueError, match="must match target"):
        DeclarationOverride(
            target="pkg.models.widget.make",
            declarations="def other() -> int: ...",
        )
    with pytest.raises(ValueError, match=r"typing\.overload"):
        DeclarationOverride(
            target="pkg.models.widget.make",
            declarations="def make() -> int: ...\ndef make() -> str: ...",
        )


def test_declaration_override_requires_generated_target(tmp_path):
    with pytest.raises(ValueError, match="did not match"):
        _process(
            "def make() -> object: ...\n",
            (
                DeclarationOverride(
                    target="pkg.models.widget.missing",
                    declarations="def missing() -> int: ...",
                ),
            ),
            tmp_path,
        )
