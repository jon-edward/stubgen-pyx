"""Tests for project-specific generated symbol rewrites."""

from __future__ import annotations

import subprocess
import sys
from typing import TYPE_CHECKING

import pytest

from stubgen_pyx.config import StubgenPyxConfig, SymbolOverride
from stubgen_pyx.postprocessing.pipeline import postprocessing_pipeline
from stubgen_pyx.postprocessing.symbol_overrides import _module_name

if TYPE_CHECKING:
    from pathlib import Path


def _config(overrides: tuple[SymbolOverride, ...], tmp_path: Path) -> StubgenPyxConfig:
    return StubgenPyxConfig(
        exclude_attribution=True,
        sort_imports=False,
        module_root="pkg",
        source_root=tmp_path / "pkg",
        symbol_overrides=overrides,
    )


def _process(
    pyi_code: str, overrides: tuple[SymbolOverride, ...], tmp_path: Path
) -> str:
    return postprocessing_pipeline(
        pyi_code,
        _config(overrides, tmp_path),
        tmp_path / "pkg" / "models" / "widget.pyx",
    )


def test_import_override_rewrites_annotations_and_defaults(tmp_path):
    result = _process(
        """
from pkg.internal import thing_kind_t

def make(kind: thing_kind_t = thing_kind_t.SMALL) -> thing_kind_t: ...
""",
        (
            SymbolOverride(
                source="pkg.internal.thing_kind_t",
                import_target="pkg.public.ThingKind",
            ),
        ),
        tmp_path,
    )

    assert "from pkg.public import ThingKind" in result
    assert "thing_kind_t" not in result
    assert "def make(kind: ThingKind=ThingKind.SMALL) -> ThingKind" in result


def test_import_override_resolves_relative_imports(tmp_path):
    result = _process(
        """
from ..internal import thing_kind_t as local_kind

def make(kind: local_kind = local_kind.SMALL) -> local_kind: ...
""",
        (
            SymbolOverride(
                source="pkg.internal.thing_kind_t",
                import_target="pkg.public.ThingKind",
            ),
        ),
        tmp_path,
    )

    assert "from pkg.public import ThingKind" in result
    assert "local_kind" not in result
    assert "def make(kind: ThingKind=ThingKind.SMALL) -> ThingKind" in result


def test_import_override_resolves_module_alias_and_prefers_longest_match(
    tmp_path,
):
    result = _process(
        """
from pkg import internal as internal_module

def make(kind: internal_module.thing_kind_t = internal_module.thing_kind_t.DEFAULT) -> internal_module.thing_kind_t: ...
""",
        (
            SymbolOverride(
                source="pkg.internal.thing_kind_t",
                import_target="pkg.public.ThingKind",
            ),
            SymbolOverride(
                source="pkg.internal.thing_kind_t.DEFAULT",
                literal="...",
            ),
        ),
        tmp_path,
    )

    assert "from pkg.public import ThingKind" in result
    assert "internal_module" not in result
    assert "def make(kind: ThingKind=...) -> ThingKind" in result


def test_literal_override_replaces_type_and_trims_source_import(tmp_path):
    result = _process(
        """
from pkg.capi import size_type

def make(size: size_type) -> size_type: ...
""",
        (SymbolOverride(source="pkg.capi.size_type", literal="int"),),
        tmp_path,
    )

    assert "pkg.capi" not in result
    assert result == "def make(size: int) -> int: ..."


def test_literal_override_can_replace_default_with_ellipsis(tmp_path):
    result = _process(
        """
from pkg.capi import DEFAULT_SEED

def make(seed: int = DEFAULT_SEED) -> int: ...
""",
        (SymbolOverride(source="pkg.capi.DEFAULT_SEED", literal="..."),),
        tmp_path,
    )

    assert "pkg.capi" not in result
    assert result == "def make(seed: int=...) -> int: ..."


def test_drop_override_removes_top_level_alias_and_source_import(tmp_path):
    result = _process(
        """
from pkg.capi import widget_type as cpp_widget
from typing_extensions import TypeAlias

Widget: TypeAlias = cpp_widget
def make() -> int: ...
""",
        (SymbolOverride(source="pkg.capi.widget_type", action="drop"),),
        tmp_path,
    )

    assert "Widget" not in result
    assert "cpp_widget" not in result
    assert "TypeAlias" not in result
    assert result == "def make() -> int: ..."


def test_drop_override_rejects_public_signature_reference(tmp_path):
    with pytest.raises(ValueError, match=r"pkg\.capi\.widget_type"):
        _process(
            """
from pkg.capi import widget_type as cpp_widget

def make(widget: cpp_widget) -> int: ...
""",
            (SymbolOverride(source="pkg.capi.widget_type", action="drop"),),
            tmp_path,
        )


def test_drop_override_ignores_unrelated_assignments(tmp_path):
    result = _process(
        """
from pkg.capi import widget_type

value: int = 1
Widget = widget_type
""",
        (SymbolOverride(source="pkg.capi.widget_type", action="drop"),),
        tmp_path,
    )

    assert result == "value: int = 1"


def test_import_override_uses_existing_target_and_avoids_name_collisions(
    tmp_path,
):
    result = _process(
        """
from pkg.capi import widget_type
from pkg.public import Widget

class WidgetType: ...

def make(value: widget_type) -> widget_type: ...
""",
        (
            SymbolOverride(
                source="pkg.capi.widget_type",
                import_target="pkg.public.Widget",
            ),
        ),
        tmp_path,
    )

    assert result.count("from pkg.public import Widget") == 1
    assert "def make(value: Widget) -> Widget" in result


def test_import_override_aliases_a_colliding_target_name(tmp_path):
    result = _process(
        """
from pkg.capi import widget_type

Widget = object
def make(value: widget_type) -> widget_type: ...
""",
        (
            SymbolOverride(
                source="pkg.capi.widget_type",
                import_target="pkg.public.Widget",
            ),
        ),
        tmp_path,
    )

    assert "from pkg.public import Widget as _stubgen_pyx_Widget" in result
    assert "def make(value: _stubgen_pyx_Widget) -> _stubgen_pyx_Widget" in result


def test_relative_import_without_context_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="cannot resolve relative import"):
        postprocessing_pipeline(
            "from .capi import widget_type\n",
            StubgenPyxConfig(
                exclude_attribution=True,
                sort_imports=False,
                symbol_overrides=(
                    SymbolOverride(source="pkg.capi.widget_type", literal="int"),
                ),
            ),
            tmp_path / "widget.pyx",
        )


def test_module_name_handles_init_modules_and_missing_context(tmp_path):
    package = tmp_path / "pkg"
    init_file = package / "models" / "__init__.pyx"
    init_file.parent.mkdir(parents=True)
    init_file.touch()

    assert _module_name("pkg", package, init_file) == "pkg.models"
    assert _module_name(None, package, init_file) is None


def test_without_overrides_output_is_unchanged(tmp_path):
    pyi_code = "from pkg.capi import size_type\n\ndef make(size: size_type) -> size_type: ...\n"
    result = postprocessing_pipeline(
        pyi_code,
        StubgenPyxConfig(exclude_attribution=True, sort_imports=False),
        tmp_path / "pkg" / "widget.pyx",
    )

    assert (
        result
        == "from pkg.capi import size_type\ndef make(size: size_type) -> size_type: ..."
    )


def test_cli_loads_overrides_and_resolves_relative_imports(tmp_path):
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "__init__.py").touch()
    (package / "widget.pyx").write_text(
        """
from .internal cimport thing_kind_t

def make(thing_kind_t kind=thing_kind_t.SMALL):
    return kind
"""
    )
    overrides = tmp_path / "overrides.toml"
    overrides.write_text(
        """
module_root = "pkg"

[[symbol_overrides]]
source = "pkg.internal.thing_kind_t"
import = "pkg.public.ThingKind"
"""
    )
    output = tmp_path / "widget.pyi"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "stubgen_pyx",
            str(package),
            "--file",
            "widget.pyx",
            "--output-file",
            str(output),
            "--symbol-overrides",
            str(overrides),
            "--exclude-attribution",
            "--exclude-docstrings",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert output.read_text() == (
        "from pkg.public import ThingKind\n\n"
        "\ndef make(kind: ThingKind=ThingKind.SMALL): ...\n"
    )
