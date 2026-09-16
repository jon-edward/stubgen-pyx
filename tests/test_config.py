"""Tests for configuration validation."""

from __future__ import annotations

import logging

import pytest

from stubgen_pyx.config import (
    DeclarationOverride,
    StubgenPyxConfig,
    SymbolOverride,
    SymbolOverridesConfig,
    load_symbol_overrides,
)


def test_config_defaults():
    """Test that default configuration values are correct."""
    config = StubgenPyxConfig()
    assert config.sort_imports is True
    assert config.trim_imports is True
    assert config.pxd_to_stubs is True
    assert config.normalize_names is True
    assert config.deduplicate_imports is True
    assert config.trim_not_defined is True
    assert config.exclude_attribution is False
    assert config.continue_on_error is False
    assert config.verbose is False


def test_config_post_init_warning_all_disabled(caplog):
    """Test that warning is logged when all postprocessing is disabled."""
    with caplog.at_level(logging.WARNING):
        StubgenPyxConfig(
            sort_imports=False,
            trim_imports=False,
            normalize_names=False,
            deduplicate_imports=False,
            trim_not_defined=False,
        )
    assert "All postprocessing steps are disabled" in caplog.text


def test_config_post_init_info_continue_on_error(caplog):
    """Test that info is logged when continue_on_error is enabled."""
    with caplog.at_level(logging.INFO):
        StubgenPyxConfig(continue_on_error=True)
    assert "Continuing on errors" in caplog.text


def test_load_symbol_overrides(tmp_path):
    config_file = tmp_path / "overrides.toml"
    config_file.write_text(
        """
module_root = "pkg"

[[symbol_overrides]]
source = "pkg.internal.thing_kind_t"
import = "pkg.public.ThingKind"

[[symbol_overrides]]
source = "pkg.capi.size_type"
literal = "int"

[[declaration_overrides]]
target = "pkg.widget.make"
declarations = "def make() -> int: ..."
"""
    )

    config = load_symbol_overrides(config_file)

    assert config.module_root == "pkg"
    assert config.overrides[0].import_target == "pkg.public.ThingKind"
    assert config.overrides[1].literal == "int"
    assert config.declaration_overrides[0].target == "pkg.widget.make"


@pytest.mark.parametrize(
    "content, message",
    [
        ('module_root = "not-valid"', "module_root"),
        (
            """module_root = "pkg"
[[symbol_overrides]]
source = "pkg.a"
literal = "int"
action = "drop"
""",
            "exactly one",
        ),
        (
            """module_root = "pkg"
[[symbol_overrides]]
source = "pkg.a"
action = "replace"
""",
            "action must be 'drop'",
        ),
        (
            """module_root = "pkg"
[[declaration_overrides]]
target = "pkg.widget.make"
declarations = "value = 1"
""",
            "must contain only function",
        ),
    ],
)
def test_load_symbol_overrides_rejects_invalid_configuration(
    tmp_path, content, message
):
    config_file = tmp_path / "overrides.toml"
    config_file.write_text(content)

    with pytest.raises(ValueError, match=message):
        load_symbol_overrides(config_file)


@pytest.mark.parametrize(
    "content, message",
    [
        ('module_root = ["pkg"]', "string module_root"),
        ('module_root = "pkg"\nsymbol_overrides = "not a list"', "array"),
        (
            'module_root = "pkg"\nsymbol_overrides = ["not a table"]',
            "must be a table",
        ),
        (
            'module_root = "pkg"\ndeclaration_overrides = "not a list"',
            "array",
        ),
        (
            'module_root = "pkg"\ndeclaration_overrides = ["not a table"]',
            "must be a table",
        ),
    ],
)
def test_load_symbol_overrides_rejects_invalid_toml_shapes(tmp_path, content, message):
    config_file = tmp_path / "overrides.toml"
    config_file.write_text(content)

    with pytest.raises(TypeError, match=message):
        load_symbol_overrides(config_file)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"source": "pkg.kind", "import_target": "Kind"},
        {"source": "pkg.kind", "literal": "not valid +"},
        {"source": "pkg.kind", "action": "replace"},
    ],
)
def test_symbol_override_validates_values(kwargs):
    with pytest.raises(ValueError):
        SymbolOverride(**kwargs)


def test_declaration_override_rejects_invalid_python():
    with pytest.raises(ValueError, match="not valid Python"):
        DeclarationOverride(
            target="pkg.widget.make",
            declarations="def make(: ...",
        )


def test_symbol_overrides_config_rejects_duplicate_rules():
    symbol = SymbolOverride(source="pkg.kind", literal="int")
    declaration = DeclarationOverride(
        target="pkg.widget.make", declarations="def make() -> int: ..."
    )
    with pytest.raises(ValueError, match="sources must be unique"):
        SymbolOverridesConfig("pkg", (symbol, symbol), ())
    with pytest.raises(ValueError, match="targets must be unique"):
        SymbolOverridesConfig("pkg", (), (declaration, declaration))


@pytest.mark.parametrize(
    "content, message",
    [
        (
            'module_root = "pkg"\n[[symbol_overrides]]\nsource = "pkg.kind"\nextra = "x"',
            "unexpected keys",
        ),
        (
            'module_root = "pkg"\n[[symbol_overrides]]\nliteral = "int"',
            "must define source",
        ),
        (
            'module_root = "pkg"\n[[declaration_overrides]]\ntarget = "pkg.widget.make"\nextra = "x"',
            "unexpected keys",
        ),
        (
            'module_root = "pkg"\n[[declaration_overrides]]\ntarget = "pkg.widget.make"',
            "must define target and declarations",
        ),
    ],
)
def test_load_symbol_overrides_rejects_invalid_override_tables(
    tmp_path, content, message
):
    config_file = tmp_path / "overrides.toml"
    config_file.write_text(content)

    with pytest.raises(ValueError, match=message):
        load_symbol_overrides(config_file)


def test_load_symbol_overrides_reports_missing_file(tmp_path):
    with pytest.raises(ValueError, match="could not read"):
        load_symbol_overrides(tmp_path / "missing.toml")


def test_load_symbol_overrides_reports_invalid_toml(tmp_path):
    config_file = tmp_path / "overrides.toml"
    config_file.write_text("not = [valid")

    with pytest.raises(ValueError, match="could not parse"):
        load_symbol_overrides(config_file)
