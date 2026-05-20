"""Tests for configuration validation."""

from __future__ import annotations

import logging

from stubgen_pyx.config import StubgenPyxConfig


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
