"""Tests for configuration validation."""

from __future__ import annotations

import logging
import pytest

from stubgen_pyx.config import StubgenPyxConfig


def test_config_defaults():
    """Test that default configuration values are correct."""
    config = StubgenPyxConfig()
    assert config.no_sort_imports is False
    assert config.no_trim_imports is False
    assert config.no_pxd_to_stubs is False
    assert config.no_normalize_names is False
    assert config.no_deduplicate_imports is False
    assert config.exclude_epilog is False
    assert config.continue_on_error is False
    assert config.verbose is False


def test_config_custom_values():
    """Test that custom configuration values are set correctly."""
    config = StubgenPyxConfig(
        no_sort_imports=True,
        no_trim_imports=True,
        continue_on_error=True,
        verbose=True,
    )
    assert config.no_sort_imports is True
    assert config.no_trim_imports is True
    assert config.continue_on_error is True
    assert config.verbose is True


def test_config_post_init_warning_all_disabled(caplog):
    """Test that warning is logged when all postprocessing is disabled."""
    with caplog.at_level(logging.WARNING):
        config = StubgenPyxConfig(
            no_sort_imports=True,
            no_trim_imports=True,
            no_normalize_names=True,
            no_deduplicate_imports=True,
        )
    assert "All postprocessing steps are disabled" in caplog.text


def test_config_post_init_info_continue_on_error(caplog):
    """Test that info is logged when continue_on_error is enabled."""
    with caplog.at_level(logging.INFO):
        config = StubgenPyxConfig(continue_on_error=True)
    assert "Continuing on errors" in caplog.text
