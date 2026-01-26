"""Tests for conversion results and error handling."""

from __future__ import annotations

from pathlib import Path

import pytest

from stubgen_pyx.stubgen import ConversionResult


def test_conversion_result_success():
    """Test ConversionResult with successful conversion."""
    pyx_file = Path("test.pyx")
    pyi_file = Path("test.pyi")

    result = ConversionResult(success=True, pyx_file=pyx_file, pyi_file=pyi_file)

    assert result.success is True
    assert result.pyx_file == pyx_file
    assert result.pyi_file == pyi_file
    assert result.error is None
    assert "Converted" in result.status_message


def test_conversion_result_failure():
    """Test ConversionResult with failed conversion."""
    pyx_file = Path("test.pyx")
    pyi_file = Path("test.pyi")
    error = ValueError("Test error")

    result = ConversionResult(
        success=False, pyx_file=pyx_file, pyi_file=pyi_file, error=error
    )

    assert result.success is False
    assert result.pyx_file == pyx_file
    assert result.pyi_file == pyi_file
    assert result.error == error
    assert "Failed to convert" in result.status_message
    assert "Test error" in result.status_message


def test_conversion_result_status_message_success():
    """Test status message for successful conversion."""
    result = ConversionResult(
        success=True, pyx_file=Path("a.pyx"), pyi_file=Path("a.pyi")
    )
    assert "Converted" in result.status_message


def test_conversion_result_status_message_failure():
    """Test status message for failed conversion."""
    error = RuntimeError("Parse failed")
    result = ConversionResult(
        success=False, pyx_file=Path("b.pyx"), pyi_file=Path("b.pyi"), error=error
    )
    assert "Failed to convert" in result.status_message
    assert "Parse failed" in result.status_message
