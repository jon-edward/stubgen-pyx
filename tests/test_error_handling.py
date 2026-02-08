"""Tests for error handling and edge cases in core modules."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch
import pytest

from stubgen_pyx.stubgen import StubgenPyx, ConversionResult
from stubgen_pyx.config import StubgenPyxConfig


class TestStubgenErrorHandling:
    """Test error handling in StubgenPyx."""

    def test_convert_file_with_encoding_error(self):
        """Test converting a file with encoding error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pyx_file = Path(tmpdir) / "bad_encoding.pyx"
            # Write file with invalid UTF-8
            pyx_file.write_bytes(b"\x80\x81\x82")

            config = StubgenPyxConfig(continue_on_error=True)
            stubgen = StubgenPyx(config=config)

            result = stubgen.convert_glob(str(pyx_file))
            assert result[0].success is False

    def test_convert_file_with_file_not_found(self):
        """Test converting a non-existent file."""
        config = StubgenPyxConfig(continue_on_error=True)
        stubgen = StubgenPyx(config=config)

        result = stubgen.convert_glob("__does_not_exist.pyx")
        assert not result  # empty list

    def test_convert_file_with_write_error(self):
        """Test when writing output file fails."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pyx_file = Path(tmpdir) / "test.pyx"
            pyx_file.write_text("def hello(): pass")

            # Create a mock output dir that can't be written to
            with patch.object(
                Path, "write_text", side_effect=IOError("Permission denied")
            ):
                config = StubgenPyxConfig(continue_on_error=True)
                stubgen = StubgenPyx(config=config)

                result = stubgen.convert_glob(str(pyx_file))
                assert result[0].success is False

    def test_convert_glob_with_error_in_middle(self):
        """Test glob conversion when one file fails."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Create valid file
            valid_file = tmppath / "valid.pyx"
            valid_file.write_text("def hello(): pass")

            # Create file with bad encoding
            bad_file = tmppath / "bad.pyx"
            bad_file.write_bytes(b"\x80\x81\x82")

            # Create another valid file
            valid_file2 = tmppath / "valid2.pyx"
            valid_file2.write_text("def goodbye(): pass")

            config = StubgenPyxConfig(continue_on_error=True)
            stubgen = StubgenPyx(config=config)

            results = stubgen.convert_glob(str(tmppath / "*.pyx"))

            # Should have results for all files
            assert len(results) >= 2
            successful = [r for r in results if r.success]
            failed = [r for r in results if not r.success]
            assert len(successful) >= 2
            assert len(failed) >= 1

    def test_convert_file_without_error_continuation(self):
        """Test that conversion stops on error when continue_on_error is False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pyx_file = Path(tmpdir) / "bad_encoding.pyx"
            pyx_file.write_bytes(b"\x80\x81\x82")

            config = StubgenPyxConfig(continue_on_error=False)
            stubgen = StubgenPyx(config=config)

            with pytest.raises(ValueError):
                stubgen.convert_glob(str(pyx_file))

    def test_conversion_result_creation(self):
        """Test ConversionResult dataclass."""
        pyx_file = Path("test.pyx")
        pyi_file = Path("test.pyi")
        result = ConversionResult(pyx_file=pyx_file, pyi_file=pyi_file, success=True)
        assert result.pyx_file == pyx_file
        assert result.pyi_file == pyi_file
        assert result.success is True

    def test_convert_file_with_pxd_file_encoding_error(self):
        """Test when .pxd file has encoding error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Create valid .pyx file
            pyx_file = tmppath / "test.pyx"
            pyx_file.write_text("def hello(): pass")

            # Create .pxd file with bad encoding
            pxd_file = tmppath / "test.pxd"
            pxd_file.write_bytes(b"\x80\x81\x82")

            config = StubgenPyxConfig(continue_on_error=True, no_pxd_to_stubs=False)
            stubgen = StubgenPyx(config=config)

            result = stubgen.convert_glob(str(tmppath / "*.pyx"))
            assert isinstance(result[0], ConversionResult)

    def test_convert_glob_with_no_files_no_error(self):
        """Test glob with no matches doesn't error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = StubgenPyxConfig()
            stubgen = StubgenPyx(config=config)

            results = stubgen.convert_glob(str(Path(tmpdir) / "*.pyx"))
            assert results == []

    def test_stubgen_initialization_with_config(self):
        """Test StubgenPyx initialization with custom config."""
        config = StubgenPyxConfig(
            verbose=True, no_sort_imports=True, include_private=True
        )
        stubgen = StubgenPyx(config=config)
        assert stubgen.config.verbose is True
        assert stubgen.config.no_sort_imports is True
        assert stubgen.config.include_private is True

    def test_stubgen_default_config(self):
        """Test StubgenPyx with default config."""
        stubgen = StubgenPyx()
        assert stubgen.config is not None

    def test_convert_str_empty_code(self):
        """Test converting empty code."""
        stubgen = StubgenPyx()
        result = stubgen.convert_str("")
        assert isinstance(result, str)

    def test_convert_str_only_comments(self):
        """Test converting code with only comments."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pyx_file = Path(tmpdir) / "test.pyx"
            code = "# This is just a comment\n# Another comment"
            result = StubgenPyx().convert_str(code, pyx_path=pyx_file)
            assert isinstance(result, str)

    def test_convert_file_with_output_dir(self):
        """Test converting file with custom output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            source_dir = tmppath / "source"
            source_dir.mkdir()
            output_dir = tmppath / "output"
            output_dir.mkdir()

            pyx_file = source_dir / "test.pyx"
            pyx_file.write_text("def hello(): pass")

            stubgen = StubgenPyx()
            (result,) = stubgen.convert_glob(str(pyx_file), output_dir)

            assert result.success is True
            assert (output_dir / "test.pyi").exists()


class TestConversionResultProperties:
    """Test ConversionResult properties."""

    def test_result_success_true(self):
        """Test result with success=True."""
        result = ConversionResult(
            pyx_file=Path("test.pyx"), success=True, pyi_file=Path("test.pyi")
        )
        assert result.success is True
        assert not hasattr(result, "error") or result.error is None

    def test_result_success_false_with_error(self):
        """Test result with success=False and error message."""
        result = ConversionResult(
            pyx_file=Path("test.pyx"),
            pyi_file=Path("test.pyi"),
            success=False,
            error=SyntaxError("Syntax error"),
        )
        assert result.success is False
        assert isinstance(result.error, SyntaxError)

    def test_result_with_pyi_file(self):
        """Test result with both pyx and pyi files."""
        pyx_file = Path("test.pyx")
        pyi_file = Path("test.pyi")
        result = ConversionResult(pyx_file=pyx_file, pyi_file=pyi_file, success=True)
        assert result.pyx_file == pyx_file
        assert result.pyi_file == pyi_file
