"""Tests for CLI module."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

from stubgen_pyx import cli


class TestCreateParser:
    """Test the argument parser creation."""

    def test_parser_creation(self):
        """Test that parser is created successfully."""
        parser = cli._create_parser()
        assert parser is not None

    def test_parser_with_version(self):
        """Test parser with --version flag."""
        parser = cli._create_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--version"])
        assert exc_info.value.code == 0

    def test_parser_with_directory_argument(self):
        """Test parser with directory argument."""
        parser = cli._create_parser()
        args = parser.parse_args(["."])
        assert args.dir == "."

    def test_parser_with_file_pattern(self):
        """Test parser with --file pattern."""
        parser = cli._create_parser()
        args = parser.parse_args([".", "--file", "**/*.pyx"])
        assert args.file == "**/*.pyx"

    def test_parser_with_output_dir(self):
        """Test parser with --output-dir."""
        parser = cli._create_parser()
        args = parser.parse_args([".", "--output-dir", "stubs/"])
        assert args.output_dir == Path("stubs/")

    def test_parser_with_verbose_flag(self):
        """Test parser with --verbose flag."""
        parser = cli._create_parser()
        args = parser.parse_args([".", "-v"])
        assert args.verbose is True

    def test_parser_with_dry_run_flag(self):
        """Test parser with --dry-run flag."""
        parser = cli._create_parser()
        args = parser.parse_args([".", "--dry-run"])
        assert args.dry_run is True

    def test_parser_with_no_sort_imports(self):
        """Test parser with --no-sort-imports flag."""
        parser = cli._create_parser()
        args = parser.parse_args([".", "--no-sort-imports"])
        assert args.no_sort_imports is True

    def test_parser_with_no_trim_imports(self):
        """Test parser with --no-trim-imports flag."""
        parser = cli._create_parser()
        args = parser.parse_args([".", "--no-trim-imports"])
        assert args.no_trim_imports is True

    def test_parser_with_no_normalize_names(self):
        """Test parser with --no-normalize-names flag."""
        parser = cli._create_parser()
        args = parser.parse_args([".", "--no-normalize-names"])
        assert args.no_normalize_names is True

    def test_parser_with_no_pxd_to_stubs(self):
        """Test parser with --no-pxd-to-stubs flag."""
        parser = cli._create_parser()
        args = parser.parse_args([".", "--no-pxd-to-stubs"])
        assert args.no_pxd_to_stubs is True

    def test_parser_with_no_deduplicate_imports(self):
        """Test parser with --no-deduplicate-imports flag."""
        parser = cli._create_parser()
        args = parser.parse_args([".", "--no-deduplicate-imports"])
        assert args.no_deduplicate_imports is True

    def test_parser_with_exclude_epilog(self):
        """Test parser with --exclude-epilog flag."""
        parser = cli._create_parser()
        args = parser.parse_args([".", "--exclude-epilog"])
        assert args.exclude_epilog is True

    def test_parser_with_continue_on_error(self):
        """Test parser with --continue-on-error flag."""
        parser = cli._create_parser()
        args = parser.parse_args([".", "--continue-on-error"])
        assert args.continue_on_error is True

    def test_parser_with_include_private(self):
        """Test parser with --include-private flag."""
        parser = cli._create_parser()
        args = parser.parse_args([".", "--include-private"])
        assert args.include_private is True

    def test_parser_default_directory(self):
        """Test parser with default directory."""
        parser = cli._create_parser()
        args = parser.parse_args([])
        assert args.dir == "."

    def test_parser_all_flags_combined(self):
        """Test parser with multiple flags."""
        parser = cli._create_parser()
        args = parser.parse_args(
            [
                "src/",
                "--file",
                "**/*.pyx",
                "--output-dir",
                "stubs/",
                "-v",
                "--dry-run",
                "--no-sort-imports",
                "--no-trim-imports",
                "--continue-on-error",
            ]
        )
        assert args.dir == "src/"
        assert args.file == "**/*.pyx"
        assert args.output_dir == Path("stubs/")
        assert args.verbose is True
        assert args.dry_run is True
        assert args.no_sort_imports is True
        assert args.no_trim_imports is True
        assert args.continue_on_error is True


class TestMain:
    """Test the main function."""

    @patch("stubgen_pyx.cli.StubgenPyx")
    @patch("stubgen_pyx.cli.logging.basicConfig")
    def test_main_basic_success(self, mock_logging, mock_stubgen_class):
        """Test main function with basic successful conversion."""
        mock_stubgen = MagicMock()
        mock_stubgen_class.return_value = mock_stubgen
        mock_result = MagicMock()
        mock_result.success = True
        mock_stubgen.convert_glob.return_value = [mock_result]

        with patch.object(sys, "argv", ["stubgen-pyx", "."]):
            with pytest.raises(SystemExit) as exc_info:
                cli.main()
            assert exc_info.value.code == 0

    @patch("stubgen_pyx.cli.StubgenPyx")
    @patch("stubgen_pyx.cli.logging.basicConfig")
    def test_main_with_verbose_logging(self, mock_logging, mock_stubgen_class):
        """Test main function with verbose logging."""
        mock_stubgen = MagicMock()
        mock_stubgen_class.return_value = mock_stubgen
        mock_result = MagicMock()
        mock_result.success = True
        mock_stubgen.convert_glob.return_value = [mock_result]

        with patch.object(sys, "argv", ["stubgen-pyx", ".", "-v"]):
            with pytest.raises(SystemExit) as exc_info:
                cli.main()
            assert exc_info.value.code == 0
            # Verify DEBUG logging was configured
            call_args = mock_logging.call_args
            assert call_args[1]["level"] == logging.DEBUG

    @patch("stubgen_pyx.cli.StubgenPyx")
    @patch("stubgen_pyx.cli.logging.basicConfig")
    def test_main_with_output_dir_creation(self, mock_logging, mock_stubgen_class):
        """Test main function creates output directory if needed."""
        import tempfile

        mock_stubgen = MagicMock()
        mock_stubgen_class.return_value = mock_stubgen
        mock_result = MagicMock()
        mock_result.success = True
        mock_stubgen.convert_glob.return_value = [mock_result]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "new_stubs"
            with patch.object(
                sys, "argv", ["stubgen-pyx", ".", "--output-dir", str(output_dir)]
            ):
                with pytest.raises(SystemExit) as exc_info:
                    cli.main()
                assert exc_info.value.code == 0
                assert output_dir.exists()

    @patch("stubgen_pyx.cli.StubgenPyx")
    @patch("stubgen_pyx.cli.logging.basicConfig")
    def test_main_with_dry_run(self, mock_logging, mock_stubgen_class):
        """Test main function with dry-run mode."""
        mock_stubgen = MagicMock()
        mock_stubgen_class.return_value = mock_stubgen
        mock_result = MagicMock()
        mock_result.success = True
        mock_stubgen.convert_glob.return_value = [mock_result]

        with patch.object(sys, "argv", ["stubgen-pyx", ".", "--dry-run"]):
            with pytest.raises(SystemExit) as exc_info:
                cli.main()
            assert exc_info.value.code == 0

    @patch("stubgen_pyx.cli.StubgenPyx")
    @patch("stubgen_pyx.cli.logging.basicConfig")
    def test_main_with_failed_conversions(self, mock_logging, mock_stubgen_class):
        """Test main function when conversions fail."""
        mock_stubgen = MagicMock()
        mock_stubgen_class.return_value = mock_stubgen
        mock_result = MagicMock()
        mock_result.success = False
        mock_stubgen.convert_glob.return_value = [mock_result]

        with patch.object(sys, "argv", ["stubgen-pyx", "."]):
            with pytest.raises(SystemExit) as exc_info:
                cli.main()
            assert exc_info.value.code == 1

    @patch("stubgen_pyx.cli.StubgenPyx")
    @patch("stubgen_pyx.cli.logging.basicConfig")
    def test_main_no_files_found(self, mock_logging, mock_stubgen_class):
        """Test main function when no files are found."""
        mock_stubgen = MagicMock()
        mock_stubgen_class.return_value = mock_stubgen
        mock_stubgen.convert_glob.return_value = []

        with patch.object(sys, "argv", ["stubgen-pyx", "."]):
            with pytest.raises(SystemExit) as exc_info:
                cli.main()
            assert exc_info.value.code == 1

    @patch("stubgen_pyx.cli.StubgenPyx")
    @patch("stubgen_pyx.cli.logging.basicConfig")
    def test_main_mixed_results(self, mock_logging, mock_stubgen_class):
        """Test main function with mixed success/failure results."""
        mock_stubgen = MagicMock()
        mock_stubgen_class.return_value = mock_stubgen
        mock_result1 = MagicMock()
        mock_result1.success = True
        mock_result2 = MagicMock()
        mock_result2.success = False
        mock_stubgen.convert_glob.return_value = [mock_result1, mock_result2]

        with patch.object(sys, "argv", ["stubgen-pyx", "."]):
            with pytest.raises(SystemExit) as exc_info:
                cli.main()
            assert exc_info.value.code == 1

    @patch("stubgen_pyx.cli.StubgenPyx")
    @patch("stubgen_pyx.cli.logging.basicConfig")
    def test_main_with_custom_file_pattern(self, mock_logging, mock_stubgen_class):
        """Test main function with custom file pattern."""
        mock_stubgen = MagicMock()
        mock_stubgen_class.return_value = mock_stubgen
        mock_result = MagicMock()
        mock_result.success = True
        mock_stubgen.convert_glob.return_value = [mock_result]

        with patch.object(sys, "argv", ["stubgen-pyx", "src/", "--file", "*.pyx"]):
            with pytest.raises(SystemExit) as exc_info:
                cli.main()
            assert exc_info.value.code == 0
