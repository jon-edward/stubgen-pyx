"""Main entry point for stubgen-pyx."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .stubgen import ConversionResult, StubgenPyx
from .config import StubgenPyxConfig
from ._version import __version__

logger = logging.getLogger(__name__)


def _create_parser() -> argparse.ArgumentParser:
    """Create the argument parser with all options."""
    parser = argparse.ArgumentParser(
        description="Generate Python stub files (.pyi) from Cython source code (.pyx/.pxd)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s .                         # Convert all .pyx files in current directory
  %(prog)s src/ --file "**/*.pyx"    # Convert all .pyx files in src/
  %(prog)s . --output-dir stubs/     # Write stubs to stubs/ directory
  %(prog)s . --dry-run               # Preview conversions without writing
  %(prog)s . --verbose               # Show detailed processing information
  %(prog)s src/ --file in.pyx --output-file out/out.pyi  # Convert a single pyx
                                                         # file to an output file
        """,
    )

    parser.add_argument(
        "--version",
        "-V",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    parser.add_argument(
        "dir",
        help="Directory to search for Cython modules",
        type=str,
        default=".",
        nargs="?",
    )

    parser.add_argument(
        "--file",
        help="Glob pattern for files to generate stubs for (default: **/*.pyx)",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--output-dir",
        help="Directory to write .pyi files (default: same as source). "
        "This option cannot be used with option '--output-file'.",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--output-file",
        help="Path to a .pyi file. Use to generate a specific file. "
        "To use this option, the list of matching input pyx files "
        "must contain a single file. "
        "This option cannot be used with option '--output-dir'.",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--verbose",
        "-v",
        help="Enable verbose logging",
        action="store_true",
    )

    parser.add_argument(
        "--dry-run",
        help="Preview conversions without writing files",
        action="store_true",
    )

    parser.add_argument(
        "--no-sort-imports",
        help="Disable sorting of imports",
        action="store_true",
    )

    parser.add_argument(
        "--no-trim-imports",
        help="Disable trimming of unused imports",
        action="store_true",
    )

    parser.add_argument(
        "--no-normalize-names",
        help="Disable normalization of Cython type names to Python equivalents",
        action="store_true",
    )

    parser.add_argument(
        "--no-pxd-to-stubs",
        help="Disable inclusion of .pxd file contents in stubs",
        action="store_true",
    )

    parser.add_argument(
        "--no-deduplicate-imports",
        help="Do not deduplicate imports in the output stub",
        action="store_true",
    )

    parser.add_argument(
        "--exclude-epilog",
        help="Disable inclusion of epilog comment",
        action="store_true",
    )

    parser.add_argument(
        "--continue-on-error",
        help="Continue processing even if a file fails to convert",
        action="store_true",
    )

    parser.add_argument(
        "--include-private",
        help="Include private functions (starting with _ and not ending with _) in the output stub",
        action="store_true",
    )

    return parser


def _create_dir(path: Path, dry_run: bool = False):
    """Create given directory if not already existing.

    dry_run controls the actual creation.
    """
    if not path.exists():
        if dry_run:
            logger.info(f"Would create output directory: {path}")
        else:
            path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created output directory: {path}")


def _parse_args() -> argparse.Namespace | None:
    parser = _create_parser()
    args = parser.parse_args()

    # check the arguments
    if args.output_dir is not None and args.output_file is not None:
        logger.error(
            "Error: options '--output-dir' and '--output-file' cannot be used together"
        )
        return None

    return args


def main():
    """Main entry point for stubgen-pyx."""

    args = _parse_args()
    if args is None:
        sys.exit(1)

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    logger.info(f"stubgen-pyx v{__version__}")

    # Create configuration
    config = StubgenPyxConfig(
        no_sort_imports=args.no_sort_imports,
        no_trim_imports=args.no_trim_imports,
        no_normalize_names=args.no_normalize_names,
        no_pxd_to_stubs=args.no_pxd_to_stubs,
        exclude_epilog=args.exclude_epilog,
        no_deduplicate_imports=args.no_deduplicate_imports,
        continue_on_error=args.continue_on_error,
        include_private=args.include_private,
        verbose=args.verbose,
    )

    # Build file pattern
    source_dir = Path(args.dir) if args.dir else Path(".")
    if args.file:
        pyx_file_pattern = str(source_dir / args.file)
    else:
        pyx_file_pattern = str(source_dir / "**" / "*.pyx")
    logger.debug(f"Using pattern: {pyx_file_pattern}")

    # Create converter and run
    stubgen = StubgenPyx(config=config)

    # Check single-file mode if requested
    pyx_files = tuple(stubgen.resolve_glob(pyx_file_pattern))
    if args.output_file is not None and ((_num := len(pyx_files)) != 1):
        logger.error(
            "Option --output-file requires a single input pyx file in "
            f"'{pyx_file_pattern}': {_num} found"
        )
        sys.exit(1)

    # Validate output directory
    output_dir = None
    if args.output_dir:
        output_dir = Path(args.output_dir)
        _create_dir(output_dir, args.dry_run)
    elif args.output_file:
        output_dir = Path(args.output_file).parent
        _create_dir(output_dir, args.dry_run)

    if args.dry_run:
        logger.info("DRY RUN MODE - no files will be written")

    results: list[ConversionResult]
    if args.output_file is None:
        # multi-files input
        results = stubgen.convert_multiple_files(
            pyx_files, output_dir=output_dir, dry_run=args.dry_run
        )
    else:
        # single-file input
        assert len(pyx_files) == 1
        results = [
            stubgen.convert_single_file(
                pyx_files[0], args.output_file, dry_run=args.dry_run
            )
        ]

    # Summary reporting
    successful_count = sum(1 for r in results if r.success)
    logger.info(f"Successfully converted {successful_count} file(s)")

    # Exit with appropriate code
    failed_count = sum(1 for r in results if not r.success)
    if failed_count > 0:
        logger.error(f"{failed_count} file(s) failed to convert")
        sys.exit(1)

    if not results:
        logger.error(f"No .pyx files found matching pattern: {pyx_file_pattern}")
        sys.exit(1)

    sys.exit(0)
