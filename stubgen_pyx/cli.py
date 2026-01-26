"""Main entry point for stubgen-pyx."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .stubgen import StubgenPyx
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
  %(prog)s .                        # Convert all .pyx files in current directory
  %(prog)s src/ --file "**/*.pyx"   # Convert all .pyx files in src/
  %(prog)s . --output-dir stubs/    # Write stubs to stubs/ directory
  %(prog)s . --dry-run              # Preview conversions without writing
  %(prog)s . --verbose              # Show detailed processing information
        """,
    )

    parser.add_argument(
        "--version",
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
        help="Directory to write .pyi files (default: same as source)",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--verbose",
        "-v",
        help="Enable verbose logging",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--dry-run",
        help="Preview conversions without writing files",
        action="store_true",
        default=False,
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

    return parser


def main():
    """Main entry point for stubgen-pyx."""
    parser = _create_parser()
    args = parser.parse_args()

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
        verbose=args.verbose,
    )

    # Validate output directory
    output_dir = None
    if args.output_dir:
        output_dir = Path(args.output_dir)
        if not output_dir.exists():
            if args.dry_run:
                logger.info(f"Would create output directory: {output_dir}")
            else:
                output_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created output directory: {output_dir}")

    # Build file pattern
    source_dir = Path(args.dir) if args.dir else Path(".")
    if args.file:
        pyx_file_pattern = str(source_dir / args.file)
    else:
        pyx_file_pattern = str(source_dir / "**" / "*.pyx")

    logger.debug(f"Using pattern: {pyx_file_pattern}")

    # Create converter and run
    stubgen = StubgenPyx(config=config)

    if args.dry_run:
        logger.info("DRY RUN MODE - no files will be written")

    results = stubgen.convert_glob(pyx_file_pattern)

    # Handle output directory relocation if specified
    if output_dir and not args.dry_run:
        logger.info(f"Moving .pyi files to {output_dir}")
        for result in results:
            if result.success:
                result.pyi_file.rename(output_dir / result.pyi_file.name)

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
