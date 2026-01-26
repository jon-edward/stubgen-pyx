"""Main entry point for stubgen-pyx."""

from __future__ import annotations

import argparse
import logging
import os

from .stubgen import StubgenPyx
from .config import StubgenPyxConfig

logger = logging.getLogger(__name__)


def main():
    """Main entry point for stubgen-pyx."""
    parser = argparse.ArgumentParser(description="Generate stubs for Cython modules")
    parser.add_argument(
        "dir", help="Directory to search for Cython modules", type=str, default="."
    )

    parser.add_argument(
        "--file", help="Glob pattern for files to generate stubs for", type=str, default=None
    )

    parser.add_argument(
        "--verbose", help="Enable verbose logging", action="store_true", default=False
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
        help="Disable normalization of names",
        action="store_true",
    )

    parser.add_argument(
        "--no-pxd-to-stubs",
        help="Disable inclusion of .pxd file contents in stubs",
        action="store_true",
    )

    parser.add_argument(
        "--no-deduplicate-imports",
        help="Do not deduplicate imports in the output stub.",
        action="store_true",
    )

    parser.add_argument(
        "--exclude-epilog", help="Disable inclusion of epilog", action="store_true"
    )

    parser.add_argument(
        "--continue-on-error",
        help="Continue on error",
        action="store_true",
    )

    args = parser.parse_args()

    config = StubgenPyxConfig(
        no_sort_imports=args.no_sort_imports,
        no_trim_imports=args.no_trim_imports,
        no_normalize_names=args.no_normalize_names,
        no_pxd_to_stubs=args.no_pxd_to_stubs,
        exclude_epilog=args.exclude_epilog,
        no_deduplicate_imports=args.no_deduplicate_imports,
        continue_on_error=args.continue_on_error,
    )

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    stubgen = StubgenPyx(config=config)

    if args.file:
        pyx_file_pattern = os.path.join(args.dir, args.file)
    else:
        pyx_file_pattern = os.path.join(args.dir, "**", "*.pyx")
    
    stubgen.convert_glob(pyx_file_pattern)
