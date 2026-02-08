"""
Profile the stubgen-pyx package
"""

import cProfile
import argparse

from stubgen_pyx.stubgen import StubgenPyx


def main():
    parser = argparse.ArgumentParser(description="Profile the stubgen-pyx package")
    parser.add_argument("file_pattern", help="Glob pattern for files to convert")
    args = parser.parse_args()

    with cProfile.Profile() as pr:
        StubgenPyx().convert_glob(args.file_pattern)

    pr.print_stats(sort="cumulative")


if __name__ == "__main__":
    main()
