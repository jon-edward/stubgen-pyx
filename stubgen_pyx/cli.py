from __future__ import annotations

import argparse
import os

from .stubgen import StubgenPyx


def main():
    parser = argparse.ArgumentParser(description="Generate stubs for Cython modules")
    parser.add_argument(
        "dir", help="Directory to search for Cython modules", type=str, default="."
    )

    args = parser.parse_args()
    dir: str = args.dir

    stubgen = StubgenPyx()

    pyx_glob = os.path.join(dir, "**", "*.pyx")
    stubgen.convert_glob(pyx_glob)
