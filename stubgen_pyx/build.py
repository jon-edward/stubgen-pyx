"""
A module for building Cython modules in place.
"""

from __future__ import annotations

import importlib
from pathlib import Path
import sys

from Cython.Build import cythonize
from setuptools import setup, Extension

from typing import Any, Generator, Callable

# See https://cython.readthedocs.io/en/latest/src/userguide/source_files_and_compilation.html#compiler-directives
#
# These are basic defaults for building most Cython. To get the most out of this tool, it is
# recommended to use the `embedsignature` directive with the `python` format. Embedded signatures will
# be used to generate stubs.
compiler_directives = {
    "embedsignature": True,
    "embedsignature.format": "python",
    "language_level": "3",
}


def _path_as_module(path: Path) -> str:
    return path.with_suffix("").as_posix().replace("/", ".")


def build(
    package_dir: str,
    filter: Callable[[Path], bool] = lambda _: True,
    /,
    **ext_kwargs: Any,
) -> Generator[tuple[Any, Path], None, None]:
    """
    Compiles all `.pyx` files in `package_dir` in place.

    Yields tuples of the form `(module, file)`.

    `filter` is a function which takes a `Path` object and returns `True` if the module's `.pyi` file should be generated.

    `ext_kwargs` are passed directly to Extensions in `cythonize`.
    """

    if not Path(package_dir).exists():
        raise ValueError(f"Package directory does not exist: {package_dir}")

    if filter is None:
        filter = lambda _: True

    if (
        Path(package_dir).is_file()
        and Path(package_dir).suffix == ".pyx"
        and filter(Path(package_dir))
    ):
        cy_files = [Path(package_dir)]
        cy_names = [_path_as_module(Path(package_dir))]
    elif Path(package_dir).is_dir():
        cy_files = [p for p in Path(package_dir).glob("**/*.pyx") if filter(p)]
        cy_names = [_path_as_module(file) for file in cy_files]
    else:
        raise ValueError(f"Invalid package directory: {package_dir}")

    extensions = [
        Extension(
            name=name,
            sources=[str(file)],
            **ext_kwargs,
        )
        for name, file in zip(cy_names, cy_files)
    ]

    ext_modules = cythonize(
        extensions,
        compiler_directives=compiler_directives,
    )

    setup(
        ext_modules=ext_modules,
        script_args=[
            "build_ext",
            "--inplace",
        ],
    )

    sys_path = sys.path
    current_dir = str(Path(".").absolute())
    sys_path.insert(0, current_dir)

    try:
        for name, file in zip(cy_names, cy_files):
            yield importlib.import_module(name), file
    finally:
        try:
            sys_path.remove(current_dir)
        except ValueError:
            pass
