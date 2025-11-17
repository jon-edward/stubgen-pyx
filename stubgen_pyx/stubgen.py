"""
The main function for stubgen-pyx.
"""

from pathlib import Path

from stubgen_pyx.build import build
from stubgen_pyx.convert import convert_module

from typing import Any, Callable, Iterable


def stubgen(
    package_dir: str,
    filter: Callable[[Path], bool] = lambda _: True,
    /,
    **ext_kwargs: Any,
) -> None:
    """
    Compiles all `.pyx` files in `package_dir` in place and generates a `.pyi` file for each module.

    `filter` is a function which takes a `Path` object and returns `True` if the module's `.pyi` file should be generated.

    `ext_kwargs` are passed directly to Extensions in `cythonize`.
    """

    modules: Iterable[tuple[Any, Path]] = build(package_dir, filter, **ext_kwargs)

    for module, file in modules:
        file.with_suffix(".pyi").write_text(convert_module(module))
