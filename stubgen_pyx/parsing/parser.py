"""Parser for Cython modules using Cython compiler internals.

This performs preprocessing of Cython code before parsing. See `preprocess.py` for details.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import typing

from Cython.Compiler.TreeFragment import parse_from_strings, StringParseContext
from Cython.Compiler import Errors
from Cython.Compiler.ModuleNode import ModuleNode

from .file_parsing import file_parsing_preprocess
from .preprocess import preprocess

Errors.init_thread()


@dataclass
class ParsedSource:
    source: str
    """The source code after preprocessing."""

    source_ast: ModuleNode
    """The AST of the source code."""


_DEFAULT_MODULE_NAME = "__pyx_module__"


def parse_pyx(
    source: str, module_name: str | None = None, pyx_path: Path | None = None
) -> ParsedSource:
    """Parse a Cython module into a ParseResult object."""
    module_name = module_name or _DEFAULT_MODULE_NAME

    if pyx_path:
        source = file_parsing_preprocess(pyx_path, source)
        module_name = path_to_module_name(pyx_path)

    return _parse_str(source, module_name)


def _parse_str(source: str, module_name: str) -> ParsedSource:
    """Parse a Cython module into a ParsedSource object."""
    context = StringParseContext(module_name, cpp=True)

    source = preprocess(source)

    ast = parse_from_strings(module_name, source, context=context)
    ast = typing.cast("ModuleNode", ast)

    parsed = ParsedSource(source, ast)
    return parsed


def _normalize_part(part: str) -> str:
    return part.replace("-", "_").replace(".", "_").replace(" ", "_")


def path_to_module_name(path: Path) -> str:
    """Convert a path to a module name for debugging."""
    return ".".join([_normalize_part(part) for part in path.with_suffix("").parts])
