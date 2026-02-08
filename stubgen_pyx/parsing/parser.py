"""Cython module parser using Cython compiler internals.

Parses Cython code after preprocessing for directives and pragmas.
See `preprocess.py` for preprocessing details.
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
    """Parsed source code and its AST.

    Attributes:
        source: The preprocessed source code text.
        source_ast: The Cython compiler AST.
    """

    source: str
    source_ast: ModuleNode


_DEFAULT_MODULE_NAME = "__pyx_module__"


def parse_pyx(
    source: str, module_name: str | None = None, pyx_path: Path | None = None
) -> ParsedSource:
    """Parse Cython source code.

    Applies file and string preprocessing, then parses with Cython compiler.

    Args:
        source: Cython source code string.
        module_name: Optional module name for error messages.
        pyx_path: Optional file path for context and preprocessing.

    Returns:
        ParsedSource with preprocessed code and AST.
    """
    module_name = module_name or _DEFAULT_MODULE_NAME

    if pyx_path:
        source = file_parsing_preprocess(pyx_path, source)
        module_name = path_to_module_name(pyx_path)

    return _parse_str(source, module_name)


def _parse_str(source: str, module_name: str) -> ParsedSource:
    """Internal: parse preprocessed source with Cython compiler."""
    context = StringParseContext(module_name, cpp=True)

    source = preprocess(source)

    ast = parse_from_strings(module_name, source, context=context)
    ast = typing.cast("ModuleNode", ast)

    parsed = ParsedSource(source, ast)
    return parsed


def _normalize_part(part: str) -> str:
    """Replace special characters with underscores for module names."""
    return part.replace("-", "_").replace(".", "_").replace(" ", "_")


def path_to_module_name(path: Path) -> str:
    """Convert a file path to a Python module name.

    Handles path separators and special characters for debugging context.
    """
    return ".".join([_normalize_part(part) for part in path.with_suffix("").parts])
