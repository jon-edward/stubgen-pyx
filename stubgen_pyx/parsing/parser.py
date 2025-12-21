from dataclasses import dataclass
from pathlib import Path
import typing

from Cython.Compiler.TreeFragment import parse_from_strings, StringParseContext
from Cython.Compiler import Errors
from Cython.Compiler.ModuleNode import ModuleNode

from .preprocess import preprocess

Errors.init_thread()

_MODULE_NAME = "__pyx_module__"


@dataclass
class ParsedSource:
    source: str
    """The source code after preprocessing."""

    source_ast: ModuleNode
    """The AST of the source code."""

@dataclass
class ParseResult:
    module_result: ParsedSource
    """The result of parsing the module."""

    pxd_result: ParsedSource | None
    """The result of parsing the pxd file if it exists."""


def parse_pyx(source: Path | str) -> ParseResult:
    """Parse a Cython module into a ParseResult object."""

    if isinstance(source, str):
        return ParseResult(_parse_str(source), None)
    
    pxd_source = source.with_suffix(".pxd")
    
    module_name = _path_to_module_name(source)

    if pxd_source.is_file():
        return ParseResult(
            _parse_str(source.read_text(encoding="utf-8")), 
            _parse_str(pxd_source.read_text(encoding="utf-8"))
        )
    
    return ParseResult(_parse_str(source.read_text(encoding="utf-8")), None)


def _parse_str(source: str, module_name: str = _MODULE_NAME) -> ParsedSource:
    """Parse a Cython module into a ParsedSource object."""
    context = StringParseContext(_MODULE_NAME)

    source = preprocess(source)
    ast = parse_from_strings(module_name, source, context=context)
    ast = typing.cast("ModuleNode", ast)
    
    return ParsedSource(source, ast)

def _normalize_part(part: str) -> str:
    return part.replace("-", "_").replace(".", "_").replace(" ", "_")

def _path_to_module_name(path: Path) -> str:
    return ".".join([_normalize_part(part) for part in path.with_suffix("").parts])
