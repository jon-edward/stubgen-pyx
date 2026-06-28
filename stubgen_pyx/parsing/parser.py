"""Cython module parser using Cython compiler internals.

Parses Cython code after preprocessing for directives and pragmas.
See `preprocess.py` for preprocessing details.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from io import StringIO
from pathlib import Path

from Cython.Compiler.TreeFragment import StringParseContext
from Cython.Compiler import Errors, Parsing
from Cython.Compiler.ModuleNode import ModuleNode
from Cython.Compiler.Scanning import PyrexScanner, StringSourceDescriptor

from .file_parsing import file_parsing_preprocess
from .preprocess import (
    preprocess,
    extract_type_comments,
    get_lines_with_newlines_in_brackets,
)

Errors.init_thread()


@dataclass
class ParsedSource:
    """Parsed source code and its AST.

    Attributes:
        source: The preprocessed source code text.
        source_ast: The Cython compiler AST.
        type_comments: Map of line number → `# type: ...` comment captured
            before comment stripping, keyed by 1-based line numbers that
            align with AST node positions for typical single-line defs.
    """

    source: str
    source_ast: ModuleNode
    type_comments: dict[int, str] = field(default_factory=dict)


_DEFAULT_MODULE_NAME = "__pyx_module__"
_PARSE_CACHE: dict[tuple[tuple[str, bool], str, str], ParsedSource] = {}


def _make_parse_cache_key(
    source: str,
    module_name: str,
    pxd: bool,
    pyx_path: Path | None,
) -> tuple[tuple[str, bool], str, str]:
    """Create a cache key for a parse operation."""
    if pyx_path is not None:
        try:
            stat_result = pyx_path.stat()
            path_key = (
                f"{pyx_path.resolve()}:{stat_result.st_mtime_ns}:{stat_result.st_size}"
            )
        except OSError:
            path_key = str(pyx_path.resolve())
    else:
        path_key = ""

    source_digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    return ((module_name, pxd), source_digest, path_key)


def clear_parse_cache() -> None:
    """Clear cached parse results."""
    _PARSE_CACHE.clear()


def parse_pyx(
    source: str,
    module_name: str | None = None,
    pyx_path: Path | None = None,
    pxd: bool = False,
) -> ParsedSource:
    """Parse Cython source code.

    Applies file and string preprocessing, then parses with Cython compiler.

    Args:
        source: Cython source code string.
        module_name: Optional module name for error messages.
        pyx_path: Optional file path for context and preprocessing.
        pxd: Whether the source is a .pxd file. If pyx_path is also provided and it's a .pxd, this is overridden.

    Returns:
        ParsedSource with preprocessed code and AST.
    """
    module_name = module_name or _DEFAULT_MODULE_NAME

    if pyx_path:
        pxd = pxd or pyx_path.suffix == ".pxd"
        source = file_parsing_preprocess(pyx_path, source)
        module_name = path_to_module_name(pyx_path)

    cache_key = _make_parse_cache_key(source, module_name, pxd, pyx_path)
    if cache_key in _PARSE_CACHE:
        return _PARSE_CACHE[cache_key]

    parsed_source = _parse_str(source, module_name, pxd)
    _PARSE_CACHE[cache_key] = parsed_source
    return parsed_source


def _parse_str(source: str, module_name: str, pxd: bool = False) -> ParsedSource:
    """Simplified version of Cython.Compiler.TreeFragment.parse_from_strings but with allowing pxd.

    Args:
        source: Cython source code string.
        module_name: Module name for error messages.
        pxd: Whether the source is a .pxd file.

    Returns:
        ParsedSource with preprocessed code and AST.
    """

    # Extract type comments from the original source, then translate
    # original line numbers to post-preprocess line numbers by subtracting
    # the count of in-bracket newlines that `preprocess` will collapse.
    # We can't run that flattening up-front: comments inside a bracketed
    # block are terminated by their newline, so removing the newline would
    # merge the comment with following code and break tokenization.
    type_comments_orig = extract_type_comments(source)
    collapsed_lines = sorted(get_lines_with_newlines_in_brackets(source))
    type_comments: dict[int, str] = {}
    for orig_line, comment in type_comments_orig.items():
        shift = sum(1 for line in collapsed_lines if line < orig_line)
        type_comments[orig_line - shift] = comment

    source = preprocess(source)

    encoding = "UTF-8"
    initial_pos = (module_name, 1, 0)

    context = StringParseContext(module_name)
    code_source = StringSourceDescriptor(module_name, source)
    buf = StringIO(source)

    scope = context.find_module(module_name, pos=initial_pos, need_pxd=False)
    scanner = PyrexScanner(
        buf,
        filename=code_source,
        source_encoding=encoding,
        scope=scope,
        context=context,
        initial_pos=initial_pos,
    )
    ctx = Parsing.Ctx(allow_struct_enum_decorator=True)

    tree = Parsing.p_module(scanner, pxd, module_name, ctx=ctx)

    return ParsedSource(source, tree, type_comments)


def _normalize_part(part: str) -> str:
    """Replace special characters with underscores for module names."""
    return part.replace("-", "_").replace(".", "_").replace(" ", "_")


def path_to_module_name(path: Path) -> str:
    """Convert a file path to a Python module name.

    Handles path separators and special characters for debugging context.
    """
    return ".".join([_normalize_part(part) for part in path.with_suffix("").parts])
