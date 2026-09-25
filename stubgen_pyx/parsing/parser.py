"""Cython module parser, backed by real files and Cython's own declaration analysis.

Replaces the previous string-preprocessing + raw-parse-only approach (see
git history for `preprocess.py`/`file_parsing.py`, removed as part of this
migration -- https://github.com/jon-edward/stubgen-pyx/issues/85): a
`ParsedSource` here comes from a real `FileSourceDescriptor` parsed
through a shared `StubgenContext` (see `parsing/context.py`) and Cython's
own `AnalyseDeclarationsTransform` (see `parsing/pipeline.py`), so
declarations carry real `Symtab.Entry`/`PyrexTypes.Type` info and
`cimport`s between files sharing one context resolve to the same, already
-parsed module scopes.

`parse_str` is kept for the existing string-only API surface (tests,
programmatic use with no file on disk): it still runs through the same
pipeline for Entry population, but has no real file to resolve `cimport`/
`include` targets against, so those simply won't cross-resolve -- which
matches what a bare string with no path could ever support.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path

from Cython.Compiler import Options, Parsing
from Cython.Compiler.ModuleNode import ModuleNode
from Cython.Compiler.Scanning import (
    FileSourceDescriptor,
    PyrexScanner,
    StringSourceDescriptor,
)

from .comments import CommentIndex, extract_comments
from .context import StubgenContext, find_root_package_dir
from .pipeline import run_stub_pipeline


class CircularIncludeError(Exception):
    """Raised when a chain of `include` statements refers back to itself.

    Cython's own parser (`Parsing.p_include_statement`) recurses into an
    `include`d file's tokens inline, with no cycle detection at all --
    an unguarded circular `include` segfaults the whole process rather
    than raising a catchable `RecursionError` (part of that recursion
    runs in compiled C code that doesn't respect
    `sys.getrecursionlimit()`). `_check_include_cycles` below is a
    lightweight, textual pre-scan -- not a real parse -- run purely to
    raise this, an ordinary Python exception, before Cython's parser ever
    gets a chance to recurse into the cycle.
    """


_INCLUDE_PATTERN = re.compile(r"""^\s*include\s+["']([^"']+)["']""", re.MULTILINE)


def _check_include_cycles(
    path: Path, context: StubgenContext, _visiting: frozenset[Path] = frozenset()
) -> None:
    """Best-effort guard against circular `include` statements. See `CircularIncludeError`."""
    resolved = path.resolve()
    if resolved in _visiting:
        raise CircularIncludeError(
            f"Circular include: {resolved} is included, directly or "
            f"indirectly, by itself (chain: "
            f"{' -> '.join(str(p) for p in _visiting)} -> {resolved})"
        )
    if not resolved.is_file():
        return
    try:
        text = resolved.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return

    from Cython.Compiler import Errors

    held = Errors.hold_errors()
    try:
        for match in _INCLUDE_PATTERN.finditer(text):
            try:
                include_path = context.find_include_file(
                    match.group(1), source_file_path=str(resolved)
                )
            except Errors.InternalError:
                # `find_include_file` raises (rather than just recording
                # an error) when called with pos=None and the target
                # can't be found -- expected for a pre-scan like this
                # one, which has no real node position to give it. Not
                # every `include` needs to be resolvable for our purposes
                # here; we only care about the ones that are.
                continue
            if include_path:
                _check_include_cycles(
                    Path(include_path), context, _visiting | {resolved}
                )
    finally:
        Errors.release_errors(ignore=True)
        del held


@dataclass
class ParsedSource:
    """One parsed file (or string) and everything derived from it.

    Attributes:
        source: The source text, exactly as read (no preprocessing).
        source_ast: The Cython compiler AST, with declarations analysed
            as far as `AnalyseDeclarationsTransform` could take them.
        scope: The `Symtab.ModuleScope` for this file, holding the real
            `Entry` objects `AnalyseDeclarationsTransform` attached.
        context: The (shared) `StubgenContext` this file was parsed against.
        comments: This file's comments, keyed by line (see
            `parsing/comments.py`). Always scoped to this one file, even
            when `context` has several files' worth of `ParsedSource`s
            alive at once. `conversion/converter.py` reads `# type: ...`
            comments directly off this (`CommentIndex.trailing`/
            `.leading_block`) rather than through a separate derived view.
        diagnostics: `CompileError`s Cython recorded while analysing this
            file (e.g. an unresolved `cimport`) -- collected rather than
            raised, since stub generation is expected to run against
            source that isn't necessarily fully resolvable in this
            environment. See `pipeline.run_stub_pipeline`.
    """

    source: str
    source_ast: ModuleNode
    scope: object
    context: StubgenContext
    comments: CommentIndex
    diagnostics: list = field(default_factory=list)


def _resolve_scope(
    context: StubgenContext,
    module_name: str,
    initial_pos: tuple,
    *,
    allow_pxd_merge: bool,
):
    """Get (creating if needed) the `ModuleScope` for `module_name` on `context`.

    When `allow_pxd_merge` is True, goes through `Context.find_module`,
    which -- as a side effect -- auto-discovers and merges a companion
    `.pxd` file's declarations into this scope if one is found on the
    include path (`Context.find_module`, the `pxd_file_loaded` handling).
    This is what replaces the previous manual pxd pre-parse-and-merge in
    `stubgen.py`.

    When False, bypasses that side effect by walking the same
    `Context._split_qualified_name` + `find_submodule` chain
    `find_module` itself uses, without the merge wrapper around it.
    Appropriate when the file being parsed *is itself* a `.pxd`: there's
    no further companion file to merge in, and going through
    `find_module` would make Cython try to auto-load-and-analyse this
    same file a second time (via `Context.process_pxd`) before we get to
    parse it ourselves.
    """
    if allow_pxd_merge:
        saved_cimport_from_pyx = Options.cimport_from_pyx
        Options.cimport_from_pyx = False
        try:
            return context.find_module(module_name, pos=initial_pos, need_pxd=0)
        finally:
            Options.cimport_from_pyx = saved_cimport_from_pyx

    scope = context
    for name, is_package in context._split_qualified_name(module_name):
        scope = scope.find_submodule(name, as_package=is_package)
    return scope


def parse_file(
    path: Path,
    context: StubgenContext,
    *,
    pxd: bool | None = None,
    module_name: str | None = None,
) -> ParsedSource:
    """Parse a real file on disk.

    Args:
        path: The file to parse. Must exist on disk.
        context: The (shared) context to parse against. Reuse the same
            context across every file in a conversion run so `cimport`s
            between them resolve to the same module scopes.
        pxd: Whether to parse as a declaration-only file. Defaults to
            `path.suffix == ".pxd"`.
        module_name: Defaults to `path_to_module_name(path)`.

    Returns:
        A `ParsedSource` for `path`.
    """
    pxd = path.suffix == ".pxd" if pxd is None else pxd
    module_name = module_name or path_to_module_name(path)

    _check_include_cycles(path, context)

    source_desc = FileSourceDescriptor(str(path))
    initial_pos = (source_desc, 1, 0)

    scope = _resolve_scope(context, module_name, initial_pos, allow_pxd_merge=not pxd)

    tree = context.parse(source_desc, scope, pxd=pxd, full_module_name=module_name)
    tree.scope = scope
    tree.is_pxd = pxd

    # Snapshot structural type info before the pipeline can clear it --
    # see `capture_static_types`'s docstring for why this has to happen
    # here, before `run_stub_pipeline`, not after.
    from ..conversion.type_parsing import capture_static_types

    capture_static_types(tree)

    result = run_stub_pipeline(context, "pxd" if pxd else "pyx", tree)

    source_text = path.read_text(encoding="utf-8")
    comments = extract_comments(source_text)

    return ParsedSource(
        source=source_text,
        source_ast=result.tree,
        scope=scope,
        context=context,
        comments=comments,
        diagnostics=result.diagnostics,
    )


def parse_str(
    source: str,
    module_name: str | None = None,
    pxd: bool = False,
    context: StubgenContext | None = None,
) -> ParsedSource:
    """Parse a Cython source string with no backing file.

    `cimport`/`include` targets won't resolve to other in-memory content
    parsed this way -- there's no real path for `Context.find_module` to
    search relative to. Kept for the existing string-only API surface
    (tests, programmatic use without files on disk); prefer `parse_file`
    whenever a real file is available.

    Args:
        source: Cython source code.
        module_name: Defaults to a synthetic name.
        pxd: Whether to parse as a declaration-only file.
        context: Reuse an existing context, or create a fresh one.
    """
    module_name = module_name or _DEFAULT_MODULE_NAME
    context = context or StubgenContext()

    from Cython.Compiler import Errors

    held = Errors.hold_errors()
    try:
        for match in _INCLUDE_PATTERN.finditer(source):
            try:
                include_path = context.find_include_file(match.group(1))
            except Errors.InternalError:
                continue
            if include_path:
                _check_include_cycles(Path(include_path), context)
    finally:
        Errors.release_errors(ignore=True)
        del held

    source_desc = StringSourceDescriptor(module_name, source)
    initial_pos = (source_desc, 1, 0)
    # Always bypass find_module's companion-.pxd auto-merge here: it needs
    # a FileSourceDescriptor (Main.search_include_directories asserts on
    # this), and there's no real path for a string input to resolve a
    # companion file against anyway.
    scope = _resolve_scope(context, module_name, initial_pos, allow_pxd_merge=False)
    scope.cpp = context.cpp

    scanner = PyrexScanner(
        StringIO(source),
        source_desc,
        source_encoding="UTF-8",
        scope=scope,
        context=context,
        initial_pos=initial_pos,
    )
    tree = Parsing.p_module(
        scanner, pxd, module_name, ctx=Parsing.Ctx(allow_struct_enum_decorator=True)
    )
    tree.scope = scope
    tree.is_pxd = pxd

    # Snapshot structural type info before the pipeline can clear it --
    # see `capture_static_types`'s docstring for why this has to happen
    # here, before `run_stub_pipeline`, not after.
    from ..conversion.type_parsing import capture_static_types

    capture_static_types(tree)

    result = run_stub_pipeline(context, "pxd" if pxd else "pyx", tree)
    comments = extract_comments(source)

    return ParsedSource(
        source=source,
        source_ast=result.tree,
        scope=scope,
        context=context,
        comments=comments,
        diagnostics=result.diagnostics,
    )


_DEFAULT_MODULE_NAME = "__pyx_module__"


def _normalize_part(part: str) -> str:
    """Replace special characters with underscores for module names."""
    return part.replace("-", "_").replace(".", "_").replace(" ", "_")


def path_to_module_name(path: Path) -> str:
    """Convert a file path to a qualified Cython module name.

    Rooted at `find_root_package_dir(path)` -- the same root Cython's own
    search machinery derives for the file -- so a module name computed
    here agrees with the name Cython computes internally when resolving a
    `cimport` of this same file from elsewhere in the project.
    """
    resolved = path.resolve()
    root = find_root_package_dir(resolved)
    stem = resolved.with_suffix("")
    try:
        relative = stem.relative_to(root)
    except ValueError:
        # Not under its own computed root (e.g. a symlink situation) --
        # fall back to just the bare file name rather than producing an
        # invalid, path-separator-laden module name.
        relative = Path(stem.name)
    return ".".join(_normalize_part(part) for part in relative.parts)
