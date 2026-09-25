"""Shared Cython compiler ``Context`` for stub generation.

Unlike ``Cython.Compiler.TreeFragment.StringParseContext`` (used by the
previous, string-based parser), a ``StubgenContext`` is backed by real
include directories and is meant to be constructed once and shared across
every file in a conversion run. Sharing one context is what lets a
``cimport`` in one file resolve to the same, already-parsed ``Entry``/
``ModuleScope`` objects for another file in the same run -- see
``Cython.Compiler.Main.Context.find_module``, which caches parsed modules
on the context instance (``self.modules``) rather than per call.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from Cython import Utils as CythonUtils
from Cython.Compiler.Main import Context
from Cython.Compiler.Options import CompilationOptions, default_options

#: Sparse overrides only -- ``InterpretCompilerDirectives`` deep-copies
#: ``Options.get_directive_defaults()`` and overlays this dict on top, so
#: there's no need (and no benefit) to pass a fully populated directives
#: dict here. See ``Cython.Compiler.ParseTreeTransforms.InterpretCompilerDirectives.__init__``.
_DEFAULT_COMPILER_DIRECTIVES: dict[str, object] = {
    "language_level": 3,
    # Cython's default (`None`) auto-generates `__reduce_cython__`/
    # `__setstate_cython__` (and a couple of supporting module-level
    # assignments) for extension types during `AnalyseDeclarationsTransform`
    # (`ParseTreeTransforms.CreatePropertyTransform`... see
    # `AnalyseDeclarationsTransform`'s pickle-support handling) -- real,
    # compiler-synthesized nodes spliced into the class body, not just
    # symbol-table entries. Stub generation should reflect what the user
    # wrote, not what Cython would add at compile time, so this is
    # disabled by default.
    "auto_pickle": False,
}


def _ensure_errors_thread_initialized() -> None:
    """Initialize Cython's thread-local error-reporting state, once.

    Cython's error-reporting state (`Cython.Compiler.Errors`) is
    thread-local and normally initialized once by whatever sets up a real
    compile (`Cython.Compiler.Main`'s CLI entry points call this). We
    never go through those, but several things we *do* call --
    `Context.parse` included, which checks `Errors.get_errors_count()`
    unconditionally -- assume it's already been done. Guarded so we don't
    reset error-listing state a host application may have set up for its
    own, unrelated use of Cython in the same process.
    """
    from Cython.Compiler import Errors

    if not hasattr(Errors.threadlocal, "cython_errors_count"):
        Errors.init_thread()


class StubgenContext(Context):
    """A ``Context`` configured for declaration-only stub generation.

    Safe to reuse across multiple ``parse_file``/``parse_str`` calls (see
    ``parsing/parser.py``): reusing the same instance for every file in a
    conversion run is what makes cross-file ``cimport`` resolution work.
    """

    def __init__(
        self,
        include_directories: list[str] | None = None,
        compiler_directives: dict[str, object] | None = None,
        cpp: bool = True,
    ) -> None:
        _ensure_errors_thread_initialized()
        super().__init__(
            list(include_directories or []),
            dict(compiler_directives or _DEFAULT_COMPILER_DIRECTIVES),
            cpp=cpp,
            language_level=3,
            # Cython's own internals reach for `context.options.*`
            # unconditionally in a few places we can't avoid going
            # through -- notably `Context.parse` (`self.options.formal_grammar`),
            # which fires even for our own primary-file parse, and
            # `Context.process_pxd`'s full pipeline (see
            # `find_root_package_dir`'s docstring for why a companion
            # `.pxd` auto-merge goes through Cython's real pipeline, not
            # ours). A plain `CompilationOptions(default_options)` is
            # enough to satisfy those attribute accesses; nothing here
            # depends on it beyond that -- we never reach codegen options.
            options=CompilationOptions(default_options),
        )

    def add_include_directory(self, directory: Path | str) -> None:
        """Add a directory to the front of the include search path.

        No-op if the directory is already present. Cython's own
        ``libc``/``libcpp``/``cpython`` ``.pxd`` files are always
        searched too (``Context.search_include_directories`` appends
        ``Main.standard_include_path`` unconditionally), so they never
        need to be added here.
        """
        directory = str(Path(directory))
        if directory not in self.include_directories:
            self.include_directories.insert(0, directory)

    def search_include_directories(self, *args, **kwargs):
        """As ``Context.search_include_directories``, tolerant of a non-file source.

        The base implementation (``Main.search_include_directories``,
        the free function) unconditionally raises ``RuntimeError("Only
        file sources for code supported")`` when asked to search relative
        to a position whose source descriptor isn't a
        ``FileSourceDescriptor`` -- which is exactly the position every
        node gets when parsed via ``parse_str`` (``StringSourceDescriptor``,
        no real file to search relative to). Confirmed empirically: this
        makes *any* ``cimport`` in a `parse_str`-parsed source -- even one
        that would resolve fine, like ``from libc.stdint cimport int32_t``
        -- fail with an internal crash instead of either resolving (for
        standard-library ``.pxd``s, found via
        ``Main.standard_include_path`` regardless of source) or failing
        gracefully (for anything else).

        Retries once without the position/source-file-path, which is
        exactly what real Cython does for a search with no position at
        all (`Context.find_include_file`, `Parsing.p_include_statement`'s
        own error path) -- still searches `self.include_directories` and
        Cython's own standard include path, just skips the "also search
        this file's own directory" step, since there's no real directory
        for a string input.
        """
        try:
            return super().search_include_directories(*args, **kwargs)
        except RuntimeError as e:
            if "Only file sources" not in str(e):
                raise
            kwargs = dict(kwargs)
            kwargs["source_pos"] = None
            kwargs["source_file_path"] = None
            return super().search_include_directories(*args, **kwargs)


def find_root_package_dir(path: Path | str) -> Path:
    """The topmost ancestor directory of `path` that's still part of the same package tree.

    Thin wrapper over `Cython.Utils.find_root_package_dir` -- the same
    function `Context.search_include_directories` uses internally
    (`Main.py`, non-`include` branch) to decide where a dotted module
    name search should be rooted. Used here so our own module naming
    (`parser.path_to_module_name`) and include-path setup agree with what
    Cython's own search machinery will independently derive for the same
    file -- otherwise a qualified module name we hand to `find_module`
    could disagree with the name Cython computes when resolving a
    `cimport` of the same file from elsewhere, and cross-file resolution
    would silently miss.

    Walks up as long as each directory contains an `__init__.{py,pyx,pxd}`
    file; stops at the first ancestor that doesn't (or at the filesystem
    root).
    """
    return Path(CythonUtils.find_root_package_dir(str(path)))


def context_for_paths(
    paths: Iterable[Path | str],
    *,
    extra_include_dirs: Iterable[Path | str] | None = None,
    cpp: bool = True,
) -> StubgenContext:
    """Build a ``StubgenContext`` whose include path covers a set of source files.

    Adds each file's parent directory (so sibling ``cimport``s within the
    same project resolve without extra configuration) plus any explicitly
    configured extra include directories (for third-party ``.pxd``-only
    packages, e.g. numpy).

    Args:
        paths: Source file paths that will be parsed with the returned context.
        extra_include_dirs: Additional directories to search for ``cimport``/
            ``include`` targets, beyond each file's own parent directory.
        cpp: Whether to parse in C++ mode.

    Returns:
        A new ``StubgenContext``.
    """
    directories: list[str] = []
    for path in paths:
        root = find_root_package_dir(Path(path).resolve())
        if str(root) not in directories:
            directories.append(str(root))
    for extra in extra_include_dirs or ():
        extra = str(Path(extra))
        if extra not in directories:
            directories.append(extra)
    return StubgenContext(include_directories=directories, cpp=cpp)
