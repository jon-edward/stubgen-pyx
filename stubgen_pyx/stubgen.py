"""StubgenPyx converts .pyx files to .pyi files."""

from __future__ import annotations

import glob
import logging
import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path

from .analysis.visitor import ModuleVisitor
from .builders.builder import Builder
from .config import StubgenPyxConfig
from .conversion.converter import Converter
from .conversion.ctypedef_aliases import pyi_module_uses_name, remove_assignment_from_module
from .conversion.fused_types import convert_fused_types
from .models.pyi_elements import PyiClass, PyiModule
from .parsing.context import StubgenContext, context_for_paths
from .parsing.parser import ParsedSource, parse_file, parse_str, path_to_module_name
from .postprocessing.pipeline import postprocessing_pipeline

_logger = logging.getLogger(__name__)


@dataclass
class ConversionResult:
    """Result of converting a single .pyx file to .pyi.

    Attributes:
        success: Whether the conversion completed without error.
        pyx_file: Path to the source .pyx file.
        pyi_file: Path to the generated .pyi file.
        error: Exception if conversion failed, otherwise None.
        diagnostics: `Cython.Compiler.Errors.CompileError`s recorded
            while parsing/analysing this file (and its companion .pxd,
            if any) but not fatal -- most commonly an unresolved
            `cimport` (e.g. a dependency that isn't installed in this
            environment). Declarations touched by a diagnostic are
            dropped from the generated stub rather than guessed at (see
            `parsing/pipeline.py::run_stub_pipeline`), so a non-empty
            list here means the .pyi may be missing something the
            source declared, even though `success` is `True`. Empty
            when nothing was recorded, including for a failed
            conversion (`success=False`) where the failure came from
            something other than this mechanism (see `error` instead).
    """

    success: bool
    pyx_file: Path
    pyi_file: Path
    error: Exception | None = None
    diagnostics: list[Exception] = field(default_factory=list)

    @property
    def status_message(self) -> str:
        """Human-readable status summary."""
        if self.success:
            if self.pyx_file != self.pyi_file:
                message = f"Converted {self.pyx_file} to {self.pyi_file}"
            else:
                message = f"Skipped {self.pyx_file}"
            if self.diagnostics:
                message += (
                    f" ({len(self.diagnostics)} declaration"
                    f"{'s' if len(self.diagnostics) != 1 else ''} skipped -- "
                    "see .diagnostics)"
                )
            return message
        return f"Failed to convert {self.pyx_file}: {self.error}"


def _log_diagnostics(diagnostics: list[Exception], pyx_path: Path | None) -> None:
    """Log each parse/analysis diagnostic as a warning.

    These are recorded (see `ParsedSource.diagnostics`,
    `parsing/pipeline.py::run_stub_pipeline`) rather than raised, by
    design -- a stub generator can't assume every dependency the source
    references is importable in this environment (the common case: an
    unresolved `cimport` for a package that isn't installed here, e.g.
    in a CI job without a full dev environment). Recording them isn't
    the same as surfacing them, though: without this, a diagnostic
    (and whatever declaration it caused to be silently dropped from the
    stub) was previously visible only by reading `ParsedSource.
    diagnostics`/`ConversionResult.diagnostics` directly -- nothing
    logged it, at any level, anywhere. A CI run watching only its
    console output (the common case) would see a clean, successful
    conversion with no indication anything was skipped.

    Formatted as one line per diagnostic (file:line:col: message)
    rather than `str(diagnostic)`, which for a `CompileError` is a
    multi-line block quoting several lines of source around the
    error -- appropriate for a human reading a single failure
    interactively, but noisy repeated once per diagnostic in a CI log
    that may have several. `message_only`/`position` (both `CompileError`
    -specific) give the same information in one line; anything else
    (a diagnostic type that doesn't have them) falls back to `str()`.
    """
    label = f" in {pyx_path}" if pyx_path is not None else ""
    for diagnostic in diagnostics:
        message = getattr(diagnostic, "message_only", None)
        position = getattr(diagnostic, "position", None)
        try:
            if message is not None and position is not None:
                source_desc, line, column = position
                # `str(source_desc)` (a `FileSourceDescriptor`) can
                # itself raise (an internal assertion, confirmed
                # empirically for a diagnostic recorded against a
                # target file that doesn't actually exist on disk -- an
                # unresolved `cimport`'s own position, say) --
                # `.filename` is what it's built from and doesn't have
                # that problem.
                filename = getattr(source_desc, "filename", None) or "<unknown>"
                _logger.warning(
                    f"Unresolved declaration{label}: {filename}:{line}:{column}: {message}"
                )
            else:
                _logger.warning(f"Unresolved declaration{label}: {diagnostic}")
        except Exception:
            # Formatting a diagnostic nicely is a courtesy, never a
            # requirement -- falling back to the raw exception (or
            # giving up on this one entirely) must never take the whole
            # conversion down with it.
            _logger.warning(f"Unresolved declaration{label} (unformattable diagnostic)")




@dataclass
class StubgenPyx:
    """Primary API for converting Cython .pyx files to .pyi stub files.

    Parses Cython source, extracts types, and generates type stubs with optional
    postprocessing (import normalization, trimming, etc.).

    Attributes:
        config: Configuration controlling generation behavior.
    """

    config: StubgenPyxConfig = field(default_factory=StubgenPyxConfig)

    def _make_converter(self) -> Converter:
        return Converter()

    def _make_builder(self) -> Builder:
        return Builder(
            include_private=self.config.include_private,
            replace_defaults_with_ellipsis=self.config.replace_defaults_with_ellipsis,
        )

    def convert_str(
        self, pyx_str: str, pxd_str: str | None = None, pyx_path: Path | None = None
    ) -> str:
        """Convert Cython source strings to .pyi stub code.

        Args:
            pyx_str: The source Cython code.
            pxd_str: Optional companion .pxd file content to merge.
            pyx_path: Optional file path for context and error messages.

        Returns:
            Generated .pyi stub file content.

        Raises:
            Various exceptions from parsing, conversion, or building.
        """
        converter = self._make_converter()
        module, diagnostics = self._compile_with_converter(
            converter, pyx_str, pxd_str, pyx_path
        )
        _log_diagnostics(diagnostics, pyx_path)
        return self._finalize(converter, module, pyx_path)

    def _finalize(
        self, converter: Converter, module: PyiModule, pyx_path: Path | None
    ) -> str:
        """`PyiModule` -> final .pyi text: build, then postprocess."""
        builder = self._make_builder()
        content = builder.build_module(module)
        return (
            postprocessing_pipeline(
                content,
                self.config,
                pyx_path,
                extra_translations=converter.cimport_alias_map,
            ).strip()
            + "\n"
        )

    def compile_str_to_module(
        self, pyx_str: str, pxd_str: str | None = None, pyx_path: Path | None = None
    ) -> PyiModule:
        """Compile Cython source strings into a PyiModule.

        Args:
            pyx_str: The source Cython code.
            pxd_str: Optional companion .pxd file content to merge.
            pyx_path: Optional file path for context and error messages.

        Returns:
            PyiModule instance as the result of the compilation.

        Raises:
            Various exceptions from parsing, conversion, or building.
        """
        module, diagnostics = self._compile_with_converter(
            self._make_converter(), pyx_str, pxd_str, pyx_path
        )
        _log_diagnostics(diagnostics, pyx_path)
        return module

    def _compile_with_converter(
        self,
        converter: Converter,
        pyx_str: str,
        pxd_str: str | None = None,
        pyx_path: Path | None = None,
    ) -> tuple[PyiModule, list[Exception]]:
        """Compile from in-memory strings (no real files required).

        Uses a fresh, single-call `StubgenContext`: each call here is
        isolated on purpose, matching this method's existing semantics
        (the public string API is meant for synthetic/programmatic
        content, not necessarily anything on disk). `cimport`/`include`
        targets won't resolve to anything outside `pyx_str`/`pxd_str`
        themselves -- see `parsing/parser.py::parse_str`. For real files,
        use `_compile_file_with_converter`, which gets real cross-file
        `cimport` resolution via a real `StubgenContext`.
        """
        module_name = path_to_module_name(pyx_path) if pyx_path else None
        context = StubgenContext()

        pxd_parse_result = None
        if pxd_str and self.config.pxd_to_stubs:
            pxd_parse_result = parse_str(
                pxd_str, module_name=module_name, pxd=True, context=context
            )

        parse_result = parse_str(
            pyx_str, module_name=module_name, pxd=False, context=context
        )

        return self._build_module(converter, parse_result, pxd_parse_result)

    def _compile_file_with_converter(
        self,
        converter: Converter,
        pyx_path: Path,
        context: StubgenContext,
        defer_ctypedef_pruning: bool = False,
        prunable_ctypedef_aliases: dict[str, "PyiAssignment"] | None = None,
    ) -> tuple[PyiModule, list[Exception]]:
        """Compile a real .pyx file (and, optionally, its companion .pxd).

        `context` is expected to be shared across every file in a
        conversion run (see `convert_multiple_files`) so that `cimport`s
        between them resolve to the same, already-parsed module scopes.

        `defer_ctypedef_pruning`/`prunable_ctypedef_aliases`: see
        `_build_module`, which this passes them straight through to.
        """
        module_name = path_to_module_name(pyx_path)

        pxd_parse_result = None
        if self.config.pxd_to_stubs:
            pxd_path = pyx_path.with_suffix(".pxd")
            if pxd_path.exists() and pxd_path != pyx_path:
                try:
                    pxd_parse_result = parse_file(
                        pxd_path, context, pxd=True, module_name=module_name
                    )
                except UnicodeDecodeError:
                    _logger.warning(f"Could not read .pxd file {pxd_path}")
                except Exception as e:
                    # See `convert_single_file`'s matching handler for
                    # why a decode error surfaces as a
                    # `Cython.Compiler.Errors.CompileError` here now,
                    # not a plain `UnicodeDecodeError`.
                    from Cython.Compiler import Errors as _CythonErrors

                    cause = e.__cause__ or e.__context__
                    if isinstance(e, _CythonErrors.CompileError) and isinstance(
                        cause, UnicodeDecodeError
                    ):
                        _logger.warning(f"Could not read .pxd file {pxd_path}")
                    else:
                        raise

        parse_result = parse_file(
            pyx_path, context, pxd=False, module_name=module_name
        )

        return self._build_module(
            converter,
            parse_result,
            pxd_parse_result,
            defer_ctypedef_pruning=defer_ctypedef_pruning,
            prunable_ctypedef_aliases=prunable_ctypedef_aliases,
        )

    def _build_module(
        self,
        converter: Converter,
        parse_result: ParsedSource,
        pxd_parse_result: ParsedSource | None,
        defer_ctypedef_pruning: bool = False,
        prunable_ctypedef_aliases: dict[str, "PyiAssignment"] | None = None,
    ) -> tuple[PyiModule, list[Exception]]:
        """Shared tail of both compile paths: `ParsedSource(s)` -> `PyiModule`.

        Full fused type support including cross-file inheritance would
        require Cython's own scope/env for symbol resolution, which
        wasn't available when this was originally written; this pre-pass
        parses the companion `.pxd` file (same stem) separately and
        merges its fused typedefs into the `.pyx` conversion, covering
        the majority of real-world use cases. Fused typedefs made visible
        via `cimport` from an unrelated module still aren't resolved this
        way.

        That restriction is now narrower than what `parse_file`'s shared
        `StubgenContext` can actually resolve -- a real cross-file
        `cimport` does get a real, shared `Entry`/scope (see
        `parsing/parser.py`). `analysis/`/`conversion/converter.py`
        haven't been rewritten to consume that yet, though -- they still
        walk the raw, un-merged AST structurally, so this pre-pass-and-
        merge step is still load-bearing for now. See the design doc's
        "delete analysis/, single-pass converter" / "Entry-driven
        conversion" sections for the follow-up that retires it.

        Returns the diagnostics `parse_result`/`pxd_parse_result`
        recorded (see `ParsedSource.diagnostics`) alongside the module,
        combined into one list -- callers decide how to surface them
        (`convert_str`/`compile_str_to_module` just log; `convert_single_file`
        also attaches them to its `ConversionResult`).

        `defer_ctypedef_pruning`/`prunable_ctypedef_aliases`: passed straight
        through to both `convert_module` calls (main file and companion
        `.pxd`, when present) -- see `Converter.convert_scope`'s docstring.
        Only `convert_multiple_files` sets these, for its whole-batch
        cross-file check.
        """
        pxd_visitor = None
        pxd_fused_types = None
        if pxd_parse_result is not None:
            pxd_visitor = ModuleVisitor(node=pxd_parse_result.source_ast)
            pxd_fused_types = convert_fused_types(pxd_visitor.scope)

        module_visitor = ModuleVisitor(node=parse_result.source_ast)
        module = converter.convert_module(
            module_visitor,
            parse_result.source,
            parse_result.comments,
            include_docstrings=self.config.include_docstrings,
            inherited_fused_types=pxd_fused_types,
            resolve_ctypedef_aliases=self.config.resolve_ctypedef_aliases,
            defer_ctypedef_pruning=defer_ctypedef_pruning,
            prunable_ctypedef_aliases=prunable_ctypedef_aliases,
        )

        if pxd_parse_result is not None and pxd_visitor is not None:
            pxd_module = converter.convert_module(
                pxd_visitor,
                pxd_parse_result.source,
                pxd_parse_result.comments,
                include_docstrings=self.config.include_docstrings,
                resolve_ctypedef_aliases=self.config.resolve_ctypedef_aliases,
                defer_ctypedef_pruning=defer_ctypedef_pruning,
                prunable_ctypedef_aliases=prunable_ctypedef_aliases,
            )
            _merge_pxd_into_module(module, pxd_module)

        diagnostics = list(parse_result.diagnostics)
        if pxd_parse_result is not None:
            diagnostics.extend(pxd_parse_result.diagnostics)
        return module, diagnostics

    def resolve_glob(
        self, pyx_file_pattern: str, exclude_patterns: list[str] | str | None = None
    ) -> tuple[Path, ...]:
        """Resolve given glob pattern.

        Args:
            pyx_file_pattern: Glob pattern (e.g., "**/*.pyx", "src/*.pyx").

        Returns:
            Iterable of Path to the resolved file names.

        When matching .pyx patterns, standalone .pxd files are also included if
        there is no corresponding .pyx file with the same stem.
        """
        file_paths = list(glob.glob(pyx_file_pattern, recursive=True))

        if pyx_file_pattern.lower().endswith(".pyx"):
            pxd_pattern = pyx_file_pattern[:-4] + ".pxd"
            pxd_files = glob.glob(pxd_pattern, recursive=True)
            pyx_stems = {Path(p).with_suffix("") for p in file_paths}
            for pxd_path in pxd_files:
                if Path(pxd_path).with_suffix("") not in pyx_stems:
                    file_paths.append(pxd_path)

        unique_paths = list(dict.fromkeys(file_paths))
        gen = (Path(p) for p in unique_paths)

        if not exclude_patterns:
            return tuple(gen)

        if isinstance(exclude_patterns, str):
            exclude_patterns = [exclude_patterns]

        output = tuple(
            f
            for f in gen
            if not any(
                fnmatch(str(f.as_posix()), p.replace("\\", "/"))
                for p in exclude_patterns
            )
        )
        if output:
            _logger.info(f"Found {len(output)} file(s) to convert")
        return output

    def convert_glob(
        self,
        pyx_file_pattern: str,
        output_dir: Path | None = None,
        dry_run: bool = False,
        exclude_patterns: list[str] | str | None = None,
    ) -> list[ConversionResult]:
        """Convert multiple .pyx files matching a glob pattern.

        Args:
            pyx_file_pattern: Glob pattern (e.g., "**/*.pyx", "src/*.pyx").
            output_dir: Optional output directory for .pyi files. If None,
                .pyi files are placed next to their source files.
            dry_run: If True, no files are actually created.

        Returns:
            List of ConversionResult objects with status for each file.
        """
        pyx_files = self.resolve_glob(
            pyx_file_pattern, exclude_patterns=exclude_patterns
        )

        if not pyx_files:
            _logger.warning("No files found to convert")
            return []

        return self.convert_multiple_files(
            pyx_files, output_dir=output_dir, dry_run=dry_run
        )

    def _resolve_pyi_path(
        self, pyx_path: Path, output_dir: Path | None, common_root: Path | None
    ) -> Path | None:
        """Where a given `pyx_path`'s .pyi output goes -- `None` means
        "generate in place" (`convert_single_file`'s own default).
        Factored out of `convert_multiple_files`'s loop so its
        deferred-pruning batch path (`_convert_multiple_files_with_ctypedef_pruning`)
        can compute the same thing without duplicating the logic.
        """
        if not output_dir:
            return None
        pyi_name = pyx_path.with_suffix(".pyi")
        if common_root is not None:
            try:
                pyi_path = output_dir / pyi_name.relative_to(common_root)
            except ValueError:
                pyi_path = output_dir / pyi_name.name
        else:
            pyi_path = output_dir / pyi_name.name
        pyi_path.parent.mkdir(parents=True, exist_ok=True)
        return pyi_path

    def convert_multiple_files(
        self,
        pyx_file_paths: Iterable[Path],
        output_dir: Path | None = None,
        dry_run: bool = False,
    ) -> list[ConversionResult]:
        """Convert multiple .pyx files, each possibly merging a companion .pxd file.

        Args:
            pyx_file_paths: Paths to the input .pyx files.
            output_dir: Optional output directory for .pyi files. If None,
                .pyi files are placed next to their source files.
            dry_run: If True, no files are actually created.

        Returns:
            ConversionResult with success status and any error details.
        """
        pyx_paths = list(pyx_file_paths)
        common_root = None
        if output_dir and pyx_paths:
            common_root = Path(os.path.commonpath([str(p.parent) for p in pyx_paths]))

        # One StubgenContext shared across every file in this batch, so a
        # `cimport` in one file resolves to the same, already-parsed
        # module scope as the file it's importing from -- see the design
        # doc, "One shared Context per conversion run". Includes each
        # companion .pxd path too (context_for_paths only needs the .pyx
        # paths in practice, since a .pxd's root package dir is the same
        # as its .pyx sibling's, but passing both is cheap and correct
        # even when they diverge).
        context = context_for_paths(
            [*pyx_paths, *(p.with_suffix(".pxd") for p in pyx_paths)],
            extra_include_dirs=self.config.include_dirs,
        )

        if self.config.resolve_ctypedef_aliases:
            # A ctypedef alias's declaration can only safely be pruned as
            # dead once every *other* file in this same batch has been
            # checked too, not just the one file that declares it -- see
            # `Converter.convert_scope`'s docstring, and
            # `StubgenPyxConfig.resolve_ctypedef_aliases`'s for why this
            # matters in practice (confirmed against a real package).
            return self._convert_multiple_files_with_ctypedef_pruning(
                pyx_paths, context, output_dir, common_root, dry_run
            )

        results: list[ConversionResult] = []
        for pyx_path in pyx_paths:
            pyi_path = self._resolve_pyi_path(pyx_path, output_dir, common_root)
            result = self.convert_single_file(pyx_path, pyi_path, dry_run, _context=context)
            results.append(result)

            if self.config.verbose or not result.success:
                _logger.info(result.status_message)

        return results

    def _convert_multiple_files_with_ctypedef_pruning(
        self,
        pyx_paths: list[Path],
        context: StubgenContext,
        output_dir: Path | None,
        common_root: Path | None,
        dry_run: bool,
    ) -> list[ConversionResult]:
        """`convert_multiple_files`'s batch-aware path for
        `resolve_ctypedef_aliases`, in three phases:

        1. Convert every file to a `PyiModule` (deferring any ctypedef
           -alias-pruning decision -- see `Converter.convert_scope`'s
           docstring) without rendering or writing any of them yet.
        2. For each file's provisionally-dead alias, check every *other*
           successfully-converted file's module in this batch
           (`pyi_module_uses_name`) -- only actually remove it from its
           origin file's assignments once nothing else in the batch
           needs it either.
        3. Render (`_finalize`) and write each file, same as
           `convert_single_file` does for one.

        `continue_on_error`/exception behavior matches
        `convert_single_file` exactly at each file, just spread across
        these phases instead of one contiguous try/except.
        """
        # (pyx_path, pyi_path, early_result | None, converter | None,
        #  module | None, diagnostics | None, prunable | None)
        prepared: list[tuple] = []
        for pyx_path in pyx_paths:
            pyi_path = self._resolve_pyi_path(pyx_path, output_dir, common_root)
            prunable: dict[str, "PyiAssignment"] = {}
            try:
                _logger.debug(f"Converting '{pyx_path}' to '{pyi_path or pyx_path.with_suffix('.pyi')}'")
                early = self._convert_single_file_to_module(
                    pyx_path,
                    context,
                    defer_ctypedef_pruning=True,
                    prunable_ctypedef_aliases=prunable,
                )
                if isinstance(early, ConversionResult):
                    prepared.append((pyx_path, pyi_path, early, None, None, None, None))
                else:
                    converter, module, diagnostics = early
                    prepared.append(
                        (pyx_path, pyi_path, None, converter, module, diagnostics, prunable)
                    )
            except Exception as e:
                _logger.exception(f"Error during conversion: {type(e).__name__}")
                if not self.config.continue_on_error:
                    raise
                early_result = ConversionResult(
                    success=False,
                    pyx_file=pyx_path,
                    pyi_file=pyi_path or pyx_path.with_suffix(".pyi"),
                    error=e,
                )
                prepared.append((pyx_path, pyi_path, early_result, None, None, None, None))

        successful_modules = [entry[4] for entry in prepared if entry[4] is not None]
        for _, _, early_result, _, module, _, prunable in prepared:
            if early_result is not None or not prunable:
                continue
            for alias_name, assignment in prunable.items():
                used_elsewhere = any(
                    other is not module and pyi_module_uses_name(other, alias_name)
                    for other in successful_modules
                )
                if not used_elsewhere:
                    remove_assignment_from_module(module, assignment)

        results: list[ConversionResult] = []
        for pyx_path, pyi_path, early_result, converter, module, diagnostics, _ in prepared:
            if early_result is not None:
                results.append(early_result)
                if self.config.verbose or not early_result.success:
                    _logger.info(early_result.status_message)
                continue

            final_pyi_path = pyi_path or pyx_path.with_suffix(".pyi")
            try:
                _log_diagnostics(diagnostics, pyx_path)
                pyi_content = self._finalize(converter, module, pyx_path)
                if not dry_run:
                    try:
                        final_pyi_path.write_text(pyi_content, encoding="utf-8")
                        _logger.debug(f"Wrote pyi file: {final_pyi_path}")
                    except OSError as e:
                        raise OSError(f"Failed to write {final_pyi_path}: {e}") from e
                else:
                    _logger.info(f"Would create output file: {final_pyi_path}")
                result = ConversionResult(
                    success=True,
                    pyx_file=pyx_path,
                    pyi_file=final_pyi_path,
                    diagnostics=diagnostics,
                )
            except Exception as e:
                _logger.exception(f"Error during conversion: {type(e).__name__}")
                if not self.config.continue_on_error:
                    raise
                result = ConversionResult(
                    success=False, pyx_file=pyx_path, pyi_file=final_pyi_path, error=e
                )
            results.append(result)
            if self.config.verbose or not result.success:
                _logger.info(result.status_message)

        return results

    def _compile_file_with_error_handling(
        self,
        converter: Converter,
        pyx_file_path: Path,
        context: StubgenContext,
        defer_ctypedef_pruning: bool = False,
        prunable_ctypedef_aliases: dict[str, "PyiAssignment"] | None = None,
    ) -> tuple[PyiModule, list[Exception]]:
        """`_compile_file_with_converter`, with the decode-error unwrapping
        `convert_single_file` needs -- factored out so `convert_multiple_files`'s
        deferred-pruning batch path (see `Converter.convert_scope`'s
        docstring) can call it directly too, without going through
        `convert_single_file`'s own immediate render-and-write.
        """
        try:
            return self._compile_file_with_converter(
                converter,
                pyx_file_path,
                context,
                defer_ctypedef_pruning=defer_ctypedef_pruning,
                prunable_ctypedef_aliases=prunable_ctypedef_aliases,
            )
        except UnicodeDecodeError as e:
            raise ValueError(f"File encoding error in {pyx_file_path}: {e}") from e
        except Exception as e:
            # Cython's own `context.parse` (used by `parse_file`,
            # replacing the previous manual file reading -- see
            # `parsing/parser.py`) catches a raw `UnicodeDecodeError`
            # itself and re-raises it wrapped as a
            # `Cython.Compiler.Errors.CompileError`
            # (`Context._report_decode_error`), so it no longer
            # surfaces here as a plain `UnicodeDecodeError` for the
            # clause above to catch. Only unwrap and convert that
            # specific, still-identifiable case (the original
            # `UnicodeDecodeError` survives as `__context__`/
            # `__cause__`); any other `CompileError` (a real syntax
            # error, say) is a different, legitimate failure and
            # re-raised as-is rather than mislabeled as an encoding
            # problem.
            from Cython.Compiler import Errors as _CythonErrors

            cause = e.__cause__ or e.__context__
            if isinstance(e, _CythonErrors.CompileError) and isinstance(
                cause, UnicodeDecodeError
            ):
                raise ValueError(
                    f"File encoding error in {pyx_file_path}: {cause}"
                ) from e
            raise

    def _convert_single_file_to_module(
        self,
        pyx_file_path: Path,
        _context: StubgenContext | None = None,
        defer_ctypedef_pruning: bool = False,
        prunable_ctypedef_aliases: dict[str, "PyiAssignment"] | None = None,
    ) -> ConversionResult | tuple[Converter, PyiModule, list[Exception]]:
        """The part of `convert_single_file` before rendering/writing: the
        existing-file and `__init__`-skip checks, then parse + convert to
        a `PyiModule`. Factored out so `convert_multiple_files`'s
        deferred-pruning batch path (see `Converter.convert_scope`'s
        docstring) can convert every file in a batch before any of them
        are rendered or written, without duplicating this logic.

        Returns a `ConversionResult` directly for the `__init__`-skip
        case (nothing to convert); otherwise `(converter, module,
        diagnostics)` for the caller to finalize and write itself. Raises
        on a real failure, same as `_compile_file_with_error_handling` --
        the caller's own try/except (`continue_on_error` handling) is
        unchanged either way.
        """
        if not pyx_file_path.exists():
            raise ValueError(f"File not found: {pyx_file_path}")

        if (
            pyx_file_path.with_suffix(".py").exists()
            and pyx_file_path.with_suffix(".py").name == "__init__.py"
        ):
            # Skip __init__.pxd/.pyx files with an existing __init__.py
            return ConversionResult(
                success=True, pyx_file=pyx_file_path, pyi_file=pyx_file_path
            )

        context = _context or context_for_paths(
            [pyx_file_path], extra_include_dirs=self.config.include_dirs
        )

        converter = self._make_converter()
        module, diagnostics = self._compile_file_with_error_handling(
            converter,
            pyx_file_path,
            context,
            defer_ctypedef_pruning=defer_ctypedef_pruning,
            prunable_ctypedef_aliases=prunable_ctypedef_aliases,
        )
        return converter, module, diagnostics

    def convert_single_file(
        self,
        pyx_file_path: Path,
        pyi_file_path: Path | None = None,
        dry_run: bool = False,
        _context: StubgenContext | None = None,
    ) -> ConversionResult:
        """Convert a single .pyx file, optionally merging a companion .pxd file.

        Args:
            pyx_file_path: Path to the input .pyx file.
            pyi_file_path: Path to write the output .pyi file. If None,
                defaults to the same location as the .pyx file with .pyi extension.
            dry_run: If True, no files are actually created.
            _context: Internal. A `StubgenContext` shared with other files
                in the same batch (see `convert_multiple_files`). Callers
                using this method standalone don't need to pass one -- a
                fresh, single-file context is created automatically, which
                is sufficient for `cimport`s that don't need cross-file
                resolution within this same batch.

        Returns:
            ConversionResult with success status and any error details.
        """
        pyi_file_path = pyi_file_path or pyx_file_path.with_suffix(".pyi")
        try:
            _logger.debug(f"Converting '{pyx_file_path}' to '{pyi_file_path}'")

            early = self._convert_single_file_to_module(pyx_file_path, _context)
            if isinstance(early, ConversionResult):
                return early
            converter, module, diagnostics = early

            _log_diagnostics(diagnostics, pyx_file_path)
            pyi_content = self._finalize(converter, module, pyx_file_path)

            if not dry_run:
                try:
                    pyi_file_path.write_text(pyi_content, encoding="utf-8")
                    _logger.debug(f"Wrote pyi file: {pyi_file_path}")
                except OSError as e:
                    raise OSError(f"Failed to write {pyi_file_path}: {e}") from e
            else:
                _logger.info(f"Would create output file: {pyi_file_path}")

            return ConversionResult(
                success=True,
                pyx_file=pyx_file_path,
                pyi_file=pyi_file_path,
                diagnostics=diagnostics,
            )

        except Exception as e:
            _logger.exception(f"Error during conversion: {type(e).__name__}")

            if not self.config.continue_on_error:
                raise

            return ConversionResult(
                success=False,
                pyx_file=pyx_file_path,
                pyi_file=pyi_file_path,
                error=e,
            )



def _merge_pxd_into_module(module: PyiModule, pxd_module: PyiModule) -> None:
    """Merge pxd module contents into the pyx module in-place.

    This is a free function rather than a method on PyiModule/PyiScope so that
    the data models stay as pure containers without merge semantics baked in.
    """
    module.scope.enums += pxd_module.scope.enums
    _deduplicate_enums(module.scope)
    module.scope.assignments += pxd_module.scope.assignments
    _deduplicate_assignments(module.scope)
    _merge_classes(module.scope, pxd_module.scope.classes)
    module.imports += pxd_module.imports


def _deduplicate_enums(scope) -> None:
    """Remove duplicate enums from a scope while preserving order.

    A pxd-declared enum is now already visible in the `.pyx`'s own
    conversion (`Converter._convert_declared_entries` walks the merged
    scope's entries directly, since `Context.find_module`'s companion-
    `.pxd` auto-merge puts pxd declarations in the *same* `Symtab.Scope`
    object), so the separate pxd-module conversion this merges in
    would otherwise add it a second time.
    """
    seen: set[str] = set()
    unique: list = []
    for enum in scope.enums:
        if enum.enum_name not in seen:
            seen.add(enum.enum_name)
            unique.append(enum)
    scope.enums = unique


def _deduplicate_assignments(scope) -> None:
    """Remove duplicate assignments from a scope while preserving order."""
    seen: set[str] = set()
    unique: list = []
    for assignment in scope.assignments:
        name = assignment.statement.partition("=")[0].partition(":")[0].strip()
        if not name or name not in seen:
            if name:
                seen.add(name)
            unique.append(assignment)
    scope.assignments = unique


def _merge_classes(scope, extra_classes: list[PyiClass]) -> None:
    """Merge extra classes into scope, combining same-name classes."""
    existing: dict[str, PyiClass] = {cls.name: cls for cls in scope.classes}
    for extra in extra_classes:
        if extra.name in existing:
            _merge_two_classes(existing[extra.name], extra)
        else:
            scope.classes.append(extra)


def _merge_two_classes(target: PyiClass, other: PyiClass) -> None:
    """Merge other into target in-place."""
    if target.doc is None:
        target.doc = other.doc
    target.bases = [*dict.fromkeys(target.bases + other.bases)]
    if target.metaclass is None:
        target.metaclass = other.metaclass
    target.decorators = [*dict.fromkeys(target.decorators + other.decorators)]
    target.keywords = {**target.keywords, **other.keywords}
    target.scope.assignments += other.scope.assignments
    _deduplicate_assignments(target.scope)
    target.scope.functions += other.scope.functions
    _merge_classes(target.scope, other.scope.classes)
    target.scope.enums += other.scope.enums
    _deduplicate_enums(target.scope)
