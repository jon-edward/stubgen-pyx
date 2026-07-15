"""StubgenPyx converts .pyx files to .pyi files."""

from __future__ import annotations

from dataclasses import dataclass, field
import glob
import logging
import os
from typing import Iterable
from pathlib import Path

from .config import StubgenPyxConfig
from .analysis.visitor import ModuleVisitor
from .conversion.converter import Converter
from .builders.builder import Builder
from .parsing.parser import parse_pyx, path_to_module_name
from .postprocessing.pipeline import postprocessing_pipeline
from .models.pyi_elements import PyiModule, PyiClass


logger = logging.getLogger(__name__)


@dataclass
class ConversionResult:
    """Result of converting a single .pyx file to .pyi.

    Attributes:
        success: Whether the conversion completed without error.
        pyx_file: Path to the source .pyx file.
        pyi_file: Path to the generated .pyi file.
        error: Exception if conversion failed, otherwise None.
    """

    success: bool
    pyx_file: Path
    pyi_file: Path
    error: Exception | None = None

    @property
    def status_message(self) -> str:
        """Human-readable status summary."""
        if self.success:
            if self.pyx_file != self.pyi_file:
                return f"Converted {self.pyx_file} to {self.pyi_file}"
            else:
                return f"Skipped {self.pyx_file}"
        return f"Failed to convert {self.pyx_file}: {self.error}"


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
        return Builder(include_private=self.config.include_private)

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
        module = self.compile_str_to_module(pyx_str, pxd_str, pyx_path)
        builder = self._make_builder()
        content = builder.build_module(module)
        return postprocessing_pipeline(content, self.config, pyx_path).strip() + "\n"

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
        converter = self._make_converter()

        module_name = path_to_module_name(pyx_path) if pyx_path else None
        # Full fused type support including cross-file inheritance would require
        # Cython's own scope/env for symbol resolution, which is not currently available
        # in this generator. Making that available would require a significant refactor,
        # so in lieu of that we restrict the code to only handle the companion .pxd file
        # (same stem) by parsing it as a separate pre-pass and merging its fused
        # typedefs into the .pyx conversion. This is a pragmatic compromise that covers
        # the majority of real-world use cases. Fused typedefs made visible via
        # `cimport` from an unrelated module are not resolved through this path.
        pxd_parse_result = None
        pxd_visitor = None
        pxd_fused_types = None
        if pxd_str and self.config.pxd_to_stubs:
            pxd_parse_result = parse_pyx(
                pxd_str, module_name=module_name, pyx_path=pyx_path, pxd=True
            )
            pxd_visitor = ModuleVisitor(node=pxd_parse_result.source_ast)
            pxd_fused_types = converter.convert_fused_types(
                pxd_visitor.scope.fused_types
            )

        parse_result = parse_pyx(pyx_str, module_name=module_name, pyx_path=pyx_path)

        module_visitor = ModuleVisitor(node=parse_result.source_ast)
        module = converter.convert_module(
            module_visitor,
            parse_result.source,
            parse_result.type_comments,
            include_docstrings=self.config.include_docstrings,
            inherited_fused_types=pxd_fused_types,
        )

        if pxd_parse_result is not None and pxd_visitor is not None:
            pxd_module = converter.convert_module(
                pxd_visitor,
                pxd_parse_result.source,
                pxd_parse_result.type_comments,
                include_docstrings=self.config.include_docstrings,
            )
            _merge_pxd_into_module(module, pxd_module)

        return module

    def resolve_glob(self, pyx_file_pattern: str) -> Iterable[Path]:
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

        if len(file_paths) == 0:
            logger.warning(f"No files matched pattern: {pyx_file_pattern}")
        else:
            logger.info(f"Found {len(file_paths)} file(s) to convert")

        unique_paths = list(dict.fromkeys(file_paths))
        return (Path(p) for p in unique_paths)

    def convert_glob(
        self,
        pyx_file_pattern: str,
        output_dir: Path | None = None,
        dry_run: bool = False,
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
        pyx_files = self.resolve_glob(pyx_file_pattern)
        return self.convert_multiple_files(
            pyx_files, output_dir=output_dir, dry_run=dry_run
        )

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
        results: list[ConversionResult] = []

        pyx_paths = list(pyx_file_paths)
        common_root = None
        if output_dir and pyx_paths:
            common_root = Path(os.path.commonpath([str(p.parent) for p in pyx_paths]))

        for pyx_path in pyx_paths:
            if output_dir:
                # place pyi files in the same dir structure as the source pyx files
                # relative to the common root of the pyx files
                pyi_name = pyx_path.with_suffix(".pyi")
                if common_root is not None:
                    try:
                        pyi_path = output_dir / pyi_name.relative_to(common_root)
                    except ValueError:
                        pyi_path = output_dir / pyi_name.name
                else:
                    pyi_path = output_dir / pyi_name.name
                pyi_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                pyi_path = None  # generate in-place
            result = self.convert_single_file(pyx_path, pyi_path, dry_run)
            results.append(result)

            if self.config.verbose or not result.success:
                logger.info(result.status_message)

        return results

    def convert_single_file(
        self,
        pyx_file_path: Path,
        pyi_file_path: Path | None = None,
        dry_run: bool = False,
    ) -> ConversionResult:
        """Convert a single .pyx file, optionally merging a companion .pxd file.

        Args:
            pyx_file_path: Path to the input .pyx file.
            pyi_file_path: Path to write the output .pyi file. If None,
                defaults to the same location as the .pyx file with .pyi extension.
            dry_run: If True, no files are actually created.

        Returns:
            ConversionResult with success status and any error details.
        """
        pyi_file_path = pyi_file_path or pyx_file_path.with_suffix(".pyi")
        try:
            logger.debug(f"Converting '{pyx_file_path}' to '{pyi_file_path}'")

            try:
                pyx_str = pyx_file_path.read_text(encoding="utf-8")
            except UnicodeDecodeError as e:
                raise ValueError(f"File encoding error in {pyx_file_path}: {e}") from e
            except FileNotFoundError as e:
                raise ValueError(f"File not found: {pyx_file_path}") from e

            if (
                pyx_file_path.with_suffix(".py").exists()
                and pyx_file_path.with_suffix(".py").name == "__init__.py"
            ):
                # Skip __init__.pxd/.pyx files with an existing __init__.py
                return ConversionResult(
                    success=True, pyx_file=pyx_file_path, pyi_file=pyx_file_path
                )

            pxd_str = None
            if self.config.pxd_to_stubs:
                pxd_file_path = pyx_file_path.with_suffix(".pxd")
                if pxd_file_path.exists() and pxd_file_path != pyx_file_path:
                    logger.debug(f"Found pxd file: {pxd_file_path}")
                    try:
                        pxd_str = pxd_file_path.read_text(encoding="utf-8")
                    except UnicodeDecodeError as e:
                        logger.warning(f"Could not read .pxd file {pxd_file_path}: {e}")

            pyi_content = self.convert_str(
                pyx_str=pyx_str,
                pxd_str=pxd_str,
                pyx_path=pyx_file_path,
            )

            if not dry_run:
                try:
                    pyi_file_path.write_text(pyi_content, encoding="utf-8")
                    logger.debug(f"Wrote pyi file: {pyi_file_path}")
                except IOError as e:
                    raise IOError(f"Failed to write {pyi_file_path}: {e}") from e
            else:
                logger.info(f"Would create output file: {pyi_file_path}")

            return ConversionResult(
                success=True,
                pyx_file=pyx_file_path,
                pyi_file=pyi_file_path,
            )

        except Exception as e:
            logger.debug(
                f"Error during conversion: {type(e).__name__}: {e}", exc_info=True
            )

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
    module.scope.assignments += pxd_module.scope.assignments
    _deduplicate_assignments(module.scope)
    _merge_classes(module.scope, pxd_module.scope.classes)
    module.imports += pxd_module.imports


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
