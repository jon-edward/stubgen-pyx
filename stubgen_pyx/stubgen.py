"""StubgenPyx converts .pyx files to .pyi files."""

from __future__ import annotations

from dataclasses import dataclass, field
import glob
import logging
from typing import Iterable
from pathlib import Path

from .config import StubgenPyxConfig
from .analysis.visitor import ModuleVisitor
from .conversion.converter import Converter
from .builders.builder import Builder
from .parsing.parser import parse_pyx, path_to_module_name
from .postprocessing.pipeline import postprocessing_pipeline
from .models.pyi_elements import PyiImport, PyiModule


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
            return f"Converted {self.pyx_file} to {self.pyi_file}"
        return f"Failed to convert {self.pyx_file}: {self.error}"


@dataclass
class StubgenPyx:
    """Primary API for converting Cython .pyx files to .pyi stub files.

    Parses Cython source, extracts types, and generates type stubs with optional
    postprocessing (import normalization, trimming, etc.).

    Attributes:
        converter: Converts Cython AST visitors to PyiElements.
        builder: Builds .pyi text from PyiElements.
        config: Configuration controlling generation behavior.
    """

    converter: Converter = field(default_factory=Converter)
    builder: Builder = field(default_factory=Builder)
    config: StubgenPyxConfig = field(default_factory=StubgenPyxConfig)

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
        content = self.builder.build_module(module)
        return postprocessing_pipeline(content, self.config, pyx_path).strip()

    def compile_str_to_module(
        self, pyx_str: str, pxd_str: str | None = None, pyx_path: Path | None = None
    ) -> PyiModule:
        """Compile Cython source strings into a PyiModule

        Args:
            pyx_str: The source Cython code.
            pxd_str: Optional companion .pxd file content to merge.
            pyx_path: Optional file path for context and error messages.

        Returns:
            PyiModule instance as the result of the compilation.

        Raises:
            Various exceptions from parsing, conversion, or building.
        """
        self.builder.include_private = self.config.include_private

        module_name = path_to_module_name(pyx_path) if pyx_path else None
        parse_result = parse_pyx(pyx_str, module_name=module_name, pyx_path=pyx_path)

        module_visitor = ModuleVisitor(node=parse_result.source_ast)
        module = self.converter.convert_module(module_visitor, parse_result.source)

        if pxd_str and not self.config.no_pxd_to_stubs:
            pxd_parse_result = parse_pyx(
                pxd_str, module_name=module_name, pyx_path=pyx_path
            )
            pxd_visitor = ModuleVisitor(node=pxd_parse_result.source_ast)
            pxd_module = self.converter.convert_module(
                pxd_visitor, pxd_parse_result.source
            )

            extra_imports = pxd_module.imports
            extra_enums = pxd_module.scope.enums
            extra_classes = pxd_module.scope.classes
            extra_assignments = pxd_module.scope.assignments
        else:
            extra_imports = []
            extra_enums = []
            extra_classes = []
            extra_assignments = []

        module.scope.enums += extra_enums
        module.scope.assignments += extra_assignments
        module.imports += extra_imports

        module_class_names = {
            class_element.name for class_element in module.scope.classes
        }

        module.scope.classes += [
            extra_class
            for extra_class in extra_classes
            if extra_class.name not in module_class_names
        ]

        module.imports.append(
            PyiImport(
                statement="from __future__ import annotations",
            )
        )

        return module

    def resolve_glob(self, pyx_file_pattern: str) -> Iterable[Path]:
        """Resolve given glob pattern.

        Args:
            pyx_file_pattern: Glob pattern (e.g., "**/*.pyx", "src/*.pyx").

        Returns:
            Iterable of Path to the resolved file names
        """

        # Note: Path.glob is not suitable with patterns containing absolute
        # paths, so glob is much better here
        pyx_files = glob.glob(pyx_file_pattern, recursive=True)

        if len(pyx_files) == 0:
            logger.warning(f"No files matched pattern: {pyx_file_pattern}")
        else:
            logger.info(f"Found {len(pyx_files)} file(s) to convert")
        return (Path(p) for p in pyx_files)

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

        Reads the .pyx files and companions .pxd (if they exist), converts them,
        and writes the resulting .pyi file.

        Args:
            pyx_file_paths: Paths to the input .pyx files.
            output_dir: Optional output directory for .pyi files. If None,
                .pyi files are placed next to their source files.
            dry_run: If True, no files are actually created.

        Returns:
            ConversionResult with success status and any error details.
        """
        results: list[ConversionResult] = []

        for pyx_path in pyx_file_paths:
            pyi_path = (
                output_dir / pyx_path.with_suffix(".pyi").name if output_dir else None
            )
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

        Reads the .pyx file and companion .pxd (if it exists), converts them,
        and writes the resulting .pyi file.

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

            pxd_str = None
            pxd_file_path = pyx_file_path.with_suffix(".pxd")
            if (
                pxd_file_path.exists()
                and not self.config.no_pxd_to_stubs
                and pxd_file_path != pyx_file_path
            ):
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
