"""StubgenPyx converts .pyx files to .pyi files."""

from __future__ import annotations

from dataclasses import dataclass, field
import glob
import logging
from pathlib import Path

from .config import StubgenPyxConfig
from .analysis.visitor import ModuleVisitor
from .conversion.converter import Converter
from .builders.builder import Builder
from .parsing.parser import parse_pyx, path_to_module_name
from .postprocessing.pipeline import postprocessing_pipeline
from .models.pyi_elements import PyiImport
from ._version import __version__


logger = logging.getLogger(__name__)


@dataclass
class ConversionResult:
    """Result of a file conversion operation."""

    success: bool
    """Whether the conversion succeeded."""

    pyx_file: Path
    """Path to the source .pyx file."""

    pyi_file: Path
    """Path to the generated .pyi file."""

    error: Exception | None = None
    """Exception if conversion failed, None otherwise."""

    @property
    def status_message(self) -> str:
        """Human-readable status message."""
        if self.success:
            return f"Converted {self.pyx_file} to {self.pyi_file}"
        return f"Failed to convert {self.pyx_file}: {self.error}"


@dataclass
class StubgenPyx:
    """
    StubgenPyx converts .pyx files to .pyi files.

    This tool parses Cython source files, extracts type information,
    and generates Python stub files (.pyi) for type checking.
    """

    converter: Converter = field(default_factory=Converter)
    """Converter converts Visitors to PyiElements."""

    builder: Builder = field(default_factory=Builder)
    """Builder builds .pyi files from PyiElements."""

    config: StubgenPyxConfig = field(default_factory=StubgenPyxConfig)
    """Configuration for StubgenPyx."""

    def convert_str(
        self, pyx_str: str, pxd_str: str | None = None, pyx_path: Path | None = None
    ) -> str:
        """
        Converts a .pyx file string to a .pyi file string.

        Args:
            pyx_str: The source Cython code
            pxd_str: Optional companion .pxd file content
            pyx_path: Optional path for better error messages and module names

        Returns:
            The generated .pyi stub file content

        Raises:
            Various exceptions from parsing, conversion, or building steps
        """
        module_name = path_to_module_name(pyx_path) if pyx_path else None
        parse_result = parse_pyx(pyx_str, module_name=module_name, pyx_path=pyx_path)

        module_visitor = ModuleVisitor(node=parse_result.source_ast)
        module = self.converter.convert_module(module_visitor, parse_result.source)

        if pxd_str and not self.config.no_pxd_to_stubs:
            # Convert extra elements from .pxd
            pxd_parse_result = parse_pyx(
                pxd_str, module_name=module_name, pyx_path=pyx_path
            )
            pxd_visitor = ModuleVisitor(node=pxd_parse_result.source_ast)
            pxd_module = self.converter.convert_module(
                pxd_visitor, pxd_parse_result.source
            )

            extra_imports = pxd_module.imports
            extra_enums = pxd_module.scope.enums
        else:
            extra_imports = []
            extra_enums = []

        module.scope.enums += extra_enums
        module.imports += extra_imports

        module.imports.append(
            PyiImport(
                statement=f"from __future__ import annotations",
            )
        )

        content = self.builder.build_module(module)
        return postprocessing_pipeline(content, self.config, pyx_path).strip()

    def convert_glob(self, pyx_file_pattern: str) -> list[ConversionResult]:
        """
        Converts a glob pattern of .pyx files to .pyi files.

        Args:
            pyx_file_pattern: Glob pattern for files to convert (e.g., "**/*.pyx")

        Returns:
            List of ConversionResult objects with status for each file
        """
        pyx_files = glob.glob(pyx_file_pattern, recursive=True)
        results: list[ConversionResult] = []

        if not pyx_files:
            logger.warning(f"No files matched pattern: {pyx_file_pattern}")
            return results

        logger.info(f"Found {len(pyx_files)} file(s) to convert")

        for pyx_file in pyx_files:
            result = self._convert_single_file(Path(pyx_file))
            results.append(result)

            if self.config.verbose or not result.success:
                logger.info(result.status_message)

        return results

    def _convert_single_file(self, pyx_file_path: Path) -> ConversionResult:
        """
        Convert a single .pyx file to .pyi format.

        Args:
            pyx_file_path: Path to the .pyx file

        Returns:
            ConversionResult with success status and any errors
        """
        try:
            logger.debug(f"Converting {pyx_file_path}")

            # Read pyx file
            try:
                pyx_str = pyx_file_path.read_text(encoding="utf-8")
            except UnicodeDecodeError as e:
                raise ValueError(f"File encoding error in {pyx_file_path}: {e}") from e
            except FileNotFoundError as e:
                raise ValueError(f"File not found: {pyx_file_path}") from e

            # Try to read companion .pxd file
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

            # Convert to pyi
            pyi_content = self.convert_str(
                pyx_str=pyx_str,
                pxd_str=pxd_str,
                pyx_path=pyx_file_path,
            )

            # Write pyi file
            pyi_file_path = pyx_file_path.with_suffix(".pyi")
            try:
                pyi_file_path.write_text(pyi_content, encoding="utf-8")
            except IOError as e:
                raise IOError(f"Failed to write {pyi_file_path}: {e}") from e

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
                pyi_file=pyx_file_path.with_suffix(".pyi"),
                error=e,
            )
