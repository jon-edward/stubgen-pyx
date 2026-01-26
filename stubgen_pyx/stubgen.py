"""StubgenPyx converts .pyx files to .pyi files."""

from __future__ import annotations

import itertools
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
class StubgenPyx:
    """StubgenPyx converts .pyx files to .pyi files."""

    converter: Converter = field(default_factory=Converter)
    """Converter converts Visitors to PyiElements."""

    builder: Builder = field(default_factory=Builder)
    """Builder builds .pyi files from PyiElements."""

    config: StubgenPyxConfig = field(default_factory=StubgenPyxConfig)
    """Configuration for StubgenPyx."""

    def convert_str(self, pyx_str: str, pxd_str: str | None = None, pyx_path: Path | None = None):
        """Converts a .pyx file to a .pyi file."""
        module_name = path_to_module_name(pyx_path) if pyx_path else None
        parse_result = parse_pyx(pyx_str, module_name=module_name, pyx_path=pyx_path)

        module_visitor = ModuleVisitor(node=parse_result.source_ast)
        module = self.converter.convert_module(
            module_visitor, parse_result.source
        )

        if pxd_str and not self.config.no_pxd_to_stubs:
            # Convert extra elements from .pxd
            pxd_parse_result = parse_pyx(pxd_str, module_name=module_name, pyx_path=pyx_path)
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


    def convert_glob(self, pyx_file_pattern: str):
        """Converts a glob pattern of .pyx files to .pyi files."""

        pyx_files = glob.glob(pyx_file_pattern, recursive=True)

        for pyx_file in pyx_files:
            try:
                logger.info(f"Converting {pyx_file}")
                
                pyx_file_path = Path(pyx_file)

                pxd_file_path = pyx_file_path.with_suffix(".pxd")
                if pxd_file_path.exists() and not self.config.no_pxd_to_stubs and pxd_file_path != pyx_file_path:
                    logger.debug(f"Found pxd file: {pxd_file_path}")
                    pxd_str = pxd_file_path.read_text(encoding="utf-8")
                else:
                    pxd_str = None

                pyx_file_path.with_suffix(".pyi").write_text(
                    self.convert_str(
                        pyx_str=pyx_file_path.read_text(encoding="utf-8"),
                        pxd_str=pxd_str,
                        pyx_path=pyx_file_path,
                    ),
                    encoding="utf-8",
                )
            except Exception as e:
                logger.error(f"Failed to convert {pyx_file}: {e}")
                if not self.config.continue_on_error:
                    raise
