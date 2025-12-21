from __future__ import annotations

from dataclasses import dataclass, field
import glob
import logging
from pathlib import Path

from .analysis.visitor import ModuleVisitor
from .conversion.converter import Converter
from .builders.builder import Builder
from .parsing.parser import parse_pyx


logger = logging.getLogger(__name__)


@dataclass
class StubgenPyx:
    converter: Converter = field(default_factory=Converter)
    builder: Builder = field(default_factory=Builder)

    def convert_glob(self, pyx_file: str):
        pyx_files = glob.glob(pyx_file, recursive=True)

        for pyx_file in pyx_files:
            pyx_file_path = Path(pyx_file)

            logger.info(f"Converting {pyx_file}")
            parse_result = parse_pyx(pyx_file_path)

            module_visitor = ModuleVisitor(node=parse_result.module_result.source_ast)
            module = self.converter.convert_module(
                module_visitor, parse_result.module_result.source
            )

            if parse_result.pxd_result:
                # Convert extra elements from .pxd
                pxd_visitor = ModuleVisitor(node=parse_result.pxd_result.source_ast)
                pxd_module = self.converter.convert_module(
                    pxd_visitor, parse_result.pxd_result.source
                )

                extra_imports = pxd_module.imports
                extra_enums = pxd_module.scope.enums
            else:
                extra_imports = []
                extra_enums = []

            module.scope.enums += extra_enums
            module.imports += extra_imports

            content = self.builder.build_module(module)
            pyx_file_path.with_suffix(".pyi").write_text(content, encoding="utf-8")
