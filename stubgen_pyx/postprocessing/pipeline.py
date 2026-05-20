"""Post-processing pipeline for generated .pyi files.

Applies transformations like import normalization, trimming, and sorting.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path

from ..config import StubgenPyxConfig
from .collect_names import collect_names
from .normalize_names import _NameNormalizer
from .trim_imports import _UnusedImportRemover
from .deduplicate_imports import _DuplicateImportRemover
from .trim_not_defined import trim_not_defined
from .sort_imports import sort_imports
from .attribution import stubgen_attribution
from .remove_identity_assignment import remove_identity_assignment

logger = logging.getLogger(__name__)


def postprocessing_pipeline(
    pyi_code: str, config: StubgenPyxConfig, pyx_path: Path | None = None
) -> str:
    """Apply post-processing transformations to .pyi code.

    Args:
        pyi_code: Generated .pyi code to postprocess.
        config: Configuration options for processing.
        pyx_path: Optional source file path for stubgen attribution comments.

    Returns:
        Processed .pyi code after all enabled transformations.
    """
    pyi_ast = ast.parse(pyi_code, type_comments=True)
    pyi_ast = _ast_transforms(pyi_ast, config)
    pyi_code = ast.unparse(pyi_ast)

    if config.sort_imports:
        pyi_code = sort_imports(pyi_code)

    if not config.exclude_attribution:
        pyi_code = f"{stubgen_attribution(pyx_path)}\n{pyi_code}"

    return pyi_code


def _ast_transforms(tree: ast.AST, config: StubgenPyxConfig) -> ast.AST:
    """Apply all enabled AST-level transforms in the correct order."""
    if config.deduplicate_imports:
        tree = _DuplicateImportRemover().visit(tree)

    if config.normalize_names:
        tree = _NameNormalizer().visit(tree)

    if config.trim_imports:
        used_names = collect_names(tree)
        tree = _UnusedImportRemover(used_names).visit(tree)

    if config.trim_not_defined:
        trim_not_defined(tree)

    tree = remove_identity_assignment(tree)
    return tree
