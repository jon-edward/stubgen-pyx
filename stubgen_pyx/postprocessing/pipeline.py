"""Postprocessing pipeline."""

from __future__ import annotations

import ast
import logging
from pathlib import Path

from ..config import StubgenPyxConfig
from .collect_names import collect_names
from .normalize_names import normalize_names
from .sort_imports import sort_imports
from .epilog import epilog

logger = logging.getLogger(__name__)


def postprocessing_pipeline(
    pyi_code: str, config: StubgenPyxConfig, pyx_path: Path | None = None
) -> str:
    """
    Apply postprocessing transformations to generated .pyi code.

    This pipeline optimizes by combining multiple AST passes where possible,
    reducing the number of times the tree is traversed.

    Args:
        pyi_code: The generated .pyi code to postprocess
        config: Configuration options for postprocessing
        pyx_path: Optional path to the source .pyx file for epilog

    Returns:
        The postprocessed .pyi code as a string
    """
    pyi_ast = ast.parse(pyi_code)

    # Collect names for trimming (must be done before normalization alters names)
    used_names = collect_names(pyi_ast) if not config.no_trim_imports else None

    # Combine import-related operations into a single transformation pass
    if (
        not config.no_deduplicate_imports
        or not config.no_trim_imports
        or not config.no_normalize_names
    ):
        pyi_ast = _combined_import_transform(
            pyi_ast,
            trim_unused=not config.no_trim_imports,
            used_names=used_names,
            normalize=not config.no_normalize_names,
            deduplicate=not config.no_deduplicate_imports,
        )
    else:
        # Still apply normalization if imports aren't being touched
        if not config.no_normalize_names:
            pyi_ast = normalize_names(pyi_ast)

    pyi_code = ast.unparse(pyi_ast)

    if not config.no_sort_imports:
        pyi_code = sort_imports(pyi_code)

    if not config.exclude_epilog:
        pyi_code = f"{pyi_code}\n\n{epilog(pyx_path)}"

    return pyi_code


def _combined_import_transform(
    tree: ast.AST,
    trim_unused: bool = True,
    used_names: set[str] | None = None,
    normalize: bool = True,
    deduplicate: bool = True,
) -> ast.AST:
    """
    Combine multiple import-related AST transformations into a single pass.

    This reduces tree traversal overhead when multiple transformations are needed.
    """
    from .normalize_names import _NameNormalizer
    from .trim_imports import _UnusedImportRemover
    from .deduplicate_imports import _DuplicateImportRemover

    # Apply transformations in the most efficient order
    if deduplicate:
        tree = _DuplicateImportRemover().visit(tree)

    if trim_unused and used_names is not None:
        tree = _UnusedImportRemover(used_names).visit(tree)

    if normalize:
        tree = _NameNormalizer().visit(tree)

    return tree
