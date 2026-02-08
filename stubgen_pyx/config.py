"""Configuration for stubgen-pyx code generation."""

import logging
from dataclasses import dataclass


logger = logging.getLogger(__name__)


@dataclass
class StubgenPyxConfig:
    """Options controlling .pyi generation behavior.

    Attributes:
        no_sort_imports: Skip import sorting.
        no_trim_imports: Skip removing unused imports.
        no_pxd_to_stubs: Skip merging .pxd file contents.
        no_normalize_names: Skip normalizing Cython type names.
        no_deduplicate_imports: Skip deduplicating imports.
        exclude_epilog: Skip adding generation epilog comment.
        continue_on_error: Continue processing files that failed.
        verbose: Enable verbose logging.
    """

    no_sort_imports: bool = False
    no_trim_imports: bool = False
    no_pxd_to_stubs: bool = False
    no_normalize_names: bool = False
    no_deduplicate_imports: bool = False
    exclude_epilog: bool = False
    continue_on_error: bool = False
    verbose: bool = False

    def __post_init__(self):
        """Validate configuration and log warnings for unusual settings."""
        if all(
            [
                self.no_sort_imports,
                self.no_trim_imports,
                self.no_normalize_names,
                self.no_deduplicate_imports,
            ]
        ):
            logger.warning(
                "All postprocessing steps are disabled. Output may be verbose."
            )

        if self.continue_on_error:
            logger.info("Continuing on errors - failed files will be skipped")
