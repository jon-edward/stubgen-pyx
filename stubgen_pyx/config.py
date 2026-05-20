"""Configuration for stubgen-pyx code generation."""

import logging
from dataclasses import dataclass


logger = logging.getLogger(__name__)


@dataclass
class StubgenPyxConfig:
    """Options controlling .pyi generation behavior.

    Attributes:
        sort_imports: Sort imports in the output (default: True).
        trim_imports: Remove unused imports (default: True).
        pxd_to_stubs: Merge .pxd file contents into stubs (default: True).
        normalize_names: Normalize Cython type names to Python equivalents (default: True).
        deduplicate_imports: Remove duplicate imports (default: True).
        trim_not_defined: Replace undefined names with ``...`` (default: True).
        exclude_attribution: Skip adding generation attribution comment (default: False).
        continue_on_error: Continue processing files that failed (default: False).
        include_private: Include private members (default: False).
        verbose: Enable verbose logging (default: False).
    """

    sort_imports: bool = True
    trim_imports: bool = True
    pxd_to_stubs: bool = True
    normalize_names: bool = True
    deduplicate_imports: bool = True
    trim_not_defined: bool = True
    exclude_attribution: bool = False
    continue_on_error: bool = False
    include_private: bool = False
    verbose: bool = False

    def __post_init__(self):
        """Validate configuration and log warnings for unusual settings."""
        if not any(
            [
                self.sort_imports,
                self.trim_imports,
                self.normalize_names,
                self.deduplicate_imports,
                self.trim_not_defined,
            ]
        ):
            logger.warning(
                "All postprocessing steps are disabled. Output may be verbose."
            )

        if self.continue_on_error:
            logger.info("Continuing on errors - failed files will be skipped")
