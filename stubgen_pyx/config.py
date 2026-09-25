"""Configuration for stubgen-pyx code generation."""

import logging
from dataclasses import dataclass, field

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
        fix_scalar_defaults: Coerce str/int literal defaults on bytes/bool
            args (from ``char *``/``bint`` params) to match their annotation
            (default: True).
        exclude_attribution: Skip adding generation attribution comment (default: False).
        continue_on_error: Continue processing files that failed (default: False).
        include_private: Include private members (default: False).
        verbose: Enable verbose logging (default: False).
        include_dirs: Extra directories to search when resolving ``cimport``/
            ``include`` targets that aren't part of the project itself (e.g.
            a third-party ``.pxd``-only package). Each converted file's own
            project root is always searched automatically; this is only for
            paths outside that (default: empty).
        resolve_ctypedef_aliases: Replace a ``ctypedef`` alias
            (``ctypedef double MyFloat``) with its underlying resolved type
            (``float``) in function/method argument and return annotations
            and in class/module attribute declarations, wherever it can be
            resolved -- rather than referencing the alias name itself. A
            ``ctypedef`` has no Python-level binding at runtime at all
            (``from mod import MyFloat`` raises ``ImportError`` even
            though ``mod.pyx`` declares ``ctypedef double MyFloat``), so
            once every usage in the stub
            is substituted this way, the alias's own
            ``MyFloat: TypeAlias = float`` declaration -- which would
            otherwise claim a module-level name that was never really
            importable -- is dropped too, if and only if nothing else
            still needs it. That "nothing else" check is whole-batch
            aware when converting more than one file at once
            (``convert_multiple_files``/``convert_glob``): another file
            in the same run that ``cimport``s the alias is enough to
            keep its declaration, even if that other file's own output
            ends up using the substituted concrete type directly and
            never mentions the alias by name -- confirmed against a
            real multi-file package (an earlier, single-file-only
            version of this check pruned a central "shared type
            aliases" module's declarations out from under files elsewhere
            in the same package that still imported them, producing a
            genuinely broken cross-file reference). A single-file
            conversion (``convert_str``/``compile_str_to_module``, or
            ``convert_single_file`` outside a batch) has no other files
            to check against and keeps the simpler, immediate, local
            -only version of this check -- correct and sufficient for a
            self-contained snippet. This only affects where the alias
            name would otherwise have been *used*. Off by default:
            referencing the alias by name is valid and often more
            meaningful/readable than its expansion, so this is only for
            callers who'd rather see the concrete type everywhere it's
            resolvable (default: False).
        replace_defaults_with_ellipsis: Render every argument default as
            ``...`` instead of its real value (``def f(x: int = ...)``
            rather than ``def f(x: int = 5)``), matching the convention
            most ``.pyi`` stubs use. Applied only at rendering time, after
            a real default of exactly ``None`` has already widened its
            argument's annotation with ``| None`` where applicable, so
            that inference is unaffected either way (default: False).
    """

    sort_imports: bool = True
    trim_imports: bool = True
    pxd_to_stubs: bool = True
    normalize_names: bool = True
    deduplicate_imports: bool = True
    trim_not_defined: bool = True
    fix_scalar_defaults: bool = True
    include_docstrings: bool = True
    exclude_attribution: bool = False
    continue_on_error: bool = False
    include_private: bool = False
    verbose: bool = False
    include_dirs: list[str] = field(default_factory=list)
    resolve_ctypedef_aliases: bool = False
    replace_defaults_with_ellipsis: bool = False

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
