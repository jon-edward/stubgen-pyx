"""Configuration for stubgen-pyx code generation."""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python < 3.11
    import tomli as tomllib

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


def _validate_dotted_name(value: str, field_name: str) -> None:
    if (
        not isinstance(value, str)
        or not value
        or any(not part.isidentifier() for part in value.split("."))
    ):
        raise ValueError(
            f"{field_name} must be a dotted Python name: {value!r}"
        )


@dataclass(frozen=True)
class SymbolOverride:
    """A project-specific rewrite for a generated Cython-facing symbol."""

    source: str
    import_target: str | None = None
    literal: str | None = None
    action: str | None = None

    def __post_init__(self) -> None:
        _validate_dotted_name(self.source, "symbol override source")
        configured_actions = sum(
            value is not None
            for value in (self.import_target, self.literal, self.action)
        )
        if configured_actions != 1:
            raise ValueError(
                f"symbol override {self.source!r} must specify exactly one of "
                "import, literal, or action"
            )
        if self.import_target is not None:
            _validate_dotted_name(self.import_target, "symbol override import")
            if "." not in self.import_target:
                raise ValueError(
                    "symbol override import must include both a module and name: "
                    f"{self.import_target!r}"
                )
        if self.literal is not None:
            try:
                ast.parse(self.literal, mode="eval")
            except SyntaxError as error:
                raise ValueError(
                    f"symbol override literal is not a Python expression: "
                    f"{self.literal!r}"
                ) from error
        if self.action is not None and self.action != "drop":
            raise ValueError(
                f"symbol override action must be 'drop': {self.action!r}"
            )


@dataclass(frozen=True)
class DeclarationOverride:
    """A project-specific replacement for a generated public declaration."""

    target: str
    declarations: str

    def __post_init__(self) -> None:
        _validate_dotted_name(self.target, "declaration override target")
        try:
            tree = ast.parse(self.declarations)
        except SyntaxError as error:
            raise ValueError(
                "declaration override declarations are not valid Python: "
                f"{self.target!r}"
            ) from error

        if not tree.body or not all(
            isinstance(statement, ast.FunctionDef) for statement in tree.body
        ):
            raise ValueError(
                "declaration override declarations must contain only function "
                f"declarations: {self.target!r}"
            )

        expected_name = self.target.rsplit(".", 1)[-1]
        if any(statement.name != expected_name for statement in tree.body):
            raise ValueError(
                "declaration override declaration names must match target "
                f"{self.target!r}"
            )

        if len(tree.body) > 1 and any(
            not any(
                isinstance(decorator, ast.Attribute)
                and isinstance(decorator.value, ast.Name)
                and decorator.value.id == "typing"
                and decorator.attr == "overload"
                for decorator in statement.decorator_list
            )
            for statement in tree.body
        ):
            raise ValueError(
                "multiple declaration override declarations must all be "
                f"typing.overload overloads: {self.target!r}"
            )


@dataclass(frozen=True)
class SymbolOverridesConfig:
    """Overrides loaded from a symbol override TOML file."""

    module_root: str
    overrides: tuple[SymbolOverride, ...]
    declaration_overrides: tuple[DeclarationOverride, ...]

    def __post_init__(self) -> None:
        _validate_dotted_name(self.module_root, "module_root")
        sources = [override.source for override in self.overrides]
        duplicates = sorted(
            source for source in set(sources) if sources.count(source) > 1
        )
        if duplicates:
            raise ValueError(
                "symbol override sources must be unique: "
                + ", ".join(duplicates)
            )
        declaration_targets = [
            override.target for override in self.declaration_overrides
        ]
        duplicate_targets = sorted(
            target
            for target in set(declaration_targets)
            if declaration_targets.count(target) > 1
        )
        if duplicate_targets:
            raise ValueError(
                "declaration override targets must be unique: "
                + ", ".join(duplicate_targets)
            )


def load_symbol_overrides(path: Path) -> SymbolOverridesConfig:
    """Load and validate a TOML symbol override file."""
    try:
        with path.open("rb") as file:
            data = tomllib.load(file)
    except OSError as error:
        raise ValueError(
            f"could not read symbol overrides file {path}: {error}"
        ) from error
    except tomllib.TOMLDecodeError as error:
        raise ValueError(
            f"could not parse symbol overrides file {path}: {error}"
        ) from error

    if not isinstance(data, dict):
        raise ValueError("symbol overrides file must contain a TOML table")
    module_root = data.get("module_root")
    if not isinstance(module_root, str):
        raise ValueError(
            "symbol overrides file must define string module_root"
        )
    raw_overrides = data.get("symbol_overrides", [])
    if not isinstance(raw_overrides, list):
        raise ValueError("symbol_overrides must be an array of tables")

    overrides = []
    for index, raw_override in enumerate(raw_overrides, start=1):
        if not isinstance(raw_override, dict):
            raise ValueError(f"symbol_overrides[{index}] must be a table")
        unexpected = set(raw_override) - {
            "source",
            "import",
            "literal",
            "action",
        }
        if unexpected:
            raise ValueError(
                f"symbol_overrides[{index}] has unexpected keys: "
                + ", ".join(sorted(unexpected))
            )
        try:
            overrides.append(
                SymbolOverride(
                    source=raw_override["source"],
                    import_target=raw_override.get("import"),
                    literal=raw_override.get("literal"),
                    action=raw_override.get("action"),
                )
            )
        except KeyError as error:
            raise ValueError(
                f"symbol_overrides[{index}] must define source"
            ) from error

    raw_declaration_overrides = data.get("declaration_overrides", [])
    if not isinstance(raw_declaration_overrides, list):
        raise ValueError("declaration_overrides must be an array of tables")

    declaration_overrides = []
    for index, raw_override in enumerate(raw_declaration_overrides, start=1):
        if not isinstance(raw_override, dict):
            raise ValueError(f"declaration_overrides[{index}] must be a table")
        unexpected = set(raw_override) - {"target", "declarations"}
        if unexpected:
            raise ValueError(
                f"declaration_overrides[{index}] has unexpected keys: "
                + ", ".join(sorted(unexpected))
            )
        try:
            declaration_overrides.append(
                DeclarationOverride(
                    target=raw_override["target"],
                    declarations=raw_override["declarations"],
                )
            )
        except KeyError as error:
            raise ValueError(
                f"declaration_overrides[{index}] must define target and declarations"
            ) from error

    return SymbolOverridesConfig(
        module_root=module_root,
        overrides=tuple(overrides),
        declaration_overrides=tuple(declaration_overrides),
    )


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
        module_root: Package root used to resolve relative imports in overrides.
        source_root: Directory containing modules under ``module_root``.
        symbol_overrides: Explicit project-specific symbol rewrite rules.
        declaration_overrides: Explicit replacements for generated declarations.
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
    module_root: str | None = None
    source_root: Path | None = None
    symbol_overrides: tuple[SymbolOverride, ...] = field(default_factory=tuple)
    declaration_overrides: tuple[DeclarationOverride, ...] = field(
        default_factory=tuple
    )

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

        if self.module_root is not None:
            _validate_dotted_name(self.module_root, "module_root")
