"""Dataclasses representing AST elements used to generate .pyi stub files."""

from __future__ import annotations

from dataclasses import dataclass, field
import re


_IMPORT_RE = re.compile(r"\bcimport\b")


@dataclass(slots=True)
class PyiElement:
    """Base class for all AST elements."""


@dataclass(slots=True)
class PyiArgument(PyiElement):
    """Represents a function argument."""

    name: str
    default: str | None = None
    annotation: str | None = None


@dataclass(slots=True)
class PyiSignature(PyiElement):
    """Represents a function signature."""

    args: list[PyiArgument] = field(default_factory=list)
    return_type: str | None = None
    var_arg: PyiArgument | None = None
    kw_arg: PyiArgument | None = None
    num_posonly_args: int = 0
    num_kwonly_args: int = 0


@dataclass(slots=True)
class PyiFunction(PyiElement):
    """Represents a function or method."""

    name: str
    is_async: bool
    doc: str | None = None
    signature: PyiSignature = field(default_factory=PyiSignature)
    decorators: list[str] = field(default_factory=list)


@dataclass(slots=True)
class PyiStatement(PyiElement):
    """Represents a statement that should be included in the pyi file as-is."""

    statement: str


@dataclass(slots=True)
class PyiAssignment(PyiStatement):
    """Represents an assignment statement that should be included in the pyi file as-is."""


@dataclass(slots=True)
class PyiImport(PyiStatement):
    """Represents an import statement. The `cimport` keyword is replaced with `import`."""

    def __post_init__(self):
        self.statement = _IMPORT_RE.sub("import", self.statement)


@dataclass(slots=True)
class PyiScope(PyiElement):
    """Represents a scope (module or class context)."""

    assignments: list[PyiAssignment] = field(default_factory=list)
    functions: list[PyiFunction] = field(default_factory=list)
    classes: list[PyiClass] = field(default_factory=list)
    enums: list[PyiEnum] = field(default_factory=list)


@dataclass(slots=True)
class PyiClass(PyiElement):
    """Represents a Python class."""

    name: str
    doc: str | None = None
    bases: list[str] = field(default_factory=list)
    metaclass: str | None = None
    decorators: list[str] = field(default_factory=list)
    scope: PyiScope = field(default_factory=PyiScope)


@dataclass(slots=True)
class PyiEnum(PyiElement):
    """Represents a cdef enum."""

    enum_name: str | None
    names: list[str] = field(default_factory=list)


@dataclass(slots=True)
class PyiModule(PyiElement):
    """Represents a Python module."""

    doc: str | None = None
    imports: list[PyiImport] = field(default_factory=list)
    scope: PyiScope = field(default_factory=PyiScope)
