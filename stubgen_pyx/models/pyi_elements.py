"""
This module contains the dataclasses that represent the elements of the AST
that are used to generate the pyi file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re


@dataclass
class PyiElement:
    """
    Base class for all elements of the AST.
    """


@dataclass
class PyiArgument(PyiElement):
    """Represents a function argument."""

    name: str
    default: str | None = None
    annotation: str | None = None


@dataclass
class PyiSignature(PyiElement):
    """Represents a function signature."""

    args: list[PyiArgument] = field(default_factory=list)
    return_type: str | None = None
    var_arg: PyiArgument | None = None
    kw_arg: PyiArgument | None = None
    num_posonly_args: int = 0
    num_kwonly_args: int = 0


@dataclass
class PyiFunction(PyiElement):
    """Represents a function or method."""

    name: str
    is_async: bool
    doc: str | None = None
    signature: PyiSignature = field(default_factory=PyiSignature)
    decorators: list[str] = field(default_factory=list)


@dataclass
class PyiStatement(PyiElement):
    """Represents a statement that should be included in the pyi file as-is."""

    statement: str


@dataclass
class PyiAssignment(PyiStatement):
    """Represents an assignment statement that should be included in the pyi file as-is."""


@dataclass
class PyiImport(PyiStatement):
    """Represents an import statement. The `cimport` keyword is replaced with `import`."""

    def __post_init__(self):
        self.statement = re.sub(r"\bcimport\b", "import", self.statement)


@dataclass
class PyiScope(PyiElement):
    """Represents a scope (module or class context)."""

    assignments: list[PyiAssignment] = field(default_factory=list)
    functions: list[PyiFunction] = field(default_factory=list)
    classes: list[PyiClass] = field(default_factory=list)
    enums: list[PyiEnum] = field(default_factory=list)


@dataclass
class PyiClass(PyiElement):
    """Represents a Python class."""

    name: str
    doc: str | None = None
    bases: list[str] = field(default_factory=list)
    metaclass: str | None = None
    decorators: list[str] = field(default_factory=list)
    scope: PyiScope = field(default_factory=PyiScope)


@dataclass
class PyiEnum(PyiElement):
    """Represents a cdef enum."""

    enum_name: str | None
    names: list[str] = field(default_factory=list)


@dataclass
class PyiModule(PyiElement):
    """Represents a Python module."""

    doc: str | None = None
    imports: list[PyiImport] = field(default_factory=list)
    scope: PyiScope = field(default_factory=PyiScope)
