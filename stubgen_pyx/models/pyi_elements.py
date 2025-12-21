"""
This module contains the dataclasses that represent the elements of the AST
that are used to generate the pyi file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re


@dataclass
class PyiElement:
    pass


@dataclass
class PyiArgument(PyiElement):
    name: str
    default: str | None = None
    annotation: str | None = None


@dataclass
class PyiSignature(PyiElement):
    args: list[PyiArgument] = field(default_factory=list)
    return_type: str | None = None
    var_arg: PyiArgument | None = None
    kw_arg: PyiArgument | None = None
    num_posonly_args: int = 0
    num_kwonly_args: int = 0


@dataclass
class PyiFunction(PyiElement):
    name: str
    doc: str | None = None
    signature: PyiSignature = field(default_factory=PyiSignature)
    decorators: list[str] = field(default_factory=list)


@dataclass
class PyiStatement(PyiElement):
    statement: str


@dataclass
class PyiAssignment(PyiStatement):
    pass


@dataclass
class PyiImport(PyiStatement):
    def __post_init__(self):
        self.statement = re.sub(r"\bcimport\b", "import", self.statement)


@dataclass
class PyiScope(PyiElement):
    assignments: list[PyiAssignment] = field(default_factory=list)
    functions: list[PyiFunction] = field(default_factory=list)
    classes: list[PyiClass] = field(default_factory=list)
    enums: list[PyiEnum] = field(default_factory=list)


@dataclass
class PyiClass(PyiElement):
    name: str
    doc: str | None = None
    bases: list[str] = field(default_factory=list)
    metaclass: str | None = None
    decorators: list[str] = field(default_factory=list)
    scope: PyiScope = field(default_factory=PyiScope)

@dataclass
class PyiEnum(PyiElement):
    enum_name: str | None
    names: list[str] = field(default_factory=list)


@dataclass
class PyiModule(PyiElement):
    doc: str | None = None
    imports: list[PyiImport] = field(default_factory=list)
    scope: PyiScope = field(default_factory=PyiScope)
