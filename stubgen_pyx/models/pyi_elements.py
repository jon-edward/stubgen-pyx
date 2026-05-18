"""Dataclasses representing AST elements used to generate .pyi stub files."""

from __future__ import annotations

from dataclasses import dataclass, field
import re


_IMPORT_RE = re.compile(r"\bcimport\b")


@dataclass
class PyiElement:
    """Base class for all AST elements."""


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
    type_comment: str | None = None


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
        self.statement = _IMPORT_RE.sub("import", self.statement)


@dataclass
class PyiScope(PyiElement):
    """Represents a scope (module or class context)."""

    assignments: list[PyiAssignment] = field(default_factory=list)
    functions: list[PyiFunction] = field(default_factory=list)
    classes: list[PyiClass] = field(default_factory=list)
    enums: list[PyiEnum | PyiAssignment] = field(default_factory=list)

    def deduplicate_assignments(self) -> None:
        """Remove duplicate assignments from the scope while preserving order."""
        seen: set[str] = set()
        unique_assignments: list[PyiAssignment] = []
        for assignment in self.assignments:
            name = assignment.statement.partition("=")[0].partition(":")[0].strip()
            if not name or name not in seen:
                if name:
                    seen.add(name)
                unique_assignments.append(assignment)
        self.assignments = unique_assignments

    def merge_classes(self, extra_classes: list[PyiClass]) -> None:
        """Merge classes with the same name and preserve unique declarations."""
        existing_classes: dict[str, PyiClass] = {cls.name: cls for cls in self.classes}
        for extra_class in extra_classes:
            if extra_class.name in existing_classes:
                existing_classes[extra_class.name].merge_with(extra_class)
            else:
                self.classes.append(extra_class)


@dataclass
class PyiClass(PyiElement):
    """Represents a Python class."""

    name: str
    doc: str | None = None
    bases: list[str] = field(default_factory=list)
    metaclass: str | None = None
    decorators: list[str] = field(default_factory=list)
    scope: PyiScope = field(default_factory=PyiScope)

    def merge_with(self, other: PyiClass) -> None:
        """Merge another class declaration into this class."""
        if self.doc is None:
            self.doc = other.doc

        self.bases = [*dict.fromkeys(self.bases + other.bases)]
        if self.metaclass is None:
            self.metaclass = other.metaclass

        self.decorators = [*dict.fromkeys(self.decorators + other.decorators)]

        self.scope.assignments += other.scope.assignments
        self.scope.deduplicate_assignments()

        self.scope.functions += other.scope.functions
        self.scope.merge_classes(other.scope.classes)
        self.scope.enums += other.scope.enums


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
