"""Dataclasses representing AST elements used to generate .pyi stub files."""

from __future__ import annotations

from dataclasses import dataclass, field


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

    # The assigned name, when cleanly known at construction time (every
    # call site already has it as a plain local variable -- it's the name
    # being assigned to). Optional/`None` only for the rare fallback path
    # that recovers an assignment straight from raw source text
    # (`Converter.convert_assignment`'s final `ast.parse` fallback) without
    # separately tracking its target. Lets consumers (e.g.
    # `Builder.build_assignment`'s privacy check) use the name directly
    # instead of re-deriving it by string-partitioning `statement`, which
    # breaks on a value/annotation containing its own `=`/`:` before the
    # real separator (e.g. `_x: Annotated[int, Field(default=5)] = 5`).
    name: str | None = None


@dataclass
class PyiImport(PyiStatement):
    """Represents an import statement."""


@dataclass
class PyiScope(PyiElement):
    """Represents a scope (module or class context)."""

    assignments: list[PyiAssignment] = field(default_factory=list)
    functions: list[PyiFunction] = field(default_factory=list)
    classes: list[PyiClass] = field(default_factory=list)
    enums: list[PyiEnum | PyiAssignment] = field(default_factory=list)


@dataclass
class PyiClass(PyiElement):
    """Represents a Python class."""

    name: str
    doc: str | None = None
    bases: list[str] = field(default_factory=list)
    metaclass: str | None = None
    keywords: dict[str, str] = field(default_factory=dict)
    decorators: list[str] = field(default_factory=list)
    scope: PyiScope = field(default_factory=PyiScope)


@dataclass
class PyiEnum(PyiElement):
    """Represents a cdef enum."""

    enum_name: str | None
    names: list[str] = field(default_factory=list)


@dataclass
class PyiFusedType(PyiElement):
    """Represents a fused type."""

    name: str
    concrete_types: tuple[str, ...]
    # Parallel to `concrete_types`: the numpy scalar-type suffix (e.g.
    # "int32", "double") for each member, or None if it has no numpy
    # equivalent (extension type, `object`, etc.). Used to render a
    # memoryview of this fused type as `numpy.typing.NDArray[...]` per
    # member instead of a bare scalar.
    numpy_scalars: tuple[str | None, ...] = ()


@dataclass
class PyiModule(PyiElement):
    """Represents a Python module."""

    doc: str | None = None
    imports: list[PyiImport] = field(default_factory=list)
    scope: PyiScope = field(default_factory=PyiScope)
