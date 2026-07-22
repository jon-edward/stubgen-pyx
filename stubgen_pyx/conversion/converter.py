"""
Converts Cython AST nodes to PyiElements.
"""

from __future__ import annotations
import ast
import re

from dataclasses import dataclass, field

from Cython.Compiler import Nodes

from ..analysis.visitor import ScopeVisitor, ClassVisitor, ModuleVisitor, ImportVisitor
from ..models.pyi_elements import (
    PyiScope,
    PyiClass,
    PyiModule,
    PyiImport,
    PyiAssignment,
    PyiFunction,
    PyiEnum,
)
from .signature import get_signature
from .source_extraction import get_decorators, get_bases, get_metaclass, get_source
from .declarators import get_enum_names, get_cdef_variables
from .unparse import unparse_expr
from .docstrings import docstring_to_string
from .type_parsing import extract_type_from_base_type
from ..postprocessing.normalize_names import _CYTHON_TRANSLATIONS

_CIMPORT_RE = re.compile(r"\bcimport\b")
_CXX_FROM_CIMPORT_RE = re.compile(
    r"^\s*from\s+(?:libcpp|libc)(?:\.[^\s]+)?\s+cimport\b"
)
_CXX_CIMPORT_RE = re.compile(r"^\s*cimport\s+(?:libcpp|libc)(?:\.|\b)")


@dataclass
class Converter:
    """Converts Cython AST visitors to PyiElements for code generation.

    Transforms Cython Compiler AST nodes (as collected by visitors) into
    intermediate PyiElement representations for building .pyi stub files.
    """

    cimport_alias_map: dict[str, str] = field(default_factory=dict)

    def _type_comment_for(
        self, node: Nodes.Node, type_comments: dict[int, str]
    ) -> str | None:
        return type_comments.get(node.pos[1])

    def convert_module(
        self,
        visitor: ModuleVisitor,
        source_code: str,
        type_comments: dict[int, str] | None = None,
        include_docstrings: bool = True,
    ) -> PyiModule:
        """Convert a ModuleVisitor to a PyiModule.

        Args:
            visitor: Module visitor containing AST information.
            source_code: The original source code text.
            type_comments: Map of line number to type comment strings.

        Returns:
            PyiModule representation with imports and scope.
        """
        tc = type_comments or {}
        doc = docstring_to_string(visitor.node.doc) if visitor.node.doc else None
        return PyiModule(
            doc=doc if include_docstrings else None,
            imports=self.convert_imports(visitor.import_visitor, source_code)
            + [
                PyiImport("from typing import Any, Callable, TypeAlias, TypedDict"),
                PyiImport("import numpy"),
            ],
            scope=self.convert_scope(
                visitor.scope, source_code, tc, include_docstrings
            ),
        )

    def convert_imports(
        self, visitor: ImportVisitor, source_code: str
    ) -> list[PyiImport]:
        """Convert import visitor nodes to PyiImport objects."""
        self.cimport_alias_map = {}
        imports = []
        for node in visitor.imports:
            raw = get_source(source_code, node)
            if _is_cxx_cimport(raw):
                self._collect_cimport_aliases(node)
                continue
            imports.append(self.convert_import(node, source_code, raw))
        return imports

    def _collect_cimport_aliases(self, node: Nodes.Node) -> None:
        if not isinstance(node, Nodes.FromCImportStatNode):
            return
        for _, original, alias in node.imported_names or ():
            if alias and alias != original:
                python_type = _CYTHON_TRANSLATIONS.get(original)
                if python_type:
                    self.cimport_alias_map[alias] = python_type

    def convert_import(
        self, node: Nodes.Node, source_code: str, raw: str | None = None
    ) -> PyiImport:
        """Convert a single import node to PyiImport, rewriting cimport -> import."""
        raw = raw if raw is not None else get_source(source_code, node)
        return PyiImport(_CIMPORT_RE.sub("import", raw))

    def convert_struct_or_union(
        self, node: Nodes.CStructOrUnionDefNode, _source_code: str
    ) -> PyiClass:
        """Convert a C struct or union definition node to PyiClass."""
        attributes = []
        for attribute in node.attributes:
            if isinstance(attribute, Nodes.CVarDefNode):
                attributes.extend(
                    [
                        PyiAssignment(f"{name}: {base_type or 'Any'}")
                        for name, base_type in get_cdef_variables(attribute)
                    ]
                )
        is_union = node.kind == "union"
        keywords = {"total": "False"} if is_union else {}
        return PyiClass(
            name=node.name,
            bases=["TypedDict"],
            keywords=keywords,
            scope=PyiScope(assignments=attributes),
        )

    def convert_scope(
        self,
        visitor: ScopeVisitor,
        source_code: str,
        type_comments: dict[int, str] | None = None,
        include_docstrings: bool = True,
    ) -> PyiScope:
        """Convert a ScopeVisitor to a PyiScope.

        Preserves source order by interleaving cdef and def functions by their
        line position, rather than emitting all cdef functions first.
        """
        tc = type_comments or {}

        cdef_assignments: list[PyiAssignment] = []
        for cdef_variable in visitor.cdef_variables:
            for name, base_type in get_cdef_variables(cdef_variable):
                resolved_type = base_type if base_type else "Any"
                cdef_assignments.append(PyiAssignment(f"{name}: {resolved_type}"))

        # Preserve source order across cdef and def functions
        cdef_funcs = [
            (
                node.pos[1],
                self.convert_cdef_func(node, source_code, tc, include_docstrings),
            )
            for node in visitor.cdef_functions
        ]
        py_funcs = [
            (
                node.pos[1],
                self.convert_py_func(node, source_code, tc, include_docstrings),
            )
            for node in visitor.py_functions
        ]

        # Replace __cinit__ with __init__ if no __init__ is present, otherwise remove
        contains_init = any(py_func.name == "__init__" for _, py_func in py_funcs)

        for idx, (_, py_func) in enumerate(py_funcs):
            if py_func.name == "__cinit__" and not contains_init:
                py_func.name = "__init__"
            elif py_func.name == "__cinit__" and contains_init:
                del py_funcs[idx]
                break

        all_funcs_sorted = sorted(cdef_funcs + py_funcs, key=lambda t: t[0])
        functions = [f for _, f in all_funcs_sorted]

        structs_or_enums = [
            self.convert_struct_or_union(node, source_code)
            for node in visitor.cdef_structs_or_unions
        ]

        return PyiScope(
            assignments=[
                self.convert_assignment(assignment, source_code)
                for assignment in visitor.assignments
            ]
            + cdef_assignments,
            functions=functions,
            classes=structs_or_enums
            + [
                self.convert_class(class_visitor, source_code, tc, include_docstrings)
                for class_visitor in visitor.classes
            ],
            enums=[self.convert_enum(enum) for enum in visitor.enums],
        )

    def convert_class(
        self,
        class_visitor: ClassVisitor,
        source_code: str,
        type_comments: dict[int, str] | None = None,
        include_docstrings: bool = True,
    ) -> PyiClass:
        """Convert a ClassVisitor to a PyiClass."""
        tc = type_comments or {}
        if isinstance(class_visitor.node, Nodes.CClassDefNode):
            name: str = class_visitor.node.class_name  # type: ignore
        else:
            name: str = class_visitor.node.name

        node_doc: str | None = getattr(class_visitor.node, "doc", None)
        doc = docstring_to_string(node_doc) if node_doc else None

        return PyiClass(
            name=name,
            doc=doc if include_docstrings else None,
            bases=get_bases(class_visitor.node),
            metaclass=get_metaclass(class_visitor.node),
            decorators=get_decorators(source_code, class_visitor.node),
            scope=self.convert_scope(
                class_visitor.scope, source_code, tc, include_docstrings
            ),
        )

    def convert_cdef_func(
        self,
        cdef_func: Nodes.CFuncDefNode,
        source_code: str,
        type_comments: dict[int, str] | None = None,
        include_docstrings: bool = True,
    ) -> PyiFunction:
        """Convert a C function definition node to PyiFunction."""
        tc = type_comments or {}
        name: str = cdef_func.declarator.base.name  # type: ignore
        doc = docstring_to_string(cdef_func.doc) if cdef_func.doc else None  # type: ignore
        return PyiFunction(
            name,
            is_async=False,
            doc=doc if include_docstrings else None,
            decorators=get_decorators(source_code, cdef_func),
            signature=get_signature(cdef_func),
            type_comment=self._type_comment_for(cdef_func, tc),
        )

    def convert_py_func(
        self,
        node: Nodes.DefNode,
        source_code: str,
        type_comments: dict[int, str] | None = None,
        include_docstrings: bool = True,
    ) -> PyiFunction:
        """Convert a Python function definition node to PyiFunction."""
        tc = type_comments or {}
        name = node.name  # type: ignore
        doc = docstring_to_string(node.doc) if node.doc else None
        return PyiFunction(
            name,
            is_async=node.is_async_def,
            doc=doc if include_docstrings else None,
            decorators=get_decorators(source_code, node),
            signature=get_signature(node),
            type_comment=self._type_comment_for(node, tc),
        )

    def convert_assignment(
        self, assignment: Nodes.AssignmentNode | Nodes.ExprStatNode, source_code: str
    ) -> PyiAssignment:
        """Convert an assignment node to PyiAssignment, extracting type annotations."""
        if isinstance(assignment, Nodes.SingleAssignmentNode):
            expr = unparse_expr(assignment.rhs)
            name: str = assignment.lhs.name
            if expr != "...":
                annotation = (
                    assignment.lhs.annotation.string.value
                    if assignment.lhs.annotation is not None
                    else None
                )
                assign: str = name
                if annotation:
                    assign = f"{assign}: {annotation}"
                assign = f"{assign} = {expr}"
                return PyiAssignment(assign)

            try:
                assignment_source = get_source(source_code, assignment)
                ast.parse(assignment_source)
                return PyiAssignment(assignment_source)
            except SyntaxError:
                pass

            return PyiAssignment(f"{name} = ...")

        if isinstance(assignment, Nodes.CTypeDefNode):
            name: str = assignment.declarator.name  # type: ignore
            type_str = extract_type_from_base_type(assignment)
            if not type_str:
                return PyiAssignment(f"{name} = ...")
            return PyiAssignment(f"{name}: TypeAlias = {type_str}")

        return PyiAssignment(get_source(source_code, assignment))

    def convert_enum(self, node: Nodes.CEnumDefNode) -> PyiEnum | PyiAssignment:
        """Convert a Cython enum definition to PyiEnum."""
        if node.create_wrapper:  # type: ignore
            name: str | None = node.name  # type: ignore
            return PyiEnum(enum_name=name, names=get_enum_names(node))
        # Make it usable as an alias for int
        return PyiAssignment(f"{node.name}: TypeAlias = int")  # type: ignore


def _is_cxx_cimport(raw: str) -> bool:
    return bool(_CXX_FROM_CIMPORT_RE.search(raw) or _CXX_CIMPORT_RE.search(raw))
