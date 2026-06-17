"""
Converts Cython AST nodes to PyiElements.
"""

from __future__ import annotations
import ast
import re

from dataclasses import dataclass

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
    PyiSignature,
)
from .signature import get_signature
from .conversion_utils import (
    get_decorators,
    get_bases,
    get_metaclass,
    get_source,
    get_enum_names,
    get_cdef_variables,
    unparse_expr,
    docstring_to_string,
)

_CIMPORT_RE = re.compile(r"\bcimport\b")


@dataclass
class PyiFusedType:
    name: str
    concrete_types: tuple[str, ...]


@dataclass
class Converter:
    """Converts Cython AST visitors to PyiElements for code generation.

    Transforms Cython Compiler AST nodes (as collected by visitors) into
    intermediate PyiElement representations for building .pyi stub files.
    """

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
        scope = self.convert_scope(
            visitor.scope, source_code, tc, include_docstrings
        )
        typing_import = "from typing import Any as _Any, TypeAlias as _TypeAlias"
        if _scope_uses_typevar(scope):
            typing_import += ", TypeVar"
        return PyiModule(
            doc=doc if include_docstrings else None,
            imports=self.convert_imports(visitor.import_visitor, source_code)
            + [PyiImport(typing_import)],
            scope=scope,
        )

    def convert_imports(
        self, visitor: ImportVisitor, source_code: str
    ) -> list[PyiImport]:
        """Convert import visitor nodes to PyiImport objects."""
        return [self.convert_import(node, source_code) for node in visitor.imports]

    def convert_import(self, node: Nodes.Node, source_code: str) -> PyiImport:
        """Convert a single import node to PyiImport, rewriting cimport -> import."""
        raw = get_source(source_code, node)
        return PyiImport(_CIMPORT_RE.sub("import", raw))

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
        fused_types = self.convert_fused_types(visitor.fused_types)

        cdef_assignments: list[PyiAssignment] = []
        for cdef_variable in visitor.cdef_variables:
            for name, base_type in get_cdef_variables(cdef_variable):
                resolved_type = base_type if base_type else "_Any"
                cdef_assignments.append(PyiAssignment(f"{name}: {resolved_type}"))

        # Preserve source order across cdef and def functions
        cdef_funcs = [
            (
                node.pos[1],
                self.convert_cdef_func(
                    node, source_code, tc, include_docstrings, fused_types
                ),
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
        all_funcs_sorted = sorted(cdef_funcs + py_funcs, key=lambda t: t[0])
        functions = [f for _, f in all_funcs_sorted]
        if functions:
            fused_typevar_names = _find_typevar_fused_types(functions, fused_types)
        else:
            fused_typevar_names = list(fused_types)

        return PyiScope(
            assignments=[
                self.convert_fused_type(fused_types[name])
                for name in fused_typevar_names
            ]
            + [
                self.convert_assignment(assignment, source_code)
                for assignment in visitor.assignments
            ]
            + cdef_assignments,
            functions=functions,
            classes=[
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

        node_doc: str | None = class_visitor.node.doc  # type: ignore

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
        fused_types: dict[str, PyiFusedType] | None = None,
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
            signature=_resolve_fused_signature(
                get_signature(cdef_func), fused_types or {}
            ),
            type_comment=self._type_comment_for(cdef_func, tc),
        )

    def convert_fused_types(
        self, fused_type_nodes: list[Nodes.FusedTypeNode]
    ) -> dict[str, PyiFusedType]:
        fused_types: dict[str, PyiFusedType] = {}
        for node in fused_type_nodes:
            name = getattr(node, "name")
            type_nodes = getattr(node, "types")
            concrete_types = tuple(
                type_name
                for type_name in (_type_name(type_node) for type_node in type_nodes)
                if type_name is not None
            )
            fused_types[name] = PyiFusedType(name, concrete_types)
        return fused_types

    def convert_fused_type(self, fused_type: PyiFusedType) -> PyiAssignment:
        concrete_types = ", ".join(fused_type.concrete_types)
        return PyiAssignment(
            f'{fused_type.name} = TypeVar("{fused_type.name}", {concrete_types})'
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
            if isinstance(assignment.base_type, Nodes.CSimpleBaseTypeNode):  # type: ignore
                node = assignment.base_type  # type: ignore
            elif isinstance(assignment.base_type, Nodes.TemplatedTypeNode):  # type: ignore
                node = assignment.base_type.base_type_node  # type: ignore
            else:
                return PyiAssignment(f"{name} = ...")

            if not isinstance(node, Nodes.CSimpleBaseTypeNode):
                return PyiAssignment(f"{name} = ...")

            base_name: str = node.name
            type_name = ".".join(node.module_path + [base_name])
            return PyiAssignment(f"{name}: _TypeAlias = {type_name}")

        return PyiAssignment(get_source(source_code, assignment))

    def convert_enum(self, node: Nodes.CEnumDefNode) -> PyiEnum | PyiAssignment:
        """Convert a Cython enum definition to PyiEnum."""
        if node.create_wrapper:  # type: ignore
            name: str | None = node.name  # type: ignore
            return PyiEnum(enum_name=name, names=get_enum_names(node))
        # Make it usable as an alias for int
        return PyiAssignment(f"{node.name}: _TypeAlias = int")  # type: ignore


def _type_name(node: Nodes.Node) -> str | None:
    if isinstance(node, Nodes.CSimpleBaseTypeNode):
        module_path = getattr(node, "module_path")
        name = getattr(node, "name")
        return ".".join(module_path + [name])
    if isinstance(node, Nodes.TemplatedTypeNode):
        return _type_name(getattr(node, "base_type_node"))
    return None


def _resolve_fused_signature(
    signature: PyiSignature, fused_types: dict[str, PyiFusedType]
) -> PyiSignature:
    usage = _signature_fused_usage(signature, fused_types)
    for arg in signature.args:
        if arg.annotation in fused_types:
            arg.annotation = _resolved_fused_annotation(arg.annotation, usage, fused_types)
    if signature.return_type in fused_types:
        signature.return_type = _resolved_fused_annotation(
            signature.return_type, usage, fused_types
        )
    return signature


def _resolved_fused_annotation(
    name: str,
    usage: dict[str, tuple[int, bool]],
    fused_types: dict[str, PyiFusedType],
) -> str:
    param_count, used_as_return = usage[name]
    if used_as_return or param_count > 1:
        return name
    return " | ".join(fused_types[name].concrete_types)


def _signature_fused_usage(
    signature: PyiSignature, fused_types: dict[str, PyiFusedType]
) -> dict[str, tuple[int, bool]]:
    usage: dict[str, tuple[int, bool]] = {}
    for name in fused_types:
        param_count = sum(arg.annotation == name for arg in signature.args)
        used_as_return = signature.return_type == name
        if param_count or used_as_return:
            usage[name] = (param_count, used_as_return)
    return usage


def _find_typevar_fused_types(
    functions: list[PyiFunction], fused_types: dict[str, PyiFusedType]
) -> list[str]:
    used: list[str] = []
    for name in fused_types:
        for function in functions:
            if _signature_uses_name(function.signature, name):
                used.append(name)
                break
    return used


def _signature_uses_name(signature: PyiSignature, name: str) -> bool:
    return any(arg.annotation == name for arg in signature.args) or signature.return_type == name


def _scope_uses_typevar(scope: PyiScope) -> bool:
    return any("TypeVar(" in assignment.statement for assignment in scope.assignments)
