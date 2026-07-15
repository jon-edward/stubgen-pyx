"""
Converts Cython AST nodes to PyiElements.
"""

from __future__ import annotations
import ast
import re

from dataclasses import dataclass
from collections.abc import Iterator

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
from .source_extraction import get_decorators, get_bases, get_metaclass, get_source
from .declarators import get_enum_names, get_cdef_variables
from .unparse import unparse_expr
from .docstrings import docstring_to_string
from .type_parsing import extract_type_from_base_type

_CIMPORT_RE = re.compile(r"\bcimport\b")

_C_TO_PYTHON = {
    "char": "int",
    "double": "float",
    "long": "int",
    "long double": "float",
    "long long": "int",
    "short": "int",
    "signed char": "int",
    "unsigned char": "int",
    "unsigned int": "int",
    "unsigned long": "int",
    "unsigned long long": "int",
    "unsigned short": "int",
}


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
        inherited_fused_types: dict[str, PyiFusedType] | None = None,
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
            visitor.scope,
            source_code,
            tc,
            include_docstrings,
            inherited_fused_types,
            emit_inherited_fused_typevars=True,
        )
        typing_import = (
            "from typing import Any, Any as _Any, TypeAlias as _TypeAlias, TypedDict"
        )
        if any("TypeVar(" in assignment.statement for assignment in scope.assignments):
            typing_import += ", TypeVar"
        return PyiModule(
            doc=doc if include_docstrings else None,
            imports=self.convert_imports(visitor.import_visitor, source_code)
            + [PyiImport(typing_import), PyiImport("import numpy")],
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

    def convert_struct_or_union(
        self, node: Nodes.CStructOrUnionDefNode, source_code: str
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
        inherited_fused_types: dict[str, PyiFusedType] | None = None,
        emit_inherited_fused_typevars: bool = False,
    ) -> PyiScope:
        """Convert a ScopeVisitor to a PyiScope.

        Preserves source order by interleaving cdef and def functions by their
        line position, rather than emitting all cdef functions first.

        The ``emit_inherited_fused_typevars`` flag encodes a scope-role
        distinction: which scope owns emitting ``TypeVar``/``TypeAlias``
        assignments for fused typedefs that were inherited from a parent
        (currently: the companion ``.pxd``, threaded in via
        ``inherited_fused_types``). Exactly one scope in a nesting chain must
        own that emission; emitting in more than one place produces
        duplicate, incorrect ``TypeVar`` declarations. Only the outermost scope
        (``convert_module``) should emit them. This solution is not the most elegant,
        but it is a sufficient encoding for differentiating modules and classes today.
        """
        tc = type_comments or {}
        local_fused_types = self.convert_fused_types(visitor.fused_types)
        fused_types = {**(inherited_fused_types or {}), **local_fused_types}

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
                self.convert_py_func(
                    node, source_code, tc, include_docstrings, fused_types
                ),
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
        classes = [
            self.convert_class(
                class_visitor, source_code, tc, include_docstrings, fused_types
            )
            for class_visitor in visitor.classes
        ]
        typevar_candidates = (
            fused_types if emit_inherited_fused_typevars else local_fused_types
        )
        typevar_scan_scope = PyiScope(functions=functions, classes=classes)
        fused_typevar_names = [
            name
            for name in typevar_candidates
            if any(
                any(
                    _annotation_uses_name(arg.annotation, name)
                    for arg in function.signature.args
                )
                or _annotation_uses_name(function.signature.return_type, name)
                for function in _scope_functions(typevar_scan_scope)
            )
        ]

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
            classes=structs_or_enums + classes,
            enums=[self.convert_enum(enum) for enum in visitor.enums],
        )

    def convert_class(
        self,
        class_visitor: ClassVisitor,
        source_code: str,
        type_comments: dict[int, str] | None = None,
        include_docstrings: bool = True,
        inherited_fused_types: dict[str, PyiFusedType] | None = None,
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
                class_visitor.scope,
                source_code,
                tc,
                include_docstrings,
                inherited_fused_types,
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
                _restore_fused_memoryview_annotations(
                    get_signature(cdef_func), cdef_func, fused_types or {}
                ),
                fused_types or {},
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
                dict.fromkeys(
                    _C_TO_PYTHON.get(type_name, type_name)
                    for type_name in (_type_name(type_node) for type_node in type_nodes)
                    if type_name is not None
                )
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
        fused_types: dict[str, PyiFusedType] | None = None,
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
            signature=_resolve_fused_signature(
                _restore_fused_memoryview_annotations(
                    get_signature(node), node, fused_types or {}
                ),
                fused_types or {},
            ),
            type_comment=self._type_comment_for(node, tc),
        )

    def convert_assignment(
        self, assignment: Nodes.AssignmentNode | Nodes.ExprStatNode, source_code: str
    ) -> PyiAssignment:
        """Convert an assignment node to PyiAssignment, extracting type annotations."""
        if isinstance(assignment, Nodes.SingleAssignmentNode):
            expr = unparse_expr(assignment.rhs)
            name: str = assignment.lhs.name  # type: ignore
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
        return PyiAssignment(f"{node.name}: _TypeAlias = int")  # type: ignore


def _restore_fused_memoryview_annotations(
    signature: PyiSignature,
    node: Nodes.CFuncDefNode | Nodes.DefNode,
    fused_types: dict[str, PyiFusedType],
) -> PyiSignature:
    """Restore fused typedef names on memoryview annotations lost during signature extraction.

    This post-processing pass is a workaround for the fact that fused types are not
    natively supported throughout the rest of stubgen-pyx's type-parsing and
    signature-extraction machinery. If fused types were threaded into
    _extract_memoryview_type we could preserve the fused typedef name there, but that
    would require treating fused types as the first-class type concept throughout
    the type-parsing layer, which is a much larger refactor. Instead, this pass
    walks the original AST alongside the extracted signature and repairs any memoryview
    annotations that were flattened to the string "memoryview" back to their fused
    typedef name, so that the downstream fused-type resolver can handle them correctly.
    """
    if not fused_types:
        return signature

    if isinstance(node, Nodes.CFuncDefNode):
        arg_nodes = getattr(getattr(node, "declarator", None), "args", [])
    else:
        arg_nodes = getattr(node, "args", [])
    for arg, arg_node in zip(signature.args, arg_nodes, strict=False):
        fused_name = _fused_memoryview_name(arg_node, fused_types)
        if arg.annotation == "memoryview" and fused_name is not None:
            arg.annotation = fused_name

    fused_name = _fused_memoryview_name(node, fused_types)
    if signature.return_type == "memoryview" and fused_name is not None:
        signature.return_type = fused_name

    return signature


def _fused_memoryview_name(
    node: Nodes.Node, fused_types: dict[str, PyiFusedType]
) -> str | None:
    """Return the fused typedef name backing a memoryview node, or None if not fused."""
    base_type = getattr(node, "base_type", None)
    if not isinstance(base_type, Nodes.MemoryViewSliceTypeNode):
        return None

    scalar = getattr(base_type, "base_type_node", None)
    name = getattr(scalar, "name", None)
    return name if name in fused_types else None


def _resolve_fused_signature(
    signature: PyiSignature, fused_types: dict[str, PyiFusedType]
) -> PyiSignature:
    """Rewrite a signature's fused-type annotations to their resolved Python form.

    Having the AST layer natively handle fused resolution would require the
    ``PyiSignature`` model to carry structured fused references rather than
    strings, and every downstream consumer (import emission,
    ``trim_not_defined``, source rendering) would need to handle those gracefully.
    Doing so is a much larger refactor. To avoid that, this function is a localized
    post-processing pass that rewrites the signature's argument and return annotations
    after the AST traversal has completed by introspecting the already-generated
    annotation strings and the fused-type definitions. A fused typedef like ``numeric``
    resolves to a ``TypeVar`` when it appears in both a parameter and the return, to a
    ``|``-union when it appears in one parameter only, or to ``object`` when its member
    set includes ``object``. That decision is a *whole-signature* property, so it cannot
    be made while an individual argument's annotation is being generated.
    """
    usage: dict[str, tuple[int, bool]] = {}
    for name in fused_types:
        param_count = sum(
            _annotation_uses_name(arg.annotation, name) for arg in signature.args
        )
        used_as_return = _annotation_uses_name(signature.return_type, name)
        if param_count or used_as_return:
            usage[name] = (param_count, used_as_return)
    for arg in signature.args:
        if arg.annotation is not None:
            arg.annotation = _resolved_fused_annotation(
                arg.annotation, usage, fused_types
            )
    if signature.return_type is not None:
        signature.return_type = _resolved_fused_annotation(
            signature.return_type, usage, fused_types
        )
    return signature


def _resolved_fused_annotation(
    annotation: str,
    usage: dict[str, tuple[int, bool]],
    fused_types: dict[str, PyiFusedType],
) -> str:
    """Resolve a single annotation string using per-name fused usage and definitions."""
    resolved = annotation
    for name in fused_types:
        if not _annotation_uses_name(resolved, name):
            continue
        param_count, used_as_return = usage[name]
        replacement = name
        if "object" in fused_types[name].concrete_types:
            replacement = "object"
        elif (not used_as_return and param_count <= 1) or param_count == 0:
            replacement = " | ".join(fused_types[name].concrete_types)
        resolved = " | ".join(
            replacement if part == name else part
            for part in _annotation_parts(resolved)
        )
    return resolved


def _scope_functions(scope: PyiScope) -> Iterator[PyiFunction]:
    """Yield all functions in the scope, recursively descending into nested classes."""
    yield from scope.functions
    for class_ in scope.classes:
        yield from _scope_functions(class_.scope)


def _annotation_uses_name(annotation: str | None, name: str) -> bool:
    """Return True if the given name appears as a part of the (union) annotation."""
    if annotation is None:
        return False
    return name in _annotation_parts(annotation)


def _annotation_parts(annotation: str) -> list[str]:
    """Split a union annotation string into its stripped alternative parts."""
    return [part.strip() for part in annotation.split("|")]


def _type_name(node: Nodes.Node) -> str | None:
    """Return a dotted type name for a simple or templated base-type node, or None."""
    while isinstance(node, Nodes.TemplatedTypeNode):
        node = getattr(node, "base_type_node")
    if isinstance(node, Nodes.CSimpleBaseTypeNode):
        return ".".join(getattr(node, "module_path") + [getattr(node, "name")])
    return None
