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
    PyiEnum
)
from .signature import get_signature
from .conversion_utils import (
    get_decorators,
    get_bases,
    get_metaclass,
    get_source,
    get_enum_names,
    unparse_expr
)


@dataclass
class Converter:
    """
    Converts Visitors to PyiElements.
    """

    def convert_module(self, visitor: ModuleVisitor, source_code: str) -> PyiModule:
        return PyiModule(
            imports=self.convert_imports(visitor.import_visitor, source_code),
            scope=self.convert_scope(visitor.scope, source_code),
        )

    def convert_imports(
        self, visitor: ImportVisitor, source_code: str
    ) -> list[PyiImport]:
        return [self.convert_import(node, source_code) for node in visitor.imports]

    def convert_import(self, node: Nodes.Node, source_code: str) -> PyiImport:
        return PyiImport(get_source(source_code, node))

    def convert_scope(self, visitor: ScopeVisitor, source_code: str) -> PyiScope:
        return PyiScope(
            assignments=[
                self.convert_assignment(assignment, source_code)
                for assignment in visitor.assignments
            ],
            functions=[
                self.convert_cdef_func(cdef_func, source_code)
                for cdef_func in visitor.cdef_functions
            ]
            + [
                self.convert_py_func(py_func, source_code)
                for py_func in visitor.py_functions
            ],
            classes=[
                self.convert_class(class_visitor, source_code)
                for class_visitor in visitor.classes
            ],
            enums=[self.convert_enum(enum) for enum in visitor.enums],
        )

    def convert_class(self, class_visitor: ClassVisitor, source_code: str) -> PyiClass:
        if isinstance(class_visitor.node, Nodes.CClassDefNode):
            name: str = class_visitor.node.class_name  # type: ignore
        else:
            name: str = class_visitor.node.name

        return PyiClass(
            name=name,
            doc=class_visitor.node.doc,  # type: ignore
            bases=get_bases(class_visitor.node),
            metaclass=get_metaclass(class_visitor.node),
            decorators=get_decorators(source_code, class_visitor.node),
            scope=self.convert_scope(class_visitor.scope, source_code),
        )

    def convert_cdef_func(
        self, cdef_func: Nodes.CFuncDefNode, source_code: str
    ) -> PyiFunction:
        name: str = cdef_func.declarator.base.name  # type: ignore
        doc: str | None = cdef_func.doc or None  # type: ignore
        return PyiFunction(
            name,
            doc,
            decorators=get_decorators(source_code, cdef_func),
            signature=get_signature(cdef_func),
        )

    def convert_py_func(self, node: Nodes.DefNode, source_code: str) -> PyiFunction:
        name = node.name  # type: ignore
        doc = node.doc or None
        return PyiFunction(
            name,
            doc,
            decorators=get_decorators(source_code, node),
            signature=get_signature(node),
        )

    def convert_assignment(
        self, assignment: Nodes.AssignmentNode | Nodes.ExprStatNode, source_code: str
    ) -> PyiAssignment:
        if isinstance(assignment, Nodes.SingleAssignmentNode):
            expr = unparse_expr(assignment.rhs)
            if expr != "...":
                name: str = assignment.lhs.name # type: ignore
                annotation = assignment.lhs.annotation.string.value if assignment.lhs.annotation is not None else None
                assign: str = name
                if annotation:
                    assign = f"{assign}: {annotation}"
                assign = f"{assign} = {expr}"
                return PyiAssignment(assign)
        return PyiAssignment(get_source(source_code, assignment))
    
    def convert_enum(self, node: Nodes.CEnumDefNode) -> PyiEnum:
        name: str | None = node.name # type: ignore
        return PyiEnum(enum_name=name, names=get_enum_names(node))
