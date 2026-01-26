"""
Visits and collects Cython AST nodes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from Cython.Compiler import Nodes, ExprNodes, ModuleNode
from Cython.Compiler.Visitor import TreeVisitor


@dataclass
class ScopeVisitor(TreeVisitor):
    """Visits and collects Cython AST nodes in a scope."""

    node: Nodes.Node
    """The node to visit."""

    assignments: list[Nodes.SingleAssignmentNode] = field(
        default_factory=list, init=False
    )
    """A list of collected assignment nodes."""

    py_functions: list[Nodes.DefNode] = field(default_factory=list, init=False)
    """A list of collected Python function nodes."""

    cdef_functions: list[Nodes.CFuncDefNode] = field(default_factory=list, init=False)
    """A list of collected C function nodes."""

    classes: list[ClassVisitor] = field(default_factory=list, init=False)
    """A list of collected class nodes."""

    enums: list[Nodes.CEnumDefNode] = field(default_factory=list, init=False)
    """A list of collected enum nodes."""

    def __post_init__(self):
        super().__init__()
        self.visitchildren(self.node)

    def visit_Node(self, node):
        """Default visitor for generic nodes."""
        return node

    def visit_CEnumDefNode(self, node):
        """Visits Cython enum definition nodes."""
        if not node.create_wrapper:
            # Only Python-visible enums
            return node
        self.enums.append(node)
        return node

    def visit_StatListNode(self, node):
        """Visits statement list nodes and their children."""
        self.visitchildren(node)
        return node

    def visit_SingleAssignmentNode(self, node):
        """Visits assignment nodes, discarding import assignments."""
        if isinstance(node.rhs, ExprNodes.ImportNode):
            return node
        if isinstance(node.lhs, ExprNodes.NameNode):
            self.assignments.append(node)
        return node

    def visit_ExprStatNode(self, node):
        """Visits expression statement nodes."""
        if (
            isinstance(node.expr, ExprNodes.NameNode)
            and node.expr.annotation is not None
        ):
            self.assignments.append(node)
        return node

    def visit_PyClassDefNode(self, node):
        """Visits Python class definition nodes."""
        self.classes.append(ClassVisitor(node=node))
        return node

    def visit_CClassDefNode(self, node):
        """Visits Cython extension type (cdef class) nodes."""
        self.classes.append(ClassVisitor(node=node))
        return node

    def visit_DefNode(self, node):
        """Visits Python function definition nodes."""
        self.py_functions.append(node)
        return node

    def visit_CFuncDefNode(self, node):
        """Visits cpdef function definition nodes."""
        if not node.declarator.overridable:
            # Only cdef, not public to Python
            return node
        self.cdef_functions.append(node)
        return node


@dataclass
class ImportVisitor(TreeVisitor):
    """Visits and collects Cython import nodes in a scope."""

    node: Nodes.Node
    """The node to visit."""

    imports: list[Nodes.Node] = field(default_factory=list, init=False)
    """A list of collected import statements."""

    def __post_init__(self):
        super().__init__()
        self.visitchildren(self.node)

    def visit_Node(self, node):
        """Default visitor for generic nodes."""
        return node

    def visit_StatListNode(self, node):
        """Visits statement list nodes and their children."""
        self.visitchildren(node)
        return node

    def visit_CImportNode(self, node):
        """Visits cimport nodes."""
        self.imports.append(node)
        return node

    def visit_CImportStatNode(self, node):
        """Visits cimport statement nodes."""
        self.imports.append(node)
        return node

    def visit_ImportNode(self, node):
        """Visits import nodes."""
        self.imports.append(node)
        return node

    def visit_FromImportNode(self, node):
        """Visits from...import nodes."""
        self.imports.append(node)
        return node

    def visit_FromImportStatNode(self, node):
        """Visits from...import statement nodes."""
        self.imports.append(node)
        return node

    def visit_FromCImportStatNode(self, node):
        """Visits from...cimport statement nodes."""
        self.imports.append(node)
        return node

    def visit_ImportStatNode(self, node):
        """Visits import statement nodes."""
        self.imports.append(node)
        return node

    def visit_SingleAssignmentNode(self, node):
        if isinstance(node.rhs, ExprNodes.ImportNode):
            self.imports.append(node)
            return node
        return node

    def visit_IfStatNode(self, node):
        """Pass through `if typing.TYPE_CHECKING: ` and `if TYPE_CHECKING: ` blocks"""
        for clause in node.if_clauses:
            condition_name = _collect_attribute(clause.condition)
            if condition_name in ("TYPE_CHECKING", "typing.TYPE_CHECKING"):
                self.visitchildren(clause)


@dataclass
class ModuleVisitor:
    """Visits and collects Cython module nodes in a scope."""

    node: ModuleNode.ModuleNode
    """The node to visit."""

    import_visitor: ImportVisitor = field(init=False)
    """A visitor for collecting import nodes."""

    scope: ScopeVisitor = field(init=False)
    """A visitor for collecting scope nodes."""

    def __post_init__(self):
        self.import_visitor = ImportVisitor(node=self.node)
        self.scope = ScopeVisitor(node=self.node)


@dataclass
class ClassVisitor:
    """Visits and collects Cython class nodes in a scope."""

    node: Nodes.CClassDefNode | Nodes.PyClassDefNode
    """The node to visit."""

    scope: ScopeVisitor = field(init=False)
    """A visitor for collecting scope nodes."""

    def __post_init__(self):
        self.scope = ScopeVisitor(node=self.node)


def _collect_attribute(node) -> str:
    names = []
    attribute = node

    while isinstance(attribute, ExprNodes.AttributeNode):
        names.append(attribute.attribute)
        attribute = attribute.obj

    if isinstance(attribute, ExprNodes.NameNode):
        names.append(attribute.name)  # type: ignore

    names.reverse()

    name = ".".join(names)
    return name
