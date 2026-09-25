"""AST visitors for collecting Cython nodes during tree traversal."""

from __future__ import annotations

from dataclasses import dataclass, field

from Cython.Compiler import ExprNodes, ModuleNode, Nodes
from Cython.Compiler.Visitor import TreeVisitor


def _collect_fused_specialization_ids(node, out: set[int], _seen: set[int] | None = None) -> None:
    """Find every ``FusedCFuncDefNode`` reachable from ``node`` and record
    the ids of the nodes its expansion leaves behind as ordinary siblings
    (each specialization, plus its own Python-visible wrapper and the
    runtime dispatcher where those exist) -- everything ``ScopeVisitor``
    should skip in favor of visiting ``FusedCFuncDefNode.node`` itself.

    Uses a raw, independent walk via ``child_attrs`` (same approach as
    ``type_parsing.capture_static_types``) rather than relying on
    traversal order within ``ScopeVisitor.visitchildren``: a
    ``FusedCFuncDefNode``'s own ``child_attrs`` doesn't include ``.node``/
    ``.nodes``/``.py_func``, so those are only reachable this way, and
    nothing guarantees a ``FusedCFuncDefNode`` is visited before the
    siblings it needs to suppress.
    """
    seen = _seen if _seen is not None else set()
    if node is None or id(node) in seen:
        return
    seen.add(id(node))

    if type(node).__name__ == "FusedCFuncDefNode":
        for specialization in getattr(node, "nodes", None) or ():
            out.add(id(specialization))
            py_func = getattr(specialization, "py_func", None)
            if py_func is not None:
                out.add(id(py_func))
        dispatcher = getattr(node, "py_func", None)
        if dispatcher is not None:
            out.add(id(dispatcher))

    for attr_name in getattr(node, "child_attrs", None) or ():
        child = getattr(node, attr_name, None)
        if isinstance(child, list):
            for item in child:
                _collect_fused_specialization_ids(item, out, seen)
        else:
            _collect_fused_specialization_ids(child, out, seen)


def _is_decorator_rebinding(
    node: Nodes.SingleAssignmentNode, name: str, def_positions: dict[str, tuple]
) -> bool:
    """Whether `node` (a `name = ...` assignment) is a
    `decorator(name)`-shaped call rebinding `name` to itself, the
    pattern `AnalyseDeclarationsTransform` produces for any decorated
    function/method once a decorator is rewritten away (see
    `ScopeVisitor.visit_SingleAssignmentNode`'s docstring).

    Structurally, this is indistinguishable from genuine, hand-written
    code with the exact same shape -- `foo = trace(foo)`, a manual,
    valid (if unusual) way to apply a wrapper, is real user code that
    must not be dropped, and parses identically. Disambiguated by
    position instead of shape: `AnalyseDeclarationsTransform` derives
    the synthetic assignment from the original decorated statement
    itself, so it sits at the *exact same* `node.pos` as the (now
    decorator-less) `DefNode`/`CFuncDefNode` it rebinds. Real,
    hand-written code necessarily sits on its own, later line.
    `def_positions` (built by `visit_DefNode`/`visit_CFuncDefNode`,
    which -- by traversal order -- always visit a function before any
    assignment rebinding it) maps a name to its function's `pos`; only
    a position match is treated as synthetic.
    """
    rhs = node.rhs
    if not isinstance(rhs, ExprNodes.SimpleCallNode):
        return False
    args = rhs.args or []
    if len(args) != 1:
        return False
    (arg,) = args
    if isinstance(arg, ExprNodes.PyCFunctionNode):
        # A `PyCFunctionNode` is a function-object *literal* -- there's
        # no source syntax that produces one directly by hand, so this
        # shape is unambiguously synthetic regardless of position.
        def_node = getattr(arg, "def_node", None)
        return getattr(def_node, "name", None) == name
    if isinstance(arg, ExprNodes.NameNode):
        arg_name = arg.name
    elif isinstance(arg, ExprNodes.AttributeNode):
        arg_name = arg.attribute
    else:
        return False
    if arg_name != name:
        return False
    return def_positions.get(name) == node.pos



def _declared_name(node) -> str | None:
    """The name a `DefNode`/`CFuncDefNode` declares, or None.

    `DefNode.name` is direct; a `CFuncDefNode`'s name lives on its
    declarator (`.base.name` for a pointer-returning function, `.name`
    otherwise -- same ambiguity `signature._to_argument` unwraps for
    arguments).
    """
    name = getattr(node, "name", None)
    if name:
        return name
    declarator = getattr(node, "declarator", None)
    if isinstance(declarator, Nodes.CPtrDeclaratorNode):
        declarator = declarator.base
    return getattr(declarator, "name", None) or None


@dataclass
class ScopeVisitor(TreeVisitor):
    """Traverses and collects Cython AST nodes in a scope.

    Attributes:
        node: The root node to visit.
        assignments: Collected variable assignments and annotated names.
        py_functions: Collected Python (def) function definitions.
        cdef_functions: Collected Cython (cdef) function definitions.
        classes: Collected class definitions.
        enums: Collected enum definitions.
        fused_types: Collected Cython fused type definitions.
    """

    node: Nodes.Node
    in_class: bool = False
    assignments: list[Nodes.SingleAssignmentNode] = field(
        default_factory=list, init=False
    )
    py_functions: list[Nodes.DefNode] = field(default_factory=list, init=False)
    cdef_functions: list[Nodes.CFuncDefNode] = field(default_factory=list, init=False)
    classes: list[ClassVisitor] = field(default_factory=list, init=False)
    enums: list[Nodes.CEnumDefNode] = field(default_factory=list, init=False)
    fused_types: list[Nodes.FusedTypeNode] = field(default_factory=list, init=False)
    cdef_variables: list[Nodes.CVarDefNode | Nodes.PropertyNode] = field(
        default_factory=list, init=False
    )
    cdef_structs_or_unions: list[Nodes.CStructOrUnionDefNode] = field(
        default_factory=list, init=False
    )
    cpp_classes: list[Nodes.CppClassNode] = field(default_factory=list, init=False)
    # The set of `id()`s of every fused-specialization/dispatcher node
    # reachable from `node`, as `_collect_fused_specialization_ids`
    # computes it. Optional constructor parameter (not `init=False`, unlike
    # the other collected-output fields above): an enclosing scope's own
    # walk already covers every nested class's subtree too, so
    # `visit_PyClassDefNode`/`visit_CClassDefNode` pass their own
    # already-computed set down to each nested `ClassVisitor` rather than
    # letting its `ScopeVisitor` redo the same walk from scratch. Left as
    # `None` (the default) only for the outermost `ScopeVisitor` in a
    # module, built directly by `ModuleVisitor` -- nothing higher up the
    # tree to inherit a set from yet -- which computes it once, here, for
    # the very first time.
    _fused_specialization_ids: set[int] | None = None
    # Maps a function/method name to its DefNode/CFuncDefNode's `pos`;
    # used to distinguish a genuine `name = decorator(name)` rebinding
    # from the synthetic sibling `AnalyseDeclarationsTransform` leaves
    # for an actually-decorated function -- see `_is_decorator_rebinding`.
    _def_positions: dict[str, tuple] = field(default_factory=dict, init=False)

    def __post_init__(self):
        super().__init__()
        if self._fused_specialization_ids is None:
            self._fused_specialization_ids = set()
            _collect_fused_specialization_ids(self.node, self._fused_specialization_ids)
        self.visitchildren(self.node)

    def visit_Node(self, node):
        """Generic node visitor."""
        return node

    def visit_CEnumDefNode(self, node):
        """Collect public enums."""
        if not node.name:
            return node
        self.enums.append(node)
        return node

    def visit_StatListNode(self, node):
        """Traverse statement lists."""
        self.visitchildren(node)
        return node

    def visit_CDefExternNode(self, node):
        """Propagate visit from "cdef extern" nodes.

        `cdef extern from "<header.h>"` allows for declaration of enumerations
        to be wrapped when `cpdef enum` is used. Here visiting is propagated
        down to the children to search for those enums.
        """
        self.visitchildren(node)
        return node

    def visit_CompilerDirectivesNode(self, node):
        """Propagate visiting into a compiler-directive-scoped block.

        Certain builtin decorators get parsed by wrapping the decorated
        statement in a `CompilerDirectivesNode` to apply a directive
        context to it -- e.g. `@staticmethod` on a module-level
        function, an implementation detail of how that decorator is
        recognized, not anything the source itself wrote. Without this,
        the default `visit_Node` fallback (no recursion) drops
        everything inside silently.
        """
        self.visitchildren(node)
        return node

    def visit_SingleAssignmentNode(self, node):
        """Collect non-import assignments.

        Skips synthetic name bindings ``AnalyseDeclarationsTransform``
        adds for a decorated function or method -- a decorator is
        rewritten away into a bare, decorator-less ``DefNode`` plus a
        separate ``name = decorator(name)`` assignment (for
        ``@classmethod``/``@staticmethod``/``@functools.singledispatch``/
        ``@overload``, and presumably any other decorator). The
        corresponding function/method is already captured directly via
        its ``DefNode`` (with the decorator itself re-emitted from the
        pre-pipeline ``_stubgen_static_decorators`` snapshot -- see
        ``type_parsing.capture_static_types``), so keeping this
        rebinding too would emit it a second time, as a bogus module/
        class-level assignment. Most of the time this shape fails to
        unparse on its own (the call's argument is usually a
        ``PyCFunctionNode``, which ``unparse_expr`` can't handle) and
        gets silently dropped later regardless -- but not always: an
        ``@overload``-decorated method inside a ``cdef class``, where
        the argument stays a plain ``NameNode`` and unparses fine,
        leaks ``__getitem__ = overload(__getitem__)`` as a real
        class-body assignment. Detected directly and generally here
        instead of relying on that accident: a single-argument call
        whose argument refers back to the same name being assigned.
        """
        if isinstance(node.rhs, ExprNodes.ImportNode):
            return node
        if isinstance(node.rhs, ExprNodes.PyCFunctionNode):
            return node
        if isinstance(node.lhs, ExprNodes.NameNode) and _is_decorator_rebinding(
            node, node.lhs.name, self._def_positions
        ):
            return node
        if (
            isinstance(node.lhs, ExprNodes.NameNode)
            and node.lhs.name == "__annotations__"
            and isinstance(node.rhs, ExprNodes.DictNode)
        ):
            # Synthetic `__annotations__ = {...}` dict `AnalyseDeclarationsTransform`
            # folds every bare-annotated attribute in this scope into (module or
            # plain-class body). The individually-typed attributes it was built
            # from are recovered separately from `_stubgen_static_annotations`
            # (captured pre-pipeline, see `type_parsing.capture_static_types`)
            # and emitted by `Converter._convert_declared_entries`; keeping this
            # synthetic node too would emit a bogus, TypedDict-invalid
            # `__annotations__ = {...}` module/class-level assignment.
            return node
        if isinstance(node.lhs, ExprNodes.NameNode):
            entry = getattr(node.lhs, "entry", None)
            if entry is not None and entry.is_cglobal and entry.visibility == "private":
                # A plain (non-`public`, non-`readonly`) `cdef` global
                # survives here, as an ordinary-looking
                # `SingleAssignmentNode`, only because it has an
                # initializer -- an uninitialized one (`cdef int X`, no
                # `= ...`) is removed from `body.stats` entirely and never
                # reaches this method at all. Both are equally invisible
                # to Python at runtime (a plain `cdef` global gets no
                # module-level binding), so both must be excluded the same
                # way, or this one leaks into the stub looking like a real,
                # accessible name -- particularly misleading for a type
                # with no Python representation at all, e.g. a C function
                # pointer.
                return node
            self.assignments.append(node)
        return node

    def visit_ExprStatNode(self, node):
        """Collect annotated name expressions."""
        if (
            isinstance(node.expr, ExprNodes.NameNode)
            and node.expr.annotation is not None
        ):
            self.assignments.append(node)
        return node

    def visit_PyClassDefNode(self, node):
        """Collect Python class definitions."""
        self.classes.append(
            ClassVisitor(node=node, _fused_specialization_ids=self._fused_specialization_ids)
        )
        return node

    def visit_CClassDefNode(self, node):
        """Collect Cython extension type (cdef class) definitions."""
        self.classes.append(
            ClassVisitor(node=node, _fused_specialization_ids=self._fused_specialization_ids)
        )
        return node

    def visit_DefNode(self, node):
        """Collect Python function definitions.

        Skips the two node shapes the fused-type expansion transform
        leaves as ordinary siblings of the real declaration once analysis
        runs: each specialization's own Python-visible wrapper (entry
        name ``__pyx_fuse_N<name>``) and the single runtime dispatcher
        (plain entry name, but the *specialized*, signature-less shape --
        ``def f(signatures, args, kwargs, defaults, _fused_sigindex={})``).
        Both are pre-collected by ``_collect_fused_specialization_ids``;
        the real, still fused-typed declaration is collected separately
        via ``visit_FusedCFuncDefNode``.
        """
        if id(node) in self._fused_specialization_ids:
            return node
        self.py_functions.append(node)
        name = _declared_name(node)
        if name is not None:
            self._def_positions[name] = node.pos
        return node

    def visit_CFuncDefNode(self, node):
        """Collect Python-visible (cpdef) function definitions.

        Skips per-specialization ``CFuncDefNode``s the fused-type
        expansion transform leaves as siblings of the real declaration
        (entry name ``__pyx_fuse_N<name>``) -- see ``visit_DefNode``.
        """
        if id(node) in self._fused_specialization_ids:
            return node
        if not node.declarator.overridable:
            return node
        self.cdef_functions.append(node)
        name = _declared_name(node)
        if name is not None:
            self._def_positions[name] = node.pos
        return node

    def visit_FusedCFuncDefNode(self, node):
        """Collect the real, still fused-typed declaration a fused
        function's own node wraps.

        Once declaration analysis runs, a fused ``cdef``/``cpdef``/``def``
        function is expanded into one specialized ``CFuncDefNode``/
        ``DefNode`` per member type of its fused parameter(s), plus (for
        ``cpdef``/``def``) a runtime dispatcher -- all inserted as plain
        siblings of the original in ``body.stats``. Emitting all of
        those individually is exactly the duplicate-def (and
        dispatcher-signature-shaped) output fused stubs were regressing
        to. ``FusedCFuncDefNode.node`` is the *original*, unexpanded
        declaration -- never itself a sibling in ``body.stats`` -- with
        its fused-typed ``base_type``/annotations still intact
        (captured by ``capture_static_types`` pre-pipeline, same as any
        other node). Visiting it here, once, is what actually gets a
        fused function into ``cdef_functions``/``py_functions`` at all;
        the specializations/dispatcher are filtered out in
        ``visit_CFuncDefNode``/``visit_DefNode`` via
        ``_fused_specialization_ids``.
        """
        original = node.node
        if isinstance(original, Nodes.CFuncDefNode):
            self.visit_CFuncDefNode(original)
        elif isinstance(original, Nodes.DefNode):
            self.visit_DefNode(original)
        return node

    def visit_CTypeDefNode(self, node):
        """Collect simple Cython type definitions.

        Skips a `ctypedef` whose underlying type is (or is a pointer to) a
        C function -- e.g. `ctypedef int (*binop_t)(int, int)`. Unlike
        every other `ctypedef`, this one has no legitimate rendering at
        all: a raw C function pointer cannot cross the Python boundary in
        any form (confirmed directly -- Cython refuses to compile a
        `cdef public` attribute, a `cpdef` parameter, or a `cpdef` return
        of a function-pointer type, in every case with "Cannot convert
        ... to/from Python object"), so a `TypeAlias = Callable[...]`
        here is not just imprecise but actively false: nothing in real
        Python code could ever be assigned to or receive that alias.
        """
        from ..conversion.type_parsing import _declarator_name  # local: avoids a circular import with conversion.converter

        name = _declarator_name(node.declarator)
        scope = getattr(self.node, "scope", None)
        entry = scope.entries.get(name) if scope is not None and name else None
        if entry is not None and entry.is_type and getattr(entry.type, "is_typedef", False):
            base = entry.type.typedef_base_type
            if getattr(base, "is_cfunction", False) or (
                getattr(base, "is_ptr", False)
                and getattr(base.base_type, "is_cfunction", False)
            ):
                return node
        self.assignments.append(node)
        return node

    def visit_FusedTypeNode(self, node):
        """Collect Cython fused type definitions."""
        self.fused_types.append(node)
        return node

    def visit_CVarDefNode(self, node: Nodes.CVarDefNode):
        """Collect C variable/constance definitions"""
        visibility: str = node.visibility  # type: ignore
        # Only ``public`` and ``readonly`` are Python-visible
        if self.in_class and (visibility == "public" or visibility == "readonly"):
            self.cdef_variables.append(node)

    def visit_PropertyNode(self, node: Nodes.PropertyNode):
        """Collect the properties `AnalyseDeclarationsTransform` synthesizes
        for ``cdef public``/``cdef readonly`` extension-type attributes.

        Declaration analysis replaces the original `CVarDefNode` for such
        an attribute with a `PropertyNode` (`__get__`/`__set__` methods) --
        `visit_CVarDefNode` above never sees these anymore. Appended to
        the same `cdef_variables` list; `type_parsing.get_cdef_variables`
        dispatches on the node type when converting.
        """
        if self.in_class:
            self.cdef_variables.append(node)
        return node

    def visit_CStructOrUnionDefNode(self, node: Nodes.CStructOrUnionDefNode):
        """Collect C struct/union definitions.

        Skips Cython's own compiler-synthesized structs (`__pyx_`-prefixed
        names, e.g. `__pyx_opt_args_...` -- generated to bundle optional
        C arguments, such as a C-typed argument with a `None` default).
        These only start showing up as real `CStructOrUnionDefNode`s in
        `body.stats` once real declaration analysis runs (confirmed
        empirically: a `cpdef int bar(self, int x=None)` produces one of
        these); they're an implementation detail of how Cython compiles
        the function, never something the user declared.
        """
        if getattr(node, "name", "").startswith("__pyx_"):
            return node
        self.cdef_structs_or_unions.append(node)
        return node

    def visit_CppClassNode(self, node: Nodes.CppClassNode):
        """Collect C++ class definitions"""
        self.cpp_classes.append(node)
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

    # Forwarded straight through to the nested `ScopeVisitor` below -- see
    # `ScopeVisitor._fused_specialization_ids`'s docstring.
    _fused_specialization_ids: set[int] | None = None

    scope: ScopeVisitor = field(init=False)
    """A visitor for collecting scope nodes."""

    def __post_init__(self):
        self.scope = ScopeVisitor(
            node=self.node,
            in_class=True,
            _fused_specialization_ids=self._fused_specialization_ids,
        )


def _collect_attribute(node) -> str:
    names = []
    attribute = node

    while isinstance(attribute, ExprNodes.AttributeNode):
        names.append(attribute.attribute)
        attribute = attribute.obj

    if isinstance(attribute, ExprNodes.NameNode):
        names.append(attribute.name)

    names.reverse()

    name = ".".join(names)
    return name
