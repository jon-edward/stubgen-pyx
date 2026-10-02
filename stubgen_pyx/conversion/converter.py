"""
Converts Cython AST nodes to PyiElements.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from Cython.Compiler import Nodes

from ..analysis.visitor import ClassVisitor, ImportVisitor, ModuleVisitor, ScopeVisitor
from ..logging_utils import with_debug_fallback
from ..models.pyi_elements import (
    PyiAssignment,
    PyiClass,
    PyiEnum,
    PyiFunction,
    PyiFusedType,
    PyiImport,
    PyiModule,
    PyiScope,
)
from ..parsing.comments import CommentIndex
from ..postprocessing.normalize_names import _CYTHON_TRANSLATIONS
from .ctypedef_aliases import (
    _scope_functions,
    _substitute_ctypedef_aliases,
    _text_uses_name,
    apply_ctypedef_aliases,
    ctypedef_alias_map,
)
from .declarations import (
    convert_assignment,
    convert_cpp_class,
    convert_enum,
    convert_import,
    convert_struct_or_union,
)
from .docstrings import docstring_to_string
from .fused_types import (
    _annotation_uses_name,
    _resolve_fused_signature,
    _restore_fused_memoryview_annotations,
    convert_fused_type,
    convert_fused_types,
)
from .signature import get_signature
from .source_extraction import get_bases, get_decorators, get_metaclass, get_source
from .static_annotations import (
    _is_readonly_cdef_attribute,
    convert_declared_entries,
    recover_static_annotations,
)
from .type_comments import apply_type_comments
from .type_parsing import extract_name_and_type, get_cdef_variables

_CXX_FROM_CIMPORT_RE = re.compile(
    r"^\s*from\s+(?:libcpp|libc)(?:\.[^\s]+)?\s+cimport\b"
)
_CXX_CIMPORT_RE = re.compile(r"^\s*cimport\s+(?:libcpp|libc)(?:\.|\b)")


def _substitute_ctypedef_aliases_in_assignment(
    assignment: PyiAssignment | None,
    ctypedef_aliases: dict[str, str],
) -> PyiAssignment | None:
    """Apply `resolve_ctypedef_aliases`'s alias -> underlying-type
    substitution to a plain assignment's whole rendered statement, not
    just an isolated type string.

    Every other substitution site (`_convert_cdef_assignments`,
    `apply_ctypedef_aliases` for a signature) only ever needs to
    replace a name inside a string that's *already* known to be nothing
    but a type. A plain assignment's value can itself reference a
    `ctypedef` alias by name -- `unparse.py`'s `visit_TypecastNode`
    renders a C-style cast's declared type straight into the value
    expression (``typing.cast(underlying_type_t_compression, 1)``) --
    and that occurrence needs the exact same substitution, or the
    now-inlined-everywhere-else alias reads as unused and gets pruned
    while this one remaining, unsubstituted reference still needs it
    (left dangling, then wiped out entirely by `trim_not_defined`
    reading it as a reference to an undefined name).
    """
    if assignment is None or not ctypedef_aliases:
        return assignment
    assignment.statement = _substitute_ctypedef_aliases(
        assignment.statement, ctypedef_aliases
    )
    return assignment


_CYTHON_IMPORT_RE = re.compile(
    r"^\s*from\s+(?:cython|cpython)(?:\.[^\s]+)*\s+c?import\b"
)
_CYTHON_FROM_IMPORT_RE = re.compile(r"^\s*c?import\s+(?:cython|cpython)(?:\.[^\s]+)*\b")

#: Matches a rendered ``@dataclass``/``@dataclasses.dataclass`` decorator
#: (any dotted prefix, e.g. ``@cython.dataclasses.dataclass``), with or
#: without a call -- `callee` is everything through the final ``dataclass``
#: segment, `args` is whatever's inside the parens, if any. See
#: `_add_init_false_to_dataclass_decorator`.
_DATACLASS_DECORATOR_RE = re.compile(
    r"^@(?P<callee>(?:\w+\.)*dataclass)(?:\((?P<args>.*)\))?$", re.DOTALL
)
#: An already-explicit ``init=`` keyword argument -- never overridden by
#: `_add_init_false_to_dataclass_decorator`, and never duplicated.
_DATACLASS_INIT_KWARG_RE = re.compile(r"(?<![\w.])init\s*=")


def _add_init_false_to_dataclass_decorator(decorator: str) -> str:
    """Add ``init=False`` to a ``@dataclass``-family decorator string,
    creating the call parens if the decorator is currently bare.

    Cython's own dataclass support (see `Converter._patch_dataclass_init_annotations`'s
    docstring) synthesizes a real, correctly-ordered ``__init__`` for a
    ``cdef class`` and stubgen-pyx always renders it explicitly in the
    class body -- so once that happens, the ``@dataclass`` decorator
    itself no longer needs to (and shouldn't) tell a type checker to
    synthesize *another* one from the class-body field annotations: the
    two are derived independently (one from Cython's own field
    collection, the other from whatever a type checker's dataclass
    support infers from the rendered attributes) and nothing keeps them
    in sync, particularly for a private field with no class-body
    annotation at all. ``init=False`` makes the explicit, real ``__init__``
    the only one a type checker considers, unambiguously.

    Leaves any other decorator, and any dataclass decorator that already
    sets ``init=`` explicitly (respecting whatever the source itself
    asked for), untouched.
    """
    match = _DATACLASS_DECORATOR_RE.match(decorator)
    if match is None:
        return decorator
    args = match.group("args")
    if args is not None and _DATACLASS_INIT_KWARG_RE.search(args):
        return decorator
    callee = match.group("callee")
    if not args:
        return f"@{callee}(init=False)"
    return f"@{callee}({args}, init=False)"


_logger = logging.getLogger(__name__)


class ConversionError(Exception):
    """An error occurred during the conversion process."""


@dataclass
class Converter:
    """Converts Cython AST visitors to PyiElements for code generation.

    Transforms Cython Compiler AST nodes (as collected by visitors) into
    intermediate PyiElement representations for building .pyi stub files.
    """

    cimport_alias_map: dict[str, str] = field(default_factory=dict)

    def convert_module(
        self,
        visitor: ModuleVisitor,
        source_code: str,
        comments: CommentIndex | None = None,
        include_docstrings: bool = True,
        inherited_fused_types: dict[str, PyiFusedType] | None = None,
        resolve_ctypedef_aliases: bool = False,
        defer_ctypedef_pruning: bool = False,
        prunable_ctypedef_aliases: dict[str, PyiAssignment] | None = None,
    ) -> PyiModule:
        """Convert a ModuleVisitor to a PyiModule.

        Args:
            visitor: Module visitor containing AST information.
            source_code: The original source code text.
            comments: This file's comments (see `parsing/comments.py`),
                used to recover `# type: ...` comments a function's
                signature and/or individual arguments didn't otherwise
                get from a real annotation or Cython declaration.
            resolve_ctypedef_aliases: See `StubgenPyxConfig`.
            defer_ctypedef_pruning: See `convert_scope`'s docstring --
                used only by `stubgen.py`'s `convert_multiple_files`, for
                its whole-batch cross-file check.
            prunable_ctypedef_aliases: Populated (when `defer_ctypedef_pruning`
                is set) with every alias this module's own scope tree
                couldn't confirm as truly unused on its own -- see
                `convert_scope`.

        Returns:
            PyiModule representation with imports and scope.
        """
        comments = comments if comments is not None else CommentIndex([])
        doc = docstring_to_string(visitor.node.doc) if visitor.node.doc else None
        scope = self.convert_scope(
            visitor.scope,
            source_code,
            comments,
            include_docstrings,
            inherited_fused_types,
            emit_inherited_fused_typevars=True,
            resolve_ctypedef_aliases=resolve_ctypedef_aliases,
            defer_ctypedef_pruning=defer_ctypedef_pruning,
            prunable_ctypedef_aliases=prunable_ctypedef_aliases,
        )

        return PyiModule(
            doc=doc if include_docstrings else None,
            imports=self.convert_imports(visitor.import_visitor, source_code),
            scope=scope,
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
            if _is_cython_import(raw):
                _logger.debug("Ignoring Cython import: %r", raw)
                continue
            imports.append(convert_import(node, source_code, raw))
        return imports

    def _collect_cimport_aliases(self, node: Nodes.Node) -> None:
        if not isinstance(node, Nodes.FromCImportStatNode):
            return
        for _, original, alias in node.imported_names or ():
            if alias and alias != original:
                python_type = _CYTHON_TRANSLATIONS.get(original)
                if python_type:
                    self.cimport_alias_map[alias] = python_type

    def _convert_cdef_assignments(
        self,
        visitor: ScopeVisitor,
        ctypedef_aliases: dict[str, str],
        include_docstrings: bool = True,
    ) -> tuple[list[PyiAssignment], set[str], dict[str, str]]:
        assignments: list[PyiAssignment] = []
        handled_names: set[str] = set()
        resolved_types: dict[str, str] = {}
        static_property_types = (
            getattr(visitor.node, "_stubgen_static_property_types", None) or {}
        )
        static_python_annotations = (
            getattr(visitor.node, "_stubgen_static_python_annotations", None) or {}
        )
        for cdef_variable in visitor.cdef_variables:
            is_readonly = _is_readonly_cdef_attribute(cdef_variable)
            for name, base_type in get_cdef_variables(cdef_variable):
                if name in static_python_annotations:
                    # A Python-style-annotated attribute (bare or
                    # defaulted; a plain Python class, a generic, or a
                    # union with `None`) always renders from its own
                    # source annotation, never from `Entry.type` --
                    # `_capture_annotated_assignment_property_type`'s
                    # docstring has the full reasoning for why the two
                    # can disagree (a generic `PyObject*` slot indifferent
                    # to which Python class it holds, a real scalar slot
                    # silently dropping a `| None`, a lost generic
                    # parameter) and why the source annotation is always
                    # the more correct answer for a `.pyi` to show.
                    base_type = static_python_annotations[name]
                else:
                    static_type = static_property_types.get(name)
                    if (
                        static_type is not None
                        and base_type is not None
                        and static_type.endswith(f".{base_type}")
                    ):
                        base_type = static_type

                resolved_type = with_debug_fallback(
                    base_type,
                    "_typeshed.Incomplete",
                    f"Unable to determine type for {name}",
                )
                resolved_type = _substitute_ctypedef_aliases(
                    resolved_type, ctypedef_aliases
                )
                # A `PropertyNode` (a get+set property, or the synthesized
                # pair for `cdef public`/`cdef readonly`) carries its own
                # docstring on `.doc`, same as a real `def` method's --
                # a raw `CVarDefNode` has no such attribute (Cython has no
                # syntax to attach a docstring to a plain `cdef` field), so
                # `getattr` with a default covers both without a type check.
                node_doc = getattr(cdef_variable, "doc", None)
                doc = (
                    docstring_to_string(node_doc)
                    if include_docstrings and node_doc
                    else None
                )
                # `cdef readonly` has no Python-level setter -- assigning
                # to it from outside the extension type raises at
                # runtime. Wrapping the annotation in `Final` (only here,
                # on the rendered attribute; `resolved_types` below keeps
                # the bare type, since a dataclass's synthesized
                # `__init__` parameter still needs its plain type) makes
                # a type checker reject that assignment statically too,
                # instead of only a real interpreter run catching it.
                annotation = (
                    f"typing.Final[{resolved_type}]" if is_readonly else resolved_type
                )
                assignments.append(
                    PyiAssignment(f"{name}: {annotation}", name=name, doc=doc)
                )
                handled_names.add(name)
                resolved_types[name] = resolved_type
        return assignments, handled_names, resolved_types

    @staticmethod
    def _patch_dataclass_init_annotations(
        visitor: ScopeVisitor,
        source_code: str,
        py_funcs: list[tuple[int, PyiFunction]],
        cdef_resolved_types: dict[str, str],
    ) -> None:
        """Fix up a compiler-synthesized dataclass ``__init__``'s parameter
        annotations for parameters that correspond to ``cdef public``/
        ``cdef readonly`` (or Python-style-annotated, still C-typed)
        extension-type attributes.

        Cython's own ``AnalyseDeclarationsTransform`` builds this
        ``__init__`` (real `dataclasses`-module support for extension
        types, not anything stubgen-pyx generates itself), correctly
        ordered and everything -- but each synthesized parameter's own
        ``annotation`` is unreliable for exactly the attributes
        `_convert_cdef_assignments` already had to resolve properly to
        render as a class-body ``name: type`` line, and for the same two
        reasons:

        - A field declared via ``cdef public``/``cdef readonly`` (C-style,
          no ``: type`` syntax to carry over) leaves its parameter with no
          annotation and no resolvable ``base_type`` node either, so
          `_to_argument` renders it bare (``z`` instead of ``z: float``)
          -- silently wrong, and for any field after it that still has a
          default, enough on its own to produce a `def __init__(...)`
          that fails to parse as Python at all (a default-less bare name
          follows a defaulted one).
        - A field declared with a Python-style annotation that happens to
          name a Cython pure-Python-mode type (``x: cython.double = 0.0``)
          *does* carry that annotation over verbatim onto the parameter --
          but verbatim means literally ``cython.double``, and stubgen-pyx
          never emits a ``cython`` import into the stub (there's no
          Python-level ``cython`` module to import at all outside a
          Cython build). `trim_not_defined` then finds no binding for
          ``cython`` and replaces the whole annotation with
          `_typeshed.Incomplete` -- again silently wrong, and
          inconsistent with the same attribute's own, correctly
          ``float``-rendered class-body annotation.

        Both are fixed the same way: every parameter that names a field
        `_convert_cdef_assignments` already resolved gets that exact
        resolved type here too, replacing whatever the compiler's own
        synthesized annotation says (or doesn't). This keeps a
        dataclass's ``__init__`` parameter types and its class-body
        attribute types in sync by construction, rather than trusting two
        separately-derived renderings of the same underlying type to
        agree.

        A private (neither ``public`` nor ``readonly``) attribute is still
        a real dataclass field -- Cython includes it in the synthesized
        ``__init__`` regardless of Python visibility -- but it never
        becomes a `PropertyNode`/`CVarDefNode` `_convert_cdef_assignments`
        walks (there's no Python-visible property to render), so it never
        makes it into `cdef_resolved_types`. `static_annotations.capture_static_types`'
        pre-pipeline ``_stubgen_static_property_types`` snapshot still has
        it (from either its raw ``cdef`` declaration or a Python-style
        annotation -- see `_capture_property_type`/
        `_capture_annotated_assignment_property_type`), as a raw Cython
        type name (``double``, not ``float``) that
        `postprocessing.normalize_names` normalizes later -- used here as
        the fallback for exactly the names `cdef_resolved_types` doesn't
        cover.
        """
        if not visitor.in_class:
            return
        static_property_types = (
            getattr(visitor.node, "_stubgen_static_property_types", None) or {}
        )
        if not cdef_resolved_types and not static_property_types:
            return
        if not any(
            "dataclass" in decorator
            for decorator in get_decorators(source_code, visitor.node)
        ):
            return
        for _, py_func in py_funcs:
            if py_func.name != "__init__":
                continue
            for argument in py_func.signature.args:
                resolved = cdef_resolved_types.get(
                    argument.name
                ) or static_property_types.get(argument.name)
                if resolved is not None:
                    argument.annotation = resolved

    def _convert_scope_members(
        self,
        visitor: ScopeVisitor,
        source_code: str,
        comments: CommentIndex,
        include_docstrings: bool,
        fused_types: dict[str, PyiFusedType],
        ctypedef_aliases: dict[str, str],
        resolve_ctypedef_aliases: bool,
        defer_ctypedef_pruning: bool,
        prunable_ctypedef_aliases: dict[str, PyiAssignment] | None,
        cdef_resolved_types: dict[str, str],
    ) -> tuple[list[PyiFunction], list[PyiClass], list[PyiClass]]:
        cdef_funcs = [
            (
                node.pos[1],
                self.convert_cdef_func(
                    node,
                    source_code,
                    comments,
                    include_docstrings,
                    fused_types,
                    ctypedef_aliases=ctypedef_aliases,
                ),
            )
            for node in visitor.cdef_functions
        ]
        py_funcs = [
            (
                node.pos[1],
                self.convert_py_func(
                    node,
                    source_code,
                    comments,
                    include_docstrings,
                    fused_types,
                    ctypedef_aliases=ctypedef_aliases,
                ),
            )
            for node in visitor.py_functions
        ]
        property_getters = [
            (
                node.pos[1],
                self.convert_property_getter(
                    node,
                    comments,
                    include_docstrings,
                    fused_types,
                    ctypedef_aliases=ctypedef_aliases,
                ),
            )
            for node in visitor.property_getters
        ]

        self._patch_dataclass_init_annotations(
            visitor, source_code, py_funcs, cdef_resolved_types
        )

        contains_init = any(py_func.name == "__init__" for _, py_func in py_funcs)
        for idx, (_, py_func) in enumerate(py_funcs):
            if py_func.name == "__cinit__" and not contains_init:
                py_func.name = "__init__"
            elif py_func.name == "__cinit__" and contains_init:
                del py_funcs[idx]
                break

        functions = [
            function
            for _, function in sorted(
                cdef_funcs + py_funcs + property_getters, key=lambda item: item[0]
            )
        ]
        structs_or_enums = [
            convert_struct_or_union(node) for node in visitor.cdef_structs_or_unions
        ]
        classes = [
            self.convert_class(
                class_visitor,
                source_code,
                comments,
                include_docstrings,
                fused_types,
                resolve_ctypedef_aliases=resolve_ctypedef_aliases,
                inherited_ctypedef_aliases=ctypedef_aliases,
                defer_ctypedef_pruning=defer_ctypedef_pruning,
                prunable_ctypedef_aliases=prunable_ctypedef_aliases,
            )
            for class_visitor in visitor.classes
        ]
        return functions, classes, structs_or_enums

    @staticmethod
    def _collect_scope_handled_names(
        visitor: ScopeVisitor,
        functions: list[PyiFunction],
        cdef_handled_names: set[str],
    ) -> set[str]:
        handled_names = set(cdef_handled_names)
        handled_names.update(function.name for function in functions)
        handled_names.update(
            (
                class_visitor.node.class_name
                if isinstance(class_visitor.node, Nodes.CClassDefNode)
                else class_visitor.node.name
            )
            for class_visitor in visitor.classes
        )
        handled_names.update(
            name
            for name in (
                getattr(node, "name", None) for node in visitor.cdef_structs_or_unions
            )
            if name
        )
        handled_names.update(
            name
            for name in (getattr(node, "name", None) for node in visitor.cpp_classes)
            if name
        )
        handled_names.update(
            name
            for name in (
                getattr(getattr(assignment, "lhs", None), "name", None)
                for assignment in visitor.assignments
            )
            if name
        )
        handled_names.update(
            name
            for name in (
                getattr(getattr(assignment, "declarator", None), "name", None)
                for assignment in visitor.assignments
            )
            if name
        )
        handled_names.update(
            name
            for name in (getattr(enum, "name", None) for enum in visitor.enums)
            if name
        )
        return handled_names

    @staticmethod
    def _find_fused_typevar_names(
        typevar_candidates: dict[str, PyiFusedType],
        functions: list[PyiFunction],
        classes: list[PyiClass],
    ) -> list[str]:
        scope = PyiScope(functions=functions, classes=classes)
        return [
            name
            for name in typevar_candidates
            if any(
                any(
                    _annotation_uses_name(argument.annotation, name)
                    for argument in function.signature.args
                )
                or _annotation_uses_name(function.signature.return_type, name)
                for function in _scope_functions(scope)
            )
        ]

    def _prune_scope_assignments(
        self,
        conv_assignments_with_source: list[tuple[object, PyiAssignment | None]],
        functions: list[PyiFunction],
        classes: list[PyiClass],
        cdef_assignments: list[PyiAssignment],
        extra_assignments: list[PyiAssignment],
        local_ctypedef_aliases: dict[str, str],
        resolve_ctypedef_aliases: bool,
        defer_ctypedef_pruning: bool,
    ) -> tuple[list[PyiAssignment], list[tuple[str, PyiAssignment]]]:
        functions_and_classes = PyiScope(functions=functions, classes=classes)
        other_statements = [assignment.statement for assignment in cdef_assignments] + [
            assignment.statement
            for assignment in extra_assignments
            if isinstance(assignment, PyiAssignment)
        ]

        def _ctypedef_alias_name(raw_node) -> str | None:
            if not (
                resolve_ctypedef_aliases and isinstance(raw_node, Nodes.CTypeDefNode)
            ):
                return None
            name, _ = extract_name_and_type(raw_node)
            return name if name in local_ctypedef_aliases else None

        def _is_used_elsewhere(
            name: str, own_statement: str | None, live: list[PyiAssignment]
        ) -> bool:
            for function in _scope_functions(functions_and_classes):
                if any(
                    _text_uses_name(argument.annotation, name)
                    for argument in function.signature.args
                ) or _text_uses_name(function.signature.return_type, name):
                    return True
            for converted in live:
                if converted.statement != own_statement and _text_uses_name(
                    converted.statement, name
                ):
                    return True
            return any(
                _text_uses_name(statement, name) for statement in other_statements
            )

        live_assignments = [
            (raw_node, converted)
            for raw_node, converted in conv_assignments_with_source
            if converted is not None
        ]
        locally_dead: list[tuple[str, PyiAssignment]] = []
        for _ in range(len(live_assignments)):
            pruned_this_pass = False
            still_live = []
            live_converted = [converted for _, converted in live_assignments]
            for raw_node, converted in live_assignments:
                alias_name = _ctypedef_alias_name(raw_node)
                if alias_name is not None and not _is_used_elsewhere(
                    alias_name, converted.statement, live_converted
                ):
                    pruned_this_pass = True
                    locally_dead.append((alias_name, converted))
                    continue
                still_live.append((raw_node, converted))
            live_assignments = still_live
            if not pruned_this_pass:
                break

        converted_assignments = [converted for _, converted in live_assignments]
        if defer_ctypedef_pruning:
            converted_assignments += [converted for _, converted in locally_dead]
        return converted_assignments, locally_dead

    def convert_scope(
        self,
        visitor: ScopeVisitor,
        source_code: str,
        comments: CommentIndex | None = None,
        include_docstrings: bool = True,
        inherited_fused_types: dict[str, PyiFusedType] | None = None,
        emit_inherited_fused_typevars: bool = False,
        resolve_ctypedef_aliases: bool = False,
        inherited_ctypedef_aliases: dict[str, str] | None = None,
        defer_ctypedef_pruning: bool = False,
        prunable_ctypedef_aliases: dict[str, PyiAssignment] | None = None,
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

        ``ctypedef`` aliases don't have this same one-owner problem --
        unlike a fused type, resolving one to its underlying type doesn't
        emit anything of its own, so every scope in the nesting chain
        (module, then each nested class) merges ``inherited_ctypedef_
        aliases`` with whatever it finds locally and just uses the result
        directly wherever `resolve_ctypedef_aliases` applies it, the same
        way `fused_types` (as opposed to ``local_fused_types``) is merged
        and used for signature resolution just below.

        ``defer_ctypedef_pruning``/``prunable_ctypedef_aliases`` exist for
        exactly one caller: `stubgen.py`'s `convert_multiple_files`, doing
        the whole-batch cross-file check described in
        `_prune_ctypedef_aliases_across_batch`'s docstring. A single
        file's own local usage -- functions, attributes, a still-live
        sibling alias's declaration -- is *never* enough on its own to
        know an alias is truly dead: another file in the same batch may
        `cimport` it and be the only thing that still needs it, and this
        scope has no way to see that on its own (see the design note on
        `StubgenPyxConfig.resolve_ctypedef_aliases`). When
        `defer_ctypedef_pruning` is set, a locally-dead alias's
        `(scope.assignments, PyiAssignment)` pair is recorded into
        `prunable_ctypedef_aliases` (keyed by name) instead of being
        dropped here -- left in the output, provisionally, for the
        caller to actually remove (by mutating `scope.assignments` in
        place, since a plain list is passed) once it has cross-checked
        every other file in the batch too. Single-file conversion
        (`convert_str`/`compile_str_to_module`, or `convert_single_file`
        called outside a batch) leaves this off, keeping today's
        immediate, local-only pruning -- the best available without a
        batch to cross-check against, and safe for exactly the reason a
        single, self-contained snippet has no other files to break.
        """
        comments = comments if comments is not None else CommentIndex([])
        local_fused_types = convert_fused_types(visitor)
        fused_types = {**(inherited_fused_types or {}), **local_fused_types}
        local_ctypedef_aliases = (
            ctypedef_alias_map(visitor) if resolve_ctypedef_aliases else {}
        )
        ctypedef_aliases = {
            **(inherited_ctypedef_aliases or {}),
            **local_ctypedef_aliases,
        }

        cdef_assignments, cdef_handled_names, cdef_resolved_types = (
            self._convert_cdef_assignments(
                visitor, ctypedef_aliases, include_docstrings
            )
        )
        functions, classes, structs_or_enums = self._convert_scope_members(
            visitor,
            source_code,
            comments,
            include_docstrings,
            fused_types,
            ctypedef_aliases,
            resolve_ctypedef_aliases,
            defer_ctypedef_pruning,
            prunable_ctypedef_aliases,
            cdef_resolved_types,
        )
        typevar_candidates = (
            fused_types if emit_inherited_fused_typevars else local_fused_types
        )
        fused_typevar_names = self._find_fused_typevar_names(
            typevar_candidates, functions, classes
        )

        conv_assignments_with_source = [
            (
                assignment,
                (
                    convert_assignment(
                        assignment, source_code, in_class=visitor.in_class
                    )
                    if isinstance(assignment, Nodes.CTypeDefNode)
                    # A `ctypedef`'s own defining statement must never go
                    # through this substitution -- it's the one place the
                    # alias name is *supposed* to appear undisturbed, and
                    # substituting it there rewrites the assignment's own
                    # target name into the resolved type
                    # (`size_type: TypeAlias = int` -> `int: TypeAlias =
                    # int`), which every other pruning/usage check then
                    # reads as the alias no longer existing at all.
                    else _substitute_ctypedef_aliases_in_assignment(
                        convert_assignment(
                            assignment, source_code, in_class=visitor.in_class
                        ),
                        ctypedef_aliases,
                    )
                ),
            )
            for assignment in visitor.assignments
        ]

        cpp_classes = (convert_cpp_class(node) for node in visitor.cpp_classes)

        handled_names = self._collect_scope_handled_names(
            visitor, functions, cdef_handled_names
        )

        # Recovered here, before `convert_declared_entries`, so its
        # `handled_names.add(name)` calls (mutating the same set) are
        # visible to that call's `scope.entries` walk and it doesn't
        # also emit a duplicate fallback entry for the same name.
        recovered_annotations = recover_static_annotations(
            visitor, handled_names, ctypedef_aliases
        )
        # Merged into `conv_assignments_with_source` by original source
        # line (see `recover_static_annotations`'s docstring) rather
        # than appended separately after pruning: a bare `x: int`
        # attribute and an ordinary `y: int = 0` one in the same class
        # are two arbitrarily-interleaved buckets by the time they reach
        # here, and rendering them bucket-by-bucket instead of by
        # source order can silently reorder a dataclass's fields --
        # turning a source order `dataclasses`/mypy accepts into one
        # they reject (a default-less field placed after a defaulted
        # one). `raw_node` is `None` for a recovered annotation (no
        # surviving AST node to carry) -- fine, since the only thing
        # `_prune_scope_assignments` does with it is a `CTypeDefNode`
        # isinstance check, which is trivially `False` for `None` too.
        positioned_assignments = [
            (raw_node.pos[1], raw_node, converted)
            for raw_node, converted in conv_assignments_with_source
        ] + [(line, None, converted) for line, converted in recovered_annotations]
        conv_assignments_with_source = [
            (raw_node, converted)
            for _, raw_node, converted in sorted(
                positioned_assignments, key=lambda item: item[0]
            )
        ]

        extra_assignments, extra_enums, extra_structs, extra_functions = (
            convert_declared_entries(visitor, handled_names, ctypedef_aliases)
        )
        functions = functions + extra_functions

        conv_assignments, locally_dead = self._prune_scope_assignments(
            conv_assignments_with_source,
            functions,
            classes,
            cdef_assignments,
            extra_assignments,
            local_ctypedef_aliases,
            resolve_ctypedef_aliases,
            defer_ctypedef_pruning,
        )

        scope = PyiScope(
            assignments=[
                convert_fused_type(fused_types[name]) for name in fused_typevar_names
            ]
            + conv_assignments
            + [c for c in cpp_classes if c]
            + cdef_assignments
            + [a for a in extra_assignments if isinstance(a, PyiAssignment)]
            + [a for a in extra_enums if isinstance(a, PyiAssignment)],
            functions=functions,
            classes=structs_or_enums + classes + extra_structs,
            enums=[convert_enum(enum) for enum in visitor.enums]
            + [e for e in extra_enums if isinstance(e, PyiEnum)],
        )
        if defer_ctypedef_pruning and prunable_ctypedef_aliases is not None:
            for alias_name, converted in locally_dead:
                prunable_ctypedef_aliases[alias_name] = converted
        return scope

    def convert_class(
        self,
        class_visitor: ClassVisitor,
        source_code: str,
        comments: CommentIndex | None = None,
        include_docstrings: bool = True,
        inherited_fused_types: dict[str, PyiFusedType] | None = None,
        resolve_ctypedef_aliases: bool = False,
        inherited_ctypedef_aliases: dict[str, str] | None = None,
        defer_ctypedef_pruning: bool = False,
        prunable_ctypedef_aliases: dict[str, PyiAssignment] | None = None,
    ) -> PyiClass:
        """Convert a ClassVisitor to a PyiClass."""
        comments = comments if comments is not None else CommentIndex([])
        if isinstance(class_visitor.node, Nodes.CClassDefNode):
            name: str = class_visitor.node.class_name  # type: ignore
        else:
            name: str = class_visitor.node.name

        node_doc: str | None = getattr(class_visitor.node, "doc", None)
        doc = docstring_to_string(node_doc) if node_doc else None

        scope = self.convert_scope(
            class_visitor.scope,
            source_code,
            comments,
            include_docstrings,
            inherited_fused_types,
            resolve_ctypedef_aliases=resolve_ctypedef_aliases,
            inherited_ctypedef_aliases=inherited_ctypedef_aliases,
            defer_ctypedef_pruning=defer_ctypedef_pruning,
            prunable_ctypedef_aliases=prunable_ctypedef_aliases,
        )
        decorators = get_decorators(source_code, class_visitor.node)
        if any(function.name == "__init__" for function in scope.functions):
            # An explicit `__init__` is already in the class body -- real
            # `dataclasses`-module behavior never overwrites one (whether
            # it's the user's own or Cython's compiler-synthesized one for
            # a `cdef class`, see `_patch_dataclass_init_annotations`'s
            # docstring), so a type checker's dataclass support shouldn't
            # synthesize a second, independently-derived one from the
            # class-body field annotations either -- `init=False` says so
            # explicitly instead of relying on each type checker's own
            # (unspecified, possibly inconsistent) handling of a
            # `@dataclass`-decorated class that already defines `__init__`.
            decorators = [
                _add_init_false_to_dataclass_decorator(decorator)
                for decorator in decorators
            ]

        return PyiClass(
            name=name,
            doc=doc if include_docstrings else None,
            bases=get_bases(class_visitor.node),
            metaclass=get_metaclass(class_visitor.node),
            decorators=decorators,
            scope=scope,
        )

    def convert_cdef_func(
        self,
        cdef_func: Nodes.CFuncDefNode,
        source_code: str,
        comments: CommentIndex | None = None,
        include_docstrings: bool = True,
        fused_types: dict[str, PyiFusedType] | None = None,
        ctypedef_aliases: dict[str, str] | None = None,
    ) -> PyiFunction:
        """Convert a C function definition node to PyiFunction."""
        comments = comments if comments is not None else CommentIndex([])
        name: str = cdef_func.declarator.base.name  # type: ignore
        doc = docstring_to_string(cdef_func.doc) if cdef_func.doc else None  # type: ignore
        signature = _resolve_fused_signature(
            _restore_fused_memoryview_annotations(
                get_signature(cdef_func), cdef_func, fused_types or {}
            ),
            fused_types or {},
        )
        raw_fallback = apply_type_comments(
            signature,
            cdef_func,
            cdef_func.declarator.args,
            comments,  # type: ignore
        )
        apply_ctypedef_aliases(signature, ctypedef_aliases)
        return PyiFunction(
            name,
            is_async=False,
            doc=doc if include_docstrings else None,
            decorators=get_decorators(source_code, cdef_func),
            signature=signature,
            type_comment=raw_fallback,
        )

    def convert_py_func(
        self,
        node: Nodes.DefNode,
        source_code: str,
        comments: CommentIndex | None = None,
        include_docstrings: bool = True,
        fused_types: dict[str, PyiFusedType] | None = None,
        ctypedef_aliases: dict[str, str] | None = None,
    ) -> PyiFunction:
        """Convert a Python function definition node to PyiFunction."""
        comments = comments if comments is not None else CommentIndex([])
        name = node.name  # type: ignore
        doc = docstring_to_string(node.doc) if node.doc else None
        signature = _resolve_fused_signature(
            _restore_fused_memoryview_annotations(
                get_signature(node), node, fused_types or {}
            ),
            fused_types or {},
        )
        raw_fallback = apply_type_comments(
            signature,
            node,
            node.args,
            comments,  # type: ignore
        )
        apply_ctypedef_aliases(signature, ctypedef_aliases)
        return PyiFunction(
            name,
            is_async=node.is_async_def,
            doc=doc if include_docstrings else None,
            decorators=get_decorators(source_code, node),
            signature=signature,
            type_comment=raw_fallback,
        )

    def convert_property_getter(
        self,
        node: Nodes.PropertyNode,
        comments: CommentIndex | None = None,
        include_docstrings: bool = True,
        fused_types: dict[str, PyiFusedType] | None = None,
        ctypedef_aliases: dict[str, str] | None = None,
    ) -> PyiFunction:
        """Convert a get-only, real ``@property`` to a ``@property``-decorated
        `PyiFunction` (see `ScopeVisitor.visit_PropertyNode`).

        Built from the property's own `__get__` `DefNode` rather than
        `convert_py_func`: its name is always the synthetic `"__get__"`
        (the real, user-given name -- ``x`` for ``def x(self): ...`` --
        lives on the enclosing `PropertyNode` instead), and its
        `decorators` never carries the original `@property` (consumed by
        the compiler when it built this node) -- re-added explicitly here.
        """
        comments = comments if comments is not None else CommentIndex([])
        getter = next(
            (s for s in node.body.stats if getattr(s, "name", None) == "__get__"),
            None,
        )
        doc = docstring_to_string(node.doc) if node.doc else None
        signature = _resolve_fused_signature(
            _restore_fused_memoryview_annotations(
                get_signature(getter), getter, fused_types or {}
            ),
            fused_types or {},
        )
        raw_fallback = apply_type_comments(
            signature,
            getter,
            getter.args,
            comments,  # type: ignore
        )
        apply_ctypedef_aliases(signature, ctypedef_aliases)
        return PyiFunction(
            node.name,
            is_async=False,
            doc=doc if include_docstrings else None,
            decorators=["@property"],
            signature=signature,
            type_comment=raw_fallback,
        )


def _is_cxx_cimport(raw: str) -> bool:
    return bool(_CXX_FROM_CIMPORT_RE.search(raw) or _CXX_CIMPORT_RE.search(raw))


def _is_cython_import(raw: str) -> bool:
    return bool(_CYTHON_FROM_IMPORT_RE.search(raw) or _CYTHON_IMPORT_RE.search(raw))
