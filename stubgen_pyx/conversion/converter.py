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
    PyiArgument,
    PyiAssignment,
    PyiClass,
    PyiEnum,
    PyiFunction,
    PyiFusedType,
    PyiImport,
    PyiModule,
    PyiScope,
    PyiSignature,
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
    convert_struct_or_union_type,
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
from .type_comments import apply_type_comments
from .type_parsing import (
    extract_name_and_type,
    get_cdef_variables,
    render_pyrex_type,
)

_CXX_FROM_CIMPORT_RE = re.compile(
    r"^\s*from\s+(?:libcpp|libc)(?:\.[^\s]+)?\s+cimport\b"
)
_CXX_CIMPORT_RE = re.compile(r"^\s*cimport\s+(?:libcpp|libc)(?:\.|\b)")

_CYTHON_IMPORT_RE = re.compile(
    r"^\s*from\s+(?:cython|cpython)(?:\.[^\s]+)*\s+c?import\b"
)
_CYTHON_FROM_IMPORT_RE = re.compile(r"^\s*c?import\s+(?:cython|cpython)(?:\.[^\s]+)*\b")


_logger = logging.getLogger(__name__)


class ConversionError(Exception):
    """An error occurred during the conversion process."""


def _convert_declared_function(name: str, t) -> PyiFunction:
    """Convert a function-shaped Entry's `CFuncType` -- no surviving
    `CFuncDefNode`/`DefNode` (e.g. a `cpdef` declared directly inside
    `cdef extern from ...:`, with no body of its own) -- to a
    `PyiFunction`.

    No default values: `CFuncTypeArg` carries no default-value
    information at all (defaults live on the original AST node, which
    doesn't survive), so an argument with a real default renders without
    one here. A narrower result than the structural path gives, but
    still a real, usable signature rather than nothing.
    """
    args = [
        PyiArgument(
            name=arg.name or f"arg{i}",
            annotation=with_debug_fallback(
                render_pyrex_type(arg.type),
                "_typeshed.Incomplete",
                lambda name_=arg.name: f"Unable to determine type for {name_}",
            ),
        )
        for i, arg in enumerate(t.args)
    ]
    return_type = with_debug_fallback(
        render_pyrex_type(t.return_type),
        "_typeshed.Incomplete",
        lambda: f"Unable to determine return type for {name}",
    )
    return PyiFunction(
        name=name,
        is_async=False,
        signature=PyiSignature(args=args, return_type=return_type),
    )


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

    def _convert_declared_entries(
        self,
        visitor: ScopeVisitor,
        handled_names: set[str],
        ctypedef_aliases: dict[str, str] | None = None,
    ) -> tuple[
        list[PyiAssignment],
        list[PyiEnum | PyiAssignment],
        list[PyiClass],
        list[PyiFunction],
    ]:
        """Catch declarations with no surviving AST node in this scope.

        Once real declaration analysis runs, a declaration with no
        runtime/executable component (an uninitialized variable, a
        ``cdef enum``, a ``cdef struct``/``union``, a ``cdef fused`` type
        with no methods) is removed from the tree entirely and exists
        only as a ``Symtab.Entry``. `visitor`'s structural walk
        (``visit_CVarDefNode``, etc.) can only find what's still a node;
        this fills in the rest by walking `visitor.node.scope.entries`
        directly and skipping anything already captured structurally
        (`handled_names`) or not actually declared in this scope
        (`entry.scope is not scope` -- an imported/foreign name, already
        handled separately by `ImportVisitor`).
        """
        scope = getattr(visitor.node, "scope", None)
        if scope is None:
            return [], [], [], []

        extra_assignments: list[PyiAssignment] = []
        extra_enums: list[PyiEnum | PyiAssignment] = []
        extra_structs: list[PyiClass] = []
        extra_functions: list[PyiFunction] = []

        # Bare annotated attributes (`x: int`, no `= ...`) at module or
        # plain-class scope are folded by `AnalyseDeclarationsTransform`
        # into a single synthetic `__annotations__ = {...}` dict assignment
        # -- `ScopeVisitor.visit_SingleAssignmentNode` drops that synthetic
        # node on sight, so the individually-typed attributes it was built
        # from are recovered here instead, from the pre-pipeline snapshot
        # `capture_static_types` stashed on the enclosing module/class node.
        for name, type_str in (
            getattr(visitor.node, "_stubgen_static_annotations", ()) or ()
        ):
            if name in handled_names:
                continue
            type_str = _substitute_ctypedef_aliases(type_str, ctypedef_aliases or {})
            extra_assignments.append(PyiAssignment(f"{name}: {type_str}", name=name))
            handled_names.add(name)

        for name, entry in scope.entries.items():
            if name in handled_names or (name.startswith("__") and name.endswith("__")):
                continue
            if name.startswith("__pyx_"):
                # Compiler-synthesized (e.g. `__pyx_opt_args_...` structs
                # bundling optional C arguments) -- never user-declared.
                # See `analysis/visitor.py::visit_CStructOrUnionDefNode`.
                continue
            if entry.scope is not scope:
                continue

            t = entry.type
            # `is_enum` alone misses a C++11 scoped `enum class`
            # (`PyrexTypes.CppScopedEnumType`, which sets `is_cpp_enum`
            # instead) -- confirmed directly: `entry.type.is_enum` is
            # `False` for one even though `.values`/`.create_wrapper`
            # (both shared via `EnumMixin`) work identically to a plain
            # enum, so it was falling through this branch entirely and
            # being silently dropped from the stub.
            if entry.is_type and (
                getattr(t, "is_enum", False) or getattr(t, "is_cpp_enum", False)
            ):
                if entry.create_wrapper:
                    extra_enums.append(PyiEnum(enum_name=name, names=list(t.values)))
                else:
                    extra_assignments.append(
                        PyiAssignment(
                            f"{name}: typing_extensions.TypeAlias = int", name=name
                        )
                    )
                continue
            if entry.is_type and getattr(t, "is_struct_or_union", False):
                extra_structs.append(convert_struct_or_union_type(t))
                continue
            if entry.is_type:
                # Fused types without a surviving node are emitted via the
                # existing `fused_types`/`convert_fused_type` machinery
                # elsewhere in `convert_scope`; other bare type entries
                # (e.g. a `ctypedef` with no further use) are skipped
                # rather than guessed at.
                continue
            if entry.is_cfunction:
                # A `cpdef` function declared with no body of its own --
                # directly inside `cdef extern from ...:`, the common
                # case -- has no surviving `CFuncDefNode`/`DefNode` to
                # convert structurally (there's nothing to keep a node
                # *for*: no body, no decorators, nothing but the
                # signature, which the pipeline resolves straight onto
                # the entry's own `CFuncType`). `create_wrapper` is the
                # same signal used for a struct/enum/variable above:
                # `False` for a plain, non-`cpdef` extern `cdef`
                # declaration, which has no Python wrapper and is
                # correctly skipped, same as any other C-only name.
                if entry.create_wrapper:
                    extra_functions.append(_convert_declared_function(name, t))
                continue
            if entry.is_variable:
                # Matches `visit_CVarDefNode`'s own filter (see
                # `analysis/visitor.py`): a bare variable is only
                # Python-visible -- and thus only worth emitting -- when
                # it's an in-class `public`/`readonly` attribute. At
                # module scope, a "private"/no-visibility `cdef` variable
                # is a pure C global with no Python binding at all (not
                # importable), so it's skipped here exactly as it always
                # was when it still had a `CVarDefNode` to filter.
                if not (
                    visitor.in_class and entry.visibility in ("public", "readonly")
                ):
                    continue
                type_name = render_pyrex_type(t)
                resolved = with_debug_fallback(
                    type_name,
                    "_typeshed.Incomplete",
                    lambda name_=name: f"Unable to determine type for {name_}",
                )
                resolved = _substitute_ctypedef_aliases(
                    resolved, ctypedef_aliases or {}
                )
                extra_assignments.append(
                    PyiAssignment(f"{name}: {resolved}", name=name)
                )

        return extra_assignments, extra_enums, extra_structs, extra_functions

    def _convert_cdef_assignments(
        self,
        visitor: ScopeVisitor,
        ctypedef_aliases: dict[str, str],
    ) -> tuple[list[PyiAssignment], set[str]]:
        assignments: list[PyiAssignment] = []
        handled_names: set[str] = set()
        static_property_types = (
            getattr(visitor.node, "_stubgen_static_property_types", None) or {}
        )
        for cdef_variable in visitor.cdef_variables:
            for name, base_type in get_cdef_variables(cdef_variable):
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
                assignments.append(PyiAssignment(f"{name}: {resolved_type}", name=name))
                handled_names.add(name)
        return assignments, handled_names

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

        contains_init = any(py_func.name == "__init__" for _, py_func in py_funcs)
        for idx, (_, py_func) in enumerate(py_funcs):
            if py_func.name == "__cinit__" and not contains_init:
                py_func.name = "__init__"
            elif py_func.name == "__cinit__" and contains_init:
                del py_funcs[idx]
                break

        functions = [
            function
            for _, function in sorted(cdef_funcs + py_funcs, key=lambda item: item[0])
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

        cdef_assignments, cdef_handled_names = self._convert_cdef_assignments(
            visitor, ctypedef_aliases
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
                convert_assignment(assignment, source_code, in_class=visitor.in_class),
            )
            for assignment in visitor.assignments
        ]

        cpp_classes = (convert_cpp_class(node) for node in visitor.cpp_classes)

        handled_names = self._collect_scope_handled_names(
            visitor, functions, cdef_handled_names
        )

        extra_assignments, extra_enums, extra_structs, extra_functions = (
            self._convert_declared_entries(visitor, handled_names, ctypedef_aliases)
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

        return PyiClass(
            name=name,
            doc=doc if include_docstrings else None,
            bases=get_bases(class_visitor.node),
            metaclass=get_metaclass(class_visitor.node),
            decorators=get_decorators(source_code, class_visitor.node),
            scope=self.convert_scope(
                class_visitor.scope,
                source_code,
                comments,
                include_docstrings,
                inherited_fused_types,
                resolve_ctypedef_aliases=resolve_ctypedef_aliases,
                inherited_ctypedef_aliases=inherited_ctypedef_aliases,
                defer_ctypedef_pruning=defer_ctypedef_pruning,
                prunable_ctypedef_aliases=prunable_ctypedef_aliases,
            ),
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


def _is_cxx_cimport(raw: str) -> bool:
    return bool(_CXX_FROM_CIMPORT_RE.search(raw) or _CXX_CIMPORT_RE.search(raw))


def _is_cython_import(raw: str) -> bool:
    return bool(_CYTHON_FROM_IMPORT_RE.search(raw) or _CYTHON_IMPORT_RE.search(raw))
