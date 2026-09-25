"""Extracts function signatures from Cython AST nodes."""

from __future__ import annotations

from Cython.Compiler import Nodes

from ..models.pyi_elements import PyiArgument, PyiSignature
from .type_parsing import (
    _declarator_name,
    extract_type_from_base_type,
    parameterize_builtin_generic,
    render_pyrex_type,
)
from .unparse import unparse_expr


def get_signature(node: Nodes.CFuncDefNode | Nodes.DefNode) -> PyiSignature:
    """Extract a PyiSignature from a Cython function node."""
    if isinstance(node, Nodes.CFuncDefNode):
        return _get_signature_cfunc(node)
    return _get_signature_def(node)


def _get_signature_def(node: Nodes.DefNode) -> PyiSignature:
    """Extract signature from a Python (def) function node."""
    pyi_args = _get_args(node.args)  # type: ignore

    var_arg = _create_argument_if_exists(node.star_arg)
    kw_arg = _create_argument_if_exists(node.starstar_arg)
    return_type = _get_return_type_annotation(node)

    return PyiSignature(
        pyi_args,
        var_arg=var_arg,
        kw_arg=kw_arg,
        return_type=return_type,
        num_posonly_args=node.num_posonly_args,
        num_kwonly_args=node.num_kwonly_args,
    )


def _get_signature_cfunc(node: Nodes.CFuncDefNode) -> PyiSignature:
    """Extract signature from a C (cdef/cpdef) function node."""
    pyi_args = _get_args(node.declarator.args)  # type: ignore
    return_type = _get_return_type_annotation(node)
    return PyiSignature(pyi_args, return_type=return_type)


def _create_argument_if_exists(arg_node) -> PyiArgument | None:
    """Convert an argument node to PyiArgument if it exists."""
    if arg_node is None:
        return None
    return PyiArgument(arg_node.name, annotation=_get_annotation(arg_node))


def _decode_or_pass(value: str | bytes) -> str:
    """Ensure value is a string, decoding bytes if needed."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, str):
        return value
    raise TypeError(f"Expected str or bytes, got {type(value)}")


def _get_annotation(arg: Nodes.CArgDeclNode | Nodes.PyArgDeclNode) -> str | None:
    """Extract type annotation from a function argument node."""
    try:
        if arg.annotation is not None:
            return _decode_or_pass(arg.annotation.string.value)
        if not isinstance(arg, Nodes.CArgDeclNode):
            return None
        return extract_type_from_base_type(arg)
    except AttributeError:
        pass
    return None


def _get_return_type_annotation(node: Nodes.CFuncDefNode | Nodes.DefNode) -> str | None:
    """Extract return type annotation from a function node."""
    if node.return_type_annotation is not None:
        return parameterize_builtin_generic(
            _decode_or_pass(node.return_type_annotation.string.value)
        )
    if isinstance(node, Nodes.DefNode):
        return None

    try:
        result = extract_type_from_base_type(node)
        if result is not None:
            return result
    except AttributeError:
        pass

    resolved_type = getattr(node, "type", None)
    if resolved_type is not None and getattr(resolved_type, "is_cfunction", False):
        # `node.base_type` (what `extract_type_from_base_type` reads) is
        # gone post-analysis for a `CFuncDefNode` whose return type needed
        # real resolution (e.g. a C++ template or ctuple return type).
        # `node.type` at that point is the function's *whole*
        # `CFuncType`, not the return type alone -- has to be narrowed
        # to `.return_type` explicitly, or `render_pyrex_type` would
        # render the callable itself instead of what it returns.
        return render_pyrex_type(resolved_type.return_type)

    return None


def _to_argument(arg: Nodes.CArgDeclNode) -> PyiArgument:
    """Convert a CArgDeclNode to a PyiArgument."""
    declarator: Nodes.CDeclaratorNode | Nodes.CPtrDeclaratorNode = arg.declarator  # type: ignore
    # `_declarator_name` recursively unwraps every declarator layer
    # (pointer, const, func, array) down to the name -- a hardcoded
    # `declarator.base.name` only handles one `CPtrDeclaratorNode` layer
    # and breaks on a const-qualified pointer (`char* const x`), which
    # inserts an extra `CConstDeclaratorNode`. Falls back to `""`: a
    # bare, unannotated argument's declarator has no name yet at this
    # point (see the bare-identifier-arg handling below).
    name = _decode_or_pass(_declarator_name(declarator) or "")

    if (
        not name
        or getattr(arg, "_stubgen_bare_identifier_arg", False)
        or getattr(arg, "is_self_arg", False)
    ):
        # A bare, unannotated argument (`self`, `cls`, or a plain
        # untyped positional like `x` in `def f(x=None)`) parses with an
        # *empty* `declarator.name`, the identifier itself sitting on
        # `base_type.name` instead -- recovered as the real name here.
        # `_stubgen_bare_identifier_arg` (captured pre-pipeline, see
        # `type_parsing.capture_static_types`) keeps this branch
        # triggering even once later analysis backfills
        # `declarator.name` from that same identifier (self-arg
        # analysis, or general inference for anything else): without
        # it, `name` above looks like a real, explicit name post
        # -pipeline, and the leftover `base_type` -- still holding that
        # same bogus identifier -- would get extracted as a real type
        # annotation instead (e.g. `self: _typeshed.Incomplete`,
        # `x: x | None`).
        #
        # `is_self_arg` is checked unconditionally too, separately from
        # that ambiguity: a plain Python-style `self`/`cls` (as opposed
        # to a C-style/`cpdef` one) parses unambiguously, with a real
        # `declarator.name` from the very start -- `not name` and
        # `_stubgen_bare_identifier_arg` are both false for it -- yet
        # Cython's own analysis still resolves `self`'s *type* to the
        # enclosing class, same as any other attribute access, which is
        # a real fact but never something the source itself wrote.
        # Confirmed empirically: without this, plain `self` rendered as
        # `self: Ops` (the enclosing class) for any method with no
        # other reason to hit the branch above. An *explicit* annotation
        # (`self: Self`, `x: int = None`) is unaffected either way --
        # `arg.annotation` is set in that case, and `_get_annotation`
        # reads that directly rather than ever consulting `base_type`/
        # inferred type.
        if not name:
            name = arg.base_type.name  # type: ignore
        annotation = _get_annotation(arg) if arg.annotation is not None else None
    else:
        annotation = _get_annotation(arg)

    annotation = parameterize_builtin_generic(annotation)

    default = unparse_expr(arg.default)  # type: ignore
    if (
        default == "None"
        and annotation
        and "None" not in annotation
        and "Optional" not in annotation
    ):
        annotation += " | None"

    return PyiArgument(name, default=default, annotation=annotation)


def _get_args(args: list[Nodes.CArgDeclNode]) -> list[PyiArgument]:
    """Convert a list of CArgDeclNodes to PyiArguments."""
    return [_to_argument(arg) for arg in args]
