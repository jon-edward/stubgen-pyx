"""Render resolved ``PyrexTypes.Type``/``CFuncType`` objects as Python annotation strings."""

from __future__ import annotations

from Cython.Compiler import PyrexTypes as _PyrexTypes

from ..logging_utils import with_debug_fallback

CYTHON_TO_NUMPY_SCALAR: dict[str, str] = {
    "bint": "bool_",
    "bool": "bool_",
    "char": "byte",
    "signed char": "int8",
    "short": "short",
    "short int": "short",
    "int": "intc",
    "long": "int_",
    "long int": "int_",
    "long long": "longlong",
    "long long int": "longlong",
    "unsigned char": "ubyte",
    "unsigned short": "ushort",
    "unsigned short int": "ushort",
    "unsigned int": "uintc",
    "unsigned long": "uint",
    "unsigned long int": "uint",
    "unsigned long long": "ulonglong",
    "unsigned long long int": "ulonglong",
    "int8_t": "int8",
    "int16_t": "int16",
    "int32_t": "int32",
    "int64_t": "int64",
    "uint8_t": "uint8",
    "uint16_t": "uint16",
    "uint32_t": "uint32",
    "uint64_t": "uint64",
    "Py_ssize_t": "intp",
    "size_t": "uintp",
    "Py_intptr_t": "intp",
    "float": "single",
    "double": "double",
    "long double": "longdouble",
    "float complex": "complex64",
    "double complex": "complex128",
}

_CYTHON_BUILTIN_GENERIC_MAPPING: dict[str, str] = {
    "tuple": "tuple[typing.Any, ...]",
    "list": "list[typing.Any]",
    "dict": "dict[typing.Any, typing.Any]",
    "set": "set[typing.Any]",
}


def parameterize_builtin_generic(name: str | None) -> str | None:
    """Map bare Cython container names to Any-filled Python generics."""
    if name is None:
        return None
    return _CYTHON_BUILTIN_GENERIC_MAPPING.get(name, name)


def render_pyrex_type(
    t: _PyrexTypes.PyrexType | None, *, _depth: int = 0
) -> str | None:
    """Render a resolved ``PyrexTypes.Type`` as a Python annotation string."""
    if t is None:
        return None
    if getattr(t, "is_cv_qualified", False):
        return render_pyrex_type(t.cv_base_type, _depth=_depth)
    return _render_unqualified_pyrex_type(t, _depth=_depth)


def _render_unqualified_pyrex_type(
    t: _PyrexTypes.PyrexType, *, _depth: int
) -> str | None:
    if t.is_void:
        return "None"
    renderers = (
        _render_pointer_type,
        _render_array_type,
        _render_ctuple_type,
        _render_memoryview_type,
        _render_cpp_template_type,
        _render_named_type,
        _render_cfunction_type,
        _render_builtin_type,
    )
    for renderer in renderers:
        rendered = renderer(t, _depth=_depth)
        if rendered is not None:
            return rendered
    return None


def _render_pointer_type(t: _PyrexTypes.PyrexType, *, _depth: int) -> str | None:
    if not t.is_ptr:
        return None
    base = t.base_type
    if base is _PyrexTypes.c_char_type:
        return "bytes"
    if base.is_void:
        return "typing.Any"
    if getattr(base, "is_cfunction", False):
        return _render_cfunction_type(base, _depth=_depth)
    return render_pyrex_type(base, _depth=_depth + 1)


def _render_array_type(t: _PyrexTypes.PyrexType, *, _depth: int) -> str | None:
    if not t.is_array:
        return None
    if t.base_type is _PyrexTypes.c_char_type:
        return "bytes"
    inner = render_pyrex_type(t.base_type, _depth=_depth + 1)
    return f"list[{inner}]" if inner is not None else None


def _render_ctuple_type(t: _PyrexTypes.PyrexType, *, _depth: int) -> str | None:
    if not getattr(t, "is_ctuple", False):
        return None
    parts = [
        with_debug_fallback(
            render_pyrex_type(component, _depth=_depth + 1),
            "object",
            lambda component_idx=component_idx: (
                f"Replaced tuple component at index {component_idx} with 'object'"
            ),
        )
        for component_idx, component in enumerate(t.components)
    ]
    return f"tuple[{', '.join(parts)}]"


def _render_memoryview_type(t: _PyrexTypes.PyrexType, *, _depth: int) -> str | None:
    if not getattr(t, "is_memoryviewslice", False):
        return None
    dtype_name = str(t.dtype) if t.dtype is not None else None
    scalar = None if dtype_name is None else CYTHON_TO_NUMPY_SCALAR.get(dtype_name)
    return f"numpy.typing.NDArray[numpy.{scalar}]" if scalar else "memoryview"


def _render_cpp_template_type(t: _PyrexTypes.PyrexType, *, _depth: int) -> str | None:
    if not getattr(t, "is_cpp_class", False) or not getattr(t, "templates", None):
        return None
    base = t.name
    parts = [
        with_debug_fallback(
            render_pyrex_type(argument, _depth=_depth + 1),
            "_typeshed.Incomplete",
            lambda argument_idx=argument_idx: (
                f"Replaced template argument of {base} at index {argument_idx} with '_typeshed.Incomplete'"
            ),
        )
        for argument_idx, argument in enumerate(t.templates)
    ]
    return f"{base}[{', '.join(parts)}]"


def _render_named_type(t: _PyrexTypes.PyrexType, *, _depth: int) -> str | None:
    name = getattr(t, "name", None)
    if name is None:
        return None
    type_flags = (
        "is_struct_or_union",
        "is_enum",
        "is_cpp_enum",
        "is_extension_type",
        "is_cpp_class",
        "is_fused",
    )
    return name if any(getattr(t, flag, False) for flag in type_flags) else None


def _render_builtin_type(t: _PyrexTypes.PyrexType, *, _depth: int) -> str | None:
    if not (t.is_pyobject or t.is_numeric or t.is_string):
        return None
    return parameterize_builtin_generic(t.py_type_name())


def _render_cfunction_type(t: _PyrexTypes.CFuncType, *, _depth: int) -> str | None:
    """Render a resolved ``CFuncType`` (a function pointer's pointee, typically) as ``Callable[[...], ...]``."""
    if not getattr(t, "is_cfunction", False):
        return None
    args = [
        with_debug_fallback(
            render_pyrex_type(arg.type, _depth=_depth + 1),
            "_typeshed.Incomplete",
            lambda arg_idx_=arg_idx: (
                f"Replaced argument {arg_idx_} type with '_typeshed.Incomplete'"
            ),
        )
        for arg_idx, arg in enumerate(t.args)
    ]
    return_type = with_debug_fallback(
        render_pyrex_type(t.return_type, _depth=_depth + 1),
        "_typeshed.Incomplete",
        lambda: "Replaced return type with '_typeshed.Incomplete'",
    )
    return f"typing.Callable[[{', '.join(args)}], {return_type}]"
