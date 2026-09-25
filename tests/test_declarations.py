"""Direct unit tests for `conversion/declarations.py`'s node-based
`convert_enum`. Real Cython source can't reach this function for a
plain module-level enum at all -- once real declaration analysis runs,
a bare enum has no surviving node, so it's converted via the
Entry/Type-based fallback in `Converter._convert_declared_entries`
instead (see that function's docstring). An enum nested inside a `cdef
class` body *does* keep a structural node, but Cython's own compiler
crashes on that shape (a real, upstream bug, not stubgen-pyx's), so
it's not reachable through the real pipeline either. `convert_enum`
itself only needs a couple of attributes (`.create_wrapper`, `.name`,
`.items`), so it's tested directly here against a minimal stand-in
rather than trying to force an unstable source shape through the full
pipeline.
"""

from __future__ import annotations

from types import SimpleNamespace

from stubgen_pyx.conversion.declarations import convert_enum
from stubgen_pyx.models.pyi_elements import PyiAssignment, PyiEnum


def _fake_enum_node(*, create_wrapper: bool, name: str, member_names: list[str]):
    return SimpleNamespace(
        create_wrapper=create_wrapper,
        name=name,
        items=[SimpleNamespace(name=member_name) for member_name in member_names],
    )


def test_convert_enum_with_wrapper_returns_pyi_enum():
    """`cpdef enum`/`cpdef enum class` -- Python-visible -- becomes a real `PyiEnum`."""
    node = _fake_enum_node(create_wrapper=True, name="Color", member_names=["RED", "GREEN"])
    result = convert_enum(node)
    assert result == PyiEnum(enum_name="Color", names=["RED", "GREEN"])


def test_convert_enum_without_wrapper_returns_int_alias():
    """A plain `cdef enum` -- not Python-visible as a real enum type --
    becomes a bare `int` `TypeAlias` fallback instead."""
    node = _fake_enum_node(create_wrapper=False, name="Color", member_names=[])
    result = convert_enum(node)
    assert isinstance(result, PyiAssignment)
    assert result.statement == "Color: typing_extensions.TypeAlias = int"
    assert result.name == "Color"
