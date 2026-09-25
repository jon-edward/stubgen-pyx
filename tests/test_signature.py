"""Direct unit tests for `conversion/signature.py`'s smaller helpers:
`_decode_or_pass`'s type validation, and the annotation-extraction
fallback chain's edge cases.
"""

from __future__ import annotations

import sys

import pytest
from Cython.Compiler import Nodes

from stubgen_pyx.conversion.signature import (
    _decode_or_pass,
    _get_annotation,
    _get_return_type_annotation,
    get_signature,
)

sys.path.insert(0, "tests")
from test_type_parsing import _first_node


def test_decode_or_pass_decodes_bytes():
    assert _decode_or_pass(b"hello") == "hello"


def test_decode_or_pass_passes_through_str():
    assert _decode_or_pass("hello") == "hello"


def test_decode_or_pass_raises_for_anything_else():
    with pytest.raises(TypeError, match="Expected str or bytes"):
        _decode_or_pass(123)  # type: ignore[arg-type]


def test_get_annotation_tolerates_attribute_error():
    """A node whose `.annotation` access itself raises `AttributeError`
    (not just a missing attribute) must not crash the whole extraction,
    same defensive style as everywhere else this attribute chain gets
    walked."""

    class _RaisesOnAnnotation:
        @property
        def annotation(self):
            raise AttributeError("simulated")

    assert _get_annotation(_RaisesOnAnnotation()) is None


def test_get_return_type_annotation_none_when_nothing_resolves():
    """A plain `cdef` function with no return annotation and no
    resolved C function type at all (the raw, pre-pipeline shape) has
    nothing to fall back to."""
    node = _first_node("cdef foo():\n    pass\n", Nodes.CFuncDefNode)
    assert node.return_type_annotation is None
    assert _get_return_type_annotation(node) is None


def test_get_signature_resolves_bare_self_via_base_type_fallback():
    """A bare, unannotated `self` parses with an empty declarator name
    and the identifier itself on `base_type.name` instead -- the same
    ambiguity `type_parsing.capture_static_types` documents for the
    entries-based path; `get_signature`'s own structural extraction
    has to recover the same way."""
    node = _first_node("def bar(self):\n    pass\n", Nodes.DefNode)
    signature = get_signature(node)
    assert signature.args[0].name == "self"


def test_to_argument_recovers_name_from_base_type_when_declarator_has_none():
    """The specific shape `_declarator_name` alone can't recover from:
    `is_self_arg` is set (a real C-level `self`), but the declarator's
    own name is genuinely empty -- `arg.base_type.name` is the only
    place the identifier survives. Constructed directly rather than
    from real source, since a real `cdef class` method's `self` arg
    already has a real declarator name by the time it's reachable here
    (`_declarator_name` resolves it first); this exercises the
    fallback's own logic regardless."""
    from types import SimpleNamespace

    from stubgen_pyx.conversion.signature import _to_argument

    fake_arg = SimpleNamespace(
        declarator=SimpleNamespace(),
        is_self_arg=True,
        base_type=SimpleNamespace(name="self"),
        annotation=None,
        default=None,
    )
    result = _to_argument(fake_arg)
    assert result.name == "self"
