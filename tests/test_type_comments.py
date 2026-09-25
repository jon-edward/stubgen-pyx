"""Direct unit tests for `conversion/type_comments.py`'s smaller helper
functions -- the specific "not a type comment at all" / "ambiguous
count" / "wrong shape" edge cases that a full end-to-end conversion
rarely exercises on its own.
"""

from __future__ import annotations

from types import SimpleNamespace

from stubgen_pyx.conversion.type_comments import (
    _align_signature_arg_types,
    _find_per_argument_type_comments,
    _find_signature_type_comment,
    _split_top_level_commas,
    _try_parse_signature_type_comment,
    _type_comment_expr,
)
from stubgen_pyx.parsing.comments import Comment, CommentIndex


def test_type_comment_expr_none_for_non_type_comment():
    """A plain, ordinary comment isn't `# type: ...` shaped at all."""
    assert _type_comment_expr("# just a regular comment") is None


def test_split_top_level_commas_respects_nested_brackets():
    """A comma inside `Dict[str, int]` must not be treated as a
    top-level argument separator."""
    assert _split_top_level_commas("Dict[str, int], List[int]") == [
        "Dict[str, int]",
        "List[int]",
    ]


def test_align_signature_arg_types_empty_on_genuine_mismatch():
    """A count that's neither equal nor exactly one short (the
    `self`/`cls`-omitted case) is genuinely ambiguous and must not be
    guessed at."""

    class _FakeArg:
        is_self_arg = False

    raw_args = [_FakeArg(), _FakeArg(), _FakeArg()]
    assert _align_signature_arg_types(["int", "str"], raw_args) == {}


def test_try_parse_signature_type_comment_none_for_non_type_comment():
    assert _try_parse_signature_type_comment("# just a regular comment") is None


def test_find_signature_type_comment_none_when_node_has_no_body():
    """A node with no `.body` at all (nothing to look for a leading
    comment above) yields no signature comment."""
    node = SimpleNamespace(pos=(None, 1, 0))
    assert _find_signature_type_comment(node, CommentIndex([])) is None


def test_find_signature_type_comment_none_when_leading_comment_isnt_type_shaped():
    """A leading comment block exists immediately above the body, but
    isn't `# type: ...` shaped at all -- not every comment there means
    something."""
    comments = CommentIndex([Comment(text="# just a note", line=2, col=4, standalone=True)])
    node = SimpleNamespace(pos=(None, 1, 0), body=SimpleNamespace(pos=(None, 3, 4)))
    assert _find_signature_type_comment(node, comments) is None


def test_find_signature_type_comment_ignore_via_leading_block():
    """`# type: ignore[...]` as the standalone leading comment (not a
    trailing one) is still recognized as the "ignore" case."""
    comments = CommentIndex(
        [Comment(text="# type: ignore[no-untyped-def]", line=2, col=4, standalone=True)]
    )
    node = SimpleNamespace(pos=(None, 1, 0), body=SimpleNamespace(pos=(None, 3, 4)))
    kind, comment = _find_signature_type_comment(node, comments)
    assert kind == "ignore"
    assert comment.text == "# type: ignore[no-untyped-def]"


def test_find_per_argument_type_comments_skips_ignore_trailing_comment():
    """A trailing `# type: ignore` on an argument's own line isn't a
    real per-argument type and must be skipped, not misread as one."""
    arg = SimpleNamespace(pos=(None, 5, 4))
    comments = CommentIndex([Comment(text="# type: ignore", line=5, col=10, standalone=False)])
    assert _find_per_argument_type_comments([arg], comments) == {}


def test_find_per_argument_type_comments_skips_whole_signature_shaped_trailing_comment():
    """A trailing comment that happens to parse as `(ARGS) -> RETURN`
    belongs to the whole-signature case, handled elsewhere -- it must
    not also be claimed here as this one argument's own type."""
    arg = SimpleNamespace(pos=(None, 6, 4))
    comments = CommentIndex(
        [Comment(text="# type: (int) -> str", line=6, col=10, standalone=False)]
    )
    assert _find_per_argument_type_comments([arg], comments) == {}
