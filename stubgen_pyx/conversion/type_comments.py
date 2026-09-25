"""PEP 484 comment-based type syntax (`# type: EXPR`): finding a
function's whole-signature or per-argument type comment(s) and applying
them to a signature wherever a real annotation is missing.
"""

from __future__ import annotations

import re

from Cython.Compiler import Nodes

from ..models.pyi_elements import PyiSignature
from ..parsing.comments import Comment, CommentIndex
from .type_parsing import parameterize_builtin_generic

#: A `# type: EXPR` comment (PEP 484's comment-based type syntax). Matches
#: both the whole-signature form (`# type: (int, str) -> bool`) and a
#: per-argument form (`# type: int`) -- distinguishing between them is
#: `_SIGNATURE_TYPE_COMMENT_PATTERN`'s job, applied to the captured EXPR.
#: Deliberately doesn't try to exclude `# type: ignore[...]` (mypy's
#: unrelated ignore-comment convention, not a type at all) via a
#: negative lookahead here -- `\s*(?!ignore\b)` doesn't work: with
#: exactly one space before "ignore", the engine can backtrack `\s*`
#: down to matching zero spaces, landing the lookahead on the space
#: itself instead of "ignore" (where it vacuously succeeds), then
#: capture the space plus "ignore[...]" as EXPR anyway. See
#: `_type_comment_expr`, which checks for "ignore" as a separate,
#: non-backtracking step instead.
_TYPE_COMMENT_PATTERN = re.compile(r"^#\s*type:\s*(.+?)\s*$")

#: A `# type: ignore` comment isn't a type at all -- mypy's unrelated
#: "suppress errors on this line" convention. `\b` so `ignoreme` (an
#: unlikely but valid type name) isn't caught by this too.
_TYPE_IGNORE_PATTERN = re.compile(r"^ignore\b")

#: The whole-signature form specifically: `(ARGS) -> RETURN`, where ARGS
#: may be `...` (PEP 484: "type comment ... consist[ing] of just the
#: return type", args deliberately left unspecified).
_SIGNATURE_TYPE_COMMENT_PATTERN = re.compile(r"^\((?P<args>.*)\)\s*->\s*(?P<ret>.+)$")


def _type_comment_expr(text: str) -> str | None:
    """Extract the EXPR from a `# type: EXPR` comment's raw `text`.

    `None` for anything that isn't this shape at all, *or* is but is
    actually mypy's unrelated `# type: ignore[...]` convention -- every
    caller that needs to know "is this a type comment I can use" should
    go through this rather than `_TYPE_COMMENT_PATTERN` directly (see
    that pattern's docstring for why the ignore-exclusion can't just
    live inside the same regex).
    """
    match = _TYPE_COMMENT_PATTERN.match(text)
    if not match:
        return None
    expr = match.group(1)
    if _TYPE_IGNORE_PATTERN.match(expr):
        return None
    return expr


def _split_top_level_commas(text: str) -> list[str]:
    """Split `text` on commas not nested inside `()`/`[]`/`{}`.

    Needed because a single argument's type can itself contain commas
    (`Dict[str, int]`, `Tuple[int, ...]`, `Union[int, str]`) that must
    not be mistaken for argument separators in a whole-signature type
    comment's `(T1, T2, ...)` portion.
    """
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    for ch in text:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return parts


def _parse_signature_type_comment(expr: str) -> tuple[list[str] | None, str] | None:
    """Parse a whole-signature `# type: (ARGS) -> RETURN` comment's EXPR.

    Returns `(arg_types, return_type)` -- `arg_types` is `None` when the
    args portion is the literal `...` (explicitly "unspecified", per
    PEP 484), or a list split on top-level commas otherwise (empty list
    for a genuinely no-argument function, `# type: () -> None`).
    Returns `None` if `expr` isn't this shape at all (e.g. a bare
    per-argument type, or something unparseable) -- the caller decides
    what to fall back to.
    """
    match = _SIGNATURE_TYPE_COMMENT_PATTERN.match(expr.strip())
    if not match:
        return None
    args_part = match.group("args").strip()
    return_part = match.group("ret").strip()
    if args_part == "...":
        return (None, return_part)
    return (_split_top_level_commas(args_part), return_part)


def _align_signature_arg_types(arg_types: list[str], raw_args: list) -> dict[int, str]:
    """Map raw-argument-index -> type string, from a whole-signature
    comment's parsed argument-type list.

    A type comment's argument list conventionally omits `self`/`cls`
    (PEP 484: "the first argument type ... in a comment for a bound
    method should be omitted"), so a count short by exactly one against
    a method whose first argument actually is `self`/`cls` is aligned
    from the second argument on. Any other count mismatch is ambiguous
    -- deliberately not guessed at; returns `{}` rather than risk
    mismatching arguments to the wrong types.
    """
    n = len(raw_args)
    if len(arg_types) == n:
        return dict(enumerate(arg_types))
    if (
        len(arg_types) == n - 1
        and raw_args
        and getattr(raw_args[0], "is_self_arg", False)
    ):
        return {i + 1: t for i, t in enumerate(arg_types)}
    return {}


def _try_parse_signature_type_comment(text: str) -> tuple[list[str] | None, str] | None:
    """Parse a comment's raw `text` as a whole-signature `# type: (ARGS) ->
    RETURN` comment.

    `None` for anything that isn't a `# type: ...` comment at all, is
    mypy's unrelated `# type: ignore[...]`, or is one but not the
    whole-signature shape (a bare per-argument type, or genuinely
    unparseable) -- none of these are distinguished here (all mean "not
    this"); see `_find_signature_type_comment` and
    `apply_type_comments` for how each is actually handled by its
    caller.
    """
    expr = _type_comment_expr(text)
    if expr is None:
        return None
    return _parse_signature_type_comment(expr)


def _find_signature_type_comment(
    node: Nodes.Node, comments: CommentIndex
) -> tuple[str, Comment] | None:
    """Find the `# type: ...` comment associated with a function's own
    signature (as opposed to one of its individual arguments).

    Checked in the two places PEP 484 allows this comment:

    1. Trailing the `def`/`cpdef` line itself (single-line signature):
       ``def f(a, b):  # type: (int, str) -> bool``. *Peeked* first
       (`CommentIndex.peek_trailing`, not `.trailing`) and only
       actually claimed if it's an ignore comment or genuinely parses
       as the whole-signature `(ARGS) -> RETURN` shape: this exact line
       can, instead, be the *first* line of a multi-line signature
       whose last argument on that line has its own, unrelated trailing
       comment (``def f(a, b,  # type: str`` continuing on a following
       line) -- claiming any trailing comment on the def line
       unconditionally breaks exactly this case, consuming the
       per-argument comment for `b` before
       `_find_per_argument_type_comments` ever gets a chance at it, and
       then discarding it entirely (it doesn't parse as a whole
       signature either). Peeking first leaves it available for that,
       if this line turns out not to be the whole-signature case.
    2. Standalone, as the first line of the function body (multi-line
       signature, or simply preferred style) -- checked *even when a
       docstring follows*, since PEP 484 requires the type comment to
       come first: ``node.body.pos`` points at the body's actual start
       (right after this comment, whether or not a docstring is there
       to skip over it), which differs from `node.body.stats[0].pos`
       specifically when a docstring is present (the docstring is
       extracted to `node.doc` and removed from `.stats` by the time
       this runs, but `.body.pos` isn't affected by that extraction).
       Using `.body.pos` instead of the first real statement's own
       position is what makes this work in both cases without
       special-casing the docstring-or-not split. No collision risk
       here the way there is for (1): a line here is never also an
       argument's own line.

    Returns `(kind, comment)`, `kind` one of:

    - ``"ignore"``: mypy's unrelated `# type: ignore[...]` convention --
      not a type at all, but still meaningful and shown verbatim in the
      output, exactly as before per-argument/whole-signature parsing
      existed for this project.
    - ``"signature"``: parses as `(ARGS) -> RETURN`; `apply_type_comments`
      applies it structurally.
    - ``"unknown"``: matches `# type: ...` but is neither of the above
      (a bare per-argument-looking type, or something genuinely
      unparseable) -- shown verbatim as a fallback, the same spirit as
      ``"ignore"``.

    `None` if neither place has anything comment-shaped at all.
    """
    candidate = comments.peek_trailing(node.pos[1])
    if candidate is not None:
        expr = _type_comment_expr(candidate.text)
        is_type_comment = _TYPE_COMMENT_PATTERN.match(candidate.text) is not None
        if (
            expr is not None
            and _try_parse_signature_type_comment(candidate.text) is not None
        ):
            return ("signature", comments.trailing(node.pos[1]))
        if is_type_comment and expr is None:
            # Matched `# type: ...` but `_type_comment_expr` rejected it
            # -- the ignore case specifically (nothing else is rejected
            # at this stage).
            return ("ignore", comments.trailing(node.pos[1]))

    body = getattr(node, "body", None)
    body_pos = getattr(body, "pos", None)
    if body_pos is None:
        return None
    leading = comments.leading_block(body_pos[1])
    if not leading:
        return None
    # The comment immediately above the body -- last in the block --
    # is the one PEP 484 means; anything further up is unrelated.
    closest = leading[-1]
    if not _TYPE_COMMENT_PATTERN.match(closest.text):
        return None
    if _type_comment_expr(closest.text) is None:
        return ("ignore", closest)
    if _try_parse_signature_type_comment(closest.text) is not None:
        return ("signature", closest)
    return ("unknown", closest)


def _find_per_argument_type_comments(
    raw_args: list, comments: CommentIndex
) -> dict[int, str]:
    """Map raw-argument-index -> type string, from each argument's own
    trailing `# type: TYPE` comment.

    Only the *last* argument on a given source line can unambiguously
    claim a trailing comment on that line -- PEP 484's rule for
    comment-based per-argument types, since the comment attaches to
    whichever argument immediately precedes it. Two arguments sharing
    a line (`def f(a, b,  # type: int`) is genuinely ambiguous for the
    earlier one(s); only `b` gets it here.

    A line's trailing comment that turns out to be whole-signature
    -shaped (`(...) -> ...`) is skipped -- this is how a single-line
    function's own `# type: (int, str) -> bool` (sitting on the same
    line as its last argument) is kept from being misread as that
    argument's own, individual type. Ordering matters for this to work
    smoothly in practice: `apply_type_comments` calls
    `_find_signature_type_comment` first, which -- specifically because
    it peeks rather than claims until it's sure (see that function's
    docstring) -- only actually claims the def line's trailing comment
    via `CommentIndex.trailing` when it turns out to be the
    whole-signature comment; by the time this runs, `.trailing()` for
    that line then correctly returns `None` in that case. The shape
    check here is a belt-and-suspenders fallback for anything that
    slips through some other way, not the only thing preventing the
    collision.
    """
    indices_by_line: dict[int, list[int]] = {}
    for i, arg in enumerate(raw_args):
        indices_by_line.setdefault(arg.pos[1], []).append(i)

    result: dict[int, str] = {}
    for line, indices in indices_by_line.items():
        comment = comments.trailing(line)
        if comment is None:
            continue
        expr = _type_comment_expr(comment.text)
        if expr is None:
            continue
        if _SIGNATURE_TYPE_COMMENT_PATTERN.match(expr.strip()):
            continue
        result[indices[-1]] = expr
    return result


def apply_type_comments(
    signature: PyiSignature,
    node: Nodes.Node,
    raw_args: list,
    comments: CommentIndex,
) -> str | None:
    """Fill in argument/return annotations `signature` is missing,
    from any `# type: ...` comment(s) associated with `node`.

    Only ever fills a gap -- an argument or return type Cython/an
    explicit annotation already resolved is left alone; a type
    comment is a fallback source of typing, never an override.
    Checked in two independent ways, both of which can contribute
    to the same signature: a whole-signature comment
    (`_find_signature_type_comment`) for the function as a whole,
    and each individual argument's own trailing comment
    (`_find_per_argument_type_comments`) -- the latter is more
    locally specific and takes precedence for any argument both
    apply to.

    Returns the associated comment's raw text, verbatim, when it's
    either mypy's `# type: ignore[...]` (not a type at all, but
    still meaningful and shown as before this function existed) or
    didn't parse as the `(ARGS) -> RETURN` shape
    (`_parse_signature_type_comment` returned `None`) -- callers use
    this to fall back to displaying it (matching this project's
    general graceful-degradation approach: something that couldn't
    be understood still shouldn't just vanish). `None` otherwise,
    including when a signature comment was found and fully applied
    -- once its information is real annotations on the signature,
    redundantly showing the original text next to them too would
    just be noise.
    """
    raw_fallback: str | None = None

    found = _find_signature_type_comment(node, comments)
    arg_type_map: dict[int, str] = {}
    if found is not None:
        kind, signature_comment = found
        if kind == "signature":
            expr = _type_comment_expr(signature_comment.text) or ""
            arg_types, return_type = _parse_signature_type_comment(expr)  # type: ignore[misc]
            if arg_types is not None:
                arg_type_map = _align_signature_arg_types(arg_types, raw_args)
            if signature.return_type is None:
                signature.return_type = parameterize_builtin_generic(return_type)
        else:  # "ignore" or "unknown"
            raw_fallback = signature_comment.text

    arg_type_map.update(_find_per_argument_type_comments(raw_args, comments))

    for i, pyi_arg in enumerate(signature.args):
        if pyi_arg.annotation is not None:
            continue
        type_str = arg_type_map.get(i)
        if type_str:
            pyi_arg.annotation = parameterize_builtin_generic(type_str)

    return raw_fallback
