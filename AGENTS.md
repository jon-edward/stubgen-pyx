# AGENTS.md — stubgen-pyx

## Setup

```bash
python -m venv venv
venv/bin/pip install -e ".[test]"
venv/bin/python -m pytest -q
```

## Architecture rule

Once Cython's real declaration analysis runs, most declarations with no
runtime component (`cdef enum`, `cdef struct`/`union`, `ctypedef`,
`ctypedef fused`, uninitialized annotated attributes, `cdef public`/`readonly`
attributes, fused functions) are **removed from `body.stats` or replaced by
synthesized nodes**. Consequences:

- A structural, `body.stats`-walking visitor can't see these. Use the
  `scope.entries`-based fallback path instead (`Entry`/`Type` objects, which
  survive regardless of the tree). `render_pyrex_type` (Entry/Type path) and
  `extract_type_from_base_type` (structural path) must stay in sync — any
  case one handles (pointers, arrays, memoryviews, const/volatile, fused
  types, C++ templates), the other generally needs too.
- Some info (a fused type's original member spelling, a bare argument's real
  name) is gone by the time analysis finishes and never reaches
  `scope.entries` either. For this, capture it **once, immediately after
  parsing, before the pipeline runs** (`type_parsing.py::capture_static_types`)
  and stash it as a `_stubgen_*` attribute directly on the node. Downstream
  code reads `getattr(node, "_stubgen_...", None)` before falling back to
  whatever survives post-pipeline.

When a declaration is silently missing, mistyped, or falls back to
`_typeshed.Incomplete`: determine whether it's (a) gone from the tree
post-pipeline (needs an entries-based fallback) or (b) present but
ambiguous/wrong post-pipeline (needs a pre-pipeline capture). Fix the
category, not the symptom.

## Hard rules

- **Never hand-roll a single-level declarator unwrap.** Declarators compose
  (`CPtrDeclaratorNode -> CConstDeclaratorNode -> CNameDeclaratorNode` for
  `char* const x`, etc.). Code doing `declarator.base.name` assuming one
  layer will crash or silently misresolve the name the moment a second layer
  appears. Always use `type_parsing._declarator_name`, which unwraps every
  layer recursively. If you find a manual `.base.name`/`.base.base.name`
  anywhere else, replace it.
- **`const`/`volatile` have two representations; handle both.** AST level:
  the const/volatile base-type node (`CQualifierTypeNode` on Cython >=3.3,
  `CConstOrVolatileTypeNode` before that — see the version-compatibility
  rule below), unwrapped in `extract_type_from_base_type`. Entry/Type
  level: `t.is_cv_qualified` / `t.cv_base_type` (attribute names unchanged
  across versions), unwrapped in `render_pyrex_type`. Neither qualifier has
  Python-level meaning — always render the underlying type. Any new
  type-rendering path needs both cases.
- **A fused memoryview must not collapse to a bare scalar.** `numeric[:, :]
  arr` is an array, not `int | float`. The fused resolver
  (`converter.py::_resolve_fused_signature`/`_resolved_fused_annotation`)
  needs to know an occurrence came from a memoryview (via
  `_stubgen_fused_was_memoryview`/`_stubgen_fused_return_was_memoryview`) to
  render `NDArray[numpy.<scalar>]` instead. When the fused type is shared
  with a non-memoryview position (normally emitted as a scalar-bound
  `TypeVar`), fall back to plain `memoryview` rather than the misleading
  TypeVar reference.
- **"No diagnostics + doesn't crash" is not proof of correct input.** This
  project's pipeline stops before full code generation and can silently
  accept invalid Cython. Before treating an edge case as a real bug, verify
  the input actually compiles with `Cython.Compiler.Main.compile()` (not
  `cythonize`, which needs `distutils`).
- **Never hardcode a `Cython.Compiler.Nodes`/`PyrexTypes` class name as a
  bare `isinstance`/attribute-access target.** `pyproject.toml` declares a
  supported Cython range (currently `>=3.0.0, <3.4.0`), and Cython's own AST
  class names are not stable across that range — e.g. Cython 3.3 renamed
  `CConstDeclaratorNode` -> `CQualifierDeclaratorNode` and
  `CConstOrVolatileTypeNode` -> `CQualifierTypeNode` (same attribute shape,
  different class, so `isinstance(node, Nodes.CConstDeclaratorNode)` raises
  `AttributeError` outright on 3.3+ rather than just failing to match).
  Resolve version-sensitive classes once, with a fallback, at module import
  time (`getattr(Nodes, "NewName", None) or Nodes.OldName`) and use that
  resolved alias everywhere, the way `type_parsing.py`'s
  `_ConstDeclaratorNode`/`_ConstOrVolatileTypeNode` do. Prefer duck-typing
  on a stable attribute (`getattr(t, "is_cv_qualified", False)`, as
  `render_pyrex_type` already does) over an `isinstance` check on an AST
  class at all, where the attribute itself is stable across versions.
  **When bumping the declared Cython version range**, update
  `.github/workflows/tests.yml`'s `test-cython-versions` job's
  `cython-version` matrix to match the new range's boundaries (it pins
  both ends explicitly, e.g. `3.2.9`/`3.3.0` — the main `test` job only
  ever installs whatever version `pyproject.toml`'s own constraint
  happens to resolve to, so it never actually exercises the bottom of
  the range on its own) — CI then verifies the new bound the same way
  this rule used to require doing by hand in a throwaway venv. Don't
  widen `pyproject.toml` on the changelog's word alone; let that job run
  first.
- **Every `StubgenPyxConfig` field needs a matching CLI flag in `cli.py`.**
  Both `resolve_ctypedef_aliases` and `include_dirs` existed on the config
  (real, documented, tested) with no CLI flag at all — unreachable except
  from the Python API — until both were added together. When adding a new
  config field, add its flag in the same change: a default-`True` field
  gets a `--no-<name>` flag (disables it); a default-`False` field gets a
  plain `--<name>` flag (enables it), matching `--include-private`/
  `--continue-on-error`/`--exclude-attribution`; a `list[str]` field gets a
  `nargs="*"` flag, matching `--exclude-pattern`/`--include-dir`. Wire it
  into `main()`'s `StubgenPyxConfig(...)` construction, and verify
  end-to-end (build the parser, parse args, run `main()` against a real
  file with the flag set) rather than trusting the plumbing from a quick
  read.

## Testing rules

- **Get ground truth before writing an assertion.** Run
  `StubgenPyx(...).convert_str(...)` and read the actual output; never guess
  at expected output.
- **Never assert only existence/truthiness on rich input.** `assert
  len(result) > 0`, `assert result is not None`, `isinstance(result, str)`
  as the *only* check on an elaborate fixture will pass even if the feature
  is completely broken. If a bare existence check is genuinely the most
  specific true thing (e.g. "empty scope renders as nothing"), assert the
  exact value (`result == ""`), not just its type.
- **Treat `or` in an assertion as a red flag.** `"import" in result or
  "Dict" in result` passes even if the specific expected content is
  corrupted, as long as either side survives anywhere in the output. Assert
  the exact expected substring/value instead.
- **Ask: would this assertion still pass if the feature were silently
  broken?** If yes, tighten it.
- **After writing/fixing a test, prove it catches the regression.**
  Temporarily revert just the production code, confirm the test fails for
  the expected reason, then restore the fix.
- **Isolate before reporting a bug.** Compare suspicious output against the
  nearest non-buggy baseline (e.g. same code without the feature in
  question) before concluding it's feature-specific rather than
  pre-existing/unrelated behavior.

## Comment and docstring style

- **State the fact, not the story.** A comment/docstring explains what the
  code does and why, not how that was discovered or what a previous version
  did wrong. Cut narrative framing: "confirmed empirically", "an earlier
  version of this did X", "used to raise/regress", "before this fix",
  "regression test:". If a claim needs backing, phrase it as the fact
  itself (e.g. "`char*` -> `bytes`; a bare `.base.name` breaks on a
  const-qualified pointer" — not "confirmed empirically that an earlier
  version broke on this").
- **Don't cite commit history, design docs, or issue numbers that don't
  ship with the code.** "See the design doc", "see git history", "see
  commit X" point the reader at something that may not be in front of them.
  Inline the one or two sentences of context that actually matter instead.
- **When editing an existing comment near backslashes/regex, verify byte-
  for-byte, don't retype.** A non-raw docstring's `\\.` (renders as one
  literal backslash) is not the same as `\.` (an unrecognized escape,
  silently different content, and a `SyntaxWarning`). If a str_replace
  touches text adjacent to a backslash sequence, leave the sequence itself
  untouched, and afterward run a compile check across the files you edited:
  `python3 -W error::SyntaxWarning -c "compile(open(path).read(), path, 'exec')"`
  for each, or a scan across the whole tree if several files were touched.

