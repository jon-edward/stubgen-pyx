"""A trimmed Cython compiler pipeline for stub generation.

Modeled on ``Cython.Compiler.Pipeline.create_pipeline``, but stops right
after ``AnalyseDeclarationsTransform`` -- the stage that walks the tree and
attaches ``Entry``/``Type`` objects (``Cython.Compiler.Symtab.Entry``,
``Cython.Compiler.PyrexTypes.Type``) to every declaration. That's all stub
generation needs: a populated symbol table. Everything Cython's real
pipeline does after that stage exists to produce compilable, optimized C
and assumes a fully resolvable compile environment (real C libraries,
real GIL/closure/buffer semantics) -- none of which stub generation should
require or benefit from.

Stages deliberately excluded, and why (so nobody "helpfully" adds them back):

- ``TrackNumpyAttributes`` / ``ParallelRangeTransform``: numpy/prange
  codegen hints, irrelevant to declarations.
- ``MarkClosureVisitor`` / ``CreateClosureClasses``: closure codegen.
- ``AutoCpdefFunctionDefinitions``: mutates ``.pxd``-driven ``cpdef``
  inference for C linkage purposes, not the Python-visible signature.
- ``RemoveUnreachableCode`` / ``ConstantFolding`` / ``FlattenInListTransform``
  / ``Optimize.*``: dead-code and constant-folding optimizations -- stub
  generation wants the code as declared, not as optimized.
- ``ControlFlowAnalysis`` / ``MarkOverflowingArithmetic`` /
  ``IntroduceBufferAuxiliaryVars``: flow analysis for C codegen correctness.
- ``check_c_declarations`` / ``check_c_declarations_pxd``: requires full
  C-level resolvability we don't want to demand of arbitrary input.
- ``AnalyseExpressionsTransform`` onward: expression-level type inference
  against a real compiled environment -- out of scope; stub generation
  needs declared signatures, not inferred expression types.
- ``GilCheck`` / ``CoerceCppTemps`` / ``FinalOptimizePhase``: pure codegen.
"""

from __future__ import annotations

from typing import Callable, Literal

from Cython.Compiler.Main import Context
from Cython.Compiler.ParseTreeTransforms import (
    AnalyseDeclarationsTransform,
    DecoratorTransform,
    ForwardDeclareTypes,
    InterpretCompilerDirectives,
    NormalizeTree,
    PostParse,
    PxdPostParse,
    WithTransform,
)

PipelineMode = Literal["pyx", "pxd"]

#: A pipeline stage: a callable that takes the tree (or, for the first
#: stage, a ``CompilationSource``/``SourceDescriptor``) and returns it,
#: possibly transformed. Matches the shape ``Pipeline.run_pipeline`` expects.
Stage = Callable[..., object]


def stub_pipeline(context: Context, mode: PipelineMode) -> list[Stage | None]:
    """Build the list of transforms to run over a freshly parsed tree.

    Does not include a parse stage -- callers are expected to have already
    produced a tree via ``context.parse(...)`` (see ``parsing/parser.py``)
    and pass it as the initial ``data`` to
    ``Cython.Compiler.Pipeline.run_pipeline``.

    Args:
        context: The (shared) ``StubgenContext`` these transforms should
            run against.
        mode: ``"pyx"`` for an implementation file, ``"pxd"`` for a
            declaration-only file. Only affects whether ``PxdPostParse``
            runs, matching ``Cython.Compiler.Pipeline.create_pipeline``.

    Returns:
        A list of callables (with a ``None`` placeholder skipped by
        ``run_pipeline``) to be run in order over the parsed tree.
    """
    if mode not in ("pyx", "pxd"):
        raise ValueError(f"mode must be 'pyx' or 'pxd', got {mode!r}")

    specific_post_parse = PxdPostParse(context) if mode == "pxd" else None

    return [
        NormalizeTree(context),
        PostParse(context),
        specific_post_parse,
        InterpretCompilerDirectives(context, context.compiler_directives),
        WithTransform(),
        DecoratorTransform(context),
        ForwardDeclareTypes(context),
        AnalyseDeclarationsTransform(context),
    ]


class StubPipelineResult:
    """The outcome of running the stub pipeline over one file.

    Attributes:
        tree: The transformed tree (``ModuleNode``), with declarations
            analysed as far as ``AnalyseDeclarationsTransform`` could take
            them -- populated even when ``diagnostics`` is non-empty, since
            Cython's error reporting (``Cython.Compiler.Errors.error``)
            records problems and keeps going rather than aborting, and
            individual failed lookups (e.g. an unresolved ``cimport``) only
            affect the entries touched by that one declaration.
        diagnostics: ``CompileError``s Cython recorded while running the
            pipeline (e.g. "'some_module.pxd' not found" for a `cimport`
            of a module that isn't on the include path). Collected rather
            than raised or printed to stderr -- stub generation is
            expected to run against source that isn't necessarily fully
            resolvable in this environment, so these are reported to the
            caller as data, not treated as fatal.
    """

    __slots__ = ("diagnostics", "tree")

    def __init__(self, tree: object, diagnostics: list) -> None:
        self.tree = tree
        self.diagnostics = diagnostics


def run_stub_pipeline(
    context: Context, mode: PipelineMode, tree: object
) -> StubPipelineResult:
    """Run ``stub_pipeline(context, mode)`` over an already-parsed tree.

    Errors Cython's own error-reporting records during the run (see
    ``Cython.Compiler.Errors.hold_errors``/``release_errors``) are
    collected into the result rather than echoed to stderr or raised --
    this is what keeps an unresolvable ``cimport`` (a common, expected
    situation for a stub generator, which can't assume every dependency is
    importable in this environment) from being fatal or noisy.

    Only re-raises if the pipeline itself signals an unrecoverable
    condition (``AbortError``/``InternalError`` from
    ``Cython.Compiler.Pipeline.run_pipeline``) -- something Cython's own
    error-accumulation couldn't route around, as opposed to an ordinary
    recorded ``CompileError``.
    """
    from Cython.Compiler import Errors
    from Cython.Compiler.Pipeline import run_pipeline

    from .context import _ensure_errors_thread_initialized

    _ensure_errors_thread_initialized()
    held = Errors.hold_errors()
    try:
        error, data = run_pipeline(stub_pipeline(context, mode), tree, printtree=False)
    finally:
        Errors.release_errors(ignore=True)

    diagnostics = list(held)
    if error is not None:
        # A CompileError here (as opposed to one merely recorded via
        # Errors.error(), which doesn't raise) means run_pipeline's own
        # try/except caught it escaping a stage -- something an ordinary
        # recorded-but-recoverable issue wouldn't do. Surface it as a
        # diagnostic too rather than raising, for the same reason: we
        # don't want one bad declaration to take down the whole file's
        # stub generation.
        diagnostics.append(error)
    return StubPipelineResult(data, diagnostics)
