"""Tests for the remove_overload_implementations postprocessing module."""

from __future__ import annotations

import ast
import textwrap

from stubgen_pyx import StubgenPyx
from stubgen_pyx.config import StubgenPyxConfig
from stubgen_pyx.postprocessing.pipeline import postprocessing_pipeline
from stubgen_pyx.postprocessing.remove_overload_implementations import (
    remove_overload_implementations,
)


def _transform(code: str) -> ast.Module:
    """Apply the pass to a module and return the resulting tree."""
    tree = remove_overload_implementations(ast.parse(textwrap.dedent(code)))
    assert isinstance(tree, ast.Module)
    return tree


def _functions(body: list[ast.stmt]) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    """Return the function definitions of a body, in order."""
    return [
        stmt
        for stmt in body
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]


def _class(tree: ast.Module, name: str) -> ast.ClassDef:
    """Return the class of the given name from a module body."""
    for stmt in tree.body:
        if isinstance(stmt, ast.ClassDef) and stmt.name == name:
            return stmt
    raise AssertionError(f"class {name!r} not found")


class TestRemoveOverloadImplementations:
    """Test the removal of implementations declared with ``@overload``."""

    def test_module_level_implementation_removed(self):
        """A module-level implementation is removed, declarations are kept."""
        tree = _transform("""
            from typing import overload

            @overload
            def f(x: int) -> int: ...
            @overload
            def f(x: str) -> str: ...
            def f(x):
                return x
            """)
        functions = _functions(tree.body)
        assert len(functions) == 2
        assert [f.returns.id for f in functions] == ["int", "str"]

    def test_method_implementation_removed(self):
        """A method implementation is removed from a class body."""
        tree = _transform("""
            from typing import overload

            class A:
                @overload
                def __getitem__(self, x: int) -> int: ...
                @overload
                def __getitem__(self, x: str) -> str: ...
                def __getitem__(self, x): ...
            """)
        assert len(_functions(_class(tree, "A").body)) == 2

    def test_dotted_decorator_recognized(self):
        """``typing.overload`` counts as an overload declaration."""
        tree = _transform("""
            import typing

            @typing.overload
            def f(x: int) -> int: ...
            @typing.overload
            def f(x: str) -> str: ...
            def f(x): ...
            """)
        assert len(_functions(tree.body)) == 2

    def test_async_implementation_removed(self):
        """An ``async def`` implementation is removed like a plain one."""
        tree = _transform("""
            from typing import overload

            @overload
            async def f(x: int) -> int: ...
            @overload
            async def f(x: str) -> str: ...
            async def f(x): ...
            """)
        assert len(_functions(tree.body)) == 2

    def test_decorated_implementation_removed(self):
        """An implementation with other decorators is still removed."""
        tree = _transform("""
            from typing import overload

            class A:
                @overload
                @staticmethod
                def f(x: int) -> int: ...
                @overload
                @staticmethod
                def f(x: str) -> str: ...
                @staticmethod
                def f(x): ...
            """)
        assert len(_functions(_class(tree, "A").body)) == 2

    def test_function_without_overloads_kept(self):
        """A definition with no overload declarations is left alone."""
        tree = _transform("""
            def f(x): ...
            def g(x): ...
            """)
        assert [f.name for f in _functions(tree.body)] == ["f", "g"]

    def test_scopes_are_independent(self):
        """An overload in a class does not affect a module-level function."""
        tree = _transform("""
            from typing import overload

            class A:
                @overload
                def f(self, x: int) -> int: ...
                @overload
                def f(self, x: str) -> str: ...
                def f(self, x): ...

            def f(x): ...
            """)
        assert [f.name for f in _functions(tree.body)] == ["f"]
        assert len(_functions(_class(tree, "A").body)) == 2

    def test_nested_class_processed(self):
        """A class body nested in another class is processed as its own scope."""
        tree = _transform("""
            from typing import overload

            class Outer:
                class Inner:
                    @overload
                    def f(self, x: int) -> int: ...
                    @overload
                    def f(self, x: str) -> str: ...
                    def f(self, x): ...
            """)
        inner = _class(_class(tree, "Outer"), "Inner")
        assert len(_functions(inner.body)) == 2

    def test_surrounding_statements_preserved(self):
        """Statements other than the implementation keep their order."""
        tree = _transform("""
            from typing import overload

            VERSION: str

            @overload
            def f(x: int) -> int: ...
            @overload
            def f(x: str) -> str: ...
            def f(x): ...

            class A: ...
            """)
        assert [type(stmt).__name__ for stmt in tree.body] == [
            "ImportFrom",
            "AnnAssign",
            "FunctionDef",
            "FunctionDef",
            "ClassDef",
        ]


class TestRemoveOverloadImplementationsInPipeline:
    """Test the pass as part of the postprocessing pipeline."""

    def test_generated_stub_omits_implementation(self):
        """Overloads written in a .pyx reach the stub without the body."""
        pyi = StubgenPyx(
            StubgenPyxConfig(exclude_attribution=True, sort_imports=False)
        ).convert_str(
            textwrap.dedent("""
                from typing import overload

                cdef class A:
                    @overload
                    def __getitem__(self, x: int) -> int: ...
                    @overload
                    def __getitem__(self, x: str) -> str: ...
                    def __getitem__(self, x):
                        return x
                """)
        )
        assert pyi == textwrap.dedent("""\
            from typing import overload
            class A:
                @overload
                def __getitem__(self, x: int) -> int: ...
                @overload
                def __getitem__(self, x: str) -> str: ...
            """)

    def test_import_used_only_by_implementation_is_trimmed(self):
        """Removal happens before import trimming, so stale imports go away."""
        pyi = postprocessing_pipeline(
            textwrap.dedent("""\
                from typing import Any, overload

                @overload
                def f(x: int) -> int: ...
                @overload
                def f(x: str) -> str: ...
                def f(x: Any) -> Any: ...
                """),
            StubgenPyxConfig(exclude_attribution=True),
        )
        assert "Any" not in pyi
        assert "overload" in pyi
