from __future__ import annotations

import ast

from stubgen_pyx.postprocessing.strip_artifacts import (
    strip_artifacts as validate_annotations,
)


def _validate(code: str) -> str:
    return ast.unparse(validate_annotations(ast.parse(code)))


def test_defined_annotation_is_untouched():
    result = _validate("class Foo: ...\ndef f(x: Foo) -> Foo: ...")

    assert "def f(x: Foo) -> Foo" in result
    assert "Any" not in result


def test_dangling_annotation_is_replaced_with_any():
    result = _validate("def f(x: StrippedCimport) -> None: ...")

    assert "from typing import Any" in result
    assert "def f(x: Any) -> None" in result


def test_nested_generic_dangling_argument_is_replaced():
    result = _validate(
        "from typing import List\ndef f(x: List[DanglingName]) -> None: ..."
    )

    assert "from typing import Any" in result
    assert "List[Any]" in result


def test_multiple_dangling_names_are_replaced():
    result = _validate("def f(x: MissingOne) -> MissingTwo: ...")

    assert "from typing import Any" in result
    assert "def f(x: Any) -> Any" in result


def test_builtin_annotations_are_preserved():
    result = _validate("def f(x: int, y: str, z: bool) -> bytes: ...")

    assert "def f(x: int, y: str, z: bool) -> bytes" in result
    assert "Any" not in result


def test_existing_any_import_is_not_duplicated():
    result = _validate("from typing import Any\ndef f(x: Missing) -> Any: ...")

    assert result.count("from typing import Any") == 1
    assert "def f(x: Any) -> Any" in result


def test_dangling_decorator_argument_is_replaced_with_any():
    result = _validate(
        "some_decorator = None\n@some_decorator(UndefinedType)\ndef foo() -> None: ..."
    )

    assert "from typing import Any" in result
    assert "@some_decorator(Any)" in result
