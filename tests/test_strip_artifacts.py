from __future__ import annotations

import ast

from stubgen_pyx.postprocessing.strip_artifacts import strip_artifacts


def _strip(code: str) -> str:
    tree = ast.parse(code)
    tree = strip_artifacts(tree)
    return ast.unparse(tree)


def test_strip_artifacts_preserves_dunder_all_list_assignment():
    code = "__all__ = ['Foo', 'Bar']\nclass Foo: pass"

    result = _strip(code)

    assert "__all__ = ['Foo', 'Bar']" in result
    assert "class Foo" in result


def test_strip_artifacts_preserves_dunder_all_tuple_assignment():
    code = "__all__ = ('foo',)\nclass Foo: pass"

    result = _strip(code)

    assert "__all__" in result
    assert "'foo'" in result
    assert "class Foo" in result


def test_strip_artifacts_removes_class_hash_none_assignment():
    code = "class Foo:\n    __hash__ = None\n    def method(self) -> None: ..."

    result = _strip(code)

    assert "__hash__" not in result
    assert "def method" in result


def test_strip_artifacts_keeps_module_level_hash_none_assignment():
    code = "__hash__ = None"

    result = _strip(code)

    assert "__hash__" in result


def test_strip_artifacts_removes_cython_and_cpython_imports():
    code = (
        "from cython import no_gc_clear\n"
        "from cpython import bool as py_bool\n"
        "import cython\n"
        "class Foo: pass"
    )

    result = _strip(code)

    assert "cython" not in result
    assert "cpython" not in result
    assert "class Foo" in result


def test_strip_artifacts_import_filters_cython_names_but_keeps_siblings():
    code = "import os, cython, sys\nclass Foo: pass"

    result = _strip(code)

    assert "cython" not in result
    assert "import os" in result
    assert "sys" in result
    assert "class Foo" in result


def test_strip_artifacts_removes_runtime_constant_call_rhs():
    code = "SIZE_TYPE = DataType(42)\nX = 42\nclass Foo: pass"

    result = _strip(code)

    assert "SIZE_TYPE" not in result
    assert "X = 42" in result
    assert "class Foo" in result


def test_strip_artifacts_removes_runtime_constant_attribute_rhs():
    code = "SEED = some_module.DEFAULT_SEED\nX = 42"

    result = _strip(code)

    assert "SEED = some_module" not in result
    assert "X = 42" in result


def test_strip_artifacts_is_idempotent():
    code = "__all__ = ['Foo']\nclass Foo:\n    __hash__ = None\n    def method(self) -> None: ..."

    once = _strip(code)
    twice = _strip(once)

    assert once == twice


def test_strip_artifacts_removes_all_supported_artifacts_together():
    code = (
        "from cython import no_gc_clear\n"
        "from cpython import bool as py_bool\n"
        "import cython\n"
        "__all__ = ['Foo']\n"
        "SIZE_TYPE = DataType(42)\n"
        "SEED = some_module.DEFAULT_SEED\n"
        "class Foo:\n"
        "    __hash__ = None\n"
        "    def method(self) -> None: ...\n"
        "X = 42\n"
        "def keep_me() -> None: ..."
    )

    result = _strip(code)

    assert "__all__ = ['Foo']" in result
    assert "__hash__" not in result
    assert "cython" not in result
    assert "cpython" not in result
    assert "SIZE_TYPE" not in result
    assert "SEED = some_module" not in result
    assert "class Foo" in result
    assert "def method" in result
    assert "X = 42" in result
    assert "keep_me" in result


def test_strip_artifacts_empty_module_result_is_valid_python():
    code = "__all__ = ['Foo']"

    result = _strip(code)

    ast.parse(result)


def test_strip_artifacts_preserves_dunder_all_call_assignment():
    code = "__all__ = build_all()\nclass Foo: pass"

    result = _strip(code)

    assert "__all__" in result
    assert "class Foo" in result


def test_strip_artifacts_preserves_dunder_all_attribute_assignment():
    code = "__all__ = exports.ALL\nclass Foo: pass"

    result = _strip(code)

    assert "__all__" in result
    assert "class Foo" in result


def test_strip_artifacts_rewrites_dangling_cython_annotation():
    code = "from cython import bint\ndef f(x: bint) -> None: ..."
    result = ast.unparse(strip_artifacts(ast.parse(code)))
    assert "from typing import Any" in result
    assert "bint" not in result
    assert "def f(x: Any) -> None" in result


def test_defined_annotation_is_untouched():
    result = _strip("class Foo: ...\ndef f(x: Foo) -> Foo: ...")

    assert "def f(x: Foo) -> Foo" in result
    assert "Any" not in result


def test_dangling_annotation_is_replaced_with_any():
    result = _strip("def f(x: StrippedCimport) -> None: ...")

    assert "from typing import Any" in result
    assert "def f(x: Any) -> None" in result


def test_nested_generic_dangling_argument_is_replaced():
    result = _strip(
        "from typing import List\ndef f(x: List[DanglingName]) -> None: ..."
    )

    assert "from typing import Any" in result
    assert "List[Any]" in result


def test_multiple_dangling_names_are_replaced():
    result = _strip("def f(x: MissingOne) -> MissingTwo: ...")

    assert "from typing import Any" in result
    assert "def f(x: Any) -> Any" in result


def test_builtin_annotations_are_preserved():
    result = _strip("def f(x: int, y: str, z: bool) -> bytes: ...")

    assert "def f(x: int, y: str, z: bool) -> bytes" in result
    assert "Any" not in result


def test_existing_any_import_is_not_duplicated():
    result = _strip("from typing import Any\ndef f(x: Missing) -> Any: ...")

    assert result.count("from typing import Any") == 1
    assert "def f(x: Any) -> Any" in result


def test_dangling_decorator_argument_is_replaced_with_any():
    result = _strip(
        "some_decorator = None\n@some_decorator(UndefinedType)\ndef foo() -> None: ..."
    )

    assert "from typing import Any" in result
    assert "@some_decorator(Any)" in result
