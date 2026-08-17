"""Tests for fix_scalar_defaults module."""

from __future__ import annotations

import ast

from stubgen_pyx.postprocessing.fix_scalar_defaults import fix_scalar_defaults


class TestFixScalarDefaults:
    """Test the fix_scalar_defaults module."""

    def test_str_default_coerced_to_bytes(self):
        code = "def f(x: bytes = '.') -> None: pass"
        tree = ast.parse(code)
        result = fix_scalar_defaults(tree)
        assert ast.unparse(result) == "def f(x: bytes=b'.') -> None:\n    pass"

    def test_int_default_coerced_to_bool_false(self):
        code = "def f(flag: bool = 0) -> None: pass"
        tree = ast.parse(code)
        result = fix_scalar_defaults(tree)
        assert ast.unparse(result) == "def f(flag: bool=False) -> None:\n    pass"

    def test_int_default_coerced_to_bool_true(self):
        code = "def f(flag: bool = 1) -> None: pass"
        tree = ast.parse(code)
        result = fix_scalar_defaults(tree)
        assert ast.unparse(result) == "def f(flag: bool=True) -> None:\n    pass"

    def test_bint_annotation_matched_when_not_normalized(self):
        """`bint` (the pre-normalize_names spelling) is matched too."""
        code = "def f(flag: bint = 1) -> None: pass"
        tree = ast.parse(code)
        result = fix_scalar_defaults(tree)
        assert ast.unparse(result) == "def f(flag: bint=True) -> None:\n    pass"

    def test_mixed_positional_args(self):
        code = "def f(a: int, x: bytes = '.', flag: bool = 0) -> None: pass"
        tree = ast.parse(code)
        result = fix_scalar_defaults(tree)
        result_str = ast.unparse(result)
        assert "x: bytes=b'.'" in result_str
        assert "flag: bool=False" in result_str

    def test_kwonly_args(self):
        code = "def f(*, x: bytes = 'ab', flag: bool = 1) -> None: pass"
        tree = ast.parse(code)
        result = fix_scalar_defaults(tree)
        result_str = ast.unparse(result)
        assert "flag: bool=True" in result_str
        assert "x: bytes=b'ab'" in result_str

    def test_kwonly_required_arg_not_touched(self):
        """A required keyword-only arg has a `None` slot in kw_defaults."""
        code = "def f(*, x: bytes, flag: bool = 0) -> None: pass"
        tree = ast.parse(code)
        result = fix_scalar_defaults(tree)
        result_str = ast.unparse(result)
        assert "x: bytes," in result_str
        assert "flag: bool=False" in result_str

    def test_posonly_args(self):
        code = "def f(x: bytes = '.', /, y: int = 1) -> None: pass"
        tree = ast.parse(code)
        result = fix_scalar_defaults(tree)
        result_str = ast.unparse(result)
        assert "y: int=1" in result_str

    def test_async_function(self):
        code = "async def f(flag: bool = 0) -> None: pass"
        tree = ast.parse(code)
        result = fix_scalar_defaults(tree)
        assert "flag: bool=False" in ast.unparse(result)

    def test_already_bytes_default_untouched(self):
        code = "def f(x: bytes = b'.') -> None: pass"
        tree = ast.parse(code)
        result = fix_scalar_defaults(tree)
        assert ast.unparse(result) == "def f(x: bytes=b'.') -> None:\n    pass"

    def test_already_bool_default_untouched(self):
        code = "def f(flag: bool = True) -> None: pass"
        tree = ast.parse(code)
        result = fix_scalar_defaults(tree)
        assert ast.unparse(result) == "def f(flag: bool=True) -> None:\n    pass"

    def test_optional_bytes_annotation_untouched(self):
        """Subscripted annotations aren't matched; only bare `bytes`/`bool`."""
        code = "def f(x: Optional[bytes] = '.') -> None: pass"
        tree = ast.parse(code)
        result = fix_scalar_defaults(tree)
        assert "'.'" in ast.unparse(result)

    def test_non_literal_default_untouched(self):
        code = "def f(x: bytes = SOME_DEFAULT) -> None: pass"
        tree = ast.parse(code)
        result = fix_scalar_defaults(tree)
        assert ast.unparse(result) == "def f(x: bytes=SOME_DEFAULT) -> None:\n    pass"

    def test_int_default_on_bytes_annotation_untouched(self):
        """A mismatch this pass doesn't claim to fix - not the char*/bint pattern."""
        code = "def f(x: bytes = 5) -> None: pass"
        tree = ast.parse(code)
        result = fix_scalar_defaults(tree)
        assert ast.unparse(result) == "def f(x: bytes=5) -> None:\n    pass"

    def test_str_default_on_bool_annotation_untouched(self):
        code = "def f(flag: bool = 'x') -> None: pass"
        tree = ast.parse(code)
        result = fix_scalar_defaults(tree)
        assert ast.unparse(result) == "def f(flag: bool='x') -> None:\n    pass"

    def test_unencodable_str_default_left_as_str(self):
        """Chars outside latin-1 range can't map to a single byte; left alone."""
        code = "def f(x: bytes = '\u2603') -> None: pass"  # snowman, U+2603
        tree = ast.parse(code)
        result = fix_scalar_defaults(tree)
        assert "'\u2603'" in ast.unparse(result)

    def test_bint_true_false_constants_untouched(self):
        code = "def f(flag: bint = True) -> None: pass"
        tree = ast.parse(code)
        result = fix_scalar_defaults(tree)
        assert ast.unparse(result) == "def f(flag: bint=True) -> None:\n    pass"
