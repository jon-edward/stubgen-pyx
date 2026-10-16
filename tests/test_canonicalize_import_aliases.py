"""Several names bound to one module collapse to the first one in the file."""

from __future__ import annotations

import ast

import pytest

from stubgen_pyx import StubgenPyx, StubgenPyxConfig
from stubgen_pyx.postprocessing.canonicalize_import_aliases import (
    canonicalize_import_aliases,
)


def _canonicalize(source: str) -> str:
    return ast.unparse(canonicalize_import_aliases(ast.parse(source)))


class TestCanonicalization:
    def test_first_alias_wins(self):
        result = _canonicalize(
            "import numpy as np\nimport numpy as cnp\nx: np.ndarray\ny: cnp.ndarray"
        )

        assert result == "import numpy as np\nx: np.ndarray\ny: np.ndarray"

    def test_second_alias_wins_when_it_comes_first(self):
        result = _canonicalize(
            "import numpy as cnp\nimport numpy as np\nx: np.ndarray\ny: cnp.ndarray"
        )

        assert result == "import numpy as cnp\nx: cnp.ndarray\ny: cnp.ndarray"

    def test_plain_import_can_be_canonical(self):
        result = _canonicalize(
            "import numpy\nimport numpy as np\nx: np.ndarray\ny: numpy.ndarray"
        )

        assert result == "import numpy\nx: numpy.ndarray\ny: numpy.ndarray"

    def test_three_names_collapse_to_the_first(self):
        result = _canonicalize(
            "import numpy as a\nimport numpy as b\nimport numpy as c\n"
            "x: b.ndarray\ny: c.ndarray"
        )

        assert result == "import numpy as a\nx: a.ndarray\ny: a.ndarray"

    def test_other_names_in_the_same_statement_are_kept(self):
        result = _canonicalize(
            "import numpy as np\nimport os, numpy as cnp\nx: cnp.ndarray\nimport sys"
        )

        assert result == ("import numpy as np\nimport os\nx: np.ndarray\nimport sys")

    def test_dotted_module_alias(self):
        result = _canonicalize(
            "import pkg.lib as pl\nimport pkg.lib as lib\nx: lib.Foo"
        )

        assert result == "import pkg.lib as pl\nx: pl.Foo"

    def test_rewrites_defaults_varargs_returns_and_class_bodies(self):
        result = _canonicalize(
            "import numpy as np\nimport numpy as cnp\n"
            "def f(a: cnp.ndarray = cnp.zeros(1), *args: cnp.ndarray)"
            " -> list[cnp.ndarray]: ...\n"
            "class A:\n    v: cnp.ndarray"
        )

        assert "cnp" not in result
        assert result.count("np.ndarray") == 4

    def test_dotted_plain_import_is_left_alone(self):
        result = _canonicalize(
            "import numpy as np\nimport numpy.typing\nimport numpy as cnp\n"
            "x: numpy.typing.NDArray[cnp.uint8]"
        )

        assert result == (
            "import numpy as np\nimport numpy.typing\nx: numpy.typing.NDArray[np.uint8]"
        )

    def test_single_alias_is_unchanged(self):
        assert _canonicalize("import numpy as np\nx: np.ndarray") == (
            "import numpy as np\nx: np.ndarray"
        )

    def test_idempotent(self):
        once = _canonicalize("import numpy as np\nimport numpy as cnp\nx: cnp.ndarray")

        assert _canonicalize(once) == once

    def test_requires_a_module(self):
        with pytest.raises(TypeError):
            canonicalize_import_aliases(ast.Name(id="x"))


class TestConflictsLeaveNamesAlone:
    @pytest.mark.parametrize(
        "source",
        [
            pytest.param(
                "import numpy as np\nimport numpy as cnp\nclass cnp: ...\nx: np.ndarray",
                id="class-named-like-alias",
            ),
            pytest.param(
                "import numpy as np\nimport numpy as cnp\n"
                "def f(cnp: int) -> np.ndarray: ...",
                id="argument-named-like-alias",
            ),
            pytest.param(
                "import numpy as np\nimport numpy as cnp\nfrom foo import cnp\n"
                "x: np.ndarray",
                id="from-import-named-like-alias",
            ),
            pytest.param(
                "import a as x\nimport b as x\nimport a as y\nz: y.T",
                id="one-name-bound-to-two-modules",
            ),
            pytest.param(
                "import numpy as np\nimport numpy as cnp\n"
                "x: 'cnp.ndarray'\ny: np.ndarray",
                id="quoted-annotation-uses-removed-name",
            ),
        ],
    )
    def test_unchanged(self, source):
        assert _canonicalize(source) == ast.unparse(ast.parse(source))


class TestPipeline:
    _SRC = (
        "cimport numpy as cnp\nimport numpy as np\n\n"
        "cdef class A:\n    cdef readonly cnp.ndarray arr\n\n"
        "    def f(self) -> np.ndarray: ...\n"
    )

    def test_enabled_by_default(self):
        result = StubgenPyx(StubgenPyxConfig(exclude_attribution=True)).convert_str(
            self._SRC
        )

        assert result == (
            "from typing import Final\n\nimport numpy as cnp\n\n\n"
            "class A:\n    arr: Final[cnp.ndarray]\n\n"
            "    def f(self) -> cnp.ndarray: ...\n"
        )

    def test_can_be_disabled(self):
        config = StubgenPyxConfig(
            exclude_attribution=True, canonicalize_import_aliases=False
        )

        result = StubgenPyx(config).convert_str(self._SRC)

        assert result == (
            "from typing import Final\n\nimport numpy as cnp\nimport numpy as np\n\n\n"
            "class A:\n    arr: Final[cnp.ndarray]\n\n"
            "    def f(self) -> np.ndarray: ...\n"
        )
