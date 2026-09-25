"""Tests for `stubgen.py`'s less-common paths: the `__init__`-skip
shortcut, batch conversion's `continue_on_error` handling at each of
its phases, encoding-error wrapping, and the small standalone helpers.
Several of these -- an OS-level write failure, a `.pxd`/`.pyx` file
that genuinely fails to decode as UTF-8 -- proved unreliable to trigger
through real broken files (Cython's own scanner tolerates more invalid
byte sequences than expected, and this sandbox runs as root, bypassing
real permission failures), so those are exercised via controlled
monkeypatching instead, verified to hit the intended branch specifically.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from Cython.Compiler import Errors as CythonErrors

from stubgen_pyx.config import StubgenPyxConfig
from stubgen_pyx.models.pyi_elements import PyiClass, PyiScope
from stubgen_pyx.stubgen import StubgenPyx, _log_diagnostics, _merge_classes


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def init_pair(temp_dir):
    """An `__init__.pyx` alongside an existing `__init__.py`."""
    (temp_dir / "__init__.py").write_text("")
    pyx_file = temp_dir / "__init__.pyx"
    pyx_file.write_text("")
    return pyx_file


class TestInitSkip:
    def test_convert_single_file_skips_when_init_py_exists(self, init_pair):
        result = StubgenPyx().convert_single_file(init_pair)
        assert result.success is True
        assert result.pyx_file == result.pyi_file
        assert result.status_message == f"Skipped {init_pair}"

    def test_batch_conversion_also_skips_it(self, temp_dir, init_pair):
        (temp_dir / "other.pyx").write_text("def f(x: int) -> int:\n    return x\n")
        stubgen = StubgenPyx(StubgenPyxConfig(resolve_ctypedef_aliases=True))
        results = stubgen.convert_glob(str(temp_dir / "*.pyx"))
        by_name = {r.pyx_file.name: r for r in results}
        assert by_name["__init__.pyx"].status_message == f"Skipped {init_pair}"
        assert by_name["other.pyx"].success is True


class TestFileNotFound:
    def test_convert_single_file_raises_value_error(self, temp_dir):
        missing = temp_dir / "does_not_exist.pyx"
        with pytest.raises(ValueError, match="File not found"):
            StubgenPyx().convert_single_file(missing)


class TestContinueOnErrorBatch:
    """A genuinely broken file in a batch: `continue_on_error` decides
    whether it's caught and reported per-file or re-raised."""

    @pytest.fixture
    def mixed_batch(self, temp_dir):
        (temp_dir / "good.pyx").write_text("def f(x: int) -> int:\n    return x\n")
        (temp_dir / "bad.pyx").write_text("def f(x: int -> int:\n    pass\n")
        return temp_dir

    def test_continue_on_error_true_reports_the_failure_and_keeps_going(
        self, mixed_batch
    ):
        stubgen = StubgenPyx(StubgenPyxConfig(continue_on_error=True))
        results = stubgen.convert_glob(str(mixed_batch / "*.pyx"))
        by_name = {r.pyx_file.name: r for r in results}
        assert by_name["good.pyx"].success is True
        assert by_name["bad.pyx"].success is False
        assert by_name["bad.pyx"].error is not None

    def test_continue_on_error_false_raises(self, mixed_batch):
        stubgen = StubgenPyx(StubgenPyxConfig(continue_on_error=False))
        with pytest.raises(CythonErrors.CompileError):
            stubgen.convert_glob(str(mixed_batch / "*.pyx"))

    def test_continue_on_error_true_with_ctypedef_pruning_batch_path(self, mixed_batch):
        """Same, but through `_convert_multiple_files_with_ctypedef_pruning`
        (the separate batch path `resolve_ctypedef_aliases` uses), which
        has its own, earlier try/except around the parse-and-convert phase."""
        stubgen = StubgenPyx(
            StubgenPyxConfig(resolve_ctypedef_aliases=True, continue_on_error=True)
        )
        results = stubgen.convert_glob(str(mixed_batch / "*.pyx"))
        by_name = {r.pyx_file.name: r for r in results}
        assert by_name["good.pyx"].success is True
        assert by_name["bad.pyx"].success is False

    def test_render_phase_failure_with_continue_on_error(self, mixed_batch):
        """A failure *after* parsing succeeds -- during rendering --
        must be caught the same way as a parse-time failure."""
        (mixed_batch / "bad.pyx").unlink()  # only the good file this time
        stubgen = StubgenPyx(
            StubgenPyxConfig(resolve_ctypedef_aliases=True, continue_on_error=True)
        )
        with patch.object(
            StubgenPyx, "_finalize", side_effect=RuntimeError("render boom")
        ):
            results = stubgen.convert_glob(str(mixed_batch / "good.pyx"))
        assert results[0].success is False
        assert "render boom" in str(results[0].error)


class TestWriteFailure:
    """A failure writing the `.pyi` file to disk -- surfaced as `OSError`
    with the target path in the message, from both the single-file and
    batch (`resolve_ctypedef_aliases`) code paths.
    """

    def test_convert_single_file_wraps_write_failure(self, temp_dir):
        pyx_file = temp_dir / "mod.pyx"
        pyx_file.write_text("def f(x: int) -> int:\n    return x\n")
        with (
            patch.object(Path, "write_text", side_effect=OSError("disk full")),
            pytest.raises(OSError, match="Failed to write"),
        ):
            StubgenPyx().convert_single_file(pyx_file)

    def test_batch_ctypedef_pruning_path_wraps_write_failure(self, temp_dir):
        pyx_file = temp_dir / "mod.pyx"
        pyx_file.write_text("def f(x: int) -> int:\n    return x\n")
        stubgen = StubgenPyx(StubgenPyxConfig(resolve_ctypedef_aliases=True))
        with (
            patch.object(Path, "write_text", side_effect=OSError("disk full")),
            pytest.raises(OSError, match="Failed to write"),
        ):
            stubgen.convert_glob(str(pyx_file))


class TestEncodingErrorWrapping:
    """Cython's own `Context.parse` catches a raw `UnicodeDecodeError`
    itself and re-raises it wrapped as a `CompileError` (see
    `_compile_file_with_error_handling`'s docstring) -- confirmed hard
    to trigger from a genuinely malformed file in practice (Cython's
    scanner tolerates more invalid byte sequences than expected), so
    the wrapped shape is constructed directly here instead.
    """

    @staticmethod
    def _compile_error_wrapping_decode_error() -> CythonErrors.CompileError:
        decode_error = UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid byte")
        error = CythonErrors.CompileError()
        error.__cause__ = decode_error
        return error

    def test_main_file_decode_error_becomes_value_error(self, temp_dir):
        pyx_file = temp_dir / "mod.pyx"
        pyx_file.write_text("def f(x: int) -> int:\n    return x\n")
        with (
            patch.object(
                StubgenPyx,
                "_compile_file_with_converter",
                side_effect=self._compile_error_wrapping_decode_error(),
            ),
            pytest.raises(ValueError, match="File encoding error"),
        ):
            StubgenPyx().convert_single_file(pyx_file)

    def test_pxd_decode_error_is_logged_and_conversion_continues(
        self, temp_dir, caplog
    ):
        """Unlike the main file, a companion `.pxd` that can't be read
        doesn't fail the whole conversion -- it's logged and the `.pyx`
        is still converted on its own."""
        pyx_file = temp_dir / "mod.pyx"
        pyx_file.write_text("def f(x: int) -> int:\n    return x\n")
        (temp_dir / "mod.pxd").write_text("ctypedef int x\n")

        import stubgen_pyx.stubgen as stubgen_mod

        real_parse_file = stubgen_mod.parse_file
        error = self._compile_error_wrapping_decode_error()

        def fake_parse_file(path, context, pxd=False, module_name=None):
            if pxd:
                raise error
            return real_parse_file(path, context, pxd=pxd, module_name=module_name)

        with patch.object(stubgen_mod, "parse_file", side_effect=fake_parse_file):
            result = StubgenPyx().convert_single_file(pyx_file)
        assert result.success is True
        assert "Could not read .pxd file" in caplog.text


class TestLogDiagnostics:
    def test_diagnostic_without_message_only_or_position_falls_back_to_str(
        self, caplog
    ):
        """Not every diagnostic is a `CompileError` with `message_only`/
        `position` -- anything else falls back to plain `str()`."""
        _log_diagnostics([ValueError("a plain, unformatted diagnostic")], None)
        assert "a plain, unformatted diagnostic" in caplog.text

    def test_unformattable_diagnostic_falls_back_gracefully(self, caplog):
        """Formatting a diagnostic nicely is a courtesy, never a
        requirement -- a diagnostic whose `position` doesn't actually
        unpack as `(source_desc, line, column)` must not crash the
        whole conversion over a logging nicety."""

        class _BadPosition:
            message_only = "oops"
            position = ("way", "too", "many", "values", "to", "unpack")

        _log_diagnostics([_BadPosition()], None)
        assert "unformattable diagnostic" in caplog.text


class TestMergeClasses:
    def test_appends_a_class_with_no_existing_same_name_match(self):
        scope = PyiScope(classes=[PyiClass("Existing", scope=PyiScope())])
        extra = PyiClass("BrandNew", scope=PyiScope())
        _merge_classes(scope, [extra])
        assert [c.name for c in scope.classes] == ["Existing", "BrandNew"]


class TestResolvePyiPath:
    def test_falls_back_to_bare_filename_when_not_relative_to_common_root(
        self, temp_dir
    ):
        """`pyx_path` outside `common_root` can't be expressed as a
        relative path -- falls back to just the file's own name under
        `output_dir` rather than raising."""
        stubgen = StubgenPyx()
        result = stubgen._resolve_pyi_path(
            Path("/some/other/place/mod.pyx"), temp_dir, Path("/tmp/unrelated-root")
        )
        assert result == temp_dir / "mod.pyi"

    def test_no_common_root_places_output_directly_under_output_dir(self, temp_dir):
        """No `common_root` at all (a single-file batch, or `output_dir`
        without one computed) -- same bare-filename placement, taken
        directly rather than via the `relative_to`/fallback path."""
        stubgen = StubgenPyx()
        result = stubgen._resolve_pyi_path(Path("/some/place/mod.pyx"), temp_dir, None)
        assert result == temp_dir / "mod.pyi"


class TestCtypedefPruningBatchPathExtras:
    """The remaining, less-common corners of
    `_convert_multiple_files_with_ctypedef_pruning` specifically (as
    opposed to the plain per-file batch loop): re-raising a genuine
    parse failure with `continue_on_error=False`, and the `dry_run`
    info log.
    """

    def test_continue_on_error_false_reraises_from_the_prep_phase(self, temp_dir):
        (temp_dir / "bad.pyx").write_text("def f(x: int -> int:\n    pass\n")
        stubgen = StubgenPyx(
            StubgenPyxConfig(resolve_ctypedef_aliases=True, continue_on_error=False)
        )
        with pytest.raises(CythonErrors.CompileError):
            stubgen.convert_glob(str(temp_dir / "bad.pyx"))

    def test_dry_run_does_not_write_and_logs_would_create(self, temp_dir, caplog):
        import logging

        pyx_file = temp_dir / "mod.pyx"
        pyx_file.write_text("def f(x: int) -> int:\n    return x\n")
        stubgen = StubgenPyx(StubgenPyxConfig(resolve_ctypedef_aliases=True))
        with caplog.at_level(logging.INFO):
            results = stubgen.convert_glob(str(pyx_file), dry_run=True)
        assert results[0].success is True
        assert not (temp_dir / "mod.pyi").exists()
        assert "Would create output file" in caplog.text
