import itertools
import pathlib
import shutil
import subprocess

THIS_DIR = pathlib.Path(__file__).parent.resolve()

def test_smoke():
    """Smoke test to ensure stubgen-pyx runs without errors on sample Cython files."""
    try:
        result = subprocess.run(
            ["stubgen-pyx", "fixtures"],
            cwd=THIS_DIR,
        )
        
        assert result.returncode == 0

        for pyx_file in THIS_DIR.joinpath("fixtures").glob("*.pyx"):
            pyi_file = pyx_file.with_suffix(".pyi")
            assert pyi_file.exists(), f"Expected .pyi file not found: {pyi_file}"
    finally:
        # Clean up generated .pyi files and compiled extensions
        to_delete = itertools.chain(
            THIS_DIR.joinpath("fixtures").glob("*.c"),
            THIS_DIR.joinpath("fixtures").glob("*.pyi"),
            THIS_DIR.joinpath("fixtures").glob("*.pyd"),
            THIS_DIR.joinpath("fixtures").glob("*.so"),
        )
        
        for file in to_delete:
            file.unlink(missing_ok=True)
        
        # Remove __pycache__ and build directories if they exist
        shutil.rmtree(THIS_DIR.joinpath("__pycache__"), ignore_errors=True)
        shutil.rmtree(THIS_DIR.joinpath("build"), ignore_errors=True)
