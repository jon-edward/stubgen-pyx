import ast
from pathlib import Path

from ..config import StubgenPyxConfig
from .collect_names import collect_names
from .normalize_names import normalize_names
from .sort_imports import sort_imports
from .deduplicate_imports import deduplicate_imports
from .trim_imports import trim_imports
from .epilog import epilog


def postprocessing_pipeline(pyi_code: str, config: StubgenPyxConfig, pyx_path: Path | None = None) -> str:
    pyi_ast = ast.parse(pyi_code)
    
    if not config.no_trim_imports:
        used_names = collect_names(pyi_ast)
        pyi_ast = trim_imports(pyi_ast, used_names)
    
    if not config.no_normalize_names:
        pyi_ast = normalize_names(pyi_ast)
    
    if not config.no_deduplicate_imports:
        pyi_ast = deduplicate_imports(pyi_ast)
    
    pyi_code = ast.unparse(pyi_ast)

    if not config.no_sort_imports:
        pyi_code = sort_imports(pyi_code)

    if not config.exclude_epilog:
        pyi_code = f"{pyi_code}\n\n{epilog(pyx_path)}"

    return pyi_code
