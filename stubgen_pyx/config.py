from dataclasses import dataclass


@dataclass
class StubgenPyxConfig:
    """
    Configuration for stubgen_pyx.
    """

    no_sort_imports: bool = False
    no_trim_imports: bool = False
    no_pxd_to_stubs: bool = False
    no_normalize_names: bool = False
    no_deduplicate_imports: bool = False
    exclude_epilog: bool = False
    continue_on_error: bool = False
