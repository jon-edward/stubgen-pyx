"""
API for stubgen-pyx, a tool for generating `.pyi` files from Cython modules.
"""

from stubgen_pyx.stubgen import stubgen
from stubgen_pyx.build import build
from stubgen_pyx.convert import convert_module


__all__ = ["stubgen", "build", "convert_module"]
