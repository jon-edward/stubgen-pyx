# stubgen-pyx

Automatic stub file generation for Cython extensions.

## Installation

```bash
pip install stubgen-pyx
```

## Usage

Generate stubs from the command line:

```bash
stubgen-pyx /path/to/package
```

Or use the Python API:

```python
import stubgen_pyx
stubgen_pyx.stubgen("/path/to/package")
```

## Overview

Cython is a widely-used extension language for Python, but type information for Cython modules is often unavailable to static analysis tools. This package generates `.pyi` stub files that expose type hints and signatures for Cython code, enabling better IDE support and type checking.

## Why not use mypy's stubgen?

While mypy's stubgen can generate stubs for compiled extension modules through runtime introspection, it cannot access Cython-specific metadata embedded in the compiled modules. This results in incomplete or inaccurate type information.

stubgen-pyx is designed specifically for Cython and leverages embedded metadata to produce more accurate and complete stub files.

### Example

A Cython module like this:

```cython
cdef class TestClass:
    """
    This is a class for testing stub file generation.
    """
    a: int

    def __init__(self):
        """
        A docstring for __init__
        """
        self.a = 1

    cpdef b(self):
        """
        A docstring for b
        """
        return self.a

    def c(self):
        """
        A docstring for c
        """
        return self.a

    cdef d(self):
        """
        A docstring for d (this should be ignored)
        """
        return self.a
```

Generates the following stub file:

```python
class TestClass:
    """
    This is a class for testing stub file generation.
    """
    def __init__(self) -> None:
        """
        A docstring for __init__
        """
        ...

    def b(self):
        """
        A docstring for b
        """
        ...

    def c(self):
        """
        A docstring for c
        """
        ...
```

Note that `cdef` methods (like `d`) are not included since they're not accessible from Python, and public attributes and `cpdef` methods are properly exposed.

## Limitations

**Import resolution for cimported types:** When types from `cimport`-ed modules appear in function signatures or class definitions, their imports are not automatically included in the generated stub file.

**Workaround:** Define a `__cimport_types__` list or tuple at the module level containing the types you want exposed:

```python
__cimport_types__ = [SomeType, AnotherType]
```

These types will then be properly imported in the generated stub file.
