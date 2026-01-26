# stubgen-pyx

**Generate Python stub files (.pyi) from Cython source code (.pyx/.pxd)**

Automatic stub file generation for Cython extensions that enables full IDE support and type checking for Cython modules.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Supported Cython Features](#supported-cython-features)
- [Configuration Options](#configuration-options)
- [Development](#development)
- [Contributing](#contributing)

## Features

**Comprehensive Cython Support**

- Extracts type information from Cython source files
- Preserves docstrings and function signatures
- Handles both `.pyx` and `.pxd` files
- Supports class hierarchies and inheritance

**Smart Processing**

- Automatic import trimming and deduplication
- Cython type normalization (e.g., `bint` → `bool`, `unicode` → `str`)
- Proper handling of positional-only and keyword-only arguments
- Preserves decorators and class metadata

## Installation

```bash
pip install stubgen-pyx
```

## Quick Start

Generate stubs for all Cython files in the current directory:

```bash
stubgen-pyx .
```

Or use the Python API:

```python
from stubgen_pyx import StubgenPyx
from pathlib import Path

stubgen = StubgenPyx()
results = stubgen.convert_glob("**/*.pyx")

for result in results:
    if result.success:
        print(f"Generated: {result.pyi_file}")
    else:
        print(f"Failed: {result.pyx_file}")
        print(f"  Error: {result.error}")
```

## Usage

### Command Line

**Basic usage:**

```bash
# Convert all .pyx files in current directory
stubgen-pyx .

# Convert files in a specific directory
stubgen-pyx /path/to/cython/package

# Use custom glob pattern
stubgen-pyx . --file "**/*.pyx"
```

**Advanced options:**

```bash
# Write .pyi files to a custom directory
stubgen-pyx . --output-dir stubs/

# Preview changes without writing
stubgen-pyx . --dry-run

# Enable verbose logging
stubgen-pyx . --verbose

# Continue processing even if some files fail
stubgen-pyx . --continue-on-error

# Customize postprocessing
stubgen-pyx . --no-sort-imports --no-trim-imports
```

**Disable specific transformations:**

```bash
# Disable import sorting
stubgen-pyx . --no-sort-imports

# Don't trim unused imports
stubgen-pyx . --no-trim-imports

# Don't normalize Cython types (keep bint, unicode, etc.)
stubgen-pyx . --no-normalize-names

# Exclude .pxd file contents
stubgen-pyx . --no-pxd-to-stubs

# Skip epilog comment
stubgen-pyx . --exclude-epilog
```

### Python API

**Basic conversion:**

```python
from stubgen_pyx import StubgenPyx
from pathlib import Path

stubgen = StubgenPyx()

# Convert a single file
pyx_code = Path("mymodule.pyx").read_text()
pyi_stub = stubgen.convert_str(pyx_code)
print(pyi_stub)
```

**Batch processing with results:**

```python
from stubgen_pyx import StubgenPyx

stubgen = StubgenPyx()
results = stubgen.convert_glob("src/**/*.pyx")

successful = sum(1 for r in results if r.success)
failed = sum(1 for r in results if not r.success)

print(f"Converted: {successful}/{len(results)} files")
if failed > 0:
    for result in results:
        if not result.success:
            print(f"  - {result.pyx_file}: {result.error}")
```

**Custom configuration:**

```python
from stubgen_pyx import StubgenPyx
from stubgen_pyx.config import StubgenPyxConfig

config = StubgenPyxConfig(
    no_trim_imports=False,      # Trim unused imports
    no_normalize_names=False,   # Normalize Cython types
    no_sort_imports=False,      # Sort imports
    continue_on_error=True,     # Don't stop on first error
    verbose=True,               # Detailed output
)

stubgen = StubgenPyx(config=config)
results = stubgen.convert_glob("**/*.pyx")
```

## How It Works

stubgen-pyx works in several stages:

1. **Parsing**: Cython source code is parsed using the Cython compiler's AST
2. **Analysis**: The AST is visited to extract type information, signatures, and docstrings
3. **Conversion**: Cython-specific constructs are converted to intermediate PyiElements
4. **Building**: PyiElements are transformed into Python stub code
5. **Postprocessing**: Generated code is optimized (imports trimmed, types normalized, etc.)

### Why stubgen-pyx vs mypy's stubgen?

While mypy's `stubgen` can generate stubs for compiled extension modules through runtime introspection, it cannot access Cython-specific metadata embedded in the source code. This results in:

- Missing type annotations
- Incomplete function signatures
- No support for Cython-specific constructs (cdef classes, memory views)

**stubgen-pyx** directly analyzes the Cython source, providing:

- Complete type information
- Accurate function signatures with annotations
- Preserved docstrings and decorators

## Supported Cython Features

### Supported

- Python functions (`def`)
- C functions (`cdef`)
- C/Python functions (`cpdef`)
- Classes (both Python `class` and Cython `cdef class`)
- Class inheritance and metaclasses
- Type annotations on arguments and return values
- Docstrings
- Decorators
- Default arguments
- \*args and \*\*kwargs
- Keyword-only arguments
- Positional-only arguments
- Cython enums (`cdef enum`)
- Import statements (including `cimport`)
- Public attributes and properties

### Limitations

- Memory views require manual type hints in some cases
- Fused types (generics) have basic support

## Configuration Options

All configuration is handled through the `StubgenPyxConfig` dataclass:

| Option                   | Type | Default | Description                                         |
| ------------------------ | ---- | ------- | --------------------------------------------------- |
| `no_sort_imports`        | bool | False   | Skip sorting imports                                |
| `no_trim_imports`        | bool | False   | Skip trimming unused imports                        |
| `no_pxd_to_stubs`        | bool | False   | Skip including .pxd file contents                   |
| `no_normalize_names`     | bool | False   | Skip normalizing Cython types to Python equivalents |
| `no_deduplicate_imports` | bool | False   | Skip deduplicating imports                          |
| `exclude_epilog`         | bool | False   | Skip adding generation epilog comment               |
| `continue_on_error`      | bool | False   | Continue processing even if a file fails            |
| `verbose`                | bool | False   | Enable verbose logging output                       |

## Example

### Input: Cython module

```cython
# math_utils.pyx
"""Mathematical utilities for scientific computing."""

cdef class Matrix:
    """A simple matrix class."""

    cdef int rows
    cdef int cols

    def __init__(self, int rows, int cols):
        """Initialize a matrix."""
        self.rows = rows
        self.cols = cols

    def shape(self) -> tuple[int, int]:
        """Get matrix dimensions."""
        return (self.rows, self.cols)

    cpdef scale(self, double factor):
        """Scale all elements."""
        pass

    cdef int _validate(self):
        """Internal validation (not exposed)."""
        return 0

def matrix_product(Matrix a, Matrix b) -> Matrix:
    """Compute matrix product."""
    return Matrix(a.rows, b.cols)
```

### Output: Generated stub

```python
# math_utils.pyi
"""Mathematical utilities for scientific computing."""

from __future__ import annotations

class Matrix:
    """A simple matrix class."""

    def __init__(self, rows: int, cols: int):
        """Initialize a matrix."""

    def shape(self) -> tuple[int, int]:
        """Get matrix dimensions."""

    def scale(self, factor: float):
        """Scale all elements."""

def matrix_product(a: Matrix, b: Matrix) -> Matrix:
    """Compute matrix product."""

# This file was generated by stubgen-pyx v0.2.0 from math_utils.pyx
```

Note:

- Public methods are included
- Type annotations are preserved
- Docstrings are preserved
- Private `cdef` method `_validate` is excluded
- Private `cdef` attributes are excluded

## Development

### Setup

```bash
# Clone the repository
git clone https://github.com/jon-edward/stubgen-pyx.git
cd stubgen-pyx

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode with test dependencies
pip install -e ".[test]"
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_config.py
```

### Project Structure

```
stubgen-pyx/
├── stubgen_pyx/              # Main package
│   ├── analysis/             # AST analysis (visitors)
│   ├── builders/             # Code generation
│   ├── conversion/           # AST conversion
│   ├── models/               # Data models (PyiElements)
│   ├── parsing/              # Cython parser
│   ├── postprocessing/       # Output optimization
│   ├── cli.py                # Command-line interface
│   ├── config.py             # Configuration
│   └── stubgen.py            # Main entry point
├── tests/                    # Test suite
├── README.md                 # This file
└── pyproject.toml            # Project metadata
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Write tests for new functionality
4. Ensure all tests pass (`pytest`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to your fork (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## License

See LICENSE file for details.

## Acknowledgments

- Built on top of the [Cython](https://cython.readthedocs.io/) compiler infrastructure
- Inspired by [mypy's stubgen](https://mypy.readthedocs.io/)
