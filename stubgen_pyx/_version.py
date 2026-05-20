import importlib.metadata

try:
    __version__: str = importlib.metadata.version(__package__)  # type: ignore
except importlib.metadata.PackageNotFoundError:
    # package is not installed
    __version__ = "Unknown"
