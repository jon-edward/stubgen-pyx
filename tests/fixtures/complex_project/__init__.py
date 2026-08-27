"""Public facade for the nested complex Cython fixture package."""

from .containers.numeric import Matrix, normalize
from .containers.numeric.vector_ops import normalize as normalize_vector
from .domains.serialization import (
    Record,
    decode,
    encode,
    make_decimal,
    make_integer,
    make_text,
    pack,
    unpack,
)
from .domains.telemetry import Event, EventKind, summarize
from .services import (
    Histogram,
    Job,
    JobState,
    RuntimeConfig,
    load_config,
    percentile,
    schedule,
)

__all__ = [
    "Event",
    "EventKind",
    "Histogram",
    "Job",
    "JobState",
    "Matrix",
    "Record",
    "RuntimeConfig",
    "decode",
    "encode",
    "load_config",
    "make_decimal",
    "make_integer",
    "make_text",
    "normalize",
    "normalize_vector",
    "pack",
    "percentile",
    "schedule",
    "summarize",
    "unpack",
]
