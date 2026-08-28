"""Streaming statistics and a typed histogram extension class."""

from __future__ import annotations

from typing import Iterable, Sequence

include "analytics.pxi"


cdef class Histogram:
    cdef dict _bins
    cdef readonly int bucket_count
    cdef readonly double width

    def __init__(self, bucket_count: int = DEFAULT_BUCKETS, width: float = 1.0) -> None:
        if bucket_count <= 0 or width <= 0:
            raise ValueError("histogram dimensions must be positive")
        self.bucket_count = bucket_count
        self.width = width
        self._bins = {}

    cpdef void add(self, double value):
        bucket = bucket_for(value, self.width)
        self._bins[bucket] = self._bins.get(bucket, 0) + 1

    cpdef dict counts(self):
        return dict(self._bins)


def percentile(values: Sequence[float], fraction: float) -> float:
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be between zero and one")
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot calculate percentile of an empty sequence")
    index = min(int(fraction * (len(ordered) - 1)), len(ordered) - 1)
    return ordered[index]
