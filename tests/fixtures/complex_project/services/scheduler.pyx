"""A small extension-backed job scheduler."""

from __future__ import annotations

from typing import Callable, Iterable, Iterator

include "scheduler.pxi"


cdef class Job:
    cdef readonly unsigned long long identifier
    cdef public JobState state
    cdef readonly str name
    cdef readonly unsigned long long scheduled_at

    def __init__(self, identifier: int, name: str, scheduled_at: int = 0) -> None:
        self.identifier = identifier
        self.name = name
        self.scheduled_at = scheduled_at
        self.state = JOB_PENDING

    cpdef bint ready(self, unsigned long long now):
        return is_due(self.scheduled_at, now) and self.state == JOB_PENDING

    def run(self, callback: Callable[[], object]) -> object:
        self.state = JOB_RUNNING
        try:
            result = callback()
        except Exception:
            self.state = JOB_FAILED
            raise
        self.state = JOB_COMPLETE
        return result


def schedule(jobs: Iterable[Job], now: int) -> Iterator[Job]:
    return (job for job in jobs if job.ready(now))
