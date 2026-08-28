"""An event model combining enums, extension classes, and iterables."""

from __future__ import annotations

from typing import Iterable

include "events.pxi"

cdef class Event:
    cdef readonly unsigned long long timestamp
    cdef readonly str name
    cdef public EventKind kind

    def __init__(self, timestamp: int, name: str, kind: EventKind = EventKind.EVENT_START) -> None:
        self.timestamp = timestamp
        self.name = name
        self.kind = kind

    cpdef str format(self):
        return f"{self.timestamp}:{self.name}:{self.kind}"

    @property
    def age(self) -> int:
        return timestamp_delta(self.timestamp, self.timestamp + 1)


def summarize(events: Iterable[Event]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for event in events:
        counts[event.name] = counts.get(event.name, 0) + 1
    return counts
