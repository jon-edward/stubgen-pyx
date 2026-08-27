"""Telemetry collection and aggregation."""

from .events import Event, EventKind, summarize

__all__ = ["Event", "EventKind", "summarize"]
