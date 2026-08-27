"""Typed entry point exercising the public complex_project API."""

from __future__ import annotations

from collections.abc import Iterable

from complex_project.containers.numeric import Matrix
from complex_project.containers.numeric.vector_ops import Vector, normalize
from complex_project.domains.serialization.formats import (
    Record,
    make_decimal,
    make_integer,
    make_text,
    pack,
    unpack,
)
from complex_project.domains.serialization.wire import decode, encode, log_wire_value
from complex_project.domains.telemetry.events import Event, EventKind, summarize
from complex_project.services.analytics import Histogram, percentile
from complex_project.services.config import RuntimeConfig, load_config
from complex_project.services.scheduler import Job, schedule


def build_records() -> list[Record]:
    return [make_integer(7), make_decimal(2.5), make_text("ready", flags=1)]


def inspect_events(names: Iterable[str]) -> tuple[int, dict[str, int]]:
    events: list[Event] = [
        Event(100 + index, name, EventKind.EVENT_START)
        for index, name in enumerate(names)
    ]
    return len(events), summarize(events)


def main() -> dict[str, object]:
    config: RuntimeConfig = load_config("demo", {"timeout": 2.5, "region": "test"})
    configured = config.with_option("environment", "smoke")
    matrix = Matrix(2, 2, [1.0, 2.0, 3.0, 4.0])
    matrix_rows: list[tuple[float, ...]] = list(matrix)
    histogram = Histogram(bucket_count=4, width=1.0)
    for value in (0.5, 1.5, 1.7):
        histogram.add(value)
    jobs = [Job(1, "warmup", scheduled_at=0), Job(2, "refresh", scheduled_at=10)]
    job_results: dict[str, object] = {}
    for job in jobs:
        if job.ready(1):

            def f(job_ident: int = job.identifier) -> str:
                return f"completed:{job_ident}"

            job_results[job.name] = job.run(f)
    ready_jobs = list(schedule(jobs, now=1))
    records: list[Record] = build_records()
    packets = [pack(record) for record in records]
    span, event_counts = inspect_events(("start", "finish", "finish"))
    wire_values: list[int | float | str] = [
        log_wire_value(11),
        log_wire_value(2.5),
        log_wire_value("payload"),
    ]
    normalized = Vector(normalize((1.0, 2.0, 3.0)))
    normalized.vector = (4.0, 5.0, 6.0)

    return {
        "config": config.name,
        "environment": configured.option("environment"),
        "matrix_item": matrix.item(1, 1),
        "matrix_rows": matrix_rows,
        "normalized": Vector(normalize((1.0, 2.0, 3.0))),
        "percentile": percentile([1.0, 4.0, 9.0], 0.5),
        "histogram": histogram.counts(),
        "ready_jobs": [job.name for job in ready_jobs],
        "job_results": job_results,
        "packets": [decode(packet) for packet in packets],
        "records": [unpack(packet) for packet in packets],
        "span": span,
        "events": event_counts,
        "wire": encode("complete"),
        "wire_values": wire_values,
    }


if __name__ == "__main__":
    main()
