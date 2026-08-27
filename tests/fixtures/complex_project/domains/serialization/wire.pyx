"""A deliberately small wire-format API with typed overload-like values."""

from __future__ import annotations

from typing import Any, Mapping

include "wire.pxi"


def encode(value: int | float | str, metadata: Mapping[str, str] | None = None) -> bytes:
    body = str(value).encode("utf-8")
    if metadata:
        body += b";" + b";".join(f"{key}={item}".encode() for key, item in metadata.items())
    return _versioned(body)


def decode(payload: bytes) -> dict[str, Any]:
    if not payload or payload[0] != WIRE_VERSION:
        raise ValueError("unsupported wire version")
    return {"version": payload[0], "body": payload[1:].decode("utf-8")}


# Function using WireValue
cpdef WireValue log_wire_value(WireValue payload):
    print(payload)
    return payload
