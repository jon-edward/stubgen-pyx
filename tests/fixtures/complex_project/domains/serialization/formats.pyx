"""Binary format helpers backed by C structs and a tagged union."""

from __future__ import annotations

from typing import Literal, TypedDict

include "formats.pxi"


class Record(TypedDict):
    version: int
    flags: int
    kind: Literal["integer", "decimal", "text"]
    value: int | float | str


def make_integer(value: int, flags: int = 0) -> Record:
    return {"version": CURRENT_VERSION, "flags": flags, "kind": "integer", "value": value}


def make_decimal(value: float, flags: int = 0) -> Record:
    return {"version": CURRENT_VERSION, "flags": flags, "kind": "decimal", "value": value}


def make_text(value: str, flags: int = 0) -> Record:
    return {"version": CURRENT_VERSION, "flags": flags, "kind": "text", "value": value}


def pack(record: Record) -> bytes:
    body = str(record["value"]).encode("utf-8")
    checked_length(body)
    return bytes((record["version"], record["flags"])) + body


def unpack(payload: bytes) -> Record:
    if len(payload) < 2:
        raise ValueError("packet is too short")
    body = payload[2:].decode("utf-8")
    return {"version": payload[0], "flags": payload[1], "kind": "text", "value": body}
