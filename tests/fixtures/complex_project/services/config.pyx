"""Configuration parsing with typed defaults and mapping protocols."""

from __future__ import annotations

from typing import Mapping


DEFAULT_TIMEOUT: float = 2.5


class RuntimeConfig:
    name: str
    options: dict[str, str]
    timeout: float

    def __init__(
        self,
        name: str,
        options: Mapping[str, str] | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self.name = name
        self.options = dict(options or {})
        self.timeout = timeout

    def option(self, key: str, fallback: str | None = None) -> str | None:
        return self.options.get(key, fallback)

    def with_option(self, key: str, value: str) -> RuntimeConfig:
        updated = dict(self.options)
        updated[key] = value
        return RuntimeConfig(self.name, updated, self.timeout)


def load_config(name: str, source: Mapping[str, object]) -> RuntimeConfig:
    raw_options = source.get("options", {})
    options = raw_options if isinstance(raw_options, Mapping) else {}
    timeout = source.get("timeout", DEFAULT_TIMEOUT)
    return RuntimeConfig(name, {str(key): str(value) for key, value in options.items()}, float(timeout))
