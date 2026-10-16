"""Validated request settings; independent of the collection crawler."""

import math
import os
from dataclasses import dataclass, fields
from typing import Mapping, Optional


@dataclass(frozen=True)
class Settings:
    request_delay: float = 1.0
    connect_timeout: float = 5.0
    read_timeout: float = 20.0
    concurrency: int = 1
    page_size: int = 50
    max_unique: int = 500
    max_page_requests: int = 100

    def __post_init__(self) -> None:
        for name in ("request_delay", "connect_timeout", "read_timeout"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
            if value < 0 or (name != "request_delay" and value == 0):
                raise ValueError(f"{name} is out of range")
        for name, upper in (("concurrency", 1), ("page_size", 50),
                            ("max_unique", 500), ("max_page_requests", 100)):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= upper:
                raise ValueError(f"{name} must be an integer in 1..{upper}")

    @classmethod
    def from_env(cls, environ: Optional[Mapping[str, str]] = None) -> "Settings":
        env = os.environ if environ is None else environ
        values = {}
        for field in fields(cls):
            key = f"PRICE_REFRESH_{field.name.upper()}"
            if key in env:
                convert = float if field.name in (
                    "request_delay", "connect_timeout", "read_timeout"
                ) else int
                try:
                    values[field.name] = convert(env[key])
                except (ValueError, TypeError) as error:
                    raise ValueError(f"Invalid {key}: {env[key]!r}") from error
        return cls(**values)
