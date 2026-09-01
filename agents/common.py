"""Shared helpers for specialized agents."""
from __future__ import annotations

from datetime import datetime
from time import perf_counter

import yaml


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def elapsed_ms(start: float) -> int:
    return round((perf_counter() - start) * 1000)


def load_rules(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)
