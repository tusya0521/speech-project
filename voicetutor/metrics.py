"""Per-stage latency timing, appended to a JSONL file (one line per turn)."""
import json
import math
import os
import time
from contextlib import contextmanager

from . import config


class Timer:
    """Collects named stage durations in milliseconds for one turn."""

    def __init__(self):
        self.stages: dict[str, float] = {}
        self._t0 = time.perf_counter()

    @contextmanager
    def stage(self, name: str):
        t = time.perf_counter()
        try:
            yield
        finally:
            self.stages[name] = self.stages.get(name, 0.0) + (time.perf_counter() - t) * 1000

    def total_ms(self) -> float:
        return (time.perf_counter() - self._t0) * 1000

    def record(self, **extra) -> dict:
        row = {"ts": time.time(), **{k: round(v, 1) for k, v in self.stages.items()},
               "total": round(self.total_ms(), 1), **extra}
        try:
            os.makedirs(os.path.dirname(config.METRICS_PATH) or ".", exist_ok=True)
            with open(config.METRICS_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
        except OSError:
            pass  # logging must never break a turn
        return row


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile; returns nan for an empty list."""
    if not values:
        return float("nan")
    s = sorted(values)
    return s[max(0, math.ceil(p / 100 * len(s)) - 1)]
