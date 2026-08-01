"""Descriptive statistics shared by the analysis modules and the summary export."""

from __future__ import annotations

import math
from typing import Any, Sequence


def describe(values: Sequence[float]) -> dict[str, Any]:
    """Mean and sample standard deviation (n-1), the convention the papers use."""
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    n = len(vals)
    if n == 0:
        return {"n": 0, "mean": None, "sd": None, "cv": None}
    mean = sum(vals) / n
    if n < 2:
        return {"n": n, "mean": mean, "sd": None, "cv": None}
    variance = sum((v - mean) ** 2 for v in vals) / (n - 1)
    sd = math.sqrt(variance)
    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "cv": (sd / mean) if mean else None,
    }


def fmt(value: float | None, digits: int = 3) -> str:
    if value is None or not math.isfinite(value):
        return "N/A"
    return f"{value:.{digits}f}"
