"""Small 1-D helpers for histogram and projection analysis.

Pure numpy on purpose. These operate on 256-bin histograms and on row/column
projections a few thousand samples long, so SciPy would buy nothing but a
dependency that has to stay version-matched to numpy.
"""

from __future__ import annotations

import numpy as np


def gaussian_smooth(data: np.ndarray, sigma: float) -> np.ndarray:
    """Convolve with a Gaussian, edges extended (equivalent to mode='nearest')."""
    sigma = float(sigma)
    if sigma <= 0:
        return np.asarray(data, dtype=np.float64)
    radius = max(1, int(round(4.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(x ** 2) / (2.0 * sigma ** 2))
    kernel /= kernel.sum()
    padded = np.pad(np.asarray(data, dtype=np.float64), radius, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def find_peaks(data: np.ndarray, prominence: float = 0.0) -> np.ndarray:
    """Indices of local maxima whose prominence exceeds the threshold.

    Prominence is measured the standard way: the peak's height above the higher
    of the two lowest points reached before hitting a taller peak on either side.
    """
    y = np.asarray(data, dtype=np.float64)
    if y.size < 3:
        return np.empty(0, dtype=int)

    # Local maxima, with plateaus reduced to their midpoint.
    candidates: list[int] = []
    i = 1
    while i < y.size - 1:
        if y[i] > y[i - 1]:
            j = i
            while j < y.size - 1 and y[j + 1] == y[i]:
                j += 1
            if j < y.size - 1 and y[j] > y[j + 1]:
                candidates.append((i + j) // 2)
            i = j + 1
        else:
            i += 1

    if not candidates or prominence <= 0:
        return np.asarray(candidates, dtype=int)

    keep = [p for p in candidates if _prominence(y, p) >= prominence]
    return np.asarray(keep, dtype=int)


def _prominence(y: np.ndarray, peak: int) -> float:
    height = y[peak]

    left_min = height
    i = peak - 1
    while i >= 0 and y[i] <= height:
        left_min = min(left_min, y[i])
        i -= 1

    right_min = height
    i = peak + 1
    while i < y.size and y[i] <= height:
        right_min = min(right_min, y[i])
        i += 1

    return float(height - max(left_min, right_min))


def gradient(data: np.ndarray) -> np.ndarray:
    return np.gradient(np.asarray(data, dtype=np.float64))


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous True stretches of a boolean array, as (start, length)."""
    flags = np.asarray(mask).astype(bool).astype(np.int8)
    if flags.size == 0:
        return []
    edges = np.diff(np.concatenate(([0], flags, [0])))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return [(int(s), int(e - s)) for s, e in zip(starts, ends)]
