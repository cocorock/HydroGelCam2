"""Tab 2 - filament uniformity test.

A serpentine of N horizontal filaments is sampled at M equally spaced x-positions
(the same positions for every filament). Each sample is the local filament
thickness, measured perpendicular to the long axis -- which, since the filaments
run horizontally, is simply vertical.

    UI = 1 - CV = 1 - SD/D_bar        precision: how consistent the filament is
    SR = D_bar / D_n                  accuracy:  how close to the nozzle bore

A filament can be perfectly uniform and uniformly too wide, which is why both are
reported.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from app import config
from app.analysis.stats import describe
from app.calib.geometry import Scale
from app.pipeline.common import Segmentation, clean_mask, preprocess
from app.pipeline.debug import DebugTrace
from app.pipeline.signal1d import runs


def analyze(
    image: np.ndarray,
    roi_json: dict[str, Any] | None,
    params: dict[str, Any],
    scale: Scale,
    debug: bool = False,
) -> dict[str, Any]:
    n_filaments = int(params.get("n_filaments", config.UNIFORMITY_FILAMENTS))
    n_positions = int(params.get("n_positions", config.UNIFORMITY_POSITIONS))
    inset = float(params.get("edge_inset", config.UNIFORMITY_EDGE_INSET))
    nozzle_mm = params.get("nozzle_id_mm")
    final_kernel = int(params.get("final_kernel", config.FINAL_MORPH_KERNEL))

    trace = DebugTrace(enabled=debug)
    seg = preprocess(
        image, roi_json,
        padding=int(params.get("padding", config.ROI_PADDING_PX)),
        polarity=params.get("polarity", config.DEFAULT_POLARITY),
        manual_threshold=params.get("manual_threshold"),
        trace=trace,
    )

    mask = clean_mask(seg.binary, final_kernel, trace,
                      name="7_clean_filaments")

    roi = seg.roi_in_crop
    x0 = max(0, roi.x)
    x1 = min(mask.shape[1], roi.x + roi.w)
    y0 = max(0, roi.y)
    y1 = min(mask.shape[0], roi.y + roi.h)

    bands, band_msg = _find_bands(mask, x0, x1, y0, y1, n_filaments, trace, seg)

    # Step 4: the same M x-positions for every filament, inset from the ROI edges.
    span = x1 - x0
    pad = int(round(span * inset))
    xs = np.linspace(x0 + pad, x1 - pad - 1, n_positions)
    xs = [int(round(v)) for v in xs]

    measurements: list[dict[str, Any]] = []
    flags: list[str] = []
    index = 0
    for b_i, band in enumerate(bands):
        for x in xs:
            edges = _measure_edge(seg.work, mask, x, band, seg.threshold)
            if edges is None:
                flags.append(f"filament {b_i + 1} has no material at x={x}")
                measurements.append({
                    "index_no": index,
                    "label": f"F{b_i + 1}@P{xs.index(x) + 1}",
                    "included": False,
                    "raw": {
                        "filament": b_i + 1,
                        "position": xs.index(x) + 1,
                        "x_crop": x,
                        "x_full": x + seg.offset[0],
                        "thickness_px": None,
                        "thickness_mm": None,
                        "missing": True,
                    },
                })
                index += 1
                continue

            top, bottom = edges
            thickness_px = bottom - top
            thickness_mm = scale.dy_mm(thickness_px)
            measurements.append({
                "index_no": index,
                "label": f"F{b_i + 1}@P{xs.index(x) + 1}",
                "included": True,
                "raw": {
                    "filament": b_i + 1,
                    "position": xs.index(x) + 1,
                    "x_crop": x,
                    "x_full": x + seg.offset[0],
                    "top_crop": top,
                    "bottom_crop": bottom,
                    "top_full": top + seg.offset[1],
                    "bottom_full": bottom + seg.offset[1],
                    "thickness_px": thickness_px,
                    "thickness_mm": thickness_mm,
                    "missing": False,
                },
            })
            index += 1

    # Continuity: a serpentine that broke cannot be assessed for uniformity.
    continuity = _check_continuity(mask, bands, x0, x1)
    if not continuity["continuous"]:
        flags.append(
            "discontinuous filament - fails at this stage "
            f"({continuity['broken_count']} of {len(bands)} filaments have gaps)"
        )

    if band_msg:
        flags.append(band_msg)

    results = compute(measurements, nozzle_mm)
    results["continuity"] = continuity
    results["x_positions_full"] = [x + seg.offset[0] for x in xs]
    results["bands"] = [
        {"index": i + 1, "y_full": int((b[0] + b[1]) / 2 + seg.offset[1]),
         "y0_full": int(b[0] + seg.offset[1]), "y1_full": int(b[1] + seg.offset[1])}
        for i, b in enumerate(bands)
    ]

    if trace.enabled:
        trace.add("9_measurements", _overlay(seg, bands, xs, measurements),
                  "Green ticks are the measured thicknesses; each is clickable in "
                  "the results list.")

    return {
        "measurements": measurements,
        "results": results,
        "flags": flags,
        "roi": seg.roi.as_dict(),
        "threshold": seg.threshold,
        "polarity": seg.polarity,
        "offset": list(seg.offset),
        "debug": trace.to_json(),
    }


def compute(measurements: list[dict[str, Any]],
            nozzle_mm: float | None) -> dict[str, Any]:
    """Formulas only -- no image processing.

    Tab 5 calls this directly when a measurement is checked or unchecked, so a
    stored run is re-scored without touching the photograph.
    """
    included = [
        m for m in measurements
        if m.get("included") and (m.get("raw") or {}).get("thickness_mm") is not None
    ]
    values = [float(m["raw"]["thickness_mm"]) for m in included]

    d = describe(values)
    n = d["n"]
    mean = d["mean"]
    sd = d["sd"]
    cv = d["cv"]

    ui = (1.0 - cv) if cv is not None else None
    sr = None
    if mean is not None and nozzle_mm:
        try:
            sr = mean / float(nozzle_mm)
        except (TypeError, ZeroDivisionError):
            sr = None

    return {
        "n_included": n,
        "n_total": len(measurements),
        "mean_mm": mean,
        "mean_um": mean * 1000.0 if mean is not None else None,
        "sd_mm": sd,
        "sd_um": sd * 1000.0 if sd is not None else None,
        "cv": cv,
        "cv_percent": cv * 100.0 if cv is not None else None,
        "uniformity_index": ui,
        "ui_valid": bool(cv is not None and cv < 1.0),
        "spreading_ratio": sr,
        "nozzle_id_mm": float(nozzle_mm) if nozzle_mm else None,
    }


# ---------------------------------------------------------------- internals


def _find_bands(
    mask: np.ndarray,
    x0: int, x1: int, y0: int, y1: int,
    n_filaments: int,
    trace: DebugTrace,
    seg: Segmentation,
) -> tuple[list[tuple[int, int]], str]:
    """Locate the horizontal filament bands, ordered top to bottom."""
    strip = mask[:, x0:x1]
    coverage = (strip > 0).sum(axis=1) / max(x1 - x0, 1)

    # A row belongs to a filament when most of the ROI width is material there.
    band_rows = coverage >= 0.5
    band_rows[:y0] = False
    band_rows[y1:] = False

    found = [(s, s + ln - 1) for s, ln in runs(band_rows)]
    message = ""

    if len(found) != n_filaments:
        # Keep the N thickest runs -- stray specks and the serpentine's curved
        # turns at the ends can otherwise add spurious bands.
        found_sorted = sorted(found, key=lambda b: b[1] - b[0], reverse=True)
        kept = sorted(found_sorted[:n_filaments], key=lambda b: b[0])
        message = (
            f"Expected {n_filaments} filaments, found {len(found)}. "
            f"Using the {len(kept)} thickest. Check the ROI covers only the "
            "straight runs, not the curved turns."
        )
        found = kept

    if trace.enabled:
        vis = cv2.cvtColor(seg.flat, cv2.COLOR_GRAY2BGR)
        for i, (a, b) in enumerate(found):
            cv2.rectangle(vis, (x0, a), (x1, b), (0, 255, 0), 1)
            cv2.putText(vis, f"F{i + 1}", (x0 - 30, (a + b) // 2 + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        trace.add("8_bands", vis,
                  f"{len(found)} filament bands detected inside the ROI, "
                  "ordered top to bottom.")

    return found, message


def _measure_edge(
    work: np.ndarray,
    mask: np.ndarray,
    x: int,
    band: tuple[int, int],
    threshold: float,
) -> tuple[float, float] | None:
    """Sub-pixel upper and lower filament/background boundary at column x.

    Two different jobs, deliberately kept separate. The step-4 threshold decides
    *which* stretch of the column is filament -- that is a segmentation question.
    Where the boundary actually lies is then measured at the half-height between
    the local background and the local filament plateau, the standard definition
    of an edge position.

    Refining at the segmentation threshold instead would bias every width: that
    threshold sits near the top of the intensity ramp, so it cuts inside the true
    boundary and under-reports thickness by roughly one blur width. UI would not
    notice (the bias is common to all samples) but the spreading ratio would,
    reporting shrinkage that is not there.
    """
    a, b = band
    pad = max(4, (b - a))
    lo = max(0, a - pad)
    hi = min(mask.shape[0] - 1, b + pad)

    column_mask = mask[lo:hi + 1, x] > 0
    stretches = runs(column_mask)
    if not stretches:
        return None

    # The stretch overlapping the band centre is this filament.
    centre = (a + b) / 2 - lo
    start, length = min(
        stretches, key=lambda s: abs((s[0] + s[1] / 2) - centre)
    )
    top_i = lo + start
    bottom_i = lo + start + length - 1

    profile = work[:, x].astype(np.float64)

    plateau = float(np.median(profile[top_i:bottom_i + 1]))
    background = _local_background(profile, top_i, bottom_i, lo, hi)
    if plateau <= background:
        return None
    level = (plateau + background) / 2.0

    top = _interp_cross(profile, top_i, -1, level)
    bottom = _interp_cross(profile, bottom_i, +1, level)
    return top, bottom


def _local_background(profile: np.ndarray, top_i: int, bottom_i: int,
                      lo: int, hi: int) -> float:
    """Background level just above and below this filament."""
    above = profile[lo:top_i]
    below = profile[bottom_i + 1:hi + 1]
    samples = np.concatenate([above, below]) if above.size or below.size else None
    if samples is None or samples.size == 0:
        return float(profile.min())
    return float(np.median(samples))


def _interp_cross(profile: np.ndarray, index: int, direction: int,
                  level: float) -> float:
    """Where the intensity profile crosses `level`, searching outward from `index`.

    Walks out until the profile drops below the level, then interpolates linearly
    between the bracketing samples.
    """
    n = profile.size
    i = index
    limit = 0 if direction < 0 else n - 1
    steps = 0
    max_steps = 64

    while steps < max_steps:
        j = i + direction
        if (direction < 0 and j < limit) or (direction > 0 and j > limit):
            return float(i)
        inside, outside = profile[i], profile[j]
        if outside <= level <= inside and inside != outside:
            t = (inside - level) / (inside - outside)
            return float(i) + direction * float(np.clip(t, 0.0, 1.0))
        if outside > inside and steps > 0:
            # Climbing again: we have left this filament without crossing.
            return float(i)
        i = j
        steps += 1
    return float(i)


def _check_continuity(mask: np.ndarray, bands: list[tuple[int, int]],
                      x0: int, x1: int) -> dict[str, Any]:
    """Does every filament span the ROI without a break?"""
    broken = []
    for i, (a, b) in enumerate(bands):
        strip = mask[a:b + 1, x0:x1] > 0
        present = strip.any(axis=0)
        gaps = [ln for _, ln in runs(~present) if ln > 3]
        if gaps:
            broken.append({"filament": i + 1, "largest_gap_px": int(max(gaps))})
    return {
        "continuous": not broken,
        "broken_count": len(broken),
        "broken": broken,
    }


def _overlay(seg: Segmentation, bands, xs, measurements) -> np.ndarray:
    vis = cv2.cvtColor(seg.flat, cv2.COLOR_GRAY2BGR)
    for m in measurements:
        raw = m["raw"]
        if raw.get("thickness_px") is None:
            continue
        x = raw["x_crop"]
        top = int(round(raw["top_crop"]))
        bottom = int(round(raw["bottom_crop"]))
        color = (0, 255, 0) if m["included"] else (140, 140, 140)
        cv2.line(vis, (x, top), (x, bottom), color, 2)
        cv2.line(vis, (x - 5, top), (x + 5, top), color, 1)
        cv2.line(vis, (x - 5, bottom), (x + 5, bottom), color, 1)
    return vis
