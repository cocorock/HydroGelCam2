"""Tab 3 - filament fusion test (Habib et al.; Ingri2024 Eqs. 3 and 4).

A 0-90 degree grid is printed with the filament-to-filament distance swept from
1 to 5 mm, so pore size grows along the diagonal from bottom-left to top-right.
One representative pore per size class is measured.

    Dfr = (At - Aa) / At x 100 %      spreading of material into the pore
    Pr  = L^2 / (16 Aa)               referenced to an ideal square
    C   = 4 pi Aa / L^2               circularity

Note that Pr and C are algebraically the same measurement: C == pi / (4 Pr) for
any area and perimeter. Comparing them can never reveal a segmentation error, so
the quality checks used here are the pass-1/pass-2 area agreement and the pore's
solidity, both computed in `pipeline.grid`.
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

from app import config
from app.calib.geometry import Scale
from app.pipeline.common import preprocess
from app.pipeline.debug import DebugTrace
from app.pipeline.grid import Pore, detect_grid


def analyze(
    image: np.ndarray,
    roi_json: dict[str, Any] | None,
    params: dict[str, Any],
    scale: Scale,
    debug: bool = False,
) -> dict[str, Any]:
    n = int(params.get("grid_n", config.FUSION_GRID_N))
    fd_list = [float(v) for v in params.get("fd_mm", config.FUSION_FD_MM)]
    tolerance = float(params.get("diagonal_tolerance", config.FUSION_DIAGONAL_TOLERANCE))
    at_overrides = params.get("at_mm2") or {}

    trace = DebugTrace(enabled=debug)
    seg = preprocess(
        image, roi_json,
        padding=int(params.get("padding", config.ROI_PADDING_PX)),
        polarity=params.get("polarity", config.DEFAULT_POLARITY),
        manual_threshold=params.get("manual_threshold"),
        trace=trace,
    )

    grid = detect_grid(
        seg, n,
        final_kernel=int(params.get("final_kernel", config.FINAL_MORPH_KERNEL)),
    )

    # The diagonal runs from the ROI's bottom-left to its top-right corner, in
    # crop coordinates. Pore size grows along it.
    roi = seg.roi_in_crop
    p_start = (float(roi.x), float(roi.y + roi.h))            # bottom-left
    p_end = (float(roi.x + roi.w), float(roi.y))              # top-right

    spacing = _mean_spacing(grid.pores, n)
    max_distance = tolerance * spacing if spacing else float("inf")

    selected = _select_diagonal(grid.pores, n, p_start, p_end, max_distance)

    measurements: list[dict[str, Any]] = []
    flags: list[str] = []
    if not grid.complete:
        flags.append(grid.message)

    for k, pore in enumerate(selected):
        fd = fd_list[k] if k < len(fd_list) else (k + 1.0)
        at_default = fd * fd                       # At = FD^2, edge-to-edge pores
        at = float(at_overrides.get(str(k), at_overrides.get(k, at_default)))

        if pore is None:
            flags.append(f"No pore found for the {fd:g}x{fd:g} mm size class.")
            measurements.append(_row(k, fd, at, None, None, None, [], "missing"))
            continue

        if pore.closed:
            # Complete fusion at minimum spacing is a valid result, not a
            # detection failure: Aa = 0, and the shape metrics do not exist.
            measurements.append(
                _row(k, fd, at, pore, 0.0, 0.0, ["closed"], "closed")
            )
            continue

        area_mm2 = scale.area_mm2(pore.contour)
        perim_mm = scale.perimeter_mm(pore.contour)
        measurements.append(
            _row(k, fd, at, pore, area_mm2, perim_mm, list(pore.flags), "open")
        )

    results = compute(measurements)
    results["grid_complete"] = grid.complete
    results["grid_message"] = grid.message
    results["diagonal"] = {
        "x1": p_start[0] + seg.offset[0], "y1": p_start[1] + seg.offset[1],
        "x2": p_end[0] + seg.offset[0], "y2": p_end[1] + seg.offset[1],
    }
    results["mean_pore_spacing_px"] = spacing

    if trace.enabled:
        trace.add("9_selected", _overlay(seg, grid.pores, selected, p_start, p_end),
                  "Red = the reference diagonal. Yellow = pores selected as the "
                  "representative of each size class.")

    return {
        # The enriched rows, not the bare geometry: Dfr, Pr and C belong on each
        # measurement so a saved run carries its own results and the replicate
        # summary can aggregate them without re-deriving.
        "measurements": results["rows"],
        "results": results,
        "flags": flags,
        "roi": seg.roi.as_dict(),
        "threshold": seg.threshold,
        "polarity": seg.polarity,
        "offset": list(seg.offset),
        "debug": trace.to_json(),
    }


def compute(measurements: list[dict[str, Any]]) -> dict[str, Any]:
    """Formulas only, so tab 5 can re-score a stored run without re-segmenting."""
    rows: list[dict[str, Any]] = []
    for m in measurements:
        raw = dict(m.get("raw") or {})
        at = raw.get("at_mm2")
        aa = raw.get("aa_mm2")
        perim = raw.get("perimeter_mm")
        status = raw.get("status", "open")

        dfr = pr = circ = None
        if at and at > 0 and aa is not None:
            dfr = (at - aa) / at * 100.0
        if status == "open" and aa and aa > 0 and perim:
            pr = (perim ** 2) / (16.0 * aa)
            circ = (4.0 * math.pi * aa) / (perim ** 2)

        raw.update({"dfr_percent": dfr, "pr": pr, "circularity": circ})
        raw["pr_in_window"] = (
            None if pr is None
            else bool(config.PR_ACCEPT_LOW <= pr <= config.PR_ACCEPT_HIGH)
        )
        rows.append({**m, "raw": raw})

    included = [r["raw"] for r in rows if r.get("included")]
    open_rows = [r for r in included if r.get("status") == "open"]

    return {
        "rows": rows,
        "n_pores": len(rows),
        "n_open": len(open_rows),
        "n_closed": sum(1 for r in included if r.get("status") == "closed"),
        "mean_dfr_percent": _mean([r["dfr_percent"] for r in included]),
        "mean_pr": _mean([r["pr"] for r in open_rows]),
        "mean_circularity": _mean([r["circularity"] for r in open_rows]),
        "pr_window": [config.PR_ACCEPT_LOW, config.PR_ACCEPT_HIGH],
    }


# ---------------------------------------------------------------- internals


def _row(k: int, fd: float, at: float, pore: Pore | None,
         area: float | None, perim: float | None,
         flags: list[str], status: str) -> dict[str, Any]:
    centroid = pore.centroid if pore is not None else None
    return {
        "index_no": k,
        "label": f"{fd:g}x{fd:g} mm",
        "included": status != "missing",
        "raw": {
            "size_class": k + 1,
            "nominal_fd_mm": fd,
            "at_mm2": at,
            "aa_mm2": area,
            "perimeter_mm": perim,
            "status": status,
            "centroid": list(centroid) if centroid else None,
            "row": pore.row if pore else None,
            "col": pore.col if pore else None,
            "solidity": pore.solidity if pore else None,
            "flags": flags,
        },
    }


def _mean(values: list[float | None]) -> float | None:
    vals = [v for v in values if v is not None and math.isfinite(v)]
    return sum(vals) / len(vals) if vals else None


def _mean_spacing(pores: list[Pore], n: int) -> float:
    """Average centre-to-centre distance between neighbouring lattice positions."""
    by_cell = {(p.row, p.col): p.centroid for p in pores}
    gaps = []
    for (r, c), pt in by_cell.items():
        right = by_cell.get((r, c + 1))
        down = by_cell.get((r + 1, c))
        if right:
            gaps.append(math.hypot(right[0] - pt[0], right[1] - pt[1]))
        if down:
            gaps.append(math.hypot(down[0] - pt[0], down[1] - pt[1]))
    return float(np.mean(gaps)) if gaps else 0.0


def _select_diagonal(
    pores: list[Pore],
    n: int,
    p_start: tuple[float, float],
    p_end: tuple[float, float],
    max_distance: float,
) -> list[Pore | None]:
    """One pore per size class, taken along the bottom-left -> top-right diagonal.

    The lattice index gives the size class unambiguously (position (n-1-k, k) is
    the k-th pore along that diagonal). The perpendicular distance to the drawn
    line is then used as a sanity filter, so a badly placed ROI is rejected rather
    than silently measuring the wrong pores.
    """
    by_cell = {(p.row, p.col): p for p in pores}
    selected: list[Pore | None] = []
    for k in range(n):
        pore = by_cell.get((n - 1 - k, k))
        if pore is None:
            selected.append(None)
            continue
        distance = _point_line_distance(pore.centroid, p_start, p_end)
        if math.isfinite(max_distance) and distance > max_distance:
            pore.flags.append(f"off_diagonal({distance:.0f}px)")
        selected.append(pore)
    return selected


def _point_line_distance(point, a, b) -> float:
    px, py = point
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    denom = math.hypot(dx, dy)
    if denom == 0:
        return math.hypot(px - ax, py - ay)
    return abs(dy * px - dx * py + bx * ay - by * ax) / denom


def _overlay(seg, pores, selected, p_start, p_end) -> np.ndarray:
    vis = cv2.cvtColor(seg.flat, cv2.COLOR_GRAY2BGR)
    cv2.line(vis, tuple(int(v) for v in p_start), tuple(int(v) for v in p_end),
             (0, 0, 255), 2)
    for p in pores:
        if p.contour is not None:
            cv2.drawContours(vis, [p.contour], -1, (120, 120, 120), 1)
    for k, p in enumerate(selected):
        if p is None:
            continue
        if p.contour is not None:
            cv2.drawContours(vis, [p.contour], -1, (0, 220, 255), 2)
        cv2.putText(vis, f"{k + 1}",
                    (int(p.centroid[0]) - 6, int(p.centroid[1]) + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 2)
    return vis
