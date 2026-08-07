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


def theoretical_area(arista_mm: float, filament_d_mm: float) -> float | None:
    """Open area of a square pore bounded by filaments a apart: (a - d)^2.

    None when the filament is as wide as the spacing, because there is no pore
    to speak of. A non-positive At has to be refused rather than divided by --
    Dfr would come back as a large positive number and read as heavy spreading
    when the truth is that the design itself leaves no opening.
    """
    side = float(arista_mm) - float(filament_d_mm)
    return side * side if side > 0 else None


def _size_classes(params: dict[str, Any], n: int) -> tuple[list[float], float]:
    """Arista per class and the shared filament diameter, padded to n classes."""
    aristas = [float(v) for v in
               (params.get("arista_mm") or config.FUSION_ARISTA_MM)]
    # A shorter list than the grid is extended with the next integers, matching
    # what the arista inputs do when the grid size is raised.
    while len(aristas) < n:
        aristas.append(float(len(aristas) + 1))
    d = float(params.get("filament_d_mm", config.FUSION_FILAMENT_D_MM))
    return aristas[:n], d


def _resolve_at(k: int, arista: float, d: float,
                overrides: dict[Any, Any]) -> tuple[float | None, str | None]:
    """At for one size class, honouring a stored override. Returns (At, warning)."""
    override = overrides.get(str(k), overrides.get(k))
    if override is not None:
        return float(override), None

    at = theoretical_area(arista, d)
    if at is None:
        return None, (
            f"Size class {k + 1}: arista {arista:g} mm is not larger than the "
            f"filament diameter {d:g} mm, so the design leaves no open pore and "
            "the theoretical area is undefined."
        )
    return at, None


def analyze(
    image: np.ndarray,
    roi_json: dict[str, Any] | None,
    params: dict[str, Any],
    scale: Scale,
    debug: bool = False,
) -> dict[str, Any]:
    n = int(params.get("grid_n", config.FUSION_GRID_N))
    aristas, filament_d = _size_classes(params, n)
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
        arista = aristas[k]
        at, warning = _resolve_at(k, arista, filament_d, at_overrides)
        if warning:
            flags.append(warning)

        geom = {"arista_mm": arista, "filament_d_mm": filament_d,
                "offset": seg.offset}

        if pore is None:
            flags.append(f"No pore found for the {arista:g} mm size class.")
            measurements.append(
                _row(k, arista, at, None, None, None, [], "missing", **geom))
            continue

        if pore.closed:
            # Complete fusion at minimum spacing is a valid result, not a
            # detection failure: Aa = 0, and the shape metrics do not exist.
            measurements.append(
                _row(k, arista, at, pore, 0.0, 0.0, ["closed"], "closed", **geom))
            continue

        area_mm2 = scale.area_mm2(pore.contour)
        perim_mm = scale.perimeter_mm(pore.contour)
        measurements.append(
            _row(k, arista, at, pore, area_mm2, perim_mm, list(pore.flags),
                 "open", **geom))

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
        # Every region the segmentation found, so the operator can override the
        # lattice's choice when a bad print confuses it.
        "candidates": _candidates(grid, scale, seg.offset, selected),
        "flags": flags,
        "roi": seg.roi.as_dict(),
        "threshold": seg.threshold,
        "polarity": seg.polarity,
        "offset": list(seg.offset),
        "arista_mm": aristas,
        "filament_d_mm": filament_d,
        "debug": trace.to_json(),
    }


def _candidates(grid, scale: Scale, offset: tuple[int, int],
                selected: list[Pore | None]) -> list[dict[str, Any]]:
    """Describe every detected region in full-frame coordinates.

    Areas and perimeters come from the *full* contour. `polygon` is a simplified
    copy for drawing only: a pore boundary traced pixel by pixel runs to well
    over a thousand points, and sending twenty-five of those would dominate the
    response for no measurable benefit.
    """
    chosen = {id(p) for p in selected if p is not None}
    ox, oy = offset

    out: list[dict[str, Any]] = []
    for i, blob in enumerate(grid.candidates):
        contour = blob["contour"]
        if contour is None or len(contour) < 3:
            continue

        epsilon = config.PORE_POLYGON_EPSILON * cv2.arcLength(contour, True)
        simplified = cv2.approxPolyDP(contour, max(epsilon, 1.0), True)

        cx, cy = blob["centroid"]
        out.append({
            "index": i,
            "centroid": [cx + ox, cy + oy],
            "polygon": [[float(px) + ox, float(py) + oy]
                        for px, py in simplified.reshape(-1, 2)],
            "aa_mm2": scale.area_mm2(contour),
            "perimeter_mm": scale.perimeter_mm(contour),
            "area_px": float(blob["area"]),
            "solidity": blob.get("solidity"),
            "auto_selected": id(blob.get("pore")) in chosen,
        })
    return out


def assign(
    candidates: list[dict[str, Any]],
    assignment: dict[Any, Any],
    params: dict[str, Any],
    manual_classes: Any = None,
) -> dict[str, Any]:
    """Build the measurement rows from an operator's choice of pores.

    `assignment` maps a size-class index to either a candidate index, the string
    "closed", or None for "leave unassigned", and covers *every* class -- classes
    the operator did not touch carry the pore the automatic pass gave them.

    `manual_classes` names the classes the operator actually clicked, so the
    untouched ones are still recorded as automatic. Without it a single manual
    correction would mark the whole table as hand-picked, which is exactly the
    distinction the flag exists to preserve.

    Rows are built through the same `_row` helper the automatic path uses, which
    is why this lives on the server: that helper is the one definition of a
    measurement row, and tab 5 and both CSV exports read the keys it produces.
    """
    n = int(params.get("grid_n", config.FUSION_GRID_N))
    aristas, filament_d = _size_classes(params, n)
    at_overrides = params.get("at_mm2") or {}
    by_index = {int(c["index"]): c for c in candidates}
    manual = {int(k) for k in (manual_classes or [])}

    measurements: list[dict[str, Any]] = []
    flags: list[str] = []

    for k in range(n):
        arista = aristas[k]
        at, warning = _resolve_at(k, arista, filament_d, at_overrides)
        if warning:
            flags.append(warning)

        geom = {"arista_mm": arista, "filament_d_mm": filament_d}
        choice = assignment.get(str(k), assignment.get(k))
        by_hand = k in manual

        if choice == "closed":
            measurements.append(
                _row(k, arista, at, None, 0.0, 0.0, ["closed"], "closed",
                     selection="manual_closed" if by_hand else "auto", **geom))
            continue

        if choice is None:
            measurements.append(
                _row(k, arista, at, None, None, None, [], "missing",
                     selection="manual" if by_hand else "auto", **geom))
            continue

        candidate = by_index.get(int(choice))
        if candidate is None:
            flags.append(f"Size class {k + 1}: no such detected region.")
            measurements.append(
                _row(k, arista, at, None, None, None, [], "missing",
                     selection="manual" if by_hand else "auto", **geom))
            continue

        measurements.append(_row(
            k, arista, at, None,
            float(candidate["aa_mm2"]), float(candidate["perimeter_mm"]),
            [], "open", selection="manual" if by_hand else "auto",
            centroid=candidate.get("centroid"),
            solidity=candidate.get("solidity"),
            candidate_index=int(candidate["index"]),
            **geom,
        ))

    # Two size classes on the same region is almost always a slip -- the second
    # assignment was meant for a neighbour -- and it is invisible in the table,
    # where both rows simply show the same area.
    used: dict[int, list[int]] = {}
    for row in measurements:
        idx = row["raw"].get("candidate_index")
        if idx is not None:
            used.setdefault(idx, []).append(row["raw"]["size_class"])
    for shared in (v for v in used.values() if len(v) > 1):
        flags.append(
            "Size classes " + ", ".join(str(c) for c in shared) +
            " are all pointing at the same region. Each class should have its "
            "own pore."
        )

    results = compute(measurements)
    return {"measurements": results["rows"], "results": results, "flags": flags}


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


def _row(k: int, arista: float, at: float | None, pore: Pore | None,
         area: float | None, perim: float | None,
         flags: list[str], status: str, *,
         selection: str = "auto",
         arista_mm: float | None = None,
         filament_d_mm: float | None = None,
         centroid: Any = None,
         solidity: float | None = None,
         candidate_index: int | None = None,
         offset: tuple[int, int] = (0, 0)) -> dict[str, Any]:
    # Centroids are stored in full-frame coordinates, never crop coordinates.
    # The crop origin moves with the ROI and the padding width, so a centroid
    # relative to it would mean something different after any parameter change,
    # and a manually assigned pore (whose position is already known in the frame)
    # could not be stored in the same field as an automatically detected one.
    if pore is not None:
        centroid = (pore.centroid[0] + offset[0], pore.centroid[1] + offset[1])
        solidity = pore.solidity
    return {
        "index_no": k,
        "label": f"a = {arista:g} mm",
        "included": status != "missing",
        "raw": {
            "size_class": k + 1,
            # Kept under the old key so a run saved before the arista change and
            # one saved after still group together in the replicate summary.
            "nominal_fd_mm": arista,
            "arista_mm": arista if arista_mm is None else arista_mm,
            "filament_d_mm": filament_d_mm,
            "at_mm2": at,
            "aa_mm2": area,
            "perimeter_mm": perim,
            "status": status,
            "selection": selection,
            "centroid": list(centroid) if centroid is not None else None,
            "row": pore.row if pore else None,
            "col": pore.col if pore else None,
            "solidity": solidity,
            "candidate_index": candidate_index,
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
