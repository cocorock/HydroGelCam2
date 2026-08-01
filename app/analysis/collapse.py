"""Tab 4 - filament collapse test (Ingri2024 Eq. 5; method from Habib & Khoda).

A single filament is bridged across a 3D-printed ABS platform with seven pillars
and six gaps of 1-6 mm, photographed from the side immediately after deposition.

    Cf = A_sag / A_max x 100 %

where A_sag is the area between the pillar-top line and the underside of the
deflected filament, and A_max = nominal gap x 6 mm pillar height (total collapse
to the floor). Flat bridge -> 0 %, fully collapsed or broken -> 100 %.

That is the direction used in the paper's Results and Fig. 3d ("1C4L ink had a
lower area collapse factor ... indicat[ing] low deformation"). Section 2.4.3 of
the same paper states the opposite convention; the two are exact complements, so
A_sag, A_max, gap and df are all persisted and the display can be switched
without re-analysing the image. See config.CF_CONVENTION.
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

from app import config
from app.analysis.stats import describe
from app.calib.geometry import Scale
from app.pipeline.debug import DebugTrace
from app.pipeline.signal1d import runs


def analyze(
    image: np.ndarray,
    roi_json: dict[str, Any] | None,
    params: dict[str, Any],
    scale: Scale,
    debug: bool = False,
) -> dict[str, Any]:
    gaps_mm = [float(v) for v in params.get("gaps_mm", config.COLLAPSE_GAPS_MM)]
    pillars_mm = [float(v) for v in
                  params.get("pillar_widths_mm", config.COLLAPSE_PILLAR_WIDTHS_MM)]
    gap_height = float(params.get("gap_height_mm", config.COLLAPSE_GAP_HEIGHT_MM))
    convention = params.get("convention", config.CF_CONVENTION)
    amax_overrides = params.get("a_max_mm2") or {}
    hsv = params.get("hsv") or config.HSV_DEFAULTS

    trace = DebugTrace(enabled=debug)

    work = image
    if params.get("quad"):
        work = rectify(image, params["quad"], trace)
    trace.add("1_source", work, "Frame after distortion correction and, if "
                                "supplied, four-point perspective rectification.")

    roi = params.get("roi") or roi_json
    if roi:
        x = max(0, int(roi.get("x", 0)))
        y = max(0, int(roi.get("y", 0)))
        w = min(work.shape[1] - x, int(roi.get("w", work.shape[1])))
        h = min(work.shape[0] - y, int(roi.get("h", work.shape[0])))
        crop = work[y:y + h, x:x + w]
        offset = (x, y)
    else:
        crop = work
        offset = (0, 0)

    pillar_mask, filament_mask = segment(crop, hsv, trace)

    platform = find_platform(pillar_mask, pillars_mm, gaps_mm, trace, crop)
    if platform is None:
        return _empty(
            gaps_mm, convention, offset, trace,
            "Could not locate the pillar platform. Check the ROI and the pillar "
            "HSV range in the tab settings.",
        )

    df_px, df_mm = filament_diameter(filament_mask, platform, scale)

    measurements: list[dict[str, Any]] = []
    flags: list[str] = []

    if platform.get("misaligned_gaps"):
        bad = ", ".join(str(i + 1) for i in platform["misaligned_gaps"])
        flags.append(
            f"Gap(s) {bad} do not line up with an actual opening in the platform. "
            f"Pillar positions are placed from the full "
            f"{sum(pillars_mm) + sum(gaps_mm):g} mm platform width, so this "
            "usually means the ROI cuts into the part, or the pillar and gap "
            "dimensions above do not match the platform in the photograph. "
            "Widen the ROI to include the whole platform and re-detect."
        )

    for i, gap in enumerate(platform["gaps"]):
        nominal = gaps_mm[i] if i < len(gaps_mm) else float(i + 1)
        a_max_default = nominal * gap_height
        a_max = float(amax_overrides.get(str(i), amax_overrides.get(i,
                                                                   a_max_default)))

        profile = sag_profile(filament_mask, gap, crop.shape)

        if profile is None:
            flags.append(f"Gap {i + 1} ({nominal:g} mm): no filament spans the gap.")
            measurements.append(_row(i, nominal, a_max, None, None, None, None,
                                     df_mm, "broken", gap))
            continue

        a_sag_mm2, depth_px, depth_x = integrate_sag(profile, gap, scale)
        theta = deflection_angle(gap, depth_px, depth_x, scale)

        measurements.append(
            _row(i, nominal, a_max, a_sag_mm2, depth_px, depth_x, theta,
                 df_mm, "bridged", gap)
        )

    results = compute(measurements, convention)
    results["df_mm"] = df_mm
    results["df_px"] = df_px
    results["platform"] = {
        "x0": platform["x0"] + offset[0],
        "x1": platform["x1"] + offset[0],
        "top_y": platform["top_y"] + offset[1],
        "px_per_mm": platform["px_per_mm"],
        "anchors": [[a[0] + offset[0], a[1] + offset[1]]
                    for a in platform["anchors"]],
    }
    results["scale_cross_check"] = _cross_check(platform, scale, pillars_mm, gaps_mm)

    if trace.enabled:
        trace.add("9_gaps", _overlay(crop, platform, filament_mask, measurements),
                  "Cyan = pillar-top reference line per gap. Yellow = measured "
                  "filament underside. Red dot = point of maximum sag.")

    return {
        # The enriched rows, not the bare geometry: the derived metrics belong
        # on each measurement so a saved run carries its own results and the
        # replicate summary can aggregate Cf without re-deriving it.
        "measurements": results["rows"],
        "results": results,
        "flags": flags,
        "roi": roi or {},
        "offset": list(offset),
        "convention": convention,
        "debug": trace.to_json(),
    }


def compute(measurements: list[dict[str, Any]],
            convention: str = config.CF_CONVENTION) -> dict[str, Any]:
    """Formulas only, so a stored run can be re-scored or its convention flipped
    without touching the image."""
    rows: list[dict[str, Any]] = []
    for m in measurements:
        raw = dict(m.get("raw") or {})
        a_sag = raw.get("a_sag_mm2")
        a_max = raw.get("a_max_mm2")
        status = raw.get("status", "bridged")

        if status == "broken":
            ratio = 1.0                       # total collapse
        elif a_max and a_max > 0 and a_sag is not None:
            ratio = max(0.0, min(1.0, a_sag / a_max))
        else:
            ratio = None

        if ratio is None:
            cf = None
        elif convention == "sag":
            cf = ratio * 100.0                # 0 % flat bridge, 100 % collapsed
        else:
            cf = (1.0 - ratio) * 100.0        # Methods-section complement

        raw["cf_percent"] = cf
        raw["sag_ratio"] = ratio
        rows.append({**m, "raw": raw})

    included = [r["raw"] for r in rows if r.get("included")]
    bridged = [r for r in included if r.get("status") == "bridged"]

    return {
        "rows": rows,
        "convention": convention,
        "convention_label": config.CF_CONVENTION_LABELS.get(convention, convention),
        "n_gaps": len(rows),
        "n_bridged": len(bridged),
        "n_broken": sum(1 for r in included if r.get("status") == "broken"),
        "cf": describe([r["cf_percent"] for r in included
                        if r.get("cf_percent") is not None]),
        "theta": describe([r["theta_deg"] for r in bridged
                           if r.get("theta_deg") is not None]),
    }


# ---------------------------------------------------------------- segmentation


def rectify(image: np.ndarray, quad: list, trace: DebugTrace | None = None
            ) -> np.ndarray:
    """Four-point perspective correction from the platform's outer corners.

    Corners are given in the order top-left, top-right, bottom-right, bottom-left.
    The output keeps the platform's real 51 x 10 mm aspect ratio, so a tilted
    photograph is measured as if taken square on.
    """
    src = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    aspect = config.COLLAPSE_PLATFORM_W_MM / config.COLLAPSE_PLATFORM_H_MM

    width = max(np.linalg.norm(src[1] - src[0]), np.linalg.norm(src[2] - src[3]))
    height = width / aspect
    dst = np.array([[0, 0], [width, 0], [width, height], [0, height]],
                   dtype=np.float32)

    H = cv2.getPerspectiveTransform(src, dst)
    out = cv2.warpPerspective(image, H, (int(round(width)), int(round(height))))
    if trace is not None:
        trace.add("1b_rectified", out,
                  "Perspective-corrected to the platform's 51 x 10 mm aspect.")
    return out


def segment(crop: np.ndarray, hsv_ranges: dict,
            trace: DebugTrace | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Split the side view into pale-red pillars and dyed filament."""
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    pillar = _hsv_mask(hsv, hsv_ranges.get("pillar"))
    if hsv_ranges.get("pillar2"):
        pillar = cv2.bitwise_or(pillar, _hsv_mask(hsv, hsv_ranges["pillar2"]))
    filament = _hsv_mask(hsv, hsv_ranges.get("filament"))

    # The pillars are a large solid body, so they tolerate a real clean-up. The
    # filament does not: a 0.41 mm bridge is only a handful of pixels tall in a
    # 51 mm-wide field, and an aggressive open erases the very thing being
    # measured. Close it to bridge dropout, then open with a 3 px element only.
    k_pillar = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    pillar = cv2.morphologyEx(pillar, cv2.MORPH_CLOSE, k_pillar)
    pillar = cv2.morphologyEx(pillar, cv2.MORPH_OPEN, k_pillar)

    k_fil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    filament = cv2.morphologyEx(filament, cv2.MORPH_CLOSE, k_fil)
    filament = cv2.morphologyEx(filament, cv2.MORPH_OPEN, k_fil)

    if trace is not None:
        trace.add_overlay("2_pillars", crop, pillar,
                          "Pillar (ABS platform) mask from the pale-red HSV range.",
                          color=(0, 0, 255))
        trace.add_overlay("3_filament", crop, filament,
                          "Filament mask from the dye HSV range.",
                          color=(0, 255, 255))
    return pillar, filament


def _hsv_mask(hsv: np.ndarray, spec: dict | None) -> np.ndarray:
    if not spec:
        return np.zeros(hsv.shape[:2], np.uint8)
    lo = np.asarray(spec["lo"], dtype=np.uint8)
    hi = np.asarray(spec["hi"], dtype=np.uint8)
    return cv2.inRange(hsv, lo, hi)


# ---------------------------------------------------------------- platform


def find_platform(
    pillar_mask: np.ndarray,
    pillar_widths_mm: list[float],
    gaps_mm: list[float],
    trace: DebugTrace | None = None,
    crop: np.ndarray | None = None,
) -> dict[str, Any] | None:
    """Locate the platform and place the seven pillars along it.

    The 2 mm inner pillars are only a few pixels wide in a 51 mm-wide field, so
    detecting each one independently is fragile. The platform outline is easy to
    find, though, and its total width is known exactly -- so the pillar and gap
    edges are placed from the nominal layout and then each anchor is refined
    against the mask locally.
    """
    contours, _ = cv2.findContours(pillar_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    body = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(body)
    if w < 20 or h < 5:
        return None

    total_mm = sum(pillar_widths_mm) + sum(gaps_mm)
    px_per_mm = w / total_mm if total_mm else 1.0

    # Column-wise topmost pillar pixel: high over a pillar, low over a gap.
    column_top = _column_top(pillar_mask, x, x + w)

    edges_mm: list[float] = []
    cursor = 0.0
    for i, pw in enumerate(pillar_widths_mm):
        edges_mm.append(cursor)
        cursor += pw
        edges_mm.append(cursor)
        if i < len(gaps_mm):
            cursor += gaps_mm[i]

    pillars = []
    for i in range(len(pillar_widths_mm)):
        left = x + edges_mm[2 * i] * px_per_mm
        right = x + edges_mm[2 * i + 1] * px_per_mm
        top = _pillar_top(column_top, left, right, y)
        pillars.append({"index": i, "left": left, "right": right, "top": top})

    gaps = []
    anchors = []
    for i in range(len(pillars) - 1):
        left_pillar, right_pillar = pillars[i], pillars[i + 1]
        gaps.append({
            "index": i,
            "x0": left_pillar["right"],
            "x1": right_pillar["left"],
            "y_left": left_pillar["top"],
            "y_right": right_pillar["top"],
            "top_y": (left_pillar["top"] + right_pillar["top"]) / 2.0,
        })
        anchors.append((left_pillar["right"], left_pillar["top"]))
        anchors.append((right_pillar["left"], right_pillar["top"]))

    if trace is not None and crop is not None:
        vis = crop.copy()
        for p in pillars:
            cv2.rectangle(vis, (int(p["left"]), int(p["top"])),
                          (int(p["right"]), y + h), (255, 0, 255), 1)
        for a in anchors:
            cv2.circle(vis, (int(a[0]), int(a[1])), 4, (0, 255, 0), -1)
        trace.add("4_platform", vis,
                  f"Seven pillars placed from the nominal layout across a "
                  f"{w} px platform ({px_per_mm:.2f} px/mm). Green = anchors.")

    # The platform's own width is the length reference for every pillar and gap
    # position, so a platform running off the edge of the ROI silently rescales
    # the whole layout rather than failing outright. Touching an edge is not
    # itself evidence -- a correctly framed platform often fills the frame --
    # so check the placement instead: over a real gap the topmost pillar pixel
    # drops to the floor, and over a pillar it does not. If the placed gaps do
    # not line up with that, the layout is wrong.
    misaligned = _misaligned_gaps(column_top, x, pillars, gaps)

    return {
        "x0": x, "x1": x + w, "y": y, "h": h,
        "top_y": min(p["top"] for p in pillars),
        "px_per_mm": px_per_mm,
        "pillars": pillars,
        "gaps": gaps,
        "anchors": anchors,
        "misaligned_gaps": misaligned,
    }


def _misaligned_gaps(column_top: np.ndarray, x0: int,
                     pillars: list[dict], gaps: list[dict]) -> list[int]:
    """Gap positions where the pillar mask does not actually fall away.

    `column_top` is indexed from the platform's left edge, so gap coordinates
    are shifted by `x0` before sampling.
    """
    pillar_tops = [p["top"] for p in pillars]
    if not pillar_tops:
        return []
    reference = float(np.median(pillar_tops))

    depths = []
    for gap in gaps:
        lo = int(round(gap["x0"] - x0))
        hi = int(round(gap["x1"] - x0))
        lo, hi = max(0, lo), min(len(column_top), hi)
        if hi - lo < 2:
            depths.append(0.0)
            continue
        depths.append(float(np.median(column_top[lo:hi])) - reference)

    # A real gap drops noticeably below the pillar tops. Judge each against the
    # deepest one found, so this works at any image scale.
    deepest = max(depths) if depths else 0.0
    if deepest <= 2.0:
        return list(range(len(gaps)))
    return [i for i, d in enumerate(depths) if d < 0.35 * deepest]


def _column_top(mask: np.ndarray, x0: int, x1: int) -> np.ndarray:
    """Top edge of the pillar per column, or the image height where empty.

    Returns the *upper edge* of the topmost set pixel, matching the pixel-edge
    convention used for the filament underside in `sag_profile`.
    """
    h = mask.shape[0]
    sub = mask[:, int(x0):int(x1)] > 0
    has = sub.any(axis=0)
    top = np.argmax(sub, axis=0).astype(np.float64) - 0.5
    top[~has] = h
    return top


def _pillar_top(column_top: np.ndarray, left: float, right: float,
                fallback: int) -> float:
    """Median top edge across a pillar's own columns, ignoring its bevelled ends."""
    lo = int(max(0, round(left - 0)))
    hi = int(min(len(column_top), round(right)))
    inset = max(1, int((hi - lo) * 0.2))
    seg = column_top[lo + inset:hi - inset] if hi - lo > 2 * inset + 1 \
        else column_top[lo:hi]
    seg = seg[np.isfinite(seg)]
    if seg.size == 0:
        return float(fallback)
    return float(np.median(seg))


def filament_diameter(filament_mask: np.ndarray, platform: dict[str, Any],
                      scale: Scale) -> tuple[float | None, float | None]:
    """Filament thickness measured on the pillar tops, where it is not deflecting."""
    thicknesses: list[float] = []
    for p in platform["pillars"]:
        lo, hi = int(round(p["left"])), int(round(p["right"]))
        for x in range(lo, min(hi, filament_mask.shape[1])):
            column = filament_mask[:, x] > 0
            stretches = runs(column)
            if not stretches:
                continue
            # The stretch sitting on this pillar's top edge.
            near = [s for s in stretches if abs(s[0] + s[1] - p["top"]) < s[1] + 12]
            chosen = near[0] if near else max(stretches, key=lambda s: s[1])
            thicknesses.append(float(chosen[1]))

    if not thicknesses:
        return None, None
    df_px = float(np.median(thicknesses))
    return df_px, scale.dy_mm(df_px)


# ---------------------------------------------------------------- per gap


def sag_profile(filament_mask: np.ndarray, gap: dict[str, Any],
                shape: tuple[int, ...]) -> np.ndarray | None:
    """Underside of the filament across one gap, as (x, y) in crop coordinates.

    Returns None when the filament does not span the gap, which is the "no
    bridge" case -- a valid outcome, not a detection failure.
    """
    x0 = int(math.ceil(gap["x0"]))
    x1 = int(math.floor(gap["x1"]))
    if x1 - x0 < 3:
        return None

    x1 = min(x1, filament_mask.shape[1])
    xs, ys = [], []
    for x in range(x0, x1):
        column = np.flatnonzero(filament_mask[:, x] > 0)
        if column.size == 0:
            continue
        xs.append(float(x))
        # Pixel *edges*, not centres: the underside of the filament is the
        # bottom edge of its lowest set pixel. The pillar-top reference uses the
        # matching convention in `_column_top`. Mixing the two leaves a one-pixel
        # bias in every gap, which at small spacings is several percent of A_max.
        ys.append(float(column.max()) + 0.5)

    if not xs:
        return None

    coverage = len(xs) / float(x1 - x0)
    if coverage < 0.85:
        return None                          # a real break, not just noise

    return np.column_stack([np.asarray(xs), np.asarray(ys)])


def integrate_sag(profile: np.ndarray, gap: dict[str, Any], scale: Scale
                  ) -> tuple[float, float, float]:
    """Area between the pillar-top line and the filament underside, in mm^2."""
    top_left = gap["y_left"]
    top_right = gap["y_right"]
    x0, x1 = gap["x0"], gap["x1"]

    # The reference line joins the two pillar tops, so a platform photographed
    # very slightly off-level does not read as sag.
    span = max(x1 - x0, 1e-6)
    reference = top_left + (profile[:, 0] - x0) * (top_right - top_left) / span

    depths = np.maximum(profile[:, 1] - reference, 0.0)
    depth_px = float(depths.max()) if depths.size else 0.0
    depth_x = float(profile[int(np.argmax(depths)), 0]) if depths.size else x0

    # Close the region: along the reference line, then back along the underside.
    top_edge = np.column_stack([profile[:, 0], reference])
    polygon = np.vstack([top_edge, profile[::-1]])
    area_mm2 = scale.polyline_area_mm2(polygon)

    return area_mm2, depth_px, depth_x


def deflection_angle(gap: dict[str, Any], depth_px: float, depth_x: float,
                     scale: Scale) -> float:
    """Angle between the pillar-top horizontal and the anchor-to-deepest-point line."""
    left_dx = abs(depth_x - gap["x0"])
    right_dx = abs(gap["x1"] - depth_x)
    run_px = min(left_dx, right_dx)
    if run_px <= 1e-6:
        return 0.0
    rise_mm = scale.dy_mm(depth_px)
    run_mm = scale.dx_mm(run_px)
    if run_mm <= 1e-9:
        return 0.0
    return float(math.degrees(math.atan2(rise_mm, run_mm)))


# ---------------------------------------------------------------- helpers


def _row(index: int, nominal: float, a_max: float, a_sag: float | None,
         depth_px: float | None, depth_x: float | None, theta: float | None,
         df_mm: float | None, status: str, gap: dict[str, Any]) -> dict[str, Any]:
    return {
        "index_no": index,
        "label": f"Gap {index + 1} ({nominal:g} mm)",
        "included": True,
        "raw": {
            "gap_no": index + 1,
            "nominal_gap_mm": nominal,
            "a_max_mm2": a_max,
            "a_sag_mm2": a_sag,
            "at_strip_mm2": (nominal * df_mm) if df_mm else None,
            "df_mm": df_mm,
            "max_sag_px": depth_px,
            "max_sag_x": depth_x,
            "theta_deg": theta,
            "status": status,
            "x0": gap["x0"], "x1": gap["x1"],
            "top_y": gap["top_y"],
        },
    }


def _empty(gaps_mm, convention, offset, trace, message) -> dict[str, Any]:
    return {
        "measurements": [],
        # Same shape as a successful result, so the results panel renders "N/A"
        # rather than "undefined" when detection fails.
        "results": {"rows": [], "convention": convention,
                    "convention_label": config.CF_CONVENTION_LABELS.get(convention),
                    "n_gaps": len(gaps_mm), "n_bridged": 0, "n_broken": 0,
                    "df_mm": None, "df_px": None,
                    "platform": None, "scale_cross_check": None,
                    "cf": describe([]), "theta": describe([])},
        "flags": [message],
        "roi": {},
        "offset": list(offset),
        "convention": convention,
        "debug": trace.to_json(),
    }


def _cross_check(platform: dict[str, Any], scale: Scale,
                 pillars_mm, gaps_mm) -> dict[str, Any]:
    """Compare the width implied by the platform against the stored calibration."""
    total_mm = sum(pillars_mm) + sum(gaps_mm)
    width_px = platform["x1"] - platform["x0"]
    implied = total_mm / width_px if width_px else None
    stored = scale.mm_per_px_x if scale.calibrated else None
    disagreement = None
    if implied and stored:
        disagreement = abs(implied - stored) / stored
    return {
        "implied_mm_per_px": implied,
        "calibration_mm_per_px": stored,
        "disagreement": disagreement,
        "warn": bool(disagreement is not None and disagreement > 0.05),
    }


def _overlay(crop, platform, filament_mask, measurements) -> np.ndarray:
    vis = crop.copy()
    for m in measurements:
        raw = m["raw"]
        x0, x1 = int(raw["x0"]), int(raw["x1"])
        y = int(raw["top_y"])
        cv2.line(vis, (x0, y), (x1, y), (255, 255, 0), 1)
        if raw["status"] == "bridged" and raw.get("max_sag_x") is not None:
            dx = int(raw["max_sag_x"])
            dy = int(raw["top_y"] + (raw["max_sag_px"] or 0))
            cv2.circle(vis, (dx, dy), 4, (0, 0, 255), -1)
            cv2.line(vis, (x0, y), (dx, dy), (0, 165, 255), 1)
    return vis
