"""Preprocessing steps 6-8: locate the pore grid and extract each pore's boundary.

Used only by the fusion test. The projection-and-morphology chain finds where the
printed lattice actually is, so pores that are clipped by the ROI edge or sit
outside the grid never enter the measurement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from app import config
from app.pipeline.common import Segmentation, clean_mask
from app.pipeline.debug import DebugTrace
from app.pipeline.signal1d import runs


@dataclass
class Pore:
    row: int
    col: int
    centroid: tuple[float, float]        # crop coordinates
    contour: np.ndarray | None           # crop coordinates, (N,1,2)
    area_px: float
    perimeter_px: float
    pass1_area_px: float
    solidity: float
    closed: bool
    flags: list[str]


@dataclass
class GridResult:
    grid_mask: np.ndarray
    final_mask: np.ndarray
    pores: list[Pore]
    # Every enclosed region that survived the pass-1 filters, *before* the
    # lattice assignment narrowed them down. When a print is bad it is usually
    # the lattice step that misfires -- broken walls, merged cells -- so the
    # region the operator actually wants may not be among `pores` at all.
    candidates: list[dict]
    n_rows: int
    n_cols: int
    complete: bool
    message: str


def detect_grid(
    seg: Segmentation,
    n: int = config.FUSION_GRID_N,
    *,
    coverage: float = config.PROJECTION_COVERAGE,
    morph: tuple[int, ...] = config.PROJECTION_MORPH,
    final_kernel: int = config.FINAL_MORPH_KERNEL,
) -> GridResult:
    trace = seg.trace
    binary = seg.binary

    # ---- step 6: where is the valid pore grid? ---------------------------
    grid_mask, col_clean, row_clean, col_wall, row_wall = _grid_mask(
        binary, coverage, morph, trace)

    # The openings between the walls are the lattice cells, and pass 1 uses them
    # to place each pore rather than clustering centroids.
    #
    # Measured on the *raw* wall map, not the cleaned one. The morphology chain
    # ends in a net dilation, so every wall in the cleaned map is about nine
    # pixels wider on each side than the filament actually is -- cells taken from
    # it are ~18 px short in each direction and every pore area comes out low.
    # The cleaned map still decides which walls are real; the raw one says where
    # their edges are.
    seg.meta["row_cells"] = (cells_between_walls(row_wall, n)
                             or cells_between_walls(row_clean, n))
    seg.meta["col_cells"] = (cells_between_walls(col_wall, n)
                             or cells_between_walls(col_clean, n))

    # ---- step 7: restrict the binary image to that region ----------------
    restricted = cv2.bitwise_and(binary, binary, mask=grid_mask)
    final_mask = clean_mask(restricted, final_kernel, trace, name="7_final_mask")

    # Confine the search to the openings between the walls. Everything else
    # inside the grid's bounding rectangle -- the wedges left in its corners, and
    # any background reachable through a break in an outer wall -- is not a pore
    # and must not be able to merge with one.
    row_cells = seg.meta.get("row_cells")
    col_cells = seg.meta.get("col_cells")
    cell_mask = _cell_mask(binary.shape, row_cells, col_cells)
    if cell_mask is not None:
        grid_mask = cv2.bitwise_and(grid_mask, cell_mask)
        trace.add_mask("6b_cells", grid_mask,
                       "Grid region narrowed to the openings between the "
                       "detected walls.")

    # ---- step 8: two-pass contour extraction ------------------------------
    pores, candidates, complete, message = _two_pass_contours(
        grid_mask, final_mask, n, trace, seg
    )

    return GridResult(
        grid_mask=grid_mask,
        final_mask=final_mask,
        pores=pores,
        candidates=candidates,
        n_rows=n,
        n_cols=n,
        complete=complete,
        message=message,
    )


# ---------------------------------------------------------------- step 6


def _grid_mask(
    binary: np.ndarray,
    coverage: float,
    morph: tuple[int, ...],
    trace: DebugTrace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Row and column projections -> cleaned 1-D wall maps -> 2-D validity mask."""
    h, w = binary.shape[:2]
    fg = (binary > 0).astype(np.float32)

    row_cov = fg.sum(axis=1) / max(w, 1)     # per-row coverage, length h
    col_cov = fg.sum(axis=0) / max(h, 1)     # per-column coverage, length w

    row_wall = (row_cov >= coverage).astype(np.uint8)
    col_wall = (col_cov >= coverage).astype(np.uint8)

    row_clean = _clean_profile(row_wall, morph)
    col_clean = _clean_profile(col_wall, morph)

    # The grid spans from the first to the last detected wall on each axis. The
    # cleaned profile is used to reject spurious walls, but its extent is taken
    # from the raw one: the dilate step in the chain grows the wall map several
    # pixels past the outermost filament, and that sliver of background outside
    # the grid forms a ring. Any weak spot in an outer wall then connects a
    # corner pore to that ring, and the pore is discarded as clipped.
    row_span = _intersect(_span(row_clean), _span(row_wall))
    col_span = _intersect(_span(col_clean), _span(col_wall))

    mask = np.zeros((h, w), np.uint8)
    if row_span and col_span:
        mask[row_span[0]:row_span[1] + 1, col_span[0]:col_span[1] + 1] = 255

    if trace.enabled:
        trace.add_plot(
            "6_projections",
            _projection_figure(row_cov, col_cov, row_clean, col_clean, coverage),
            "Row/column coverage projections, thresholded at "
            f"{coverage:.0%} and cleaned with the "
            f"{'->'.join(str(k) for k in morph)} morphology chain.",
        )
        trace.add_mask("6_grid_mask", mask,
                       "Region the two cleaned projections agree is inside the "
                       "printed lattice.")

    return mask, col_clean, row_clean, col_wall, row_wall


def _clean_profile(profile: np.ndarray, morph: tuple[int, ...]) -> np.ndarray:
    """open -> close -> erode -> dilate -> close on a 1-D binary profile."""
    k_open, k_close, k_erode, k_dilate, k_close2 = morph[:5]
    p = profile.reshape(1, -1).astype(np.uint8)

    def se(size: int) -> np.ndarray:
        return cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, int(size)), 1))

    p = cv2.morphologyEx(p, cv2.MORPH_OPEN, se(k_open))
    p = cv2.morphologyEx(p, cv2.MORPH_CLOSE, se(k_close))
    p = cv2.erode(p, se(k_erode))
    p = cv2.dilate(p, se(k_dilate))
    p = cv2.morphologyEx(p, cv2.MORPH_CLOSE, se(k_close2))
    return p.ravel()


def _span(profile: np.ndarray) -> tuple[int, int] | None:
    idx = np.flatnonzero(profile)
    if idx.size == 0:
        return None
    return int(idx[0]), int(idx[-1])


def _intersect(a: tuple[int, int] | None,
               b: tuple[int, int] | None) -> tuple[int, int] | None:
    if a is None:
        return b
    if b is None:
        return a
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    return (lo, hi) if hi > lo else a


def _projection_figure(row_cov, col_cov, row_clean, col_clean, coverage):
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 4.6))
    ax1.plot(row_cov, lw=1.0, color="#1f77b4", label="row coverage")
    ax1.plot(row_clean * row_cov.max(), lw=1.0, color="#d62728",
             alpha=0.7, label="cleaned wall map")
    ax1.axhline(coverage, color="#999", ls="--", lw=0.9)
    ax1.set_xlabel("row (y)")
    ax1.legend(fontsize=7)

    ax2.plot(col_cov, lw=1.0, color="#2ca02c", label="column coverage")
    ax2.plot(col_clean * col_cov.max(), lw=1.0, color="#d62728",
             alpha=0.7, label="cleaned wall map")
    ax2.axhline(coverage, color="#999", ls="--", lw=0.9)
    ax2.set_xlabel("column (x)")
    ax2.legend(fontsize=7)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- step 8


def _two_pass_contours(
    grid_mask: np.ndarray,
    final_mask: np.ndarray,
    n: int,
    trace: DebugTrace,
    seg: Segmentation,
) -> tuple[list[Pore], list[dict], bool, str]:
    """Pass 1 finds expected pore positions; pass 2 re-contours each one locally."""
    # Pass 1: pores are the holes in the material, so invert inside the grid.
    inside = cv2.bitwise_and(cv2.bitwise_not(final_mask), grid_mask)

    # Connected components rather than findContours: the cleaned projections
    # overshoot the outermost walls by a few pixels, leaving a thin background
    # ring around the lattice. That ring is the outermost contour, so RETR_EXTERNAL
    # would return it alone and hide every pore nested inside it. Components let
    # us drop it by the fact that it reaches the grid-mask border -- which also
    # correctly discards any pore clipped by the ROI edge.
    blobs = _components_inside(inside, grid_mask, seg.valid)

    if trace.enabled:
        vis = cv2.cvtColor(final_mask, cv2.COLOR_GRAY2BGR)
        for b in blobs:
            cv2.circle(vis, tuple(int(v) for v in b["centroid"]), 6, (0, 255, 0), -1)
        trace.add("8a_pass1", vis,
                  f"Pass 1 on the grid mask: {len(blobs)} candidate pores.")

    rows, cols, complete, message = _assign_lattice(
        blobs, n, seg.meta.get("row_cells"), seg.meta.get("col_cells")
    )

    # Pass 2: re-contour each expected position on a local crop, which keeps a
    # neighbouring pore from being merged in by a thin break in the wall.
    row_cells = seg.meta.get("row_cells")
    col_cells = seg.meta.get("col_cells")

    pores: list[Pore] = []
    for (r, c), blob in sorted(rows.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        # Bound the refinement window by the pore's own cell, so a break in a
        # wall cannot let the component spill into the neighbouring pore and be
        # measured as one oversized opening.
        cell = None
        if row_cells and col_cells and r < len(row_cells) and c < len(col_cells):
            cell = (col_cells[c][0], row_cells[r][0],
                    col_cells[c][1], row_cells[r][1])
        pore = _refine(blob, final_mask, r, c, cell)
        # Back-reference, so the caller can mark which candidates the automatic
        # pass ended up choosing without comparing geometry.
        blob["pore"] = pore
        pores.append(pore)

    if trace.enabled and pores:
        vis = cv2.cvtColor(seg.flat, cv2.COLOR_GRAY2BGR)
        for p in pores:
            if p.contour is not None:
                cv2.drawContours(vis, [p.contour], -1, (0, 200, 255), 2)
            cv2.putText(vis, f"{p.row},{p.col}",
                        (int(p.centroid[0]) - 12, int(p.centroid[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        trace.add("8b_pass2", vis,
                  "Pass 2: boundary re-extracted on a local crop around each "
                  "expected pore position.")

    _ = cols
    return pores, blobs, complete, message


def _components_inside(inside: np.ndarray, grid_mask: np.ndarray,
                       valid: np.ndarray | None = None) -> list[dict]:
    """Background components that are plausibly single pores."""
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(inside, 8)
    if count <= 1:
        return []

    ys, xs = np.nonzero(grid_mask)
    if xs.size == 0:
        return []
    gx0, gx1 = int(xs.min()), int(xs.max())
    gy0, gy1 = int(ys.min()), int(ys.max())
    grid_w = max(1, gx1 - gx0)
    grid_h = max(1, gy1 - gy0)

    # Only the ROI's own edge marks a genuinely half-measured pore. The grid
    # mask's edge does not: an outer wall dimmed by vignetting can have a small
    # break in it, and a corner pore leaking a few pixels through that break is
    # still perfectly measurable.
    if valid is not None:
        vys, vxs = np.nonzero(valid)
        vx0, vx1 = int(vxs.min()), int(vxs.max())
        vy0, vy1 = int(vys.min()), int(vys.max())
    else:
        vx0, vx1, vy0, vy1 = gx0, gx1, gy0, gy1

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = float(areas.max()) if areas.size else 0.0
    # In a 1..N mm sweep the smallest pore is 1/N^2 of the largest, so anything
    # well under that is a fragment rather than a pore.
    min_area = max(20.0, largest * config.PORE_MIN_AREA_FRACTION)

    blobs: list[dict] = []
    for i in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[i])
        if area < min_area:
            continue
        # A component spanning most of the grid in both directions is the ring of
        # background outside the lattice, or several pores merged through
        # collapsed walls. Either way it is not one pore.
        if w > 0.55 * grid_w and h > 0.55 * grid_h:
            continue
        if x <= vx0 or y <= vy0 or x + w - 1 >= vx1 or y + h - 1 >= vy1:
            continue
        # Extent -- area over bounding-box area -- is what separates a pore from
        # the leftovers. A pore very nearly fills its own box (0.99 for a clean
        # square, 0.79 even for a circle). The wedges left in the corners of the
        # rectangular grid mask, and the remnant of a pore that has fused shut,
        # sit far below that while still being large enough to pass an area test.
        if area < config.PORE_MIN_EXTENT * w * h:
            continue
        component = (labels == i).astype(np.uint8)
        contours, _ = cv2.findContours(
            component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        if not contours:
            continue
        blobs.append({
            "centroid": (float(centroids[i][0]), float(centroids[i][1])),
            "area": float(area),
            "contour": max(contours, key=cv2.contourArea),
        })
    return blobs


def cells_between_walls(profile: np.ndarray, n: int) -> list[tuple[int, int]] | None:
    """The n openings between the n+1 walls of a cleaned 1-D wall map.

    Deriving the lattice from the walls themselves is far steadier than
    clustering pore centroids. A break in a vignette-dimmed outer wall leaves
    small slivers of background that are indistinguishable from a genuine pore by
    position alone, and a handful of those is enough to drag a k-means cluster
    centre onto the wrong row.
    """
    walls = [(s, s + ln - 1) for s, ln in runs(profile.astype(bool))]
    if len(walls) < 2:
        return None
    if len(walls) != n + 1:
        # Keep the widest n+1: thin spurious runs are the usual excess.
        if len(walls) < n + 1:
            return None
        walls = sorted(sorted(walls, key=lambda r: r[1] - r[0],
                              reverse=True)[:n + 1])
    return [(walls[i][1] + 1, walls[i + 1][0] - 1) for i in range(n)]


def _assign_lattice(
    blobs: list[dict],
    n: int,
    row_cells: list[tuple[int, int]] | None = None,
    col_cells: list[tuple[int, int]] | None = None,
) -> tuple[dict, dict, bool, str]:
    """Place each candidate pore into the lattice cell its centroid falls in."""
    if not blobs:
        return {}, {}, False, "No pores detected inside the grid region."

    xs = np.array([b["centroid"][0] for b in blobs])
    ys = np.array([b["centroid"][1] for b in blobs])

    if row_cells and col_cells:
        row_centers = np.array([(a + b) / 2 for a, b in row_cells])
        col_centers = np.array([(a + b) / 2 for a, b in col_cells])
        locate_r = lambda v: _cell_index(v, row_cells)      # noqa: E731
        locate_c = lambda v: _cell_index(v, col_cells)      # noqa: E731
    else:
        row_centers = _cluster_1d(ys, n)
        col_centers = _cluster_1d(xs, n)
        locate_r = lambda v: int(np.argmin(np.abs(row_centers - v)))  # noqa: E731
        locate_c = lambda v: int(np.argmin(np.abs(col_centers - v)))  # noqa: E731

    assigned: dict[tuple[int, int], dict] = {}
    for b in blobs:
        c = locate_c(b["centroid"][0])
        r = locate_r(b["centroid"][1])
        if r is None or c is None:
            continue                      # outside every cell: a sliver
        # Keep the largest blob if two land on the same cell.
        prev = assigned.get((r, c))
        if prev is None or b["area"] > prev["area"]:
            assigned[(r, c)] = b

    expected = {(r, c) for r in range(n) for c in range(n)}
    missing = sorted(expected - set(assigned))
    complete = not missing

    if complete:
        message = f"Complete {n}x{n} lattice."
    else:
        rows_missing = sorted({r for r, _ in missing})
        cols_missing = sorted({c for _, c in missing})
        message = (
            f"{len(missing)} of {n * n} lattice positions had no open pore "
            f"(rows {rows_missing}, columns {cols_missing}). Positions that are "
            "fully fused are reported as closed pores."
        )
        for r, c in missing:
            assigned[(r, c)] = {
                "centroid": (float(col_centers[c]), float(row_centers[r])),
                "area": 0.0,
                "contour": None,
            }

    return assigned, {"rows": row_centers, "cols": col_centers}, complete, message


def _cell_mask(shape: tuple[int, ...],
               row_cells: list[tuple[int, int]] | None,
               col_cells: list[tuple[int, int]] | None) -> np.ndarray | None:
    """Union of the lattice cell rectangles, or None if the walls were unclear."""
    if not row_cells or not col_cells:
        return None
    mask = np.zeros(shape[:2], np.uint8)
    for y0, y1 in row_cells:
        for x0, x1 in col_cells:
            mask[max(0, y0):y1 + 1, max(0, x0):x1 + 1] = 255
    return mask


def _cell_index(value: float, cells: list[tuple[int, int]]) -> int | None:
    for i, (lo, hi) in enumerate(cells):
        if lo <= value <= hi:
            return i
    return None


def _cluster_1d(values: np.ndarray, k: int) -> np.ndarray:
    """k evenly-ordered cluster centres along one axis (1-D k-means, seeded by
    quantiles so it converges deterministically)."""
    if values.size == 0:
        return np.zeros(k)
    if values.size <= k:
        centers = np.sort(values)
        return np.pad(centers, (0, k - centers.size), mode="edge")

    centers = np.quantile(values, np.linspace(0.5 / k, 1 - 0.5 / k, k))
    for _ in range(50):
        labels = np.argmin(np.abs(values[:, None] - centers[None, :]), axis=1)
        new = centers.copy()
        for i in range(k):
            sel = values[labels == i]
            if sel.size:
                new[i] = sel.mean()
        if np.allclose(new, centers, atol=1e-3):
            break
        centers = new
    return np.sort(centers)


def _refine(blob: dict, final_mask: np.ndarray, row: int, col: int,
            cell: tuple[int, int, int, int] | None = None) -> Pore:
    """Re-run contour detection on a window around one expected pore."""
    cx, cy = blob["centroid"]
    flags: list[str] = []

    if blob["contour"] is None:
        return Pore(row, col, (cx, cy), None, 0.0, 0.0, 0.0, 0.0, True,
                    ["closed"])

    x, y, w, h = cv2.boundingRect(blob["contour"])
    pad = int(max(w, h) * 0.35) + 6
    H, W = final_mask.shape[:2]
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(W, x + w + pad), min(H, y + h + pad)

    if cell is not None:
        cx0, cy0, cx1, cy1 = cell
        x0, y0 = max(x0, cx0), max(y0, cy0)
        x1, y1 = min(x1, cx1 + 1), min(y1, cy1 + 1)
        if x1 - x0 < 3 or y1 - y0 < 3:
            return Pore(row, col, (cx, cy), None, 0.0, 0.0, blob["area"], 0.0,
                        True, ["closed"])

    # Take the component containing this pore's centroid, rather than every
    # contour in the window: the padding pulls in slivers of the neighbouring
    # pores, and picking "nearest centroid" among nested contours can select the
    # wrong one.
    window = cv2.bitwise_not(final_mask[y0:y1, x0:x1])
    count, labels = cv2.connectedComponents(window, 8)
    local_x = int(round(cx - x0))
    local_y = int(round(cy - y0))
    if count <= 1 or not (0 <= local_y < labels.shape[0] and 0 <= local_x < labels.shape[1]):
        return Pore(row, col, (cx, cy), None, 0.0, 0.0, blob["area"], 0.0, True,
                    ["closed"])

    label = int(labels[local_y, local_x])
    if label == 0:
        return Pore(row, col, (cx, cy), None, 0.0, 0.0, blob["area"], 0.0, True,
                    ["closed"])

    component = (labels == label).astype(np.uint8)
    contours, _ = cv2.findContours(
        component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        return Pore(row, col, (cx, cy), None, 0.0, 0.0, blob["area"], 0.0, True,
                    ["closed"])

    best = max(contours, key=cv2.contourArea)
    best = best + np.array([[x0, y0]], dtype=best.dtype)

    area = float(cv2.contourArea(best))
    perimeter = float(cv2.arcLength(best, True))
    hull_area = float(cv2.contourArea(cv2.convexHull(best)))
    solidity = area / hull_area if hull_area > 0 else 0.0

    # The cross-checks that can actually catch a bad segmentation. Comparing Pr
    # against C cannot: C == pi/(4*Pr) identically for any area and perimeter.
    if blob["area"] > 0 and abs(area - blob["area"]) / blob["area"] > config.AREA_AGREEMENT_TOL:
        flags.append("pass1_pass2_area_mismatch")
    if solidity < config.SOLIDITY_MIN:
        flags.append("low_solidity")

    m = cv2.moments(best)
    centroid = (m["m10"] / m["m00"], m["m01"] / m["m00"]) if m["m00"] else (cx, cy)

    return Pore(row, col, centroid, best, area, perimeter,
                float(blob["area"]), solidity, False, flags)


def _centroid_distance(contour: np.ndarray, point: tuple[float, float]) -> float:
    m = cv2.moments(contour)
    if m["m00"] == 0:
        return 1e9
    return float(np.hypot(m["m10"] / m["m00"] - point[0],
                          m["m01"] / m["m00"] - point[1]))
