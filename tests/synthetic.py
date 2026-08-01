"""Synthetic sample images with known ground truth.

These exist so the metrics can be checked against values that are known exactly
rather than eyeballed: a serpentine of a stated thickness must give that
thickness back, a square pore must give C = pi/4, a parabolic sag must give
(2/3) x depth x span.
"""

from __future__ import annotations

import cv2
import numpy as np


def vignette(image: np.ndarray, strength: float, floor: float = 0.25,
             kind: str = "radial") -> np.ndarray:
    """Multiply by a brightness falloff, as an endoscope's optics would."""
    if strength <= 0:
        return image
    h, w = image.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    if kind == "radial":
        r = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
        field = np.clip(1.0 - strength * r ** 2, floor, 1.0)
    else:
        field = np.clip(1.0 - strength * ((xx / w) * 0.7 + (yy / h) * 0.5), floor, 1.0)
    return np.clip(image * field, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- tab 2


def serpentine(
    n_filaments: int = 6,
    thickness_px: int = 24,
    blur: int = 3,
    dark: bool = False,
    vignette_strength: float = 0.0,
    vignette_kind: str = "radial",
    break_filament: int | None = None,
) -> tuple[np.ndarray, dict]:
    """Horizontal filaments of exactly `thickness_px`, on a plain background."""
    h, w = 700, 1000
    background, material = (210, 30) if dark else (30, 210)
    img = np.full((h, w), background, np.uint8)

    for i in range(n_filaments):
        y = 120 + i * 80
        top = y - thickness_px // 2
        cv2.rectangle(img, (150, top), (850, top + thickness_px - 1), material, -1)
        if break_filament is not None and i == break_filament:
            cv2.rectangle(img, (460, top - 2), (560, top + thickness_px + 1),
                          background, -1)

    img = cv2.GaussianBlur(img, (blur, blur), 0)
    img = vignette(img, vignette_strength, 0.25, vignette_kind)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), {
        "roi": {"x": 160, "y": 95, "w": 680, "h": 520},
        "thickness_px": thickness_px,
        "n_filaments": n_filaments,
        "polarity": "dark" if dark else "bright",
    }


# ---------------------------------------------------------------- tab 3


def pore_grid(
    fds_mm: tuple[float, ...] = (1, 2, 3, 4, 5),
    px_per_mm: float = 77.0,
    wall_mm: float = 0.41,
    fuse_smallest: bool = False,
    dark: bool = False,
    vignette_strength: float = 0.0,
) -> tuple[np.ndarray, dict]:
    """0-90 lattice whose pores grow from bottom-left to top-right."""
    wall = max(1, int(round(wall_mm * px_per_mm)))
    margin = int(2 * px_per_mm)

    xs = [margin]
    for fd in fds_mm:
        xs.append(xs[-1] + int(fd * px_per_mm) + wall)
    ys = [margin]
    for fd in reversed(fds_mm):
        ys.append(ys[-1] + int(fd * px_per_mm) + wall)

    w, h = xs[-1] + margin, ys[-1] + margin
    background, material = (205, 25) if dark else (25, 205)
    img = np.full((h, w), background, np.uint8)

    for x in xs:
        cv2.rectangle(img, (x - wall, ys[0] - wall), (x - 1, ys[-1] - 1), material, -1)
    for y in ys:
        cv2.rectangle(img, (xs[0] - wall, y - wall), (xs[-1] - 1, y - 1), material, -1)
    if fuse_smallest:
        cv2.rectangle(img, (xs[0], ys[-2]),
                      (xs[1] - wall - 1, ys[-1] - wall - 1), material, -1)

    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = vignette(img, vignette_strength, 0.3)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), {
        "roi": {"x": xs[0] - wall, "y": ys[0] - wall,
                "w": xs[-1] - xs[0] + wall, "h": ys[-1] - ys[0] + wall},
        "px_per_mm": px_per_mm,
        "fds_mm": list(fds_mm),
        "polarity": "dark" if dark else "bright",
    }


# ---------------------------------------------------------------- tab 4


def _hsv_bgr(hue: int, sat: int, val: int) -> tuple[int, int, int]:
    px = cv2.cvtColor(np.uint8([[[hue, sat, val]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return tuple(int(c) for c in px)


def collapse_platform(
    sag_fraction: tuple[float, ...] = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3),
    px_per_mm: float = 20.0,
    df_px: int = 8,
    broken: tuple[int, ...] = (),
) -> tuple[np.ndarray, dict]:
    """Side view of the ABS platform with a parabolic sag in each gap.

    The sag under a parabola of depth d over span s is exactly (2/3) d s, which
    is what the returned ground truth records.
    """
    pillar_bgr = _hsv_bgr(6, 90, 220)       # pale red ABS
    filament_bgr = _hsv_bgr(115, 180, 200)  # dyed filament
    background_bgr = _hsv_bgr(30, 15, 245)

    pillars_mm = [10, 2, 2, 2, 2, 2, 10]
    gaps_mm = [1, 2, 3, 4, 5, 6]

    w = int(sum(pillars_mm + gaps_mm) * px_per_mm)
    h = int(10 * px_per_mm) + 40
    img = np.full((h, w, 3), background_bgr, np.uint8)

    floor_y = int(40 + 10 * px_per_mm)
    top_y = int(40 + 4 * px_per_mm)
    cv2.rectangle(img, (0, int(40 + 6 * px_per_mm)), (w, floor_y), pillar_bgr, -1)

    spans = []
    cursor = 0.0
    for i, pw in enumerate(pillars_mm):
        x0 = int(cursor * px_per_mm)
        cursor += pw
        x1 = int(cursor * px_per_mm)
        cv2.rectangle(img, (x0, top_y), (x1, floor_y), pillar_bgr, -1)
        spans.append((x0, x1))
        if i < len(gaps_mm):
            cursor += gaps_mm[i]

    analytic: list[float | None] = []
    rasterised: list[float | None] = []
    underside = np.full(w, -1, dtype=int)

    for i in range(len(gaps_mm)):
        gx0, gx1 = spans[i][1], spans[i + 1][0]
        if i in broken:
            analytic.append(None)
            rasterised.append(None)
            continue
        depth = sag_fraction[i] * 6 * px_per_mm
        for x in range(gx0, gx1 + 1):
            t = (x - gx0) / max(gx1 - gx0, 1)
            y = int(top_y + 4 * depth * t * (1 - t))
            cv2.line(img, (x, y - df_px), (x, y), filament_bgr, 1)
            underside[x] = y

        # Two ground truths, because they are not the same number at small
        # spans. The analytic area of a parabola of depth d over span s is
        # (2/3) d s; the drawn image only approximates that, since each column's
        # depth is truncated to a whole pixel. At 1 mm (20 px) the two differ by
        # over 10 %, so a measurement must be checked against what was actually
        # rasterised -- the analytic value is the limit it converges to.
        analytic.append((2 / 3) * (depth / px_per_mm) * ((gx1 - gx0) / px_per_mm))
        depths = [max(0, underside[x] - top_y) for x in range(gx0 + 1, gx1)]
        rasterised.append(float(sum(depths)) / (px_per_mm * px_per_mm))

    for x0, x1 in spans:
        cv2.rectangle(img, (x0, top_y - df_px), (x1, top_y), filament_bgr, -1)

    return img, {
        "px_per_mm": px_per_mm,
        "gaps_mm": gaps_mm,
        "a_sag_mm2": rasterised,
        "a_sag_mm2_analytic": analytic,
        "df_mm": (df_px + 1) / px_per_mm,
        "broken": list(broken),
    }


# ---------------------------------------------------------------- calibration


def chessboard(
    cols: int = 9, rows: int = 6, square_px: int = 60,
    K: np.ndarray | None = None, dist: np.ndarray | None = None,
) -> np.ndarray:
    """A chessboard with `cols` x `rows` inner corners, optionally distorted."""
    w = (cols + 1) * square_px
    h = (rows + 1) * square_px
    board = np.zeros((h, w), np.uint8)
    for r in range(rows + 1):
        for c in range(cols + 1):
            if (r + c) % 2 == 0:
                board[r * square_px:(r + 1) * square_px,
                      c * square_px:(c + 1) * square_px] = 255

    # A quiet border, so findChessboardCorners can see the outer squares.
    pad = square_px
    framed = np.full((h + 2 * pad, w + 2 * pad), 255, np.uint8)
    framed[pad:pad + h, pad:pad + w] = board
    bgr = cv2.cvtColor(framed, cv2.COLOR_GRAY2BGR)

    if K is not None and dist is not None:
        bgr = _distort(bgr, K, dist)
    return bgr


def _distort(image: np.ndarray, K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """Apply lens distortion, so undistort() has something real to undo."""
    h, w = image.shape[:2]
    map_x, map_y = cv2.initUndistortRectifyMap(
        K, dist, None, K, (w, h), cv2.CV_32FC1
    )
    return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))


DEFAULT_K = np.array([[1000.0, 0, 640.0],
                      [0, 1000.0, 480.0],
                      [0, 0, 1.0]])

# Tilts about x and y, in degrees, plus the board's distance in mm. Camera
# calibration cannot recover a focal length from fronto-parallel views alone --
# a plane seen only head-on leaves focal length and distance indistinguishable --
# so every pose here is genuinely oblique, and in more than one direction.
DEFAULT_POSES = (
    (-22, -18, 300), (-20, 12, 310), (-8, 24, 290), (10, -22, 300),
    (18, 16, 320), (24, -6, 280), (-14, 0, 260), (6, 8, 340),
    (-26, 20, 330), (20, -20, 270),
)


def chessboard_views(
    cols: int = 9, rows: int = 6, square_px: int = 60, square_mm: float = 25.0,
    K: np.ndarray | None = None, dist: np.ndarray | None = None,
    poses=DEFAULT_POSES, size: tuple[int, int] = (1280, 960),
) -> list[np.ndarray]:
    """Perspective-correct views of one board seen from several orientations.

    Each view is produced by projecting the board's four physical corners through
    a real pinhole model and warping the flat board image onto them, so
    `calibrateCamera` is given exactly the geometry it expects.
    """
    K = DEFAULT_K if K is None else K
    dist = np.zeros(5) if dist is None else np.asarray(dist, float).ravel()

    flat = chessboard(cols=cols, rows=rows, square_px=square_px)
    bh, bw = flat.shape[:2]
    mm_per_px = square_mm / square_px

    # Board corners in millimetres, centred on the origin so rotations tilt the
    # board about its middle rather than swinging it out of frame.
    half_w, half_h = bw * mm_per_px / 2, bh * mm_per_px / 2
    corners_mm = np.array([
        [-half_w, -half_h, 0], [half_w, -half_h, 0],
        [half_w, half_h, 0], [-half_w, half_h, 0],
    ], dtype=np.float64)
    corners_px = np.array([[0, 0], [bw, 0], [bw, bh], [0, bh]], dtype=np.float32)

    views = []
    for tilt_x, tilt_y, distance in poses:
        rvec, _ = cv2.Rodrigues(
            _euler(np.deg2rad(tilt_x), np.deg2rad(tilt_y)))
        tvec = np.array([[0.0], [0.0], [float(distance)]])
        projected, _ = cv2.projectPoints(corners_mm, rvec, tvec, K, dist)

        H = cv2.getPerspectiveTransform(
            corners_px, projected.reshape(4, 2).astype(np.float32))
        views.append(cv2.warpPerspective(
            flat, H, size, borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255)))
    return views


def _euler(rx: float, ry: float) -> np.ndarray:
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    return Ry @ Rx
