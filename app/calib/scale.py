"""Stage B: pixel -> millimetre scale, measured in the plane the sample sits in.

The checker square size entered here is what sets the absolute scale, so it is
kept independent of the board used for stage A: a coarse board is easier to shoot
from many angles for the intrinsic solve, while a fine board laid on the glass
plate gives a better in-plane scale.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from app.calib.intrinsic import find_corners, draw_corners, object_points


def undistort(image: np.ndarray, K: Any, dist: Any) -> np.ndarray:
    """Apply the stage-A correction. Returns the input unchanged if K is absent."""
    if K is None or dist is None:
        return image
    K_arr = np.asarray(K, dtype=np.float64).reshape(3, 3)
    d_arr = np.asarray(dist, dtype=np.float64).ravel()
    return cv2.undistort(image, K_arr, d_arr)


def compute(
    image: np.ndarray,
    cols: int,
    rows: int,
    square_mm: float,
    K: Any = None,
    dist: Any = None,
    mode: str = "scalar",
) -> dict[str, Any]:
    """Derive mm-per-pixel (and optionally a plane homography) from one board image.

    Returns the scale plus a sanity readout of the whole board's measured size, so
    a mistyped square size is obvious before the profile is saved.
    """
    flat = undistort(image, K, dist)
    corners = find_corners(flat, cols, rows)
    if corners is None:
        raise ValueError(
            f"No {cols}x{rows} chessboard found in the scale image. Check the "
            "inner-corner counts and that the board lies flat in the sample plane."
        )

    grid = corners.reshape(rows, cols, 2)

    # Spacing between horizontally and vertically adjacent corners, in pixels.
    dx = np.linalg.norm(grid[:, 1:, :] - grid[:, :-1, :], axis=2)
    dy = np.linalg.norm(grid[1:, :, :] - grid[:-1, :, :], axis=2)
    px_per_square_x = float(np.mean(dx))
    px_per_square_y = float(np.mean(dy))

    mm_per_px_x = float(square_mm) / px_per_square_x
    mm_per_px_y = float(square_mm) / px_per_square_y
    mean = (mm_per_px_x + mm_per_px_y) / 2.0
    anisotropy = abs(mm_per_px_x - mm_per_px_y) / mean if mean else 0.0

    H = None
    if mode == "homography":
        objp = object_points(cols, rows, square_mm)[:, :2].astype(np.float64)
        H_mat, _ = cv2.findHomography(
            corners.reshape(-1, 2).astype(np.float64), objp, method=0
        )
        if H_mat is None:
            raise ValueError("Homography fit failed; try scalar mode.")
        H = H_mat.tolist()

    return {
        "mode": mode,
        "mm_per_px_x": mm_per_px_x,
        "mm_per_px_y": mm_per_px_y,
        "anisotropy": anisotropy,
        "H": H,
        "px_per_square_x": px_per_square_x,
        "px_per_square_y": px_per_square_y,
        # Sanity readout: the full board spans (cols-1) x (rows-1) squares.
        "board_width_mm": px_per_square_x * (cols - 1) * mm_per_px_x,
        "board_height_mm": px_per_square_y * (rows - 1) * mm_per_px_y,
        "board_span_px": [
            px_per_square_x * (cols - 1),
            px_per_square_y * (rows - 1),
        ],
        "overlay": draw_corners(flat, corners, cols, rows),
    }
