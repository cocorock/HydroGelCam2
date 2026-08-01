"""Pixel -> millimetre conversions, driven by a stored calibration profile.

Every measurement in tabs 2-4 goes through this module, so switching a profile
between scalar and homography mode changes nothing downstream. In homography
mode points are mapped into the board plane before any length or area is taken,
which is what makes a tilted view measurable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class Scale:
    mm_per_px_x: float = 1.0
    mm_per_px_y: float = 1.0
    H: np.ndarray | None = None          # image px -> board plane mm
    K: np.ndarray | None = None
    dist: np.ndarray | None = None
    calibrated: bool = False
    name: str = "uncalibrated (1 px = 1 mm)"

    # -------------------------------------------------------------- lengths

    @property
    def mm_per_px(self) -> float:
        return (self.mm_per_px_x + self.mm_per_px_y) / 2.0

    def dx_mm(self, px: float) -> float:
        """Horizontal distance. Homography mode measures along the image x axis."""
        if self.H is not None:
            a, b = self.to_mm([[0.0, 0.0], [float(px), 0.0]])
            return float(np.linalg.norm(b - a))
        return float(px) * self.mm_per_px_x

    def dy_mm(self, px: float) -> float:
        if self.H is not None:
            a, b = self.to_mm([[0.0, 0.0], [0.0, float(px)]])
            return float(np.linalg.norm(b - a))
        return float(px) * self.mm_per_px_y

    # -------------------------------------------------------------- mapping

    def to_mm(self, points: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
        """Map image points (N,2) into millimetres."""
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
        if self.H is not None:
            out = cv2.perspectiveTransform(pts, self.H)
            return out.reshape(-1, 2)
        flat = pts.reshape(-1, 2).copy()
        flat[:, 0] *= self.mm_per_px_x
        flat[:, 1] *= self.mm_per_px_y
        return flat

    # -------------------------------------------------------------- contours

    def area_mm2(self, contour: np.ndarray) -> float:
        """Enclosed area of a closed polygon, in mm^2."""
        pts = self.to_mm(np.asarray(contour, dtype=np.float64).reshape(-1, 2))
        return float(abs(cv2.contourArea(pts.astype(np.float32))))

    def perimeter_mm(self, contour: np.ndarray, closed: bool = True) -> float:
        pts = self.to_mm(np.asarray(contour, dtype=np.float64).reshape(-1, 2))
        return float(cv2.arcLength(pts.astype(np.float32), closed))

    def polyline_area_mm2(self, points: np.ndarray) -> float:
        """Shoelace area of an open polyline closed back on itself.

        Used for the collapse sag region, which is bounded above by the straight
        pillar-top line and below by the measured filament profile.
        """
        pts = self.to_mm(np.asarray(points, dtype=np.float64).reshape(-1, 2))
        x, y = pts[:, 0], pts[:, 1]
        return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


UNCALIBRATED = Scale()


def from_profile(profile: dict[str, Any] | None) -> Scale:
    """Build a Scale from a calibration row. None yields the 1 px = 1 mm identity."""
    if not profile:
        return UNCALIBRATED

    K = profile.get("K_json")
    dist = profile.get("dist_json")
    H = profile.get("H_json")
    mode = profile.get("mode") or "scalar"

    mx = profile.get("mm_per_px_x")
    my = profile.get("mm_per_px_y")

    use_H = mode == "homography" and H is not None
    if not use_H and (mx is None or my is None):
        return UNCALIBRATED

    return Scale(
        mm_per_px_x=float(mx) if mx else 1.0,
        mm_per_px_y=float(my) if my else 1.0,
        H=np.asarray(H, dtype=np.float64).reshape(3, 3) if use_H else None,
        K=np.asarray(K, dtype=np.float64).reshape(3, 3) if K else None,
        dist=np.asarray(dist, dtype=np.float64).ravel() if dist else None,
        calibrated=True,
        name=profile.get("name", "unnamed"),
    )


def undistort(image: np.ndarray, scale: Scale) -> np.ndarray:
    if scale.K is None or scale.dist is None:
        return image
    return cv2.undistort(image, scale.K, scale.dist)
