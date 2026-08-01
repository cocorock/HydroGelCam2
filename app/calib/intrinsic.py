"""Stage A: intrinsic matrix and distortion coefficients from a chessboard set."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

FIND_FLAGS = (
    cv2.CALIB_CB_ADAPTIVE_THRESH
    | cv2.CALIB_CB_NORMALIZE_IMAGE
    | cv2.CALIB_CB_FAST_CHECK
)
SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
SUBPIX_WINDOW = (11, 11)


def object_points(cols: int, rows: int, square_mm: float) -> np.ndarray:
    """Ideal board corners in board coordinates, in millimetres, z = 0."""
    grid = np.zeros((rows * cols, 3), np.float32)
    grid[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    return grid * float(square_mm)


def find_corners(image: np.ndarray, cols: int, rows: int) -> np.ndarray | None:
    """Locate inner chessboard corners, refined to sub-pixel. None if not found."""
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ok, corners = cv2.findChessboardCorners(gray, (cols, rows), FIND_FLAGS)
    if not ok:
        # FAST_CHECK bails early on blurry or oblique views; retry without it.
        ok, corners = cv2.findChessboardCorners(
            gray, (cols, rows),
            cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
    if not ok:
        return None
    return cv2.cornerSubPix(gray, corners, SUBPIX_WINDOW, (-1, -1), SUBPIX_CRITERIA)


def draw_corners(image: np.ndarray, corners: np.ndarray,
                 cols: int, rows: int) -> np.ndarray:
    out = image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.drawChessboardCorners(out, (cols, rows), corners, True)
    return out


@dataclass
class Frame:
    index: int
    corners: np.ndarray
    thumb_png: bytes
    error_px: float | None = None


@dataclass
class IntrinsicSession:
    """Accumulates chessboard views, then solves for K and dist.

    Held in memory so the user can shoot views, see per-frame reprojection error,
    delete the bad ones and re-solve without re-shooting the whole set.
    """

    cols: int
    rows: int
    square_mm: float
    image_size: tuple[int, int] | None = None
    frames: list[Frame] = field(default_factory=list)
    result: dict[str, Any] | None = None
    _next_index: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def reset(self, cols: int, rows: int, square_mm: float) -> None:
        with self._lock:
            self.cols, self.rows, self.square_mm = cols, rows, square_mm
            self.frames.clear()
            self.image_size = None
            self.result = None
            self._next_index = 0

    def add(self, image: np.ndarray) -> dict[str, Any]:
        corners = find_corners(image, self.cols, self.rows)
        if corners is None:
            return {
                "accepted": False,
                "message": (
                    f"No {self.cols}x{self.rows} chessboard found. Check the inner-"
                    "corner counts and that the whole board is in frame."
                ),
            }
        with self._lock:
            h, w = image.shape[:2]
            if self.image_size is None:
                self.image_size = (w, h)
            elif self.image_size != (w, h):
                return {
                    "accepted": False,
                    "message": (
                        "Frame size changed mid-calibration "
                        f"({self.image_size[0]}x{self.image_size[1]} -> {w}x{h}). "
                        "Reset the set or restore the previous resolution."
                    ),
                }

            overlay = draw_corners(image, corners, self.cols, self.rows)
            thumb = _thumbnail_png(overlay)
            frame = Frame(index=self._next_index, corners=corners, thumb_png=thumb)
            self._next_index += 1
            self.frames.append(frame)
            self.result = None
        return {"accepted": True, "index": frame.index, "count": len(self.frames)}

    def remove(self, index: int) -> None:
        with self._lock:
            self.frames = [f for f in self.frames if f.index != index]
            self.result = None

    def solve(self) -> dict[str, Any]:
        with self._lock:
            if len(self.frames) < 3 or self.image_size is None:
                raise ValueError(
                    f"Need at least 3 views to calibrate; have {len(self.frames)}."
                )
            objp = object_points(self.cols, self.rows, self.square_mm)
            obj_points = [objp for _ in self.frames]
            img_points = [f.corners for f in self.frames]

            rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
                obj_points, img_points, self.image_size, None, None
            )

            for frame, rvec, tvec in zip(self.frames, rvecs, tvecs):
                projected, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
                frame.error_px = float(
                    cv2.norm(frame.corners, projected, cv2.NORM_L2)
                    / np.sqrt(len(projected))
                )

            self.result = {
                "rms_px": float(rms),
                "K": K.tolist(),
                "dist": dist.ravel().tolist(),
                "image_size": list(self.image_size),
                "n_frames": len(self.frames),
                "per_frame": [
                    {"index": f.index, "error_px": f.error_px} for f in self.frames
                ],
            }
            return self.result

    def state(self) -> dict[str, Any]:
        with self._lock:
            return {
                "cols": self.cols,
                "rows": self.rows,
                "square_mm": self.square_mm,
                "image_size": list(self.image_size) if self.image_size else None,
                "frames": [
                    {"index": f.index, "error_px": f.error_px} for f in self.frames
                ],
                "result": self.result,
            }

    def thumbnail(self, index: int) -> bytes | None:
        with self._lock:
            for f in self.frames:
                if f.index == index:
                    return f.thumb_png
        return None


def _thumbnail_png(image: np.ndarray, width: int = 320) -> bytes:
    h, w = image.shape[:2]
    scale = width / float(w)
    small = cv2.resize(image, (width, max(1, int(round(h * scale)))),
                       interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", small)
    return buf.tobytes() if ok else b""


session = IntrinsicSession(cols=9, rows=6, square_mm=5.0)
