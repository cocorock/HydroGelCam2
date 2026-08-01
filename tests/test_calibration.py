"""Calibration: corner detection, intrinsic recovery, and pixel -> mm scale."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.calib import geometry, intrinsic, scale as scale_mod
from tests import synthetic


def test_finds_chessboard_corners():
    board = synthetic.chessboard(cols=9, rows=6, square_px=60)
    corners = intrinsic.find_corners(board, 9, 6)
    assert corners is not None
    assert corners.shape == (9 * 6, 1, 2)


def test_scale_recovers_known_millimetres_per_pixel():
    square_px, square_mm = 60, 5.0
    board = synthetic.chessboard(cols=9, rows=6, square_px=square_px)

    out = scale_mod.compute(board, cols=9, rows=6, square_mm=square_mm)

    expected = square_mm / square_px
    assert out["mm_per_px_x"] == pytest.approx(expected, rel=0.01)
    assert out["mm_per_px_y"] == pytest.approx(expected, rel=0.01)
    assert out["anisotropy"] < 0.01
    # The sanity readout is what catches a mistyped checker size.
    assert out["board_width_mm"] == pytest.approx((9 - 1) * square_mm, rel=0.01)
    assert out["board_height_mm"] == pytest.approx((6 - 1) * square_mm, rel=0.01)


def test_wrong_checker_size_scales_the_readout_proportionally():
    """Entering twice the true square size doubles every reported millimetre."""
    board = synthetic.chessboard(cols=9, rows=6, square_px=60)
    right = scale_mod.compute(board, cols=9, rows=6, square_mm=5.0)
    wrong = scale_mod.compute(board, cols=9, rows=6, square_mm=10.0)
    assert wrong["mm_per_px_x"] == pytest.approx(right["mm_per_px_x"] * 2, rel=1e-6)
    assert wrong["board_width_mm"] == pytest.approx(right["board_width_mm"] * 2, rel=1e-6)


def test_scale_rejects_an_image_with_no_board():
    blank = np.full((400, 600, 3), 128, np.uint8)
    with pytest.raises(ValueError, match="No 9x6 chessboard"):
        scale_mod.compute(blank, cols=9, rows=6, square_mm=5.0)


def test_intrinsic_solve_recovers_the_known_camera_matrix():
    square_mm = 25.0
    views = synthetic.chessboard_views(cols=9, rows=6, square_mm=square_mm)

    session = intrinsic.IntrinsicSession(cols=9, rows=6, square_mm=square_mm)
    for view in views:
        assert session.add(view)["accepted"], "every synthetic view must be found"

    result = session.solve()
    assert result["n_frames"] == len(views)
    assert result["rms_px"] < 0.5

    K = np.asarray(result["K"])
    assert K.shape == (3, 3)
    assert K[0, 0] == pytest.approx(synthetic.DEFAULT_K[0, 0], rel=0.02)  # fx
    assert K[1, 1] == pytest.approx(synthetic.DEFAULT_K[1, 1], rel=0.02)  # fy
    assert K[0, 2] == pytest.approx(synthetic.DEFAULT_K[0, 2], rel=0.05)  # cx
    assert K[1, 2] == pytest.approx(synthetic.DEFAULT_K[1, 2], rel=0.05)  # cy

    assert all(f["error_px"] is not None for f in session.state()["frames"])


def test_removing_a_frame_invalidates_the_previous_solution():
    views = synthetic.chessboard_views(cols=9, rows=6)
    session = intrinsic.IntrinsicSession(cols=9, rows=6, square_mm=25.0)
    for view in views[:5]:
        session.add(view)
    session.solve()
    assert session.result is not None

    session.remove(0)
    assert session.result is None, "a stale solution must not survive an edit"
    assert len(session.frames) == 4


def test_intrinsic_rejects_a_frame_whose_size_changed():
    views = synthetic.chessboard_views(cols=9, rows=6)
    session = intrinsic.IntrinsicSession(cols=9, rows=6, square_mm=25.0)
    session.add(views[0])

    resized = cv2.resize(views[1], (640, 480))
    result = session.add(resized)
    assert result["accepted"] is False
    assert "size changed" in result["message"]


# ---------------------------------------------------------------- geometry


def test_uncalibrated_scale_is_the_identity():
    s = geometry.from_profile(None)
    assert s.calibrated is False
    assert s.dx_mm(100) == 100
    assert s.dy_mm(100) == 100


def test_scalar_profile_converts_lengths_and_areas():
    s = geometry.from_profile({
        "name": "test", "mode": "scalar",
        "mm_per_px_x": 0.05, "mm_per_px_y": 0.05,
    })
    assert s.calibrated is True
    assert s.dx_mm(200) == pytest.approx(10.0)

    square = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], np.float32)
    assert s.area_mm2(square) == pytest.approx(25.0)      # 5 mm x 5 mm
    assert s.perimeter_mm(square) == pytest.approx(20.0)


def test_homography_profile_measures_in_the_board_plane():
    """A profile in homography mode measures on the rectified plane."""
    H = np.array([[0.05, 0, 0], [0, 0.05, 0], [0, 0, 1]], float)
    s = geometry.from_profile({
        "name": "h", "mode": "homography", "H_json": H.tolist(),
        "mm_per_px_x": 1.0, "mm_per_px_y": 1.0,
    })
    square = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], np.float32)
    assert s.area_mm2(square) == pytest.approx(25.0)
    assert s.dx_mm(200) == pytest.approx(10.0)


def test_polyline_area_matches_a_known_triangle():
    s = geometry.from_profile({
        "name": "t", "mode": "scalar", "mm_per_px_x": 1.0, "mm_per_px_y": 1.0,
    })
    triangle = np.array([[0, 0], [10, 0], [0, 10]], float)
    assert s.polyline_area_mm2(triangle) == pytest.approx(50.0)
