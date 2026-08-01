"""The collapse platform's own width is the length reference, so a cropped ROI
must be reported rather than silently rescaling every result."""

from __future__ import annotations

import pytest

from app.analysis import collapse
from app.calib.geometry import Scale
from tests import synthetic


def scale_for(px_per_mm: float) -> Scale:
    return Scale(mm_per_px_x=1 / px_per_mm, mm_per_px_y=1 / px_per_mm,
                 calibrated=True)


def test_full_frame_measures_correctly():
    px_per_mm = 20.0
    image, truth = synthetic.collapse_platform(px_per_mm=px_per_mm)
    out = collapse.analyze(image, None, {}, scale_for(px_per_mm))

    assert not any("do not line up" in f for f in out["flags"])
    for row, expected in zip(out["results"]["rows"], truth["a_sag_mm2"]):
        assert row["raw"]["a_sag_mm2"] == pytest.approx(expected, rel=0.02)


def test_cropped_platform_is_flagged():
    px_per_mm = 20.0
    image, _ = synthetic.collapse_platform(px_per_mm=px_per_mm)
    h, w = image.shape[:2]
    # A default 15 % inset would cut into the wide outer pillars.
    roi = {"x": int(w * 0.15), "y": 0, "w": int(w * 0.70), "h": h}

    out = collapse.analyze(image, roi, {}, scale_for(px_per_mm))
    assert any("do not line up" in f for f in out["flags"]), \
        "a clipped platform must be reported, not silently rescaled"
    # The platform's implied scale also disagrees with the calibration.
    assert out["results"]["scale_cross_check"]["warn"] is True


def test_scale_cross_check_catches_a_mismatched_calibration():
    """A calibration from the wrong plane disagrees with the platform's width."""
    px_per_mm = 20.0
    image, _ = synthetic.collapse_platform(px_per_mm=px_per_mm)
    # Calibration claims 40 px/mm; the platform says 20.
    out = collapse.analyze(image, None, {}, scale_for(40.0))

    check = out["results"]["scale_cross_check"]
    assert check["warn"] is True
    assert check["disagreement"] == pytest.approx(1.0, rel=0.05)
