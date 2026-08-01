"""The ROI is padded, not expanded: nothing outside it may affect a result."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from app import config
from app.analysis import uniformity
from app.calib.geometry import Scale
from app.pipeline.common import preprocess
from tests import synthetic


def scale_1px() -> Scale:
    return Scale(mm_per_px_x=1.0, mm_per_px_y=1.0, calibrated=True)


def test_crop_is_the_roi_plus_padding():
    image, truth = synthetic.serpentine()
    roi = truth["roi"]
    padding = 85

    seg = preprocess(image, roi, padding=padding, polarity="bright")

    assert seg.crop.shape[0] == roi["h"] + 2 * padding
    assert seg.crop.shape[1] == roi["w"] + 2 * padding
    assert seg.padding == padding
    assert seg.roi_in_crop.as_dict() == {
        "x": padding, "y": padding, "w": roi["w"], "h": roi["h"],
    }
    # Crop coordinates map back to the frame through the offset.
    assert seg.offset == (roi["x"] - padding, roi["y"] - padding)


@pytest.mark.parametrize("polarity,expected", [("bright", 0), ("dark", 255)])
def test_padding_colour_follows_material_appearance(polarity, expected):
    image, truth = synthetic.serpentine(dark=(polarity == "dark"))
    seg = preprocess(image, truth["roi"], padding=30, polarity=polarity)

    assert seg.pad_value == expected
    border = seg.gray[0, 0]
    assert border == expected
    # Padding is never material, whichever way round the polarity is.
    assert seg.binary[~seg.valid].max() == 0


def test_content_outside_the_roi_cannot_change_the_result():
    """A bright blob just outside the ROI must not move a single measurement."""
    clean, truth = synthetic.serpentine()
    roi = truth["roi"]

    noisy = clean.copy()
    # A glare patch hugging the ROI's left edge, entirely outside it.
    cv2.rectangle(noisy, (roi["x"] - 60, roi["y"] + 40),
                  (roi["x"] - 5, roi["y"] + 300), (255, 255, 255), -1)

    params = {"n_filaments": 6, "n_positions": 5, "polarity": "bright",
              "nozzle_id_mm": 1.0}
    a = uniformity.analyze(clean, roi, params, scale_1px())["results"]
    b = uniformity.analyze(noisy, roi, params, scale_1px())["results"]

    assert b["mean_mm"] == pytest.approx(a["mean_mm"], rel=1e-9)
    assert b["sd_mm"] == pytest.approx(a["sd_mm"], rel=1e-9)
    assert b["n_included"] == a["n_included"]


def test_histogram_uses_only_roi_pixels():
    """The padding is excluded, so its huge single-value spike cannot dominate."""
    image, truth = synthetic.serpentine()
    roi = truth["roi"]

    small = preprocess(image, roi, padding=10, polarity="bright")
    large = preprocess(image, roi, padding=200, polarity="bright")

    # Twenty times the padding is a vast number of extra identical pixels. If
    # they entered the histogram the threshold would move; it must not.
    assert small.meta["threshold"]["n_pixels"] == roi["w"] * roi["h"]
    assert large.meta["threshold"]["n_pixels"] == roi["w"] * roi["h"]
    assert large.threshold == pytest.approx(small.threshold, abs=1.0)


def test_measurements_are_stable_across_padding_widths():
    image, truth = synthetic.serpentine(vignette_strength=0.55)
    params = {"n_filaments": 6, "n_positions": 5, "polarity": "bright",
              "nozzle_id_mm": 1.0}

    means = []
    for padding in (10, 40, 85, 150):
        out = uniformity.analyze(image, truth["roi"],
                                 {**params, "padding": padding}, scale_1px())
        assert out["results"]["n_included"] == 30
        means.append(out["results"]["mean_mm"])

    assert max(means) - min(means) < 0.05, "padding width must not shift the result"


def test_material_touching_the_roi_edge_is_still_measured():
    """A filament against the ROI boundary is the case padding exists to fix."""
    image, _ = synthetic.serpentine(thickness_px=40, vignette_strength=0.55)
    # Top edge cuts within a few pixels of the first filament (centre y = 120).
    roi = {"x": 160, "y": 96, "w": 680, "h": 500}

    out = uniformity.analyze(
        image, roi,
        {"n_filaments": 6, "n_positions": 5, "polarity": "bright",
         "nozzle_id_mm": 1.0},
        scale_1px(),
    )
    r = out["results"]
    assert r["n_included"] == 30, "the edge-most filament must not be lost"
    assert r["mean_mm"] == pytest.approx(40.0, rel=0.02)


def test_cleanup_kernel_is_configurable():
    image, truth = synthetic.serpentine()
    params = {"n_filaments": 6, "n_positions": 5, "polarity": "bright",
              "nozzle_id_mm": 1.0}

    for kernel in (3, 9, 15):
        out = uniformity.analyze(image, truth["roi"],
                                 {**params, "final_kernel": kernel}, scale_1px())
        assert out["results"]["n_included"] == 30
        assert out["results"]["mean_mm"] == pytest.approx(24.0, rel=0.03)


def test_debug_trace_shows_the_padded_crop():
    image, truth = synthetic.serpentine()
    out = uniformity.analyze(
        image, truth["roi"],
        {"n_filaments": 6, "n_positions": 5, "polarity": "bright"},
        scale_1px(), debug=True,
    )
    names = [s["name"] for s in out["debug"]]
    assert "2_roi" in names
    assert "2b_padded" in names
