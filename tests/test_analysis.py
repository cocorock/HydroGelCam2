"""Metric correctness against synthetic images with known ground truth."""

from __future__ import annotations

import math

import pytest

from app.analysis import collapse, fusion, uniformity
from app.calib.geometry import Scale
from tests import synthetic


def scale_for(px_per_mm: float) -> Scale:
    return Scale(mm_per_px_x=1 / px_per_mm, mm_per_px_y=1 / px_per_mm,
                 calibrated=True, name="synthetic")


# ============================================================ tab 2


UNIFORMITY_CASES = [
    pytest.param(0.0, "radial", False, 24, 3, id="flat-bright"),
    pytest.param(0.55, "radial", False, 24, 3, id="radial-vignette"),
    pytest.param(0.80, "radial", False, 24, 3, id="heavy-vignette"),
    pytest.param(0.60, "gradient", False, 24, 3, id="gradient-vignette"),
    pytest.param(0.55, "radial", True, 24, 3, id="dark-material"),
    pytest.param(0.55, "radial", False, 40, 3, id="thick-filament"),
    pytest.param(0.55, "radial", False, 24, 9, id="soft-focus"),
]


@pytest.mark.parametrize("vig,kind,dark,thickness,blur", UNIFORMITY_CASES)
def test_uniformity_recovers_known_thickness(vig, kind, dark, thickness, blur):
    px_per_mm = 10.0
    image, truth = synthetic.serpentine(
        thickness_px=thickness, blur=blur, dark=dark,
        vignette_strength=vig, vignette_kind=kind,
    )
    out = uniformity.analyze(
        image, truth["roi"],
        {"n_filaments": 6, "n_positions": 5,
         "polarity": truth["polarity"], "nozzle_id_mm": 2.0},
        scale_for(px_per_mm),
    )
    r = out["results"]
    expected_mm = thickness / px_per_mm

    assert r["n_included"] == 30, "every filament should be sampled at every position"
    assert r["mean_mm"] == pytest.approx(expected_mm, rel=0.01)
    assert r["spreading_ratio"] == pytest.approx(expected_mm / 2.0, rel=0.01)
    # A constant-width filament is perfectly uniform.
    assert r["uniformity_index"] > 0.99
    assert r["ui_valid"] is True


def test_uniformity_is_precision_not_accuracy():
    """UI stays high for a uniformly over-wide filament; SR is what catches it."""
    image, truth = synthetic.serpentine(thickness_px=40)
    out = uniformity.analyze(
        image, truth["roi"],
        {"n_filaments": 6, "n_positions": 5, "polarity": "bright",
         "nozzle_id_mm": 2.0},
        scale_for(10.0),
    )
    r = out["results"]
    assert r["uniformity_index"] > 0.99          # perfectly consistent
    assert r["spreading_ratio"] == pytest.approx(2.0, rel=0.01)   # and twice too wide


def test_uniformity_flags_a_broken_filament():
    image, truth = synthetic.serpentine(break_filament=2)
    out = uniformity.analyze(
        image, truth["roi"],
        {"n_filaments": 6, "n_positions": 5, "polarity": "bright",
         "nozzle_id_mm": 2.0},
        scale_for(10.0),
    )
    assert out["results"]["continuity"]["continuous"] is False
    assert any("discontinuous" in f.lower() for f in out["flags"])


def test_uniformity_excluding_measurements_changes_n_and_sd():
    image, truth = synthetic.serpentine()
    out = uniformity.analyze(
        image, truth["roi"],
        {"n_filaments": 6, "n_positions": 5, "polarity": "bright",
         "nozzle_id_mm": 2.0},
        scale_for(10.0),
    )
    measurements = out["measurements"]
    for m in measurements[:5]:
        m["included"] = False

    rescored = uniformity.compute(measurements, 2.0)
    assert rescored["n_included"] == 25
    assert rescored["n_total"] == 30
    # SD must use the reduced count, not the original one.
    assert rescored["sd_mm"] != out["results"]["sd_mm"]


# ============================================================ tab 3


FUSION_CASES = [
    pytest.param(0.0, False, False, id="flat-open"),
    pytest.param(0.0, True, False, id="flat-fused-corner"),
    pytest.param(0.5, False, False, id="vignette-open"),
    pytest.param(0.5, True, False, id="vignette-fused-corner"),
    pytest.param(0.8, True, False, id="heavy-vignette-fused"),
    pytest.param(0.5, True, True, id="dark-material-fused"),
]


@pytest.mark.parametrize("vig,fused,dark", FUSION_CASES)
def test_fusion_recovers_known_pore_areas(vig, fused, dark):
    px_per_mm = 77.0
    image, truth = synthetic.pore_grid(
        fuse_smallest=fused, dark=dark, vignette_strength=vig, px_per_mm=px_per_mm)
    out = fusion.analyze(
        image, truth["roi"],
        {"grid_n": 5, "fd_mm": truth["fds_mm"], "polarity": truth["polarity"]},
        scale_for(px_per_mm),
    )
    rows = out["results"]["rows"]
    assert len(rows) == 5, "the table must have one row per size class, never more"

    for row in rows:
        w = row["raw"]
        fd = w["nominal_fd_mm"]
        if fused and fd == 1:
            assert w["status"] == "closed"
            assert w["aa_mm2"] == 0
            assert w["dfr_percent"] == pytest.approx(100.0)
            assert w["pr"] is None and w["circularity"] is None
            continue

        assert w["status"] == "open"
        assert w["aa_mm2"] == pytest.approx(fd * fd, rel=0.05)
        # A square pore: C = pi/4, Pr = 1. Small pores carry more pixel
        # quantisation at the corners, so the tolerance loosens as fd shrinks.
        tol = 0.04 if fd >= 3 else 0.08
        assert w["circularity"] == pytest.approx(math.pi / 4, abs=tol)
        assert w["pr"] == pytest.approx(1.0, abs=tol)


def test_fusion_pr_and_circularity_are_the_same_measurement():
    """C == pi/(4 Pr) identically, which is why they cannot cross-check."""
    image, truth = synthetic.pore_grid()
    out = fusion.analyze(image, truth["roi"],
                         {"grid_n": 5, "polarity": "bright"}, scale_for(77.0))
    for row in out["results"]["rows"]:
        w = row["raw"]
        if w["pr"] is None:
            continue
        assert w["circularity"] == pytest.approx(math.pi / (4 * w["pr"]), rel=1e-9)


def test_fusion_theoretical_area_is_overridable():
    image, truth = synthetic.pore_grid()
    out = fusion.analyze(
        image, truth["roi"],
        {"grid_n": 5, "polarity": "bright", "at_mm2": {"0": 2.5}},
        scale_for(77.0),
    )
    assert out["results"]["rows"][0]["raw"]["at_mm2"] == pytest.approx(2.5)


# ============================================================ tab 4


def test_collapse_recovers_parabolic_sag_area():
    px_per_mm = 20.0
    image, truth = synthetic.collapse_platform(px_per_mm=px_per_mm)
    out = collapse.analyze(image, None, {"convention": "sag"}, scale_for(px_per_mm))

    rows = out["results"]["rows"]
    assert len(rows) == 6
    assert out["results"]["df_mm"] == pytest.approx(truth["df_mm"], rel=0.05)

    for row, expected in zip(rows, truth["a_sag_mm2"]):
        w = row["raw"]
        assert w["status"] == "bridged"
        assert w["a_sag_mm2"] == pytest.approx(expected, rel=0.02)
        assert w["cf_percent"] == pytest.approx(
            w["a_sag_mm2"] / w["a_max_mm2"] * 100, rel=1e-6)

    # Above ~2 mm the rasterised sag has converged on the analytic parabola, so
    # the measurement should match the closed form too.
    for row, analytic in zip(rows[1:], truth["a_sag_mm2_analytic"][1:]):
        assert row["raw"]["a_sag_mm2"] == pytest.approx(analytic, rel=0.06)


def test_collapse_flat_bridge_reads_zero():
    px_per_mm = 20.0
    image, _ = synthetic.collapse_platform(
        sag_fraction=(0.0,) * 6, px_per_mm=px_per_mm)
    out = collapse.analyze(image, None, {"convention": "sag"}, scale_for(px_per_mm))
    for row in out["results"]["rows"]:
        assert row["raw"]["cf_percent"] == pytest.approx(0.0, abs=0.5)
        assert row["raw"]["theta_deg"] == pytest.approx(0.0, abs=1.0)


def test_collapse_broken_gap_is_total_collapse_not_a_failure():
    px_per_mm = 20.0
    image, _ = synthetic.collapse_platform(px_per_mm=px_per_mm, broken=(3,))
    out = collapse.analyze(image, None, {"convention": "sag"}, scale_for(px_per_mm))

    broken = out["results"]["rows"][3]["raw"]
    assert broken["status"] == "broken"
    assert broken["cf_percent"] == pytest.approx(100.0)
    assert any("no filament spans" in f for f in out["flags"])
    # The other gaps still measure normally.
    assert all(out["results"]["rows"][i]["raw"]["status"] == "bridged"
               for i in (0, 1, 2, 4, 5))


def test_collapse_conventions_are_exact_complements():
    px_per_mm = 20.0
    image, _ = synthetic.collapse_platform(px_per_mm=px_per_mm)
    out = collapse.analyze(image, None, {"convention": "sag"}, scale_for(px_per_mm))

    flipped = collapse.compute(out["measurements"], "bridge")
    for a, b in zip(out["results"]["rows"], flipped["rows"]):
        assert a["raw"]["cf_percent"] + b["raw"]["cf_percent"] == pytest.approx(100.0)


def test_collapse_deflection_angle_matches_triangle_approximation():
    px_per_mm = 20.0
    depth_fraction = 0.25
    image, truth = synthetic.collapse_platform(
        sag_fraction=(depth_fraction,) * 6, px_per_mm=px_per_mm)
    out = collapse.analyze(image, None, {"convention": "sag"}, scale_for(px_per_mm))

    for row in out["results"]["rows"]:
        w = row["raw"]
        # Deepest point of a parabola is mid-span, so the run is half the gap.
        expected = math.degrees(math.atan2(
            depth_fraction * 6.0, w["nominal_gap_mm"] / 2.0))
        assert w["theta_deg"] == pytest.approx(expected, abs=2.0)
