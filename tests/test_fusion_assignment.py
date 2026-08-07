"""Arista-based theoretical area, and operator-assigned pores."""

from __future__ import annotations

import math

import pytest

from app import config
from app.analysis import fusion
from app.calib.geometry import Scale
from tests import synthetic

PX_PER_MM = 77.0
D_MM = 0.41
ARISTAS = [1.5, 2.0, 3.0, 4.0, 5.0]


def scale() -> Scale:
    return Scale(mm_per_px_x=1 / PX_PER_MM, mm_per_px_y=1 / PX_PER_MM,
                 calibrated=True)


def grid_for_aristas(aristas=ARISTAS, d=D_MM):
    """A lattice whose walls are d wide and whose openings are (a - d) across.

    That is the geometry the arista convention describes, so a correct
    measurement should return At almost exactly.
    """
    return synthetic.pore_grid(
        fds_mm=tuple(a - d for a in aristas), wall_mm=d, px_per_mm=PX_PER_MM)


def params(**over):
    return {"grid_n": 5, "polarity": "bright",
            "arista_mm": ARISTAS, "filament_d_mm": D_MM, **over}


# ============================================================ At = (a - d)^2


@pytest.mark.parametrize("a,d,expected", [
    (1.0, 0.41, 0.3481),
    (2.0, 0.41, 2.5281),
    (5.0, 0.41, 21.0681),
    (3.0, 0.0, 9.0),
    (1.5, 0.5, 1.0),
])
def test_theoretical_area(a, d, expected):
    assert fusion.theoretical_area(a, d) == pytest.approx(expected)


@pytest.mark.parametrize("a,d", [(0.41, 0.41), (0.3, 0.41), (0.0, 0.41)])
def test_theoretical_area_is_undefined_when_the_filament_fills_the_spacing(a, d):
    """Refused, not divided by: a non-positive At makes Dfr meaningless."""
    assert fusion.theoretical_area(a, d) is None


def test_arista_not_larger_than_the_filament_is_flagged():
    image, truth = grid_for_aristas()
    out = fusion.analyze(image, truth["roi"],
                         params(arista_mm=[0.3, 2.0, 3.0, 4.0, 5.0]), scale())

    assert any("not larger than the filament" in f for f in out["flags"])
    first = out["results"]["rows"][0]["raw"]
    assert first["at_mm2"] is None
    assert first["dfr_percent"] is None, "Dfr must not be computed from a bad At"


def test_measured_area_matches_the_arista_geometry():
    image, truth = grid_for_aristas()
    out = fusion.analyze(image, truth["roi"], params(), scale())

    for row, a in zip(out["results"]["rows"], ARISTAS):
        w = row["raw"]
        assert w["arista_mm"] == pytest.approx(a)
        assert w["filament_d_mm"] == pytest.approx(D_MM)
        assert w["at_mm2"] == pytest.approx((a - D_MM) ** 2)
        if w["status"] == "open":
            # The lattice was drawn to this convention, so Dfr should be ~0.
            assert abs(w["dfr_percent"]) < 2.0


def test_the_arista_list_is_padded_to_the_grid_size():
    image, truth = grid_for_aristas()
    out = fusion.analyze(image, truth["roi"],
                         params(arista_mm=[1.5, 2.0]), scale())
    aristas = [r["raw"]["arista_mm"] for r in out["results"]["rows"]]
    assert len(aristas) == 5
    assert aristas[:2] == [1.5, 2.0]
    assert aristas[2:] == [3.0, 4.0, 5.0], "short lists extend by the next integers"


def test_a_stored_override_still_wins():
    """Runs saved before the arista change carry their own At."""
    image, truth = grid_for_aristas()
    out = fusion.analyze(image, truth["roi"],
                         params(at_mm2={"0": 9.99}), scale())
    assert out["results"]["rows"][0]["raw"]["at_mm2"] == pytest.approx(9.99)


# ============================================================ candidates


def test_candidates_cover_every_detected_region():
    image, truth = grid_for_aristas()
    out = fusion.analyze(image, truth["roi"], params(), scale())

    candidates = out["candidates"]
    assert len(candidates) >= 5
    assert sum(1 for c in candidates if c["auto_selected"]) == 5

    for c in candidates:
        assert len(c["polygon"]) >= 3
        assert c["aa_mm2"] > 0
        assert c["perimeter_mm"] > 0


def test_candidate_centroids_are_in_full_frame_coordinates():
    """The same frame of reference the measurement rows use, so a manual pick
    and an automatic one can be compared and stored identically."""
    image, truth = grid_for_aristas()
    out = fusion.analyze(image, truth["roi"], params(), scale())

    by_centroid = {(round(c["centroid"][0]), round(c["centroid"][1]))
                   for c in out["candidates"]}
    for row in out["results"]["rows"]:
        c = row["raw"]["centroid"]
        if c and row["raw"]["status"] == "open":
            assert (round(c[0]), round(c[1])) in by_centroid


def test_candidate_polygon_is_only_for_drawing():
    """The simplified outline must not be what the metrics came from."""
    image, truth = grid_for_aristas()
    out = fusion.analyze(image, truth["roi"], params(), scale())

    for c in out["candidates"]:
        # A pixel-traced boundary has hundreds of points; the drawing copy has
        # far fewer, yet the reported area is the full-contour one.
        assert len(c["polygon"]) < 60
        side = math.sqrt(c["aa_mm2"])
        assert side > 0.3, "area must come from the real contour, not the outline"


# ============================================================ assignment


def auto_assignment(out):
    """The automatic choice expressed as an assignment map."""
    by_centroid = {(round(c["centroid"][0]), round(c["centroid"][1])): c["index"]
                   for c in out["candidates"]}
    assignment = {}
    for k, row in enumerate(out["results"]["rows"]):
        c = row["raw"]["centroid"]
        assignment[str(k)] = (
            by_centroid.get((round(c[0]), round(c[1]))) if c else None)
    return assignment


def test_assign_reproduces_the_automatic_result():
    """The two paths must not disagree when handed the same pores."""
    image, truth = grid_for_aristas()
    auto = fusion.analyze(image, truth["roi"], params(), scale())

    manual = fusion.assign(auto["candidates"], auto_assignment(auto), params())

    for a_row, m_row in zip(auto["results"]["rows"], manual["results"]["rows"]):
        aw, mw = a_row["raw"], m_row["raw"]
        for key in ("at_mm2", "aa_mm2", "perimeter_mm",
                    "dfr_percent", "pr", "circularity"):
            if aw[key] is None:
                assert mw[key] is None
            else:
                assert mw[key] == pytest.approx(aw[key])


def test_a_manual_pick_takes_that_regions_values():
    image, truth = grid_for_aristas()
    auto = fusion.analyze(image, truth["roi"], params(), scale())
    assignment = auto_assignment(auto)

    # Point the 4 mm class at the pore the 5 mm class was using.
    swapped = {**assignment, "3": assignment["4"]}
    manual = fusion.assign(auto["candidates"], swapped, params(),
                           manual_classes=[3])

    moved = manual["results"]["rows"][3]["raw"]
    source = auto["results"]["rows"][4]["raw"]
    assert moved["aa_mm2"] == pytest.approx(source["aa_mm2"])
    assert moved["perimeter_mm"] == pytest.approx(source["perimeter_mm"])
    # At follows the class, not the pore.
    assert moved["at_mm2"] == pytest.approx((ARISTAS[3] - D_MM) ** 2)
    assert moved["dfr_percent"] != pytest.approx(source["dfr_percent"])


def test_clicking_outside_the_roi_records_a_closed_pore():
    image, truth = grid_for_aristas()
    auto = fusion.analyze(image, truth["roi"], params(), scale())
    assignment = {**auto_assignment(auto), "2": "closed"}

    manual = fusion.assign(auto["candidates"], assignment, params(),
                           manual_classes=[2])

    w = manual["results"]["rows"][2]["raw"]
    assert w["status"] == "closed"
    assert w["aa_mm2"] == 0
    assert w["dfr_percent"] == pytest.approx(100.0)
    assert w["pr"] is None and w["circularity"] is None
    assert w["selection"] == "manual_closed"


def test_only_the_classes_the_operator_touched_read_as_manual():
    """One correction must not relabel the whole table as hand-picked."""
    image, truth = grid_for_aristas()
    auto = fusion.analyze(image, truth["roi"], params(), scale())

    manual = fusion.assign(auto["candidates"], auto_assignment(auto), params(),
                           manual_classes=[3])

    sources = [r["raw"]["selection"] for r in manual["results"]["rows"]]
    assert sources == ["auto", "auto", "auto", "manual", "auto"]


def test_an_unknown_region_is_reported_rather_than_guessed():
    image, truth = grid_for_aristas()
    auto = fusion.analyze(image, truth["roi"], params(), scale())

    manual = fusion.assign(auto["candidates"], {"0": 9999}, params(),
                           manual_classes=[0])

    assert any("no such detected region" in f for f in manual["flags"])
    assert manual["results"]["rows"][0]["raw"]["status"] == "missing"


def test_provenance_reaches_the_measurement_rows():
    image, truth = grid_for_aristas()
    auto = fusion.analyze(image, truth["roi"], params(), scale())

    for row in auto["measurements"]:
        raw = row["raw"]
        # The flat CSV export walks raw.items(), so these keys reach it.
        assert raw["selection"] == "auto"
        assert "arista_mm" in raw and "filament_d_mm" in raw


def test_two_classes_on_the_same_region_is_reported():
    """Invisible in the table otherwise -- both rows just show the same area."""
    image, truth = grid_for_aristas()
    auto = fusion.analyze(image, truth["roi"], params(), scale())
    assignment = auto_assignment(auto)

    clash = {**assignment, "3": assignment["4"]}
    manual = fusion.assign(auto["candidates"], clash, params(),
                           manual_classes=[3])

    assert any("same region" in f for f in manual["flags"])


def test_distinct_regions_raise_no_clash_warning():
    image, truth = grid_for_aristas()
    auto = fusion.analyze(image, truth["roi"], params(), scale())

    manual = fusion.assign(auto["candidates"], auto_assignment(auto), params())
    assert not any("same region" in f for f in manual["flags"])
