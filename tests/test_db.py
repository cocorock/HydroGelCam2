"""Database round-trip: a saved run must reload and re-score identically."""

from __future__ import annotations

import importlib

import pytest

from app.analysis import uniformity
from app.calib.geometry import Scale
from tests import synthetic


@pytest.fixture()
def repo(tmp_path, monkeypatch):
    """A throwaway database, so tests never touch the real storage directory."""
    from app import config
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test.db")

    from app.db import repo as repo_module
    importlib.reload(repo_module)
    repo_module.init_db()
    yield repo_module
    repo_module.db().close()


def make_run():
    image, truth = synthetic.serpentine()
    scale = Scale(mm_per_px_x=0.1, mm_per_px_y=0.1, calibrated=True)
    return uniformity.analyze(
        image, truth["roi"],
        {"n_filaments": 6, "n_positions": 5, "polarity": "bright",
         "nozzle_id_mm": 2.0},
        scale,
    )


def test_run_survives_a_save_and_reload_unchanged(repo):
    out = make_run()
    run_id = repo.create_run({
        "test_type": "uniformity", "name": "sample A", "replicate_no": 1,
        "nozzle_id_mm": 2.0, "roi_json": out["roi"],
        "params_json": {"n_filaments": 6, "n_positions": 5},
        "results_json": out["results"], "flags_json": out["flags"],
    }, out["measurements"])

    loaded = repo.get_run(run_id)
    assert loaded["name"] == "sample A"
    assert loaded["roi_json"] == out["roi"]
    assert len(loaded["measurements"]) == len(out["measurements"])

    # Recomputing from the reloaded rows must reproduce the original numbers.
    rescored = uniformity.compute(loaded["measurements"], loaded["nozzle_id_mm"])
    assert rescored["mean_mm"] == pytest.approx(out["results"]["mean_mm"])
    assert rescored["uniformity_index"] == pytest.approx(
        out["results"]["uniformity_index"])
    assert rescored["n_included"] == out["results"]["n_included"]


def test_excluding_a_measurement_persists(repo):
    out = make_run()
    run_id = repo.create_run(
        {"test_type": "uniformity", "name": "sample B", "nozzle_id_mm": 2.0},
        out["measurements"])

    measurements = repo.get_measurements(run_id)
    repo.set_measurement_included(measurements[0]["id"], False)

    reloaded = repo.get_measurements(run_id)
    assert reloaded[0]["included"] is False
    assert uniformity.compute(reloaded, 2.0)["n_included"] == 29


def test_deleting_a_run_removes_its_measurements(repo):
    out = make_run()
    run_id = repo.create_run(
        {"test_type": "uniformity", "name": "gone"}, out["measurements"])
    assert repo.get_measurements(run_id)

    repo.delete_run(run_id)
    assert repo.get_run(run_id) is None
    assert repo.get_measurements(run_id) == []


def test_listing_filters_by_test_type_and_name(repo):
    out = make_run()
    repo.create_run({"test_type": "uniformity", "name": "alpha"}, out["measurements"])
    repo.create_run({"test_type": "fusion", "name": "beta"}, [])

    assert len(repo.list_runs()) == 2
    assert len(repo.list_runs(test_type="fusion")) == 1
    assert repo.list_runs(name="alph")[0]["name"] == "alpha"


def test_saving_a_calibration_twice_updates_rather_than_duplicates(repo):
    first = repo.save_calibration({
        "name": "bench", "board_cols": 9, "board_rows": 6, "square_mm": 5.0,
        "mm_per_px_x": 0.05, "mm_per_px_y": 0.05,
    })
    second = repo.save_calibration({
        "name": "bench", "board_cols": 9, "board_rows": 6, "square_mm": 5.0,
        "mm_per_px_x": 0.02, "mm_per_px_y": 0.02,
    })
    assert first == second
    assert len(repo.list_calibrations()) == 1
    assert repo.get_calibration(first)["mm_per_px_x"] == pytest.approx(0.02)


def test_json_columns_round_trip_as_python_objects(repo):
    run_id = repo.create_run({
        "test_type": "collapse", "name": "json",
        "roi_json": {"x": 1, "y": 2, "w": 3, "h": 4},
        "flags_json": ["one", "two"],
        "results_json": {"cf": {"mean": 12.5, "sd": None}},
    }, [])
    loaded = repo.get_run(run_id)
    assert loaded["roi_json"]["w"] == 3
    assert loaded["flags_json"] == ["one", "two"]
    assert loaded["results_json"]["cf"]["mean"] == 12.5
