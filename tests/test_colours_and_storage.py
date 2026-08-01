"""Picked colour ranges for tab 4, and the overlay image stored beside a run."""

from __future__ import annotations

import importlib

import cv2
import numpy as np
import pytest

from app import config
from app.analysis import collapse
from app.calib.geometry import Scale
from tests import synthetic


def scale_for(px_per_mm: float) -> Scale:
    return Scale(mm_per_px_x=1 / px_per_mm, mm_per_px_y=1 / px_per_mm,
                 calibrated=True)


def hsv_window(bgr, hue_tol=12):
    """The same window the front end builds from a sampled pixel."""
    h, s, v = cv2.cvtColor(np.uint8([[list(bgr)]]), cv2.COLOR_BGR2HSV)[0, 0]
    s_lo, v_lo = max(0, int(s) - 90), max(0, int(v) - 110)
    lo, hi = int(h) - hue_tol, int(h) + hue_tol
    if lo < 0:
        return [{"lo": [0, s_lo, v_lo], "hi": [hi, 255, 255]},
                {"lo": [180 + lo, s_lo, v_lo], "hi": [179, 255, 255]}]
    if hi > 179:
        return [{"lo": [lo, s_lo, v_lo], "hi": [179, 255, 255]},
                {"lo": [0, s_lo, v_lo], "hi": [hi - 180, 255, 255]}]
    return [{"lo": [lo, s_lo, v_lo], "hi": [hi, 255, 255]}]


def test_colours_sampled_from_the_image_drive_the_measurement():
    px_per_mm = 20.0
    image, truth = synthetic.collapse_platform(px_per_mm=px_per_mm)

    # Sample the two materials exactly as a click on the canvas would.
    pillar_bgr = tuple(int(v) for v in image[180, 30])
    filament_bgr = tuple(int(v) for v in image[117, 215])

    pillar = hsv_window(pillar_bgr)
    filament = hsv_window(filament_bgr)
    hsv = {"filament": filament[0]}
    hsv["pillar"] = pillar[0]
    if len(pillar) > 1:
        hsv["pillar2"] = pillar[1]

    out = collapse.analyze(image, None, {"hsv": hsv}, scale_for(px_per_mm))

    assert not out["flags"]
    for row, expected in zip(out["results"]["rows"], truth["a_sag_mm2"]):
        assert row["raw"]["status"] == "bridged"
        assert row["raw"]["a_sag_mm2"] == pytest.approx(expected, rel=0.02)


def test_red_hue_wrap_produces_two_ranges():
    """Pale red sits at H ~ 0, so its window must straddle the wrap."""
    ranges = hsv_window((142, 158, 220), hue_tol=12)   # BGR of the ABS
    assert len(ranges) == 2
    assert ranges[0]["lo"][0] == 0
    assert ranges[1]["hi"][0] == 179


def test_a_wrong_colour_choice_fails_loudly_rather_than_silently():
    px_per_mm = 20.0
    image, _ = synthetic.collapse_platform(px_per_mm=px_per_mm)
    # Green matches neither material.
    hsv = {"pillar": {"lo": [60, 80, 80], "hi": [80, 255, 255]},
           "filament": {"lo": [60, 80, 80], "hi": [80, 255, 255]}}

    out = collapse.analyze(image, None, {"hsv": hsv}, scale_for(px_per_mm))
    assert out["flags"], "an unmatched colour range must be reported"
    assert out["results"]["rows"] == []
    # The results still have the shape the UI expects.
    assert out["results"]["n_gaps"] == len(config.COLLAPSE_GAPS_MM)
    assert out["results"]["cf"]["mean"] is None


# ---------------------------------------------------------------- storage


@pytest.fixture()
def repo(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test.db")
    from app.db import repo as repo_module
    importlib.reload(repo_module)
    repo_module.init_db()
    yield repo_module
    repo_module.db().close()


def test_overlay_path_is_stored_and_reloaded(repo):
    run_id = repo.create_run({
        "test_type": "uniformity", "name": "with overlay",
        "image_path": "/tmp/capture.png",
        "overlay_path": "/tmp/capture_overlay.png",
    }, [])
    loaded = repo.get_run(run_id)
    assert loaded["image_path"] == "/tmp/capture.png"
    assert loaded["overlay_path"] == "/tmp/capture_overlay.png"


def test_a_run_without_an_overlay_is_still_valid(repo):
    run_id = repo.create_run(
        {"test_type": "fusion", "name": "no overlay",
         "image_path": "/tmp/only-capture.png"}, [])
    assert repo.get_run(run_id)["overlay_path"] is None


def test_migration_adds_the_overlay_column_to_an_older_database(repo):
    """A database created before overlays existed must keep working."""
    repo.db().execute("DROP TABLE test_run")
    repo.db().execute("""
        CREATE TABLE test_run (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            test_type TEXT NOT NULL,
            name TEXT NOT NULL DEFAULT '',
            image_path TEXT,
            roi_json TEXT NOT NULL DEFAULT '{}',
            params_json TEXT NOT NULL DEFAULT '{}',
            results_json TEXT NOT NULL DEFAULT '{}',
            flags_json TEXT NOT NULL DEFAULT '[]',
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        )""")
    repo.db().execute(
        "INSERT INTO test_run (test_type, name) VALUES ('uniformity', 'legacy')")
    repo.db().commit()

    repo.init_db()   # runs the migration

    columns = {r["name"] for r in
               repo.db().execute("PRAGMA table_info(test_run)").fetchall()}
    assert "overlay_path" in columns
    row = repo.db().execute("SELECT * FROM test_run WHERE name='legacy'").fetchone()
    assert row["overlay_path"] is None, "existing rows survive the migration"


def test_pending_purge_removes_only_stale_files(tmp_path, monkeypatch):
    """Staged captures are scratch; old ones must not accumulate forever."""
    import os
    import time as time_module
    from app import main

    monkeypatch.setattr(main, "PENDING", tmp_path)

    fresh = tmp_path / "fresh.png"
    stale = tmp_path / "stale.png"
    fresh.write_bytes(b"x")
    stale.write_bytes(b"x")
    old = time_module.time() - 48 * 3600
    os.utime(stale, (old, old))

    removed = main._purge_pending(max_age_hours=24.0)

    assert removed == 1
    assert fresh.exists()
    assert not stale.exists()
