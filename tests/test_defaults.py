"""Per-tab stored defaults, so a rig that never changes is not re-entered."""

from __future__ import annotations

import importlib

import pytest

from app import config


@pytest.fixture()
def repo(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test.db")
    from app.db import repo as repo_module
    importlib.reload(repo_module)
    repo_module.init_db()
    yield repo_module
    repo_module.db().close()


UNIFORMITY = {"nozzle_id_mm": 0.25, "feed_rate_mms": 7.5, "n_filaments": 8,
              "padding": 60, "polarity": "dark", "final_kernel": 5}
FUSION = {"grid_n": 4, "filament_d_mm": 0.55,
          "arista_mm": [1.2, 2.4, 3.6, 4.8], "diagonal_tolerance": 0.4}
COLLAPSE = {"gap_height_mm": 5.5, "convention": "bridge",
            "gaps_mm": [1, 2, 3, 4, 5, 6],
            "hsv": {"pillar": {"lo": [0, 30, 120], "hi": [15, 140, 255]}}}


@pytest.mark.parametrize("test_type,values", [
    ("uniformity", UNIFORMITY), ("fusion", FUSION), ("collapse", COLLAPSE),
])
def test_defaults_round_trip(repo, test_type, values):
    repo.save_tab_defaults(test_type, values)

    stored = repo.get_tab_defaults(test_type)
    assert stored["test_type"] == test_type
    assert stored["values"] == values
    assert stored["updated_at"]


def test_each_tab_keeps_its_own_set(repo):
    repo.save_tab_defaults("uniformity", UNIFORMITY)
    repo.save_tab_defaults("fusion", FUSION)

    assert repo.get_tab_defaults("uniformity")["values"] == UNIFORMITY
    assert repo.get_tab_defaults("fusion")["values"] == FUSION
    assert repo.get_tab_defaults("collapse") is None


def test_saving_twice_replaces_rather_than_duplicates(repo):
    repo.save_tab_defaults("uniformity", UNIFORMITY)
    repo.save_tab_defaults("uniformity", {**UNIFORMITY, "padding": 20})

    assert repo.get_tab_defaults("uniformity")["values"]["padding"] == 20
    rows = repo.db().execute(
        "SELECT COUNT(*) AS n FROM tab_defaults WHERE test_type = 'uniformity'"
    ).fetchone()
    assert rows["n"] == 1


def test_clearing_removes_the_set(repo):
    repo.save_tab_defaults("fusion", FUSION)
    repo.clear_tab_defaults("fusion")
    assert repo.get_tab_defaults("fusion") is None


def test_a_set_saved_before_a_field_existed_still_loads(repo):
    """Forward compatibility: a missing key falls back to the factory value.

    This is the case that keeps an old default set usable after the tab gains an
    input, rather than forcing the operator to re-save everything.
    """
    partial = {"nozzle_id_mm": 0.25}
    repo.save_tab_defaults("uniformity", partial)

    values = repo.get_tab_defaults("uniformity")["values"]
    assert values == partial
    assert "padding" not in values, "absent keys stay absent, not None"


def test_unknown_keys_survive_a_round_trip_without_raising(repo):
    """A key the current UI no longer has is stored and returned untouched; the
    front end ignores it because no input matches."""
    repo.save_tab_defaults("uniformity", {**UNIFORMITY, "retired_setting": 3})
    values = repo.get_tab_defaults("uniformity")["values"]
    assert values["retired_setting"] == 3
    assert values["padding"] == 60


def test_corrupt_json_does_not_break_the_tab(repo):
    repo.db().execute(
        "INSERT INTO tab_defaults (test_type, values_json) VALUES (?, ?)",
        ("fusion", "{not json"))
    repo.db().commit()
    assert repo.get_tab_defaults("fusion")["values"] == {}


def test_migration_adds_the_table_to_an_older_database(repo):
    """A database created before defaults existed must keep working."""
    repo.db().execute("DROP TABLE tab_defaults")
    repo.db().commit()

    repo.init_db()

    tables = {r["name"] for r in repo.db().execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    assert "tab_defaults" in tables
    assert repo.get_tab_defaults("uniformity") is None


# ---------------------------------------------------------------- API rules


def test_identity_fields_are_never_stored():
    """Carrying a sample name or replicate number forward mislabels a run."""
    from app.main import DEFAULTS_EXCLUDED

    submitted = {**UNIFORMITY, "name": "yesterday's sample", "replicate_no": 3}
    kept = {k: v for k, v in submitted.items() if k not in DEFAULTS_EXCLUDED}

    assert "name" not in kept
    assert "replicate_no" not in kept
    assert kept["nozzle_id_mm"] == 0.25


def test_every_test_type_is_accepted():
    from app.main import _check_test_type
    from fastapi import HTTPException

    for test_type in config.TEST_TYPES:
        _check_test_type(test_type)

    with pytest.raises(HTTPException):
        _check_test_type("not-a-tab")
