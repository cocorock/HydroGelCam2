"""SQLite access layer.

Deliberately thin: raw sqlite3 with row factories, no ORM. Every JSON column is
decoded on read and encoded on write so callers only ever see Python objects.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Iterable

from app import config

_JSON_COLUMNS = {
    "props_json", "hsv_ranges_json", "K_json", "dist_json", "H_json",
    "roi_json", "params_json", "results_json", "flags_json", "raw_json",
}

_local = threading.local()


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(config.DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def db() -> sqlite3.Connection:
    """One connection per thread; FastAPI's threadpool reuses threads."""
    conn = getattr(_local, "conn", None)
    if conn is None:
        conn = _local.conn = _connect()
    return conn


def init_db() -> None:
    schema = (Path(__file__).parent / "schema.sql").read_text(encoding="utf-8")
    db().executescript(schema)
    _migrate()
    db().commit()


# Columns added after the first release. SQLite has no ADD COLUMN IF NOT EXISTS,
# so each is checked against the live table and added when missing, leaving
# existing databases usable rather than needing a rebuild.
_ADDED_COLUMNS = {
    "test_run": {"overlay_path": "TEXT"},
}


def _migrate() -> None:
    for table, columns in _ADDED_COLUMNS.items():
        existing = {
            row["name"]
            for row in db().execute(f"PRAGMA table_info({table})").fetchall()
        }
        for name, decl in columns.items():
            if name not in existing:
                db().execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")


# ---------------------------------------------------------------- helpers


def _decode(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    out: dict[str, Any] = dict(row)
    for key in list(out):
        if key in _JSON_COLUMNS and isinstance(out[key], str):
            try:
                out[key] = json.loads(out[key])
            except (json.JSONDecodeError, TypeError):
                out[key] = None
    return out


def _decode_all(rows: Iterable[sqlite3.Row]) -> list[dict[str, Any]]:
    return [d for d in (_decode(r) for r in rows) if d is not None]


def _encode(values: dict[str, Any]) -> dict[str, Any]:
    out = dict(values)
    for key, val in out.items():
        if key in _JSON_COLUMNS and not isinstance(val, (str, type(None))):
            out[key] = json.dumps(val)
    return out


def _insert(table: str, values: dict[str, Any]) -> int:
    values = _encode(values)
    cols = ", ".join(values)
    marks = ", ".join("?" for _ in values)
    cur = db().execute(
        f"INSERT INTO {table} ({cols}) VALUES ({marks})", tuple(values.values())
    )
    db().commit()
    return int(cur.lastrowid)


def _update(table: str, row_id: int, values: dict[str, Any]) -> None:
    if not values:
        return
    values = _encode(values)
    sets = ", ".join(f"{k} = ?" for k in values)
    db().execute(
        f"UPDATE {table} SET {sets} WHERE id = ?", (*values.values(), row_id)
    )
    db().commit()


def _delete(table: str, row_id: int) -> None:
    db().execute(f"DELETE FROM {table} WHERE id = ?", (row_id,))
    db().commit()


# ---------------------------------------------------------------- camera profiles


def list_camera_profiles() -> list[dict[str, Any]]:
    return _decode_all(
        db().execute("SELECT * FROM camera_profile ORDER BY name").fetchall()
    )


def get_camera_profile(profile_id: int) -> dict[str, Any] | None:
    return _decode(
        db().execute(
            "SELECT * FROM camera_profile WHERE id = ?", (profile_id,)
        ).fetchone()
    )


def save_camera_profile(values: dict[str, Any]) -> int:
    """Upsert by name so re-saving a profile overwrites it instead of erroring."""
    existing = db().execute(
        "SELECT id FROM camera_profile WHERE name = ?", (values["name"],)
    ).fetchone()
    if existing:
        _update("camera_profile", existing["id"], values)
        return int(existing["id"])
    return _insert("camera_profile", values)


def delete_camera_profile(profile_id: int) -> None:
    _delete("camera_profile", profile_id)


# ---------------------------------------------------------------- calibrations


def list_calibrations(intended_use: str | None = None) -> list[dict[str, Any]]:
    if intended_use:
        rows = db().execute(
            "SELECT * FROM calibration WHERE intended_use = ? "
            "ORDER BY created_at DESC",
            (intended_use,),
        ).fetchall()
    else:
        rows = db().execute(
            "SELECT * FROM calibration ORDER BY created_at DESC"
        ).fetchall()
    return _decode_all(rows)


def get_calibration(calib_id: int) -> dict[str, Any] | None:
    return _decode(
        db().execute("SELECT * FROM calibration WHERE id = ?", (calib_id,)).fetchone()
    )


def save_calibration(values: dict[str, Any]) -> int:
    existing = db().execute(
        "SELECT id FROM calibration WHERE name = ?", (values["name"],)
    ).fetchone()
    if existing:
        _update("calibration", existing["id"], values)
        return int(existing["id"])
    return _insert("calibration", values)


def update_calibration(calib_id: int, values: dict[str, Any]) -> None:
    _update("calibration", calib_id, values)


def delete_calibration(calib_id: int) -> None:
    _delete("calibration", calib_id)


# ---------------------------------------------------------------- test runs


def create_run(values: dict[str, Any], measurements: list[dict[str, Any]]) -> int:
    run_id = _insert("test_run", values)
    _replace_measurements(run_id, measurements)
    return run_id


def update_run(
    run_id: int,
    values: dict[str, Any],
    measurements: list[dict[str, Any]] | None = None,
) -> None:
    values = {**values, "updated_at": _now()}
    _update("test_run", run_id, values)
    if measurements is not None:
        _replace_measurements(run_id, measurements)


def _now() -> str:
    return db().execute("SELECT datetime('now')").fetchone()[0]


def _replace_measurements(run_id: int, measurements: list[dict[str, Any]]) -> None:
    db().execute("DELETE FROM measurement WHERE test_run_id = ?", (run_id,))
    for i, m in enumerate(measurements):
        db().execute(
            "INSERT INTO measurement (test_run_id, index_no, label, included, raw_json)"
            " VALUES (?, ?, ?, ?, ?)",
            (
                run_id,
                m.get("index_no", i),
                m.get("label"),
                1 if m.get("included", True) else 0,
                json.dumps(m.get("raw", m.get("raw_json", {}))),
            ),
        )
    db().commit()


def get_run(run_id: int) -> dict[str, Any] | None:
    run = _decode(
        db().execute("SELECT * FROM test_run WHERE id = ?", (run_id,)).fetchone()
    )
    if run is None:
        return None
    run["measurements"] = get_measurements(run_id)
    return run


def get_measurements(run_id: int) -> list[dict[str, Any]]:
    rows = db().execute(
        "SELECT * FROM measurement WHERE test_run_id = ? ORDER BY index_no",
        (run_id,),
    ).fetchall()
    out = _decode_all(rows)
    for m in out:
        m["included"] = bool(m["included"])
        m["raw"] = m.pop("raw_json") or {}
    return out


def set_measurement_included(measurement_id: int, included: bool) -> None:
    db().execute(
        "UPDATE measurement SET included = ? WHERE id = ?",
        (1 if included else 0, measurement_id),
    )
    db().commit()


def list_runs(
    test_type: str | None = None,
    name: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    limit: int = 500,
) -> list[dict[str, Any]]:
    where: list[str] = []
    args: list[Any] = []
    if test_type:
        where.append("test_type = ?")
        args.append(test_type)
    if name:
        where.append("name LIKE ?")
        args.append(f"%{name}%")
    if date_from:
        where.append("created_at >= ?")
        args.append(date_from)
    if date_to:
        where.append("created_at <= ?")
        args.append(date_to)
    clause = f"WHERE {' AND '.join(where)}" if where else ""
    args.append(limit)
    rows = db().execute(
        f"SELECT * FROM test_run {clause} ORDER BY created_at DESC LIMIT ?", args
    ).fetchall()
    return _decode_all(rows)


def delete_run(run_id: int) -> None:
    _delete("test_run", run_id)
