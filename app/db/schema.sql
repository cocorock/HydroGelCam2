-- HydroGelCam2 schema. SQLite, WAL, no authentication (single local user).
-- Images live on disk; only their paths are stored here, so the .db stays small
-- and the photos remain openable in ImageJ for cross-checking.

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS camera_profile (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    name             TEXT NOT NULL UNIQUE,
    backend          TEXT NOT NULL DEFAULT 'dshow',
    device_index     INTEGER NOT NULL DEFAULT 0,
    device_name      TEXT,
    width            INTEGER,
    height           INTEGER,
    props_json       TEXT NOT NULL DEFAULT '{}',
    hsv_ranges_json  TEXT NOT NULL DEFAULT '{}',
    created_at       TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS calibration (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    name                TEXT NOT NULL UNIQUE,
    -- 'top_down' for the uniformity/fusion glass-plate view,
    -- 'lateral'  for the collapse platform side view.
    intended_use        TEXT NOT NULL DEFAULT 'top_down',

    -- Stage A: intrinsic / distortion
    board_cols          INTEGER NOT NULL,
    board_rows          INTEGER NOT NULL,
    square_mm           REAL NOT NULL,
    K_json              TEXT,
    dist_json           TEXT,
    rms_px              REAL,
    n_frames            INTEGER DEFAULT 0,

    -- Stage B: pixel -> mm, measured in the measurement plane.
    -- The checker size here is independent of stage A's.
    scale_board_cols    INTEGER,
    scale_board_rows    INTEGER,
    scale_square_mm     REAL,
    mode                TEXT DEFAULT 'scalar',      -- 'scalar' | 'homography'
    mm_per_px_x         REAL,
    mm_per_px_y         REAL,
    anisotropy          REAL,
    H_json              TEXT,
    scale_image_path    TEXT,

    notes               TEXT,
    created_at          TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS test_run (
    id                     INTEGER PRIMARY KEY AUTOINCREMENT,
    test_type              TEXT NOT NULL,           -- uniformity | fusion | collapse
    name                   TEXT NOT NULL DEFAULT '',
    replicate_no           INTEGER,

    flow_rate_mms          REAL,
    feed_rate_mms          REAL,
    nozzle_id_mm           REAL,
    first_layer_height_mm  REAL,

    calibration_id         INTEGER REFERENCES calibration(id) ON DELETE SET NULL,
    camera_profile_id      INTEGER REFERENCES camera_profile(id) ON DELETE SET NULL,

    image_path             TEXT,
    -- The same frame with the measurement drawn on it: ROI, ticks, contours.
    -- Kept as a separate file so the original capture stays untouched.
    overlay_path           TEXT,
    roi_json               TEXT NOT NULL DEFAULT '{}',
    params_json            TEXT NOT NULL DEFAULT '{}',
    results_json           TEXT NOT NULL DEFAULT '{}',

    convention             TEXT,                    -- Cf convention for collapse runs
    flags_json             TEXT NOT NULL DEFAULT '[]',
    notes                  TEXT,
    created_at             TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at             TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_run_type    ON test_run(test_type);
CREATE INDEX IF NOT EXISTS idx_run_name    ON test_run(name);
CREATE INDEX IF NOT EXISTS idx_run_created ON test_run(created_at DESC);

-- One row per thickness (tab 2), per pore (tab 3) or per gap (tab 4).
-- raw_json holds the geometry, which is what lets tab 5 recompute the
-- formulas without re-running any image processing.
CREATE TABLE IF NOT EXISTS measurement (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    test_run_id  INTEGER NOT NULL REFERENCES test_run(id) ON DELETE CASCADE,
    index_no     INTEGER NOT NULL,
    label        TEXT,
    included     INTEGER NOT NULL DEFAULT 1,
    raw_json     TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_meas_run ON measurement(test_run_id, index_no);
