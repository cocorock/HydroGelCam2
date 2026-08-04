"""FastAPI application: routes for the camera, calibration, the three tests and
the database browser."""

from __future__ import annotations

import csv
import io
import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import (
    FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app import config
from app.analysis import collapse, fusion, uniformity
from app.calib import geometry, intrinsic, scale as scale_mod
from app.camera import devices, stream
from app.db import repo

class NoCacheStatic(StaticFiles):
    """Serve the front end with caching off.

    This runs on the same machine as the browser, so there is nothing to gain
    from caching, and plenty to lose: an edited script silently not taking
    effect looks exactly like a bug in the code you just changed. ES modules
    make it worse, since an import specifier cannot carry a cache-busting query.
    """

    def is_not_modified(self, response_headers, request_headers) -> bool:
        return False

    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-store, must-revalidate"
        return response


app = FastAPI(title="HydroGelCam2")
app.mount("/static", NoCacheStatic(directory=str(config.STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(config.TEMPLATE_DIR))

PENDING = config.CAPTURE_DIR / "_pending"
PENDING.mkdir(parents=True, exist_ok=True)

ANALYZERS = {
    "uniformity": uniformity,
    "fusion": fusion,
    "collapse": collapse,
}


@app.on_event("startup")
def _startup() -> None:
    repo.init_db()
    _purge_pending()


def _purge_pending(max_age_hours: float = 24.0) -> int:
    """Drop stale files from the capture staging area.

    Every captured frame, uploaded file and overlay lands here before it is
    either saved to a run or abandoned. Saving copies the file out, so anything
    left behind is scratch -- without this the directory grows by a few megabytes
    per session and never shrinks.
    """
    cutoff = time.time() - max_age_hours * 3600
    removed = 0
    for path in PENDING.glob("*.png"):
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
                removed += 1
        except OSError:
            continue
    return removed


@app.on_event("shutdown")
def _shutdown() -> None:
    stream.session.close()


# ---------------------------------------------------------------- pages


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {
        "test_types": config.TEST_TYPES,
        "polarity_choices": config.POLARITY_CHOICES,
        "polarity_labels": config.POLARITY_LABELS,
        "default_polarity": config.DEFAULT_POLARITY,
        "defaults": {
            "board_cols": config.DEFAULT_BOARD_COLS,
            "board_rows": config.DEFAULT_BOARD_ROWS,
            "square_mm": config.DEFAULT_SQUARE_MM,
            "min_frames": config.MIN_CALIB_FRAMES,
            "uniformity_filaments": config.UNIFORMITY_FILAMENTS,
            "uniformity_positions": config.UNIFORMITY_POSITIONS,
            "fusion_grid_n": config.FUSION_GRID_N,
            "fusion_fd_mm": list(config.FUSION_FD_MM),
            "collapse_gaps_mm": list(config.COLLAPSE_GAPS_MM),
            "collapse_pillars_mm": list(config.COLLAPSE_PILLAR_WIDTHS_MM),
            "collapse_gap_height_mm": config.COLLAPSE_GAP_HEIGHT_MM,
            "cf_convention": config.CF_CONVENTION,
            "cf_convention_labels": config.CF_CONVENTION_LABELS,
            "pr_window": [config.PR_ACCEPT_LOW, config.PR_ACCEPT_HIGH],
            "roi_padding_px": config.ROI_PADDING_PX,
            "final_kernel": config.FINAL_MORPH_KERNEL,
            "hsv_defaults": config.HSV_DEFAULTS,
        },
    })


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    """Browsers request this path directly, whatever the <link> tag says."""
    path = config.STATIC_DIR / "favicon.ico"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No favicon.")
    return FileResponse(path, media_type="image/x-icon")


# ---------------------------------------------------------------- camera


@app.get("/api/camera/devices")
def camera_devices():
    return {"devices": devices.list_devices(),
            "backends": list(devices.BACKENDS),
            "default_backend": devices.DEFAULT_BACKEND}


@app.post("/api/camera/open")
async def camera_open(payload: dict):
    try:
        return stream.session.open(
            index=int(payload.get("index", 0)),
            backend=payload.get("backend", devices.DEFAULT_BACKEND),
            width=payload.get("width"),
            height=payload.get("height"),
            name=payload.get("name"),
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@app.post("/api/camera/close")
def camera_close():
    stream.session.close()
    return {"open": False}


@app.get("/api/camera/status")
def camera_status():
    return stream.session.status()


@app.get("/api/camera/stream")
def camera_stream():
    if not stream.session.is_open:
        raise HTTPException(status_code=409, detail="No camera is open.")
    return StreamingResponse(
        stream.session.mjpeg(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.post("/api/camera/props")
def camera_props(payload: dict):
    try:
        return {"values": stream.session.set_props(payload.get("values", {}))}
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@app.post("/api/camera/capture")
def camera_capture():
    frame = stream.session.latest_frame()
    if frame is None:
        raise HTTPException(status_code=409, detail="No frame available.")
    return _stage(frame)


@app.post("/api/image/upload")
async def image_upload(file: UploadFile = File(...)):
    data = np.frombuffer(await file.read(), np.uint8)
    frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode that image.")
    return _stage(frame)


def _stage(frame: np.ndarray) -> dict[str, Any]:
    image_id = uuid.uuid4().hex
    path = PENDING / f"{image_id}.png"
    cv2.imwrite(str(path), frame)
    return {
        "image_id": image_id,
        "width": int(frame.shape[1]),
        "height": int(frame.shape[0]),
        "url": f"/api/image/{image_id}",
    }


@app.get("/api/image/{image_id}")
def image_get(image_id: str):
    path = PENDING / f"{image_id}.png"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Image not found.")
    return FileResponse(path, media_type="image/png")


@app.get("/api/run-image/{run_id}")
def run_image(run_id: int, kind: str = "capture"):
    """The stored photograph, or the annotated overlay saved beside it."""
    run = repo.get_run(run_id)
    column = "overlay_path" if kind == "overlay" else "image_path"
    if not run or not run.get(column):
        raise HTTPException(status_code=404, detail=f"No {kind} image for that run.")
    path = Path(run[column])
    if not path.exists():
        raise HTTPException(status_code=404, detail="Image file is missing.")
    return FileResponse(path, media_type="image/png")


@app.post("/api/shutdown")
def shutdown():
    """Release the camera and stop the server.

    A local single-user tool has no other way to hand the camera back, and
    leaving uvicorn running in a terminal the user has closed is worse than an
    abrupt exit. The camera is released and the database checkpointed first;
    everything after that is already committed to disk.
    """
    stream.session.close()
    try:
        repo.db().execute("PRAGMA wal_checkpoint(TRUNCATE)")
        repo.db().commit()
    except Exception:
        pass

    def _stop() -> None:
        # Long enough for this response to reach the browser.
        time.sleep(0.4)
        os._exit(0)

    threading.Thread(target=_stop, daemon=True).start()
    return {"stopping": True}


# ---------------------------------------------------------------- camera profiles


@app.get("/api/camera/profiles")
def camera_profiles():
    return {"profiles": repo.list_camera_profiles()}


@app.post("/api/camera/profiles")
def camera_profile_save(payload: dict):
    if not payload.get("name"):
        raise HTTPException(status_code=400, detail="A profile name is required.")
    return {"id": repo.save_camera_profile(payload)}


@app.delete("/api/camera/profiles/{profile_id}")
def camera_profile_delete(profile_id: int):
    repo.delete_camera_profile(profile_id)
    return {"deleted": profile_id}


# ---------------------------------------------------------------- calibration


@app.get("/api/calib/intrinsic/state")
def calib_state():
    return intrinsic.session.state()


@app.post("/api/calib/intrinsic/reset")
def calib_reset(payload: dict):
    intrinsic.session.reset(
        int(payload.get("cols", config.DEFAULT_BOARD_COLS)),
        int(payload.get("rows", config.DEFAULT_BOARD_ROWS)),
        float(payload.get("square_mm", config.DEFAULT_SQUARE_MM)),
    )
    return intrinsic.session.state()


@app.post("/api/calib/intrinsic/add")
def calib_add(payload: dict | None = None):
    payload = payload or {}
    frame = _resolve_frame(payload)
    result = intrinsic.session.add(frame)
    result["state"] = intrinsic.session.state()
    return result


@app.delete("/api/calib/intrinsic/frame/{index}")
def calib_remove(index: int):
    intrinsic.session.remove(index)
    return intrinsic.session.state()


@app.get("/api/calib/intrinsic/thumb/{index}")
def calib_thumb(index: int):
    png = intrinsic.session.thumbnail(index)
    if png is None:
        raise HTTPException(status_code=404, detail="No such frame.")
    return Response(content=png, media_type="image/png")


@app.post("/api/calib/intrinsic/solve")
def calib_solve():
    try:
        return intrinsic.session.solve()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/calib/scale")
def calib_scale(payload: dict):
    frame = _resolve_frame(payload)

    K = dist = None
    result = intrinsic.session.result
    if result:
        K, dist = result["K"], result["dist"]
    elif payload.get("calibration_id"):
        stored = repo.get_calibration(int(payload["calibration_id"]))
        if stored:
            K, dist = stored.get("K_json"), stored.get("dist_json")

    try:
        out = scale_mod.compute(
            frame,
            cols=int(payload.get("cols", config.DEFAULT_BOARD_COLS)),
            rows=int(payload.get("rows", config.DEFAULT_BOARD_ROWS)),
            square_mm=float(payload.get("square_mm", config.DEFAULT_SQUARE_MM)),
            K=K, dist=dist,
            mode=payload.get("mode", "scalar"),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    overlay = out.pop("overlay")
    ok, buf = cv2.imencode(".png", overlay)
    out["overlay_url"] = None
    if ok:
        staged = PENDING / f"scale_{uuid.uuid4().hex}.png"
        staged.write_bytes(buf.tobytes())
        out["overlay_url"] = f"/api/image/{staged.stem}"
    out["anisotropy_warn"] = out["anisotropy"] > config.ANISOTROPY_WARN
    out["has_intrinsic"] = K is not None
    return out


@app.get("/api/calib/list")
def calib_list(intended_use: str | None = None):
    return {"calibrations": repo.list_calibrations(intended_use)}


@app.post("/api/calib/save")
def calib_save(payload: dict):
    if not payload.get("name"):
        raise HTTPException(status_code=400, detail="A calibration name is required.")

    values: dict[str, Any] = {
        "name": payload["name"],
        "intended_use": payload.get("intended_use", "top_down"),
        "board_cols": int(payload.get("board_cols", config.DEFAULT_BOARD_COLS)),
        "board_rows": int(payload.get("board_rows", config.DEFAULT_BOARD_ROWS)),
        "square_mm": float(payload.get("square_mm", config.DEFAULT_SQUARE_MM)),
        "notes": payload.get("notes"),
    }

    result = intrinsic.session.result
    if result:
        values.update({
            "K_json": result["K"],
            "dist_json": result["dist"],
            "rms_px": result["rms_px"],
            "n_frames": result["n_frames"],
        })

    for key in ("scale_board_cols", "scale_board_rows", "scale_square_mm",
                "mode", "mm_per_px_x", "mm_per_px_y", "anisotropy"):
        if payload.get(key) is not None:
            values[key] = payload[key]
    if payload.get("H") is not None:
        values["H_json"] = payload["H"]

    return {"id": repo.save_calibration(values)}


@app.delete("/api/calib/{calib_id}")
def calib_delete(calib_id: int):
    repo.delete_calibration(calib_id)
    return {"deleted": calib_id}


# ---------------------------------------------------------------- analysis


@app.post("/api/test/{test_type}/analyze")
def test_analyze(test_type: str, payload: dict):
    module = ANALYZERS.get(test_type)
    if module is None:
        raise HTTPException(status_code=404, detail=f"Unknown test '{test_type}'.")

    frame = _resolve_frame(payload)
    profile = None
    if payload.get("calibration_id"):
        profile = repo.get_calibration(int(payload["calibration_id"]))
    scale = geometry.from_profile(profile)

    frame = geometry.undistort(frame, scale)

    try:
        out = module.analyze(
            frame,
            payload.get("roi"),
            payload.get("params") or {},
            scale,
            debug=bool(payload.get("debug")),
        )
    except Exception as exc:  # analysis must never 500 the tab
        raise HTTPException(status_code=422,
                            detail=f"{type(exc).__name__}: {exc}")

    out["calibrated"] = scale.calibrated
    out["calibration_name"] = scale.name
    out["image_id"] = payload.get("image_id")
    return out


@app.post("/api/test/{test_type}/recompute")
def test_recompute(test_type: str, payload: dict):
    """Re-run the formulas over stored measurements, with no image processing."""
    measurements = payload.get("measurements") or []
    params = payload.get("params") or {}

    if test_type == "uniformity":
        return {"results": uniformity.compute(measurements,
                                              params.get("nozzle_id_mm"))}
    if test_type == "fusion":
        return {"results": fusion.compute(measurements)}
    if test_type == "collapse":
        return {"results": collapse.compute(
            measurements, params.get("convention", config.CF_CONVENTION))}
    raise HTTPException(status_code=404, detail=f"Unknown test '{test_type}'.")


# ---------------------------------------------------------------- runs


@app.post("/api/runs")
def run_create(payload: dict):
    test_type = payload.get("test_type")
    if test_type not in config.TEST_TYPES:
        raise HTTPException(status_code=400, detail="Unknown test type.")

    image_path = _persist_image(payload.get("image_id"), test_type)
    overlay_path = _persist_image(payload.get("overlay_id"), test_type,
                                  suffix="_overlay")

    values = {
        "test_type": test_type,
        "name": payload.get("name") or "",
        "replicate_no": payload.get("replicate_no"),
        "flow_rate_mms": payload.get("flow_rate_mms"),
        "feed_rate_mms": payload.get("feed_rate_mms"),
        "nozzle_id_mm": payload.get("nozzle_id_mm"),
        "first_layer_height_mm": payload.get("first_layer_height_mm"),
        "calibration_id": payload.get("calibration_id"),
        "camera_profile_id": payload.get("camera_profile_id"),
        "image_path": image_path,
        "overlay_path": overlay_path,
        "roi_json": payload.get("roi") or {},
        "params_json": payload.get("params") or {},
        "results_json": payload.get("results") or {},
        "convention": payload.get("convention"),
        "flags_json": payload.get("flags") or [],
        "notes": payload.get("notes"),
    }
    run_id = repo.create_run(values, payload.get("measurements") or [])
    return {"id": run_id}


def _persist_image(image_id: str | None, test_type: str,
                   suffix: str = "") -> str | None:
    if not image_id:
        return None
    src = PENDING / f"{image_id}.png"
    if not src.exists():
        return None
    dest_dir = config.CAPTURE_DIR / test_type
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{image_id}{suffix}.png"
    dest.write_bytes(src.read_bytes())
    return str(dest)


@app.get("/api/runs")
def run_list(test_type: str | None = None, name: str | None = None,
             date_from: str | None = None, date_to: str | None = None):
    return {"runs": repo.list_runs(test_type, name, date_from, date_to)}


@app.get("/api/runs/{run_id}")
def run_get(run_id: int):
    run = repo.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="No such run.")
    return run


@app.put("/api/runs/{run_id}")
def run_update(run_id: int, payload: dict):
    if repo.get_run(run_id) is None:
        raise HTTPException(status_code=404, detail="No such run.")
    values = {k: v for k, v in payload.items()
              if k in {"name", "replicate_no", "flow_rate_mms", "feed_rate_mms",
                       "nozzle_id_mm", "first_layer_height_mm", "notes",
                       "convention", "calibration_id"}}
    for src, dst in (("roi", "roi_json"), ("params", "params_json"),
                     ("results", "results_json"), ("flags", "flags_json")):
        if payload.get(src) is not None:
            values[dst] = payload[src]
    repo.update_run(run_id, values, payload.get("measurements"))
    return repo.get_run(run_id)


@app.delete("/api/runs/{run_id}")
def run_delete(run_id: int):
    repo.delete_run(run_id)
    return {"deleted": run_id}


@app.post("/api/runs/{run_id}/duplicate")
def run_duplicate(run_id: int):
    run = repo.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="No such run.")
    values = {k: v for k, v in run.items()
              if k not in {"id", "created_at", "updated_at", "measurements"}}
    values["name"] = f"{run['name']} (copy)"
    new_id = repo.create_run(values, run["measurements"])
    return {"id": new_id}


@app.post("/api/runs/{run_id}/reanalyze")
def run_reanalyze(run_id: int, payload: dict | None = None):
    """Re-run the full pipeline on a stored image with its stored ROI/params."""
    payload = payload or {}
    run = repo.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="No such run.")
    if not run.get("image_path") or not Path(run["image_path"]).exists():
        raise HTTPException(status_code=400, detail="This run has no stored image.")

    frame = cv2.imread(run["image_path"], cv2.IMREAD_COLOR)
    profile = repo.get_calibration(run["calibration_id"]) if run["calibration_id"] else None
    scale = geometry.from_profile(profile)
    frame = geometry.undistort(frame, scale)

    module = ANALYZERS[run["test_type"]]
    params = payload.get("params") or run.get("params_json") or {}
    roi = payload.get("roi") or run.get("roi_json") or {}
    out = module.analyze(frame, roi, params, scale,
                         debug=bool(payload.get("debug")))
    return out


# ---------------------------------------------------------------- export


@app.get("/api/export/runs.csv")
def export_flat(test_type: str | None = None):
    runs = repo.list_runs(test_type)
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow([
        "id", "test_type", "name", "replicate_no", "created_at",
        "flow_rate_mms", "feed_rate_mms", "nozzle_id_mm", "first_layer_height_mm",
        "calibration_id", "convention", "measurement", "field", "value",
    ])
    for run in runs:
        for m in repo.get_measurements(run["id"]):
            for key, value in (m.get("raw") or {}).items():
                if isinstance(value, (list, dict)):
                    value = json.dumps(value)
                writer.writerow([
                    run["id"], run["test_type"], run["name"], run["replicate_no"],
                    run["created_at"], run["flow_rate_mms"], run["feed_rate_mms"],
                    run["nozzle_id_mm"], run["first_layer_height_mm"],
                    run["calibration_id"], run["convention"],
                    m.get("label"), key, value,
                ])
    return _csv_response(buf, "hydrogelcam_runs.csv")


@app.get("/api/export/summary.csv")
def export_summary(test_type: str | None = None):
    """Mean +/- SD across replicates, grouped by sample and size class."""
    from app.analysis.stats import describe

    runs = repo.list_runs(test_type)
    groups: dict[tuple, dict[str, list[float]]] = {}

    metric_keys = {
        "uniformity": ("thickness_mm",),
        "fusion": ("aa_mm2", "dfr_percent", "pr", "circularity"),
        "collapse": ("a_sag_mm2", "cf_percent", "theta_deg"),
    }
    class_keys = {
        "uniformity": "filament",
        "fusion": "nominal_fd_mm",
        "collapse": "nominal_gap_mm",
    }

    for run in runs:
        tt = run["test_type"]
        ck = class_keys.get(tt)
        for m in repo.get_measurements(run["id"]):
            if not m.get("included"):
                continue
            raw = m.get("raw") or {}
            key = (tt, run["name"], raw.get(ck))
            bucket = groups.setdefault(key, {})
            for metric in metric_keys.get(tt, ()):
                value = raw.get(metric)
                if isinstance(value, (int, float)):
                    bucket.setdefault(metric, []).append(float(value))

    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(["test_type", "name", "size_class", "metric", "n", "mean", "sd"])
    for (tt, name, size_class), metrics in sorted(
        groups.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]), str(kv[0][2]))
    ):
        for metric, values in metrics.items():
            d = describe(values)
            writer.writerow([tt, name, size_class, metric,
                             d["n"], d["mean"], d["sd"]])
    return _csv_response(buf, "hydrogelcam_summary.csv")


def _csv_response(buf: io.StringIO, filename: str) -> Response:
    return Response(
        content=buf.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ---------------------------------------------------------------- helpers


def _resolve_frame(payload: dict) -> np.ndarray:
    """Take a frame from a staged image id, or grab one from the live camera."""
    image_id = payload.get("image_id")
    if image_id:
        path = PENDING / f"{image_id}.png"
        if not path.exists():
            raise HTTPException(status_code=404, detail="Staged image not found.")
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(status_code=400, detail="Could not read that image.")
        return frame

    frame = stream.session.latest_frame()
    if frame is None:
        raise HTTPException(
            status_code=409,
            detail="No image supplied and no camera is streaming.",
        )
    return frame


@app.exception_handler(HTTPException)
async def _http_error(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})
