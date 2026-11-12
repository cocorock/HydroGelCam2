"""USB camera enumeration. DirectShow only.

OpenCV cannot report device names, only indices; DirectShow lists the friendly
names ("Logi Webcam C920e") in the same order OpenCV's DirectShow backend
numbers the devices, which is what makes the dropdown readable.
"""

from __future__ import annotations

import cv2

from app.camera import dshow

# DirectShow is the only backend: it is the one whose devices carry the UVC
# controls (see controls.py), and the only one in use on the measurement rig.
BACKEND = "dshow"
BACKEND_ID = cv2.CAP_DSHOW

# Modes offered in the resolution dropdown, filtered at connect time to the ones
# the device actually accepts.
CANDIDATE_MODES = (
    (640, 480), (800, 600), (1024, 768), (1280, 720), (1280, 960),
    (1600, 1200), (1920, 1080), (2560, 1440), (3840, 2160),
)


def list_devices() -> list[dict]:
    """Return [{index, name}] for every attached camera."""
    names = dshow.device_names()
    if names:
        return [{"index": i, "name": n} for i, n in enumerate(names)]
    return [{"index": i, "name": f"Camera {i}"} for i in _probe_indices()]


def _probe_indices(limit: int = 8) -> list[int]:
    found = []
    for i in range(limit):
        cap = cv2.VideoCapture(i, BACKEND_ID)
        if cap.isOpened():
            found.append(i)
        cap.release()
    return found


def supported_modes(cap: cv2.VideoCapture) -> list[dict]:
    """Ask the device to adopt each candidate mode and keep the ones it honours."""
    original = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    modes: list[dict] = []
    seen: set[tuple[int, int]] = set()
    for w, h in CANDIDATE_MODES:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        got = (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        if got not in seen and got[0] > 0:
            seen.add(got)
            modes.append({"width": got[0], "height": got[1]})
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, original[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, original[1])
    return sorted(modes, key=lambda m: m["width"] * m["height"])
