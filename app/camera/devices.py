"""USB camera enumeration and backend selection.

OpenCV cannot report device names, only indices. On Windows pygrabber reads the
DirectShow friendly names, which is what makes "USB Endoscope" distinguishable
from "Integrated Webcam" in the dropdown. Everywhere else we fall back to
probing indices.
"""

from __future__ import annotations

import sys

import cv2

BACKENDS = {
    "dshow": cv2.CAP_DSHOW,   # Windows default: exposes the most UVC properties
    "msmf": cv2.CAP_MSMF,     # Windows fallback: better with some 4K sensors
    "any": cv2.CAP_ANY,
}

DEFAULT_BACKEND = "dshow" if sys.platform == "win32" else "any"

# Modes offered in the resolution dropdown, filtered at connect time to the ones
# the device actually accepts.
CANDIDATE_MODES = (
    (640, 480), (800, 600), (1024, 768), (1280, 720), (1280, 960),
    (1600, 1200), (1920, 1080), (2560, 1440), (3840, 2160),
)


def backend_id(name: str) -> int:
    return BACKENDS.get(name, cv2.CAP_ANY)


def list_devices() -> list[dict]:
    """Return [{index, name}] for every attached camera."""
    names = _directshow_names()
    if names:
        return [{"index": i, "name": n} for i, n in enumerate(names)]
    return [{"index": i, "name": f"Camera {i}"} for i in _probe_indices()]


def _directshow_names() -> list[str]:
    if sys.platform != "win32":
        return []
    try:
        from pygrabber.dshow_graph import FilterGraph
    except ImportError:
        return []
    try:
        return list(FilterGraph().get_input_devices())
    except Exception:
        # pygrabber raises assorted COM errors when no device is present.
        return []


def _probe_indices(limit: int = 8) -> list[int]:
    found = []
    for i in range(limit):
        cap = cv2.VideoCapture(i, backend_id(DEFAULT_BACKEND))
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
