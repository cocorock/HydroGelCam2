"""Single-owner camera session.

Only one thread ever holds the cv2.VideoCapture handle. It grabs continuously
into a latest-frame slot; the MJPEG stream and the still-capture endpoint both
read that slot. This matters on Windows, where opening a second handle on a UVC
device that is already in use fails outright.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Iterator

import cv2
import numpy as np

from app import config
from app.camera import controls, devices


class CameraSession:
    def __init__(self) -> None:
        self._cap: cv2.VideoCapture | None = None
        self._lock = threading.Lock()
        self._frame_lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._frame_seq = 0
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

        self.device_index: int | None = None
        self.device_name: str | None = None
        self.backend: str = devices.DEFAULT_BACKEND
        self.capabilities: dict[str, dict[str, Any]] = {}
        self.modes: list[dict] = []
        self.last_error: str | None = None

    # ------------------------------------------------------------ lifecycle

    @property
    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def open(
        self,
        index: int,
        backend: str = devices.DEFAULT_BACKEND,
        width: int | None = None,
        height: int | None = None,
        name: str | None = None,
    ) -> dict[str, Any]:
        # A resolution not requested at all is a resolution the driver picks for
        # itself, and that default is very often a low-bandwidth mode like
        # 1280x720 rather than the sensor's native resolution. Asking for the
        # configured preferred size up front is what makes a fresh connect land
        # on it; the device still negotiates to the nearest mode it actually
        # supports, and the true result is read back into `status()` below.
        width = width or config.DEFAULT_WIDTH
        height = height or config.DEFAULT_HEIGHT

        with self._lock:
            self._close_locked()

            cap = cv2.VideoCapture(index, devices.backend_id(backend))
            if not cap.isOpened():
                cap.release()
                self.last_error = (
                    f"Could not open camera {index} with the {backend} backend. "
                    "It may be in use by another application."
                )
                raise RuntimeError(self.last_error)

            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

            self._cap = cap
            self.device_index = index
            self.device_name = name
            self.backend = backend
            self.last_error = None
            # Not a full sweep of every candidate resolution here -- on many
            # DirectShow drivers each mode change briefly restarts the video
            # stream, and sweeping nine of them was the single biggest cost in
            # how long a connect took. `detect_modes()` runs that sweep, but only
            # when explicitly asked for.
            self.modes = []
            self.capabilities = controls.probe(cap)

            self._stop.clear()
            self._thread = threading.Thread(
                target=self._grab_loop, name="camera-grab", daemon=True
            )
            self._thread.start()

        self._wait_for_frame(timeout=3.0)
        return self.status()

    def detect_modes(self) -> list[dict]:
        """On-demand full sweep of every candidate resolution.

        Deliberately not run on every connect -- see `open()`. The sweep needs
        exclusive use of the capture handle while it cycles modes, so the grab
        thread is paused and restarted around it rather than left reading
        concurrently, which OpenCV's DirectShow backend does not guarantee is
        safe from two threads at once.
        """
        with self._lock:
            cap = self._cap
            if cap is None:
                raise RuntimeError("No camera is open.")

            self._stop.set()
            if self._thread is not None and self._thread.is_alive():
                self._thread.join(timeout=2.0)
            try:
                self.modes = devices.supported_modes(cap)
            finally:
                self._stop.clear()
                self._thread = threading.Thread(
                    target=self._grab_loop, name="camera-grab", daemon=True
                )
                self._thread.start()
            return self.modes

    def close(self) -> None:
        with self._lock:
            self._close_locked()

    def _close_locked(self) -> None:
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        with self._frame_lock:
            self._frame = None
        self.capabilities = {}
        self.modes = []
        self.device_index = None
        self.device_name = None

    # ------------------------------------------------------------ grabbing

    def _grab_loop(self) -> None:
        interval = 1.0 / max(config.STREAM_FPS, 1)
        misses = 0
        while not self._stop.is_set():
            cap = self._cap
            if cap is None:
                break
            ok, frame = cap.read()
            if ok and frame is not None:
                misses = 0
                with self._frame_lock:
                    self._frame = frame
                    self._frame_seq += 1
            else:
                misses += 1
                if misses > 50:
                    self.last_error = "Camera stopped delivering frames."
                    break
                time.sleep(0.02)
            time.sleep(interval * 0.5)

    def _wait_for_frame(self, timeout: float) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._frame_lock:
                if self._frame is not None:
                    return
            time.sleep(0.02)

    def latest_frame(self) -> np.ndarray | None:
        with self._frame_lock:
            return None if self._frame is None else self._frame.copy()

    # ------------------------------------------------------------ properties

    def set_props(self, values: dict[str, Any]) -> dict[str, float]:
        with self._lock:
            if self._cap is None:
                raise RuntimeError("No camera is open.")
            applied = controls.apply(self._cap, values)
        for key, val in applied.items():
            if key in self.capabilities:
                self.capabilities[key]["value"] = val
        return applied

    def get_props(self) -> dict[str, float]:
        with self._lock:
            if self._cap is None:
                return {}
            return controls.read_all(self._cap)

    def status(self) -> dict[str, Any]:
        frame = self.latest_frame()
        return {
            "open": self.is_open,
            "device_index": self.device_index,
            "device_name": self.device_name,
            "backend": self.backend,
            "width": 0 if frame is None else int(frame.shape[1]),
            "height": 0 if frame is None else int(frame.shape[0]),
            "capabilities": self.capabilities,
            "modes": self.modes,
            "error": self.last_error,
        }

    # ------------------------------------------------------------ streaming

    def mjpeg(self) -> Iterator[bytes]:
        boundary = b"--frame\r\n"
        interval = 1.0 / max(config.STREAM_FPS, 1)
        params = [int(cv2.IMWRITE_JPEG_QUALITY), config.JPEG_QUALITY]
        last_seq = -1
        while self.is_open:
            with self._frame_lock:
                frame = None if self._frame is None else self._frame
                seq = self._frame_seq
                if frame is not None and seq != last_seq:
                    frame = frame.copy()
                    last_seq = seq
                else:
                    frame = None
            if frame is None:
                time.sleep(interval / 2)
                continue
            ok, buf = cv2.imencode(".jpg", frame, params)
            if not ok:
                continue
            payload = buf.tobytes()
            yield (
                boundary
                + b"Content-Type: image/jpeg\r\n"
                + f"Content-Length: {len(payload)}\r\n\r\n".encode()
                + payload
                + b"\r\n"
            )
            time.sleep(interval)


session = CameraSession()
