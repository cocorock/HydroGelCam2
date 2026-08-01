"""UVC camera property map and capability probing.

OpenCV happily accepts a `set()` on a property the device does not implement and
returns True, so the only reliable way to know what a camera supports is to write
a value and read it back. `probe()` does that once at connect time; the UI then
renders a slider only for properties that actually moved.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import cv2


@dataclass(frozen=True)
class PropSpec:
    key: str
    label: str
    cv_id: int
    lo: float
    hi: float
    step: float = 1.0
    kind: str = "range"      # "range" | "toggle"
    group: str = "image"


# Ranges are the UVC-typical spans; the probe narrows them to what the device
# actually reports where it can. Order here is the order rendered in the UI.
PROPS: tuple[PropSpec, ...] = (
    PropSpec("brightness", "Brightness", cv2.CAP_PROP_BRIGHTNESS, -64, 64),
    PropSpec("contrast", "Contrast", cv2.CAP_PROP_CONTRAST, 0, 100),
    PropSpec("saturation", "Saturation", cv2.CAP_PROP_SATURATION, 0, 100),
    PropSpec("sharpness", "Sharpness", cv2.CAP_PROP_SHARPNESS, 0, 100),
    PropSpec("gamma", "Gamma", cv2.CAP_PROP_GAMMA, 30, 300),
    PropSpec("hue", "Hue", cv2.CAP_PROP_HUE, -180, 180),
    PropSpec("gain", "Gain", cv2.CAP_PROP_GAIN, 0, 100),

    # UVC calls this "backlight compensation"; on most endoscopy cameras it is
    # the control marketed as low-light compensation.
    PropSpec("backlight", "Low-light compensation", cv2.CAP_PROP_BACKLIGHT,
             0, 2, group="exposure"),

    PropSpec("auto_exposure", "Auto exposure", cv2.CAP_PROP_AUTO_EXPOSURE,
             0.25, 0.75, step=0.25, kind="toggle", group="exposure"),
    PropSpec("exposure", "Exposure", cv2.CAP_PROP_EXPOSURE, -13, 0,
             group="exposure"),

    PropSpec("auto_wb", "Auto white balance", cv2.CAP_PROP_AUTO_WB,
             0, 1, kind="toggle", group="white_balance"),
    PropSpec("wb_temperature", "WB temperature (K)",
             cv2.CAP_PROP_WB_TEMPERATURE, 2000, 10000, step=10,
             group="white_balance"),

    PropSpec("autofocus", "Autofocus", cv2.CAP_PROP_AUTOFOCUS,
             0, 1, kind="toggle", group="focus"),
    PropSpec("focus", "Focus", cv2.CAP_PROP_FOCUS, 0, 255, group="focus"),
    PropSpec("zoom", "Zoom", cv2.CAP_PROP_ZOOM, 0, 500, group="focus"),
)

BY_KEY = {p.key: p for p in PROPS}

# Toggles whose "auto" state disables a companion manual control.
AUTO_PAIRS = {
    "auto_exposure": "exposure",
    "auto_wb": "wb_temperature",
    "autofocus": "focus",
}


def read_all(cap: cv2.VideoCapture) -> dict[str, float]:
    return {p.key: float(cap.get(p.cv_id)) for p in PROPS}


def apply(cap: cv2.VideoCapture, values: dict[str, Any]) -> dict[str, float]:
    """Set the given properties, then read everything back.

    Auto flags are written first so that a manual value sent in the same request
    is not immediately overridden by the driver re-enabling automatic control.
    """
    ordered = sorted(values.items(), key=lambda kv: kv[0] not in AUTO_PAIRS)
    for key, val in ordered:
        spec = BY_KEY.get(key)
        if spec is None or val is None:
            continue
        try:
            cap.set(spec.cv_id, float(val))
        except (TypeError, ValueError):
            continue
    return read_all(cap)


def probe(cap: cv2.VideoCapture) -> dict[str, dict[str, Any]]:
    """Write-then-read each property to find out which ones the device honours.

    A property is considered supported when the device either reports a
    plausible current value or accepts a nudged value. The original value is
    always restored.
    """
    caps: dict[str, dict[str, Any]] = {}
    for spec in PROPS:
        original = cap.get(spec.cv_id)
        supported = False

        # A device that does not implement a property typically returns 0 or -1
        # for every read. Nudge it and see whether the read follows.
        probe_value = spec.hi if original <= (spec.lo + spec.hi) / 2 else spec.lo
        try:
            cap.set(spec.cv_id, float(probe_value))
            after = cap.get(spec.cv_id)
            supported = abs(after - original) > 1e-6 or _plausible(spec, original)
            cap.set(spec.cv_id, float(original))
        except (TypeError, ValueError, cv2.error):
            supported = False

        caps[spec.key] = {
            **asdict(spec),
            "supported": bool(supported),
            "value": float(cap.get(spec.cv_id)),
            "pairs_with": AUTO_PAIRS.get(spec.key),
        }
    return caps


def _plausible(spec: PropSpec, value: float) -> bool:
    """A read strictly inside the range (and not a sentinel) suggests support."""
    if value in (-1.0, 0.0):
        return False
    return spec.lo <= value <= spec.hi
