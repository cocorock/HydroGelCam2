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
    """Find out which properties the device honours, and their real range.

    Range ("slider") properties get their true min/max discovered directly, not
    just a supported/unsupported flag -- see `_probe_range`. Toggle properties
    have a fixed two-value domain, so they keep the simpler nudge-and-check.
    """
    caps: dict[str, dict[str, Any]] = {}
    for spec in PROPS:
        caps[spec.key] = (_probe_toggle(cap, spec) if spec.kind == "toggle"
                          else _probe_range(cap, spec))
    return caps


def _probe_toggle(cap: cv2.VideoCapture, spec: PropSpec) -> dict[str, Any]:
    """Write-then-read: does a nudge to the spec's own hi/lo actually move it?

    A device that does not implement a property typically returns 0 or -1 for
    every read; nudging it and checking whether the read follows is how support
    is inferred. The original value is always restored.
    """
    original = cap.get(spec.cv_id)
    supported = False

    probe_value = spec.hi if original <= (spec.lo + spec.hi) / 2 else spec.lo
    try:
        cap.set(spec.cv_id, float(probe_value))
        after = cap.get(spec.cv_id)
        supported = abs(after - original) > 1e-6 or _plausible(spec, original)
        cap.set(spec.cv_id, float(original))
    except (TypeError, ValueError, cv2.error):
        supported = False

    return {
        **asdict(spec),
        "supported": bool(supported),
        "value": float(cap.get(spec.cv_id)),
        "pairs_with": AUTO_PAIRS.get(spec.key),
    }


# Clamp probes set well outside any plausible UVC value, so the driver's own
# Set() clamp reveals the true bound instead of us guessing one. DirectShow's
# IAMVideoProcAmp/IAMCameraControl always clamp a Set() to GetRange()'s bounds,
# and OpenCV passes the clamped value straight through on the following Get() --
# OpenCV has no direct range-query API, so this is the reliable way to learn a
# specific device's actual span (which can be -64..64, 0..255, or something
# else entirely) instead of rendering a slider against a generic guess that
# either clips off part of the real range or accepts values the device ignores.
_CLAMP_LOW = -1_000_000.0
_CLAMP_HIGH = 1_000_000.0
# A "discovered" bound further from zero than this means the driver did not
# actually clamp (e.g. it silently ignored the out-of-range Set and the read
# reflects something else), so the result is discarded in favour of the guess.
_SANE_BOUND = 100_000.0


def _probe_range(cap: cv2.VideoCapture, spec: PropSpec) -> dict[str, Any]:
    """Discover a slider property's true min/max via the Set-and-clamp trick.

    The original value is always restored, even if a Set/Get call raises
    partway through. A partial result -- the low probe succeeding and the high
    one raising, or vice versa -- is discarded rather than trusted: pairing a
    freshly discovered bound with whatever `discovered_hi`/`discovered_lo`
    happened to hold before the failure can look like a perfectly plausible
    (but wrong) range, so both probes have to complete before either is used.
    """
    original = cap.get(spec.cv_id)
    discovered_lo = discovered_hi = None
    probed_ok = False

    try:
        cap.set(spec.cv_id, _CLAMP_LOW)
        discovered_lo = cap.get(spec.cv_id)
        cap.set(spec.cv_id, _CLAMP_HIGH)
        discovered_hi = cap.get(spec.cv_id)
        probed_ok = True
    except (TypeError, ValueError, cv2.error):
        probed_ok = False
    finally:
        try:
            cap.set(spec.cv_id, float(original))
        except (TypeError, ValueError, cv2.error):
            pass

    lo, hi, supported = spec.lo, spec.hi, False
    if (probed_ok
            and discovered_hi > discovered_lo
            and abs(discovered_lo) < _SANE_BOUND
            and abs(discovered_hi) < _SANE_BOUND):
        lo, hi, supported = discovered_lo, discovered_hi, True

    return {
        **asdict(spec),
        "lo": lo, "hi": hi,
        "supported": bool(supported),
        "value": float(cap.get(spec.cv_id)),
        "pairs_with": AUTO_PAIRS.get(spec.key),
    }


def _plausible(spec: PropSpec, value: float) -> bool:
    """A read strictly inside the range (and not a sentinel) suggests support."""
    if value in (-1.0, 0.0):
        return False
    return spec.lo <= value <= spec.hi
