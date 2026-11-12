"""Camera image controls: the ones the device really offers, with its own ranges.

Every control is read and written through DirectShow's IAMVideoProcAmp and
IAMCameraControl (see dshow.py) -- the interfaces Logi Tune and the Windows
camera dialog use. GetRange reports the device's true minimum, maximum, step and
default, and whether it can adjust the property automatically, so a slider spans
exactly what the camera accepts (0-255 for a C920e's brightness, 2000-6500 K for
its white balance) and a property the device lacks is simply absent.

This replaces reading the same properties through OpenCV, which exposes no
range, cannot tell automatic from manual, has no anti-flicker property at all,
and reports -1 for some controls that do exist. Discovering a range by writing a
wildly out-of-range value and reading back where the driver clamped it does not
work either: the C920e rejects such a write outright instead of clamping, which
made every control except exposure look unsupported.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

from app.camera import dshow
from app.camera.dshow import FLAG_AUTO, FLAG_MANUAL

AMP = "amp"  # IAMVideoProcAmp
CAM = "cam"  # IAMCameraControl


@dataclass(frozen=True)
class Control:
    key: str
    label: str
    iface: str
    prop: int
    group: str
    kind: str = "range"            # "range" | "toggle" | "choice"
    # Key of the companion "Auto ..." toggle, offered when the device reports it
    # can adjust this property by itself.
    auto_key: str | None = None
    auto_label: str | None = None
    choices: tuple[tuple[int, str], ...] = ()
    hint: str = ""


GROUP_LABELS = {
    "image": "Image",
    "exposure": "Exposure & light",
    "white_balance": "White balance",
    "focus": "Focus & zoom",
}

# Property ids are VideoProcAmpProperty / CameraControlProperty (strmif.h), plus
# kernel-streaming ids past the end of those enums that the same interfaces pass
# straight through (ksmedia.h): 13 is KSPROPERTY_VIDEOPROCAMP_POWERLINE_FREQUENCY
# and 19 is KSPROPERTY_CAMERACONTROL_AUTO_EXPOSURE_PRIORITY. Order here is the
# order rendered in the UI.
CONTROLS: tuple[Control, ...] = (
    Control("brightness", "Brightness", AMP, 0, "image"),
    Control("contrast", "Contrast", AMP, 1, "image"),
    Control("saturation", "Saturation", AMP, 3, "image"),
    Control("sharpness", "Sharpness", AMP, 4, "image"),
    Control("gamma", "Gamma", AMP, 5, "image"),
    Control("hue", "Hue", AMP, 2, "image"),
    Control("gain", "Gain", AMP, 9, "image"),

    Control("exposure", "Exposure", CAM, 4, "exposure",
            auto_key="auto_exposure", auto_label="Auto exposure",
            hint="log2 seconds: -5 is 1/32 s, -2 is 1/4 s"),
    # Labelled "Low Light Compensation" by Logi Tune and the Windows camera
    # dialog: the camera may lower its frame rate to expose longer in dim light.
    Control("low_light", "Low-light compensation", CAM, 19, "exposure", kind="toggle",
            hint="Lets the frame rate drop to expose longer in dim light"),
    Control("backlight", "Backlight compensation", AMP, 8, "exposure"),
    # UVC power-line frequency. Only the values inside the device's reported
    # range are offered -- a C920e has 50 Hz and 60 Hz.
    Control("anti_flicker", "Anti-flicker", AMP, 13, "exposure", kind="choice",
            choices=((0, "Off"), (1, "50 Hz"), (2, "60 Hz"), (3, "Auto")),
            hint="Match the mains frequency of the room lighting"),

    Control("wb_temperature", "Temperature (K)", AMP, 7, "white_balance",
            auto_key="auto_wb", auto_label="Auto white balance"),

    Control("focus", "Focus", CAM, 6, "focus",
            auto_key="autofocus", auto_label="Autofocus"),
    Control("zoom", "Zoom", CAM, 3, "focus"),
    Control("pan", "Pan", CAM, 0, "focus",
            hint="Digital: only moves the picture while zoomed in"),
    Control("tilt", "Tilt", CAM, 1, "focus",
            hint="Digital: only moves the picture while zoomed in"),
)

BY_KEY = {c.key: c for c in CONTROLS}
BY_AUTO_KEY = {c.auto_key: c for c in CONTROLS if c.auto_key}


class CameraControls:
    """The controls of one open camera.

    `bind(index)` returns the (IAMVideoProcAmp, IAMCameraControl) pair and `run`
    executes a callable on the thread that owns them; both default to the real
    DirectShow ones and are replaceable so the logic can be tested without a
    camera.
    """

    def __init__(self, bind: Callable[[int], tuple[Any, Any]] | None = None,
                 run: Callable[..., Any] | None = None) -> None:
        self._bind = bind or dshow.bind
        self._run = run or dshow.run
        self._ifaces: dict[str, Any] = {}
        self._ranges: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------ lifecycle

    def open(self, index: int) -> dict[str, dict[str, Any]]:
        """Bind device `index` and describe every control.

        Never raises: a camera whose controls cannot be reached still streams,
        it just has no sliders.
        """
        self.close()
        try:
            return self._run(self._open, index)
        except Exception:
            self._ifaces, self._ranges = {}, {}
            return {}

    def close(self) -> None:
        if not self._ifaces:
            return
        try:
            # Released on the thread that owns them, as COM requires.
            self._run(self._drop)
        except Exception:
            self._ifaces, self._ranges = {}, {}

    def _open(self, index: int) -> dict[str, dict[str, Any]]:
        amp, cam = self._bind(index)
        self._ifaces = {AMP: amp, CAM: cam}
        self._ranges = {}
        for c in CONTROLS:
            r = self._range(c)
            if r is not None:
                self._ranges[c.key] = r
        return self._describe()

    def _drop(self) -> None:
        self._ifaces, self._ranges = {}, {}

    # ------------------------------------------------------------ public API

    def read_all(self) -> dict[str, float]:
        if not self._ifaces:
            return {}
        return self._run(self._read_all)

    def apply(self, values: dict[str, Any]) -> dict[str, float]:
        """Write the given controls, then read everything back.

        Auto toggles are written first. A manual value sent in the same request
        as its own Auto switched on is ignored -- the device owns that value
        now. A manual value sent on its own switches Auto off, which is what
        dragging a slider means and what a saved profile always did.
        """
        if not self._ifaces:
            return {}
        return self._run(self._apply, dict(values))

    # ------------------------------------------------------------ COM-thread side

    def _range(self, c: Control) -> dict[str, Any] | None:
        iface = self._ifaces.get(c.iface)
        if iface is None:
            return None
        try:
            lo, hi, step, default, caps = iface.GetRange(c.prop)
            iface.Get(c.prop)  # some drivers answer GetRange for what they cannot read
        except Exception:
            return None

        kind = c.kind
        choices: list[dict[str, Any]] = []
        if kind == "toggle":
            # Boolean controls report a meaningless range ([0, 0] on a C920e).
            lo, hi, step = 0, 1, 1
        elif kind == "choice":
            choices = [{"value": v, "label": label}
                       for v, label in c.choices if lo <= v <= hi]
            if len(choices) < 2:
                return None
        else:
            if hi <= lo:
                return None
            if hi - lo == 1:
                kind = "toggle"
        return {
            "kind": kind, "lo": int(lo), "hi": int(hi), "step": max(int(step), 1),
            "default": int(default), "can_auto": bool(caps & FLAG_AUTO) and bool(c.auto_key),
            "choices": choices,
        }

    def _get(self, c: Control) -> tuple[int, int]:
        value, flags = self._ifaces[c.iface].Get(c.prop)
        return int(value), int(flags)

    def _set(self, c: Control, value: int, flags: int) -> None:
        try:
            self._ifaces[c.iface].Set(c.prop, int(value), flags)
        except Exception:
            # The read-back that follows every write shows it did not take.
            pass

    def _describe(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for c in CONTROLS:
            base = {"key": c.key, "label": c.label, "group": c.group,
                    "group_label": GROUP_LABELS[c.group]}
            r = self._ranges.get(c.key)
            if r is None:
                out[c.key] = {**base, "kind": c.kind, "supported": False}
                continue
            try:
                value, flags = self._get(c)
            except Exception:
                out[c.key] = {**base, "kind": c.kind, "supported": False}
                continue
            if r["can_auto"]:
                out[c.auto_key] = {
                    **base, "key": c.auto_key, "label": c.auto_label, "kind": "toggle",
                    "lo": 0, "hi": 1, "step": 1, "supported": True,
                    "value": 1.0 if flags & FLAG_AUTO else 0.0, "pairs_with": c.key,
                }
            out[c.key] = {
                **base, "kind": r["kind"], "lo": r["lo"], "hi": r["hi"],
                "step": r["step"], "default": r["default"], "value": float(value),
                "supported": True, "hint": c.hint, "choices": r["choices"],
                "auto_key": c.auto_key if r["can_auto"] else None,
            }
        return out

    def _read_all(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for c in CONTROLS:
            r = self._ranges.get(c.key)
            if r is None:
                continue
            try:
                value, flags = self._get(c)
            except Exception:
                continue
            out[c.key] = float(value)
            if r["can_auto"]:
                out[c.auto_key] = 1.0 if flags & FLAG_AUTO else 0.0
        return out

    def _apply(self, values: dict[str, Any]) -> dict[str, float]:
        auto_on: set[str] = set()
        for key, raw in values.items():
            c = BY_AUTO_KEY.get(key)
            r = self._ranges.get(c.key) if c else None
            if r is None or not r["can_auto"]:
                continue
            on = _as_switch(raw)
            if on is None:
                continue
            # Switching Auto off keeps whatever value the device had settled on,
            # which is how Logi Tune behaves too.
            current, _ = self._get(c)
            self._set(c, current, FLAG_AUTO if on else FLAG_MANUAL)
            if on:
                auto_on.add(c.key)

        for key, raw in values.items():
            c = BY_KEY.get(key)
            r = self._ranges.get(key)
            if c is None or r is None or key in auto_on:
                continue
            if r["kind"] == "choice":
                # Exact match only: clamping would quietly turn an unoffered
                # "Off" into the nearest offered frequency.
                value = _as_int(raw)
                if value not in {ch["value"] for ch in r["choices"]}:
                    continue
            else:
                value = _snap(raw, r)
                if value is None:
                    continue
            self._set(c, value, FLAG_MANUAL)

        return self._read_all()


def _as_switch(raw: Any) -> bool | None:
    """On/off from a toggle value; None for nothing usable.

    Negative values are skipped rather than read as "off": -1 is what OpenCV
    reported for controls it could not read, and profiles saved before this
    module existed may carry it.
    """
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return None
    if math.isnan(v) or v < 0:
        return None
    return v >= 0.5


def _as_int(raw: Any) -> int | None:
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) or v != int(v) else int(v)


def _snap(raw: Any, r: dict[str, Any]) -> int | None:
    """Clamp into the device's range and onto its step grid.

    The C920e rejects an out-of-range or off-grid write outright rather than
    clamping it, so a value from a profile saved on another camera, or a focus
    of 83 on a step-5 control, must be made valid before it is sent.
    """
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return None
    if math.isnan(v):
        return None
    lo, hi, step = r["lo"], r["hi"], r["step"]
    v = min(max(v, lo), hi)
    snapped = lo + round((v - lo) / step) * step
    return int(min(snapped, hi))
