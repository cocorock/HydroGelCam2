"""Camera controls read through DirectShow's control interfaces.

No camera is needed: fake IAMVideoProcAmp / IAMCameraControl objects reproduce
what a Logitech C920e actually reports (captured from the real device), including
the behaviour that broke the previous approach -- an out-of-range Set() is
rejected outright, not clamped.
"""

from __future__ import annotations

import pytest

from app.camera import controls
from app.camera.dshow import FLAG_AUTO, FLAG_MANUAL


class DriverError(Exception):
    """Stands in for comtypes.COMError (E_INVALIDARG / ERROR_NOT_FOUND)."""


class FakeIface:
    """One control interface; `props` maps property id -> its state."""

    def __init__(self, props: dict[int, dict]):
        self.props = {pid: dict(p) for pid, p in props.items()}
        self.writes: list[tuple[int, int, int]] = []

    def GetRange(self, pid):
        p = self._prop(pid)
        return p["lo"], p["hi"], p["step"], p["default"], p["caps"]

    def Get(self, pid):
        p = self._prop(pid)
        return p["value"], p["flags"]

    def Set(self, pid, value, flags):
        p = self._prop(pid)
        lo, hi = p.get("accept", (p["lo"], p["hi"]))
        if not lo <= value <= hi or (value - lo) % max(p["step"], 1):
            raise DriverError("E_INVALIDARG")  # the C920e rejects, never clamps
        p["value"], p["flags"] = value, flags
        self.writes.append((pid, value, flags))

    def _prop(self, pid):
        if pid not in self.props:
            raise DriverError("ERROR_NOT_FOUND")
        return self.props[pid]


def prop(lo, hi, value, *, step=1, default=None, caps=FLAG_MANUAL, flags=FLAG_MANUAL, **extra):
    return {"lo": lo, "hi": hi, "step": step, "default": lo if default is None else default,
            "caps": caps, "value": value, "flags": flags, **extra}


def c920e() -> tuple[FakeIface, FakeIface]:
    """The ranges and states read from a real C920e."""
    amp = FakeIface({
        0: prop(0, 255, 136, default=128),                       # brightness
        1: prop(0, 255, 135, default=128),                       # contrast
        3: prop(0, 255, 139, default=128),                       # saturation
        4: prop(0, 255, 132, default=128),                       # sharpness
        7: prop(2000, 6500, 3811, default=4000,                  # white balance, auto on
                caps=FLAG_AUTO | FLAG_MANUAL, flags=FLAG_AUTO),
        8: prop(0, 1, 0),                                        # backlight compensation
        9: prop(0, 255, 185),                                    # gain
        13: prop(1, 2, 2, default=2, caps=0, flags=0),           # power-line frequency
    })
    cam = FakeIface({
        0: prop(-10, 10, 0),                                     # pan
        1: prop(-10, 10, 0),                                     # tilt
        3: prop(100, 500, 100, default=100),                     # zoom
        4: prop(-11, -2, -4, default=-5, caps=FLAG_AUTO | FLAG_MANUAL),  # exposure
        6: prop(0, 250, 30, step=5, caps=FLAG_AUTO | FLAG_MANUAL),       # focus
        # Auto-exposure priority reports a meaningless [0, 0] range but takes 0/1.
        19: prop(0, 0, 1, step=0, caps=0, flags=0, accept=(0, 1)),
    })
    return amp, cam


def make(amp=None, cam=None) -> tuple[controls.CameraControls, FakeIface, FakeIface]:
    if amp is None and cam is None:
        amp, cam = c920e()
    ctl = controls.CameraControls(bind=lambda _index: (amp, cam),
                                  run=lambda fn, *args: fn(*args))
    return ctl, amp, cam


# ---------------------------------------------------------------- describing


def test_every_control_the_c920e_offers_is_supported_with_its_real_range():
    ctl, _, _ = make()
    caps = ctl.open(0)

    for key in ("brightness", "contrast", "saturation", "sharpness", "gain"):
        assert caps[key]["supported"] is True
        assert (caps[key]["lo"], caps[key]["hi"]) == (0, 255)
    assert caps["brightness"]["default"] == 128
    assert (caps["wb_temperature"]["lo"], caps["wb_temperature"]["hi"]) == (2000, 6500)
    assert (caps["exposure"]["lo"], caps["exposure"]["hi"]) == (-11, -2)
    assert caps["focus"]["step"] == 5
    assert caps["zoom"]["hi"] == 500


def test_properties_the_device_lacks_are_reported_unsupported():
    ctl, _, _ = make()
    caps = ctl.open(0)
    assert caps["hue"]["supported"] is False
    assert caps["gamma"]["supported"] is False
    assert "lo" not in caps["hue"]


def test_auto_toggles_exist_only_where_the_device_can_auto_adjust():
    ctl, _, _ = make()
    caps = ctl.open(0)

    assert caps["auto_wb"]["value"] == 1.0, "white balance is in auto on this camera"
    assert caps["auto_exposure"]["value"] == 0.0
    assert caps["autofocus"]["value"] == 0.0
    assert caps["auto_wb"]["pairs_with"] == "wb_temperature"
    assert caps["wb_temperature"]["auto_key"] == "auto_wb"
    # Brightness has no auto mode, so no toggle and no pairing.
    assert "auto_brightness" not in caps
    assert caps["brightness"]["auto_key"] is None


def test_auto_toggle_is_listed_just_before_its_manual_control():
    ctl, _, _ = make()
    keys = list(ctl.open(0))
    assert keys.index("auto_exposure") + 1 == keys.index("exposure")
    assert keys.index("autofocus") + 1 == keys.index("focus")


def test_anti_flicker_offers_only_the_frequencies_the_device_supports():
    ctl, _, _ = make()
    flicker = ctl.open(0)["anti_flicker"]
    assert flicker["kind"] == "choice"
    assert [c["label"] for c in flicker["choices"]] == ["50 Hz", "60 Hz"]
    assert flicker["value"] == 2.0


def test_low_light_compensation_is_a_toggle_despite_its_degenerate_range():
    ctl, _, _ = make()
    low = ctl.open(0)["low_light"]
    assert low["supported"] is True
    assert low["kind"] == "toggle"
    assert (low["lo"], low["hi"]) == (0, 1)
    assert low["value"] == 1.0


def test_a_zero_or_one_range_is_presented_as_a_toggle():
    ctl, _, _ = make()
    assert ctl.open(0)["backlight"]["kind"] == "toggle"


def test_open_never_raises_when_the_device_cannot_be_bound():
    def boom(_index):
        raise DriverError("no such device")

    ctl = controls.CameraControls(bind=boom, run=lambda fn, *a: fn(*a))
    assert ctl.open(3) == {}
    assert ctl.apply({"brightness": 10}) == {}
    assert ctl.read_all() == {}


# ---------------------------------------------------------------- writing


def test_a_slider_value_is_written_in_manual_mode_and_read_back():
    ctl, amp, _ = make()
    ctl.open(0)
    out = ctl.apply({"brightness": 200})
    assert out["brightness"] == 200.0
    assert amp.writes[-1] == (0, 200, FLAG_MANUAL)


def test_out_of_range_and_off_step_values_are_made_valid_before_sending():
    """The C920e rejects these writes outright, so they must be clamped and
    snapped first -- e.g. a profile saved on a camera with a -64..64 range."""
    ctl, _, cam = make()
    ctl.open(0)
    out = ctl.apply({"brightness": -21, "focus": 83, "zoom": 9999})
    assert out["brightness"] == 0.0
    assert out["focus"] == 85.0
    assert out["zoom"] == 500.0


def test_switching_auto_on_hands_the_value_to_the_device():
    ctl, _, cam = make()
    ctl.open(0)
    out = ctl.apply({"autofocus": 1})
    assert out["autofocus"] == 1.0
    assert cam.props[6]["flags"] == FLAG_AUTO


def test_a_manual_value_sent_alongside_auto_on_is_ignored():
    """Applying a profile saved with autofocus on must not switch it off again
    by writing the focus value it also happens to carry."""
    ctl, _, cam = make()
    ctl.open(0)
    out = ctl.apply({"autofocus": 1, "focus": 120})
    assert out["autofocus"] == 1.0
    assert cam.props[6]["value"] == 30, "focus left to the camera"


def test_a_manual_value_on_its_own_switches_auto_off():
    """What dragging the slider means, and what older profiles relied on."""
    ctl, amp, _ = make()
    ctl.open(0)
    out = ctl.apply({"wb_temperature": 5000})
    assert out["wb_temperature"] == 5000.0
    assert out["auto_wb"] == 0.0


def test_switching_auto_off_keeps_the_value_the_device_settled_on():
    ctl, amp, _ = make()
    ctl.open(0)
    out = ctl.apply({"auto_wb": 0})
    assert out["auto_wb"] == 0.0
    assert out["wb_temperature"] == 3811.0


def test_a_legacy_minus_one_auto_value_is_ignored_not_read_as_off():
    """OpenCV reported -1 for auto controls it could not read; a profile saved
    then must not silently switch auto white balance off."""
    ctl, _, _ = make()
    ctl.open(0)
    out = ctl.apply({"auto_wb": -1})
    assert out["auto_wb"] == 1.0


def test_anti_flicker_and_low_light_can_be_changed():
    ctl, _, _ = make()
    ctl.open(0)
    out = ctl.apply({"anti_flicker": 1, "low_light": 0})
    assert out["anti_flicker"] == 1.0
    assert out["low_light"] == 0.0


def test_a_choice_outside_the_offered_list_is_not_sent():
    ctl, amp, _ = make()
    ctl.open(0)
    out = ctl.apply({"anti_flicker": 0})  # "Off" is not offered by a C920e
    assert out["anti_flicker"] == 2.0
    assert not any(pid == 13 for pid, _, _ in amp.writes)


def test_unknown_keys_and_unsupported_controls_are_ignored():
    ctl, amp, cam = make()
    ctl.open(0)
    ctl.apply({"hue": 10, "not_a_control": 5, "brightness": None})
    assert amp.writes == [] and cam.writes == []


def test_a_write_the_driver_rejects_does_not_raise():
    amp, cam = c920e()
    amp.props[0]["accept"] = (999, 999)  # this driver refuses every brightness
    ctl, _, _ = make(amp, cam)
    ctl.open(0)
    assert ctl.apply({"brightness": 50})["brightness"] == 136.0


def test_close_releases_the_interfaces():
    ctl, _, _ = make()
    ctl.open(0)
    ctl.close()
    assert ctl.read_all() == {}
