"""Real min/max discovery for camera sliders, and the toggle probe it left alone.

No physical camera is needed: a fake `cv2.VideoCapture` stand-in simulates a
driver that clamps Set() to its own range, which is the behaviour the whole
scheme depends on.
"""

from __future__ import annotations

import pytest

from app.camera import controls


class FakeCap:
    """Mimics one property: DirectShow clamps Set() to [true_lo, true_hi]."""

    def __init__(self, initial: float, true_lo: float, true_hi: float):
        self.value = float(initial)
        self.true_lo = true_lo
        self.true_hi = true_hi

    def get(self, _cv_id):
        return self.value

    def set(self, _cv_id, value):
        self.value = max(self.true_lo, min(self.true_hi, float(value)))
        return True


class UnsupportedCap:
    """Mimics a property the driver does not implement: Set() is a no-op."""

    def __init__(self, sentinel: float = -1.0):
        self.value = sentinel

    def get(self, _cv_id):
        return self.value

    def set(self, _cv_id, _value):
        return True  # OpenCV reports success even when nothing changed


def spec_for(key: str) -> controls.PropSpec:
    return controls.BY_KEY[key]


# ---------------------------------------------------------------- range probe


def test_discovers_a_range_wider_than_the_generic_guess():
    """Brightness is guessed as -64..64; this device's real range is -100..200."""
    cap = FakeCap(initial=10.0, true_lo=-100.0, true_hi=200.0)
    result = controls._probe_range(cap, spec_for("brightness"))

    assert result["supported"] is True
    assert result["lo"] == pytest.approx(-100.0)
    assert result["hi"] == pytest.approx(200.0)


def test_discovers_a_range_narrower_than_the_generic_guess():
    """Gamma is guessed as 30..300; this device only goes 100..250."""
    cap = FakeCap(initial=150.0, true_lo=100.0, true_hi=250.0)
    result = controls._probe_range(cap, spec_for("gamma"))

    assert result["supported"] is True
    assert result["lo"] == pytest.approx(100.0)
    assert result["hi"] == pytest.approx(250.0)


def test_original_value_is_always_restored():
    cap = FakeCap(initial=42.0, true_lo=-64.0, true_hi=64.0)
    controls._probe_range(cap, spec_for("brightness"))
    assert cap.value == pytest.approx(42.0)


def test_original_value_is_restored_even_when_a_call_raises():
    class RaisingCap(FakeCap):
        def set(self, cv_id, value):
            if value == controls._CLAMP_HIGH:
                raise TypeError("driver rejected the call")
            return super().set(cv_id, value)

    cap = RaisingCap(initial=7.0, true_lo=-64.0, true_hi=64.0)
    result = controls._probe_range(cap, spec_for("brightness"))

    assert cap.value == pytest.approx(7.0)
    assert result["supported"] is False, "a failed discovery must not be trusted"


def test_an_unimplemented_property_is_reported_unsupported_not_zero_span():
    cap = UnsupportedCap(sentinel=-1.0)
    result = controls._probe_range(cap, spec_for("sharpness"))

    assert result["supported"] is False
    # Falls back to the generic guess rather than a real but useless [-1, -1].
    spec = spec_for("sharpness")
    assert result["lo"] == spec.lo
    assert result["hi"] == spec.hi


def test_an_absurd_clamp_reading_is_rejected_as_unsupported():
    """A driver that ignores the Set entirely but still reports some large,
    unrelated value must not hand the UI a runaway slider range."""
    cap = FakeCap(initial=5.0, true_lo=-2_000_000.0, true_hi=2_000_000.0)
    result = controls._probe_range(cap, spec_for("brightness"))
    assert result["supported"] is False


def test_probe_dispatches_range_and_toggle_properties_differently():
    """probe() must route sliders through range discovery and toggles through
    the simpler nudge check, not the same path for both."""
    class AnyCap(FakeCap):
        pass

    cap = AnyCap(initial=0.0, true_lo=-64.0, true_hi=64.0)
    caps = controls.probe(cap)

    assert "lo" in caps["brightness"] and "hi" in caps["brightness"]
    assert caps["auto_exposure"]["kind"] == "toggle"
    assert set(caps.keys()) == {p.key for p in controls.PROPS}


# ---------------------------------------------------------------- toggle probe


def test_toggle_probe_reports_support_when_the_value_moves():
    cap = FakeCap(initial=0.25, true_lo=0.25, true_hi=0.75)
    result = controls._probe_toggle(cap, spec_for("auto_exposure"))
    assert result["supported"] is True
    assert cap.value == pytest.approx(0.25), "restored to the original state"


def test_toggle_probe_reports_unsupported_for_a_fixed_property():
    cap = UnsupportedCap(sentinel=0.0)
    result = controls._probe_toggle(cap, spec_for("auto_wb"))
    assert result["supported"] is False
