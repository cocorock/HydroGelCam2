"""Preprocessing steps 1-5, shared by every test tab.

    1. load / undistort
    2. crop to the ROI, then surround it with an 85 px constant-colour border
    3. flatten the vignette
    4. derive a threshold from the rising edge of the histogram's second peak
    5. binarize

Tabs 2 and 4 stop here. Tab 3 continues into `pipeline.grid` for steps 6-8.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from app import config
from app.pipeline.debug import DebugTrace
from app.pipeline.signal1d import find_peaks, gaussian_smooth


@dataclass
class Roi:
    x: int
    y: int
    w: int
    h: int

    @classmethod
    def from_json(cls, data: dict[str, Any] | None, shape: tuple[int, ...]) -> "Roi":
        h, w = shape[:2]
        if not data:
            return cls.default(shape)
        return cls(
            int(round(data.get("x", 0))),
            int(round(data.get("y", 0))),
            int(round(data.get("w", w))),
            int(round(data.get("h", h))),
        ).clamped(shape)

    @classmethod
    def default(cls, shape: tuple[int, ...], inset: float = 0.15) -> "Roi":
        """The pre-drawn red rectangle: a centred box covering the middle ~70 %."""
        h, w = shape[:2]
        dx, dy = int(w * inset), int(h * inset)
        return cls(dx, dy, w - 2 * dx, h - 2 * dy)

    def clamped(self, shape: tuple[int, ...]) -> "Roi":
        h, w = shape[:2]
        x = max(0, min(self.x, w - 1))
        y = max(0, min(self.y, h - 1))
        return Roi(x, y, max(1, min(self.w, w - x)), max(1, min(self.h, h - y)))

    def slice(self, image: np.ndarray) -> np.ndarray:
        return image[self.y:self.y + self.h, self.x:self.x + self.w]

    def as_dict(self) -> dict[str, int]:
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h}


@dataclass
class Segmentation:
    """Everything downstream needs, in one place.

    `offset` maps crop coordinates back to the full frame. With the ROI padded
    rather than expanded, the padding sits *outside* the source image, so the
    offset is the ROI's own origin minus the padding width and can be negative.
    """

    source: np.ndarray            # full undistorted frame (BGR)
    crop: np.ndarray              # padded ROI crop (BGR)
    gray: np.ndarray              # crop, grayscale
    flat: np.ndarray              # crop after vignette flattening
    work: np.ndarray              # flat, oriented so material is always bright
    binary: np.ndarray            # step 5 output, 0/255
    valid: np.ndarray             # True for real image pixels, False for padding
    threshold: float
    roi: Roi                      # user ROI, full-frame coordinates
    padding: int
    offset: tuple[int, int]
    roi_in_crop: Roi              # user ROI expressed in crop coordinates
    polarity: str
    pad_value: int
    trace: DebugTrace = field(default_factory=DebugTrace)
    meta: dict[str, Any] = field(default_factory=dict)


def preprocess(
    image: np.ndarray,
    roi_json: dict[str, Any] | None = None,
    *,
    padding: int = config.ROI_PADDING_PX,
    vignette_kernel: int = config.VIGNETTE_KERNEL,
    smooth_passes: int = config.HIST_SMOOTH_PASSES,
    smooth_sigma: float = config.HIST_SMOOTH_SIGMA,
    threshold_margin: float = config.THRESHOLD_MARGIN,
    polarity: str = config.DEFAULT_POLARITY,
    manual_threshold: float | None = None,
    trace: DebugTrace | None = None,
) -> Segmentation:
    trace = trace or DebugTrace()

    trace.add("1_source", image, "Captured frame after lens-distortion correction.")

    # ---- step 2: crop to the ROI, then pad -------------------------------
    roi = Roi.from_json(roi_json, image.shape)
    inner = roi.slice(image).copy()

    gray_inner = (cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY)
                  if inner.ndim == 3 else inner.copy())

    # Polarity is decided on the ROI's own pixels, before any padding is added,
    # and before flattening -- the flattening needs to know which way round the
    # material is, and the padding colour depends on the same answer.
    if polarity == "auto":
        polarity = detect_polarity(gray_inner)

    # The border is filled with the background colour, so it can never read as
    # material: black behind bright material, white behind dark material.
    pad_value = 0 if polarity == "bright" else 255
    crop = cv2.copyMakeBorder(
        inner, padding, padding, padding, padding,
        cv2.BORDER_CONSTANT,
        value=(pad_value, pad_value, pad_value) if inner.ndim == 3 else pad_value,
    )

    valid = np.zeros(crop.shape[:2], dtype=bool)
    valid[padding:padding + roi.h, padding:padding + roi.w] = True

    if trace.enabled:
        marked = image.copy()
        cv2.rectangle(marked, (roi.x, roi.y),
                      (roi.x + roi.w, roi.y + roi.h), (0, 0, 255), 2)
        trace.add("2_roi", marked,
                  f"Red = user ROI, {roi.w}x{roi.h} px. Only these pixels are "
                  "analysed.")
        trace.add("2b_padded", crop,
                  f"Cropped to the ROI and surrounded by a {padding} px "
                  f"{'black' if pad_value == 0 else 'white'} border, matching the "
                  "background for this material appearance. Nothing outside the "
                  "ROI can influence the result.")

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop.copy()

    # ---- step 3: vignette flattening -------------------------------------
    flat = flatten_vignette(gray, vignette_kernel, polarity, valid)
    trace.add("3_flattened", flat,
              f"Divided by a {vignette_kernel} px estimate of the illumination "
              "field, rescaled by that field's median. Removes the "
              "darker-toward-the-edges falloff.")

    # ---- step 4: adaptive threshold --------------------------------------
    work = flat if polarity == "bright" else cv2.bitwise_not(flat)

    if manual_threshold is not None:
        threshold = float(manual_threshold)
        thr_meta: dict[str, Any] = {"source": "manual"}
    else:
        threshold, thr_meta = adaptive_threshold(
            work, smooth_passes, smooth_sigma, threshold_margin, trace, valid
        )

    # ---- step 5: binarize -------------------------------------------------
    _, binary = cv2.threshold(work, threshold, 255, cv2.THRESH_BINARY)
    # Belt and braces: the padding is background-coloured, so it should already
    # fall below the threshold, but forcing it keeps a stray bright border from
    # ever being measured as material.
    binary[~valid] = 0
    trace.add_mask("5_binary", binary,
                   f"Binarized at {threshold:.1f} ({polarity} material). "
                   "White = deposited material.")

    roi_in_crop = Roi(padding, padding, roi.w, roi.h)

    return Segmentation(
        source=image,
        crop=crop,
        gray=gray,
        flat=flat,
        work=work,
        binary=binary,
        valid=valid,
        threshold=threshold,
        roi=roi,
        padding=padding,
        # Crop coordinate (padding, padding) is the ROI's top-left in the frame,
        # so the offset is the ROI origin shifted back by the padding width.
        offset=(roi.x - padding, roi.y - padding),
        roi_in_crop=roi_in_crop,
        polarity=polarity,
        pad_value=pad_value,
        trace=trace,
        meta={"threshold": thr_meta, "padding": padding, "pad_value": pad_value},
    )


# ---------------------------------------------------------------- step 3


def flatten_vignette(
    gray: np.ndarray,
    kernel: int = config.VIGNETTE_KERNEL,
    polarity: str = "bright",
    valid: np.ndarray | None = None,
) -> np.ndarray:
    """Estimate the slow-varying illumination field and divide it out.

    The field is fitted as a 2-D quadratic over *background pixels only*. Two
    cheaper estimators were tried first and both fail on real prints:

    - A plain blur of the image is inflated by the filaments themselves, so
      background near the print normalises darker than background far from it,
      and no single global threshold can work.
    - Suppressing the material with a morphological open first fixes that for
      thin filaments, but not for a solid region wider than the structuring
      element -- and a fully fused corner is exactly the expected result of the
      fusion test at 1 mm spacing. That region survives into the field estimate
      and, once the result is rescaled by the field's maximum, amplifies the
      whole frame until everything thresholds as material.

    A low-order surface cannot represent a filament at all, so no amount of
    deposited material can corrupt it, and vignetting is genuinely low-order.

    Rescaling uses the field's median rather than its maximum: both are just a
    global gain, but the median maps background back to its own original level
    and so preserves the material-to-background contrast, where the maximum
    pushes the material past 255 and clips it away.
    """
    k = int(kernel) | 1  # cv2 requires an odd kernel

    # Estimate the field from the real pixels only. The ROI's own padding is a
    # flat 0 or 255 that is not part of the scene's illumination; dividing by a
    # field dragged towards it throws bright artefacts around the whole border.
    if valid is not None and not valid.all():
        ys, xs = np.nonzero(valid)
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        inner = gray[y0:y1, x0:x1]
        inner_field = _illumination_field(inner, k, polarity)
        # Extend the field outwards by replicating its edge, so the padding is
        # divided by a plausible local background rather than by nothing.
        field = cv2.copyMakeBorder(
            inner_field, y0, gray.shape[0] - y1, x0, gray.shape[1] - x1,
            cv2.BORDER_REPLICATE,
        )
        median = float(np.median(inner_field))
    else:
        field = _illumination_field(gray, k, polarity)
        median = float(np.median(field))

    flat = gray.astype(np.float32) / field * median
    return np.clip(flat, 0, 255).astype(np.uint8)


def _illumination_field(
    gray: np.ndarray,
    k: int,
    polarity: str,
    coarse: int = config.FIELD_COARSE_PX,
    se_divisor: int = config.FIELD_SE_DIVISOR,
) -> np.ndarray:
    """Smooth positive estimate of the background illumination across the crop.

    A grayscale open (bright material) or close (dark material) removes anything
    smaller than the structuring element, leaving the illumination behind. The
    work is done at heavily reduced resolution, which is what makes it robust:
    a fully fused corner hundreds of pixels across at full size covers only a
    handful of pixels here, so a modest structuring element removes it, while at
    full resolution no practical element could.

    Before the opening the coarse image is framed with a band of background
    sampled along each of its own edges. An opening cannot remove a feature that
    runs into the border -- there is no background on that side to erode from --
    and a filament near the edge of the ROI does exactly that, surviving into the
    field and flattening its own surroundings far too dark. Sampling the frame
    per column and per row keeps the illumination gradient intact, which a single
    constant fill would not.

    Deliberately threshold-free. Classifying background first with a global Otsu
    fails exactly when the field matters most -- under strong falloff, material
    in the dark corner reads below the global threshold and is counted as
    background, corrupting the very estimate meant to fix that.
    """
    h, w = gray.shape[:2]
    scale = min(1.0, coarse / max(h, w))
    cw = max(16, int(round(w * scale)))
    ch = max(16, int(round(h * scale)))

    small = cv2.resize(gray, (cw, ch), interpolation=cv2.INTER_AREA)

    se_size = max(9, min(cw, ch) // max(1, se_divisor)) | 1
    pad = se_size
    framed = _frame_with_edge_background(small, pad, polarity)

    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (se_size, se_size))
    op = cv2.MORPH_OPEN if polarity == "bright" else cv2.MORPH_CLOSE
    background = cv2.morphologyEx(framed, op, se).astype(np.float32)
    background = cv2.GaussianBlur(background, (9, 9), 0)
    background = background[pad:pad + ch, pad:pad + cw]

    field = cv2.resize(background, (w, h), interpolation=cv2.INTER_CUBIC)
    return np.maximum(field, 1.0)


def _frame_with_edge_background(small: np.ndarray, pad: int,
                                polarity: str) -> np.ndarray:
    """Surround `small` with background sampled along its own four edges."""
    q = config.FIELD_PERCENTILE
    if polarity != "bright":
        q = 100.0 - q
    band = max(2, pad)
    h, w = small.shape[:2]
    band_y = min(band, h)
    band_x = min(band, w)

    out = np.empty((h + 2 * pad, w + 2 * pad), dtype=small.dtype)
    out[pad:pad + h, pad:pad + w] = small

    top = np.percentile(small[:band_y, :], q, axis=0)
    bottom = np.percentile(small[-band_y:, :], q, axis=0)
    left = np.percentile(small[:, :band_x], q, axis=1)
    right = np.percentile(small[:, -band_x:], q, axis=1)

    out[:pad, pad:pad + w] = top.astype(small.dtype)
    out[pad + h:, pad:pad + w] = bottom.astype(small.dtype)
    out[pad:pad + h, :pad] = left.astype(small.dtype)[:, None]
    out[pad:pad + h, pad + w:] = right.astype(small.dtype)[:, None]

    # Corners take the nearest edge value, so the frame stays continuous.
    out[:pad, :pad] = out[pad, pad]
    out[:pad, pad + w:] = out[pad, pad + w - 1]
    out[pad + h:, :pad] = out[pad + h - 1, pad]
    out[pad + h:, pad + w:] = out[pad + h - 1, pad + w - 1]
    return out


def detect_polarity(gray: np.ndarray,
                    kernel: int = config.VIGNETTE_KERNEL) -> str:
    """Best guess at whether the material is brighter or darker than background.

    Flattens both ways and keeps whichever yields the smaller plausible material
    fraction, on the grounds that the printed pattern never fills the ROI.

    This is a fallback, not the default -- see `config.DEFAULT_POLARITY`. On
    synthetic sweeps it is right about 90 % of the time, and the cases it misses
    are genuinely ambiguous: a wide filament covering a quarter of the ROI can
    produce nearly the same material fraction either way. Whether a given dye
    reads bright or dark is a fixed property of the rig and the lighting, so it
    is far safer to state it once in the UI than to re-guess it per frame and be
    silently wrong.
    """
    best: tuple[float, str] | None = None
    for polarity in ("bright", "dark"):
        flat = flatten_vignette(gray, kernel, polarity)
        work = flat if polarity == "bright" else cv2.bitwise_not(flat)
        threshold, _ = adaptive_threshold(work)
        fraction = float((work > threshold).mean())
        # Anything covering essentially the whole crop is the inverted reading.
        score = fraction if 0.005 <= fraction <= 0.80 else 1.0 + fraction
        if best is None or score < best[0]:
            best = (score, polarity)
    return best[1] if best else "bright"


# ---------------------------------------------------------------- step 4


def adaptive_threshold(
    image: np.ndarray,
    smooth_passes: int = config.HIST_SMOOTH_PASSES,
    smooth_sigma: float = config.HIST_SMOOTH_SIGMA,
    margin: float = config.THRESHOLD_MARGIN,
    trace: DebugTrace | None = None,
    valid: np.ndarray | None = None,
) -> tuple[float, dict[str, Any]]:
    """Threshold from the rising edge of the histogram's second peak.

    Otsu's own split sits midway between the two modes, which cuts into the
    blurred halo around each filament and inflates every measured width. The
    transition *into* solid material is the rising edge of the upper peak, so we
    find that instead and back off by a small margin.

    The histogram is built from the pixels inside the ROI and nothing else.
    `valid` excludes the constant-coloured padding: it is not scene content, and
    a border of identical pixels would put a spike of tens of thousands of counts
    into a single bin. That spike sets the scale for the prominence test, which
    can then suppress the genuine background and material peaks entirely.
    """
    otsu_thr, _ = cv2.threshold(
        image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    values = image[valid] if valid is not None else image.ravel()
    hist = np.bincount(values.ravel(), minlength=256).astype(np.float64)
    smoothed = hist.copy()
    for _ in range(max(1, int(smooth_passes))):
        smoothed = gaussian_smooth(smoothed, smooth_sigma)

    derivative = np.gradient(smoothed)

    peaks = find_peaks(smoothed, prominence=float(smoothed.max()) * 0.02)
    peak: int | None = None
    edge: int | None = None

    if len(peaks) >= 2:
        peak = int(peaks[-1])           # the material peak, highest intensity
        edge = _rising_edge(smoothed, peak, int(peaks[-2]))
    elif len(peaks) == 1:
        peak = int(peaks[0])
        edge = _rising_edge(smoothed, peak, None)

    if edge is None:
        threshold = float(otsu_thr)
        meta = {"source": "otsu_fallback", "otsu": float(otsu_thr),
                "reason": "fewer than two resolvable histogram peaks"}
    else:
        threshold = float(edge) * float(margin)
        meta = {
            "source": "second_peak_rising_edge",
            "otsu": float(otsu_thr),
            "peak": peak,
            "edge": edge,
            "margin": float(margin),
        }

    if trace is not None:
        trace.add_histogram(hist, smoothed, derivative, peak, edge, threshold)

    meta["threshold"] = threshold
    meta["n_pixels"] = int(values.size)
    return threshold, meta


def _rising_edge(smoothed: np.ndarray, peak: int, prev_peak: int | None,
                 fraction: float = 0.10) -> int | None:
    """Intensity at the foot of the material peak.

    Anchored to the valley between the two peaks: walk up from the valley to
    where the count first reaches a small fraction of the way to the crest. That
    is the transition from blurry background into solid material.

    Locating the foot from the derivative instead (first point below some
    fraction of the maximum slope) lands partway up the flank, not at the base.
    For a tightly distributed material peak the flank is only a few grey levels
    wide, so even after the 0.9 safety factor the threshold can sit inside the
    material's own lower tail and start eating into solid regions.
    """
    lo = 0 if prev_peak is None else int(prev_peak)
    if peak <= lo:
        return None

    segment = smoothed[lo:peak + 1]
    if segment.size < 2:
        return None

    valley_i = lo + int(np.argmin(segment))
    valley = float(smoothed[valley_i])
    crest = float(smoothed[peak])
    if crest <= valley:
        return None

    level = valley + (crest - valley) * fraction
    for i in range(valley_i, peak + 1):
        if smoothed[i] >= level:
            return int(i)
    return None


# ---------------------------------------------------------------- step 7


def clean_mask(mask: np.ndarray, kernel: int = config.FINAL_MORPH_KERNEL,
               trace: DebugTrace | None = None, name: str = "7_clean") -> np.ndarray:
    """One open -> close pass to drop specks and fill pinholes."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
    out = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k)
    if trace is not None:
        trace.add_mask(name, out,
                       f"Open then close at kernel {kernel}: removes leftover "
                       "specks and fills pinholes.")
    return out
