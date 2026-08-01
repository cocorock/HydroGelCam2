"""Ordered capture of every preprocessing stage, for the debug panel.

Each tab's debug checkbox renders whatever a DebugTrace collected during the last
detection run. Stages are stored as PNG bytes and handed to the browser as data
URIs, so nothing needs to be written to disk unless the run is saved.
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


@dataclass
class Stage:
    name: str
    caption: str
    png: bytes

    def data_uri(self) -> str:
        return "data:image/png;base64," + base64.b64encode(self.png).decode("ascii")


@dataclass
class DebugTrace:
    enabled: bool = False
    stages: list[Stage] = field(default_factory=list)
    max_width: int = 900

    def add(self, name: str, image: np.ndarray, caption: str = "") -> None:
        if not self.enabled or image is None:
            return
        self.stages.append(Stage(name, caption, _encode(image, self.max_width)))

    def add_mask(self, name: str, mask: np.ndarray, caption: str = "") -> None:
        """Add a binary mask, scaled to full contrast so it is actually visible."""
        if not self.enabled or mask is None:
            return
        vis = mask.astype(np.uint8)
        if vis.max() <= 1:
            vis = vis * 255
        self.add(name, vis, caption)

    def add_overlay(
        self,
        name: str,
        base: np.ndarray,
        mask: np.ndarray,
        caption: str = "",
        color: tuple[int, int, int] = (0, 0, 255),
    ) -> None:
        """Tint `base` where `mask` is set, to show what the mask actually caught."""
        if not self.enabled:
            return
        bgr = base if base.ndim == 3 else cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
        tint = np.zeros_like(bgr)
        tint[mask.astype(bool)] = color
        self.add(name, cv2.addWeighted(bgr, 0.65, tint, 0.35, 0), caption)

    def add_plot(self, name: str, figure, caption: str = "") -> None:
        if not self.enabled:
            plt.close(figure)
            return
        buf = io.BytesIO()
        figure.savefig(buf, format="png", dpi=110, bbox_inches="tight")
        plt.close(figure)
        self.stages.append(Stage(name, caption, buf.getvalue()))

    def add_histogram(
        self,
        hist: np.ndarray,
        smoothed: np.ndarray,
        derivative: np.ndarray,
        peak: int | None,
        edge: int | None,
        threshold: float,
    ) -> None:
        """The step-4 diagnostic: where the threshold came from and why."""
        if not self.enabled:
            return
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 5.0), sharex=True)
        x = np.arange(len(hist))
        ax1.plot(x, hist, lw=0.8, color="#999", label="raw histogram")
        ax1.plot(x, smoothed, lw=1.6, color="#1f77b4", label="smoothed")
        if peak is not None:
            ax1.axvline(peak, color="#2ca02c", ls="--", lw=1.2,
                        label=f"2nd peak @ {peak}")
        if edge is not None:
            ax1.axvline(edge, color="#ff7f0e", ls="--", lw=1.2,
                        label=f"rising edge @ {edge}")
        ax1.axvline(threshold, color="#d62728", lw=1.8,
                    label=f"threshold @ {threshold:.1f} (edge x 0.9)")
        ax1.set_ylabel("count")
        ax1.legend(fontsize=7, loc="upper right")
        ax1.set_title("Step 4 - adaptive threshold from the histogram's second peak",
                      fontsize=9)

        ax2.plot(x, derivative, lw=1.2, color="#9467bd")
        ax2.axhline(0, color="#ccc", lw=0.8)
        if edge is not None:
            ax2.axvline(edge, color="#ff7f0e", ls="--", lw=1.2)
        ax2.set_xlabel("intensity")
        ax2.set_ylabel("d(count)/d(intensity)")
        fig.tight_layout()
        self.add_plot("histogram", fig,
                      "Smoothed histogram and its derivative inside the "
                      "Otsu-isolated material region.")

    # ------------------------------------------------------------ output

    def to_json(self) -> list[dict[str, str]]:
        return [
            {"name": s.name, "caption": s.caption, "image": s.data_uri()}
            for s in self.stages
        ]

    def save(self, directory: Path) -> list[str]:
        directory.mkdir(parents=True, exist_ok=True)
        paths = []
        for i, stage in enumerate(self.stages):
            path = directory / f"{i:02d}_{_slug(stage.name)}.png"
            path.write_bytes(stage.png)
            paths.append(str(path))
        return paths


def _encode(image: np.ndarray, max_width: int) -> bytes:
    img = image
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    h, w = img.shape[:2]
    if w > max_width:
        scale = max_width / float(w)
        img = cv2.resize(img, (max_width, max(1, int(round(h * scale)))),
                         interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes() if ok else b""


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in text.lower()).strip("_")
