/*
 * Interactive overlay for the captured still.
 *
 * Reproduces the cv2.selectROI gesture in the browser: press anywhere on the
 * image and drag corner to corner to define a new rectangle, discarding the
 * previous one. Once a rectangle exists, eight handles resize it and dragging
 * the interior moves it whole.
 *
 * All coordinates are kept in *image* space, never canvas space, so the overlay
 * stays correct when the browser scales the image and the stored ROI means the
 * same thing at any zoom.
 */

const HANDLE = 7;          // half-size of a grab handle, in screen px
const MIN_SIZE = 20;       // smallest permitted ROI edge, in image px

/** Ray casting: is the point inside the closed polygon? */
function pointInPolygon(p, points) {
  if (!points || points.length < 3) return false;
  let inside = false;
  for (let i = 0, j = points.length - 1; i < points.length; j = i++) {
    const [xi, yi] = points[i];
    const [xj, yj] = points[j];
    if ((yi > p.y) !== (yj > p.y) &&
        p.x < ((xj - xi) * (p.y - yi)) / (yj - yi) + xi) {
      inside = !inside;
    }
  }
  return inside;
}

export class RoiCanvas {
  constructor(canvas, opts = {}) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.image = null;
    this.roi = null;
    this.defaultRoi = null;
    this.mode = opts.mode || "rect";      // "rect" | "quad"
    this.showDiagonal = !!opts.diagonal;
    // Fraction of the frame left outside the default ROI. Zero means the whole
    // frame, which is what the collapse test needs: its platform width is the
    // measurement's own length reference, so cropping into it rescales
    // everything.
    this.defaultInset = opts.defaultInset ?? 0.15;
    this.quad = [];
    this.marks = [];                      // clickable measurement overlays
    this.onChange = opts.onChange || (() => {});
    this.onMarkClick = opts.onMarkClick || (() => {});
    this.onPick = opts.onPick || (() => {});
    this.onAssign = opts.onAssign || (() => {});
    this.picking = null;          // key of the colour being sampled, or null
    this.assigning = false;       // waiting for a click that picks a pore
    this.scaleBar = null;         // {mmPerPx, approximate} or null
    this.sampler = null;          // offscreen copy, for reading true pixel values
    this.scale = 1;
    // {cropX, cropY, cropW, cropH, factor, offsetX, offsetY} while "Zoom to
    // ROI" is active, else null. A pure display transform -- the working image
    // and the ROI's own coordinates are never touched by it.
    this.zoom = null;

    this.drag = null;
    this._bind();
  }

  /* ------------------------------------------------------------ image */

  async setImage(url, roi) {
    const img = new Image();
    img.src = url;
    await img.decode();
    this.image = img;
    const inset = this.defaultInset;
    this.defaultRoi = {
      x: Math.round(img.width * inset),
      y: Math.round(img.height * inset),
      w: Math.round(img.width * (1 - 2 * inset)),
      h: Math.round(img.height * (1 - 2 * inset)),
    };
    this.roi = roi ? { ...roi } : { ...this.defaultRoi };
    this.quad = [];
    this.marks = [];
    this.zoom = null;             // a new image always starts at full view

    // An unscaled offscreen copy, so a colour sample reads the image's real
    // pixel rather than whatever the on-screen canvas resampled it to.
    this.sampler = document.createElement("canvas");
    this.sampler.width = img.width;
    this.sampler.height = img.height;
    this.sampler.getContext("2d", { willReadFrequently: true })
      .drawImage(img, 0, 0);

    this._resize();
    this.draw();
    this.onChange(this.roi);
  }

  /* ------------------------------------------------------- colour pick */

  armPicker(key) {
    this.picking = key;
    this.canvas.classList.toggle("picking", !!key);
  }

  /**
   * Wait for a click that chooses a pore.
   *
   * Stays armed until the caller disarms it, because assigning a whole sequence
   * of size classes is several clicks in a row.
   */
  armAssign(on) {
    this.assigning = !!on;
    this.canvas.classList.toggle("picking", !!on);
  }

  sampleAt(x, y) {
    if (!this.sampler) return null;
    const px = Math.max(0, Math.min(this.image.width - 1, Math.round(x)));
    const py = Math.max(0, Math.min(this.image.height - 1, Math.round(y)));
    const d = this.sampler.getContext("2d").getImageData(px, py, 1, 1).data;
    return { r: d[0], g: d[1], b: d[2], x: px, y: py };
  }

  clear() {
    this.image = null;
    this.roi = null;
    this.marks = [];
    this.quad = [];
    this.zoom = null;
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
  }

  setMarks(marks) {
    this.marks = marks || [];
    this.draw();
  }

  /**
   * Show a scale bar from the selected calibration, or nothing without one.
   *
   * `approximate` is set for a homography profile: there the scale varies across
   * the frame, so a single bar is only representative and says so.
   */
  setScaleBar(spec) {
    this.scaleBar = (spec && spec.mmPerPx > 0) ? spec : null;
    this.draw();
  }

  resetRoi() {
    if (!this.defaultRoi) return;
    this.roi = { ...this.defaultRoi };
    this.quad = [];
    this.draw();
    this.onChange(this.roi);
  }

  setMode(mode) {
    this.mode = mode;
    this.quad = [];
    this.draw();
  }

  /**
   * Toggle "Zoom to ROI": crop to the current ROI and scale it to fill this
   * same canvas, aspect ratio preserved (contain, not cover -- letterboxed on
   * whichever axis does not match, never stretched).
   *
   * A pure display transform. The ROI stays pinned to whatever it was when the
   * button was pressed, so refining it afterwards works the same way it does
   * at full view -- drag a handle, draw a new rectangle -- all while looking at
   * the magnified image; toggling again snaps back to the full frame. Calling
   * this a second time exits zoom rather than re-snapping, so a second click
   * always means "done looking closely," not "zoom in again on whatever the
   * ROI has become."
   */
  toggleZoom() {
    if (this.zoom) {
      this.zoom = null;
      this.draw();
      return;
    }
    if (!this.roi || this.roi.w < 1 || this.roi.h < 1) return;
    this.zoom = { cropX: this.roi.x, cropY: this.roi.y,
                 cropW: this.roi.w, cropH: this.roi.h };
    this._recomputeZoom();
    this.draw();
  }

  /** Refresh the zoom's fit-to-canvas factor and letterbox offsets against the
   * current canvas size, without moving the pinned crop window itself. Needed
   * after a resize, since the canvas the zoom fits into just changed size. */
  _recomputeZoom() {
    if (!this.zoom) return;
    const { cropX, cropY, cropW, cropH } = this.zoom;
    const factor = Math.min(this.canvas.width / cropW, this.canvas.height / cropH);
    this.zoom = {
      cropX, cropY, cropW, cropH, factor,
      offsetX: (this.canvas.width - cropW * factor) / 2,
      offsetY: (this.canvas.height - cropH * factor) / 2,
    };
  }

  _resize() {
    if (!this.image) return;
    // Canvas dimensions always track the FULL image's own aspect ratio -- this
    // is "the original capture frame" that a zoomed view fits into, and it must
    // stay fixed regardless of what is currently drawn inside it.
    const width = this.canvas.parentElement.clientWidth;
    this.scale = width / this.image.width;
    this.canvas.width = width;
    this.canvas.height = Math.round(this.image.height * this.scale);
    this._recomputeZoom();
  }

  /* ------------------------------------------------------ coordinates */

  /** Canvas px per image px at the current view -- the zoom factor while
   * zoomed, the plain fit-to-container scale otherwise. Everything sized in
   * canvas pixels from image-space content (hit tolerances, the scale bar)
   * goes through this rather than `this.scale` directly, so it stays correct
   * whichever view is active. */
  _viewScale() {
    return this.zoom ? this.zoom.factor : this.scale;
  }

  toImage(ev) {
    const rect = this.canvas.getBoundingClientRect();
    // CSS-pixel offset within the element, converted to canvas backing-store
    // pixels (the two differ when the element is styled to a different size
    // than its width/height attributes, e.g. on a high-DPI display).
    const cx = (ev.clientX - rect.left) * (this.canvas.width / rect.width);
    const cy = (ev.clientY - rect.top) * (this.canvas.height / rect.height);

    if (this.zoom) {
      const z = this.zoom;
      return { x: z.cropX + (cx - z.offsetX) / z.factor,
               y: z.cropY + (cy - z.offsetY) / z.factor };
    }
    return { x: cx / this.scale, y: cy / this.scale };
  }

  toCanvas(p) {
    if (this.zoom) {
      const z = this.zoom;
      return { x: z.offsetX + (p.x - z.cropX) * z.factor,
               y: z.offsetY + (p.y - z.cropY) * z.factor };
    }
    return { x: p.x * this.scale, y: p.y * this.scale };
  }

  /* ------------------------------------------------------------ input */

  _bind() {
    this.canvas.addEventListener("mousedown", (e) => this._down(e));
    window.addEventListener("mousemove", (e) => this._move(e));
    window.addEventListener("mouseup", () => this._up());
    window.addEventListener("keydown", (e) => {
      if (e.key !== "Escape") return;
      if (this.drag) {
        this.roi = this.drag.original ? { ...this.drag.original } : this.roi;
        this.drag = null;
        this.draw();
      } else if (this.assigning) {
        this.armAssign(false);
        this.onAssign({ cancelled: true });
      } else if (this.picking) {
        this.armPicker(null);
      }
    });
    window.addEventListener("resize", () => { this._resize(); this.draw(); });
  }

  _down(ev) {
    if (!this.image) return;
    const p = this.toImage(ev);

    // Sampling a colour takes priority over every other gesture, and is armed
    // for one click only so the ROI stays editable straight afterwards.
    if (this.picking) {
      const sample = this.sampleAt(p.x, p.y);
      const key = this.picking;
      this.armPicker(null);
      if (sample) this.onPick(key, sample);
      return;
    }

    // Checked before the ROI-drag branch below. A click outside the ROI is a
    // meaningful answer here -- it means "this size class is a closed pore" --
    // and that gesture would otherwise start drawing a new ROI rectangle.
    if (this.assigning) {
      this.onAssign({
        markId: this._hitMark(p),
        inside: this._inside(p),
        x: p.x, y: p.y,
      });
      return;
    }

    if (this.mode === "quad") {
      this.quad.push([p.x, p.y]);
      if (this.quad.length > 4) this.quad = [[p.x, p.y]];
      this.draw();
      this.onChange(this.roi, this.quad);
      return;
    }

    // Clicking a measurement tick toggles it rather than starting a drag.
    const hit = this._hitMark(p);
    if (hit !== null) {
      this.onMarkClick(hit);
      return;
    }

    const handle = this._hitHandle(p);
    if (handle) {
      this.drag = { kind: "resize", handle, original: { ...this.roi } };
      return;
    }

    // Shift is the only way to move the box wholesale. Plain dragging always
    // starts a new rectangle, even from inside the current one -- otherwise a
    // large ROI would cover the whole image and leave nowhere to begin a fresh
    // corner-to-corner drag, which is the primary gesture.
    if (ev.shiftKey && this._inside(p)) {
      this.drag = {
        kind: "move", start: p, origin: { ...this.roi }, original: { ...this.roi },
      };
      return;
    }

    this.drag = { kind: "draw", anchor: p, original: { ...this.roi } };
    this.roi = { x: p.x, y: p.y, w: 0, h: 0 };
    this.draw();
  }

  _move(ev) {
    if (!this.drag || !this.image) return;
    const p = this._clampToImage(this.toImage(ev));

    if (this.drag.kind === "draw") {
      const a = this.drag.anchor;
      this.roi = {
        x: Math.min(a.x, p.x), y: Math.min(a.y, p.y),
        w: Math.abs(p.x - a.x), h: Math.abs(p.y - a.y),
      };
    } else if (this.drag.kind === "move") {
      const dx = p.x - this.drag.start.x;
      const dy = p.y - this.drag.start.y;
      const o = this.drag.origin;
      this.roi = {
        x: Math.max(0, Math.min(this.image.width - o.w, o.x + dx)),
        y: Math.max(0, Math.min(this.image.height - o.h, o.y + dy)),
        w: o.w, h: o.h,
      };
    } else if (this.drag.kind === "resize") {
      this._resizeBy(this.drag.handle, p);
    }
    this.draw();
  }

  _up() {
    if (!this.drag) return;
    const wasDraw = this.drag.kind === "draw";
    const previous = this.drag.original;
    this.drag = null;
    if (!this.roi) return;

    // A click, or a slip of a few pixels, is not an attempt to define a region.
    // Restore what was there rather than the default: the user may have spent
    // time positioning it, and losing that to a stray click is worse than the
    // click doing nothing.
    if (wasDraw && (this.roi.w < MIN_SIZE || this.roi.h < MIN_SIZE)) {
      this.roi = previous ? { ...previous } : { ...this.defaultRoi };
    }
    this.roi = this._round(this.roi);
    this.draw();
    this.onChange(this.roi);
  }

  _round(r) {
    return {
      x: Math.round(r.x), y: Math.round(r.y),
      w: Math.max(MIN_SIZE, Math.round(r.w)),
      h: Math.max(MIN_SIZE, Math.round(r.h)),
    };
  }

  _clampToImage(p) {
    return {
      x: Math.max(0, Math.min(this.image.width, p.x)),
      y: Math.max(0, Math.min(this.image.height, p.y)),
    };
  }

  _resizeBy(handle, p) {
    const r = { ...this.roi };
    let x0 = r.x, y0 = r.y, x1 = r.x + r.w, y1 = r.y + r.h;
    if (handle.includes("w")) x0 = p.x;
    if (handle.includes("e")) x1 = p.x;
    if (handle.includes("n")) y0 = p.y;
    if (handle.includes("s")) y1 = p.y;
    this.roi = {
      x: Math.min(x0, x1), y: Math.min(y0, y1),
      w: Math.abs(x1 - x0), h: Math.abs(y1 - y0),
    };
  }

  /* ------------------------------------------------------------ hits */

  _handlePoints() {
    const r = this.roi;
    if (!r) return {};
    const mx = r.x + r.w / 2, my = r.y + r.h / 2;
    return {
      nw: [r.x, r.y], n: [mx, r.y], ne: [r.x + r.w, r.y],
      w: [r.x, my], e: [r.x + r.w, my],
      sw: [r.x, r.y + r.h], s: [mx, r.y + r.h], se: [r.x + r.w, r.y + r.h],
    };
  }

  _hitHandle(p) {
    const tol = HANDLE / this._viewScale();
    for (const [name, [hx, hy]] of Object.entries(this._handlePoints())) {
      if (Math.abs(p.x - hx) <= tol && Math.abs(p.y - hy) <= tol) return name;
    }
    return null;
  }

  _hitMark(p) {
    const tol = 8 / this._viewScale();
    // Smallest first, so a pore nested inside a larger region still wins.
    const ordered = [...this.marks].sort((a, b) => (a.area || 0) - (b.area || 0));
    for (const m of ordered) {
      if (m.kind === "tick") {
        if (Math.abs(p.x - m.x) <= tol &&
            p.y >= Math.min(m.y0, m.y1) - tol && p.y <= Math.max(m.y0, m.y1) + tol) {
          return m.id;
        }
      } else if (m.kind === "point") {
        if (Math.hypot(p.x - m.x, p.y - m.y) <= Math.max(tol, m.r || 0)) return m.id;
      } else if (m.kind === "pore") {
        // Anywhere on the pore counts, not just the centroid dot.
        if (Math.hypot(p.x - m.x, p.y - m.y) <= Math.max(tol, 7 / this._viewScale())) {
          return m.id;
        }
        if (pointInPolygon(p, m.points)) return m.id;
      }
    }
    return null;
  }

  _inside(p) {
    const r = this.roi;
    return r && p.x > r.x && p.x < r.x + r.w && p.y > r.y && p.y < r.y + r.h;
  }

  /* ------------------------------------------------------------ paint */

  /**
   * The same annotated view rendered at the image's native resolution.
   *
   * The on-screen canvas is scaled to fit its column, so exporting it directly
   * would archive a downscaled record of a measurement -- readable, but no
   * longer pixel-aligned with the capture it annotates.
   */
  exportFullSize() {
    if (!this.image) return null;
    const target = document.createElement("canvas");
    target.width = this.image.width;
    target.height = this.image.height;

    const savedCanvas = this.canvas;
    const savedCtx = this.ctx;
    const savedScale = this.scale;
    const savedZoom = this.zoom;
    try {
      this.canvas = target;
      this.ctx = target.getContext("2d");
      this.scale = 1;
      // The saved overlay is an archival record and must always show the whole
      // frame -- an ephemeral on-screen "Zoom to ROI" view has no bearing on it.
      this.zoom = null;
      this.draw();
    } finally {
      this.canvas = savedCanvas;
      this.ctx = savedCtx;
      this.scale = savedScale;
      this.zoom = savedZoom;
    }
    return target;
  }

  draw() {
    const ctx = this.ctx;
    if (!this.image) return;
    ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

    if (this.zoom) {
      // Letterbox bars (wherever the crop's aspect ratio does not match the
      // canvas's) get the viewer's own background rather than staying
      // transparent, so they read as intentional, not broken.
      ctx.fillStyle = "#0d0f12";
      ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
      const z = this.zoom;
      ctx.drawImage(this.image, z.cropX, z.cropY, z.cropW, z.cropH,
                    z.offsetX, z.offsetY, z.cropW * z.factor, z.cropH * z.factor);
    } else {
      ctx.drawImage(this.image, 0, 0, this.canvas.width, this.canvas.height);
    }

    if (this.mode === "quad") {
      this._drawQuad();
    } else if (this.roi) {
      this._drawRect();
    }
    this._drawMarks();
    this._drawScaleBar();
  }

  _drawRect() {
    const ctx = this.ctx;
    const r = this.roi;
    const a = this.toCanvas({ x: r.x, y: r.y });
    const scale = this._viewScale();
    const w = r.w * scale, h = r.h * scale;

    // Dim everything outside the ROI so the working area reads clearly.
    ctx.save();
    ctx.fillStyle = "rgba(0,0,0,0.35)";
    ctx.beginPath();
    ctx.rect(0, 0, this.canvas.width, this.canvas.height);
    ctx.rect(a.x, a.y, w, h);
    ctx.fill("evenodd");
    ctx.restore();

    ctx.strokeStyle = "#ff2d2d";
    ctx.lineWidth = 2;
    ctx.strokeRect(a.x, a.y, w, h);

    if (this.showDiagonal) {
      ctx.save();
      ctx.strokeStyle = "#ff2d2d";
      ctx.setLineDash([7, 5]);
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(a.x, a.y + h);          // bottom-left: smallest pore
      ctx.lineTo(a.x + w, a.y);          // top-right: largest pore
      ctx.stroke();
      ctx.restore();
    }

    ctx.fillStyle = "#ff2d2d";
    for (const [, [hx, hy]] of Object.entries(this._handlePoints())) {
      const c = this.toCanvas({ x: hx, y: hy });
      ctx.fillRect(c.x - HANDLE / 2, c.y - HANDLE / 2, HANDLE, HANDLE);
    }
  }

  _drawQuad() {
    const ctx = this.ctx;
    ctx.strokeStyle = "#ff2d2d";
    ctx.fillStyle = "#ff2d2d";
    ctx.lineWidth = 2;
    ctx.beginPath();
    this.quad.forEach((p, i) => {
      const c = this.toCanvas({ x: p[0], y: p[1] });
      if (i === 0) ctx.moveTo(c.x, c.y); else ctx.lineTo(c.x, c.y);
    });
    if (this.quad.length === 4) ctx.closePath();
    ctx.stroke();
    this.quad.forEach((p, i) => {
      const c = this.toCanvas({ x: p[0], y: p[1] });
      ctx.beginPath();
      ctx.arc(c.x, c.y, 5, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = "#fff";
      ctx.font = "11px sans-serif";
      ctx.fillText(String(i + 1), c.x + 7, c.y - 7);
      ctx.fillStyle = "#ff2d2d";
    });
  }

  /**
   * A 20 / 10 / 5 mm scale bar in the bottom-right corner.
   *
   * Three nested bars sharing a left edge rather than one subdivided bar, so
   * each length can be read straight off without counting ticks. Drawn last, on
   * top of everything, and sized in canvas pixels from the calibration — which
   * means it comes out correct both on screen and in the full-resolution overlay
   * export, where the canvas is the image's own size.
   *
   * Lengths that would take more than `MAX_FRACTION` of the frame are dropped;
   * a 20 mm bar across a 15 mm field would otherwise run off the image. If none
   * of the three fit, smaller decades are tried so there is always some scale.
   */
  _drawScaleBar() {
    const bar = this.scaleBar;
    if (!bar || !this.image) return;

    const MAX_FRACTION = 0.55;
    const pxPerMm = 1 / bar.mmPerPx;
    const canvasPxPerMm = pxPerMm * this._viewScale();
    const limit = this.canvas.width * MAX_FRACTION;

    let lengths = [20, 10, 5].filter((mm) => mm * canvasPxPerMm <= limit);
    if (!lengths.length) {
      lengths = [2, 1, 0.5, 0.2, 0.1]
        .filter((mm) => mm * canvasPxPerMm <= limit).slice(0, 3);
    }
    if (!lengths.length) return;

    const ctx = this.ctx;
    // Scale the furniture with the canvas so the exported overlay is not a
    // hairline drawing on a 1500 px image.
    const font = Math.max(10, Math.round(this.canvas.width / 68));
    const barH = Math.max(4, Math.round(font * 0.5));
    const gap = Math.round(font * 0.45);
    const pad = Math.round(font * 0.7);

    const longest = Math.max(...lengths) * canvasPxPerMm;
    ctx.save();
    ctx.font = `600 ${font}px ui-monospace, monospace`;
    ctx.textBaseline = "middle";
    const labelW = Math.max(...lengths.map((mm) => ctx.measureText(`${mm} mm`).width));

    const boxW = longest + labelW + pad * 3;
    const boxH = lengths.length * (barH + gap) + gap + (bar.approximate ? font + 2 : 0);
    const boxX = this.canvas.width - boxW - pad;
    const boxY = this.canvas.height - boxH - pad;

    ctx.fillStyle = "rgba(12,14,18,0.72)";
    ctx.strokeStyle = "rgba(255,255,255,0.28)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.rect(boxX, boxY, boxW, boxH);
    ctx.fill();
    ctx.stroke();

    const left = boxX + pad;
    lengths.forEach((mm, i) => {
      const w = mm * canvasPxPerMm;
      const y = boxY + gap + i * (barH + gap);

      // Alternating 1 mm cells, so the bar is readable as a ruler rather than
      // just a line of a stated length.
      const cells = Math.max(1, Math.round(mm));
      const cellW = w / cells;
      for (let c = 0; c < cells; c++) {
        ctx.fillStyle = c % 2 ? "#0c0e12" : "#ffffff";
        ctx.fillRect(left + c * cellW, y, cellW, barH);
      }
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 1;
      ctx.strokeRect(left + 0.5, y + 0.5, w - 1, barH - 1);

      ctx.fillStyle = "#ffffff";
      ctx.fillText(`${mm} mm`, left + longest + pad, y + barH / 2);
    });

    if (bar.approximate) {
      ctx.fillStyle = "#ffb547";
      ctx.font = `${Math.max(9, font - 2)}px ui-monospace, monospace`;
      ctx.fillText("approx. (homography)", left,
                   boxY + boxH - (font + 2) / 2 - gap / 2);
    }
    ctx.restore();
  }

  _drawMarks() {
    const ctx = this.ctx;
    for (const m of this.marks) {
      const on = m.included !== false;
      ctx.strokeStyle = m.color || (on ? "#25e06a" : "#8a8a8a");
      ctx.fillStyle = ctx.strokeStyle;
      ctx.lineWidth = m.lineWidth || 2;
      if (m.kind === "pore") {
        // Contour in its own colour, centroid dot in another: the two are
        // separate visual jobs, since the contour says where the region is and
        // the dot is the thing small pores are actually clicked by.
        ctx.beginPath();
        m.points.forEach((p, i) => {
          const c = this.toCanvas({ x: p[0], y: p[1] });
          if (i === 0) ctx.moveTo(c.x, c.y); else ctx.lineTo(c.x, c.y);
        });
        ctx.closePath();
        ctx.stroke();

        const c = this.toCanvas({ x: m.x, y: m.y });
        ctx.fillStyle = m.centroidColor || "#25e06a";
        ctx.beginPath();
        ctx.arc(c.x, c.y, 4, 0, Math.PI * 2);
        ctx.fill();

        if (m.label) {
          ctx.fillStyle = m.labelColor || m.color || "#25e06a";
          ctx.font = "bold 13px ui-monospace, monospace";
          ctx.strokeStyle = "rgba(0,0,0,0.75)";
          ctx.lineWidth = 3;
          ctx.strokeText(m.label, c.x + 8, c.y - 6);
          ctx.fillText(m.label, c.x + 8, c.y - 6);
        }
        continue;
      }
      if (m.kind === "tick") {
        const a = this.toCanvas({ x: m.x, y: m.y0 });
        const b = this.toCanvas({ x: m.x, y: m.y1 });
        ctx.beginPath();
        ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y);
        ctx.moveTo(a.x - 5, a.y); ctx.lineTo(a.x + 5, a.y);
        ctx.moveTo(b.x - 5, b.y); ctx.lineTo(b.x + 5, b.y);
        ctx.stroke();
      } else if (m.kind === "poly") {
        ctx.beginPath();
        m.points.forEach((p, i) => {
          const c = this.toCanvas({ x: p[0], y: p[1] });
          if (i === 0) ctx.moveTo(c.x, c.y); else ctx.lineTo(c.x, c.y);
        });
        ctx.closePath();
        ctx.stroke();
      } else if (m.kind === "point") {
        const c = this.toCanvas({ x: m.x, y: m.y });
        ctx.beginPath();
        ctx.arc(c.x, c.y, 5, 0, Math.PI * 2);
        ctx.fill();
      }
      if (m.label) {
        const c = this.toCanvas({
          x: m.x ?? m.points?.[0]?.[0] ?? 0,
          y: m.y ?? m.y0 ?? m.points?.[0]?.[1] ?? 0,
        });
        ctx.fillStyle = m.color || (on ? "#25e06a" : "#8a8a8a");
        ctx.font = "12px ui-monospace, monospace";
        ctx.fillText(m.label, c.x + 7, c.y - 5);
      }
    }
  }
}
