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
    this.picking = null;          // key of the colour being sampled, or null
    this.sampler = null;          // offscreen copy, for reading true pixel values
    this.scale = 1;

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
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
  }

  setMarks(marks) {
    this.marks = marks || [];
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

  _resize() {
    if (!this.image) return;
    const width = this.canvas.parentElement.clientWidth;
    this.scale = width / this.image.width;
    this.canvas.width = width;
    this.canvas.height = Math.round(this.image.height * this.scale);
  }

  /* ------------------------------------------------------ coordinates */

  toImage(ev) {
    const rect = this.canvas.getBoundingClientRect();
    return {
      x: (ev.clientX - rect.left) / this.scale * (this.canvas.width / rect.width),
      y: (ev.clientY - rect.top) / this.scale * (this.canvas.height / rect.height),
    };
  }

  toCanvas(p) {
    return { x: p.x * this.scale, y: p.y * this.scale };
  }

  /* ------------------------------------------------------------ input */

  _bind() {
    this.canvas.addEventListener("mousedown", (e) => this._down(e));
    window.addEventListener("mousemove", (e) => this._move(e));
    window.addEventListener("mouseup", () => this._up());
    window.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && this.drag) {
        this.roi = this.drag.original ? { ...this.drag.original } : this.roi;
        this.drag = null;
        this.draw();
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
    const tol = HANDLE / this.scale;
    for (const [name, [hx, hy]] of Object.entries(this._handlePoints())) {
      if (Math.abs(p.x - hx) <= tol && Math.abs(p.y - hy) <= tol) return name;
    }
    return null;
  }

  _hitMark(p) {
    const tol = 8 / this.scale;
    for (const m of this.marks) {
      if (m.kind === "tick") {
        if (Math.abs(p.x - m.x) <= tol &&
            p.y >= Math.min(m.y0, m.y1) - tol && p.y <= Math.max(m.y0, m.y1) + tol) {
          return m.id;
        }
      } else if (m.kind === "point") {
        if (Math.hypot(p.x - m.x, p.y - m.y) <= Math.max(tol, m.r || 0)) return m.id;
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
    try {
      this.canvas = target;
      this.ctx = target.getContext("2d");
      this.scale = 1;
      this.draw();
    } finally {
      this.canvas = savedCanvas;
      this.ctx = savedCtx;
      this.scale = savedScale;
    }
    return target;
  }

  draw() {
    const ctx = this.ctx;
    if (!this.image) return;
    ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    ctx.drawImage(this.image, 0, 0, this.canvas.width, this.canvas.height);

    if (this.mode === "quad") {
      this._drawQuad();
    } else if (this.roi) {
      this._drawRect();
    }
    this._drawMarks();
  }

  _drawRect() {
    const ctx = this.ctx;
    const r = this.roi;
    const a = this.toCanvas({ x: r.x, y: r.y });
    const w = r.w * this.scale, h = r.h * this.scale;

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

  _drawMarks() {
    const ctx = this.ctx;
    for (const m of this.marks) {
      const on = m.included !== false;
      ctx.strokeStyle = on ? "#25e06a" : "#8a8a8a";
      ctx.fillStyle = ctx.strokeStyle;
      ctx.lineWidth = 2;
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
        ctx.fillStyle = on ? "#25e06a" : "#8a8a8a";
        ctx.font = "12px ui-monospace, monospace";
        ctx.fillText(m.label, c.x + 7, c.y - 5);
      }
    }
  }
}
