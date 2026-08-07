import { RoiCanvas } from "./roi-canvas.js";

const D = window.APP_DEFAULTS || {};

/* ------------------------------------------------------------ utilities */

async function api(path, opts = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...opts,
  });
  const text = await res.text();
  let body;
  try { body = text ? JSON.parse(text) : {}; } catch { body = { error: text }; }
  if (!res.ok) throw new Error(body.error || `${res.status} ${res.statusText}`);
  return body;
}

const post = (p, b) => api(p, { method: "POST", body: JSON.stringify(b ?? {}) });
const put = (p, b) => api(p, { method: "PUT", body: JSON.stringify(b ?? {}) });
const del = (p) => api(p, { method: "DELETE" });

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k === "html") node.innerHTML = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else if (v !== null && v !== undefined) node.setAttribute(k, v);
  }
  for (const c of children.flat()) {
    if (c === null || c === undefined || c === false) continue;
    node.append(c.nodeType ? c : document.createTextNode(String(c)));
  }
  return node;
}

const num = (v, digits = 3) =>
  v === null || v === undefined || Number.isNaN(v) ? "N/A" : Number(v).toFixed(digits);

function status(message, kind = "") {
  const node = $("#global-status");
  node.textContent = message || "";
  node.style.color = kind === "bad" ? "var(--bad)"
    : kind === "good" ? "var(--good)" : "var(--muted)";
  if (message) clearTimeout(status._t), status._t = setTimeout(() => {
    if (node.textContent === message) node.textContent = "";
  }, 6000);
}

function parseList(text) {
  return String(text || "").split(",").map((s) => parseFloat(s.trim()))
    .filter((v) => Number.isFinite(v));
}

function alertBox(kind, message) {
  return el("div", { class: `alert ${kind}` }, message);
}

/* ------------------------------------------------------------ colour */

/** RGB (0-255) to OpenCV HSV: H 0-179, S 0-255, V 0-255. */
function rgbToOpenCvHsv({ r, g, b }) {
  const rn = r / 255, gn = g / 255, bn = b / 255;
  const max = Math.max(rn, gn, bn), min = Math.min(rn, gn, bn);
  const d = max - min;
  let h = 0;
  if (d !== 0) {
    if (max === rn) h = ((gn - bn) / d) % 6;
    else if (max === gn) h = (bn - rn) / d + 2;
    else h = (rn - gn) / d + 4;
  }
  h = Math.round(h * 30);                       // 60°/2, OpenCV's half-degree hue
  if (h < 0) h += 180;
  return { h, s: Math.round(max === 0 ? 0 : (d / max) * 255), v: Math.round(max * 255) };
}

/**
 * An HSV window around a sampled colour.
 *
 * Saturation and value get a wide band because shading across a curved filament
 * or a matte ABS face moves both a long way while the hue barely shifts; hue is
 * what actually identifies the material, so it carries the tolerance the user
 * controls. Red wraps at H = 0, so a window spanning the wrap is returned as two
 * ranges rather than one impossible one.
 */
function hsvWindow(sample, hueTol = 12) {
  const { h, s, v } = rgbToOpenCvHsv(sample);
  const sLo = Math.max(0, s - 90), sHi = 255;
  const vLo = Math.max(0, v - 110), vHi = 255;
  const lo = h - hueTol, hi = h + hueTol;

  if (lo < 0) {
    return [{ lo: [0, sLo, vLo], hi: [hi, sHi, vHi] },
            { lo: [180 + lo, sLo, vLo], hi: [179, sHi, vHi] }];
  }
  if (hi > 179) {
    return [{ lo: [lo, sLo, vLo], hi: [179, sHi, vHi] },
            { lo: [0, sLo, vLo], hi: [hi - 180, sHi, vHi] }];
  }
  return [{ lo: [lo, sLo, vLo], hi: [hi, sHi, vHi] }];
}

const rgbCss = ({ r, g, b }) => `rgb(${r},${g},${b})`;

/* ------------------------------------------------------------ tabs */

$$("nav.tabs button").forEach((btn) => {
  btn.addEventListener("click", () => {
    $$("nav.tabs button").forEach((b) => b.classList.toggle("active", b === btn));
    $$(".panel").forEach((p) =>
      p.classList.toggle("active", p.id === `panel-${btn.dataset.tab}`));
    if (btn.dataset.tab === "database") Database.refresh();
    else if (btn.dataset.tab !== "camera") Tests[btn.dataset.tab]?.onShow();
  });
});

/* ============================================================ CAMERA TAB */

const Camera = {
  capabilities: {},

  async init() {
    $("#cam-refresh").onclick = () => this.scan();
    $("#cam-open").onclick = () => this.open();
    $("#cam-close").onclick = () => this.close();
    $("#cam-profile-save").onclick = () => this.saveProfile();
    await this.scan();
    await this.loadProfiles();
    await this.refreshStatus();
  },

  async scan() {
    const info = await api("/api/camera/devices");
    $("#cam-device").replaceChildren(
      ...info.devices.map((d) => el("option", { value: d.index }, `${d.index}: ${d.name}`)));
    $("#cam-backend").replaceChildren(
      ...info.backends.map((b) =>
        el("option", { value: b, ...(b === info.default_backend ? { selected: "" } : {}) }, b)));
    if (!info.devices.length) status("No cameras found.", "bad");
  },

  async open() {
    const mode = $("#cam-mode").value ? JSON.parse($("#cam-mode").value) : {};
    const sel = $("#cam-device");
    try {
      status("Connecting…");
      const st = await post("/api/camera/open", {
        index: parseInt(sel.value, 10),
        backend: $("#cam-backend").value,
        name: sel.selectedOptions[0]?.textContent,
        ...mode,
      });
      this.applyStatus(st);
      status("Camera connected.", "good");
    } catch (e) { status(e.message, "bad"); }
  },

  async close() {
    await post("/api/camera/close");
    await this.refreshStatus();
  },

  async refreshStatus() {
    this.applyStatus(await api("/api/camera/status"));
  },

  applyStatus(st) {
    const viewer = $("#cam-viewer");
    if (st.open) {
      viewer.replaceChildren(
        el("img", { src: `/api/camera/stream?t=${Date.now()}`, alt: "Live camera" }));
      $("#cam-status").textContent =
        `Connected: ${st.device_name || st.device_index} via ${st.backend} — ${st.width}×${st.height}`;
      if (st.modes?.length && !$("#cam-mode").options.length) {
        $("#cam-mode").replaceChildren(...st.modes.map((m) =>
          el("option", { value: JSON.stringify(m) }, `${m.width} × ${m.height}`)));
      }
    } else {
      viewer.replaceChildren(el("div", { class: "placeholder" },
        st.error || "No camera connected."));
      $("#cam-status").textContent = st.error || "Not connected.";
    }
    this.capabilities = st.capabilities || {};
    this.renderControls();
  },

  renderControls() {
    const host = $("#cam-controls");
    const props = Object.values(this.capabilities);
    if (!props.length) {
      host.replaceChildren(el("div", { class: "hint" },
        "Connect a camera to probe its controls."));
      return;
    }
    host.replaceChildren(...props.map((p) => this.controlRow(p)));
  },

  controlRow(p) {
    const supported = p.supported;
    if (p.kind === "toggle") {
      const box = el("input", {
        type: "checkbox", ...(p.value > (p.lo + p.hi) / 2 ? { checked: "" } : {}),
        ...(supported ? {} : { disabled: "" }),
        onchange: (e) => this.set(p.key, e.target.checked ? p.hi : p.lo),
      });
      return el("div", { class: `slider-row ${supported ? "" : "unsupported"}` },
        el("label", {}, p.label), box,
        el("span", { class: "val" }, supported ? "" : "n/a"));
    }
    const out = el("span", { class: "val" }, supported ? num(p.value, 0) : "n/a");
    const range = el("input", {
      type: "range", min: p.lo, max: p.hi, step: p.step || 1, value: p.value,
      ...(supported ? {} : { disabled: "" }),
      oninput: (e) => { out.textContent = e.target.value; },
      onchange: (e) => this.set(p.key, parseFloat(e.target.value)),
    });
    return el("div", { class: `slider-row ${supported ? "" : "unsupported"}` },
      el("label", {}, p.label), range, out);
  },

  async set(key, value) {
    try {
      const r = await post("/api/camera/props", { values: { [key]: value } });
      for (const [k, v] of Object.entries(r.values)) {
        if (this.capabilities[k]) this.capabilities[k].value = v;
      }
    } catch (e) { status(e.message, "bad"); }
  },

  async saveProfile() {
    const name = $("#cam-profile-name").value.trim();
    if (!name) return status("Give the profile a name first.", "bad");
    const st = await api("/api/camera/status");
    const props = {};
    for (const [k, c] of Object.entries(this.capabilities)) {
      if (c.supported) props[k] = c.value;
    }
    await post("/api/camera/profiles", {
      name, backend: st.backend, device_index: st.device_index,
      device_name: st.device_name, width: st.width, height: st.height,
      props_json: props,
      hsv_ranges_json: { polarity: $("#cam-polarity").value },
    });
    status("Camera profile saved.", "good");
    await this.loadProfiles();
  },

  async loadProfiles() {
    const { profiles } = await api("/api/camera/profiles");
    $("#cam-profile-list").replaceChildren(...profiles.map((p) =>
      el("div", { class: "row tight" },
        el("span", { class: "status", style: "flex:1" }, p.name),
        el("button", {
          onclick: async () => {
            await post("/api/camera/props", { values: p.props_json || {} });
            if (p.hsv_ranges_json?.polarity) $("#cam-polarity").value = p.hsv_ranges_json.polarity;
            await this.refreshStatus();
            status(`Applied “${p.name}”.`, "good");
          },
        }, "Apply"),
        el("button", {
          class: "danger",
          onclick: async () => {
            await del(`/api/camera/profiles/${p.id}`);
            await this.loadProfiles();
          },
        }, "×"))));
  },
};

/* ======================================================= CALIBRATION */

const Calib = {
  scaleResult: null,

  init() {
    $("#cal-add").onclick = () => this.addView();
    $("#cal-solve").onclick = () => this.solve();
    $("#cal-reset").onclick = () => this.reset();
    $("#scale-measure").onclick = () => this.measureScale();
    $("#calib-save").onclick = () => this.save();
    this.reset();
    this.loadList();
  },

  async reset() {
    const st = await post("/api/calib/intrinsic/reset", {
      cols: +$("#cal-cols").value, rows: +$("#cal-rows").value,
      square_mm: +$("#cal-square").value,
    });
    this.renderState(st);
    $("#cal-result").replaceChildren();
  },

  async addView() {
    try {
      const r = await post("/api/calib/intrinsic/add", {});
      if (!r.accepted) return status(r.message, "bad");
      this.renderState(r.state);
      status(`View ${r.count} accepted.`, "good");
    } catch (e) { status(e.message, "bad"); }
  },

  async solve() {
    try {
      status("Calibrating…");
      const r = await post("/api/calib/intrinsic/solve");
      const kind = r.rms_px < 0.5 ? "good" : r.rms_px < 1.0 ? "warn" : "bad";
      $("#cal-result").replaceChildren(
        alertBox(kind, `Reprojection RMS ${r.rms_px.toFixed(3)} px over ${r.n_frames} views` +
          (r.rms_px < 0.5 ? " — good." : r.rms_px < 1.0
            ? " — usable, but delete the worst views and re-run for a tighter fit."
            : " — too high. Delete outlier views, or re-shoot with a flatter board.")));
      this.renderState(await api("/api/calib/intrinsic/state"));
      status("Calibrated.", "good");
    } catch (e) { status(e.message, "bad"); }
  },

  renderState(st) {
    $("#cal-count").textContent = `${st.frames.length} views`;
    $("#cal-thumbs").replaceChildren(...st.frames.map((f) =>
      el("div", { class: "thumb" },
        el("img", { src: `/api/calib/intrinsic/thumb/${f.index}` }),
        el("div", { class: "err" },
          f.error_px === null || f.error_px === undefined
            ? "not solved" : `${f.error_px.toFixed(3)} px`),
        el("button", {
          class: "danger",
          onclick: async () => this.renderState(
            await del(`/api/calib/intrinsic/frame/${f.index}`)),
        }, "×"))));
  },

  async measureScale() {
    try {
      status("Measuring scale…");
      const r = await post("/api/calib/scale", {
        cols: +$("#scale-cols").value, rows: +$("#scale-rows").value,
        square_mm: +$("#scale-square").value, mode: $("#scale-mode").value,
      });
      this.scaleResult = r;
      const boxes = [];
      if (!r.has_intrinsic) {
        boxes.push(alertBox("warn",
          "No intrinsic solution in this session — the scale was measured on the raw " +
          "image, so lens distortion is not corrected. Run stage A first for best results."));
      }
      if (r.anisotropy_warn) {
        boxes.push(alertBox("warn",
          `Horizontal and vertical scales differ by ${(r.anisotropy * 100).toFixed(1)} %. ` +
          "The board is probably tilted — use homography mode, or re-shoot it flat."));
      }
      boxes.push(el("div", { class: "metrics" },
        metric("mm per px (x)", num(r.mm_per_px_x, 5)),
        metric("mm per px (y)", num(r.mm_per_px_y, 5)),
        metric("board width", num(r.board_width_mm, 2), "mm measured"),
        metric("board height", num(r.board_height_mm, 2), "mm measured")));
      boxes.push(el("div", { class: "hint" },
        `Sanity check: with ${$("#scale-cols").value}×${$("#scale-rows").value} inner corners ` +
        `at ${$("#scale-square").value} mm, the board should span ` +
        `${(($("#scale-cols").value - 1) * $("#scale-square").value).toFixed(2)} × ` +
        `${(($("#scale-rows").value - 1) * $("#scale-square").value).toFixed(2)} mm. ` +
        "If the measured values disagree, the checker size is wrong."));
      if (r.overlay_url) {
        boxes.push(el("img", {
          src: r.overlay_url,
          style: "max-width:100%;margin-top:10px;border:1px solid var(--line);border-radius:5px",
        }));
      }
      $("#scale-result").replaceChildren(...boxes);
      status("Scale measured.", "good");
    } catch (e) {
      $("#scale-result").replaceChildren(alertBox("bad", e.message));
      status(e.message, "bad");
    }
  },

  async save() {
    const name = $("#calib-name").value.trim();
    if (!name) return status("Give the calibration a name first.", "bad");
    const s = this.scaleResult;
    try {
      await post("/api/calib/save", {
        name, intended_use: $("#calib-use").value, notes: $("#calib-notes").value,
        board_cols: +$("#cal-cols").value, board_rows: +$("#cal-rows").value,
        square_mm: +$("#cal-square").value,
        scale_board_cols: +$("#scale-cols").value,
        scale_board_rows: +$("#scale-rows").value,
        scale_square_mm: +$("#scale-square").value,
        mode: s?.mode, mm_per_px_x: s?.mm_per_px_x, mm_per_px_y: s?.mm_per_px_y,
        anisotropy: s?.anisotropy, H: s?.H,
      });
      status("Calibration profile saved.", "good");
      await this.loadList();
      for (const t of Object.values(Tests)) await t.loadCalibrations();
    } catch (e) { status(e.message, "bad"); }
  },

  async loadList() {
    const { calibrations } = await api("/api/calib/list");
    if (!calibrations.length) {
      $("#calib-list").replaceChildren(el("div", { class: "hint" },
        "No calibration profiles saved yet."));
      return;
    }
    $("#calib-list").replaceChildren(el("table", {},
      el("thead", {}, el("tr", {},
        ...["Name", "Use", "Mode", "mm/px", "RMS px", "Checker mm", "Created", ""]
          .map((h) => el("th", {}, h)))),
      el("tbody", {}, ...calibrations.map((c) => el("tr", {},
        el("td", {}, c.name),
        el("td", {}, c.intended_use),
        el("td", {}, c.mode || "—"),
        el("td", { class: "num" }, num(c.mm_per_px_x, 5)),
        el("td", { class: "num" }, c.rms_px === null ? "—" : num(c.rms_px, 3)),
        el("td", { class: "num" }, num(c.scale_square_mm, 2)),
        el("td", {}, (c.created_at || "").slice(0, 16)),
        el("td", {}, el("button", {
          class: "danger",
          onclick: async () => {
            await del(`/api/calib/${c.id}`);
            await this.loadList();
            for (const t of Object.values(Tests)) await t.loadCalibrations();
          },
        }, "×")))))));
  },
};

function metric(key, value, unit, kind = "") {
  return el("div", { class: `metric ${kind}` },
    el("div", { class: "k" }, key),
    el("div", { class: "v" }, value),
    unit ? el("div", { class: "u" }, unit) : null);
}

/* ========================================================== TEST TABS */

class TestTab {
  constructor(type) {
    this.type = type;
    this.panel = $(`#panel-${type}`);
    this.imageId = null;
    this.roi = null;
    this.quad = null;
    this.analysis = null;
    this.measurements = [];

    this.canvas = null;
    this.calibrations = [];
    this._wire();
  }

  q(sel) { return $(sel, this.panel); }
  qa(sel) { return $$(sel, this.panel); }

  _wire() {
    const act = (name) => this.q(`[data-act="${name}"]`);
    act("capture").onclick = () => this.capture();
    act("retake").onclick = () => this.retake();
    act("reset-roi").onclick = () => this.canvas?.resetRoi();
    act("analyze").onclick = () => this.analyze();
    act("redetect").onclick = () => this.analyze();
    act("save").onclick = () => this.save();
    act("upload").onchange = (e) => this.upload(e.target.files[0]);

    if (this.type === "collapse") {
      act("quad").onclick = () => this.canvas?.setMode("quad");
      act("quad-clear").onclick = () => { this.quad = null; this.canvas?.setMode("rect"); };
      act("colours-reset").onclick = () => this.resetColours();
      this.qa("[data-pick]").forEach((btn) => {
        btn.onclick = () => this.armPicker(btn.dataset.pick, btn);
      });
      this.resetColours();
    }

    if (this.type === "fusion") {
      act("assign-all").onclick = () => this.armAssign({ sequential: true, next: 0 });
      act("assign-revert").onclick = () => this.revertToAutomatic();
    }

    act("defaults-save").onclick = () => this.saveDefaults();
    act("defaults-restore").onclick = () => this.loadDefaults();
    act("defaults-reset").onclick = () => this.resetDefaults();

    this.q('[data-role="calibration"]').onchange = () => this.showCalibHint();
    this.buildParamRows();
  }

  /* ------------------------------------------------- pore assignment */

  armAssign(mode) {
    if (!this.analysis?.candidates?.length) {
      return status("Run Calculate first — there are no detected regions yet.", "bad");
    }
    this.assignMode = mode;
    this.canvas?.armAssign(true);
    this.showAssignStatus();
    this.updateMarks();
  }

  disarmAssign() {
    this.assignMode = null;
    this.canvas?.armAssign(false);
    this.showAssignStatus();
    this.updateMarks();
  }

  showAssignStatus() {
    const node = this.q('[data-role="assign-status"]');
    if (!node) return;
    if (!this.assignMode) { node.textContent = ""; return; }
    const k = this.assignMode.sequential ? this.assignMode.next : this.assignMode.classIndex;
    const label = this.analysis?.results?.rows?.[k]?.label ?? `class ${k + 1}`;
    node.textContent = this.assignMode.sequential
      ? `Click the pore for ${label} (${k + 1} of ${this.analysis.results.rows.length}), ` +
        "or outside the ROI if it is closed. Esc to stop."
      : `Click the pore for ${label}, or outside the ROI if it is closed. Esc to cancel.`;
    node.style.color = "var(--accent)";
  }

  /**
   * The assignment the table currently reflects, as class index -> choice.
   *
   * Rows produced by the automatic pass do not carry a candidate index, so they
   * are matched back to a candidate by centroid; both are stored in full-frame
   * coordinates, which makes that exact rather than approximate.
   */
  currentAssignment() {
    if (this.assignment) return this.assignment;
    const byCentroid = new Map(
      (this.analysis?.candidates || []).map((c) =>
        [`${Math.round(c.centroid[0])},${Math.round(c.centroid[1])}`, c.index]));

    const out = {};
    (this.analysis?.results?.rows || []).forEach((row, k) => {
      const w = row.raw;
      if (w.status === "closed") { out[k] = "closed"; return; }
      if (w.candidate_index !== null && w.candidate_index !== undefined) {
        out[k] = w.candidate_index; return;
      }
      const c = w.centroid;
      out[k] = c
        ? byCentroid.get(`${Math.round(c[0])},${Math.round(c[1])}`) ?? null
        : null;
    });
    return out;
  }

  /** A click arrived while an assignment was armed. */
  async onAssignClick(ev) {
    if (ev.cancelled) return this.disarmAssign();
    if (!this.assignMode) return;

    const k = this.assignMode.sequential
      ? this.assignMode.next : this.assignMode.classIndex;

    let choice;
    if (ev.markId !== null && ev.markId !== undefined) {
      choice = ev.markId;                       // a detected region
    } else if (!ev.inside) {
      choice = "closed";                        // outside the ROI = fused shut
    } else {
      // Inside the ROI but not on a region: almost certainly a misclick, so do
      // nothing rather than silently recording a closed pore.
      status("No detected region there. Click a pore, or outside the ROI for closed.");
      return;
    }

    // Seeded from whatever is on screen, so assigning one class leaves the other
    // four pointing at the pores they already had. Sending only the class just
    // clicked would blank the rest.
    this.assignment = { ...this.currentAssignment(), [k]: choice };
    this.manualClasses = new Set([...(this.manualClasses || []), k]);

    if (this.assignMode.sequential && k + 1 < this.analysis.results.rows.length) {
      this.assignMode = { sequential: true, next: k + 1 };
      this.showAssignStatus();
    } else {
      this.disarmAssign();
    }
    await this.applyAssignment();
  }

  async applyAssignment() {
    try {
      const out = await post("/api/test/fusion/assign", {
        candidates: this.analysis.candidates.map((c) => ({
          index: c.index, aa_mm2: c.aa_mm2, perimeter_mm: c.perimeter_mm,
          centroid: c.centroid, solidity: c.solidity,
        })),
        assignment: this.assignment || {},
        manual_classes: [...(this.manualClasses || [])],
        params: this.params(),
      });
      this.analysis = { ...this.analysis, ...out };
      this.measurements = out.measurements;
      this.render();
    } catch (e) { status(e.message, "bad"); }
  }

  async revertToAutomatic() {
    this.assignment = null;
    this.manualClasses = null;
    this.disarmAssign();
    await this.analyze();
  }

  async onShow() {
    if (!this.calibrations.length) await this.loadCalibrations();
    // Share the in-flight load rather than guarding with a boolean set before
    // the await: the tab-switch handler does not await this, so a second caller
    // would otherwise sail past a load that has not finished and read fields
    // that are still showing the factory values.
    this.defaultsPromise ||= this.loadDefaults();
    await this.defaultsPromise;
  }

  /* --------------------------------------------------- stored defaults */

  async loadDefaults() {
    try {
      const stored = await api(`/api/defaults/${this.type}`);
      if (stored.values && Object.keys(stored.values).length) {
        this.applyDefaults(stored.values);
      }
      this.showDefaultsStatus(stored.updated_at);
    } catch (e) { status(e.message, "bad"); }
  }

  /**
   * Write a stored set into the tab's inputs.
   *
   * Driven off the same [data-f]/[data-p] selectors that `fields()` and
   * `params()` read, so the list of what a tab contains lives in one place. A
   * key with no matching input is ignored and a field with no stored value keeps
   * its factory default, which is what lets a set saved by an older build still
   * load after the tab gains an input.
   */
  applyDefaults(values) {
    for (const [key, value] of Object.entries(values)) {
      if (value === null || value === undefined) continue;
      const input = this.q(`[data-f="${key}"]`) || this.q(`[data-p="${key}"]`);
      if (!input) continue;
      if (input.type === "checkbox") input.checked = !!value;
      else if (Array.isArray(value)) input.value = value.join(", ");
      else input.value = value;
    }

    // Structured values that are not plain inputs.
    if (this.type === "fusion" && Array.isArray(values.arista_mm)) {
      this.pendingAristas = values.arista_mm;
      this.rebuildAristaRows?.();
      // Rebuild reads the existing inputs, so seed them and rebuild once more.
      const host = $("#fusion-at");
      values.arista_mm.forEach((a, i) => {
        const input = host.querySelector(`input[data-arista="${i}"]`);
        if (input) input.value = a;
      });
      this.rebuildAristaRows?.();
    }
    if (this.type === "collapse" && values.hsv) {
      for (const key of ["pillar", "filament"]) {
        if (values.hsv[`${key}_sample`]) {
          this.colours[key] = {
            sample: values.hsv[`${key}_sample`],
            ranges: values.hsv[key] || [],
          };
        }
      }
      this.paintSwatches();
    }
    if (this.type === "collapse") this.rebuildAmaxRows?.();
  }

  collectDefaults() {
    const values = { ...this.fields(), ...this.params() };
    delete values.name;
    delete values.replicate_no;
    if (this.type === "collapse") {
      values.hsv = {
        ...this.hsvParams(),
        pillar_sample: this.colours?.pillar?.sample,
        filament_sample: this.colours?.filament?.sample,
      };
    }
    return values;
  }

  showDefaultsStatus(updatedAt) {
    const node = this.q('[data-role="defaults-status"]');
    if (!node) return;
    node.textContent = updatedAt
      ? `Using defaults saved ${updatedAt}.`
      : "Using the factory values. Save as defaults to keep the current settings.";
  }

  async saveDefaults() {
    try {
      const stored = await put(`/api/defaults/${this.type}`,
                               { values: this.collectDefaults() });
      this.showDefaultsStatus(stored.updated_at);
      status("Saved as the defaults for this tab.", "good");
    } catch (e) { status(e.message, "bad"); }
  }

  async resetDefaults() {
    if (!confirm(
      "Discard the saved defaults for this tab and go back to the factory values?"
    )) return;
    try {
      await del(`/api/defaults/${this.type}`);
      this.showDefaultsStatus(null);
      status("Defaults cleared. Reload the page to see the factory values.", "good");
    } catch (e) { status(e.message, "bad"); }
  }

  /* ------------------------------------------------- colour sampling */

  resetColours() {
    const d = D.hsv_defaults || {};
    // Midpoint of each default range, purely so the swatches start meaningful.
    this.colours = {
      pillar: { sample: { r: 220, g: 158, b: 142 }, ranges: [d.pillar, d.pillar2].filter(Boolean) },
      filament: { sample: { r: 59, g: 82, b: 200 }, ranges: [d.filament].filter(Boolean) },
    };
    this.paintSwatches();
  }

  armPicker(key, button) {
    if (!this.canvas) return status("Capture or load an image first.", "bad");
    this.qa("[data-pick]").forEach((b) => b.classList.remove("armed"));
    button.classList.add("armed");
    this.canvas.armPicker(key);
    status(`Click the ${key === "pillar" ? "ABS platform" : "filament"} in the image.`);
  }

  onColourPicked(key, sample) {
    const tol = parseFloat(this.q('[data-p="hue_tolerance"]').value) || 12;
    this.colours[key] = { sample, ranges: hsvWindow(sample, tol) };
    this.qa("[data-pick]").forEach((b) => b.classList.remove("armed"));
    this.paintSwatches();
    status(`${key === "pillar" ? "Platform" : "Filament"} colour set from ` +
           `(${sample.x}, ${sample.y}).`, "good");
  }

  paintSwatches() {
    for (const [key, entry] of Object.entries(this.colours || {})) {
      const sw = this.q(`[data-swatch="${key}"]`);
      if (sw && entry?.sample) sw.style.background = rgbCss(entry.sample);
    }
  }

  hsvParams() {
    const out = {};
    const pillar = this.colours?.pillar?.ranges || [];
    const filament = this.colours?.filament?.ranges || [];
    if (pillar[0]) out.pillar = pillar[0];
    if (pillar[1]) out.pillar2 = pillar[1];
    if (filament[0]) out.filament = filament[0];
    return out;
  }

  /* ---------------------------------------------------- calibrations */

  async loadCalibrations() {
    const { calibrations } = await api("/api/calib/list");
    this.calibrations = calibrations;
    const preferred = this.type === "collapse" ? "lateral" : "top_down";
    const select = this.q('[data-role="calibration"]');
    select.replaceChildren(
      el("option", { value: "" }, "None — results stay in pixels"),
      ...calibrations.map((c) => el("option", { value: c.id },
        `${c.name} (${c.intended_use}${c.mode ? ", " + c.mode : ""})`)));
    const match = calibrations.find((c) => c.intended_use === preferred);
    if (match) select.value = match.id;
    this.showCalibHint();
  }

  showCalibHint() {
    const id = this.q('[data-role="calibration"]').value;
    const c = this.calibrations.find((x) => String(x.id) === String(id));
    const hint = this.q('[data-role="calib-hint"]');
    if (!c) {
      hint.textContent =
        "Without a calibration every length is in pixels, so mm results will be wrong.";
      hint.style.color = "var(--warn)";
      return;
    }
    const wanted = this.type === "collapse" ? "lateral" : "top_down";
    const mismatch = c.intended_use !== wanted;
    hint.textContent = mismatch
      ? `This profile is tagged “${c.intended_use}” but this test expects “${wanted}”. ` +
        "The scale will be wrong unless the board was shot in this test's plane."
      : `${num(c.mm_per_px_x, 5)} mm/px · checker ${num(c.scale_square_mm, 2)} mm` +
        (c.rms_px ? ` · RMS ${num(c.rms_px, 3)} px` : "");
    hint.style.color = mismatch ? "var(--warn)" : "var(--muted)";
  }

  /* --------------------------------------------------- parameter rows */

  buildParamRows() {
    if (this.type === "fusion") {
      this.rebuildAristaRows = () => {
        const host = $("#fusion-at");
        const n = Math.max(1, parseInt(this.q('[data-p="grid_n"]').value, 10) || 5);
        const d = parseFloat(this.q('[data-p="filament_d_mm"]').value) || 0;

        // Values already typed are kept; a larger N appends the next integers
        // rather than resetting the list, so raising the grid size does not
        // throw away edits.
        const current = $$("#fusion-at input[data-arista]")
          .map((i) => parseFloat(i.value));
        const aristas = [];
        for (let i = 0; i < n; i++) {
          aristas.push(Number.isFinite(current[i]) ? current[i]
            : (Number.isFinite(D.fusion_arista_mm?.[i]) ? D.fusion_arista_mm[i] : i + 1));
        }

        host.replaceChildren(...aristas.map((a, i) => {
          const side = a - d;
          const at = side > 0 ? (side * side).toFixed(4) : "undefined";
          const atCell = el("span", { class: "val" }, `Aₜ ${at}`);
          if (side <= 0) atCell.style.color = "var(--bad)";
          return el("div", { class: "slider-row" },
            el("label", {}, `Class ${i + 1} — a`),
            el("input", {
              type: "number", step: "0.01", min: "0", value: a, "data-arista": i,
              onchange: () => this.rebuildAristaRows(),
            }),
            atCell);
        }));
      };
      this.q('[data-p="grid_n"]').addEventListener("change", () => this.rebuildAristaRows());
      this.q('[data-p="filament_d_mm"]').addEventListener("change", () => this.rebuildAristaRows());
      this.rebuildAristaRows();
    }
    if (this.type === "collapse") {
      const host = $("#collapse-amax");
      this.rebuildAmaxRows = () => {
        const gaps = parseList(this.q('[data-p="gaps_mm"]').value);
        const h = parseFloat(this.q('[data-p="gap_height_mm"]').value) || 6;
        host.replaceChildren(...gaps.map((g, i) =>
          el("div", { class: "slider-row" },
            el("label", {}, `Gap ${i + 1} — ${g} mm`),
            el("input", { type: "number", step: "0.001", value: (g * h).toFixed(3), "data-amax": i }),
            el("span", { class: "val" }, "mm²"))));
      };
      this.q('[data-p="gaps_mm"]').addEventListener("change", () => this.rebuildAmaxRows());
      this.q('[data-p="gap_height_mm"]').addEventListener("change", () => this.rebuildAmaxRows());
      this.rebuildAmaxRows();
    }
  }

  params() {
    const p = {};
    this.qa("[data-p]").forEach((input) => {
      const key = input.dataset.p;
      if (key === "gaps_mm" || key === "pillar_widths_mm") {
        p[key] = parseList(input.value);
      } else if (input.type === "number") {
        p[key] = parseFloat(input.value);
      } else {
        p[key] = input.value;
      }
    });
    p.nozzle_id_mm = parseFloat(this.q('[data-f="nozzle_id_mm"]').value);

    if (this.type === "fusion") {
      p.arista_mm = $$("#fusion-at input[data-arista]").map((i) => parseFloat(i.value));
    }
    if (this.type === "collapse") {
      const amax = {};
      $$("#collapse-amax input").forEach((i) => { amax[i.dataset.amax] = parseFloat(i.value); });
      p.a_max_mm2 = amax;
      p.hsv = this.hsvParams();
      if (this.quad) p.quad = this.quad;
    }
    return p;
  }

  fields() {
    const f = {};
    this.qa("[data-f]").forEach((input) => {
      const v = input.value;
      f[input.dataset.f] = input.type === "number"
        ? (v === "" ? null : parseFloat(v)) : v;
    });
    return f;
  }

  /* -------------------------------------------------------- capture */

  async capture() {
    try {
      const r = await post("/api/camera/capture");
      await this.showImage(r);
    } catch (e) { status(e.message, "bad"); }
  }

  async upload(file) {
    if (!file) return;
    const form = new FormData();
    form.append("file", file);
    const res = await fetch("/api/image/upload", { method: "POST", body: form });
    const body = await res.json();
    if (!res.ok) return status(body.error || "Upload failed.", "bad");
    await this.showImage(body);
  }

  async showImage(r) {
    // Keep the ROI only when the new frame is the same size as the last one --
    // that is a retake of the same setup, and re-drawing the region every time
    // would be tedious. A different size is a different image, and carrying a
    // stale rectangle onto it silently measures the wrong part of the frame.
    const sameFraming = this.imageSize
      && this.imageSize[0] === r.width && this.imageSize[1] === r.height;
    if (!sameFraming) this.roi = null;
    this.imageSize = [r.width, r.height];

    this.imageId = r.image_id;
    const viewer = this.q('[data-role="viewer"]');
    let canvas = $("canvas", viewer);
    if (!canvas) {
      canvas = el("canvas");
      viewer.replaceChildren(canvas);
      this.canvas = new RoiCanvas(canvas, {
        diagonal: this.type === "fusion",
        // The collapse platform's own 51 mm width is the length reference, so
        // its ROI must start as the whole frame; cropping into the platform
        // rescales every measurement.
        defaultInset: this.type === "collapse" ? 0 : 0.15,
        onChange: (roi, quad) => {
          this.roi = roi;
          if (quad?.length === 4) this.quad = quad;
          this.q('[data-role="roi-readout"]').textContent =
            roi ? `ROI ${roi.w} × ${roi.h} px at (${roi.x}, ${roi.y})` : "";
        },
        onMarkClick: (id) => this.toggleMeasurement(id),
        onPick: (key, sample) => this.onColourPicked(key, sample),
        onAssign: (ev) => this.onAssignClick(ev),
      });
    }
    await this.canvas.setImage(r.url, this.roi);
    status("Frame captured. Drag corner to corner on the image to set the ROI.");
  }

  retake() {
    this.imageId = null;
    this.imageSize = null;
    this.roi = null;
    this.analysis = null;
    this.measurements = [];
    this.assignment = null;
    this.manualClasses = null;
    this.assignMode = null;
    this.canvas = null;
    this.q('[data-role="viewer"]').replaceChildren(
      el("div", { class: "placeholder" },
        "Capture a frame, or load an image file, to begin."));
    this.q('[data-role="metrics"]').replaceChildren(
      el("div", { class: "hint" }, "No analysis yet."));
    this.q('[data-role="table"]').replaceChildren();
    this.q('[data-role="flags"]').replaceChildren();
    this.q('[data-role="debug-card"]').style.display = "none";
  }

  /* -------------------------------------------------------- analyze */

  async analyze() {
    if (!this.imageId) return status("Capture or load an image first.", "bad");
    const debug = this.q('[data-act="debug"]').checked;

    // Re-detect always regenerates the debug panel, so stale stage images can
    // never be mistaken for the current run.
    this.q('[data-role="debug-card"]').style.display = "none";
    this.q('[data-role="debug"]').replaceChildren();

    // A fresh detection supersedes any manual assignment: the candidate indices
    // it referred to belong to the previous run and mean nothing now.
    this.assignment = null;
    this.manualClasses = null;
    if (this.assignMode) this.disarmAssign();

    try {
      status("Analysing…");
      const calibId = this.q('[data-role="calibration"]').value;
      const out = await post(`/api/test/${this.type}/analyze`, {
        image_id: this.imageId,
        roi: this.roi,
        params: this.params(),
        calibration_id: calibId ? parseInt(calibId, 10) : null,
        debug,
      });
      this.analysis = out;
      this.measurements = out.measurements || [];
      this.render();
      if (debug) this.renderDebug(out.debug || []);
      status("Analysis complete.", "good");
    } catch (e) {
      this.q('[data-role="flags"]').replaceChildren(alertBox("bad", e.message));
      status(e.message, "bad");
    }
  }

  render() {
    const out = this.analysis;
    if (!out) return;

    const flags = this.q('[data-role="flags"]');
    const boxes = [];
    if (!out.calibrated) {
      boxes.push(alertBox("warn",
        "No calibration selected — all lengths below are in pixels, not millimetres."));
    }
    (out.flags || []).forEach((f) => boxes.push(alertBox("warn", f)));
    flags.replaceChildren(...boxes);

    if (this.type === "uniformity") this.renderUniformity();
    else if (this.type === "fusion") this.renderFusion();
    else this.renderCollapse();

    this.updateMarks();
  }

  /* -------------------------------------------------- uniformity view */

  renderUniformity() {
    const r = this.analysis.results;
    const uiKind = !r.ui_valid ? "bad" : r.uniformity_index > 0.95 ? "good" : "warn";
    this.q('[data-role="metrics"]').replaceChildren(el("div", { class: "metrics" },
      metric("Uniformity index", r.ui_valid ? num(r.uniformity_index, 4) : "invalid",
        r.ui_valid ? "UI = 1 − CV" : "CV ≥ 1", uiKind),
      metric("Spreading ratio", num(r.spreading_ratio, 4),
        `D̄ / Dₙ (Dₙ = ${num(r.nozzle_id_mm, 2)} mm)`),
      metric("Mean thickness D̄", num(r.mean_mm, 4), `mm — ${num(r.mean_um, 1)} µm`),
      metric("SD", num(r.sd_mm, 4), "mm"),
      metric("CV", num(r.cv_percent, 2), "%"),
      metric("N included", `${r.n_included} / ${r.n_total}`, "measurements")));

    if (!r.continuity?.continuous) {
      this.q('[data-role="flags"]').prepend(alertBox("bad",
        "Discontinuous filament — the sample fails at this stage. " +
        "Uniformity statistics are still shown, but a broken filament cannot be assessed."));
    }

    const list = this.q('[data-role="checklist"]');
    list.replaceChildren(...this.measurements.map((m) => {
      const box = el("input", {
        type: "checkbox", ...(m.included ? { checked: "" } : {}),
        ...(m.raw.thickness_mm === null ? { disabled: "" } : {}),
        onchange: () => this.toggleMeasurement(m.index_no),
      });
      return el("div", {},
        box, el("span", {}, m.label),
        el("span", { class: "val" },
          m.raw.thickness_mm === null ? "not found"
            : `${num(m.raw.thickness_mm, 4)} mm`));
    }));
    this.q('[data-role="table"]').replaceChildren();
  }

  /* ------------------------------------------------------ fusion view */

  renderFusion() {
    const r = this.analysis.results;
    this.q('[data-role="metrics"]').replaceChildren(el("div", { class: "metrics" },
      metric("Mean Dfr", num(r.mean_dfr_percent, 2), "%"),
      metric("Mean Pr", num(r.mean_pr, 4), `window ${D.pr_window?.join("–")}`),
      metric("Mean circularity", num(r.mean_circularity, 4), "C (square = 0.785)"),
      metric("Pores", `${r.n_open} open / ${r.n_closed} closed`, `of ${r.n_pores}`)));

    this.q('[data-role="assign-card"]').style.display =
      this.analysis.candidates?.length ? "block" : "none";

    const head = ["", "Pore", "a mm", "Centroid", "Status", "Source", "At mm²",
      "Aa mm²", "L mm", "Dfr %", "Pr", "C", "Flags"];
    this.q('[data-role="table"]').replaceChildren(el("table", {},
      el("thead", {}, el("tr", {}, ...head.map((h) => el("th", {}, h)))),
      el("tbody", {}, ...r.rows.map((row, k) => {
        const w = row.raw;
        const prCell = el("td", { class: "num" }, num(w.pr, 4));
        if (w.pr_in_window === false) prCell.style.color = "var(--warn)";

        const armed = this.assignMode && !this.assignMode.sequential
          && this.assignMode.classIndex === k;
        const assignBtn = el("button", {
          class: armed ? "armed" : "",
          style: "padding:1px 8px;font-size:11px",
          onclick: () => (armed ? this.disarmAssign()
            : this.armAssign({ classIndex: k })),
        }, armed ? "Click…" : "Assign");

        const manual = (w.selection || "auto").startsWith("manual");
        return el("tr", {},
          el("td", {}, assignBtn),
          el("td", {}, `#${w.size_class}`),
          el("td", { class: "num" }, num(w.arista_mm, 2)),
          el("td", { class: "num" }, w.centroid
            ? `${w.centroid.map((v) => Math.round(v)).join(", ")}` : "—"),
          el("td", {}, el("span", {
            class: `pill ${w.status === "open" ? "good" : w.status === "closed" ? "warn" : "bad"}`,
          }, w.status)),
          el("td", {}, el("span", {
            class: `pill ${manual ? "warn" : ""}`,
          }, manual ? "manual" : "auto")),
          el("td", { class: "num" }, num(w.at_mm2, 3)),
          el("td", { class: "num" }, num(w.aa_mm2, 3)),
          el("td", { class: "num" }, num(w.perimeter_mm, 3)),
          el("td", { class: "num" }, num(w.dfr_percent, 2)),
          prCell,
          el("td", { class: "num" }, num(w.circularity, 4)),
          el("td", {}, (w.flags || []).join(" ")));
      }))));
  }

  /* ---------------------------------------------------- collapse view */

  renderCollapse() {
    const r = this.analysis.results;
    this.q('[data-role="metrics"]').replaceChildren(
      alertBox("info", r.convention_label),
      el("div", { class: "metrics" },
        metric("Mean Cf", num(r.cf?.mean, 2), "%"),
        metric("SD Cf", num(r.cf?.sd, 2), "%"),
        metric("Mean θ", num(r.theta?.mean, 2), "°"),
        metric("Bridged", `${r.n_bridged} / ${r.n_gaps}`,
          `${r.n_broken} broken`, r.n_broken ? "warn" : "good"),
        metric("Filament ⌀ df", num(r.df_mm, 4), "mm (on pillar tops)")));

    const cc = r.scale_cross_check;
    if (cc?.warn) {
      this.q('[data-role="flags"]').prepend(alertBox("warn",
        `The platform's 51 mm width implies ${num(cc.implied_mm_per_px, 5)} mm/px, but the ` +
        `selected calibration says ${num(cc.calibration_mm_per_px, 5)} mm/px — a ` +
        `${(cc.disagreement * 100).toFixed(1)} % disagreement. Check that the calibration ` +
        "board was photographed in the pillar plane."));
    }

    const head = ["Gap", "Nominal mm", "Status", "A_sag mm²", "A_max mm²",
      "At strip mm²", "df mm", "Cf %", "θ °"];
    this.q('[data-role="table"]').replaceChildren(el("table", {},
      el("thead", {}, el("tr", {}, ...head.map((h) => el("th", {}, h)))),
      el("tbody", {}, ...r.rows.map((row) => {
        const w = row.raw;
        return el("tr", {},
          el("td", {}, `#${w.gap_no}`),
          el("td", { class: "num" }, num(w.nominal_gap_mm, 1)),
          el("td", {}, el("span", {
            class: `pill ${w.status === "bridged" ? "good" : "bad"}`,
          }, w.status)),
          el("td", { class: "num" }, num(w.a_sag_mm2, 3)),
          el("td", { class: "num" }, num(w.a_max_mm2, 3)),
          el("td", { class: "num" }, num(w.at_strip_mm2, 3)),
          el("td", { class: "num" }, num(w.df_mm, 4)),
          el("td", { class: "num" }, num(w.cf_percent, 2)),
          el("td", { class: "num" }, num(w.theta_deg, 2)));
      }))));
  }

  /* -------------------------------------------------------- overlays */

  updateMarks() {
    if (!this.canvas || !this.analysis) return;
    const marks = [];
    if (this.type === "uniformity") {
      for (const m of this.measurements) {
        if (m.raw.thickness_px === null) continue;
        marks.push({
          kind: "tick", id: m.index_no, included: m.included,
          x: m.raw.x_full, y0: m.raw.top_full, y1: m.raw.bottom_full,
          label: m.label,
        });
      }
    } else if (this.type === "fusion") {
      // Which candidate is currently standing in for which size class, matched
      // on centroid because a row does not otherwise carry its candidate index
      // when it came from the automatic pass.
      const assignedTo = new Map();
      (this.analysis.results.rows || []).forEach((row, k) => {
        const c = row.raw.centroid;
        if (!c) return;
        assignedTo.set(`${Math.round(c[0])},${Math.round(c[1])}`, k + 1);
      });

      for (const cand of this.analysis.candidates || []) {
        const key = `${Math.round(cand.centroid[0])},${Math.round(cand.centroid[1])}`;
        const sizeClass = assignedTo.get(key);
        marks.push({
          kind: "pore", id: cand.index,
          x: cand.centroid[0], y: cand.centroid[1],
          points: cand.polygon,
          area: cand.area_px,
          // Yellow contour, green centroid. An assigned pore is drawn brighter
          // and thicker, and carries its size-class number.
          color: sizeClass ? "#ff9d2d" : "#ffd23f",
          centroidColor: "#25e06a",
          lineWidth: sizeClass ? 3 : 1.5,
          label: sizeClass ? String(sizeClass) : "",
          labelColor: "#ff9d2d",
        });
      }
    } else {
      const off = this.analysis.offset || [0, 0];
      for (const row of this.analysis.results.rows) {
        const w = row.raw;
        if (w.max_sag_x === null || w.max_sag_x === undefined) continue;
        marks.push({
          kind: "point", id: row.index_no, included: row.included,
          x: w.max_sag_x + off[0], y: w.top_y + (w.max_sag_px || 0) + off[1],
          r: 5, label: `#${w.gap_no}`,
        });
      }
    }
    this.canvas.setMarks(marks);
  }

  async toggleMeasurement(id) {
    const m = this.measurements.find((x) => x.index_no === id);
    if (!m || m.raw.thickness_mm === null) return;
    m.included = !m.included;
    try {
      const r = await post(`/api/test/${this.type}/recompute`, {
        measurements: this.measurements,
        params: { nozzle_id_mm: parseFloat(this.q('[data-f="nozzle_id_mm"]').value),
                  convention: this.params().convention },
      });
      this.analysis.results = { ...this.analysis.results, ...r.results };
      this.render();
    } catch (e) { status(e.message, "bad"); }
  }

  renderDebug(stages) {
    const card = this.q('[data-role="debug-card"]');
    const host = this.q('[data-role="debug"]');
    host.replaceChildren(...stages.map((s) =>
      el("div", { class: "debug-stage" },
        el("h4", {}, s.name),
        s.caption ? el("p", {}, s.caption) : null,
        el("img", { src: s.image, alt: s.name }))));
    card.style.display = stages.length ? "block" : "none";
  }

  /* ------------------------------------------------------------ save */

  /**
   * Upload the annotated canvas as a second image.
   *
   * Exported from the canvas rather than re-drawn on the server, so what is
   * stored is exactly what was on screen when the numbers were accepted --
   * including which measurements were left unchecked.
   */
  async saveOverlay() {
    const full = this.canvas?.exportFullSize();
    if (!full) return null;
    const blob = await new Promise((resolve) => full.toBlob(resolve, "image/png"));
    if (!blob) return null;
    const form = new FormData();
    form.append("file", blob, "overlay.png");
    const res = await fetch("/api/image/upload", { method: "POST", body: form });
    if (!res.ok) return null;
    return (await res.json()).image_id;
  }

  async save() {
    if (!this.analysis) return status("Run the analysis before saving.", "bad");
    const calibId = this.q('[data-role="calibration"]').value;
    try {
      const overlayId = await this.saveOverlay();
      const r = await post("/api/runs", {
        test_type: this.type,
        ...this.fields(),
        calibration_id: calibId ? parseInt(calibId, 10) : null,
        image_id: this.imageId,
        overlay_id: overlayId,
        roi: this.roi,
        params: this.params(),
        results: this.analysis.results,
        measurements: this.measurements,
        flags: this.analysis.flags,
        convention: this.params().convention || null,
      });
      status(`Saved as run #${r.id}.`, "good");
    } catch (e) { status(e.message, "bad"); }
  }
}

const Tests = {};

/* ======================================================== DATABASE TAB */

const Database = {
  current: null,

  init() {
    $("#db-refresh").onclick = () => this.refresh();
    $("#db-type").onchange = () => this.refresh();
    $("#db-name").oninput = () => this.refresh();
    $("#db-recompute").onclick = () => this.recompute();
    $("#db-reanalyze").onclick = () => this.reanalyze();
    $("#db-save").onclick = () => this.save();
    $("#db-duplicate").onclick = () => this.duplicate();
    $("#db-delete").onclick = () => this.remove();
  },

  async refresh() {
    const params = new URLSearchParams();
    if ($("#db-type").value) params.set("test_type", $("#db-type").value);
    if ($("#db-name").value) params.set("name", $("#db-name").value);
    const { runs } = await api(`/api/runs?${params}`);
    const head = ["ID", "Test", "Name", "Rep", "Created", "Key result", ""];
    $("#db-list").replaceChildren(el("table", {},
      el("thead", {}, el("tr", {}, ...head.map((h) => el("th", {}, h)))),
      el("tbody", {}, ...runs.map((run) => el("tr", {},
        el("td", {}, `#${run.id}`),
        el("td", {}, run.test_type),
        el("td", {}, run.name || "—"),
        el("td", { class: "num" }, run.replicate_no ?? "—"),
        el("td", {}, (run.created_at || "").slice(0, 16)),
        el("td", { class: "num" }, this.summary(run)),
        el("td", {}, el("button", { onclick: () => this.open(run.id) }, "Open")))))));
  },

  summary(run) {
    const r = run.results_json || {};
    if (run.test_type === "uniformity") {
      return `UI ${num(r.uniformity_index, 4)} · SR ${num(r.spreading_ratio, 3)}`;
    }
    if (run.test_type === "fusion") {
      return `Pr ${num(r.mean_pr, 3)} · Dfr ${num(r.mean_dfr_percent, 1)} %`;
    }
    return `Cf ${num(r.cf?.mean, 1)} % · θ ${num(r.theta?.mean, 1)}°`;
  },

  async open(id) {
    const run = await api(`/api/runs/${id}`);
    this.current = run;
    $("#db-detail").style.display = "block";

    const viewer = $("#db-viewer");
    if (run.image_path) {
      // Default to the annotated overlay where one was stored: it is what shows
      // which measurements produced these numbers.
      const showOverlay = !!run.overlay_path;
      const img = el("img", {
        src: `/api/run-image/${run.id}?kind=${showOverlay ? "overlay" : "capture"}`,
      });
      viewer.replaceChildren(img);
      $("#db-image-toggle").style.display = run.overlay_path ? "" : "none";
      $("#db-image-toggle").value = showOverlay ? "overlay" : "capture";
      $("#db-image-toggle").onchange = (e) => {
        img.src = `/api/run-image/${run.id}?kind=${e.target.value}`;
      };
    } else {
      viewer.replaceChildren(el("div", { class: "placeholder" }, "No image stored."));
      $("#db-image-toggle").style.display = "none";
    }

    const F = [
      ["name", "Name", "text"], ["replicate_no", "Replicate no.", "number"],
      ["flow_rate_mms", "Flow rate (mm/s)", "number"],
      ["feed_rate_mms", "Feed rate (mm/s)", "number"],
      ["nozzle_id_mm", "Nozzle inner ⌀ (mm)", "number"],
      ["first_layer_height_mm", "First layer height (mm)", "number"],
      ["notes", "Notes", "text"],
    ];
    $("#db-fields").replaceChildren(...F.map(([key, label, type]) =>
      el("div", { class: "field" },
        el("label", {}, label),
        el("input", { type, id: `db-f-${key}`, value: run[key] ?? "" }))));

    $("#db-flags").replaceChildren(
      ...(run.flags_json || []).map((f) => alertBox("warn", f)));

    this.renderMetrics();
    this.renderMeasurements();
  },

  renderMetrics() {
    const run = this.current;
    const r = run.results_json || {};
    let items;
    if (run.test_type === "uniformity") {
      items = [metric("UI", num(r.uniformity_index, 4)),
        metric("SR", num(r.spreading_ratio, 4)),
        metric("D̄", num(r.mean_mm, 4), "mm"),
        metric("CV", num(r.cv_percent, 2), "%"),
        metric("N", `${r.n_included} / ${r.n_total}`)];
    } else if (run.test_type === "fusion") {
      items = [metric("Mean Dfr", num(r.mean_dfr_percent, 2), "%"),
        metric("Mean Pr", num(r.mean_pr, 4)),
        metric("Mean C", num(r.mean_circularity, 4))];
    } else {
      items = [metric("Mean Cf", num(r.cf?.mean, 2), "%"),
        metric("SD Cf", num(r.cf?.sd, 2), "%"),
        metric("Mean θ", num(r.theta?.mean, 2), "°")];
    }
    $("#db-metrics").replaceChildren(
      run.test_type === "collapse" && r.convention_label
        ? alertBox("info", r.convention_label) : el("span"),
      el("div", { class: "metrics" }, ...items));
  },

  renderMeasurements() {
    const run = this.current;
    $("#db-measurements").replaceChildren(...run.measurements.map((m) => {
      const box = el("input", {
        type: "checkbox", ...(m.included ? { checked: "" } : {}),
        onchange: () => { m.included = box.checked; },
      });
      const raw = m.raw || {};
      const value = raw.thickness_mm !== undefined ? `${num(raw.thickness_mm, 4)} mm`
        : raw.aa_mm2 !== undefined ? `Aa ${num(raw.aa_mm2, 3)} mm²`
        : `Cf ${num(raw.cf_percent, 2)} %`;
      return el("div", {}, box, el("span", {}, m.label || `#${m.index_no}`),
        el("span", { class: "val" }, value));
    }));
  },

  async recompute() {
    const run = this.current;
    const r = await post(`/api/test/${run.test_type}/recompute`, {
      measurements: run.measurements,
      params: {
        nozzle_id_mm: parseFloat($("#db-f-nozzle_id_mm").value) || run.nozzle_id_mm,
        convention: run.convention || D.cf_convention,
      },
    });
    run.results_json = { ...run.results_json, ...r.results };
    this.renderMetrics();
    status("Recomputed from stored measurements.", "good");
  },

  async reanalyze() {
    const run = this.current;
    try {
      status("Re-analysing the stored image…");
      const out = await post(`/api/runs/${run.id}/reanalyze`, {});
      run.results_json = out.results;
      run.measurements = out.measurements;
      run.flags_json = out.flags;
      this.renderMetrics();
      this.renderMeasurements();
      $("#db-flags").replaceChildren(...(out.flags || []).map((f) => alertBox("warn", f)));
      status("Re-analysed. Save to keep the new values.", "good");
    } catch (e) { status(e.message, "bad"); }
  },

  async save() {
    const run = this.current;
    const payload = { measurements: run.measurements, results: run.results_json };
    for (const key of ["name", "replicate_no", "flow_rate_mms", "feed_rate_mms",
      "nozzle_id_mm", "first_layer_height_mm", "notes"]) {
      const input = $(`#db-f-${key}`);
      if (!input) continue;
      payload[key] = input.type === "number"
        ? (input.value === "" ? null : parseFloat(input.value)) : input.value;
    }
    await put(`/api/runs/${run.id}`, payload);
    status("Saved.", "good");
    await this.refresh();
  },

  async duplicate() {
    const r = await post(`/api/runs/${this.current.id}/duplicate`);
    status(`Duplicated as run #${r.id}.`, "good");
    await this.refresh();
  },

  async remove() {
    if (!confirm(`Delete run #${this.current.id} permanently? This cannot be undone.`)) return;
    await del(`/api/runs/${this.current.id}`);
    $("#db-detail").style.display = "none";
    this.current = null;
    await this.refresh();
    status("Run deleted.");
  },
};

/* ------------------------------------------------------------ boot */

$("#shutdown").onclick = async () => {
  if (!confirm(
    "Disconnect the camera and shut the server down?\n\n" +
    "Saved runs are kept. Anything captured but not yet saved is lost, and the " +
    "page will stop working until you start the server again."
  )) return;
  $("#shutdown").disabled = true;
  try {
    await post("/api/shutdown");
  } catch {
    // The process exits mid-response often enough that a network error here is
    // the expected outcome, not a failure.
  }
  document.body.innerHTML =
    '<div style="padding:60px;text-align:center;color:#9aa3b2;font:14px system-ui">' +
    "<h2 style='color:#e6e9ef'>Server stopped</h2>" +
    "<p>The camera has been released. Restart with " +
    "<code style='color:#4da3ff'>python -m uvicorn app.main:app --port 8000</code>" +
    " and reload this page.</p></div>";
};

for (const t of ["uniformity", "fusion", "collapse"]) Tests[t] = new TestTab(t);

Camera.init().catch((e) => status(e.message, "bad"));
Calib.init();
Database.init();
for (const t of Object.values(Tests)) t.loadCalibrations().catch(() => {});

// Exposed so the tabs can be driven from the console when diagnosing a bad
// detection without clicking through the whole capture flow.
window.HydroGelCam = { Tests, Camera, Calib, Database, api };
