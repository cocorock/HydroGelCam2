# HydroGelCam2

Measures 3D-printability of hydrogel bioinks from photographs taken with a USB
endoscopy camera. Implements the three printability tests from Mancilla Corzo
et al., *Bioprinting* **43** (2024) e00358 (`Ingri2024.pdf`, §2.4):

| Tab | Test | Metrics |
|---|---|---|
| 2 | Filament uniformity | Uniformity index `UI = 1 − CV`, spreading ratio `SR = D̄/Dₙ` |
| 3 | Filament fusion | Diffusion rate `Dfr`, printability `Pr`, circularity `C` (Eqs. 3, 4) |
| 4 | Filament collapse | Collapse factor `Cf`, deflection angle `θ` (Eq. 5) |

Everything runs locally: a Python/OpenCV backend, a browser front end, and a
SQLite file. No account, no password, no network.

## Running it

```bash
pip install -r requirements.txt
```

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Then open <http://127.0.0.1:8000>.

## Workflow

1. **Tab 1** — pick the camera, set its image controls, then calibrate in two
   stages: the intrinsic matrix and distortion from ~10 chessboard views, and
   then pixel→millimetre from one board photographed *in the plane the sample
   will occupy*. Save as a named profile tagged `top_down` (tabs 2–3) or
   `lateral` (tab 4).
2. **Tabs 2–4** — fill in the sample fields, capture a frame, drag corner to
   corner on the image to set the region of interest, then **Calculate**.
   Tick **Show preprocessing steps** to see every stage of the pipeline.
   **Save to database** stores the photo, the parameters and the results together.
3. **Tab 5** — browse, edit, recompute and export. *Recompute* re-runs the
   formulas over the stored measurements instantly; *Re-analyze* runs the whole
   pipeline again on the stored photograph.

Each test tab carries the equations it uses in a panel on the right, with its
symbols defined, so the numbers in the results table can be read against the
formula that produced them.

The **Disconnect & shut down** button at the top right releases the camera and
stops the server. Saved runs are kept; anything captured but not saved is lost.

### Region of interest

Dragging corner to corner always starts a **new** rectangle, matching
`cv2.selectROI`. Corner and edge handles resize it, and **Shift**-drag moves it
whole. A stray click leaves the current rectangle alone. Loading a frame of a
different size resets the ROI; a retake at the same size keeps it.

The ROI is **cropped exactly and then padded** with an 85 px border — black
behind bright material, white behind dark. Nothing outside the region you drew
can reach the threshold or the morphology, so a neighbouring filament, the edge
of the glass plate, or a reflection just outside the ROI no longer influences a
measurement. The padding width and the open/close cleanup kernel are both
editable per tab. The histogram in step 4 is built from the ROI's own pixels
only; the constant-coloured padding is excluded, since a border of tens of
thousands of identical pixels would put a spike in one bin large enough to
suppress the real background and material peaks.

The collapse tab defaults to the **whole frame**, because that test derives its
pillar and gap positions from the platform's full 51 mm width — cropping into
the platform rescales every result. The app flags it if you do.

### Choosing pores by hand (fusion tab)

The lattice step is what misfires on a bad print — walls break, cells merge — so
after *Calculate* every detected region is drawn with a **yellow contour** and a
**green centroid**, not just the five the automatic pass chose. The ones in use
are drawn brighter and carry their size-class number.

**Assign** on a results row points that class at a different region: click the
pore or its centroid. **Assign all sequentially** walks from class 1 to N.
Clicking **outside the ROI** records that class as a closed pore (`Aa` = 0,
`Dfr` = 100 %); clicking inside on empty space is ignored rather than guessed.
Esc cancels, and **Revert to automatic** re-runs the detection.

Classes you do not touch keep the pore the automatic pass gave them, and each row
records whether it was chosen automatically or by hand. That provenance shows in
the results table, is stored with the run, and appears in the CSV — for a number
that ends up in a paper, being able to say which pores were operator-selected is
worth having. Two classes landing on the same region is flagged.

### Saved defaults (tabs 2–4)

Each test tab has **Save as defaults**, **Restore defaults** and **Reset to
factory**. Saving snapshots the current parameters into the database so they
come back on the next launch. Nothing is captured automatically — a value typed
once for an odd sample should not silently become the new normal.

**Name and Replicate no. are never stored**, since carrying yesterday's sample
name forward is how a run ends up mislabelled. A stored set that predates a new
input still loads; the missing field just falls back to its factory value.

### Colours (collapse tab)

Two eyedroppers set the segmentation: click **Pick**, then click the ABS platform
or the filament in the image, and the pixel under the cursor becomes the centre
of that material's HSV range. Hue carries the tolerance you control; saturation
and value get a wide band, because shading across a curved filament or a matte
ABS face moves both a long way while the hue barely shifts. Pale red ABS sits at
H ≈ 0, so its window straddles the hue wrap and is applied as two ranges.

### Scale bar

Once a calibration is selected, a scale bar is drawn in the bottom-right corner
of the captured image showing **20 mm, 10 mm and 5 mm** as three nested bars,
each divided into 1 mm cells so it reads as a ruler rather than a line of a
stated length. Because it is drawn on the canvas, it lands in the saved overlay
automatically and is therefore in whatever you export from tab 5.

A length that would take more than about half the frame is dropped — a 20 mm bar
across a 19.5 mm field cannot be drawn honestly — so a close-up shows 10 and
5 mm, and a very small field falls back to sub-millimetre steps rather than
showing nothing. A homography profile is labelled *approx.*, since there the
scale varies across the frame and one bar can only be representative.

### What gets stored

Each saved run keeps **two images**: the original capture, and the annotated
overlay exactly as it appeared on screen — ROI rectangle, measurement ticks, pore
contours, and which measurements were left unchecked. The overlay is exported at
the capture's native resolution, not the scaled-to-fit view. Tab 5 shows the
overlay by default with a selector to fall back to the untouched capture.

### Material appearance

Whether the dyed material reads brighter or darker than the background is a
property of your dye and lighting, so it is a **setting** rather than something
guessed per frame. Auto-detection exists but is only reliable when the printed
pattern covers a clear minority of the frame; a wrong guess silently inverts the
whole segmentation.

## Conventions, and where they differ from the paper

**`Cf` direction.** The paper contradicts itself. §2.4.3 says "C_f = 100 % for
filaments that do not collapse"; the Results (p. 6, Fig. 3d) say "1C4L ink had a
**lower** area collapse factor … indicat[ing] low deformation." These are exact
complements. This app follows the **Results/Fig. 3d** direction so new data is
comparable with the published figure:

```
Cf = A_sag / A_max × 100      0 % = flat bridge, 100 % = fully collapsed or broken
A_sag = area between the pillar-top line and the filament underside
A_max = nominal gap × 6 mm pillar height
```

`A_sag`, `A_max`, `gap` and `df` are all stored, so the complementary convention
can be displayed at any time without re-analysing an image. The active
convention is written into every results table and CSV export.

**`At` for the fusion test** is `(a − d)²`, where `a` is the arista (the design
nozzle-path spacing for that size class) and `d` the filament diameter shared
across classes. The open pore is what is left between two filaments laid `a`
apart, so it is one filament width smaller than `a`: at `a` = 1 mm and
`d` = 0.41 mm the pore is 0.59 mm across and `At` = 0.348 mm². The label
"a = 1 mm" names the **spacing**, not the opening. `At` is undefined when
`a ≤ d` and is refused rather than divided by — a non-positive `At` would make
`Dfr` read as heavy spreading when the design simply leaves no pore.

There is one arista input per grid position, so the count follows N; raising N
appends the next integers and keeps whatever you have typed. Runs saved before
this change carry their own stored `at_mm2` and re-score exactly as before.

**Uniformity N** is `filaments × positions`, default 6 × 5 = 30. The paper's
formulas divide by 25; here the mean divides by the number of measurements left
checked and the SD by N−1, so unchecking outliers is handled correctly.

**`Pr` and `C` cannot cross-check each other.** `C ≡ π/(4·Pr)` identically for
any area and perimeter, so a disagreement between them is impossible by
construction. Segmentation quality is judged instead by the pass-1/pass-2
contour area agreement and each pore's solidity, both surfaced as per-pore flags.

## Layout

```
app/
  main.py            FastAPI routes
  config.py          paths, defaults, pipeline tunables
  camera/            device enumeration, single-owner capture thread, UVC controls
  calib/             intrinsic solve, pixel→mm scale, px↔mm geometry
  pipeline/          preprocessing steps 1–5 (common) and 6–8 (grid), debug trace
  analysis/          uniformity, fusion, collapse
  db/                SQLite schema and access
  web/               templates and front end (no build step)
storage/             database, captures, debug images  (created on first run)
tests/               synthetic images with known ground truth
```

## Tests

```bash
python -m pytest tests/ -q
```

The suite checks metrics against images whose answers are known exactly — a
serpentine of a stated thickness, square pores that must give `C = π/4` and
`Pr = 1`, a parabolic sag whose area is `⅔·depth·span`, and a chessboard rendered
through a known camera matrix that `calibrateCamera` must recover. It also covers
the awkward cases: a fully fused corner pore, a broken bridge, heavy vignetting,
dark-on-light material, and a cropped collapse platform.

## Notes on the image pipeline

The preprocessing follows the specified eight steps, with two deviations that
were necessary to make it work:

- **Illumination field.** The field is estimated by a morphological open on a
  heavily downscaled copy, not by blurring the image directly. A plain blur is
  inflated by the filaments themselves; a full-resolution morphological open
  cannot remove a fully fused corner, which is an *expected* result of the
  fusion test at 1 mm spacing. Rescaling uses the field's median rather than its
  maximum, which preserves material-to-background contrast instead of clipping
  the material away. Before the opening, the downscaled image is framed with
  background sampled along each of its own edges: an opening cannot remove a
  feature that runs into the border, and a filament near the edge of the ROI does
  exactly that — without the frame it survives into the field and flattens its
  own surroundings far too dark to threshold.
- **Fusion lattice.** Cells are taken from the *raw* wall projections, not the
  cleaned ones. The specified morphology chain ends in a net dilation, so walls in
  the cleaned map are ~9 px wider on each side than the filament actually is, and
  cells measured from it come out 18 px short in each direction — every pore area
  low. The cleaned map still decides which walls are real. Each pore is confined
  to its own cell, so a break in a vignette-dimmed wall cannot merge two pores
  into one oversized opening.
- **Threshold.** The rising edge of the histogram's second peak is located from
  the valley between the two peaks, not from the derivative's magnitude. For a
  tightly distributed material peak the derivative-based foot sits partway up
  the flank, so even after the ×0.9 margin the threshold could cut into solid
  material.

Filament edges are refined to sub-pixel at the half-height between the local
background and the filament plateau, not at the segmentation threshold —
refining at the threshold biases every width low by roughly one blur width,
which `UI` would not notice but `SR` would.
