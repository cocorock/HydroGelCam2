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

### Colours (collapse tab)

Two eyedroppers set the segmentation: click **Pick**, then click the ABS platform
or the filament in the image, and the pixel under the cursor becomes the centre
of that material's HSV range. Hue carries the tolerance you control; saturation
and value get a wide band, because shading across a curved filament or a matte
ABS face moves both a long way while the hue barely shifts. Pale red ABS sits at
H ≈ 0, so its window straddles the hue wrap and is applied as two ranges.

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

**`At` for the fusion test** is `FD²` — FD is taken as the edge-to-edge pore gap,
so nominal pores are exactly 1×1 … 5×5 mm. Every value is editable per pore.

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
