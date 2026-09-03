"""Paths, defaults and pipeline tunables for HydroGelCam2."""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------- paths

ROOT = Path(__file__).resolve().parent.parent
APP_DIR = ROOT / "app"
WEB_DIR = APP_DIR / "web"
TEMPLATE_DIR = WEB_DIR / "templates"
STATIC_DIR = WEB_DIR / "static"

STORAGE = ROOT / "storage"
DB_PATH = STORAGE / "hydrogelcam.db"
CAPTURE_DIR = STORAGE / "captures"
DEBUG_DIR = STORAGE / "debug"
CALIB_DIR = STORAGE / "calib"

for _d in (STORAGE, CAPTURE_DIR, DEBUG_DIR, CALIB_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- server

HOST = "127.0.0.1"
PORT = 8000

TEST_TYPES = ("uniformity", "fusion", "collapse")

# ---------------------------------------------------------------- camera

DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
JPEG_QUALITY = 85
STREAM_FPS = 15

# ---------------------------------------------------------------- calibration

DEFAULT_BOARD_COLS = 9  # inner corners along x
DEFAULT_BOARD_ROWS = 6  # inner corners along y
DEFAULT_SQUARE_MM = 5.0
MIN_CALIB_FRAMES = 10
ANISOTROPY_WARN = 0.02  # warn when |mm_per_px_x - mm_per_px_y| / mean exceeds this

# ---------------------------------------------------------------- pipeline
# Values below are the ones specified in the measurement protocol. They are
# exposed through the API so each tab can override them per run.

# Step 2: the ROI is cropped exactly, then surrounded by a border of constant
# background-coloured pixels. Padding rather than an enlarged crop means nothing
# outside the region the user selected can ever influence a measurement -- a
# neighbouring filament, the edge of the glass plate, or a bright reflection just
# outside the ROI no longer leaks into the threshold or the morphology.
ROI_PADDING_PX = 85
VIGNETTE_KERNEL = 85        # step 3: blur kernel estimating the brightness field
# The field estimate runs on a downscaled copy so its structuring element can
# exceed any solid patch of deposited material without also erasing the
# illumination gradient. FIELD_SE_DIVISOR sets the element to 1/N of the coarse
# image's short side.
FIELD_COARSE_PX = 96
FIELD_SE_DIVISOR = 6
# Percentile of each window taken as the background level (mirrored to
# 100 - this for dark material). Low enough to reject deposited material, high
# enough not to chase sensor noise.
FIELD_PERCENTILE = 20.0

# Does the deposited material read brighter or darker than its background?
# "bright" | "dark" | "auto". This is a property of the dye and the lighting, so
# it is a setting rather than something re-guessed per frame: auto-detection is
# reliable only when the pattern covers a clear minority of the ROI, and a wrong
# guess silently inverts the whole segmentation. Selectable per test tab and
# saved on the camera profile.
DEFAULT_POLARITY = "bright"
POLARITY_CHOICES = ("bright", "dark", "auto")
POLARITY_LABELS = {
    "bright": "Material brighter than background",
    "dark": "Material darker than background",
    "auto": "Auto-detect (less reliable)",
}
HIST_SMOOTH_PASSES = 6      # step 4: repeated smoothing of the intensity histogram
HIST_SMOOTH_SIGMA = 2.0
THRESHOLD_MARGIN = 0.9      # step 4: safety factor on the detected rising edge
PROJECTION_COVERAGE = 0.15  # step 6: fraction of a row/column that marks a "wall"
PROJECTION_MORPH = (7, 9, 15, 33, 39, 9)  # open, close, erode, dilate, close, (final)
FINAL_MORPH_KERNEL = 9      # step 7

# ---------------------------------------------------------------- tests

# Tab 2 - filament uniformity
UNIFORMITY_FILAMENTS = 6
UNIFORMITY_POSITIONS = 5
UNIFORMITY_EDGE_INSET = 0.05  # fraction of ROI width kept clear at each end

# Tab 3 - filament fusion
FUSION_GRID_N = 5
# Arista (design nozzle-path spacing) per size class, and the filament diameter
# shared across them. The theoretical pore is what is left between two filaments
# laid a apart:  At = (a - d)^2.
FUSION_ARISTA_MM = (1.0, 2.0, 3.0, 4.0, 5.0)
FUSION_FILAMENT_D_MM = 0.41
# Legacy: earlier runs stored a filament-distance list with At = FD^2 taken
# edge-to-edge. Kept so a saved run from before the change still re-analyses.
FUSION_FD_MM = (1.0, 2.0, 3.0, 4.0, 5.0)
FUSION_DIAGONAL_TOLERANCE = 0.35  # fraction of mean pore spacing
PR_ACCEPT_LOW = 0.9               # Ouyang et al. acceptance window
PR_ACCEPT_HIGH = 1.1
AREA_AGREEMENT_TOL = 0.10         # pass1 vs pass2 contour area disagreement
SOLIDITY_MIN = 0.85
# What counts as a pore in pass 1, rather than a fragment. Area is judged against
# the largest pore found; extent is area over bounding-box area.
PORE_MIN_AREA_FRACTION = 0.02
PORE_MIN_EXTENT = 0.50
# Contour simplification used only to draw a candidate pore in the browser, as a
# fraction of its perimeter. Every reported metric is measured from the full
# pixel-accurate contour; this exists so twenty-five boundaries of a thousand
# points each do not dominate the response.
PORE_POLYGON_EPSILON = 0.004

# Tab 4 - filament collapse (ABS platform, Ultimaker 3 Extended)
COLLAPSE_PILLAR_WIDTHS_MM = (10.0, 2.0, 2.0, 2.0, 2.0, 2.0, 10.0)
COLLAPSE_GAPS_MM = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
COLLAPSE_PLATFORM_W_MM = 51.0
COLLAPSE_PLATFORM_H_MM = 10.0
COLLAPSE_FLOOR_MM = 4.0
COLLAPSE_GAP_HEIGHT_MM = 6.0

# Cf convention. "sag" is the Results/Fig. 3d direction of Ingri2024:
#   Cf = A_sag / A_max * 100  ->  flat bridge 0 %, full collapse 100 %.
# "bridge" is the complementary Methods-section direction, kept so a stored run
# can be re-displayed either way without re-analysing the image.
CF_CONVENTION = "sag"
CF_CONVENTION_LABELS = {
    "sag": "Cf = A_sag / A_max x 100 (0 % = flat bridge, 100 % = collapsed)",
    "bridge": "Cf = (1 - A_sag / A_max) x 100 (100 % = flat bridge, 0 % = collapsed)",
}

# Default HSV ranges (OpenCV: H 0-179, S 0-255, V 0-255).
# Pillars are pale red ABS; the filament carries a contrasting dye.
HSV_DEFAULTS = {
    "pillar": {"lo": [0, 30, 120], "hi": [15, 140, 255]},
    "pillar2": {"lo": [165, 30, 120], "hi": [179, 140, 255]},  # red wraps around H=0
    "filament": {"lo": [90, 60, 40], "hi": [140, 255, 255]},
}
