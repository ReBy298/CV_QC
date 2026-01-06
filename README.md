# Slide Offset Fix (CV Registration)

Small Python script to reduce the **global “offset”** between two scans of the same slide by:

- registering **moving** (annotation scan / RefScan) → **fixed** (reference scan / Decoverslipped 5X)
- applying the resulting transform to ROI polygon points stored in a **QCF-like JSON**
- exporting a **corrected QCF** + a **QC overlay PNG** for quick verification

---

## What it does

- Reads a thumbnail from each `.svs` (fast)
- Estimates a robust **affine transform** (translation + rotation + scale) using **ORB + RANSAC**
- Converts ROI coordinates:
  - **moving full-res → moving thumbnail → fixed thumbnail → fixed full-res**
- Writes:
  - `corrected.qcf.json` (ROIs in **fixed** coordinate space)
  - `qc_overlay.png` (fixed thumbnail with ROIs drawn in green)

Optional (recommended for box-shift cases):

- `--autofit_box` shifts/expands the **existing scan box** (RECTANGLE) so it covers the other ROIs.
  - This is meant to **fix the symptom** (box missing tissue) in the output annotation.
  - It **does not** recover tissue that was never scanned.

---

## Requirements

- Python 3.9+
- OpenSlide installed on your system (needed for `.svs`)
- Python packages from `requirements.txt`

---

## Setup (macOS)

```bash
# 1) System dependency for SVS
brew install openslide

# 2) Create venv
python3 -m venv .venv
source .venv/bin/activate

# 3) Install deps
pip install --upgrade pip
pip install -r requirements.txt

## Run 

python offset_fix.py \
  --moving_img "F00048061-sansfids-RefScan-2025-12-04_16_03_40.Group.svs" \
  --fixed_img  "F00048061-6XFIDV3-Decoverslipped-5X-2025-12-05_16_49_24.svs" \
  --qcf_in     "F00048061-sansfids-RefScan-2025-12-04_16_03_40.Group..json" \
  --qcf_out    "corrected.qcf.json" \
  --qc_out     "qc_overlay.png" \
  --max_dim 8000 \
  --autofit_box \
  --outside_threshold 0.01 \
  --box_margin 0.03
