# Slide Offset Fix (CV Registration)

Small Python tool to reduce the **global “offset”** between two scans of the same slide by:
- registering **moving** (annotation scan / RefScan) to **fixed** (reference scan / Decoverslipped 5X)
- applying the resulting transform to ROI polygon points stored in a **QCF-like JSON**
- producing a **corrected QCF** and a **QC overlay** image

## What it does
- Reads a thumbnail from each `.svs` (fast)
- Computes a robust **affine transform** (translation + rotation + scale) using **ORB + RANSAC**
- Converts ROI coordinates from **moving full-res → moving thumbnail → fixed thumbnail → fixed full-res**
- Writes:
  - `corrected.qcf.json` (ROIs in fixed coordinate space)
  - `qc_overlay.png` (fixed thumbnail with ROIs drawn in green)

## Requirements
- Python 3.9+
- OpenSlide installed on your system (needed for `.svs`)
- Python packages from `requirements.txt`

## Setup (macOS)
```bash
# Install OpenSlide
brew install openslide

# Create venv
python3 -m venv .venv
source .venv/bin/activate

# Install deps
pip install --upgrade pip
pip install -r requirements.txt

# Run Script
python offset_fix_en.py \
  --moving_img "F00048061-sansfids-RefScan-2025-12-04_16_03_40.Group.svs" \
  --fixed_img  "F00048061-6XFIDV3-Decoverslipped-5X-2025-12-05_16_49_24.svs" \
  --qcf_in     "F00048061-sansfids-RefScan-2025-12-04_16_03_40.Group..json" \
  --qcf_out    "corrected.qcf.json" \
  --qc_out     "qc_overlay.png" \
  --max_dim 8000