#!/usr/bin/env python3
import argparse
import json
import os
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import numpy as np
import cv2


def _try_import_openslide():
    try:
        import openslide  # type: ignore
        return openslide
    except Exception:
        return None


@dataclass
class Loaded:
    img_bgr: np.ndarray
    w: int
    h: int
    full_w: int
    full_h: int
    mpp: Optional[float]
    path: str
    is_wsi: bool


def load_any(path: str, max_dim: int = 12000) -> Loaded:
    ext = os.path.splitext(path)[1].lower()
    openslide = _try_import_openslide()

    if ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Could not read image: {path}")
        h, w = img.shape[:2]
        if max(h, w) > max_dim:
            s = max_dim / float(max(h, w))
            img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
            h, w = img.shape[:2]
        return Loaded(img, w, h, w, h, None, path, False)

    if ext == ".svs":
        if openslide is None:
            raise RuntimeError(
                "OpenSlide is not available. On macOS: brew install openslide && pip install openslide-python"
            )

        slide = openslide.OpenSlide(path)
        full_w, full_h = slide.dimensions

        # pick a level that is not too large
        best_level = slide.level_count - 1
        for lvl in range(slide.level_count):
            lw, lh = slide.level_dimensions[lvl]
            if max(lw, lh) <= max_dim:
                best_level = lvl
                break

        lw, lh = slide.level_dimensions[best_level]
        region = slide.read_region((0, 0), best_level, (lw, lh)).convert("RGB")
        img_rgb = np.array(region)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        mpp = None
        try:
            mx = slide.properties.get("openslide.mpp-x", None)
            my = slide.properties.get("openslide.mpp-y", None)
            if mx and my:
                mpp = (float(mx) + float(my)) / 2.0
        except Exception:
            pass

        return Loaded(img_bgr, lw, lh, full_w, full_h, mpp, path, True)

    raise ValueError(f"Unsupported file extension: {ext}")


def remove_green_overlay(img_bgr: np.ndarray) -> np.ndarray:
    # masks common bright green overlays
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([35, 40, 40], dtype=np.uint8)
    upper = np.array([95, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    out = img_bgr.copy()
    out[mask > 0] = (255, 255, 255)
    return out


def preprocess(img_bgr: np.ndarray) -> np.ndarray:
    img = remove_green_overlay(img_bgr)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(g)


def estimate_affine_orb(moving_gray: np.ndarray, fixed_gray: np.ndarray) -> Optional[np.ndarray]:
    orb = cv2.ORB_create(8000)
    k1, d1 = orb.detectAndCompute(moving_gray, None)
    k2, d2 = orb.detectAndCompute(fixed_gray, None)

    if d1 is None or d2 is None or len(k1) < 30 or len(k2) < 30:
        return None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(d1, d2, k=2)

    good = []
    for pair in matches:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) < 15:
        return None

    pts_m = np.float32([k1[m.queryIdx].pt for m in good])
    pts_f = np.float32([k2[m.trainIdx].pt for m in good])

    A, inliers = cv2.estimateAffinePartial2D(
        pts_m,
        pts_f,
        method=cv2.RANSAC,
        ransacReprojThreshold=5.0,
        maxIters=8000,
        confidence=0.995,
    )
    return A.astype(np.float64) if A is not None else None


def to_homography(A2x3: np.ndarray) -> np.ndarray:
    H = np.eye(3, dtype=np.float64)
    H[:2, :3] = A2x3
    return H


def apply_homography(H: np.ndarray, pts_xy: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts_xy, dtype=np.float64)
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    ph = np.concatenate([pts, ones], axis=1)
    out = (H @ ph.T).T
    out_xy = out[:, :2] / np.clip(out[:, 2:3], 1e-12, None)
    return out_xy


def scale_matrix(sx: float, sy: float) -> np.ndarray:
    return np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], dtype=np.float64)


def transform_qcf_full(
    qcf_in: Dict[str, Any],
    H_full: np.ndarray,
    fixed_name: str,
    fixed_full_w: int,
    fixed_full_h: int,
    fixed_mpp: Optional[float],
) -> Dict[str, Any]:
    out = json.loads(json.dumps(qcf_in))

    out.setdefault("fileContainer", {})
    out["fileContainer"]["FileName"] = os.path.basename(fixed_name)
    out["fileContainer"]["Width"] = str(int(fixed_full_w))
    out["fileContainer"]["Height"] = str(int(fixed_full_h))
    if fixed_mpp is not None:
        out["fileContainer"]["MicronPerPixels"] = str(float(fixed_mpp))

    rois = out.get("ROIContainer", {}).get("ROIs", [])
    for roi in rois:
        coords = roi.get("ROICoordinate", [])
        if isinstance(coords, list) and len(coords) >= 3:
            pts = np.array(coords, dtype=np.float64)
            roi["ROICoordinate"] = apply_homography(H_full, pts).tolist()

        holes = roi.get("Holes", None)
        if isinstance(holes, list):
            new_holes = []
            for hole in holes:
                if isinstance(hole, list) and len(hole) >= 3:
                    hp = np.array(hole, dtype=np.float64)
                    new_holes.append(apply_homography(H_full, hp).tolist())
                else:
                    new_holes.append(hole)
            roi["Holes"] = new_holes

    return out


def draw_rois_on_thumb(fixed_thumb_bgr: np.ndarray, rois_thumb: List[np.ndarray]) -> np.ndarray:
    out = fixed_thumb_bgr.copy()
    for pts in rois_thumb:
        pts_i = np.round(pts).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(out, [pts_i], isClosed=True, color=(0, 255, 0), thickness=2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--moving_img", required=True, help="WSI/PNG where the ROIs were created (moving).")
    ap.add_argument("--fixed_img", required=True, help="WSI/PNG target reference (fixed).")
    ap.add_argument("--qcf_in", required=True, help="Input QCF/JSON (ROIs in moving full-res coordinates).")
    ap.add_argument("--qcf_out", required=True, help="Output QCF/JSON (ROIs in fixed full-res coordinates).")
    ap.add_argument("--qc_out", required=True, help="QC overlay PNG (fixed thumbnail + ROIs).")
    ap.add_argument("--max_dim", type=int, default=12000, help="Max thumbnail size used for registration.")
    args = ap.parse_args()

    with open(args.qcf_in, "r", encoding="utf-8") as f:
        qcf_in = json.load(f)

    fc = qcf_in.get("fileContainer", {})
    moving_full_w = int(float(fc.get("Width")))
    moving_full_h = int(float(fc.get("Height")))

    moving = load_any(args.moving_img, max_dim=args.max_dim)
    fixed = load_any(args.fixed_img, max_dim=args.max_dim)

    fixed_full_w = fixed.full_w
    fixed_full_h = fixed.full_h

    mov_gray = preprocess(moving.img_bgr)
    fix_gray = preprocess(fixed.img_bgr)

    A = estimate_affine_orb(mov_gray, fix_gray)
    if A is None:
        raise RuntimeError("Registration failed (not enough stable matches).")

    H_thumb = to_homography(A)  # moving_thumb -> fixed_thumb

    Sm = scale_matrix(moving.w / float(moving_full_w), moving.h / float(moving_full_h))      # moving_full -> moving_thumb
    Sf_inv = scale_matrix(fixed_full_w / float(fixed.w), fixed_full_h / float(fixed.h))     # fixed_thumb -> fixed_full

    H_full = Sf_inv @ H_thumb @ Sm  # moving_full -> fixed_full

    qcf_out = transform_qcf_full(qcf_in, H_full, fixed.path, fixed_full_w, fixed_full_h, fixed.mpp)
    with open(args.qcf_out, "w", encoding="utf-8") as f:
        json.dump(qcf_out, f, indent=2)

    rois_thumb = []
    for roi in qcf_in.get("ROIContainer", {}).get("ROIs", []):
        coords = roi.get("ROICoordinate", [])
        if isinstance(coords, list) and len(coords) >= 3:
            pts_full = np.array(coords, dtype=np.float64)
            pts_mthumb = apply_homography(Sm, pts_full)
            pts_fthumb = apply_homography(H_thumb, pts_mthumb)
            rois_thumb.append(pts_fthumb)

    qc_img = draw_rois_on_thumb(fixed.img_bgr, rois_thumb)
    cv2.imwrite(args.qc_out, qc_img)

    print("[OK] Wrote:", args.qcf_out)
    print("[OK] Wrote:", args.qc_out)
    print(f"[INFO] moving_full={moving_full_w}x{moving_full_h} | moving_thumb={moving.w}x{moving.h}")
    print(f"[INFO] fixed_full={fixed_full_w}x{fixed_full_h} | fixed_thumb={fixed.w}x{fixed.h}")


if __name__ == "__main__":
    main()
