#!/usr/bin/env python3
import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import cv2


# -----------------------------
# IO helpers
# -----------------------------
def _try_import_openslide():
    """Try OpenSlide (SVS reader). If it isn't installed, we can still run with PNG/JPG."""
    try:
        import openslide  # type: ignore
        return openslide
    except Exception:
        return None


class Loaded:
    """Small wrapper so I can treat PNG and SVS the same way after loading."""
    def __init__(self, img_bgr, w, h, full_w, full_h, mpp, path, is_wsi):
        self.img_bgr = img_bgr      # thumbnail in BGR (OpenCV)
        self.w = w                  # thumbnail width
        self.h = h                  # thumbnail height
        self.full_w = full_w        # original full-res width (SVS only, else == w)
        self.full_h = full_h        # original full-res height (SVS only, else == h)
        self.mpp = mpp              # microns-per-pixel (if available)
        self.path = path
        self.is_wsi = is_wsi


def load_any(path: str, max_dim: int = 12000) -> Loaded:
    """Load either a normal image or a WSI (.svs). For SVS, I pick a level <= max_dim for fast registration."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    openslide = _try_import_openslide()

    # Normal image path (png/jpg/...)
    if ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Could not read image: {path}")

        h, w = img.shape[:2]
        full_w, full_h = w, h

        # Keep thumbnails manageable (faster ORB)
        if max(h, w) > max_dim:
            s = max_dim / float(max(h, w))
            img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
            h, w = img.shape[:2]

        return Loaded(img, w, h, full_w, full_h, None, path, False)

    # WSI path (svs)
    if ext == ".svs":
        if openslide is None:
            raise RuntimeError(
                "OpenSlide not available. macOS: brew install openslide && pip install openslide-python"
            )

        # If this throws "Unsupported or missing image file", usually the path is wrong or not a real SVS
        try:
            slide = openslide.OpenSlide(path)
        except Exception as e:
            raise RuntimeError(f"OpenSlide failed to open: {path}\n{e}")

        full_w, full_h = slide.dimensions

        # Pick the first level that fits max_dim; otherwise fall back to the smallest
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

        # mpp is optional but nice to keep in output metadata
        mpp = None
        try:
            mx = slide.properties.get("openslide.mpp-x", None)
            my = slide.properties.get("openslide.mpp-y", None)
            if mx and my:
                mpp = (float(mx) + float(my)) / 2.0
        except Exception:
            mpp = None

        return Loaded(img_bgr, lw, lh, full_w, full_h, mpp, path, True)

    raise ValueError(f"Unsupported file extension: {ext}")


# -----------------------------
# Registration helpers
# -----------------------------
def remove_green_overlay(img_bgr: np.ndarray) -> np.ndarray:
    """Some of the QC previews have a bright green overlay. I nuke it so ORB doesn't lock onto it."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([35, 40, 40], dtype=np.uint8)
    upper = np.array([95, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    out = img_bgr.copy()
    out[mask > 0] = (255, 255, 255)
    return out


def preprocess(img_bgr: np.ndarray) -> np.ndarray:
    """Grayscale + CLAHE to make keypoints more stable."""
    img = remove_green_overlay(img_bgr)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(g)


def estimate_affine_orb(moving_gray: np.ndarray, fixed_gray: np.ndarray) -> Optional[np.ndarray]:
    """ORB feature matching + RANSAC -> affine (2x3). Good enough for slide-level alignment."""
    orb = cv2.ORB_create(10000)
    k1, d1 = orb.detectAndCompute(moving_gray, None)
    k2, d2 = orb.detectAndCompute(fixed_gray, None)

    if d1 is None or d2 is None or len(k1) < 50 or len(k2) < 50:
        return None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches_knn = bf.knnMatch(d1, d2, k=2)

    # Lowe ratio test to filter garbage matches
    good = []
    for pair in matches_knn:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) < 20:
        return None

    pts_m = np.float32([k1[m.queryIdx].pt for m in good])
    pts_f = np.float32([k2[m.trainIdx].pt for m in good])

    A, _ = cv2.estimateAffinePartial2D(
        pts_m,
        pts_f,
        method=cv2.RANSAC,
        ransacReprojThreshold=5.0,
        maxIters=8000,
        confidence=0.995,
    )
    if A is None:
        return None
    return A.astype(np.float64)


def to_homography(A2x3: np.ndarray) -> np.ndarray:
    """Convert 2x3 affine -> 3x3 homography (so I can use one apply function everywhere)."""
    H = np.eye(3, dtype=np.float64)
    H[:2, :3] = A2x3
    return H


def scale_matrix(sx: float, sy: float) -> np.ndarray:
    """Homography for scaling between coordinate spaces."""
    return np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], dtype=np.float64)


def apply_homography(H: np.ndarray, pts_xy: np.ndarray) -> np.ndarray:
    """Apply homography to Nx2 points."""
    pts = np.asarray(pts_xy, dtype=np.float64)
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    ph = np.concatenate([pts, ones], axis=1)
    out = (H @ ph.T).T
    return out[:, :2] / np.clip(out[:, 2:3], 1e-12, None)


# -----------------------------
# QCF helpers
# -----------------------------
def transform_qcf_full(
    qcf_in: Dict[str, Any],
    H_full: np.ndarray,
    fixed_name: str,
    fixed_full_w: int,
    fixed_full_h: int,
    fixed_mpp: Optional[float],
) -> Dict[str, Any]:
    """Clone QCF and remap every ROI coordinate into FIXED full-res space."""
    out = json.loads(json.dumps(qcf_in))

    # Update fileContainer so downstream knows this QCF belongs to the fixed slide now
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

    return out


def draw_rois_on_thumb(fixed_thumb_bgr, rois_thumb_with_labels, alpha=0.30):
    """
    rois_thumb_with_labels: List[Tuple[str, np.ndarray]] = [(label, pts_thumb), ...]
    """
    base = fixed_thumb_bgr.copy()
    overlay = base.copy()

    for label, pts in rois_thumb_with_labels:
        if pts.shape[0] < 3:
            continue
        pts_i = np.round(pts).astype(np.int32).reshape((-1, 1, 2))
        lab = (label or "").upper()

        if lab == "RECTANGLE":
            # Scan box: outline only (do NOT fill)
            cv2.polylines(base, [pts_i], True, (0, 255, 0), 4)
        else:
            # Tissue/outside ROIs: filled + outline
            cv2.fillPoly(overlay, [pts_i], (0, 255, 0))
            cv2.polylines(base, [pts_i], True, (0, 255, 0), 2)

    return cv2.addWeighted(overlay, alpha, base, 1 - alpha, 0)



# -----------------------------
# Scan-box symptom fix (shift/expand)
# -----------------------------
def clamp_int(v: float, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(round(v))))


def rect_poly(x0: float, y0: float, x1: float, y1: float) -> List[List[float]]:
    # QCF expects a polygon even for rectangles, so I store 4 points
    return [[float(x0), float(y0)], [float(x1), float(y0)], [float(x1), float(y1)], [float(x0), float(y1)]]


def roi_bbox(coords_list: List[List[float]]) -> Tuple[float, float, float, float]:
    pts = np.asarray(coords_list, dtype=np.float64)
    return float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max())


def bbox_intersection_area(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    iw = max(0.0, ix1 - ix0)
    ih = max(0.0, iy1 - iy0)
    return iw * ih


def bbox_area(a: Tuple[float, float, float, float]) -> float:
    x0, y0, x1, y1 = a
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def bbox_outside_ratio(target: Tuple[float, float, float, float], box: Tuple[float, float, float, float]) -> float:
    """How much of target bbox is outside the current scan box (0 = fully covered)."""
    ta = bbox_area(target)
    if ta <= 0:
        return 0.0
    inter = bbox_intersection_area(target, box)
    return max(0.0, 1.0 - (inter / ta))


def find_scan_box_index(rois: List[Dict[str, Any]]) -> Optional[int]:
    """Try to find the scan box ROI. I prefer Label=='RECTANGLE'; otherwise I pick the largest ROI."""
    rect_ids = []
    for i, r in enumerate(rois):
        if (r.get("Label") or "").upper() == "RECTANGLE":
            rect_ids.append(i)

    def area_of(i: int) -> float:
        pts = np.array(rois[i].get("ROICoordinate", []), dtype=np.float32)
        if pts.shape[0] < 4:
            return -1.0
        return float(abs(cv2.contourArea(pts)))

    if len(rect_ids) == 1:
        return rect_ids[0]
    if len(rect_ids) > 1:
        return max(rect_ids, key=area_of)

    best_i, best_a = None, -1.0
    for i, r in enumerate(rois):
        pts = np.array(r.get("ROICoordinate", []), dtype=np.float32)
        if pts.shape[0] < 4:
            continue
        a = float(abs(cv2.contourArea(pts)))
        if a > best_a:
            best_i, best_a = i, a
    return best_i


def union_bbox_of_rois(rois: List[Dict[str, Any]], skip_idx: int) -> Optional[Tuple[float, float, float, float]]:
    """Union bbox of all ROIs except the scan box (this is my proxy for 'tissue we care about')."""
    x0, y0 = float("inf"), float("inf")
    x1, y1 = float("-inf"), float("-inf")
    found = False

    for i, roi in enumerate(rois):
        if i == skip_idx:
            continue
        coords = roi.get("ROICoordinate", [])
        if not (isinstance(coords, list) and len(coords) >= 3):
            continue
        bx0, by0, bx1, by1 = roi_bbox(coords)
        x0 = min(x0, bx0)
        y0 = min(y0, by0)
        x1 = max(x1, bx1)
        y1 = max(y1, by1)
        found = True

    if not found:
        return None
    return (x0, y0, x1, y1)


def apply_margin_to_bbox(b: Tuple[float, float, float, float], margin_frac: float) -> Tuple[float, float, float, float]:
    """Pad a bbox by a fraction of its size (keeps things from being too tight)."""
    x0, y0, x1, y1 = b
    w = max(1.0, x1 - x0)
    h = max(1.0, y1 - y0)
    mx = w * margin_frac
    my = h * margin_frac
    return (x0 - mx, y0 - my, x1 + mx, y1 + my)


def clamp_bbox_to_slide(b: Tuple[float, float, float, float], W: int, H: int) -> Tuple[float, float, float, float]:
    """Keep bbox inside slide bounds."""
    x0, y0, x1, y1 = b
    x0 = float(clamp_int(x0, 0, W - 1))
    y0 = float(clamp_int(y0, 0, H - 1))
    x1 = float(clamp_int(x1, 0, W - 1))
    y1 = float(clamp_int(y1, 0, H - 1))
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    return (x0, y0, x1, y1)


def shift_then_expand_box(
    box: Tuple[float, float, float, float],
    target: Tuple[float, float, float, float],
) -> Tuple[float, float, float, float]:
    """
    Main "symptom fix":
    - If target is smaller than box: shift box just enough to include it.
    - If target is bigger than box: expand box to target.
    """
    bx0, by0, bx1, by1 = box
    tx0, ty0, tx1, ty1 = target

    bw = bx1 - bx0
    bh = by1 - by0
    tw = tx1 - tx0
    th = ty1 - ty0

    new_bx0, new_by0, new_bx1, new_by1 = bx0, by0, bx1, by1

    # X direction
    if tw <= bw:
        dx = 0.0
        if tx0 < bx0:
            dx = tx0 - bx0
        elif tx1 > bx1:
            dx = tx1 - bx1
        new_bx0 += dx
        new_bx1 += dx
    else:
        new_bx0 = tx0
        new_bx1 = tx1

    # Y direction
    if th <= bh:
        dy = 0.0
        if ty0 < by0:
            dy = ty0 - by0
        elif ty1 > by1:
            dy = ty1 - by1
        new_by0 += dy
        new_by1 += dy
    else:
        new_by0 = ty0
        new_by1 = ty1

    return (new_bx0, new_by0, new_bx1, new_by1)


def main():
    ap = argparse.ArgumentParser()

    # Moving = where ROIs were created, Fixed = where we want them to land
    ap.add_argument("--moving_img", required=True, help="WSI/PNG where the ROIs were created (moving).")
    ap.add_argument("--fixed_img", required=True, help="WSI/PNG target reference (fixed).")

    ap.add_argument("--qcf_in", required=True, help="Input QCF/JSON (ROIs in moving full-res coordinates).")
    ap.add_argument("--qcf_out", required=True, help="Output QCF/JSON (ROIs in fixed full-res coordinates).")
    ap.add_argument("--qc_out", required=True, help="QC overlay PNG (fixed thumbnail + ROIs).")

    ap.add_argument("--max_dim", type=int, default=12000, help="Max thumbnail size used for registration.")

    # This is the "second fix" you asked for: shift/expand the existing scan box based on the other ROIs
    ap.add_argument(
        "--autofit_box",
        action="store_true",
        help="Shift/expand the scan-box using existing ROIs (fix symptom).",
    )
    ap.add_argument(
        "--box_margin",
        type=float,
        default=0.03,
        help="Margin as a fraction of tissue bbox size (e.g., 0.03 = 3%%).",
    )
    ap.add_argument(
        "--outside_threshold",
        type=float,
        default=0.01,
        help="If target bbox outside existing box is above this, refit the box.",
    )

    args = ap.parse_args()

    # Read QCF JSON (moving full-res coords)
    with open(args.qcf_in, "r", encoding="utf-8") as f:
        qcf_in = json.load(f)

    fc = qcf_in.get("fileContainer", {})
    moving_full_w = int(float(fc.get("Width")))
    moving_full_h = int(float(fc.get("Height")))

    # Load thumbnails (and full dimensions) for both slides
    moving = load_any(args.moving_img, max_dim=args.max_dim)
    fixed = load_any(args.fixed_img, max_dim=args.max_dim)

    fixed_full_w = fixed.full_w
    fixed_full_h = fixed.full_h

    # -----------------------------
    # Optional: adjust scan box in MOVING space before mapping to FIXED
    # -----------------------------
    if args.autofit_box:
        in_rois = qcf_in.get("ROIContainer", {}).get("ROIs", [])
        idx_box = find_scan_box_index(in_rois)

        if idx_box is not None:
            box_coords = in_rois[idx_box].get("ROICoordinate", [])
            if isinstance(box_coords, list) and len(box_coords) >= 4:
                box_bb = roi_bbox(box_coords)

                # union bbox of all other ROIs is my "target"
                target_bb = union_bbox_of_rois(in_rois, skip_idx=idx_box)

                if target_bb is not None:
                    target_bb = apply_margin_to_bbox(target_bb, args.box_margin)
                    target_bb = clamp_bbox_to_slide(target_bb, moving_full_w, moving_full_h)

                    outside_before = bbox_outside_ratio(target_bb, box_bb)

                    if outside_before > args.outside_threshold:
                        new_box_bb = shift_then_expand_box(box_bb, target_bb)
                        new_box_bb = clamp_bbox_to_slide(new_box_bb, moving_full_w, moving_full_h)
                        in_rois[idx_box]["ROICoordinate"] = rect_poly(*new_box_bb)

                        outside_after = bbox_outside_ratio(target_bb, new_box_bb)
                        print(f"[INFO] outside_ratio_before={outside_before:.4f} -> refit (shift_then_expand)")
                        print(f"[INFO] outside_ratio_after={outside_after:.4f}")
                    else:
                        print(f"[INFO] outside_ratio_before={outside_before:.4f} -> no refit")
                else:
                    print("[INFO] No tissue ROIs found (only box?). Skipping autofit.")
            else:
                print("[INFO] Scan box ROI has no valid rectangle coordinates. Skipping autofit.")
        else:
            print("[INFO] No scan box ROI found. Skipping autofit.")

    # -----------------------------
    # Registration on thumbnails
    # -----------------------------
    mov_gray = preprocess(moving.img_bgr)
    fix_gray = preprocess(fixed.img_bgr)

    A = estimate_affine_orb(mov_gray, fix_gray)
    if A is None:
        raise RuntimeError("Registration failed (not enough stable matches).")

    H_thumb = to_homography(A)  # moving_thumb -> fixed_thumb

    # moving_full -> moving_thumb
    Sm = scale_matrix(moving.w / float(moving_full_w), moving.h / float(moving_full_h))

    # fixed_thumb -> fixed_full
    Sf_inv = scale_matrix(fixed_full_w / float(fixed.w), fixed_full_h / float(fixed.h))

    # moving_full -> fixed_full
    H_full = Sf_inv @ H_thumb @ Sm

    # Transform all ROIs (including updated box) into FIXED full-res
    qcf_out = transform_qcf_full(qcf_in, H_full, fixed.path, fixed_full_w, fixed_full_h, fixed.mpp)

    with open(args.qcf_out, "w", encoding="utf-8") as f:
        json.dump(qcf_out, f, indent=2)

    # -----------------------------
    # QC overlay: FIXED full -> FIXED thumb
    # -----------------------------
    Sf = scale_matrix(fixed.w / float(fixed_full_w), fixed.h / float(fixed_full_h))

    rois_thumb_with_labels: List[Tuple[str, np.ndarray]] = []
    for roi in qcf_out.get("ROIContainer", {}).get("ROIs", []):
        coords = roi.get("ROICoordinate", [])
        if isinstance(coords, list) and len(coords) >= 3:
            pts_full = np.array(coords, dtype=np.float64)
            pts_thumb = apply_homography(Sf, pts_full)
            label = roi.get("Label", "")
            rois_thumb_with_labels.append((label, pts_thumb))
    qc_img = draw_rois_on_thumb(fixed.img_bgr, rois_thumb_with_labels)
    cv2.imwrite(args.qc_out, qc_img)

    print("[OK] Wrote:", args.qcf_out)
    print("[OK] Wrote:", args.qc_out)
    print(f"[INFO] moving_full={moving_full_w}x{moving_full_h} | moving_thumb={moving.w}x{moving.h}")
    print(f"[INFO] fixed_full={fixed_full_w}x{fixed_full_h} | fixed_thumb={fixed.w}x{fixed.h}")


if __name__ == "__main__":
    main()
