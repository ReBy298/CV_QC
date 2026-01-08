#!/usr/bin/env python3
"""
svs_to_png.py

Convert .svs (Whole Slide Image) files into small PNG previews.
- Uses OpenSlide if available.
- Saves a thumbnail (default max 4000 px on the longest side).
"""

import argparse
import os
from pathlib import Path

def try_import_openslide():
    try:
        import openslide  # type: ignore
        return openslide
    except Exception:
        return None

def ensure_outdir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

def save_pil_png(img, out_path: Path) -> None:
    # Lazy import to avoid needing PIL when only copying files
    from PIL import Image
    if not isinstance(img, Image.Image):
        raise TypeError("Expected PIL Image")
    img.save(str(out_path), format="PNG", optimize=True)

def convert_svs_to_png(openslide_mod, svs_path: Path, out_dir: Path, max_dim: int, also_associated: bool) -> None:
    from PIL import Image

    slide = openslide_mod.OpenSlide(str(svs_path))

    # Main thumbnail
    thumb = slide.get_thumbnail((max_dim, max_dim)).convert("RGB")
    out_main = out_dir / (svs_path.stem + ".png")
    thumb.save(str(out_main), format="PNG", optimize=True)
    print(f"[OK] SVS -> PNG: {svs_path.name} -> {out_main.name}")

    if also_associated:
        # Common associated images: "macro" and "label" (if present)
        for key in ["macro", "label"]:
            if key in slide.associated_images:
                assoc = slide.associated_images[key].convert("RGB")
                out_assoc = out_dir / f"{svs_path.stem}.{key}.png"
                assoc.save(str(out_assoc), format="PNG", optimize=True)
                print(f"[OK] Associated '{key}': {out_assoc.name}")

def convert_regular_image_to_png(img_path: Path, out_dir: Path, max_dim: int) -> None:
    from PIL import Image

    img = Image.open(str(img_path)).convert("RGB")

    # Resize if huge (keep aspect ratio)
    w, h = img.size
    scale = min(1.0, max_dim / float(max(w, h)))
    if scale < 1.0:
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        img = img.resize((new_w, new_h), Image.LANCZOS)

    out_path = out_dir / (img_path.stem + ".png")
    img.save(str(out_path), format="PNG", optimize=True)
    print(f"[OK] IMG -> PNG: {img_path.name} -> {out_path.name}")

def main():
    parser = argparse.ArgumentParser(description="Convert SVS/images to PNG previews.")
    parser.add_argument("inputs", nargs="+", help="Paths to .svs / .png / .jpg files")
    parser.add_argument("--out", default="out_png", help="Output folder (default: out_png)")
    parser.add_argument("--max-dim", type=int, default=4000, help="Max size for the longest side (default: 4000)")
    parser.add_argument("--associated", action="store_true", help="Also export associated images (macro/label) for SVS")
    args = parser.parse_args()

    out_dir = Path(args.out)
    ensure_outdir(out_dir)

    openslide_mod = try_import_openslide()

    for inp in args.inputs:
        p = Path(inp)
        if not p.exists():
            print(f"[SKIP] Not found: {p}")
            continue

        ext = p.suffix.lower()

        # Handle ".svs.png" as normal image
        if p.name.lower().endswith(".svs.png"):
            convert_regular_image_to_png(p, out_dir, args.max_dim)
            continue

        if ext == ".svs":
            if openslide_mod is None:
                print("[ERROR] OpenSlide not available. Install it to read .svs files.")
                print("        macOS (brew): brew install openslide")
                print("        python: pip install openslide-python pillow")
                continue
            convert_svs_to_png(openslide_mod, p, out_dir, args.max_dim, args.associated)
        elif ext in [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]:
            convert_regular_image_to_png(p, out_dir, args.max_dim)
        else:
            print(f"[SKIP] Unsupported: {p.name}")

if __name__ == "__main__":
    main()
