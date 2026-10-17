"""
Fast, scalable dominant-color extraction for AuraSync's Visionary agent.

Replaces the original per-image sklearn.cluster.KMeans(n_init=5) approach
(which never got past ~3 images before being killed) with:
  - a downsampled thumbnail before any pixel-level work (biggest single win)
  - a single vectorized histogram-mode pass instead of iterative clustering
  - multiprocessing across CPU cores (this is a CPU-bound, embarrassingly
    parallel task -- no GPU needed)
  - resumable, atomic checkpointing keyed on article_id (not position),
    so a killed/restarted run is safe

Usage:
    pip install pillow numpy pandas scikit-image tqdm
    python extract_colors.py --images-root /path/to/images_resized --out color_features.csv
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Tuple

import numpy as np
import pandas as pd
from PIL import Image
from skimage.filters import threshold_otsu
from tqdm import tqdm

ARTICLE_ID_RE = re.compile(r"^(\d{10})\.jpg$")

try:
    _RESAMPLE = Image.Resampling.BILINEAR  # Pillow >= 9.1
except AttributeError:  # pragma: no cover - older Pillow
    _RESAMPLE = Image.BILINEAR


def find_images(root: str) -> pd.DataFrame:
    """Index images and sort deterministically by article_id.

    Deterministic ordering matters: glob.glob's traversal order is not
    guaranteed to be stable across runs/filesystems, and it's good hygiene
    for any downstream resume logic to not depend on incidental ordering.
    """
    paths = glob.glob(os.path.join(root, "**", "*.jpg"), recursive=True)
    records = []
    for p in paths:
        m = ARTICLE_ID_RE.match(os.path.basename(p))
        if m:
            records.append({"article_id": int(m.group(1)), "filepath": p})
    df = pd.DataFrame(records)
    return df.sort_values("article_id").reset_index(drop=True)


def _dominant_color_fast(pixels: np.ndarray, bin_size: int = 16) -> np.ndarray:
    """Vectorized dominant-color estimate via coarse histogram binning.

    O(n), single pass, no iterative refinement -- this is what replaces the
    per-image KMeans(n_init=5) call. Bins RGB space into (256/bin_size)^3
    buckets, finds the most populated bucket, and returns the mean color of
    the pixels in it. That's "mode of the most common color region," not a
    blended average across the whole image, so it still behaves reasonably
    on patterned garments (stripes, prints) rather than muddying to gray.
    """
    q = (pixels // bin_size).astype(np.int32)
    n_bins = 256 // bin_size
    keys = q[:, 0] * n_bins * n_bins + q[:, 1] * n_bins + q[:, 2]
    values, counts = np.unique(keys, return_counts=True)
    mode_key = values[np.argmax(counts)]
    return pixels[keys == mode_key].mean(axis=0)


def dominant_color(
    filepath: str, thumb_size: int = 64, bin_size: int = 16
) -> Tuple[Tuple[int, int, int], str, float]:
    """Otsu-segment the garment from its background, then extract its
    dominant color from a downsampled thumbnail.

    thumb_size=64 caps every image at <=4096 pixels before any per-pixel
    work happens -- the single biggest lever on runtime. A typical H&M
    product photo has 10-100x more pixels than that, and none of the extra
    resolution changes the answer to "what color is this garment."
    """
    img = Image.open(filepath).convert("RGB")
    img.thumbnail((thumb_size, thumb_size), _RESAMPLE)
    arr = np.asarray(img, dtype=np.float32)

    gray = arr.mean(axis=2)
    thresh = threshold_otsu(gray)
    fg_mask = gray < thresh
    fg_pixels = arr[fg_mask]

    if fg_pixels.shape[0] < 8:
        # segmentation degenerated (near-empty or near-full mask) -- fall
        # back to the whole thumbnail instead of crashing or returning junk
        fg_pixels = arr.reshape(-1, 3)
        fg_mask = np.ones_like(fg_mask)

    rgb = _dominant_color_fast(fg_pixels, bin_size=bin_size).round().astype(int)
    hex_code = "#{:02x}{:02x}{:02x}".format(*rgb.tolist())
    return tuple(rgb.tolist()), hex_code, float(fg_mask.mean())


def _process_one(task: Tuple[int, str, int, int]) -> dict:
    article_id, filepath, thumb_size, bin_size = task
    try:
        rgb, hexcode, fg_pct = dominant_color(filepath, thumb_size, bin_size)
        return {
            "article_id": article_id,
            "filepath": filepath,
            "dominant_rgb": rgb,
            "dominant_hex": hexcode,
            "foreground_pct": round(fg_pct * 100, 1),
            "error": None,
        }
    except Exception as e:
        # deliberately broad: one bad image must never take down a
        # 105k-image batch job
        return {
            "article_id": article_id,
            "filepath": filepath,
            "dominant_rgb": None,
            "dominant_hex": None,
            "foreground_pct": None,
            "error": str(e),
        }


def _flush_checkpoint(rows: list, out_csv: str) -> None:
    new_df = pd.DataFrame(rows)
    if os.path.exists(out_csv):
        combined = pd.concat([pd.read_csv(out_csv), new_df], ignore_index=True)
    else:
        combined = new_df
    tmp_path = f"{out_csv}.tmp"
    combined.to_csv(tmp_path, index=False)
    os.replace(tmp_path, out_csv)  # atomic on POSIX and Windows


def extract_colors(
    images_root: str,
    out_csv: str,
    n_workers: int | None = None,
    checkpoint_every: int = 2000,
    thumb_size: int = 64,
    bin_size: int = 16,
) -> pd.DataFrame:
    n_workers = n_workers or (os.cpu_count() or 4)
    idx = find_images(images_root)
    logging.info("Indexed %d images under %s", len(idx), images_root)
    if idx.empty:
        raise SystemExit(
            f"No .jpg files found under {images_root!r} matching the "
            f"<10-digit-article-id>.jpg pattern -- check --images-root."
        )

    done_ids: set = set()
    if os.path.exists(out_csv):
        done_ids = set(pd.read_csv(out_csv, usecols=["article_id"])["article_id"])
        logging.info("Resuming: %d/%d already processed", len(done_ids), len(idx))

    todo = idx[~idx["article_id"].isin(done_ids)]
    if todo.empty:
        logging.info("Nothing left to process -- %s is already complete.", out_csv)
        return pd.read_csv(out_csv)

    tasks = [
        (aid, fp, thumb_size, bin_size)
        for aid, fp in zip(todo["article_id"], todo["filepath"])
    ]
    buffer: list = []

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_process_one, t): t for t in tasks}
        for n_done, fut in enumerate(
            tqdm(as_completed(futures), total=len(futures), desc="colors"), start=1
        ):
            buffer.append(fut.result())
            if n_done % checkpoint_every == 0:
                _flush_checkpoint(buffer, out_csv)
                buffer.clear()

    if buffer:
        _flush_checkpoint(buffer, out_csv)

    result = pd.read_csv(out_csv)
    n_failed = int(result["error"].notna().sum())
    if n_failed:
        logging.warning("%d/%d images failed -- see the 'error' column", n_failed, len(result))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract dominant garment colors at scale.")
    parser.add_argument("--images-root", required=True)
    parser.add_argument("--out", default="color_features.csv")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    parser.add_argument("--checkpoint-every", type=int, default=2000)
    parser.add_argument("--thumb-size", type=int, default=64)
    parser.add_argument("--bin-size", type=int, default=16)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    extract_colors(
        args.images_root,
        args.out,
        n_workers=args.workers,
        checkpoint_every=args.checkpoint_every,
        thumb_size=args.thumb_size,
        bin_size=args.bin_size,
    )


if __name__ == "__main__":
    main()
