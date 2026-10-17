"""
AuraSync — Visionary Agent
Generate vanilla CLIP catalog embeddings
---------------------------------------------------------------------------
Generates catalog_emb_clip.npy — the vanilla-CLIP counterpart to
catalog_emb_marqo_siglip.npy — so colour_regression_diagnostic.py can run
its direct side-by-side comparison instead of the SigLIP-only fallback.

IMPORTANT: set CLIP_MODEL / CLIP_PRETRAINED below to whatever checkpoint
you actually used to get the original 0.502 baseline. If you don't
remember, "ViT-B-32" / "openai" is the most common default for a "vanilla
CLIP" baseline and a reasonable choice if you didn't deliberately pick a
different one — but a mismatched checkpoint means this comparison measures
the wrong thing.

Mirrors your existing image-embedding pipeline conventions:
    - reads catalog_metadata.csv for article_id order
    - image path: <DATA_DIR>/images_resized/<first-3-digits>/<10-digit
      zero-padded article_id>.jpg
    - runs on MPS if available, else CPU
    - output row order matches catalog_metadata.csv exactly, so it plugs
      directly into align() in colour_regression_diagnostic.py

Requires open_clip (should already be installed, since it's used for the
Marqo-FashionSigLIP pipeline):
    pip install open_clip_torch --break-system-packages   # if needed

Run locally:
    python generate_clip_embeddings.py

Override the data directory with the AURASYNC_DATA_DIR env var.
Override LIMIT below for a quick smoke test before committing to a full run.
"""

import os
import time
import numpy as np
import pandas as pd
from PIL import Image
import torch
import open_clip

DATA_DIR = os.environ.get("AURASYNC_DATA_DIR", "/Users/mac/Downloads")

METADATA_PATH = os.path.join(DATA_DIR, "catalog_metadata.csv")
IMAGES_DIR = os.path.join(DATA_DIR, "images_resized")
OUT_PATH = os.path.join(DATA_DIR, "catalog_emb_clip.npy")
CHECKPOINT_PATH = os.path.join(DATA_DIR, "catalog_emb_clip.checkpoint.npz")

CLIP_MODEL = "ViT-B-32"
CLIP_PRETRAINED = "openai"

BATCH_SIZE = 64
LIMIT = None          # set to e.g. 500 for a quick smoke test; None = full catalog
SAVE_EVERY_N_BATCHES = 50  # checkpoint frequency, in case of interruption


def image_path_for(article_id: str) -> str:
    article_id = article_id.zfill(10)
    return os.path.join(IMAGES_DIR, article_id[:3], f"{article_id}.jpg")


def load_metadata():
    meta = pd.read_csv(METADATA_PATH, dtype={"article_id": str})
    meta["article_id"] = meta["article_id"].str.zfill(10)
    if LIMIT is not None:
        meta = meta.iloc[:LIMIT].reset_index(drop=True)
    return meta


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    meta = load_metadata()
    n = len(meta)
    device = get_device()
    print(f"Loading {CLIP_MODEL} ({CLIP_PRETRAINED}) on {device} for {n} articles...")

    model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL, pretrained=CLIP_PRETRAINED
    )
    model = model.to(device).eval()

    embed_dim = model.visual.output_dim
    embeddings = np.zeros((n, embed_dim), dtype=np.float32)
    done_mask = np.zeros(n, dtype=bool)
    missing = []

    start_batch = 0
    if os.path.exists(CHECKPOINT_PATH):
        ckpt = np.load(CHECKPOINT_PATH)
        if ckpt["embeddings"].shape == embeddings.shape:
            embeddings = ckpt["embeddings"]
            done_mask = ckpt["done_mask"]
            start_batch = int(done_mask.sum() // BATCH_SIZE)
            print(f"Resuming from checkpoint: {done_mask.sum()}/{n} already done.")

    t0 = time.time()
    n_batches = (n + BATCH_SIZE - 1) // BATCH_SIZE

    for b in range(start_batch, n_batches):
        lo, hi = b * BATCH_SIZE, min((b + 1) * BATCH_SIZE, n)
        if done_mask[lo:hi].all():
            continue

        batch_imgs, batch_positions = [], []
        for i in range(lo, hi):
            aid = meta.iloc[i]["article_id"]
            path = image_path_for(aid)
            try:
                img = Image.open(path).convert("RGB")
                batch_imgs.append(preprocess(img))
                batch_positions.append(i)
            except (FileNotFoundError, OSError):
                missing.append(aid)

        if batch_imgs:
            batch_tensor = torch.stack(batch_imgs).to(device)
            with torch.no_grad():
                feats = model.encode_image(batch_tensor)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            feats = feats.cpu().numpy()
            for pos, feat in zip(batch_positions, feats):
                embeddings[pos] = feat
                done_mask[pos] = True

        if b % SAVE_EVERY_N_BATCHES == 0 or b == n_batches - 1:
            np.savez(CHECKPOINT_PATH, embeddings=embeddings, done_mask=done_mask)
            elapsed = time.time() - t0
            processed = done_mask.sum()
            rate = processed / elapsed if elapsed > 0 else 0
            eta_min = (n - processed) / rate / 60 if rate > 0 else float("inf")
            print(
                f"  batch {b+1}/{n_batches} | {processed}/{n} done | "
                f"{elapsed:.0f}s elapsed | ETA {eta_min:.1f} min | "
                f"{len(missing)} missing images so far"
            )

    np.save(OUT_PATH, embeddings)
    print(f"\nSaved {OUT_PATH} — shape {embeddings.shape}")
    if missing:
        print(f"{len(missing)} images were missing on disk (embedded as zero vectors).")
        print(f"First few missing article_ids: {missing[:10]}")
    print(
        "\nDone. Set CLIP_EMB_PATH in colour_regression_diagnostic.py to this "
        "file (it already defaults to this exact path) and rerun for the "
        "full side-by-side comparison."
    )


if __name__ == "__main__":
    main()
