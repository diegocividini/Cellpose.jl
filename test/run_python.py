# run_python.py
import os
import numpy as np
from cellpose import models, io
from skimage import io as skio
import pandas as pd

# === CONFIG ===
IMG_DIR = "dataset/images"
OUT_DIR = "masks_python"
RESULTS_CSV = "results_python.csv"
MODEL_TYPE = "cyto3"  # o "nuclei" a seconda del dataset
DIAMETER = 30         # adatta al tuo dataset
FLOW_THRESH = 0.4
CELLPROB_THRESH = 0.0
USE_GPU = False       # cambia se vuoi testare GPU
# ================

os.makedirs(OUT_DIR, exist_ok=True)
model = models.Cellpose(gpu=USE_GPU, model_type=MODEL_TYPE)

results = []

for fname in sorted(os.listdir(IMG_DIR)):
    if not fname.endswith((".png", ".jpg", ".tif")):
        continue
    path = os.path.join(IMG_DIR, fname)
    img = skio.imread(path)
    if img.ndim == 2:
        # Cellpose expects channels last
        img = np.stack([img, img, img], axis=-1)

    masks, flows, styles = model.eval(
        [img], diameter=DIAMETER,
        flow_threshold=FLOW_THRESH,
        cellprob_threshold=CELLPROB_THRESH
    )
    mask = masks[0]

    # Salva maschera
    mask_path = os.path.join(OUT_DIR, f"{os.path.splitext(fname)[0]}.png")
    skio.imsave(mask_path, mask.astype(np.uint16))

    # Metriche rapide per debug
    n_cells = len(np.unique(mask)) - 1
    areas = [np.sum(mask == i) for i in range(1, n_cells + 1)]
    results.append({
        "image": fname,
        "n_cells": n_cells,
        "mean_area": np.mean(areas) if areas else 0,
        "median_area": np.median(areas) if areas else 0,
        "std_area": np.std(areas) if areas else 0
    })
    print(f"✅ Python: {fname} -> {n_cells} cells")

pd.DataFrame(results).to_csv(RESULTS_CSV, index=False)
print(f"📊 Risultati Python salvati in {RESULTS_CSV}")
