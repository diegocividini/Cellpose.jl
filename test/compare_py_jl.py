# compare_py_jl.py
import os
import numpy as np
import pandas as pd
from skimage import io
import matplotlib.pyplot as plt
from pathlib import Path

# === CONFIG ===
PY_DIR = "masks_python"
JL_DIR = "masks_julia"
OUT_FIGURES = "figures_comparison"
OUT_TABLE = "comparison_table.tex"
# ================

os.makedirs(OUT_FIGURES, exist_ok=True)


def compute_metrics(mask_py, mask_jl):
    bin_py = (mask_py > 0).astype(np.uint8)
    bin_jl = (mask_jl > 0).astype(np.uint8)

    tp = np.logical_and(bin_py, bin_jl).sum()
    fp = np.logical_and(bin_jl, ~bin_py).sum()
    fn = np.logical_and(bin_py, ~bin_jl).sum()
    tn = np.logical_and(~bin_py, ~bin_jl).sum()

    pixel_acc = (tp + tn) / (tp + tn + fp + fn)
    binary_iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    n_py = len(np.unique(mask_py)) - 1
    n_jl = len(np.unique(mask_jl)) - 1

    areas_py = [np.sum(mask_py == i)
                for i in range(1, n_py + 1)] if n_py > 0 else []
    areas_jl = [np.sum(mask_jl == i)
                for i in range(1, n_jl + 1)] if n_jl > 0 else []

    return {
        "pixel_acc": pixel_acc,
        "binary_iou": binary_iou,
        "count_diff": n_jl - n_py,
        "count_diff_pct": ((n_jl - n_py) / n_py * 100) if n_py > 0 else 0,
        "mean_area_py": np.mean(areas_py) if areas_py else 0,
        "mean_area_jl": np.mean(areas_jl) if areas_jl else 0,
        "median_area_py": np.median(areas_py) if areas_py else 0,
        "median_area_jl": np.median(areas_jl) if areas_jl else 0,
        "fp_pixels": int(fp),
        "fn_pixels": int(fn)
    }


def generate_diff_map(mask_py, mask_jl, save_path):
    bin_py = (mask_py > 0).astype(bool)
    bin_jl = (mask_jl > 0).astype(bool)

    agreement = bin_py & bin_jl
    py_only = bin_py & ~bin_jl
    jl_only = ~bin_py & bin_jl

    # RGB: Blue=Agreement, Green=Py-only, Red=Jl-only, Black=Background
    diff = np.zeros((*mask_py.shape, 3), dtype=np.uint8)
    diff[agreement] = [0, 0, 255]
    diff[py_only] = [0, 255, 0]
    diff[jl_only] = [255, 0, 0]

    plt.imsave(save_path, diff)


# Esecuzione
rows = []
for fname in sorted(os.listdir(PY_DIR)):
    if not fname.endswith(".png"):
        continue

    m_py = io.imread(os.path.join(PY_DIR, fname))
    m_jl = io.imread(os.path.join(JL_DIR, fname))

    if m_py.shape != m_jl.shape:
        print(f"⚠️ Skip {fname}: shape mismatch")
        continue

    met = compute_metrics(m_py, m_jl)
    met["image"] = fname
    rows.append(met)

    # Genera difference map solo per le prime 3 immagini (per risparmiare spazio)
    if len(rows) <= 3:
        generate_diff_map(m_py, m_jl, os.path.join(
            OUT_FIGURES, f"diff_{fname}"))

df = pd.DataFrame(rows)

# === SALVATAGGIO TABELLA LATEX ===
latex_code = df.to_latex(
    columns=["image", "count_diff_pct", "binary_iou",
             "pixel_acc", "fp_pixels", "fn_pixels"],
    index=False,
    float_format="%.2f",
    caption="Confronto quantitativo Python vs Julia (Python come riferimento).",
    label="tab:py_jl_comparison",
    escape=False
)
# Pulizia formattazione per compilazione LaTeX
latex_code = latex_code.replace("\\_", "_").replace("\\%", "\\%")
with open(OUT_TABLE, "w") as f:
    f.write(latex_code)

print(f"✅ Confronto completato. Tabella LaTeX salvata in {OUT_TABLE}")
print(
    f"📊 Media IoU: {df['binary_iou'].mean():.2%} | Media Pixel Acc: {df['pixel_acc'].mean():.2%}")
