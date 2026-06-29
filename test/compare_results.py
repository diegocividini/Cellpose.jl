#!/usr/bin/env python3
"""
compare_results.py - Confronto quantitativo tra maschere Python e Julia
Genera:
  - Metriche di accordo (Binary IoU, Pixel Accuracy, F1)
  - Difference map visiva
  - Tabella LaTeX pronta per la tesi
"""
import os
import numpy as np
import pandas as pd
from pathlib import Path
from skimage import io, metrics, measure
import matplotlib.pyplot as plt

# === CONFIGURAZIONE ===
device = "cpu"  # Cambia in "gpu" quando farai il confronto della versione GPU
PY_CSV = Path(f"results/results_python_{device}.csv")
JL_CSV = Path(f"results/results_julia_{device}.csv")
PY_MASKS_DIR = Path(f"masks_python_{device}")
JL_MASKS_DIR = Path(f"masks_julia_{device}")
OUT_FIGURES = Path(f"results/figures_comparison_{device}")
OUT_TABLE_LATEX = Path(f"results/comparison_table_{device}.tex")
OUT_SUMMARY = Path(f"results/summary_{device}.txt")
# =====================


def compute_binary_metrics(mask_py, mask_jl):
    """Calcola Binary IoU, Pixel Accuracy, F1 a livello pixel"""
    bin_py = (mask_py > 0).astype(np.uint8)
    bin_jl = (mask_jl > 0).astype(np.uint8)

    # Binary IoU (Jaccard)
    binary_iou = metrics.jaccard_score(bin_py.flatten(), bin_jl.flatten())

    # Pixel Accuracy
    pixel_acc = np.mean(bin_py == bin_jl)

    # Precision, Recall, F1
    tp = np.logical_and(bin_py, bin_jl).sum()
    fp = np.logical_and(bin_jl, ~bin_py).sum()
    fn = np.logical_and(bin_py, ~bin_jl).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / \
        (precision + recall) if (precision + recall) > 0 else 0

    return {
        "binary_iou": binary_iou,
        "pixel_acc": pixel_acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "fp_pixels": int(fp),
        "fn_pixels": int(fn)
    }


def compute_count_metrics(mask_py, mask_jl):
    """Calcola discrepanza nel conteggio cellule"""
    n_py = len(np.unique(mask_py)) - 1
    n_jl = len(np.unique(mask_jl)) - 1
    diff = n_jl - n_py
    diff_pct = (diff / n_py * 100) if n_py > 0 else 0
    return {
        "count_py": n_py,
        "count_jl": n_jl,
        "count_diff": diff,
        "count_diff_pct": diff_pct
    }


def generate_diff_map(mask_py, mask_jl, save_path):
    """Genera difference map: Blue=Agreement, Green=Py-only, Red=Jl-only"""
    bin_py = (mask_py > 0).astype(bool)
    bin_jl = (mask_jl > 0).astype(bool)

    agreement = bin_py & bin_jl
    py_only = bin_py & ~bin_jl
    jl_only = ~bin_py & bin_jl

    # RGB: Blue=Agreement, Green=Py-only, Red=Jl-only, Black=Background
    diff = np.zeros((*mask_py.shape, 3), dtype=np.uint8)
    diff[agreement] = [0, 0, 255]   # Blue
    diff[py_only] = [0, 255, 0]   # Green
    diff[jl_only] = [255, 0, 0]   # Red

    plt.figure(figsize=(8, 8))
    plt.imshow(diff)
    plt.axis('off')
    plt.title("Difference Map (Blue=Agreement, Green=Python-only, Red=Julia-only)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def generate_latex_table(df):
    """Genera tabella LaTeX pronta per l'inclusione nella tesi"""
    # Seleziona colonne rilevanti
    cols = ["image", "count_diff_pct", "binary_iou",
            "pixel_acc", "f1_score", "fp_pixels", "fn_pixels"]
    df_latex = df[cols].copy()

    # Formattazione
    df_latex["binary_iou"] = df_latex["binary_iou"].apply(
        lambda x: f"{x*100:.2f}\\%")
    df_latex["pixel_acc"] = df_latex["pixel_acc"].apply(
        lambda x: f"{x*100:.2f}\\%")
    df_latex["f1_score"] = df_latex["f1_score"].apply(lambda x: f"{x:.3f}")
    df_latex["count_diff_pct"] = df_latex["count_diff_pct"].apply(
        lambda x: f"{x:+.2f}\\%")

    # Genera LaTeX
    latex = df_latex.to_latex(
        index=False,
        column_format="lrrrrrr",
        caption="Confronto quantitativo Python vs Julia (Python come riferimento).",
        label="tab:py_jl_comparison",
        escape=False
    )

    # Pulizia per compilazione LaTeX
    latex = latex.replace("\\_", "_").replace("\\%", "\\%")
    return latex


def main():
    OUT_FIGURES.mkdir(parents=True, exist_ok=True)

    # Carica CSV risultati
    if not PY_CSV.exists() or not JL_CSV.exists():
        print("❌ Errore: Esegui prima run_python.py e run_julia.jl per generare i CSV.")
        return

    df_py = pd.read_csv(PY_CSV)
    df_jl = pd.read_csv(JL_CSV)

    # Merge per immagine
    df = pd.merge(df_py, df_jl, on="image",
                  suffixes=("_py", "_jl"), how="inner")

    results = []
    diff_maps_generated = 0

    for _, row in df.iterrows():
        fname = row["image"]
        mask_py = io.imread(PY_MASKS_DIR / f"{Path(fname).stem}.png")
        mask_jl = io.imread(JL_MASKS_DIR / f"{Path(fname).stem}.png")

        if mask_py.shape != mask_jl.shape:
            print(
                f"⚠️ Skip {fname}: shape mismatch {mask_py.shape} vs {mask_jl.shape}")
            continue

        # Calcola metriche
        binary_metrics = compute_binary_metrics(mask_py, mask_jl)
        count_metrics = compute_count_metrics(mask_py, mask_jl)

        # Unisci risultati
        result = {"image": fname, "shape": mask_py.shape}
        result.update(binary_metrics)
        result.update(count_metrics)
        results.append(result)

        # Genera difference map per le prime 3 immagini (per risparmiare spazio)
        if diff_maps_generated < 3:
            diff_path = OUT_FIGURES / f"diff_{Path(fname).stem}.png"
            generate_diff_map(mask_py, mask_jl, diff_path)
            print(f"🗺️ Difference map saved: {diff_path}")
            diff_maps_generated += 1

        print(
            f"✅ {fname}: IoU={binary_metrics['binary_iou']:.2%}, Δcount={count_metrics['count_diff_pct']:+.2f}%")

    # Salva risultati completi
    df_results = pd.DataFrame(results)
    df_results.to_csv("results/full_comparison.csv", index=False)

    # Genera tabella LaTeX
    latex_table = generate_latex_table(df_results)
    with open(OUT_TABLE_LATEX, "w") as f:
        f.write(latex_table)
    print(f"📄 Tabella LaTeX salvata in {OUT_TABLE_LATEX}")

    # Genera summary testuale
    summary = f"""
=== BENCHMARK SUMMARY: Python vs Julia Cellpose ===
Images compared: {len(df_results)}
Mean Binary IoU: {df_results['binary_iou'].mean():.2%} ± {df_results['binary_iou'].std():.2%}
Mean Pixel Accuracy: {df_results['pixel_acc'].mean():.2%} ± {df_results['pixel_acc'].std():.2%}
Mean F1 Score: {df_results['f1_score'].mean():.3f} ± {df_results['f1_score'].std():.3f}
Mean Cell Count Difference: {df_results['count_diff_pct'].mean():+.2f}% ± {df_results['count_diff_pct'].std():.2f}%
Total False Positive Pixels: {df_results['fp_pixels'].sum():,}
Total False Negative Pixels: {df_results['fn_pixels'].sum():,}
"""
    with open(OUT_SUMMARY, "w") as f:
        f.write(summary)
    print(summary)

    print(f"🎯 Confronto completato. Risultati in results/")


if __name__ == "__main__":
    main()
