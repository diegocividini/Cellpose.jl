#!/usr/bin/env python3
"""
run_python.py - Segmentazione con Cellpose Python v4.1.1 (modello cpsam)
Genera maschere e metriche descrittive per il benchmark vs Julia.
"""
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from skimage import io as skio

# === CONFIGURAZIONE ===
IMG_DIR = Path("dataset/images")
OUT_DIR = Path("masks_python")
RESULTS_CSV = Path("results/results_python.csv")
MODEL_TYPE = "cpsam"          # Modello default in Cellpose v4.1.1
DIAMETER = 0.0                # 0.0 = auto-estimate
FLOW_THRESH = 0.4
CELLPROB_THRESH = 0.0
USE_GPU = False               # True se hai CUDA configurato
# [cyto, nuclei] = 0,0 = usa tutti i canali (cpsam)
CHANNELS = [0, 0]
# =====================


def setup():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)

    # Import Cellpose v4.1.1 API
    try:
        from cellpose import models
        # v4.1.1: usa CellposeModel con model_type o pretrained_model
        if MODEL_TYPE in ["cpsam", "cyto3", "nuclei"]:
            model = models.CellposeModel(gpu=USE_GPU, model_type=MODEL_TYPE)
        else:
            # Se MODEL_TYPE è un percorso a file .onnx o cartella
            model = models.CellposeModel(
                gpu=USE_GPU, pretrained_model=MODEL_TYPE)
        print(f"✅ Modello Cellpose caricato: {MODEL_TYPE} (GPU={USE_GPU})")
        return model
    except ImportError:
        print("❌ Errore: cellpose non installato. Esegui: pip install 'cellpose>=4.1.1'")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Errore nel caricamento del modello: {e}")
        sys.exit(1)


def preprocess_image(img):
    """Assicura formato (H, W, C) per Cellpose"""
    if img.ndim == 2:
        # Grayscale → fake RGB (cpsam accetta anche grayscale, ma meglio esplicito)
        return np.stack([img, img, img], axis=-1)
    elif img.ndim == 3:
        if img.shape[0] in [1, 3] and img.shape[0] < img.shape[1]:
            # (C, H, W) → (H, W, C)
            return np.moveaxis(img, 0, -1)
        return img  # Già (H, W, C)
    else:
        raise ValueError(
            f"Immagine con dimensione non supportata: {img.shape}")


def compute_metrics(mask):
    """Calcola metriche descrittive per una maschera"""
    n_cells = len(np.unique(mask)) - 1  # Escludi background (0)
    if n_cells > 0:
        areas = [np.sum(mask == i) for i in range(1, n_cells + 1)]
        return {
            "n_cells": n_cells,
            "mean_area": float(np.mean(areas)),
            "median_area": float(np.median(areas)),
            "std_area": float(np.std(areas)),
            "min_area": int(np.min(areas)),
            "max_area": int(np.max(areas))
        }
    return {
        "n_cells": 0, "mean_area": 0, "median_area": 0,
        "std_area": 0, "min_area": 0, "max_area": 0
    }


def main():
    model = setup()
    results = []

    for img_path in sorted(IMG_DIR.glob("*")):
        if img_path.suffix.lower() not in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
            continue

        print(f"🔄 Processing: {img_path.name}")

        # Carica immagine
        img = skio.imread(str(img_path))
        img_preprocessed = preprocess_image(img)

        # Segmentazione con Cellpose v4.1.1
        try:
            masks, flows, styles = model.eval(
                [img_preprocessed],
                diameter=DIAMETER if DIAMETER > 0 else None,  # None = auto
                flow_threshold=FLOW_THRESH,
                cellprob_threshold=CELLPROB_THRESH,
                channels=CHANNELS,
                augment=False,  # Disabilita augment per riproducibilità
                batch_size=8    # Ottimizza per CPU/GPU
            )
            mask = masks[0]
        except Exception as e:
            print(f"⚠️ Errore in {img_path.name}: {e}")
            continue

        # Salva maschera (16-bit per preservare ID)
        mask_fname = f"{img_path.stem}.png"
        skio.imsave(str(OUT_DIR / mask_fname), mask.astype(np.uint16))

        # Calcola e salva metriche
        metrics = compute_metrics(mask)
        metrics["image"] = img_path.name
        metrics["shape_H"] = img.shape[0]
        metrics["shape_W"] = img.shape[1]
        results.append(metrics)
        print(f"✅ {img_path.name}: {metrics['n_cells']} cells, IoU-ready")

    # Salva risultati CSV
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"\n📊 Risultati Python salvati in {RESULTS_CSV}")
    if not df.empty:
        print(
            f"📈 Stats: {df['n_cells'].mean():.1f} ± {df['n_cells'].std():.1f} cells/image")


if __name__ == "__main__":
    main()
