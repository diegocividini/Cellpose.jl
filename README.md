# Cellpose.jl 🔬

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://github.com/MouseLand/cellpose)
[![Language](https://img.shields.io/badge/language-Julia-9558B2.svg)](https://julialang.org/)

**Cellpose.jl** is an unofficial, native Julia port of the popular [Cellpose](https://github.com/MouseLand/cellpose) anatomical segmentation algorithm, originally developed in Python by Carsen Stringer and Marius Pachitariu.

This package provides a **pure Julia** pipeline for cellular segmentation using exported ONNX models (like the Vision Transformer-based `cpsam`). It completely bypasses Python interoperability overhead (`PyCall`/`PythonCall`), making it ideal for high-performance, native Julia bio-imaging workflows.

## ✨ Key Features

* **No Python Dependency:** The entire pre-processing, inference, and post-processing pipeline is written in native Julia.
* **Multithreaded Tiling:** Automatically divides large biomedical images into overlapping patches (256x256) and processes them in parallel across available CPU cores using `Threads.@threads`.
* **Exact Mathematical Porting:** Replicates Cellpose v4's exact logic:
  * 10% tile overlap.
  * Flat-top window for seamless bilinear blending and stitching.
  * RGB channel formatting and 1-99th percentile normalization.
* **Custom Dynamics Engine:** The core Euler integration for fluid dynamics (`follow_flows`) and spatial gradient mask recovery (`compute_masks`) have been rewritten and optimized for Julia arrays.
* **Hardware Acceleration Ready:** Supports GPU inference on NVIDIA hardware via `ONNXRunTime.jl` with a simple flag toggle.

## 🛠 Installation

Currently, the package is in development. You can run it locally by cloning the repository and instantiating the Julia environment:

```julia
# Open the Julia REPL and type ']' to enter the Pkg manager
pkg> activate .
pkg> instantiate
```

## 🚀 Quick Start

```julia
using Cellpose
using NPZ

# 1. Load your 2D or 3D (RGB) biomedical image
img = npzread("data/my_image.npy")

# 2. Define the path to your exported ONNX model
model_path = "models/cpsam.onnx"

# 3. Run the full segmentation pipeline
# Note: Set use_gpu=true if you are running on an NVIDIA GPU machine
masks = segment(img, model_path, use_gpu=false)

println("Segmentation complete. Found $(maximum(masks)) cells!")
```

## 🧠 How it Works (The Pipeline)

1. Global Normalization: The image is normalized per-channel (1st to 99th percentile) to prevent noise hallucinations in empty areas.

2. Overlapping Tiling: The image is divided into 256x256 patches with a 10% overlap (stride of 230 pixels).

3. Parallel Inference: Each patch is formatted to 3 channels and passed through the ONNX network to predict cell probabilities and spatial gradients (dP).

4. Flat-Top Blending: Output patches are stitched back together using a flat-top window to smoothly blend the overlapping borders without edge amplification.

5. Fluid Dynamics: Pixels are treated as particles and pushed along the predicted vector field using Euler integration until they converge at the cell centers.

6. Quality Control: Small masks (area < 15 pixels) or masks with inconsistent flow fields are automatically rejected.

## 📜 Acknowledgments & Citation

This is an independent port built for educational and research integration purposes. All credit for the original machine learning architecture, pretrained models, and algorithmic concepts goes to the original Cellpose authors.

If you use the Cellpose algorithm in your research, please cite their original papers:

> Stringer, C., Wang, T., Michaelos, M., & Pachitariu, M. (2021). Cellpose: a generalist algorithm for cellular anatomy. Nature Methods, 18(1), 100-106.

