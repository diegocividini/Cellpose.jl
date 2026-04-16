# Cellpose.jl

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://github.com/MouseLand/cellpose)
[![Language](https://img.shields.io/badge/language-Julia-9558B2.svg)](https://julialang.org/)

**Cellpose.jl** is an unofficial, native Julia port of the popular [Cellpose](https://github.com/MouseLand/cellpose) anatomical segmentation algorithm, originally developed in Python by Carsen Stringer and Marius Pachitariu.

This package provides a **pure Julia** pipeline for cellular segmentation using exported ONNX models (like the Vision Transformer-based `cpsam`). It completely bypasses Python interoperability overhead (`PyCall`/`PythonCall`), making it ideal for high-performance, native Julia bio-imaging workflows.

## 🚀 Key Features

* **No Python Dependency:** The entire pre-processing, inference, and post-processing pipeline is written in native Julia.
* **Multithreaded Tiling:** Automatically divides large biomedical images into overlapping patches (256x256) and processes them in parallel across available CPU cores using `Threads.@threads`.
* **Exact Mathematical Porting:** Replicates Cellpose v4's exact logic:
  * 10% tile overlap.
  * Flat-top window for seamless bilinear blending and stitching.
  * RGB channel formatting and 1-99th percentile normalization.
* **Custom Dynamics Engine:** The core Euler integration for fluid dynamics (`follow_flows`) and spatial gradient mask recovery (`compute_masks`) have been rewritten and optimized for Julia arrays.
* **Hardware Acceleration Ready:** Supports GPU inference on NVIDIA hardware via `ONNXRuntime.jl` with a simple flag toggle.

---

## ⚙️ System Requirements

| Component | Requirement |
|-----------|-------------|
| **Julia** | ≥ 1.9 (recommended: 1.10+) |
| **OS** | Linux, macOS, Windows (WSL2 recommended for Windows) |
| **RAM** | ≥ 8 GB (16+ GB recommended for images > 2048px) |
| **GPU** *(optional)* | NVIDIA GPU with CUDA ≥ 11.8 + cuDNN ≥ 8.9 |
| **Disk** | ~200 MB for ONNX model + temporary tile cache |

---

## ⚒️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/diegocividini/Cellpose.jl.git
cd Cellpose.jl
```

### 2. Instantiate the Julia Environment

```julia
# Open Julia REPL and enter package mode with ']'
julia> ]
pkg> activate .
pkg> instantiate
```

This will install all required dependencies:
* **ONNXRuntime.jl** – ONNX model inference engine
* **Images.jl, FileIO.jl** – Image I/O and processing
* **Statistics.jl**, **FixedPointNumbers.jl** – Numerical utilities


### 3. Export or Download the ONNX Model

The package does not include pretrained models due to size constraints. You have two options:
#### Option A: Export from Python (Recommended)
If you have the official Cellpose Python package installed:

```bash
# Run the provided export script
python scripts/py_to_onnx.py --model cpsam --output models/
```

This generates `models/cpsam.onnx` (and optionally `cpsam.onnx.data`).

#### Option B: Use a Pre-exported Model

If you already have a compatible ONNX model:

1. Place it in the `models/` directory
2. Ensure the model outputs a tensor named `"flows_and_probs"` with shape `(1, 3, H, W)` where:
    * Channel 1: vertical flow (`dy`)
    * Channel 2: horizontal flow (`dx`)
    * Channel 3: cell probability map

### 4. (Optional) Configure GPU Support

If using NVIDIA GPU acceleration:
1. Install `CUDA Toolkit ≥ 11.8`
2. Install `cuDNN ≥ 8.9` and add to LD_LIBRARY_PATH (Linux) or system PATH (Windows)
3. In Julia, verify CUDA is detected:

    ```julia
    using CUDA
    CUDA.functional()  # should return true
    ```

---

### ✨ Quick Start

#### Basic Usage (Image Array)

```julia
using Cellpose
using FileIO, Images

# Load image (supports TIFF, PNG, JPEG, NPZ, etc.)
img = load("data/my_image.tif")  # Returns Array{<:Colorant} or Matrix{<:Real}

# Convert to Float32 array if needed (Cellpose handles this internally)
# But you can pre-convert for control:
img_data = Float32.(channelview(img)[1:3, :, :])  # For RGB; drop [1:3, :, :] for grayscale

# Run segmentation
model_path = "models/cpsam.onnx"
masks = segment(img_data, model_path; use_gpu=false, diameter=0.0)  # diameter=0.0 → auto-estimate

println("Segmentation complete. Found $(maximum(masks)) cells.")
```

#### Basic Usage (File Path – Simplest)

```julia
using Cellpose

# Let Cellpose handle loading and conversion internally
masks = segment("data/my_image.tif", "models/cpsam.onnx"; use_gpu=true, diameter=30.0)

println("Found $(maximum(masks)) cells.")
```

#### Save Results

```julia
# Save both analytical (16-bit TIFF) and visual (PNG overlay) outputs
path_tiff, path_png = save_masks("data/my_image.tif", masks, "results/output")

# Outputs:
#   results/output.tif          → Raw mask IDs (UInt16, 0=background)
#   results/output_overlay.png  → Original image + colored cell boundaries
```

#### 🧠 Pipeline Overview

1. **Global Normalization**: Per-channel scaling to [0,1] using 1st–99th percentiles (robust to outliers).
2. **Overlapping Tiling**: Image split into 256×256 tiles with 10% overlap (stride = 230 px).
3. **Parallel Inference**: Each tile processed independently via ONNX runtime (CPU or CUDA).
4. **Flat-Top Blending**: Tiles stitched using a custom window function to avoid edge artifacts.
5. **Flow Integration**: Pixels "flow" along predicted vector field via Euler integration (`follow_flows`).
6. **Seed Detection**: Local maxima in converged positions identify cell centers.
7. **Mask Assignment**: Pixels assigned to nearest seed via BFS-like expansion.
8. **Quality Control**: Masks filtered by size (`min_size`, `max_size`) and flow consistency.

---

### 📚 API Reference

#### Core Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `segment` | `segment(img::AbstractArray, model_path::String; use_gpu=false, diameter=0.0)` | Main entry point. Returns `Matrix{Int32}` mask array |
| `segment` | `segment(img_path::String, model_path::String; use_gpu=false, diameter=0.0)` | Convenience overload: loads image internally |
| `save_masks` | `save_masks(img_path, masks, output_path)` | Saves analytical (TIFF) and visual (PNG) outputs |
| `normalize99` | `normalize99(img::AbstractArray)` | Percentile-based normalization [0,1] |
| `estimate_diameter` | `estimate_diameter(cellprob::Matrix; threshold=0.0)` | Auto-estimates cell diameter from probability map |
| `compute_masks` | `compute_masks(dP, cellprob; niter=200, min_size=100, max_size=1800)` | Low-level mask computation from flows + probabilities |

#### Key Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------| ----------- |
| `use_gpu` | `Bool` | `false` | Enable CUDA acceleration (requires NVIDIA GPU + CUDA toolkit) |
| `diameter` | `Float64` | `0.0` | Expected cell diameter in pixels. `0.0` triggers auto-estimation |
| `min_size` | `Int` | dynamic | Minimum mask area (px²). Auto-scaled from `diameter` if not set |
| `max_size` | `Int` | dynamic | Maximum mask area (px²). Auto-scaled from `diameter` |
| `cellprob_threshold` | `Float64` | `0.0` | Pixels with probability < threshold are treated as background |
| `flow_threshold` | `Float64` | `0.0` | Masks with mean flow error > threshold are discarded |
| `niter` | `Int` | 200 | Number of Euler integration steps for flow following |

---

### 🐛 Troubleshooting

| Issue | Likely Cause | Solution |
|-----------|------|---------|
| `MethodError: no method matching Float32(::Matrix{Float32})` | Missing `.` in broadcasting (e.g., `Float32(arr)` vs `Float32.(arr)`) | Ensure all scalar conversions use broadcasting or are inside loops with explicit indexing. |
| `ONNX model not found` | Incorrect path or missing `.onnx.data` file | Verify `model_path` points to the `.onnx` file; keep `.data` sibling file if present. |
| `CUDA not functional` | Missing cuDNN or incompatible CUDA version | Run `using CUDA; CUDA.versioninfo()` to diagnose; ensure cuDNN is in library path. |
| `Out of memory` | Large image + GPU + many tiles | Reduce image size, use CPU (use_gpu=false), or increase system RAM. |
| `Too few cells detected` | Overly strict `min_size` or wrong `diameter` | Let `diameter=0.0` auto-estimate, or manually tune `min_size`/`max_size`. |
| `Edge artifacts in masks` | Insufficient tile overlap | Do not modify `TILE_OVERLAP=0.1`; the flat-top window requires this value for correct blending. |

---

### ⚡ Performance Tips

1. **Thread Count**: Set `JULIA_NUM_THREADS` before starting Julia:
    ```bash
    export JULIA_NUM_THREADS=8  # Linux/macOS
    set JULIA_NUM_THREADS=8     # Windows CMD
    ```
    You can also automatically set the number of threads starting Julia with the flag: `-t auto`.
2. **GPU vs CPU**:
    * GPU: Faster for large images (>1024px) but requires data transfer overhead.
    * CPU: More predictable memory usage; scales well with core count.
3. **Memory Management**:
    * For very large images (>4096px), consider pre-cropping regions of interest.
    * The tiling system limits peak memory to ~3× tile size × channels.
4. **Diameter Hint**: Providing an approximate diameter (even ±30%) improves speed and accuracy by reducing false positives in estimate_diameter.

---

### 📜 Acknowledgments & Citation

This is an independent port built for educational and research integration purposes. All credit for the original machine learning architecture, pretrained models, and algorithmic concepts goes to the original Cellpose authors.

If you use the Cellpose algorithm in your research, please cite their original papers:
> Stringer, C., Wang, T., Michaelos, M., & Pachitariu, M. (2021). Cellpose: a generalist algorithm for cellular anatomy. Nature Methods, 18(1), 100-106. https://doi.org/10.1038/s41592-020-01018-x

> Pachitariu, M., & Stringer, C. (2022). Cellpose 2.0: how to train your own model. Nature Methods, 19(12), 1634-1641. https://doi.org/10.1038/s41592-022-01663-4