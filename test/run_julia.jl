#!/usr/bin/env julia
using Cellpose
using FileIO, Images, DataFrames, CSV, Dates, Statistics

# Parse command line arguments
device = "cpu"
if "--device" in ARGS
    idx = findfirst(==("--device"), ARGS)
    if idx !== nothing && idx < length(ARGS)
        device = ARGS[idx+1]
    end
end
use_gpu = device == "gpu"

# Configuration
IMG_DIR = "dataset/images"
OUT_DIR = "masks_julia_$device"
OUT_CSV = "results/results_julia_$device.csv"
MODEL_PATH = "models/cpsam.onnx"

mkpath(OUT_DIR)
mkpath("results")
@info "Julia Cellpose.jl | Device: $(uppercase(device)) | Model: $MODEL_PATH"

df = DataFrame(
    image=String[], device=String[], n_cells=Int[],
    mean_area=Float64[], time_sec=Float64[]
)

for fname in readdir(IMG_DIR)
    endswith(fname, (".png", ".jpg", ".tif", ".tiff")) || continue
    img_path = joinpath(IMG_DIR, fname)

    if use_gpu
        using CUDA
        CUDA.synchronize()
    end

    t0 = time_ns()
    try
        masks = Cellpose.segment(img_path, MODEL_PATH;
            use_gpu=use_gpu, diameter=0.0, cellprob_threshold=0.0,
            flow_threshold=0.4, min_size=nothing, max_size=nothing, niter=200)

        if use_gpu
            CUDA.synchronize()
        end
        elapsed = (time_ns() - t0) / 1e9

        save(joinpath(OUT_DIR, replace(fname, r"\.\w+$" => ".png")), masks)

        n_cells = maximum(masks)
        areas = n_cells > 0 ? [sum(masks .== i) for i in 1:n_cells] : Float64[]
        push!(df, (
            image=fname, device=device, n_cells=n_cells,
            mean_area=isempty(areas) ? 0.0 : mean(areas),
            time_sec=round(elapsed, digits=3)
        ))
        @info "✅ $fname | $n_cells cells | $(round(elapsed, digits=3))s"
    catch e
        @warn "❌ Error in $fname: $e"
    end
end

CSV.write(OUT_CSV, df)
@info "📊 Results saved to $OUT_CSV"
if !isempty(df)
    @info "📈 Stats: $(round(mean(df.n_cells), digits=1)) ± $(round(std(df.n_cells), digits=1)) cells/image"
end