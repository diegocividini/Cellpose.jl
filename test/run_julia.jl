#!/usr/bin/env julia
"""
run_julia.jl - Segmentazione con Cellpose.jl (modello cpsam via ONNX)
Genera maschere e metriche descrittive per il benchmark vs Python.
"""
using Cellpose          # Il tuo package Cellpose.jl
using FileIO, Images    # Per caricare/salvare immagini
using DataFrames, CSV   # Per risultati tabellari
using Statistics        # Per mean, median, std
using Dates             # Per timestamp

# === CONFIGURAZIONE ===
IMG_DIR = "dataset/images"
OUT_DIR = "masks_julia"
RESULTS_CSV = "results/results_julia.csv"
MODEL_PATH = "models/cpsam.onnx"  # Percorso al tuo modello ONNX

# Parametri di segmentazione (DEVONO essere identici a Python!)
DIAMETER = 0.0          # 0.0 = auto-estimate, oppure valore fisso es. 30.0
USE_GPU = false         # true se vuoi testare accelerazione CUDA
CELLPROB_THRESH = 0.0   # Soglia probabilità cellula
FLOW_THRESH = 0.4       # Soglia consistenza flusso (default Cellpose)
MIN_SIZE = nothing      # nothing = auto, oppure valore fisso in px²
MAX_SIZE = nothing
NITER = 200             # Iterazioni Euler integration
# =====================

function setup()
    mkpath(OUT_DIR)
    mkpath("results")
    @info "Cellpose.jl benchmark initialized" MODEL_PATH USE_GPU DIAMETER
end

function compute_metrics(mask::Matrix{Int32})
    n_cells = maximum(mask)
    if n_cells > 0
        areas = [sum(mask .== i) for i in 1:n_cells]
        return (
            n_cells=n_cells,
            mean_area=mean(areas),
            median_area=median(areas),
            std_area=std(areas),
            min_area=minimum(areas),
            max_area=maximum(areas)
        )
    else
        return (n_cells=0, mean_area=0.0, median_area=0.0, std_area=0.0, min_area=0, max_area=0)
    end
end

function main()
    setup()
    results = DataFrame(
        image=String[], n_cells=Int[], mean_area=Float64[],
        median_area=Float64[], std_area=Float64[],
        min_area=Int[], max_area=Int[], shape_H=Int[], shape_W=Int[]
    )

    for fname in readdir(IMG_DIR)
        # Filtra estensioni supportate
        endswith(fname, (".png", ".jpg", ".jpeg", ".tif", ".tiff")) || continue
        img_path = joinpath(IMG_DIR, fname)

        @info "Processing" fname
        start_time = time()

        try
            # 🎯 CHIAMATA REALE A Cellpose.jl (API aggiornata)
            masks = Cellpose.segment(
                img_path,
                MODEL_PATH;
                use_gpu=USE_GPU,
                diameter=DIAMETER,
                cellprob_threshold=CELLPROB_THRESH,
                flow_threshold=FLOW_THRESH,
                min_size=MIN_SIZE,
                max_size=MAX_SIZE,
                niter=NITER
            )

            elapsed = time() - start_time

            # Salva maschera come PNG (16-bit per preservare ID)
            mask_fname = replace(fname, r"\.\w+$" => ".png")
            mask_path = joinpath(OUT_DIR, mask_fname)
            save(mask_path, masks)  # FileIO.jl gestisce Int32 → PNG

            # Calcola metriche
            metrics = compute_metrics(masks)

            # Aggiungi alla tabella risultati
            push!(results, (
                image=fname,
                n_cells=metrics.n_cells,
                mean_area=metrics.mean_area,
                median_area=metrics.median_area,
                std_area=metrics.std_area,
                min_area=metrics.min_area,
                max_area=metrics.max_area,
                shape_H=size(masks, 1),
                shape_W=size(masks, 2)
            ))

            @info "✅ Julia: $fname → $(metrics.n_cells) cells in $(round(elapsed, digits=2))s"

        catch e
            @warn "❌ Errore in $fname: $e"
            continue
        end
    end

    # Salva risultati CSV
    CSV.write(RESULTS_CSV, results)
    @info "📊 Risultati Julia salvati in $RESULTS_CSV"
    if !isempty(results)
        @info "📈 Stats: $(round(mean(results.n_cells), digits=1)) ± $(round(std(results.n_cells), digits=1)) cells/image"
    end
end

main()