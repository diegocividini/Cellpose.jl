# run_julia.jl
using Cellpose          # Il tuo package
using FileIO, Images    # Per caricare/salvare (opzionale, se usi path stringa non serve)
using DataFrames, CSV   # Per risultati tabellari
using Statistics        # Per mean, median, std

# === CONFIGURAZIONE ===
IMG_DIR = "dataset/images"
OUT_DIR = "masks_julia"
RESULTS_CSV = "results_julia.csv"
MODEL_PATH = "models/cpsam.onnx"  # Percorso al tuo modello ONNX

# Parametri di segmentazione (DEVONO essere identici a Python!)
DIAMETER = 0.0          # 0.0 = auto-estimate, oppure valore fisso es. 30.0
USE_GPU = false         # true se vuoi testare accelerazione CUDA
CELLPROB_THRESH = 0.0   # Soglia probabilità cellula
FLOW_THRESH = 0.0       # Soglia consistenza flusso
MIN_SIZE = nothing      # nothing = auto, oppure valore fisso in px²
MAX_SIZE = nothing
NITER = 200             # Iterazioni Euler integration
# =====================

mkpath(OUT_DIR)
results = DataFrame(
    image=String[],
    n_cells=Int[],
    mean_area=Float64[],
    median_area=Float64[],
    std_area=Float64[]
)

for fname in readdir(IMG_DIR)
    # Filtra estensioni supportate
    endswith(fname, (".png", ".jpg", ".jpeg", ".tif", ".tiff")) || continue

    img_path = joinpath(IMG_DIR, fname)

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

    # Salva maschera come PNG (16-bit per preservare ID)
    mask_fname = replace(fname, r"\.\w+$" => ".png")
    mask_path = joinpath(OUT_DIR, mask_fname)
    save(mask_path, masks)  # FileIO.jl gestisce Int32 → PNG

    # Calcola metriche descrittive
    n_cells = maximum(masks)  # ID massimi = numero di cellule
    if n_cells > 0
        areas = [sum(masks .== i) for i in 1:n_cells]
        mean_area = mean(areas)
        median_area = median(areas)
        std_area = std(areas)
    else
        mean_area = median_area = std_area = 0.0
    end

    # Aggiungi alla tabella risultati
    push!(results, (
        image=fname,
        n_cells=n_cells,
        mean_area=mean_area,
        median_area=median_area,
        std_area=std_area
    ))

    println("✅ Julia: $fname → $n_cells cells")
end

# Salva risultati CSV
CSV.write(RESULTS_CSV, results)
println("📊 Risultati Julia salvati in $RESULTS_CSV")
println("📈 Media cellule: $(mean(results.n_cells)) ± $(std(results.n_cells))")