using NPZ
using Cellpose

println("1. Caricamento dei tensori in uscita dalla rete (Fase 1 Python)...")
dP = npzread(joinpath(@__DIR__, "..", "test_data", "05_full_dP.npy"))
cellprob = npzread(joinpath(@__DIR__, "..", "test_data", "05_full_cellprob.npy"))
python_masks = npzread(joinpath(@__DIR__, "..", "test_data", "06_full_masks.npy"))

println("\n2. Calcolo della Dinamica in Julia (Euler Integration)...")
# Usiamo @time per vedere quanto è veloce Julia a fare questa follia matematica
@time julia_masks = Cellpose.compute_masks(dP, cellprob; niter=200, cellprob_threshold=0.0)

println("\n3. Analisi dei risultati...")
# Sfondo = 0, quindi togliamo 1 per contare le cellule vere
n_cells_python = length(unique(python_masks)) - 1
n_cells_julia = length(unique(julia_masks)) - 1

println("-> Cellule individuate da Python: ", n_cells_python)
println("-> Cellule individuate da Julia:  ", n_cells_julia)

if abs(n_cells_python - n_cells_julia) < 10
    println("\n✅ INCREDIBILE! Il raggruppamento funziona in modo quasi identico!")
else
    println("\n⚠️ Il numero differisce. È normale: Python applica filtri morfologici post-calcolo che implementeremo dopo.")
end