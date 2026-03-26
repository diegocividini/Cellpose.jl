using NPZ
using Cellpose

println("=== Test Pipeline Completa (Analisi Discrepanze) ===")

img_path = joinpath(@__DIR__, "..", "test_data", "01_img_original.npy")
img = npzread(img_path)

model_path = joinpath(@__DIR__, "..", "models", "cpsam.onnx")

println("\nLancio l'inferenza Julia...")
masks = Cellpose.segment(img, model_path)

println("\nCarico i tensori salvati da Python...")
dP_python = npzread(joinpath(@__DIR__, "..", "test_data", "05_full_dP.npy"))
cellprob_python = npzread(joinpath(@__DIR__, "..", "test_data", "05_full_cellprob.npy"))
python_masks = npzread(joinpath(@__DIR__, "..", "test_data", "06_full_masks.npy"))

println("\n--- DIAGNOSI ---")
n_cells_julia = maximum(masks)
n_cells_python = maximum(python_masks)
println("Cellule Julia: $n_cells_julia")
println("Cellule Python: $n_cells_python")

println("\nRicalcolo le maschere Julia usando ESATTAMENTE i tensori di Python...")
masks_from_python_tensors = Cellpose.compute_masks(dP_python, cellprob_python; niter=200, cellprob_threshold=0.0, flow_threshold=0.4)
println("Cellule Julia (con dati Python): ", maximum(masks_from_python_tensors))

println("\nVERDETTO:")
if maximum(masks_from_python_tensors) ≈ n_cells_python
    println("Le dinamiche (Dynamics) sono perfette.")
    println("ERRORE TROVATO: La differenza è nello step 1-3 (Tiling, Stitching o Normalizzazione).")
else
    println("ERRORE TROVATO: La differenza è nello step 4 (Dynamics, calcolo flussi, etc).")
end