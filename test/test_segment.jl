using NPZ
using Cellpose

println("=== Test Pipeline Completa Cellpose.jl ===")

# 1. Carica l'immagine originale gigante (2832x2296)
img_path = joinpath(@__DIR__, "..", "test_data", "01_img_original.npy")
img = npzread(img_path)

# 2. Punta al modello ONNX
model_path = joinpath(@__DIR__, "..", "models", "cpsam.onnx")

# 3. Lancia il macchinario!
@time masks = Cellpose.segment(img, model_path)

println("Dimensioni maschera generata: ", size(masks))
println("Numero di cellule: ", maximum(masks))
println("==========================================")