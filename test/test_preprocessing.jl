using NPZ
using Statistics
using Cellpose 

println("1. Caricamento dell'immagine originale (da Python)...")
img_original = npzread(joinpath(@__DIR__, "..", "test_data", "01_img_original.npy"))

println("\n2. Estrazione del tile centrale...")
sz = size(img_original)
H = sz[1]
W = sz[2]
ch = H ÷ 2
cw = W ÷ 2

# Julia richiede di specificare la terza dimensione se esiste!
if length(sz) == 2
    tile = img_original[ch-128+1:ch+128, cw-128+1:cw+128]
else
    # Mettiamo i ':' per dire "tutti i canali"
    tile = img_original[ch-128+1:ch+128, cw-128+1:cw+128, :] 
end

println("\n3. Normalizzazione 99 in Julia...")
tile_norm_julia = Cellpose.normalize99(tile)

println("\n4. Preparazione del tensore 4D per ONNX...")
tensor_julia = Cellpose.prepare_tensor(tile_norm_julia)

println("\n5. Confronto con il tensore preparato da Python...")
tensor_python = npzread(joinpath(@__DIR__, "..", "test_data", "02_tile_256_input.npy"))

max_diff = maximum(abs.(tensor_julia .- tensor_python))

println("Dimensioni tensore Julia: ", size(tensor_julia))
println("Dimensioni tensore Python: ", size(tensor_python))
println("Differenza massima: ", max_diff)

if max_diff < 1e-4
    println("\n✅ PRE-PROCESSING PERFETTO! La tua logica Julia è allineata con Python.")
else
    println("\n⚠️ Discrepanza trovata. Controlliamo la logica dei percentili.")
end