module Cellpose

using Statistics
using ONNXRunTime

# Importa il nostro motore matematico
include("Dynamics.jl")

export normalize99, prepare_tensor, segment, compute_masks

function normalize99(img::AbstractArray)
    p1 = quantile(vec(img), 0.01)
    p99 = quantile(vec(img), 0.99)
    
    if p99 - p1 < 1e-6
        return zeros(Float32, size(img))
    end
    
    img_norm = (img .- p1) ./ (p99 - p1)
    return Float32.(img_norm)
end

function prepare_tensor(tile_norm::AbstractArray)
    if ndims(tile_norm) == 2
        H, W = size(tile_norm)
        tensor = zeros(Float32, 1, 3, H, W)
        for c in 1:3
            tensor[1, c, :, :] .= tile_norm
        end
        return tensor
    elseif ndims(tile_norm) == 3
        H, W, C = size(tile_norm)
        tensor = zeros(Float32, 1, 3, H, W)
        for c in 1:min(C, 3)
            tensor[1, c, :, :] .= tile_norm[:, :, c]
        end
        return tensor
    else
        error("Formato immagine non supportato. Richiesto 2D o 3D.")
    end
end

"""
    segment(img::AbstractArray, model_path::String)

Pipeline completa: pre-processing, inferenza ONNX con Tiling in sovrapposizione (50%), 
fusione bilineare (blending) e calcolo dinamico delle maschere.
"""
function segment(img::AbstractArray, model_path::String)
    println("1. Inizializzazione modello ONNX...")
    model = load_inference(model_path)
    
    TILE_SIZE = 256
    STRIDE = 128 # Sovrapposizione del 50%
    
    H, W = size(img)[1:2]
    
    pad_h = max(0, TILE_SIZE - H)
    pad_w = max(0, TILE_SIZE - W)
    
    H_pad = H + pad_h
    W_pad = W + pad_w
    
    if ndims(img) == 2
        img_padded = zeros(eltype(img), H_pad, W_pad)
        img_padded[1:H, 1:W] .= img
    else
        C = size(img, 3)
        img_padded = zeros(eltype(img), H_pad, W_pad, C)
        img_padded[1:H, 1:W, :] .= img
    end
    
    y_starts = collect(1:STRIDE:max(1, H_pad - TILE_SIZE + 1))
    if y_starts[end] != H_pad - TILE_SIZE + 1
        push!(y_starts, H_pad - TILE_SIZE + 1)
    end
    
    x_starts = collect(1:STRIDE:max(1, W_pad - TILE_SIZE + 1))
    if x_starts[end] != W_pad - TILE_SIZE + 1
        push!(x_starts, W_pad - TILE_SIZE + 1)
    end
    
    total_tiles = length(y_starts) * length(x_starts)
    
    dP_full = zeros(Float32, 2, H_pad, W_pad)
    cellprob_full = zeros(Float32, H_pad, W_pad)
    weight_sum = zeros(Float32, H_pad, W_pad)
    
    # Finestra Bilineare
    w1 = Float32[min(i, TILE_SIZE - i + 1) for i in 1:TILE_SIZE]
    window = w1 .* w1'
    
    println("2. Esecuzione Inferenza su $total_tiles riquadri (con sovrapposizione bilineare)...")
    println("   (Nota: impiegherà più tempo per via della precisione extra)")
    
    tile_count = 0
    for y_start in y_starts
        for x_start in x_starts
            tile_count += 1
            y_end = y_start + TILE_SIZE - 1
            x_end = x_start + TILE_SIZE - 1
            
            if ndims(img_padded) == 2
                tile = img_padded[y_start:y_end, x_start:x_end]
            else
                tile = img_padded[y_start:y_end, x_start:x_end, :]
            end
            
            tile_norm = normalize99(tile)
            input_tensor = prepare_tensor(tile_norm)
            
            outputs = model(Dict("input_image" => input_tensor))
            out_tensor = outputs["flows_and_probs"]
            
            cellprob_full[y_start:y_end, x_start:x_end] .+= out_tensor[1, 1, :, :] .* window
            dP_full[1, y_start:y_end, x_start:x_end] .+= out_tensor[1, 2, :, :] .* window
            dP_full[2, y_start:y_end, x_start:x_end] .+= out_tensor[1, 3, :, :] .* window
            weight_sum[y_start:y_end, x_start:x_end] .+= window
        end
    end
    
    println("3. Normalizzazione dei bordi e Rimozione padding...")
    cellprob_full ./= weight_sum
    dP_full[1, :, :] ./= weight_sum
    dP_full[2, :, :] ./= weight_sum
    
    cellprob_crop = cellprob_full[1:H, 1:W]
    dP_crop = dP_full[:, 1:H, 1:W]
    
    println("4. Calcolo Dinamica dei fluidi (Euler Integration)...")
    masks = compute_masks(dP_crop, cellprob_crop; niter=200, cellprob_threshold=0.0, flow_threshold=0.4)
    
    println("Completato! Trovate $(maximum(masks)) cellule.")
    return masks
end

end # module