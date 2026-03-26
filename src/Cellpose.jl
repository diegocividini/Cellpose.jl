module Cellpose

using Statistics
using ONNXRunTime

include("Dynamics.jl")

export normalize99, prepare_tensor, segment, compute_masks

function normalize99(img::AbstractArray)
    out = zeros(Float32, size(img))
    if ndims(img) == 2
        p1 = quantile(vec(img), 0.01)
        p99 = quantile(vec(img), 0.99)
        if p99 - p1 > 1e-6
            out .= (img .- p1) ./ (p99 - p1)
        end
    elseif ndims(img) == 3
        # Normalizzazione indipendente per ogni canale RGB!
        for c in 1:size(img, 3)
            channel_data = vec(img[:, :, c])
            p1 = quantile(channel_data, 0.01)
            p99 = quantile(channel_data, 0.99)
            if p99 - p1 > 1e-6
                out[:, :, c] .= (img[:, :, c] .- p1) ./ (p99 - p1)
            end
        end
    end
    return out
end

"""
    prepare_tensor(tile_norm::AbstractArray)

Se è a colori (3D), passa i canali R, G, B direttamente alla rete.
Se è in scala di grigi (2D), clona il grigio su 3 canali.
"""
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
        error("Formato immagine non supportato")
    end
end

function pad_reflect(img::AbstractArray, pad_h::Int, pad_w::Int)
    H, W = size(img)[1:2]
    if ndims(img) == 2
        out = zeros(Float32, H + pad_h, W + pad_w)
        out[1:H, 1:W] .= img
        for y in 1:pad_h, x in 1:W; out[H+y, x] = img[H-y+1, x]; end
        for y in 1:H+pad_h, x in 1:pad_w; out[y, W+x] = out[y, W-x+1]; end
        return out
    else
        C = size(img, 3)
        out = zeros(Float32, H + pad_h, W + pad_w, C)
        out[1:H, 1:W, :] .= img
        for y in 1:pad_h, x in 1:W, c in 1:C; out[H+y, x, c] = img[H-y+1, x, c]; end
        for y in 1:H+pad_h, x in 1:pad_w, c in 1:C; out[y, W+x, c] = out[y, W-x+1, c]; end
        return out
    end
end

"""
    segment(img::AbstractArray, model_path::String)

Pipeline nativa con replica esatta di Cellpose v4 (cpsam).
"""
function segment(img::AbstractArray, model_path::String)
    println("1. Inizializzazione modello ONNX...")
    model = load_inference(model_path)
    
    TILE_SIZE = 256
    TILE_OVERLAP = 0.1 # 10% overlap 
    STRIDE = round(Int, TILE_SIZE * (1.0 - TILE_OVERLAP)) # 230 pixel
    
    println("1.5 Normalizzazione globale dell'immagine (per canale)...")
    img_norm = normalize99(img)
    
    H, W = size(img_norm)[1:2]
    
    # Pad solo se l'immagine è più piccola di 256x256 (nel tuo caso non farà nulla)
    pad_h = max(0, TILE_SIZE - H)
    pad_w = max(0, TILE_SIZE - W)
    
    if pad_h > 0 || pad_w > 0
        img_padded = pad_reflect(img_norm, pad_h, pad_w)
    else
        img_padded = img_norm
    end
    
    H_pad, W_pad = size(img_padded)[1:2]
    
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
    
    # Finestra A Tetto Piatto (Esatta)
    tap = Float32.(abs.(range(-1.0, 1.0, length=TILE_SIZE)))
    tap .= 1.0f0 .- tap
    for i in 1:TILE_SIZE
        if tap[i] < TILE_OVERLAP
            tap[i] = tap[i] / Float32(TILE_OVERLAP)
        else
            tap[i] = 1.0f0
        end
    end
    window = tap .* tap'
    
    println("2. Esecuzione Inferenza su $total_tiles riquadri (Multithreading)...")
    
    stitching_lock = ReentrantLock()
    tiles_coords = [(y, x) for y in y_starts for x in x_starts]
    
    Threads.@threads for (y_start, x_start) in tiles_coords
        y_end = y_start + TILE_SIZE - 1
        x_end = x_start + TILE_SIZE - 1
        
        if ndims(img_padded) == 2
            tile = img_padded[y_start:y_end, x_start:x_end]
        else
            tile = img_padded[y_start:y_end, x_start:x_end, :]
        end
        
        input_tensor = prepare_tensor(tile)
        
        outputs = model(Dict("input_image" => input_tensor))
        out_tensor = outputs["flows_and_probs"]
        
        lock(stitching_lock) do
            cellprob_full[y_start:y_end, x_start:x_end] .+= out_tensor[1, 1, :, :] .* window
            dP_full[1, y_start:y_end, x_start:x_end] .+= out_tensor[1, 2, :, :] .* window
            dP_full[2, y_start:y_end, x_start:x_end] .+= out_tensor[1, 3, :, :] .* window
            weight_sum[y_start:y_end, x_start:x_end] .+= window
        end
    end
    
    println("3. Normalizzazione dei bordi e Crop finale...")
    weight_sum[weight_sum .== 0] .= 1.0f0
    
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