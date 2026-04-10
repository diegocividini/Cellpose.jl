module Cellpose

using Statistics
using ONNXRunTime
using Images
using Random
using FixedPointNumbers
using FileIO

include("Dynamics.jl")

export normalize99, prepare_tensor, segment, compute_masks, save_masks

function normalize99(img::AbstractArray)
    out = zeros(Float32, size(img))
    if ndims(img) == 2
        p1 = quantile(vec(img), 0.01)
        p99 = quantile(vec(img), 0.99)
        if p99 - p1 > 1e-6
            out .= (img .- p1) ./ (p99 - p1)
        end
    elseif ndims(img) == 3
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

function prepare_tensor(tile_norm::AbstractArray)
    H, W = size(tile_norm)[1:2]
    tensor = zeros(Float32, W, H, 3, 1)
    if ndims(tile_norm) == 2
        for c in 1:3
            for y in 1:H, x in 1:W
                tensor[x, y, c, 1] = tile_norm[y, x]
            end
        end
    else
        C = size(tile_norm, 3)
        for c in 1:min(C, 3)
            for y in 1:H, x in 1:W
                tensor[x, y, c, 1] = tile_norm[y, x, c]
            end
        end
    end
    return reshape(tensor, (1, 3, H, W))
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

function segment(img::AbstractArray, model_path::String; use_gpu::Bool=false)
    println("1. Initializing ONNX model...")
    if use_gpu
        println("   --> Hardware acceleration activated (CUDA provider).")
        model = load_inference(model_path, execution_provider=:cuda)
    else
        model = load_inference(model_path)
    end
    
    TILE_SIZE = 256 
    TILE_OVERLAP = 0.1
    STRIDE = round(Int, TILE_SIZE * (1.0 - TILE_OVERLAP)) 
    
    println("1.5 Applying global image normalization (per channel)...")
    img_norm = normalize99(img)
    
    H, W = size(img_norm)[1:2]
    
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
    
    println("2. Executing inference on $total_tiles tiles (Multithreading)...")
    
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
        
        out_rev = reshape(out_tensor, (TILE_SIZE, TILE_SIZE, 3, 1))
        
        prob_tile = permutedims(out_rev[:, :, 1, 1], (2, 1))
        dy_tile   = permutedims(out_rev[:, :, 2, 1], (2, 1))
        dx_tile   = permutedims(out_rev[:, :, 3, 1], (2, 1))
        
        lock(stitching_lock) do
            cellprob_full[y_start:y_end, x_start:x_end] .+= prob_tile .* window
            dP_full[1, y_start:y_end, x_start:x_end] .+= dy_tile .* window
            dP_full[2, y_start:y_end, x_start:x_end] .+= dx_tile .* window
            weight_sum[y_start:y_end, x_start:x_end] .+= window
        end
    end
    
    println("3. Normalizing borders and applying final crop...")
    weight_sum[weight_sum .== 0] .= 1.0f0
    
    cellprob_full ./= weight_sum
    dP_full[1, :, :] ./= weight_sum
    dP_full[2, :, :] ./= weight_sum
    
    cellprob_crop = cellprob_full[1:H, 1:W]
    dP_crop = dP_full[:, 1:H, 1:W]
    
    println("4. Computing dynamic flows (Euler Integration)...")
    
    masks = compute_masks(dP_crop, cellprob_crop; niter=200, cellprob_threshold=0.5, flow_threshold=0.4)
    
    println("Segmentation completed! Found $(maximum(masks)) cells.")
    return masks
end

function segment(img_path::String, model_path::String; use_gpu::Bool=false)
    println("Loading image from: $img_path ...")
    raw_img = load(img_path)
    
    if eltype(raw_img) <: Colorant
        img_data = Float32.(permutedims(channelview(raw_img), (2, 3, 1)))
    else
        img_data = Float32.(raw_img)
    end
    
    return segment(img_data, model_path; use_gpu=use_gpu)
end

function save_masks(img_path::String, masks::AbstractMatrix{<:Integer}, output_path::String)
    base_path = replace(output_path, r"\.(tif|tiff|png|jpg|jpeg)$"i => "")
    
    path_analytical = base_path * ".tif"
    path_visual = base_path * "_overlay.png"
    
    println("Salvataggio risultati in corso...")
    
    # 1. Maschera Analitica
    analytical_mask = reinterpret(Gray{N0f16}, UInt16.(masks))
    save(path_analytical, analytical_mask)
    println("  [+] Maschera analitica (16-bit) salvata in: ", path_analytical)
    
    # 2. Overlay
    raw_img = load(img_path)
    
    if eltype(raw_img) <: RGB
        img_data = Float32.(permutedims(channelview(raw_img), (2, 3, 1)))
    elseif eltype(raw_img) <: Colorant
        img_data = Float32.(permutedims(channelview(RGB.(raw_img)), (2, 3, 1)))
    else
        img_data = Float32.(raw_img)
    end
    
    img_norm = normalize99(img_data)
    H, W = size(masks)
    
    overlay = zeros(RGB{Float32}, H, W)
    if ndims(img_norm) == 2
        for y in 1:H, x in 1:W
            v = img_norm[y, x]
            overlay[y, x] = RGB(v, v, v)
        end
    else
        for y in 1:H, x in 1:W
            overlay[y, x] = RGB(img_norm[y, x, 1], img_norm[y, x, 2], img_norm[y, x, 3])
        end
    end
    
    n_cells = maximum(masks)
    if n_cells > 0
        Random.seed!(42) 
        colors = [RGB(0.0, 0.0, 0.0)]
        for _ in 1:n_cells
            push!(colors, RGB(rand(0.3:0.1:1.0), rand(0.3:0.1:1.0), rand(0.3:0.1:1.0)))
        end
        
        for y in 1:H, x in 1:W
            m = masks[y, x]
            if m > 0
                c = colors[m + 1]
                is_boundary = false
                for dy in -1:1, dx in -1:1
                    ny, nx = y + dy, x + dx
                    if 1 <= ny <= H && 1 <= nx <= W
                        if masks[ny, nx] != m
                            is_boundary = true
                            break
                        end
                    end
                end
                
                if is_boundary
                    overlay[y, x] = c
                else
                    bg = overlay[y, x]
                    overlay[y, x] = bg * 0.7f0 + c * 0.3f0
                end
            end
        end
    end
    
    # FIX: Converte forzatamente i colori a 8-bit per non corrompere il PNG!
    overlay_u8 = map(c -> RGB{N0f8}(clamp(red(c), 0.0f0, 1.0f0), clamp(green(c), 0.0f0, 1.0f0), clamp(blue(c), 0.0f0, 1.0f0)), overlay)
    
    save(path_visual, overlay_u8)
    println("  [+] Overlay visivo (Immagine + Maschere) salvato in: ", path_visual)
    
    return path_analytical, path_visual
end

end # module