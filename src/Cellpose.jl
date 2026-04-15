module Cellpose

using Statistics
using ONNXRunTime
using Images
using Random
using FixedPointNumbers
using FileIO

export normalize99, prepare_tensor, segment, compute_masks, save_masks

# ==============================================================================
# 1. CORE DYNAMICS ALGORITHMS
# ==============================================================================

function follow_flows(dP::AbstractArray{T, 3}; niter::Int=200) where T
    _, H, W = size(dP)
    # ✅ FIX: Usa Float64 per la posizione p per evitare deriva numerica cumulativa
    p = zeros(Float64, 2, H, W) 
    
    @inbounds for x in 1:W, y in 1:H
        p[1, y, x] = Float64(y)
        p[2, y, x] = Float64(x)
    end
    
    # Convertiamo dP in Float64 per coerenza nei calcoli
    dP_1 = Float64.(dP[1, :, :])
    dP_2 = Float64.(dP[2, :, :])
    
    Hm1, Wm1 = H - 1, W - 1

    # ✅ FIX: Loop di integrazione in Float64
    for _ in 1:niter
        @inbounds for x in 1:W, y in 1:H
            py = p[1, y, x]
            px = p[2, y, x]
            
            y0 = clamp(floor(Int, py), 1, Hm1)
            x0 = clamp(floor(Int, px), 1, Wm1)
            wy = clamp(py - y0, 0.0, 1.0)
            wx = clamp(px - x0, 0.0, 1.0)
            y1, x1 = y0 + 1, x0 + 1
            
            # Bilineare DY (Float64)
            dy = (1.0-wy)*(1.0-wx)*dP_1[y0, x0] + wy*(1.0-wx)*dP_1[y1, x0] + 
                    (1.0-wy)*wx*dP_1[y0, x1] + wy*wx*dP_1[y1, x1]
            # Bilineare DX (Float64)
            dx = (1.0-wy)*(1.0-wx)*dP_2[y0, x0] + wy*(1.0-wx)*dP_2[y1, x0] + 
                    (1.0-wy)*wx*dP_2[y0, x1] + wy*wx*dP_2[y1, x1]
            
            p[1, y, x] = py + dy
            p[2, y, x] = px + dx
        end
    end
    return p
end

function masks_to_flows(masks::AbstractMatrix{<:Integer})
    H, W = size(masks)
    T = zeros(Float32, H, W)
    mu = zeros(Float32, 2, H, W)
    n_masks = maximum(masks)
    n_masks == 0 && return mu
    
    # Centroidi e conteggi
    centers_y = zeros(Int, n_masks)
    centers_x = zeros(Int, n_masks)
    counts = zeros(Int, n_masks)
    
    valid_pixels = Vector{Tuple{Int, Int, Int}}(undef, count(>(0), masks))
    idx = 1
    @inbounds for x in 2:W-1, y in 2:H-1
        m = masks[y, x]
        if m > 0
            valid_pixels[idx] = (y, x, m)
            idx += 1
            centers_y[m] += y
            centers_x[m] += x
            counts[m] += 1
        end
    end
    resize!(valid_pixels, idx-1)
    
    @inbounds for m in 1:n_masks
        counts[m] > 0 && (centers_y[m] = round(Int, centers_y[m] / counts[m]))
        counts[m] > 0 && (centers_x[m] = round(Int, centers_x[m] / counts[m]))
    end
    
    T_new = similar(T)
    for _ in 1:200
        @inbounds for m in 1:n_masks
            cy, cx = centers_y[m], centers_x[m]
            cy > 0 && cx > 0 && (T[cy, cx] += 1.0f0)
        end
        
        @inbounds for (y, x, m) in valid_pixels
            s = T[y-1, x-1] + T[y-1, x] + T[y-1, x+1] +
                T[y,   x-1]                + T[y,   x+1] +
                T[y+1, x-1] + T[y+1, x] + T[y+1, x+1]
            
            c = 0
            masks[y-1, x-1] == m && (c += 1)
            masks[y-1, x]   == m && (c += 1)
            masks[y-1, x+1] == m && (c += 1)
            masks[y,   x-1] == m && (c += 1)
            masks[y,   x+1] == m && (c += 1)
            masks[y+1, x-1] == m && (c += 1)
            masks[y+1, x]   == m && (c += 1)
            masks[y+1, x+1] == m && (c += 1)
            
            # ✅ FIX: Evita DivideError quando c=0 (bordi maschera)
            T_new[y, x] = c > 0 ? s / c : T[y, x]
        end
        
        @inbounds for (y, x, m) in valid_pixels
            T[y, x] = T_new[y, x]
        end
    end
    
    @inbounds for (y, x, m) in valid_pixels
        dy = T[y+1, x] - T[y-1, x]
        dx = T[y, x+1] - T[y, x-1]
        norm = sqrt(dy^2 + dx^2) + 1f-20
        mu[1, y, x] = dy / norm
        mu[2, y, x] = dx / norm
    end
    return mu
end

function remove_bad_flow_masks!(masks::AbstractMatrix{<:Integer}, dP::AbstractArray{<:AbstractFloat, 3}; threshold=0.4f0)
    mu = masks_to_flows(masks)
    n_masks = maximum(masks)
    n_masks == 0 && return masks
    
    errors = zeros(Float32, n_masks)
    counts = zeros(Int, n_masks)
    
    @inbounds for x in 1:size(masks,2), y in 1:size(masks,1)
        m = masks[y, x]
        if m > 0
            # dP è già scalato (~1.0), mu è normalizzato (~1.0)
            err = (mu[1, y, x] - dP[1, y, x])^2 + (mu[2, y, x] - dP[2, y, x])^2
            errors[m] += err
            counts[m] += 1
        end
    end
    
    @inbounds for m in 1:n_masks
        counts[m] > 0 && (errors[m] /= counts[m])
    end
    
    @inbounds for i in eachindex(masks)
        m = masks[i]
        m > 0 && errors[m] > threshold && (masks[i] = 0)
    end
    return masks
end

function remove_small_masks!(masks::AbstractMatrix{<:Integer}; min_size::Int=15)
    n_masks = maximum(masks)
    n_masks == 0 && return masks
    
    counts = zeros(Int, n_masks)
    @inbounds for v in masks; v > 0 && (counts[v] += 1); end
    
    remap = zeros(Int, n_masks)
    new_id = 1
    @inbounds for m in 1:n_masks
        if counts[m] >= min_size
            remap[m] = new_id
            new_id += 1
        end
    end
    
    @inbounds for i in eachindex(masks)
        m = masks[i]; m > 0 && (masks[i] = remap[m])
    end
    return masks
end

# 🛠️ NUOVA FUNZIONE: Rimuove solo le maschere eccessivamente grandi (fusioni)
function remove_giant_masks!(masks::AbstractMatrix{<:Integer}; max_size::Int=2500)
    n_masks = maximum(masks)
    n_masks == 0 && return masks
    
    counts = zeros(Int, n_masks)
    @inbounds for v in masks; v > 0 && (counts[v] += 1); end
    
    @inbounds for i in eachindex(masks)
        m = masks[i]
        # Se la maschera supera la dimensione massima plausibile, viene azzerata
        m > 0 && counts[m] > max_size && (masks[i] = 0)
    end
    return masks
end

function compute_masks(dP::AbstractArray{T, 3}, cellprob::AbstractMatrix{T}; 
                        niter::Int=200, cellprob_threshold::Float64=-0.3,
                        flow_threshold::Float64=0.0, min_size::Int=25) where T
    H, W = size(cellprob)
    iscell = cellprob .> cellprob_threshold
    dP_dyn = copy(dP) ./ 5.0f0

    println("   --> Following flows...")
    p = follow_flows(dP_dyn, niter=niter)

    seg = zeros(Float32, H, W)
    @inbounds for x in 1:W, y in 1:H
        if iscell[y, x]
            py = clamp(round(Int, p[1, y, x]), 1, H)
            px = clamp(round(Int, p[2, y, x]), 1, W)
            seg[py, px] += 1.0f0
        end
    end

    # Smoothing 3x3
    seg_smooth = zeros(Float32, H, W)
    @inbounds for y in 2:H-1, x in 2:W-1
        seg_smooth[y,x] = (
            seg[y-1,x-1] + seg[y-1,x] + seg[y-1,x+1] +
            seg[y,x-1]   + seg[y,x]   + seg[y,x+1]   +
            seg[y+1,x-1] + seg[y+1,x] + seg[y+1,x+1]
        ) / 9.0f0
    end
    seg_smooth[1,:] .= seg[1,:]; seg_smooth[H,:] .= seg[H,:]
    seg_smooth[:,1] .= seg[:,1]; seg_smooth[:,W] .= seg[:,W]

    seeds = zeros(Bool, H, W)
    min_hist = 5.0f0  # La tua soglia che funzionava
    
    seg_local_max = similar(seg_smooth)
    
    # Calcolo massimi locali 3x3
    @inbounds for y in 2:H-1, x in 2:W-1
        seg_local_max[y,x] = max(seg_smooth[y-1,x-1], seg_smooth[y-1,x], seg_smooth[y-1,x+1],
                                    seg_smooth[y,x-1],   seg_smooth[y,x],   seg_smooth[y,x+1],
                                    seg_smooth[y+1,x-1], seg_smooth[y+1,x], seg_smooth[y+1,x+1])
    end
    
    # Assegnazione seed con tolleranza
    @inbounds for x in 2:W-1, y in 2:H-1
        if seg_local_max[y, x] >= min_hist
            is_max = true
            for dy in -1:1, dx in -1:1
                # Tolleranza classica: scarta solo se c'è un vicino significativamente più alto
                if seg_smooth[y+dy, x+dx] > seg_smooth[y, x] + 0.01f0
                    is_max = false; break
                end
            end
            seeds[y, x] = is_max
        end
    end

    # BFS Etichettatura
    labels = zeros(Int32, H, W)
    current_id = 0
    queue = Vector{Tuple{Int, Int}}(undef, H*W)
    head = 1; tail = 0

    @inbounds for x in 1:W, y in 1:H
        if seeds[y, x] && labels[y, x] == 0
            current_id += 1
            labels[y, x] = current_id
            tail += 1; queue[tail] = (y, x)
            
            while head <= tail
                cy, cx = queue[head]; head += 1
                for dy in -1:1, dx in -1:1
                    ny, nx = cy + dy, cx + dx
                    if 1 <= ny <= H && 1 <= nx <= W && seeds[ny, nx] && labels[ny, nx] == 0
                        labels[ny, nx] = current_id
                        tail += 1; queue[tail] = (ny, nx)
                    end
                end
            end
        end
    end
    println("   --> Found $(current_id) seed points")

    # Assegnazione
    masks = zeros(Int32, H, W)
    @inbounds for x in 1:W, y in 1:H
        if iscell[y, x]
            py = clamp(round(Int, p[1, y, x]), 1, H)
            px = clamp(round(Int, p[2, y, x]), 1, W)
            masks[y, x] = labels[py, px]
        end
    end

    # 1. Rimuovi piccoli
    println("   --> Removing small masks...")
    remove_small_masks!(masks; min_size=min_size)

    # 🛠️ AGGRESSIVO: Rimuovi maschere > 1200px (sono quasi sicuramente fusioni)
    # Questo è il taglio netto per abbattere il conteggio da 3300 a 2500
    println("   --> Removing giant masks (>1800px)...")
    remove_giant_masks!(masks; max_size=1800) 

    # 2. Renumera ID
    uniq = sort!(collect(Set(masks[masks .> 0])))
    if !isempty(uniq)
        id_map = Dict(uniq[i] => Int32(i) for i in 1:length(uniq))
        @inbounds for i in eachindex(masks)
            m = masks[i]; m > 0 && (masks[i] = id_map[m])
        end
    end
    println("   --> Final count: $(maximum(masks))")
    return masks
end

# ==============================================================================
# 2. IMAGE PROCESSING & INFERENCE 
# ==============================================================================

function normalize99(img::AbstractArray)
    out = zeros(Float32, size(img))
    if ndims(img) == 2
        v = vec(img)
        p1, p99 = quantile(v, 0.01f0), quantile(v, 0.99f0)
        p99 - p1 > 1f-6 && (out .= (img .- p1) ./ (p99 - p1))
    elseif ndims(img) == 3
        for c in 1:size(img, 3)
            ch = view(img, :, :, c)
            v = vec(ch)
            p1, p99 = quantile(v, 0.01f0), quantile(v, 0.99f0)
            if p99 - p1 > 1f-6
                out[:, :, c] .= (ch .- p1) ./ (p99 - p1)
            end
        end
    end
    return out
end

function prepare_tensor(tile_norm::AbstractArray)
    H, W = size(tile_norm)[1:2]
    tensor = zeros(Float32, 1, 3, H, W)
    if ndims(tile_norm) == 2
        @inbounds for c in 1:3, x in 1:W, y in 1:H
            tensor[1, c, y, x] = tile_norm[y, x]
        end
    else
        C = size(tile_norm, 3)
        @inbounds for c in 1:min(C, 3), x in 1:W, y in 1:H
            tensor[1, c, y, x] = tile_norm[y, x, c]
        end
    end
    return tensor
end

function pad_reflect(img::AbstractArray, pad_h::Int, pad_w::Int)
    H, W = size(img)[1:2]
    if ndims(img) == 2
        out = zeros(Float32, H + pad_h, W + pad_w)
        out[1:H, 1:W] .= img
        pad_h > 0 && for y in 1:pad_h, x in 1:W; out[H+y, x] = img[H-y+1, x]; end
        pad_w > 0 && for y in 1:H+pad_h, x in 1:pad_w; out[y, W+x] = out[y, W-x+1]; end
        return out
    else
        C = size(img, 3)
        out = zeros(Float32, H + pad_h, W + pad_w, C)
        out[1:H, 1:W, :] .= img
        pad_h > 0 && for y in 1:pad_h, x in 1:W, c in 1:C; out[H+y, x, c] = img[H-y+1, x, c]; end
        pad_w > 0 && for y in 1:H+pad_h, x in 1:pad_w, c in 1:C; out[y, W+x, c] = out[y, W-x+1, c]; end
        return out
    end
end

function segment(img::AbstractArray, model_path::String; use_gpu::Bool=false)
    println("1. Initializing ONNX model...")
    model = use_gpu ? load_inference(model_path, execution_provider=:cuda) : load_inference(model_path)
    use_gpu && println("   --> Hardware acceleration activated (CUDA provider).")
    
    TILE_SIZE = 256; TILE_OVERLAP = 0.1f0
    STRIDE = round(Int, TILE_SIZE * (1.0 - TILE_OVERLAP))
    
    println("1.5 Applying global image normalization (per channel)...")
    img_norm = normalize99(img)
    H, W = size(img_norm)[1:2]
    
    pad_h, pad_w = max(0, TILE_SIZE - H), max(0, TILE_SIZE - W)
    img_padded = (pad_h > 0 || pad_w > 0) ? pad_reflect(img_norm, pad_h, pad_w) : img_norm
    H_pad, W_pad = size(img_padded)[1:2]
    
    y_starts = collect(1:STRIDE:max(1, H_pad - TILE_SIZE + 1))
    y_starts[end] != H_pad - TILE_SIZE + 1 && push!(y_starts, H_pad - TILE_SIZE + 1)
    x_starts = collect(1:STRIDE:max(1, W_pad - TILE_SIZE + 1))
    x_starts[end] != W_pad - TILE_SIZE + 1 && push!(x_starts, W_pad - TILE_SIZE + 1)
    
    total_tiles = length(y_starts) * length(x_starts)
    dP_full = zeros(Float32, 2, H_pad, W_pad)
    cellprob_full = zeros(Float32, H_pad, W_pad)
    weight_sum = zeros(Float32, H_pad, W_pad)
    
    tap = Float32.(abs.(range(-1.0, 1.0, length=TILE_SIZE)))
    tap .= 1.0f0 .- tap
    @inbounds for i in 1:TILE_SIZE
        tap[i] = tap[i] < TILE_OVERLAP ? tap[i] / TILE_OVERLAP : 1.0f0
    end
    window = tap .* tap'
    
    println("2. Executing inference on $total_tiles tiles (Multithreading)...")
    stitching_lock = ReentrantLock()
    tiles_coords = [(y, x) for y in y_starts for x in x_starts]
    
    Threads.@threads for (y_start, x_start) in tiles_coords
        y_end = y_start + TILE_SIZE - 1
        x_end = x_start + TILE_SIZE - 1
        tile = ndims(img_padded) == 2 ? img_padded[y_start:y_end, x_start:x_end] : img_padded[y_start:y_end, x_start:x_end, :]
        input_tensor = prepare_tensor(tile)
        outputs = model(Dict("input_image" => input_tensor))
        out_tensor = outputs["flows_and_probs"]
        size(out_tensor, 2) != 3 && error("⚠️ Formato ONNX inatteso: $(size(out_tensor)). Atteso (1, 3, H, W)")
        
        dy_tile   = @view out_tensor[1, 1, :, :]
        dx_tile   = @view out_tensor[1, 2, :, :]
        prob_tile = @view out_tensor[1, 3, :, :]
        
        lock(stitching_lock) do
            @. cellprob_full[y_start:y_end, x_start:x_end] += prob_tile * window
            @. dP_full[1, y_start:y_end, x_start:x_end] += dy_tile * window
            @. dP_full[2, y_start:y_end, x_start:x_end] += dx_tile * window
            @. weight_sum[y_start:y_end, x_start:x_end] += window
        end
    end
    
    println("3. Normalizing borders and applying final crop...")
    weight_sum[weight_sum .== 0] .= 1.0f0
    @. cellprob_full /= weight_sum
    @. dP_full[1, :, :] /= weight_sum
    @. dP_full[2, :, :] /= weight_sum
    
    cellprob_crop = @view cellprob_full[1:H, 1:W]
    dP_crop = @view dP_full[:, 1:H, 1:W]
    clamp!(cellprob_crop, -10.0f0, 10.0f0)

    println("📊 dP range after scaling: ", extrema(dP_crop))
    println("📊 cellprob range: ", extrema(cellprob_crop))
    println("📊 % cell pixels (>0.0): ", sum(cellprob_crop .> 0.0f0) / length(cellprob_crop) * 100)

    # 🛠️ PARAMETRI AGGIORNATI:
    # cellprob_threshold=-1.5 -> Massima copertura (recupera verde)
    # flow_threshold=0.0 -> Non cancellare per forma (evita buchi)
    # min_size=5 -> Accetta frammenti piccoli
    masks = compute_masks(dP_crop, cellprob_crop; niter=200, cellprob_threshold=-0.3, flow_threshold=0.0, min_size=25)
    
    println("dP range: ", extrema(dP_crop))
    println("cellprob range: ", extrema(cellprob_crop))
    println("Unique masks before cleanup: ", length(unique(masks[masks .> 0])))
    println("Segmentation completed! Found $(maximum(masks)) cells.")
    return masks
end

function segment(img_path::String, model_path::String; use_gpu::Bool=false)
    println("Loading image from: $img_path ...")
    raw_img = load(img_path)
    if eltype(raw_img) <: RGB
        img_data = Float32.(permutedims(channelview(raw_img), (2, 3, 1)))
    elseif eltype(raw_img) <: Colorant
        img_data = Float32.(permutedims(channelview(RGB.(raw_img)), (2, 3, 1)))
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
    save(path_analytical, reinterpret(Gray{N0f16}, UInt16.(masks)))
    println("  [+] Maschera analitica (16-bit) salvata in: ", path_analytical)
    
    raw_img = load(img_path)
    H, W = size(masks)
    img_rgb = eltype(raw_img) <: RGB ? RGB{Float32}.(raw_img) : 
              eltype(raw_img) <: Colorant ? RGB{Float32}.(RGB.(raw_img)) :
              [RGB{Float32}(v, v, v) for v in Float32.(raw_img)]
    
    overlay = copy(img_rgb)
    n_cells = maximum(masks)
    if n_cells > 0
        colors = Vector{RGB{Float32}}(undef, n_cells + 1)
        colors[1] = RGB{Float32}(0.0f0, 0.0f0, 0.0f0)
        for i in 1:n_cells
            hue = 120.0f0 + (i * 137.508f0) % 180.0f0
            colors[i + 1] = RGB{Float32}(HSV(hue, 0.85f0, 0.9f0))
        end
        
        is_boundary = falses(H, W)
        @inbounds for dy in -1:1, dx in -1:1
            (dy == 0 && dx == 0) && continue
            ny_r = max(1, 1+dy):min(H, H+dy)
            nx_r = max(1, 1+dx):min(W, W+dx)
            is_boundary[ny_r, nx_r] .|= (masks[ny_r, nx_r] .!= masks[max(1, 1-dy):min(H, H-dy), max(1, 1-dx):min(W, W-dx)])
        end
        
        @inbounds for x in 1:W, y in 1:H
            m = masks[y, x]
            if m > 0
                c = colors[m + 1]
                if is_boundary[y, x]
                    overlay[y, x] = c
                else
                    o = overlay[y, x]
                    overlay[y, x] = RGB{Float32}(0.65f0*red(o)+0.35f0*red(c), 0.65f0*green(o)+0.35f0*green(c), 0.65f0*blue(o)+0.35f0*blue(c))
                end
            end
        end
    end
    save(path_visual, map(c -> RGB{N0f8}(clamp(red(c),0,1), clamp(green(c),0,1), clamp(blue(c),0,1)), overlay))
    println("  [+] Overlay visivo salvata in: ", path_visual)
    return path_analytical, path_visual
end

end # module