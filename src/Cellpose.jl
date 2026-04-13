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
    p = zeros(Float32, 2, H, W)
    
    # Inizializza posizioni (ordine cache-friendly)
    @inbounds for x in 1:W, y in 1:H
        p[1, y, x] = Float32(y)
        p[2, y, x] = Float32(x)
    end
    
    dP_1 = @view dP[1, :, :]
    dP_2 = @view dP[2, :, :]
    p_1 = @view p[1, :, :]
    p_2 = @view p[2, :, :]

    # Pre-calc bounds
    Hm1, Wm1 = H - 1, W - 1

    for _ in 1:niter
        @inbounds for x in 1:W, y in 1:H
            py = p_1[y, x]
            px = p_2[y, x]
            
            # Clamp sicuro per interpolazione
            py_clamp = clamp(py, 1.0f0, Float32(H))
            px_clamp = clamp(px, 1.0f0, Float32(W))
            
            y0 = floor(Int, py_clamp)
            x0 = floor(Int, px_clamp)
            
            # Pesi frazionari
            wy = py_clamp - y0
            wx = px_clamp - x0
            
            # Gestione bordi: se siamo all'ultimo indice, il peso verso il "prossimo" è 0
            if y0 == H; wy = 0.0f0; end
            if x0 == W; wx = 0.0f0; end
            
            y1 = min(y0 + 1, H)
            x1 = min(x0 + 1, W)
            
            # Interpolazione Bilineare DY
            dy_interp = (1.0f0-wy)*(1.0f0-wx)*dP_1[y0, x0] + 
                        wy*(1.0f0-wx)*dP_1[y1, x0] + 
                        (1.0f0-wy)*wx*dP_1[y0, x1] + 
                        wy*wx*dP_1[y1, x1]
            
            # Interpolazione Bilineare DX
            dx_interp = (1.0f0-wy)*(1.0f0-wx)*dP_2[y0, x0] + 
                        wy*(1.0f0-wx)*dP_2[y1, x0] + 
                        (1.0f0-wy)*wx*dP_2[y0, x1] + 
                        wy*wx*dP_2[y1, x1]
            
            p_1[y, x] = py + dy_interp
            p_2[y, x] = px + dx_interp
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
    
    centers_y = zeros(Int, n_masks)
    centers_x = zeros(Int, n_masks)
    counts = zeros(Int, n_masks)
    
    # Cache-valid pixels for diffusion
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
    
    # Calcolo centroidi
    @inbounds for m in 1:n_masks
        counts[m] > 0 && (centers_y[m] = round(Int, centers_y[m] / counts[m]))
        counts[m] > 0 && (centers_x[m] = round(Int, centers_x[m] / counts[m]))
    end
    
    n_iter = 200
    T_new = similar(T)
    
    for _ in 1:n_iter
        # Heat injection
        @inbounds for m in 1:n_masks
            cy, cx = centers_y[m], centers_x[m]
            if cy > 0 && cx > 0
                T[cy, cx] += 1.0f0
            end
        end
        
        # Diffusion step
        @inbounds for (y, x, m) in valid_pixels
            s = T[y-1, x-1] + T[y-1, x] + T[y-1, x+1] +
                T[y,   x-1]                + T[y,   x+1] +
                T[y+1, x-1] + T[y+1, x] + T[y+1, x+1]
            
            c = 0
            if masks[y-1, x-1] == m; c += 1; end
            if masks[y-1, x]   == m; c += 1; end
            if masks[y-1, x+1] == m; c += 1; end
            if masks[y,   x-1] == m; c += 1; end
            if masks[y,   x+1] == m; c += 1; end
            if masks[y+1, x-1] == m; c += 1; end
            if masks[y+1, x]   == m; c += 1; end
            if masks[y+1, x+1] == m; c += 1; end
            
            T_new[y, x] = c > 0 ? s / c : T[y, x]
        end
        
        @inbounds for (y, x, m) in valid_pixels
            T[y, x] = T_new[y, x]
        end
    end
    
    # Final gradients
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
    H, W = size(masks)
    
    inv_scale = 1.0f0 / 5.0f0
    
    @inbounds for x in 1:W, y in 1:H
        m = masks[y, x]
        if m > 0
            dy_net = dP[1, y, x] * inv_scale
            dx_net = dP[2, y, x] * inv_scale
            err = (mu[1, y, x] - dy_net)^2 + (mu[2, y, x] - dx_net)^2
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
    
    # Array diretto invece di Dict: ~10x più veloce e deterministico
    counts = zeros(Int, n_masks)
    @inbounds for v in masks
        v > 0 && (counts[v] += 1)
    end
    
    remap = zeros(Int, n_masks)
    new_id = 1
    @inbounds for m in 1:n_masks
        if counts[m] >= min_size
            remap[m] = new_id
            new_id += 1
        end
    end
    
    @inbounds for i in eachindex(masks)
        m = masks[i]
        m > 0 && (masks[i] = remap[m])
    end
    
    return masks
end

function compute_masks(dP::AbstractArray{T, 3}, cellprob::AbstractMatrix{T}; 
                        niter::Int=200, cellprob_threshold::Float64=0.0, 
                        flow_threshold::Float64=0.4) where T
    
    H, W = size(cellprob)
    iscell = cellprob .> cellprob_threshold
    
    dP_dyn = copy(dP) ./ 5.0f0
    @inbounds for x in 1:W, y in 1:H
        if !iscell[y, x]
            dP_dyn[1, y, x] = 0.0f0
            dP_dyn[2, y, x] = 0.0f0
        end
    end
    
    println("   --> Following flows...")
    p = follow_flows(dP_dyn, niter=niter)
    
    # Istogramma convergenza
    hist = zeros(Int32, H, W)
    @inbounds for x in 1:W, y in 1:H
        if iscell[y, x]
            py = clamp(round(Int, p[1, y, x]), 1, H)
            px = clamp(round(Int, p[2, y, x]), 1, W)
            hist[py, px] += 1
        end
    end
    
    # Trova semi
    seeds = Vector{Tuple{Int, Int, Int32}}()
    current_id = Int32(1)
    min_hist = 10
    
    @inbounds for x in 1:W, y in 1:H
        if hist[y, x] >= min_hist
            is_max = true
            for dy in -1:1, dx in -1:1
                (dx == 0 && dy == 0) && continue
                ny, nx = y + dy, x + dx
                if 1 <= ny <= H && 1 <= nx <= W && hist[ny, nx] > hist[y, x]
                    is_max = false; break
                end
            end
            if is_max
                push!(seeds, (y, x, current_id))
                current_id += 1
            end
        end
    end
    
    println("   --> Found $(length(seeds)) seed points")
    
    # Inizializza griglia e propaga con BFS (convergenza garantita O(N))
    grid = zeros(Int32, H, W)
    for (sy, sx, id) in seeds
        grid[sy, sx] = id
    end
    
    # Queue-based flood fill
    queue = Vector{Tuple{Int, Int}}()
    sizehint!(queue, sum(iscell))
    for (sy, sx) in seeds
        push!(queue, (sy, sx))
    end
    
    head = 1
    @inbounds while head <= length(queue)
        y, x = queue[head]
        head += 1
        lbl = grid[y, x]
        
        for dy in -1:1, dx in -1:1
            (dy == 0 && dx == 0) && continue
            ny, nx = y + dy, x + dx
            if 1 <= ny <= H && 1 <= nx <= W && iscell[ny, nx] && grid[ny, nx] == 0
                grid[ny, nx] = lbl
                push!(queue, (ny, nx))
            end
        end
    end
    
    # Assegna maschere finali
    masks = zeros(Int32, H, W)
    @inbounds for x in 1:W, y in 1:H
        if iscell[y, x]
            py = clamp(round(Int, p[1, y, x]), 1, H)
            px = clamp(round(Int, p[2, y, x]), 1, W)
            masks[y, x] = grid[py, px]
        end
    end
    
    # Post-processing
    if flow_threshold > 0.0
        println("   --> Removing bad flow masks...")
        remove_bad_flow_masks!(masks, dP; threshold=flow_threshold)
    end
    println("   --> Removing small masks...")
    remove_small_masks!(masks; min_size=15)
    
    # Rinumera consecutivamente
    uniq = sort!(collect(Set(masks[masks .> 0])))
    if !isempty(uniq)
        id_map = Dict(uniq[i] => Int32(i) for i in 1:length(uniq))
        @inbounds for i in eachindex(masks)
            m = masks[i]
            m > 0 && (masks[i] = id_map[m])
        end
    end
    
    return masks
end

# ==============================================================================
# 2. IMAGE PROCESSING & INFERENCE 
# ==============================================================================

function normalize99(img::AbstractArray)
    out = zeros(Float32, size(img))
    if ndims(img) == 2
        p1 = quantile(vec(img), 0.01f0)
        p99 = quantile(vec(img), 0.99f0)
        if p99 - p1 > 1f-6
            out .= (img .- p1) ./ (p99 - p1)
        end
    elseif ndims(img) == 3
        for c in 1:size(img, 3)
            channel_data = view(img, :, :, c)
            p1 = quantile(channel_data, 0.01f0)
            p99 = quantile(channel_data, 0.99f0)
            if p99 - p1 > 1f-6
                out[:, :, c] .= (channel_data .- p1) ./ (p99 - p1)
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
        if pad_h > 0
            @inbounds for x in 1:W, y in 1:pad_h
                out[H+y, x] = img[H-y+1, x]
            end
        end
        if pad_w > 0
            @inbounds for y in 1:H+pad_h, x in 1:pad_w
                out[y, W+x] = out[y, W-x+1]
            end
        end
        return out
    else
        C = size(img, 3)
        out = zeros(Float32, H + pad_h, W + pad_w, C)
        out[1:H, 1:W, :] .= img
        if pad_h > 0
            @inbounds for c in 1:C, x in 1:W, y in 1:pad_h
                out[H+y, x, c] = img[H-y+1, x, c]
            end
        end
        if pad_w > 0
            @inbounds for c in 1:C, y in 1:H+pad_h, x in 1:pad_w
                out[y, W+x, c] = out[y, W-x+1, c]
            end
        end
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
    TILE_OVERLAP = 0.1f0
    STRIDE = round(Int, TILE_SIZE * (1.0 - TILE_OVERLAP)) 
    
    println("1.5 Applying global image normalization (per channel)...")
    img_norm = normalize99(img)
    
    H, W = size(img_norm)[1:2]
    pad_h = max(0, TILE_SIZE - H)
    pad_w = max(0, TILE_SIZE - W)
    
    img_padded = (pad_h > 0 || pad_w > 0) ? pad_reflect(img_norm, pad_h, pad_w) : img_norm
    H_pad, W_pad = size(img_padded)[1:2]
    
    y_starts = collect(1:STRIDE:max(1, H_pad - TILE_SIZE + 1))
    if y_starts[end] != H_pad - TILE_SIZE + 1; push!(y_starts, H_pad - TILE_SIZE + 1); end
    x_starts = collect(1:STRIDE:max(1, W_pad - TILE_SIZE + 1))
    if x_starts[end] != W_pad - TILE_SIZE + 1; push!(x_starts, W_pad - TILE_SIZE + 1); end
    
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

    masks = compute_masks(dP_crop, cellprob_crop; niter=200, cellprob_threshold=0.0, flow_threshold=0.4)

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
    
    analytical_mask = reinterpret(Gray{N0f16}, UInt16.(masks))
    save(path_analytical, analytical_mask)
    println("  [+] Maschera analitica (16-bit) salvata in: ", path_analytical)
    
    raw_img = load(img_path)
    H, W = size(masks)
    
    if eltype(raw_img) <: RGB
        img_rgb = RGB{Float32}.(raw_img)
    elseif eltype(raw_img) <: Colorant
        img_rgb = RGB{Float32}.(RGB.(raw_img))
    else
        gray_img = Float32.(raw_img)
        img_rgb = [RGB{Float32}(v, v, v) for v in gray_img]
    end
    
    overlay = copy(img_rgb)
    n_cells = maximum(masks)
    
    if n_cells > 0
        colors = Vector{RGB{Float32}}(undef, n_cells + 1)
        colors[1] = RGB{Float32}(0.0f0, 0.0f0, 0.0f0)
        
        for i in 1:n_cells
            hue = 120.0f0 + (i * 137.508f0) % 180.0f0
            c = HSV(hue, 0.85f0, 0.9f0)
            colors[i + 1] = RGB{Float32}(c)
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
                    overlay[y, x] = RGB{Float32}(
                        0.65f0 * red(o) + 0.35f0 * red(c),
                        0.65f0 * green(o) + 0.35f0 * green(c),
                        0.65f0 * blue(o) + 0.35f0 * blue(c)
                    )
                end
            end
        end
    end
    
    overlay_u8 = map(c -> RGB{N0f8}(clamp(red(c), 0.0f0, 1.0f0), clamp(green(c), 0.0f0, 1.0f0), clamp(blue(c), 0.0f0, 1.0f0)), overlay)
    save(path_visual, overlay_u8)
    println("  [+] Overlay visivo salvato in: ", path_visual)
    
    return path_analytical, path_visual
end

end # module