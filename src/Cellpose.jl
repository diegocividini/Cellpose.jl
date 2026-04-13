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
    
    for x in 1:W, y in 1:H
        p[1, y, x] = y
        p[2, y, x] = x
    end
    
    # Integrazione di Eulero: i pixel si muovono dolcemente
    for i in 1:niter
        for x in 1:W, y in 1:H
            py = p[1, y, x]
            px = p[2, y, x]
            
            y_idx = clamp(round(Int, py), 1, H)
            x_idx = clamp(round(Int, px), 1, W)
            
            p[1, y, x] = py + dP[1, y_idx, x_idx]
            p[2, y, x] = px + dP[2, y_idx, x_idx]
        end
    end
    
    return p
end

function masks_to_flows(masks::AbstractMatrix{<:Integer})
    H, W = size(masks)
    T = zeros(Float32, H, W)
    mu = zeros(Float32, 2, H, W)
    
    n_masks = maximum(masks)
    if n_masks == 0
        return mu
    end
    
    # Ottimizzazione: Usare array invece di Dict per i centri
    centers_y = zeros(Int, n_masks)
    centers_x = zeros(Int, n_masks)
    counts = zeros(Int, n_masks)
    
    # Preallocazione per evitare riallocazioni dinamiche
    valid_pixels = Tuple{Int, Int, Int}[]
    sizehint!(valid_pixels, H * W ÷ 4) 
    
    for y in 2:H-1, x in 2:W-1
        m = masks[y, x]
        if m > 0
            push!(valid_pixels, (y, x, m))
            centers_y[m] += y
            centers_x[m] += x
            counts[m] += 1
        end
    end
    
    # Calcolo dei centroidi
    for m in 1:n_masks
        if counts[m] > 0
            centers_y[m] = round(Int, centers_y[m] / counts[m])
            centers_x[m] = round(Int, centers_x[m] / counts[m])
        end
    end
    
    n_iter = 50 
    T_new = zeros(Float32, H, W)
    
    for iter in 1:n_iter
        # Aggiungi "calore" ai centri
        for m in 1:n_masks
            cy, cx = centers_y[m], centers_x[m]
            if cy > 0 && cx > 0
                T[cy, cx] += 1.0f0
            end
        end
        
        # Diffusione
        for (y, x, m) in valid_pixels
            # Somma valori vicini (unrolling manuale per velocità)
            # Siccome valid_pixels è generato da 2:H-1, siamo sicuri che y-1 e y+1 esistano
            s = T[y-1, x-1] + T[y-1, x] + T[y-1, x+1] +
                T[y,   x-1]                + T[y,   x+1] +
                T[y+1, x-1] + T[y+1, x] + T[y+1, x+1]
            
            # Conta vicini con stessa maschera
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
        
        # Aggiorna T
        for (y, x, m) in valid_pixels
            T[y, x] = T_new[y, x]
        end
    end
    
    # Calcolo gradienti finali
    for (y, x, m) in valid_pixels
        dy = T[y+1, x] - T[y-1, x]
        dx = T[y, x+1] - T[y, x-1]
        norm = sqrt(dy^2 + dx^2) + 1f-20
        mu[1, y, x] = dy / norm
        mu[2, y, x] = dx / norm
    end
    
    return mu
end

function remove_bad_flow_masks!(masks::AbstractMatrix{<:Integer}, dP::AbstractArray{<:AbstractFloat, 3}; threshold=0.4)
    mu = masks_to_flows(masks)
    
    n_masks = maximum(masks)
    errors = zeros(Float32, n_masks)
    counts = zeros(Int, n_masks)
    
    H, W = size(masks)
    
    for y in 1:H, x in 1:W
        m = masks[y, x]
        if m > 0
            # FIX: dP è già stato diviso per 5.0, ora ha magnitudine 1 come "mu"
            dy_net = dP[1, y, x]
            dx_net = dP[2, y, x]
            
            dy_mask = mu[1, y, x]
            dx_mask = mu[2, y, x]
            
            err = (dy_mask - dy_net)^2 + (dx_mask - dx_net)^2
            errors[m] += err
            counts[m] += 1
        end
    end
    
    for m in 1:n_masks
        if counts[m] > 0
            errors[m] /= counts[m]
        end
    end
    
    for i in eachindex(masks)
        m = masks[i]
        if m > 0 && errors[m] > threshold
            masks[i] = 0
        end
    end
    
    return masks
end

function remove_small_masks!(masks::AbstractMatrix{<:Integer}; min_size::Int=15)
    sizes = Dict{Int, Int}()
    for val in masks
        if val > 0
            sizes[val] = get(sizes, val, 0) + 1
        end
    end
    
    new_ids = Dict{Int, Int}()
    current_id = 1
    for (val, count) in sizes
        if count >= min_size
            new_ids[val] = current_id
            current_id += 1
        else
            new_ids[val] = 0
        end
    end
    
    for i in eachindex(masks)
        val = masks[i]
        if val > 0
            masks[i] = new_ids[val]
        end
    end
    
    return masks
end

function compute_masks(dP::AbstractArray{T, 3}, cellprob::AbstractMatrix{T}; 
                        niter::Int=200, cellprob_threshold::Float64=0.0, 
                        flow_threshold::Float64=0.4) where T
    
    H, W = size(cellprob)
    iscell = cellprob .> cellprob_threshold
    
    # Python: dP * (cellprob > cellprob_threshold) / 5.
    dP_dyn = copy(dP) ./ 5.0f0
    dP_dyn[:, .!iscell] .= 0.0f0  # Azzera flussi nello sfondo
    
    println("   --> Following flows...")
    p = follow_flows(dP_dyn, niter=niter)
    
    # Istogramma convergenza (get_masks_torch in Python)
    hist = zeros(Int32, H, W)
    for x in 1:W, y in 1:H
        if iscell[y, x]
            py = clamp(round(Int, p[1, y, x]), 1, H)
            px = clamp(round(Int, p[2, y, x]), 1, W)
            hist[py, px] += 1
        end
    end
    
    # Trova semi (Python usa soglia fissa 10)
    seeds = Vector{Tuple{Int, Int, Int32}}()
    current_id = Int32(1)
    min_hist = 10  # ✅ RIPRISTINATO A 10
    
    for x in 1:W, y in 1:H
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
    
    # Inizializza griglia e propaga label
    grid = zeros(Int32, H, W)
    for (sy, sx, id) in seeds
        grid[sy, sx] = id
    end
    
    changed = true
    iter = 0
    while changed && iter < 20
        changed = false
        iter += 1
        for y in 2:H-1, x in 2:W-1
            if iscell[y, x] && grid[y, x] == 0
                best_label = Int32(0)
                max_count = 0
                for dy in -1:1, dx in -1:1
                    lbl = grid[y+dy, x+dx]
                    if lbl > 0
                        c = 1
                        for dy2 in -1:1, dx2 in -1:1
                            if grid[y+dy2, x+dx2] == lbl; c += 1; end
                        end
                        if c > max_count
                            max_count = c
                            best_label = lbl
                        end
                    end
                end
                if best_label > 0
                    grid[y, x] = best_label
                    changed = true
                end
            end
        end
    end
    
    # Assegna maschere finali mappando posizioni finali ai semi
    masks = zeros(Int32, H, W)
    for x in 1:W, y in 1:H
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
        for i in eachindex(masks)
            masks[i] = masks[i] > 0 ? id_map[masks[i]] : 0
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
    # Alloca direttamente nel formato NCHW (1, 3, H, W)
    tensor = zeros(Float32, 1, 3, H, W)
    
    if ndims(tile_norm) == 2
        # Grayscale: duplica su 3 canali
        for c in 1:3
            for y in 1:H, x in 1:W
                tensor[1, c, y, x] = tile_norm[y, x]
            end
        end
    else
        # Color: prendi i primi 3 canali
        C = size(tile_norm, 3)
        for c in 1:min(C, 3)
            for y in 1:H, x in 1:W
                tensor[1, c, y, x] = tile_norm[y, x, c]
            end
        end
    end
    return tensor
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
        
        # 1. Estrai la tile dall'immagine padded
        if ndims(img_padded) == 2
            tile = img_padded[y_start:y_end, x_start:x_end]
        else
            tile = img_padded[y_start:y_end, x_start:x_end, :]
        end
        
        # 2. Prepara il tensore NCHW per ONNX
        input_tensor = prepare_tensor(tile)
        
        # 3. Esegui inference
        outputs = model(Dict("input_image" => input_tensor))
        out_tensor = outputs["flows_and_probs"]
        
        # 🔍 Verifica formato (deve essere NCHW: 1 batch, 3 canali, H, W)
        if size(out_tensor, 2) != 3
            error("⚠️ Formato ONNX inatteso: $(size(out_tensor)). Atteso (1, 3, H, W)")
        end
        
        # 4. Estrai canali direttamente (più veloce e sicuro di reshape+permutedims)
        dy_tile   = out_tensor[1, 1, :, :]
        dx_tile   = out_tensor[1, 2, :, :]
        prob_tile = out_tensor[1, 3, :, :]
        
        # 5. Accumula con pesatura e thread-safety
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
    
    # Normalizza correttamente i flussi
    println("5. Scaling flows for dynamics (Cellpose standard)...")
    # Cellpose calibration: i flussi sono addestrati con scala ~5.0
    # dP_crop ./= 5.0f0

    # Opzionale: clip di sicurezza per evitare valori estremi da artefatti ONNX
    clamp!(cellprob_crop, -10.0f0, 10.0f0)

    println("📊 dP range after scaling: ", extrema(dP_crop))
    println("📊 cellprob range: ", extrema(cellprob_crop))
    println("📊 % cell pixels (>0.0): ", sum(cellprob_crop .> 0.0) / length(cellprob_crop) * 100)

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
    
    # 1. Maschera Analitica
    analytical_mask = reinterpret(Gray{N0f16}, UInt16.(masks))
    save(path_analytical, analytical_mask)
    println("  [+] Maschera analitica (16-bit) salvata in: ", path_analytical)
    
    # 2. Overlay ottimizzato per H&E
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
        # 🟢 PALETTE FREDDA (Verde/Blu) per massimo contrasto su H&E
        colors = Vector{RGB{Float32}}(undef, n_cells + 1)
        colors[1] = RGB{Float32}(0.0f0, 0.0f0, 0.0f0)
        
        for i in 1:n_cells
            # Usa l'angolo aureo ma shiftato nel range 120°-300° (Verde -> Ciano -> Blu)
            # Evitiamo deliberatamente il Rosso (0°-60°) che si mimetizza col tessuto
            hue = 120.0f0 + (i * 137.508) % 180.0f0
            
            # Saturazione alta (0.85) e Valore alto (0.9) per colori vivaci
            c = HSV(hue, 0.85f0, 0.9f0)
            colors[i + 1] = RGB{Float32}(c)
        end
        
        # Calcola bordi in modo efficiente
        is_boundary = falses(H, W)
        
        for dy in -1:1, dx in -1:1
            (dy == 0 && dx == 0) && continue
            ny_range = max(1, 1+dy):min(H, H+dy)
            nx_range = max(1, 1+dx):min(W, W+dx)
            
            shifted_masks = zeros(Int32, H, W)
            shifted_masks[ny_range, nx_range] .= masks[max(1, 1-dy):min(H, H-dy), max(1, 1-dx):min(W, W-dx)]
            
            is_boundary .|= (shifted_masks .!= masks)
        end
        
        # Applica colori
        for y in 1:H, x in 1:W
            m = masks[y, x]
            if m > 0
                c = colors[m + 1]
                if is_boundary[y, x]
                    overlay[y, x] = c
                else
                    orig = overlay[y, x]
                    # 🟢 Blending più trasparente per vedere meglio il tessuto
                    overlay[y, x] = RGB{Float32}(
                        0.65f0 * red(orig) + 0.35f0 * red(c),
                        0.65f0 * green(orig) + 0.35f0 * green(c),
                        0.65f0 * blue(orig) + 0.35f0 * blue(c)
                    )
                end
            end
        end
    end
    
    overlay_u8 = map(c -> RGB{N0f8}(
        clamp(red(c), 0.0f0, 1.0f0),
        clamp(green(c), 0.0f0, 1.0f0),
        clamp(blue(c), 0.0f0, 1.0f0)
    ), overlay)
    
    save(path_visual, overlay_u8)
    println("  [+] Overlay visivo salvato in: ", path_visual)
    
    return path_analytical, path_visual
end

end # module