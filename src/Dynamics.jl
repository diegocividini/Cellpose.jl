# src/Dynamics.jl

"""
    follow_flows(dP::AbstractArray{T, 3}; niter::Int=200)

Simula il movimento dei pixel seguendo i gradienti (dP) per `niter` iterazioni.
È l'implementazione Julia dell'integrazione di Eulero di Cellpose.
"""
function follow_flows(dP::AbstractArray{T, 3}; niter::Int=200) where T
    _, H, W = size(dP)
    
    # Inizializziamo la griglia delle coordinate di partenza
    p = zeros(Float32, 2, H, W)
    p_new = zeros(Float32, 2, H, W)
    
    for x in 1:W, y in 1:H
        p[1, y, x] = y
        p[2, y, x] = x
    end
    
    # Integrazione: muoviamo ogni punto lungo il campo vettoriale
    for i in 1:niter
        for x in 1:W, y in 1:H
            py = p[1, y, x]
            px = p[2, y, x]
            
            # Arrotondiamo per trovare l'indice della matrice del gradiente
            y_idx = clamp(round(Int, py), 1, H)
            x_idx = clamp(round(Int, px), 1, W)
            
            # Aggiorniamo la posizione sommando il vettore direzionale
            p_new[1, y, x] = py + dP[1, y_idx, x_idx]
            p_new[2, y, x] = px + dP[2, y_idx, x_idx]
        end
        
        # Scambiamo i buffer per l'iterazione successiva (molto efficiente in memoria)
        temp = p
        p = p_new
        p_new = temp
    end
    
    return p
end

"""
    compute_masks(dP, cellprob; niter=200, cellprob_threshold=0.0)

Prende i flussi e le probabilità, traccia i pixel e genera le maschere delle cellule.
"""
function compute_masks(dP::AbstractArray{T, 3}, cellprob::AbstractMatrix{T}; 
                       niter::Int=200, cellprob_threshold::Float64=0.0, flow_threshold::Float64=0.4) where T
    
    # 1. Spostiamo i pixel verso il centro delle cellule
    p = follow_flows(dP, niter=niter)
    
    H, W = size(cellprob)
    iscell = cellprob .> cellprob_threshold
    
    # 2. Creiamo l'istogramma delle posizioni finali
    hist = zeros(Int32, H, W)
    for x in 1:W, y in 1:H
        if iscell[y, x]
            py = clamp(round(Int, p[1, y, x]), 1, H)
            px = clamp(round(Int, p[2, y, x]), 1, W)
            hist[py, px] += 1
        end
    end
    
    # 3. Troviamo i "picchi" locali (i centri) che hanno più di 10 pixel
    seeds = Vector{Tuple{Int, Int, Int32}}()
    current_id = Int32(1)
    for x in 1:W, y in 1:H
        if hist[y, x] > 10
            is_max = true
            for dx in -2:2, dy in -2:2
                (dx == 0 && dy == 0) && continue
                nx, ny = x + dx, y + dy
                if 1 <= nx <= W && 1 <= ny <= H
                    if hist[ny, nx] > hist[y, x]
                        is_max = false
                        break
                    end
                end
            end
            if is_max
                push!(seeds, (y, x, current_id))
                current_id += 1
            end
        end
    end
    
    # 4. Espandiamo i centri di 5 pixel per accorpare i mucchi sparsi
    grid = zeros(Int32, H, W)
    for (sy, sx, id) in seeds
        for dx in -5:5, dy in -5:5
            nx, ny = sx + dx, sy + dy
            if 1 <= nx <= W && 1 <= ny <= H
                grid[ny, nx] = id
            end
        end
    end
    
    # 5. Assegniamo l'ID finale della maschera
    masks = zeros(Int32, H, W)
    for x in 1:W, y in 1:H
        if iscell[y, x]
            py = clamp(round(Int, p[1, y, x]), 1, H)
            px = clamp(round(Int, p[2, y, x]), 1, W)
            masks[y, x] = grid[py, px]
        end
    end
    
    # 6. Filtri di errore e grandezza
    if flow_threshold > 0.0
        remove_bad_flow_masks!(masks, dP; threshold=flow_threshold)
    end
    remove_small_masks!(masks; min_size=15)

    return masks
end

"""
    remove_small_masks!(masks::AbstractMatrix{<:Integer}; min_size::Int=15)

Elimina le maschere (cellule) che hanno un'area inferiore a `min_size` pixel.
Riordina gli ID rimanenti in modo che siano contigui da 1 a N.
"""
function remove_small_masks!(masks::AbstractMatrix{<:Integer}; min_size::Int=15)
    # 1. Contiamo quanto è grande ogni cellula
    sizes = Dict{Int, Int}()
    for val in masks
        if val > 0
            sizes[val] = get(sizes, val, 0) + 1
        end
    end
    
    # 2. Creiamo una mappa: ID vecchio -> ID nuovo
    # Se la cellula è troppo piccola, il suo nuovo ID sarà 0 (sfondo)
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
    
    # 3. Applichiamo la mappa all'immagine
    for i in eachindex(masks)
        val = masks[i]
        if val > 0
            masks[i] = new_ids[val]
        end
    end
    
    return masks
end

"""
    masks_to_flows(masks::AbstractMatrix{<:Integer})

Calcola i flussi teorici (ideali) a partire dalle maschere 2D.
Replica la funzione `masks_to_flows_gpu` e `_extend_centers_gpu` di Cellpose 
usando la diffusione del calore.
"""
function masks_to_flows(masks::AbstractMatrix{<:Integer})
    H, W = size(masks)
    T = zeros(Float32, H, W)
    mu = zeros(Float32, 2, H, W)
    
    n_masks = maximum(masks)
    if n_masks == 0
        return mu
    end
    
    # OTTIMIZZAZIONE: Raccogliamo solo le coordinate dei pixel validi
    valid_pixels = Tuple{Int, Int, Int}[] 
    coords_dict = Dict{Int, Vector{Tuple{Int, Int}}}()
    
    for y in 2:H-1, x in 2:W-1
        m = masks[y, x]
        if m > 0
            push!(valid_pixels, (y, x, m))
            if !haskey(coords_dict, m)
                coords_dict[m] = Tuple{Int, Int}[]
            end
            push!(coords_dict[m], (y, x))
        end
    end
    
    # 1. Trova il centro di massa
    centers = fill((0, 0), n_masks)
    for i in 1:n_masks
        if !haskey(coords_dict, i)
            continue
        end
        coords = coords_dict[i]
        sum_y = sum(c[1] for c in coords)
        sum_x = sum(c[2] for c in coords)
        mean_y = round(Int, sum_y / length(coords))
        mean_x = round(Int, sum_x / length(coords))
        
        min_dist = Inf
        center = coords[1]
        for c in coords
            dist = (c[1] - mean_y)^2 + (c[2] - mean_x)^2
            if dist < min_dist
                min_dist = dist
                center = c
            end
        end
        centers[i] = center
    end
    
    # 2. Diffusione del calore SUPER VELOCE (solo sui pixel delle cellule)
    n_iter = 200 
    T_new = zeros(Float32, H, W)
    
    for iter in 1:n_iter
        for i in 1:n_masks
            cy, cx = centers[i]
            if cy != 0
                T[cy, cx] += 1.0f0
            end
        end
        
        for (y, x, m) in valid_pixels
            s = 0.0f0
            c = 0
            for dy in -1:1, dx in -1:1
                if masks[y+dy, x+dx] == m
                    s += T[y+dy, x+dx]
                    c += 1
                end
            end
            T_new[y, x] = s / c
        end
        
        # Scambia i valori 
        for (y, x, m) in valid_pixels
            T[y, x] = T_new[y, x]
        end
    end
    
    # 3. Calcola i gradienti (derivate spaziali)
    for (y, x, m) in valid_pixels
        dy = T[y+1, x] - T[y-1, x]
        dx = T[y, x+1] - T[y, x-1]
        norm = sqrt(dy^2 + dx^2) + 1f-20
        mu[1, y, x] = dy / norm
        mu[2, y, x] = dx / norm
    end
    
    return mu
end

"""
    remove_bad_flow_masks!(masks, dP; threshold=0.4)

Elimina le maschere il cui flusso teorico non coincide con quello predetto dalla rete.
Replica esattamente la logica di `remove_bad_flow_masks` e `flow_error`.
"""
function remove_bad_flow_masks!(masks::AbstractMatrix{<:Integer}, dP::AbstractArray{<:AbstractFloat, 3}; threshold=0.4)
    # Calcola i flussi teorici
    mu = masks_to_flows(masks)
    
    n_masks = maximum(masks)
    errors = zeros(Float32, n_masks)
    counts = zeros(Int, n_masks)
    
    H, W = size(masks)
    
    # Calcola l'errore quadratico medio per ogni cellula
    for y in 1:H, x in 1:W
        m = masks[y, x]
        if m > 0
            # Cellpose scala sempre i flussi di rete per 5.0 durante l'inferenza
            dy_net = dP[1, y, x] / 5.0f0
            dx_net = dP[2, y, x] / 5.0f0
            
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
    
    # Elimina le cellule con errore > threshold (0.4)
    for i in eachindex(masks)
        m = masks[i]
        if m > 0 && errors[m] > threshold
            masks[i] = 0
        end
    end
    
    return masks
end