### Dynamics.jl

"""
    follow_flows(dP::AbstractArray{T, 3}; niter::Int=200)

    It takes the predicted flows (dP) and iteratively moves each pixel 
    according to these flows for a specified number of iterations (niter).
    
        The output is a new set of coordinates (p) that represent the final positions 
    of each pixel after following the flows.
"""
function follow_flows(dP::AbstractArray{T, 3}; niter::Int=200) where T
    _, H, W = size(dP)
    
    # Initializing the coordinates of the starting points (p) for each pixel
    p = zeros(Float32, 2, H, W)
    p_new = zeros(Float32, 2, H, W)
    
    for x in 1:W, y in 1:H
        p[1, y, x] = y
        p[2, y, x] = x
    end
    
    # Moving each point according to the flow vectors for niter iterations, using the nearest neighbor approach
    for i in 1:niter
        for x in 1:W, y in 1:H
            py = p[1, y, x]
            px = p[2, y, x]
            
            # Rounding to the nearest pixel to get the flow vector at that location
            y_idx = clamp(round(Int, py), 1, H)
            x_idx = clamp(round(Int, px), 1, W)
            
            # Updating the position of the pixel according to the flow vector
            p_new[1, y, x] = py + dP[1, y_idx, x_idx]
            p_new[2, y, x] = px + dP[2, y_idx, x_idx]
        end
        
        # Swapping the buffers for the next iteration (high memory efficiency)
        temp = p
        p = p_new
        p_new = temp
    end
    
    return p
end

"""
    compute_masks(dP, cellprob; niter=200, cellprob_threshold=0.0)

    1. Moves each pixel towards the center of the cells by following the flow vectors (dP) for niter iterations.
    2. Creates a histogram of the final positions of the pixels that exceed the cell probability threshold (cellprob_threshold).
    3. Identifies local "peaks" in the histogram that represent the centers of the cells.
    4. Expands the identified centers by 5 pixels to merge scattered groups of pixels belonging to the same cell.
    5. Assigns a mask ID to each pixel based on the center it is associated with.
    6. Applies error and size filters to remove invalid masks.
"""
function compute_masks(dP::AbstractArray{T, 3}, cellprob::AbstractMatrix{T}; 
                        niter::Int=200, cellprob_threshold::Float64=0.0, flow_threshold::Float64=0.4) where T
    
    # 1. Moving each pixel according to the flow vectors (dP) for niter iterations
    p = follow_flows(dP, niter=niter)
    
    H, W = size(cellprob)
    iscell = cellprob .> cellprob_threshold
    
    # 2. Creates the histogram of the final positions
    hist = zeros(Int32, H, W)
    for x in 1:W, y in 1:H
        if iscell[y, x]
            py = clamp(round(Int, p[1, y, x]), 1, H)
            px = clamp(round(Int, p[2, y, x]), 1, W)
            hist[py, px] += 1
        end
    end
    
    # 3. Identifies local "peaks" (the centers) that have more than 10 pixels
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
    
    # 4. Expands the identified centers by 5 pixels to merge scattered groups of pixels belonging to the same cell
    grid = zeros(Int32, H, W)
    for (sy, sx, id) in seeds
        for dx in -5:5, dy in -5:5
            nx, ny = sx + dx, sy + dy
            if 1 <= nx <= W && 1 <= ny <= H
                grid[ny, nx] = id
            end
        end
    end
    
    # 5. Assigns a mask ID to each pixel based on the center it is associated with
    masks = zeros(Int32, H, W)
    for x in 1:W, y in 1:H
        if iscell[y, x]
            py = clamp(round(Int, p[1, y, x]), 1, H)
            px = clamp(round(Int, p[2, y, x]), 1, W)
            masks[y, x] = grid[py, px]
        end
    end
    
    # 6. Applies error and size filters to remove invalid masks
    if flow_threshold > 0.0
        remove_bad_flow_masks!(masks, dP; threshold=flow_threshold)
    end
    remove_small_masks!(masks; min_size=15)

    return masks
end

"""
    remove_small_masks!(masks::AbstractMatrix{<:Integer}; min_size::Int=15)

    It deletes masks (cells) that have an area smaller than `min_size` pixels.
    It then reorders the remaining IDs so that they are contiguous from 1 to N.
    This replicates the logic of `remove_small_masks` in Cellpose.
"""
function remove_small_masks!(masks::AbstractMatrix{<:Integer}; min_size::Int=15)
    # 1. Using a dict to count the size of each mask (cell)
    sizes = Dict{Int, Int}()
    for val in masks
        if val > 0
            sizes[val] = get(sizes, val, 0) + 1
        end
    end
    
    # 2. Creating a mapping from old IDs to new IDs, where small masks get ID 0 (background)
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
    
    # 3. Applying the new IDs to the masks
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

    Calculates the theoretical (ideal) flows from 2D masks.
    It replicates the logic of `masks_to_flows_gpu` and `_extend_centers_gpu` in Cellpose using heat diffusion.
"""
function masks_to_flows(masks::AbstractMatrix{<:Integer})
    H, W = size(masks)
    T = zeros(Float32, H, W)
    mu = zeros(Float32, 2, H, W)
    
    n_masks = maximum(masks)
    if n_masks == 0
        return mu
    end
    
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
    
    # 1. Finding the centers of each mask by calculating the mean position and then selecting the pixel closest to the mean as the center
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
    
    # 2. Diffusion process to create a "heat map" (T) where the center of the cell 
    # has the highest value and it decreases as we move away from the center
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
        
        for (y, x, m) in valid_pixels
            T[y, x] = T_new[y, x]
        end
    end
    
    # 3. Calculating the flow vectors (mu) as the normalized gradient of the heat map (T)
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

    It deletes masks (cells) whose theoretical flow (calculated from the mask) does not match the flow predicted by the network (dP).
    It calculates the mean squared error between the theoretical flow and the predicted flow for each mask, and removes those whose error exceeds the specified threshold (default 0.4).
"""
function remove_bad_flow_masks!(masks::AbstractMatrix{<:Integer}, dP::AbstractArray{<:AbstractFloat, 3}; threshold=0.4)
    # Calculates the theoretical flow from the masks
    mu = masks_to_flows(masks)
    
    n_masks = maximum(masks)
    errors = zeros(Float32, n_masks)
    counts = zeros(Int, n_masks)
    
    H, W = size(masks)
    
    # Calculates the mean squared error for each cell
    for y in 1:H, x in 1:W
        m = masks[y, x]
        if m > 0
            # Cellpose scales the network flows by 5.0 during inference
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
    
    # Deletes masks whose error exceeds the threshold
    for i in eachindex(masks)
        m = masks[i]
        if m > 0 && errors[m] > threshold
            masks[i] = 0
        end
    end
    
    return masks
end