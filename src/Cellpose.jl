module Cellpose

using Statistics
using ONNXRunTime
using Images

include("Dynamics.jl")

export normalize99, prepare_tensor, segment, compute_masks

"""
    normalize99(img::AbstractArray)

    Normalizes the input image (img) by scaling its pixel values based on the 1st and 99th percentiles. 
    For grayscale images (2D), it applies the same normalization to all pixels. 
    For RGB images (3D), it normalizes each channel independently, ensuring that the color information is preserved while enhancing contrast.
"""
function normalize99(img::AbstractArray)
    out = zeros(Float32, size(img))
    if ndims(img) == 2
        p1 = quantile(vec(img), 0.01)
        p99 = quantile(vec(img), 0.99)
        if p99 - p1 > 1e-6
            out .= (img .- p1) ./ (p99 - p1)
        end
    elseif ndims(img) == 3
        # Normalize each channel independently
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

    Prepares the normalized tile (tile_norm) for input into the ONNX model.
    For grayscale images (2D), it replicates the single channel into three channels to create a 4D tensor of shape (1, 3, H, W).
    For RGB images (3D), it rearranges the dimensions to create a 4D tensor of shape (1, 3, H, W) while preserving the color information.
    This function ensures that the input data is in the correct format expected by the ONNX model, facilitating accurate inference.
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
        error("Image format not supported. Expected 2D (grayscale) or 3D (RGB) array.")
    end
end

"""
    pad_reflect(img::AbstractArray, pad_h::Int, pad_w::Int)
    Pads the input image (img) by reflecting its borders.
    This function is used to ensure that the input image meets the minimum size requirements for processing, while avoiding the introduction of artificial borders that could affect the segmentation results.
    The padding is applied by reflecting the pixel values at the borders of the image, creating a seamless extension that preserves the natural structure of the image.
"""
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

    Native pipeline that replicates exactly Cellpose v4 (cpsam).
"""
function segment(img::AbstractArray, model_path::String; use_gpu::Bool=false)
    println("1. ONNX model initialization")
    if use_gpu
        println("   --> Activating hardware acceleration (CUDA)...")
        model = load_inference(model_path, execution_provider=:cuda)
    else
        model = load_inference(model_path)
    end
    
    TILE_SIZE = 256
    TILE_OVERLAP = 0.1 # 10% overlap 
    STRIDE = round(Int, TILE_SIZE * (1.0 - TILE_OVERLAP)) # 230 pixel
    
    println("1.5 Global image normalization (per channel)")
    img_norm = normalize99(img)
    
    H, W = size(img_norm)[1:2]
    
    # Pad only if the image is smaller than the tile size (256x256)
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
    
    # Precompute the blending window (tapering) to apply to each tile's output for smooth stitching
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
        
        lock(stitching_lock) do
            cellprob_full[y_start:y_end, x_start:x_end] .+= out_tensor[1, 1, :, :] .* window
            dP_full[1, y_start:y_end, x_start:x_end] .+= out_tensor[1, 2, :, :] .* window
            dP_full[2, y_start:y_end, x_start:x_end] .+= out_tensor[1, 3, :, :] .* window
            weight_sum[y_start:y_end, x_start:x_end] .+= window
        end
    end
    
    println("3. Normalizing borders and final cropping")
    weight_sum[weight_sum .== 0] .= 1.0f0
    
    cellprob_full ./= weight_sum
    dP_full[1, :, :] ./= weight_sum
    dP_full[2, :, :] ./= weight_sum
    
    cellprob_crop = cellprob_full[1:H, 1:W]
    dP_crop = dP_full[:, 1:H, 1:W]
    
    println("4. Dynamic calculation of flows (Euler Integration)...")
    masks = compute_masks(dP_crop, cellprob_crop; niter=200, cellprob_threshold=0.0, flow_threshold=0.4)
    
    println("Completed! Found $(maximum(masks)) cells.")
    return masks
end

"""
    segment(img_path::String, model_path::String; use_gpu::Bool=false)

    Wrapper function that allows users to input an image file path directly.
    This function loads the image from the specified path, applies the necessary preprocessing to convert it into a format suitable for the ONNX model, and then calls the main `segment` function to perform the segmentation.
    It supports both grayscale and RGB images, ensuring that the input data is correctly formatted for the segmentation pipeline while providing a user-friendly interface for loading images from disk.
"""
function segment(img_path::String, model_path::String; use_gpu::Bool=false)
    println("Caricamento immagine da: $img_path ...")
    raw_img = load(img_path)
    
    # Converte l'immagine in matrici Float32 digeribili dalla rete neurale
    if eltype(raw_img) <: Colorant
        # Se è a colori (RGB), srotola i canali e riordina le dimensioni
        img_data = Float32.(permutedims(channelview(raw_img), (2, 3, 1)))
    else
        # Se è in scala di grigi
        img_data = Float32.(raw_img)
    end
    
    # Chiama la TUA funzione segment originale (quella vera con gli Array)
    return segment(img_data, model_path; use_gpu=use_gpu)
end

end # module