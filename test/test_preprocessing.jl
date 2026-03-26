using NPZ
using Statistics
using Cellpose 

println("1. Loading the original image from Python npy file")
img_original = npzread(joinpath(@__DIR__, "..", "test_data", "01_img_original.npy"))

println("\n2. Extracting the central tile")
sz = size(img_original)
H = sz[1]
W = sz[2]
ch = H ÷ 2
cw = W ÷ 2

# Julia requires 3 dimensions
if length(sz) == 2
    tile = img_original[ch-128+1:ch+128, cw-128+1:cw+128]
else
    tile = img_original[ch-128+1:ch+128, cw-128+1:cw+128, :] 
end

println("\n3. Normalization 99 in Julia...")
tile_norm_julia = Cellpose.normalize99(tile)

println("\n4. Preparing the 4D tensor for ONNX...")
tensor_julia = Cellpose.prepare_tensor(tile_norm_julia)

println("\n5. Comparison with the tensor prepared by Python...")
tensor_python = npzread(joinpath(@__DIR__, "..", "test_data", "02_tile_256_input.npy"))

max_diff = maximum(abs.(tensor_julia .- tensor_python))

println("Julia's tensor dimensions: ", size(tensor_julia))
println("Python's tensor dimensions: ", size(tensor_python))
println("Maximum difference: ", max_diff)

if max_diff < 1e-4
    println("\nPerfect pre-processing match")
else
    println("\nDiscrepancy in pre-processing")
end