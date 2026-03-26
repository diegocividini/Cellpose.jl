using ONNXRunTime
using NPZ

println("1. Loading ONNX model...")

model_path = joinpath(@__DIR__, "models", "cpsam.onnx") # loading cpsam model
model = load_inference(model_path)
println("Successfully loaded ONNX model from: ", model_path)

println("\n2. Loading test data...")
# Load the pre-processed input tile from Python
input_tensor = npzread(joinpath(@__DIR__, "test_data", "02_tile_256_input.npy"))

# Load the raw output saved from Python for comparison
python_output = npzread(joinpath(@__DIR__, "test_data", "03_tile_raw_output.npy"))

println("\n3. Execution of inference in Julia (ONNX Runtime)...")
# ONNXRunTime expects a dictionary of inputs, where the key is the name of the input defined in the ONNX model
inputs = Dict("input_image" => input_tensor)
julia_output_dict = model(inputs)

# Extract the actual output tensor from the dictionary (the key depends on how the ONNX model is defined)
# The output is named "flows_and_probs" in the ONNX model
julia_output = julia_output_dict["flows_and_probs"]

println("\n4. Results Analysis")
# Calculate the maximum absolute difference between the Julia output and the Python output
max_diff = maximum(abs.(julia_output .- python_output))

println("Dimensions output Julia: ", size(julia_output))
println("Dimensions output Python: ", size(python_output))
println("Maximum numerical difference: ", max_diff)

if max_diff < 1e-4
    println("\nSuccess! The ONNX inference in Julia produces results that are numerically very close to the Python output.")
else
    println("\nMmm, there is a numerical discrepancy. We need to investigate.")
end