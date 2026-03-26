using NPZ
using Cellpose

println("=== Complete Pipeline Test (with Discrepancy Analysis) ===")

# loading image from Python npy file
img_path = joinpath(@__DIR__, "..", "test_data", "01_img_original.npy")
img = npzread(img_path)

# loading the cpsam ONNX model path
model_path = joinpath(@__DIR__, "..", "models", "cpsam.onnx")

println("\nRunning the complete segmentation pipeline in Julia...")
masks = Cellpose.segment(img, model_path)

println("\nLoading the tensors saved by Python...")
dP_python = npzread(joinpath(@__DIR__, "..", "test_data", "05_full_dP.npy"))
cellprob_python = npzread(joinpath(@__DIR__, "..", "test_data", "05_full_cellprob.npy"))
python_masks = npzread(joinpath(@__DIR__, "..", "test_data", "06_full_masks.npy"))

println("\n--- Diagnosis ---")
n_cells_julia = maximum(masks)
n_cells_python = maximum(python_masks)
println("Cells found by Julia: $n_cells_julia")
println("Cells found by Python: $n_cells_python")

println("\nRecomputing masks with Python tensors...")
masks_from_python_tensors = Cellpose.compute_masks(dP_python, cellprob_python; niter=200, cellprob_threshold=0.0, flow_threshold=0.4)
println("Cells found by Julia (with Python data): ", maximum(masks_from_python_tensors))