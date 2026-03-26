using NPZ
using Cellpose

println("1. Loading tensors from Python output (Stage 1 Python)")
dP = npzread(joinpath(@__DIR__, "..", "test_data", "05_full_dP.npy"))
cellprob = npzread(joinpath(@__DIR__, "..", "test_data", "05_full_cellprob.npy"))
python_masks = npzread(joinpath(@__DIR__, "..", "test_data", "06_full_masks.npy"))

println("\n2. Running Julia's compute_masks with the Python tensors")

# check the time taken by compute_masks to see if it's reasonable
@time julia_masks = Cellpose.compute_masks(dP, cellprob; niter=200, cellprob_threshold=0.0)

println("\n3. Analysis of results")
# The background is labeled as 0, so we subtract 1 to get the number of cells
n_cells_python = length(unique(python_masks)) - 1
n_cells_julia = length(unique(julia_masks)) - 1

println("-> Cells found by Python: ", n_cells_python)
println("-> Cells found by Julia:  ", n_cells_julia)

if abs(n_cells_python - n_cells_julia) < 10
    println("\nThe masks are reasonably close in terms of cell count")
else
    println("\nThe number of the masks differs between Python and Julia")
end