using Images, Statistics, Printf, ColorTypes

# ==============================================================================
# PATHS
# ==============================================================================
PATH_PYTHON = "/Users/diegocividini/Desktop/new_image/maschera_wrapper.tif"
PATH_JULIA  = "/Users/diegocividini/Desktop/new_image/maschera_native.tif"
OUTPUT_DIFF = "./difference_map_NEW.png"

# ==============================================================================
# SUPPORT FUNCTIONS
# ==============================================================================

"""
    load_mask(path::String)

    Uploads a mask image and converts it to an integer matrix. Handles various pixel types and color formats.
"""
function load_mask(path::String)
    println("Caricamento: $path")
    img = load(path)
    
    if eltype(img) <: ColorTypes.Gray
        raw = channelview(img)
        if eltype(raw) <: FixedPointNumbers.N0f16
            return Int.(reinterpret(UInt16, raw))
        elseif eltype(raw) <: FixedPointNumbers.N0f8
            return Int.(reinterpret(UInt8, raw))
        else
            @warn "Tipo di pixel non standard rilevato: $(eltype(raw)). Tentativo di conversione diretta."
            return Int.(round.(raw .* 65535))
        end
    else
        return Int.(img)
    end
end

"""
    masks_to_flows_gpu(masks::AbstractMatrix{<:Integer})

    Calculates the theoretical (ideal) flows from 2D masks using GPU. 
    This is a reference function for comparing results with the CPU version.
"""
function create_diff_map(masks_py, masks_jl)
    H, W = size(masks_py)
    diff_img = zeros(RGB{N0f8}, H, W)
    
    for y in 1:H, x in 1:W
        p = masks_py[y, x]
        j = masks_jl[y, x]
        
        if p == 0 && j == 0
            diff_img[y, x] = RGB{N0f8}(0.0, 0.0, 0.0) # Black: Background
        elseif p == j
            diff_img[y, x] = RGB{N0f8}(1.0, 1.0, 0.0) # Yellow: Perfect Agreement (Same ID)
        elseif p > 0 && j == 0
            diff_img[y, x] = RGB{N0f8}(0.0, 1.0, 0.0) # Green: Python Only
        elseif p == 0 && j > 0
            diff_img[y, x] = RGB{N0f8}(1.0, 0.0, 0.0) # Red: Julia Only
        else
            diff_img[y, x] = RGB{N0f8}(0.0, 0.0, 1.0) # Blue: ID Mismatch (both > 0 but different IDs)
        end
    end
    return diff_img
end

# ==============================================================================
# TEST COMPARISON FUNCTION
# ==============================================================================

function main()
    println("==================================================")
    println("QUANTITATIVE TEST: PYTHON vs JULIA")
    println("==================================================")

    try
        masks_py = load_mask(PATH_PYTHON)
        masks_jl = load_mask(PATH_JULIA)

        n_cells_py = maximum(masks_py)
        n_cells_jl = maximum(masks_jl)
        diff_cells = abs(n_cells_py - n_cells_jl)

        println("\nCELLS COUNT:")
        println("  Python         : $n_cells_py")
        println("  Julia Native   : $n_cells_jl")
        println("  Differenza     : $diff_cells ($(round(diff_cells/n_cells_py*100, digits=2))%)")

        aree_py = [sum(masks_py .== i) for i in 1:n_cells_py]
        aree_jl = [sum(masks_jl .== i) for i in 1:n_cells_jl]

        println("\n📏 AREA DISTRIBUTION (pixel):")
        println("  Python - Average: $(round(mean(aree_py), digits=1)) ± $(round(std(aree_py), digits=1))")
        println("           Median : $(median(aree_py)), Min: $(minimum(aree_py)), Max: $(maximum(aree_py))")
        
        println("  Julia  - Average: $(round(mean(aree_jl), digits=1)) ± $(round(std(aree_jl), digits=1))")
        println("           Median : $(median(aree_jl)), Min: $(minimum(aree_jl)), Max: $(maximum(aree_jl))")

        is_cell_py = masks_py .> 0
        is_cell_jl = masks_jl .> 0
        
        intersection = sum(is_cell_py .& is_cell_jl)
        union = sum(is_cell_py .| is_cell_jl)
        iou_binary = intersection / union
        
        println("\n🎯 SEGMENTATION QUALITY (Pixel Accuracy & IoU):")
        @printf("  Pixel Accuracy: %.2f%%\n", (sum(is_cell_py .== is_cell_jl) / length(masks_py)) * 100)
        @printf("  Binary IoU:    %.2f%% (Soil/Soil overlap)\n", iou_binary * 100)

        println("\n CELLS PER DIMENSION (Range px²):")
        ranges = [(0, 500), (500, 1000), (1000, 2000), (2000, 5000), (5000, Inf)]
        println("  Range (px²)      | Python | Julia | Diff")
        println("  ------------------------------------------")
        for (r_min, r_max) in ranges
            count_py = sum((aree_py .>= r_min) .& (aree_py .< r_max))
            count_jl = sum((aree_jl .>= r_min) .& (aree_jl .< r_max))
            diff = count_py - count_jl
            range_str = "$r_min - $(r_max == Inf ? "∞" : r_max)"
            println("  $(lpad(range_str, 14)) | $(lpad(count_py, 6)) | $(lpad(count_jl, 5)) | $(diff > 0 ? "-" : "+")$(abs(diff))")
        end

        println("\n ANALYSIS OF ERRORS:")
        false_neg = sum(is_cell_py .& .!is_cell_jl) # Python vede, Julia no
        false_pos = sum(.!is_cell_py .& is_cell_jl) # Julia vede, Python no
        
        println("  False Negatives (Pixels lost by Julia) : $false_neg")
        println("  False Positives (Extra pixels by Julia): $false_pos")
        
        if false_neg > 0
            avg_loss_per_cell = false_neg / max(1, diff_cells)
            println("  ➤ Estimated size of 'lost' cells    : ~$(round(Int, avg_loss_per_cell)) px/cell (theoretical average)")
        end
        
        if false_pos > 0
            avg_gain_per_cell = false_pos / max(1, diff_cells)
            println("  ➤ Estimated size of 'extra' cells   : ~$(round(Int, avg_gain_per_cell)) px/cell (theoretical average)")
        end

        println("\nGenerating difference map...")
        diff_map = create_diff_map(masks_py, masks_jl)
        save(OUTPUT_DIFF, diff_map)
        println("  Difference map saved in: $OUTPUT_DIFF")
        println("  Legend: Yellow=ID Agreement | Green=Python Only | Red=Julia Only | Blue=ID Disagreement | Black=Background")

        println("\n==================================================")
        if iou_binary > 0.80
            println("OUTCOME: EXCELLENT. Versions are highly related")
        elseif iou_binary > 0.60
            println("OUTCOME: GOOD. There are systematic differences (probably threshold/fusion), but the structure is preserved.")
        else
            println("OUTCOME: TO REVIEW. The differences are significant.")
        end
        println("==================================================")

    catch e
        println("\nERROR OCCURRED IN EXECUTION:")
        println(e)
        println("\nVerify that the file paths are correct in the code.")
    end
end

main()