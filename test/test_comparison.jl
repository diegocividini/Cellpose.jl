using Images, Statistics, Printf, ColorTypes

# ==============================================================================
# CONFIGURAZIONE PERCORSI
# ==============================================================================
# Modifica questi percorsi per puntare ai tuoi file reali
PATH_PYTHON = "/Users/diegocividini/Desktop/vm/maschera_wrapper.tif"
PATH_JULIA  = "/Users/diegocividini/Desktop/vm/maschera_native.tif"
OUTPUT_DIFF = "/Users/diegocividini/Desktop/vm/differenze_map.png"

# ==============================================================================
# FUNZIONI DI SUPPORTO
# ==============================================================================

"""
Carica una maschera da file TIFF gestendo correttamente i tipi di dati.
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
Crea una mappa delle differenze per visualizzazione.
"""
function create_diff_map(masks_py, masks_jl)
    H, W = size(masks_py)
    diff_img = zeros(RGB{N0f8}, H, W)
    
    for y in 1:H, x in 1:W
        p = masks_py[y, x]
        j = masks_jl[y, x]
        
        if p == 0 && j == 0
            diff_img[y, x] = RGB{N0f8}(0.0, 0.0, 0.0) # Nero: Sfondo
        elseif p == j
            diff_img[y, x] = RGB{N0f8}(1.0, 1.0, 0.0) # Giallo: Accordo perfetto (ID stesso)
        elseif p > 0 && j == 0
            diff_img[y, x] = RGB{N0f8}(0.0, 1.0, 0.0) # Verde: Solo Python
        elseif p == 0 && j > 0
            diff_img[y, x] = RGB{N0f8}(1.0, 0.0, 0.0) # Rosso: Solo Julia
        else
            diff_img[y, x] = RGB{N0f8}(0.0, 0.0, 1.0) # Blu: Disaccordo ID (entrambe > 0 ma ID diversi)
        end
    end
    return diff_img
end

# ==============================================================================
# ESECUZIONE TEST
# ==============================================================================

function main()
    println("==================================================")
    println("🔬 CONFRONTO QUANTITATIVO: PYTHON vs JULIA NATIVE")
    println("==================================================")

    try
        masks_py = load_mask(PATH_PYTHON)
        masks_jl = load_mask(PATH_JULIA)

        # 1. Statistiche di base
        n_cells_py = maximum(masks_py)
        n_cells_jl = maximum(masks_jl)
        diff_cells = abs(n_cells_py - n_cells_jl)

        println("\n📊 CONTEGGIO CELLULE:")
        println("  Python Wrapper : $n_cells_py")
        println("  Julia Native   : $n_cells_jl")
        println("  Differenza     : $diff_cells ($(round(diff_cells/n_cells_py*100, digits=2))%)")

        # 2. Analisi delle Aree Globali
        aree_py = [sum(masks_py .== i) for i in 1:n_cells_py]
        aree_jl = [sum(masks_jl .== i) for i in 1:n_cells_jl]

        println("\n📏 DISTRIBUZIONE AREE (pixel):")
        println("  Python - Media: $(round(mean(aree_py), digits=1)) ± $(round(std(aree_py), digits=1))")
        println("           Mediana: $(median(aree_py)), Min: $(minimum(aree_py)), Max: $(maximum(aree_py))")
        
        println("  Julia  - Media: $(round(mean(aree_jl), digits=1)) ± $(round(std(aree_jl), digits=1))")
        println("           Mediana: $(median(aree_jl)), Min: $(minimum(aree_jl)), Max: $(maximum(aree_jl))")

        # 3. IoU Binario (Metrica fondamentale di qualità)
        is_cell_py = masks_py .> 0
        is_cell_jl = masks_jl .> 0
        
        intersection = sum(is_cell_py .& is_cell_jl)
        union = sum(is_cell_py .| is_cell_jl)
        iou_binary = intersection / union
        
        println("\n🎯 QUALITÀ SEGMENTAZIONE (Pixel Accuracy & IoU):")
        @printf("  Pixel Accuracy: %.2f%%\n", (sum(is_cell_py .== is_cell_jl) / length(masks_py)) * 100)
        @printf("  IoU Binario:    %.2f%% (Soil/Soil sovrapposizione)\n", iou_binary * 100)

        # 4. Distribuzione Aree per Range (Nuovo)
        println("\n📊 CELLULE PER DIMENSIONE (Range px²):")
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

        # 5. Analisi Errori (Nuovo)
        println("\n❌ ANALISI DISACCORDI:")
        false_neg = sum(is_cell_py .& .!is_cell_jl) # Python vede, Julia no
        false_pos = sum(.!is_cell_py .& is_cell_jl) # Julia vede, Python no
        
        println("  Falsi Negativi (Pixel persi da Julia): $false_neg")
        println("  Falsi Positivi (Pixel extra da Julia): $false_pos")
        
        # Stima dimensione media errori
        if false_neg > 0
            # Calcolo approssimativo basato sui pixel persi e il numero di cellule perse
            # Non avendo un matching 1:1 esatto, usiamo la media dei pixel persi divisa la stima di cellule perse
            avg_loss_per_cell = false_neg / max(1, diff_cells)
            println("  ➤ Stima dimensione cellule 'perse': ~$(round(Int, avg_loss_per_cell)) px/cellula (media teorica)")
        end
        
        if false_pos > 0
            avg_gain_per_cell = false_pos / max(1, diff_cells)
            println("  ➤ Stima dimensione cellule 'extra'  : ~$(round(Int, avg_gain_per_cell)) px/cellula (media teorica)")
        end

        # 6. Generazione Mappa Differenze
        println("\n🎨 Generazione mappa differenze...")
        diff_map = create_diff_map(masks_py, masks_jl)
        save(OUTPUT_DIFF, diff_map)
        println("  Mappa salvata in: $OUTPUT_DIFF")
        println("  Legenda: Giallo=Accordo ID | Verde=Solo Python | Rosso=Solo Julia | Blu=Disaccordo ID")

        # VERDETTO FINALE
        println("\n==================================================")
        if iou_binary > 0.80
            println("✅ ESITO: ECCELLENTE. Le versioni sono altamente correlate.")
        elseif iou_binary > 0.60
            println("⚠️ ESITO: BUONO. Ci sono differenze sistematiche (probabilmente soglia/fusione), ma la struttura è preservata.")
        else
            println("❌ ESITO: DA RIVEDERE. Le differenze sono significative.")
        end
        println("==================================================")

    catch e
        println("\n❌ ERRORE DURANTE L'ESECUZIONE:")
        println(e)
        println("\nVerifica che i percorsi dei file siano corretti nel codice.")
    end
end

# Avvio dello script
main()