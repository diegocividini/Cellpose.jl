using ONNXRunTime
using NPZ

println("1. Caricamento del modello ONNX...")
# Carichiamo il modello. (Assicurati che il percorso sia corretto)
model_path = joinpath(@__DIR__, "models", "cpsam.onnx")
model = load_inference(model_path)
println("Modello caricato con successo!")

println("\n2. Caricamento dei dati di test...")
# Carichiamo il tile di input pre-processato da Python
input_tensor = npzread(joinpath(@__DIR__, "test_data", "02_tile_256_input.npy"))

# Carichiamo l'output grezzo salvato da Python per fare il confronto
python_output = npzread(joinpath(@__DIR__, "test_data", "03_tile_raw_output.npy"))

println("\n3. Esecuzione inferenza in Julia (ONNX Runtime)...")
# ONNXRunTime si aspetta un dizionario dove la chiave è il nome dell'input 
# che abbiamo definito durante l'esportazione in Python ('input_image')
inputs = Dict("input_image" => input_tensor)
julia_output_dict = model(inputs)

# Estraiamo il tensore risultante usando il nome dell'output ('flows_and_probs')
julia_output = julia_output_dict["flows_and_probs"]

println("\n4. Confronto dei risultati (La resa dei conti!)")
# Calcoliamo la differenza massima assoluta tra i numeri di Julia e quelli di Python
max_diff = maximum(abs.(julia_output .- python_output))

println("Dimensioni output Julia: ", size(julia_output))
println("Dimensioni output Python: ", size(python_output))
println("Differenza numerica massima: ", max_diff)

if max_diff < 1e-4
    println("\nSUCCESSO TOTALE! L'inferenza in Julia è matematicamente identica a PyTorch!")
else
    println("\nMmm, c'è una discrepanza numerica. Dobbiamo indagare.")
end