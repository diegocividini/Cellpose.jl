import numpy as np
import torch
import cv2
from cellpose import models, transforms, io

# --- 1. CARICAMENTO IMMAGINE ---
# Inserisci qui il percorso di una tua immagine reale (es. 512x512 o simile)
img_path = "../h1.tif" # <--- CAMBIA QUESTO
try:
    img_original = io.imread(img_path)
except:
    # Se non metti un'immagine, genero un gradiente fittizio per non far crashare lo script,
    # ma le maschere finali saranno probabilmente vuote (zero cellule trovate).
    print("Immagine non trovata, uso un'immagine fittizia 512x512...")
    img_original = np.random.rand(512, 512).astype(np.float32)

np.save("01_img_original.npy", img_original)
print("1. Immagine originale salvata (01_img_original.npy)")

# --- 2. PIPELINE COMPLETA (Il traguardo finale) ---
# Usiamo l'API standard di Python per avere i risultati perfetti
model = models.CellposeModel(pretrained_model='cpsam', gpu=True)

# Eseguiamo l'inferenza. channels=[0,0] dice a Cellpose di processare in scala di grigi.
masks, flows, styles = model.eval(img_original, diameter=None, channels=[0,0])

# Salviamo i gradienti (dP) e la probabilità delle cellule
# In Cellpose, flows[1] sono i gradienti (Y, X) e flows[2] è la mappa di probabilità
np.save("05_full_dP.npy", flows[1])
np.save("05_full_cellprob.npy", flows[2])
np.save("06_full_masks.npy", masks)
print("2. Output finali salvati (Maschere e Gradienti di tutto l'intero campo visivo)")

# --- 3. ISOLIAMO UN SINGOLO TILE (La prova per l'ONNX in Julia) ---
# ONNX richiede un riquadro fisso 256x256x3. Ritagliamo il centro della tua immagine
h, w = img_original.shape[:2]
ch = h//2; cw = w//2
# Prendiamo un riquadro di 256x256 dal centro
tile = img_original[ch-128:ch+128, cw-128:cw+128]

# Normalizziamo il tile esattamente come fa Cellpose (percentili 1-99)
tile_norm = transforms.normalize99(tile)

# SAM vuole sempre 3 canali (RGB). Se l'immagine è in scala di grigi, la duplichiamo su 3 canali
if tile_norm.ndim == 2:
    tile_norm = np.stack((tile_norm, tile_norm, tile_norm), axis=-1)

# PyTorch/ONNX vogliono il formato (Batch, Canali, Altezza, Larghezza) -> (1, 3, 256, 256)
tile_tensor = tile_norm.transpose(2, 0, 1) # Da (H,W,C) a (C,H,W)
tile_tensor = tile_tensor[np.newaxis, ...].astype(np.float32) # Aggiunge la dimensione Batch

np.save("02_tile_256_input.npy", tile_tensor)
print("3. Tile di input per ONNX salvato (02_tile_256_input.npy)")

# --- 4. OUTPUT RAW DELLA RETE ---
# Passiamo il tile al modello PyTorch per vedere i numeri grezzi che escono
pytorch_net = model.net
pytorch_net.eval()
pytorch_net = pytorch_net.float()

with torch.no_grad():
    # Creiamo il tensore e LO SPOSTIAMO sulla stessa memoria del modello (MPS/GPU)
    input_pt = torch.from_numpy(tile_tensor).to(model.device)
    
    # Ora calcoliamo l'output
    output_pt, _ = pytorch_net(input_pt) 
    
    # Riportiamo il risultato sulla CPU per poterlo salvare con Numpy
    raw_output = output_pt.cpu().numpy()

np.save("03_tile_raw_output.npy", raw_output)
print("4. Output grezzo della rete salvato (03_tile_raw_output.npy)")

print("\n✅ Fase 1 completata con successo! Hai creato i file di Ground Truth.")