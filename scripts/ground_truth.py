import numpy as np
import torch
import cv2
from cellpose import models, transforms, io

# Image path of the original image to be processed.
img_path = "../resources/test_image.tif"
try:
    img_original = io.imread(img_path)
except:    
    # If the image is not found, generate a dummy image (512x512) to avoid crashing the script,
    # but the final masks will likely be empty (zero cells found).
    print("Image not found, using a dummy 512x512 image...")
    img_original = np.random.rand(512, 512).astype(np.float32)

np.save("01_img_original.npy", img_original)
print("1. Original image saved (01_img_original.npy)")

# Using the pretrained Cellpose SAM model (Python) to get the ground truth for the ONNX test in Julia
model = models.CellposeModel(pretrained_model='cpsam', gpu=True)

# Run the model on the original image to get the masks, flows, and styles
masks, flows, styles = model.eval(img_original, diameter=None, channels=[0,0])

# Saving the full flows and cell probability maps for the entire field of view (not just the tile) as ground truth for Julia
# In Cellpose, flows[1] are the gradients (Y, X) and flows[2] is the cell probability map
np.save("05_full_dP.npy", flows[1]) # Gradient maps (Y,X)
np.save("05_full_cellprob.npy", flows[2]) # Cell probability map
np.save("06_full_masks.npy", masks) # Full masks for the entire field of view (not just the tile) - this is the "ground truth" for Julia to compare against
print("2. Final outputs saved (Masks and Gradients for the entire field of view)")

# ONNX requires a fixed input size, so we need to create a tile of 256x256 from the original image. We will take a central crop of 256x256 pixels.
h, w = img_original.shape[:2]
ch = h//2; cw = w//2
# the tile is centered on the middle of the image, but you can choose any other region as long as it's 256x256
tile = img_original[ch-128:ch+128, cw-128:cw+128]

# Normalizing the tile to the range [0,1] using Cellpose's normalize99 function, which is what the model expects as input
tile_norm = transforms.normalize99(tile)

# Cellpose models expect 3 channels (even if the input is grayscale), so we need to stack the single channel 3 times if it's a 2D image
if tile_norm.ndim == 2:
    tile_norm = np.stack((tile_norm, tile_norm, tile_norm), axis=-1)

# Now we have a tile of shape (256, 256, 3) that we can feed into the ONNX model. We need to transpose it to (3, 256, 256) and add a batch dimension to make it (1, 3, 256, 256).
tile_tensor = tile_norm.transpose(2, 0, 1) # From (H,W,C) to (C,H,W)
tile_tensor = tile_tensor[np.newaxis, ...].astype(np.float32) # Add batch dimension

np.save("02_tile_256_input.npy", tile_tensor)
print("3. Tile of input for ONNX saved (02_tile_256_input.npy)")

# Now we will run the same tile through the original PyTorch model to get the "ground truth" output that we will compare against in Julia. This is important because the ONNX export and inference in Julia should ideally produce the same output as the original PyTorch model for the same input tile.
pytorch_net = model.net
pytorch_net.eval()
pytorch_net = pytorch_net.float()

with torch.no_grad():
    # Creating the tensor input for PyTorch from the tile we created, and moving it to the same device as the model (CPU or GPU)
    input_pt = torch.from_numpy(tile_tensor).to(model.device)
    
    output_pt, _ = pytorch_net(input_pt) 
    
    raw_output = output_pt.cpu().numpy()

np.save("03_tile_raw_output.npy", raw_output)
print("4. Raw output of the network saved (03_tile_raw_output.npy)")

print("\n✅ Ground truth generation completed successfully!")