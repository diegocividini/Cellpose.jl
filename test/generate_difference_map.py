import numpy as np
import matplotlib.pyplot as plt
from PIL import Image # Assumendo che tu abbia Pillow installato

mask_py = np.array(Image.open("/Users/diegocividini/Desktop/new_image/maschera_wrapper.tif"))
mask_jl = np.array(Image.open("/Users/diegocividini/Desktop/new_image/maschera_native.tif"))

py_bin = (mask_py > 0).astype(np.uint8)
jl_bin = (mask_jl > 0).astype(np.uint8)


py_bin = (mask_py > 0).astype(np.uint8)
jl_bin = (mask_jl > 0).astype(np.uint8)

# AGREEMENT (Blue): Both Python and Julia found a cell (1 in both)
agreement = (py_bin == 1) & (jl_bin == 1)

# PYTHON ONLY (Green): Python found a cell, Julia did not (1 in Python, 0 in Julia)
py_only = (py_bin == 1) & (jl_bin == 0)

# JULIA ONLY (Red): Julia found a cell, Python did not (1 in Julia, 0 in Python)
jl_only = (py_bin == 0) & (jl_bin == 1)

diff_map = np.zeros((agreement.shape[0], agreement.shape[1], 3), dtype=np.uint8)

diff_map[agreement] = [0, 0, 255]      # BLUE = AGREEMENT (Both found a cell)
diff_map[py_only]   = [0, 255, 0]      # GREEN = PYTHON ONLY
diff_map[jl_only]   = [255, 0, 0]      # RED = JULIA ONLY

plt.imsave('difference_map_presence_NEW.png', diff_map)