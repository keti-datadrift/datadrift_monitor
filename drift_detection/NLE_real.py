import os
import re
from PIL import Image
import numpy as np
from pathlib import Path
import numpy as np
from NLE import noise_lev_est

#IMG_DIR = Path("D:/Datasets/tid2008/noisy/bmp")
IMG_DIR = Path("D:/Datasets/temp/1-xtumfdrpkgoqykstesco")
PATCH_M1 = 7
DELTA = 0.8
MAX_ITERS = 3
DECIM = 0


files = [f for f in os.listdir(IMG_DIR)]

noise_levels = []
thresholds = []
patch_counts = []
timestamps = []

for fname in files:
    path = os.path.join(IMG_DIR, fname)
    img = np.array(Image.open(path).convert("RGB"), dtype=np.float32)
    noise_level, threshold, n_patches = noise_lev_est(img, show=False, w=PATCH_M1, delta=DELTA, decim=DECIM, conf=1 - 1e-6, itr=MAX_ITERS)
    
    noise_levels.append(noise_level)
    thresholds.append(threshold)
    patch_counts.append(n_patches)

    print(f"{fname}: noise={np.mean(noise_level):.4f}, threshold={np.mean(threshold):.4f}, patches={np.mean(n_patches):.0f}")