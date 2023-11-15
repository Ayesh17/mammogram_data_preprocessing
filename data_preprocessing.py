#imports
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
import pydicom
import os
import skimage
from pathlib import Path

import config


def ShowHist255(img, ignore_zero=False):
    hist, bin_edges = np.histogram(img, bins=255, density=False)

    if ignore_zero:
        plt.plot(bin_edges[1:-1], hist[1:])
    else:
        plt.plot(bin_edges[0:-1], hist)

    plt.show()



# Read the selected .dcm files
# Paths of selected .dcm files
selected_paths = os.path.join(config.CBIS_PATH, "Calc-Training_P_00288_RIGHT_CC_1","09-06-2017-DDSM-NA-38470","1.000000-ROI mask images-13046")

ds = [pydicom.dcmread(selected_paths[i]) for i in range(len(selected_paths))]

arr = [_ds.pixel_array for _ds in ds]

# # Paths of corresponding masks
# mask_paths = ["../data/raw_data/Mass/Train/Mass-Training_P_00001_LEFT_CC_MASK_1.dcm",
#               "../data/raw_data/Mass/Train/Mass-Training_P_00009_RIGHT_MLO_MASK_1.dcm",
#               "../data/raw_data/Mass/Train/Mass-Training_P_00572_RIGHT_CC_MASK_1.dcm",
#               "../data/raw_data/Mass/Train/Mass-Training_P_00146_RIGHT_CC_MASK_1.dcm",
#               "../data/raw_data/Mass/Train/Mass-Training_P_00710_LEFT_MLO_MASK_1.dcm",
#               "../data/raw_data/Mass/Train/Mass-Training_P_01343_LEFT_CC_MASK_1.dcm",
#               "../data/raw_data/Mass/Train/Mass-Training_P_01343_LEFT_CC_MASK_2.dcm"]

# ds_masks = [pydicom.dcmread(mask_paths[i]) for i in range(len(mask_paths))]
#
# arr_masks = [_ds.pixel_array for _ds in ds_masks]

# Visualise the original images
# Plot together
fig, ax = plt.subplots(nrows=1, ncols=len(selected_paths), figsize=(22, 5))

for i in range(len(selected_paths)):
    ax[i].imshow(arr[i], cmap="gray")
    ax[i].set_title(f"{ds[i].PatientID}")

plt.tight_layout()
plt.savefig(fname="../data/raw_data/visualisations_for_slides/raw.png", dpi=300)