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

base_path = config.CBIS_PATH

# Read the selected .dcm files
# Paths of selected .dcm files
selected_paths = [os.path.join(base_path, "Mass-Test_P_00016_LEFT_CC_full", "1-1.dcm"),
                  os.path.join(base_path, "Mass-Test_P_00016_LEFT_MLO_full", "1-1.dcm"),
                  os.path.join(base_path, "Mass-Test_P_00017_LEFT_CC_full", "1-1.dcm"),
                  os.path.join(base_path, "Mass-Test_P_00017_LEFT_MLO_full", "1-1.dcm"),
                  os.path.join(base_path, "Mass-Test_P_00032_RIGHT_CC_full", "1-1.dcm")]

ds = [pydicom.dcmread(selected_paths[i]) for i in range(len(selected_paths))]

arr = [_ds.pixel_array for _ds in ds]

# Paths of corresponding masks
mask_paths = [os.path.join(base_path, "Mass-Test_P_00016_LEFT_CC_mask_1", "1-1.dcm"),
              os.path.join(base_path, "Mass-Test_P_00016_LEFT_MLO_mask_1", "1-1.dcm"),
              os.path.join(base_path, "Mass-Test_P_00017_LEFT_CC_mask_1", "1-2.dcm"),
              os.path.join(base_path, "Mass-Test_P_00017_LEFT_MLO_mask_1", "1-2.dcm"),
              os.path.join(base_path, "Mass-Test_P_00032_RIGHT_CC_mask_1", "1-2.dcm")]

ds_masks = [pydicom.dcmread(mask_paths[i]) for i in range(len(mask_paths))]

arr_masks = [_ds.pixel_array for _ds in ds_masks]



#Visualizing

# Ensure the directory exists or create it
save_dir = "../data/raw_data/visualisations_for_slides/"
os.makedirs(save_dir, exist_ok=True)


# Visualise the original images
fig, ax = plt.subplots(nrows=1, ncols=len(selected_paths), figsize=(22, 5))

for i in range(len(selected_paths)):
    ax[i].imshow(arr[i], cmap="gray")
    ax[i].set_title(f"{ds[i].PatientID}")

plt.tight_layout()
plt.savefig(fname="../data/raw_data/visualisations_for_slides/raw.png", dpi=300)
# plt.show()



# Visualise the corresponding mask
fig, ax = plt.subplots(nrows=1, ncols=len(mask_paths), figsize=(22, 5))

for i in range(len(mask_paths)):
    ax[i].imshow(arr_masks[i], cmap="gray")
    ax[i].set_title(f"{ds_masks[i].PatientID}")

plt.tight_layout()
plt.savefig(fname="../data/raw_data/visualisations_for_slides/raw_masks.png", dpi=300)
# plt.show()



# Understanding the images

for a in arr:
    print("Shape:", a.shape)
    print("Dimensions:", a.ndim)
    print("Type:", type(a))
    print("Data type:", a.dtype)
    print(f"min value, max value: {a.min(), a.max()}")
    print("---")