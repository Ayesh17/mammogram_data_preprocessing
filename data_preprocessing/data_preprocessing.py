import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
import pydicom
import os
import skimage
import pydicom
from pydicom.pixel_data_handlers.util import convert_color_space

import config
from removing_artifacts import MinMaxNormalise, CropBorders, Binarisation, OwnGlobalBinarise, OpenMask, XLargestBlobs, \
    InvertMask, HorizontalFlip, clahe, ApplyMask, Pad



cbis_path = config.CBIS_PATH
output_path = os.path.join(config.CBIS_BASE_PATH, "output")

if not os.path.exists(output_path):
    os.makedirs(output_path)


# Initialize empty lists to store image and mask paths
image_paths = []
mask_paths = []

for root, dirs, files in os.walk(cbis_path):
    for dir_name in dirs:
        subfolder_path = os.path.join(root, dir_name)
        # Check if the subfolder name ends with "full" for images or "mask" for masks
        if dir_name.endswith("_full"):
            # Collect image paths
            for file_name in os.listdir(subfolder_path):
                if file_name.endswith(".dcm"):
                    image_paths.append(os.path.join(subfolder_path, file_name))
        else:
            # Collect mask paths
            for file_name in os.listdir(subfolder_path):
                if file_name.endswith(".dcm"):
                    mask_paths.append(os.path.join(subfolder_path, file_name))

# Read the selected .dcm files
ds = [pydicom.dcmread(path) for path in image_paths]
arr = [_ds.pixel_array for _ds in ds]

# Read the corresponding masks
ds_masks = [pydicom.dcmread(path) for path in mask_paths]
arr_masks = [_ds.pixel_array for _ds in ds_masks]


# ## Understanding the images
for a in arr:
    print("Shape:", a.shape)
    print("Dimensions:", a.ndim)
    print("Type:", type(a))
    print("Data type:", a.dtype)
    print(f"min value, max value: {a.min(), a.max()}")
    print("---")


# Normalise to range [0, 1]
arr_norm = [MinMaxNormalise(a) for a in arr]



# Step 1.1 - Initial crop around the image boundaries
cropped_img_list = []

for i in range(len(arr_norm)):
    cropped_img = CropBorders(img=arr_norm[i])
    cropped_img_list.append(cropped_img)



# image binarization
# th1_list = []
# th2_list = []
# th3_list = []
# th4_list = []
# #
# # # Plot binarised images
# # # fig, ax = plt.subplots(nrows=5, ncols=5, figsize=(22, 25))
# #
# for i in range(len(arr_norm)):
#
#     # Plot binarised images.
#     th1, th2, th3, th4 = Binarisation(img=arr_norm[i], maxval=1.0, show=False)
#     th1_list.append(th1)
#     th2_list.append(th2)
#     th3_list.append(th3)
#     th4_list.append(th4)


# Step 2.1 - binarization of images
own_binarised_img_list = []

# Plot binarised images
for i in range(len(arr_norm)):
    # Plot own binarised image.
    binarised_img = OwnGlobalBinarise(img=cropped_img_list[i], thresh=0.1, maxval=1.0)
    own_binarised_img_list.append(binarised_img)



# Step 2.2 - Removing noise from mask
edited_mask_list = []
for i in range(len(arr_norm)):
    edited_mask = OpenMask(mask=own_binarised_img_list[i], ksize=(33, 33), operation="open")
    edited_mask_list.append(edited_mask)



# Step 2.3 - Remove the breast region from the mask
X_largest_blobs_list = []
for i in range(len(arr_norm)):
    _, X_largest_blobs = XLargestBlobs(mask=edited_mask_list[i], top_X=1)
    X_largest_blobs_list.append(X_largest_blobs)



# Step 3.1 - invert mask
inverted_mask_list = []
for i in range(len(arr_norm)):
    inverted_mask = InvertMask(X_largest_blobs_list[i])
    inverted_mask_list.append(inverted_mask)


# step 3.2 - inpainting
own_masked_img_list = []

for i in range(len(arr_norm)):
    # Plot applying largest-blob mask
    masked_img = ApplyMask(img=cropped_img_list[i], mask=X_largest_blobs_list[i])
    own_masked_img_list.append(masked_img)



# step 3.3 - Orientating the mammograms - Horizontal flip first AFTER removing pectoral muscle
flipped_img_list = []
for i in range(len(arr_norm)):
    # Plot flipped image.
    horizontal_flip = HorizontalFlip(mask=X_largest_blobs_list[i])
    if horizontal_flip:
        flipped_img = np.fliplr(own_masked_img_list[i])
        flipped_img_list.append(flipped_img)
    else:
        flipped_img_list.append(own_masked_img_list[i])




# step 3.4 - Contrast-Limited Adaptive Histogram Equalisation (CLAHE)
clahe_img_list = []
for i in range(len(arr_norm)):
    # CLAHE enhancement.
    clahe_img = clahe(img=flipped_img_list[i])
    clahe_img_list.append(clahe_img)




# Step 4 - Pad into a square
padded_img_list = []
for i in range(len(arr_norm)):
    padded_img = Pad(img=clahe_img_list[i])
    padded_img_list.append(padded_img)



# Step 5 - Resize all images in the padded_img_list to 256x256x3
resized_img_list = []

for img in padded_img_list:
    # print(f"Image dimensions: {img.shape}")
    resized_img = cv2.resize(img, (256, 256))
    resized_img_list.append(resized_img)



# Step 6 - Generate outputs

# if want the output in png format
output_png = os.path.join(config.CBIS_BASE_PATH, "output_png")
if not os.path.exists(output_png):
    os.makedirs(output_png)

for i in range(len(resized_img_list)):
    save_path = os.path.join(output_png, f"{ds[i].PatientID}_pad.png")
    cv2.imwrite(filename=save_path, img=resized_img_list[i])

#
#
#
# # if want the output in dcm format
# output_dcm = os.path.join(config.CBIS_BASE_PATH, "output_dcm")
# if not os.path.exists(output_dcm):
#     os.makedirs(output_dcm)
#
# for i in range(len(resized_img_list)):
#     # Create a new DICOM dataset
#     new_dcm = pydicom.Dataset()
#
#     # Set necessary DICOM tags
#     new_dcm.PatientID = ds[i].PatientID  # Set the PatientID
#     new_dcm.Rows = resized_img_list[i].shape[0]  # Set the number of rows
#     new_dcm.Columns = resized_img_list[i].shape[1]  # Set the number of columns
#     new_dcm.PixelData = resized_img_list[i].tobytes()  # Set pixel data
#
#     # Set transfer syntax and endianness
#     new_dcm.file_meta = pydicom.Dataset()
#     new_dcm.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian  # Set transfer syntax
#     new_dcm.is_little_endian = True
#     new_dcm.is_implicit_VR = True
#     new_dcm.PixelRepresentation = 0  # 0 for unsigned data, 1 for signed data
#     new_dcm.BitsAllocated = 16  # Adjust as per your image data
#
#     # You might need to set more DICOM tags according to your requirements
#
#     # Save the DICOM file
#     save_path = os.path.join(output_dcm, f"{ds[i].PatientID}_pad.dcm")
#     pydicom.filewriter.write_file(save_path, new_dcm)




# # if want the output in npy format
# # Loop through padded images and save as NumPy files
# for i, img in enumerate(resized_img_list):
#     # Set the save path for the NumPy file
#     save_path = os.path.join(output_path, f"{ds[i].PatientID}_pad.npy")
#
#     # Save the image as a NumPy file
#     np.save(save_path, img)





