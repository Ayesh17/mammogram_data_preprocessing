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
output_path_images = os.path.join(config.CBIS_BASE_PATH, "output", "images")
output_path_masks = os.path.join(config.CBIS_BASE_PATH, "output", "masks")

if not os.path.exists(output_path_images):
    os.makedirs(output_path_images)

if not os.path.exists(output_path_masks):
    os.makedirs(output_path_masks)


# Initialize empty lists to store image and mask paths
print("Initializing paths")
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
print("Understanding the images")
for a in arr:
    print("Shape:", a.shape)
    print("Dimensions:", a.ndim)
    print("Type:", type(a))
    print("Data type:", a.dtype)
    print(f"min value, max value: {a.min(), a.max()}")
    print("---")


# Normalise to range [0, 1]
# arr_norm = [MinMaxNormalise(a) for a in arr]



# Step 1.1 - Initial crop around the image boundaries
print("Initial cropping around the boundaries")
cropped_img_list = []
cropped_msk_list = []

for i in range(len(arr)):
    cropped_img = CropBorders(img=arr[i])
    cropped_img_list.append(cropped_img)

for i in range(len(arr_masks)):
    cropped_msk = CropBorders(img=arr_masks[i])
    cropped_msk_list.append(cropped_msk)


# image binarization
# th1_list = []
# th2_list = []
# th3_list = []
# th4_list = []
# #
# # # Plot binarised images
# # # fig, ax = plt.subplots(nrows=5, ncols=5, figsize=(22, 25))
# #
# for i in range(len(arr)):
#
#     # Plot binarised images.
#     th1, th2, th3, th4 = Binarisation(img=arr[i], maxval=1.0, show=False)
#     th1_list.append(th1)
#     th2_list.append(th2)
#     th3_list.append(th3)
#     th4_list.append(th4)


# Step 2.1 - binarization of images
print("Image binarization")
own_binarised_img_list = []

# Plot binarised images
for i in range(len(arr)):
    # Plot own binarised image.
    binarised_img = OwnGlobalBinarise(img=cropped_img_list[i], thresh=0.1, maxval=1.0)
    own_binarised_img_list.append(binarised_img)



# Step 2.2 - Removing noise from mask
print("Removing noise from mask")
edited_mask_list = []
for i in range(len(arr)):
    edited_mask = OpenMask(mask=own_binarised_img_list[i], ksize=(33, 33), operation="open")
    edited_mask_list.append(edited_mask)



# Step 2.3 - Removing the breast region from the mask
print("Removing the breast region from the mask")
X_largest_blobs_list = []
for i in range(len(arr)):
    _, X_largest_blobs = XLargestBlobs(mask=edited_mask_list[i], top_X=1)
    X_largest_blobs_list.append(X_largest_blobs)


# X_largest_blobs_list_masks = []
# for i in range(len(arr_masks)):
#     _, X_largest_blobs_masks = XLargestBlobs(mask=edited_mask_list[i], top_X=1)
#     X_largest_blobs_list_masks.append(X_largest_blobs_masks)

# Step 3.1 - inverting mask
print("Inverting mask")
inverted_mask_list = []
for i in range(len(arr)):
    inverted_mask = InvertMask(X_largest_blobs_list[i])
    inverted_mask_list.append(inverted_mask)


# step 3.2 - inpainting
print("Inpainting")
own_masked_img_list = []

for i in range(len(arr)):
    # Plot applying largest-blob mask
    masked_img = ApplyMask(img=cropped_img_list[i], mask=X_largest_blobs_list[i])
    own_masked_img_list.append(masked_img)




# step 3.3 - Orientating the mammograms - Horizontal flip first AFTER removing pectoral muscle
print("Horizontal flipping")
flipped_img_list = []
flipped_msk_list = []

for i in range(len(arr)):
    # Plot flipped image.
    horizontal_flip = HorizontalFlip(mask=X_largest_blobs_list[i])
    if horizontal_flip:
        flipped_img = np.fliplr(own_masked_img_list[i])
        flipped_img_list.append(flipped_img)

    else:
        flipped_img_list.append(own_masked_img_list[i])



for i in range(len(arr_masks)):
    # Plot flipped image.
    horizontal_flip = HorizontalFlip(mask=cropped_msk_list[i])
    if horizontal_flip:
        flipped_msk = np.fliplr(cropped_msk_list[i])
        flipped_msk_list.append(flipped_msk)

    else:
        flipped_msk_list.append(cropped_msk_list[i])



# step 3.4 - Contrast-Limited Adaptive Histogram Equalisation (CLAHE)
print("CLAHE")
clahe_img_list = []
for i in range(len(arr)):
    # CLAHE enhancement.
    clahe_img = clahe(img=flipped_img_list[i])
    clahe_img_list.append(clahe_img)




# Step 4 - Padding into a square
print("Padding")
padded_img_list = []
padded_msk_list = []
for i in range(len(arr)):
    padded_img = Pad(img=clahe_img_list[i])
    padded_img_list.append(padded_img)

for i in range(len(arr_masks)):
    padded_msk = Pad(img=flipped_msk_list[i])
    padded_msk_list.append(padded_msk)



# Step 5 - Resize all images in the padded_img_list to 256x256x3
print("Resizing")
resized_img_list = []
resized_msk_list = []

for i in range(len(padded_img_list)):
    # print(f"Image dimensions: {img.shape}")
    resized_img = cv2.resize(padded_img_list[i], (256, 256))
    resized_img_list.append(resized_img)

for i in range(len(padded_msk_list)):
    resized_msk = cv2.resize(padded_msk_list[i], (256, 256))
    resized_msk_list.append(resized_msk)



# Step 6 - Generate outputs
print("Generating outputs")
# if want the output in png format
output_images_png = os.path.join(config.CBIS_BASE_PATH, "output_png", "images")
if not os.path.exists(output_images_png):
    os.makedirs(output_images_png)

for i in range(len(resized_img_list)):
    save_path = os.path.join(output_images_png, f"{ds[i].PatientID}_image.png")
    cv2.imwrite(filename=save_path, img=resized_img_list[i])



output_masks_png = os.path.join(config.CBIS_BASE_PATH, "output_png", "masks")
if not os.path.exists(output_masks_png):
    os.makedirs(output_masks_png)

for i in range(len(resized_msk_list)):
    save_path = os.path.join(output_masks_png, f"{ds_masks[i].PatientID}_mask.png")
    cv2.imwrite(filename=save_path, img=resized_msk_list[i])




# # if want the output in dcm format
# output_dcm_images = os.path.join(config.CBIS_BASE_PATH, "output_dcm", "images")
# output_dcm_masks = os.path.join(config.CBIS_BASE_PATH, "output_dcm", "masks")
#
# if not os.path.exists(output_dcm_images):
#     os.makedirs(output_dcm_images)
#
# if not os.path.exists(output_dcm_masks):
#     os.makedirs(output_dcm_masks)
#
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
#     save_path = os.path.join(output_dcm_images, f"{ds[i].PatientID}_pad.dcm")
#     pydicom.filewriter.write_file(save_path, new_dcm)
#
# for i in range(len(resized_msk_list)):
#     # Create a new DICOM dataset
#     new_dcm = pydicom.Dataset()
#
#     # Set necessary DICOM tags
#     new_dcm.PatientID = ds_masks[i].PatientID  # Set the PatientID
#     new_dcm.Rows = resized_msk_list[i].shape[0]  # Set the number of rows
#     new_dcm.Columns = resized_msk_list[i].shape[1]  # Set the number of columns
#     new_dcm.PixelData = resized_msk_list[i].tobytes()  # Set pixel data
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
#     save_path = os.path.join(output_dcm_masks, f"{ds_masks[i].PatientID}_pad.dcm")
#     pydicom.filewriter.write_file(save_path, new_dcm)




# # if want the output in npy format
# # Loop through padded images and save as NumPy files
# for i, img in enumerate(resized_img_list):
#     # Set the save path for the NumPy file
#     save_path = os.path.join(output_path_images, f"{ds_masks[i].PatientID}_pad.npy")
#
#     # Save the image as a NumPy file
#     np.save(save_path, img)
#
#
# for i, img in enumerate(resized_msk_list):
#     # Set the save path for the NumPy file
#     save_path = os.path.join(output_path_masks, f"{ds_masks[i].PatientID}_pad.npy")
#
#     # Save the image as a NumPy file
#     np.save(save_path, img)



