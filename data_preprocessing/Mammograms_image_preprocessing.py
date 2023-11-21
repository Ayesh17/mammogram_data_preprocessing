import matplotlib.pyplot as plt
import pandas            as pd
import numpy             as np
import cv2
import pydicom
import os
import skimage

from pathlib             import Path

import config
from removing_artifacts import MinMaxNormalise, CropBorders, Binarisation, OwnGlobalBinarise, OpenMask, XLargestBlobs, \
    InvertMask, HorizontalFlip, clahe, ApplyMask, Pad

cbis_path = config.CBIS_PATH
base_path = "../data/raw_data"


# Read the selected .dcm files
# Paths of selected .dcm files
selected_paths = [os.path.join(cbis_path, "Mass-Test_P_00016_LEFT_CC_full", "1-1.dcm"),
                  os.path.join(cbis_path, "Mass-Test_P_00016_LEFT_MLO_full", "1-1.dcm"),
                  os.path.join(cbis_path, "Mass-Test_P_00017_LEFT_CC_full", "1-1.dcm"),
                  os.path.join(cbis_path, "Mass-Test_P_00032_RIGHT_CC_full", "1-1.dcm"),
                  os.path.join(cbis_path, "Mass-Test_P_00124_RIGHT_CC_full", "1-1.dcm")]

ds = [pydicom.dcmread(selected_paths[i]) for i in range(len(selected_paths))]
arr = [_ds.pixel_array for _ds in ds]


# Paths of corresponding masks
mask_paths = [os.path.join(cbis_path, "Mass-Test_P_00016_LEFT_CC_mask_1", "1-1.dcm"),
              os.path.join(cbis_path, "Mass-Test_P_00016_LEFT_MLO_mask_1", "1-1.dcm"),
              os.path.join(cbis_path, "Mass-Test_P_00017_LEFT_CC_mask_1", "1-2.dcm"),
              os.path.join(cbis_path, "Mass-Test_P_00032_RIGHT_CC_mask_1", "1-2.dcm"),
              os.path.join(cbis_path, "Mass-Test_P_00124_RIGHT_CC_mask_1", "1-2.dcm")
              ]

ds_masks = [pydicom.dcmread(mask_paths[i]) for i in range(len(mask_paths))]
arr_masks = [_ds.pixel_array for _ds in ds_masks]




# Plot together
fig, ax = plt.subplots(nrows=1, ncols=len(selected_paths), figsize = (22, 5))

for i in range(len(selected_paths)):
    ax[i].imshow(arr[i], cmap="gray")
    ax[i].set_title(f"{ds[i].PatientID}")
    
plt.tight_layout()
plt.savefig(fname= os.path.join(base_path,"raw.png"), dpi=300)


# Plot individually
for i in range(len(arr)):
    save_path = os.path.join(base_path, f"{ds[i].PatientID}_raw_{i}.png")
    cv2.imwrite(filename=save_path, img=arr[i])




# ### Visualise the corresponding masks
# Plot together
fig, ax = plt.subplots(nrows=1, ncols=len(mask_paths), figsize = (22, 5))

for i in range(len(mask_paths)):
    ax[i].imshow(arr_masks[i], cmap="gray")
    ax[i].set_title(f"{ds_masks[i].PatientID}")
    
plt.tight_layout()
plt.savefig(fname=os.path.join(base_path,"raw_masks.png"), dpi=300)


# Plot individually
for i in range(len(arr_masks)):
    save_path = os.path.join(base_path, f"{ds_masks[i].PatientID}.png")
    cv2.imwrite(filename=save_path, img=arr_masks[i])



# ## Understanding the images
for a in arr:
    print("Shape:", a.shape)
    print("Dimensions:", a.ndim)
    print("Type:", type(a))
    print("Data type:", a.dtype)
    print(f"min value, max value: {a.min(), a.max()}")
    print("---")


# ### 2. Normalise to range [0, 1]?
# 
# - **Observations:**
#     1. Doing min-max normalisation changes the intensities in the image (i.e. **the image is DISTORTED using min-max scaling!!!**)
#         - [Image datatypes and what they mean](https://scikit-image.org/docs/dev/user_guide/data_types.html)
#     2. Normalisation using the below method changes the image array to `dtype = float32`.
#     
# - **Conclusions:**
#     1. Don't need to normalise for image preprocessing. Only normalise to \[0, 1\] after image preprocessing (before feeding into model).

# In[13]:




arr_norm = [MinMaxNormalise(a) for a in arr]
fig, ax = plt.subplots(nrows=2, ncols=len(selected_paths), figsize = (22, 10))
fig.suptitle("Min-Max Normalisation")


# Plot original
for i in range(len(selected_paths)):
    ax[0][i].imshow(arr[i], cmap="gray")
    ax[0][i].set_title(f"{ds[i].PatientID}")
    
# Plot normalised
for i in range(len(selected_paths)):
    ax[1][i].imshow(arr_norm[i], cmap="gray")
    ax[1][i].set_title("NORMLISED")
    
plt.tight_layout()
plt.savefig(fname=os.path.join(base_path,"normalised.png"), dpi=300)



for a in arr:
    print(a.dtype)
    print({a.min(), a.max()})

for a in arr_norm:
    print(a.dtype)
    print({a.min(), a.max()})


# ### 3. uint16 (16-bit) -> uint8 (8-bit)
# 
# - **Why:**
#     1. We have to change from 16-bit to 8-bit because `cv2.threshold` and `cv2.adaptiveThreshold()` requires 8-bit image array.
#     
# <br>
# 
# - **Observations:**
#     1. `skimage.img_as_ubyte()` does not seem to change the relative intensities within each image as drastically as min-max normalisation.
#     2. BUT converting from `uint16` to `uint8` removes the granularity of the information in an image.
# 
# <br>
# 
# - **Conclusions:**
#     1. Should not convert to `uint8`.

# In[17]:


arr_uint8 = [skimage.img_as_ubyte(a) for a in arr]


fig, ax = plt.subplots(nrows=3, ncols=len(selected_paths), figsize = (22, 20))
fig.suptitle("uint16 to uint8")

# Plot original uint16
for i in range(len(selected_paths)):
    ax[0][i].imshow(arr[i], cmap="gray")
    ax[0][i].set_title(f"{ds[i].PatientID}")
    
# Plot uint8
for i in range(len(selected_paths)):
    ax[1][i].imshow(arr_uint8[i], cmap="gray")
    ax[1][i].set_title(f"{ds[i].PatientID} \nuint8")
    
# Plot histogram of uint8 (ignoring 0)
for i in range(len(selected_paths)):
    hist, bin_edges = np.histogram(arr_uint8[i], bins=255, density=False)
    ax[2][i].plot(bin_edges[1:-1], hist[1:])
    
# plt.tight_layout()
# plt.savefig(fname="../outputs/image-preprocessing/uint8.png", dpi=300)


# In[19]:


for a in arr:
    print(a.dtype)
    print(a.min(), a.max())

for a in arr_uint8:
    print(a.dtype)
    print(a.min(), a.max())


# ---
# 
# ## Initial crop around the image boundaries
# 
# - **Why:**
#     - Some scans have a white border on some/all sides of the frame. Taking away a small percentage of the image on all sides might remove this border while not removing too much of the image such that we lose valuable information.

# In[20]:





# In[21]:


cropped_img_list = []

fig, ax = plt.subplots(nrows=2, ncols=len(selected_paths), figsize = (22, 10))

for i in range(len(arr_norm)):
    
    # Plot original
    ax[0][i].imshow(arr_norm[i], cmap="gray")
    ax[0][i].set_title(f"{ds[i].PatientID}")
    
    # Plot cropped
    cropped_img = CropBorders(img=arr_norm[i])
    cropped_img_list.append(cropped_img)
    ax[1][i].imshow(cropped_img, cmap="gray")
    ax[1][i].set_title("Cropped image")

    
plt.tight_layout()
plt.savefig(fname=os.path.join(base_path,"1_cropped.png"), dpi=150)


# In[22]:


# Plot individually
for i in range(len(cropped_img_list)):
    save_path = os.path.join(os.path.join(base_path,"1_cropped"), f"{ds[i].PatientID}_cropped.png")
    cv2.imwrite(filename=save_path, img=cropped_img_list[i]*255)







# In[24]:


th1_list = []
th2_list = []
th3_list = []
th4_list = []

# Plot binarised images
fig, ax = plt.subplots(nrows=5, ncols=5, figsize=(22, 25))

for i in range(len(arr_norm)):
    
    # Plot original image.
    ax[0][i].imshow(arr_norm[i], cmap="gray")
    ax[0][i].set_title(f"{ds[i].PatientID}")
    
    # Plot binarised images.
    th1, th2, th3, th4 = Binarisation(img=arr_norm[i], maxval=1.0, show=False)
    th1_list.append(th1)
    th2_list.append(th2)
    th3_list.append(th3)
    th4_list.append(th4)
    
    ax[1][i].imshow(th1, cmap="gray")
    ax[1][i].set_title("Global thresholding (v = 0.1)")
    
    ax[2][i].imshow(th2, cmap="gray")
    ax[2][i].set_title("Otsu's thresholding")
    
    ax[3][i].imshow(th3, cmap="gray")
    ax[3][i].set_title("Adaptive mean thresholding")
    
    ax[4][i].imshow(th4, cmap="gray")
    ax[4][i].set_title("Adaptive gaussian thresholding")
    
# plt.tight_layout()
# plt.savefig(fname="../outputs/image-preprocessing/binarised.png", dpi=300)


# #### Step 1.1 - Try to binarise using my manual function




# In[26]:


own_binarised_img_list = []

# Plot binarised images
fig, ax = plt.subplots(nrows=2, ncols=len(arr_norm), figsize=(22, 10))

for i in range(len(arr_norm)):
    
    # Plot original image.
    ax[0][i].imshow(cropped_img_list[i], cmap="gray")
    ax[0][i].set_title(f"{ds[i].PatientID}")
    
    # Plot own binarised image.
    binarised_img = OwnGlobalBinarise(img=cropped_img_list[i], thresh=0.1, maxval=1.0)
    own_binarised_img_list.append(binarised_img)
    ax[1][i].imshow(binarised_img, cmap="gray")
    ax[1][i].set_title("Binarised")
    
plt.tight_layout()
plt.savefig(fname=os.path.join(base_path,"binarised.png"), dpi=300)


# In[27]:


# Plot individually
for i in range(len(own_binarised_img_list)):
    save_path = os.path.join(os.path.join(base_path,"2_binarised"), f"{ds[i].PatientID}_binarised.png")
    cv2.imwrite(filename=save_path, img=own_binarised_img_list[i]*255)


# #### Step 1.2 - Removing noise from mask
edited_mask_list = []

fig, ax = plt.subplots(nrows=2, ncols=len(arr_norm), figsize=(22, 10))

for i in range(len(arr_norm)):
    
    # Plot original image.
    ax[0][i].imshow(cropped_img_list[i], cmap="gray")
    ax[0][i].set_title(f"{ds[i].PatientID}")
    
#     # Plot original mask.
#     ax[1][i].imshow(own_binarised_img_list[i], cmap="gray")
#     ax[1][i].set_title("Binarised")
    
    # Plot edited mask.
    edited_mask = OpenMask(mask=own_binarised_img_list[i], ksize=(33, 33), operation="open")
    edited_mask_list.append(edited_mask)
    ax[1][i].imshow(edited_mask, cmap="gray")
    ax[1][i].set_title("Edited mask")
    
plt.tight_layout()
plt.savefig(fname=os.path.join(base_path,"remove_noise.png"), dpi=300)


# In[30]:


# Plot individually
for i in range(len(edited_mask_list)):
    save_path = os.path.join(os.path.join(base_path,"3_remove_noise"), f"{ds[i].PatientID}_remove_noise.png")
    cv2.imwrite(filename=save_path, img=edited_mask_list[i]*255)


# #### Step 1.3 - Remove the breast region from the mask


# In[34]:


X_largest_blobs_list = []

fig, ax = plt.subplots(nrows=2, ncols=len(arr_norm), figsize=(22, 10))

for i in range(len(arr_norm)):
    
    # Plot original image.
    ax[0][i].imshow(arr_norm[i], cmap="gray")
    ax[0][i].set_title(f"{ds[i].PatientID}")
    
#     # Plot original mask.
#     ax[1][i].imshow(own_binarised_img_list[i], cmap="gray")
#     ax[1][i].set_title("Binarised")
    
#     # Plot edited mask.
#     ax[2][i].imshow(edited_mask_list[i], cmap="gray")
#     ax[2][i].set_title("Edited masks")
    
    # Plot largest-blob mask.
    _, X_largest_blobs = XLargestBlobs(mask=edited_mask_list[i], top_X=1)
    X_largest_blobs_list.append(X_largest_blobs)
    ax[1][i].imshow(X_largest_blobs, cmap="gray")
    ax[1][i].set_title("Largest blob")
    
plt.tight_layout()
plt.savefig(fname=os.path.join(base_path,"largest_blob.png"), dpi=300)


# In[35]:


# Plot individually
for i in range(len(X_largest_blobs_list)):
    save_path = os.path.join(os.path.join(base_path,"4_largest_blob"), f"{ds[i].PatientID}_largest_blob.png")
    cv2.imwrite(filename=save_path, img=X_largest_blobs_list[i]*255)


# ### Step 2 - `cv2.inpaint()` with the mask

# #### Step 2.1 - Invert mask

# In[36]:



# In[37]:


inverted_mask_list = []

fig, ax = plt.subplots(nrows=2, ncols=len(arr_norm), figsize=(22, 10))

for i in range(len(arr_norm)):
    
    # Plot original image.
    ax[0][i].imshow(cropped_img_list[i], cmap="gray")
    ax[0][i].set_title(f"{ds[i].PatientID}")
    
#     # Plot original mask.
#     ax[1][i].imshow(binarised_img_list[i], cmap="gray")
#     ax[1][i].set_title("Binarised")
    
#     # Plot edited mask.
#     ax[2][i].imshow(edited_mask_list[i], cmap="gray")
#     ax[2][i].set_title("Edited masks")
    
#     # Plot largest-blob mask.
#     ax[3][i].imshow(X_largest_blobs_list[i], cmap="gray")
#     ax[3][i].set_title("Largest blob")
    
    # Plot inverted largest-blob mask
    inverted_mask = InvertMask(X_largest_blobs_list[i])
    inverted_mask_list.append(inverted_mask)
    ax[1][i].imshow(inverted_mask, cmap="gray")
    ax[1][i].set_title("Inverted largest blob")
    
plt.tight_layout()
plt.savefig(fname=os.path.join(base_path,"inverted.png"), dpi=300)


# In[38]:


# Plot individually
for i in range(len(inverted_mask_list)):
    save_path = os.path.join(os.path.join(base_path,"5_inverted"), f"{ds[i].PatientID}_inverted.png")
    cv2.imwrite(filename=save_path, img=inverted_mask_list[i]*255)




own_masked_img_list = []

fig, ax = plt.subplots(nrows=2, ncols=len(arr_norm), figsize=(22, 10))

for i in range(len(arr_norm)):
    
    # Plot original image.
    ax[0][i].imshow(cropped_img_list[i], cmap="gray")
    ax[0][i].set_title(f"{ds[i].PatientID}")
    
#     # Plot original mask.
#     ax[1][i].imshow(binarised_img_list[i], cmap="gray")
#     ax[1][i].set_title("Binarised")
    
#     # Plot edited mask.
#     ax[2][i].imshow(edited_mask_list[i], cmap="gray")
#     ax[2][i].set_title("Edited masks")
    
#     # Plot largest-blob mask.
#     ax[3][i].imshow(X_largest_blobs_list[i], cmap="gray")
#     ax[3][i].set_title("Largest blob")
    
    # Plot applying largest-blob mask
    masked_img = ApplyMask(img=cropped_img_list[i], mask=X_largest_blobs_list[i])
    own_masked_img_list.append(masked_img)
    ax[1][i].imshow(masked_img, cmap="gray")
    ax[1][i].set_title("Masked image")

    
plt.tight_layout()
plt.savefig(fname=os.path.join(base_path,"apply_mask.png"), dpi=300)


# In[44]:


# Plot individually
for i in range(len(own_masked_img_list)):
    save_path = os.path.join(os.path.join(base_path,"6_apply_mask"), f"{ds[i].PatientID}_apply_mask.png")
    cv2.imwrite(filename=save_path, img=own_masked_img_list[i]*255)


# ---
# 
# ## Orientating the mammograms - Horizontal flip first AFTER removing pectoral muscle




# In[46]:


flipped_img_list = []

fig, ax = plt.subplots(nrows=2, ncols=len(arr_norm), figsize=(22, 10))

for i in range(len(arr_norm)):
    
    # Plot original image.
    ax[0][i].imshow(cropped_img_list[i], cmap="gray")
    ax[0][i].set_title(f"{ds[i].PatientID}")
    
#     # Plot largest-blob mask.
#     ax[1][i].imshow(X_largest_blobs_list[i], cmap="gray")
#     ax[1][i].set_title("Largest blob")
    
#     # Plot final image.
#     ax[2][i].imshow(final_result_1_list[i], cmap="gray")
#     ax[2][i].set_title("FINAL RESULT")
    
    # Plot flipped image.
    horizontal_flip = HorizontalFlip(mask=X_largest_blobs_list[i])
    if horizontal_flip:
        flipped_img = np.fliplr(own_masked_img_list[i])
        flipped_img_list.append(flipped_img)
    else:
        flipped_img_list.append(own_masked_img_list[i])
    
    ax[1][i].imshow(flipped_img_list[i], cmap="gray")
    ax[1][i].set_title("Flipped image")

plt.tight_layout()
plt.savefig(fname=os.path.join(base_path,"flipped.png"), dpi=300)


# In[47]:


# Plot individually
for i in range(len(flipped_img_list)):
    save_path = os.path.join(os.path.join(base_path,"7_flipped"), f"{ds[i].PatientID}_flipped.png")
    cv2.imwrite(filename=save_path, img=flipped_img_list[i]*255)


# ---
# 
# ## Contrast-Limited Adaptive Histogram Equalisation (CLAHE) 

# In[48]:




# In[49]:


clahe_img_list = []

fig, ax = plt.subplots(nrows=2, ncols=len(arr_norm), figsize=(22, 10))

for i in range(len(arr_norm)):
    
    # Plot original image.
    ax[0][i].imshow(flipped_img_list[i], cmap="gray")
    ax[0][i].set_title(f"{ds[i].PatientID}")
    
#     # Plot largest-blob mask.
#     ax[1][i].imshow(X_largest_blobs_list[i], cmap="gray")
#     ax[1][i].set_title("Largest blob")
    
#     # Plot final image.
#     ax[2][i].imshow(final_result_1_list[i], cmap="gray")
#     ax[2][i].set_title("FINAL RESULT")
    
    # CLAHE enhancement.
    clahe_img = clahe(img=flipped_img_list[i])
    clahe_img_list.append(clahe_img)
    
    ax[1][i].imshow(clahe_img_list[i], cmap="gray")
    ax[1][i].set_title("CLAHE image")

plt.tight_layout()
plt.savefig(fname=os.path.join(base_path,"clahe.png"), dpi=300)


# In[50]:


# Plot individually
for i in range(len(clahe_img_list)):
    save_path = os.path.join(os.path.join(base_path,"8_clahe"), f"{ds[i].PatientID}_clahe.png")
    cv2.imwrite(filename=save_path, img=clahe_img_list[i])


# ---
# 
# ## Removal of pectoral muscle

# ### Step 1 - Pad into a square

# In[51]:





# In[52]:


padded_img_list = []

fig, ax = plt.subplots(nrows=2, ncols=len(arr_norm), figsize=(22, 10))

for i in range(len(arr_norm)):
    
    # Plot original image.
    ax[0][i].imshow(cropped_img_list[i], cmap="gray")
    ax[0][i].set_title(f"{ds[i].PatientID}")
    
#     # Plot flipped image.
#     ax[1][i].imshow(flipped_img_list[i], cmap="gray")
#     ax[1][i].set_title("Flipped image")
    
    # Plot padded image.
    padded_img = Pad(img=clahe_img_list[i])
    padded_img_list.append(padded_img)
    ax[1][i].imshow(padded_img, cmap="gray")
    ax[1][i].set_title("Padded image")

plt.tight_layout()
plt.savefig(fname=os.path.join(base_path,"pad.png"), dpi=300)


# In[53]:


# Plot individually
for i in range(len(padded_img_list)):
    save_path = os.path.join(os.path.join(base_path,"9_pad"), f"{ds[i].PatientID}_pad.png")
    cv2.imwrite(filename=save_path, img=padded_img_list[i])

# # Loop through padded images and save as NumPy files
# for i, img in enumerate(padded_img_list):
#     # Set the save path for the NumPy file
#     save_path = os.path.join(base_path, f"{ds[i].PatientID}_pad.npy")
#
#     # Save the image as a NumPy file
#     np.save(save_path, img)


