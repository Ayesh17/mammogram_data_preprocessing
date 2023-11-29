import os
import shutil
from pathlib import Path

import config

# Define input path
dataset_images_path = os.path.join(config.CBIS_BASE_PATH, "output", "images")
dataset_masks_path = os.path.join(config.CBIS_BASE_PATH, "output", "masks")

# Define paths for training and testing data
train_images_path = os.path.join(config.CBIS_BASE_PATH, "train", "images")
train_masks_path = os.path.join(config.CBIS_BASE_PATH, "train", "masks")
test_images_path = os.path.join(config.CBIS_BASE_PATH, "test", "images")
test_masks_path = os.path.join(config.CBIS_BASE_PATH, "test", "masks")

if not os.path.exists(train_images_path):
    os.makedirs(train_images_path)

if not os.path.exists(train_masks_path):
    os.makedirs(train_masks_path)

if not os.path.exists(test_images_path):
    os.makedirs(test_images_path)

if not os.path.exists(test_masks_path):
    os.makedirs(test_masks_path)


# Splitting the dataset into train and test based on file names

# images
for file_name in os.listdir(dataset_images_path):
    file_path = os.path.join(dataset_images_path, file_name)  # Full file path
    # Check if the file name indicates it's for training or testing
    if "Training" in file_name:
        # print("train_img_path", file_name)
        shutil.move(file_path, os.path.join(train_images_path, file_name))
    elif "Test" in file_name:
        # print("test_img_path", file_name)
        shutil.move(file_path, os.path.join(test_images_path, file_name))

# masks
for file_name in os.listdir(dataset_masks_path):
    file_path = os.path.join(dataset_masks_path, file_name)  # Full file path
    # Check if the file name indicates it's for training or testing
    if "Training" in file_name:
        # print("train_msk_path", file_name)
        shutil.move(file_path, os.path.join(train_masks_path, file_name))
    elif "Test" in file_name:
        # print("test_msk_path", file_name)
        shutil.move(file_path, os.path.join(test_masks_path, file_name))