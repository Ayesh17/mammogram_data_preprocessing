# imports
import numpy as np
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import config


def select_path(train_or_test, images_or_masks):
    #base folder paths
    input_folder = config.DATASET_PATH
    output_folder = config.OUTPUT_PATH

    # select the dataset
    if train_or_test == "train" and images_or_masks == "images":
        input_folder = os.path.join(input_folder, "train", "images")
        output_folder = os.path.join(output_folder, "train", "images")

    elif (train_or_test == "test" and images_or_masks == "images"):
        input_folder = os.path.join(input_folder, "test", "images")
        output_folder = os.path.join(output_folder, "test", "images")

    elif (train_or_test == "train" and images_or_masks == "masks"):
        input_folder = os.path.join(input_folder, "train", "masks")
        output_folder = os.path.join(output_folder, "train", "masks")

    elif (train_or_test == "test" and images_or_masks == "masks"):
        input_folder = os.path.join(input_folder, "test", "masks")
        output_folder = os.path.join(output_folder, "test", "masks")

    return input_folder, output_folder


def convert_npy_to_png(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    npy_files = [f for f in os.listdir(input_folder) if f.endswith('.npy')]

    for npy_file in npy_files:
        npy_path = os.path.join(input_folder, npy_file)
        image_array = np.load(npy_path)
        # Normalize the pixel values to be in the range of 0 to 1
        image_array = (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array))

        plt.imsave(os.path.join(output_folder, f"{os.path.splitext(npy_file)[0]}.png"), image_array, cmap='gray')



# When usingINbreast
input_folder, output_folder = select_path(train_or_test = "train", images_or_masks = "masks")


# # When using CBIS-DDSm
# cur_dir = os.getcwd()
# input_folder = os.path.join(cur_dir, "data", "preprocessed_data")
# output_folder = os.path.join(cur_dir, "data", "converted_pngs")


convert_npy_to_png(input_folder, output_folder)

