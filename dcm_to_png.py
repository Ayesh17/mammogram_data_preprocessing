import os

import numpy as np
import pydicom
import cv2
from matplotlib import pyplot as plt

import config


def convert_dicom_to_png_INbreast(dicom_folder, png_folder):
    # Create the output directory if it doesn't exist
    os.makedirs(png_folder, exist_ok=True)

    # Iterate through each DICOM file in the folder
    for filename in os.listdir(dicom_folder):
        if filename.endswith(".dcm"):
            file_path = os.path.join(dicom_folder, filename)
            print("Converting file : ", file_path)
            # Read the DICOM file
            ds = pydicom.dcmread(file_path)

            # Normalize the pixel values to be between 0 and 1
            img = ds.pixel_array.astype(float)
            img /= img.max()

            # Save as PNG
            output_filename = os.path.splitext(filename)[0] + ".png"
            output_path = os.path.join(png_folder, output_filename)
            plt.imsave(output_path, img, cmap='gray')


def convert_dicom_to_png_CBIS(dicom_folder, png_folder):
    # Create the output directory if it doesn't exist
    os.makedirs(png_folder, exist_ok=True)

    # Iterate through each subdirectory in the DICOM folder
    for root, _, files in os.walk(dicom_folder):
        for filename in files:
            if filename.endswith(".dcm"):
                file_path = os.path.join(root, filename)
                print("Converting file:", file_path)

                # Create a similar directory structure in the PNG folder
                relative_path = os.path.relpath(file_path, dicom_folder)
                output_directory = os.path.join(png_folder, os.path.dirname(relative_path))
                os.makedirs(output_directory, exist_ok=True)

                # Read the DICOM file
                ds = pydicom.dcmread(file_path)

                # Normalize the pixel values to be between 0 and 1
                img = ds.pixel_array.astype(float)
                img /= img.max()

                # Save as PNG
                output_filename = os.path.splitext(filename)[0] + ".png"
                output_path = os.path.join(output_directory, output_filename)
                plt.imsave(output_path, img, cmap='gray')


# Input and output folder paths

# for INbreast
dicom_folder_path_INbreast = os.path.join(config.INBREAST_PATH, "AllDICOMs")
png_output_folder_path_INbreast = os.path.join(config.INBREAST_PATH, "Converted_PNG")

# for CBID-DDSM
dicom_folder_path_CBIS = config.CBIS_PATH
png_output_folder_path_CBIS = os.path.join(config.CBIS_BASE_PATH, "Converted_PNG")

if __name__ == '__main__':

    # # Convert the DICOM file to PNG when using the INbreast datset
    # convert_dicom_to_png_INbreast(dicom_folder_path_INbreast, png_output_folder_path_INbreast)

    # Convert the DICOM file to PNG when using the CBIS-DDSM datset
    convert_dicom_to_png_CBIS(dicom_folder_path_CBIS, png_output_folder_path_CBIS)


