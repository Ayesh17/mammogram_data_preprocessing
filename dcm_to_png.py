import os

import numpy as np
import pydicom
import cv2
from matplotlib import pyplot as plt

import config


def convert_dicom_to_png(dicom_folder, png_folder):
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


# Replace with your DICOM folder path and output PNG folder path
dicom_folder_path = os.path.join(config.INBREAST_PATH, "AllDICOMs")
png_output_folder_path = os.path.join(config.INBREAST_PATH, "Converted_PNG")

if __name__ == '__main__':
    # Convert the DICOM file to PNG
    convert_dicom_to_png(dicom_folder_path, png_output_folder_path)


