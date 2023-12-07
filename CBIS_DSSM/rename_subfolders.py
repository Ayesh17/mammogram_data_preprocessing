import os
import re
import shutil
import matplotlib.pyplot as plt
import config

def rename_subfolders(dicom_folder):
    # Iterate through each subdirectory in the DICOM folder
    for root, dirs, files in os.walk(dicom_folder):
        for dir_name in dirs:
            subfolder_path = os.path.join(root, dir_name)
            if os.path.isdir(subfolder_path):
                # Check if the subfolder name ends with a digit
                match = re.search(r'(\d)$', dir_name)
                if match:
                    digit = match.group(1)
                    rest_of_name = dir_name[:-1]  # Get the name excluding the digit
                    new_dir_name = f"{rest_of_name}mask_{digit}"  # Rearrange the name
                else:
                    new_dir_name = dir_name + "_full"

                # Rename the subfolder
                new_subfolder_path = os.path.join(root, new_dir_name)
                os.rename(subfolder_path, new_subfolder_path)

if __name__ == '__main__':
    # when using the CBIS-DDSM datset
    rename_subfolders(config.CBIS_BASE_PATH)
