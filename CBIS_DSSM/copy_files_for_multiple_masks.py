import os
import shutil

import config

# Path to the main folder
main_folder = config.CBIS_BASE_PATH


# Iterate through all folders in the main folder
for folder in os.listdir(main_folder):
    if folder.endswith('_mask_2'):
        # Get the base folder name
        base_name = folder.replace('_mask_2', '')

        # Define paths for full and full_2 folders
        full_folder = os.path.join(main_folder, base_name + '_full')
        full_2_folder = os.path.join(main_folder, base_name + '_2_full')

        # Check if the full folder exists and full_2 does not exist
        if os.path.exists(full_folder) and not os.path.exists(full_2_folder):
            # Copy the entire content of the 'full' folder to 'full_2'
            shutil.copytree(full_folder, full_2_folder)

# import os
# import re
# import shutil
#
# import config
#
# # Define the directory paths
# folder_path = os.path.join(config.CBIS_BASE_PATH, "data_1")  # Replace with your full folder path
#
# # Collect file names with similar digits
# full_files = [file for file in os.listdir(folder_path) if file.startswith("Mass-Test_P_")]
#
# # Check and copy files
# for file in full_files:
#     # Extract the number between "Mass-Test_P_" and the next "_"
#     matches = re.findall(r"Mass-Test_P_(\d+)_mask|Mass-Test_P_(\d+)_full", file)
#     digit = matches[0][0] or matches[0][1] if matches else None
#     print("file", file)
#     print("digit", digit)
#     if not file.endswith("_full"):
#         count = sum(1 for f in os.listdir(folder_path) if f.startswith(f"Mass-Test_P_{digit}") and not f.endswith("_full"))
#         print("count", count)
#         if count > 1:
#             # Find the file that ends with "_full"
#             full_files = [f for f in os.listdir(folder_path) if f.startswith(f"Mass-Test_P_{digit}") and f.endswith("_full")]
#             if full_files:
#                 full_file = full_files[0]
#                 # Copy the file to the masks folder
#                 new_name = full_file.replace("_full", "_1_full")
#                 shutil.copy(os.path.join(folder_path, full_file), os.path.join(folder_path, new_name))

