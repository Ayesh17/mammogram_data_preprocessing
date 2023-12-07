import numpy as np
import pydicom
import os
import config

# Path to the main directory
main_directory = config.CBIS_PATH



def combine_masks(folder_path):
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    for subfolder in subfolders:
        if subfolder.endswith("mask"):
            subfolder_path = os.path.join(folder_path, subfolder)
            masks = [file for file in os.listdir(subfolder_path) if file.endswith('.dcm')]
            if len(masks) > 1:
                combined_mask = None
                for mask_file in masks:
                    mask_path = os.path.join(subfolder_path, mask_file)
                    print("mask_path", mask_path)
                    dcm_data = pydicom.dcmread(mask_path)
                    mask_array = dcm_data.pixel_array.astype(bool)
                    print("mask_array", mask_array)
                    if combined_mask is None:
                        combined_mask = mask_array
                    else:
                        combined_mask |= mask_array  # Perform logical OR operation


                # Save the combined mask
                print("combined_mask", combined_mask)



                # Convert boolean array to integers (True becomes 1, False becomes 0)
                mask_array = combined_mask.astype(np.float32)
                # Normalize the array
                normalized_mask = (mask_array - np.min(mask_array)) / (np.max(mask_array) - np.min(mask_array))

                # combined_dcm = pydicom.Dataset()
                #
                # combined_dcm.PixelData = combined_mask.tobytes()
                # combined_dcm.Rows, combined_dcm.Columns = combined_mask.shape
                np.save(os.path.join(subfolder_path, "combined_mask.npy"), normalized_mask)



combine_masks(main_directory)




