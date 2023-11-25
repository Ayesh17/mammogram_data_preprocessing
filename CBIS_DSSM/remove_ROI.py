import os

import config

def remove_larger_dcm(dicom_folder):
    for root, dirs, files in os.walk(dicom_folder):
        for dir_name in dirs:
            subfolder_path = os.path.join(root, dir_name)
            if os.path.isdir(subfolder_path) and not dir_name.endswith('_full'):
                dcm_files = [f for f in os.listdir(subfolder_path) if f.endswith('.dcm')]
                if len(dcm_files) == 2:  # Ensure there are exactly 2 .dcm files
                    file_1 = os.path.join(subfolder_path, dcm_files[0])
                    file_2 = os.path.join(subfolder_path, dcm_files[1])
                    if os.path.getsize(file_1) < os.path.getsize(file_2):
                        os.remove(file_1)
                    else:
                        os.remove(file_2)

# Example usage:
remove_larger_dcm(config.CBIS_PATH)
