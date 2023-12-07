import os
import shutil
import config  # Import the module where CBIS_PATH is defined

batch_size = 100  # Number of subfolders in each batch

data_folder = config.CBIS_BASE_PATH  # Your main data folder containing subfolders
subfolders = [f.path for f in os.scandir(data_folder) if f.is_dir()]
total_subfolders = len(subfolders)

# Create folders for batches
for i in range(0, total_subfolders, batch_size):
    # Create a new folder for the batch
    batch_folder_name = f"data_{i // batch_size + 1}"
    batch_folder_path = os.path.join(data_folder, batch_folder_name)
    os.makedirs(batch_folder_path, exist_ok=True)

    # Determine the range for the current batch
    batch_subfolders = subfolders[i:i + batch_size]

    # Move 100 subfolders into the batch folder
    for subfolder in batch_subfolders:
        subfolder_name = os.path.basename(subfolder)
        new_subfolder_path = os.path.join(batch_folder_path, subfolder_name)
        shutil.move(subfolder, new_subfolder_path)
