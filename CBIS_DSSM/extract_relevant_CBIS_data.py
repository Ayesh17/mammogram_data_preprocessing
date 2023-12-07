import os
import shutil
import config

# Set the directory path
directory_path = config.CBIS_BASE_PATH

# Get a list of all subdirectories in the given directory
subdirectories = next(os.walk(directory_path))[1]

# Remove subdirectories that don't start with 'MSS'
for subdir in subdirectories:
    if subdir.startswith('Calc'):
        subdir_path = os.path.join(directory_path, subdir)
        # Remove the directory and its contents
        try:
            shutil.rmtree(subdir_path)
            print(f"Removed: {subdir_path}")
        except OSError as e:
            print(f"Error: {subdir_path} - {e.strerror}")