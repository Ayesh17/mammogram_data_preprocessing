import os
import shutil
import config

# Set the directory path
directory_path = config.CBIS_PATH

# Get a list of all subdirectories in the given directory
subdirectories = next(os.walk(directory_path))[1]

# Iterate through each subdirectory
for subdir in subdirectories:
    subdir_path = os.path.join(directory_path, subdir)
    print("subdir_path", subdir_path)
    if os.path.isdir(subdir_path):
        # Iterate through sub-subdirectories
        for root, dirs, files in os.walk(subdir_path):
            for file in files:
                print("file", file)
                if file.endswith('.dcm'):
                    file_path = os.path.join(root, file)
                    destination = os.path.join(subdir_path, file)
                    # Check if a file with the same name exists in the 'mass' subdirectory
                    if os.path.exists(destination):
                        # Generate a new name by appending an incremental number
                        i = 1
                        new_destination = os.path.join(subdir_path, f"{os.path.splitext(file)[0]}_{i}.dcm")
                        while os.path.exists(new_destination):
                            i += 1
                            new_destination = os.path.join(subdir_path, f"{os.path.splitext(file)[0]}_{i}.dcm")
                        shutil.move(file_path, new_destination)
                    else:
                        shutil.move(file_path, destination)



#Deleet now empty subdirectories
# Get a list of all subdirectories in the given directory
subdirectories = next(os.walk(directory_path))[1]

# Remove subdirectories that don't start with 'Mass'
# Iterate through the main directory
for root, dirs, files in os.walk(directory_path):
    for subdir in dirs:
        subdir_path = os.path.join(root, subdir)
        if not subdir.startswith('mass') and not os.listdir(subdir_path):
            # Remove empty subdirectories
            shutil.rmtree(subdir_path)
            print(f"Removed: {subdir_path}")



# Remove subdirectories that don't start with 'Mass'
# Iterate through the main directory
for root, dirs, files in os.walk(directory_path):
    for subdir in dirs:
        subdir_path = os.path.join(root, subdir)
        if not subdir.startswith('mass') and not os.listdir(subdir_path):
            # Remove empty subdirectories
            shutil.rmtree(subdir_path)
            print(f"Removed: {subdir_path}")
