import os
import shutil
import config

# Set the root directory path
root_directory = config.CBIS_PATH

# Create two directories for train and test data
train_directory = os.path.join(root_directory, "train")
test_directory = os.path.join(root_directory, "test")

if not os.path.exists(train_directory):
    os.makedirs(train_directory)

if not os.path.exists(test_directory):
    os.makedirs(test_directory)


# Iterate through subdirectories in the root directory
for subdir in os.listdir(root_directory):
    subdir_path = os.path.join(root_directory, subdir)
    if subdir.startswith("Mass-"):
        # Determine if it's a train or test directory
        if "training" in subdir.lower():
            destination = train_directory
        elif "test" in subdir.lower():
            destination = test_directory
        else:
            print("subdir", subdir)
            continue

        # Move the directory to the appropriate train or test directory
        destination_path = os.path.join(destination, subdir)
        os.makedirs(destination_path, exist_ok=True)
        shutil.move(subdir_path, destination_path)
