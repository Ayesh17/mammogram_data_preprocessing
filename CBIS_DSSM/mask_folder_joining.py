import os
import re
import shutil

import config

main_folder = config.CBIS_BASE_PATH  # Replace this with the path to your main folder

for root, dirs, files in os.walk(main_folder):
    for dir_name in dirs:
        subfolder_path = os.path.join(root, dir_name)
        if os.path.isdir(subfolder_path):
            # Check if the subfolder name ends with a digit
            match = re.search(r'(\d)$', dir_name)
            if match:
                digit = match.group(1)
                rest_of_name = dir_name[:-1]  # Get the name excluding the digit and the last underscore
                if rest_of_name.endswith('_'):
                    rest_of_name = rest_of_name[:-1]  # Remove the final underscore
                source_names = [d for d in dirs if d.startswith(rest_of_name)]
                destination_folder = os.path.join(main_folder, rest_of_name)
                if not os.path.exists(destination_folder):
                    os.makedirs(destination_folder)
                for source_name in source_names:
                    source_folder = os.path.join(root, source_name)

                    # Cut .dcm files with the subfolder name
                    for item in os.listdir(source_folder):
                        if item.endswith('.dcm'):
                            source_item = os.path.join(source_folder, item)
                            new_name = os.path.join(destination_folder, source_name + '.dcm')
                            shutil.move(source_item, new_name)
                        else:
                            # Move other files
                            shutil.move(os.path.join(source_folder, item), destination_folder)

                    # Delete the original folder after moving all files
                    shutil.rmtree(source_folder)
