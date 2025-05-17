import os
import shutil
import random

# Calculate the number of files to copy (e.g., 30% of the files)

# Define the source and destination directories

path = './dataset/data/'
for seed in [10,20,30,40,50]:
    for source_dir,data_ratio in zip([path+'avoiding/data', path+'aligning/all_data/state',path+'stacking/all_data'],([10,5,2,1],[10,2,1,0.6],[50,25,10,5])):
        for r in data_ratio:
            destination_dir = source_dir+f'_ratio{r}_seed{seed}'

            # Make sure the destination directory exists
            os.makedirs(destination_dir, exist_ok=True)

            # Get a list of files in the source directory
            files = [file for file in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, file))]
            number_of_files_to_copy = round(len(files) * r/100)


            # Randomly select the files to copy
            files_to_copy = random.sample(files, number_of_files_to_copy)

            # Copy the selected files to the destination directory
            for file in files_to_copy:
                source_file_path = os.path.join(source_dir, file)
                destination_file_path = os.path.join(destination_dir, file)
                shutil.copy2(source_file_path, destination_file_path)

            print(f"Copied {number_of_files_to_copy} files from {source_dir} to {destination_dir}.")
