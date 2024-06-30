import os

# Directory paths
base_dir = './data'

# Function to rename images in a directory
def rename_images_in_directory(directory, label):
    for count, filename in enumerate(os.listdir(directory)):
        src = os.path.join(directory, filename)
        if os.path.isfile(src):
            # New filename
            dst = os.path.join(directory, f"{count}_{label}.jpg")
            os.rename(src, dst)

# Iterate through each subdirectory
for subdir in os.listdir(base_dir):
    subdir_path = os.path.join(base_dir, subdir)
    if os.path.isdir(subdir_path):
        label = subdir.replace('-', '_').lower()
        rename_images_in_directory(subdir_path, label)

print("Images renamed successfully.")
