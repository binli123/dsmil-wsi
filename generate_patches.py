import os
import numpy as np
from patchify import patchify
from PIL import Image

# Input and output directories
input_dir = "/ssd_scratch/karan.p/train"
output_dir = "/ssd_scratch/karan.p/patches_20X"

# Define patch size and step size
patch_size = (224, 224, 3)
step_size = 224  # Adjust as needed for overlapping patches

def resize_to_nearest_multiple(image, patch_size):
    width, height = image.size
    new_width = ((width//5) // patch_size[0]) * patch_size[0]
    new_height = ((height//5) // patch_size[1]) * patch_size[1]

    new_height = 224 if new_height == 0 else new_height
    new_width = 224 if new_width == 0  else new_width
    
    return image.resize((new_width, new_height))

# Function to create patches
def create_patches(input_path, output_path, patch_size, step_size):
    for root, dirs, files in os.walk(input_path):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg', '.tiff')):  # Add extensions as needed
                img_path = os.path.join(root, file)
                img = Image.open(img_path)
                img = resize_to_nearest_multiple(img, patch_size)
                img = np.array(img)

                # Create patches
                patches = patchify(img, patch_size, step=step_size)
                relative_path = os.path.relpath(root, input_path)

                # Ensure output directories exist
                save_dir = os.path.join(output_path, relative_path)
                os.makedirs(save_dir, exist_ok=True)

                # Save patches
                for i in range(patches.shape[0]):
                    for j in range(patches.shape[1]):
                        patch = patches[i, j, 0]  # Remove the channel dimension if not 3D
                        patch_img = Image.fromarray(patch)
                        patch_filename = f"{file.split('.')[0]}_patch_{i}_{j}.png"
                        patch_img.save(os.path.join(save_dir, patch_filename))

# Iterate through each category folder
for category in os.listdir(input_dir):
    category_path = os.path.join(input_dir, category)
    if os.path.isdir(category_path):
        create_patches(category_path, os.path.join(output_dir, category), patch_size, step_size)

print("Patch generation completed.")
