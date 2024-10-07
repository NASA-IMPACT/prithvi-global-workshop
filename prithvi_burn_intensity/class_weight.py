import os
import torch
import numpy as np
from PIL import Image
import rasterio
NO_DATA = -9999
NO_DATA_FLOAT = 0

# Directory where your images are stored
mask_dir = "./mask_path.txt"

with open(mask_dir,"r") as f:
    path_list=f.readlines()
#print(path_list)
mask_dir="/rgroup/aifm/akumar/Burn_Scar/Burn_Scar_cropped_v1/"

num_classes = 5

class_counts = torch.zeros(num_classes)

for mask_name in path_list:
    mask_name=mask_name.strip()
    mask_path = os.path.join(mask_dir, mask_name)
    
    with rasterio.open(mask_path) as src:
            mask = src.read()
    mask = np.where(mask == NO_DATA, NO_DATA_FLOAT, mask)
    
    
    mask_tensor = torch.from_numpy(mask).long()
    #print("mask tensor shape",mask_tensor.shape)
    #print("mask labels", torch.unique(mask_tensor))
    

    if mask_tensor.shape != (1,224, 224):
        raise ValueError(f"Mask {mask_name} does not have the expected shape (224, 224)")

    # Count occurrences of each class in the mask
    class_counts += torch.bincount(mask_tensor.view(-1), minlength=num_classes)

# Total number of pixels across all masks
total_pixels = class_counts.sum()

# Calculate the percentage of each class
class_percentages = (class_counts / total_pixels) * 100

# Print the percentage of each class
for i in range(num_classes):
    print(f"Class {i}: {class_percentages[i]:.2f}%")

'''
Class 0: 64.44%
Class 1: 8.24%
Class 2: 15.10%
Class 3: 10.44%
Class 4: 1.78%
'''