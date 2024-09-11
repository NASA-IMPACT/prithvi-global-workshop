import os
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import rasterio
NO_DATA = -9999
NO_DATA_FLOAT = 0.0001

# Directory where your images are stored
image_dir = "./image_path.txt"

with open(image_dir,"r") as f:
    path_list=f.readlines()
#print(path_list)
image_dir="/rgroup/aifm/akumar/Burn_Scar/Burn_Scar_cropped_v1/"


# Initialize lists to store the sums and squared sums of the pixels
channel_sums = torch.zeros(6)
channel_squared_sums = torch.zeros(6)
num_pixels_per_channel = 0

# Loop through all the images in the directory
for image_name in path_list:

    image_name=image_name.strip()
    image_path = os.path.join(image_dir, image_name)

    with rasterio.open(image_path) as src:
            img = src.read()
    img = np.where(img == NO_DATA, NO_DATA_FLOAT, img)
    

    tensor_image = torch.tensor(img)

    # Update the sum and squared sum for each channel
    channel_sums += tensor_image.sum(dim=[1, 2])  # Sum over height and width (dim 1 and 2)
    channel_squared_sums += (tensor_image ** 2).sum(dim=[1, 2])  # Squared sum over height and width
    num_pixels_per_channel += tensor_image.size(1) * tensor_image.size(2)  # Number of pixels per channel

# Calculate mean and standard deviation for each channel
means = channel_sums / num_pixels_per_channel
stds = torch.sqrt((channel_squared_sums / num_pixels_per_channel) - (means ** 2))

print("Means: ", means)
print("Standard Deviations: ", stds)

'''
Means:  tensor([1027.0479, 1230.4219, 1270.1404, 1895.1371, 1366.3008, 2467.9395])
Standard Deviations:  tensor([1735.9897, 1725.6100, 1687.5278, 1252.5571, 1122.0760, 1522.4668])
'''