#data_process.py
import torch
import numpy as np
from torch.utils.data import Dataset
import rasterio
import os
import random

img_norm_cfg = dict(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Example values
crop_size = (224, 224)  
tile_size = 224 
bands = [0, 1, 2]  
num_frames = 1 

# Define the training pipeline
train_pipeline = [
    dict(type="LoadGeospatialImageFromFile", to_float32=True),
    dict(type="LoadGeospatialAnnotations", reduce_zero_label=True),
    dict(type="RandomFlip", prob=0.5),
    dict(type="ToTensor", keys=["img", "gt_semantic_seg"]),
    # Change to channels first
    dict(type="TorchPermute", keys=["img"], order=(2, 0, 1)),  # Assuming img initially has shape (H, W, C)
    dict(type="TorchNormalize", **img_norm_cfg),
    dict(type="TorchRandomCrop", crop_size=crop_size),  # Randomly crop the image
    dict(
        type="Reshape",
        keys=["img"],
        new_shape=(len(bands), num_frames, tile_size, tile_size),  # Reshape the image
    ),
    dict(type="Reshape", keys=["gt_semantic_seg"], new_shape=(1, tile_size, tile_size)),  # Reshape the target
    dict(type="CastTensor", keys=["gt_semantic_seg"], new_type="torch.LongTensor"),  # Cast target to LongTensor
    dict(type="Collect", keys=["img", "gt_semantic_seg"]),  # Collect processed tensors
]

##################################################################################################################

NO_DATA = -9999
NO_DATA_FLOAT = 0.0001
PERCENTILES = (0.1, 99.9)

def load_raster(path,if_img,crop=None):
        with rasterio.open(path) as src:
            img = src.read()

            img = np.where(img == NO_DATA, NO_DATA_FLOAT, img)
            
            #crops only lower right corner
            #if crop:
                #img = img[:, -crop[0]:, -crop[1]:]
        return img

def random_crop(tensor, crop_size=(224, 224)):
    # Get original dimensions: channel, height (H), width (W)
    C, H, W = tensor["img"].shape

    # Ensure the crop size fits within the original dimensions
    crop_h, crop_w = crop_size
    if H < crop_h or W < crop_w:
        raise ValueError(f"Original size ({H}, {W}) is smaller than the crop size ({crop_h}, {crop_w})")

    # Randomly select the top-left corner for the crop
    top = random.randint(0, H - crop_h)
    left = random.randint(0, W - crop_w)

    # Perform the crop (channel dimension remains unchanged)
    tensor["img"] = tensor["img"][:, top:top + crop_h, left:left + crop_w]
    tensor["mask"] = tensor["mask"][top:top + crop_h, left:left + crop_w]

    return tensor



# Example processing function to simulate the pipeline
def process_input(input_array,mask,img_norm_cfg):
    
    img_tensor = torch.from_numpy(input_array).float()  # Assuming input_array is of type np.float32
    mask_tensor = torch.from_numpy(mask)

    processed_data = {}
    
    # step['type'] == "RandomFlip":
    p=np.random.rand()
    if p < 0.5:
        processed_data['img'] = torch.flip(img_tensor, [2])  # Flip along width
        processed_data['mask'] = torch.flip(mask_tensor, [2])  # Flip along width

    print("flipped img shape",processed_data['img'].shape)
    print("flipped mask shape",processed_data['mask'].shape)

    mean=torch.tensor(img_norm_cfg['mean']).view(-1, 1, 1)
    std=torch.tensor(img_norm_cfg['std']).view(-1, 1, 1)

    print("mean shape",mean.shape)
    print("std shape",std.shape)

    # step['type'] == "TorchNormalize":
    processed_data['img'] = (processed_data['img'] - mean)/ std
    print("normalized img shape",processed_data['img'].shape)

    # step['type'] == "TorchRandomCrop":
    processed_data = random_crop(processed_data, (224, 224))
    print("cropped img shape",processed_data['img'].shape)
    print("cropped mask shape",processed_data['mask'].shape)

    return processed_data

# Example usage
#image_path=self.data_dir[idx][0]
input_array = np.random.rand(18, 512, 512).astype(np.float32)  # Example input
mask = np.random.randint(0, 1, size=(512, 512))  # Example input
img_norm_cfg={}
img_norm_cfg["mean"]=[494.905781,815.239594,924.335066,2968.881459,2634.621962,1739.579917,
        494.905781,815.239594,924.335066,2968.881459,2634.621962,1739.579917,
        494.905781,815.239594,924.335066,2968.881459,2634.621962,1739.579917]

img_norm_cfg["std"]=[
        284.925432,
        357.84876,
        575.566823,
        896.601013,
        951.900334,
        921.407808,
        284.925432,
        357.84876,
        575.566823,
        896.601013,
        951.900334,
        921.407808,
        284.925432,
        357.84876,
        575.566823,
        896.601013,
        951.900334,
        921.407808
    ]
processed_data = process_input(input_array,mask,img_norm_cfg)