import numpy as np
import torch
from torch.utils.data import Dataset
import rasterio
import os
import random

NO_DATA = -9999
NO_DATA_FLOAT = 0.0001
PERCENTILES = (0.1, 99.9)

def load_raster(path,if_img,crop=None):
        with rasterio.open(path) as src:
            img = src.read()

            img = np.where(img == NO_DATA, NO_DATA_FLOAT, img)
            
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
    tensor["mask"] = tensor["mask"][:,top:top + crop_h, left:left + crop_w]

    return tensor



# Example processing function to simulate the pipeline
def process_input(input_array,mask,img_norm_cfg):
    
    input_array=input_array.astype(np.float32)
    img_tensor = torch.from_numpy(input_array)  # Assuming input_array is of type np.float32
    img_tensor=img_tensor.float()
    mask_tensor = torch.from_numpy(mask)

    processed_data = {}
    processed_data['img'] = img_tensor  
    processed_data['mask'] = mask_tensor
    # step['type'] == "RandomFlip":
    p=np.random.rand()
    if p < 0.5:
        processed_data['img'] = torch.flip(img_tensor, [2])  # Flip along width
        processed_data['mask'] = torch.flip(mask_tensor, [2])  # Flip along width

    #print("flipped img shape",processed_data['img'].shape)
    #print("flipped mask shape",processed_data['mask'].shape)

    mean=torch.tensor(img_norm_cfg['mean']).view(-1, 1, 1)
    std=torch.tensor(img_norm_cfg['std']).view(-1, 1, 1)

    #print("mean shape",mean.shape)
    #print("std shape",std.shape)

    # step['type'] == "TorchNormalize":
    processed_data['img'] = (processed_data['img'] - mean)/ std
    #print("normalized img shape",processed_data['img'].shape)

    # step['type'] == "TorchRandomCrop":
    processed_data = random_crop(processed_data, (224, 224))
    #print("cropped img shape",processed_data['img'].shape)
    #print("cropped mask shape",processed_data['mask'].shape)

    return processed_data['img'],processed_data['mask']



class crop_dataset(Dataset):
    def __init__(self,path,means,stds):
        self.data_dir=path
        self.means=means
        self.stds=stds
        

    def __len__(self):
        #print("dataset length",len(self.input_plus_mask_path))
        return len(self.data_dir)
    
    
    def __getitem__(self,idx):
        
        image_path=self.data_dir[idx][0]
        mask_path=self.data_dir[idx][1]
        #print("image path:",image_path)

        if_img=1
        input_array = load_raster(image_path,if_img,crop=None)

        if_img=0
        mask_array = load_raster(mask_path,if_img,crop=None)

        img_norm_cfg={}
        img_norm_cfg["mean"]=self.means
        img_norm_cfg["std"]=self.stds
         

        img,mask = process_input(input_array,mask_array,img_norm_cfg)
        img=img.unsqueeze(1)
        mask=mask.squeeze(0)
        return img.float(),mask



