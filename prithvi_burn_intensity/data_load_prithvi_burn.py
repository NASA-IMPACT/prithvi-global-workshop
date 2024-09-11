import numpy as np
import torch
from torch.utils.data import Dataset
import rasterio
import os

NO_DATA = -9999
NO_DATA_FLOAT = 0.0001
PERCENTILES = (0.1, 99.9)

def load_raster(path,if_img,crop=None):
        with rasterio.open(path) as src:
            img = src.read()

            img = np.where(img == NO_DATA, NO_DATA_FLOAT, img)
            #print("img size",img.shape)
           
            
            if crop:
                img = img[:, -crop[0]:, -crop[1]:]
        return img

def preprocess_image(image,means,stds):
        # normalize image
        means1 = means.reshape(-1,1,1)  # Mean across height and width, for each channel
        stds1 = stds.reshape(-1,1,1)    # Std deviation across height and width, for each channel
        normalized = image.copy()
        #print("means1  shape",means1.shape)
        #print("stds1 shape",stds1.shape)
        normalized = ((image - means1) / stds1)
        #print("normalized image",normalized)
        
        normalized = torch.from_numpy(normalized.reshape(1, normalized.shape[0], 1, *normalized.shape[-2:])).to(torch.float32)
        return normalized

class burn_dataset(Dataset):
    def __init__(self,path,means,stds):
        self.data_dir=path
        self.means=means
        self.stds=stds
        

    def __len__(self):
        #print("dataset length",len(self.input_plus_mask_path))
        return len(self.data_dir)
    
    
    def __getitem__(self,idx):
        
        image_path_pre=self.data_dir[idx][0]
        image_path_current=self.data_dir[idx][1]
        image_path_post=self.data_dir[idx][2]
        image_path=[image_path_pre,image_path_current,image_path_post]

        mask_path=self.data_dir[idx][3]
        #print("image path:",image_path)

        if_image=1
        image_all=torch.tensor([])
        i=0
        for i in range(3):
            image=load_raster(image_path[i],if_image,crop=(224, 224))
            #print("image size",image.shape)
            #print("means shape",self.means[i].shape)
            #print("stds shape",self.stds[i].shape)
            final_image=preprocess_image(image,self.means[i],self.stds[i])
            #print("final image size",final_image.shape)
            image_all=torch.cat((image_all,final_image),dim=2)
        
        #print("final image size",image_all.shape)

        if_image=0
        final_mask=load_raster(mask_path,if_image,crop=(224, 224))
        #print("mask shape",final_mask.shape)

        image_all=image_all.squeeze(0)
        final_mask=final_mask.squeeze(0)
        #print("squeezed image size",final_image.shape)
        
        
        return image_all,final_mask



