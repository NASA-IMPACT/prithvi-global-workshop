import numpy as np
import torch
from torch.utils.data import Dataset
import rasterio
import os
import random

from lib.utils import load_raster, process_input


class DownstreamDataset(Dataset):
    def __init__(self,path,means,stds,config):
        self.data_dir=path
        self.means=means
        self.stds=stds
        self.case=config["case"]


    def __len__(self):
        #print("dataset length",len(self.input_plus_mask_path))
        return len(self.data_dir)


    def __getitem__(self,idx):

        image_path=self.data_dir[idx][0]
        mask_path=self.data_dir[idx][1]
        #print("image path:",image_path)

        if_img=1
        input_array = load_raster(image_path,if_img,crop=None)
        if self.case=="flood":
            input_array=input_array[[1,2,3,8,11,12],:,:]

        if_img=0
        mask_array = load_raster(mask_path,if_img,crop=None)

        img_norm_cfg={}
        img_norm_cfg["mean"]=self.means
        img_norm_cfg["std"]=self.stds


        img,mask = process_input(input_array,mask_array,img_norm_cfg)
        img=img.unsqueeze(1)
        mask=mask.squeeze(0)
        return img.float(),mask
