import torch
import torch.nn as nn
import torch.optim as optim
import os
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import yaml
from Prithvi_global_v1.mae.models_mae import MaskedAutoencoderViT
import pickle
from Prithvi_global_v1.mae.config import get_config
import argparse
from typing import Optional
from functools import partial
import peft

def LORA_peft(model,Lora_peft_layer_name_pre,Lora_peft_layer_name_suffix,LP_layer_no_start,LP_layer_no_end):

    target_modules: list[str] = [f"{Lora_peft_layer_name_pre}.{i}.{Lora_peft_layer_name_suffix}{1}" 
                                 for i in range(LP_layer_no_start,LP_layer_no_end+1)
                                ] + [
                                f"{Lora_peft_layer_name_pre}.{i}.{Lora_peft_layer_name_suffix}{2}" 
                                for i in range(LP_layer_no_start,LP_layer_no_end+1)
                                ]

    
    peft_config = peft.LoraConfig(
        r=10,
        lora_alpha=8,
        lora_dropout=0,
        bias="none",
        target_modules=target_modules,
        # modules_to_save=["model.head"],
        )
    lora_model = peft.get_peft_model(model=model, peft_config=peft_config)
    return lora_model



class prithvi(nn.Module):
    def __init__(self,prithvi_weight,prithvi_config,n_frame,input_size,config):
        super(prithvi,self).__init__()

        self.weights_path =prithvi_weight
        self.checkpoint = torch.load(self.weights_path)

        #prithvi_config = prithvi_config

        
        self.prithvi_model=MaskedAutoencoderViT(input_size=input_size,
                                 patch_size=prithvi_config.MODEL.PATCH_SIZE,
                                 in_chans=len(prithvi_config.DATA.BANDS),
                                 embed_dim=prithvi_config.MODEL.EMBED_DIM,
                                 depth=prithvi_config.MODEL.DEPTH,
                                 num_heads=prithvi_config.MODEL.NUM_HEADS,
                                 decoder_embed_dim=prithvi_config.MODEL.DECODER_EMBED_DIM,
                                 decoder_depth=prithvi_config.MODEL.DECODER_DEPTH,
                                 decoder_num_heads=prithvi_config.MODEL.DECODER_NUM_HEADS,
                                 mlp_ratio=prithvi_config.MODEL.MLP_RATIO,
                                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                 norm_pix_loss=prithvi_config.MODEL.NORM_PIX_LOSS,
                                 coords_encoding=prithvi_config.MODEL.COORDS_ENCODING,
                                 coords_drop_rate=prithvi_config.MODEL.COORDS_DROP_RATE,
                                 coords_scale_learn=prithvi_config.MODEL.COORDS_SCALE_LEARN,
                                 drop_channels_rate=prithvi_config.MODEL.DROP_CHANNELS_RATE)

        
        print("checkpoint names:",self.checkpoint.keys())
         
        _ = self.prithvi_model.load_state_dict(self.checkpoint, strict=False)
        #print("initialized model")

        #Apply LORA peft on prithvi Transformer's encoder's linear layers only 
        Lora_peft_layer_name_pre=config["Lora_peft_layer_name"][0]
        Lora_peft_layer_name_suffix=config["Lora_peft_layer_name"][1]
        LP_layer_no_start=config["Lora_peft_layer_no"][0]
        LP_layer_no_end=config["Lora_peft_layer_no"][1]
        self.prithvi_model=LORA_peft(self.prithvi_model,Lora_peft_layer_name_pre,Lora_peft_layer_name_suffix,LP_layer_no_start,LP_layer_no_end)
    

    def forward(self,x,temp,loc,mask):
        
        latent, mask, ids_restore = self.prithvi_model.forward_encoder(x, temp, loc, 0)#mask_ratio=0
        
        pred=latent

        return pred

