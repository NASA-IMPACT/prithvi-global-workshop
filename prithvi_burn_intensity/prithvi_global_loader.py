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


class prithvi(nn.Module):
    def __init__(self,prithvi_weight,prithvi_config,n_frame,input_size):
        super(prithvi,self).__init__()

        self.weights_path =prithvi_weight
        self.checkpoint = torch.load(self.weights_path)

        config = prithvi_config

        
        self.prithvi_model=MaskedAutoencoderViT(input_size=input_size,
                                 patch_size=config.MODEL.PATCH_SIZE,
                                 in_chans=len(config.DATA.BANDS),
                                 embed_dim=config.MODEL.EMBED_DIM,
                                 depth=config.MODEL.DEPTH,
                                 num_heads=config.MODEL.NUM_HEADS,
                                 decoder_embed_dim=config.MODEL.DECODER_EMBED_DIM,
                                 decoder_depth=config.MODEL.DECODER_DEPTH,
                                 decoder_num_heads=config.MODEL.DECODER_NUM_HEADS,
                                 mlp_ratio=config.MODEL.MLP_RATIO,
                                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                 norm_pix_loss=config.MODEL.NORM_PIX_LOSS,
                                 coords_encoding=config.MODEL.COORDS_ENCODING,
                                 coords_drop_rate=config.MODEL.COORDS_DROP_RATE,
                                 coords_scale_learn=config.MODEL.COORDS_SCALE_LEARN,
                                 drop_channels_rate=config.MODEL.DROP_CHANNELS_RATE)

        
        print("checkpoint names:",self.checkpoint.keys())
         
        _ = self.prithvi_model.load_state_dict(self.checkpoint, strict=False)
        #print("initialized model")
        
    def forward(self,x,temp,loc,mask):
        #print("forward function")
        latent, mask, ids_restore = self.prithvi_model.forward_encoder(x, temp, loc, 0)#mask_ratio=0
        #print("latent shape",latent.shape)
        #pred = self.prithvi_model.forward_decoder(latent, ids_restore, temp, loc)
        #print("pred shape",pred.shape)
        return latent

'''
#initialize model    
pr_weight="/rhome/rghosal/Rinki/rinki-hls-foundation-os/Prithvi_global.pt"

pr_config= get_config(None)  
print("model:",pr_config)

n_frame=3
model_prithvi=prithvi(pr_weight,pr_config,n_frame)

#check blocks of the model after loading its weights
######print("loaded model blocks",model_pvi.pvi_model)

input_tensor = torch.randn(1, 6, 3, 224, 224)  # Input batch,channel,frame,height,width
temp_coord=None #torch.randn((1,366),1, 3,2)
loc_coord=None #torch.randn((0,90),1, 2)
with torch.no_grad():
    prithvi_loaded=model_prithvi(input_tensor,temp_coord,loc_coord,0)

print("prithvi forward passed")'''

