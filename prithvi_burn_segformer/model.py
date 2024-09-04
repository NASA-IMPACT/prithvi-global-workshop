import torch
import torch.nn as nn
import torch.nn.functional as F
from prithvi_global_loader import prithvi
from head import SegmentationHead
###########################################################################
########################################################################################

class prithvi_wrapper(nn.Module):
    def __init__(self, segmentor_config,n_channels, n_classes,n_frame,embed_size,input_size,patch_size,prithvi_weight,prithvi_config):
        super(prithvi_wrapper, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.pr_weight=prithvi_weight
        self.pr_config=prithvi_config
        self.n_frame=n_frame
        self.input_size=input_size
        self.embed_size=embed_size
        self.patch_size=patch_size
        self.seg_conf=segmentor_config
 
        self.prithvi=prithvi(self.pr_weight,self.pr_config,n_frame,input_size)
        #print("initialize prithvi")

        n_layers=self.seg_conf["n_layers"]
        n_heads=self.seg_conf["n_heads"]
        d_model=self.seg_conf["decoder_embed_dim"]
        d_ff=self.seg_conf["mlp"]
        drop_path_rate=self.seg_conf["drop_path_rate"]
        dropout=self.seg_conf["dropout"]

        self.head=SegmentationHead(self.n_classes,self.patch_size[1],self.embed_size,n_layers,
                                   n_heads,d_model,d_ff,drop_path_rate,dropout)
        

    def forward(self, x):

        #print("x shape",x.shape)
        pri_out=self.prithvi(x,None,None,0)

        #eliminate class token
        pri_out=pri_out[:,1:,]

        final=self.head(pri_out,(224,224))

        return final



