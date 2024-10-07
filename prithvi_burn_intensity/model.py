import torch
import torch.nn as nn
import torch.nn.functional as F
from prithvi_global_loader import prithvi

def upscaling_block(in_channels, out_channels):

        block=nn.Sequential(
                nn.ConvTranspose3d(in_channels, out_channels, kernel_size=(1,2,2),stride=(1,2,2),padding=(0,0,0)),
                nn.BatchNorm3d(out_channels),
                nn.GELU()
            )
        return block

def upscaling_block2(in_channels, out_channels):

        block=nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm3d(out_channels),
                nn.GELU()
            )
        return block

def upscaling_block3(n_channels):

        block=nn.Sequential(
                nn.Conv3d(n_channels, n_channels, kernel_size=(3,1,1),stride=(1,1,1)),
                nn.BatchNorm3d(n_channels),
                nn.GELU()
            )
        return block
    

class SegmentationHead(nn.Module):
    def __init__(self,embed_size,n_classes,n_frame,input_size,patch_size):
        super(SegmentationHead, self).__init__()
        
        self.image_size = input_size
        self.grid_dim=input_size[-1]//patch_size[-1]
        self.embed_dim=embed_size
        self.n_frame=n_frame

        # Conv layer to upscale the token grid to the desired segmented image size
        self.up1 = upscaling_block(self.embed_dim,256)
        self.up2 = upscaling_block(256,128)
        self.up3 = upscaling_block(128,64)
        self.up4 = upscaling_block(64,32)
        self.up5 = upscaling_block2(32,n_classes)
        self.up6 = upscaling_block3(n_classes)

    def forward(self, x):
        batch_size = x.size(0)
        
        # Reshape x to (batch, embed_size, grid_dim, grid_dim)

        x = x.reshape(batch_size,-1,3,self.grid_dim, self.grid_dim)
        #print("x shape:",x.shape)
        x = self.up1(x)
        #print("x1 shape:",x.shape)
        x = self.up2(x)
        #print("x2 shape:",x.shape)
        x = self.up3(x)
        #print("x3 shape:",x.shape)
        x = self.up4(x)
        #print("x4 shape:",x.shape)
        x=self.up5(x)
        #print("x5 shape:",x.shape)
        x=self.up6(x)
        #print("x6 shape:",x.shape)
        x=x.squeeze(2)
        #print("x end shape:",x.shape)
       
        return x

###########################################################################
########################################################################################

class prithvi_wrapper(nn.Module):
    def __init__(self, n_channels, n_classes,n_frame,embed_size,input_size,patch_size,prithvi_weight,prithvi_config):
        super(prithvi_wrapper, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.pr_weight=prithvi_weight
        self.pr_config=prithvi_config
        self.n_frame=n_frame
        self.input_size=input_size
        self.embed_size=embed_size
        self.patch_size=patch_size
 
        self.prithvi=prithvi(self.pr_weight,self.pr_config,n_frame,input_size)
        #print("initialize prithvi")
        

        self.head=SegmentationHead(self.embed_size, self.n_classes, self.n_frame,self.input_size,self.patch_size)

    def forward(self, x):

        #print("x shape",x.shape)
        pri_out=self.prithvi(x,None,None,0)[:,1:,:] #eliminate class token
        #print("prithvi out shape",pri_out.shape)

        pri_out=pri_out.transpose(1,2)
        #print("prithvi out shape",pri_out.shape)
        
        out=self.head(pri_out)
        return out

    
    



