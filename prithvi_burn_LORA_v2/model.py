import torch
import torch.nn as nn
import torch.nn.functional as F
from prithvi_global_loader import prithvi

class Norm2d(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x
    
def upscaling_block(in_channels, out_channels):

       fpn1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(1,1),
                stride=(1,1)
            ),
            Norm2d(out_channels),
            nn.GELU(),
            nn.ConvTranspose2d(
                out_channels,
                out_channels,
                kernel_size=(2,2),
                stride=(2,2)
            )
        )
       
       return fpn1

def upscaling_block2(out_channels):

        fpn2 = nn.Sequential(
            nn.ConvTranspose2d(
                out_channels,
                out_channels,
                kernel_size=(2,2),
                stride=(2,2)
            ),
            Norm2d(out_channels),
            nn.GELU(),
            nn.ConvTranspose2d(
                out_channels,
                out_channels,
                kernel_size=(2,2),
                stride=(2,2)
            )
        )
         
        return fpn2

def upscaling_block3(out_channels):
    fpn3=nn.Sequential(
            nn.ConvTranspose2d(
                out_channels,
                out_channels,
                kernel_size=(2,2),
                stride=(2,2)
            ),
            Norm2d(out_channels),
            nn.GELU())
    
    return fpn3
    

class SegmentationHead(nn.Module):
    def __init__(self,embed_size,n_classes,n_frame,input_size,patch_size):
        super(SegmentationHead, self).__init__()
        
        self.image_size = input_size
        self.grid_dim=input_size[-1]//patch_size[-1]
        self.embed_dim=embed_size
        self.n_frame=n_frame

        # Conv layer to upscale the token grid to the desired segmented image size
        self.up1=upscaling_block(self.embed_dim,n_classes)
        self.up2=upscaling_block2(n_classes)
        self.up3=upscaling_block3(n_classes)

    def forward(self, x):
        batch_size = x.size(0)
        
        # Reshape x to (batch, embed_size, grid_dim, grid_dim)

        x = x.reshape(batch_size,self.embed_dim,self.grid_dim, self.grid_dim)
        
        x = self.up1(x)
        x = self.up2(x)
        x=self.up3(x)

        #print("x shape",x.shape)
        
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

    
    
'''
input_tensor = torch.randn(1, 12, 1, 224, 224)  # Input batch,channel,frame,height,width
mask=torch.randint(0,2,(1,224,224))
print("mask shape",mask.shape)
#print("input shape",input_tensor.shape)
unet=UNet(12, 2) #in_channel=12,n_class=2  #burn scar has 2 classes
pred=unet(input_tensor)
print("pred shape",pred.shape)
print("unet output shape",pred.shape)
loss=UNet.segmentation_loss(mask,pred)
print("loss",loss)'''


