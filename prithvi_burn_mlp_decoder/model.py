import torch
import torch.nn as nn
import torch.nn.functional as F
from prithvi_global_loader import prithvi

class MLPBlock(nn.Module):
    def __init__(self, embed_dim, mlp_dim, dropout=0.1):
        super(MLPBlock, self).__init__()
        self.fc1 = nn.Linear(embed_dim, mlp_dim)
        self.gelu = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
       

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout1(x)
        return x

def upscaling_block(scale_factor=16):

        block=nn.Sequential(
                nn.Upsample(scale_factor=16)
            )
        return block
    

class SegmentationHead(nn.Module):
    def __init__(self,dec_embed_size,n_classes,n_frame,input_size,patch_size):
        super(SegmentationHead, self).__init__()
        
        self.image_size = input_size
        self.grid_dim=input_size[-1]//patch_size[-1]
        self.dec_embed_dim=dec_embed_size
        self.n_frame=n_frame
        self.n_classes=n_classes
        scale_factor=16

        self.mlp1=MLPBlock(self.dec_embed_dim,n_classes)
        self.up1 = upscaling_block(scale_factor)
        #self.mlp2=MLPBlock(6,self.n_classes)

    def forward(self, x):
        batch_size = x.size(0)
        
        # Reshape x to (batch, embed_size, grid_dim* grid_dim)
        #print("x shape",x.shape)
        x = x.reshape(batch_size,self.dec_embed_dim,-1)
        #print("x shape:",x.shape)
        x=x.reshape(batch_size,x.shape[1],self.grid_dim,self.grid_dim)
        #print("x shape:",x.shape)
        x=self.up1(x)

        x=x.transpose(1,2)
        #print("x shape:",x.shape)
        x=self.mlp1(x)
        #print("x shape:",x.shape)
        x=x.transpose(1,2)
        
        #print("x shape:",x.shape)
        
        return x

###########################################################################
########################################################################################

class prithvi_wrapper(nn.Module):
    def __init__(self, n_channels, n_classes,n_frame,dec_embed_size,input_size,patch_size,prithvi_weight,prithvi_config):
        super(prithvi_wrapper, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.pr_weight=prithvi_weight
        self.pr_config=prithvi_config
        self.n_frame=n_frame
        self.input_size=input_size
        self.dec_embed_size=dec_embed_size
        self.patch_size=patch_size
 
        self.prithvi=prithvi(self.pr_weight,self.pr_config,n_frame,input_size)
        #print("initialize prithvi")
        

        self.head=SegmentationHead(self.dec_embed_size, self.n_classes, self.n_frame,self.input_size,self.patch_size)

    def forward(self, x):

        #print("x shape",x.shape)
        pri_out=self.prithvi(x,None,None,0)
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


