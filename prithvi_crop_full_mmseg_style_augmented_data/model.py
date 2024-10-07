import torch
import torch.nn as nn
import torch.nn.functional as F
#from prithvi_global_loader import prithvi
from neck import Neck
from Head import TemporalViTEncoder
from Seg_head import FCNHead


###########################################################################
########################################################################################

def print_model_details(model):
    for name, module in model.named_modules():
        #print(f"Layer Name: {name}")
        #print(f"Layer Type: {module.__class__.__name__}")
        print("name",name)        


class prithvi_wrapper(nn.Module):
    def __init__(self,n_channels, n_classes,n_frame,embed_size,input_size,patch_size,prithvi_weight,prithvi_config,in_chans):
        super(prithvi_wrapper, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.pr_weight=prithvi_weight
        self.pr_config=prithvi_config
        self.n_frame=n_frame
        self.input_size=input_size
        self.embed_size=embed_size
        self.patch_size=patch_size

        self.tubelet_size=1
        self.depth = 24
        self.num_heads = 16
        self.mlp_ratio = 4.0
        self.norm_layer= nn.LayerNorm
        self.norm_pix_loss = False
        
       
        #(self.pr_weight,self.pr_config,n_frame,input_size,in_chans)

        #initialize and load weights for backbone from prithvi
        self.prithvi_backbone=TemporalViTEncoder(
            self.input_size, 
            self.patch_size,
            self.n_frame,
            self.tubelet_size,
            self.n_channels,
            self.embed_size,
            self.depth,
            self.num_heads,
            self.mlp_ratio,
            self.norm_layer,
            self.norm_pix_loss, 
            self.pr_weight)
        
        #print("model details",print_model_details(self.prithvi_backbone))
        
        #initialize neck
        self.neck_embedding=self.embed_size*self.n_frame
        self.neck=Neck(self.neck_embedding)

        #initialize seg_head
        self.Seg_head=FCNHead(self.neck_embedding, 256, self.n_classes, dropout_p=0.1)

        #print("model details",print_model_details(self.neck))


    def forward(self, x):

        B,C,T,H,W=x.shape  #8,18,1,224,224
        

        #Reshape the input into 6 channel and 3 timeframes
        C=int(C/self.n_frame) #18/3=6
        
        x=x.reshape(B,self.n_frame,C,H,W).contiguous() #8,3,6,224,224
        x=x.transpose(1,2).contiguous() #8,6,3,224,224
        
        #print("x shape",x.shape)
        pri_out=self.prithvi_backbone(x)  #8,589,1024

        #eliminate class token
        pri_out=pri_out[:,1:,:]  #8,588,1024
        
        # change no_patch and embedding size of prithvi_out to fit embedding size of neck(=prithvi_encoder_embed_size*no_frame)
        n_patch=int(pri_out.shape[1]/self.n_frame) #588/3=196
        embed_size_neck=int(self.embed_size*self.n_frame) #1024*3=3096
        
        pri_out=pri_out.reshape(B,self.n_frame,n_patch,self.embed_size).contiguous() #8,3,196,1024
        pri_out=pri_out.transpose(1,2).contiguous() #8,196,3,1024
        pri_out=pri_out.flatten(2) #8,196,3*1024 =8,196,3072
        pri_out=pri_out.transpose(1,2).contiguous() #8,3072,196

        H=int(self.input_size[1]/self.patch_size[1]) #224/16=14
        pri_out=pri_out.reshape(B,embed_size_neck,H,H).contiguous() #8,3072,14,14

        neck_out=self.neck(pri_out) #8, 3072, 224, 224

        out=self.Seg_head(neck_out)#8,13,224,224
        
        return out


