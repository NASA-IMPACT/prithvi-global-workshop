import torch.nn as nn
#from prithvi_global_loader import prithvi
from neck import Neck
from prithvi_global.mae.models_mae import TemporalViTEncoder
from segmentation_head import FCNHead
from lib.utils import print_model_details


class PrithviWrapper(nn.Module):
    def __init__(
        self,
        n_channels,
        n_classes,
        n_frame,
        embed_size,
        input_size,
        patch_size,
        prithvi_weight,
        prithvi_config,
        in_chans,
        config
    ):
        super(PrithviWrapper, self).__init__()

        self.n_channels  =  n_channels
        self.n_classes  =  n_classes
        self.pr_weight = prithvi_weight
        self.pr_config = prithvi_config
        self.n_frame = n_frame
        self.input_size = input_size
        self.embed_size = embed_size
        self.patch_size = patch_size

        self.tubelet_size = 1
        self.depth = config["prithvi_backbone"]["depth"]
        self.num_heads = config["prithvi_backbone"]["num_heads"]
        self.mlp_ratio = config["prithvi_backbone"]["mlp_ratio"]
        self.norm_layer = nn.LayerNorm
        self.norm_pix_loss = False

        #initialize and load weights for backbone from prithvi
        self.prithvi_backbone = TemporalViTEncoder(
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
            self.pr_weight
        )

        #initialize neck
        self.neck_embedding = self.embed_size * self.n_frame
        self.neck = Neck(self.neck_embedding)

        #initialize seg_head
        self.Seg_head = FCNHead(self.neck_embedding, 256, self.n_classes, dropout_p=0.1)

        #print("model details",print_model_details(self.neck))


    def forward(self, x):

        B, C, T, H, W = x.shape  #8,18,1,224,224


        #Reshape the input into 6 channel and 3 timeframes
        C = int(C/self.n_frame) #18/3 = 6

        x1 = x
        x2 = x1.reshape(B,self.n_frame,C,H,W) #8,3,6,224,224
        x3 = x2.transpose(1,2) #8,6,3,224,224


        #print("x shape",x.shape)
        pri_out = self.prithvi_backbone(x3)  #8,589,1024

        #eliminate class token
        pri_out = pri_out[:,1:,:]  #8,588,1024

        # change no_patch and embedding size of prithvi_out to fit embedding size of neck( = prithvi_encoder_embed_size*no_frame)
        n_patch = int(pri_out.shape[1] / self.n_frame) #588/3 = 196
        embed_size_neck = int(self.embed_size * self.n_frame) #1024*3 = 3096

        pri_op1 = pri_out
        pri_op2 = pri_op1.reshape(B, self.n_frame, n_patch, self.embed_size) #8,3,196,1024
        pri_op3 = pri_op2.transpose(1, 2) #8,196,3,1024
        pri_op4 = pri_op3.flatten(2) #8,196,3*1024  = 8,196,3072
        pri_op5 = pri_op4.transpose(1, 2) #8,3072,196

        H = int(self.input_size[1] / self.patch_size[1]) #224/16 = 14
        pri_out = pri_op5.reshape(B, embed_size_neck, H, H) #8,3072,14,14

        neck_out = self.neck(pri_out) #8, 3072, 224, 224

        out = self.Seg_head(neck_out)#8,13,224,224

        return out
