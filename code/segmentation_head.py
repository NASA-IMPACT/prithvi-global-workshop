from torch.nn import nn
from lib.model_utils import upscaling_block, upscaling_block2

class SegmentationHead(nn.Module):
    def __init__(self,embed_size,n_classes,n_frame,input_size,patch_size):
        super(SegmentationHead, self).__init__()

        self.image_size = input_size
        self.grid_dim = input_size[-1]//patch_size[-1]
        self.embed_dim = embed_size
        self.n_frame = n_frame

        # Conv layer to upscale the token grid to the desired segmented image size
        self.up1 = upscaling_block(self.embed_dim,256)
        self.up2 = upscaling_block(256,128)
        self.up3 = upscaling_block(128,64)
        self.up4 = upscaling_block(64,32)
        self.up5 = upscaling_block2(32,n_classes)

    def forward(self, x):
        batch_size = x.size(0)

        # Reshape x to (batch, embed_size, grid_dim, grid_dim)

        x = x.reshape(batch_size,self.embed_dim,self.grid_dim, self.grid_dim)
        #print("x shape:",x.shape)
        x = self.up1(x)
        #print("x shape:",x.shape)
        x = self.up2(x)
        #print("x shape:",x.shape)
        x = self.up3(x)
        #print("x shape:",x.shape)
        x = self.up4(x)
        #print("x shape:",x.shape)
        x=self.up5(x)
        #print("x shape:",x.shape)

        return x
