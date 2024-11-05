import torch.nn as nn
import torch.nn.functional as F
from prithvi import Prithvi
from segmentation_head import SegmentationHead


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
        prithvi_config
    ):
        super(PrithviWrapper, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.pr_weight = prithvi_weight
        self.pr_config = prithvi_config
        self.n_frame = n_frame
        self.input_size = input_size
        self.embed_size = embed_size
        self.patch_size = patch_size

        self.prithvi = Prithvi(self.pr_weight, self.pr_config, n_frame, input_size)

        self.head = SegmentationHead(self.embed_size, self.n_classes, self.n_frame, self.input_size,self.patch_size)

    def forward(self, x):
        pri_out = self.prithvi(x, None, None, 0)[:,1:,:] #eliminate class token
        pri_out = pri_out.transpose(1, 2)
        out = self.head(pri_out)

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


