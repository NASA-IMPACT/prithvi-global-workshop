import torch
import torch.nn as nn
import torch.nn.functional as F

class Norm2d(nn.Module):
    def __init__(self, num_features, eps=1e-6):
        super(Norm2d, self).__init__()
        # LayerNorm that operates on each channel, so we normalize across channels only
        self.ln = nn.LayerNorm((num_features,), eps=eps, elementwise_affine=True)

    def forward(self, x):
        # x has shape [batch_size, channels, height, width]
        # First permute to [batch_size, height, width, channels] for LayerNorm
        x1=x
        x2 = x1.permute(0, 2, 3, 1)

        # Apply LayerNorm
        x3 = self.ln(x2)

        # Permute back to original shape [batch_size, channels, height, width]
        x4 = x3.permute(0, 3, 1, 2)
        return x4


class FPN1(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(FPN1, self).__init__()
        self.fpn1 = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size=(2, 2), stride=(2, 2)),
            Norm2d(out_channel,eps=1e-6),  # Custom Norm2d with LayerNorm
            nn.GELU(),
            nn.ConvTranspose2d(out_channel, out_channel, kernel_size=(2, 2), stride=(2, 2))
        )

    def forward(self, x):
        return self.fpn1(x)


class FPN2(nn.Module):
    def __init__(self,out_channel):
        super(FPN2, self).__init__()
        self.fpn1 = nn.Sequential(
            nn.ConvTranspose2d(out_channel, out_channel, kernel_size=(2, 2), stride=(2, 2)),
            Norm2d(out_channel,eps=1e-6),  # Custom Norm2d with LayerNorm
            nn.GELU(),
            nn.ConvTranspose2d(out_channel, out_channel, kernel_size=(2, 2), stride=(2, 2))
        )

    def forward(self, x):
        return self.fpn1(x)



class Neck(nn.Module):
    def __init__(self,embed_size):
        super(Neck, self).__init__()

        self.embed_dim=embed_size

        # Conv layer to upscale the token grid to the desired segmented image size
        self.fpn1=FPN1(self.embed_dim,self.embed_dim)
        self.fpn2=FPN2(self.embed_dim)


    def forward(self, x):

        x = self.fpn1(x)
        x = self.fpn2(x)

        #print("x shape",x.shape)

        return x
