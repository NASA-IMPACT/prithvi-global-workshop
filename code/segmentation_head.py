import torch
import torch.nn as nn

class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvModule, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activate = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activate(x)
        return x

# Define FCNHead
class FCNHead(nn.Module):
    def __init__(self, in_channels, channels, num_classes, dropout_p=0.1):
        super(FCNHead, self).__init__()

        #self.loss_decode = DiceLoss()  # Dice loss

        self.convs = nn.Sequential(
            ConvModule(in_channels, channels, kernel_size=3, stride=1, padding=1)
        )
        self.dropout = nn.Dropout2d(p=dropout_p)
        self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)  # Conv2d for segmentation

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize conv_seg weights with Normal distribution (std=0.01)
        nn.init.normal_(self.conv_seg.weight, std=0.01)
        if self.conv_seg.bias is not None:
            nn.init.constant_(self.conv_seg.bias, 0)

    def forward(self, x):
        x = self.convs(x)
        x = self.dropout(x)
        x = self.conv_seg(x)  # Final segmentation output
        return x
