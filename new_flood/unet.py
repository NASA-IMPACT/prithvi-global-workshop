import torch
import torch.nn as nn
import torch.nn.functional as F
from prithvi_global_loader import prithvi

class Conv(nn.Module):
    """(convolution-BatchNorm-ReLU)"""

    def __init__(self, in_channels, out_channels,type):

        super(Conv, self).__init__()

        if type==1:
            self.conv_layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=(1,3,3),stride=(1,2,2),padding=(0,1,1)),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout3d(p=0.5)
            )
        if type==2:
            self.conv_layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=(1,3,3),stride=(1,1,1),padding=(0,1,1)),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout3d(p=0.5))
            
        if type==3:
            self.conv_layer = nn.Sequential(
                nn.ConvTranspose3d(in_channels, out_channels, kernel_size=(1,2,2),stride=(1,2,2),padding=(0,0,0)),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout3d(p=0.5))
            
        if type==4:
            self.conv_layer = nn.Sequential(
                nn.ConvTranspose3d(in_channels, out_channels, kernel_size=(1,1,1),stride=(1,1,1),padding=(0,0,0)),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout3d(p=0.5))
            
        if type==5:
            self.conv_layer = nn.Sequential(
                nn.ConvTranspose3d(in_channels, out_channels, kernel_size=(1,2,2),stride=(1,2,2),padding=(0,0,0)),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True),
                )

    def forward(self, x):
        return self.conv_layer(x)
    
########################################################################################
########################################################################################

class UNet(nn.Module):

    def __init__(self, n_channels, n_classes,n_frame,prithvi_weight,prithvi_config,input_size):
        
        super(UNet, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.pr_weight=prithvi_weight
        self.pr_config=prithvi_config

        #down leg of UNet
        self.down1 = Conv(n_channels, 64,1)
        self.down2 = Conv(64,128,1)
        self.down3 = Conv(128,256,1)
        self.down4 = Conv(256,512,1)
        self.down5 = Conv(512,1024,2) 

        self.prithvi=prithvi(self.pr_weight,self.pr_config,n_frame,input_size)
        #print("initialize prithvi")

        self.up1 = Conv(1024,512,4) 
        self.up2 = Conv(1024, 256,3)
        self.up3 = Conv(512,128,3)
        self.up4 = Conv(256, 64,3)
        self.up5 =Conv(128, self.n_classes,5) #eg: n_classes=2 for flood'''
        

    def forward(self, x):
        
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        #print("prithvi input shape",x5.shape)

        
        pri_out=self.prithvi(x5,None,None,0)[:, 1:, :]#eliminate cls token
        #print("prithvi out shape",pri_out.shape)

        pri_out=pri_out.transpose(1,2).reshape(x5.shape[-5],x5.shape[-4],x5.shape[-3],x5.shape[-2],x5.shape[-1])
        #print("prithvi out shape",pri_out.shape)
        
        x6 = self.up1(pri_out) #([1, 512, 1, 14, 14])
        x6_concatted=torch.cat((x6,x4),dim=1)#([1, 1024, 1, 14, 14])
        x7=self.up2(x6_concatted)#([1, 256, 1, 28, 28])
        x7_concatted=torch.cat((x7,x3),dim=1)#([1, 512, 1, 28, 28])
        x8 = self.up3(x7_concatted) #([1, 128, 1, 56, 56])
        x8_concatted=torch.cat((x8,x2),dim=1)#([1, 256, 1, 56, 56])
        x9=self.up4(x8_concatted)#([1, 64, 1, 112, 112])
        x9_concatted=torch.cat((x9,x1),dim=1)#([1, 128, 1, 112, 112])
        x10=self.up5(x9_concatted)#([1, n, 1, 224, 224])
        
        #print("x10 shape",x10.shape)
        return x10 



