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
                nn.Conv3d(in_channels, out_channels, kernel_size=(1,16,16),stride=(1,16,16),padding=(0,0,0)),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True)
            )
        if type==2:
            self.conv_layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=(1,1,1),stride=(1,1,1),padding=(0,0,0)),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True)
            )
            

    def forward(self, x):
        return self.conv_layer(x)
    
class convT(nn.Module):
    def __init__(self,in_channels, out_channels):
        super(convT, self).__init__()
        
        self.conv_layer = nn.Sequential(
                nn.ConvTranspose3d(in_channels, out_channels, kernel_size=(1,16,16),stride=(1,16,16),padding=(0,0,0)),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True)
                )
    def forward(self, x):
        return self.conv_layer(x)
    
###########################################################################
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
        self.down2 = Conv(64,128,2)
        self.down3 = Conv(128,256,2)
        self.down4 = Conv(256,512,2)
        self.down5 = Conv(512,1024,2) 

        self.prithvi=prithvi(self.pr_weight,self.pr_config,n_frame,input_size)
        #print("initialize prithvi")

        
        self.up1 = Conv(1024, 256,2)
        self.up2 = Conv(512,128,2)
        self.up3 = Conv(256, 64,2)
        self.up4=convT(128,128)
        self.up5 =Conv(128, self.n_classes,2) #eg: n_classes=2 for flood'''
        

    def forward(self, x):
        
        x1 = self.down1(x)  #([batch, 64, 1, 14, 14])
        x2 = self.down2(x1) #([batch, 128, 1, 14, 14])
        x3 = self.down3(x2) #([batch, 256, 1, 14, 14])
        x4 = self.down4(x3) #([batch, 512, 1, 14, 14])
        x5 = self.down5(x4) #([batch, 1024, 1, 14, 14])
        #print("prithvi input shape",x5.shape)

        
        pri_out=self.prithvi(x5,None,None,0) #mask_ratio=0
        #print("prithvi out shape",pri_out.shape)
        
        pri_out=pri_out.transpose(1,2).reshape(x5.shape[-5],-1,x5.shape[-3],x5.shape[-2],x5.shape[-1])
        #print("prithvi out shape",pri_out.shape) #([batch, 512, 1, 14, 14])
        
        x5_concatted=torch.cat((pri_out,x4),dim=1) #([batch, 1024, 1, 14, 14])
        x6 = self.up1(x5_concatted) #([batch, 256, 1, 14, 14])
        x6_concatted=torch.cat((x6,x3),dim=1)#([batch, 512, 1, 14, 14])
        x7=self.up2(x6_concatted)#([batch, 128, 1, 14, 14])
        x7_concatted=torch.cat((x7,x2),dim=1)#([batch, 256, 1, 14, 14])
        x8 = self.up3(x7_concatted) #([batch, 64,1, 14,14])
        x8_concatted=torch.cat((x8,x1),dim=1)#([batch, 128,1, 14, 14])
        x9=self.up4(x8_concatted)#([batch, 128,1, 224,224])
        x10=self.up5(x9)#([batch, n,1, 224, 224])
        
        #print("x10 shape",x10.shape)
        
        return x10 

    
    


