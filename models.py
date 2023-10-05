import torch
import torch.nn as nn
import numpy as np

class UNet(nn.Module):
    
    def __init__(self):
        super(UNet, self).__init__()

        self.max_pool = nn.MaxPool3d(kernel_size = 2, stride = 2)
        self.down_conv_1 = double_conv(1, 64)
        self.down_conv_2 = double_conv(64, 128)
        self.down_conv_3 = double_conv(128, 256)
        self.down_conv_4 = double_conv(256, 512)
 
        self.up_trans_1 = nn.ConvTranspose3d(
                            in_channels = 512,
                            out_channels = 256,
                            kernel_size = 2,
                            stride = 2 
                        )   
        self.up_conv_1 = double_conv(512, 256)

        self.up_trans_2 = nn.ConvTranspose3d(
                            in_channels = 256,
                            out_channels = 128,
                            kernel_size = 2,
                            stride = 2
                        )
        self.up_conv_2 = double_conv(256, 128)

        self.up_trans_3 = nn.ConvTranspose3d(
                            in_channels = 128,
                            out_channels = 64,
                            kernel_size = 2,
                            stride = 2
                        )
        self.up_conv_3 = double_conv(128, 64)

        self.out = nn.Conv3d(
                    in_channels = 64,
                    out_channels = 1,
                    kernel_size=1
                  )  

                    

    def forward(self, image):
        #bs, c, d, h, w
        #encoder
        #
        x1 = self.down_conv_1(image) #
        #
        x2 = self.max_pool(x1) 
        #
        x3 = self.down_conv_2(x2) #
        #
        x4 = self.max_pool(x3) 
        #
        x5 = self.down_conv_3(x4) #
        #
        x6 = self.max_pool(x5)
        #
        x7 = self.down_conv_4(x6) #

        #decoder
        #
        x = self.up_trans_1(x7)
        #
        y = self.up_conv_1(torch.cat([x5, x], 1))
        #
        x = self.up_trans_2(y)
        #
        y = self.up_conv_2(torch.cat([x3, x], 1))
        #
        x = self.up_trans_3(y)
        y = self.up_conv_3(torch.cat([x1, x], 1))
        #
        x = self.out(y)
        #
        return x


def double_conv(in_c, out_c):
    #define double convolution each followed by a ReLU activation
    conv = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=3, padding=(1, 1, 1)),
            nn.ReLU(inplace=True),
            #nn.Tanh(),
            nn.Conv3d(out_c, out_c, kernel_size=3, padding=(1, 1, 1)),
            #nn.Tanh(),
            nn.ReLU(inplace=True)
           )
    return conv

