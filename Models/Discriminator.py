import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self,channels,features):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels,features,4,2,1),
            nn.LeakyReLU(0.2),
            self.basicBlock(features,features*2),
            self.basicBlock(features*2,features*4),
            self.basicBlock(features*4,features*8),
            self.basicBlock(features*8,1,4,2,0),
            nn.Sigmoid()
        )

    def basicBlock(self,inChannels,outChannels,kernelSize=4,stride=2,padding=1):
        return nn.Sequential(
            nn.Conv2d(in_channels=inChannels,out_channels=outChannels,kernel_size=kernelSize,stride=stride,padding=padding,bias=False),
            nn.BatchNorm2d(outChannels),
            nn.LeakyReLU(0.2),
        )

    def forward(self,X):
        return self.layers(X)
        