import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self,channels,features):
        super().__init__()
        self.layers = nn.Sequential(
            self.basicBlock(channels,features),
            self.basicBlock(features,features*2),
            self.basicBlock(features*2,features*4),
            nn.Flatten(),
            nn.Linear(1234,32),
            nn.Linear(32,16),
            nn.Linear(16,1),
            nn.Sigmoid()
        )

    def basicBlock(inChannels,outChannels,kernelSize=3,padding=1):
        return nn.Sequential(
            nn.Conv2d(in_channels=inChannels,out_channels=outChannels,kernel_size=kernelSize,padding=padding),
            nn.ReLU(),
            nn.BatchNorm2d(outChannels),
            nn.MaxPool2d(2)
        )

    def forward(self,X):
        return self.layers(X)
        