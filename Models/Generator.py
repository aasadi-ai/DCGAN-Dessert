import torch.nn as nn

class Generator(nn.Module):
    def __init__(self,zDim,channels,features):
        super().__init__()
        self.layers = nn.Sequential(
            self.basicBlock(zDim,features*16,4,1,0),
            self.basicBlock(features*16,features*8,4,2,1),
            self.basicBlock(features*8,features*4,4,2,1),
            self.basicBlock(features*4,features*2,4,2,1),
            nn.ConvTranspose2d(features*2,channels,4,2,1),
            nn.Tanh()
        )
    
     def basicBlock(self,inChannels,outChannels,kernelSize,stride=1,padding=1):
        return nn.Sequential(
            nn.ConvTranspose2d(inChannels,outChannels,kernelSize,stride,padding,bias=False),
            nn.BatchNorm2d(outChannels),
            nn.ReLU()
        )