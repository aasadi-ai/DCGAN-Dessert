import torch.nn as nn

def initializeWeights(model,Mean,Std):
    for nnModule in model.modules():
        if isinstance(nnModule,(nn.Conv2d,nn.ConvTranspose2d,nn.BatchNorm2d)):
            nn.init.normal_(nnModule.weight.data,Mean,Std)