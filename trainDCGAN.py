import torch
from torch._C import device
import torch.optim as optim
import torch.nn as nn
from Models.Discriminator import Discriminator
from Models.Generator import Generator
from torch.utils.data import DataLoader
from tennisballDataset import TennisBallDataset

#Init constants
BATCH_SIZE = 128
IMAGE_DIM = 64
CHANNELS = 3
Z_NOISE_DIM = 100
FEATURES = 64
LEARNING_RATE = 2e-4
EPOCHS = 11
OPTIM_BETAS = (0.5,0.999)

#Use GPU if found
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Initialize Models
generator = Generator(Z_NOISE_DIM,CHANNELS,FEATURES).to(device)
discriminator = Discriminator(CHANNELS,FEATURES).to(device)
generator.train()
discriminator.train()

#Intialize Optimizers
optimizerGen = optim.Adam(generator.parameters(),lr=LEARNING_RATE,betas=OPTIM_BETAS)
optimizerDisc = optim.Adam(discriminator.parameters(),lr=LEARNING_RATE,betas=OPTIM_BETAS)
criterion = nn.BCELoss()

#Init DataLoader
dataloader = 2

for epoch in range(EPOCHS):
    for batchIdx,(realImages,labels) in enumerate(dataloader):
        realImages = realImages.to(device)
        zNoise = torch.randn(BATCH_SIZE,Z_NOISE_DIM,1,1).to(device)
        fakeImages = generator(zNoise)

        ##Discriminator Training Step
        #compute loss on real images
        yHatReal = discriminator(realImages)
        lossDiscReal = criterion(yHatReal,torch.ones_like(yHatReal))
        #compute loss on fake images
        yHatFake = discriminator(fakeImages.detach())
        lossDiscFake = criterion(yHatFake,torch.zeros_like(yHatFake))
        #compute total loss and back prop
        totalLoss = (lossDiscReal+lossDiscFake)/2
        discriminator.zero_grad()
        totalLoss.backward()
        optimizerDisc.step()

        #Generator Training Step
        yHatDisc = discriminator(fakeImages)
        lossGen = criterion(yHatDisc,torch.ones_like(yHatDisc))
        generator.zero_grad()
        lossGen.backward()
        optimizerGen.step()
