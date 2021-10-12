import torch
from torch._C import device
import torch.optim as optim
import torch.nn as nn
from Models.Discriminator import Discriminator
from Models.Generator import Generator
from torch.utils.data import DataLoader, dataloader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision
from torch.utils.tensorboard import SummaryWriter
from Models.modelSetup import initializeWeights

#Init constants
BATCH_SIZE = 128
IMAGE_DIM = 64
CHANNELS = 1
Z_NOISE_DIM = 100
FEATURES = 64
LEARNING_RATE = 2e-4
EPOCHS = 11
OPTIM_BETAS = (0.5,0.999)

#Use GPU if found
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Initialize Models and Fixed Noise Vector
FIXED_NOISE = torch.randn(10,Z_NOISE_DIM,1,1).to(device)
generator = Generator(Z_NOISE_DIM,CHANNELS,FEATURES).to(device)
discriminator = Discriminator(CHANNELS,FEATURES).to(device)
initializeWeights(generator,0.0,0.02)
initializeWeights(discriminator,0.0,0.02)
generator.train()
discriminator.train()

#Intialize Optimizers
optimizerGen = optim.Adam(generator.parameters(),lr=LEARNING_RATE,betas=OPTIM_BETAS)
optimizerDisc = optim.Adam(discriminator.parameters(),lr=LEARNING_RATE,betas=OPTIM_BETAS)
criterion = nn.BCELoss()

#Define transforms on train images
transforms = transforms.Compose(
    [
        transforms.Resize(IMAGE_DIM),
        transforms.ToTensor(),
        transforms.Normalize([0.5 for i in range(CHANNELS)],[0.5 for i in range(CHANNELS)]),
    ]
)

#Init dataset and dataloader (TEST)
dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms,download=True)
dataloader = DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=True)

#Init Summary Writer
writer = SummaryWriter()

#Batch count 
count = 0

for epoch in range(EPOCHS):
    print(f"Epoch:{epoch}")
    for batchIdx,(realImages,labels) in enumerate(dataloader):
        print(f"Batch:{batchIdx}")
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

        #Write fake images to writer every 10 batches
        if batchIdx%10==0:
            count+=1
            with torch.no_grad():
                fakeImages = generator(FIXED_NOISE)
                imgGrid = torchvision.utils.make_grid(fakeImages[:10],normalize=True)
                writer.add_image("Fakes:",imgGrid,global_step=count)

            #Print current Loss
            print(f"Loss Generator:{lossGen.item()} Loss Discriminator:{totalLoss.item()}")

writer.close()