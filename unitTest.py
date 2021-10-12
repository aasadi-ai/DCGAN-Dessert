import torch
from Models.Generator import Generator
from Models.Discriminator import Discriminator

BATCH_SIZE = 10
CHANNELS = 3
IMAGE_X,IMAGE_Y = 64,64
Z_SIZE = 100
FEATURES = 8

#Random Images and Random Noise Vector
sampleImageBatch = torch.rand((BATCH_SIZE,CHANNELS,IMAGE_Y,IMAGE_X))
sampleNoiseVector = torch.rand((BATCH_SIZE,Z_SIZE,1,1))

#Initialize Discriminator and Generator
discriminator = Discriminator(CHANNELS,FEATURES)
generator = Generator(Z_SIZE,CHANNELS,FEATURES)

#Shape Assertions
expectedShapeDiscriminator = (BATCH_SIZE,1,1,1)
assert(discriminator(sampleImageBatch).shape==expectedShapeDiscriminator)

expectedShapeGenerator = (BATCH_SIZE,CHANNELS,IMAGE_Y,IMAGE_X)
assert(generator(sampleNoiseVector).shape==expectedShapeGenerator)

#Tests Passed
print("We're golden!")