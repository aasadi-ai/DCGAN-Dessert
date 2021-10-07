import torch.optim as optim
import torch.nn as nn
from Discriminator import Discriminator

def trainingLoop(model,trainLoader,valLoader,optimizer,criterion,epochs=100):
    trainingLoss = []
    validationLoss = []
    
    for epoch in range(epochs):
        for imgs,labels in trainLoader:
            optimizer.zero_grad()
            yHat = model(imgs)
            loss = criterion(yHat,labels)
            loss.backward()
            optimizer.step()
            if epoch%5==0:
                trainingLoss.append(loss.item())
                for valImgs,valLabels in valLoader:
                    optimizer.zero_grad()
                    yHat = model(valImgs)
                    valLoss = criterion(yHat,valLabels)
                    validationLoss.append(valLoss.item())
    return model,trainingLoss,validationLoss

def train(learningRate=1e-2,epochs=100):
    model = Discriminator(3,16)
    optimizer = optim.SGD(model.parameters(), lr=learningRate)
    criterion = nn.CrossEntropyLoss()
    trainLoader = 1
    valLoader = 1
    return trainingLoop(model,trainLoader,valLoader,optimizer,criterion,epochs)
    


