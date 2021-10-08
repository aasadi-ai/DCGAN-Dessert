import torch
import torch.optim as optim
import torch.nn as nn
from Models.Discriminator import Discriminator
from torch.utils.data import DataLoader
from tennisballDataset import TennisBallDataset

def trainingLoop(model,trainLoader,valLoader,optimizer,criterion,epochs=100):
    trainingLoss = []
    validationLoss = []
    
    for epoch in range(epochs):
        print(epoch)
        runningLoss = []
        for imgs,labels in trainLoader:
            optimizer.zero_grad()
            yHat = model(imgs)
            loss = criterion(yHat.squeeze(),labels)
            loss.backward()
            optimizer.step()
            runningLoss.append(loss.item())
        trainingLoss.append(sum(runningLoss)/len(runningLoss))
       
        if epoch%5==0:
            for valImgs,valLabels in valLoader:
                optimizer.zero_grad()
                yHat = model(valImgs).squeeze()
                valLoss = criterion(yHat,valLabels)
                validationLoss.append(valLoss.item())
                prediction = yHat >=0.5
                print("Accuracy:",10*torch.sum(torch.logical_and(prediction,valLabels))/len(valLabels),"%")

    return model,trainingLoss,validationLoss

def train(learningRate=1e-3,epochs=100):
    model = Discriminator(3,16)
    optimizer = optim.SGD(model.parameters(), lr=learningRate)
    criterion = nn.BCELoss()
    trainLoader = DataLoader(TennisBallDataset("Data/",(96,96),train=True),batch_size=24)
    valLoader = DataLoader(TennisBallDataset("Data/",(96,96),train=False),batch_size=100)
    return trainingLoop(model,trainLoader,valLoader,optimizer,criterion,epochs)
    


