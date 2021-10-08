import torch
from torch.utils.data import Dataset,DataLoader
from torchvision.transforms import Normalize
import os
import cv2

class TennisBallDataset(Dataset):
    def __init__(self,root_dir,imgDim,train=True):
        self.imgs = []
        self.imgDim = imgDim

        if train==True:
            root_dir = os.path.join(root_dir,"trainData")
        else:
            root_dir = os.path.join(root_dir,"validationData")

        ballPath = os.path.join(root_dir,"balls")
        for imgPath in os.listdir(ballPath):
            self.imgs.append((os.path.join(ballPath,imgPath),1))

        emptyPath = os.path.join(root_dir,"empty")
        for imgPath in os.listdir(emptyPath):
            self.imgs.append((os.path.join(emptyPath,imgPath),0))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = cv2.imread(self.imgs[idx][0])
        img = cv2.resize(img,self.imgDim)
        img = torch.from_numpy(img)
        img = img.permute(2,0,1).float()
        normTransform = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        img = normTransform(img)
        label = torch.tensor(self.imgs[idx][1]).float()
        return img,label

