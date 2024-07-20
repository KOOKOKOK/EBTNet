

import os


from utils.dataloader import ClassIdMapDataSet,ImgGradAngleDataSet
from utils.transform import NormalizeImgGradAngle
from albumentations.pytorch import ToTensorV2
import albumentations as A
from torch.utils.data import DataLoader
from model import *
from loss.loss import BiSeNetV2Loss,CrossEntropyLoss
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import torch
from tqdm import tqdm
import wandb
import torch.nn.functional as F
from torch.utils.data.dataset import random_split


def train():
    model.train()
    
    step = 0
    accuLoss = 0

    for data in tqdm(dataLoader,desc='train'):
        imgs,label = data[0].to(device),data[1].to(device).long()
        out = model(imgs)
        optimizer.zero_grad()
        loss = creterion(out,label)
        loss.backward()
        optimizer.step()
        step += 1

        accuLoss += loss.item()
        if step % 50 == 0:
            accuLoss /= 5
            wandb.log({'loss':accuLoss})
            accuLoss = 0

    
            

                
            
      

        
  
      


if __name__ == '__main__':

    epoch = 300
    batchSize = 4
    imgSize = (640, 640)
    loadList = []#填写模型属性名称
    frozeenList = []

    
    trainDir = r'/root/pan/data/heart6420/target/train'
    valDir = r'/root/pan/data/heart6420/target/val'

    color2label = {(0, 255, 0): 1, (0, 0, 255): 2, (255, 255,0): 3}
    label2color = {value: key for key, value in color2label.items()}



    transTrain = A.Compose([
        A.Resize(640, 640), 
        A.HorizontalFlip(),
        A.CoarseDropout(min_holes=8 ,max_holes=16,min_height=8,max_height=24,min_width=8,max_width=24),
        A.AdvancedBlur(),
        A.OpticalDistortion(),
        A.Rotate(),
        A.Normalize(

        mean=[0.2194, 0.2194, 0.2210],
        std=[0.2394, 0.2377, 0.2391],
    ),
    ToTensorV2()
    ])
    transVal = A.Compose([
    A.Resize(640, 640), 
    A.Normalize(

    mean=[0.2194, 0.2194, 0.2210],
    std=[0.2394, 0.2377, 0.2391],
    ),
    ToTensorV2()
    ])

# # 
    train_dataset = ClassIdMapDataSet(trainDir,transTrain,color2label)
    val_dataset = ClassIdMapDataSet(valDir,transVal,color2label)
   
    dataLoader = DataLoader(train_dataset,batchSize,True,num_workers=8,drop_last=True)
    
    dataLoaderVal = DataLoader(val_dataset,batchSize,True,num_workers=8,drop_last=True)
#   
    model = CUSNet(3,4)

    
    # creterion = CrossEntropyLoss()
    creterion = BiSeNetV2Loss()
    optimizer = Adam(model.parameters(),0.001)
    scheduler = StepLR(optimizer,50,0.5)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    model.to(device=device)


    for i in range(epoch):
        train()
        scheduler.step()
        if (i+1) % 50 == 0:
            
            torch.save({'model':model.state_dict()}, '/%d.pth'%(i+1))
