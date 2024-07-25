import os
from torch.utils.data import Dataset
import cv2
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from .transform import NormalizeImgGradAngle
class ClassIdMapDataSet(Dataset):
    def __init__(self,dataDir:str,transform:A.Compose,color2label:dict) -> None:
        super().__init__()
        self.imagesPath = os.path.join(dataDir,'images')
        self.imageDir = os.listdir(self.imagesPath)
        self.labelsPath = os.path.join(dataDir,'labels')
        self.labelDir = os.listdir(self.labelsPath)        
        self.transform = transform
        self.color2label = color2label
    def __getitem__(self, index):
        imagePath = os.path.join(self.imagesPath,self.imageDir[index])
        labelPath = os.path.join(self.labelsPath,self.imageDir[index])
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        maskG = cv2.imread(labelPath)
        mask = np.zeros((image.shape[0],image.shape[1]),dtype=np.uint8)
        for key in self.color2label.keys():
            idx = np.where((maskG==key).all(axis=2))
            mask[idx[0],idx[1]] = self.color2label[key]
        transformed  = self.transform(image=image, mask=mask)
        transformed_image = transformed['image']
        transformed_mask = transformed['mask']
        return transformed_image.to(torch.float32),transformed_mask.to(torch.float32)

    def __len__(self):
        return len(self.imageDir)


class ImgGradAngleDataSet(Dataset):
    def __init__(self,dataDir:str,transform:A.Compose,color2label:dict) -> None:
        super().__init__()
        self.imagesPath = os.path.join(dataDir,'images')
        self.imageDir = os.listdir(self.imagesPath)
        self.labelsPath = os.path.join(dataDir,'labels')
        self.labelDir = os.listdir(self.labelsPath)        
        self.transform = transform
        self.color2label = color2label
    def __getitem__(self, index):
        imagePath = os.path.join(self.imagesPath,self.imageDir[index])
        labelPath = os.path.join(self.labelsPath,self.labelDir[index])
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
        sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
        angle = np.arctan2(sobel_y, sobel_x) * 180 / np.pi

        maskG = cv2.imread(labelPath)
        mask = np.zeros((image.shape[0],image.shape[1]),dtype=np.uint8)
        for key in self.color2label.keys():
            idx = np.where((maskG==key).all(axis=2))
            mask[idx[0],idx[1]] = self.color2label[key]
        
        image = np.dstack((image,angle))
        transformed  = self.transform(image=image, mask=mask)
        transformed_image = transformed['image']
        transformed_mask = transformed['mask']
        return transformed_image.to(torch.float32),transformed_mask.to(torch.float32)

    def __len__(self):
        return len(self.imageDir)



