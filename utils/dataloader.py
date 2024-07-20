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

if __name__ == '__main__':
    from albumentations.pytorch import ToTensorV2
    import matplotlib.pyplot as plt
    import numpy as np

    trans = A.Compose([
        A.Resize(256, 256), 
        A.HorizontalFlip(),
        NormalizeImgGradAngle(
        mean=[0.2194, 0.2194, 0.2210],
        std=[0.2394, 0.2377, 0.2391],
    ),
    ToTensorV2()
    ])
    # {(255, 0, 0): 'RA', (0, 255, 0): 'RV', (0, 0, 255): 'LA', (255, 255, 0): 'LV', (255, 0, 255): 'DAO'}, 'processor': 'cv2'}
    dataset = ImgGradAngleDataSet('/root/pan/data/heart6420/target/train',trans,\
                     {(255, 0, 0): 1, (0, 255, 0): 2, (0, 0, 255): 3, (255, 255, 0): 4, (255, 0, 255): 5})
    img,label = dataset[0]
    color2label = {(255, 0, 0): 1, (0, 255, 0): 2, (0, 0, 255): 3, (255, 255, 0): 4, (255, 0, 255): 5}
    label2color = {value: key for key, value in color2label.items()}

    mask = np.zeros_like(img,dtype=np.uint8)
    for key in label2color.keys():
            idx = np.where((label==key).all())
            # print(np.nonzero(label==key))
            mask[label==key] = label2color[key]
    alpha = 0.5  # 标签的透明度
    overlay = mask  # 创建一个与图像大小相同的三通道标签
    output = cv2.addWeighted(img, 1-alpha, overlay, alpha, 0)

    img1 = cv2.cvtColor(label*50 ,cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

    # 创建一个包含两个子图的图像
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # 在第一个子图中显示第一张图像
    axs[0].imshow(img1)
    axs[0].set_title('Image 1')

    # 在第二个子图中显示第二张图像
    axs[1].imshow(img2)
    axs[1].set_title('Image 2')

    # 显示图像
    plt.show()

