
import torch
import wandb
import torch.nn.functional as F
import numpy as np
import cv2
from torch.utils.data.dataset import random_split
from typing import List
import torch.nn as nn
from torch.utils.data import Dataset


        

'''
description: 按照固定随机种子划分数据集
param {Dataset} dataset
param {List} ratios
return {*}
'''
def randomSliptData(dataset:Dataset,ratios:List[int]):
        generator = torch.Generator().manual_seed(42)
        train_dataset, val_dataset = random_split(dataset, ratios,generator)
        return train_dataset, val_dataset

'''
description: 初始化权重未实现！！！
param {torch} model
return {*}
'''
def initModelWeights(model:torch.nn.Module):
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(0)#让显卡产生的随机数一致
    torch.cuda.manual_seed_all(0)
    with torch.no_grad():
        for param in model.parameters():
            if len(param.data.size()) > 1:
                nn.init.xavier_normal_(param.data)
            else:
                nn.init.zeros_(param.data)

    return model

