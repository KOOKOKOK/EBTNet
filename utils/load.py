

import torch
import torchvision.models as models
import torch.nn as nn
from tqdm import tqdm

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        # 定义两个卷积层和批归一化层，用ReLU函数作为激活函数
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        x = self.conv(x)
        return x
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # UNet的下采样部分
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature
        # UNet的上采样部分
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature*2, feature))
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        
    def forward(self, x):
        skip_connections = []
        
        # UNet的下采样部分
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        
        # UNet的上采样部分
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            if x.shape != skip_connection.shape:
                x = nn.functional.interpolate(x, size=skip_connection.shape[2:], mode='bilinear', align_corners=True)
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)#上采样
        
        out = self.final_conv(x)
        if self.training:
            return {'pred':out}
        else:
            return out



'''
description: 需要预先知道要冻结的属性名称
param {list} moduleList  
param {*} modelPath
param {*} model
return {*}
'''
def loadSelectedModule(moduleList:list,modelPath,model):
    if len(moduleList) == 0:
        print('没有加载任何权重')
        return
    print('加载了 {} 参数'.format(moduleList))
    pretrained_dict = torch.load(modelPath,map_location=torch.device('cpu'))['model']
    d = {}
    for name in tqdm(moduleList,desc=''):
        # name 是 state_dict 中的一个字段 key才是真正的 name一般为模型中的属性名称
        for key, value in pretrained_dict.items():
            if name in key:
                d[key] = pretrained_dict[key]

    model_dict = model.state_dict()
    model_dict.update(d)
    model.load_state_dict(model_dict)

if __name__ == '__main__':
    loadSelectedModule(['ups'],'wandb/run-20230620_084841-lhrmh2tn/files/300.pth',UNet(3,4))
