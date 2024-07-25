

import torch
import torch.nn as nn
from typing import Any, List, Optional
from torch.nn import functional as F
from torchvision.ops.misc import Conv2dNormActivation
from torch import Tensor

class GSPB(nn.Module):
    def __init__(self, inChannels,outChannels,dilations=(1, 6, 12, 18)) -> None:
        super().__init__()

        self.down = inChannels!=outChannels
        l = []
        self.patch = nn.Conv2d(inChannels,outChannels,3,stride=2,padding=1)if inChannels!= outChannels\
        else nn.Identity()

        for (i,d) in enumerate(dilations):
            l.append(
                nn.Sequential(
                 nn.Conv2d(
                    outChannels,
                    outChannels,
                    (1,3),
                    dilation=d,
                    padding= (0,d)),
                nn.Conv2d(
                    outChannels,
                    outChannels,
                    (3,1),
                    dilation=d,
                    padding=(d,0)),
                nn.BatchNorm2d(outChannels) if (i+1)%2==0 else nn.GELU()  
                )
                
            )
        self.conv = nn.Sequential(*l)
    def forward(self,x):
        x = self.patch(x)
        out = self.conv(x)
        if not self.down:
            out = out + x
        return out

class DetailBranchBlock(nn.Module):
    '''
    description:  
    param {*} self
    param {*} inChannels 
    param {*} outChannels
    param {*} repeat 
    return {*}
    '''
    def __init__(self,inChannels,outChannels,repeat) -> None:
        super().__init__()
        self.block = nn.Sequential(
                    nn.Conv2d(in_channels=inChannels,out_channels=outChannels,kernel_size=3,stride=2,padding=1),
                    nn.BatchNorm2d(outChannels),
                    nn.GELU(),
                    *[GSPB(outChannels,outChannels) for i in range(repeat)]
        )
       
    def forward(self,x):
        out = self.block(x)
        
        return out 


class StemBlock(nn.Module):

    '''
    description:  
    param {*} self
    param {*} inChannels 
    param {*} outChannels 
    return {*}
    '''
    def __init__(self,inChannels,outChannels) -> None:
        super().__init__()

        self.convFirst = nn.Sequential(nn.Conv2d(in_channels=inChannels,out_channels=outChannels,kernel_size=3,stride=2,padding=1),\
                                       nn.BatchNorm2d(outChannels),\
                                       nn.GELU(),\
                                       ) 
        self.leftBranch = nn.Sequential(
            nn.Conv2d(in_channels=outChannels,out_channels=outChannels // 2,kernel_size=1),\
            nn.BatchNorm2d(outChannels//2),\
            nn.GELU(),\
            nn.Conv2d(in_channels=outChannels//2,out_channels=outChannels,kernel_size=3,stride=2,padding=1),\
            nn.BatchNorm2d(outChannels),\
            nn.GELU(),\
        )

        # self.rightBranch = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.rightBranch = nn.Conv2d(outChannels,outChannels,kernel_size=3,stride=2,padding=1)

        self.lastConv = nn.Sequential(
            nn.Conv2d(in_channels=outChannels*2,out_channels=outChannels,kernel_size=3,padding=1),\
            nn.BatchNorm2d(outChannels),\
            nn.GELU(),\
        )

    def forward(self,x):
        x = self.convFirst(x)
        l = self.leftBranch(x)
        r = self.rightBranch(x)
        catX = torch.concat([l,r],dim=1)
        out = self.lastConv(catX)

        return out



class EDSConv(nn.Module):
    expansion = 4
    def __init__(self, inChannels,outChannel,groups=1) -> None:
        super().__init__()
        groups = inChannels
        self.downSample = True if outChannel>inChannels  else False
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=inChannels,out_channels=inChannels
                                             ,stride=2 if self.downSample else 1,
                               kernel_size=7,padding=3,groups=groups),
                               nn.BatchNorm2d(inChannels))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=inChannels,out_channels=self.expansion*inChannels,kernel_size=1),
            nn.GELU())
        self.conv3 = nn.Conv2d(in_channels=self.expansion*inChannels,out_channels=outChannel,kernel_size=1)

    def forward(self,x):
        fea0 = self.conv1(x)
        fea1 = self.conv2(fea0)
        fea2 = self.conv3(fea1)

        if self.downSample:
            return fea2
        else:
            out = x + fea2
        return out
    
class DepthwiseConv2d(nn.Module):
    '''
    description: 
    param {*} self
    param {*} in_channels 
    param {*} out_channels 
    param {*} kernel_size
    param {*} stride
    param {*} padding
    return {*}
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(DepthwiseConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
class LHPB(nn.Module):
    '''
    description:
    param {*} self
    param {*} inChannel 
    param {*} outChannel
    param {*} expansionFactor 
    return {*}
    '''
    def __init__(self,inChannel,outChannel,expansionFactor) -> None:
        super().__init__()

        self.downOp = inChannel < outChannel
        hidenChannel = outChannel 
        if self.downOp:
            self.conv = nn.Sequential(
                Conv2dNormActivation(in_channels=inChannel,out_channels=outChannel,kernel_size=3,stride=2,padding=1,norm_layer=nn.BatchNorm2d,activation_layer=nn.GELU,inplace=None),\
                EDSConv(inChannels=outChannel,outChannel=hidenChannel),\
                # DepthwiseConv2d(in_channels=outChannel,out_channels=hidenChannel,kernel_size=3,stride=2,padding=1),
                nn.BatchNorm2d(hidenChannel),\
                EDSConv(inChannels=outChannel,outChannel=hidenChannel),\
                # DepthwiseConv2d(in_channels=hidenChannel,out_channels=hidenChannel,kernel_size=3,padding=1),
                nn.BatchNorm2d(hidenChannel),\
                nn.Conv2d(in_channels=hidenChannel,out_channels=outChannel,kernel_size=1,stride=1),\
                nn.BatchNorm2d(outChannel),\
            )
            self.branch = nn.Sequential(
                EDSConv(inChannels=inChannel,outChannel=hidenChannel),\
                # DepthwiseConv2d(in_channels=inChannel,out_channels=outChannel,kernel_size=3,stride=2,padding=1),
                nn.BatchNorm2d(outChannel),\
                nn.Conv2d(in_channels=outChannel,out_channels=outChannel,kernel_size=1,stride=1),\
                nn.BatchNorm2d(outChannel),\
            )
        else:
            self.conv = nn.Sequential(
                Conv2dNormActivation(in_channels=inChannel,out_channels=outChannel,kernel_size=3,stride=1,padding=1,norm_layer=nn.BatchNorm2d,activation_layer=nn.GELU,inplace=None),\
                EDSConv(inChannels=outChannel,outChannel=hidenChannel),\
                # DepthwiseConv2d(in_channels=outChannel,out_channels=hidenChannel,kernel_size=3,padding=1),
                nn.BatchNorm2d(hidenChannel),\
                nn.Conv2d(in_channels=hidenChannel,out_channels=outChannel,kernel_size=1,stride=1),\
                nn.BatchNorm2d(outChannel),\
            )
        self.lastAct = nn.GELU()
        
    def forward(self,x):
        if self.downOp:
            fea0 = self.conv(x)
            branchX = self.branch(x)
            fea1 = fea0 + branchX
        else:
            fea0 = self.conv(x)
            fea1 = fea0 + x

        out = self.lastAct(fea1)
        return out

class SemanticBranch(nn.Module):
    '''
    description: 
    param {*} self
    param {*} channels
    param {*} repeat
    param {*} expandRatio
    return {*}
    '''
    def __init__(self, inChannel,outChannel,channels=[16,32,32,64,64,128,128],repeat=[1,1,1,1,1,3],expandRatio=[6,6,6,6,6,6],headApply=[True,True,True,True]) -> None:
        super().__init__()

        self.branch = nn.ModuleList()
        self.headApply = headApply
        
        l = []
        inChannels = inChannel
        i = 0
        self.branch.append(StemBlock(inChannels=inChannels,outChannels=channels[0]))
        
        inChannels = channels[0]
        channels = channels[1:]
# 
        for c,r,e in zip(channels,repeat,expandRatio):
            if r > 1:
                for j in range(r):
                    l.append(LHPB(inChannel=inChannels,outChannel=c,expansionFactor=e))
                    inChannels = c
                i += 1
            else:
                l.append(LHPB(inChannel=inChannels,outChannel=c,expansionFactor=e))
                inChannels = c
                i += 1
            if i % 2 == 0:
                self.branch.append(nn.Sequential(*l))
                l.clear()
        
        self.branch.append(LHPB(channels[-1],channels[-1],0))

    def forward(self,x):
        fea = []
        fea.append(self.branch[0](x))
        # 
        for item in self.branch[1:-1]:
            fea.append(item(fea[-1]))#
        
        semantic = fea[-1]
        out = {}
        for i,t in enumerate(self.headApply):#
            if t:
                out[i] = fea[i]
                
        # out[len(out.keys())] = semantic#out
        out[-1] = semantic#out

        return out
class ChannelAttentionModule(nn.Module):
    def __init__(self, in_channels, reduction_ratio=4):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc1 = nn.Conv2d(in_channels, in_channels * reduction_ratio, 1)
        self.ReLU1 = nn.GELU()
        self.fc2 = nn.Conv2d(in_channels * reduction_ratio, in_channels, 1)
        
    def forward(self, x):
        # avg_pool = self.avg_pool(x)
        max_pool = self.max_pool(x)
        
        # avg_pool = self.fc2(self.ReLU1(self.fc1(avg_pool)))
        max_pool = self.fc2(self.ReLU1(self.fc1(max_pool)))
        
        # channel_attention = torch.sigmoid(avg_pool + max_pool)
        channel_attention = torch.sigmoid(max_pool)
        return channel_attention

class SpatialAttentionModule(nn.Module):
    def __init__(self, kernel_size=7,embbed_size=20):
        super(SpatialAttentionModule, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(1, 1, kernel_size, padding=padding)
        self.embbeding = nn.Parameter(torch.randn((1,1,embbed_size,embbed_size)))
    def forward(self, x):
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        
        combined = max_pool + self.embbeding
        spatial_attention = torch.sigmoid(self.conv(combined))

        return spatial_attention

class Mixer(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7,embbed_size=20):
        super(Mixer, self).__init__()
        self.channel_attention = ChannelAttentionModule(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttentionModule(kernel_size,embbed_size=embbed_size)
        
    def forward(self, x):
        channel_attention = self.channel_attention(x)
        x = x * channel_attention
        spatial_attention = self.spatial_attention(x)
        x = x * spatial_attention
        return x
class MixFormer(nn.Module):
    '''
    description: 
    param {*} self
    param {*} inChannel
    param {*} outChannel
    return {*}
    ''' 
    def __init__(self, inChannel,outChannel) -> None:
        super().__init__()

        self.detailProcess = Mixer(128,embbed_size=80)
        self.semanticProcess = Mixer(128,embbed_size=20)
        self.toSemantic = nn.AvgPool2d(kernel_size=4,stride=4)
        self.toDetail = nn.Sequential(
            nn.Upsample(scale_factor=4,mode='bilinear'))

        self.up4Ratio = nn.Upsample(scale_factor=4,mode='bilinear')
        self.lastConv = Conv2dNormActivation(in_channels=outChannel,out_channels=outChannel,kernel_size=3,stride=1,padding=1,norm_layer=nn.BatchNorm2d,activation_layer=None)
    def forward(self,d:torch.Tensor,s:torch.Tensor):
        assert d.size()[-1] // s.size()[-1] == 4,'4倍'
        detail_att = self.detailProcess(d)
        semantic_att = self.semanticProcess(s)
#mixer
        left = d + self.toDetail(semantic_att)
        right = s  + self.toSemantic(detail_att)
        right = self.up4Ratio(right)#

        fea0 = left + right

        out = self.lastConv(fea0)

        return out


class SegHead(nn.Module):
    '''
    description: 
    param {*} self
    param {*} inChannel
    param {*} outChannel
    param {*} scale 
    param {int} midChannel
    return {*}
    '''
    def __init__(self, inChannel,outChannel,scale,midChannel:int=0) -> None:
        super().__init__()

        if midChannel!=0:
            scale = scale // 2

        self.conv = nn.Sequential( nn.Upsample(scale_factor=2) if midChannel!=0 else nn.Identity(),\
                                Conv2dNormActivation(in_channels=inChannel,out_channels=midChannel if midChannel!=0 else inChannel,\
                                         kernel_size=3,stride=1,padding=1,norm_layer=nn.BatchNorm2d,\
                                            activation_layer=nn.GELU,inplace=None),\
                                nn.Conv2d(in_channels=midChannel if midChannel!=0 else inChannel,\
                                          out_channels=outChannel,kernel_size=1),\
                                nn.Upsample(scale_factor=scale,mode='bilinear')\
        )

    def forward(self,x):
        out = self.conv(x)
        return out



class BoostBranch(nn.Module):
    '''
    description: 
    param {*} self
    param {*} inChannel 16
    param {*} outChannel 128
    param {*} minScale 
    param {*} classes 
    param {*} midChannel 
    param {*} headApply 
    return {*}
    '''
    def __init__(self,inChannel,outChannel,minScale,classes,midChannel=[128,128,128,128],headApply=[True,True,True,True]) -> None:
        super().__init__()
        
        # 
        
        cAndS = []#len = 4(channel,scale)
        cAndS.append((inChannel,minScale))#minScale=4 
        while cAndS[-1][0] != outChannel:
            cAndS.append((cAndS[-1][0]*2,cAndS[-1][1]*2))
        
        assert len(cAndS) == len(headApply) and len(midChannel) == len(headApply),'headApply'
        self.branch = nn.ModuleDict()

# 
        for i,(c,s) in enumerate(cAndS):
            self.branch[str(i)] = SegHead(inChannel=c,outChannel=classes,scale=s,midChannel=midChannel[i]) if headApply[i] else nn.Identity()
    

    def forward(self,x):
        out = []
        for k,block in self.branch.items():
            k = int(k)
            if type(block) != nn.Identity and k in x.keys():
                out.append(block(x[k]))
        return out
    

class EBTNet(nn.Module):
    '''
    description: 
    param {*} self
    param {*} inChannel 
    param {*} outChannel 
    param {*} classes 
    param {*} stemChannel stem
    param {*} headApply 
    return {*} 
    '''
    def __init__(self, inChannel,classes,outChannel=128,stemChannel=16,headApply=[True,True,True,True],
                 detailBranchChannels=[64,64,128],semanticBranchChannels=[16,32,32,64,64,128,128]) -> None:
        super().__init__()

        # outChannel *= 2
        # stemChannel *= 2
        # detailBranchChannels = [item * 2 for item in detailBranchChannels]
        # semanticBranchChannels = [item * 2 for item in semanticBranchChannels]


        self.inChannel = inChannel
        self.outChannel = outChannel


        self.detailBranch = self._loadDetailBranch(channels=detailBranchChannels)
        self.semanticBranch = self._loadSemanticBranch(headApply=headApply,channels=semanticBranchChannels,
                                                       repeat=[1,3,1,3,1,5])#stem
        self.MixFormer = MixFormer(inChannel=outChannel,outChannel=outChannel)
        self.segHead = SegHead(inChannel=outChannel,outChannel=classes,scale=8)
        self.boostBranch = BoostBranch(inChannel=stemChannel,outChannel=outChannel,minScale=4,classes=classes)
        # print('done')
    '''
    description: 
    param {*} self
    param {*} channels
    param {*} repeat 
    return {*}
    '''
    def _loadDetailBranch(self,channels=[64,64,128],repeat=[1,2,2]):
        l = []
        inChannel = self.inChannel
        for c,r in zip(channels,repeat):
            l.append(DetailBranchBlock(inChannels=inChannel,outChannels=c,repeat=r))
            inChannel = c
        return nn.Sequential(*l)

    '''
    description: 
    param {*} self _loadSemanticBranch
    param {*} channels 
    param {*} repeat 
    param {*} expandRatio 
    return {*} 
    '''
    def _loadSemanticBranch(self,channels=[16,32,32,64,64,128,128],repeat=[1,1,1,1,1,3],expandRatio=[6,6,6,6,6,6],headApply=[False,False,False,False]):
        m = SemanticBranch(inChannel=self.inChannel,outChannel=self.outChannel,\
                           channels=channels,repeat=repeat,\
                            expandRatio=[6,6,6,6,6,6],headApply=headApply)
        
        return m
    
    def forward(self,x):
        detail = self.detailBranch(x)#tensor
        semantic = self.semanticBranch(x)#dict[int:tensor]
        
        # fea0 = self.MixFormer(detail,semantic[len(semantic.keys())-1])#dict[int:tensor]
        fea0 = self.MixFormer(detail,semantic[-1])#dict[int:tensor]
        out = self.segHead(fea0)
        # out = F.interpolate(out, 640,)

        if self.training:
            boostOut = self.boostBranch(semantic)
            return {'pred':out,'boost':boostOut}
        else:
            return out#
def CUSNet(inChannels,outChannels,headApply=[True,True,True,True]):
    model = EBTNet(inChannel=inChannels,classes=outChannels,headApply=headApply)
    # model.boostBranch = nn.Identity()
    return model
if __name__ == '__main__':

    model = EBTNet(inChannel=3,outChannel=128,classes=5)
    # model.eval()
    
    t = torch.randn((4,3,640,640))
    fea0 = model(t)
    print(fea0['pred'].size())
