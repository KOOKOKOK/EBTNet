
import torch
import torch.nn as nn



class LossWrapper(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    

    '''
    description: 
    param {*} self
    param {*} pred 推荐使用键值对
    param {*} labels 推荐使用键值对
    return {*}
    '''
    def forward(self,pred,labels):
        raise NotImplementedError('该方法必须实现')

class CrossEntropyLoss(LossWrapper):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
        self.creterion = nn.CrossEntropyLoss()
    def forward(self,pred,labels):
        loss = self.creterion(pred['pred'],labels)
        return loss

            
if __name__ == '__main__':
    # creterion = nn.CrossEntropyLoss()
    # res = torch.randn(2,6,640,640)
    # label = torch.randint(0,6,(2,640,640))
    # loss = creterion(res,label)
    # print(loss)
    res = torch.randn(2,6,640,640)
    boost = {0:res.clone(),1:res.clone()}
    label = torch.randint(0,6,(2,640,640))
    creterion = BiSeNetV2Loss()
    pre = {'pred':res,'boost':boost}
    loss = creterion(pre,label)
    print(loss)