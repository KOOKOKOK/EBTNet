

from tqdm import tqdm

'''
description: 需要预先知道要冻结的属性名称
param {list} moduleList
param {*} model
return {*}
'''
def frozenModules(moduleList:list,model):
    if len(moduleList) == 0:
        print('没有需要冻结的层')
        return
    for name in tqdm(moduleList,desc=''):
        if hasattr(model, name):
            m = getattr(model,name)
            for param in m.parameters():
                param.requires_grad = False
            print('冻结了 {} 模块'.format(name))
        else:
            print("attribute 属性不存在")
       