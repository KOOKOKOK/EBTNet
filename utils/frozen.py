

from tqdm import tqdm

'''
description: 
param {list} moduleList
param {*} model
return {*}
'''
def frozenModules(moduleList:list,model):
    if len(moduleList) == 0:
        
        return
    for name in tqdm(moduleList,desc=''):
        if hasattr(model, name):
            m = getattr(model,name)
            for param in m.parameters():
                param.requires_grad = False
            
        else:
            print("attribute none")
       
