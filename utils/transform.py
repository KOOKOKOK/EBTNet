
from albumentations.core.transforms_interface import ImageOnlyTransform 
import albumentations.augmentations.functional as F
import numpy as np

class NormalizeImgGradAngle(ImageOnlyTransform):
    """Normalization is applied by the formula: `img = (img - mean * max_pixel_value) / (std * max_pixel_value)`
        照着官方的类修改的使其能够对一个4维的输入包含图片梯度方向的矩阵进行归一化
        使用梯度方向是因为如果使用梯度值不好归一化,不知道最大值和最小值,而反正切有确定的域
    Args:
        mean (float, list of float): mean values
        std  (float, list of float): std values
        max_pixel_value (float): maximum possible pixel value

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(
        self,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        max_pixel_value=255.0,
        always_apply=False,
        p=1.0,
    ):
        super(NormalizeImgGradAngle, self).__init__(always_apply, p)
        self.mean = mean
        self.std = std
        self.max_pixel_value = max_pixel_value

    '''
    description: 将先在通道上分离然后再分别归一化,最后在通道上叠加在一起
    param {*} self
    param {*} image[:,:,image,gradAngel],一般image都是uint8,为了迎合
    angle就统一为float
    param {object} params
    return {*}
    '''
    def apply(self, image, **params):
        image = image.astype(np.float32)
        min_value = -np.pi/2
        max_value = np.pi/2

        image,gradAngle = image[:,:,0:3],image[:,:,3:]
        image = F.normalize(image, self.mean, self.std, self.max_pixel_value)
        gradAngle = (gradAngle - min_value) / (max_value - min_value)
        # 对齐维度
        if len(gradAngle.shape) != len(image.shape):
            gradAngle = np.expand_dims(gradAngle,axis=2)
        
        data = np.dstack((image,gradAngle))
        return data

    def get_transform_init_args_names(self):
        return ("mean", "std", "max_pixel_value")
    

if __name__ == '__main__':
    trans = NormalizeImgGradAngle()
    t = np.random.randn(640,640,4)
    out = trans(image=t)['image']
    print(out.shape)
