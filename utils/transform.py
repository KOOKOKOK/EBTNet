
from albumentations.core.transforms_interface import ImageOnlyTransform 
import albumentations.augmentations.functional as F
import numpy as np

class NormalizeImgGradAngle(ImageOnlyTransform):
    """Normalization is applied by the formula: `img = (img - mean * max_pixel_value) / (std * max_pixel_value)`
       
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
    description: 
    param {*} self
    param {*} image[:,:,image,gradAngel]
    angle float
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
        # 
        if len(gradAngle.shape) != len(image.shape):
            gradAngle = np.expand_dims(gradAngle,axis=2)
        
        data = np.dstack((image,gradAngle))
        return data

    def get_transform_init_args_names(self):
        return ("mean", "std", "max_pixel_value")
    


