from batchgenerators.transforms.abstract_transforms import AbstractTransform
from monai.transforms import NormalizeIntensity
import numpy as np

class Min_Max_normalize(AbstractTransform):
    '''
    data_dict['output_key'] will be a list of segmentations scaled according to ds_scales
    '''
    def __init__(self, input_key="data", output_key="data"):
        self.output_key = output_key
        self.input_key = input_key

    def __call__(self, **data_dict):

        image_list = data_dict[self.input_key]

        output_list = []
        for t in range(len(image_list)):
            current_image = image_list[t]
            for b in range(len(current_image)):
                img_min = current_image[b].min().item()
                img_max = current_image[b].max().item()
                current_image[b] = NormalizeIntensity(subtrahend=img_min, divisor=(img_max - img_min), nonzero=True)(current_image[b])
            assert np.all(current_image <= 1.0) and np.all(current_image >= 0.0)
            output_list.append(current_image)

        data_dict[self.output_key] = output_list
        return data_dict