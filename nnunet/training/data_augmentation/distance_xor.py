from batchgenerators.transforms.abstract_transforms import AbstractTransform
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.ndimage import distance_transform_edt as eucl_distance


class DistanceXor(AbstractTransform):
    def __init__(self, input_key1="target", input_key2="middle_target", output_key="xor_distance"):
        self.output_key = output_key
        self.input_key1 = input_key1
        self.input_key2 = input_key2
    
    def __call__(self, **data_dict):
        label1 = data_dict[self.input_key1]
        label2 = data_dict[self.input_key2]
        assert label1.shape[1] == 1
        batch_distance = []
        for t in range(len(label1)):
            distance = get_xor_distance(np.squeeze(label1[t]), np.squeeze(label2[t]))
            batch_distance.append(distance)
        batch_distance = np.stack(batch_distance, axis=0)
        data_dict[self.output_key] = batch_distance
        return data_dict

def get_xor_distance(label1, label2):
    out_list = []
    for t in range(0, 4):
        current_class1 = (label1 == t)
        current_class2 = (label2 == t)
        both_x = np.logical_xor(current_class1, current_class2)
        #if np.all(both_x == 0):
        #    d = np.zeros_like(both_x)
        #else:
        #    negmask = ~both_x
        #    d = eucl_distance(negmask) * negmask - (eucl_distance(both_x) - 1) * both_x
        #    d = (d - d.mean()) / d.std()
        out_list.append(both_x)
    out = np.stack(out_list, axis=0)

    matplotlib.use('QtAgg')
    fig, ax = plt.subplots(1, 5)
    print(np.unique(out[-2:].sum(axis=0)))
    ax[0].imshow(out[0], cmap='gray')
    ax[1].imshow(out[1], cmap='gray')
    ax[2].imshow(out[2], cmap='gray')
    ax[3].imshow(out[3], cmap='gray')
    ax[4].imshow(out.sum(axis=0), cmap='gray')
    plt.show()

    return out