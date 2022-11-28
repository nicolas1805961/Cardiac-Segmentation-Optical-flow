import torch
from batchgenerators.augmentations.utils import convert_seg_image_to_one_hot_encoding_batched, resize_segmentation
from batchgenerators.transforms.abstract_transforms import AbstractTransform
from torch.nn.functional import avg_pool2d, avg_pool3d
import numpy as np
import cv2 as cv
from skimage.measure import regionprops
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import matplotlib


class HeatMap(AbstractTransform):
    def __init__(self, input_key="middle_target", output_key="middle_heatmap"):
        self.output_key = output_key
        self.input_key = input_key
    
    def __call__(self, **data_dict):
        seg = data_dict[self.input_key]
        assert seg.shape[1] == 1
        batch_heatmap = []
        for t in range(len(seg)):
            heatmap = get_heatmap(np.squeeze(seg[t]).astype(int))
            batch_heatmap.append(heatmap)
        batch_heatmap = np.stack(batch_heatmap, axis=0)
        data_dict[self.output_key] = batch_heatmap
        return data_dict

def get_heatmap(label):
    a = np.zeros_like(label).astype(np.float32)
    label[label == 2] = 3.
    for region in regionprops(label):
        c = region.centroid
        a[int(round(c[0])), int(round(c[1]))] = 1.
    out = gaussian_filter(a, sigma=11)[None, :, :]
    out = (out - out.mean()) / out.std()

    return out
        