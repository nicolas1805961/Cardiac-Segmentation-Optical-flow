import torch
from batchgenerators.augmentations.utils import convert_seg_image_to_one_hot_encoding_batched, resize_segmentation
from batchgenerators.transforms.abstract_transforms import AbstractTransform
from torch.nn.functional import avg_pool2d, avg_pool3d
import numpy as np
import cv2 as cv


class DistanceMap(AbstractTransform):
    def __init__(self, input_key="target", output_key="directional_field"):
        self.output_key = output_key
        self.input_key = input_key
    
    def __call__(self, **data_dict):
        seg = data_dict[self.input_key]
        assert seg.shape[1] == 1
        batch_df = []
        for t in range(len(seg)):
            df = get_distance_image(np.squeeze(seg[t]), norm=True)
            batch_df.append(df)
        batch_df = np.stack(batch_df, axis=0)
        data_dict[self.output_key] = batch_df
        return data_dict

def get_distance_image(label, norm):

        h, w = label.shape

        accumulation = np.zeros((2, h, w), dtype=np.float32)
        for t in range(1, 4):
            current_class = (label == t).astype(np.uint8)
            dst, labels = cv.distanceTransformWithLabels(current_class, cv.DIST_L2, cv.DIST_MASK_PRECISE, labelType=cv.DIST_LABEL_PIXEL)
            # labels is a LABEL map indicating LABEL (not index) of nearest zero pixel. Zero pixels have different labels.
            #  As a result som labels in backgound and in heart structure can have the same label.
            index = np.copy(labels)
            index[current_class > 0] = 0
            place = np.argwhere(index > 0) # get coords of background pixels
            nearCord = place[labels-1,:] # get coords of nearest zero pixel of EVERY pixels of the image. For background this is current coords.
            nearPixel = np.transpose(nearCord, axes=(2, 0, 1))
            grid = np.indices(current_class.shape).astype(float)
            diff = grid - nearPixel

            if norm:
                dr = np.sqrt(np.sum(diff**2, axis = 0))
            else:
                dr = np.ones_like(current_class)

            direction = np.zeros((2, h, w), dtype=np.float32)
            direction[0, current_class>0] = np.divide(diff[0, current_class>0], dr[current_class>0])
            direction[1, current_class>0] = np.divide(diff[1, current_class>0], dr[current_class>0])

            accumulation[:, current_class>0] = 0
            accumulation = accumulation + direction

            #fig, ax = plt.subplots(1, 5)
            #ax[0].imshow(labels, cmap='gray')
            #ax[1].imshow(index, cmap='gray')
            #ax[2].imshow(nearPixel[0], cmap='gray')
            #ax[3].imshow(diff[0], cmap='gray')
            #ax[4].imshow(accumulation[0], cmap='gray')
            #plt.show()
            #plt.waitforbuttonpress()
            #plt.close(fig)

        assert accumulation.max() <= 1.0 and accumulation.min() >= -1.0
        return accumulation