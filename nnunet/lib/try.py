from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from torch.distributions.uniform import Uniform
import torchvision.transforms.functional as TF
import torch
from dataset_utils import normalize_0_1
import elasticdeform.torch as etorch
from dataset_utils import process_labeled_image, process_unlabeled_image, normalize_0_1, distance_transform, rand_bbox

class RandomBrightnessAdjust(object):
    def __init__(self, brightness_factor_range):
        self.brightness_factor_range = brightness_factor_range

    def augment_labeled(self, img, label, dist_map_tensor):
        brightness_factor = Uniform(self.brightness_factor_range[0], self.brightness_factor_range[1]).sample((1,)).item()
        img = TF.adjust_brightness(img, brightness_factor)
        return img, label, dist_map_tensor
    
    def augment_unlabeled(self, img):
        brightness_factor = Uniform(self.brightness_factor_range[0], self.brightness_factor_range[1]).sample((1,)).item()
        img = TF.adjust_brightness(img, brightness_factor)
        return img

class RandomGaussianBlur(object):
    def __init__(self, blurr_sigma_range):
        self.blurr_sigma_range = blurr_sigma_range

    def augment_labeled(self, img, label, dist_map_tensor):
        sigma = Uniform(self.blurr_sigma_range[0], self.blurr_sigma_range[1]).sample((1,)).item()
        img = TF.gaussian_blur(img, kernel_size=5, sigma=sigma)
        return img, label, dist_map_tensor
    
    def augment_unlabeled(self, img):
        img = TF.gaussian_blur(img, kernel_size=5, sigma=[self.sigma])
        return img

class GaussianNoise(object):
    def __init__(self, std, mean=0.):
        self.std = std
        self.mean = mean
        
    def augment_labeled(self, img, label, dist_map_tensor):
        image_noise = torch.randn(img.shape)
        img = img + image_noise * self.std + self.mean
        return img, label, dist_map_tensor

    def augment_unlabeled(self, img):
        img = img + self.image_noise * self.std + self.mean
        return img

class AdjustGamma(object):
    def __init__(self, gamma_range):
        self.gamma_range = gamma_range

    def augment_labeled(self, img, label, dist_map_tensor):
        gamma = Uniform(self.gamma_range[0], self.gamma_range[1]).sample((1,)).item()
        img = TF.adjust_gamma(img, gamma=gamma, gain=1)
        return img, label, dist_map_tensor
    
    def augment_unlabeled(self, img):
        img = TF.adjust_gamma(img, gamma=self.gamma, gain=1)
        return img

class RandomElasticDeformation(object):
    def __init__(self, displacement_range):
        self.displacement_range = displacement_range

    def augment_labeled(self, img, label, dist_map_tensor):
        label = torch.argmax(label, dim=0)

        a = self.displacement_range[0]
        b = self.displacement_range[1]
        displacement = torch.randint(a, b, (2, 3, 3))

        aug_img, aug_label = etorch.deform_grid([img.squeeze(0), label], displacement, order=[3, 0])
        aug_label = torch.nn.functional.one_hot(aug_label, num_classes=4).permute(2, 0, 1).float()
        new_dist_map_tensor = distance_transform(aug_label.numpy())

        aug_img = normalize_0_1(aug_img)

        return aug_img.unsqueeze(0), aug_label, new_dist_map_tensor

    def augment_unlabeled(self, img):
        aug_img = etorch.deform_grid(img.squeeze(0), self.displacement, order=3)
        return aug_img.unsqueeze(0)

#path_list = glob('ACDC_data/*/labeled/*.npz')
#clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#r_object = RandomElasticDeformation([-15, -14])
#for path in path_list:
#    print(path)
#    data = np.load(path)
#    image = data['arr_0']
#    label = data['arr_1']
#    label = torch.from_numpy(label)
#    image_clahe = clahe.apply(image.astype(np.uint16)).astype(np.int32)
#    image_clahe = normalize_0_1(torch.from_numpy(image_clahe).float())
#    image_brightness_adjusted, _, _ = r_object.augment_labeled(image_clahe[None], label[None], image_clahe)
#    print(torch.unique(image_brightness_adjusted))
#    fig, ax = plt.subplots(1, 2)
#    ax[0].imshow(image_clahe, cmap='gray')
#    ax[1].imshow(image_brightness_adjusted.squeeze(), cmap='gray', vmin=0, vmax=1)
#    plt.show()