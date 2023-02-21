import torch
import random
from torch.distributions.uniform import Uniform
import torchvision.transforms.functional as TF
import numpy as np
from datasets import normalize_0_1, distance_transform, rand_bbox
import math
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from skimage.exposure import match_histograms
from datasets import process_image_lib
import elasticdeform.torch as etorch

class GaussianNoise(object):
    def __init__(self, std, mean=0., img_size=(224, 224)):
        self.std = std
        self.mean = mean
        self.img_size = img_size
        
    def augment_labeled(self, img, label, dist_map_tensor, index):
        img = img + self.image_noise * self.std + self.mean
        return img, label, dist_map_tensor

    def augment_unlabeled(self, img, index):
        img = img + self.image_noise * self.std + self.mean
        return img

    def reset(self, samples):
        self.image_noise = torch.randn(self.img_size)

class RandomCenterZoom(object):
    def __init__(self, scale):
        self.scale = scale

    def augment_labeled(self, img, label, dist_map_tensor, index):
        C, H, W = img.shape
        label = torch.argmax(label, dim=0, keepdim=True)
        scale = int(self.r*H)
        aug_img = TF.resize(TF.center_crop(img, scale), H, interpolation=TF.InterpolationMode.BILINEAR)
        aug_label = TF.resize(TF.center_crop(label, scale), H, interpolation=TF.InterpolationMode.NEAREST).squeeze(dim=0)
        aug_label = torch.nn.functional.one_hot(aug_label, num_classes=4).permute(2, 0, 1)
        new_dist_map_tensor = distance_transform(aug_label.numpy())
        return aug_img, aug_label, new_dist_map_tensor
    
    def augment_unlabeled(self, img, index):
        C, H, W = img.shape
        scale = int(self.r*H)
        aug_img = TF.resize(TF.center_crop(img, scale), H, interpolation=TF.InterpolationMode.BILINEAR)
        return aug_img
    
    def reset(self, samples):
        self.r = Uniform(self.scale[0], self.scale[1]).sample((1,)).item()

class RandomRotation(object):
    def __init__(self, degrees):
        self.angle = degrees

    def augment_labeled(self, img, label, dist_map_tensor, index):
        label = torch.argmax(label, dim=0, keepdim=True)
        aug_img = TF.rotate(img, self.alpha, interpolation=TF.InterpolationMode.BILINEAR, fill=0)
        aug_label = TF.rotate(label, self.alpha, interpolation=TF.InterpolationMode.NEAREST, fill=0).squeeze(dim=0)
        aug_label = torch.nn.functional.one_hot(aug_label, num_classes=4).permute(2, 0, 1)
        new_dist_map_tensor = distance_transform(aug_label.numpy())
        return aug_img, aug_label, new_dist_map_tensor
    
    def augment_unlabeled(self, img, index):
        aug_img = TF.rotate(img, self.alpha, interpolation=TF.InterpolationMode.BILINEAR, fill=0)
        return aug_img
    
    def reset(self, samples):
        self.alpha = random.randint(-self.angle, self.angle)

class RandomVerticalFlip(object):
    def __init__(self):
        pass

    def augment_labeled(self, img, label, dist_map_tensor, index):
        label = torch.argmax(label, dim=0, keepdim=True)
        aug_img = TF.vflip(img)
        aug_label = TF.vflip(label).squeeze(dim=0)
        aug_label = torch.nn.functional.one_hot(aug_label, num_classes=4).permute(2, 0, 1)
        new_dist_map_tensor = distance_transform(aug_label.numpy())
        return aug_img, aug_label, new_dist_map_tensor

    def augment_unlabeled(self, img, index):
        aug_img = TF.vflip(img)
        return aug_img
    
    def reset(self, samples):
        pass

class RandomElasticDeformation(object):
    def __init__(self, displacement_range):
        self.displacement_range = displacement_range

    def augment_labeled(self, img, label, dist_map_tensor, index):
        label = torch.argmax(label, dim=0)
        aug_img, aug_label = etorch.deform_grid([img.squeeze(0), label], self.displacement, order=[3, 0])
        aug_label = torch.nn.functional.one_hot(aug_label, num_classes=4).permute(2, 0, 1)
        new_dist_map_tensor = distance_transform(aug_label.numpy())
        return aug_img.unsqueeze(0), aug_label, new_dist_map_tensor

    def augment_unlabeled(self, img, index):
        aug_img = etorch.deform_grid(img, self.displacement, order=3)
        return aug_img.unsqueeze(0)
    
    def reset(self, samples):
        a = self.displacement_range[0]
        b = self.displacement_range[1]
        self.displacement = torch.randint(a, b, (2, 3, 3))

class RandomHorizontalFlip(object):
    def __init__(self):
        pass

    def augment_labeled(self, img, label, dist_map_tensor, index):
        label = torch.argmax(label, dim=0, keepdim=True)
        aug_img = TF.hflip(img)
        aug_label = TF.hflip(label).squeeze(dim=0)
        aug_label = torch.nn.functional.one_hot(aug_label, num_classes=4).permute(2, 0, 1)
        new_dist_map_tensor = distance_transform(aug_label.numpy())
        return aug_img, aug_label, new_dist_map_tensor

    def augment_unlabeled(self, img, index):
        aug_img = TF.hflip(img)
        return aug_img
    
    def reset(self, samples):
        pass

class RandomShearing(object):
    def __init__(self, shear_range):
        self.shear_range = shear_range

    def augment_labeled(self, img, label, dist_map_tensor, index):
        label = torch.argmax(label, dim=0, keepdim=True)
        aug_img = TF.affine(img, angle=0, translate=[0, 0], scale=1, shear=[self.x_angle, self.y_angle], interpolation=TF.InterpolationMode.BILINEAR, fill=0)
        aug_label = TF.affine(label, angle=0, translate=[0, 0], scale=1, shear=[self.x_angle, self.y_angle], interpolation=TF.InterpolationMode.NEAREST, fill=0).squeeze(dim=0)
        aug_label = torch.nn.functional.one_hot(aug_label, num_classes=4).permute(2, 0, 1)
        new_dist_map_tensor = distance_transform(aug_label.numpy())
        return aug_img, aug_label, new_dist_map_tensor
    
    def augment_unlabeled(self, img, index):
        aug_img = TF.affine(img, angle=0, translate=[0, 0], scale=1, shear=[self.x_angle, self.y_angle], interpolation=TF.InterpolationMode.BILINEAR, fill=0)
        return aug_img
    
    def reset(self, samples):
        self.x_angle = random.randint(-self.shear_range[0], self.shear_range[0])
        self.y_angle = random.randint(-self.shear_range[1], self.shear_range[1])

class RandomTranslate(object):
    def __init__(self, translate_scale):
        self.translate_scale = translate_scale

    def augment_labeled(self, img, label, dist_map_tensor, index):
        label = torch.argmax(label, dim=0, keepdim=True)
        aug_img = TF.affine(img, angle=0, translate=[self.x_shift, self.y_shift], scale=1, shear=0, interpolation=TF.InterpolationMode.BILINEAR, fill=0)
        aug_label = TF.affine(label, angle=0, translate=[self.x_shift, self.y_shift], scale=1, shear=0, interpolation=TF.InterpolationMode.NEAREST, fill=0).squeeze(dim=0)
        aug_label = torch.nn.functional.one_hot(aug_label, num_classes=4).permute(2, 0, 1)
        new_dist_map_tensor = distance_transform(aug_label.numpy())
        return aug_img, aug_label, new_dist_map_tensor
    
    def augment_unlabeled(self, img, index):
        aug_img = TF.affine(img, angle=0, translate=[self.x_shift, self.y_shift], scale=1, shear=0, interpolation=TF.InterpolationMode.BILINEAR, fill=0)
        return aug_img
    
    def reset(self, samples):
        self.x_shift = random.randint(-self.translate_scale[0], self.translate_scale[0])
        self.y_shift = random.randint(-self.translate_scale[1], self.translate_scale[1])

class RandomGaussianBlur(object):
    def __init__(self, blurr_sigma_range):
        self.blurr_sigma_range = blurr_sigma_range

    def augment_labeled(self, img, label, dist_map_tensor, index):
        img = TF.gaussian_blur(img, kernel_size=5, sigma=[self.sigma])
        return img, label, dist_map_tensor
    
    def augment_unlabeled(self, img, index):
        img = TF.gaussian_blur(img, kernel_size=5, sigma=[self.sigma])
        return img
    
    def reset(self, samples):
        self.sigma = Uniform(self.blurr_sigma_range[0], self.blurr_sigma_range[1]).sample((1,)).item()

class AdjustGamma(object):
    def __init__(self, gamma_range):
        self.gamma_range = gamma_range

    def augment_labeled(self, img, label, dist_map_tensor, index):
        img = TF.adjust_gamma(img, gamma=self.gamma, gain=1)
        return img, label, dist_map_tensor
    
    def augment_unlabeled(self, img, index):
        img = TF.adjust_gamma(img, gamma=self.gamma, gain=1)
        return img
    
    def reset(self, samples):
        self.gamma = Uniform(self.gamma_range[0], self.gamma_range[1]).sample((1,)).item()
        print(self.gamma)

class AdjustSharpness(object):
    def __init__(self, sharpness_factor_range):
        self.sharpness_factor_range = sharpness_factor_range

    def augment_labeled(self, img, label, dist_map_tensor, index):
        img = TF.adjust_sharpness(img, sharpness_factor=self.factor)
        return img, label, dist_map_tensor

    def augment_unlabeled(self, img, index):
        img = TF.adjust_sharpness(img, sharpness_factor=self.factor)
        return img
    
    def reset(self, samples):
        self.factor = Uniform(self.sharpness_factor_range[0], self.sharpness_factor_range[1]).sample((1,)).item()

class Mixup(object):
    def __init__(self, device, use_spatial_transformer):
        self.alpha = 0.5
        self.device=device
        self.use_spatial_transformer = use_spatial_transformer

    def augment_labeled(self, image1, label1, dist_map1, index):
        image2, label2, dist_map2, metadata = process_image_lib(self.sample[index], use_spatial_transformer=self.use_spatial_transformer, device=self.device, augmentation=None)
        image = self.lam * image1 + (1 - self.lam) * image2
        label = self.lam * label1 + (1 - self.lam) * label2
        dist_map = distance_transform(label.numpy())
        return image, label, dist_map
    
    def augment_unlabeled(self, image1, index):
        image2 = process_image_lib(self.sample[index], augmentation=None)
        image = self.lam * image1 + (1 - self.lam) * image2
        return image
    
    def reset(self, samples):
        self.lam = np.random.beta(self.alpha, self.alpha)
        new_idx = random.randint(0, len(samples)-1)
        self.sample = samples[new_idx]

class Cutmix(object):
    def __init__(self, device, use_spatial_transformer):
        self.alpha = 1
        self.device=device
        self.use_spatial_transformer = use_spatial_transformer

    def augment_labeled(self, image1, label1, dist_map1, index):
        C, H, W = image1.shape
        image2, label2, dist_map2, metadata = process_image_lib(self.sample[index], use_spatial_transformer=self.use_spatial_transformer, device=self.device, augmentation=None)
        bbx1, bby1, bbx2, bby2 = rand_bbox(image1.size(), self.lam)
        image1[:, bbx1:bbx2, bby1:bby2] = image2[:, bbx1:bbx2, bby1:bby2]
        label1[:, bbx1:bbx2, bby1:bby2] = label2[:, bbx1:bbx2, bby1:bby2]
        dist_map = distance_transform(label1.numpy())
        return image1, label1, dist_map
    
    def augment_unlabeled(self, image1, index):
        C, H, W = image1.shape
        image2 = process_image_lib(self.sample[index], augmentation=None)
        bbx1, bby1, bbx2, bby2 = rand_bbox(image1.size(), self.lam)
        image1[:, bbx1:bbx2, bby1:bby2] = image2[:, bbx1:bbx2, bby1:bby2]
        return image1
    
    def reset(self, samples):
        self.lam = np.random.beta(self.alpha, self.alpha)
        new_idx = random.randint(0, len(samples)-1)
        self.sample = samples[new_idx]

class Cowmix(object):
    def __init__(self, sigma_range, proportion_range, kernel_size, use_spatial_transformer, device):
        self.alpha = 1
        self.sigma_range = sigma_range
        self.proportion_range = proportion_range
        self.kernel_size = kernel_size
        self.device = device
        self.use_spatial_transformer = use_spatial_transformer

    def augment_labeled(self, image1, label1, dist_map1, index):
        C, H, W = image1.shape

        noise = torch.normal(0.0, 1.0, size=(1,) + image1.size())
        smoothed_noise = self.gaussian_smoothing(noise).squeeze(dim=0)
        noise_mean = smoothed_noise.mean()
        noise_std = smoothed_noise.std()
        thresholds = ((torch.erfinv(torch.tensor([2 * self.p - 1])).item() * math.sqrt(2.0)) * noise_std) + noise_mean
        masks = (smoothed_noise <= thresholds).float()

        image2, label2, dist_map2, metadata = process_image_lib(self.sample[index], use_spatial_transformer=self.use_spatial_transformer, device=self.device, augmentation=None)
        image = image1 * masks + image2 * (1 - masks)
        label = label1 * masks + label2 * (1 - masks)
        dist_map = distance_transform(label.numpy())
        return image, label, dist_map
    
    def augment_unlabeled(self, image1, index):
        C, H, W = image1.shape

        noise = torch.normal(0.0, 1.0, size=(1,) + image1.size())
        smoothed_noise = self.gaussian_smoothing(noise).squeeze(dim=0)
        noise_mean = smoothed_noise.mean()
        noise_std = smoothed_noise.std()
        thresholds = ((torch.erfinv(torch.tensor([2 * self.p - 1])).item() * math.sqrt(2.0)) * noise_std) + noise_mean
        masks = (smoothed_noise <= thresholds).float()

        image2 = process_image_lib(self.sample[index], augmentation=None)
        image = image1 * masks + image2 * (1 - masks)
        return image
    
    def reset(self, samples):
        sigma = Uniform(self.sigma_range[0], self.sigma_range[1]).sample((1,)).item()
        self.gaussian_smoothing = GaussianSmoothing(1, self.kernel_size, sigma)
        self.p = Uniform(self.proportion_range[0], self.proportion_range[1]).sample((1,)).item()
        new_idx = random.randint(0, len(samples)-1)
        self.sample = samples[new_idx]


class MyAugment(object):
    def __init__(self, device, use_spatial_transformer):
        self.device = device
        self.use_spatial_transformer = use_spatial_transformer

    def augment_labeled(self, image1, label1, dist_map1, index):
        image2, label2, dist_map2, metadata = process_image_lib(self.sample[index], use_spatial_transformer=self.use_spatial_transformer, device=self.device, augmentation=None)

        binary_mask1 = torch.zeros_like(image1).float()
        binary_mask1[torch.argmax(label1, dim=0, keepdim=True) > 0] = 1.0
        binary_mask2 = torch.zeros_like(image2).float()
        binary_mask2[torch.argmax(label2, dim=0, keepdim=True) > 0] = 1.0

        union = binary_mask1 + binary_mask2
        union[union > 1] = 1

        matched = match_histograms(image1.numpy(), image2.numpy()).astype(np.float32)

        part1_image = union * matched
        part2_image = (1 - union) * image2

        new_image = part1_image + part2_image

        return new_image, label1, dist_map1

    def augment_unlabeled(self, image1):
        return image1

    def reset(self, samples):
        new_idx = random.randint(0, len(samples)-1)
        self.sample = samples[new_idx]