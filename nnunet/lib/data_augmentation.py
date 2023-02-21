import torch
import random
from torch.distributions.uniform import Uniform
import torchvision.transforms.functional as TF
import numpy as np
import math
import matplotlib.pyplot as plt
import cv2 as cv
from skimage.exposure import match_histograms
from dataset_utils import process_2d_image, process_unlabeled_image, normalize_0_1, distance_transform, rand_bbox
import elasticdeform.torch as etorch
import torch.nn as nn
import torch.nn.functional as F
from monai.transforms import Rotate, Zoom, Affine, Rand2DElastic


class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma):
        super(GaussianSmoothing, self).__init__()
        kernel_size = [kernel_size] * 2
        sigma = [sigma] * 2

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in kernel_size])
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        self.conv = F.conv2d

    def forward(self, x):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(x, weight=self.weight, groups=self.groups, padding='same')


class GaussianNoise(object):
    def __init__(self, shape, std, mean=0.):
        self.std = std
        self.mean = mean
        self.shape = shape
        
    def augment_labeled(self, img, label):

        img_min = img.min()
        img_max = img.max()

        img = (img - img_min) / (img_max - img_min)

        aug_img = img + self.image_noise * self.std + self.mean
        aug_img = aug_img * (img_max - img_min) + img_min
        #aug_img = torch.clamp(aug_img, img_min, img_max)

        return aug_img, label
    
    def augment_images(self, img, label, logits_2d, logits_3d):

        img_min = img.min()
        img_max = img.max()

        non_zero_mask = (img != 0)
        non_zero_img = img[non_zero_mask]
        non_zero_noise = self.image_noise[non_zero_mask]
        
        aug_img = non_zero_img + non_zero_noise * self.std + self.mean
        aug_img = torch.clamp(aug_img, img_min, img_max)

        img[non_zero_mask] = aug_img

        return img, label, logits_2d, logits_3d
    
    def reset(self):
        self.image_noise = torch.randn(self.shape).unsqueeze(0)

    def augment_unlabeled(self, img):
        non_zero_mask = (img != 0)
        non_zero_img = img[non_zero_mask]
        non_zero_noise = self.image_noise[non_zero_mask]
        
        aug_img = non_zero_img + non_zero_noise * self.std + self.mean
        aug_img = torch.clamp(aug_img, 0, 1)

        img[non_zero_mask] = aug_img

        return img
        

class RandomCenterZoom(object):
    def __init__(self, scale):
        self.scale = scale

    def augment_labeled(self, img, label):

        img_min = img.min()
        img_max = img.max()

        C, H, W = img.shape
        label = torch.argmax(label, dim=0, keepdim=True)
        aug_label = self.zoom(label, mode='nearest', padding_mode='constant').long().squeeze(dim=0)
        aug_label = torch.nn.functional.one_hot(aug_label, num_classes=4).permute(2, 0, 1).float()

        aug_img = self.zoom(img, mode='bicubic', padding_mode='constant')
        aug_img[aug_img != 0] = torch.clamp(aug_img[aug_img != 0], img_min, img_max)

        return aug_img, aug_label
    
    def augment_images(self, img, label, logits_2d, logits_3d):
        C, H, W = img.shape
        label = torch.argmax(label, dim=0, keepdim=True)
        aug_label = self.zoom(label, mode='nearest', padding_mode='constant').long().squeeze(dim=0)
        aug_label = torch.nn.functional.one_hot(aug_label, num_classes=4).permute(2, 0, 1).float()

        aug_logits_2d = self.zoom(logits_2d, mode='bicubic', padding_mode='constant')
        aug_logits_2d = torch.clamp(aug_logits_2d, 0, 1)

        aug_logits_3d = self.zoom(logits_3d, mode='bicubic', padding_mode='constant')
        aug_logits_3d = torch.clamp(aug_logits_3d, 0, 1)

        aug_img = self.zoom(img, mode='bicubic', padding_mode='constant')
        aug_img = torch.clamp(aug_img, 0, 1)

        #fig, ax = plt.subplots(1, 3)
        #ax[0].imshow(aug_softmax_output[0], cmap='plasma')
        #ax[1].imshow(aug_img[0], cmap='gray')
        #ax[2].imshow(aug_label[0], cmap='gray')
        #plt.show()

        return aug_img, aug_label, aug_logits_2d, aug_logits_3d
    
    def reset(self):
        self.r = Uniform(self.scale[0], self.scale[1]).sample((1,)).item()
        self.zoom = Zoom(zoom=self.r)

    def augment_unlabeled(self, img):
        C, H, W = img.shape
        scale = int(self.r * H)
        aug_img = TF.resize(TF.center_crop(img, scale), H, interpolation=TF.InterpolationMode.bicubic)
        return aug_img
    
class RandomRotation(object):
    def __init__(self, degree_range):
        self.degree_range = degree_range

    def augment_labeled(self, img, label):
        img_min = img.min()
        img_max = img.max()

        label = torch.argmax(label, dim=0, keepdim=True)
        aug_label = self.rotate(label, mode='nearest', padding_mode='zeros').long().squeeze(dim=0)
        aug_label = torch.nn.functional.one_hot(aug_label, num_classes=4).permute(2, 0, 1).float()
        aug_img = self.rotate(img, mode='bicubic', padding_mode='zeros')
        aug_img[aug_img != 0] = torch.clamp(aug_img[aug_img != 0], img_min, img_max)

        #fig, ax = plt.subplots(1, 3)
        #ax[0].imshow(img[0], cmap='gray')
        #ax[1].imshow(aug_img[0], cmap='gray')
        #ax[2].imshow(torch.argmax(aug_label, dim=0), cmap='gray')
        #plt.show()

        return aug_img, aug_label
    
    def augment_images(self, img, label, logits_2d, logits_3d):
        label = torch.argmax(label, dim=0, keepdim=True)
        aug_label = self.rotate(label, mode='nearest', padding_mode='zeros').long().squeeze(dim=0)
        aug_label = torch.nn.functional.one_hot(aug_label, num_classes=4).permute(2, 0, 1).float()

        aug_img = self.rotate(img, mode='bicubic', padding_mode='zeros')
        aug_img = torch.clamp(aug_img, 0, 1)

        aug_logits_2d = self.rotate(logits_2d, mode='bicubic', padding_mode='zeros')
        aug_logits_2d = torch.clamp(aug_logits_2d, 0, 1)

        aug_logits_3d = self.rotate(logits_3d, mode='bicubic', padding_mode='zeros')
        aug_logits_3d = torch.clamp(aug_logits_3d, 0, 1)

        #fig, ax = plt.subplots(1, 3)
        #ax[0].imshow(img[0], cmap='gray')
        #ax[1].imshow(aug_img[0], cmap='gray')
        #ax[2].imshow(aug_label[0], cmap='gray')
        #plt.show()

        return aug_img, aug_label, aug_logits_2d, aug_logits_3d
    
    def reset(self):
        self.alpha = random.randint(self.degree_range[0], self.degree_range[1])
        self.alpha = math.radians(self.alpha)
        self.rotate = Rotate(angle=self.alpha)
    
    def augment_unlabeled(self, img):
        aug_img = TF.rotate(img, self.alpha, interpolation=TF.InterpolationMode.bicubic, fill=0)
        return aug_img


class RandomVerticalFlip(object):
    def __init__(self):
        pass

    def augment_labeled(self, img, label, dist_map_tensor):
        label = torch.argmax(label, dim=0, keepdim=True)
        aug_img = TF.vflip(img)
        aug_label = TF.vflip(label).squeeze(dim=0)
        aug_label = torch.nn.functional.one_hot(aug_label, num_classes=4).permute(2, 0, 1).float()
        new_dist_map_tensor = distance_transform(aug_label.numpy())
        return aug_img, aug_label, new_dist_map_tensor

    def augment_unlabeled(self, img):
        aug_img = TF.vflip(img)
        return aug_img
    
    def reset(self, samples):
        pass

class RandomHorizontalFlip(object):
    def __init__(self):
        pass

    def augment_labeled(self, img, label, dist_map_tensor):
        label = torch.argmax(label, dim=0, keepdim=True)
        aug_img = TF.hflip(img)
        aug_label = TF.hflip(label).squeeze(dim=0)
        aug_label = torch.nn.functional.one_hot(aug_label, num_classes=4).permute(2, 0, 1).float()
        new_dist_map_tensor = distance_transform(aug_label.numpy())
        return aug_img, aug_label, new_dist_map_tensor

    def augment_unlabeled(self, img):
        aug_img = TF.hflip(img)
        return aug_img
    
    def reset(self, samples):
        pass

class RandomElasticDeformation(object):
    def __init__(self, std):
        self.std = std

    def augment_labeled(self, img, label):
        label = torch.argmax(label, dim=0)
        aug_img, aug_label = etorch.deform_grid([img.squeeze(0), label], self.displacement, order=[3, 0], prefilter=False)
        aug_img = aug_img.unsqueeze(0)
        #aug_img = torch.clamp(aug_img, 0, 1).unsqueeze(0)
        aug_label = torch.nn.functional.one_hot(aug_label, num_classes=4).permute(2, 0, 1).float()
        return aug_img, aug_label
    
    def augment_images(self, img, label, logits_2d, logits_3d):
        label = torch.argmax(label, dim=0)
        aug_img, aug_label = etorch.deform_grid([img.squeeze(0), label], self.displacement, order=[3, 0], prefilter=False)
        aug_label = torch.nn.functional.one_hot(aug_label, num_classes=4).permute(2, 0, 1).float()
        aug_img = torch.clamp(aug_img, 0, 1).unsqueeze(0)

        aug_logits_2d = etorch.deform_grid(logits_2d, self.displacement, order=3, prefilter=False, axis=(1, 2))
        aug_logits_2d = torch.clamp(aug_logits_2d, 0, 1)

        aug_logits_3d = etorch.deform_grid(logits_3d, self.displacement, order=3, prefilter=False, axis=(1, 2))
        aug_logits_3d = torch.clamp(aug_logits_3d, 0, 1)

        #fig, ax = plt.subplots(1, 3)
        #ax[0].imshow(aug_softmax_output[0], cmap='plasma')
        #ax[1].imshow(aug_img[0], cmap='gray')
        #ax[2].imshow(aug_label[0], cmap='gray')
        #plt.show()

        return aug_img, aug_label, aug_logits_2d, aug_logits_3d
    
    def reset(self):
        #self.deform = Rand2DElastic(prob=1.0, spacing=(30, 30), magnitude_range=self.magnitude_range)
        #self.seed=random.randint(0, sys.maxsize)
        self.displacement = torch.normal(mean=0.0, std=self.std, size=(2, 3, 3))

    def augment_unlabeled(self, img):
        aug_img = etorch.deform_grid(img.squeeze(0), self.displacement, order=3, prefilter=False)
        aug_img = torch.clamp(aug_img, 0, 1).unsqueeze(0)
        return aug_img
        

class RandomShearing(object):
    def __init__(self, shear_range):
        self.shear_range = shear_range

    def augment_labeled(self, img, label, dist_map_tensor):
        label = torch.argmax(label, dim=0, keepdim=True)
        aug_img = TF.affine(img, angle=0, translate=[0, 0], scale=1, shear=[self.x_angle, self.y_angle], interpolation=TF.InterpolationMode.bicubic, fill=0)
        aug_label = TF.affine(label, angle=0, translate=[0, 0], scale=1, shear=[self.x_angle, self.y_angle], interpolation=TF.InterpolationMode.NEAREST, fill=0).squeeze(dim=0)
        aug_label = torch.nn.functional.one_hot(aug_label, num_classes=4).permute(2, 0, 1).float()
        new_dist_map_tensor = distance_transform(aug_label.numpy())
        return aug_img, aug_label, new_dist_map_tensor
    
    def augment_unlabeled(self, img):
        aug_img = TF.affine(img, angle=0, translate=[0, 0], scale=1, shear=[self.x_angle, self.y_angle], interpolation=TF.InterpolationMode.bicubic, fill=0)
        return aug_img
    
    def reset(self, samples):
        self.x_angle = random.randint(-self.shear_range[0], self.shear_range[0])
        self.y_angle = random.randint(-self.shear_range[1], self.shear_range[1])

class RandomTranslate(object):
    def __init__(self, translate_scale):
        self.translate_scale = translate_scale

    def augment_labeled(self, img, label):
        img_min = img.min()
        img_max = img.max()

        label = torch.argmax(label, dim=0, keepdim=True)
        aug_label = self.affine(label, mode='nearest', padding_mode='zeros').long().squeeze(dim=0)
        aug_label = torch.nn.functional.one_hot(aug_label, num_classes=4).permute(2, 0, 1).float()
        aug_img = self.affine(img, mode='bicubic', padding_mode='zeros')
        aug_img[aug_img != 0] = torch.clamp(aug_img[aug_img != 0], img_min, img_max)
        return aug_img, aug_label
    
    def augment_images(self, img, label, logits_2d, logits_3d):
        label = torch.argmax(label, dim=0, keepdim=True)
        aug_label = self.affine(label, mode='nearest', padding_mode='zeros').long().squeeze(dim=0)
        aug_label = torch.nn.functional.one_hot(aug_label, num_classes=4).permute(2, 0, 1).float()

        aug_img = self.affine(img, mode='bicubic', padding_mode='zeros')
        aug_img = torch.clamp(aug_img, 0, 1)

        aug_logits_2d = self.affine(logits_2d, mode='bicubic', padding_mode='zeros')
        aug_logits_2d = torch.clamp(aug_logits_2d, 0, 1)

        aug_logits_3d = self.affine(logits_3d, mode='bicubic', padding_mode='zeros')
        aug_logits_3d = torch.clamp(aug_logits_3d, 0, 1)

        #fig, ax = plt.subplots(1, 3)
        #ax[0].imshow(aug_softmax_output[0], cmap='plasma')
        #ax[1].imshow(aug_img[0], cmap='gray')
        #ax[2].imshow(aug_label[0], cmap='gray')
        #plt.show()

        return aug_img, aug_label, aug_logits_2d, aug_logits_3d
    
    def reset(self):
        self.x_shift = random.randint(-self.translate_scale[0], self.translate_scale[0])
        self.y_shift = random.randint(-self.translate_scale[1], self.translate_scale[1])
        self.affine = Affine(translate_params=(self.y_shift, self.x_shift), image_only=True)
    
    def augment_unlabeled(self, img):
        aug_img = TF.affine(img, angle=0, translate=[self.x_shift, self.y_shift], scale=1, shear=0, interpolation=TF.InterpolationMode.bicubic, fill=0)
        return aug_img
        

class RandomGaussianBlur(object):
    def __init__(self, blurr_sigma_range):
        self.blurr_sigma_range = blurr_sigma_range

    def augment_labeled(self, img, label):
        aug_img = TF.gaussian_blur(img, kernel_size=5, sigma=self.sigma)
        aug_img = torch.clamp(aug_img, 0, 1)
        return aug_img, label
    
    def augment_images(self, img, label, softmax_output):
        aug_img = TF.gaussian_blur(img, kernel_size=5, sigma=self.sigma)
        aug_img = torch.clamp(aug_img, 0, 1)
        return aug_img, label, softmax_output
    
    def reset(self):
        self.sigma = Uniform(self.blurr_sigma_range[0], self.blurr_sigma_range[1]).sample((1,)).item()
    
    def augment_unlabeled(self, img):
        aug_img = TF.gaussian_blur(img, kernel_size=5, sigma=self.sigma)
        aug_img = torch.clamp(aug_img, 0, 1)
        return aug_img

class AdjustGamma(object):
    def __init__(self, gamma_range):
        self.gamma_range = gamma_range

    def augment_labeled(self, img, label):
        img_min = img.min()
        img_max = img.max()
        img = (img - img_min) / (img_max - img_min)
        aug_img = TF.adjust_gamma(img, gamma=self.gamma, gain=1) * (img_max - img_min) + img_min
        
        return aug_img, label
    
    def augment_images(self, img, label, logits_2d, logits_3d):
        aug_img = TF.adjust_gamma(img, gamma=self.gamma, gain=1)
        return aug_img, label, logits_2d, logits_3d
    
    def reset(self):
        self.gamma = Uniform(self.gamma_range[0], self.gamma_range[1]).sample((1,)).item()
    
    def augment_unlabeled(self, img):
        aug_img = TF.adjust_gamma(img, gamma=self.gamma, gain=1)
        return aug_img

class RandomBrightnessAdjust(object):
    def __init__(self, brightness_factor_range):
        self.brightness_factor_range = brightness_factor_range

    def augment_labeled(self, img, label):
        img_min = img.min()
        img_max = img.max()
        img = (img - img_min) / (img_max - img_min)
        aug_img = TF.adjust_brightness(img, self.brightness_factor) * (img_max - img_min) + img_min
        return aug_img, label
    
    def augment_images(self, img, label, logits_2d, logits_3d):
        aug_img = TF.adjust_brightness(img, self.brightness_factor)
        aug_img = torch.clamp(aug_img, 0, 1)
        return aug_img, label, logits_2d, logits_3d
    
    def reset(self):
        self.brightness_factor = Uniform(self.brightness_factor_range[0], self.brightness_factor_range[1]).sample((1,)).item()
    
    def augment_unlabeled(self, img):
        aug_img = TF.adjust_brightness(img, self.brightness_factor)
        aug_img = torch.clamp(aug_img, 0, 1)
        return aug_img

class RandomAdjustSharpness(object):
    def __init__(self, sharpness_factor_range):
        self.sharpness_factor_range = sharpness_factor_range

    def augment_labeled(self, img, label):
        img_min = img.min()
        img_max = img.max()
        img = (img - img_min) / (img_max - img_min)
        aug_img = TF.adjust_sharpness(img, sharpness_factor=self.factor) * (img_max - img_min) + img_min

        return aug_img, label
    
    def augment_images(self, img, label, logits_2d, logits_3d):
        aug_img = TF.adjust_sharpness(img, sharpness_factor=self.factor)
        aug_img = torch.clamp(aug_img, 0, 1)
        return aug_img, label, logits_2d, logits_3d

    def augment_unlabeled(self, img):
        aug_img = TF.adjust_sharpness(img, sharpness_factor=self.factor)
        aug_img = torch.clamp(aug_img, 0, 1)
        return aug_img
    
    def reset(self):
        self.factor = Uniform(self.sharpness_factor_range[0], self.sharpness_factor_range[1]).sample((1,)).item()

class Mixup(object):
    def __init__(self, device, use_spatial_transformer, dataset):
        if dataset == 'acdc':
            self.rotate = True
            self.mean = 3258.0759
            self.std = 3507.0581
        elif dataset == 'lib':
            self.rotate = False
            self.mean = 3605.3267
            self.std = 3475.4136
        elif dataset == 'M&Ms':
            self.rotate = False
            self.mean = 6525.8843
            self.std = 5997.3311

        self.alpha = 0.5
        self.device=device
        self.use_spatial_transformer = use_spatial_transformer

    def augment_labeled(self, image1, label1, dist_map1):
        image2, label2, dist_map2, metadata = process_labeled_image_2d(self.sample, rotate=self.rotate, mean=self.mean, std=self.std, use_spatial_transformer=self.use_spatial_transformer, device=self.device, augmentation=None)
        image = self.lam * image1 + (1 - self.lam) * image2
        label = self.lam * label1 + (1 - self.lam) * label2
        dist_map = distance_transform(label.numpy())
        return image, label, dist_map
    
    def augment_unlabeled(self, image1):
        image2 = process_unlabeled_image(self.sample, augmentation=None)
        image = self.lam * image1 + (1 - self.lam) * image2
        return image
    
    def reset(self, samples):
        self.lam = np.random.beta(self.alpha, self.alpha)
        new_idx = random.randint(0, len(samples)-1)
        self.sample = samples[new_idx]

class Cutmix(object):
    def __init__(self, device, use_spatial_transformer, dataset):
        if dataset == 'acdc':
            self.rotate = True
            self.mean = 3258.0759
            self.std = 3507.0581
        elif dataset == 'lib':
            self.rotate = False
            self.mean = 3605.3267
            self.std = 3475.4136
        elif dataset == 'M&Ms':
            self.rotate = False
            self.mean = 6525.8843
            self.std = 5997.3311
        self.alpha = 1
        self.device=device
        self.use_spatial_transformer = use_spatial_transformer

    def augment_labeled(self, image1, label1, dist_map1):
        C, H, W = image1.shape
        image2, label2, dist_map2, metadata = process_labeled_image_2d(self.sample, rotate=self.rotate, mean=self.mean, std=self.std, use_spatial_transformer=self.use_spatial_transformer, device=self.device, augmentation=None)
        bbx1, bby1, bbx2, bby2 = rand_bbox(image1.size(), self.lam)
        image1[:, bbx1:bbx2, bby1:bby2] = image2[:, bbx1:bbx2, bby1:bby2]
        label1[:, bbx1:bbx2, bby1:bby2] = label2[:, bbx1:bbx2, bby1:bby2]
        dist_map = distance_transform(label1.numpy())
        return image1, label1, dist_map
    
    def augment_unlabeled(self, image1):
        C, H, W = image1.shape
        image2 = process_unlabeled_image(self.sample, augmentation=None)
        bbx1, bby1, bbx2, bby2 = rand_bbox(image1.size(), self.lam)
        image1[:, bbx1:bbx2, bby1:bby2] = image2[:, bbx1:bbx2, bby1:bby2]
        return image1
    
    def reset(self, samples):
        self.lam = np.random.beta(self.alpha, self.alpha)
        new_idx = random.randint(0, len(samples)-1)
        self.sample = samples[new_idx]

class Cowmix(object):
    def __init__(self, sigma_range, proportion_range, kernel_size, use_spatial_transformer, dataset, device):
        if dataset == 'acdc':
            self.rotate = True
            self.mean = 3258.0759
            self.std = 3507.0581
        elif dataset == 'lib':
            self.rotate = False
            self.mean = 3605.3267
            self.std = 3475.4136
        elif dataset == 'M&Ms':
            self.rotate = False
            self.mean = 6525.8843
            self.std = 5997.3311
        self.alpha = 1
        self.sigma_range = sigma_range
        self.proportion_range = proportion_range
        self.kernel_size = kernel_size
        self.device = device
        self.use_spatial_transformer = use_spatial_transformer

    def augment_labeled(self, image1, label1, dist_map1):
        C, H, W = image1.shape

        noise = torch.normal(0.0, 1.0, size=(1,) + image1.size())
        smoothed_noise = self.gaussian_smoothing(noise).squeeze(dim=0)
        noise_mean = smoothed_noise.mean()
        noise_std = smoothed_noise.std()
        thresholds = ((torch.erfinv(torch.tensor([2 * self.p - 1])).item() * math.sqrt(2.0)) * noise_std) + noise_mean
        masks = (smoothed_noise <= thresholds).float()

        image2, label2, dist_map2, metadata = process_labeled_image_2d(self.sample, rotate=self.rotate, mean=self.mean, std=self.std, use_spatial_transformer=self.use_spatial_transformer, device=self.device, augmentation=None)
        image = image1 * masks + image2 * (1 - masks)
        label = label1 * masks + label2 * (1 - masks)
        dist_map = distance_transform(label.numpy())
        return image, label, dist_map
    
    def augment_unlabeled(self, image1):
        C, H, W = image1.shape

        noise = torch.normal(0.0, 1.0, size=(1,) + image1.size())
        smoothed_noise = self.gaussian_smoothing(noise).squeeze(dim=0)
        noise_mean = smoothed_noise.mean()
        noise_std = smoothed_noise.std()
        thresholds = ((torch.erfinv(torch.tensor([2 * self.p - 1])).item() * math.sqrt(2.0)) * noise_std) + noise_mean
        masks = (smoothed_noise <= thresholds).float()

        image2 = process_unlabeled_image(self.sample, augmentation=None)
        image = image1 * masks + image2 * (1 - masks)
        return image
    
    def reset(self, samples):
        sigma = Uniform(self.sigma_range[0], self.sigma_range[1]).sample((1,)).item()
        self.gaussian_smoothing = GaussianSmoothing(1, self.kernel_size, sigma)
        self.p = Uniform(self.proportion_range[0], self.proportion_range[1]).sample((1,)).item()
        new_idx = random.randint(0, len(samples)-1)
        self.sample = samples[new_idx]

def process_my_augment(image1, label1, image2, label2, device):
    binary_mask1 = torch.zeros_like(image1).float()
    binary_mask1[torch.argmax(label1, dim=1, keepdim=True) > 0] = 1.0
    binary_mask2 = torch.zeros_like(image2).float()
    binary_mask2[torch.argmax(label2, dim=1, keepdim=True) > 0] = 1.0

    union = binary_mask1 + binary_mask2
    union[union > 1] = 1

    matched = torch.from_numpy(match_histograms(image1.cpu().numpy(), image2.cpu().numpy()).astype(np.float32)).to(device)

    part1_image = union * matched
    part2_image = (1 - union) * image2

    new_image = part1_image + part2_image
    new_image = normalize_0_1(new_image)

    return new_image

class MyAugment(object):
    def __init__(self, device, use_spatial_transformer, dataset):
        if dataset == 'acdc':
            self.rotate = True
            self.mean = 3258.0759
            self.std = 3507.0581
        elif dataset == 'lib':
            self.rotate = False
            self.mean = 3605.3267
            self.std = 3475.4136
        elif dataset == 'M&Ms':
            self.rotate = False
            self.mean = 6525.8843
            self.std = 5997.3311
        self.device = device
        self.use_spatial_transformer = use_spatial_transformer

    def augment_labeled(self, image1, label1, dist_map1):
        fig, ax = plt.subplots(2, 3)
        image2, label2, dist_map2, metadata = process_labeled_image_2d(self.sample, rotate=self.rotate, mean=self.mean, std=self.std, use_spatial_transformer=self.use_spatial_transformer, device=self.device, augmentation=None)

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