import torch
from torch.utils.data import Dataset, Subset
from torch.utils.data import RandomSampler, DataLoader
import cv2 as cv
from glob import glob
import numpy as np
import os
from sklearn.model_selection import train_test_split
from random import shuffle
from torchvision.transforms.functional import adjust_gamma
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import random
import nibabel as nib
import torchvision.transforms.functional as TF
from scipy.ndimage import distance_transform_edt as eucl_distance
from boundary_utils import one_hot
from typing import cast
from torch import Tensor
from torch.utils.data import Sampler
import torch.nn.functional as F
import math
from skimage.measure import regionprops


def create_gan_acdc_dataset(path, batch_size, device, image_or_label, data_augmentation_utils=None):
    path_list = glob(path)
    train = []
    val = []
    test = []
    for i in range(5):
        array_path_list = np.array(path_list[i*20:(i+1)*20])
        group_train, group_val, group_test = np.split(array_path_list, indices_or_sections=[int(0.7*len(array_path_list)), int(0.8*len(array_path_list))])
        train.append(group_train)
        val.append(group_val)
        test.append(group_test)
    train_list = np.concatenate(train)
    val_list = np.concatenate(val)
    test_list = np.concatenate(test)

    labeled_train_dataset = MyLabeledGANACDCDataset(slice_lists=train_list, device=device, image_or_label=image_or_label, data_augmentation_utils=data_augmentation_utils)
    unlabeled_train_dataset = MyUnlabeledGANACDCDataset(slice_lists=train_list, device=device, data_augmentation_utils=data_augmentation_utils)
    train_dataset = torch.utils.data.ConcatDataset([labeled_train_dataset, unlabeled_train_dataset])

    labeled_val_dataset = MyLabeledGANACDCDataset(slice_lists=val_list, device=device, image_or_label=image_or_label, data_augmentation_utils=None)
    unlabeled_val_dataset = MyUnlabeledGANACDCDataset(slice_lists=val_list, device=device, data_augmentation_utils=None)
    val_dataset = torch.utils.data.ConcatDataset([labeled_val_dataset, unlabeled_val_dataset])

    labeled_test_dataset = MyLabeledGANACDCDataset(slice_lists=test_list, device=device, image_or_label=image_or_label, data_augmentation_utils=None)
    unlabeled_test_dataset = MyUnlabeledGANACDCDataset(slice_lists=test_list, device=device, data_augmentation_utils=None)
    test_dataset = torch.utils.data.ConcatDataset([labeled_test_dataset, unlabeled_test_dataset])

    val_random_sampler = RandomSampler(val_dataset, replacement=True, num_samples=1, generator=None)

    val_random_dataloader = DataLoader(val_dataset, sampler=val_random_sampler)
    labeled_train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1)

    return {'labeled_train_dataloader': labeled_train_dataloader,
            'val_dataloader': val_dataloader, 
            'val_random_dataloader': val_random_dataloader, 
            'test_dataloader': test_dataloader,
            'val_dataloader_subset': val_dataloader}


class MyUnlabeledGANACDCDataset(Dataset):
    def __init__(self, slice_lists, device, data_augmentation_utils=None):

        samples = []
        for path in slice_lists:
            patient_files = glob(os.path.join(path, 'unlabeled', '*'))
            samples = samples + patient_files

        self.device = device
        self.samples = samples
        self.data_augmentation_utils = data_augmentation_utils

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        augmentation = None
        if self.data_augmentation_utils is not None:
            augmentation, probability = random.choice(list(self.data_augmentation_utils.values()))
            if random.random() > probability:
                augmentation = None
            else:
                augmentation.reset(self.samples)
        
        path = self.samples[idx]
        image = process_unlabeled_image(path, rotate=True, mean=, std=, augmentation=augmentation)
        
        return image.to(self.device)

class MyLabeledGANACDCDataset(Dataset):
    def __init__(self, slice_lists, device, image_or_label, data_augmentation_utils=None):

        samples = []
        for path in slice_lists:
            patient_files = glob(os.path.join(path, 'labeled', '*'))
            samples = samples + patient_files

        self.device = device
        self.samples = samples
        self.data_augmentation_utils = data_augmentation_utils
        self.image_or_label=image_or_label

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        augmentation = None
        if self.data_augmentation_utils is not None:
            augmentation, probability = random.choice(list(self.data_augmentation_utils.values()))
            if random.random() > probability:
                augmentation = None
            else:
                augmentation.reset(self.samples)
        
        path = self.samples[idx]
        image, label, dist_map_tensor, metadata = process_labeled_image(path, rotate=True, mean=, std=, augmentation=augmentation)
        
        if self.image_or_label == 'image':
            return image.to(self.device)
        elif self.image_or_label == 'label':
            return label.to(self.device)