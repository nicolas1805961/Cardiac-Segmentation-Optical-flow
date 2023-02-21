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
from dataset_utils import process_labeled_image


def create_lib_datasets(path, img_size, batch_size, val_subset_size, device, use_spatial_transformer, binary, data_augmentation_utils=None):
    train = []
    val = []
    test = []
    for folder in glob(path):
        if any(x in folder for x in ['Wallet', 'classes', 'quality']):
            continue
        subjects = glob(os.path.join(folder, '*'))
        group_train, group_val, group_test = np.split(subjects, indices_or_sections=[int(0.7 * len(subjects)), int(0.8 * len(subjects))])
        train.append(group_train)
        val.append(group_val)
        test.append(group_test)
    train_list = np.concatenate(train)
    val_list = np.concatenate(val)
    test_list = np.concatenate(test)

    labeled_train_dataset = MyLabeledLIBDataset(use_spatial_transformer=use_spatial_transformer, img_size=img_size, slice_lists=train_list, binary=binary, device=device, data_augmentation_utils=data_augmentation_utils)
    val_dataset = MyLabeledLIBDataset(use_spatial_transformer=use_spatial_transformer, img_size=img_size, slice_lists=val_list, binary=binary, device=device, data_augmentation_utils=None)
    test_dataset = MyLabeledLIBDataset(use_spatial_transformer=use_spatial_transformer, img_size=img_size, slice_lists=test_list, binary=binary, device=device, data_augmentation_utils=None)

    indices = np.random.randint(0, len(val_dataset), size=val_subset_size)
    val_dataset_subset = Subset(val_dataset, indices.tolist())

    val_random_sampler = RandomSampler(val_dataset, replacement=True, num_samples=1, generator=None)

    val_random_dataloader = DataLoader(val_dataset, sampler=val_random_sampler)
    train_dataloader = DataLoader(labeled_train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)
    val_dataloader_subset = DataLoader(val_dataset_subset, batch_size=1, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1)

    return {'labeled_train_dataloader': train_dataloader,
            'val_dataloader': val_dataloader, 
            'val_random_dataloader': val_random_dataloader, 
            'test_dataloader': test_dataloader,
            'val_dataloader_subset': val_dataloader_subset}


class MyLabeledLIBDataset(Dataset):
    def __init__(self, slice_lists, use_spatial_transformer, device, img_size, binary, data_augmentation_utils=None):

        samples = []
        for path in slice_lists:
            patient_files = glob(os.path.join(path, '*'))
            samples = samples + patient_files

        self.device = device
        self.samples = samples
        self.data_augmentation_utils = data_augmentation_utils
        self.binary = binary
        self.img_size = img_size
        self.use_spatial_transformer=use_spatial_transformer

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
        image, label, dist_map_tensor, metadata = process_labeled_image(path=path, 
                                                                        rotate=False,
                                                                        mean=3605.3267,
                                                                        std=3475.4136,
                                                                        img_size=self.img_size, 
                                                                        use_spatial_transformer=self.use_spatial_transformer, 
                                                                        device=self.device, 
                                                                        augmentation=augmentation)

        if self.binary:
            label = torch.argmax(label, dim=0)
            label[label > 0] = 1
            label = torch.nn.functional.one_hot(label, num_classes=2).permute(2, 0, 1).float()

        #print(metadata['tx'])
        #print(metadata['ty'])
        #print(metadata['scale'])
        #print(metadata['angle'])
        #fig, ax = plt.subplots(1, 2, figsize=(20, 8))
        #ax[0].imshow(torch.argmax(label, dim=0).cpu(), cmap='gray')
        #ax[1].imshow(image.cpu()[0], cmap='gray')
        #plt.show()
        
        return {"x": image.to(self.device),
                "y": label.to(self.device),
                "distance_map": dist_map_tensor.to(self.device),
                "metadata": metadata}