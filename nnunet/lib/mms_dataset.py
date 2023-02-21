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
from dataset_utils import process_labeled_image, process_unlabeled_image, rand_bbox


def create_mms_dataset(path, img_size, batch_size, use_spatial_transformer, device, binary, data_augmentation_utils=None):
    train_list_labeled = glob(os.path.join(path, 'Training', 'Labeled', '*'))
    train_list_unlabeled = glob(os.path.join(path, 'Training', 'Unlabeled', '*'))
    val_list = glob(os.path.join(path, 'Validation', '*'))
    test_list = glob(os.path.join(path, 'Testing', '*'))

    unlabeled_train_dataset1 = MyUnlabeledMMsDataset(slice_lists=train_list_unlabeled, device=device, data_augmentation_utils=data_augmentation_utils)
    unlabeled_train_dataset2 = MyUnlabeledMMsDataset(slice_lists=train_list_unlabeled, device=device, data_augmentation_utils=data_augmentation_utils)
    labeled_train_dataset = MyLabeledMMsDataset(use_spatial_transformer=use_spatial_transformer, img_size=img_size, slice_lists=train_list_labeled, binary=binary, device=device, data_augmentation_utils=data_augmentation_utils)
    val_dataset = MyLabeledMMsDataset(use_spatial_transformer=use_spatial_transformer, img_size=img_size, slice_lists=val_list, binary=binary, device=device, data_augmentation_utils=None)
    test_dataset = MyLabeledMMsDataset(use_spatial_transformer=use_spatial_transformer, img_size=img_size, slice_lists=test_list, binary=binary, device=device, data_augmentation_utils=None)

    val_random_sampler = RandomSampler(val_dataset, replacement=True, num_samples=1, generator=None)

    val_random_dataloader = DataLoader(val_dataset, sampler=val_random_sampler)
    unlabeled_train_dataloader1 = DataLoader(unlabeled_train_dataset1, batch_size=batch_size, shuffle=True)
    unlabeled_train_dataloader2 = DataLoader(unlabeled_train_dataset2, batch_size=batch_size, shuffle=True)
    labeled_train_dataloader = DataLoader(labeled_train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1)

    return {'labeled_train_dataloader': labeled_train_dataloader,
            'unlabeled_train_dataloader1': unlabeled_train_dataloader1,
            'unlabeled_train_dataloader2': unlabeled_train_dataloader2,
            'val_dataloader': val_dataloader, 
            'val_random_dataloader': val_random_dataloader, 
            'test_dataloader': test_dataloader,
            'val_dataloader_subset': val_dataloader}


class MyUnlabeledMMsDataset(Dataset):
    def __init__(self, slice_lists, device, data_augmentation_utils=None):

        samples = []
        for patient in slice_lists:
            patient_files = glob(os.path.join(patient, '*'))
            samples = samples + patient_files

        self.device = device
        self.samples = samples
        self.data_augmentation_utils = data_augmentation_utils
        self.alpha = 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        lam = np.random.beta(self.alpha, self.alpha)
        bbx1, bby1, bbx2, bby2 = rand_bbox((224, 224), lam)
        mask = torch.ones((224, 224))
        mask[bbx1:bbx2, bby1:bby2] = 0
        mask = mask.unsqueeze(dim=0)

        augmentation = None
        if self.data_augmentation_utils is not None:
            augmentation, probability = random.choice(list(self.data_augmentation_utils.values()))
            if random.random() > probability:
                augmentation = None
            else:
                augmentation.reset(self.samples)
        
        path = self.samples[idx]
        image = process_unlabeled_image(path, rotate=False, mean=6525.8843, std=5997.3311, augmentation=augmentation)
        
        return {"x": image.to(self.device), "mask": mask.to(self.device)}


class MyLabeledMMsDataset(Dataset):
    def __init__(self, slice_lists, use_spatial_transformer, device, img_size, binary, data_augmentation_utils=None):

        samples = []
        for patient in slice_lists:
            patient_files = glob(os.path.join(patient, '*'))
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
                                                                        mean=6525.8843,
                                                                        std=5997.3311,
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