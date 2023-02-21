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


def get_phases(paths):
    slice_lists = []
    for x in paths:
        patient_images = glob(os.path.join(x, '*.npz'))
        for i, patient_image in enumerate(patient_images):
            slice_name = patient_image.split('_')[-2]
            if i == 0:
                slice_list = []
                slice_list.append(patient_image)
            elif i == len(patient_images) - 1:
                slice_list.append(patient_image)
                slice_lists.append(slice_list)
            elif previous_slice_name == slice_name:
                slice_list.append(patient_image)
            else:
                slice_lists.append(slice_list)
                slice_list = []
                slice_list.append(patient_image)
            previous_slice_name = slice_name

    min_nb_frames = 1000
    for frames in slice_lists:
        a = len(frames)
        if a < min_nb_frames:
            min_nb_frames = a
    
    return slice_lists, min_nb_frames


def create_lib_datasets(path, nb_frames, val_subset_size, batch_size, method, device, data_augmentation_utils=None):
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
    
    
    #path_list_racine = [x for x in path_list if os.path.basename(x).startswith('RA_')]
    #path_list_temoin = [x for x in path_list if not os.path.basename(x).startswith('RA_')]
    #train_racine, test_racine = train_test_split(path_list_racine, test_size=0.2, random_state=0)
    #train_temoin, test_temoin = train_test_split(path_list_temoin, test_size=0.2, random_state=0)
#
    #train_racine, val_racine = train_test_split(train_racine, test_size=0.2, random_state=0)
    #train_temoin, val_temoin = train_test_split(train_temoin, test_size=0.2, random_state=0)
#
    #train_paths = train_racine + train_temoin
    #val_paths = val_racine + val_temoin
    #test_paths = test_racine + test_temoin
#
    slice_list_train, min_train = get_phases(train_list)
    slice_list_val, min_val = get_phases(val_list)
    slice_list_test, min_test = get_phases(test_list)
    min_nb_frames = min(min_train, min_val, min_test)

    train_dataset = MyDataset(slice_list_train, min_nb_frames=min_nb_frames, nb_frames=nb_frames, method=method, device=device,
                                         data_augmentation_utils=data_augmentation_utils)

    val_dataset = MyDataset(slice_list_val, min_nb_frames=min_nb_frames, nb_frames=nb_frames, method=method, device=device,
                                             data_augmentation_utils=None)

    test_dataset = MyDataset(slice_list_test, min_nb_frames=min_nb_frames, nb_frames=nb_frames, method=method, device=device,
                                             data_augmentation_utils=None)

    indices = np.random.randint(0, len(val_dataset), size=val_subset_size)
    val_dataset_subset = Subset(val_dataset, indices.tolist())

    val_random_sampler = RandomSampler(val_dataset, replacement=True, num_samples=1, generator=None)

    val_random_dataloader = DataLoader(val_dataset, sampler=val_random_sampler)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)
    val_dataloader_subset = DataLoader(val_dataset_subset, batch_size=1, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1)

    return {'labeled_train_dataloader': train_dataloader,
            'val_dataloader': val_dataloader, 
            'val_random_dataloader': val_random_dataloader, 
            'test_dataloader': test_dataloader,
            'val_dataloader_subset': val_dataloader_subset}

def process_image_lib(path, index=None, augmentation=None):
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    data = np.load(path)
    image = data['arr_0']
    label = data['arr_1']
    image = clahe.apply(image.astype(np.uint16)).astype(np.int32)
    image = torch.from_numpy(image).unsqueeze(dim=0).float()
    label = torch.from_numpy(label).long()
    label = TF.center_crop(label, 224)
    image = TF.center_crop(image, 224)
    label = torch.nn.functional.one_hot(label, num_classes=4).permute(2, 0, 1).float()
    dist_map_tensor = distance_transform(label.numpy())
    image = standardize(image, mean=3605.3267, std=3475.4136)
    image = normalize_0_1(image)
    if augmentation is not None:
        print(path)
        print(augmentation)
        fig, ax = plt.subplots(2, 3, figsize=(20, 8))
        ax[0, 0].imshow(image[0], cmap='gray')
        ax[0, 1].imshow(torch.argmax(label, 0), cmap='gray')
        ax[0, 2].imshow(dist_map_tensor[2], cmap='jet')
        image, label, dist_map_tensor = augmentation.augment_labeled(image, label, dist_map_tensor, index)
        image = normalize_0_1(image)
        ax[1, 0].imshow(image[0], cmap='gray')
        ax[1, 1].imshow(torch.argmax(label, 0), cmap='gray')
        ax[1, 2].imshow(dist_map_tensor[2], cmap='jet')
        plt.show()
    return image, label, dist_map_tensor

class MyDataset(Dataset):
    def __init__(self, slice_lists, min_nb_frames, nb_frames, method, device, data_augmentation_utils=None):

        samples = []
        if nb_frames == 1:
            for frames in slice_lists: # slice_lists = patient phases
                for frame in frames:
                    samples.append([frame])
        elif method == 'equal':
            for frames in slice_lists:
                indices = np.linspace(0, len(frames) - 1, nb_frames).astype(np.int16)
                path_indices = [frames[x] for x in indices]
                samples.append(path_indices)
        elif method == 'same_stride':
            for frames in slice_lists:
                stride = int(len(frames) / min_nb_frames)
                for i in range(len(frames) - stride*nb_frames + stride):
                    indices = np.arange(start=i, stop=i + stride*nb_frames, step=stride)
                    path_indices = [frames[x] for x in indices]
                    samples.append(path_indices)
        elif method == 'multiple_stride':
            for frames in slice_lists:
                for stride in range(1, int(len(frames) / (nb_frames - 1))):
                    for i in range(len(frames) - stride*(nb_frames - 1)):
                        indices = np.arange(start=i, stop=i + stride*nb_frames, step=stride)
                        path_indices = [frames[x] for x in indices]
                        samples.append(path_indices)
        elif method == 'block':
            for frames in slice_lists:
                for i in range(0, len(frames), nb_frames):
                    if i + nb_frames > len(frames):
                        break
                    indices = np.arange(start=i, stop=i + nb_frames, step=1)
                    path_indices = [frames[x] for x in indices]
                    samples.append(path_indices)

        self.device = device
        self.samples = samples
        self.data_augmentation_utils = data_augmentation_utils
        self.nb_frames = nb_frames
        #self.clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

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
        
        video_x = []
        video_y = []
        distance_maps = []
        for i, path in enumerate(self.samples[idx]):
            image, label, dist_map_tensor = process_image_lib(path, index=i, augmentation=augmentation)
            distance_maps.append(dist_map_tensor.to(self.device))
            video_x.append(image)
            video_y.append(label)

        video_x = torch.stack(video_x).to(self.device)
        video_y = torch.stack(video_y).to(self.device)

        return {"x": video_x,
                "y": video_y,
                "distance_map": distance_maps}