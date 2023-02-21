import torch
from torch.utils.data import Dataset, Subset
from torch.utils.data import RandomSampler, DataLoader
from glob import glob
import numpy as np
import os
import matplotlib.pyplot as plt
import random
from monai.transforms import ResizeWithPadOrCrop
from dataset_utils import process_2d_image_crop, process_2d_image, process_labeled_image_3d_crop, process_labeled_image_2d_logits, process_labeled_image_3d_logits, process_unlabeled_image, rand_bbox, normalize_0_1_autoencoder, process_labeled_image_3d
from torch import Tensor
from typing import Iterator, Iterable, Optional, Sequence, List, TypeVar, Generic, Sized, Union
from torch.utils.data.sampler import Sampler, SequentialSampler

def get_cases(slice_lists):
    samples = []
    for patient_path in slice_lists:
        phase_list = [[], []]
        current_index = 0
        n_prev = -1
        for file_path in glob(os.path.join(patient_path, 'labeled/*.npz')):
            n = int(file_path.split(os.sep)[-1].split('_')[0].split('frame')[-1])
            if n != n_prev:
                if current_index == 0:
                    current_index = 1
                else:
                    current_index = 0
            phase_list[current_index].append(file_path)
            n_prev = n
        phase_list[0] = sorted(phase_list[0], key=lambda x: int(x.split('.')[0].split('slice')[-1]))
        phase_list[1] = sorted(phase_list[1], key=lambda x: int(x.split('.')[0].split('slice')[-1]))
        samples = samples + phase_list
    return samples


class CustomSamplerUniform(Sampler[int]):
    r"""Samples elements from ``[0,..,len(weights)-1]`` with given probabilities (weights).

    Args:
        weights (sequence)   : a sequence of weights, not necessary summing up to one
        num_samples (int): number of samples to draw
        replacement (bool): if ``True``, samples are drawn with replacement.
            If not, they are drawn without replacement, which means that when a
            sample index is drawn for a row, it cannot be drawn again for that row.
        generator (Generator): Generator used in sampling.

    Example:
        >>> list(WeightedRandomSampler([0.1, 0.9, 0.4, 0.7, 3.0, 0.6], 5, replacement=True))
        [4, 4, 1, 4, 5]
        >>> list(WeightedRandomSampler([0.9, 0.4, 0.05, 0.2, 0.3, 0.1], 5, replacement=False))
        [0, 1, 4, 3, 2]
    """
    weights: Tensor
    num_samples: int
    replacement: bool

    def __init__(self, num_samples: int, nb_epochs, writer, sampler_param, replacement: bool = True, generator=None) -> None:
        if not isinstance(num_samples, int) or isinstance(num_samples, bool) or \
                num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(num_samples))
        if not isinstance(replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(replacement))
        self.num_samples = num_samples
        self.replacement = replacement
        self.generator = generator
        self.current_epoch = -1
        self.nb_epochs = nb_epochs
        self.writer = writer
        self.full_dataset_epoch = self.nb_epochs / sampler_param

    def __iter__(self) -> Iterator[int]:
        self.current_epoch += 1
        b = self.num_samples / 2
        a = (self.num_samples - b) / self.full_dataset_epoch
        high = int(round(a * self.current_epoch + b))
        if high > self.num_samples:
            high = self.num_samples
            self.replacement = False
        weights = torch.ones(size=(high,))
        weights = weights / weights.sum()
        rand_tensor = torch.multinomial(weights, self.num_samples, self.replacement, generator=self.generator)
        self.writer.add_histogram('Epoch/Samples_drawn', rand_tensor, self.current_epoch)
        shuffle_idx = torch.randperm(rand_tensor.numel())
        rand_tensor = rand_tensor[shuffle_idx]
        yield from iter(rand_tensor.tolist())

    def __len__(self) -> int:
        return self.num_samples


class CustomSamplerNormal(Sampler[int]):
    r"""Samples elements from ``[0,..,len(weights)-1]`` with given probabilities (weights).

    Args:
        weights (sequence)   : a sequence of weights, not necessary summing up to one
        num_samples (int): number of samples to draw
        replacement (bool): if ``True``, samples are drawn with replacement.
            If not, they are drawn without replacement, which means that when a
            sample index is drawn for a row, it cannot be drawn again for that row.
        generator (Generator): Generator used in sampling.

    Example:
        >>> list(WeightedRandomSampler([0.1, 0.9, 0.4, 0.7, 3.0, 0.6], 5, replacement=True))
        [4, 4, 1, 4, 5]
        >>> list(WeightedRandomSampler([0.9, 0.4, 0.05, 0.2, 0.3, 0.1], 5, replacement=False))
        [0, 1, 4, 3, 2]
    """
    weights: Tensor
    num_samples: int
    replacement: bool

    def __init__(self, num_samples: int, nb_epochs, writer, sampler_param, replacement: bool = True, generator=None) -> None:
        if not isinstance(num_samples, int) or isinstance(num_samples, bool) or \
                num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(num_samples))
        if not isinstance(replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(replacement))
        self.num_samples = num_samples
        self.replacement = replacement
        self.generator = generator
        self.current_epoch = -1
        self.nb_epochs = nb_epochs
        self.writer = writer
        self.sampler_param = sampler_param
        self.reached_epoch = self.nb_epochs / 4

    def __iter__(self) -> Iterator[int]:
        self.current_epoch += 1
        current_mean = self.current_epoch * (self.num_samples / self.reached_epoch)
        if current_mean > self.num_samples:
            current_mean = self.num_samples
        distribution = torch.distributions.normal.Normal(loc=current_mean, scale=self.num_samples / self.sampler_param)
        weights = distribution.log_prob(torch.arange(start=0, end=self.num_samples)).exp()
        rand_tensor = torch.multinomial(weights, self.num_samples, self.replacement, generator=self.generator)
        self.writer.add_histogram('Epoch/Samples_drawn', rand_tensor, self.current_epoch)
        shuffle_idx = torch.randperm(rand_tensor.numel())
        rand_tensor = rand_tensor[shuffle_idx]
        yield from iter(rand_tensor.tolist())

    def __len__(self) -> int:
        return self.num_samples

def my_collate(batch):
    out = {}
    out['x'] = torch.stack([batch[i]['x'] for i in range(len(batch))], dim=0)
    out['y'] = torch.stack([batch[i]['y'] for i in range(len(batch))], dim=0)
    out['original_label'] = [batch[i]['original_label'] for i in range(len(batch))]
    out['original_image'] = [batch[i]['original_image'] for i in range(len(batch))]
    out['image'] = [batch[i]['image'] for i in range(len(batch))]
    out['label'] = [batch[i]['label'] for i in range(len(batch))]
    if 'directional_field' in batch[0]:
        out['directional_field'] = torch.stack([batch[i]['directional_field'] for i in range(len(batch))], dim=0)
    if 'parameters' in batch[0]:
        out['parameters'] = torch.stack([batch[i]['parameters'] for i in range(len(batch))], dim=0)
    if 'padding' in batch[0]:
        out['padding'] = torch.stack([batch[i]['padding'] for i in range(len(batch))], dim=0)

    return out

def get_size_train_val(path):
    path_list = glob(path)
    groups = [[] for i in range(5)]
    for patient_path in path_list:
        nb = int(patient_path[-3:])
        idx = (nb - 1) // 20
        for file_path in glob(os.path.join(patient_path, 'labeled/*.npz')):
            groups[idx].append(file_path)
    
    train = []
    val = []
    for group in groups:
        group_train, group_val = np.split(group, indices_or_sections=[int(0.8 * len(group))])
        train.append(group_train)
        val.append(group_val)
    train_list = np.concatenate(train)
    val_list = np.concatenate(val)
    return train_list, val_list

def get_patient_files(slice_lists):
    if slice_lists[0].count('\\') < 2:
        samples = []
        for path in slice_lists:
            patient_files = glob(os.path.join(path, 'labeled', '*'))
            samples = samples + patient_files
    else:
        samples = slice_lists
    return samples

def my_sort(path_list, crop_size):
    sort_list = []
    for path in path_list:
        label = np.load(path)['label']
        label[label > 0] = 1
        label = torch.from_numpy(label)
        r = torch.count_nonzero(label) / crop_size**2
        sort_list.append((path, r))
    sort_list = sorted(sort_list, key=lambda x: x[1], reverse=True)
    return [x[0] for x in sort_list]


def get_train_val_list(path):
    path_list = glob(path)
    train = []
    val = []
    for i in range(5):
        array_path_list = np.array(path_list[i*20:(i+1)*20])
        group_train, group_val = np.split(array_path_list, indices_or_sections=[int(0.8 * len(array_path_list))])
        train.append(group_train)
        val.append(group_val)
    train_list = np.concatenate(train)
    val_list = np.concatenate(val)
    return train_list, val_list


def create_3d_acdc_dataset_crop(train_path, val_path, img_size, batch_size, device, val_subset_size, data_augmentation_utils):
    if train_path == val_path:
        train_list, val_list = get_train_val_list(train_path)
    else:
        train_list, _ = get_train_val_list(train_path)
        _, val_list = get_train_val_list(val_path)

    labeled_train_dataset = MyLabeledACDCDataset3DCrop(device=device, img_size=img_size, slice_lists=train_list, data_augmentation_utils=data_augmentation_utils)
    val_dataset = MyLabeledACDCDataset3DCrop(device=device, img_size=img_size, slice_lists=val_list, data_augmentation_utils=None)

    val_random_sampler = RandomSampler(val_dataset, replacement=True, num_samples=1, generator=None)
    val_random_dataloader = DataLoader(val_dataset, sampler=val_random_sampler)

    train_random_sampler = RandomSampler(labeled_train_dataset, replacement=True, num_samples=1, generator=None)
    train_random_dataloader = DataLoader(labeled_train_dataset, sampler=train_random_sampler)

    labeled_train_dataloader = DataLoader(labeled_train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)
    
    if val_subset_size > 0:
        indices = np.random.randint(0, len(val_dataset), size=val_subset_size)
        val_dataset_subset = Subset(val_dataset, indices.tolist())
        val_dataloader_subset = DataLoader(val_dataset_subset, batch_size=1, shuffle=True)
    else:
        val_dataloader_subset = val_dataloader

    train_indices = np.random.randint(0, len(labeled_train_dataset), size=len(val_dataloader_subset))
    train_dataset_subset = Subset(labeled_train_dataset, train_indices.tolist())
    train_dataloader_subset = DataLoader(train_dataset_subset, batch_size=1, shuffle=True)

    return {'labeled_train_dataloader': labeled_train_dataloader,
            'val_dataloader': val_dataloader, 
            'val_random_dataloader': val_random_dataloader,
            'train_random_dataloader': train_random_dataloader,
            'val_dataloader_subset': val_dataloader_subset,
            'train_dataloader_subset': train_dataloader_subset}


def create_3d_acdc_dataset(train_path, val_path, img_size, batch_size, device, val_subset_size, data_augmentation_utils):
    if train_path == val_path:
        train_list, val_list = get_train_val_list(train_path)
    else:
        train_list, _ = get_train_val_list(train_path)
        _, val_list = get_train_val_list(val_path)

    labeled_train_dataset = MyLabeledACDCDataset3D(dimensions=3, device=device, img_size=img_size, slice_lists=train_list, data_augmentation_utils=data_augmentation_utils)
    val_dataset = MyLabeledACDCDataset3D(dimensions=3, device=device, img_size=img_size, slice_lists=val_list, data_augmentation_utils=None)

    val_random_sampler = RandomSampler(val_dataset, replacement=True, num_samples=1, generator=None)
    val_random_dataloader = DataLoader(val_dataset, sampler=val_random_sampler)

    train_random_sampler = RandomSampler(labeled_train_dataset, replacement=True, num_samples=1, generator=None)
    train_random_dataloader = DataLoader(labeled_train_dataset, sampler=train_random_sampler)

    labeled_train_dataloader = DataLoader(labeled_train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)
    
    if val_subset_size > 0:
        indices = np.random.randint(0, len(val_dataset), size=val_subset_size)
        val_dataset_subset = Subset(val_dataset, indices.tolist())
        val_dataloader_subset = DataLoader(val_dataset_subset, batch_size=1, shuffle=True)
    else:
        val_dataloader_subset = val_dataloader

    train_indices = np.random.randint(0, len(labeled_train_dataset), size=len(val_dataloader_subset))
    train_dataset_subset = Subset(labeled_train_dataset, train_indices.tolist())
    train_dataloader_subset = DataLoader(train_dataset_subset, batch_size=1, shuffle=True)

    return {'labeled_train_dataloader': labeled_train_dataloader,
            'val_dataloader': val_dataloader, 
            'val_random_dataloader': val_random_dataloader,
            'train_random_dataloader': train_random_dataloader,
            'val_dataloader_subset': val_dataloader_subset,
            'train_dataloader_subset': train_dataloader_subset}


def create_2d_acdc_dataset(train_path, 
                            val_path, 
                            img_size, 
                            batch_size, 
                            device, 
                            val_subset_size, 
                            data_augmentation_utils, 
                            directional_field, 
                            nb_epochs, 
                            writer, 
                            sampler_param,
                            learn_transforms,
                            binary,
                            use_cropped_images,
                            apply_clahe):
    keys = ['small', 'middle', 'big']
    if any(x in train_path for x in keys):
        if train_path == val_path:
            train_list, val_list = get_size_train_val(train_path)
        else:
            train_list, _ = get_size_train_val(train_path)
            _, val_list = get_size_train_val(val_path)
    else:
        if train_path == val_path:
            train_list, val_list = get_train_val_list(train_path)
            train_list = get_patient_files(train_list)
            #train_list = my_sort(train_list, img_size)
        else:
            train_list, _ = get_train_val_list(train_path)
            _, val_list = get_train_val_list(val_path)

    if use_cropped_images:
        labeled_train_dataset = MyLabeledACDCDataset(device=device, img_size=img_size, slice_lists=train_list, data_augmentation_utils=data_augmentation_utils, directional_field=directional_field, apply_clahe=apply_clahe)
        val_dataset = MyLabeledTestACDCDataset3D(device=device, img_size=img_size, slice_lists=val_list, directional_field=directional_field, apply_clahe=apply_clahe)
    else:
        labeled_train_dataset = MyLabeledACDCDataset2DCrop(device=device, img_size=img_size, slice_lists=train_list, data_augmentation_utils=data_augmentation_utils, target_ratio=-1, binary=binary, learn_transforms=learn_transforms, directional_field=directional_field, apply_clahe=apply_clahe)
        val_dataset = MyLabeledTestACDCDataset3DCrop(device=device, img_size=img_size, slice_lists=val_list, target_ratio=-1, binary=binary, learn_transforms=learn_transforms, directional_field=directional_field, apply_clahe=apply_clahe)

    val_random_sampler = RandomSampler(val_dataset, replacement=True, num_samples=1, generator=None)
    val_random_dataloader = DataLoader(val_dataset, sampler=val_random_sampler)

    train_random_sampler = RandomSampler(labeled_train_dataset, replacement=True, num_samples=1, generator=None)
    train_random_dataloader = DataLoader(labeled_train_dataset, sampler=train_random_sampler)

    #labeled_train_dataloader = DataLoader(labeled_train_dataset, batch_size=batch_size, collate_fn=my_collate, shuffle=True)
    #custom_sampler = CustomSamplerUniform(num_samples=len(labeled_train_dataset), nb_epochs=nb_epochs, sampler_param=sampler_param, replacement=True, writer=writer)
    #custom_sampler = CustomSamplerNormal(num_samples=len(labeled_train_dataset), nb_epochs=nb_epochs, replacement=True, writer=writer, sampler_param=sampler_param)
    labeled_train_dataloader = DataLoader(labeled_train_dataset, batch_size=batch_size, collate_fn=my_collate, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, collate_fn=my_collate, shuffle=True)
    
    if val_subset_size > 0:
        indices = np.random.randint(0, len(val_dataset), size=val_subset_size)
        val_dataset_subset = Subset(val_dataset, indices.tolist())
        val_dataloader_subset = DataLoader(val_dataset_subset, batch_size=1, shuffle=True)
    else:
        val_dataloader_subset = val_dataloader
    
    #train_indices = np.random.randint(0, len(labeled_train_dataset), size=len(val_dataloader_subset))
    #train_dataset_subset = Subset(labeled_train_dataset, train_indices.tolist())
    overfitting_dataloader = DataLoader(labeled_train_dataset, batch_size=1, collate_fn=my_collate, shuffle=True)

    return {'labeled_train_dataloader': labeled_train_dataloader,
            'val_dataloader': val_dataloader, 
            'val_random_dataloader': val_random_dataloader,
            'train_random_dataloader': train_random_dataloader,
            'val_dataloader_subset': val_dataloader_subset,
            'overfitting_dataloader': overfitting_dataloader}
    

def create_2d_acdc_dataset_crop(train_path, val_path, img_size, batch_size, device, val_subset_size, data_augmentation_utils, target_ratio, binary, learn_transforms):
    if train_path == val_path:
        train_list, val_list = get_train_val_list(train_path)
    else:
        train_list, _ = get_train_val_list(train_path)
        _, val_list = get_train_val_list(val_path)

    labeled_train_dataset = MyLabeledACDCDataset2DCrop(device=device, img_size=img_size, slice_lists=train_list, data_augmentation_utils=data_augmentation_utils, target_ratio=target_ratio, binary=binary, learn_transforms=learn_transforms)
    val_dataset = MyLabeledTestACDCDataset3DCrop(device=device, img_size=img_size, slice_lists=val_list, target_ratio=target_ratio, binary=binary, learn_transforms=learn_transforms)

    val_random_sampler = RandomSampler(val_dataset, replacement=True, num_samples=1, generator=None)
    val_random_dataloader = DataLoader(val_dataset, sampler=val_random_sampler)

    train_random_sampler = RandomSampler(labeled_train_dataset, replacement=True, num_samples=1, generator=None)
    train_random_dataloader = DataLoader(labeled_train_dataset, sampler=train_random_sampler)

    #labeled_train_dataloader = DataLoader(labeled_train_dataset, collate_fn=my_collate, batch_size=batch_size, shuffle=True)
    labeled_train_dataloader = DataLoader(labeled_train_dataset, collate_fn=my_collate, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, collate_fn=my_collate, batch_size=1, shuffle=True)
    
    if val_subset_size > 0:
        indices = np.random.randint(0, len(val_dataset), size=val_subset_size)
        val_dataset_subset = Subset(val_dataset, indices.tolist())
        val_dataloader_subset = DataLoader(val_dataset_subset, batch_size=1, shuffle=True)
    else:
        val_dataloader_subset = val_dataloader
    
    train_indices = np.random.randint(0, len(labeled_train_dataset), size=len(val_dataloader_subset))
    train_dataset_subset = Subset(labeled_train_dataset, train_indices.tolist())
    train_dataloader_subset = DataLoader(train_dataset_subset, batch_size=1, shuffle=True)

    return {'labeled_train_dataloader': labeled_train_dataloader,
            'val_dataloader': val_dataloader, 
            'val_random_dataloader': val_random_dataloader,
            'train_random_dataloader': train_random_dataloader,
            'val_dataloader_subset': val_dataloader_subset,
            'train_dataloader_subset': train_dataloader_subset}


def create_acdc_dataset_logits_2d(train_path, val_path, img_size, batch_size, device, val_subset_size, data_augmentation_utils):
    keys = ['small', 'middle', 'big']
    if any(x in train_path for x in keys):
        if train_path == val_path:
            train_list, val_list = get_size_train_val(train_path)
        else:
            train_list, _ = get_size_train_val(train_path)
            _, val_list = get_size_train_val(val_path)
    else:
        if train_path == val_path:
            train_list, val_list = get_train_val_list(train_path)
        else:
            train_list, _ = get_train_val_list(train_path)
            _, val_list = get_train_val_list(val_path)

    labeled_train_dataset = MyLabeledACDCDatasetLogits2D(device=device, img_size=img_size, slice_lists=train_list, data_augmentation_utils=data_augmentation_utils)
    val_dataset = MyLabeledACDCDatasetLogits2D(device=device, img_size=img_size, slice_lists=val_list, data_augmentation_utils=None)

    val_random_sampler = RandomSampler(val_dataset, replacement=True, num_samples=1, generator=None)
    val_random_dataloader = DataLoader(val_dataset, sampler=val_random_sampler)

    train_random_sampler = RandomSampler(labeled_train_dataset, replacement=True, num_samples=1, generator=None)
    train_random_dataloader = DataLoader(labeled_train_dataset, sampler=train_random_sampler)

    labeled_train_dataloader = DataLoader(labeled_train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)
    
    if val_subset_size > 0:
        indices = np.random.randint(0, len(val_dataset), size=val_subset_size)
        val_dataset_subset = Subset(val_dataset, indices.tolist())
        val_dataloader_subset = DataLoader(val_dataset_subset, batch_size=1, shuffle=True)
    else:
        val_dataloader_subset = val_dataloader
    
    train_indices = np.random.randint(0, len(labeled_train_dataset), size=len(val_dataloader_subset))
    train_dataset_subset = Subset(labeled_train_dataset, train_indices.tolist())
    train_dataloader_subset = DataLoader(train_dataset_subset, batch_size=1, shuffle=True)

    return {'labeled_train_dataloader': labeled_train_dataloader,
            'val_dataloader': val_dataloader, 
            'val_random_dataloader': val_random_dataloader,
            'train_random_dataloader': train_random_dataloader,
            'val_dataloader_subset': val_dataloader_subset,
            'train_dataloader_subset': train_dataloader_subset}

def create_acdc_dataset_logits_3d(train_path, val_path, batch_size, device, val_subset_size, data_augmentation_utils):
    keys = ['small', 'middle', 'big']
    if any(x in train_path for x in keys):
        if train_path == val_path:
            train_list, val_list = get_size_train_val(train_path)
        else:
            train_list, _ = get_size_train_val(train_path)
            _, val_list = get_size_train_val(val_path)
    else:
        if train_path == val_path:
            train_list, val_list = get_train_val_list(train_path)
        else:
            train_list, _ = get_train_val_list(train_path)
            _, val_list = get_train_val_list(val_path)

    labeled_train_dataset = MyLabeledACDCDatasetLogits3D(device=device, slice_lists=train_list, data_augmentation_utils=data_augmentation_utils)
    val_dataset = MyLabeledACDCDatasetLogits3D(device=device, slice_lists=val_list, data_augmentation_utils=None)

    val_random_sampler = RandomSampler(val_dataset, replacement=True, num_samples=1, generator=None)
    val_random_dataloader = DataLoader(val_dataset, sampler=val_random_sampler)

    train_random_sampler = RandomSampler(labeled_train_dataset, replacement=True, num_samples=1, generator=None)
    train_random_dataloader = DataLoader(labeled_train_dataset, sampler=train_random_sampler)

    labeled_train_dataloader = DataLoader(labeled_train_dataset, collate_fn=my_collate, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)
    
    if val_subset_size > 0:
        indices = np.random.randint(0, len(val_dataset), size=val_subset_size)
        val_dataset_subset = Subset(val_dataset, indices.tolist())
        val_dataloader_subset = DataLoader(val_dataset_subset, batch_size=1, shuffle=True)
    else:
        val_dataloader_subset = val_dataloader
    
    train_indices = np.random.randint(0, len(labeled_train_dataset), size=len(val_dataloader_subset))
    train_dataset_subset = Subset(labeled_train_dataset, train_indices.tolist())
    train_dataloader_subset = DataLoader(train_dataset_subset, batch_size=1, shuffle=True)

    return {'labeled_train_dataloader': labeled_train_dataloader,
            'val_dataloader': val_dataloader, 
            'val_random_dataloader': val_random_dataloader,
            'train_random_dataloader': train_random_dataloader,
            'val_dataloader_subset': val_dataloader_subset,
            'train_dataloader_subset': train_dataloader_subset}

def create_acdc_autoencoder_dataset(path, img_size, batch_size, device, data_augmentation_utils=None):
    path_list = glob(path)
    train = []
    val = []
    for i in range(5):
        array_path_list = np.array(path_list[i*20:(i+1)*20])
        group_train, group_val = np.split(array_path_list, indices_or_sections=[int(0.8 * len(array_path_list))])
        train.append(group_train)
        val.append(group_val)
    train_list = np.concatenate(train)
    val_list = np.concatenate(val)

    training_dataset = AutoencoderACDCDataset(train_list, device, img_size, data_augmentation_utils)
    val_dataset = AutoencoderACDCDataset(val_list, device, img_size, data_augmentation_utils=None)

    val_random_sampler = RandomSampler(val_dataset, replacement=True, num_samples=1, generator=None)

    val_random_dataloader = DataLoader(val_dataset, sampler=val_random_sampler)
    labeled_train_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    return {'labeled_train_dataloader': labeled_train_dataloader,
            'val_dataloader': val_dataloader, 
            'val_random_dataloader': val_random_dataloader, 
            'test_dataloader': None,
            'val_dataloader_subset': val_dataloader}

#def create_ACDC_test(path, crop_size, device):
#    labeled_path_list = glob(os.path.join(path, '*'))
#    unlabeled_path_list = glob(os.path.join(path, '*'))
#    labeled_test_dataset = MyLabeledTestACDCDataset(labeled_path_list, device, crop_size)
#    unlabeled_test_dataset = MyUnlabeledTestACDCDataset(unlabeled_path_list, device, crop_size)
#    labeled_test_dataloader = DataLoader(labeled_test_dataset, batch_size=1)
#    unlabeled_test_dataloader = DataLoader(unlabeled_test_dataset, batch_size=1)
#    return labeled_test_dataloader, unlabeled_test_dataloader

def create_ACDC_test(path, crop_size, device):
    labeled_path_list = glob(os.path.join(path, '*/labeled/*.npz'))
    unlabeled_path_list = glob(os.path.join(path, '*/unlabeled/*.npz'))
    labeled_test_dataset = MyLabeledTestACDCDataset(labeled_path_list, device, crop_size)
    unlabeled_test_dataset = MyUnlabeledTestACDCDataset(unlabeled_path_list, device, crop_size)
    labeled_test_dataloader = DataLoader(labeled_test_dataset, batch_size=1)
    unlabeled_test_dataloader = DataLoader(unlabeled_test_dataset, batch_size=1)
    return labeled_test_dataloader, unlabeled_test_dataloader

def create_ACDC_test_3d(path, crop_size, device):
    labeled_path_list = glob(path)
    labeled_test_dataset = MyLabeledTestACDCDataset3D(labeled_path_list, device, crop_size)
    labeled_test_dataloader = DataLoader(labeled_test_dataset, batch_size=1)
    return labeled_test_dataloader

def create_ACDC_test_3d_crop(path, crop_size, device, target_ratio, binary, learn_transforms, directional_field):
    labeled_path_list = glob(path)
    labeled_test_dataset = MyLabeledTestACDCDataset3DCrop(labeled_path_list, device, crop_size, target_ratio, binary, learn_transforms, directional_field)
    labeled_test_dataloader = DataLoader(labeled_test_dataset, batch_size=1)
    return labeled_test_dataloader
    
class MyUnlabeledACDCDataset(Dataset):
    def __init__(self, slice_lists, device, data_augmentation_utils=None):

        samples = []
        for path in slice_lists:
            patient_files = glob(os.path.join(path, 'unlabeled', '*'))
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
        image = process_unlabeled_image(path, rotate=True, mean=3258.0759, std=3507.0581, augmentation=augmentation)
        
        return {"x": image.to(self.device), "mask": mask.to(self.device)}


class MyLabeledTestACDCDataset(Dataset):
    def __init__(self, slice_lists, device, img_size):

        samples = []
        for path in slice_lists:
            samples.append(path)

        self.device = device
        self.samples = samples
        self.img_size = img_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        
        path = self.samples[idx]
        image, label, dist_map_tensor, parameters = process_labeled_image_2d(path=path, 
                                                                        img_size=self.img_size,
                                                                        augmentations=None)
        
        return {'x': image.to(self.device), 'y': label.to(self.device), 'path': path}


class MyLabeledTestACDCDataset3DCrop(Dataset):
    def __init__(self, slice_lists, device, img_size, target_ratio, binary, learn_transforms, directional_field, apply_clahe):

        self.samples = get_cases(slice_lists)

        self.device = device
        self.img_size = img_size
        self.target_ratio = target_ratio
        self.binary = binary
        self.learn_transforms = learn_transforms
        self.directional_field = directional_field
        self.apply_clahe = apply_clahe

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        
        path_list = self.samples[idx]
        out = process_labeled_image_3d_crop(path_list=path_list, 
                                            img_size=self.img_size, 
                                            augmentations=None, 
                                            target_ratio=self.target_ratio, 
                                            binary=self.binary, 
                                            learn_transforms=self.learn_transforms,
                                            directional_field=self.directional_field,
                                            apply_clahe=self.apply_clahe)

        out['path_list'] = path_list

        assert len(path_list) == out['x'].shape[1]
        
        return out

class MyLabeledACDCDataset(Dataset):
    def __init__(self, slice_lists, img_size, device, data_augmentation_utils, directional_field, apply_clahe):

        self.samples = slice_lists
        self.data_augmentation_utils = data_augmentation_utils
        self.img_size = img_size
        self.device = device
        self.directional_field = directional_field
        self.apply_clahe = apply_clahe
            

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        augmentations = None
        if self.data_augmentation_utils is not None:
            probabilities = np.array([x[1] for x in self.data_augmentation_utils])
            r = np.random.rand(len(self.data_augmentation_utils))
            augmentations = self.data_augmentation_utils[np.where(r < probabilities)]
            augmentations = sorted(augmentations.tolist(), key=lambda x: x[2])
            augmentations = [x[0] for x in augmentations]
        
        path = self.samples[idx]
        out = process_2d_image(path=path, img_size=self.img_size, augmentations=augmentations, directional_field=self.directional_field, apply_clahe=self.apply_clahe)
        
        return out


class MyLabeledACDCDataset2DCrop(Dataset):
    def __init__(self, slice_lists, img_size, device, data_augmentation_utils, target_ratio, binary, learn_transforms, directional_field, apply_clahe):

        self.samples = slice_lists
        self.data_augmentation_utils = data_augmentation_utils
        self.img_size = img_size
        self.device = device
        self.target_ratio = target_ratio
        self.binary = binary
        self.learn_transforms = learn_transforms
        self.directional_field = directional_field
        self.apply_clahe = apply_clahe
            

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        augmentations = None
        if self.data_augmentation_utils is not None:
            probabilities = np.array([x[1] for x in self.data_augmentation_utils])
            r = np.random.rand(len(self.data_augmentation_utils))
            augmentations = self.data_augmentation_utils[np.where(r < probabilities)]
            augmentations = sorted(augmentations.tolist(), key=lambda x: x[2])
            augmentations = [x[0] for x in augmentations]
        
        path = self.samples[idx]
        out = process_2d_image_crop(path=path, 
                                    img_size=self.img_size, 
                                    augmentations=augmentations, 
                                    target_ratio=self.target_ratio, 
                                    binary=self.binary, 
                                    learn_transforms=self.learn_transforms,
                                    directional_field=self.directional_field,
                                    apply_clahe=self.apply_clahe)
        
        return out


class MyLabeledACDCDatasetLogits2D(Dataset):
    def __init__(self, slice_lists, img_size, device, data_augmentation_utils):

        if slice_lists[0].count('\\') < 2:
            samples = []
            for path in slice_lists:
                patient_files = glob(os.path.join(path, 'labeled', '*'))
                samples = samples + patient_files
        else:
            samples = slice_lists

        self.samples = samples
        self.data_augmentation_utils = data_augmentation_utils
        self.img_size = img_size
        self.device = device
            

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        augmentations = None
        if self.data_augmentation_utils is not None:
            probabilities = np.array([x[1] for x in self.data_augmentation_utils])
            r = np.random.rand(len(self.data_augmentation_utils))
            augmentations = self.data_augmentation_utils[np.where(r < probabilities)]
            augmentations = sorted(augmentations.tolist(), key=lambda x: x[2])
            augmentations = [x[0] for x in augmentations]
        
        path = self.samples[idx]
        input_2d, input_3d, label, angle, unrotated_label, image_in = process_labeled_image_2d_logits(path=path,
                                                                                        img_size=self.img_size,
                                                                                        augmentations=augmentations)


        out = {"input_2d": input_2d.to(self.device), "input_3d": input_3d.to(self.device), "y": label.to(self.device), "angle": angle.to(self.device), "unrotated_label": unrotated_label.to(self.device)}
        
        return out


class MyLabeledACDCDatasetLogits3D(Dataset):
    def __init__(self, slice_lists, device, data_augmentation_utils):

        if slice_lists[0].count('\\') < 2:
            samples = []
            for path in slice_lists:
                patient_files = glob(os.path.join(path, 'labeled', '*'))
                samples = samples + patient_files
        else:
            samples = slice_lists

        self.samples = samples
        self.data_augmentation_utils = data_augmentation_utils
        self.device = device
            

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        augmentations = None
        if self.data_augmentation_utils is not None:
            probabilities = np.array([x[1] for x in self.data_augmentation_utils])
            r = np.random.rand(len(self.data_augmentation_utils))
            augmentations = self.data_augmentation_utils[np.where(r < probabilities)]
            augmentations = sorted(augmentations.tolist(), key=lambda x: x[2])
            augmentations = [x[0] for x in augmentations]
        
        path = self.samples[idx]
        input_volume, label, angle, translation_parameters, padding, original_label, original_image = process_labeled_image_3d_logits(path=path,
                                                                                                                                    augmentations=augmentations)

        out = {"x": input_volume.to(self.device), 
                "y": label.to(self.device), 
                "angle": angle.to(self.device), 
                "translation_parameters": translation_parameters.to(self.device), 
                "padding": padding.to(self.device),
                "original_label": original_label.to(self.device), 
                "original_image": original_image.to(self.device)}
        
        return out


class MyLabeledTestACDCDataset3D(Dataset):
    def __init__(self, slice_lists, img_size, device, directional_field, apply_clahe):

        self.samples = get_cases(slice_lists)
        self.img_size = img_size
        self.device = device
        self.directional_field = directional_field
        self.apply_clahe = apply_clahe
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        
        path_list = self.samples[idx]
        out = process_labeled_image_3d(path_list=path_list, img_size=self.img_size, augmentations=None, directional_field=self.directional_field, apply_clahe=self.apply_clahe)

        #print(path_list)
        #for i in range(10):
        #    fig, ax = plt.subplots(1, 2)
        #    ax[0].imshow(torch.argmax(label_volume, dim=0)[i], cmap='gray')
        #    ax[1].imshow(image_volume[0, i], cmap='gray')
        #    plt.show()

        return out


class MyLabeledACDCDataset3DCrop(Dataset):
    def __init__(self, slice_lists, img_size, device, data_augmentation_utils, dimensions, target_ratio):

        self.samples = get_cases(slice_lists)
        self.data_augmentation_utils = data_augmentation_utils
        self.img_size = img_size
        self.device = device
        self.dimensions = dimensions
        self.target_ratio = target_ratio

        if dimensions == 3:
            self.r_obj = ResizeWithPadOrCrop((12, img_size, img_size))
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        augmentations = None
        if self.data_augmentation_utils is not None:
            probabilities = np.array([x[1] for x in self.data_augmentation_utils])
            r = np.random.rand(len(self.data_augmentation_utils))
            augmentations = self.data_augmentation_utils[np.where(r < probabilities)]
            augmentations = sorted(augmentations.tolist(), key=lambda x: x[2])
            augmentations = [x[0] for x in augmentations]
        
        path_list = self.samples[idx]
        out = process_labeled_image_3d_crop(path_list=path_list, img_size=self.img_size, augmentations=augmentations, target_ratio=self.target_ratio)

        if self.dimensions == 3:
            out['x'] = self.r_obj(out['x'])

            label_volume = torch.argmax(out['y'], dim=0, keepdim=True)
            label_volume = self.r_obj(label_volume).squeeze()
            out['y'] = torch.nn.functional.one_hot(label_volume, num_classes=4).permute(3, 0, 1, 2).float()

        #print(path_list)
        #for i in range(10):
        #    fig, ax = plt.subplots(1, 2)
        #    ax[0].imshow(torch.argmax(label_volume, dim=0)[i], cmap='gray')
        #    ax[1].imshow(image_volume[0, i], cmap='gray')
        #    plt.show()

        return out

class AutoencoderACDCDataset(Dataset):
    def __init__(self, slice_lists, device, img_size, data_augmentation_utils=None):

        samples = []
        for path in slice_lists:
            patient_files = glob(os.path.join(path, 'labeled', '*'))
            samples = samples + patient_files

        self.device = device
        self.samples = samples
        self.data_augmentation_utils = data_augmentation_utils
        self.img_size = img_size

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
        image, label, dist_map_tensor, metadata = process_labeled_image_2d(path=path, 
                                                                        rotate=True,
                                                                        mean=3258.0759,
                                                                        std=3507.0581,
                                                                        img_size=self.img_size, 
                                                                        use_spatial_transformer=False, 
                                                                        device=self.device, 
                                                                        augmentation=augmentation)
        
        label = torch.argmax(label, dim=0, keepdim=True).float()
        label = normalize_0_1_autoencoder(label)


        #print(metadata['tx'])
        #print(metadata['ty'])
        #print(metadata['scale'])
        #print(metadata['angle'])
        #fig, ax = plt.subplots(1, 2, figsize=(20, 8))
        #ax[0].imshow(torch.argmax(label, dim=0).cpu(), cmap='gray')
        #ax[1].imshow(image.cpu()[0], cmap='gray')
        #plt.show()
        
        return label.to(self.device)