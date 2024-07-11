#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import math
import torch
import torch.nn as nn
from monai.transforms import NormalizeIntensity
import cv2 as cv
from collections import OrderedDict
import numpy as np
import sys
from numpy.random import RandomState
from multiprocessing import Pool
from math import ceil, floor
from scipy.signal import savgol_filter
from numpy.lib.stride_tricks import sliding_window_view
import monai.transforms as T
from monai.transforms import Compose
from monai.utils.misc import set_determinism
import random
from skimage.measure import perimeter
import nibabel as nib

from batchgenerators.dataloading.data_loader import SlimDataLoaderBase

from nnunet.configuration import default_num_threads
from nnunet.paths import preprocessing_output_dir
from batchgenerators.utilities.file_and_folder_operations import *
import matplotlib
import matplotlib.pyplot as plt
from copy import copy
from monai.transforms import Lambda

class SinusoidalPositionEmbeddings(nn.Module):
    """Positional embeddings"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
    

def get_slice_nb_one_vs_all(random_slice):
    if random_slice >= 1:
        middle_slice = random_slice - 1
    else:
        middle_slice = 1
    return random_slice, middle_slice

def select_idx(nb_slices, random_slice, keys):
    random_percent = random_slice / nb_slices
    min_d = 100
    for i in range(len(keys)):
        d = abs(keys[i] - random_percent)
        if d < min_d:
            min_d = d
            idx = i
    if get_idx(keys[idx] * nb_slices) >= nb_slices:
        idx = idx - 1
    return keys[idx]

def get_idx(x):
    if x % 1 > 0.5:
        return int(ceil(x))
    else:
        return int(floor(x))

def get_case_identifiers(folder):
    case_identifiers = [i[:-4] for i in os.listdir(folder) if i.endswith("npz") and (i.find("segFromPrevStage") == -1)]
    return case_identifiers


def get_case_identifiers_from_raw_folder(folder):
    case_identifiers = np.unique(
        [i[:-12] for i in os.listdir(folder) if i.endswith(".nii.gz") and (i.find("segFromPrevStage") == -1)])
    return case_identifiers


def convert_to_npy(args):
    if not isinstance(args, tuple):
        key = "data"
        npz_file = args
    else:
        npz_file, key = args
    if not isfile(npz_file[:-3] + "npy"):
        a = np.load(npz_file)[key]
        np.save(npz_file[:-3] + "npy", a)


def save_as_npz(args):
    if not isinstance(args, tuple):
        key = "data"
        npy_file = args
    else:
        npy_file, key = args
    d = np.load(npy_file)
    np.savez_compressed(npy_file[:-3] + "npz", **{key: d})


def unpack_dataset(folder, threads=default_num_threads, key="data"):
    """
    unpacks all npz files in a folder to npy (whatever you want to have unpacked must be saved unter key)
    :param folder:
    :param threads:
    :param key:
    :return:
    """
    p = Pool(threads)
    npz_files = subfiles(folder, True, None, ".npz", True)
    p.map(convert_to_npy, zip(npz_files, [key] * len(npz_files)))
    p.close()
    p.join()


def pack_dataset(folder, threads=default_num_threads, key="data"):
    p = Pool(threads)
    npy_files = subfiles(folder, True, None, ".npy", True)
    p.map(save_as_npz, zip(npy_files, [key] * len(npy_files)))
    p.close()
    p.join()


def delete_npy(folder):
    case_identifiers = get_case_identifiers(folder)
    npy_files = [join(folder, i + ".npy") for i in case_identifiers]
    npy_files = [i for i in npy_files if isfile(i)]
    for n in npy_files:
        os.remove(n)


def load_dataset(folder, num_cases_properties_loading_threshold=1000):
    # we don't load the actual data but instead return the filename to the np file.
    print('loading labeled dataset')
    case_identifiers = get_case_identifiers(folder)
    case_identifiers = [x for x in case_identifiers if '_u' not in x]
    case_identifiers.sort()
    dataset = OrderedDict()
    for c in case_identifiers:
        dataset[c] = OrderedDict()
        dataset[c]['data_file'] = join(folder, "%s.npz" % c)

        # dataset[c]['properties'] = load_pickle(join(folder, "%s.pkl" % c))
        dataset[c]['properties_file'] = join(folder, "%s.pkl" % c)

        if dataset[c].get('seg_from_prev_stage_file') is not None:
            dataset[c]['seg_from_prev_stage_file'] = join(folder, "%s_segs.npz" % c)

    if len(case_identifiers) <= num_cases_properties_loading_threshold:
        print('loading all case properties')
        for i in dataset.keys():
            dataset[i]['properties'] = load_pickle(dataset[i]['properties_file'])

    return dataset


def load_unlabeled_dataset(folder, num_cases_properties_loading_threshold=1000):
    # we don't load the actual data but instead return the filename to the np file.
    print('loading unlabeled dataset')
    case_identifiers = get_case_identifiers(folder)
    case_identifiers = [x for x in case_identifiers if '_u' in x]
    case_identifiers.sort()
    dataset = OrderedDict()
    for c in case_identifiers:
        dataset[c] = OrderedDict()
        dataset[c]['data_file'] = join(folder, "%s.npz" % c)

        # dataset[c]['properties'] = load_pickle(join(folder, "%s.pkl" % c))
        dataset[c]['properties_file'] = join(folder, "%s.pkl" % c)

        if dataset[c].get('seg_from_prev_stage_file') is not None:
            dataset[c]['seg_from_prev_stage_file'] = join(folder, "%s_segs.npz" % c)

    if len(case_identifiers) <= num_cases_properties_loading_threshold:
        print('loading all case properties')
        for i in dataset.keys():
            dataset[i]['properties'] = load_pickle(dataset[i]['properties_file'])

    return dataset


def crop_2D_image_force_fg(img, crop_size, valid_voxels):
    """
    img must be [c, x, y]
    img[-1] must be the segmentation with segmentation>0 being foreground
    :param img:
    :param crop_size:
    :param valid_voxels: voxels belonging to the selected class
    :return:
    """
    assert len(valid_voxels.shape) == 2

    if type(crop_size) not in (tuple, list):
        crop_size = [crop_size] * (len(img.shape) - 1)
    else:
        assert len(crop_size) == (len(
            img.shape) - 1), "If you provide a list/tuple as center crop make sure it has the same len as your data has dims (3d)"

    # we need to find the center coords that we can crop to without exceeding the image border
    lb_x = crop_size[0] // 2
    ub_x = img.shape[1] - crop_size[0] // 2 - crop_size[0] % 2
    lb_y = crop_size[1] // 2
    ub_y = img.shape[2] - crop_size[1] // 2 - crop_size[1] % 2

    if len(valid_voxels) == 0:
        selected_center_voxel = (np.random.random_integers(lb_x, ub_x),
                                 np.random.random_integers(lb_y, ub_y))
    else:
        selected_center_voxel = valid_voxels[np.random.choice(valid_voxels.shape[1]), :]

    selected_center_voxel = np.array(selected_center_voxel)
    for i in range(2):
        selected_center_voxel[i] = max(crop_size[i] // 2, selected_center_voxel[i])
        selected_center_voxel[i] = min(img.shape[i + 1] - crop_size[i] // 2 - crop_size[i] % 2,
                                       selected_center_voxel[i])

    result = img[:, (selected_center_voxel[0] - crop_size[0] // 2):(
            selected_center_voxel[0] + crop_size[0] // 2 + crop_size[0] % 2),
             (selected_center_voxel[1] - crop_size[1] // 2):(
                     selected_center_voxel[1] + crop_size[1] // 2 + crop_size[1] % 2)]
    return result


class DataLoader3D(SlimDataLoaderBase):
    def __init__(self, data, patch_size, final_patch_size, batch_size, has_prev_stage=False,
                 oversample_foreground_percent=0.0, memmap_mode="r", pad_mode="edge", pad_kwargs_data=None,
                 pad_sides=None):
        """
        This is the basic data loader for 3D networks. It uses preprocessed data as produced by my (Fabian) preprocessing.
        You can load the data with load_dataset(folder) where folder is the folder where the npz files are located. If there
        are only npz files present in that folder, the data loader will unpack them on the fly. This may take a while
        and increase CPU usage. Therefore, I advise you to call unpack_dataset(folder) first, which will unpack all npz
        to npy. Don't forget to call delete_npy(folder) after you are done with training?
        Why all the hassle? Well the decathlon dataset is huge. Using npy for everything will consume >1 TB and that is uncool
        given that I (Fabian) will have to store that permanently on /datasets and my local computer. With this strategy all
        data is stored in a compressed format (factor 10 smaller) and only unpacked when needed.
        :param data: get this with load_dataset(folder, stage=0). Plug the return value in here and you are g2g (good to go)
        :param patch_size: what patch size will this data loader return? it is common practice to first load larger
        patches so that a central crop after data augmentation can be done to reduce border artifacts. If unsure, use
        get_patch_size() from data_augmentation.default_data_augmentation
        :param final_patch_size: what will the patch finally be cropped to (after data augmentation)? this is the patch
        size that goes into your network. We need this here because we will pad patients in here so that patches at the
        border of patients are sampled properly
        :param batch_size:
        :param num_batches: how many batches will the data loader produce before stopping? None=endless
        :param seed:
        :param stage: ignore this (Fabian only)
        :param random: Sample keys randomly; CAREFUL! non-random sampling requires batch_size=1, otherwise you will iterate batch_size times over the dataset
        :param oversample_foreground: half the batch will be forced to contain at least some foreground (equal prob for each of the foreground classes)
        """
        super(DataLoader3D, self).__init__(data, batch_size, None)
        if pad_kwargs_data is None:
            pad_kwargs_data = OrderedDict()
        self.pad_kwargs_data = pad_kwargs_data
        self.pad_mode = pad_mode
        self.oversample_foreground_percent = oversample_foreground_percent
        self.final_patch_size = final_patch_size
        self.has_prev_stage = has_prev_stage
        self.patch_size = patch_size
        self.list_of_keys = list(self._data.keys())
        # need_to_pad denotes by how much we need to pad the data so that if we sample a patch of size final_patch_size
        # (which is what the network will get) these patches will also cover the border of the patients
        self.need_to_pad = (np.array(patch_size) - np.array(final_patch_size)).astype(int)
        if pad_sides is not None:
            if not isinstance(pad_sides, np.ndarray):
                pad_sides = np.array(pad_sides)
            self.need_to_pad += pad_sides
        self.memmap_mode = memmap_mode
        self.num_channels = None
        self.pad_sides = pad_sides
        self.data_shape, self.seg_shape = self.determine_shapes()

    def get_do_oversample(self, batch_idx):
        return not batch_idx < round(self.batch_size * (1 - self.oversample_foreground_percent))

    def determine_shapes(self):
        if self.has_prev_stage:
            num_seg = 2
        else:
            num_seg = 1

        k = list(self._data.keys())[0]
        if isfile(self._data[k]['data_file'][:-4] + ".npy"):
            case_all_data = np.load(self._data[k]['data_file'][:-4] + ".npy", self.memmap_mode)
        else:
            case_all_data = np.load(self._data[k]['data_file'])['data']
        num_color_channels = case_all_data.shape[0] - 1
        data_shape = (self.batch_size, num_color_channels, *self.patch_size)
        seg_shape = (self.batch_size, num_seg, *self.patch_size)
        return data_shape, seg_shape

    def generate_train_batch(self):
        selected_keys = np.random.choice(self.list_of_keys, self.batch_size, True, None)
        data = np.zeros(self.data_shape, dtype=np.float32)
        seg = np.zeros(self.seg_shape, dtype=np.float32)
        case_properties = []
        for j, i in enumerate(selected_keys):
            # oversampling foreground will improve stability of model training, especially if many patches are empty
            # (Lung for example)
            if self.get_do_oversample(j):
                force_fg = True
            else:
                force_fg = False

            if 'properties' in self._data[i].keys():
                properties = self._data[i]['properties']
            else:
                properties = load_pickle(self._data[i]['properties_file'])
            case_properties.append(properties)

            # cases are stored as npz, but we require unpack_dataset to be run. This will decompress them into npy
            # which is much faster to access
            if isfile(self._data[i]['data_file'][:-4] + ".npy"):
                case_all_data = np.load(self._data[i]['data_file'][:-4] + ".npy", self.memmap_mode)
            else:
                case_all_data = np.load(self._data[i]['data_file'])['data']

            # If we are doing the cascade then we will also need to load the segmentation of the previous stage and
            # concatenate it. Here it will be concatenates to the segmentation because the augmentations need to be
            # applied to it in segmentation mode. Later in the data augmentation we move it from the segmentations to
            # the last channel of the data
            if self.has_prev_stage:
                if isfile(self._data[i]['seg_from_prev_stage_file'][:-4] + ".npy"):
                    segs_from_previous_stage = np.load(self._data[i]['seg_from_prev_stage_file'][:-4] + ".npy",
                                                       mmap_mode=self.memmap_mode)[None]
                else:
                    segs_from_previous_stage = np.load(self._data[i]['seg_from_prev_stage_file'])['data'][None]
                # we theoretically support several possible previsous segmentations from which only one is sampled. But
                # in practice this feature was never used so it's always only one segmentation
                seg_key = np.random.choice(segs_from_previous_stage.shape[0])
                seg_from_previous_stage = segs_from_previous_stage[seg_key:seg_key + 1]
                assert all([i == j for i, j in zip(seg_from_previous_stage.shape[1:], case_all_data.shape[1:])]), \
                    "seg_from_previous_stage does not match the shape of case_all_data: %s vs %s" % \
                    (str(seg_from_previous_stage.shape[1:]), str(case_all_data.shape[1:]))
            else:
                seg_from_previous_stage = None

            # do you trust me? You better do. Otherwise you'll have to go through this mess and honestly there are
            # better things you could do right now

            # (above) documentation of the day. Nice. Even myself coming back 1 months later I have not friggin idea
            # what's going on. I keep the above documentation just for fun but attempt to make things clearer now

            need_to_pad = self.need_to_pad.copy()
            for d in range(3):
                # if case_all_data.shape + need_to_pad is still < patch size we need to pad more! We pad on both sides
                # always
                if need_to_pad[d] + case_all_data.shape[d + 1] < self.patch_size[d]:
                    need_to_pad[d] = self.patch_size[d] - case_all_data.shape[d + 1]

            # we can now choose the bbox from -need_to_pad // 2 to shape - patch_size + need_to_pad // 2. Here we
            # define what the upper and lower bound can be to then sample from them with np.random.randint
            shape = case_all_data.shape[1:]
            lb_x = - need_to_pad[0] // 2
            ub_x = shape[0] + need_to_pad[0] // 2 + need_to_pad[0] % 2 - self.patch_size[0]
            lb_y = - need_to_pad[1] // 2
            ub_y = shape[1] + need_to_pad[1] // 2 + need_to_pad[1] % 2 - self.patch_size[1]
            lb_z = - need_to_pad[2] // 2
            ub_z = shape[2] + need_to_pad[2] // 2 + need_to_pad[2] % 2 - self.patch_size[2]

            # if not force_fg then we can just sample the bbox randomly from lb and ub. Else we need to make sure we get
            # at least one of the foreground classes in the patch
            if not force_fg:
                bbox_x_lb = np.random.randint(lb_x, ub_x + 1)
                bbox_y_lb = np.random.randint(lb_y, ub_y + 1)
                bbox_z_lb = np.random.randint(lb_z, ub_z + 1)
            else:
                # these values should have been precomputed
                if 'class_locations' not in properties.keys():
                    raise RuntimeError("Please rerun the preprocessing with the newest version of nnU-Net!")

                # this saves us a np.unique. Preprocessing already did that for all cases. Neat.
                foreground_classes = np.array(
                    [i for i in properties['class_locations'].keys() if len(properties['class_locations'][i]) != 0])
                foreground_classes = foreground_classes[foreground_classes > 0]

                if len(foreground_classes) == 0:
                    # this only happens if some image does not contain foreground voxels at all
                    selected_class = None
                    voxels_of_that_class = None
                    print('case does not contain any foreground classes', i)
                else:
                    selected_class = np.random.choice(foreground_classes)

                    voxels_of_that_class = properties['class_locations'][selected_class]

                if voxels_of_that_class is not None:
                    selected_voxel = voxels_of_that_class[np.random.choice(len(voxels_of_that_class))]
                    # selected voxel is center voxel. Subtract half the patch size to get lower bbox voxel.
                    # Make sure it is within the bounds of lb and ub
                    bbox_x_lb = max(lb_x, selected_voxel[0] - self.patch_size[0] // 2)
                    bbox_y_lb = max(lb_y, selected_voxel[1] - self.patch_size[1] // 2)
                    bbox_z_lb = max(lb_z, selected_voxel[2] - self.patch_size[2] // 2)
                else:
                    # If the image does not contain any foreground classes, we fall back to random cropping
                    bbox_x_lb = np.random.randint(lb_x, ub_x + 1)
                    bbox_y_lb = np.random.randint(lb_y, ub_y + 1)
                    bbox_z_lb = np.random.randint(lb_z, ub_z + 1)

            bbox_x_ub = bbox_x_lb + self.patch_size[0]
            bbox_y_ub = bbox_y_lb + self.patch_size[1]
            bbox_z_ub = bbox_z_lb + self.patch_size[2]

            # whoever wrote this knew what he was doing (hint: it was me). We first crop the data to the region of the
            # bbox that actually lies within the data. This will result in a smaller array which is then faster to pad.
            # valid_bbox is just the coord that lied within the data cube. It will be padded to match the patch size
            # later
            valid_bbox_x_lb = max(0, bbox_x_lb)
            valid_bbox_x_ub = min(shape[0], bbox_x_ub)
            valid_bbox_y_lb = max(0, bbox_y_lb)
            valid_bbox_y_ub = min(shape[1], bbox_y_ub)
            valid_bbox_z_lb = max(0, bbox_z_lb)
            valid_bbox_z_ub = min(shape[2], bbox_z_ub)

            # At this point you might ask yourself why we would treat seg differently from seg_from_previous_stage.
            # Why not just concatenate them here and forget about the if statements? Well that's because segneeds to
            # be padded with -1 constant whereas seg_from_previous_stage needs to be padded with 0s (we could also
            # remove label -1 in the data augmentation but this way it is less error prone)
            case_all_data = np.copy(case_all_data[:, valid_bbox_x_lb:valid_bbox_x_ub,
                                    valid_bbox_y_lb:valid_bbox_y_ub,
                                    valid_bbox_z_lb:valid_bbox_z_ub])
            if seg_from_previous_stage is not None:
                seg_from_previous_stage = seg_from_previous_stage[:, valid_bbox_x_lb:valid_bbox_x_ub,
                                          valid_bbox_y_lb:valid_bbox_y_ub,
                                          valid_bbox_z_lb:valid_bbox_z_ub]

            data[j] = np.pad(case_all_data[:-1], ((0, 0),
                                                  (-min(0, bbox_x_lb), max(bbox_x_ub - shape[0], 0)),
                                                  (-min(0, bbox_y_lb), max(bbox_y_ub - shape[1], 0)),
                                                  (-min(0, bbox_z_lb), max(bbox_z_ub - shape[2], 0))),
                             self.pad_mode, **self.pad_kwargs_data)

            seg[j, 0] = np.pad(case_all_data[-1:], ((0, 0),
                                                    (-min(0, bbox_x_lb), max(bbox_x_ub - shape[0], 0)),
                                                    (-min(0, bbox_y_lb), max(bbox_y_ub - shape[1], 0)),
                                                    (-min(0, bbox_z_lb), max(bbox_z_ub - shape[2], 0))),
                               'constant', **{'constant_values': -1})
            if seg_from_previous_stage is not None:
                seg[j, 1] = np.pad(seg_from_previous_stage, ((0, 0),
                                                             (-min(0, bbox_x_lb),
                                                              max(bbox_x_ub - shape[0], 0)),
                                                             (-min(0, bbox_y_lb),
                                                              max(bbox_y_ub - shape[1], 0)),
                                                             (-min(0, bbox_z_lb),
                                                              max(bbox_z_ub - shape[2], 0))),
                                   'constant', **{'constant_values': 0})

        return {'data': data, 'seg': seg, 'properties': case_properties, 'keys': selected_keys}


class DataLoader2D(SlimDataLoaderBase):
    def __init__(self, data, patch_size, final_patch_size, batch_size, oversample_foreground_percent=0.0,
                 memmap_mode="r", pseudo_3d_slices=1, pad_mode="edge",
                 pad_kwargs_data=None, pad_sides=None, filter_phase=False):
        """
        This is the basic data loader for 2D networks. It uses preprocessed data as produced by my (Fabian) preprocessing.
        You can load the data with load_dataset(folder) where folder is the folder where the npz files are located. If there
        are only npz files present in that folder, the data loader will unpack them on the fly. This may take a while
        and increase CPU usage. Therefore, I advise you to call unpack_dataset(folder) first, which will unpack all npz
        to npy. Don't forget to call delete_npy(folder) after you are done with training?
        Why all the hassle? Well the decathlon dataset is huge. Using npy for everything will consume >1 TB and that is uncool
        given that I (Fabian) will have to store that permanently on /datasets and my local computer. With this strategy all
        data is stored in a compressed format (factor 10 smaller) and only unpacked when needed.
        :param data: get this with load_dataset(folder, stage=0). Plug the return value in here and you are g2g (good to go)
        :param patch_size: what patch size will this data loader return? it is common practice to first load larger
        patches so that a central crop after data augmentation can be done to reduce border artifacts. If unsure, use
        get_patch_size() from data_augmentation.default_data_augmentation
        :param final_patch_size: what will the patch finally be cropped to (after data augmentation)? this is the patch
        size that goes into your network. We need this here because we will pad patients in here so that patches at the
        border of patients are sampled properly
        :param batch_size:
        :param num_batches: how many batches will the data loader produce before stopping? None=endless
        :param seed:
        :param stage: ignore this (Fabian only)
        :param transpose: ignore this
        :param random: sample randomly; CAREFUL! non-random sampling requires batch_size=1, otherwise you will iterate batch_size times over the dataset
        :param pseudo_3d_slices: 7 = 3 below and 3 above the center slice
        """
        super(DataLoader2D, self).__init__(data, batch_size, None)
        if pad_kwargs_data is None:
            pad_kwargs_data = OrderedDict()
        self.pad_kwargs_data = pad_kwargs_data
        self.pad_mode = pad_mode
        self.pseudo_3d_slices = pseudo_3d_slices
        self.oversample_foreground_percent = oversample_foreground_percent
        self.final_patch_size = final_patch_size
        self.patch_size = patch_size
        self.list_of_keys = list(self._data.keys())
        self.need_to_pad = np.array(patch_size) - np.array(final_patch_size)
        self.memmap_mode = memmap_mode
        if pad_sides is not None:
            if not isinstance(pad_sides, np.ndarray):
                pad_sides = np.array(pad_sides)
            self.need_to_pad += pad_sides
        self.pad_sides = pad_sides
        self.data_shape, self.seg_shape = self.determine_shapes()

        if filter_phase:
            self.list_of_keys = self.filter_phase()

    def filter_phase(self):
        list_of_keys_filtered = []
        patient_list = []
        for key in self.list_of_keys:
            patient_list.append(key.split('_')[0])
            if 'properties' in self._data[key].keys():
                properties = self._data[key]['properties']
            else:
                properties = load_pickle(self._data[key]['properties_file'])
            ed_idx = np.rint(properties['ed_number']).astype(int)
            es_idx = np.rint(properties['es_number']).astype(int)
            frame_nb = int(key.split('frame')[-1])
            if frame_nb == ed_idx + 1 or frame_nb == es_idx + 1:
                list_of_keys_filtered.append(key)
        
        patient_list = sorted(list(set(patient_list)), key=lambda x:int(x[-3:]))
        print(list_of_keys_filtered)
        
        assert len(list_of_keys_filtered) == len(patient_list) * 2
        return list_of_keys_filtered


    def determine_shapes(self):
        num_seg = 1

        k = list(self._data.keys())[0]
        if isfile(self._data[k]['data_file'][:-4] + ".npy"):
            case_all_data = np.load(self._data[k]['data_file'][:-4] + ".npy", self.memmap_mode)
        else:
            case_all_data = np.load(self._data[k]['data_file'])['data']
        num_color_channels = case_all_data.shape[0] - num_seg
        data_shape = (self.batch_size, num_color_channels, *self.patch_size)
        seg_shape = (self.batch_size, num_seg, *self.patch_size)
        return data_shape, seg_shape

    def get_do_oversample(self, batch_idx):
        return not batch_idx < round(self.batch_size * (1 - self.oversample_foreground_percent))

    def generate_train_batch(self):
        selected_keys = np.random.choice(self.list_of_keys, self.batch_size, True, None)

        data = np.zeros(self.data_shape, dtype=np.float32)
        seg = np.zeros(self.seg_shape, dtype=np.float32)

        case_properties = []
        for j, i in enumerate(selected_keys):
            if 'properties' in self._data[i].keys():
                properties = self._data[i]['properties']
            else:
                properties = load_pickle(self._data[i]['properties_file'])
            case_properties.append(properties)

            if self.get_do_oversample(j):
                force_fg = True
            else:
                force_fg = False

            if not isfile(self._data[i]['data_file'][:-4] + ".npy"):
                # lets hope you know what you're doing
                case_all_data = np.load(self._data[i]['data_file'][:-4] + ".npz")['data']
            else:
                case_all_data = np.load(self._data[i]['data_file'][:-4] + ".npy", self.memmap_mode)

            # this is for when there is just a 2d slice in case_all_data (2d support)
            if len(case_all_data.shape) == 3:
                case_all_data = case_all_data[:, None]

            #matplotlib.use('QtAgg')
            #print(case_all_data.shape)
            #print(i)
            #fig, ax = plt.subplots(1, case_all_data.shape[1])
            #for u in range(case_all_data.shape[1]):
            #    ax[u].imshow(case_all_data[0, u], cmap='gray')
            #plt.show()

            # first select a slice. This can be either random (no force fg) or guaranteed to contain some class
            if not force_fg:
                random_slice = np.random.choice(case_all_data.shape[1])
                selected_class = None
            else:
                # these values should have been precomputed
                if 'class_locations' not in properties.keys():
                    raise RuntimeError("Please rerun the preprocessing with the newest version of nnU-Net!")

                foreground_classes = np.array(
                    [i for i in properties['class_locations'].keys() if len(properties['class_locations'][i]) != 0])
                foreground_classes = foreground_classes[foreground_classes > 0]
                if len(foreground_classes) == 0:
                    selected_class = None
                    random_slice = np.random.choice(case_all_data.shape[1])
                    print('case does not contain any foreground classes', i)
                else:
                    selected_class = np.random.choice(foreground_classes)

                    voxels_of_that_class = properties['class_locations'][selected_class]
                    valid_slices = np.unique(voxels_of_that_class[:, 0])
                    random_slice = np.random.choice(valid_slices)
                    voxels_of_that_class = voxels_of_that_class[voxels_of_that_class[:, 0] == random_slice]
                    voxels_of_that_class = voxels_of_that_class[:, 1:]

            # now crop case_all_data to contain just the slice of interest. If we want additional slice above and
            # below the current slice, here is where we get them. We stack those as additional color channels
            if self.pseudo_3d_slices == 1:
                case_all_data = case_all_data[:, random_slice]
            else:
                # this is very deprecated and will probably not work anymore. If you intend to use this you need to
                # check this!
                mn = random_slice - (self.pseudo_3d_slices - 1) // 2
                mx = random_slice + (self.pseudo_3d_slices - 1) // 2 + 1
                valid_mn = max(mn, 0)
                valid_mx = min(mx, case_all_data.shape[1])
                case_all_seg = case_all_data[-1:]
                case_all_data = case_all_data[:-1]
                case_all_data = case_all_data[:, valid_mn:valid_mx]
                case_all_seg = case_all_seg[:, random_slice]
                need_to_pad_below = valid_mn - mn
                need_to_pad_above = mx - valid_mx
                if need_to_pad_below > 0:
                    shp_for_pad = np.array(case_all_data.shape)
                    shp_for_pad[1] = need_to_pad_below
                    case_all_data = np.concatenate((np.zeros(shp_for_pad), case_all_data), 1)
                if need_to_pad_above > 0:
                    shp_for_pad = np.array(case_all_data.shape)
                    shp_for_pad[1] = need_to_pad_above
                    case_all_data = np.concatenate((case_all_data, np.zeros(shp_for_pad)), 1)
                case_all_data = case_all_data.reshape((-1, case_all_data.shape[-2], case_all_data.shape[-1]))
                case_all_data = np.concatenate((case_all_data, case_all_seg), 0)

            # case all data should now be (c, x, y)
            assert len(case_all_data.shape) == 3

            # we can now choose the bbox from -need_to_pad // 2 to shape - patch_size + need_to_pad // 2. Here we
            # define what the upper and lower bound can be to then sample from them with np.random.randint

            need_to_pad = self.need_to_pad.copy()
            for d in range(2):
                # if case_all_data.shape + need_to_pad is still < patch size we need to pad more! We pad on both sides
                # always
                if need_to_pad[d] + case_all_data.shape[d + 1] < self.patch_size[d]:
                    need_to_pad[d] = self.patch_size[d] - case_all_data.shape[d + 1]

            shape = case_all_data.shape[1:]
            lb_x = - need_to_pad[0] // 2
            ub_x = shape[0] + need_to_pad[0] // 2 + need_to_pad[0] % 2 - self.patch_size[0]
            lb_y = - need_to_pad[1] // 2
            ub_y = shape[1] + need_to_pad[1] // 2 + need_to_pad[1] % 2 - self.patch_size[1]

            # if not force_fg then we can just sample the bbox randomly from lb and ub. Else we need to make sure we get
            # at least one of the foreground classes in the patch
            if not force_fg or selected_class is None:
                bbox_x_lb = np.random.randint(lb_x, ub_x + 1)
                bbox_y_lb = np.random.randint(lb_y, ub_y + 1)
            else:
                # this saves us a np.unique. Preprocessing already did that for all cases. Neat.
                selected_voxel = voxels_of_that_class[np.random.choice(len(voxels_of_that_class))]
                # selected voxel is center voxel. Subtract half the patch size to get lower bbox voxel.
                # Make sure it is within the bounds of lb and ub
                bbox_x_lb = max(lb_x, selected_voxel[0] - self.patch_size[0] // 2)
                bbox_y_lb = max(lb_y, selected_voxel[1] - self.patch_size[1] // 2)

            bbox_x_ub = bbox_x_lb + self.patch_size[0]
            bbox_y_ub = bbox_y_lb + self.patch_size[1]

            # whoever wrote this knew what he was doing (hint: it was me). We first crop the data to the region of the
            # bbox that actually lies within the data. This will result in a smaller array which is then faster to pad.
            # valid_bbox is just the coord that lied within the data cube. It will be padded to match the patch size
            # later
            valid_bbox_x_lb = max(0, bbox_x_lb)
            valid_bbox_x_ub = min(shape[0], bbox_x_ub)
            valid_bbox_y_lb = max(0, bbox_y_lb)
            valid_bbox_y_ub = min(shape[1], bbox_y_ub)

            # At this point you might ask yourself why we would treat seg differently from seg_from_previous_stage.
            # Why not just concatenate them here and forget about the if statements? Well that's because segneeds to
            # be padded with -1 constant whereas seg_from_previous_stage needs to be padded with 0s (we could also
            # remove label -1 in the data augmentation but this way it is less error prone)

            case_all_data = case_all_data[:, valid_bbox_x_lb:valid_bbox_x_ub,
                            valid_bbox_y_lb:valid_bbox_y_ub]

            case_all_data_donly = np.pad(case_all_data[:-1], ((0, 0),
                                                              (-min(0, bbox_x_lb), max(bbox_x_ub - shape[0], 0)),
                                                              (-min(0, bbox_y_lb), max(bbox_y_ub - shape[1], 0))),
                                         self.pad_mode, **self.pad_kwargs_data)

            case_all_data_segonly = np.pad(case_all_data[-1:], ((0, 0),
                                                                (-min(0, bbox_x_lb), max(bbox_x_ub - shape[0], 0)),
                                                                (-min(0, bbox_y_lb), max(bbox_y_ub - shape[1], 0))),
                                           'constant', **{'constant_values': -1})

            data[j] = case_all_data_donly
            seg[j] = case_all_data_segonly
        
        #matplotlib.use('QtAgg')
        #print(data.shape)
        #fig, ax = plt.subplots(1, 1)
        #ax.imshow(data[0, 0], cmap='gray')
        #plt.show()

        keys = selected_keys
        return {'data': data, 'seg': seg, 'properties': case_properties, "keys": keys}
    



class DataLoader2DCropped(SlimDataLoaderBase):
    def __init__(self, data, patch_size, final_patch_size, batch_size, oversample_foreground_percent=0.0,
                 memmap_mode="r", pseudo_3d_slices=1, pad_mode="edge",
                 pad_kwargs_data=None, pad_sides=None, filter_phase=False):
        """
        This is the basic data loader for 2D networks. It uses preprocessed data as produced by my (Fabian) preprocessing.
        You can load the data with load_dataset(folder) where folder is the folder where the npz files are located. If there
        are only npz files present in that folder, the data loader will unpack them on the fly. This may take a while
        and increase CPU usage. Therefore, I advise you to call unpack_dataset(folder) first, which will unpack all npz
        to npy. Don't forget to call delete_npy(folder) after you are done with training?
        Why all the hassle? Well the decathlon dataset is huge. Using npy for everything will consume >1 TB and that is uncool
        given that I (Fabian) will have to store that permanently on /datasets and my local computer. With this strategy all
        data is stored in a compressed format (factor 10 smaller) and only unpacked when needed.
        :param data: get this with load_dataset(folder, stage=0). Plug the return value in here and you are g2g (good to go)
        :param patch_size: what patch size will this data loader return? it is common practice to first load larger
        patches so that a central crop after data augmentation can be done to reduce border artifacts. If unsure, use
        get_patch_size() from data_augmentation.default_data_augmentation
        :param final_patch_size: what will the patch finally be cropped to (after data augmentation)? this is the patch
        size that goes into your network. We need this here because we will pad patients in here so that patches at the
        border of patients are sampled properly
        :param batch_size:
        :param num_batches: how many batches will the data loader produce before stopping? None=endless
        :param seed:
        :param stage: ignore this (Fabian only)
        :param transpose: ignore this
        :param random: sample randomly; CAREFUL! non-random sampling requires batch_size=1, otherwise you will iterate batch_size times over the dataset
        :param pseudo_3d_slices: 7 = 3 below and 3 above the center slice
        """
        super(DataLoader2DCropped, self).__init__(data, batch_size, None)
        if pad_kwargs_data is None:
            pad_kwargs_data = OrderedDict()
        self.pad_kwargs_data = pad_kwargs_data
        self.pad_mode = pad_mode
        self.pseudo_3d_slices = pseudo_3d_slices
        self.oversample_foreground_percent = oversample_foreground_percent
        self.final_patch_size = final_patch_size
        self.patch_size = patch_size
        self.list_of_keys = list(self._data.keys())
        self.need_to_pad = np.array(patch_size) - np.array(final_patch_size)
        self.memmap_mode = memmap_mode
        if pad_sides is not None:
            if not isinstance(pad_sides, np.ndarray):
                pad_sides = np.array(pad_sides)
            self.need_to_pad += pad_sides
        self.pad_sides = pad_sides
        self.data_shape, self.seg_shape = self.determine_shapes()
        self.data_path = 'Lib_resampling_training_mask'

        if filter_phase:
            self.list_of_keys = self.filter_phase()

    def filter_phase(self):
        list_of_keys_filtered = []
        patient_list = []
        for key in self.list_of_keys:
            patient_list.append(key.split('_')[0])
            if 'properties' in self._data[key].keys():
                properties = self._data[key]['properties']
            else:
                properties = load_pickle(self._data[key]['properties_file'])
            ed_idx = np.rint(properties['ed_number']).astype(int)
            es_idx = np.rint(properties['es_number']).astype(int)
            frame_nb = int(key.split('frame')[-1])
            if frame_nb == ed_idx + 1 or frame_nb == es_idx + 1:
                list_of_keys_filtered.append(key)
        
        patient_list = sorted(list(set(patient_list)), key=lambda x:int(x[-3:]))
        print(list_of_keys_filtered)
        
        assert len(list_of_keys_filtered) == len(patient_list) * 2
        return list_of_keys_filtered


    def determine_shapes(self):
        num_seg = 1

        k = list(self._data.keys())[0]
        if isfile(self._data[k]['data_file'][:-4] + ".npy"):
            case_all_data = np.load(self._data[k]['data_file'][:-4] + ".npy", self.memmap_mode)
        else:
            case_all_data = np.load(self._data[k]['data_file'])['data']
        num_color_channels = case_all_data.shape[0] - num_seg
        data_shape = (self.batch_size, num_color_channels, *self.patch_size)
        seg_shape = (self.batch_size, num_seg, *self.patch_size)
        return data_shape, seg_shape

    def get_do_oversample(self, batch_idx):
        return not batch_idx < round(self.batch_size * (1 - self.oversample_foreground_percent))

    def generate_train_batch(self):
        selected_keys = np.random.choice(self.list_of_keys, self.batch_size, True, None)

        data = np.zeros(self.data_shape, dtype=np.float32)
        seg = np.zeros(self.seg_shape, dtype=np.float32)

        case_properties = []
        for j, i in enumerate(selected_keys):
            if 'properties' in self._data[i].keys():
                properties = self._data[i]['properties']
            else:
                properties = load_pickle(self._data[i]['properties_file'])
            case_properties.append(properties)

            if self.get_do_oversample(j):
                force_fg = True
            else:
                force_fg = False


            filename = i + ".npy"
            case_all_data = np.load(os.path.join(self.data_path, filename))
            case_all_data = case_all_data[:2]
            case_all_data = np.transpose(case_all_data, (0, 3, 1, 2))

            #if not isfile(self._data[i]['data_file'][:-4] + ".npy"):
            #    # lets hope you know what you're doing
            #    case_all_data = np.load(self._data[i]['data_file'][:-4] + ".npz")['data']
            #else:
            #    case_all_data = np.load(self._data[i]['data_file'][:-4] + ".npy", self.memmap_mode)

            # this is for when there is just a 2d slice in case_all_data (2d support)
            if len(case_all_data.shape) == 3:
                case_all_data = case_all_data[:, None]

            #matplotlib.use('QtAgg')
            #print(case_all_data.shape)
            #print(i)
            #fig, ax = plt.subplots(1, case_all_data.shape[1])
            #for u in range(case_all_data.shape[1]):
            #    ax[u].imshow(case_all_data[0, u], cmap='gray')
            #plt.show()

            # first select a slice. This can be either random (no force fg) or guaranteed to contain some class
            if not force_fg:
                random_slice = np.random.choice(case_all_data.shape[1])
                selected_class = None
            else:
                # these values should have been precomputed
                if 'class_locations' not in properties.keys():
                    raise RuntimeError("Please rerun the preprocessing with the newest version of nnU-Net!")

                foreground_classes = np.array(
                    [i for i in properties['class_locations'].keys() if len(properties['class_locations'][i]) != 0])
                foreground_classes = foreground_classes[foreground_classes > 0]
                if len(foreground_classes) == 0:
                    selected_class = None
                    random_slice = np.random.choice(case_all_data.shape[1])
                    print('case does not contain any foreground classes', i)
                else:
                    selected_class = np.random.choice(foreground_classes)

                    voxels_of_that_class = properties['class_locations'][selected_class]
                    valid_slices = np.unique(voxels_of_that_class[:, 0])
                    random_slice = np.random.choice(valid_slices)
                    voxels_of_that_class = voxels_of_that_class[voxels_of_that_class[:, 0] == random_slice]
                    voxels_of_that_class = voxels_of_that_class[:, 1:]

            # now crop case_all_data to contain just the slice of interest. If we want additional slice above and
            # below the current slice, here is where we get them. We stack those as additional color channels
            if self.pseudo_3d_slices == 1:
                case_all_data = case_all_data[:, random_slice]
            else:
                # this is very deprecated and will probably not work anymore. If you intend to use this you need to
                # check this!
                mn = random_slice - (self.pseudo_3d_slices - 1) // 2
                mx = random_slice + (self.pseudo_3d_slices - 1) // 2 + 1
                valid_mn = max(mn, 0)
                valid_mx = min(mx, case_all_data.shape[1])
                case_all_seg = case_all_data[-1:]
                case_all_data = case_all_data[:-1]
                case_all_data = case_all_data[:, valid_mn:valid_mx]
                case_all_seg = case_all_seg[:, random_slice]
                need_to_pad_below = valid_mn - mn
                need_to_pad_above = mx - valid_mx
                if need_to_pad_below > 0:
                    shp_for_pad = np.array(case_all_data.shape)
                    shp_for_pad[1] = need_to_pad_below
                    case_all_data = np.concatenate((np.zeros(shp_for_pad), case_all_data), 1)
                if need_to_pad_above > 0:
                    shp_for_pad = np.array(case_all_data.shape)
                    shp_for_pad[1] = need_to_pad_above
                    case_all_data = np.concatenate((case_all_data, np.zeros(shp_for_pad)), 1)
                case_all_data = case_all_data.reshape((-1, case_all_data.shape[-2], case_all_data.shape[-1]))
                case_all_data = np.concatenate((case_all_data, case_all_seg), 0)

            # case all data should now be (c, x, y)
            assert len(case_all_data.shape) == 3

            # we can now choose the bbox from -need_to_pad // 2 to shape - patch_size + need_to_pad // 2. Here we
            # define what the upper and lower bound can be to then sample from them with np.random.randint

            #need_to_pad = self.need_to_pad.copy()
            #for d in range(2):
            #    # if case_all_data.shape + need_to_pad is still < patch size we need to pad more! We pad on both sides
            #    # always
            #    if need_to_pad[d] + case_all_data.shape[d + 1] < self.patch_size[d]:
            #        need_to_pad[d] = self.patch_size[d] - case_all_data.shape[d + 1]
#
            #shape = case_all_data.shape[1:]
            #lb_x = - need_to_pad[0] // 2
            #ub_x = shape[0] + need_to_pad[0] // 2 + need_to_pad[0] % 2 - self.patch_size[0]
            #lb_y = - need_to_pad[1] // 2
            #ub_y = shape[1] + need_to_pad[1] // 2 + need_to_pad[1] % 2 - self.patch_size[1]
#
            ## if not force_fg then we can just sample the bbox randomly from lb and ub. Else we need to make sure we get
            ## at least one of the foreground classes in the patch
            #if not force_fg or selected_class is None:
            #    bbox_x_lb = np.random.randint(lb_x, ub_x + 1)
            #    bbox_y_lb = np.random.randint(lb_y, ub_y + 1)
            #else:
            #    # this saves us a np.unique. Preprocessing already did that for all cases. Neat.
            #    selected_voxel = voxels_of_that_class[np.random.choice(len(voxels_of_that_class))]
            #    # selected voxel is center voxel. Subtract half the patch size to get lower bbox voxel.
            #    # Make sure it is within the bounds of lb and ub
            #    bbox_x_lb = max(lb_x, selected_voxel[0] - self.patch_size[0] // 2)
            #    bbox_y_lb = max(lb_y, selected_voxel[1] - self.patch_size[1] // 2)
#
            #bbox_x_ub = bbox_x_lb + self.patch_size[0]
            #bbox_y_ub = bbox_y_lb + self.patch_size[1]
#
            ## whoever wrote this knew what he was doing (hint: it was me). We first crop the data to the region of the
            ## bbox that actually lies within the data. This will result in a smaller array which is then faster to pad.
            ## valid_bbox is just the coord that lied within the data cube. It will be padded to match the patch size
            ## later
            #valid_bbox_x_lb = max(0, bbox_x_lb)
            #valid_bbox_x_ub = min(shape[0], bbox_x_ub)
            #valid_bbox_y_lb = max(0, bbox_y_lb)
            #valid_bbox_y_ub = min(shape[1], bbox_y_ub)

            # At this point you might ask yourself why we would treat seg differently from seg_from_previous_stage.
            # Why not just concatenate them here and forget about the if statements? Well that's because segneeds to
            # be padded with -1 constant whereas seg_from_previous_stage needs to be padded with 0s (we could also
            # remove label -1 in the data augmentation but this way it is less error prone)

            #case_all_data = case_all_data[:, valid_bbox_x_lb:valid_bbox_x_ub,
            #                valid_bbox_y_lb:valid_bbox_y_ub]
            #case_all_data_donly = np.pad(case_all_data[:-1], ((0, 0),
            #                                                  (-min(0, bbox_x_lb), max(bbox_x_ub - shape[0], 0)),
            #                                                  (-min(0, bbox_y_lb), max(bbox_y_ub - shape[1], 0))),
            #                             self.pad_mode, **self.pad_kwargs_data)
#
            #case_all_data_segonly = np.pad(case_all_data[-1:], ((0, 0),
            #                                                    (-min(0, bbox_x_lb), max(bbox_x_ub - shape[0], 0)),
            #                                                    (-min(0, bbox_y_lb), max(bbox_y_ub - shape[1], 0))),
            #                               'constant', **{'constant_values': -1})

            data[j] = case_all_data[0]
            seg[j] = case_all_data[1]
        
        #matplotlib.use('QtAgg')
        #print(data.shape)
        #fig, ax = plt.subplots(1, 1)
        #ax.imshow(data[0, 0], cmap='gray')
        #plt.show()

        keys = selected_keys
        return {'data': data, 'seg': seg, 'properties': case_properties, "keys": keys}



class DataLoader2DUnlabeled(SlimDataLoaderBase):
    def __init__(self, data, patch_size, final_patch_size, batch_size, oversample_foreground_percent=0.0,
                 memmap_mode="r", pseudo_3d_slices=1, pad_mode="edge",
                 pad_kwargs_data=None, pad_sides=None):
        """
        This is the basic data loader for 2D networks. It uses preprocessed data as produced by my (Fabian) preprocessing.
        You can load the data with load_dataset(folder) where folder is the folder where the npz files are located. If there
        are only npz files present in that folder, the data loader will unpack them on the fly. This may take a while
        and increase CPU usage. Therefore, I advise you to call unpack_dataset(folder) first, which will unpack all npz
        to npy. Don't forget to call delete_npy(folder) after you are done with training?
        Why all the hassle? Well the decathlon dataset is huge. Using npy for everything will consume >1 TB and that is uncool
        given that I (Fabian) will have to store that permanently on /datasets and my local computer. With this strategy all
        data is stored in a compressed format (factor 10 smaller) and only unpacked when needed.
        :param data: get this with load_dataset(folder, stage=0). Plug the return value in here and you are g2g (good to go)
        :param patch_size: what patch size will this data loader return? it is common practice to first load larger
        patches so that a central crop after data augmentation can be done to reduce border artifacts. If unsure, use
        get_patch_size() from data_augmentation.default_data_augmentation
        :param final_patch_size: what will the patch finally be cropped to (after data augmentation)? this is the patch
        size that goes into your network. We need this here because we will pad patients in here so that patches at the
        border of patients are sampled properly
        :param batch_size:
        :param num_batches: how many batches will the data loader produce before stopping? None=endless
        :param seed:
        :param stage: ignore this (Fabian only)
        :param transpose: ignore this
        :param random: sample randomly; CAREFUL! non-random sampling requires batch_size=1, otherwise you will iterate batch_size times over the dataset
        :param pseudo_3d_slices: 7 = 3 below and 3 above the center slice
        """
        super(DataLoader2DUnlabeled, self).__init__(data, batch_size, None)
        if pad_kwargs_data is None:
            pad_kwargs_data = OrderedDict()
        self.pad_kwargs_data = pad_kwargs_data
        self.pad_mode = pad_mode
        self.pseudo_3d_slices = pseudo_3d_slices
        self.oversample_foreground_percent = oversample_foreground_percent
        self.final_patch_size = final_patch_size
        self.patch_size = patch_size
        self.list_of_keys = list(self._data.keys())
        self.need_to_pad = np.array(patch_size) - np.array(final_patch_size)
        self.memmap_mode = memmap_mode
        if pad_sides is not None:
            if not isinstance(pad_sides, np.ndarray):
                pad_sides = np.array(pad_sides)
            self.need_to_pad += pad_sides
        self.pad_sides = pad_sides
        self.data_shape = self.determine_shapes()

    def determine_shapes(self):
        num_seg = 0

        k = list(self._data.keys())[0]
        if isfile(self._data[k]['data_file'][:-4] + ".npy"):
            case_all_data = np.load(self._data[k]['data_file'][:-4] + ".npy", self.memmap_mode)
        else:
            case_all_data = np.load(self._data[k]['data_file'])['data']
        num_color_channels = case_all_data.shape[0] - num_seg
        data_shape = (self.batch_size, num_color_channels, *self.patch_size)
        return data_shape

    def get_do_oversample(self, batch_idx):
        return not batch_idx < round(self.batch_size * (1 - self.oversample_foreground_percent))

    def generate_train_batch(self):
        selected_keys = np.random.choice(self.list_of_keys, self.batch_size, True, None)

        data = np.zeros(self.data_shape, dtype=np.float32)

        case_properties = []
        for j, i in enumerate(selected_keys):
            if 'properties' in self._data[i].keys():
                properties = self._data[i]['properties']
            else:
                properties = load_pickle(self._data[i]['properties_file'])
            case_properties.append(properties)

            force_fg = False

            if not isfile(self._data[i]['data_file'][:-4] + ".npy"):
                # lets hope you know what you're doing
                case_all_data = np.load(self._data[i]['data_file'][:-4] + ".npz")['data']
            else:
                case_all_data = np.load(self._data[i]['data_file'][:-4] + ".npy", self.memmap_mode)

            # this is for when there is just a 2d slice in case_all_data (2d support)
            if len(case_all_data.shape) == 3:
                case_all_data = case_all_data[:, None]

            # first select a slice. This can be either random (no force fg) or guaranteed to contain some class
            if not force_fg:
                random_slice = np.random.choice(case_all_data.shape[1])
                selected_class = None
            else:
                # these values should have been precomputed
                if 'class_locations' not in properties.keys():
                    raise RuntimeError("Please rerun the preprocessing with the newest version of nnU-Net!")

                foreground_classes = np.array(
                    [i for i in properties['class_locations'].keys() if len(properties['class_locations'][i]) != 0])
                foreground_classes = foreground_classes[foreground_classes > 0]
                if len(foreground_classes) == 0:
                    selected_class = None
                    random_slice = np.random.choice(case_all_data.shape[1])
                    print('case does not contain any foreground classes', i)
                else:
                    selected_class = np.random.choice(foreground_classes)

                    voxels_of_that_class = properties['class_locations'][selected_class]
                    valid_slices = np.unique(voxels_of_that_class[:, 0])
                    random_slice = np.random.choice(valid_slices)
                    voxels_of_that_class = voxels_of_that_class[voxels_of_that_class[:, 0] == random_slice]
                    voxels_of_that_class = voxels_of_that_class[:, 1:]

            # now crop case_all_data to contain just the slice of interest. If we want additional slice above and
            # below the current slice, here is where we get them. We stack those as additional color channels
            if self.pseudo_3d_slices == 1:
                case_all_data = case_all_data[:, random_slice]
            else:
                # this is very deprecated and will probably not work anymore. If you intend to use this you need to
                # check this!
                mn = random_slice - (self.pseudo_3d_slices - 1) // 2
                mx = random_slice + (self.pseudo_3d_slices - 1) // 2 + 1
                valid_mn = max(mn, 0)
                valid_mx = min(mx, case_all_data.shape[1])
                case_all_seg = case_all_data[-1:]
                case_all_data = case_all_data[:-1]
                case_all_data = case_all_data[:, valid_mn:valid_mx]
                case_all_seg = case_all_seg[:, random_slice]
                need_to_pad_below = valid_mn - mn
                need_to_pad_above = mx - valid_mx
                if need_to_pad_below > 0:
                    shp_for_pad = np.array(case_all_data.shape)
                    shp_for_pad[1] = need_to_pad_below
                    case_all_data = np.concatenate((np.zeros(shp_for_pad), case_all_data), 1)
                if need_to_pad_above > 0:
                    shp_for_pad = np.array(case_all_data.shape)
                    shp_for_pad[1] = need_to_pad_above
                    case_all_data = np.concatenate((case_all_data, np.zeros(shp_for_pad)), 1)
                case_all_data = case_all_data.reshape((-1, case_all_data.shape[-2], case_all_data.shape[-1]))
                case_all_data = np.concatenate((case_all_data, case_all_seg), 0)

            # case all data should now be (c, x, y)
            assert len(case_all_data.shape) == 3
            assert case_all_data.shape[0] == 1

            # we can now choose the bbox from -need_to_pad // 2 to shape - patch_size + need_to_pad // 2. Here we
            # define what the upper and lower bound can be to then sample from them with np.random.randint

            need_to_pad = self.need_to_pad.copy()
            for d in range(2):
                # if case_all_data.shape + need_to_pad is still < patch size we need to pad more! We pad on both sides
                # always
                if need_to_pad[d] + case_all_data.shape[d + 1] < self.patch_size[d]:
                    need_to_pad[d] = self.patch_size[d] - case_all_data.shape[d + 1]

            shape = case_all_data.shape[1:]
            lb_x = - need_to_pad[0] // 2
            ub_x = shape[0] + need_to_pad[0] // 2 + need_to_pad[0] % 2 - self.patch_size[0]
            lb_y = - need_to_pad[1] // 2
            ub_y = shape[1] + need_to_pad[1] // 2 + need_to_pad[1] % 2 - self.patch_size[1]

            # if not force_fg then we can just sample the bbox randomly from lb and ub. Else we need to make sure we get
            # at least one of the foreground classes in the patch
            if not force_fg or selected_class is None:
                bbox_x_lb = np.random.randint(lb_x, ub_x + 1)
                bbox_y_lb = np.random.randint(lb_y, ub_y + 1)
            else:
                # this saves us a np.unique. Preprocessing already did that for all cases. Neat.
                selected_voxel = voxels_of_that_class[np.random.choice(len(voxels_of_that_class))]
                # selected voxel is center voxel. Subtract half the patch size to get lower bbox voxel.
                # Make sure it is within the bounds of lb and ub
                bbox_x_lb = max(lb_x, selected_voxel[0] - self.patch_size[0] // 2)
                bbox_y_lb = max(lb_y, selected_voxel[1] - self.patch_size[1] // 2)

            bbox_x_ub = bbox_x_lb + self.patch_size[0]
            bbox_y_ub = bbox_y_lb + self.patch_size[1]

            # whoever wrote this knew what he was doing (hint: it was me). We first crop the data to the region of the
            # bbox that actually lies within the data. This will result in a smaller array which is then faster to pad.
            # valid_bbox is just the coord that lied within the data cube. It will be padded to match the patch size
            # later
            valid_bbox_x_lb = max(0, bbox_x_lb)
            valid_bbox_x_ub = min(shape[0], bbox_x_ub)
            valid_bbox_y_lb = max(0, bbox_y_lb)
            valid_bbox_y_ub = min(shape[1], bbox_y_ub)

            # At this point you might ask yourself why we would treat seg differently from seg_from_previous_stage.
            # Why not just concatenate them here and forget about the if statements? Well that's because segneeds to
            # be padded with -1 constant whereas seg_from_previous_stage needs to be padded with 0s (we could also
            # remove label -1 in the data augmentation but this way it is less error prone)

            case_all_data = case_all_data[:, valid_bbox_x_lb:valid_bbox_x_ub,
                            valid_bbox_y_lb:valid_bbox_y_ub]

            case_all_data_donly = np.pad(case_all_data, ((0, 0),
                                                              (-min(0, bbox_x_lb), max(bbox_x_ub - shape[0], 0)),
                                                              (-min(0, bbox_y_lb), max(bbox_y_ub - shape[1], 0))),
                                         self.pad_mode, **self.pad_kwargs_data)

            data[j] = case_all_data_donly

        keys = selected_keys
        return {'data': data, 'seg': None, 'properties': case_properties, "keys": keys}


class DataLoader2DMiddle(SlimDataLoaderBase):
    def __init__(self, data, patch_size, final_patch_size, batch_size, is_val, one_vs_all, oversample_foreground_percent=0.0,
                 memmap_mode="r", pseudo_3d_slices=1, pad_mode="edge",
                 pad_kwargs_data=None, pad_sides=None):
        """
        This is the basic data loader for 2D networks. It uses preprocessed data as produced by my (Fabian) preprocessing.
        You can load the data with load_dataset(folder) where folder is the folder where the npz files are located. If there
        are only npz files present in that folder, the data loader will unpack them on the fly. This may take a while
        and increase CPU usage. Therefore, I advise you to call unpack_dataset(folder) first, which will unpack all npz
        to npy. Don't forget to call delete_npy(folder) after you are done with training?
        Why all the hassle? Well the decathlon dataset is huge. Using npy for everything will consume >1 TB and that is uncool
        given that I (Fabian) will have to store that permanently on /datasets and my local computer. With this strategy all
        data is stored in a compressed format (factor 10 smaller) and only unpacked when needed.
        :param data: get this with load_dataset(folder, stage=0). Plug the return value in here and you are g2g (good to go)
        :param patch_size: what patch size will this data loader return? it is common practice to first load larger
        patches so that a central crop after data augmentation can be done to reduce border artifacts. If unsure, use
        get_patch_size() from data_augmentation.default_data_augmentation
        :param final_patch_size: what will the patch finally be cropped to (after data augmentation)? this is the patch
        size that goes into your network. We need this here because we will pad patients in here so that patches at the
        border of patients are sampled properly
        :param batch_size:
        :param num_batches: how many batches will the data loader produce before stopping? None=endless
        :param seed:
        :param stage: ignore this (Fabian only)
        :param transpose: ignore this
        :param random: sample randomly; CAREFUL! non-random sampling requires batch_size=1, otherwise you will iterate batch_size times over the dataset
        :param pseudo_3d_slices: 7 = 3 below and 3 above the center slice
        """
        super(DataLoader2DMiddle, self).__init__(data, batch_size, None)
        if pad_kwargs_data is None:
            pad_kwargs_data = OrderedDict()
        self.pad_kwargs_data = pad_kwargs_data
        self.is_val = is_val
        self.pad_mode = pad_mode
        self.pseudo_3d_slices = pseudo_3d_slices
        self.oversample_foreground_percent = oversample_foreground_percent
        self.final_patch_size = final_patch_size
        self.patch_size = patch_size
        self.list_of_keys = list(self._data.keys())
        self.need_to_pad = np.array(patch_size) - np.array(final_patch_size)
        self.memmap_mode = memmap_mode
        self.one_vs_all = one_vs_all
        self.percent = None
        if pad_sides is not None:
            if not isinstance(pad_sides, np.ndarray):
                pad_sides = np.array(pad_sides)
            self.need_to_pad += pad_sides
        self.pad_sides = pad_sides
        self.data_shape, self.seg_shape = self.determine_shapes()

    def determine_shapes(self):
        num_seg = 1

        k = list(self._data.keys())[0]
        if isfile(self._data[k]['data_file'][:-4] + ".npy"):
            case_all_data = np.load(self._data[k]['data_file'][:-4] + ".npy", self.memmap_mode)
        else:
            case_all_data = np.load(self._data[k]['data_file'])['data']
        num_color_channels = case_all_data.shape[0] - num_seg
        data_shape = (self.batch_size, num_color_channels, *self.patch_size)
        seg_shape = (self.batch_size, num_seg, *self.patch_size)
        return data_shape, seg_shape

    def get_do_oversample(self, batch_idx):
        return not batch_idx < round(self.batch_size * (1 - self.oversample_foreground_percent))

    def generate_train_batch(self):
        selected_keys = np.random.choice(self.list_of_keys, self.batch_size, True, None)

        data = np.zeros(self.data_shape, dtype=np.float32)
        seg = np.zeros(self.seg_shape, dtype=np.float32)
        middle = np.zeros(self.data_shape, dtype=np.float32)
        middle_seg = np.zeros(self.seg_shape, dtype=np.float32)
        batched_slice_distance = np.zeros(shape=(self.batch_size,), dtype=np.float32)

        case_properties = []
        for j, i in enumerate(selected_keys):
            if 'properties' in self._data[i].keys():
                properties = self._data[i]['properties']
            else:
                properties = load_pickle(self._data[i]['properties_file'])
            case_properties.append(properties)

            if not isfile(self._data[i]['data_file'][:-4] + ".npy"):
                # lets hope you know what you're doing
                case_all_data = np.load(self._data[i]['data_file'][:-4] + ".npz")['data']
            else:
                case_all_data = np.load(self._data[i]['data_file'][:-4] + ".npy", self.memmap_mode)

            # this is for when there is just a 2d slice in case_all_data (2d support)
            if len(case_all_data.shape) == 3:
                case_all_data = case_all_data[:, None]

            # first select a slice. This can be either random (no force fg) or guaranteed to contain some class
            if self.is_val:
                if self.one_vs_all:
                    random_slice = np.random.choice(case_all_data.shape[1]) 
                    random_slice, middle_slice = get_slice_nb_one_vs_all(random_slice=random_slice)
                else:
                    #middle_slice = get_idx(case_all_data.shape[1] * 0.33)
                    middle_slice = get_idx(case_all_data.shape[1] * self.percent)
                    p_non_middle = np.full(shape=(case_all_data.shape[1],), fill_value=1 / case_all_data.shape[1])
                    #p_non_middle = np.full(shape=(case_all_data.shape[1],), fill_value=1 / (case_all_data.shape[1] - 1)).astype(float)
                    #p_non_middle[middle_slice] = 0.0
                    random_slice = np.random.choice(case_all_data.shape[1], p=p_non_middle)
                batched_slice_distance = None
            else:
                if self.one_vs_all:
                    random_slice = np.random.choice(case_all_data.shape[1])
                    if random_slice == 0:
                        middle_slice = 1
                    elif random_slice == case_all_data.shape[1] - 1:
                        middle_slice = case_all_data.shape[1] - 2
                    else:
                        p_middle = np.full(shape=(case_all_data.shape[1],), fill_value=0.0)
                        p_middle[random_slice + 1] = 0.5
                        p_middle[random_slice - 1] = 0.5
                        middle_slice = np.random.choice(case_all_data.shape[1], p=p_middle)
                    #p_middle = np.full(shape=(case_all_data.shape[1],), fill_value=1 / (case_all_data.shape[1] - 1))
                    #p_middle[random_slice] = 0
                    #middle_slice = np.random.choice(case_all_data.shape[1], p=p_middle)
                    #assert middle_slice != random_slice
                    #slice_distance = abs(middle_slice - random_slice) / case_all_data.shape[1]
                    #batched_slice_distance[j] = slice_distance
                else:
                    quart1 = case_all_data.shape[1] * 0.056
                    quart2 = case_all_data.shape[1] * 0.56
                    idx1 = get_idx(quart1)
                    idx2 = get_idx(quart2)
                    p = np.zeros(shape=(case_all_data.shape[1],))
                    p_middle = np.copy(p)
                    p_non_middle = np.copy(p)
                    middle_nb = idx2 - idx1
                    non_middle_nb = case_all_data.shape[1] - middle_nb
                    p_middle[idx1:idx2] = 1 / middle_nb
                    middle_slice = np.random.choice(case_all_data.shape[1], p=p_middle)
                    p_non_middle[p_middle == 0] = 1 / non_middle_nb
                    if middle_slice == 0:
                        middle_slice_percent[j] = 0.49 / case_all_data.shape[1]
                    else:
                        middle_slice_percent[j] = middle_slice / case_all_data.shape[1]
                    random_slice = np.random.choice(case_all_data.shape[1], p=p_non_middle)
                    random_slice_percent = None
                    if self.interpolate:
                        p_non_middle[middle_slice + 1] = 0
                        p_non_middle[middle_slice - 1] = 0
                        p_non_middle[p_non_middle != 0] = 1 / np.count_nonzero(p_non_middle)
                        random_slice = np.random.choice(case_all_data.shape[1], p=p_non_middle)
                        inter_slice = (random_slice + middle_slice) // 2
                        assert inter_slice != random_slice and inter_slice != middle_slice
                        #print(middle_slice)
                        #print(inter_slice)
                        #print(random_slice)

            # now crop case_all_data to contain just the slice of interest. If we want additional slice above and
            # below the current slice, here is where we get them. We stack those as additional color channels
            middle_data = case_all_data[:, middle_slice]
            if self.pseudo_3d_slices == 1:
                case_all_data = case_all_data[:, random_slice]
            else:
                # this is very deprecated and will probably not work anymore. If you intend to use this you need to
                # check this!
                mn = random_slice - (self.pseudo_3d_slices - 1) // 2
                mx = random_slice + (self.pseudo_3d_slices - 1) // 2 + 1
                valid_mn = max(mn, 0)
                valid_mx = min(mx, case_all_data.shape[1])
                case_all_seg = case_all_data[-1:]
                case_all_data = case_all_data[:-1]
                case_all_data = case_all_data[:, valid_mn:valid_mx]
                case_all_seg = case_all_seg[:, random_slice]
                need_to_pad_below = valid_mn - mn
                need_to_pad_above = mx - valid_mx
                if need_to_pad_below > 0:
                    shp_for_pad = np.array(case_all_data.shape)
                    shp_for_pad[1] = need_to_pad_below
                    case_all_data = np.concatenate((np.zeros(shp_for_pad), case_all_data), 1)
                if need_to_pad_above > 0:
                    shp_for_pad = np.array(case_all_data.shape)
                    shp_for_pad[1] = need_to_pad_above
                    case_all_data = np.concatenate((case_all_data, np.zeros(shp_for_pad)), 1)
                case_all_data = case_all_data.reshape((-1, case_all_data.shape[-2], case_all_data.shape[-1]))
                case_all_data = np.concatenate((case_all_data, case_all_seg), 0)

            # case all data should now be (c, x, y)
            assert len(case_all_data.shape) == 3
            assert len(middle_data.shape) == 3

            # we can now choose the bbox from -need_to_pad // 2 to shape - patch_size + need_to_pad // 2. Here we
            # define what the upper and lower bound can be to then sample from them with np.random.randint

            need_to_pad = self.need_to_pad.copy()
            for d in range(2):
                # if case_all_data.shape + need_to_pad is still < patch size we need to pad more! We pad on both sides
                # always
                if need_to_pad[d] + case_all_data.shape[d + 1] < self.patch_size[d]:
                    need_to_pad[d] = self.patch_size[d] - case_all_data.shape[d + 1]

            shape = case_all_data.shape[1:]
            lb_x = - need_to_pad[0] // 2
            ub_x = shape[0] + need_to_pad[0] // 2 + need_to_pad[0] % 2 - self.patch_size[0]
            lb_y = - need_to_pad[1] // 2
            ub_y = shape[1] + need_to_pad[1] // 2 + need_to_pad[1] % 2 - self.patch_size[1]

            # if not force_fg then we can just sample the bbox randomly from lb and ub. Else we need to make sure we get
            # at least one of the foreground classes in the patch
            bbox_x_lb = np.random.randint(lb_x, ub_x + 1)
            bbox_y_lb = np.random.randint(lb_y, ub_y + 1)

            bbox_x_ub = bbox_x_lb + self.patch_size[0]
            bbox_y_ub = bbox_y_lb + self.patch_size[1]

            # whoever wrote this knew what he was doing (hint: it was me). We first crop the data to the region of the
            # bbox that actually lies within the data. This will result in a smaller array which is then faster to pad.
            # valid_bbox is just the coord that lied within the data cube. It will be padded to match the patch size
            # later
            valid_bbox_x_lb = max(0, bbox_x_lb)
            valid_bbox_x_ub = min(shape[0], bbox_x_ub)
            valid_bbox_y_lb = max(0, bbox_y_lb)
            valid_bbox_y_ub = min(shape[1], bbox_y_ub)

            # At this point you might ask yourself why we would treat seg differently from seg_from_previous_stage.
            # Why not just concatenate them here and forget about the if statements? Well that's because segneeds to
            # be padded with -1 constant whereas seg_from_previous_stage needs to be padded with 0s (we could also
            # remove label -1 in the data augmentation but this way it is less error prone)

            middle_data = middle_data[:, valid_bbox_x_lb:valid_bbox_x_ub,
                        valid_bbox_y_lb:valid_bbox_y_ub]
                
            middle_data_donly = np.pad(middle_data[:-1], ((0, 0),
                                                            (-min(0, bbox_x_lb), max(bbox_x_ub - shape[0], 0)),
                                                            (-min(0, bbox_y_lb), max(bbox_y_ub - shape[1], 0))),
                                        self.pad_mode, **self.pad_kwargs_data)
            
            middle_data_segonly = np.pad(middle_data[-1:], ((0, 0),
                                                                (-min(0, bbox_x_lb), max(bbox_x_ub - shape[0], 0)),
                                                                (-min(0, bbox_y_lb), max(bbox_y_ub - shape[1], 0))),
                                           'constant', **{'constant_values': -1})
                
            middle[j] = middle_data_donly
            middle_seg[j] = middle_data_segonly

            case_all_data = case_all_data[:, valid_bbox_x_lb:valid_bbox_x_ub,
                            valid_bbox_y_lb:valid_bbox_y_ub]

            case_all_data_donly = np.pad(case_all_data[:-1], ((0, 0),
                                                              (-min(0, bbox_x_lb), max(bbox_x_ub - shape[0], 0)),
                                                              (-min(0, bbox_y_lb), max(bbox_y_ub - shape[1], 0))),
                                         self.pad_mode, **self.pad_kwargs_data)

            case_all_data_segonly = np.pad(case_all_data[-1:], ((0, 0),
                                                                (-min(0, bbox_x_lb), max(bbox_x_ub - shape[0], 0)),
                                                                (-min(0, bbox_y_lb), max(bbox_y_ub - shape[1], 0))),
                                           'constant', **{'constant_values': -1})

            data[j] = case_all_data_donly
            seg[j] = case_all_data_segonly

        keys = selected_keys
        return {'data': [data, middle], 'seg': [seg, middle_seg], 'properties': case_properties, "keys": keys, "slice_distance": batched_slice_distance}


class DataLoader2DMiddleUnlabeled(SlimDataLoaderBase):
    def __init__(self, data, patch_size, final_patch_size, batch_size, is_val, v1, one_vs_all, unlabeled_dataset, oversample_foreground_percent=0.0,
                 memmap_mode="r", pseudo_3d_slices=1, pad_mode="edge",
                 pad_kwargs_data=None, pad_sides=None):
        """
        This is the basic data loader for 2D networks. It uses preprocessed data as produced by my (Fabian) preprocessing.
        You can load the data with load_dataset(folder) where folder is the folder where the npz files are located. If there
        are only npz files present in that folder, the data loader will unpack them on the fly. This may take a while
        and increase CPU usage. Therefore, I advise you to call unpack_dataset(folder) first, which will unpack all npz
        to npy. Don't forget to call delete_npy(folder) after you are done with training?
        Why all the hassle? Well the decathlon dataset is huge. Using npy for everything will consume >1 TB and that is uncool
        given that I (Fabian) will have to store that permanently on /datasets and my local computer. With this strategy all
        data is stored in a compressed format (factor 10 smaller) and only unpacked when needed.
        :param data: get this with load_dataset(folder, stage=0). Plug the return value in here and you are g2g (good to go)
        :param patch_size: what patch size will this data loader return? it is common practice to first load larger
        patches so that a central crop after data augmentation can be done to reduce border artifacts. If unsure, use
        get_patch_size() from data_augmentation.default_data_augmentation
        :param final_patch_size: what will the patch finally be cropped to (after data augmentation)? this is the patch
        size that goes into your network. We need this here because we will pad patients in here so that patches at the
        border of patients are sampled properly
        :param batch_size:
        :param num_batches: how many batches will the data loader produce before stopping? None=endless
        :param seed:
        :param stage: ignore this (Fabian only)
        :param transpose: ignore this
        :param random: sample randomly; CAREFUL! non-random sampling requires batch_size=1, otherwise you will iterate batch_size times over the dataset
        :param pseudo_3d_slices: 7 = 3 below and 3 above the center slice
        """
        super(DataLoader2DMiddleUnlabeled, self).__init__(data, batch_size, None)
        if pad_kwargs_data is None:
            pad_kwargs_data = OrderedDict()
        self.pad_kwargs_data = pad_kwargs_data
        self.is_val = is_val
        self.pad_mode = pad_mode
        self.v1 = v1
        self.pseudo_3d_slices = pseudo_3d_slices
        self.oversample_foreground_percent = oversample_foreground_percent
        self.final_patch_size = final_patch_size
        self.patch_size = patch_size
        self.list_of_keys = list(self._data.keys())
        self.un_list_of_keys = list(unlabeled_dataset.keys())
        self.need_to_pad = np.array(patch_size) - np.array(final_patch_size)
        self.memmap_mode = memmap_mode
        self.one_vs_all = one_vs_all
        self.percent = None
        if pad_sides is not None:
            if not isinstance(pad_sides, np.ndarray):
                pad_sides = np.array(pad_sides)
            self.need_to_pad += pad_sides
        self.pad_sides = pad_sides
        self.data_shape, self.seg_shape = self.determine_shapes()
        self.un_data = unlabeled_dataset

    def determine_shapes(self):
        num_seg = 1

        k = list(self._data.keys())[0]
        if isfile(self._data[k]['data_file'][:-4] + ".npy"):
            case_all_data = np.load(self._data[k]['data_file'][:-4] + ".npy", self.memmap_mode)
        else:
            case_all_data = np.load(self._data[k]['data_file'])['data']
        num_color_channels = case_all_data.shape[0] - num_seg
        data_shape = (self.batch_size, num_color_channels, *self.patch_size)
        seg_shape = (self.batch_size, num_seg, *self.patch_size)
        return data_shape, seg_shape

    def get_do_oversample(self, batch_idx):
        return not batch_idx < round(self.batch_size * (1 - self.oversample_foreground_percent))
    
    def get_slice_nb(self, random_slice, case_all_data):
        if random_slice == 0:
            middle_slice = 1
        else:
            middle_slice = random_slice - 1
        #if self.v1:
        #    if random_slice == 0:
        #        middle_slice = 1
        #    else:
        #        middle_slice = random_slice - 1
        #else:
        #    if self.is_val:
        #        if random_slice == 0:
        #            middle_slice = 1
        #        else:
        #            middle_slice = random_slice - 1
        #    else:
        #        if random_slice == 0:
        #            middle_slice = 1
        #        elif random_slice == case_all_data.shape[1] - 1:
        #            middle_slice = case_all_data.shape[1] - 2
        #        else:
        #            p_middle = np.full(shape=(case_all_data.shape[1],), fill_value=0.0)
        #            p_middle[random_slice + 1] = 0.5
        #            p_middle[random_slice - 1] = 0.5
        #            middle_slice = np.random.choice(case_all_data.shape[1], p=p_middle)
        return middle_slice

    def generate_train_batch(self):
        selected_keys = np.random.choice(self.list_of_keys, self.batch_size, True, None)
        three_lists = []
        for i in range(len(selected_keys)):
            if self.v1:
                key_l_1 = key_l_2 = selected_keys[i]
                patient_id = key_l_1[:10]
                filtered = self.un_list_of_keys + self.list_of_keys
                filtered.remove(key_l_1)
                filtered = [x for x in filtered if x[:10] in patient_id]
                key_u = np.random.choice(filtered)
                three_lists.append([key_l_1, key_l_2, key_u, key_u])
            else:
                key_l_1 = selected_keys[i]
                patient_id = key_l_1[:10]
                key_l_2 = [x for x in self.list_of_keys if x[:10] in patient_id and x != key_l_1][0]
                if self.is_val:
                    key_u_1 = min(key_l_1, key_l_2, key=lambda x: int(x[-2:]))
                else:
                    filtered = [x for x in self.un_list_of_keys if x[:10] in patient_id]
                    key_u_1 = np.random.choice(filtered)
                three_lists.append([key_l_1, key_l_2, key_u_1])

        data_l_1 = np.zeros(self.data_shape, dtype=np.float32)
        data_l_2 = np.zeros(self.data_shape, dtype=np.float32)
        data_u_1 = np.zeros(self.data_shape, dtype=np.float32)
        if self.v1:
            data_u_2 = np.zeros(self.data_shape, dtype=np.float32)
        seg_1 = np.zeros(self.seg_shape, dtype=np.float32)
        seg_2 = np.zeros(self.seg_shape, dtype=np.float32)

        case_properties = []
        for j, three_list in enumerate(three_lists):
            key_l_1 = three_list[0]
            key_l_2 = three_list[1]
            key_u_1 = three_list[2]
            if self.v1:
                key_u_2 = three_list[3]

            if 'properties' in self._data[key_l_1].keys():
                properties = self._data[key_l_1]['properties']
            else:
                properties = load_pickle(self._data[key_l_1]['properties_file'])
            case_properties.append(properties)

            if not isfile(self._data[key_l_1]['data_file'][:-4] + ".npy"):
                # lets hope you know what you're doing
                case_all_data_1 = np.load(self._data[key_l_1]['data_file'][:-4] + ".npz")['data']
                case_all_data_2 = np.load(self._data[key_l_2]['data_file'][:-4] + ".npz")['data']
                if '_u' in key_u_1:
                    case_all_data_un_1 = np.load(self.un_data[key_u_1]['data_file'][:-4] + ".npz")['data']
                else:
                    case_all_data_un_1 = np.load(self._data[key_u_1]['data_file'][:-4] + ".npz")['data']
                if self.v1:
                    if '_u' in key_u_2:
                        case_all_data_un_2 = np.load(self.un_data[key_u_2]['data_file'][:-4] + ".npz")['data']
                    else:
                        case_all_data_un_2 = np.load(self._data[key_u_2]['data_file'][:-4] + ".npz")['data']
            else:
                case_all_data_1 = np.load(self._data[key_l_1]['data_file'][:-4] + ".npy", self.memmap_mode)
                case_all_data_2 = np.load(self._data[key_l_2]['data_file'][:-4] + ".npy", self.memmap_mode)
                if '_u' in key_u_1:
                    case_all_data_un_1 = np.load(self.un_data[key_u_1]['data_file'][:-4] + ".npy", self.memmap_mode)
                else:
                    case_all_data_un_1 = np.load(self._data[key_u_1]['data_file'][:-4] + ".npy", self.memmap_mode)
                if self.v1:
                    if '_u' in key_u_2:
                        case_all_data_un_2 = np.load(self.un_data[key_u_2]['data_file'][:-4] + ".npy", self.memmap_mode)
                    else:
                        case_all_data_un_2 = np.load(self._data[key_u_2]['data_file'][:-4] + ".npy", self.memmap_mode)

            # this is for when there is just a 2d slice in case_all_data (2d support)
            if len(case_all_data_1.shape) == 3:
                case_all_data_1 = case_all_data_1[:, None]
                case_all_data_2 = case_all_data_2[:, None]
                case_all_data_un_1 = case_all_data_un_1[:, None]
                if self.v1:
                    case_all_data_un_2 = case_all_data_un_2[:, None]
            
            assert case_all_data_1.shape[1:] == case_all_data_2.shape[1:] == case_all_data_un_1.shape[1:]

            # first select a slice. This can be either random (no force fg) or guaranteed to contain some class
            random_slice = np.random.choice(case_all_data_1.shape[1])
            middle_slice = self.get_slice_nb(random_slice, case_all_data_1)
            if self.v1:
                random_slice2 = np.random.choice(case_all_data_un_1.shape[1])
                middle_slice2 = self.get_slice_nb(random_slice2, case_all_data_un_1)

            # now crop case_all_data to contain just the slice of interest. If we want additional slice above and
            # below the current slice, here is where we get them. We stack those as additional color channels
            if self.pseudo_3d_slices == 1:
                if self.v1:
                    case_all_data_1 = case_all_data_1[:, random_slice]
                    case_all_data_2 = case_all_data_2[:, middle_slice]
                    case_all_data_un_1 = case_all_data_un_1[:, random_slice2]
                    case_all_data_un_2 = case_all_data_un_2[:, middle_slice2]
                    
                else:
                    case_all_data_1 = case_all_data_1[:, random_slice]
                    case_all_data_2 = case_all_data_2[:, random_slice]
                    case_all_data_un_1 = case_all_data_un_1[:, middle_slice]
            else:
                # this is very deprecated and will probably not work anymore. If you intend to use this you need to
                # check this!
                mn = random_slice - (self.pseudo_3d_slices - 1) // 2
                mx = random_slice + (self.pseudo_3d_slices - 1) // 2 + 1
                valid_mn = max(mn, 0)
                valid_mx = min(mx, case_all_data.shape[1])
                case_all_seg = case_all_data[-1:]
                case_all_data = case_all_data[:-1]
                case_all_data = case_all_data[:, valid_mn:valid_mx]
                case_all_seg = case_all_seg[:, random_slice]
                need_to_pad_below = valid_mn - mn
                need_to_pad_above = mx - valid_mx
                if need_to_pad_below > 0:
                    shp_for_pad = np.array(case_all_data.shape)
                    shp_for_pad[1] = need_to_pad_below
                    case_all_data = np.concatenate((np.zeros(shp_for_pad), case_all_data), 1)
                if need_to_pad_above > 0:
                    shp_for_pad = np.array(case_all_data.shape)
                    shp_for_pad[1] = need_to_pad_above
                    case_all_data = np.concatenate((case_all_data, np.zeros(shp_for_pad)), 1)
                case_all_data = case_all_data.reshape((-1, case_all_data.shape[-2], case_all_data.shape[-1]))
                case_all_data = np.concatenate((case_all_data, case_all_seg), 0)

            # case all data should now be (c, x, y)
            assert len(case_all_data_1.shape) == 3
            assert len(case_all_data_2.shape) == 3
            assert len(case_all_data_un_1.shape) == 3

            # we can now choose the bbox from -need_to_pad // 2 to shape - patch_size + need_to_pad // 2. Here we
            # define what the upper and lower bound can be to then sample from them with np.random.randint

            need_to_pad = self.need_to_pad.copy()
            for d in range(2):
                # if case_all_data.shape + need_to_pad is still < patch size we need to pad more! We pad on both sides
                # always
                if need_to_pad[d] + case_all_data_1.shape[d + 1] < self.patch_size[d]:
                    need_to_pad[d] = self.patch_size[d] - case_all_data_1.shape[d + 1]

            shape = case_all_data_1.shape[1:]
            lb_x = - need_to_pad[0] // 2
            ub_x = shape[0] + need_to_pad[0] // 2 + need_to_pad[0] % 2 - self.patch_size[0]
            lb_y = - need_to_pad[1] // 2
            ub_y = shape[1] + need_to_pad[1] // 2 + need_to_pad[1] % 2 - self.patch_size[1]

            # if not force_fg then we can just sample the bbox randomly from lb and ub. Else we need to make sure we get
            # at least one of the foreground classes in the patch
            bbox_x_lb = np.random.randint(lb_x, ub_x + 1)
            bbox_y_lb = np.random.randint(lb_y, ub_y + 1)

            bbox_x_ub = bbox_x_lb + self.patch_size[0]
            bbox_y_ub = bbox_y_lb + self.patch_size[1]

            # whoever wrote this knew what he was doing (hint: it was me). We first crop the data to the region of the
            # bbox that actually lies within the data. This will result in a smaller array which is then faster to pad.
            # valid_bbox is just the coord that lied within the data cube. It will be padded to match the patch size
            # later
            valid_bbox_x_lb = max(0, bbox_x_lb)
            valid_bbox_x_ub = min(shape[0], bbox_x_ub)
            valid_bbox_y_lb = max(0, bbox_y_lb)
            valid_bbox_y_ub = min(shape[1], bbox_y_ub)

            # At this point you might ask yourself why we would treat seg differently from seg_from_previous_stage.
            # Why not just concatenate them here and forget about the if statements? Well that's because segneeds to
            # be padded with -1 constant whereas seg_from_previous_stage needs to be padded with 0s (we could also
            # remove label -1 in the data augmentation but this way it is less error prone)

            case_all_data_un_1 = case_all_data_un_1[:, valid_bbox_x_lb:valid_bbox_x_ub,
                        valid_bbox_y_lb:valid_bbox_y_ub]

            if len(case_all_data_un_1) > 1:
                case_all_data_un_1 = case_all_data_un_1[:-1]
                
            case_all_data_un_1_donly = np.pad(case_all_data_un_1, ((0, 0),
                                                            (-min(0, bbox_x_lb), max(bbox_x_ub - shape[0], 0)),
                                                            (-min(0, bbox_y_lb), max(bbox_y_ub - shape[1], 0))),
                                        self.pad_mode, **self.pad_kwargs_data)
            
            data_u_1[j] = case_all_data_un_1_donly

            if self.v1:
                case_all_data_un_2 = case_all_data_un_2[:, valid_bbox_x_lb:valid_bbox_x_ub,
                            valid_bbox_y_lb:valid_bbox_y_ub]

                if len(case_all_data_un_2) > 1:
                    case_all_data_un_2 = case_all_data_un_2[:-1]

                case_all_data_un_2_donly = np.pad(case_all_data_un_2, ((0, 0),
                                                            (-min(0, bbox_x_lb), max(bbox_x_ub - shape[0], 0)),
                                                            (-min(0, bbox_y_lb), max(bbox_y_ub - shape[1], 0))),
                                        self.pad_mode, **self.pad_kwargs_data)

                data_u_2[j] = case_all_data_un_2_donly

            case_all_data_1 = case_all_data_1[:, valid_bbox_x_lb:valid_bbox_x_ub,
                            valid_bbox_y_lb:valid_bbox_y_ub]
            case_all_data_2 = case_all_data_2[:, valid_bbox_x_lb:valid_bbox_x_ub,
                            valid_bbox_y_lb:valid_bbox_y_ub]

            case_all_data_1_donly = np.pad(case_all_data_1[:-1], ((0, 0),
                                                              (-min(0, bbox_x_lb), max(bbox_x_ub - shape[0], 0)),
                                                              (-min(0, bbox_y_lb), max(bbox_y_ub - shape[1], 0))),
                                         self.pad_mode, **self.pad_kwargs_data)
            case_all_data_2_donly = np.pad(case_all_data_2[:-1], ((0, 0),
                                                              (-min(0, bbox_x_lb), max(bbox_x_ub - shape[0], 0)),
                                                              (-min(0, bbox_y_lb), max(bbox_y_ub - shape[1], 0))),
                                         self.pad_mode, **self.pad_kwargs_data)

            case_all_data_1_segonly = np.pad(case_all_data_1[-1:], ((0, 0),
                                                                (-min(0, bbox_x_lb), max(bbox_x_ub - shape[0], 0)),
                                                                (-min(0, bbox_y_lb), max(bbox_y_ub - shape[1], 0))),
                                           'constant', **{'constant_values': -1})
            case_all_data_2_segonly = np.pad(case_all_data_2[-1:], ((0, 0),
                                                                (-min(0, bbox_x_lb), max(bbox_x_ub - shape[0], 0)),
                                                                (-min(0, bbox_y_lb), max(bbox_y_ub - shape[1], 0))),
                                           'constant', **{'constant_values': -1})

            data_l_1[j] = case_all_data_1_donly
            data_l_2[j] = case_all_data_2_donly
            seg_1[j] = case_all_data_1_segonly
            seg_2[j] = case_all_data_2_segonly

        for i in range(data_l_1.shape[0]):
            assert np.count_nonzero(data_l_1[i]) > 0
            assert np.count_nonzero(data_l_2[i]) > 0
            assert np.count_nonzero(data_u_1[i]) > 0

        keys = selected_keys
        data = [data_l_1, data_l_2, data_u_1]
        if self.v1:
            data.append(data_u_2)
        return {'data': data, 'seg': [seg_1, seg_2], 'properties': case_properties, "keys": keys}


class DataLoaderVideoUnlabeled(SlimDataLoaderBase):
    def __init__(self, data, patch_size, final_patch_size, batch_size, is_val, unlabeled_dataset, video_length, step, force_one_label, oversample_foreground_percent=0.0,
                 memmap_mode="r", pseudo_3d_slices=1, pad_mode="edge",
                 pad_kwargs_data=None, pad_sides=None):
        """
        This is the basic data loader for 2D networks. It uses preprocessed data as produced by my (Fabian) preprocessing.
        You can load the data with load_dataset(folder) where folder is the folder where the npz files are located. If there
        are only npz files present in that folder, the data loader will unpack them on the fly. This may take a while
        and increase CPU usage. Therefore, I advise you to call unpack_dataset(folder) first, which will unpack all npz
        to npy. Don't forget to call delete_npy(folder) after you are done with training?
        Why all the hassle? Well the decathlon dataset is huge. Using npy for everything will consume >1 TB and that is uncool
        given that I (Fabian) will have to store that permanently on /datasets and my local computer. With this strategy all
        data is stored in a compressed format (factor 10 smaller) and only unpacked when needed.
        :param data: get this with load_dataset(folder, stage=0). Plug the return value in here and you are g2g (good to go)
        :param patch_size: what patch size will this data loader return? it is common practice to first load larger
        patches so that a central crop after data augmentation can be done to reduce border artifacts. If unsure, use
        get_patch_size() from data_augmentation.default_data_augmentation
        :param final_patch_size: what will the patch finally be cropped to (after data augmentation)? this is the patch
        size that goes into your network. We need this here because we will pad patients in here so that patches at the
        border of patients are sampled properly
        :param batch_size:
        :param num_batches: how many batches will the data loader produce before stopping? None=endless
        :param seed:
        :param stage: ignore this (Fabian only)
        :param transpose: ignore this
        :param random: sample randomly; CAREFUL! non-random sampling requires batch_size=1, otherwise you will iterate batch_size times over the dataset
        :param pseudo_3d_slices: 7 = 3 below and 3 above the center slice
        """
        super(DataLoaderVideoUnlabeled, self).__init__(data, batch_size, None)
        if pad_kwargs_data is None:
            pad_kwargs_data = OrderedDict()
        self.pad_kwargs_data = pad_kwargs_data
        self.is_val = is_val
        self.pad_mode = pad_mode
        self.pseudo_3d_slices = pseudo_3d_slices
        self.oversample_foreground_percent = oversample_foreground_percent
        self.final_patch_size = final_patch_size
        self.patch_size = patch_size
        self.list_of_keys = list(self._data.keys())
        self.un_list_of_keys = list(unlabeled_dataset.keys())
        self.need_to_pad = np.array(patch_size) - np.array(final_patch_size)
        self.memmap_mode = memmap_mode
        self.video_length = video_length
        self.step = step
        self.percent = None
        self.force_one_label = force_one_label
        if pad_sides is not None:
            if not isinstance(pad_sides, np.ndarray):
                pad_sides = np.array(pad_sides)
            self.need_to_pad += pad_sides
        self.pad_sides = pad_sides
        self.data_shape, self.seg_shape = self.determine_shapes()
        self.un_data = unlabeled_dataset

    def determine_shapes(self):
        num_seg = 1

        k = list(self._data.keys())[0]
        if isfile(self._data[k]['data_file'][:-4] + ".npy"):
            case_all_data = np.load(self._data[k]['data_file'][:-4] + ".npy", self.memmap_mode)
        else:
            case_all_data = np.load(self._data[k]['data_file'])['data']
        num_color_channels = case_all_data.shape[0] - num_seg
        data_shape = (self.batch_size, num_color_channels, *self.patch_size)
        seg_shape = (self.batch_size, num_seg, *self.patch_size)
        return data_shape, seg_shape

    def get_do_oversample(self, batch_idx):
        return not batch_idx < round(self.batch_size * (1 - self.oversample_foreground_percent))
    
    def select_deep_supervision(self, x):
        if self.deep_supervision:
            return x[0]
        else:
            return x
    
    
    def get_slice_nb(self, random_slice, case_all_data):
        if random_slice == 0:
            middle_slice = 1
        else:
            middle_slice = random_slice - 1
        #if self.v1:
        #    if random_slice == 0:
        #        middle_slice = 1
        #    else:
        #        middle_slice = random_slice - 1
        #else:
        #    if self.is_val:
        #        if random_slice == 0:
        #            middle_slice = 1
        #        else:
        #            middle_slice = random_slice - 1
        #    else:
        #        if random_slice == 0:
        #            middle_slice = 1
        #        elif random_slice == case_all_data.shape[1] - 1:
        #            middle_slice = case_all_data.shape[1] - 2
        #        else:
        #            p_middle = np.full(shape=(case_all_data.shape[1],), fill_value=0.0)
        #            p_middle[random_slice + 1] = 0.5
        #            p_middle[random_slice - 1] = 0.5
        #            middle_slice = np.random.choice(case_all_data.shape[1], p=p_middle)
        return middle_slice

    def generate_train_batch(self):
        #eval_idx = None
        selected_keys = np.random.choice(self.list_of_keys, self.batch_size, True, None)
        list_of_frames = []
        for i in range(len(selected_keys)):
            patient_id = selected_keys[i][:10]
            l_filtered = [x for x in self.list_of_keys if x[:10] in patient_id]
            un_filtered = [x for x in self.un_list_of_keys if x[:10] in patient_id]
            filtered = l_filtered + un_filtered
            list_of_frames.append(filtered)

        labeled_binary = np.zeros(shape=(self.video_length, self.batch_size), dtype=bool)
        data = np.zeros(shape=(self.video_length, self.batch_size, self.data_shape[1], self.data_shape[2], self.data_shape[3]))
        seg = np.zeros(shape=(self.video_length, self.batch_size, self.data_shape[1], self.data_shape[2], self.data_shape[3]))
        data.fill(np.nan)
        #registered_idx = np.zeros(shape=(self.batch_size,), dtype=int)

        case_properties = []
        for j, frames in enumerate(list_of_frames):
            labeled_frame = frames[0]
            if 'properties' in self._data[labeled_frame].keys():
                properties = self._data[labeled_frame]['properties']
            else:
                properties = load_pickle(self._data[labeled_frame]['properties_file'])
            case_properties.append(properties)

            depth = properties['size_after_resampling'][0]
            depth_idx = np.random.choice(depth)

            frames = sorted(frames, key=lambda x: int(x[16:18]))
            frames = np.array(frames)
            labeled_idx = np.where(~np.char.endswith(frames, '_u'))[0]

            if self.step > 1:
                if np.all(labeled_idx % 2 == 0):
                    frames = frames[0::self.step]
                elif np.all(labeled_idx % 2 != 0):
                    frames = frames[1::self.step]
                else:
                    start = np.random.randint(0, 2)
                    frames = frames[start::self.step]
                labeled_idx = np.where(~np.char.endswith(frames, '_u'))[0]

            values = np.arange(len(frames))

            if self.is_val:
                before = self.video_length // 2
                after = before + (self.video_length % 2)
                values = np.pad(values, (before, after), mode='wrap')
                mask = np.isin(values, labeled_idx)
                possible_indices = np.argwhere(mask)
                possible_indices = possible_indices[np.logical_and(possible_indices >= before, possible_indices <= len(values) - after)]
                m = np.random.choice(possible_indices)
                #start = min(max(s - step, 0), len(frames) - self.video_length)
                start = m - before
                end = m + after
                assert start >= 0
                assert end <= len(values)
                frame_indices = values[start:end]
                assert len(frame_indices) == self.video_length
                #eval_idx = int(np.where(indices == s)[0])
            else:
                values = np.pad(values, (10000, 10000), mode='wrap')
                windows = sliding_window_view(values, self.video_length)
                if self.force_one_label:
                    mask = np.isin(windows, labeled_idx)
                    mask = np.any(mask, axis=1)
                    windows = windows[mask]
                    np.set_printoptions(threshold=sys.maxsize)
                window_idx = np.random.choice(len(windows))
                frame_indices = windows[window_idx]
                assert len(frame_indices) == self.video_length
            video = frames[frame_indices]

            seed = np.random.randint(0, 4294967296, dtype=np.int64)
            for idx, t in enumerate(video):
                prng = RandomState(seed)

                if '_u' in t:
                    if not isfile(self.un_data[t]['data_file'][:-4] + ".npy"):
                        # lets hope you know what you're doing
                        case_all_data = np.load(self.un_data[t]['data_file'][:-4] + ".npz")['data']
                    else:
                        case_all_data = np.load(self.un_data[t]['data_file'][:-4] + ".npy", self.memmap_mode)
                else:
                    if not isfile(self._data[t]['data_file'][:-4] + ".npy"):
                        # lets hope you know what you're doing
                        case_all_data = np.load(self._data[t]['data_file'][:-4] + ".npz")['data']
                    else:
                        case_all_data = np.load(self._data[t]['data_file'][:-4] + ".npy", self.memmap_mode)

                # this is for when there is just a 2d slice in case_all_data (2d support)
                if len(case_all_data.shape) == 3:
                    case_all_data = case_all_data[:, None]
                
                assert case_all_data.shape[1] == depth

                # now crop case_all_data to contain just the slice of interest. If we want additional slice above and
                # below the current slice, here is where we get them. We stack those as additional color channels
                slice_data = case_all_data[:, depth_idx]

                # case all data should now be (c, x, y)
                assert len(slice_data.shape) == 3

                # we can now choose the bbox from -need_to_pad // 2 to shape - patch_size + need_to_pad // 2. Here we
                # define what the upper and lower bound can be to then sample from them with np.random.randint

                need_to_pad = self.need_to_pad.copy()
                for d in range(2):
                    # if slice_data.shape + need_to_pad is still < patch size we need to pad more! We pad on both sides
                    # always
                    if need_to_pad[d] + slice_data.shape[d + 1] < self.patch_size[d]:
                        need_to_pad[d] = self.patch_size[d] - slice_data.shape[d + 1]

                shape = slice_data.shape[1:]
                lb_x = - need_to_pad[0] // 2
                ub_x = shape[0] + need_to_pad[0] // 2 + need_to_pad[0] % 2 - self.patch_size[0]
                lb_y = - need_to_pad[1] // 2
                ub_y = shape[1] + need_to_pad[1] // 2 + need_to_pad[1] % 2 - self.patch_size[1]

                # if not force_fg then we can just sample the bbox randomly from lb and ub. Else we need to make sure we get
                # at least one of the foreground classes in the patch
                bbox_x_lb = prng.randint(lb_x, ub_x + 1)
                bbox_y_lb = prng.randint(lb_y, ub_y + 1)

                bbox_x_ub = bbox_x_lb + self.patch_size[0]
                bbox_y_ub = bbox_y_lb + self.patch_size[1]

                # whoever wrote this knew what he was doing (hint: it was me). We first crop the data to the region of the
                # bbox that actually lies within the data. This will result in a smaller array which is then faster to pad.
                # valid_bbox is just the coord that lied within the data cube. It will be padded to match the patch size
                # later
                valid_bbox_x_lb = max(0, bbox_x_lb)
                valid_bbox_x_ub = min(shape[0], bbox_x_ub)
                valid_bbox_y_lb = max(0, bbox_y_lb)
                valid_bbox_y_ub = min(shape[1], bbox_y_ub)

                # At this point you might ask yourself why we would treat seg differently from seg_from_previous_stage.
                # Why not just concatenate them here and forget about the if statements? Well that's because segneeds to
                # be padded with -1 constant whereas seg_from_previous_stage needs to be padded with 0s (we could also
                # remove label -1 in the data augmentation but this way it is less error prone)

                slice_data = slice_data[:, valid_bbox_x_lb:valid_bbox_x_ub,
                                valid_bbox_y_lb:valid_bbox_y_ub]
                
                video_idx = idx

                if '_u' in t:
                    slice_data_donly = np.pad(slice_data, ((0, 0),
                                                                (-min(0, bbox_x_lb), max(bbox_x_ub - shape[0], 0)),
                                                                (-min(0, bbox_y_lb), max(bbox_y_ub - shape[1], 0))),
                                            self.pad_mode, **self.pad_kwargs_data)
                else:
                    slice_data_donly = np.pad(slice_data[:-1], ((0, 0),
                                                                    (-min(0, bbox_x_lb), max(bbox_x_ub - shape[0], 0)),
                                                                    (-min(0, bbox_y_lb), max(bbox_y_ub - shape[1], 0))),
                                                self.pad_mode, **self.pad_kwargs_data)

                    slice_data_segonly = np.pad(slice_data[-1:], ((0, 0),
                                                                        (-min(0, bbox_x_lb), max(bbox_x_ub - shape[0], 0)),
                                                                        (-min(0, bbox_y_lb), max(bbox_y_ub - shape[1], 0))),
                                                'constant', **{'constant_values': -1})
                    seg[video_idx, j] = slice_data_segonly
                    labeled_binary[video_idx, j] = True

                data[video_idx, j] = slice_data_donly

        keys = selected_keys
        data = [data[i] for i in range(self.video_length)]
        seg = [seg[i] for i in range(self.video_length)]
        #labeled_binary = [labeled_binary[i] for i in range(len(labeled_binary))]

        #if self.is_val:
        #    matplotlib.use('QtAgg')
        #    print(video_padding)
        #    print(labeled_binary)
        #    fig, ax = plt.subplots(2, self.video_length)
        #    for i in range(self.video_length):
        #        ax[0, i].imshow(data[i][0, 0], cmap='gray')
        #        ax[1, i].imshow(seg[i][0, 0], cmap='gray')
        #    plt.show()

        return {'data': data, 'seg': seg, 'properties': case_properties, "keys": keys, "labeled_binary": labeled_binary}





#class DataLoaderFlowTrain5(SlimDataLoaderBase):
#    def __init__(self, data, patch_size, final_patch_size, batch_size, unlabeled_dataset, video_length, processor, crop_size, is_val, oversample_foreground_percent=0.0,
#                 memmap_mode="r", pseudo_3d_slices=1, pad_mode="edge",
#                 pad_kwargs_data=None, pad_sides=None):
#        """
#        This is the basic data loader for 2D networks. It uses preprocessed data as produced by my (Fabian) preprocessing.
#        You can load the data with load_dataset(folder) where folder is the folder where the npz files are located. If there
#        are only npz files present in that folder, the data loader will unpack them on the fly. This may take a while
#        and increase CPU usage. Therefore, I advise you to call unpack_dataset(folder) first, which will unpack all npz
#        to npy. Don't forget to call delete_npy(folder) after you are done with training?
#        Why all the hassle? Well the decathlon dataset is huge. Using npy for everything will consume >1 TB and that is uncool
#        given that I (Fabian) will have to store that permanently on /datasets and my local computer. With this strategy all
#        data is stored in a compressed format (factor 10 smaller) and only unpacked when needed.
#        :param data: get this with load_dataset(folder, stage=0). Plug the return value in here and you are g2g (good to go)
#        :param patch_size: what patch size will this data loader return? it is common practice to first load larger
#        patches so that a central crop after data augmentation can be done to reduce border artifacts. If unsure, use
#        get_patch_size() from data_augmentation.default_data_augmentation
#        :param final_patch_size: what will the patch finally be cropped to (after data augmentation)? this is the patch
#        size that goes into your network. We need this here because we will pad patients in here so that patches at the
#        border of patients are sampled properly
#        :param batch_size:
#        :param num_batches: how many batches will the data loader produce before stopping? None=endless
#        :param seed:
#        :param stage: ignore this (Fabian only)
#        :param transpose: ignore this
#        :param random: sample randomly; CAREFUL! non-random sampling requires batch_size=1, otherwise you will iterate batch_size times over the dataset
#        :param pseudo_3d_slices: 7 = 3 below and 3 above the center slice
#        """
#        super(DataLoaderFlowTrain5, self).__init__(data, batch_size, None)
#        if pad_kwargs_data is None:
#            pad_kwargs_data = OrderedDict()
#        self.pad_kwargs_data = pad_kwargs_data
#        self.pad_mode = pad_mode
#        self.is_val = is_val
#        self.crop_size = crop_size
#        self.pseudo_3d_slices = pseudo_3d_slices
#        self.oversample_foreground_percent = oversample_foreground_percent
#        self.final_patch_size = final_patch_size
#        self.patch_size = patch_size
#        self.list_of_keys = list(self._data.keys())
#        self.un_list_of_keys = list(unlabeled_dataset.keys())
#        self.need_to_pad = np.array(patch_size) - np.array(final_patch_size)
#        self.memmap_mode = memmap_mode
#        self.video_length = video_length
#        self.percent = None
#        if pad_sides is not None:
#            if not isinstance(pad_sides, np.ndarray):
#                pad_sides = np.array(pad_sides)
#            self.need_to_pad += pad_sides
#        self.pad_sides = pad_sides
#        self.data_shape, self.seg_shape = self.determine_shapes()
#        self.un_data = unlabeled_dataset
#
#        self.preprocessing_transform = self.set_up_preprocessing_pipeline()
#        self.pixel_transform, self.spatial_transform = self.set_up_augmentation_pipeline()
#
#        self.processor = processor
#
#    def determine_shapes(self):
#        num_seg = 1
#
#        k = list(self._data.keys())[0]
#        if isfile(self._data[k]['data_file'][:-4] + ".npy"):
#            case_all_data = np.load(self._data[k]['data_file'][:-4] + ".npy", self.memmap_mode)
#        else:
#            case_all_data = np.load(self._data[k]['data_file'])['data']
#        num_color_channels = case_all_data.shape[0] - num_seg
#        data_shape = (self.batch_size, num_color_channels, *self.patch_size)
#        seg_shape = (self.batch_size, num_seg, *self.patch_size)
#        return data_shape, seg_shape
#
#    def get_do_oversample(self, batch_idx):
#        return not batch_idx < round(self.batch_size * (1 - self.oversample_foreground_percent))
#    
#    def select_deep_supervision(self, x):
#        if self.deep_supervision:
#            return x[0]
#        else:
#            return x
#        
#
#    def my_crop(self, labeled, seg, unlabeled):
#
#        xm = labeled.shape[-1] / 2
#        ym = labeled.shape[-2] / 2
#
#        x1 = int(round(xm - self.final_patch_size[1] / 2))
#        x2 = int(round(xm + self.final_patch_size[1] / 2))
#        y1 = int(round(ym - self.final_patch_size[0] / 2))
#        y2 = int(round(ym + self.final_patch_size[0] / 2))
#
#        labeled = labeled[:, :, y1:y2, x1:x2]
#        seg = seg[:, :, y1:y2, x1:x2]
#
#        xm = unlabeled.shape[-1] / 2
#        ym = unlabeled.shape[-2] / 2
#
#        x1 = int(round(xm - self.final_patch_size[1] / 2))
#        x2 = int(round(xm + self.final_patch_size[1] / 2))
#        y1 = int(round(ym - self.final_patch_size[0] / 2))
#        y2 = int(round(ym + self.final_patch_size[0] / 2))
#
#        unlabeled = unlabeled[:, :, :, y1:y2, x1:x2]
#
#        return labeled, seg, unlabeled
#    
#    def preprocess(self, image, mask):
#        if mask is not None:
#            data = {'image': image, 'mask': mask}
#            transformed = self.preprocessing_transform(data)
#            image = transformed['image']
#            mask = transformed['mask']
#        else:
#            data = {'image': image}
#            transformed = self.preprocessing_transform(data)
#            image = transformed['image']
#
#        return image, mask
#    
#    
#    def augment(self, image, mask, seed):
#        self.pixel_transform = self.pixel_transform.set_random_state(seed=seed)
#        self.spatial_transform = self.spatial_transform.set_random_state(seed=seed)
#        
#        temp = torch.clone(image)
#        padding_mask = image == 0
#
#        if mask is not None:
#            data = {'image': image, 'mask': mask}
#            pixel_transformed = self.pixel_transform(data)
#            image = pixel_transformed['image']
#            mask = pixel_transformed['mask']
#            image[padding_mask] = 0
#            data = {'image': image, 'mask': mask}
#            spatial_transformed = self.spatial_transform(data)
#            image = spatial_transformed['image']
#            mask = spatial_transformed['mask']
#
#            #matplotlib.use('QtAgg')
#            #fig, ax = plt.subplots(1, 3)
#            #ax[0].imshow(temp[0].cpu(), cmap='gray')
#            #ax[1].imshow(image[0].cpu(), cmap='gray')
#            #ax[2].imshow(mask[0].cpu(), cmap='gray')
#            #plt.show()
#        else:
#            data = {'image': image}
#            pixel_transformed = self.pixel_transform(data)
#            image = pixel_transformed['image']
#            image[padding_mask] = 0
#            data = {'image': image}
#            spatial_transformed = self.spatial_transform(data)
#            image = spatial_transformed['image']
#
#            #matplotlib.use('QtAgg')
#            #fig, ax = plt.subplots(1, 2)
#            #ax[0].imshow(temp[0].cpu(), cmap='gray')
#            #ax[1].imshow(image[0].cpu(), cmap='gray')
#            #plt.show()
#
#        return image, mask
#    
#    def set_up_preprocessing_pipeline(self):
#        preprocess = Compose([
#                            T.SpatialPadd(keys=['image', 'mask'], spatial_size=self.final_patch_size, allow_missing_keys=True),
#                            T.CenterSpatialCropd(keys=['image', 'mask'], roi_size=self.final_patch_size, allow_missing_keys=True),
#                            T.ToTensor(dtype=torch.float32, device='cuda:0')
#                            ])
#
#        return preprocess
#    
#    def set_up_augmentation_pipeline(self):
#        spatial_transform = Compose([
#                                    T.RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=0, allow_missing_keys=True),
#                                    T.RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=1, allow_missing_keys=True),
#                                    T.RandRotated(keys=['image', 'mask'], prob=0.2, range_x=np.pi, mode=["bilinear", "nearest"], padding_mode="zeros", allow_missing_keys=True),
#                                    T.RandZoomd(keys=['image', 'mask'], prob=0.2, min_zoom=0.5, max_zoom=1.5, mode=["bilinear", "nearest"], padding_mode="constant", allow_missing_keys=True),
#                                    T.RandAffined(keys=['image', 'mask'], prob=0.2, translate_range=(-26, 26), mode=["bilinear", "nearest"], padding_mode="zeros", allow_missing_keys=True)
#                                    ])
#
#        pixel_transform = Compose([
#                                #RandInvertd(keys=['image'], prob=0.5),
#                                T.RandAdjustContrastd(keys=['image'], prob=0.2, gamma=(0.7, 1.5), allow_missing_keys=True),
#                                T.RandGaussianNoised(keys=['image'], prob=0.2, std=0.04, allow_missing_keys=True),
#                                T.RandScaleIntensityd(keys=['image'], prob=0.2, factors=0.2, allow_missing_keys=True),
#                                T.RandGaussianSmoothd(keys=['image'], prob=0.2, sigma_x=(0.25, 0.5), sigma_y=(0.25, 0.5), allow_missing_keys=True),
#                                T.RandGaussianSharpend(keys=['image'], prob=0.2, sigma1_x=(0.1, 0.2), sigma1_y=(0.1, 0.2), sigma2_x=(0.2, 0.4), sigma2_y=(0.2, 0.4), alpha=(2.0, 3.0), allow_missing_keys=True)
#                                ])
#
#        return pixel_transform, spatial_transform
#
#    def generate_train_batch(self):
#        #eval_idx = None
#        selected_keys = np.random.choice(self.list_of_keys, self.batch_size, True, None)
#        print(selected_keys)
#        list_of_frames = []
#        for i in range(len(selected_keys)):
#            patient_id = selected_keys[i][:10]
#            l_filtered = [x for x in self.list_of_keys if x[:10] in patient_id]
#            un_filtered = [x for x in self.un_list_of_keys if x[:10] in patient_id]
#            filtered = l_filtered + un_filtered
#            list_of_frames.append(filtered)
#        
#        target_list = []
#        unlabeled_list = []
#        padding_need_list = []
#        target_mask_list = []
#
#        case_properties = []
#        for j, frames in enumerate(list_of_frames):
#            labeled_frame = frames[0]
#            if 'properties' in self._data[labeled_frame].keys():
#                properties = self._data[labeled_frame]['properties']
#            else:
#                properties = load_pickle(self._data[labeled_frame]['properties_file'])
#            case_properties.append(properties)
#
#            depth = properties['size_after_resampling'][0]
#            depth_idx = np.random.choice(depth)
#
#            frames = sorted(frames, key=lambda x: int(x[16:18]))
#            frames = np.array(frames)
#            global_labeled_idx = np.where(~np.char.endswith(frames, '_u'))[0]
#            assert len(global_labeled_idx) == 2
#
#            unlabeled = torch.full(size=(self.video_length, 1, *self.final_patch_size), fill_value=torch.nan, device='cuda:0')
#            seg = torch.full(size=(self.video_length, 1, *self.final_patch_size), fill_value=torch.nan, device='cuda:0')
#            cropped_unlabeled = torch.full(size=(self.video_length, 1, self.crop_size, self.crop_size), fill_value=torch.nan, device='cuda:0')
#            cropped_seg = torch.full(size=(self.video_length, 1, self.crop_size, self.crop_size), fill_value=torch.nan, device='cuda:0')
#            target_mask = torch.zeros(size=(self.video_length - 2,), dtype=bool, device='cuda:0')
#
#            possible_indices = np.arange(0, len(frames))
#            possible_indices = set(global_labeled_idx).symmetric_difference(possible_indices)
#            possible_indices = np.array(list(possible_indices))
#
#            random_indices = np.random.choice(possible_indices, size=self.video_length - 2)
#            frame_indices = np.concatenate([global_labeled_idx.reshape((-1,)), random_indices])
#            sorted_indices = np.argsort(frame_indices)
#            frame_indices = frame_indices[sorted_indices]
#            target_mask = torch.cat([torch.tensor([True, True], device='cuda:0').reshape((-1,)), target_mask])
#            target_mask = target_mask[sorted_indices]
#
#            before_where = np.argwhere(frame_indices < global_labeled_idx[0]).reshape(-1,)
#            after_where = np.argwhere(frame_indices >= global_labeled_idx[0]).reshape(-1,)
#
#            before_indices = frame_indices[before_where]
#            after_indices = frame_indices[after_where]
#
#            before_mask = target_mask[before_where]
#            after_mask = target_mask[after_where]
#
#            frame_indices = np.concatenate([after_indices, before_indices])
#            target_mask = torch.cat([after_mask, before_mask])
#            assert frame_indices[0] == global_labeled_idx[0]
#
#            assert len(frame_indices) == self.video_length
#            video = frames[frame_indices]
#
#            labeled_idx = np.where(~np.char.endswith(video, '_u'))[0]
#
#            for frame_idx, t in enumerate(video):
#
#                if '_u' in t:
#                    if not isfile(self.un_data[t]['data_file'][:-4] + ".npy"):
#                        # lets hope you know what you're doing
#                        case_all_data = np.load(self.un_data[t]['data_file'][:-4] + ".npz")['data']
#                    else:
#                        case_all_data = np.load(self.un_data[t]['data_file'][:-4] + ".npy", self.memmap_mode)
#                else:
#                    if not isfile(self._data[t]['data_file'][:-4] + ".npy"):
#                        # lets hope you know what you're doing
#                        case_all_data = np.load(self._data[t]['data_file'][:-4] + ".npz")['data']
#                    else:
#                        case_all_data = np.load(self._data[t]['data_file'][:-4] + ".npy", self.memmap_mode)
#
#                # this is for when there is just a 2d slice in case_all_data (2d support)
#                if len(case_all_data.shape) == 3:
#                    case_all_data = case_all_data[:, None]
#                
#                assert case_all_data.shape[1] == depth
#
#                # now crop case_all_data to contain just the slice of interest. If we want additional slice above and
#                # below the current slice, here is where we get them. We stack those as additional color channels
#                slice_data = case_all_data[:, depth_idx]
#
#                # case all data should now be (c, x, y)
#                assert len(slice_data.shape) == 3
#
#                if '_u' not in t:
#                    image = slice_data[:-1].copy()
#                    mask = slice_data[-1:].copy()
#                    mask[mask < 0] = 0
#                    image = image + 1e-8
#
#                    image, mask = self.preprocess(image, mask)
#
#                    if target_mask[frame_idx] == True:
#                        seg[frame_idx] = mask
#                        assert frame_idx in labeled_idx
#                else:
#                    image = slice_data.copy()
#                    image = image + 1e-8
#
#                    image, _ = self.preprocess(image, None)
#
#                unlabeled[frame_idx] = image
#
#            with torch.no_grad():
#                mean_centroid, _ = self.processor.preprocess_no_registration(data=unlabeled) # T, C(1), H, W
#
#                cropped_unlabeled, padding_need = self.processor.crop_and_pad(data=unlabeled, mean_centroid=mean_centroid)
#                padding_need_list.append(padding_need)
#
#                cropped_seg, _ = self.processor.crop_and_pad(data=seg, mean_centroid=mean_centroid)
#
#            cropped_unlabeled = (cropped_unlabeled - cropped_unlabeled.min()) / (cropped_unlabeled.max() - cropped_unlabeled.min() + 1e-8)
#            cropped_unlabeled = cropped_unlabeled + 1e-8
#
#            assert torch.count_nonzero(torch.any(~torch.isnan(cropped_seg.reshape(self.video_length, -1)), dim=-1)) == 2
#            assert target_mask[0] == True
#
#            if not self.is_val:
#                temp_before_augment = torch.clone(cropped_unlabeled).to('cpu')
#                
#                seed = random.randint(0, 2**32-1)
#
#                for t in range(len(cropped_unlabeled)):
#                    cropped_unlabeled[t], cropped_seg[t] = self.augment(image=cropped_unlabeled[t], mask=cropped_seg[t], seed=seed)
#
#            #matplotlib.use('QtAgg')
#            #fig, ax = plt.subplots(3, len(cropped_unlabeled))
#            #for t in range(len(cropped_unlabeled)):
#            #    ax[0, t].imshow(temp_before_augment[t, 0].cpu(), cmap='gray')
#            #    ax[1, t].imshow(cropped_unlabeled[t, 0].cpu(), cmap='gray')
#            #    ax[2, t].imshow(cropped_seg[t, 0].cpu(), cmap='gray')
#            #plt.show()
#                    
#            cropped_unlabeled = NormalizeIntensity()(cropped_unlabeled)
#
#            unlabeled_list.append(cropped_unlabeled)
#            target_list.append(cropped_seg)
#            target_mask_list.append(target_mask)
#
#        unlabeled = torch.stack(unlabeled_list, dim=1) # T, B, 1, H, W
#        target = torch.stack(target_list, dim=1) # T, B, 1, H, W
#        target_mask = torch.stack(target_mask_list, dim=1) # T, B
#        padding_need = torch.stack(padding_need_list, dim=0) # B, 4
#
#        keys = selected_keys
#
#        #print(seg.shape)
#        #print(np.unique(seg))
#        #matplotlib.use('QtAgg')
#        #fig, ax = plt.subplots(1, 2)
#        #ax[0].imshow(labeled[3, 0], cmap='gray')
#        #ax[1].imshow(seg[3, 0], cmap='gray')
#        #plt.show()
#
#        return {'unlabeled':unlabeled, 
#                'target': target,
#                'padding_need': padding_need,
#                'properties': case_properties, 
#                "keys": keys,
#                'target_mask': target_mask}
                

class DataLoaderFlowTrain5(SlimDataLoaderBase):
    def __init__(self, data, patch_size, final_patch_size, batch_size, unlabeled_dataset, video_length, processor, crop_size, is_val, oversample_foreground_percent=0.0,
                 memmap_mode="r", pseudo_3d_slices=1, pad_mode="edge",
                 pad_kwargs_data=None, pad_sides=None):
        """
        This is the basic data loader for 2D networks. It uses preprocessed data as produced by my (Fabian) preprocessing.
        You can load the data with load_dataset(folder) where folder is the folder where the npz files are located. If there
        are only npz files present in that folder, the data loader will unpack them on the fly. This may take a while
        and increase CPU usage. Therefore, I advise you to call unpack_dataset(folder) first, which will unpack all npz
        to npy. Don't forget to call delete_npy(folder) after you are done with training?
        Why all the hassle? Well the decathlon dataset is huge. Using npy for everything will consume >1 TB and that is uncool
        given that I (Fabian) will have to store that permanently on /datasets and my local computer. With this strategy all
        data is stored in a compressed format (factor 10 smaller) and only unpacked when needed.
        :param data: get this with load_dataset(folder, stage=0). Plug the return value in here and you are g2g (good to go)
        :param patch_size: what patch size will this data loader return? it is common practice to first load larger
        patches so that a central crop after data augmentation can be done to reduce border artifacts. If unsure, use
        get_patch_size() from data_augmentation.default_data_augmentation
        :param final_patch_size: what will the patch finally be cropped to (after data augmentation)? this is the patch
        size that goes into your network. We need this here because we will pad patients in here so that patches at the
        border of patients are sampled properly
        :param batch_size:
        :param num_batches: how many batches will the data loader produce before stopping? None=endless
        :param seed:
        :param stage: ignore this (Fabian only)
        :param transpose: ignore this
        :param random: sample randomly; CAREFUL! non-random sampling requires batch_size=1, otherwise you will iterate batch_size times over the dataset
        :param pseudo_3d_slices: 7 = 3 below and 3 above the center slice
        """
        super(DataLoaderFlowTrain5, self).__init__(data, batch_size, None)
        if pad_kwargs_data is None:
            pad_kwargs_data = OrderedDict()
        self.pad_kwargs_data = pad_kwargs_data
        self.pad_mode = pad_mode
        self.is_val = is_val
        self.crop_size = crop_size
        self.pseudo_3d_slices = pseudo_3d_slices
        self.oversample_foreground_percent = oversample_foreground_percent
        self.final_patch_size = final_patch_size
        self.patch_size = patch_size
        self.list_of_keys = list(self._data.keys())
        self.un_list_of_keys = list(unlabeled_dataset.keys())
        self.need_to_pad = np.array(patch_size) - np.array(final_patch_size)
        self.memmap_mode = memmap_mode
        self.video_length = video_length
        self.percent = None
        if pad_sides is not None:
            if not isinstance(pad_sides, np.ndarray):
                pad_sides = np.array(pad_sides)
            self.need_to_pad += pad_sides
        self.pad_sides = pad_sides
        self.data_shape, self.seg_shape = self.determine_shapes()
        self.un_data = unlabeled_dataset

        self.preprocessing_transform = self.set_up_preprocessing_pipeline()
        self.pixel_transform, self.spatial_transform = self.set_up_augmentation_pipeline()

        self.processor = processor

    def determine_shapes(self):
        num_seg = 1

        k = list(self._data.keys())[0]
        if isfile(self._data[k]['data_file'][:-4] + ".npy"):
            case_all_data = np.load(self._data[k]['data_file'][:-4] + ".npy", self.memmap_mode)
        else:
            case_all_data = np.load(self._data[k]['data_file'])['data']
        num_color_channels = case_all_data.shape[0] - num_seg
        data_shape = (self.batch_size, num_color_channels, *self.patch_size)
        seg_shape = (self.batch_size, num_seg, *self.patch_size)
        return data_shape, seg_shape

    def get_do_oversample(self, batch_idx):
        return not batch_idx < round(self.batch_size * (1 - self.oversample_foreground_percent))
    
    def select_deep_supervision(self, x):
        if self.deep_supervision:
            return x[0]
        else:
            return x
        

    def my_crop(self, labeled, seg, unlabeled):

        xm = labeled.shape[-1] / 2
        ym = labeled.shape[-2] / 2

        x1 = int(round(xm - self.final_patch_size[1] / 2))
        x2 = int(round(xm + self.final_patch_size[1] / 2))
        y1 = int(round(ym - self.final_patch_size[0] / 2))
        y2 = int(round(ym + self.final_patch_size[0] / 2))

        labeled = labeled[:, :, y1:y2, x1:x2]
        seg = seg[:, :, y1:y2, x1:x2]

        xm = unlabeled.shape[-1] / 2
        ym = unlabeled.shape[-2] / 2

        x1 = int(round(xm - self.final_patch_size[1] / 2))
        x2 = int(round(xm + self.final_patch_size[1] / 2))
        y1 = int(round(ym - self.final_patch_size[0] / 2))
        y2 = int(round(ym + self.final_patch_size[0] / 2))

        unlabeled = unlabeled[:, :, :, y1:y2, x1:x2]

        return labeled, seg, unlabeled
    
    def preprocess(self, image, mask):
        if mask is not None:
            data = {'image': image, 'mask': mask}
            transformed = self.preprocessing_transform(data)
            image = transformed['image']
            mask = transformed['mask']
        else:
            data = {'image': image}
            transformed = self.preprocessing_transform(data)
            image = transformed['image']

        return image, mask
    
    
    def augment(self, image, mask, seed):
        self.pixel_transform = self.pixel_transform.set_random_state(seed=seed)
        self.spatial_transform = self.spatial_transform.set_random_state(seed=seed)
        
        temp = torch.clone(image)
        padding_mask = image == 0

        if mask is not None:
            data = {'image': image, 'mask': mask}
            pixel_transformed = self.pixel_transform(data)
            image = pixel_transformed['image']
            mask = pixel_transformed['mask']
            image[padding_mask] = 0
            data = {'image': image, 'mask': mask}
            spatial_transformed = self.spatial_transform(data)
            image = spatial_transformed['image']
            mask = spatial_transformed['mask']

            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(1, 3)
            #ax[0].imshow(temp[0].cpu(), cmap='gray')
            #ax[1].imshow(image[0].cpu(), cmap='gray')
            #ax[2].imshow(mask[0].cpu(), cmap='gray')
            #plt.show()
        else:
            data = {'image': image}
            pixel_transformed = self.pixel_transform(data)
            image = pixel_transformed['image']
            image[padding_mask] = 0
            data = {'image': image}
            spatial_transformed = self.spatial_transform(data)
            image = spatial_transformed['image']

            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(1, 2)
            #ax[0].imshow(temp[0].cpu(), cmap='gray')
            #ax[1].imshow(image[0].cpu(), cmap='gray')
            #plt.show()

        return image, mask
    
    def set_up_preprocessing_pipeline(self):
        preprocess = Compose([
                            T.SpatialPadd(keys=['image', 'mask'], spatial_size=self.final_patch_size, allow_missing_keys=True),
                            T.CenterSpatialCropd(keys=['image', 'mask'], roi_size=self.final_patch_size, allow_missing_keys=True),
                            T.ToTensor(dtype=torch.float32, device='cuda:0')
                            ])

        return preprocess
    
    def set_up_augmentation_pipeline(self):
        spatial_transform = Compose([
                                    T.RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=0, allow_missing_keys=True),
                                    T.RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=1, allow_missing_keys=True),
                                    T.RandRotated(keys=['image', 'mask'], prob=0.2, range_x=np.pi, mode=["bilinear", "nearest"], padding_mode="zeros", allow_missing_keys=True),
                                    T.RandZoomd(keys=['image', 'mask'], prob=0.2, min_zoom=0.5, max_zoom=1.5, mode=["bilinear", "nearest"], padding_mode="constant", allow_missing_keys=True),
                                    T.RandAffined(keys=['image', 'mask'], prob=0.2, translate_range=(-26, 26), mode=["bilinear", "nearest"], padding_mode="zeros", allow_missing_keys=True)
                                    ])

        pixel_transform = Compose([
                                #RandInvertd(keys=['image'], prob=0.5),
                                T.RandAdjustContrastd(keys=['image'], prob=0.2, gamma=(0.7, 1.5), allow_missing_keys=True),
                                T.RandGaussianNoised(keys=['image'], prob=0.2, std=0.04, allow_missing_keys=True),
                                T.RandScaleIntensityd(keys=['image'], prob=0.2, factors=0.2, allow_missing_keys=True),
                                T.RandGaussianSmoothd(keys=['image'], prob=0.2, sigma_x=(0.25, 0.5), sigma_y=(0.25, 0.5), allow_missing_keys=True),
                                T.RandGaussianSharpend(keys=['image'], prob=0.2, sigma1_x=(0.1, 0.2), sigma1_y=(0.1, 0.2), sigma2_x=(0.2, 0.4), sigma2_y=(0.2, 0.4), alpha=(2.0, 3.0), allow_missing_keys=True)
                                ])

        return pixel_transform, spatial_transform

    def generate_train_batch(self):
        #eval_idx = None
        selected_keys = np.random.choice(self.list_of_keys, self.batch_size, True, None)
        list_of_frames = []
        for i in range(len(selected_keys)):
            patient_id = selected_keys[i][:10]
            l_filtered = [x for x in self.list_of_keys if x[:10] in patient_id]
            un_filtered = [x for x in self.un_list_of_keys if x[:10] in patient_id]
            filtered = l_filtered + un_filtered
            list_of_frames.append(filtered)
        
        target_list = []
        unlabeled_list = []
        padding_need_list = []
        target_mask_list = []
        distances_list = []

        case_properties = []
        for j, frames in enumerate(list_of_frames):
            labeled_frame = frames[0]
            if 'properties' in self._data[labeled_frame].keys():
                properties = self._data[labeled_frame]['properties']
            else:
                properties = load_pickle(self._data[labeled_frame]['properties_file'])
            case_properties.append(properties)

            depth = properties['size_after_resampling'][0]
            depth_idx = np.random.choice(depth)

            frames = sorted(frames, key=lambda x: int(x[16:18]))
            frames = np.array(frames)
            global_labeled_idx = np.where(~np.char.endswith(frames, '_u'))[0]
            assert len(global_labeled_idx) == 2

            unlabeled = torch.full(size=(self.video_length, 1, *self.final_patch_size), fill_value=torch.nan, device='cuda:0')
            seg = torch.full(size=(self.video_length, 1, *self.final_patch_size), fill_value=torch.nan, device='cuda:0')
            cropped_unlabeled = torch.full(size=(self.video_length, 1, self.crop_size, self.crop_size), fill_value=torch.nan, device='cuda:0')
            cropped_seg = torch.full(size=(self.video_length, 1, self.crop_size, self.crop_size), fill_value=torch.nan, device='cuda:0')
            target_mask = torch.zeros(size=(self.video_length - 2,), dtype=bool, device='cuda:0')

            possible_indices = np.arange(0, len(frames))
            possible_indices = set(global_labeled_idx).symmetric_difference(possible_indices)
            possible_indices = np.array(list(possible_indices))

            random_indices = np.random.choice(possible_indices, size=self.video_length - 2)
            frame_indices = np.concatenate([global_labeled_idx.reshape((-1,)), random_indices])
            sorted_indices = np.argsort(frame_indices)
            frame_indices = frame_indices[sorted_indices]
            target_mask = torch.cat([torch.tensor([True, True], device='cuda:0').reshape((-1,)), target_mask])
            target_mask = target_mask[sorted_indices]

            before_where = np.argwhere(frame_indices < global_labeled_idx[0]).reshape(-1,)
            after_where = np.argwhere(frame_indices >= global_labeled_idx[0]).reshape(-1,)

            before_indices = frame_indices[before_where]
            after_indices = frame_indices[after_where]

            before_mask = target_mask[before_where]
            after_mask = target_mask[after_where]

            frame_indices = np.concatenate([after_indices, before_indices])
            target_mask = torch.cat([after_mask, before_mask])
            assert frame_indices[0] == global_labeled_idx[0]

            assert len(frame_indices) == self.video_length
            distances_list.append(np.nan)
            video = frames[frame_indices]

            labeled_idx = np.where(~np.char.endswith(video, '_u'))[0]

            for frame_idx, t in enumerate(video):

                if '_u' in t:
                    if not isfile(self.un_data[t]['data_file'][:-4] + ".npy"):
                        # lets hope you know what you're doing
                        case_all_data = np.load(self.un_data[t]['data_file'][:-4] + ".npz")['data']
                    else:
                        case_all_data = np.load(self.un_data[t]['data_file'][:-4] + ".npy", self.memmap_mode)
                else:
                    if not isfile(self._data[t]['data_file'][:-4] + ".npy"):
                        # lets hope you know what you're doing
                        case_all_data = np.load(self._data[t]['data_file'][:-4] + ".npz")['data']
                    else:
                        case_all_data = np.load(self._data[t]['data_file'][:-4] + ".npy", self.memmap_mode)

                # this is for when there is just a 2d slice in case_all_data (2d support)
                if len(case_all_data.shape) == 3:
                    case_all_data = case_all_data[:, None]
                
                assert case_all_data.shape[1] == depth

                # now crop case_all_data to contain just the slice of interest. If we want additional slice above and
                # below the current slice, here is where we get them. We stack those as additional color channels
                slice_data = case_all_data[:, depth_idx]

                # case all data should now be (c, x, y)
                assert len(slice_data.shape) == 3

                if '_u' not in t:
                    image = slice_data[:-1].copy()
                    mask = slice_data[-1:].copy()
                    mask[mask < 0] = 0

                    image, mask = self.preprocess(image, mask)

                    if target_mask[frame_idx] == True:
                        seg[frame_idx] = mask
                        assert frame_idx in labeled_idx
                else:
                    image = slice_data.copy()

                    image, _ = self.preprocess(image, None)

                unlabeled[frame_idx] = image

            with torch.no_grad():
                mean_centroid, _ = self.processor.preprocess_no_registration(data=torch.clone(unlabeled)) # T, C(1), H, W

                cropped_unlabeled, padding_need = self.processor.crop_and_pad(data=unlabeled, mean_centroid=mean_centroid)
                padding_need_list.append(padding_need)

                cropped_seg, _ = self.processor.crop_and_pad(data=seg, mean_centroid=mean_centroid)

            cropped_unlabeled = (cropped_unlabeled - cropped_unlabeled.min()) / (cropped_unlabeled.max() - cropped_unlabeled.min() + 1e-8)

            assert torch.count_nonzero(torch.any(~torch.isnan(cropped_seg.reshape(self.video_length, -1)), dim=-1)) == 2
            assert target_mask[0] == True

            if not self.is_val:
                temp_before_augment = torch.clone(cropped_unlabeled).to('cpu')
                
                seed = random.randint(0, 2**32-1)

                for t in range(len(cropped_unlabeled)):
                    cropped_unlabeled[t], cropped_seg[t] = self.augment(image=cropped_unlabeled[t], mask=cropped_seg[t], seed=seed)

            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(3, len(cropped_unlabeled))
            #for t in range(len(cropped_unlabeled)):
            #    ax[0, t].imshow(temp_before_augment[t, 0].cpu(), cmap='gray')
            #    ax[1, t].imshow(cropped_unlabeled[t, 0].cpu(), cmap='gray')
            #    ax[2, t].imshow(cropped_seg[t, 0].cpu(), cmap='gray')
            #plt.show()
                    
            cropped_unlabeled = NormalizeIntensity()(cropped_unlabeled)

            unlabeled_list.append(cropped_unlabeled)
            target_list.append(cropped_seg)
            target_mask_list.append(target_mask)

        unlabeled = torch.stack(unlabeled_list, dim=1) # T, B, 1, H, W
        target = torch.stack(target_list, dim=1) # T, B, 1, H, W
        target_mask = torch.stack(target_mask_list, dim=1) # T, B
        padding_need = torch.stack(padding_need_list, dim=0) # B, 4
        distances = torch.stack(distances_list, dim=1).float().to('cuda:0') # T - 1, B
        keys = selected_keys

        #print(seg.shape)
        #print(np.unique(seg))
        #matplotlib.use('QtAgg')
        #fig, ax = plt.subplots(1, 2)
        #ax[0].imshow(labeled[3, 0], cmap='gray')
        #ax[1].imshow(seg[3, 0], cmap='gray')
        #plt.show()

        return {'unlabeled':unlabeled, 
                'target': target,
                'padding_need': padding_need,
                'properties': case_properties, 
                "distances": distances,
                'target_mask': target_mask}
    




class DataLoaderFlowTrain5Progressive(SlimDataLoaderBase):
    def __init__(self, data, patch_size, final_patch_size, batch_size, unlabeled_dataset, video_length, processor, crop_size, is_val, oversample_foreground_percent=0.0,
                 memmap_mode="r", pseudo_3d_slices=1, pad_mode="edge",
                 pad_kwargs_data=None, pad_sides=None):
        """
        This is the basic data loader for 2D networks. It uses preprocessed data as produced by my (Fabian) preprocessing.
        You can load the data with load_dataset(folder) where folder is the folder where the npz files are located. If there
        are only npz files present in that folder, the data loader will unpack them on the fly. This may take a while
        and increase CPU usage. Therefore, I advise you to call unpack_dataset(folder) first, which will unpack all npz
        to npy. Don't forget to call delete_npy(folder) after you are done with training?
        Why all the hassle? Well the decathlon dataset is huge. Using npy for everything will consume >1 TB and that is uncool
        given that I (Fabian) will have to store that permanently on /datasets and my local computer. With this strategy all
        data is stored in a compressed format (factor 10 smaller) and only unpacked when needed.
        :param data: get this with load_dataset(folder, stage=0). Plug the return value in here and you are g2g (good to go)
        :param patch_size: what patch size will this data loader return? it is common practice to first load larger
        patches so that a central crop after data augmentation can be done to reduce border artifacts. If unsure, use
        get_patch_size() from data_augmentation.default_data_augmentation
        :param final_patch_size: what will the patch finally be cropped to (after data augmentation)? this is the patch
        size that goes into your network. We need this here because we will pad patients in here so that patches at the
        border of patients are sampled properly
        :param batch_size:
        :param num_batches: how many batches will the data loader produce before stopping? None=endless
        :param seed:
        :param stage: ignore this (Fabian only)
        :param transpose: ignore this
        :param random: sample randomly; CAREFUL! non-random sampling requires batch_size=1, otherwise you will iterate batch_size times over the dataset
        :param pseudo_3d_slices: 7 = 3 below and 3 above the center slice
        """
        super(DataLoaderFlowTrain5Progressive, self).__init__(data, batch_size, None)
        if pad_kwargs_data is None:
            pad_kwargs_data = OrderedDict()
        self.pad_kwargs_data = pad_kwargs_data
        self.pad_mode = pad_mode
        self.is_val = is_val
        self.crop_size = crop_size
        self.pseudo_3d_slices = pseudo_3d_slices
        self.oversample_foreground_percent = oversample_foreground_percent
        self.final_patch_size = final_patch_size
        self.patch_size = patch_size
        self.list_of_keys = list(self._data.keys())
        self.un_list_of_keys = list(unlabeled_dataset.keys())
        self.need_to_pad = np.array(patch_size) - np.array(final_patch_size)
        self.memmap_mode = memmap_mode
        self.video_length = video_length
        self.percent = None
        if pad_sides is not None:
            if not isinstance(pad_sides, np.ndarray):
                pad_sides = np.array(pad_sides)
            self.need_to_pad += pad_sides
        self.pad_sides = pad_sides
        self.data_shape, self.seg_shape = self.determine_shapes()
        self.un_data = unlabeled_dataset

        self.preprocessing_transform = self.set_up_preprocessing_pipeline()
        self.pixel_transform, self.spatial_transform = self.set_up_augmentation_pipeline()

        self.processor = processor

    def determine_shapes(self):
        num_seg = 1

        k = list(self._data.keys())[0]
        if isfile(self._data[k]['data_file'][:-4] + ".npy"):
            case_all_data = np.load(self._data[k]['data_file'][:-4] + ".npy", self.memmap_mode)
        else:
            case_all_data = np.load(self._data[k]['data_file'])['data']
        num_color_channels = case_all_data.shape[0] - num_seg
        data_shape = (self.batch_size, num_color_channels, *self.patch_size)
        seg_shape = (self.batch_size, num_seg, *self.patch_size)
        return data_shape, seg_shape

    def get_do_oversample(self, batch_idx):
        return not batch_idx < round(self.batch_size * (1 - self.oversample_foreground_percent))
    
    def select_deep_supervision(self, x):
        if self.deep_supervision:
            return x[0]
        else:
            return x
        

    def my_crop(self, labeled, seg, unlabeled):

        xm = labeled.shape[-1] / 2
        ym = labeled.shape[-2] / 2

        x1 = int(round(xm - self.final_patch_size[1] / 2))
        x2 = int(round(xm + self.final_patch_size[1] / 2))
        y1 = int(round(ym - self.final_patch_size[0] / 2))
        y2 = int(round(ym + self.final_patch_size[0] / 2))

        labeled = labeled[:, :, y1:y2, x1:x2]
        seg = seg[:, :, y1:y2, x1:x2]

        xm = unlabeled.shape[-1] / 2
        ym = unlabeled.shape[-2] / 2

        x1 = int(round(xm - self.final_patch_size[1] / 2))
        x2 = int(round(xm + self.final_patch_size[1] / 2))
        y1 = int(round(ym - self.final_patch_size[0] / 2))
        y2 = int(round(ym + self.final_patch_size[0] / 2))

        unlabeled = unlabeled[:, :, :, y1:y2, x1:x2]

        return labeled, seg, unlabeled
    
    def preprocess(self, image, mask):
        if mask is not None:
            data = {'image': image, 'mask': mask}
            transformed = self.preprocessing_transform(data)
            image = transformed['image']
            mask = transformed['mask']
        else:
            data = {'image': image}
            transformed = self.preprocessing_transform(data)
            image = transformed['image']

        return image, mask
    
    
    def augment(self, image, mask, seed):
        self.pixel_transform = self.pixel_transform.set_random_state(seed=seed)
        self.spatial_transform = self.spatial_transform.set_random_state(seed=seed)
        
        temp = torch.clone(image)
        padding_mask = image == 0

        if mask is not None:
            data = {'image': image, 'mask': mask}
            pixel_transformed = self.pixel_transform(data)
            image = pixel_transformed['image']
            mask = pixel_transformed['mask']
            image[padding_mask] = 0
            data = {'image': image, 'mask': mask}
            spatial_transformed = self.spatial_transform(data)
            image = spatial_transformed['image']
            mask = spatial_transformed['mask']

            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(1, 3)
            #ax[0].imshow(temp[0].cpu(), cmap='gray')
            #ax[1].imshow(image[0].cpu(), cmap='gray')
            #ax[2].imshow(mask[0].cpu(), cmap='gray')
            #plt.show()
        else:
            data = {'image': image}
            pixel_transformed = self.pixel_transform(data)
            image = pixel_transformed['image']
            image[padding_mask] = 0
            data = {'image': image}
            spatial_transformed = self.spatial_transform(data)
            image = spatial_transformed['image']

            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(1, 2)
            #ax[0].imshow(temp[0].cpu(), cmap='gray')
            #ax[1].imshow(image[0].cpu(), cmap='gray')
            #plt.show()

        return image, mask
    
    def set_up_preprocessing_pipeline(self):
        preprocess = Compose([
                            T.SpatialPadd(keys=['image', 'mask'], spatial_size=self.final_patch_size, allow_missing_keys=True),
                            T.CenterSpatialCropd(keys=['image', 'mask'], roi_size=self.final_patch_size, allow_missing_keys=True),
                            T.ToTensor(dtype=torch.float32, device='cuda:0')
                            ])

        return preprocess
    
    def set_up_augmentation_pipeline(self):
        spatial_transform = Compose([
                                    T.RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=0, allow_missing_keys=True),
                                    T.RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=1, allow_missing_keys=True),
                                    T.RandRotated(keys=['image', 'mask'], prob=0.2, range_x=np.pi, mode=["bilinear", "nearest"], padding_mode="zeros", allow_missing_keys=True),
                                    T.RandZoomd(keys=['image', 'mask'], prob=0.2, min_zoom=0.5, max_zoom=1.5, mode=["bilinear", "nearest"], padding_mode="constant", allow_missing_keys=True),
                                    T.RandAffined(keys=['image', 'mask'], prob=0.2, translate_range=(-26, 26), mode=["bilinear", "nearest"], padding_mode="zeros", allow_missing_keys=True)
                                    ])

        pixel_transform = Compose([
                                #RandInvertd(keys=['image'], prob=0.5),
                                T.RandAdjustContrastd(keys=['image'], prob=0.2, gamma=(0.7, 1.5), allow_missing_keys=True),
                                T.RandGaussianNoised(keys=['image'], prob=0.2, std=0.04, allow_missing_keys=True),
                                T.RandScaleIntensityd(keys=['image'], prob=0.2, factors=0.2, allow_missing_keys=True),
                                T.RandGaussianSmoothd(keys=['image'], prob=0.2, sigma_x=(0.25, 0.5), sigma_y=(0.25, 0.5), allow_missing_keys=True),
                                T.RandGaussianSharpend(keys=['image'], prob=0.2, sigma1_x=(0.1, 0.2), sigma1_y=(0.1, 0.2), sigma2_x=(0.2, 0.4), sigma2_y=(0.2, 0.4), alpha=(2.0, 3.0), allow_missing_keys=True)
                                ])

        return pixel_transform, spatial_transform

    def generate_train_batch(self):
        #eval_idx = None
        selected_keys = np.random.choice(self.list_of_keys, self.batch_size, True, None)
        list_of_frames = []
        for i in range(len(selected_keys)):
            patient_id = selected_keys[i][:10]
            l_filtered = [x for x in self.list_of_keys if x[:10] in patient_id]
            un_filtered = [x for x in self.un_list_of_keys if x[:10] in patient_id]
            filtered = l_filtered + un_filtered
            list_of_frames.append(filtered)
        
        target_list = []
        unlabeled_list = []
        padding_need_list = []
        target_mask_list = []

        case_properties = []
        for j, frames in enumerate(list_of_frames):
            labeled_frame = frames[0]
            if 'properties' in self._data[labeled_frame].keys():
                properties = self._data[labeled_frame]['properties']
            else:
                properties = load_pickle(self._data[labeled_frame]['properties_file'])
            case_properties.append(properties)

            depth = properties['size_after_resampling'][0]
            depth_idx = np.random.choice(depth)

            frames = sorted(frames, key=lambda x: int(x[16:18]))
            frames = np.array(frames)
            global_labeled_idx = np.where(~np.char.endswith(frames, '_u'))[0]
            ed_idx = global_labeled_idx[0]
            es_idx = global_labeled_idx[1]
            assert len(global_labeled_idx) == 2

            unlabeled = torch.full(size=(self.video_length, 1, *self.final_patch_size), fill_value=torch.nan, device='cuda:0')
            seg = torch.full(size=(self.video_length, 1, *self.final_patch_size), fill_value=torch.nan, device='cuda:0')
            cropped_unlabeled = torch.full(size=(self.video_length, 1, self.crop_size, self.crop_size), fill_value=torch.nan, device='cuda:0')
            cropped_seg = torch.full(size=(self.video_length, 1, self.crop_size, self.crop_size), fill_value=torch.nan, device='cuda:0')

            possible_indices = np.arange(len(frames))

            before_where = np.argwhere(possible_indices < ed_idx).reshape(-1,)
            after_where = np.argwhere(possible_indices >= ed_idx).reshape(-1,)

            before_indices = possible_indices[before_where]
            after_indices = possible_indices[after_where]

            possible_indices = np.concatenate([after_indices, before_indices])

            stop_idx = np.argwhere(possible_indices == es_idx)[0][0]
            chunk1 = possible_indices[:stop_idx + 1]
            chunk2 = possible_indices[stop_idx:]
            chunk2 = np.concatenate([np.array([possible_indices[0]]), chunk2[::-1]])

            r = np.random.randint(0, 2)
            if r == 0:
                possible_indices = chunk1
            elif r == 1:
                possible_indices = chunk2
            
            assert possible_indices[0] == ed_idx
            assert possible_indices[-1] == es_idx

            random_indices = np.random.choice(np.arange(0, len(possible_indices)), size=self.video_length - 2)
            target_mask = np.concatenate([np.array([True]), np.full_like(random_indices, fill_value=False), np.array([True])]).astype(bool)
            random_indices = np.concatenate([np.array([0]), random_indices, np.array([len(possible_indices) - 1])])
            sorted_indices = np.argsort(random_indices)

            random_indices = random_indices[sorted_indices]
            target_mask = target_mask[sorted_indices]
            target_mask = torch.from_numpy(target_mask).to('cuda:0')

            frame_indices = possible_indices[random_indices]

            assert len(frame_indices) == self.video_length
            video = frames[frame_indices]

            labeled_idx = np.where(~np.char.endswith(video, '_u'))[0]

            for frame_idx, t in enumerate(video):

                if '_u' in t:
                    if not isfile(self.un_data[t]['data_file'][:-4] + ".npy"):
                        # lets hope you know what you're doing
                        case_all_data = np.load(self.un_data[t]['data_file'][:-4] + ".npz")['data']
                    else:
                        case_all_data = np.load(self.un_data[t]['data_file'][:-4] + ".npy", self.memmap_mode)
                else:
                    if not isfile(self._data[t]['data_file'][:-4] + ".npy"):
                        # lets hope you know what you're doing
                        case_all_data = np.load(self._data[t]['data_file'][:-4] + ".npz")['data']
                    else:
                        case_all_data = np.load(self._data[t]['data_file'][:-4] + ".npy", self.memmap_mode)

                # this is for when there is just a 2d slice in case_all_data (2d support)
                if len(case_all_data.shape) == 3:
                    case_all_data = case_all_data[:, None]
                
                assert case_all_data.shape[1] == depth

                # now crop case_all_data to contain just the slice of interest. If we want additional slice above and
                # below the current slice, here is where we get them. We stack those as additional color channels
                slice_data = case_all_data[:, depth_idx]

                # case all data should now be (c, x, y)
                assert len(slice_data.shape) == 3

                if '_u' not in t:
                    image = slice_data[:-1].copy()
                    mask = slice_data[-1:].copy()
                    mask[mask < 0] = 0

                    image, mask = self.preprocess(image, mask)

                    if target_mask[frame_idx] == True:
                        seg[frame_idx] = mask
                        assert frame_idx in labeled_idx
                else:
                    image = slice_data.copy()

                    image, _ = self.preprocess(image, None)

                unlabeled[frame_idx] = image

            with torch.no_grad():
                mean_centroid, _ = self.processor.preprocess_no_registration(data=torch.clone(unlabeled)) # T, C(1), H, W

                cropped_unlabeled, padding_need = self.processor.crop_and_pad(data=unlabeled, mean_centroid=mean_centroid)
                padding_need_list.append(padding_need)

                cropped_seg, _ = self.processor.crop_and_pad(data=seg, mean_centroid=mean_centroid)

            cropped_unlabeled = (cropped_unlabeled - cropped_unlabeled.min()) / (cropped_unlabeled.max() - cropped_unlabeled.min() + 1e-8)

            assert torch.count_nonzero(torch.any(~torch.isnan(cropped_seg.reshape(self.video_length, -1)), dim=-1)) == 2
            assert target_mask[0] == True

            if not self.is_val:
                temp_before_augment = torch.clone(cropped_unlabeled).to('cpu')
                
                seed = random.randint(0, 2**32-1)

                for t in range(len(cropped_unlabeled)):
                    cropped_unlabeled[t], cropped_seg[t] = self.augment(image=cropped_unlabeled[t], mask=cropped_seg[t], seed=seed)

            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(3, len(cropped_unlabeled))
            #for t in range(len(cropped_unlabeled)):
            #    ax[0, t].imshow(temp_before_augment[t, 0].cpu(), cmap='gray')
            #    ax[1, t].imshow(cropped_unlabeled[t, 0].cpu(), cmap='gray')
            #    ax[2, t].imshow(cropped_seg[t, 0].cpu(), cmap='gray')
            #plt.show()
                    
            cropped_unlabeled = NormalizeIntensity()(cropped_unlabeled)

            unlabeled_list.append(cropped_unlabeled)
            target_list.append(cropped_seg)
            target_mask_list.append(target_mask)

        unlabeled = torch.stack(unlabeled_list, dim=1) # T, B, 1, H, W
        target = torch.stack(target_list, dim=1) # T, B, 1, H, W
        target_mask = torch.stack(target_mask_list, dim=1) # T, B
        padding_need = torch.stack(padding_need_list, dim=0) # B, 4
        keys = selected_keys


        #print(seg.shape)
        #print(np.unique(seg))
        #matplotlib.use('QtAgg')
        #fig, ax = plt.subplots(1, 2)
        #ax[0].imshow(labeled[3, 0], cmap='gray')
        #ax[1].imshow(seg[3, 0], cmap='gray')
        #plt.show()

        return {'unlabeled':unlabeled, 
                'target': target,
                'in_between': None,
                'padding_need': padding_need,
                'properties': case_properties,
                'target_mask': target_mask}



class DataLoaderFlowACDCProgressiveAllDataFirst(SlimDataLoaderBase):
    def __init__(self, data, patch_size, final_patch_size, batch_size, unlabeled_dataset, video_length, processor, crop_size, is_val, oversample_foreground_percent=0.0,
                 memmap_mode="r", pseudo_3d_slices=1, pad_mode="edge",
                 pad_kwargs_data=None, pad_sides=None):
        """
        This is the basic data loader for 2D networks. It uses preprocessed data as produced by my (Fabian) preprocessing.
        You can load the data with load_dataset(folder) where folder is the folder where the npz files are located. If there
        are only npz files present in that folder, the data loader will unpack them on the fly. This may take a while
        and increase CPU usage. Therefore, I advise you to call unpack_dataset(folder) first, which will unpack all npz
        to npy. Don't forget to call delete_npy(folder) after you are done with training?
        Why all the hassle? Well the decathlon dataset is huge. Using npy for everything will consume >1 TB and that is uncool
        given that I (Fabian) will have to store that permanently on /datasets and my local computer. With this strategy all
        data is stored in a compressed format (factor 10 smaller) and only unpacked when needed.
        :param data: get this with load_dataset(folder, stage=0). Plug the return value in here and you are g2g (good to go)
        :param patch_size: what patch size will this data loader return? it is common practice to first load larger
        patches so that a central crop after data augmentation can be done to reduce border artifacts. If unsure, use
        get_patch_size() from data_augmentation.default_data_augmentation
        :param final_patch_size: what will the patch finally be cropped to (after data augmentation)? this is the patch
        size that goes into your network. We need this here because we will pad patients in here so that patches at the
        border of patients are sampled properly
        :param batch_size:
        :param num_batches: how many batches will the data loader produce before stopping? None=endless
        :param seed:
        :param stage: ignore this (Fabian only)
        :param transpose: ignore this
        :param random: sample randomly; CAREFUL! non-random sampling requires batch_size=1, otherwise you will iterate batch_size times over the dataset
        :param pseudo_3d_slices: 7 = 3 below and 3 above the center slice
        """
        super(DataLoaderFlowACDCProgressiveAllDataFirst, self).__init__(data, batch_size, None)
        if pad_kwargs_data is None:
            pad_kwargs_data = OrderedDict()
        self.pad_kwargs_data = pad_kwargs_data
        self.pad_mode = pad_mode
        self.is_val = is_val
        self.crop_size = crop_size
        self.pseudo_3d_slices = pseudo_3d_slices
        self.oversample_foreground_percent = oversample_foreground_percent
        self.final_patch_size = final_patch_size
        self.patch_size = patch_size
        self.list_of_keys = list(self._data.keys())
        self.un_list_of_keys = list(unlabeled_dataset.keys())
        self.need_to_pad = np.array(patch_size) - np.array(final_patch_size)
        self.memmap_mode = memmap_mode
        self.video_length = video_length
        self.percent = None
        if pad_sides is not None:
            if not isinstance(pad_sides, np.ndarray):
                pad_sides = np.array(pad_sides)
            self.need_to_pad += pad_sides
        self.pad_sides = pad_sides
        self.data_shape, self.seg_shape = self.determine_shapes()
        self.un_data = unlabeled_dataset

        self.preprocessing_transform = self.set_up_preprocessing_pipeline()
        self.pixel_transform, self.spatial_transform = self.set_up_augmentation_pipeline()

        self.processor = processor

    def determine_shapes(self):
        num_seg = 1

        k = list(self._data.keys())[0]
        if isfile(self._data[k]['data_file'][:-4] + ".npy"):
            case_all_data = np.load(self._data[k]['data_file'][:-4] + ".npy", self.memmap_mode)
        else:
            case_all_data = np.load(self._data[k]['data_file'])['data']
        num_color_channels = case_all_data.shape[0] - num_seg
        data_shape = (self.batch_size, num_color_channels, *self.patch_size)
        seg_shape = (self.batch_size, num_seg, *self.patch_size)
        return data_shape, seg_shape

    def get_do_oversample(self, batch_idx):
        return not batch_idx < round(self.batch_size * (1 - self.oversample_foreground_percent))
    
    def select_deep_supervision(self, x):
        if self.deep_supervision:
            return x[0]
        else:
            return x
        

    def my_crop(self, labeled, seg, unlabeled):

        xm = labeled.shape[-1] / 2
        ym = labeled.shape[-2] / 2

        x1 = int(round(xm - self.final_patch_size[1] / 2))
        x2 = int(round(xm + self.final_patch_size[1] / 2))
        y1 = int(round(ym - self.final_patch_size[0] / 2))
        y2 = int(round(ym + self.final_patch_size[0] / 2))

        labeled = labeled[:, :, y1:y2, x1:x2]
        seg = seg[:, :, y1:y2, x1:x2]

        xm = unlabeled.shape[-1] / 2
        ym = unlabeled.shape[-2] / 2

        x1 = int(round(xm - self.final_patch_size[1] / 2))
        x2 = int(round(xm + self.final_patch_size[1] / 2))
        y1 = int(round(ym - self.final_patch_size[0] / 2))
        y2 = int(round(ym + self.final_patch_size[0] / 2))

        unlabeled = unlabeled[:, :, :, y1:y2, x1:x2]

        return labeled, seg, unlabeled
    
    def preprocess(self, image, mask):
        if mask is not None:
            data = {'image': image, 'mask': mask}
            transformed = self.preprocessing_transform(data)
            image = transformed['image']
            mask = transformed['mask']
        else:
            data = {'image': image}
            transformed = self.preprocessing_transform(data)
            image = transformed['image']

        return image, mask
    
    
    def augment(self, image, mask, seed):
        self.pixel_transform = self.pixel_transform.set_random_state(seed=seed)
        self.spatial_transform = self.spatial_transform.set_random_state(seed=seed)
        
        temp = torch.clone(image)
        padding_mask = image == 0

        if mask is not None:
            data = {'image': image, 'mask': mask}
            pixel_transformed = self.pixel_transform(data)
            image = pixel_transformed['image']
            mask = pixel_transformed['mask']
            image[padding_mask] = 0
            data = {'image': image, 'mask': mask}
            spatial_transformed = self.spatial_transform(data)
            image = spatial_transformed['image']
            mask = spatial_transformed['mask']

            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(1, 3)
            #ax[0].imshow(temp[0].cpu(), cmap='gray')
            #ax[1].imshow(image[0].cpu(), cmap='gray')
            #ax[2].imshow(mask[0].cpu(), cmap='gray')
            #plt.show()
        else:
            data = {'image': image}
            pixel_transformed = self.pixel_transform(data)
            image = pixel_transformed['image']
            image[padding_mask] = 0
            data = {'image': image}
            spatial_transformed = self.spatial_transform(data)
            image = spatial_transformed['image']

            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(1, 2)
            #ax[0].imshow(temp[0].cpu(), cmap='gray')
            #ax[1].imshow(image[0].cpu(), cmap='gray')
            #plt.show()

        return image, mask
    
    def set_up_preprocessing_pipeline(self):
        preprocess = Compose([
                            T.SpatialPadd(keys=['image', 'mask'], spatial_size=self.final_patch_size, allow_missing_keys=True),
                            T.CenterSpatialCropd(keys=['image', 'mask'], roi_size=self.final_patch_size, allow_missing_keys=True),
                            T.ToTensor(dtype=torch.float32, device='cuda:0')
                            ])

        return preprocess
    
    def set_up_augmentation_pipeline(self):
        spatial_transform = Compose([
                                    T.RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=0, allow_missing_keys=True),
                                    T.RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=1, allow_missing_keys=True),
                                    T.RandRotated(keys=['image', 'mask'], prob=0.2, range_x=np.pi, mode=["bilinear", "nearest"], padding_mode="zeros", allow_missing_keys=True),
                                    T.RandZoomd(keys=['image', 'mask'], prob=0.2, min_zoom=0.5, max_zoom=1.5, mode=["bilinear", "nearest"], padding_mode="constant", allow_missing_keys=True),
                                    T.RandAffined(keys=['image', 'mask'], prob=0.2, translate_range=(-26, 26), mode=["bilinear", "nearest"], padding_mode="zeros", allow_missing_keys=True)
                                    ])

        pixel_transform = Compose([
                                #RandInvertd(keys=['image'], prob=0.5),
                                T.RandAdjustContrastd(keys=['image'], prob=0.2, gamma=(0.7, 1.5), allow_missing_keys=True),
                                T.RandGaussianNoised(keys=['image'], prob=0.2, std=0.04, allow_missing_keys=True),
                                T.RandScaleIntensityd(keys=['image'], prob=0.2, factors=0.2, allow_missing_keys=True),
                                T.RandGaussianSmoothd(keys=['image'], prob=0.2, sigma_x=(0.25, 0.5), sigma_y=(0.25, 0.5), allow_missing_keys=True),
                                T.RandGaussianSharpend(keys=['image'], prob=0.2, sigma1_x=(0.1, 0.2), sigma1_y=(0.1, 0.2), sigma2_x=(0.2, 0.4), sigma2_y=(0.2, 0.4), alpha=(2.0, 3.0), allow_missing_keys=True)
                                ])

        return pixel_transform, spatial_transform

    def generate_train_batch(self):
        #eval_idx = None
        selected_keys = np.random.choice(self.list_of_keys, self.batch_size, True, None)
        list_of_frames = []
        for i in range(len(selected_keys)):
            patient_id = selected_keys[i][:10]
            l_filtered = [x for x in self.list_of_keys if x[:10] in patient_id]
            un_filtered = [x for x in self.un_list_of_keys if x[:10] in patient_id]
            filtered = l_filtered + un_filtered
            list_of_frames.append(filtered)
        
        target_list = []
        unlabeled_list = []
        padding_need_list = []
        target_mask_list = []

        case_properties = []
        for j, frames in enumerate(list_of_frames):
            labeled_frame = frames[0]
            if 'properties' in self._data[labeled_frame].keys():
                properties = self._data[labeled_frame]['properties']
            else:
                properties = load_pickle(self._data[labeled_frame]['properties_file'])
            case_properties.append(properties)

            depth = properties['size_after_resampling'][0]
            depth_idx = np.random.choice(depth)

            frames = sorted(frames, key=lambda x: int(x[16:18]))
            frames = np.array(frames)
            global_labeled_idx = np.where(~np.char.endswith(frames, '_u'))[0]
            ed_idx = global_labeled_idx[0]
            es_idx = global_labeled_idx[1]
            assert len(global_labeled_idx) == 2

            unlabeled = torch.full(size=(self.video_length, 1, *self.final_patch_size), fill_value=torch.nan, device='cuda:0')
            seg = torch.full(size=(self.video_length, 1, *self.final_patch_size), fill_value=torch.nan, device='cuda:0')
            cropped_unlabeled = torch.full(size=(self.video_length, 1, self.crop_size, self.crop_size), fill_value=torch.nan, device='cuda:0')
            cropped_seg = torch.full(size=(self.video_length, 1, self.crop_size, self.crop_size), fill_value=torch.nan, device='cuda:0')

            possible_indices = np.arange(len(frames))

            before_where = np.argwhere(possible_indices < ed_idx).reshape(-1,)
            after_where = np.argwhere(possible_indices >= ed_idx).reshape(-1,)

            before_indices = possible_indices[before_where]
            after_indices = possible_indices[after_where]

            possible_indices = np.concatenate([after_indices, before_indices])

            stop_idx = np.argwhere(possible_indices == es_idx)[0][0]
            chunk1 = possible_indices[:stop_idx + 1]
            chunk2 = possible_indices[stop_idx:]
            chunk2 = np.concatenate([np.array([possible_indices[0]]), chunk2[::-1]])

            r = np.random.randint(0, 2)
            if r == 0:
                possible_indices = chunk1
            elif r == 1:
                possible_indices = chunk2
            
            assert possible_indices[0] == ed_idx
            assert possible_indices[-1] == es_idx

            random_indices = np.random.choice(np.arange(1, len(possible_indices)), size=1)
            target_mask = np.concatenate([np.array([True]), np.full_like(random_indices, fill_value=False)]).astype(bool)
            random_indices = np.concatenate([np.array([0]), random_indices])
            sorted_indices = np.argsort(random_indices)

            random_indices = random_indices[sorted_indices]
            target_mask = target_mask[sorted_indices]
            target_mask = torch.from_numpy(target_mask).to('cuda:0')

            frame_indices = possible_indices[random_indices]

            assert len(frame_indices) == self.video_length
            video = frames[frame_indices]

            labeled_idx = np.where(~np.char.endswith(video, '_u'))[0]

            for frame_idx, t in enumerate(video):

                if '_u' in t:
                    if not isfile(self.un_data[t]['data_file'][:-4] + ".npy"):
                        # lets hope you know what you're doing
                        case_all_data = np.load(self.un_data[t]['data_file'][:-4] + ".npz")['data']
                    else:
                        case_all_data = np.load(self.un_data[t]['data_file'][:-4] + ".npy", self.memmap_mode)
                else:
                    if not isfile(self._data[t]['data_file'][:-4] + ".npy"):
                        # lets hope you know what you're doing
                        case_all_data = np.load(self._data[t]['data_file'][:-4] + ".npz")['data']
                    else:
                        case_all_data = np.load(self._data[t]['data_file'][:-4] + ".npy", self.memmap_mode)

                # this is for when there is just a 2d slice in case_all_data (2d support)
                if len(case_all_data.shape) == 3:
                    case_all_data = case_all_data[:, None]
                
                assert case_all_data.shape[1] == depth

                # now crop case_all_data to contain just the slice of interest. If we want additional slice above and
                # below the current slice, here is where we get them. We stack those as additional color channels
                slice_data = case_all_data[:, depth_idx]

                # case all data should now be (c, x, y)
                assert len(slice_data.shape) == 3

                if '_u' not in t:
                    image = slice_data[:-1].copy()
                    mask = slice_data[-1:].copy()
                    mask[mask < 0] = 0

                    image, mask = self.preprocess(image, mask)

                    if target_mask[frame_idx] == True:
                        seg[frame_idx] = mask
                        assert frame_idx in labeled_idx
                else:
                    image = slice_data.copy()

                    image, _ = self.preprocess(image, None)

                unlabeled[frame_idx] = image

            with torch.no_grad():
                mean_centroid, _ = self.processor.preprocess_no_registration(data=torch.clone(unlabeled)) # T, C(1), H, W

                cropped_unlabeled, padding_need = self.processor.crop_and_pad(data=unlabeled, mean_centroid=mean_centroid)
                padding_need_list.append(padding_need)

                cropped_seg, _ = self.processor.crop_and_pad(data=seg, mean_centroid=mean_centroid)

            cropped_unlabeled = (cropped_unlabeled - cropped_unlabeled.min()) / (cropped_unlabeled.max() - cropped_unlabeled.min() + 1e-8)

            assert torch.count_nonzero(torch.any(~torch.isnan(cropped_seg.reshape(self.video_length, -1)), dim=-1)) == 1
            assert target_mask[0] == True

            if not self.is_val:
                temp_before_augment = torch.clone(cropped_unlabeled).to('cpu')
                
                seed = random.randint(0, 2**32-1)

                for t in range(len(cropped_unlabeled)):
                    cropped_unlabeled[t], cropped_seg[t] = self.augment(image=cropped_unlabeled[t], mask=cropped_seg[t], seed=seed)

            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(3, len(cropped_unlabeled))
            #for t in range(len(cropped_unlabeled)):
            #    ax[0, t].imshow(temp_before_augment[t, 0].cpu(), cmap='gray')
            #    ax[1, t].imshow(cropped_unlabeled[t, 0].cpu(), cmap='gray')
            #    ax[2, t].imshow(cropped_seg[t, 0].cpu(), cmap='gray')
            #plt.show()
                    
            cropped_unlabeled = NormalizeIntensity()(cropped_unlabeled)

            unlabeled_list.append(cropped_unlabeled)
            target_list.append(cropped_seg)
            target_mask_list.append(target_mask)

        unlabeled = torch.stack(unlabeled_list, dim=1) # T, B, 1, H, W
        target = torch.stack(target_list, dim=1) # T, B, 1, H, W
        target_mask = torch.stack(target_mask_list, dim=1) # T, B
        padding_need = torch.stack(padding_need_list, dim=0) # B, 4
        keys = selected_keys


        #print(seg.shape)
        #print(np.unique(seg))
        #matplotlib.use('QtAgg')
        #fig, ax = plt.subplots(1, 2)
        #ax[0].imshow(labeled[3, 0], cmap='gray')
        #ax[1].imshow(seg[3, 0], cmap='gray')
        #plt.show()

        return {'unlabeled':unlabeled, 
                'target': target,
                'in_between': None,
                'padding_need': padding_need,
                'properties': case_properties,
                'target_mask': target_mask}
    



class DataLoaderFlowACDCProgressiveAllDataAdjacent(SlimDataLoaderBase):
    def __init__(self, data, patch_size, final_patch_size, batch_size, unlabeled_dataset, video_length, processor, crop_size, is_val, oversample_foreground_percent=0.0,
                 memmap_mode="r", pseudo_3d_slices=1, pad_mode="edge",
                 pad_kwargs_data=None, pad_sides=None):
        """
        This is the basic data loader for 2D networks. It uses preprocessed data as produced by my (Fabian) preprocessing.
        You can load the data with load_dataset(folder) where folder is the folder where the npz files are located. If there
        are only npz files present in that folder, the data loader will unpack them on the fly. This may take a while
        and increase CPU usage. Therefore, I advise you to call unpack_dataset(folder) first, which will unpack all npz
        to npy. Don't forget to call delete_npy(folder) after you are done with training?
        Why all the hassle? Well the decathlon dataset is huge. Using npy for everything will consume >1 TB and that is uncool
        given that I (Fabian) will have to store that permanently on /datasets and my local computer. With this strategy all
        data is stored in a compressed format (factor 10 smaller) and only unpacked when needed.
        :param data: get this with load_dataset(folder, stage=0). Plug the return value in here and you are g2g (good to go)
        :param patch_size: what patch size will this data loader return? it is common practice to first load larger
        patches so that a central crop after data augmentation can be done to reduce border artifacts. If unsure, use
        get_patch_size() from data_augmentation.default_data_augmentation
        :param final_patch_size: what will the patch finally be cropped to (after data augmentation)? this is the patch
        size that goes into your network. We need this here because we will pad patients in here so that patches at the
        border of patients are sampled properly
        :param batch_size:
        :param num_batches: how many batches will the data loader produce before stopping? None=endless
        :param seed:
        :param stage: ignore this (Fabian only)
        :param transpose: ignore this
        :param random: sample randomly; CAREFUL! non-random sampling requires batch_size=1, otherwise you will iterate batch_size times over the dataset
        :param pseudo_3d_slices: 7 = 3 below and 3 above the center slice
        """
        super(DataLoaderFlowACDCProgressiveAllDataAdjacent, self).__init__(data, batch_size, None)
        if pad_kwargs_data is None:
            pad_kwargs_data = OrderedDict()
        self.pad_kwargs_data = pad_kwargs_data
        self.pad_mode = pad_mode
        self.is_val = is_val
        self.crop_size = crop_size
        self.pseudo_3d_slices = pseudo_3d_slices
        self.oversample_foreground_percent = oversample_foreground_percent
        self.final_patch_size = final_patch_size
        self.patch_size = patch_size
        self.list_of_keys = list(self._data.keys())
        self.un_list_of_keys = list(unlabeled_dataset.keys())
        self.need_to_pad = np.array(patch_size) - np.array(final_patch_size)
        self.memmap_mode = memmap_mode
        self.video_length = video_length
        self.percent = None
        if pad_sides is not None:
            if not isinstance(pad_sides, np.ndarray):
                pad_sides = np.array(pad_sides)
            self.need_to_pad += pad_sides
        self.pad_sides = pad_sides
        self.data_shape, self.seg_shape = self.determine_shapes()
        self.un_data = unlabeled_dataset

        self.preprocessing_transform = self.set_up_preprocessing_pipeline()
        self.pixel_transform, self.spatial_transform = self.set_up_augmentation_pipeline()

        self.processor = processor

    def determine_shapes(self):
        num_seg = 1

        k = list(self._data.keys())[0]
        if isfile(self._data[k]['data_file'][:-4] + ".npy"):
            case_all_data = np.load(self._data[k]['data_file'][:-4] + ".npy", self.memmap_mode)
        else:
            case_all_data = np.load(self._data[k]['data_file'])['data']
        num_color_channels = case_all_data.shape[0] - num_seg
        data_shape = (self.batch_size, num_color_channels, *self.patch_size)
        seg_shape = (self.batch_size, num_seg, *self.patch_size)
        return data_shape, seg_shape

    def get_do_oversample(self, batch_idx):
        return not batch_idx < round(self.batch_size * (1 - self.oversample_foreground_percent))
    
    def select_deep_supervision(self, x):
        if self.deep_supervision:
            return x[0]
        else:
            return x
        

    def my_crop(self, labeled, seg, unlabeled):

        xm = labeled.shape[-1] / 2
        ym = labeled.shape[-2] / 2

        x1 = int(round(xm - self.final_patch_size[1] / 2))
        x2 = int(round(xm + self.final_patch_size[1] / 2))
        y1 = int(round(ym - self.final_patch_size[0] / 2))
        y2 = int(round(ym + self.final_patch_size[0] / 2))

        labeled = labeled[:, :, y1:y2, x1:x2]
        seg = seg[:, :, y1:y2, x1:x2]

        xm = unlabeled.shape[-1] / 2
        ym = unlabeled.shape[-2] / 2

        x1 = int(round(xm - self.final_patch_size[1] / 2))
        x2 = int(round(xm + self.final_patch_size[1] / 2))
        y1 = int(round(ym - self.final_patch_size[0] / 2))
        y2 = int(round(ym + self.final_patch_size[0] / 2))

        unlabeled = unlabeled[:, :, :, y1:y2, x1:x2]

        return labeled, seg, unlabeled
    
    def preprocess(self, image, mask):
        if mask is not None:
            data = {'image': image, 'mask': mask}
            transformed = self.preprocessing_transform(data)
            image = transformed['image']
            mask = transformed['mask']
        else:
            data = {'image': image}
            transformed = self.preprocessing_transform(data)
            image = transformed['image']

        return image, mask
    
    
    def augment(self, image, mask, seed):
        self.pixel_transform = self.pixel_transform.set_random_state(seed=seed)
        self.spatial_transform = self.spatial_transform.set_random_state(seed=seed)
        
        temp = torch.clone(image)
        padding_mask = image == 0

        if mask is not None:
            data = {'image': image, 'mask': mask}
            pixel_transformed = self.pixel_transform(data)
            image = pixel_transformed['image']
            mask = pixel_transformed['mask']
            image[padding_mask] = 0
            data = {'image': image, 'mask': mask}
            spatial_transformed = self.spatial_transform(data)
            image = spatial_transformed['image']
            mask = spatial_transformed['mask']

            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(1, 3)
            #ax[0].imshow(temp[0].cpu(), cmap='gray')
            #ax[1].imshow(image[0].cpu(), cmap='gray')
            #ax[2].imshow(mask[0].cpu(), cmap='gray')
            #plt.show()
        else:
            data = {'image': image}
            pixel_transformed = self.pixel_transform(data)
            image = pixel_transformed['image']
            image[padding_mask] = 0
            data = {'image': image}
            spatial_transformed = self.spatial_transform(data)
            image = spatial_transformed['image']

            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(1, 2)
            #ax[0].imshow(temp[0].cpu(), cmap='gray')
            #ax[1].imshow(image[0].cpu(), cmap='gray')
            #plt.show()

        return image, mask
    
    def set_up_preprocessing_pipeline(self):
        preprocess = Compose([
                            T.SpatialPadd(keys=['image', 'mask'], spatial_size=self.final_patch_size, allow_missing_keys=True),
                            T.CenterSpatialCropd(keys=['image', 'mask'], roi_size=self.final_patch_size, allow_missing_keys=True),
                            T.ToTensor(dtype=torch.float32, device='cuda:0')
                            ])

        return preprocess
    
    def set_up_augmentation_pipeline(self):
        spatial_transform = Compose([
                                    T.RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=0, allow_missing_keys=True),
                                    T.RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=1, allow_missing_keys=True),
                                    T.RandRotated(keys=['image', 'mask'], prob=0.2, range_x=np.pi, mode=["bilinear", "nearest"], padding_mode="zeros", allow_missing_keys=True),
                                    T.RandZoomd(keys=['image', 'mask'], prob=0.2, min_zoom=0.5, max_zoom=1.5, mode=["bilinear", "nearest"], padding_mode="constant", allow_missing_keys=True),
                                    T.RandAffined(keys=['image', 'mask'], prob=0.2, translate_range=(-26, 26), mode=["bilinear", "nearest"], padding_mode="zeros", allow_missing_keys=True)
                                    ])

        pixel_transform = Compose([
                                #RandInvertd(keys=['image'], prob=0.5),
                                T.RandAdjustContrastd(keys=['image'], prob=0.2, gamma=(0.7, 1.5), allow_missing_keys=True),
                                T.RandGaussianNoised(keys=['image'], prob=0.2, std=0.04, allow_missing_keys=True),
                                T.RandScaleIntensityd(keys=['image'], prob=0.2, factors=0.2, allow_missing_keys=True),
                                T.RandGaussianSmoothd(keys=['image'], prob=0.2, sigma_x=(0.25, 0.5), sigma_y=(0.25, 0.5), allow_missing_keys=True),
                                T.RandGaussianSharpend(keys=['image'], prob=0.2, sigma1_x=(0.1, 0.2), sigma1_y=(0.1, 0.2), sigma2_x=(0.2, 0.4), sigma2_y=(0.2, 0.4), alpha=(2.0, 3.0), allow_missing_keys=True)
                                ])

        return pixel_transform, spatial_transform

    def generate_train_batch(self):
        #eval_idx = None
        selected_keys = np.random.choice(self.list_of_keys, self.batch_size, True, None)
        list_of_frames = []
        for i in range(len(selected_keys)):
            patient_id = selected_keys[i][:10]
            l_filtered = [x for x in self.list_of_keys if x[:10] in patient_id]
            un_filtered = [x for x in self.un_list_of_keys if x[:10] in patient_id]
            filtered = l_filtered + un_filtered
            list_of_frames.append(filtered)
        
        target_list = []
        unlabeled_list = []
        padding_need_list = []
        target_mask_list = []

        case_properties = []
        for j, frames in enumerate(list_of_frames):
            labeled_frame = frames[0]
            if 'properties' in self._data[labeled_frame].keys():
                properties = self._data[labeled_frame]['properties']
            else:
                properties = load_pickle(self._data[labeled_frame]['properties_file'])
            case_properties.append(properties)

            depth = properties['size_after_resampling'][0]
            depth_idx = np.random.choice(depth)

            frames = sorted(frames, key=lambda x: int(x[16:18]))
            frames = np.array(frames)
            global_labeled_idx = np.where(~np.char.endswith(frames, '_u'))[0]
            ed_idx = global_labeled_idx[0]
            es_idx = global_labeled_idx[1]
            assert len(global_labeled_idx) == 2

            unlabeled = torch.full(size=(self.video_length, 1, *self.final_patch_size), fill_value=torch.nan, device='cuda:0')
            seg = torch.full(size=(self.video_length, 1, *self.final_patch_size), fill_value=torch.nan, device='cuda:0')
            cropped_unlabeled = torch.full(size=(self.video_length, 1, self.crop_size, self.crop_size), fill_value=torch.nan, device='cuda:0')
            cropped_seg = torch.full(size=(self.video_length, 1, self.crop_size, self.crop_size), fill_value=torch.nan, device='cuda:0')

            possible_indices = np.arange(len(frames))

            before_where = np.argwhere(possible_indices < ed_idx).reshape(-1,)
            after_where = np.argwhere(possible_indices >= ed_idx).reshape(-1,)

            before_indices = possible_indices[before_where]
            after_indices = possible_indices[after_where]

            possible_indices = np.concatenate([after_indices, before_indices])

            stop_idx = np.argwhere(possible_indices == es_idx)[0][0]
            chunk1 = possible_indices[:stop_idx + 1]
            chunk2 = possible_indices[stop_idx:]
            chunk2 = np.concatenate([np.array([possible_indices[0]]), chunk2[::-1]])

            r = np.random.randint(0, 2)
            if r == 0:
                possible_indices = chunk1
            elif r == 1:
                possible_indices = chunk2
            
            assert possible_indices[0] == ed_idx
            assert possible_indices[-1] == es_idx

            random_indices = np.random.choice(np.arange(0, len(possible_indices) - 1), size=1)
            random_indices = np.concatenate([random_indices, np.array(random_indices + 1)], axis=0)
            target_mask = np.full_like(random_indices, fill_value=False).astype(bool)
            sorted_indices = np.argsort(random_indices)

            random_indices = random_indices[sorted_indices]
            target_mask = target_mask[sorted_indices]
            target_mask = torch.from_numpy(target_mask).to('cuda:0')

            frame_indices = possible_indices[random_indices]

            assert len(frame_indices) == self.video_length
            video = frames[frame_indices]

            labeled_idx = np.where(~np.char.endswith(video, '_u'))[0]

            for frame_idx, t in enumerate(video):

                if '_u' in t:
                    if not isfile(self.un_data[t]['data_file'][:-4] + ".npy"):
                        # lets hope you know what you're doing
                        case_all_data = np.load(self.un_data[t]['data_file'][:-4] + ".npz")['data']
                    else:
                        case_all_data = np.load(self.un_data[t]['data_file'][:-4] + ".npy", self.memmap_mode)
                else:
                    if not isfile(self._data[t]['data_file'][:-4] + ".npy"):
                        # lets hope you know what you're doing
                        case_all_data = np.load(self._data[t]['data_file'][:-4] + ".npz")['data']
                    else:
                        case_all_data = np.load(self._data[t]['data_file'][:-4] + ".npy", self.memmap_mode)

                # this is for when there is just a 2d slice in case_all_data (2d support)
                if len(case_all_data.shape) == 3:
                    case_all_data = case_all_data[:, None]
                
                assert case_all_data.shape[1] == depth

                # now crop case_all_data to contain just the slice of interest. If we want additional slice above and
                # below the current slice, here is where we get them. We stack those as additional color channels
                slice_data = case_all_data[:, depth_idx]

                # case all data should now be (c, x, y)
                assert len(slice_data.shape) == 3

                if '_u' not in t:
                    image = slice_data[:-1].copy()
                    mask = slice_data[-1:].copy()
                    mask[mask < 0] = 0

                    image, mask = self.preprocess(image, mask)

                    if target_mask[frame_idx] == True:
                        seg[frame_idx] = mask
                        assert frame_idx in labeled_idx
                else:
                    image = slice_data.copy()

                    image, _ = self.preprocess(image, None)

                unlabeled[frame_idx] = image

            with torch.no_grad():
                mean_centroid, _ = self.processor.preprocess_no_registration(data=torch.clone(unlabeled)) # T, C(1), H, W

                cropped_unlabeled, padding_need = self.processor.crop_and_pad(data=unlabeled, mean_centroid=mean_centroid)
                padding_need_list.append(padding_need)

                cropped_seg, _ = self.processor.crop_and_pad(data=seg, mean_centroid=mean_centroid)

            cropped_unlabeled = (cropped_unlabeled - cropped_unlabeled.min()) / (cropped_unlabeled.max() - cropped_unlabeled.min() + 1e-8)

            if not self.is_val:
                temp_before_augment = torch.clone(cropped_unlabeled).to('cpu')
                
                seed = random.randint(0, 2**32-1)

                for t in range(len(cropped_unlabeled)):
                    cropped_unlabeled[t], cropped_seg[t] = self.augment(image=cropped_unlabeled[t], mask=cropped_seg[t], seed=seed)

            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(3, len(cropped_unlabeled))
            #for t in range(len(cropped_unlabeled)):
            #    ax[0, t].imshow(temp_before_augment[t, 0].cpu(), cmap='gray')
            #    ax[1, t].imshow(cropped_unlabeled[t, 0].cpu(), cmap='gray')
            #    ax[2, t].imshow(cropped_seg[t, 0].cpu(), cmap='gray')
            #plt.show()
                    
            cropped_unlabeled = NormalizeIntensity()(cropped_unlabeled)

            unlabeled_list.append(cropped_unlabeled)
            target_list.append(cropped_seg)
            target_mask_list.append(target_mask)

        unlabeled = torch.stack(unlabeled_list, dim=1) # T, B, 1, H, W
        target = torch.stack(target_list, dim=1) # T, B, 1, H, W
        target_mask = torch.stack(target_mask_list, dim=1) # T, B
        padding_need = torch.stack(padding_need_list, dim=0) # B, 4
        keys = selected_keys


        #print(seg.shape)
        #print(np.unique(seg))
        #matplotlib.use('QtAgg')
        #fig, ax = plt.subplots(1, 2)
        #ax[0].imshow(labeled[3, 0], cmap='gray')
        #ax[1].imshow(seg[3, 0], cmap='gray')
        #plt.show()

        return {'unlabeled':unlabeled, 
                'target': target,
                'in_between': None,
                'padding_need': padding_need,
                'properties': case_properties,
                'target_mask': target_mask}
    


class DataLoaderFlowTrain5Interpolation(SlimDataLoaderBase):
    def __init__(self, data, patch_size, final_patch_size, batch_size, unlabeled_dataset, video_length, processor, crop_size, is_val, oversample_foreground_percent=0.0,
                 memmap_mode="r", pseudo_3d_slices=1, pad_mode="edge",
                 pad_kwargs_data=None, pad_sides=None):
        """
        This is the basic data loader for 2D networks. It uses preprocessed data as produced by my (Fabian) preprocessing.
        You can load the data with load_dataset(folder) where folder is the folder where the npz files are located. If there
        are only npz files present in that folder, the data loader will unpack them on the fly. This may take a while
        and increase CPU usage. Therefore, I advise you to call unpack_dataset(folder) first, which will unpack all npz
        to npy. Don't forget to call delete_npy(folder) after you are done with training?
        Why all the hassle? Well the decathlon dataset is huge. Using npy for everything will consume >1 TB and that is uncool
        given that I (Fabian) will have to store that permanently on /datasets and my local computer. With this strategy all
        data is stored in a compressed format (factor 10 smaller) and only unpacked when needed.
        :param data: get this with load_dataset(folder, stage=0). Plug the return value in here and you are g2g (good to go)
        :param patch_size: what patch size will this data loader return? it is common practice to first load larger
        patches so that a central crop after data augmentation can be done to reduce border artifacts. If unsure, use
        get_patch_size() from data_augmentation.default_data_augmentation
        :param final_patch_size: what will the patch finally be cropped to (after data augmentation)? this is the patch
        size that goes into your network. We need this here because we will pad patients in here so that patches at the
        border of patients are sampled properly
        :param batch_size:
        :param num_batches: how many batches will the data loader produce before stopping? None=endless
        :param seed:
        :param stage: ignore this (Fabian only)
        :param transpose: ignore this
        :param random: sample randomly; CAREFUL! non-random sampling requires batch_size=1, otherwise you will iterate batch_size times over the dataset
        :param pseudo_3d_slices: 7 = 3 below and 3 above the center slice
        """
        super(DataLoaderFlowTrain5Interpolation, self).__init__(data, batch_size, None)
        if pad_kwargs_data is None:
            pad_kwargs_data = OrderedDict()
        self.pad_kwargs_data = pad_kwargs_data
        self.pad_mode = pad_mode
        self.is_val = is_val
        self.crop_size = crop_size
        self.pseudo_3d_slices = pseudo_3d_slices
        self.oversample_foreground_percent = oversample_foreground_percent
        self.final_patch_size = final_patch_size
        self.patch_size = patch_size
        self.list_of_keys = list(self._data.keys())
        self.un_list_of_keys = list(unlabeled_dataset.keys())
        self.need_to_pad = np.array(patch_size) - np.array(final_patch_size)
        self.memmap_mode = memmap_mode
        self.video_length = video_length
        self.percent = None
        if pad_sides is not None:
            if not isinstance(pad_sides, np.ndarray):
                pad_sides = np.array(pad_sides)
            self.need_to_pad += pad_sides
        self.pad_sides = pad_sides
        self.data_shape, self.seg_shape = self.determine_shapes()
        self.un_data = unlabeled_dataset

        self.preprocessing_transform = self.set_up_preprocessing_pipeline()
        self.pixel_transform, self.spatial_transform = self.set_up_augmentation_pipeline()

        self.processor = processor

    def determine_shapes(self):
        num_seg = 1

        k = list(self._data.keys())[0]
        if isfile(self._data[k]['data_file'][:-4] + ".npy"):
            case_all_data = np.load(self._data[k]['data_file'][:-4] + ".npy", self.memmap_mode)
        else:
            case_all_data = np.load(self._data[k]['data_file'])['data']
        num_color_channels = case_all_data.shape[0] - num_seg
        data_shape = (self.batch_size, num_color_channels, *self.patch_size)
        seg_shape = (self.batch_size, num_seg, *self.patch_size)
        return data_shape, seg_shape

    def get_do_oversample(self, batch_idx):
        return not batch_idx < round(self.batch_size * (1 - self.oversample_foreground_percent))
    
    def select_deep_supervision(self, x):
        if self.deep_supervision:
            return x[0]
        else:
            return x
        

    def my_crop(self, labeled, seg, unlabeled):

        xm = labeled.shape[-1] / 2
        ym = labeled.shape[-2] / 2

        x1 = int(round(xm - self.final_patch_size[1] / 2))
        x2 = int(round(xm + self.final_patch_size[1] / 2))
        y1 = int(round(ym - self.final_patch_size[0] / 2))
        y2 = int(round(ym + self.final_patch_size[0] / 2))

        labeled = labeled[:, :, y1:y2, x1:x2]
        seg = seg[:, :, y1:y2, x1:x2]

        xm = unlabeled.shape[-1] / 2
        ym = unlabeled.shape[-2] / 2

        x1 = int(round(xm - self.final_patch_size[1] / 2))
        x2 = int(round(xm + self.final_patch_size[1] / 2))
        y1 = int(round(ym - self.final_patch_size[0] / 2))
        y2 = int(round(ym + self.final_patch_size[0] / 2))

        unlabeled = unlabeled[:, :, :, y1:y2, x1:x2]

        return labeled, seg, unlabeled
    
    def preprocess(self, image, mask):
        if mask is not None:
            data = {'image': image, 'mask': mask}
            transformed = self.preprocessing_transform(data)
            image = transformed['image']
            mask = transformed['mask']
        else:
            data = {'image': image}
            transformed = self.preprocessing_transform(data)
            image = transformed['image']

        return image, mask
    
    
    def augment(self, image, mask, seed):
        self.pixel_transform = self.pixel_transform.set_random_state(seed=seed)
        self.spatial_transform = self.spatial_transform.set_random_state(seed=seed)
        
        temp = torch.clone(image)
        padding_mask = image == 0

        if mask is not None:
            data = {'image': image, 'mask': mask}
            pixel_transformed = self.pixel_transform(data)
            image = pixel_transformed['image']
            mask = pixel_transformed['mask']
            image[padding_mask] = 0
            data = {'image': image, 'mask': mask}
            spatial_transformed = self.spatial_transform(data)
            image = spatial_transformed['image']
            mask = spatial_transformed['mask']

            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(1, 3)
            #ax[0].imshow(temp[0].cpu(), cmap='gray')
            #ax[1].imshow(image[0].cpu(), cmap='gray')
            #ax[2].imshow(mask[0].cpu(), cmap='gray')
            #plt.show()
        else:
            data = {'image': image}
            pixel_transformed = self.pixel_transform(data)
            image = pixel_transformed['image']
            image[padding_mask] = 0
            data = {'image': image}
            spatial_transformed = self.spatial_transform(data)
            image = spatial_transformed['image']

            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(1, 2)
            #ax[0].imshow(temp[0].cpu(), cmap='gray')
            #ax[1].imshow(image[0].cpu(), cmap='gray')
            #plt.show()

        return image, mask
    
    def set_up_preprocessing_pipeline(self):
        preprocess = Compose([
                            T.SpatialPadd(keys=['image', 'mask'], spatial_size=self.final_patch_size, allow_missing_keys=True),
                            T.CenterSpatialCropd(keys=['image', 'mask'], roi_size=self.final_patch_size, allow_missing_keys=True),
                            T.ToTensor(dtype=torch.float32, device='cuda:0')
                            ])

        return preprocess
    
    def set_up_augmentation_pipeline(self):
        spatial_transform = Compose([
                                    T.RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=0, allow_missing_keys=True),
                                    T.RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=1, allow_missing_keys=True),
                                    T.RandRotated(keys=['image', 'mask'], prob=0.2, range_x=np.pi, mode=["bilinear", "nearest"], padding_mode="zeros", allow_missing_keys=True),
                                    T.RandZoomd(keys=['image', 'mask'], prob=0.2, min_zoom=0.5, max_zoom=1.5, mode=["bilinear", "nearest"], padding_mode="constant", allow_missing_keys=True),
                                    T.RandAffined(keys=['image', 'mask'], prob=0.2, translate_range=(-26, 26), mode=["bilinear", "nearest"], padding_mode="zeros", allow_missing_keys=True)
                                    ])

        pixel_transform = Compose([
                                #RandInvertd(keys=['image'], prob=0.5),
                                T.RandAdjustContrastd(keys=['image'], prob=0.2, gamma=(0.7, 1.5), allow_missing_keys=True),
                                T.RandGaussianNoised(keys=['image'], prob=0.2, std=0.04, allow_missing_keys=True),
                                T.RandScaleIntensityd(keys=['image'], prob=0.2, factors=0.2, allow_missing_keys=True),
                                T.RandGaussianSmoothd(keys=['image'], prob=0.2, sigma_x=(0.25, 0.5), sigma_y=(0.25, 0.5), allow_missing_keys=True),
                                T.RandGaussianSharpend(keys=['image'], prob=0.2, sigma1_x=(0.1, 0.2), sigma1_y=(0.1, 0.2), sigma2_x=(0.2, 0.4), sigma2_y=(0.2, 0.4), alpha=(2.0, 3.0), allow_missing_keys=True)
                                ])

        return pixel_transform, spatial_transform

    def generate_train_batch(self):
        #eval_idx = None
        selected_keys = np.random.choice(self.list_of_keys, self.batch_size, True, None)
        list_of_frames = []
        for i in range(len(selected_keys)):
            patient_id = selected_keys[i][:10]
            l_filtered = [x for x in self.list_of_keys if x[:10] in patient_id]
            un_filtered = [x for x in self.un_list_of_keys if x[:10] in patient_id]
            filtered = l_filtered + un_filtered
            list_of_frames.append(filtered)
        
        target_list = []
        unlabeled_list = []
        padding_need_list = []
        target_mask_list = []
        distances_list_local = []
        distances_list_global = []

        case_properties = []
        for j, frames in enumerate(list_of_frames):
            labeled_frame = frames[0]
            if 'properties' in self._data[labeled_frame].keys():
                properties = self._data[labeled_frame]['properties']
            else:
                properties = load_pickle(self._data[labeled_frame]['properties_file'])
            case_properties.append(properties)

            depth = properties['size_after_resampling'][0]
            depth_idx = np.random.choice(depth)

            frames = sorted(frames, key=lambda x: int(x[16:18]))
            frames = np.array(frames)
            global_labeled_idx = np.where(~np.char.endswith(frames, '_u'))[0]
            ed_idx = global_labeled_idx[0]
            es_idx = global_labeled_idx[1]
            assert len(global_labeled_idx) == 2

            unlabeled = torch.full(size=(self.video_length * 2 - 2, 1, *self.final_patch_size), fill_value=torch.nan, device='cuda:0')
            seg = torch.full(size=(self.video_length * 2 - 2, 1, *self.final_patch_size), fill_value=torch.nan, device='cuda:0')
            cropped_unlabeled = torch.full(size=(self.video_length * 2 - 2, 1, self.crop_size, self.crop_size), fill_value=torch.nan, device='cuda:0')
            cropped_seg = torch.full(size=(self.video_length * 2 - 2, 1, self.crop_size, self.crop_size), fill_value=torch.nan, device='cuda:0')

            possible_indices = np.arange(len(frames))

            before_where = np.argwhere(possible_indices < ed_idx).reshape(-1,)
            after_where = np.argwhere(possible_indices >= ed_idx).reshape(-1,)

            before_indices = possible_indices[before_where]
            after_indices = possible_indices[after_where]

            possible_indices = np.concatenate([after_indices, before_indices])

            stop_idx = np.argwhere(possible_indices == es_idx)[0][0]

            gap = 2
            indices = np.arange(len(frames))
            indices = np.pad(indices, pad_width=(0, 1000), mode='wrap')
            indices = indices[::gap][:self.video_length - 1]

            indices = np.concatenate([indices, np.array([stop_idx])])
            target_mask = np.concatenate([np.array([True]), np.full(shape=(self.video_length - 2,), fill_value=False), np.array([True])]).astype(bool)
            target_mask = torch.from_numpy(target_mask).to('cuda:0')

            diff = np.diff(indices) % len(frames)
            in_between_indices = ((indices[:-1] + (diff / 2)) % len(frames)).astype(int)
            in_between_indices = in_between_indices[:-1]

            frame_indices = possible_indices[indices]
            in_between_indices = possible_indices[in_between_indices]

            assert frame_indices[0] == ed_idx
            assert frame_indices[-1] == es_idx

            assert len(frame_indices) == self.video_length
            assert len(in_between_indices) == self.video_length - 2

            frame_indices = np.concatenate([frame_indices, in_between_indices], axis=0)
            target_mask = torch.nn.functional.pad(target_mask, pad=(0, len(in_between_indices)))
            video = frames[frame_indices]

            labeled_idx = np.where(np.isin(frame_indices, global_labeled_idx))[0]

            for frame_idx, t in enumerate(video):

                if '_u' in t:
                    if not isfile(self.un_data[t]['data_file'][:-4] + ".npy"):
                        # lets hope you know what you're doing
                        case_all_data = np.load(self.un_data[t]['data_file'][:-4] + ".npz")['data']
                    else:
                        case_all_data = np.load(self.un_data[t]['data_file'][:-4] + ".npy", self.memmap_mode)
                else:
                    if not isfile(self._data[t]['data_file'][:-4] + ".npy"):
                        # lets hope you know what you're doing
                        case_all_data = np.load(self._data[t]['data_file'][:-4] + ".npz")['data']
                    else:
                        case_all_data = np.load(self._data[t]['data_file'][:-4] + ".npy", self.memmap_mode)

                # this is for when there is just a 2d slice in case_all_data (2d support)
                if len(case_all_data.shape) == 3:
                    case_all_data = case_all_data[:, None]
                
                assert case_all_data.shape[1] == depth

                # now crop case_all_data to contain just the slice of interest. If we want additional slice above and
                # below the current slice, here is where we get them. We stack those as additional color channels
                slice_data = case_all_data[:, depth_idx]

                # case all data should now be (c, x, y)
                assert len(slice_data.shape) == 3

                if '_u' not in t:
                    image = slice_data[:-1].copy()
                    mask = slice_data[-1:].copy()
                    mask[mask < 0] = 0

                    image, mask = self.preprocess(image, mask)

                    if target_mask[frame_idx] == True:
                        seg[frame_idx] = mask
                        assert frame_idx in labeled_idx
                else:
                    image = slice_data.copy()

                    image, _ = self.preprocess(image, None)

                unlabeled[frame_idx] = image

            with torch.no_grad():
                mean_centroid, _ = self.processor.preprocess_no_registration(data=torch.clone(unlabeled)) # T, C(1), H, W

                cropped_unlabeled, padding_need = self.processor.crop_and_pad(data=unlabeled, mean_centroid=mean_centroid)
                padding_need_list.append(padding_need)

                cropped_seg, _ = self.processor.crop_and_pad(data=seg, mean_centroid=mean_centroid)

            cropped_unlabeled = (cropped_unlabeled - cropped_unlabeled.min()) / (cropped_unlabeled.max() - cropped_unlabeled.min() + 1e-8)

            assert torch.count_nonzero(torch.any(~torch.isnan(cropped_seg.reshape(self.video_length * 2 - 2, -1)), dim=-1)) == 2
            assert target_mask[0] == True
            assert torch.count_nonzero(target_mask) == 2

            if not self.is_val:
                temp_before_augment = torch.clone(cropped_unlabeled).to('cpu')
                
                seed = random.randint(0, 2**32-1)

                for t in range(len(cropped_unlabeled)):
                    cropped_unlabeled[t], cropped_seg[t] = self.augment(image=cropped_unlabeled[t], mask=cropped_seg[t], seed=seed)

            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(3, len(cropped_unlabeled))
            #for t in range(len(cropped_unlabeled)):
            #    ax[0, t].imshow(temp_before_augment[t, 0].cpu(), cmap='gray')
            #    ax[1, t].imshow(cropped_unlabeled[t, 0].cpu(), cmap='gray')
            #    ax[2, t].imshow(cropped_seg[t, 0].cpu(), cmap='gray')
            #plt.show()
                    
            cropped_unlabeled = NormalizeIntensity()(cropped_unlabeled)

            unlabeled_list.append(cropped_unlabeled)
            target_list.append(cropped_seg)
            target_mask_list.append(target_mask)

        unlabeled = torch.stack(unlabeled_list, dim=1) # T, B, 1, H, W
        target = torch.stack(target_list, dim=1) # T, B, 1, H, W
        target_mask = torch.stack(target_mask_list, dim=1) # T, B
        padding_need = torch.stack(padding_need_list, dim=0) # B, 4

        in_between = unlabeled[self.video_length:]
        unlabeled = unlabeled[:self.video_length]
        target = target[:self.video_length]
        target_mask = target_mask[:self.video_length]

        #print(seg.shape)
        #print(np.unique(seg))
        #matplotlib.use('QtAgg')
        #fig, ax = plt.subplots(1, 2)
        #ax[0].imshow(labeled[3, 0], cmap='gray')
        #ax[1].imshow(seg[3, 0], cmap='gray')
        #plt.show()

        return {'unlabeled':unlabeled, 
                'target': target,
                'in_between': in_between,
                'padding_need': padding_need,
                'properties': case_properties,
                'target_mask': target_mask}





class DataLoaderFlowTrainRecursiveVideo(SlimDataLoaderBase):
    def __init__(self, data, patch_size, final_patch_size, batch_size, unlabeled_dataset, video_length, processor, crop_size, is_val, oversample_foreground_percent=0.0,
                 memmap_mode="r", pseudo_3d_slices=1, pad_mode="edge",
                 pad_kwargs_data=None, pad_sides=None):
        """
        This is the basic data loader for 2D networks. It uses preprocessed data as produced by my (Fabian) preprocessing.
        You can load the data with load_dataset(folder) where folder is the folder where the npz files are located. If there
        are only npz files present in that folder, the data loader will unpack them on the fly. This may take a while
        and increase CPU usage. Therefore, I advise you to call unpack_dataset(folder) first, which will unpack all npz
        to npy. Don't forget to call delete_npy(folder) after you are done with training?
        Why all the hassle? Well the decathlon dataset is huge. Using npy for everything will consume >1 TB and that is uncool
        given that I (Fabian) will have to store that permanently on /datasets and my local computer. With this strategy all
        data is stored in a compressed format (factor 10 smaller) and only unpacked when needed.
        :param data: get this with load_dataset(folder, stage=0). Plug the return value in here and you are g2g (good to go)
        :param patch_size: what patch size will this data loader return? it is common practice to first load larger
        patches so that a central crop after data augmentation can be done to reduce border artifacts. If unsure, use
        get_patch_size() from data_augmentation.default_data_augmentation
        :param final_patch_size: what will the patch finally be cropped to (after data augmentation)? this is the patch
        size that goes into your network. We need this here because we will pad patients in here so that patches at the
        border of patients are sampled properly
        :param batch_size:
        :param num_batches: how many batches will the data loader produce before stopping? None=endless
        :param seed:
        :param stage: ignore this (Fabian only)
        :param transpose: ignore this
        :param random: sample randomly; CAREFUL! non-random sampling requires batch_size=1, otherwise you will iterate batch_size times over the dataset
        :param pseudo_3d_slices: 7 = 3 below and 3 above the center slice
        """
        super(DataLoaderFlowTrainRecursiveVideo, self).__init__(data, batch_size, None)
        if pad_kwargs_data is None:
            pad_kwargs_data = OrderedDict()
        self.pad_kwargs_data = pad_kwargs_data
        self.pad_mode = pad_mode
        self.is_val = is_val
        self.crop_size = crop_size
        self.pseudo_3d_slices = pseudo_3d_slices
        self.oversample_foreground_percent = oversample_foreground_percent
        self.final_patch_size = final_patch_size
        self.patch_size = patch_size
        self.list_of_keys = list(self._data.keys())
        self.un_list_of_keys = list(unlabeled_dataset.keys())
        self.need_to_pad = np.array(patch_size) - np.array(final_patch_size)
        self.memmap_mode = memmap_mode
        self.video_length = video_length
        self.percent = None
        if pad_sides is not None:
            if not isinstance(pad_sides, np.ndarray):
                pad_sides = np.array(pad_sides)
            self.need_to_pad += pad_sides
        self.pad_sides = pad_sides
        self.data_shape, self.seg_shape = self.determine_shapes()
        self.un_data = unlabeled_dataset

        self.preprocessing_transform = self.set_up_preprocessing_pipeline()
        self.pixel_transform, self.spatial_transform = self.set_up_augmentation_pipeline()

        self.processor = processor

    def determine_shapes(self):
        num_seg = 1

        k = list(self._data.keys())[0]
        if isfile(self._data[k]['data_file'][:-4] + ".npy"):
            case_all_data = np.load(self._data[k]['data_file'][:-4] + ".npy", self.memmap_mode)
        else:
            case_all_data = np.load(self._data[k]['data_file'])['data']
        num_color_channels = case_all_data.shape[0] - num_seg
        data_shape = (self.batch_size, num_color_channels, *self.patch_size)
        seg_shape = (self.batch_size, num_seg, *self.patch_size)
        return data_shape, seg_shape

    def get_do_oversample(self, batch_idx):
        return not batch_idx < round(self.batch_size * (1 - self.oversample_foreground_percent))
    
    def select_deep_supervision(self, x):
        if self.deep_supervision:
            return x[0]
        else:
            return x
        

    def my_crop(self, labeled, seg, unlabeled):

        xm = labeled.shape[-1] / 2
        ym = labeled.shape[-2] / 2

        x1 = int(round(xm - self.final_patch_size[1] / 2))
        x2 = int(round(xm + self.final_patch_size[1] / 2))
        y1 = int(round(ym - self.final_patch_size[0] / 2))
        y2 = int(round(ym + self.final_patch_size[0] / 2))

        labeled = labeled[:, :, y1:y2, x1:x2]
        seg = seg[:, :, y1:y2, x1:x2]

        xm = unlabeled.shape[-1] / 2
        ym = unlabeled.shape[-2] / 2

        x1 = int(round(xm - self.final_patch_size[1] / 2))
        x2 = int(round(xm + self.final_patch_size[1] / 2))
        y1 = int(round(ym - self.final_patch_size[0] / 2))
        y2 = int(round(ym + self.final_patch_size[0] / 2))

        unlabeled = unlabeled[:, :, :, y1:y2, x1:x2]

        return labeled, seg, unlabeled
    
    def preprocess(self, image, mask):
        if mask is not None:
            data = {'image': image, 'mask': mask}
            transformed = self.preprocessing_transform(data)
            image = transformed['image']
            mask = transformed['mask']
        else:
            data = {'image': image}
            transformed = self.preprocessing_transform(data)
            image = transformed['image']

        return image, mask
    
    
    def augment(self, image, mask, seed):
        self.pixel_transform = self.pixel_transform.set_random_state(seed=seed)
        self.spatial_transform = self.spatial_transform.set_random_state(seed=seed)
        
        temp = torch.clone(image)
        padding_mask = image == 0

        if mask is not None:
            data = {'image': image, 'mask': mask}
            pixel_transformed = self.pixel_transform(data)
            image = pixel_transformed['image']
            mask = pixel_transformed['mask']
            image[padding_mask] = 0
            data = {'image': image, 'mask': mask}
            spatial_transformed = self.spatial_transform(data)
            image = spatial_transformed['image']
            mask = spatial_transformed['mask']

            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(1, 3)
            #ax[0].imshow(temp[0].cpu(), cmap='gray')
            #ax[1].imshow(image[0].cpu(), cmap='gray')
            #ax[2].imshow(mask[0].cpu(), cmap='gray')
            #plt.show()
        else:
            data = {'image': image}
            pixel_transformed = self.pixel_transform(data)
            image = pixel_transformed['image']
            image[padding_mask] = 0
            data = {'image': image}
            spatial_transformed = self.spatial_transform(data)
            image = spatial_transformed['image']

            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(1, 2)
            #ax[0].imshow(temp[0].cpu(), cmap='gray')
            #ax[1].imshow(image[0].cpu(), cmap='gray')
            #plt.show()

        return image, mask
    
    def set_up_preprocessing_pipeline(self):
        preprocess = Compose([
                            T.SpatialPadd(keys=['image', 'mask'], spatial_size=self.final_patch_size, allow_missing_keys=True),
                            T.CenterSpatialCropd(keys=['image', 'mask'], roi_size=self.final_patch_size, allow_missing_keys=True),
                            T.ToTensor(dtype=torch.float32, device='cuda:0')
                            ])

        return preprocess
    
    def set_up_augmentation_pipeline(self):
        spatial_transform = Compose([
                                    T.RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=0, allow_missing_keys=True),
                                    T.RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=1, allow_missing_keys=True),
                                    T.RandRotated(keys=['image', 'mask'], prob=0.2, range_x=np.pi, mode=["bilinear", "nearest"], padding_mode="zeros", allow_missing_keys=True),
                                    T.RandZoomd(keys=['image', 'mask'], prob=0.2, min_zoom=0.5, max_zoom=1.5, mode=["bilinear", "nearest"], padding_mode="constant", allow_missing_keys=True),
                                    T.RandAffined(keys=['image', 'mask'], prob=0.2, translate_range=(-26, 26), mode=["bilinear", "nearest"], padding_mode="zeros", allow_missing_keys=True)
                                    ])

        pixel_transform = Compose([
                                #RandInvertd(keys=['image'], prob=0.5),
                                T.RandAdjustContrastd(keys=['image'], prob=0.2, gamma=(0.7, 1.5), allow_missing_keys=True),
                                T.RandGaussianNoised(keys=['image'], prob=0.2, std=0.04, allow_missing_keys=True),
                                T.RandScaleIntensityd(keys=['image'], prob=0.2, factors=0.2, allow_missing_keys=True),
                                T.RandGaussianSmoothd(keys=['image'], prob=0.2, sigma_x=(0.25, 0.5), sigma_y=(0.25, 0.5), allow_missing_keys=True),
                                T.RandGaussianSharpend(keys=['image'], prob=0.2, sigma1_x=(0.1, 0.2), sigma1_y=(0.1, 0.2), sigma2_x=(0.2, 0.4), sigma2_y=(0.2, 0.4), alpha=(2.0, 3.0), allow_missing_keys=True)
                                ])

        return pixel_transform, spatial_transform

    def generate_train_batch(self):
        #eval_idx = None
        selected_keys = np.random.choice(self.list_of_keys, self.batch_size, True, None)
        list_of_frames = []
        for i in range(len(selected_keys)):
            patient_id = selected_keys[i][:10]
            l_filtered = [x for x in self.list_of_keys if x[:10] in patient_id]
            un_filtered = [x for x in self.un_list_of_keys if x[:10] in patient_id]
            filtered = l_filtered + un_filtered
            list_of_frames.append(filtered)
        
        target_list = []
        unlabeled_list = []
        padding_need_list = []
        target_mask_list = []

        case_properties = []
        for j, frames in enumerate(list_of_frames):
            labeled_frame = frames[0]
            if 'properties' in self._data[labeled_frame].keys():
                properties = self._data[labeled_frame]['properties']
            else:
                properties = load_pickle(self._data[labeled_frame]['properties_file'])
            case_properties.append(properties)

            depth = properties['size_after_resampling'][0]
            depth_idx = np.random.choice(depth)

            frames = sorted(frames, key=lambda x: int(x[16:18]))
            frames = np.array(frames)
            global_labeled_idx = np.where(~np.char.endswith(frames, '_u'))[0]
            assert len(global_labeled_idx) == 2

            unlabeled = torch.full(size=(self.video_length, 1, *self.final_patch_size), fill_value=torch.nan, device='cuda:0')
            seg = torch.full(size=(self.video_length, 1, *self.final_patch_size), fill_value=torch.nan, device='cuda:0')
            cropped_unlabeled = torch.full(size=(self.video_length, 1, self.crop_size, self.crop_size), fill_value=torch.nan, device='cuda:0')
            cropped_seg = torch.full(size=(self.video_length, 1, self.crop_size, self.crop_size), fill_value=torch.nan, device='cuda:0')

            possible_indices = np.arange(global_labeled_idx[0], global_labeled_idx[-1]+1)

            random_indices = np.random.choice(possible_indices, size=self.video_length - 2)
            frame_indices = np.concatenate([global_labeled_idx.reshape((-1,)), random_indices])

            sorted_indices = np.argsort(frame_indices)
            frame_indices = frame_indices[sorted_indices]
            target_mask = np.isin(frame_indices, global_labeled_idx)
            
            target_mask[1:-1] = False
            target_mask = torch.from_numpy(target_mask).to('cuda:0')

            before_where = np.argwhere(frame_indices < global_labeled_idx[0]).reshape(-1,)
            after_where = np.argwhere(frame_indices >= global_labeled_idx[0]).reshape(-1,)

            before_indices = frame_indices[before_where]
            after_indices = frame_indices[after_where]

            before_mask = target_mask[before_where]
            after_mask = target_mask[after_where]

            frame_indices = np.concatenate([after_indices, before_indices])
            target_mask = torch.cat([after_mask, before_mask])
            assert frame_indices[0] == global_labeled_idx[0]
            assert frame_indices[-1] == global_labeled_idx[-1]

            assert len(frame_indices) == self.video_length
            video = frames[frame_indices]

            labeled_idx = np.where(~np.char.endswith(video, '_u'))[0]

            for frame_idx, t in enumerate(video):

                if '_u' in t:
                    if not isfile(self.un_data[t]['data_file'][:-4] + ".npy"):
                        # lets hope you know what you're doing
                        case_all_data = np.load(self.un_data[t]['data_file'][:-4] + ".npz")['data']
                    else:
                        case_all_data = np.load(self.un_data[t]['data_file'][:-4] + ".npy", self.memmap_mode)
                else:
                    if not isfile(self._data[t]['data_file'][:-4] + ".npy"):
                        # lets hope you know what you're doing
                        case_all_data = np.load(self._data[t]['data_file'][:-4] + ".npz")['data']
                    else:
                        case_all_data = np.load(self._data[t]['data_file'][:-4] + ".npy", self.memmap_mode)

                # this is for when there is just a 2d slice in case_all_data (2d support)
                if len(case_all_data.shape) == 3:
                    case_all_data = case_all_data[:, None]
                
                assert case_all_data.shape[1] == depth

                # now crop case_all_data to contain just the slice of interest. If we want additional slice above and
                # below the current slice, here is where we get them. We stack those as additional color channels
                slice_data = case_all_data[:, depth_idx]

                # case all data should now be (c, x, y)
                assert len(slice_data.shape) == 3

                if '_u' not in t:
                    image = slice_data[:-1].copy()
                    mask = slice_data[-1:].copy()
                    mask[mask < 0] = 0

                    image, mask = self.preprocess(image, mask)

                    if target_mask[frame_idx] == True:
                        seg[frame_idx] = mask
                        assert frame_idx in labeled_idx
                else:
                    image = slice_data.copy()

                    image, _ = self.preprocess(image, None)

                unlabeled[frame_idx] = image

            with torch.no_grad():
                mean_centroid, _ = self.processor.preprocess_no_registration(data=torch.clone(unlabeled)) # T, C(1), H, W

                cropped_unlabeled, padding_need = self.processor.crop_and_pad(data=unlabeled, mean_centroid=mean_centroid)
                padding_need_list.append(padding_need)

                cropped_seg, _ = self.processor.crop_and_pad(data=seg, mean_centroid=mean_centroid)

            cropped_unlabeled = (cropped_unlabeled - cropped_unlabeled.min()) / (cropped_unlabeled.max() - cropped_unlabeled.min() + 1e-8)

            assert torch.count_nonzero(torch.any(~torch.isnan(cropped_seg.reshape(self.video_length, -1)), dim=-1)) == 2
            assert target_mask[0] == True
            assert target_mask[-1] == True

            if not self.is_val:
                temp_before_augment = torch.clone(cropped_unlabeled).to('cpu')
                
                seed = random.randint(0, 2**32-1)

                for t in range(len(cropped_unlabeled)):
                    cropped_unlabeled[t], cropped_seg[t] = self.augment(image=cropped_unlabeled[t], mask=cropped_seg[t], seed=seed)

            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(3, len(cropped_unlabeled))
            #for t in range(len(cropped_unlabeled)):
            #    ax[0, t].imshow(temp_before_augment[t, 0].cpu(), cmap='gray')
            #    ax[1, t].imshow(cropped_unlabeled[t, 0].cpu(), cmap='gray')
            #    ax[2, t].imshow(cropped_seg[t, 0].cpu(), cmap='gray')
            #plt.show()
                    
            cropped_unlabeled = NormalizeIntensity()(cropped_unlabeled)

            unlabeled_list.append(cropped_unlabeled)
            target_list.append(cropped_seg)
            target_mask_list.append(target_mask)

        unlabeled = torch.stack(unlabeled_list, dim=1) # T, B, 1, H, W
        target = torch.stack(target_list, dim=1) # T, B, 1, H, W
        target_mask = torch.stack(target_mask_list, dim=1) # T, B
        padding_need = torch.stack(padding_need_list, dim=0) # B, 4

        keys = selected_keys

        #print(seg.shape)
        #print(np.unique(seg))
        #matplotlib.use('QtAgg')
        #fig, ax = plt.subplots(1, 2)
        #ax[0].imshow(labeled[3, 0], cmap='gray')
        #ax[1].imshow(seg[3, 0], cmap='gray')
        #plt.show()

        return {'unlabeled':unlabeled, 
                'target': target,
                'padding_need': padding_need,
                'properties': case_properties, 
                "keys": keys,
                'target_mask': target_mask}



class DataLoaderFlowTrainRecursiveVideoLib(SlimDataLoaderBase):
    def __init__(self, data, patch_size, final_patch_size, batch_size, video_length, processor, crop_size, is_val, oversample_foreground_percent=0.0,
                 memmap_mode="r", pseudo_3d_slices=1, pad_mode="edge",
                 pad_kwargs_data=None, pad_sides=None):
        """
        This is the basic data loader for 2D networks. It uses preprocessed data as produced by my (Fabian) preprocessing.
        You can load the data with load_dataset(folder) where folder is the folder where the npz files are located. If there
        are only npz files present in that folder, the data loader will unpack them on the fly. This may take a while
        and increase CPU usage. Therefore, I advise you to call unpack_dataset(folder) first, which will unpack all npz
        to npy. Don't forget to call delete_npy(folder) after you are done with training?
        Why all the hassle? Well the decathlon dataset is huge. Using npy for everything will consume >1 TB and that is uncool
        given that I (Fabian) will have to store that permanently on /datasets and my local computer. With this strategy all
        data is stored in a compressed format (factor 10 smaller) and only unpacked when needed.
        :param data: get this with load_dataset(folder, stage=0). Plug the return value in here and you are g2g (good to go)
        :param patch_size: what patch size will this data loader return? it is common practice to first load larger
        patches so that a central crop after data augmentation can be done to reduce border artifacts. If unsure, use
        get_patch_size() from data_augmentation.default_data_augmentation
        :param final_patch_size: what will the patch finally be cropped to (after data augmentation)? this is the patch
        size that goes into your network. We need this here because we will pad patients in here so that patches at the
        border of patients are sampled properly
        :param batch_size:
        :param num_batches: how many batches will the data loader produce before stopping? None=endless
        :param seed:
        :param stage: ignore this (Fabian only)
        :param transpose: ignore this
        :param random: sample randomly; CAREFUL! non-random sampling requires batch_size=1, otherwise you will iterate batch_size times over the dataset
        :param pseudo_3d_slices: 7 = 3 below and 3 above the center slice
        """
        super(DataLoaderFlowTrainRecursiveVideoLib, self).__init__(data, batch_size, None)
        if pad_kwargs_data is None:
            pad_kwargs_data = OrderedDict()
        self.pad_kwargs_data = pad_kwargs_data
        self.pad_mode = pad_mode
        self.is_val = is_val
        self.crop_size = crop_size
        self.pseudo_3d_slices = pseudo_3d_slices
        self.oversample_foreground_percent = oversample_foreground_percent
        self.final_patch_size = final_patch_size
        self.patch_size = patch_size
        self.list_of_keys = list(self._data.keys())
        self.need_to_pad = np.array(patch_size) - np.array(final_patch_size)
        self.memmap_mode = memmap_mode
        self.video_length = video_length
        self.percent = None
        if pad_sides is not None:
            if not isinstance(pad_sides, np.ndarray):
                pad_sides = np.array(pad_sides)
            self.need_to_pad += pad_sides
        self.pad_sides = pad_sides
        self.data_shape, self.seg_shape = self.determine_shapes()

        self.preprocessing_transform = self.set_up_preprocessing_pipeline()
        self.pixel_transform, self.spatial_transform = self.set_up_augmentation_pipeline()

        self.processor = processor

    def determine_shapes(self):
        num_seg = 1

        k = list(self._data.keys())[0]
        if isfile(self._data[k]['data_file'][:-4] + ".npy"):
            case_all_data = np.load(self._data[k]['data_file'][:-4] + ".npy", self.memmap_mode)
        else:
            case_all_data = np.load(self._data[k]['data_file'])['data']
        num_color_channels = case_all_data.shape[0] - num_seg
        data_shape = (self.batch_size, num_color_channels, *self.patch_size)
        seg_shape = (self.batch_size, num_seg, *self.patch_size)
        return data_shape, seg_shape

    def get_do_oversample(self, batch_idx):
        return not batch_idx < round(self.batch_size * (1 - self.oversample_foreground_percent))
    
    def select_deep_supervision(self, x):
        if self.deep_supervision:
            return x[0]
        else:
            return x
        

    def my_crop(self, labeled, seg, unlabeled):

        xm = labeled.shape[-1] / 2
        ym = labeled.shape[-2] / 2

        x1 = int(round(xm - self.final_patch_size[1] / 2))
        x2 = int(round(xm + self.final_patch_size[1] / 2))
        y1 = int(round(ym - self.final_patch_size[0] / 2))
        y2 = int(round(ym + self.final_patch_size[0] / 2))

        labeled = labeled[:, :, y1:y2, x1:x2]
        seg = seg[:, :, y1:y2, x1:x2]

        xm = unlabeled.shape[-1] / 2
        ym = unlabeled.shape[-2] / 2

        x1 = int(round(xm - self.final_patch_size[1] / 2))
        x2 = int(round(xm + self.final_patch_size[1] / 2))
        y1 = int(round(ym - self.final_patch_size[0] / 2))
        y2 = int(round(ym + self.final_patch_size[0] / 2))

        unlabeled = unlabeled[:, :, :, y1:y2, x1:x2]

        return labeled, seg, unlabeled
    
    def preprocess(self, image, mask):
        if mask is not None:
            data = {'image': image, 'mask': mask}
            transformed = self.preprocessing_transform(data)
            image = transformed['image']
            mask = transformed['mask']
        else:
            data = {'image': image}
            transformed = self.preprocessing_transform(data)
            image = transformed['image']

        return image, mask
    
    
    def augment(self, image, mask, seed):
        self.pixel_transform = self.pixel_transform.set_random_state(seed=seed)
        self.spatial_transform = self.spatial_transform.set_random_state(seed=seed)
        
        temp = torch.clone(image)
        padding_mask = image == 0

        if mask is not None:
            data = {'image': image, 'mask': mask}
            pixel_transformed = self.pixel_transform(data)
            image = pixel_transformed['image']
            mask = pixel_transformed['mask']
            image[padding_mask] = 0
            data = {'image': image, 'mask': mask}
            spatial_transformed = self.spatial_transform(data)
            image = spatial_transformed['image']
            mask = spatial_transformed['mask']

            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(1, 3)
            #ax[0].imshow(temp[0].cpu(), cmap='gray')
            #ax[1].imshow(image[0].cpu(), cmap='gray')
            #ax[2].imshow(mask[0].cpu(), cmap='gray')
            #plt.show()
        else:
            data = {'image': image}
            pixel_transformed = self.pixel_transform(data)
            image = pixel_transformed['image']
            image[padding_mask] = 0
            data = {'image': image}
            spatial_transformed = self.spatial_transform(data)
            image = spatial_transformed['image']

            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(1, 2)
            #ax[0].imshow(temp[0].cpu(), cmap='gray')
            #ax[1].imshow(image[0].cpu(), cmap='gray')
            #plt.show()

        return image, mask
    
    def set_up_preprocessing_pipeline(self):
        preprocess = Compose([
                            T.SpatialPadd(keys=['image', 'mask'], spatial_size=self.final_patch_size, allow_missing_keys=True),
                            T.CenterSpatialCropd(keys=['image', 'mask'], roi_size=self.final_patch_size, allow_missing_keys=True),
                            T.ToTensor(dtype=torch.float32, device='cuda:0')
                            ])

        return preprocess
    
    def set_up_augmentation_pipeline(self):
        spatial_transform = Compose([
                                    T.RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=0, allow_missing_keys=True),
                                    T.RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=1, allow_missing_keys=True),
                                    T.RandRotated(keys=['image', 'mask'], prob=0.2, range_x=np.pi, mode=["bilinear", "nearest"], padding_mode="zeros", allow_missing_keys=True),
                                    T.RandZoomd(keys=['image', 'mask'], prob=0.2, min_zoom=0.5, max_zoom=1.5, mode=["bilinear", "nearest"], padding_mode="constant", allow_missing_keys=True),
                                    T.RandAffined(keys=['image', 'mask'], prob=0.2, translate_range=(-26, 26), mode=["bilinear", "nearest"], padding_mode="zeros", allow_missing_keys=True)
                                    ])

        pixel_transform = Compose([
                                #RandInvertd(keys=['image'], prob=0.5),
                                T.RandAdjustContrastd(keys=['image'], prob=0.2, gamma=(0.7, 1.5), allow_missing_keys=True),
                                T.RandGaussianNoised(keys=['image'], prob=0.2, std=0.04, allow_missing_keys=True),
                                T.RandScaleIntensityd(keys=['image'], prob=0.2, factors=0.2, allow_missing_keys=True),
                                T.RandGaussianSmoothd(keys=['image'], prob=0.2, sigma_x=(0.25, 0.5), sigma_y=(0.25, 0.5), allow_missing_keys=True),
                                T.RandGaussianSharpend(keys=['image'], prob=0.2, sigma1_x=(0.1, 0.2), sigma1_y=(0.1, 0.2), sigma2_x=(0.2, 0.4), sigma2_y=(0.2, 0.4), alpha=(2.0, 3.0), allow_missing_keys=True)
                                ])

        return pixel_transform, spatial_transform

    def generate_train_batch(self):
        #eval_idx = None
        selected_keys = np.random.choice(self.list_of_keys, self.batch_size, True, None)
        list_of_frames = []
        for i in range(len(selected_keys)):
            patient_id = selected_keys[i][:10]
            filtered = [x for x in self.list_of_keys if x[:10] in patient_id]
            list_of_frames.append(filtered)
        
        target_list = []
        unlabeled_list = []
        padding_need_list = []
        target_mask_list = []

        case_properties = []
        for j, frames in enumerate(list_of_frames):
            labeled_frame = frames[0]
            if 'properties' in self._data[labeled_frame].keys():
                properties = self._data[labeled_frame]['properties']
            else:
                properties = load_pickle(self._data[labeled_frame]['properties_file'])
            case_properties.append(properties)

            ed_idx = np.rint(np.array(properties['ed_number'])).astype(np.uint8) % len(frames)
            es_idx = np.rint(np.array(properties['es_number'])).astype(np.uint8) % len(frames)

            depth = properties['size_after_resampling'][0]
            depth_idx = np.random.choice(depth)

            global_labeled_idx = np.array([ed_idx, es_idx])

            frames = sorted(frames, key=lambda x: int(x[16:18]))
            frames = np.array(frames)

            unlabeled = torch.full(size=(self.video_length, 1, *self.final_patch_size), fill_value=torch.nan, device='cuda:0')
            seg = torch.full(size=(self.video_length, 1, *self.final_patch_size), fill_value=torch.nan, device='cuda:0')
            cropped_unlabeled = torch.full(size=(self.video_length, 1, self.crop_size, self.crop_size), fill_value=torch.nan, device='cuda:0')
            cropped_seg = torch.full(size=(self.video_length, 1, self.crop_size, self.crop_size), fill_value=torch.nan, device='cuda:0')

            if ed_idx > es_idx:
                after_possible = np.arange(ed_idx + 1, len(frames))
                before_possible = np.arange(es_idx)
                possible_indices = np.concatenate([after_possible, before_possible])
            else:
                possible_indices = np.arange(ed_idx + 1, es_idx)


            random_indices = np.random.choice(possible_indices, size=self.video_length - 2)
            frame_indices = np.concatenate([global_labeled_idx.reshape((-1,)), random_indices])

            sorted_indices = np.argsort(frame_indices)
            frame_indices = frame_indices[sorted_indices]
            target_mask = np.isin(frame_indices, global_labeled_idx)
            
            target_mask = torch.from_numpy(target_mask).to('cuda:0')

            before_where = np.argwhere(frame_indices < ed_idx).reshape(-1,)
            after_where = np.argwhere(frame_indices >= ed_idx).reshape(-1,)

            before_indices = frame_indices[before_where]
            after_indices = frame_indices[after_where]

            before_mask = target_mask[before_where]
            after_mask = target_mask[after_where]

            frame_indices = np.concatenate([after_indices, before_indices])
            target_mask = torch.cat([after_mask, before_mask])
            assert frame_indices[0] == ed_idx
            assert frame_indices[-1] == es_idx

            assert len(frame_indices) == self.video_length
            video = frames[frame_indices]

            labeled_idx = np.where(np.isin(frame_indices, global_labeled_idx))[0]

            for frame_idx, t in enumerate(video):

                if '_u' in t:
                    if not isfile(self.un_data[t]['data_file'][:-4] + ".npy"):
                        # lets hope you know what you're doing
                        case_all_data = np.load(self.un_data[t]['data_file'][:-4] + ".npz")['data']
                    else:
                        case_all_data = np.load(self.un_data[t]['data_file'][:-4] + ".npy", self.memmap_mode)
                else:
                    if not isfile(self._data[t]['data_file'][:-4] + ".npy"):
                        # lets hope you know what you're doing
                        case_all_data = np.load(self._data[t]['data_file'][:-4] + ".npz")['data']
                    else:
                        case_all_data = np.load(self._data[t]['data_file'][:-4] + ".npy", self.memmap_mode)

                # this is for when there is just a 2d slice in case_all_data (2d support)
                if len(case_all_data.shape) == 3:
                    case_all_data = case_all_data[:, None]
                
                assert case_all_data.shape[1] == depth

                # now crop case_all_data to contain just the slice of interest. If we want additional slice above and
                # below the current slice, here is where we get them. We stack those as additional color channels
                slice_data = case_all_data[:, depth_idx]

                # case all data should now be (c, x, y)
                assert len(slice_data.shape) == 3

                if '_u' not in t:
                    image = slice_data[:-1].copy()
                    mask = slice_data[-1:].copy()
                    mask[mask < 0] = 0

                    image, mask = self.preprocess(image, mask)

                    if target_mask[frame_idx] == True:
                        seg[frame_idx] = mask
                        assert frame_idx in labeled_idx
                else:
                    image = slice_data.copy()

                    image, _ = self.preprocess(image, None)

                unlabeled[frame_idx] = image

            with torch.no_grad():
                mean_centroid, _ = self.processor.preprocess_no_registration(data=torch.clone(unlabeled)) # T, C(1), H, W

                cropped_unlabeled, padding_need = self.processor.crop_and_pad(data=unlabeled, mean_centroid=mean_centroid)
                padding_need_list.append(padding_need)

                cropped_seg, _ = self.processor.crop_and_pad(data=seg, mean_centroid=mean_centroid)

            cropped_unlabeled = (cropped_unlabeled - cropped_unlabeled.min()) / (cropped_unlabeled.max() - cropped_unlabeled.min() + 1e-8)

            assert target_mask[0] == True
            assert target_mask[-1] == True
            assert torch.all(target_mask[1:-1] == False)

            if not self.is_val:
                temp_before_augment = torch.clone(cropped_unlabeled).to('cpu')
                
                seed = random.randint(0, 2**32-1)

                for t in range(len(cropped_unlabeled)):
                    cropped_unlabeled[t], cropped_seg[t] = self.augment(image=cropped_unlabeled[t], mask=cropped_seg[t], seed=seed)

            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(3, len(cropped_unlabeled))
            #for t in range(len(cropped_unlabeled)):
            #    ax[0, t].imshow(temp_before_augment[t, 0].cpu(), cmap='gray')
            #    ax[1, t].imshow(cropped_unlabeled[t, 0].cpu(), cmap='gray')
            #    ax[2, t].imshow(cropped_seg[t, 0].cpu(), cmap='gray')
            #plt.show()
                    
            cropped_unlabeled = NormalizeIntensity()(cropped_unlabeled)

            unlabeled_list.append(cropped_unlabeled)
            target_list.append(cropped_seg)
            target_mask_list.append(target_mask)

        unlabeled = torch.stack(unlabeled_list, dim=1) # T, B, 1, H, W
        target = torch.stack(target_list, dim=1) # T, B, 1, H, W
        target_mask = torch.stack(target_mask_list, dim=1) # T, B
        padding_need = torch.stack(padding_need_list, dim=0) # B, 4

        keys = selected_keys

        #print(seg.shape)
        #print(np.unique(seg))
        #matplotlib.use('QtAgg')
        #fig, ax = plt.subplots(1, 2)
        #ax[0].imshow(labeled[3, 0], cmap='gray')
        #ax[1].imshow(seg[3, 0], cmap='gray')
        #plt.show()

        return {'unlabeled':unlabeled, 
                'target': target,
                'padding_need': padding_need,
                'properties': case_properties, 
                "keys": keys,
                'target_mask': target_mask}
    


class DataLoaderFlowTrainLib(SlimDataLoaderBase):
    def __init__(self, data, patch_size, final_patch_size, batch_size, video_length, processor, crop_size, is_val, oversample_foreground_percent=0.0,
                 memmap_mode="r", pseudo_3d_slices=1, pad_mode="edge",
                 pad_kwargs_data=None, pad_sides=None):
        """
        This is the basic data loader for 2D networks. It uses preprocessed data as produced by my (Fabian) preprocessing.
        You can load the data with load_dataset(folder) where folder is the folder where the npz files are located. If there
        are only npz files present in that folder, the data loader will unpack them on the fly. This may take a while
        and increase CPU usage. Therefore, I advise you to call unpack_dataset(folder) first, which will unpack all npz
        to npy. Don't forget to call delete_npy(folder) after you are done with training?
        Why all the hassle? Well the decathlon dataset is huge. Using npy for everything will consume >1 TB and that is uncool
        given that I (Fabian) will have to store that permanently on /datasets and my local computer. With this strategy all
        data is stored in a compressed format (factor 10 smaller) and only unpacked when needed.
        :param data: get this with load_dataset(folder, stage=0). Plug the return value in here and you are g2g (good to go)
        :param patch_size: what patch size will this data loader return? it is common practice to first load larger
        patches so that a central crop after data augmentation can be done to reduce border artifacts. If unsure, use
        get_patch_size() from data_augmentation.default_data_augmentation
        :param final_patch_size: what will the patch finally be cropped to (after data augmentation)? this is the patch
        size that goes into your network. We need this here because we will pad patients in here so that patches at the
        border of patients are sampled properly
        :param batch_size:
        :param num_batches: how many batches will the data loader produce before stopping? None=endless
        :param seed:
        :param stage: ignore this (Fabian only)
        :param transpose: ignore this
        :param random: sample randomly; CAREFUL! non-random sampling requires batch_size=1, otherwise you will iterate batch_size times over the dataset
        :param pseudo_3d_slices: 7 = 3 below and 3 above the center slice
        """
        super(DataLoaderFlowTrainLib, self).__init__(data, batch_size, None)
        if pad_kwargs_data is None:
            pad_kwargs_data = OrderedDict()
        self.pad_kwargs_data = pad_kwargs_data
        self.pad_mode = pad_mode
        self.is_val = is_val
        self.crop_size = crop_size
        self.pseudo_3d_slices = pseudo_3d_slices
        self.oversample_foreground_percent = oversample_foreground_percent
        self.final_patch_size = final_patch_size
        self.patch_size = patch_size
        self.list_of_keys = list(self._data.keys())
        self.need_to_pad = np.array(patch_size) - np.array(final_patch_size)
        self.memmap_mode = memmap_mode
        self.video_length = video_length
        self.percent = None
        if pad_sides is not None:
            if not isinstance(pad_sides, np.ndarray):
                pad_sides = np.array(pad_sides)
            self.need_to_pad += pad_sides
        self.pad_sides = pad_sides
        self.data_shape, self.seg_shape = self.determine_shapes()

        self.preprocessing_transform = self.set_up_preprocessing_pipeline()
        self.pixel_transform, self.spatial_transform = self.set_up_augmentation_pipeline()

        self.processor = processor

    def determine_shapes(self):
        num_seg = 1

        k = list(self._data.keys())[0]
        if isfile(self._data[k]['data_file'][:-4] + ".npy"):
            case_all_data = np.load(self._data[k]['data_file'][:-4] + ".npy", self.memmap_mode)
        else:
            case_all_data = np.load(self._data[k]['data_file'])['data']
        num_color_channels = case_all_data.shape[0] - num_seg
        data_shape = (self.batch_size, num_color_channels, *self.patch_size)
        seg_shape = (self.batch_size, num_seg, *self.patch_size)
        return data_shape, seg_shape

    def get_do_oversample(self, batch_idx):
        return not batch_idx < round(self.batch_size * (1 - self.oversample_foreground_percent))
    
    def select_deep_supervision(self, x):
        if self.deep_supervision:
            return x[0]
        else:
            return x
    
    def preprocess(self, image, mask):
        data = {'image': image, 'mask': mask}
        transformed = self.preprocessing_transform(data)
        image = transformed['image']
        mask = transformed['mask']

        return image, mask
    
    
    def augment(self, image, mask, seed):
        self.pixel_transform = self.pixel_transform.set_random_state(seed=seed)
        self.spatial_transform = self.spatial_transform.set_random_state(seed=seed)
        
        temp = torch.clone(image)
        padding_mask = image == 0

        data = {'image': image, 'mask': mask}
        pixel_transformed = self.pixel_transform(data)
        image = pixel_transformed['image']
        mask = pixel_transformed['mask']
        image[padding_mask] = 0
        data = {'image': image, 'mask': mask}
        spatial_transformed = self.spatial_transform(data)
        image = spatial_transformed['image']
        mask = spatial_transformed['mask']

        #matplotlib.use('QtAgg')
        #fig, ax = plt.subplots(1, 3)
        #ax[0].imshow(temp[0].cpu(), cmap='gray')
        #ax[1].imshow(image[0].cpu(), cmap='gray')
        #ax[2].imshow(mask[0].cpu(), cmap='gray')
        #plt.show()

        return image, mask
    
    def set_up_preprocessing_pipeline(self):
        preprocess = Compose([
                            T.SpatialPadd(keys=['image', 'mask'], spatial_size=self.final_patch_size, allow_missing_keys=True),
                            T.CenterSpatialCropd(keys=['image', 'mask'], roi_size=self.final_patch_size, allow_missing_keys=True),
                            #T.ToTensor(dtype=torch.float32, device='cpu')
                            ])

        return preprocess
    
    def set_up_augmentation_pipeline(self):
        spatial_transform = Compose([
                                    T.RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=0, allow_missing_keys=True),
                                    T.RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=1, allow_missing_keys=True),
                                    T.RandRotated(keys=['image', 'mask'], prob=0.2, range_x=np.pi, mode=["bilinear", "nearest"], padding_mode="zeros", allow_missing_keys=True),
                                    T.RandZoomd(keys=['image', 'mask'], prob=0.2, min_zoom=0.5, max_zoom=1.5, mode=["bilinear", "nearest"], padding_mode="constant", allow_missing_keys=True),
                                    T.RandAffined(keys=['image', 'mask'], prob=0.2, translate_range=(-26, 26), mode=["bilinear", "nearest"], padding_mode="zeros", allow_missing_keys=True)
                                    ])

        pixel_transform = Compose([
                                #RandInvertd(keys=['image'], prob=0.5),
                                T.RandAdjustContrastd(keys=['image'], prob=0.2, gamma=(0.7, 1.5), allow_missing_keys=True),
                                T.RandGaussianNoised(keys=['image'], prob=0.2, std=0.04, allow_missing_keys=True),
                                T.RandScaleIntensityd(keys=['image'], prob=0.2, factors=0.2, allow_missing_keys=True),
                                T.RandGaussianSmoothd(keys=['image'], prob=0.2, sigma_x=(0.25, 0.5), sigma_y=(0.25, 0.5), allow_missing_keys=True),
                                T.RandGaussianSharpend(keys=['image'], prob=0.2, sigma1_x=(0.1, 0.2), sigma1_y=(0.1, 0.2), sigma2_x=(0.2, 0.4), sigma2_y=(0.2, 0.4), alpha=(2.0, 3.0), allow_missing_keys=True)
                                ])

        return pixel_transform, spatial_transform
    
    
    def get_strain(self, target):
        rv_perim_list = []
        endo_perim_list = []
        epi_perim_list = []
        for t in range(len(target)):
            current_arr = target[t]
            binarized_rv = current_arr == 1
            binarized_endo = current_arr == 3
            binarized_epi = np.logical_or(current_arr == 2, binarized_endo)
            perim_rv = perimeter(binarized_rv)
            perim_endo = perimeter(binarized_endo)
            perim_epi = perimeter(binarized_epi)
            rv_perim_list.append(perim_rv)
            endo_perim_list.append(perim_endo)
            epi_perim_list.append(perim_epi)
        
        rv_strain = [(rv_perim_list[i] - rv_perim_list[0]) / (rv_perim_list[0] + 1e-8) for i in range(len(rv_perim_list))]
        endo_strain = [(endo_perim_list[i] - endo_perim_list[0]) / (endo_perim_list[0] + 1e-8) for i in range(len(endo_perim_list))]
        epi_strain = [(epi_perim_list[i] - epi_perim_list[0]) / (epi_perim_list[0] + 1e-8) for i in range(len(epi_perim_list))]

        rv_strain = np.array(rv_strain)
        endo_strain = np.array(endo_strain)
        epi_strain = np.array(epi_strain)

        lv_strain = (endo_strain + epi_strain) / 2

        return rv_strain, lv_strain


    def generate_train_batch(self):
        #eval_idx = None
        selected_keys = np.random.choice(self.list_of_keys, self.batch_size, True, None)
        
        list_of_frames = []
        for i in range(len(selected_keys)):
            patient_id = selected_keys[i][:10]
            filtered = [x for x in self.list_of_keys if x[:10] in patient_id]
            list_of_frames.append(filtered)

        target_list = []
        unlabeled_list = []
        padding_need_list = []
        lv_strain_list = []
        rv_strain_list = []

        case_properties = []
        for j, frames in enumerate(list_of_frames):
            if 'properties' in self._data[frames[0]].keys():
                properties = self._data[frames[0]]['properties']
            else:
                properties = load_pickle(self._data[frames[0]]['properties_file'])
            case_properties.append(properties)

            depth = properties['size_after_resampling'][0]
            depth_idx = np.random.choice(depth)

            frames = sorted(frames, key=lambda x: int(x[16:18]))
            frames = np.array(frames)

            unlabeled = torch.full(size=(self.video_length, 1, *self.final_patch_size), fill_value=torch.nan, device='cuda:0')
            seg = np.full(shape=(self.video_length, 1, *self.final_patch_size), fill_value=torch.nan)
            cropped_unlabeled = torch.full(size=(self.video_length, 1, self.crop_size, self.crop_size), fill_value=torch.nan, device='cuda:0')
            cropped_seg = torch.full(size=(self.video_length, 1, self.crop_size, self.crop_size), fill_value=torch.nan, device='cpu')

            possible_indices = np.arange(0, len(frames))
            frame_indices = np.random.choice(possible_indices, size=self.video_length)
            sorted_indices = np.argsort(frame_indices)
            frame_indices = frame_indices[sorted_indices]

            assert len(frame_indices) == self.video_length
            video = frames[frame_indices]

            for frame_idx, t in enumerate(video):

                if not isfile(self._data[t]['data_file'][:-4] + ".npy"):
                    # lets hope you know what you're doing
                    case_all_data = np.load(self._data[t]['data_file'][:-4] + ".npz")['data']
                else:
                    case_all_data = np.load(self._data[t]['data_file'][:-4] + ".npy", self.memmap_mode)
                
                # this is for when there is just a 2d slice in case_all_data (2d support)
                if len(case_all_data.shape) == 3:
                    case_all_data = case_all_data[:, None]
                
                assert case_all_data.shape[1] == depth

                # now crop case_all_data to contain just the slice of interest. If we want additional slice above and
                # below the current slice, here is where we get them. We stack those as additional color channels
                slice_data = case_all_data[:, depth_idx]

                # case all data should now be (c, x, y)
                assert len(slice_data.shape) == 3

                image = slice_data[:-1].copy()
                mask = slice_data[-1:].copy()
                mask[mask < 0] = 0

                image, mask = self.preprocess(image, mask)
                image = torch.from_numpy(image).to('cuda:0')

                seg[frame_idx] = mask
                unlabeled[frame_idx] = image

            with torch.no_grad():
                mean_centroid, _ = self.processor.preprocess_no_registration(data=torch.clone(unlabeled)) # T, C(1), H, W

                cropped_unlabeled, padding_need = self.processor.crop_and_pad(data=unlabeled, mean_centroid=mean_centroid)
                padding_need_list.append(padding_need)

                cropped_seg, _ = self.processor.crop_and_pad(data=seg, mean_centroid=mean_centroid)

            cropped_unlabeled = (cropped_unlabeled - cropped_unlabeled.min()) / (cropped_unlabeled.max() - cropped_unlabeled.min() + 1e-8)

            if not self.is_val:
                temp_before_augment = torch.clone(cropped_unlabeled).to('cpu')
                
                seed = random.randint(0, 2**32-1)

                for t in range(len(cropped_unlabeled)):
                    cropped_unlabeled[t], cropped_seg[t] = self.augment(image=cropped_unlabeled[t], mask=cropped_seg[t], seed=seed)

            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(3, len(cropped_unlabeled))
            #for t in range(len(cropped_unlabeled)):
            #    ax[0, t].imshow(temp_before_augment[t, 0].cpu(), cmap='gray')
            #    ax[1, t].imshow(cropped_unlabeled[t, 0].cpu(), cmap='gray')
            #    ax[2, t].imshow(cropped_seg[t, 0], cmap='gray')
            #plt.show()

            rv_strain, lv_strain = self.get_strain(cropped_seg[:, 0])

            rv_strain = torch.from_numpy(rv_strain).view(-1,).float()
            lv_strain = torch.from_numpy(lv_strain).view(-1,).float()
            cropped_seg = torch.from_numpy(cropped_seg)
                    
            cropped_unlabeled = NormalizeIntensity()(cropped_unlabeled)

            unlabeled_list.append(cropped_unlabeled)
            target_list.append(cropped_seg)
            lv_strain_list.append(lv_strain)
            rv_strain_list.append(rv_strain)

        unlabeled = torch.stack(unlabeled_list, dim=1) # T, B, 1, H, W
        target = torch.stack(target_list, dim=1) # T, B, 1, H, W
        padding_need = torch.stack(padding_need_list, dim=0) # B, 4
        lv_strain = torch.stack(lv_strain_list, dim=1) # T, B
        rv_strain = torch.stack(rv_strain_list, dim=1) # T, B

        keys = selected_keys

        unlabeled = unlabeled.to('cuda:0')
        target = target.to('cuda:0')
        padding_need = padding_need.to('cuda:0')
        rv_strain = rv_strain.to('cuda:0')
        lv_strain = lv_strain.to('cuda:0')

        #print(seg.shape)
        #print(np.unique(seg))
        #matplotlib.use('QtAgg')
        #fig, ax = plt.subplots(1, 2)
        #ax[0].imshow(labeled[3, 0], cmap='gray')
        #ax[1].imshow(seg[3, 0], cmap='gray')
        #plt.show()

        return {'unlabeled':unlabeled, 
                'target': target,
                'padding_need': padding_need,
                'properties': case_properties, 
                'lv_strain': lv_strain,
                'rv_strain': rv_strain,
                "keys": keys}
    



class DataLoaderFlowTrain5Lib(SlimDataLoaderBase):
    def __init__(self, data, patch_size, final_patch_size, batch_size, video_length, processor, crop_size, is_val, do_data_aug, oversample_foreground_percent=0.0,
                 memmap_mode="r", pseudo_3d_slices=1, pad_mode="edge",
                 pad_kwargs_data=None, pad_sides=None):
        """
        This is the basic data loader for 2D networks. It uses preprocessed data as produced by my (Fabian) preprocessing.
        You can load the data with load_dataset(folder) where folder is the folder where the npz files are located. If there
        are only npz files present in that folder, the data loader will unpack them on the fly. This may take a while
        and increase CPU usage. Therefore, I advise you to call unpack_dataset(folder) first, which will unpack all npz
        to npy. Don't forget to call delete_npy(folder) after you are done with training?
        Why all the hassle? Well the decathlon dataset is huge. Using npy for everything will consume >1 TB and that is uncool
        given that I (Fabian) will have to store that permanently on /datasets and my local computer. With this strategy all
        data is stored in a compressed format (factor 10 smaller) and only unpacked when needed.
        :param data: get this with load_dataset(folder, stage=0). Plug the return value in here and you are g2g (good to go)
        :param patch_size: what patch size will this data loader return? it is common practice to first load larger
        patches so that a central crop after data augmentation can be done to reduce border artifacts. If unsure, use
        get_patch_size() from data_augmentation.default_data_augmentation
        :param final_patch_size: what will the patch finally be cropped to (after data augmentation)? this is the patch
        size that goes into your network. We need this here because we will pad patients in here so that patches at the
        border of patients are sampled properly
        :param batch_size:
        :param num_batches: how many batches will the data loader produce before stopping? None=endless
        :param seed:
        :param stage: ignore this (Fabian only)
        :param transpose: ignore this
        :param random: sample randomly; CAREFUL! non-random sampling requires batch_size=1, otherwise you will iterate batch_size times over the dataset
        :param pseudo_3d_slices: 7 = 3 below and 3 above the center slice
        """
        super(DataLoaderFlowTrain5Lib, self).__init__(data, batch_size, None)
        if pad_kwargs_data is None:
            pad_kwargs_data = OrderedDict()
        self.pad_kwargs_data = pad_kwargs_data
        self.pad_mode = pad_mode
        self.is_val = is_val
        self.crop_size = crop_size
        self.pseudo_3d_slices = pseudo_3d_slices
        self.oversample_foreground_percent = oversample_foreground_percent
        self.final_patch_size = final_patch_size
        self.patch_size = patch_size
        self.list_of_keys = list(self._data.keys())
        self.need_to_pad = np.array(patch_size) - np.array(final_patch_size)
        self.memmap_mode = memmap_mode
        self.video_length = video_length
        self.percent = None
        if pad_sides is not None:
            if not isinstance(pad_sides, np.ndarray):
                pad_sides = np.array(pad_sides)
            self.need_to_pad += pad_sides
        self.pad_sides = pad_sides
        self.data_shape, self.seg_shape = self.determine_shapes()
        self.do_data_aug = do_data_aug

        self.preprocessing_transform = self.set_up_preprocessing_pipeline()
        self.pixel_transform, self.spatial_transform = self.set_up_augmentation_pipeline()

        self.processor = processor
        

    def determine_shapes(self):
        num_seg = 1

        k = list(self._data.keys())[0]
        if isfile(self._data[k]['data_file'][:-4] + ".npy"):
            case_all_data = np.load(self._data[k]['data_file'][:-4] + ".npy", self.memmap_mode)
        else:
            case_all_data = np.load(self._data[k]['data_file'])['data']
        num_color_channels = case_all_data.shape[0] - num_seg
        data_shape = (self.batch_size, num_color_channels, *self.patch_size)
        seg_shape = (self.batch_size, num_seg, *self.patch_size)
        return data_shape, seg_shape

    def get_do_oversample(self, batch_idx):
        return not batch_idx < round(self.batch_size * (1 - self.oversample_foreground_percent))
    
    def select_deep_supervision(self, x):
        if self.deep_supervision:
            return x[0]
        else:
            return x
        

    def my_crop(self, labeled, seg, unlabeled):

        xm = labeled.shape[-1] / 2
        ym = labeled.shape[-2] / 2

        x1 = int(round(xm - self.final_patch_size[1] / 2))
        x2 = int(round(xm + self.final_patch_size[1] / 2))
        y1 = int(round(ym - self.final_patch_size[0] / 2))
        y2 = int(round(ym + self.final_patch_size[0] / 2))

        labeled = labeled[:, :, y1:y2, x1:x2]
        seg = seg[:, :, y1:y2, x1:x2]

        xm = unlabeled.shape[-1] / 2
        ym = unlabeled.shape[-2] / 2

        x1 = int(round(xm - self.final_patch_size[1] / 2))
        x2 = int(round(xm + self.final_patch_size[1] / 2))
        y1 = int(round(ym - self.final_patch_size[0] / 2))
        y2 = int(round(ym + self.final_patch_size[0] / 2))

        unlabeled = unlabeled[:, :, :, y1:y2, x1:x2]

        return labeled, seg, unlabeled
    
    def preprocess(self, image, mask):
        if mask is not None:
            data = {'image': image, 'mask': mask}
            transformed = self.preprocessing_transform(data)
            image = transformed['image']
            mask = transformed['mask']
        else:
            data = {'image': image}
            transformed = self.preprocessing_transform(data)
            image = transformed['image']

        return image, mask
    
    
    def augment(self, image, mask, seed):
        self.pixel_transform = self.pixel_transform.set_random_state(seed=seed)
        self.spatial_transform = self.spatial_transform.set_random_state(seed=seed)
        
        temp = torch.clone(image)
        padding_mask = image == 0

        if mask is not None:
            data = {'image': image, 'mask': mask}
            pixel_transformed = self.pixel_transform(data)
            image = pixel_transformed['image']
            mask = pixel_transformed['mask']
            image[padding_mask] = 0
            data = {'image': image, 'mask': mask}
            spatial_transformed = self.spatial_transform(data)
            image = spatial_transformed['image']
            mask = spatial_transformed['mask']

            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(1, 3)
            #ax[0].imshow(temp[0].cpu(), cmap='gray')
            #ax[1].imshow(image[0].cpu(), cmap='gray')
            #ax[2].imshow(mask[0].cpu(), cmap='gray')
            #plt.show()
        else:
            data = {'image': image}
            pixel_transformed = self.pixel_transform(data)
            image = pixel_transformed['image']
            image[padding_mask] = 0
            data = {'image': image}
            spatial_transformed = self.spatial_transform(data)
            image = spatial_transformed['image']

            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(1, 2)
            #ax[0].imshow(temp[0].cpu(), cmap='gray')
            #ax[1].imshow(image[0].cpu(), cmap='gray')
            #plt.show()

        return image, mask
    
    def set_up_preprocessing_pipeline(self):
        preprocess = Compose([
                            T.SpatialPadd(keys=['image', 'mask'], spatial_size=self.final_patch_size, allow_missing_keys=True),
                            T.CenterSpatialCropd(keys=['image', 'mask'], roi_size=self.final_patch_size, allow_missing_keys=True),
                            T.ToTensor(dtype=torch.float32, device='cuda:0')
                            ])

        return preprocess
    
    def set_up_augmentation_pipeline(self):
        spatial_transform = Compose([
                                    T.RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=0, allow_missing_keys=True),
                                    T.RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=1, allow_missing_keys=True),
                                    T.RandRotated(keys=['image', 'mask'], prob=0.2, range_x=np.pi, mode=["bilinear", "nearest"], padding_mode="zeros", allow_missing_keys=True),
                                    T.RandZoomd(keys=['image', 'mask'], prob=0.2, min_zoom=0.5, max_zoom=1.5, mode=["bilinear", "nearest"], padding_mode="constant", allow_missing_keys=True),
                                    T.RandAffined(keys=['image', 'mask'], prob=0.2, translate_range=(-26, 26), mode=["bilinear", "nearest"], padding_mode="zeros", allow_missing_keys=True)
                                    ])

        pixel_transform = Compose([
                                #RandInvertd(keys=['image'], prob=0.5),
                                T.RandAdjustContrastd(keys=['image'], prob=0.2, gamma=(0.7, 1.5), allow_missing_keys=True),
                                T.RandGaussianNoised(keys=['image'], prob=0.2, std=0.04, allow_missing_keys=True),
                                T.RandScaleIntensityd(keys=['image'], prob=0.2, factors=0.2, allow_missing_keys=True),
                                T.RandGaussianSmoothd(keys=['image'], prob=0.2, sigma_x=(0.25, 0.5), sigma_y=(0.25, 0.5), allow_missing_keys=True),
                                T.RandGaussianSharpend(keys=['image'], prob=0.2, sigma1_x=(0.1, 0.2), sigma1_y=(0.1, 0.2), sigma2_x=(0.2, 0.4), sigma2_y=(0.2, 0.4), alpha=(2.0, 3.0), allow_missing_keys=True)
                                ])

        return pixel_transform, spatial_transform

    def generate_train_batch(self):
        #eval_idx = None
        selected_keys = np.random.choice(self.list_of_keys, self.batch_size, True, None)
        list_of_frames = []
        for i in range(len(selected_keys)):
            patient_id = selected_keys[i][:10]
            filtered = [x for x in self.list_of_keys if x[:10] in patient_id]
            list_of_frames.append(filtered)
        
        target_list = []
        unlabeled_list = []
        padding_need_list = []
        target_mask_list = []
        distances_list = []

        case_properties = []
        for j, frames in enumerate(list_of_frames):
            labeled_frame = frames[0]
            if 'properties' in self._data[labeled_frame].keys():
                properties = self._data[labeled_frame]['properties']
            else:
                properties = load_pickle(self._data[labeled_frame]['properties_file'])
            case_properties.append(properties)

            #print(properties['ed_number'])
            #print(np.rint(np.array(properties['ed_number'])))
            #print('******************')

            ed_idx = np.rint(np.array(properties['ed_number'])).astype(np.uint8) % len(frames)
            es_idx = np.rint(np.array(properties['es_number'])).astype(np.uint8) % len(frames)

            depth = properties['size_after_resampling'][0]
            depth_idx = np.random.choice(depth)

            global_labeled_idx = np.array([ed_idx, es_idx])

            frames = sorted(frames, key=lambda x: int(x[16:18]))
            frames = np.array(frames)

            unlabeled = torch.full(size=(self.video_length, 1, *self.final_patch_size), fill_value=torch.nan, device='cuda:0')
            seg = torch.full(size=(self.video_length, 1, *self.final_patch_size), fill_value=torch.nan, device='cuda:0')
            cropped_unlabeled = torch.full(size=(self.video_length, 1, self.crop_size, self.crop_size), fill_value=torch.nan, device='cuda:0')
            cropped_seg = torch.full(size=(self.video_length, 1, self.crop_size, self.crop_size), fill_value=torch.nan, device='cuda:0')
            target_mask = torch.zeros(size=(self.video_length - 2,), dtype=bool, device='cuda:0')

            possible_indices = np.arange(0, len(frames))
            possible_indices = set(global_labeled_idx).symmetric_difference(possible_indices)
            possible_indices = np.array(list(possible_indices))

            random_indices = np.random.choice(possible_indices, size=self.video_length - 2)
            frame_indices = np.concatenate([global_labeled_idx.reshape((-1,)), random_indices])
            sorted_indices = np.argsort(frame_indices)
            frame_indices = frame_indices[sorted_indices]
            target_mask = torch.cat([torch.tensor([True, True], device='cuda:0').reshape((-1,)), target_mask])
            target_mask = target_mask[sorted_indices]

            before_where = np.argwhere(frame_indices < ed_idx).reshape(-1,)
            after_where = np.argwhere(frame_indices >= ed_idx).reshape(-1,)

            before_indices = frame_indices[before_where]
            after_indices = frame_indices[after_where]

            before_mask = target_mask[before_where]
            after_mask = target_mask[after_where]

            frame_indices = np.concatenate([after_indices, before_indices])
            target_mask = torch.cat([after_mask, before_mask])
            assert frame_indices[0] == ed_idx

            assert len(frame_indices) == self.video_length
            distances_list.append(torch.tensor([torch.nan]))
            video = frames[frame_indices]

            labeled_idx = np.where(np.isin(frame_indices, global_labeled_idx))[0]

            for frame_idx, t in enumerate(video):

                if '_u' in t:
                    if not isfile(self.un_data[t]['data_file'][:-4] + ".npy"):
                        # lets hope you know what you're doing
                        case_all_data = np.load(self.un_data[t]['data_file'][:-4] + ".npz")['data']
                    else:
                        case_all_data = np.load(self.un_data[t]['data_file'][:-4] + ".npy", self.memmap_mode)
                else:
                    if not isfile(self._data[t]['data_file'][:-4] + ".npy"):
                        # lets hope you know what you're doing
                        case_all_data = np.load(self._data[t]['data_file'][:-4] + ".npz")['data']
                    else:
                        case_all_data = np.load(self._data[t]['data_file'][:-4] + ".npy", self.memmap_mode)

                # this is for when there is just a 2d slice in case_all_data (2d support)
                if len(case_all_data.shape) == 3:
                    case_all_data = case_all_data[:, None]
                
                assert case_all_data.shape[1] == depth

                # now crop case_all_data to contain just the slice of interest. If we want additional slice above and
                # below the current slice, here is where we get them. We stack those as additional color channels
                slice_data = case_all_data[:, depth_idx]

                # case all data should now be (c, x, y)
                assert len(slice_data.shape) == 3

                if '_u' not in t:
                    image = slice_data[:-1].copy()
                    mask = slice_data[-1:].copy()
                    mask[mask < 0] = 0

                    image, mask = self.preprocess(image, mask)

                    if target_mask[frame_idx] == True:
                        seg[frame_idx] = mask
                        assert frame_idx in labeled_idx
                else:
                    image = slice_data.copy()

                    image, _ = self.preprocess(image, None)

                unlabeled[frame_idx] = image

            with torch.no_grad():
                mean_centroid, _ = self.processor.preprocess_no_registration(data=torch.clone(unlabeled)) # T, C(1), H, W

                cropped_unlabeled, padding_need = self.processor.crop_and_pad(data=unlabeled, mean_centroid=mean_centroid)
                padding_need_list.append(padding_need)

                cropped_seg, _ = self.processor.crop_and_pad(data=seg, mean_centroid=mean_centroid)

            cropped_unlabeled = (cropped_unlabeled - cropped_unlabeled.min()) / (cropped_unlabeled.max() - cropped_unlabeled.min() + 1e-8)

            assert torch.count_nonzero(torch.any(~torch.isnan(cropped_seg.reshape(self.video_length, -1)), dim=-1)) == 2
            assert target_mask[0] == True
            assert torch.count_nonzero(target_mask) == 2

            if not self.is_val and self.do_data_aug:
                temp_before_augment = torch.clone(cropped_unlabeled).to('cpu')
                
                seed = random.randint(0, 2**32-1)

                for t in range(len(cropped_unlabeled)):
                    cropped_unlabeled[t], cropped_seg[t] = self.augment(image=cropped_unlabeled[t], mask=cropped_seg[t], seed=seed)

            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(3, len(cropped_unlabeled))
            #for t in range(len(cropped_unlabeled)):
            #    ax[0, t].imshow(temp_before_augment[t, 0].cpu(), cmap='gray')
            #    ax[1, t].imshow(cropped_unlabeled[t, 0].cpu(), cmap='gray')
            #    ax[2, t].imshow(cropped_seg[t, 0].cpu(), cmap='gray')
            #plt.show()
                    
            cropped_unlabeled = NormalizeIntensity()(cropped_unlabeled)

            unlabeled_list.append(cropped_unlabeled)
            target_list.append(cropped_seg)
            target_mask_list.append(target_mask)

        unlabeled = torch.stack(unlabeled_list, dim=1) # T, B, 1, H, W
        target = torch.stack(target_list, dim=1) # T, B, 1, H, W
        target_mask = torch.stack(target_mask_list, dim=1) # T, B
        padding_need = torch.stack(padding_need_list, dim=0) # B, 4
        distances = torch.stack(distances_list, dim=1).float().to('cuda:0') # T - 1, B

        keys = selected_keys

        #print(seg.shape)
        #print(np.unique(seg))
        #matplotlib.use('QtAgg')
        #fig, ax = plt.subplots(1, 2)
        #ax[0].imshow(labeled[3, 0], cmap='gray')
        #ax[1].imshow(seg[3, 0], cmap='gray')
        #plt.show()

        return {'unlabeled':unlabeled, 
                'target': target,
                'padding_need': padding_need,
                'properties': case_properties, 
                "distances": distances,
                'target_mask': target_mask}



class DataLoaderAugment(SlimDataLoaderBase):
    def __init__(self, data, patch_size, final_patch_size, batch_size, video_length, processor, crop_size, is_val, do_data_aug, oversample_foreground_percent=0.0,
                 memmap_mode="r", pseudo_3d_slices=1, pad_mode="edge",
                 pad_kwargs_data=None, pad_sides=None):
        """
        This is the basic data loader for 2D networks. It uses preprocessed data as produced by my (Fabian) preprocessing.
        You can load the data with load_dataset(folder) where folder is the folder where the npz files are located. If there
        are only npz files present in that folder, the data loader will unpack them on the fly. This may take a while
        and increase CPU usage. Therefore, I advise you to call unpack_dataset(folder) first, which will unpack all npz
        to npy. Don't forget to call delete_npy(folder) after you are done with training?
        Why all the hassle? Well the decathlon dataset is huge. Using npy for everything will consume >1 TB and that is uncool
        given that I (Fabian) will have to store that permanently on /datasets and my local computer. With this strategy all
        data is stored in a compressed format (factor 10 smaller) and only unpacked when needed.
        :param data: get this with load_dataset(folder, stage=0). Plug the return value in here and you are g2g (good to go)
        :param patch_size: what patch size will this data loader return? it is common practice to first load larger
        patches so that a central crop after data augmentation can be done to reduce border artifacts. If unsure, use
        get_patch_size() from data_augmentation.default_data_augmentation
        :param final_patch_size: what will the patch finally be cropped to (after data augmentation)? this is the patch
        size that goes into your network. We need this here because we will pad patients in here so that patches at the
        border of patients are sampled properly
        :param batch_size:
        :param num_batches: how many batches will the data loader produce before stopping? None=endless
        :param seed:
        :param stage: ignore this (Fabian only)
        :param transpose: ignore this
        :param random: sample randomly; CAREFUL! non-random sampling requires batch_size=1, otherwise you will iterate batch_size times over the dataset
        :param pseudo_3d_slices: 7 = 3 below and 3 above the center slice
        """
        super(DataLoaderAugment, self).__init__(data, batch_size, None)
        if pad_kwargs_data is None:
            pad_kwargs_data = OrderedDict()
        self.pad_kwargs_data = pad_kwargs_data
        self.pad_mode = pad_mode
        self.is_val = is_val
        self.crop_size = crop_size
        self.pseudo_3d_slices = pseudo_3d_slices
        self.oversample_foreground_percent = oversample_foreground_percent
        self.final_patch_size = final_patch_size
        self.patch_size = patch_size
        self.list_of_keys = list(self._data.keys())
        self.need_to_pad = np.array(patch_size) - np.array(final_patch_size)
        self.memmap_mode = memmap_mode
        self.video_length = video_length
        self.percent = None
        if pad_sides is not None:
            if not isinstance(pad_sides, np.ndarray):
                pad_sides = np.array(pad_sides)
            self.need_to_pad += pad_sides
        self.pad_sides = pad_sides
        self.data_shape, self.seg_shape = self.determine_shapes()
        self.do_data_aug = do_data_aug

        self.preprocessing_transform = self.set_up_preprocessing_pipeline()
        self.pixel_transform, self.spatial_transform = self.set_up_augmentation_pipeline()

        self.processor = processor
        

    def determine_shapes(self):
        num_seg = 1

        k = list(self._data.keys())[0]
        if isfile(self._data[k]['data_file'][:-4] + ".npy"):
            case_all_data = np.load(self._data[k]['data_file'][:-4] + ".npy", self.memmap_mode)
        else:
            case_all_data = np.load(self._data[k]['data_file'])['data']
        num_color_channels = case_all_data.shape[0] - num_seg
        data_shape = (self.batch_size, num_color_channels, *self.patch_size)
        seg_shape = (self.batch_size, num_seg, *self.patch_size)
        return data_shape, seg_shape

    def get_do_oversample(self, batch_idx):
        return not batch_idx < round(self.batch_size * (1 - self.oversample_foreground_percent))
    
    def select_deep_supervision(self, x):
        if self.deep_supervision:
            return x[0]
        else:
            return x
        

    def my_crop(self, labeled, seg, unlabeled):

        xm = labeled.shape[-1] / 2
        ym = labeled.shape[-2] / 2

        x1 = int(round(xm - self.final_patch_size[1] / 2))
        x2 = int(round(xm + self.final_patch_size[1] / 2))
        y1 = int(round(ym - self.final_patch_size[0] / 2))
        y2 = int(round(ym + self.final_patch_size[0] / 2))

        labeled = labeled[:, :, y1:y2, x1:x2]
        seg = seg[:, :, y1:y2, x1:x2]

        xm = unlabeled.shape[-1] / 2
        ym = unlabeled.shape[-2] / 2

        x1 = int(round(xm - self.final_patch_size[1] / 2))
        x2 = int(round(xm + self.final_patch_size[1] / 2))
        y1 = int(round(ym - self.final_patch_size[0] / 2))
        y2 = int(round(ym + self.final_patch_size[0] / 2))

        unlabeled = unlabeled[:, :, :, y1:y2, x1:x2]

        return labeled, seg, unlabeled
    
    def preprocess(self, image, mask):
        if mask is not None:
            data = {'image': image, 'mask': mask}
            transformed = self.preprocessing_transform(data)
            image = transformed['image']
            mask = transformed['mask']
        else:
            data = {'image': image}
            transformed = self.preprocessing_transform(data)
            image = transformed['image']

        return image, mask
    
    
    def augment(self, image, mask, pixel_transform, spatial_transform):
        
        temp = torch.clone(image)
        padding_mask = image == 0

        if mask is not None:
            data = {'image': image, 'mask': mask}
            pixel_transformed = pixel_transform(data)
            image = pixel_transformed['image']
            mask = pixel_transformed['mask']
            image[padding_mask] = 0
            data = {'image': image, 'mask': mask}
            spatial_transformed = spatial_transform(data)
            image = spatial_transformed['image']
            mask = spatial_transformed['mask']

            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(1, 3)
            #ax[0].imshow(temp[0].cpu(), cmap='gray')
            #ax[1].imshow(image[0].cpu(), cmap='gray')
            #ax[2].imshow(mask[0].cpu(), cmap='gray')
            #plt.show()
        else:
            data = {'image': image}
            pixel_transformed = pixel_transform(data)
            image = pixel_transformed['image']
            image[padding_mask] = 0
            data = {'image': image}
            spatial_transformed = spatial_transform(data)
            image = spatial_transformed['image']

            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(1, 2)
            #ax[0].imshow(temp[0].cpu(), cmap='gray')
            #ax[1].imshow(image[0].cpu(), cmap='gray')
            #plt.show()

        return image, mask
    
    def set_up_preprocessing_pipeline(self):
        preprocess = Compose([
                            T.SpatialPadd(keys=['image', 'mask'], spatial_size=self.final_patch_size, allow_missing_keys=True),
                            T.CenterSpatialCropd(keys=['image', 'mask'], roi_size=self.final_patch_size, allow_missing_keys=True),
                            T.ToTensor(dtype=torch.float32, device='cuda:0')
                            ])

        return preprocess
    
    def set_up_augmentation_pipeline(self):
        spatial_transform = [
                                    T.RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=0, allow_missing_keys=True), 
                                    T.RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=1, allow_missing_keys=True),
                                    T.RandRotated(keys=['image', 'mask'], prob=0.5, range_x=np.pi, mode=["bilinear", "nearest"], padding_mode="zeros", allow_missing_keys=True),
                                    T.RandZoomd(keys=['image', 'mask'], prob=0.5, min_zoom=0.5, max_zoom=1.5, mode=["bilinear", "nearest"], padding_mode="constant", allow_missing_keys=True),
                                    T.RandAffined(keys=['image', 'mask'], prob=0.5, translate_range=(-26, 26), mode=["bilinear", "nearest"], padding_mode="zeros", allow_missing_keys=True)
                            ]

        pixel_transform = [
                                        T.RandRicianNoised(keys=['image'], prob=0.5, std=0.075),
                                        T.RandGibbsNoised(keys=['image'], prob=0.5, alpha=[0.45, 0.75]),
                                        T.RandAdjustContrastd(keys=['image'], prob=0.5, gamma=(0.7, 1.5), allow_missing_keys=True),
                                        T.RandGaussianNoised(keys=['image'], prob=0.5, std=0.04, allow_missing_keys=True),
                                        T.RandGaussianSharpend(keys=['image'], prob=0.5, sigma1_x=(0.1, 0.2), sigma1_y=(0.1, 0.2), sigma2_x=(0.2, 0.4), sigma2_y=(0.2, 0.4), alpha=(2.0, 3.0), allow_missing_keys=True)
                                ]

        return pixel_transform, spatial_transform
    

    def generate_train_batch(self):
        #eval_idx = None
        selected_keys = np.random.choice(self.list_of_keys, self.batch_size, True, None)
        list_of_frames = []
        for i in range(len(selected_keys)):
            patient_id = selected_keys[i][:10]
            filtered = [x for x in self.list_of_keys if x[:10] in patient_id]
            list_of_frames.append(filtered)
        
        target_list = []
        unlabeled_list = []
        padding_need_list = []
        target_mask_list = []
        database_list = []

        case_properties = []
        for j, frames in enumerate(list_of_frames):
            labeled_frame = frames[0]
            if 'properties' in self._data[labeled_frame].keys():
                properties = self._data[labeled_frame]['properties']
            else:
                properties = load_pickle(self._data[labeled_frame]['properties_file'])
            case_properties.append(properties)

            database_list.append(properties['Database'])

            #print(properties['ed_number'])
            #print(np.rint(np.array(properties['ed_number'])))
            #print('******************')

            ed_idx = np.rint(np.array(properties['ed_number'])).astype(np.uint8) % len(frames)
            es_idx = np.rint(np.array(properties['es_number'])).astype(np.uint8) % len(frames)

            depth = properties['size_after_resampling'][0]
            depth_idx = np.random.choice(depth)

            global_labeled_idx = np.array([ed_idx, es_idx])

            frames = sorted(frames, key=lambda x: int(x[16:18]))
            frames = np.array(frames)

            unlabeled = torch.full(size=(self.video_length, 1, *self.final_patch_size), fill_value=torch.nan, device='cuda:0')
            seg = torch.full(size=(self.video_length, 1, *self.final_patch_size), fill_value=torch.nan, device='cuda:0')
            cropped_unlabeled = torch.full(size=(self.video_length, 1, self.crop_size, self.crop_size), fill_value=torch.nan, device='cuda:0')
            cropped_seg = torch.full(size=(self.video_length, 1, self.crop_size, self.crop_size), fill_value=torch.nan, device='cuda:0')

            possible_indices = np.arange(len(frames))

            before_where = np.argwhere(possible_indices < ed_idx).reshape(-1,)
            after_where = np.argwhere(possible_indices >= ed_idx).reshape(-1,)

            before_indices = possible_indices[before_where]
            after_indices = possible_indices[after_where]

            possible_indices = np.concatenate([after_indices, before_indices])

            stop_idx = np.argwhere(possible_indices == es_idx)[0][0]
            chunk1 = possible_indices[:stop_idx + 1]
            chunk2 = possible_indices[stop_idx:]
            chunk2 = np.concatenate([np.array([possible_indices[0]]), chunk2[::-1]])

            r = np.random.randint(0, 2)
            if r == 0:
                possible_indices = chunk1
            elif r == 1:
                possible_indices = chunk2
            
            assert possible_indices[0] == ed_idx
            assert possible_indices[-1] == es_idx

            random_indices = np.random.choice(np.arange(0, len(possible_indices)), size=self.video_length - 2)
            target_mask = np.concatenate([np.array([True]), np.full_like(random_indices, fill_value=False), np.array([True])]).astype(bool)
            random_indices = np.concatenate([np.array([0]), random_indices, np.array([len(possible_indices) - 1])])
            sorted_indices = np.argsort(random_indices)

            random_indices = random_indices[sorted_indices]
            target_mask = target_mask[sorted_indices]
            target_mask = torch.from_numpy(target_mask).to('cuda:0')

            frame_indices = possible_indices[random_indices]

            assert len(frame_indices) == self.video_length
            video = frames[frame_indices]


            labeled_idx = np.where(np.isin(frame_indices, global_labeled_idx))[0]

            for frame_idx, t in enumerate(video):

                if '_u' in t:
                    if not isfile(self.un_data[t]['data_file'][:-4] + ".npy"):
                        # lets hope you know what you're doing
                        case_all_data = np.load(self.un_data[t]['data_file'][:-4] + ".npz")['data']
                    else:
                        case_all_data = np.load(self.un_data[t]['data_file'][:-4] + ".npy", self.memmap_mode)
                else:
                    if not isfile(self._data[t]['data_file'][:-4] + ".npy"):
                        # lets hope you know what you're doing
                        case_all_data = np.load(self._data[t]['data_file'][:-4] + ".npz")['data']
                    else:
                        print(self._data[t]['data_file'][:-4])
                        case_all_data = np.load(self._data[t]['data_file'][:-4] + ".npy", self.memmap_mode)

                # this is for when there is just a 2d slice in case_all_data (2d support)
                if len(case_all_data.shape) == 3:
                    case_all_data = case_all_data[:, None]
                
                assert case_all_data.shape[1] == depth

                # now crop case_all_data to contain just the slice of interest. If we want additional slice above and
                # below the current slice, here is where we get them. We stack those as additional color channels
                slice_data = case_all_data[:, depth_idx]

                # case all data should now be (c, x, y)
                assert len(slice_data.shape) == 3

                if '_u' not in t:
                    image = slice_data[:-1].copy()
                    mask = slice_data[-1:].copy()
                    mask[mask < 0] = 0

                    image, mask = self.preprocess(image, mask)

                    if target_mask[frame_idx] == True:
                        seg[frame_idx] = mask
                        assert frame_idx in labeled_idx
                else:
                    image = slice_data.copy()

                    image, _ = self.preprocess(image, None)
                
                unlabeled[frame_idx] = image

            with torch.no_grad():
                mean_centroid, _ = self.processor.preprocess_no_registration(data=torch.clone(unlabeled)) # T, C(1), H, W

                cropped_unlabeled, padding_need = self.processor.crop_and_pad(data=unlabeled, mean_centroid=mean_centroid)
                padding_need_list.append(padding_need)

                cropped_seg, _ = self.processor.crop_and_pad(data=seg, mean_centroid=mean_centroid)

            cropped_unlabeled = (cropped_unlabeled - cropped_unlabeled.min()) / (cropped_unlabeled.max() - cropped_unlabeled.min() + 1e-8)

            assert torch.count_nonzero(torch.any(~torch.isnan(cropped_seg.reshape(self.video_length, -1)), dim=-1)) == 2
            assert target_mask[0] == True
            assert target_mask[-1] == True
            assert torch.count_nonzero(target_mask) == 2

            if not self.is_val and self.do_data_aug:
                temp_before_augment = torch.clone(cropped_unlabeled).to('cpu')

                current_pixel_transform = np.random.choice(self.pixel_transform)
                current_spatial_transform = np.random.choice(self.spatial_transform)

                #print(current_pixel_transform)
                
                seed = random.randint(0, 2**32-1)

                for t in range(len(cropped_unlabeled)):

                    current_pixel_transform = current_pixel_transform.set_random_state(seed=seed)
                    current_spatial_transform = current_spatial_transform.set_random_state(seed=seed)

                    cropped_unlabeled[t], cropped_seg[t] = self.augment(image=cropped_unlabeled[t], mask=cropped_seg[t], pixel_transform=current_pixel_transform, spatial_transform=current_spatial_transform)

            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(3, len(cropped_unlabeled))
            #for t in range(len(cropped_unlabeled)):
            #    ax[0, t].imshow(temp_before_augment[t, 0].cpu(), cmap='gray')
            #    ax[1, t].imshow(cropped_unlabeled[t, 0].cpu(), cmap='gray')
            #    ax[2, t].imshow(cropped_seg[t, 0].cpu(), cmap='gray')
            #plt.show()
                    
            cropped_unlabeled = NormalizeIntensity()(cropped_unlabeled)

            unlabeled_list.append(cropped_unlabeled)
            target_list.append(cropped_seg)
            target_mask_list.append(target_mask)

        unlabeled = torch.stack(unlabeled_list, dim=1) # T, B, 1, H, W
        target = torch.stack(target_list, dim=1) # T, B, 1, H, W
        target_mask = torch.stack(target_mask_list, dim=1) # T, B
        padding_need = torch.stack(padding_need_list, dim=0) # B, 4

        #print(seg.shape)
        #print(np.unique(seg))
        #matplotlib.use('QtAgg')
        #fig, ax = plt.subplots(1, 2)
        #ax[0].imshow(labeled[3, 0], cmap='gray')
        #ax[1].imshow(seg[3, 0], cmap='gray')
        #plt.show()

        return {'unlabeled':unlabeled, 
                'target': target,
                'in_between': None,
                'database': database_list,
                'padding_need': padding_need,
                'properties': case_properties,
                'target_mask': target_mask}



class DataLoaderPreprocessed(SlimDataLoaderBase):
    def __init__(self, data, patch_size, final_patch_size, batch_size, video_length, processor, crop_size, is_val, do_data_aug, data_path, distance_map_power, binary_distance, oversample_foreground_percent=0.0,
                 memmap_mode="r", pseudo_3d_slices=1, pad_mode="edge",
                 pad_kwargs_data=None, pad_sides=None):
        """
        This is the basic data loader for 2D networks. It uses preprocessed data as produced by my (Fabian) preprocessing.
        You can load the data with load_dataset(folder) where folder is the folder where the npz files are located. If there
        are only npz files present in that folder, the data loader will unpack them on the fly. This may take a while
        and increase CPU usage. Therefore, I advise you to call unpack_dataset(folder) first, which will unpack all npz
        to npy. Don't forget to call delete_npy(folder) after you are done with training?
        Why all the hassle? Well the decathlon dataset is huge. Using npy for everything will consume >1 TB and that is uncool
        given that I (Fabian) will have to store that permanently on /datasets and my local computer. With this strategy all
        data is stored in a compressed format (factor 10 smaller) and only unpacked when needed.
        :param data: get this with load_dataset(folder, stage=0). Plug the return value in here and you are g2g (good to go)
        :param patch_size: what patch size will this data loader return? it is common practice to first load larger
        patches so that a central crop after data augmentation can be done to reduce border artifacts. If unsure, use
        get_patch_size() from data_augmentation.default_data_augmentation
        :param final_patch_size: what will the patch finally be cropped to (after data augmentation)? this is the patch
        size that goes into your network. We need this here because we will pad patients in here so that patches at the
        border of patients are sampled properly
        :param batch_size:
        :param num_batches: how many batches will the data loader produce before stopping? None=endless
        :param seed:
        :param stage: ignore this (Fabian only)
        :param transpose: ignore this
        :param random: sample randomly; CAREFUL! non-random sampling requires batch_size=1, otherwise you will iterate batch_size times over the dataset
        :param pseudo_3d_slices: 7 = 3 below and 3 above the center slice
        """
        super(DataLoaderPreprocessed, self).__init__(data, batch_size, None)
        if pad_kwargs_data is None:
            pad_kwargs_data = OrderedDict()
        self.pad_kwargs_data = pad_kwargs_data
        self.pad_mode = pad_mode
        self.is_val = is_val
        self.crop_size = crop_size
        self.pseudo_3d_slices = pseudo_3d_slices
        self.oversample_foreground_percent = oversample_foreground_percent
        self.final_patch_size = final_patch_size
        self.patch_size = patch_size
        self.list_of_keys = list(self._data.keys())
        self.need_to_pad = np.array(patch_size) - np.array(final_patch_size)
        self.memmap_mode = memmap_mode
        self.video_length = video_length
        self.percent = None
        if pad_sides is not None:
            if not isinstance(pad_sides, np.ndarray):
                pad_sides = np.array(pad_sides)
            self.need_to_pad += pad_sides
        self.pad_sides = pad_sides
        self.data_shape, self.seg_shape = self.determine_shapes()
        self.do_data_aug = do_data_aug
        self.distance_map_power = distance_map_power
        self.binary_distance = binary_distance

        self.preprocessing_transform = self.set_up_preprocessing_pipeline()
        self.pixel_transform, self.spatial_transform = self.set_up_augmentation_pipeline()

        self.processor = processor
        self.data_path = data_path
        

    def determine_shapes(self):
        num_seg = 1

        k = list(self._data.keys())[0]
        if isfile(self._data[k]['data_file'][:-4] + ".npy"):
            case_all_data = np.load(self._data[k]['data_file'][:-4] + ".npy", self.memmap_mode)
        else:
            case_all_data = np.load(self._data[k]['data_file'])['data']
        num_color_channels = case_all_data.shape[0] - num_seg
        data_shape = (self.batch_size, num_color_channels, *self.patch_size)
        seg_shape = (self.batch_size, num_seg, *self.patch_size)
        return data_shape, seg_shape

    def get_do_oversample(self, batch_idx):
        return not batch_idx < round(self.batch_size * (1 - self.oversample_foreground_percent))
    
    def select_deep_supervision(self, x):
        if self.deep_supervision:
            return x[0]
        else:
            return x
        

    def my_crop(self, labeled, seg, unlabeled):

        xm = labeled.shape[-1] / 2
        ym = labeled.shape[-2] / 2

        x1 = int(round(xm - self.final_patch_size[1] / 2))
        x2 = int(round(xm + self.final_patch_size[1] / 2))
        y1 = int(round(ym - self.final_patch_size[0] / 2))
        y2 = int(round(ym + self.final_patch_size[0] / 2))

        labeled = labeled[:, :, y1:y2, x1:x2]
        seg = seg[:, :, y1:y2, x1:x2]

        xm = unlabeled.shape[-1] / 2
        ym = unlabeled.shape[-2] / 2

        x1 = int(round(xm - self.final_patch_size[1] / 2))
        x2 = int(round(xm + self.final_patch_size[1] / 2))
        y1 = int(round(ym - self.final_patch_size[0] / 2))
        y2 = int(round(ym + self.final_patch_size[0] / 2))

        unlabeled = unlabeled[:, :, :, y1:y2, x1:x2]

        return labeled, seg, unlabeled
    
    def preprocess(self, image, mask):
        if mask is not None:
            data = {'image': image, 'mask': mask}
            transformed = self.preprocessing_transform(data)
            image = transformed['image']
            mask = transformed['mask']
        else:
            data = {'image': image}
            transformed = self.preprocessing_transform(data)
            image = transformed['image']

        return image, mask
    
    
    def augment(self, image, mask, pixel_transform, spatial_transform):
        
        temp = torch.clone(image)
        padding_mask = image == 0

        if mask is not None:
            data = {'image': image, 'mask': mask}
            pixel_transformed = pixel_transform(data)
            image = pixel_transformed['image']
            mask = pixel_transformed['mask']
            image[padding_mask] = 0
            data = {'image': image, 'mask': mask}
            spatial_transformed = spatial_transform(data)
            image = spatial_transformed['image']
            mask = spatial_transformed['mask']

            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(1, 3)
            #ax[0].imshow(temp[0].cpu(), cmap='gray')
            #ax[1].imshow(image[0].cpu(), cmap='gray')
            #ax[2].imshow(mask[0].cpu(), cmap='gray')
            #plt.show()
        else:
            data = {'image': image}
            pixel_transformed = pixel_transform(data)
            image = pixel_transformed['image']
            image[padding_mask] = 0
            data = {'image': image}
            spatial_transformed = spatial_transform(data)
            image = spatial_transformed['image']

            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(1, 2)
            #ax[0].imshow(temp[0].cpu(), cmap='gray')
            #ax[1].imshow(image[0].cpu(), cmap='gray')
            #plt.show()

        return image, mask
    
    def set_up_preprocessing_pipeline(self):
        preprocess = Compose([
                            T.SpatialPadd(keys=['image', 'mask'], spatial_size=self.final_patch_size, allow_missing_keys=True),
                            T.CenterSpatialCropd(keys=['image', 'mask'], roi_size=self.final_patch_size, allow_missing_keys=True),
                            T.ToTensor(dtype=torch.float32, device='cuda:0')
                            ])

        return preprocess
    
    def set_up_augmentation_pipeline(self):
        spatial_transform = [
                                    T.RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=0, allow_missing_keys=True), 
                                    T.RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=1, allow_missing_keys=True),
                                    T.RandRotated(keys=['image', 'mask'], prob=0.5, range_x=np.pi, mode=["bilinear", "nearest"], padding_mode="zeros", allow_missing_keys=True),
                                    T.RandZoomd(keys=['image', 'mask'], prob=0.5, min_zoom=0.5, max_zoom=1.5, mode=["bilinear", "nearest"], padding_mode="constant", allow_missing_keys=True),
                                    T.RandAffined(keys=['image', 'mask'], prob=0.5, translate_range=(-26, 26), mode=["bilinear", "nearest"], padding_mode="zeros", allow_missing_keys=True)
                            ]

        pixel_transform = [
                                        T.RandRicianNoised(keys=['image'], prob=0.5, std=0.075),
                                        T.RandGibbsNoised(keys=['image'], prob=0.5, alpha=[0.45, 0.75]),
                                        T.RandAdjustContrastd(keys=['image'], prob=0.5, gamma=(0.7, 1.5), allow_missing_keys=True),
                                        T.RandGaussianNoised(keys=['image'], prob=0.5, std=0.04, allow_missing_keys=True),
                                        T.RandGaussianSharpend(keys=['image'], prob=0.5, sigma1_x=(0.1, 0.2), sigma1_y=(0.1, 0.2), sigma2_x=(0.2, 0.4), sigma2_y=(0.2, 0.4), alpha=(2.0, 3.0), allow_missing_keys=True)
                                ]

        return pixel_transform, spatial_transform
    

    def generate_train_batch(self):
        #eval_idx = None
        selected_keys = np.random.choice(self.list_of_keys, self.batch_size, True, None)
        list_of_frames = []
        for i in range(len(selected_keys)):
            patient_id = selected_keys[i][:10]
            filtered = [x for x in self.list_of_keys if x[:10] in patient_id]
            list_of_frames.append(filtered)
        
        target_list = []
        unlabeled_list = []
        padding_need_list = []
        target_mask_list = []
        strain_mask_list = []
        database_list = []

        case_properties = []
        for j, frames in enumerate(list_of_frames):
            labeled_frame = frames[0]
            if 'properties' in self._data[labeled_frame].keys():
                properties = self._data[labeled_frame]['properties']
            else:
                properties = load_pickle(self._data[labeled_frame]['properties_file'])
            case_properties.append(properties)

            database_list.append(properties['Database'])

            #print(properties['ed_number'])
            #print(np.rint(np.array(properties['ed_number'])))
            #print('******************')

            ed_idx = np.rint(np.array(properties['ed_number'])).astype(np.uint8) % len(frames)
            es_idx = np.rint(np.array(properties['es_number'])).astype(np.uint8) % len(frames)

            depth = properties['size_after_resampling'][0]
            depth_idx = np.random.choice(depth)
            if 'patient021' in frames[0]:
                while depth_idx == 1:
                    depth_idx = np.random.choice(depth)

            global_labeled_idx = np.array([ed_idx, es_idx])

            frames = sorted(frames, key=lambda x: int(x[16:18]))
            frames = np.array(frames)

            cropped_unlabeled = torch.full(size=(self.video_length, 1, self.crop_size, self.crop_size), fill_value=torch.nan, device='cuda:0')
            cropped_seg = torch.full(size=(self.video_length, 5, self.crop_size, self.crop_size), fill_value=torch.nan, device='cuda:0')

            possible_indices = np.arange(len(frames))

            before_where = np.argwhere(possible_indices < ed_idx).reshape(-1,)
            after_where = np.argwhere(possible_indices >= ed_idx).reshape(-1,)

            before_indices = possible_indices[before_where]
            after_indices = possible_indices[after_where]

            possible_indices = np.concatenate([after_indices, before_indices])

            stop_idx = np.argwhere(possible_indices == es_idx)[0][0]
            chunk1 = possible_indices[:stop_idx + 1]
            chunk2 = possible_indices[stop_idx:]
            chunk2 = np.concatenate([np.array([possible_indices[0]]), chunk2[::-1]])

            r = np.random.randint(0, 2)
            if r == 0:
                possible_indices = chunk1
            elif r == 1:
                possible_indices = chunk2
            
            assert possible_indices[0] == ed_idx
            assert possible_indices[-1] == es_idx

            r2 = np.random.randint(0, 2)
            #if self.also_start_es and r2 == 0:
            #    possible_indices = np.flip(possible_indices)

            random_indices = np.random.choice(np.arange(0, len(possible_indices)), size=self.video_length - 2)
            target_mask = np.concatenate([np.array([True]), np.full_like(random_indices, fill_value=False), np.array([True])]).astype(bool)
            random_indices = np.concatenate([np.array([0]), random_indices, np.array([len(possible_indices) - 1])])
            sorted_indices = np.argsort(random_indices)

            random_indices = random_indices[sorted_indices]
            target_mask = target_mask[sorted_indices]
            target_mask = torch.from_numpy(target_mask).to('cuda:0')

            frame_indices = possible_indices[random_indices]

            assert len(frame_indices) == self.video_length
            video = frames[frame_indices]


            labeled_idx = np.where(np.isin(frame_indices, global_labeled_idx))[0]

            for frame_idx, t in enumerate(video):

                filename = os.path.basename(t) + ".npy"

                data = np.load(os.path.join(self.data_path, filename))

                img = data[0]
                gt = data[1]
                mask = data[2:]

                with open(os.path.join(self.data_path, filename[:-4] + '.pkl'), 'rb') as f:
                    pkl_data = pickle.load(f)
                    padding_need = pkl_data['padding_need']
                    assert padding_need.shape[-1] == mask.shape[-1] == depth
                    if frame_idx == 0:
                        padding_need_list.append(torch.from_numpy(padding_need[:, depth_idx]).to('cuda:0'))
                
                assert img.shape[-1] == depth

                img = torch.from_numpy(img[:, :, depth_idx]).to('cuda:0')
                gt = torch.from_numpy(gt[:, :, depth_idx]).to('cuda:0')
                mask = torch.from_numpy(mask[:, :, :, depth_idx]).to('cuda:0')

                mask[mask < 0] = 0
                gt[gt < 0] = 0

                if target_mask[frame_idx] == True:
                    to_cat = gt[None]
                else:
                    to_cat = torch.full_like(gt[None], fill_value=torch.nan)

                cropped_seg[frame_idx] = torch.cat([to_cat, mask], dim=0)
                cropped_unlabeled[frame_idx] = img


            cropped_unlabeled = (cropped_unlabeled - cropped_unlabeled.min()) / (cropped_unlabeled.max() - cropped_unlabeled.min() + 1e-8)

            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(1, 4)
            #ax[0].imshow(cropped_seg[:, 0][:, None][0, 0].cpu(), cmap='gray')
            #ax[1].imshow(cropped_seg[:, 0][:, None][1, 0].cpu(), cmap='gray')
            #ax[2].imshow(cropped_seg[:, 0][:, None][2, 0].cpu(), cmap='gray')
            #ax[3].imshow(cropped_seg[:, 0][:, None][3, 0].cpu(), cmap='gray')
            #plt.show()

            assert torch.count_nonzero(torch.any(~torch.isnan(cropped_seg[:, 0][:, None].reshape(self.video_length, -1)), dim=-1)) == 2
            assert target_mask[0] == True
            assert target_mask[-1] == True
            assert torch.count_nonzero(target_mask) == 2

            if not self.is_val and self.do_data_aug:
                temp_before_augment = torch.clone(cropped_unlabeled).to('cpu')

                current_pixel_transform = np.random.choice(self.pixel_transform)
                current_spatial_transform = np.random.choice(self.spatial_transform)

                #print(current_pixel_transform)
                
                seed = random.randint(0, 2**32-1)

                for t in range(len(cropped_unlabeled)):

                    current_pixel_transform = current_pixel_transform.set_random_state(seed=seed)
                    current_spatial_transform = current_spatial_transform.set_random_state(seed=seed)

                    cropped_unlabeled[t], cropped_seg[t] = self.augment(image=cropped_unlabeled[t], mask=cropped_seg[t], pixel_transform=current_pixel_transform, spatial_transform=current_spatial_transform)

            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(3, len(cropped_unlabeled))
            #for t in range(len(cropped_unlabeled)):
            #    ax[0, t].imshow(temp_before_augment[t, 0].cpu(), cmap='gray')
            #    ax[1, t].imshow(cropped_unlabeled[t, 0].cpu(), cmap='gray')
            #    ax[2, t].imshow(cropped_seg[t, 0].cpu(), cmap='gray')
            #plt.show()
                    
            cropped_unlabeled = NormalizeIntensity()(cropped_unlabeled)

            #print(cropped_seg.shape)
            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(1, 4)
            #ax[0].imshow(cropped_seg[0, 0].cpu(), cmap='gray')
            #ax[1].imshow(cropped_seg[1, 0].cpu(), cmap='gray')
            #ax[2].imshow(cropped_seg[2, 0].cpu(), cmap='gray')
            #ax[3].imshow(cropped_seg[3, 0].cpu(), cmap='gray')
            #plt.show()


            unlabeled_list.append(cropped_unlabeled)
            target_list.append(cropped_seg[:, 0][:, None])
            target_mask_list.append(target_mask)
            strain_mask_list.append(cropped_seg[:, 1:])

        unlabeled = torch.stack(unlabeled_list, dim=1) # T, B, 1, H, W
        target = torch.stack(target_list, dim=1) # T, B, 1, H, W
        strain_mask = torch.stack(strain_mask_list, dim=1) # T, B, 4, H, W
        target_mask = torch.stack(target_mask_list, dim=1) # T, B
        padding_need = torch.stack(padding_need_list, dim=0) # B, 4

        strain_mask = torch.pow(strain_mask, self.distance_map_power)
        if self.binary_distance:
            strain_mask = strain_mask.long()
        strain_mask_not_one_hot = strain_mask[:, :, -1, :, :][:, :, None, :, :]
        strain_mask_one_hot = strain_mask[:, :, :-1, :, :]

        #print(seg.shape)
        #print(np.unique(seg))
        #matplotlib.use('QtAgg')
        #fig, ax = plt.subplots(1, 2)
        #ax[0].imshow(labeled[3, 0], cmap='gray')
        #ax[1].imshow(seg[3, 0], cmap='gray')
        #plt.show()

        return {'unlabeled':unlabeled, 
                'target': target,
                'in_between': None,
                'database': database_list,
                'padding_need': padding_need,
                'properties': case_properties,
                'target_mask': target_mask,
                'strain_mask': strain_mask_not_one_hot,
                'strain_mask_one_hot': strain_mask_one_hot}





class DataLoaderPreprocessedAdjacent(SlimDataLoaderBase):
    def __init__(self, data, patch_size, final_patch_size, batch_size, video_length, processor, crop_size, is_val, do_data_aug, data_path, distance_map_power, binary_distance, oversample_foreground_percent=0.0,
                 memmap_mode="r", pseudo_3d_slices=1, pad_mode="edge",
                 pad_kwargs_data=None, pad_sides=None):
        """
        This is the basic data loader for 2D networks. It uses preprocessed data as produced by my (Fabian) preprocessing.
        You can load the data with load_dataset(folder) where folder is the folder where the npz files are located. If there
        are only npz files present in that folder, the data loader will unpack them on the fly. This may take a while
        and increase CPU usage. Therefore, I advise you to call unpack_dataset(folder) first, which will unpack all npz
        to npy. Don't forget to call delete_npy(folder) after you are done with training?
        Why all the hassle? Well the decathlon dataset is huge. Using npy for everything will consume >1 TB and that is uncool
        given that I (Fabian) will have to store that permanently on /datasets and my local computer. With this strategy all
        data is stored in a compressed format (factor 10 smaller) and only unpacked when needed.
        :param data: get this with load_dataset(folder, stage=0). Plug the return value in here and you are g2g (good to go)
        :param patch_size: what patch size will this data loader return? it is common practice to first load larger
        patches so that a central crop after data augmentation can be done to reduce border artifacts. If unsure, use
        get_patch_size() from data_augmentation.default_data_augmentation
        :param final_patch_size: what will the patch finally be cropped to (after data augmentation)? this is the patch
        size that goes into your network. We need this here because we will pad patients in here so that patches at the
        border of patients are sampled properly
        :param batch_size:
        :param num_batches: how many batches will the data loader produce before stopping? None=endless
        :param seed:
        :param stage: ignore this (Fabian only)
        :param transpose: ignore this
        :param random: sample randomly; CAREFUL! non-random sampling requires batch_size=1, otherwise you will iterate batch_size times over the dataset
        :param pseudo_3d_slices: 7 = 3 below and 3 above the center slice
        """
        super(DataLoaderPreprocessedAdjacent, self).__init__(data, batch_size, None)
        if pad_kwargs_data is None:
            pad_kwargs_data = OrderedDict()
        self.pad_kwargs_data = pad_kwargs_data
        self.pad_mode = pad_mode
        self.is_val = is_val
        self.crop_size = crop_size
        self.pseudo_3d_slices = pseudo_3d_slices
        self.oversample_foreground_percent = oversample_foreground_percent
        self.final_patch_size = final_patch_size
        self.patch_size = patch_size
        self.list_of_keys = list(self._data.keys())
        self.need_to_pad = np.array(patch_size) - np.array(final_patch_size)
        self.memmap_mode = memmap_mode
        self.video_length = video_length
        self.percent = None
        if pad_sides is not None:
            if not isinstance(pad_sides, np.ndarray):
                pad_sides = np.array(pad_sides)
            self.need_to_pad += pad_sides
        self.pad_sides = pad_sides
        self.data_shape, self.seg_shape = self.determine_shapes()
        self.do_data_aug = do_data_aug
        self.distance_map_power = distance_map_power

        self.preprocessing_transform = self.set_up_preprocessing_pipeline()
        self.pixel_transform, self.spatial_transform = self.set_up_augmentation_pipeline()

        self.processor = processor
        self.data_path = data_path
        

    def determine_shapes(self):
        num_seg = 1

        k = list(self._data.keys())[0]
        if isfile(self._data[k]['data_file'][:-4] + ".npy"):
            case_all_data = np.load(self._data[k]['data_file'][:-4] + ".npy", self.memmap_mode)
        else:
            case_all_data = np.load(self._data[k]['data_file'])['data']
        num_color_channels = case_all_data.shape[0] - num_seg
        data_shape = (self.batch_size, num_color_channels, *self.patch_size)
        seg_shape = (self.batch_size, num_seg, *self.patch_size)
        return data_shape, seg_shape

    def get_do_oversample(self, batch_idx):
        return not batch_idx < round(self.batch_size * (1 - self.oversample_foreground_percent))
    
    def select_deep_supervision(self, x):
        if self.deep_supervision:
            return x[0]
        else:
            return x
        

    def my_crop(self, labeled, seg, unlabeled):

        xm = labeled.shape[-1] / 2
        ym = labeled.shape[-2] / 2

        x1 = int(round(xm - self.final_patch_size[1] / 2))
        x2 = int(round(xm + self.final_patch_size[1] / 2))
        y1 = int(round(ym - self.final_patch_size[0] / 2))
        y2 = int(round(ym + self.final_patch_size[0] / 2))

        labeled = labeled[:, :, y1:y2, x1:x2]
        seg = seg[:, :, y1:y2, x1:x2]

        xm = unlabeled.shape[-1] / 2
        ym = unlabeled.shape[-2] / 2

        x1 = int(round(xm - self.final_patch_size[1] / 2))
        x2 = int(round(xm + self.final_patch_size[1] / 2))
        y1 = int(round(ym - self.final_patch_size[0] / 2))
        y2 = int(round(ym + self.final_patch_size[0] / 2))

        unlabeled = unlabeled[:, :, :, y1:y2, x1:x2]

        return labeled, seg, unlabeled
    
    def preprocess(self, image, mask):
        if mask is not None:
            data = {'image': image, 'mask': mask}
            transformed = self.preprocessing_transform(data)
            image = transformed['image']
            mask = transformed['mask']
        else:
            data = {'image': image}
            transformed = self.preprocessing_transform(data)
            image = transformed['image']

        return image, mask
    
    
    def augment(self, image, mask, pixel_transform, spatial_transform):
        
        temp = torch.clone(image)
        padding_mask = image == 0

        if mask is not None:
            data = {'image': image, 'mask': mask}
            pixel_transformed = pixel_transform(data)
            image = pixel_transformed['image']
            mask = pixel_transformed['mask']
            image[padding_mask] = 0
            data = {'image': image, 'mask': mask}
            spatial_transformed = spatial_transform(data)
            image = spatial_transformed['image']
            mask = spatial_transformed['mask']

            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(1, 3)
            #ax[0].imshow(temp[0].cpu(), cmap='gray')
            #ax[1].imshow(image[0].cpu(), cmap='gray')
            #ax[2].imshow(mask[0].cpu(), cmap='gray')
            #plt.show()
        else:
            data = {'image': image}
            pixel_transformed = pixel_transform(data)
            image = pixel_transformed['image']
            image[padding_mask] = 0
            data = {'image': image}
            spatial_transformed = spatial_transform(data)
            image = spatial_transformed['image']

            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(1, 2)
            #ax[0].imshow(temp[0].cpu(), cmap='gray')
            #ax[1].imshow(image[0].cpu(), cmap='gray')
            #plt.show()

        return image, mask
    
    def set_up_preprocessing_pipeline(self):
        preprocess = Compose([
                            T.SpatialPadd(keys=['image', 'mask'], spatial_size=self.final_patch_size, allow_missing_keys=True),
                            T.CenterSpatialCropd(keys=['image', 'mask'], roi_size=self.final_patch_size, allow_missing_keys=True),
                            T.ToTensor(dtype=torch.float32, device='cuda:0')
                            ])

        return preprocess
    
    def set_up_augmentation_pipeline(self):
        spatial_transform = [
                                    T.RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=0, allow_missing_keys=True), 
                                    T.RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=1, allow_missing_keys=True),
                                    T.RandRotated(keys=['image', 'mask'], prob=0.5, range_x=np.pi, mode=["bilinear", "nearest"], padding_mode="zeros", allow_missing_keys=True),
                                    T.RandZoomd(keys=['image', 'mask'], prob=0.5, min_zoom=0.5, max_zoom=1.5, mode=["bilinear", "nearest"], padding_mode="constant", allow_missing_keys=True),
                                    T.RandAffined(keys=['image', 'mask'], prob=0.5, translate_range=(-26, 26), mode=["bilinear", "nearest"], padding_mode="zeros", allow_missing_keys=True)
                            ]

        pixel_transform = [
                                        T.RandRicianNoised(keys=['image'], prob=0.5, std=0.075),
                                        T.RandGibbsNoised(keys=['image'], prob=0.5, alpha=[0.45, 0.75]),
                                        T.RandAdjustContrastd(keys=['image'], prob=0.5, gamma=(0.7, 1.5), allow_missing_keys=True),
                                        T.RandGaussianNoised(keys=['image'], prob=0.5, std=0.04, allow_missing_keys=True),
                                        T.RandGaussianSharpend(keys=['image'], prob=0.5, sigma1_x=(0.1, 0.2), sigma1_y=(0.1, 0.2), sigma2_x=(0.2, 0.4), sigma2_y=(0.2, 0.4), alpha=(2.0, 3.0), allow_missing_keys=True)
                                ]

        return pixel_transform, spatial_transform
    

    def generate_train_batch(self):
        #eval_idx = None
        selected_keys = np.random.choice(self.list_of_keys, self.batch_size, True, None)
        list_of_frames = []
        for i in range(len(selected_keys)):
            patient_id = selected_keys[i][:10]
            filtered = [x for x in self.list_of_keys if x[:10] in patient_id]
            list_of_frames.append(filtered)
        
        target_list = []
        unlabeled_list = []
        padding_need_list = []
        target_mask_list = []
        strain_mask_list = []
        database_list = []

        case_properties = []
        for j, frames in enumerate(list_of_frames):
            labeled_frame = frames[0]
            if 'properties' in self._data[labeled_frame].keys():
                properties = self._data[labeled_frame]['properties']
            else:
                properties = load_pickle(self._data[labeled_frame]['properties_file'])
            case_properties.append(properties)

            database_list.append(properties['Database'])

            #print(properties['ed_number'])
            #print(np.rint(np.array(properties['ed_number'])))
            #print('******************')

            ed_idx = np.rint(np.array(properties['ed_number'])).astype(np.uint8) % len(frames)
            es_idx = np.rint(np.array(properties['es_number'])).astype(np.uint8) % len(frames)

            depth = properties['size_after_resampling'][0]
            depth_idx = np.random.choice(depth)
            if 'patient021' in frames[0]:
                while depth_idx == 1:
                    depth_idx = np.random.choice(depth)

            global_labeled_idx = np.array([ed_idx, es_idx])

            frames = sorted(frames, key=lambda x: int(x[16:18]))
            frames = np.array(frames)

            cropped_unlabeled = torch.full(size=(self.video_length, 1, self.crop_size, self.crop_size), fill_value=torch.nan, device='cuda:0')
            cropped_seg = torch.full(size=(self.video_length, 5, self.crop_size, self.crop_size), fill_value=torch.nan, device='cuda:0')

            possible_indices = np.arange(len(frames))

            before_where = np.argwhere(possible_indices < ed_idx).reshape(-1,)
            after_where = np.argwhere(possible_indices >= ed_idx).reshape(-1,)

            before_indices = possible_indices[before_where]
            after_indices = possible_indices[after_where]

            possible_indices = np.concatenate([after_indices, before_indices])

            assert possible_indices[0] == ed_idx

            random_indices = np.random.choice(np.arange(0, len(possible_indices) - 1), size=1)
            random_indices = np.concatenate([random_indices, np.array(random_indices + 1)], axis=0)
            target_mask = np.full_like(random_indices, fill_value=False).astype(bool)
            sorted_indices = np.argsort(random_indices)

            random_indices = random_indices[sorted_indices]
            target_mask = target_mask[sorted_indices]
            target_mask = torch.from_numpy(target_mask).to('cuda:0')

            frame_indices = possible_indices[random_indices]

            assert len(frame_indices) == self.video_length
            video = frames[frame_indices]

            labeled_idx = np.where(np.isin(frame_indices, global_labeled_idx))[0]

            for frame_idx, t in enumerate(video):

                filename = os.path.basename(t) + ".npy"

                data = np.load(os.path.join(self.data_path, filename))

                img = data[0]
                gt = data[1]
                mask = data[2:]

                with open(os.path.join(self.data_path, filename[:-4] + '.pkl'), 'rb') as f:
                    pkl_data = pickle.load(f)
                    padding_need = pkl_data['padding_need']
                    assert padding_need.shape[-1] == mask.shape[-1] == depth
                    padding_need_list.append(torch.from_numpy(padding_need[:, depth_idx]).to('cuda:0'))
                
                assert img.shape[-1] == depth

                img = torch.from_numpy(img[:, :, depth_idx]).to('cuda:0')
                gt = torch.from_numpy(gt[:, :, depth_idx]).to('cuda:0')
                mask = torch.from_numpy(mask[:, :, :, depth_idx]).to('cuda:0')

                mask[mask < 0] = 0
                gt[gt < 0] = 0

                if target_mask[frame_idx] == True:
                    to_cat = gt[None]
                else:
                    to_cat = torch.full_like(gt[None], fill_value=torch.nan)

                cropped_seg[frame_idx] = torch.cat([to_cat, mask], dim=0)
                cropped_unlabeled[frame_idx] = img


            cropped_unlabeled = (cropped_unlabeled - cropped_unlabeled.min()) / (cropped_unlabeled.max() - cropped_unlabeled.min() + 1e-8)

            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(1, 4)
            #ax[0].imshow(cropped_seg[:, 0][:, None][0, 0].cpu(), cmap='gray')
            #ax[1].imshow(cropped_seg[:, 0][:, None][1, 0].cpu(), cmap='gray')
            #ax[2].imshow(cropped_seg[:, 0][:, None][2, 0].cpu(), cmap='gray')
            #ax[3].imshow(cropped_seg[:, 0][:, None][3, 0].cpu(), cmap='gray')
            #plt.show()

            if not self.is_val and self.do_data_aug:
                temp_before_augment = torch.clone(cropped_unlabeled).to('cpu')

                current_pixel_transform = np.random.choice(self.pixel_transform)
                current_spatial_transform = np.random.choice(self.spatial_transform)

                #print(current_pixel_transform)
                
                seed = random.randint(0, 2**32-1)

                for t in range(len(cropped_unlabeled)):

                    current_pixel_transform = current_pixel_transform.set_random_state(seed=seed)
                    current_spatial_transform = current_spatial_transform.set_random_state(seed=seed)

                    cropped_unlabeled[t], cropped_seg[t] = self.augment(image=cropped_unlabeled[t], mask=cropped_seg[t], pixel_transform=current_pixel_transform, spatial_transform=current_spatial_transform)

            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(3, len(cropped_unlabeled))
            #for t in range(len(cropped_unlabeled)):
            #    ax[0, t].imshow(temp_before_augment[t, 0].cpu(), cmap='gray')
            #    ax[1, t].imshow(cropped_unlabeled[t, 0].cpu(), cmap='gray')
            #    ax[2, t].imshow(cropped_seg[t, 0].cpu(), cmap='gray')
            #plt.show()
                    
            cropped_unlabeled = NormalizeIntensity()(cropped_unlabeled)

            #print(cropped_seg.shape)
            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(1, 4)
            #ax[0].imshow(cropped_seg[0, 0].cpu(), cmap='gray')
            #ax[1].imshow(cropped_seg[1, 0].cpu(), cmap='gray')
            #ax[2].imshow(cropped_seg[2, 0].cpu(), cmap='gray')
            #ax[3].imshow(cropped_seg[3, 0].cpu(), cmap='gray')
            #plt.show()


            unlabeled_list.append(cropped_unlabeled)
            target_list.append(cropped_seg[:, 0][:, None])
            target_mask_list.append(target_mask)
            strain_mask_list.append(cropped_seg[:, 1:])

        unlabeled = torch.stack(unlabeled_list, dim=1) # T, B, 1, H, W
        target = torch.stack(target_list, dim=1) # T, B, 1, H, W
        strain_mask = torch.stack(strain_mask_list, dim=1) # T, B, 4, H, W
        target_mask = torch.stack(target_mask_list, dim=1) # T, B
        padding_need = torch.stack(padding_need_list, dim=0) # B, 4

        strain_mask = torch.pow(strain_mask, self.distance_map_power)
        strain_mask_not_one_hot = strain_mask[:, :, -1, :, :][:, :, None, :, :]
        strain_mask_one_hot = strain_mask[:, :, :-1, :, :]

        #print(seg.shape)
        #print(np.unique(seg))
        #matplotlib.use('QtAgg')
        #fig, ax = plt.subplots(1, 2)
        #ax[0].imshow(labeled[3, 0], cmap='gray')
        #ax[1].imshow(seg[3, 0], cmap='gray')
        #plt.show()

        return {'unlabeled':unlabeled, 
                'target': target,
                'in_between': None,
                'database': database_list,
                'padding_need': padding_need,
                'properties': case_properties,
                'target_mask': target_mask,
                'strain_mask': strain_mask_not_one_hot,
                'strain_mask_one_hot': strain_mask_one_hot}
    



class DataLoaderPreprocessedSupervised(SlimDataLoaderBase):
    def __init__(self, data, patch_size, final_patch_size, batch_size, video_length, processor, crop_size, is_val, do_data_aug, data_path, distance_map_power, binary_distance, oversample_foreground_percent=0.0,
                 memmap_mode="r", pseudo_3d_slices=1, pad_mode="edge",
                 pad_kwargs_data=None, pad_sides=None):
        """
        This is the basic data loader for 2D networks. It uses preprocessed data as produced by my (Fabian) preprocessing.
        You can load the data with load_dataset(folder) where folder is the folder where the npz files are located. If there
        are only npz files present in that folder, the data loader will unpack them on the fly. This may take a while
        and increase CPU usage. Therefore, I advise you to call unpack_dataset(folder) first, which will unpack all npz
        to npy. Don't forget to call delete_npy(folder) after you are done with training?
        Why all the hassle? Well the decathlon dataset is huge. Using npy for everything will consume >1 TB and that is uncool
        given that I (Fabian) will have to store that permanently on /datasets and my local computer. With this strategy all
        data is stored in a compressed format (factor 10 smaller) and only unpacked when needed.
        :param data: get this with load_dataset(folder, stage=0). Plug the return value in here and you are g2g (good to go)
        :param patch_size: what patch size will this data loader return? it is common practice to first load larger
        patches so that a central crop after data augmentation can be done to reduce border artifacts. If unsure, use
        get_patch_size() from data_augmentation.default_data_augmentation
        :param final_patch_size: what will the patch finally be cropped to (after data augmentation)? this is the patch
        size that goes into your network. We need this here because we will pad patients in here so that patches at the
        border of patients are sampled properly
        :param batch_size:
        :param num_batches: how many batches will the data loader produce before stopping? None=endless
        :param seed:
        :param stage: ignore this (Fabian only)
        :param transpose: ignore this
        :param random: sample randomly; CAREFUL! non-random sampling requires batch_size=1, otherwise you will iterate batch_size times over the dataset
        :param pseudo_3d_slices: 7 = 3 below and 3 above the center slice
        """
        super(DataLoaderPreprocessedSupervised, self).__init__(data, batch_size, None)
        if pad_kwargs_data is None:
            pad_kwargs_data = OrderedDict()
        self.pad_kwargs_data = pad_kwargs_data
        self.pad_mode = pad_mode
        self.is_val = is_val
        self.crop_size = crop_size
        self.pseudo_3d_slices = pseudo_3d_slices
        self.oversample_foreground_percent = oversample_foreground_percent
        self.final_patch_size = final_patch_size
        self.patch_size = patch_size
        self.list_of_keys = list(self._data.keys())
        self.need_to_pad = np.array(patch_size) - np.array(final_patch_size)
        self.memmap_mode = memmap_mode
        self.video_length = video_length
        self.percent = None
        if pad_sides is not None:
            if not isinstance(pad_sides, np.ndarray):
                pad_sides = np.array(pad_sides)
            self.need_to_pad += pad_sides
        self.pad_sides = pad_sides
        self.data_shape, self.seg_shape = self.determine_shapes()
        self.do_data_aug = do_data_aug
        self.distance_map_power = distance_map_power
        self.binary_distance = binary_distance

        self.preprocessing_transform = self.set_up_preprocessing_pipeline()
        self.pixel_transform, self.spatial_transform = self.set_up_augmentation_pipeline()

        self.processor = processor
        self.data_path = data_path
        

    def determine_shapes(self):
        num_seg = 1

        k = list(self._data.keys())[0]
        if isfile(self._data[k]['data_file'][:-4] + ".npy"):
            case_all_data = np.load(self._data[k]['data_file'][:-4] + ".npy", self.memmap_mode)
        else:
            case_all_data = np.load(self._data[k]['data_file'])['data']
        num_color_channels = case_all_data.shape[0] - num_seg
        data_shape = (self.batch_size, num_color_channels, *self.patch_size)
        seg_shape = (self.batch_size, num_seg, *self.patch_size)
        return data_shape, seg_shape

    def get_do_oversample(self, batch_idx):
        return not batch_idx < round(self.batch_size * (1 - self.oversample_foreground_percent))
    
    def select_deep_supervision(self, x):
        if self.deep_supervision:
            return x[0]
        else:
            return x
        

    def my_crop(self, labeled, seg, unlabeled):

        xm = labeled.shape[-1] / 2
        ym = labeled.shape[-2] / 2

        x1 = int(round(xm - self.final_patch_size[1] / 2))
        x2 = int(round(xm + self.final_patch_size[1] / 2))
        y1 = int(round(ym - self.final_patch_size[0] / 2))
        y2 = int(round(ym + self.final_patch_size[0] / 2))

        labeled = labeled[:, :, y1:y2, x1:x2]
        seg = seg[:, :, y1:y2, x1:x2]

        xm = unlabeled.shape[-1] / 2
        ym = unlabeled.shape[-2] / 2

        x1 = int(round(xm - self.final_patch_size[1] / 2))
        x2 = int(round(xm + self.final_patch_size[1] / 2))
        y1 = int(round(ym - self.final_patch_size[0] / 2))
        y2 = int(round(ym + self.final_patch_size[0] / 2))

        unlabeled = unlabeled[:, :, :, y1:y2, x1:x2]

        return labeled, seg, unlabeled
    
    def preprocess(self, image, mask):
        if mask is not None:
            data = {'image': image, 'mask': mask}
            transformed = self.preprocessing_transform(data)
            image = transformed['image']
            mask = transformed['mask']
        else:
            data = {'image': image}
            transformed = self.preprocessing_transform(data)
            image = transformed['image']

        return image, mask
    
    
    def augment(self, image, mask, pixel_transform, spatial_transform):
        
        temp = torch.clone(image)
        padding_mask = image == 0

        if mask is not None:
            data = {'image': image, 'mask': mask}
            pixel_transformed = pixel_transform(data)
            image = pixel_transformed['image']
            mask = pixel_transformed['mask']
            image[padding_mask] = 0
            data = {'image': image, 'mask': mask}
            spatial_transformed = spatial_transform(data)
            image = spatial_transformed['image']
            mask = spatial_transformed['mask']

            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(1, 3)
            #ax[0].imshow(temp[0].cpu(), cmap='gray')
            #ax[1].imshow(image[0].cpu(), cmap='gray')
            #ax[2].imshow(mask[0].cpu(), cmap='gray')
            #plt.show()
        else:
            data = {'image': image}
            pixel_transformed = pixel_transform(data)
            image = pixel_transformed['image']
            image[padding_mask] = 0
            data = {'image': image}
            spatial_transformed = spatial_transform(data)
            image = spatial_transformed['image']

            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(1, 2)
            #ax[0].imshow(temp[0].cpu(), cmap='gray')
            #ax[1].imshow(image[0].cpu(), cmap='gray')
            #plt.show()

        return image, mask
    
    def set_up_preprocessing_pipeline(self):
        preprocess = Compose([
                            T.SpatialPadd(keys=['image', 'mask'], spatial_size=self.final_patch_size, allow_missing_keys=True),
                            T.CenterSpatialCropd(keys=['image', 'mask'], roi_size=self.final_patch_size, allow_missing_keys=True),
                            T.ToTensor(dtype=torch.float32, device='cuda:0')
                            ])

        return preprocess
    
    def set_up_augmentation_pipeline(self):
        spatial_transform = [
                                    T.RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=0, allow_missing_keys=True), 
                                    T.RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=1, allow_missing_keys=True),
                                    T.RandRotated(keys=['image', 'mask'], prob=0.5, range_x=np.pi, mode=["bilinear", "nearest"], padding_mode="zeros", allow_missing_keys=True),
                                    T.RandZoomd(keys=['image', 'mask'], prob=0.5, min_zoom=0.5, max_zoom=1.5, mode=["bilinear", "nearest"], padding_mode="constant", allow_missing_keys=True),
                                    T.RandAffined(keys=['image', 'mask'], prob=0.5, translate_range=(-26, 26), mode=["bilinear", "nearest"], padding_mode="zeros", allow_missing_keys=True)
                            ]

        pixel_transform = [
                                        T.RandRicianNoised(keys=['image'], prob=0.5, std=0.075),
                                        T.RandGibbsNoised(keys=['image'], prob=0.5, alpha=[0.45, 0.75]),
                                        T.RandAdjustContrastd(keys=['image'], prob=0.5, gamma=(0.7, 1.5), allow_missing_keys=True),
                                        T.RandGaussianNoised(keys=['image'], prob=0.5, std=0.04, allow_missing_keys=True),
                                        T.RandGaussianSharpend(keys=['image'], prob=0.5, sigma1_x=(0.1, 0.2), sigma1_y=(0.1, 0.2), sigma2_x=(0.2, 0.4), sigma2_y=(0.2, 0.4), alpha=(2.0, 3.0), allow_missing_keys=True)
                                ]

        return pixel_transform, spatial_transform
    

    def generate_train_batch(self):
        #eval_idx = None
        selected_keys = np.random.choice(self.list_of_keys, self.batch_size, True, None)
        list_of_frames = []
        for i in range(len(selected_keys)):
            patient_id = selected_keys[i][:10]
            filtered = [x for x in self.list_of_keys if x[:10] in patient_id]
            list_of_frames.append(filtered)
        
        target_list = []
        unlabeled_list = []
        padding_need_list = []
        target_mask_list = []
        strain_mask_list = []
        database_list = []

        case_properties = []
        for j, frames in enumerate(list_of_frames):
            labeled_frame = frames[0]
            if 'properties' in self._data[labeled_frame].keys():
                properties = self._data[labeled_frame]['properties']
            else:
                properties = load_pickle(self._data[labeled_frame]['properties_file'])
            case_properties.append(properties)

            database_list.append(properties['Database'])

            #print(properties['ed_number'])
            #print(np.rint(np.array(properties['ed_number'])))
            #print('******************')

            ed_idx = np.rint(np.array(properties['ed_number'])).astype(np.uint8) % len(frames)
            es_idx = np.rint(np.array(properties['es_number'])).astype(np.uint8) % len(frames)

            depth = properties['size_after_resampling'][0]
            depth_idx = np.random.choice(depth)
            if 'patient021' in frames[0]:
                while depth_idx == 1:
                    depth_idx = np.random.choice(depth)

            global_labeled_idx = np.array([ed_idx, es_idx])

            frames = sorted(frames, key=lambda x: int(x[16:18]))
            frames = np.array(frames)

            cropped_unlabeled = torch.full(size=(self.video_length, 1, self.crop_size, self.crop_size), fill_value=torch.nan, device='cuda:0')
            cropped_seg = torch.full(size=(self.video_length, 5, self.crop_size, self.crop_size), fill_value=torch.nan, device='cuda:0')

            possible_indices = np.arange(len(frames))

            before_where = np.argwhere(possible_indices < ed_idx).reshape(-1,)
            after_where = np.argwhere(possible_indices >= ed_idx).reshape(-1,)

            before_indices = possible_indices[before_where]
            after_indices = possible_indices[after_where]

            possible_indices = np.concatenate([after_indices, before_indices])

            stop_idx = np.argwhere(possible_indices == es_idx)[0][0]
            chunk1 = possible_indices[:stop_idx + 1]
            chunk2 = possible_indices[stop_idx:]
            chunk2 = np.concatenate([np.array([possible_indices[0]]), chunk2[::-1]])

            r = np.random.randint(0, 2)
            if r == 0:
                possible_indices = chunk1
            elif r == 1:
                possible_indices = chunk2
            
            assert possible_indices[0] == ed_idx
            assert possible_indices[-1] == es_idx

            random_indices = np.random.choice(np.arange(0, len(possible_indices)), size=self.video_length - 2)
            target_mask = np.concatenate([np.array([True]), np.full_like(random_indices, fill_value=True), np.array([True])]).astype(bool)
            random_indices = np.concatenate([np.array([0]), random_indices, np.array([len(possible_indices) - 1])])
            sorted_indices = np.argsort(random_indices)

            random_indices = random_indices[sorted_indices]
            target_mask = target_mask[sorted_indices]
            target_mask = torch.from_numpy(target_mask).to('cuda:0')

            frame_indices = possible_indices[random_indices]

            assert len(frame_indices) == self.video_length
            video = frames[frame_indices]


            labeled_idx = np.where(np.isin(frame_indices, global_labeled_idx))[0]

            for frame_idx, t in enumerate(video):

                filename = os.path.basename(t) + ".npy"

                data = np.load(os.path.join(self.data_path, filename))

                img = data[0]
                gt = data[1]
                mask = data[2:]

                with open(os.path.join(self.data_path, filename[:-4] + '.pkl'), 'rb') as f:
                    pkl_data = pickle.load(f)
                    padding_need = pkl_data['padding_need']
                    assert padding_need.shape[-1] == mask.shape[-1] == depth
                    padding_need_list.append(torch.from_numpy(padding_need[:, depth_idx]).to('cuda:0'))
                
                assert img.shape[-1] == depth

                img = torch.from_numpy(img[:, :, depth_idx]).to('cuda:0')
                gt = torch.from_numpy(gt[:, :, depth_idx]).to('cuda:0')
                mask = torch.from_numpy(mask[:, :, :, depth_idx]).to('cuda:0')

                mask[mask < 0] = 0
                gt[gt < 0] = 0

                to_cat = gt[None]

                cropped_seg[frame_idx] = torch.cat([to_cat, mask], dim=0)
                cropped_unlabeled[frame_idx] = img


            cropped_unlabeled = (cropped_unlabeled - cropped_unlabeled.min()) / (cropped_unlabeled.max() - cropped_unlabeled.min() + 1e-8)

            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(1, 4)
            #ax[0].imshow(cropped_seg[:, 0][:, None][0, 0].cpu(), cmap='gray')
            #ax[1].imshow(cropped_seg[:, 0][:, None][1, 0].cpu(), cmap='gray')
            #ax[2].imshow(cropped_seg[:, 0][:, None][2, 0].cpu(), cmap='gray')
            #ax[3].imshow(cropped_seg[:, 0][:, None][3, 0].cpu(), cmap='gray')
            #plt.show()

            assert torch.count_nonzero(torch.any(~torch.isnan(cropped_seg[:, 0][:, None].reshape(self.video_length, -1)), dim=-1)) == len(video)
            assert torch.count_nonzero(target_mask) == len(video)

            if not self.is_val and self.do_data_aug:
                temp_before_augment = torch.clone(cropped_unlabeled).to('cpu')

                current_pixel_transform = np.random.choice(self.pixel_transform)
                current_spatial_transform = np.random.choice(self.spatial_transform)

                #print(current_pixel_transform)
                
                seed = random.randint(0, 2**32-1)

                for t in range(len(cropped_unlabeled)):

                    current_pixel_transform = current_pixel_transform.set_random_state(seed=seed)
                    current_spatial_transform = current_spatial_transform.set_random_state(seed=seed)

                    cropped_unlabeled[t], cropped_seg[t] = self.augment(image=cropped_unlabeled[t], mask=cropped_seg[t], pixel_transform=current_pixel_transform, spatial_transform=current_spatial_transform)

            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(3, len(cropped_unlabeled))
            #for t in range(len(cropped_unlabeled)):
            #    ax[0, t].imshow(temp_before_augment[t, 0].cpu(), cmap='gray')
            #    ax[1, t].imshow(cropped_unlabeled[t, 0].cpu(), cmap='gray')
            #    ax[2, t].imshow(cropped_seg[t, 0].cpu(), cmap='gray')
            #plt.show()
                    
            cropped_unlabeled = NormalizeIntensity()(cropped_unlabeled)

            #print(cropped_seg.shape)
            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(1, 4)
            #ax[0].imshow(cropped_seg[0, 0].cpu(), cmap='gray')
            #ax[1].imshow(cropped_seg[1, 0].cpu(), cmap='gray')
            #ax[2].imshow(cropped_seg[2, 0].cpu(), cmap='gray')
            #ax[3].imshow(cropped_seg[3, 0].cpu(), cmap='gray')
            #plt.show()


            unlabeled_list.append(cropped_unlabeled)
            target_list.append(cropped_seg[:, 0][:, None])
            target_mask_list.append(target_mask)
            strain_mask_list.append(cropped_seg[:, 1:])

        unlabeled = torch.stack(unlabeled_list, dim=1) # T, B, 1, H, W
        target = torch.stack(target_list, dim=1) # T, B, 1, H, W
        strain_mask = torch.stack(strain_mask_list, dim=1) # T, B, 4, H, W
        target_mask = torch.stack(target_mask_list, dim=1) # T, B
        padding_need = torch.stack(padding_need_list, dim=0) # B, 4

        strain_mask = torch.pow(strain_mask, self.distance_map_power)
        if self.binary_distance:
            strain_mask = strain_mask.long()
        strain_mask_not_one_hot = strain_mask[:, :, -1, :, :][:, :, None, :, :]
        strain_mask_one_hot = strain_mask[:, :, :-1, :, :]

        #print(seg.shape)
        #print(np.unique(seg))
        #matplotlib.use('QtAgg')
        #fig, ax = plt.subplots(1, 2)
        #ax[0].imshow(labeled[3, 0], cmap='gray')
        #ax[1].imshow(seg[3, 0], cmap='gray')
        #plt.show()

        return {'unlabeled':unlabeled, 
                'target': target,
                'in_between': None,
                'database': database_list,
                'padding_need': padding_need,
                'properties': case_properties,
                'target_mask': target_mask,
                'strain_mask': strain_mask_not_one_hot,
                'strain_mask_one_hot': strain_mask_one_hot}
    



class DataLoaderPreprocessedValidation(SlimDataLoaderBase):
    def __init__(self, data, patch_size, final_patch_size, batch_size, video_length, processor, crop_size, is_val, do_data_aug, distance_map_power, binary_distance, oversample_foreground_percent=0.0,
                 memmap_mode="r", pseudo_3d_slices=1, pad_mode="edge",
                 pad_kwargs_data=None, pad_sides=None):
        """
        This is the basic data loader for 2D networks. It uses preprocessed data as produced by my (Fabian) preprocessing.
        You can load the data with load_dataset(folder) where folder is the folder where the npz files are located. If there
        are only npz files present in that folder, the data loader will unpack them on the fly. This may take a while
        and increase CPU usage. Therefore, I advise you to call unpack_dataset(folder) first, which will unpack all npz
        to npy. Don't forget to call delete_npy(folder) after you are done with training?
        Why all the hassle? Well the decathlon dataset is huge. Using npy for everything will consume >1 TB and that is uncool
        given that I (Fabian) will have to store that permanently on /datasets and my local computer. With this strategy all
        data is stored in a compressed format (factor 10 smaller) and only unpacked when needed.
        :param data: get this with load_dataset(folder, stage=0). Plug the return value in here and you are g2g (good to go)
        :param patch_size: what patch size will this data loader return? it is common practice to first load larger
        patches so that a central crop after data augmentation can be done to reduce border artifacts. If unsure, use
        get_patch_size() from data_augmentation.default_data_augmentation
        :param final_patch_size: what will the patch finally be cropped to (after data augmentation)? this is the patch
        size that goes into your network. We need this here because we will pad patients in here so that patches at the
        border of patients are sampled properly
        :param batch_size:
        :param num_batches: how many batches will the data loader produce before stopping? None=endless
        :param seed:
        :param stage: ignore this (Fabian only)
        :param transpose: ignore this
        :param random: sample randomly; CAREFUL! non-random sampling requires batch_size=1, otherwise you will iterate batch_size times over the dataset
        :param pseudo_3d_slices: 7 = 3 below and 3 above the center slice
        """
        super(DataLoaderPreprocessedValidation, self).__init__(data, batch_size, None)
        if pad_kwargs_data is None:
            pad_kwargs_data = OrderedDict()
        self.pad_kwargs_data = pad_kwargs_data
        self.pad_mode = pad_mode
        self.is_val = is_val
        self.crop_size = crop_size
        self.pseudo_3d_slices = pseudo_3d_slices
        self.oversample_foreground_percent = oversample_foreground_percent
        self.final_patch_size = final_patch_size
        self.patch_size = patch_size
        self.list_of_keys = list(self._data.keys())
        self.need_to_pad = np.array(patch_size) - np.array(final_patch_size)
        self.memmap_mode = memmap_mode
        self.video_length = video_length
        self.percent = None
        if pad_sides is not None:
            if not isinstance(pad_sides, np.ndarray):
                pad_sides = np.array(pad_sides)
            self.need_to_pad += pad_sides
        self.pad_sides = pad_sides
        self.data_shape, self.seg_shape = self.determine_shapes()
        self.do_data_aug = do_data_aug
        self.binary_distance = binary_distance

        self.preprocessing_transform = self.set_up_preprocessing_pipeline()
        self.pixel_transform, self.spatial_transform = self.set_up_augmentation_pipeline()

        self.processor = processor
        self.distance_map_power = distance_map_power
        

    def determine_shapes(self):
        num_seg = 1

        k = list(self._data.keys())[0]
        if isfile(self._data[k]['data_file'][:-4] + ".npy"):
            case_all_data = np.load(self._data[k]['data_file'][:-4] + ".npy", self.memmap_mode)
        else:
            case_all_data = np.load(self._data[k]['data_file'])['data']
        num_color_channels = case_all_data.shape[0] - num_seg
        data_shape = (self.batch_size, num_color_channels, *self.patch_size)
        seg_shape = (self.batch_size, num_seg, *self.patch_size)
        return data_shape, seg_shape

    def get_do_oversample(self, batch_idx):
        return not batch_idx < round(self.batch_size * (1 - self.oversample_foreground_percent))
    
    def select_deep_supervision(self, x):
        if self.deep_supervision:
            return x[0]
        else:
            return x
        

    def my_crop(self, labeled, seg, unlabeled):

        xm = labeled.shape[-1] / 2
        ym = labeled.shape[-2] / 2

        x1 = int(round(xm - self.final_patch_size[1] / 2))
        x2 = int(round(xm + self.final_patch_size[1] / 2))
        y1 = int(round(ym - self.final_patch_size[0] / 2))
        y2 = int(round(ym + self.final_patch_size[0] / 2))

        labeled = labeled[:, :, y1:y2, x1:x2]
        seg = seg[:, :, y1:y2, x1:x2]

        xm = unlabeled.shape[-1] / 2
        ym = unlabeled.shape[-2] / 2

        x1 = int(round(xm - self.final_patch_size[1] / 2))
        x2 = int(round(xm + self.final_patch_size[1] / 2))
        y1 = int(round(ym - self.final_patch_size[0] / 2))
        y2 = int(round(ym + self.final_patch_size[0] / 2))

        unlabeled = unlabeled[:, :, :, y1:y2, x1:x2]

        return labeled, seg, unlabeled
    
    def preprocess(self, image, mask):
        if mask is not None:
            data = {'image': image, 'mask': mask}
            transformed = self.preprocessing_transform(data)
            image = transformed['image']
            mask = transformed['mask']
        else:
            data = {'image': image}
            transformed = self.preprocessing_transform(data)
            image = transformed['image']

        return image, mask
    
    
    def augment(self, image, mask, pixel_transform, spatial_transform):
        
        temp = torch.clone(image)
        padding_mask = image == 0

        if mask is not None:
            data = {'image': image, 'mask': mask}
            pixel_transformed = pixel_transform(data)
            image = pixel_transformed['image']
            mask = pixel_transformed['mask']
            image[padding_mask] = 0
            data = {'image': image, 'mask': mask}
            spatial_transformed = spatial_transform(data)
            image = spatial_transformed['image']
            mask = spatial_transformed['mask']

            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(1, 3)
            #ax[0].imshow(temp[0].cpu(), cmap='gray')
            #ax[1].imshow(image[0].cpu(), cmap='gray')
            #ax[2].imshow(mask[0].cpu(), cmap='gray')
            #plt.show()
        else:
            data = {'image': image}
            pixel_transformed = pixel_transform(data)
            image = pixel_transformed['image']
            image[padding_mask] = 0
            data = {'image': image}
            spatial_transformed = spatial_transform(data)
            image = spatial_transformed['image']

            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(1, 2)
            #ax[0].imshow(temp[0].cpu(), cmap='gray')
            #ax[1].imshow(image[0].cpu(), cmap='gray')
            #plt.show()

        return image, mask
    
    def set_up_preprocessing_pipeline(self):
        preprocess = Compose([
                            T.SpatialPadd(keys=['image', 'mask'], spatial_size=self.final_patch_size, allow_missing_keys=True),
                            T.CenterSpatialCropd(keys=['image', 'mask'], roi_size=self.final_patch_size, allow_missing_keys=True),
                            T.ToTensor(dtype=torch.float32, device='cuda:0')
                            ])

        return preprocess
    
    def set_up_augmentation_pipeline(self):
        spatial_transform = [
                                    T.RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=0, allow_missing_keys=True), 
                                    T.RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=1, allow_missing_keys=True),
                                    T.RandRotated(keys=['image', 'mask'], prob=0.5, range_x=np.pi, mode=["bilinear", "nearest"], padding_mode="zeros", allow_missing_keys=True),
                                    T.RandZoomd(keys=['image', 'mask'], prob=0.5, min_zoom=0.5, max_zoom=1.5, mode=["bilinear", "nearest"], padding_mode="constant", allow_missing_keys=True),
                                    T.RandAffined(keys=['image', 'mask'], prob=0.5, translate_range=(-26, 26), mode=["bilinear", "nearest"], padding_mode="zeros", allow_missing_keys=True)
                            ]

        pixel_transform = [
                                        T.RandRicianNoised(keys=['image'], prob=0.5, std=0.075),
                                        T.RandGibbsNoised(keys=['image'], prob=0.5, alpha=[0.45, 0.75]),
                                        T.RandAdjustContrastd(keys=['image'], prob=0.5, gamma=(0.7, 1.5), allow_missing_keys=True),
                                        T.RandGaussianNoised(keys=['image'], prob=0.5, std=0.04, allow_missing_keys=True),
                                        T.RandGaussianSharpend(keys=['image'], prob=0.5, sigma1_x=(0.1, 0.2), sigma1_y=(0.1, 0.2), sigma2_x=(0.2, 0.4), sigma2_y=(0.2, 0.4), alpha=(2.0, 3.0), allow_missing_keys=True)
                                ]

        return pixel_transform, spatial_transform
    

    def generate_train_batch(self):
        #eval_idx = None
        selected_keys = np.random.choice(self.list_of_keys, self.batch_size, True, None)
        list_of_frames = []
        for i in range(len(selected_keys)):
            patient_id = selected_keys[i][:10]
            filtered = [x for x in self.list_of_keys if x[:10] in patient_id]
            list_of_frames.append(filtered)
        
        target_list = []
        unlabeled_list = []
        padding_need_list = []
        target_mask_list = []
        strain_mask_list = []
        database_list = []

        case_properties = []
        for j, frames in enumerate(list_of_frames):
            labeled_frame = frames[0]
            if 'properties' in self._data[labeled_frame].keys():
                properties = self._data[labeled_frame]['properties']
            else:
                properties = load_pickle(self._data[labeled_frame]['properties_file'])
            case_properties.append(properties)

            database_list.append(properties['Database'])

            #print(properties['ed_number'])
            #print(np.rint(np.array(properties['ed_number'])))
            #print('******************')

            ed_idx = np.rint(np.array(properties['ed_number'])).astype(np.uint8) % len(frames)
            es_idx = np.rint(np.array(properties['es_number'])).astype(np.uint8) % len(frames)

            depth = properties['size_after_resampling'][0]
            depth_idx = np.random.choice(depth)

            global_labeled_idx = np.array([ed_idx, es_idx])

            frames = sorted(frames, key=lambda x: int(x[16:18]))
            frames = np.array(frames)

            cropped_unlabeled = torch.full(size=(self.video_length, 1, self.crop_size, self.crop_size), fill_value=torch.nan, device='cuda:0')
            cropped_seg = torch.full(size=(self.video_length, 4, self.crop_size, self.crop_size), fill_value=torch.nan, device='cuda:0')

            possible_indices = np.arange(len(frames))

            before_where = np.argwhere(possible_indices < ed_idx).reshape(-1,)
            after_where = np.argwhere(possible_indices >= ed_idx).reshape(-1,)

            before_indices = possible_indices[before_where]
            after_indices = possible_indices[after_where]

            possible_indices = np.concatenate([after_indices, before_indices])

            stop_idx = np.argwhere(possible_indices == es_idx)[0][0]
            chunk1 = possible_indices[:stop_idx + 1]
            chunk2 = possible_indices[stop_idx:]
            chunk2 = np.concatenate([np.array([possible_indices[0]]), chunk2[::-1]])

            r = np.random.randint(0, 2)
            if r == 0:
                possible_indices = chunk1
            elif r == 1:
                possible_indices = chunk2
            
            assert possible_indices[0] == ed_idx
            assert possible_indices[-1] == es_idx

            target_mask = np.full_like(possible_indices, fill_value=False).astype(bool)
            target_mask[0] = True
            target_mask[-1] = True
            target_mask = torch.from_numpy(target_mask).to('cuda:0')

            video = frames[possible_indices]

            labeled_idx = np.where(np.isin(possible_indices, global_labeled_idx))[0]

            cropped_unlabeled = torch.full(size=(len(possible_indices), 1, self.crop_size, self.crop_size), fill_value=torch.nan, device='cuda:0')
            cropped_seg = torch.full(size=(len(possible_indices), 5, self.crop_size, self.crop_size), fill_value=torch.nan, device='cuda:0')

            for frame_idx, t in enumerate(video):

                filename = os.path.basename(t) + ".npy"

                data = np.load(os.path.join(r'Lib_resampling_training_mask', filename))

                img = data[0]
                gt = data[1]
                mask = data[2:]

                with open(os.path.join(r'Lib_resampling_training_mask', filename[:-4] + '.pkl'), 'rb') as f:
                    pkl_data = pickle.load(f)
                    padding_need = pkl_data['padding_need']
                    assert padding_need.shape[-1] == mask.shape[-1] == depth
                    padding_need_list.append(torch.from_numpy(padding_need[:, depth_idx]).to('cuda:0'))
                
                assert img.shape[-1] == depth

                img = torch.from_numpy(img[:, :, depth_idx]).to('cuda:0')
                gt = torch.from_numpy(gt[:, :, depth_idx]).to('cuda:0')
                mask = torch.from_numpy(mask[:, :, :, depth_idx]).to('cuda:0')

                mask[mask < 0] = 0
                gt[gt < 0] = 0

                if target_mask[frame_idx] == True:
                    to_cat = gt[None]
                else:
                    to_cat = torch.full_like(gt[None], fill_value=torch.nan)

                cropped_seg[frame_idx] = torch.cat([to_cat, mask], dim=0)
                cropped_unlabeled[frame_idx] = img


            cropped_unlabeled = (cropped_unlabeled - cropped_unlabeled.min()) / (cropped_unlabeled.max() - cropped_unlabeled.min() + 1e-8)

            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(1, 4)
            #ax[0].imshow(cropped_seg[:, 0][:, None][0, 0].cpu(), cmap='gray')
            #ax[1].imshow(cropped_seg[:, 0][:, None][1, 0].cpu(), cmap='gray')
            #ax[2].imshow(cropped_seg[:, 0][:, None][2, 0].cpu(), cmap='gray')
            #ax[3].imshow(cropped_seg[:, 0][:, None][3, 0].cpu(), cmap='gray')
            #plt.show()

            assert torch.count_nonzero(torch.any(~torch.isnan(cropped_seg[:, 0][:, None].reshape(len(video), -1)), dim=-1)) == 2
            assert target_mask[0] == True
            assert target_mask[-1] == True
            assert torch.count_nonzero(target_mask) == 2
                    
            cropped_unlabeled = NormalizeIntensity()(cropped_unlabeled)

            #print(cropped_seg.shape)
            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(1, 4)
            #ax[0].imshow(cropped_seg[0, 0].cpu(), cmap='gray')
            #ax[1].imshow(cropped_seg[1, 0].cpu(), cmap='gray')
            #ax[2].imshow(cropped_seg[2, 0].cpu(), cmap='gray')
            #ax[3].imshow(cropped_seg[3, 0].cpu(), cmap='gray')
            #plt.show()


            unlabeled_list.append(cropped_unlabeled)
            target_list.append(cropped_seg[:, 0][:, None])
            target_mask_list.append(target_mask)
            strain_mask_list.append(cropped_seg[:, 1:])

        unlabeled = torch.stack(unlabeled_list, dim=1) # T, B, 1, H, W
        target = torch.stack(target_list, dim=1) # T, B, 1, H, W
        strain_mask = torch.stack(strain_mask_list, dim=1) # T, B, 3, H, W
        target_mask = torch.stack(target_mask_list, dim=1) # T, B
        padding_need = torch.stack(padding_need_list, dim=0) # T, 4

        padding_need_reduced = padding_need.sum(-1)
        assert torch.all(padding_need_reduced == padding_need_reduced[0])
        padding_need = padding_need[0][None]

        strain_mask = torch.pow(strain_mask, self.distance_map_power)
        if self.binary_distance:
            strain_mask = strain_mask.long()
        strain_mask_not_one_hot = strain_mask[:, :, -1, :, :][:, :, None, :, :]
        strain_mask_one_hot = strain_mask[:, :, :-1, :, :]

        #print(seg.shape)
        #print(np.unique(seg))
        #matplotlib.use('QtAgg')
        #fig, ax = plt.subplots(1, 2)
        #ax[0].imshow(labeled[3, 0], cmap='gray')
        #ax[1].imshow(seg[3, 0], cmap='gray')
        #plt.show()

        return {'unlabeled':unlabeled, 
                'target': target,
                'in_between': None,
                'database': database_list,
                'padding_need': padding_need,
                'properties': case_properties,
                'target_mask': target_mask,
                'strain_mask': strain_mask_not_one_hot,
                'strain_mask_one_hot': strain_mask_one_hot}
    


class DataLoaderAugmentRegular(SlimDataLoaderBase):
    def __init__(self, data, patch_size, final_patch_size, batch_size, video_length, processor, crop_size, is_val, do_data_aug, oversample_foreground_percent=0.0,
                 memmap_mode="r", pseudo_3d_slices=1, pad_mode="edge",
                 pad_kwargs_data=None, pad_sides=None):
        """
        This is the basic data loader for 2D networks. It uses preprocessed data as produced by my (Fabian) preprocessing.
        You can load the data with load_dataset(folder) where folder is the folder where the npz files are located. If there
        are only npz files present in that folder, the data loader will unpack them on the fly. This may take a while
        and increase CPU usage. Therefore, I advise you to call unpack_dataset(folder) first, which will unpack all npz
        to npy. Don't forget to call delete_npy(folder) after you are done with training?
        Why all the hassle? Well the decathlon dataset is huge. Using npy for everything will consume >1 TB and that is uncool
        given that I (Fabian) will have to store that permanently on /datasets and my local computer. With this strategy all
        data is stored in a compressed format (factor 10 smaller) and only unpacked when needed.
        :param data: get this with load_dataset(folder, stage=0). Plug the return value in here and you are g2g (good to go)
        :param patch_size: what patch size will this data loader return? it is common practice to first load larger
        patches so that a central crop after data augmentation can be done to reduce border artifacts. If unsure, use
        get_patch_size() from data_augmentation.default_data_augmentation
        :param final_patch_size: what will the patch finally be cropped to (after data augmentation)? this is the patch
        size that goes into your network. We need this here because we will pad patients in here so that patches at the
        border of patients are sampled properly
        :param batch_size:
        :param num_batches: how many batches will the data loader produce before stopping? None=endless
        :param seed:
        :param stage: ignore this (Fabian only)
        :param transpose: ignore this
        :param random: sample randomly; CAREFUL! non-random sampling requires batch_size=1, otherwise you will iterate batch_size times over the dataset
        :param pseudo_3d_slices: 7 = 3 below and 3 above the center slice
        """
        super(DataLoaderAugmentRegular, self).__init__(data, batch_size, None)
        if pad_kwargs_data is None:
            pad_kwargs_data = OrderedDict()
        self.pad_kwargs_data = pad_kwargs_data
        self.pad_mode = pad_mode
        self.is_val = is_val
        self.crop_size = crop_size
        self.pseudo_3d_slices = pseudo_3d_slices
        self.oversample_foreground_percent = oversample_foreground_percent
        self.final_patch_size = final_patch_size
        self.patch_size = patch_size
        self.list_of_keys = list(self._data.keys())
        self.need_to_pad = np.array(patch_size) - np.array(final_patch_size)
        self.memmap_mode = memmap_mode
        self.video_length = video_length
        self.percent = None
        if pad_sides is not None:
            if not isinstance(pad_sides, np.ndarray):
                pad_sides = np.array(pad_sides)
            self.need_to_pad += pad_sides
        self.pad_sides = pad_sides
        self.data_shape, self.seg_shape = self.determine_shapes()
        self.do_data_aug = do_data_aug

        self.preprocessing_transform = self.set_up_preprocessing_pipeline()
        self.pixel_transform, self.spatial_transform = self.set_up_augmentation_pipeline()

        self.processor = processor

    
    def custom_next(self, training_percent):
        return self.generate_train_batch(training_percent)
        

    def determine_shapes(self):
        num_seg = 1

        k = list(self._data.keys())[0]
        if isfile(self._data[k]['data_file'][:-4] + ".npy"):
            case_all_data = np.load(self._data[k]['data_file'][:-4] + ".npy", self.memmap_mode)
        else:
            case_all_data = np.load(self._data[k]['data_file'])['data']
        num_color_channels = case_all_data.shape[0] - num_seg
        data_shape = (self.batch_size, num_color_channels, *self.patch_size)
        seg_shape = (self.batch_size, num_seg, *self.patch_size)
        return data_shape, seg_shape

    def get_do_oversample(self, batch_idx):
        return not batch_idx < round(self.batch_size * (1 - self.oversample_foreground_percent))
    
    def select_deep_supervision(self, x):
        if self.deep_supervision:
            return x[0]
        else:
            return x
        

    def my_crop(self, labeled, seg, unlabeled):

        xm = labeled.shape[-1] / 2
        ym = labeled.shape[-2] / 2

        x1 = int(round(xm - self.final_patch_size[1] / 2))
        x2 = int(round(xm + self.final_patch_size[1] / 2))
        y1 = int(round(ym - self.final_patch_size[0] / 2))
        y2 = int(round(ym + self.final_patch_size[0] / 2))

        labeled = labeled[:, :, y1:y2, x1:x2]
        seg = seg[:, :, y1:y2, x1:x2]

        xm = unlabeled.shape[-1] / 2
        ym = unlabeled.shape[-2] / 2

        x1 = int(round(xm - self.final_patch_size[1] / 2))
        x2 = int(round(xm + self.final_patch_size[1] / 2))
        y1 = int(round(ym - self.final_patch_size[0] / 2))
        y2 = int(round(ym + self.final_patch_size[0] / 2))

        unlabeled = unlabeled[:, :, :, y1:y2, x1:x2]

        return labeled, seg, unlabeled
    
    def preprocess(self, image, mask):
        if mask is not None:
            data = {'image': image, 'mask': mask}
            transformed = self.preprocessing_transform(data)
            image = transformed['image']
            mask = transformed['mask']
        else:
            data = {'image': image}
            transformed = self.preprocessing_transform(data)
            image = transformed['image']

        return image, mask
    
    
    def augment(self, image, mask, pixel_transform, spatial_transform):
        
        temp = torch.clone(image)
        padding_mask = image == 0

        if mask is not None:
            data = {'image': image, 'mask': mask}
            pixel_transformed = pixel_transform(data)
            image = pixel_transformed['image']
            mask = pixel_transformed['mask']
            image[padding_mask] = 0
            data = {'image': image, 'mask': mask}
            spatial_transformed = spatial_transform(data)
            image = spatial_transformed['image']
            mask = spatial_transformed['mask']

            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(1, 3)
            #ax[0].imshow(temp[0].cpu(), cmap='gray')
            #ax[1].imshow(image[0].cpu(), cmap='gray')
            #ax[2].imshow(mask[0].cpu(), cmap='gray')
            #plt.show()
        else:
            data = {'image': image}
            pixel_transformed = pixel_transform(data)
            image = pixel_transformed['image']
            image[padding_mask] = 0
            data = {'image': image}
            spatial_transformed = spatial_transform(data)
            image = spatial_transformed['image']

            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(1, 2)
            #ax[0].imshow(temp[0].cpu(), cmap='gray')
            #ax[1].imshow(image[0].cpu(), cmap='gray')
            #plt.show()

        return image, mask
    
    def set_up_preprocessing_pipeline(self):
        preprocess = Compose([
                            T.SpatialPadd(keys=['image', 'mask'], spatial_size=self.final_patch_size, allow_missing_keys=True),
                            T.CenterSpatialCropd(keys=['image', 'mask'], roi_size=self.final_patch_size, allow_missing_keys=True),
                            T.ToTensor(dtype=torch.float32, device='cuda:0')
                            ])

        return preprocess
    
    def set_up_augmentation_pipeline(self):
        spatial_transform = [
                                    T.RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=0, allow_missing_keys=True), 
                                    T.RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=1, allow_missing_keys=True),
                                    T.RandRotated(keys=['image', 'mask'], prob=0.5, range_x=np.pi, mode=["bilinear", "nearest"], padding_mode="zeros", allow_missing_keys=True),
                                    T.RandZoomd(keys=['image', 'mask'], prob=0.5, min_zoom=0.5, max_zoom=1.5, mode=["bilinear", "nearest"], padding_mode="constant", allow_missing_keys=True),
                                    T.RandAffined(keys=['image', 'mask'], prob=0.5, translate_range=(-26, 26), mode=["bilinear", "nearest"], padding_mode="zeros", allow_missing_keys=True)
                            ]

        pixel_transform = [
                                        T.RandRicianNoised(keys=['image'], prob=0.5, std=0.075),
                                        T.RandGibbsNoised(keys=['image'], prob=0.5, alpha=[0.45, 0.75]),
                                        T.RandAdjustContrastd(keys=['image'], prob=0.5, gamma=(0.7, 1.5), allow_missing_keys=True),
                                        T.RandGaussianNoised(keys=['image'], prob=0.5, std=0.04, allow_missing_keys=True),
                                        T.RandGaussianSharpend(keys=['image'], prob=0.5, sigma1_x=(0.1, 0.2), sigma1_y=(0.1, 0.2), sigma2_x=(0.2, 0.4), sigma2_y=(0.2, 0.4), alpha=(2.0, 3.0), allow_missing_keys=True)
                                ]

        return pixel_transform, spatial_transform
    

    def generate_train_batch(self, training_percent):
        gap = int(math.ceil(20 * (training_percent + 1e-7)))
        #eval_idx = None
        selected_keys = np.random.choice(self.list_of_keys, self.batch_size, True, None)
        list_of_frames = []
        for i in range(len(selected_keys)):
            patient_id = selected_keys[i][:10]
            filtered = [x for x in self.list_of_keys if x[:10] in patient_id]
            list_of_frames.append(filtered)
        
        target_list = []
        unlabeled_list = []
        padding_need_list = []
        target_mask_list = []
        database_list = []

        case_properties = []
        for j, frames in enumerate(list_of_frames):
            labeled_frame = frames[0]
            if 'properties' in self._data[labeled_frame].keys():
                properties = self._data[labeled_frame]['properties']
            else:
                properties = load_pickle(self._data[labeled_frame]['properties_file'])
            case_properties.append(properties)

            database_list.append(properties['Database'])

            #print(properties['ed_number'])
            #print(np.rint(np.array(properties['ed_number'])))
            #print('******************')

            ed_idx = np.rint(np.array(properties['ed_number'])).astype(np.uint8) % len(frames)
            es_idx = np.rint(np.array(properties['es_number'])).astype(np.uint8) % len(frames)

            depth = properties['size_after_resampling'][0]
            depth_idx = np.random.choice(depth)

            global_labeled_idx = np.array([ed_idx, es_idx])

            frames = sorted(frames, key=lambda x: int(x[16:18]))
            frames = np.array(frames)

            unlabeled = torch.full(size=(self.video_length, 1, *self.final_patch_size), fill_value=torch.nan, device='cuda:0')
            seg = torch.full(size=(self.video_length, 1, *self.final_patch_size), fill_value=torch.nan, device='cuda:0')
            cropped_unlabeled = torch.full(size=(self.video_length, 1, self.crop_size, self.crop_size), fill_value=torch.nan, device='cuda:0')
            cropped_seg = torch.full(size=(self.video_length, 1, self.crop_size, self.crop_size), fill_value=torch.nan, device='cuda:0')

            possible_indices = np.arange(len(frames))

            before_where = np.argwhere(possible_indices < ed_idx).reshape(-1,)
            after_where = np.argwhere(possible_indices >= ed_idx).reshape(-1,)

            before_indices = possible_indices[before_where]
            after_indices = possible_indices[after_where]

            possible_indices = np.concatenate([after_indices, before_indices])

            possible_indices = np.pad(possible_indices, pad_width=(1000, 1000), mode='wrap')
            start = np.random.randint(0, len(possible_indices) - (self.video_length * gap))
            filter_indices = np.zeros(shape=(self.video_length,), dtype=int)
            filter_indices[0] = start
            for o in range(1, len(filter_indices)):
                filter_indices[o] = filter_indices[o - 1] + gap
            random_indices = possible_indices[filter_indices]
            
            target_mask = np.full_like(random_indices, fill_value=False)
            sorted_indices = np.argsort(random_indices)

            random_indices = random_indices[sorted_indices]
            target_mask = target_mask[sorted_indices]
            target_mask = torch.from_numpy(target_mask).to('cuda:0')

            frame_indices = possible_indices[random_indices]

            assert len(frame_indices) == self.video_length
            video = frames[frame_indices]


            labeled_idx = np.where(np.isin(frame_indices, global_labeled_idx))[0]

            for frame_idx, t in enumerate(video):

                if '_u' in t:
                    if not isfile(self.un_data[t]['data_file'][:-4] + ".npy"):
                        # lets hope you know what you're doing
                        case_all_data = np.load(self.un_data[t]['data_file'][:-4] + ".npz")['data']
                    else:
                        case_all_data = np.load(self.un_data[t]['data_file'][:-4] + ".npy", self.memmap_mode)
                else:
                    if not isfile(self._data[t]['data_file'][:-4] + ".npy"):
                        # lets hope you know what you're doing
                        case_all_data = np.load(self._data[t]['data_file'][:-4] + ".npz")['data']
                    else:
                        case_all_data = np.load(self._data[t]['data_file'][:-4] + ".npy", self.memmap_mode)

                # this is for when there is just a 2d slice in case_all_data (2d support)
                if len(case_all_data.shape) == 3:
                    case_all_data = case_all_data[:, None]
                
                assert case_all_data.shape[1] == depth

                # now crop case_all_data to contain just the slice of interest. If we want additional slice above and
                # below the current slice, here is where we get them. We stack those as additional color channels
                slice_data = case_all_data[:, depth_idx]

                # case all data should now be (c, x, y)
                assert len(slice_data.shape) == 3

                if '_u' not in t:
                    image = slice_data[:-1].copy()
                    mask = slice_data[-1:].copy()
                    mask[mask < 0] = 0

                    image, mask = self.preprocess(image, mask)

                    if target_mask[frame_idx] == True:
                        seg[frame_idx] = mask
                        assert frame_idx in labeled_idx
                else:
                    image = slice_data.copy()

                    image, _ = self.preprocess(image, None)
                
                unlabeled[frame_idx] = image

            with torch.no_grad():
                mean_centroid, _ = self.processor.preprocess_no_registration(data=torch.clone(unlabeled)) # T, C(1), H, W

                cropped_unlabeled, padding_need = self.processor.crop_and_pad(data=unlabeled, mean_centroid=mean_centroid)
                padding_need_list.append(padding_need)

                cropped_seg, _ = self.processor.crop_and_pad(data=seg, mean_centroid=mean_centroid)

            cropped_unlabeled = (cropped_unlabeled - cropped_unlabeled.min()) / (cropped_unlabeled.max() - cropped_unlabeled.min() + 1e-8)

            assert torch.count_nonzero(torch.any(~torch.isnan(cropped_seg.reshape(self.video_length, -1)), dim=-1)) == 0
            assert torch.count_nonzero(target_mask) == 0

            if not self.is_val and self.do_data_aug:
                temp_before_augment = torch.clone(cropped_unlabeled).to('cpu')

                current_pixel_transform = np.random.choice(self.pixel_transform)
                current_spatial_transform = np.random.choice(self.spatial_transform)

                #print(current_pixel_transform)
                
                seed = random.randint(0, 2**32-1)

                for t in range(len(cropped_unlabeled)):

                    current_pixel_transform = current_pixel_transform.set_random_state(seed=seed)
                    current_spatial_transform = current_spatial_transform.set_random_state(seed=seed)

                    cropped_unlabeled[t], cropped_seg[t] = self.augment(image=cropped_unlabeled[t], mask=cropped_seg[t], pixel_transform=current_pixel_transform, spatial_transform=current_spatial_transform)

            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(3, len(cropped_unlabeled))
            #for t in range(len(cropped_unlabeled)):
            #    ax[0, t].imshow(temp_before_augment[t, 0].cpu(), cmap='gray')
            #    ax[1, t].imshow(cropped_unlabeled[t, 0].cpu(), cmap='gray')
            #    ax[2, t].imshow(cropped_seg[t, 0].cpu(), cmap='gray')
            #plt.show()
                    
            cropped_unlabeled = NormalizeIntensity()(cropped_unlabeled)

            unlabeled_list.append(cropped_unlabeled)
            target_list.append(cropped_seg)
            target_mask_list.append(target_mask)

        unlabeled = torch.stack(unlabeled_list, dim=1) # T, B, 1, H, W
        target = torch.stack(target_list, dim=1) # T, B, 1, H, W
        target_mask = torch.stack(target_mask_list, dim=1) # T, B
        padding_need = torch.stack(padding_need_list, dim=0) # B, 4

        #print(seg.shape)
        #print(np.unique(seg))
        #matplotlib.use('QtAgg')
        #fig, ax = plt.subplots(1, 2)
        #ax[0].imshow(labeled[3, 0], cmap='gray')
        #ax[1].imshow(seg[3, 0], cmap='gray')
        #plt.show()

        return {'unlabeled':unlabeled, 
                'target': target,
                'in_between': None,
                'database': database_list,
                'padding_need': padding_need,
                'properties': case_properties,
                'target_mask': target_mask}






class PreTrainedDataloader(SlimDataLoaderBase):
    def __init__(self, data, patch_size, final_patch_size, batch_size, unlabeled_dataset, video_length, processor, crop_size, is_val, do_data_aug, oversample_foreground_percent=0.0,
                 memmap_mode="r", pseudo_3d_slices=1, pad_mode="edge",
                 pad_kwargs_data=None, pad_sides=None):
        """
        This is the basic data loader for 2D networks. It uses preprocessed data as produced by my (Fabian) preprocessing.
        You can load the data with load_dataset(folder) where folder is the folder where the npz files are located. If there
        are only npz files present in that folder, the data loader will unpack them on the fly. This may take a while
        and increase CPU usage. Therefore, I advise you to call unpack_dataset(folder) first, which will unpack all npz
        to npy. Don't forget to call delete_npy(folder) after you are done with training?
        Why all the hassle? Well the decathlon dataset is huge. Using npy for everything will consume >1 TB and that is uncool
        given that I (Fabian) will have to store that permanently on /datasets and my local computer. With this strategy all
        data is stored in a compressed format (factor 10 smaller) and only unpacked when needed.
        :param data: get this with load_dataset(folder, stage=0). Plug the return value in here and you are g2g (good to go)
        :param patch_size: what patch size will this data loader return? it is common practice to first load larger
        patches so that a central crop after data augmentation can be done to reduce border artifacts. If unsure, use
        get_patch_size() from data_augmentation.default_data_augmentation
        :param final_patch_size: what will the patch finally be cropped to (after data augmentation)? this is the patch
        size that goes into your network. We need this here because we will pad patients in here so that patches at the
        border of patients are sampled properly
        :param batch_size:
        :param num_batches: how many batches will the data loader produce before stopping? None=endless
        :param seed:
        :param stage: ignore this (Fabian only)
        :param transpose: ignore this
        :param random: sample randomly; CAREFUL! non-random sampling requires batch_size=1, otherwise you will iterate batch_size times over the dataset
        :param pseudo_3d_slices: 7 = 3 below and 3 above the center slice
        """
        super(PreTrainedDataloader, self).__init__(data, batch_size, None)
        if pad_kwargs_data is None:
            pad_kwargs_data = OrderedDict()
        self.pad_kwargs_data = pad_kwargs_data
        self.pad_mode = pad_mode
        self.is_val = is_val
        self.crop_size = crop_size
        self.pseudo_3d_slices = pseudo_3d_slices
        self.oversample_foreground_percent = oversample_foreground_percent
        self.final_patch_size = final_patch_size
        self.patch_size = patch_size
        self.list_of_keys = list(self._data.keys())
        self.un_list_of_keys = list(unlabeled_dataset.keys())
        self.need_to_pad = np.array(patch_size) - np.array(final_patch_size)
        self.memmap_mode = memmap_mode
        self.video_length = video_length
        self.percent = None
        if pad_sides is not None:
            if not isinstance(pad_sides, np.ndarray):
                pad_sides = np.array(pad_sides)
            self.need_to_pad += pad_sides
        self.pad_sides = pad_sides
        self.data_shape, self.seg_shape = self.determine_shapes()
        self.do_data_aug = do_data_aug
        self.un_data = unlabeled_dataset

        self.preprocessing_transform = self.set_up_preprocessing_pipeline()
        self.pixel_transform, self.spatial_transform = self.set_up_augmentation_pipeline()

        self.processor = processor

        with open('ACDC_indices.pkl', 'rb') as f:
            self.depth_indices = pickle.load(f)
        

    def determine_shapes(self):
        num_seg = 1

        k = list(self._data.keys())[0]
        if isfile(self._data[k]['data_file'][:-4] + ".npy"):
            case_all_data = np.load(self._data[k]['data_file'][:-4] + ".npy", self.memmap_mode)
        else:
            case_all_data = np.load(self._data[k]['data_file'])['data']
        num_color_channels = case_all_data.shape[0] - num_seg
        data_shape = (self.batch_size, num_color_channels, *self.patch_size)
        seg_shape = (self.batch_size, num_seg, *self.patch_size)
        return data_shape, seg_shape

    def get_do_oversample(self, batch_idx):
        return not batch_idx < round(self.batch_size * (1 - self.oversample_foreground_percent))
    
    def select_deep_supervision(self, x):
        if self.deep_supervision:
            return x[0]
        else:
            return x
        

    def my_crop(self, labeled, seg, unlabeled):

        xm = labeled.shape[-1] / 2
        ym = labeled.shape[-2] / 2

        x1 = int(round(xm - self.final_patch_size[1] / 2))
        x2 = int(round(xm + self.final_patch_size[1] / 2))
        y1 = int(round(ym - self.final_patch_size[0] / 2))
        y2 = int(round(ym + self.final_patch_size[0] / 2))

        labeled = labeled[:, :, y1:y2, x1:x2]
        seg = seg[:, :, y1:y2, x1:x2]

        xm = unlabeled.shape[-1] / 2
        ym = unlabeled.shape[-2] / 2

        x1 = int(round(xm - self.final_patch_size[1] / 2))
        x2 = int(round(xm + self.final_patch_size[1] / 2))
        y1 = int(round(ym - self.final_patch_size[0] / 2))
        y2 = int(round(ym + self.final_patch_size[0] / 2))

        unlabeled = unlabeled[:, :, :, y1:y2, x1:x2]

        return labeled, seg, unlabeled
    
    def preprocess(self, image, mask):
        if mask is not None:
            data = {'image': image, 'mask': mask}
            transformed = self.preprocessing_transform(data)
            image = transformed['image']
            mask = transformed['mask']
        else:
            data = {'image': image}
            transformed = self.preprocessing_transform(data)
            image = transformed['image']

        return image, mask
    
    
    def augment(self, image, mask, pixel_transform, spatial_transform):
        
        temp = torch.clone(image)
        padding_mask = image == 0

        if mask is not None:
            data = {'image': image, 'mask': mask}
            pixel_transformed = pixel_transform(data)
            image = pixel_transformed['image']
            mask = pixel_transformed['mask']
            image[padding_mask] = 0
            data = {'image': image, 'mask': mask}
            spatial_transformed = spatial_transform(data)
            image = spatial_transformed['image']
            mask = spatial_transformed['mask']

            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(1, 3)
            #ax[0].imshow(temp[0].cpu(), cmap='gray')
            #ax[1].imshow(image[0].cpu(), cmap='gray')
            #ax[2].imshow(mask[0].cpu(), cmap='gray')
            #plt.show()
        else:
            data = {'image': image}
            pixel_transformed = pixel_transform(data)
            image = pixel_transformed['image']
            image[padding_mask] = 0
            data = {'image': image}
            spatial_transformed = spatial_transform(data)
            image = spatial_transformed['image']

            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(1, 2)
            #ax[0].imshow(temp[0].cpu(), cmap='gray')
            #ax[1].imshow(image[0].cpu(), cmap='gray')
            #plt.show()

        return image, mask
    
    def set_up_preprocessing_pipeline(self):
        preprocess = Compose([
                            T.SpatialPadd(keys=['image', 'mask'], spatial_size=self.final_patch_size, allow_missing_keys=True),
                            T.CenterSpatialCropd(keys=['image', 'mask'], roi_size=self.final_patch_size, allow_missing_keys=True),
                            T.ToTensor(dtype=torch.float32, device='cuda:0')
                            ])

        return preprocess
    
    def set_up_augmentation_pipeline(self):
        spatial_transform = [
                                    T.RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=0, allow_missing_keys=True), 
                                    T.RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=1, allow_missing_keys=True),
                                    T.RandRotated(keys=['image', 'mask'], prob=0.5, range_x=np.pi, mode=["bilinear", "nearest"], padding_mode="zeros", allow_missing_keys=True),
                                    T.RandZoomd(keys=['image', 'mask'], prob=0.5, min_zoom=0.5, max_zoom=1.5, mode=["bilinear", "nearest"], padding_mode="constant", allow_missing_keys=True),
                                    T.RandAffined(keys=['image', 'mask'], prob=0.5, translate_range=(-26, 26), mode=["bilinear", "nearest"], padding_mode="zeros", allow_missing_keys=True)
                            ]

        pixel_transform = [
                                        T.RandRicianNoised(keys=['image'], prob=0.5, std=0.075),
                                        T.RandGibbsNoised(keys=['image'], prob=0.5, alpha=[0.45, 0.75]),
                                        T.RandAdjustContrastd(keys=['image'], prob=0.5, gamma=(0.7, 1.5), allow_missing_keys=True),
                                        T.RandGaussianNoised(keys=['image'], prob=0.5, std=0.04, allow_missing_keys=True),
                                        T.RandGaussianSharpend(keys=['image'], prob=0.5, sigma1_x=(0.1, 0.2), sigma1_y=(0.1, 0.2), sigma2_x=(0.2, 0.4), sigma2_y=(0.2, 0.4), alpha=(2.0, 3.0), allow_missing_keys=True)
                                ]

        return pixel_transform, spatial_transform
    

    def generate_train_batch(self):
        #eval_idx = None
        selected_keys = np.random.choice(self.list_of_keys, self.batch_size, True, None)
        list_of_frames = []
        for i in range(len(selected_keys)):
            patient_id = selected_keys[i][:10]
            l_filtered = [x for x in self.list_of_keys if x[:10] in patient_id]
            un_filtered = [x for x in self.un_list_of_keys if x[:10] in patient_id]
            filtered = l_filtered + un_filtered
            list_of_frames.append(filtered)
        
        target_list = []
        unlabeled_list = []
        padding_need_list = []
        target_mask_list = []
        database_list = []

        case_properties = []
        for j, frames in enumerate(list_of_frames):
            labeled_frame = frames[0]
            if 'properties' in self._data[labeled_frame].keys():
                properties = self._data[labeled_frame]['properties']
            else:
                properties = load_pickle(self._data[labeled_frame]['properties_file'])
            case_properties.append(properties)

            patient_name = frames[0].split('_')[0]

            depth = properties['size_after_resampling'][0]
            depth_array = np.arange(depth)
            mask = np.isin(depth_array, self.depth_indices[patient_name])
            depth_array = depth_array[~mask]
            depth_idx = np.random.choice(depth_array)

            frames = sorted(frames, key=lambda x: int(x[16:18]))
            frames = np.array(frames)
            global_labeled_idx = np.where(~np.char.endswith(frames, '_u'))[0]
            ed_idx = global_labeled_idx[0]
            es_idx = global_labeled_idx[1]
            assert len(global_labeled_idx) == 2

            unlabeled = torch.full(size=(self.video_length, 1, *self.final_patch_size), fill_value=torch.nan, device='cuda:0')
            seg = torch.full(size=(self.video_length, 1, *self.final_patch_size), fill_value=torch.nan, device='cuda:0')
            cropped_unlabeled = torch.full(size=(self.video_length, 1, self.crop_size, self.crop_size), fill_value=torch.nan, device='cuda:0')
            cropped_seg = torch.full(size=(self.video_length, 1, self.crop_size, self.crop_size), fill_value=torch.nan, device='cuda:0')

            possible_indices = np.arange(len(frames))

            before_where = np.argwhere(possible_indices < ed_idx).reshape(-1,)
            after_where = np.argwhere(possible_indices >= ed_idx).reshape(-1,)

            before_indices = possible_indices[before_where]
            after_indices = possible_indices[after_where]

            possible_indices = np.concatenate([after_indices, before_indices])

            stop_idx = np.argwhere(possible_indices == es_idx)[0][0]
            chunk1 = possible_indices[:stop_idx + 1]
            chunk2 = possible_indices[stop_idx:]
            chunk2 = np.concatenate([np.array([possible_indices[0]]), chunk2[::-1]])

            r = np.random.randint(0, 2)
            if r == 0:
                possible_indices = chunk1
            elif r == 1:
                possible_indices = chunk2
            
            assert possible_indices[0] == ed_idx
            assert possible_indices[-1] == es_idx

            random_indices = np.random.choice(np.arange(0, len(possible_indices)), size=self.video_length - 2)
            target_mask = np.concatenate([np.array([True]), np.full_like(random_indices, fill_value=False), np.array([True])]).astype(bool)
            random_indices = np.concatenate([np.array([0]), random_indices, np.array([len(possible_indices) - 1])])
            sorted_indices = np.argsort(random_indices)

            random_indices = random_indices[sorted_indices]
            target_mask = target_mask[sorted_indices]
            target_mask = torch.from_numpy(target_mask).to('cuda:0')

            frame_indices = possible_indices[random_indices]

            assert len(frame_indices) == self.video_length
            video = frames[frame_indices]

            labeled_idx = np.where(~np.char.endswith(video, '_u'))[0]

            for frame_idx, t in enumerate(video):

                if '_u' in t:
                    if not isfile(self.un_data[t]['data_file'][:-4] + ".npy"):
                        # lets hope you know what you're doing
                        case_all_data = np.load(self.un_data[t]['data_file'][:-4] + ".npz")['data']
                    else:
                        case_all_data = np.load(self.un_data[t]['data_file'][:-4] + ".npy", self.memmap_mode)
                else:
                    if not isfile(self._data[t]['data_file'][:-4] + ".npy"):
                        # lets hope you know what you're doing
                        case_all_data = np.load(self._data[t]['data_file'][:-4] + ".npz")['data']
                    else:
                        case_all_data = np.load(self._data[t]['data_file'][:-4] + ".npy", self.memmap_mode)

                # this is for when there is just a 2d slice in case_all_data (2d support)
                if len(case_all_data.shape) == 3:
                    case_all_data = case_all_data[:, None]
                
                assert case_all_data.shape[1] == depth

                # now crop case_all_data to contain just the slice of interest. If we want additional slice above and
                # below the current slice, here is where we get them. We stack those as additional color channels
                slice_data = case_all_data[:, depth_idx]

                # case all data should now be (c, x, y)
                assert len(slice_data.shape) == 3

                if '_u' not in t:
                    image = slice_data[:-1].copy()
                    mask = slice_data[-1:].copy()
                    mask[mask < 0] = 0

                    image, mask = self.preprocess(image, mask)

                    if target_mask[frame_idx] == True:
                        seg[frame_idx] = mask
                        assert frame_idx in labeled_idx
                else:
                    image = slice_data.copy()

                    image, _ = self.preprocess(image, None)
                
                unlabeled[frame_idx] = image

            with torch.no_grad():
                mean_centroid, _ = self.processor.preprocess_no_registration(data=torch.clone(unlabeled)) # T, C(1), H, W

                cropped_unlabeled, padding_need = self.processor.crop_and_pad(data=unlabeled, mean_centroid=mean_centroid)
                padding_need_list.append(padding_need)

                cropped_seg, _ = self.processor.crop_and_pad(data=seg, mean_centroid=mean_centroid)

            cropped_unlabeled = (cropped_unlabeled - cropped_unlabeled.min()) / (cropped_unlabeled.max() - cropped_unlabeled.min() + 1e-8)

            assert torch.count_nonzero(torch.any(~torch.isnan(cropped_seg.reshape(self.video_length, -1)), dim=-1)) == 2
            assert target_mask[0] == True
            assert target_mask[-1] == True
            assert torch.count_nonzero(target_mask) == 2

            if not self.is_val and self.do_data_aug:
                temp_before_augment = torch.clone(cropped_unlabeled).to('cpu')

                current_pixel_transform = np.random.choice(self.pixel_transform)
                current_spatial_transform = np.random.choice(self.spatial_transform)

                #print(current_pixel_transform)
                
                seed = random.randint(0, 2**32-1)

                for t in range(len(cropped_unlabeled)):

                    current_pixel_transform = current_pixel_transform.set_random_state(seed=seed)
                    current_spatial_transform = current_spatial_transform.set_random_state(seed=seed)

                    cropped_unlabeled[t], cropped_seg[t] = self.augment(image=cropped_unlabeled[t], mask=cropped_seg[t], pixel_transform=current_pixel_transform, spatial_transform=current_spatial_transform)

            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(3, len(cropped_unlabeled))
            #for t in range(len(cropped_unlabeled)):
            #    ax[0, t].imshow(temp_before_augment[t, 0].cpu(), cmap='gray')
            #    ax[1, t].imshow(cropped_unlabeled[t, 0].cpu(), cmap='gray')
            #    ax[2, t].imshow(cropped_seg[t, 0].cpu(), cmap='gray')
            #plt.show()
                    
            cropped_unlabeled = NormalizeIntensity()(cropped_unlabeled)
            assert torch.all(torch.isfinite(cropped_unlabeled))
            assert -1 not in cropped_seg

            unlabeled_list.append(cropped_unlabeled)
            target_list.append(cropped_seg)
            target_mask_list.append(target_mask)

        unlabeled = torch.stack(unlabeled_list, dim=1) # T, B, 1, H, W
        target = torch.stack(target_list, dim=1) # T, B, 1, H, W
        target_mask = torch.stack(target_mask_list, dim=1) # T, B
        padding_need = torch.stack(padding_need_list, dim=0) # B, 4

        #print(seg.shape)
        #print(np.unique(seg))
        #matplotlib.use('QtAgg')
        #fig, ax = plt.subplots(1, 2)
        #ax[0].imshow(labeled[3, 0], cmap='gray')
        #ax[1].imshow(seg[3, 0], cmap='gray')
        #plt.show()

        return {'unlabeled':unlabeled, 
                'target': target,
                'name': selected_keys[0],
                'depth_idx': depth_idx,
                'database': database_list,
                'padding_need': padding_need,
                'properties': case_properties,
                'target_mask': target_mask}






class DataLoaderAugmentValidation(SlimDataLoaderBase):
    def __init__(self, data, patch_size, final_patch_size, batch_size, video_length, processor, crop_size, is_val, do_data_aug, oversample_foreground_percent=0.0,
                 memmap_mode="r", pseudo_3d_slices=1, pad_mode="edge",
                 pad_kwargs_data=None, pad_sides=None):
        """
        This is the basic data loader for 2D networks. It uses preprocessed data as produced by my (Fabian) preprocessing.
        You can load the data with load_dataset(folder) where folder is the folder where the npz files are located. If there
        are only npz files present in that folder, the data loader will unpack them on the fly. This may take a while
        and increase CPU usage. Therefore, I advise you to call unpack_dataset(folder) first, which will unpack all npz
        to npy. Don't forget to call delete_npy(folder) after you are done with training?
        Why all the hassle? Well the decathlon dataset is huge. Using npy for everything will consume >1 TB and that is uncool
        given that I (Fabian) will have to store that permanently on /datasets and my local computer. With this strategy all
        data is stored in a compressed format (factor 10 smaller) and only unpacked when needed.
        :param data: get this with load_dataset(folder, stage=0). Plug the return value in here and you are g2g (good to go)
        :param patch_size: what patch size will this data loader return? it is common practice to first load larger
        patches so that a central crop after data augmentation can be done to reduce border artifacts. If unsure, use
        get_patch_size() from data_augmentation.default_data_augmentation
        :param final_patch_size: what will the patch finally be cropped to (after data augmentation)? this is the patch
        size that goes into your network. We need this here because we will pad patients in here so that patches at the
        border of patients are sampled properly
        :param batch_size:
        :param num_batches: how many batches will the data loader produce before stopping? None=endless
        :param seed:
        :param stage: ignore this (Fabian only)
        :param transpose: ignore this
        :param random: sample randomly; CAREFUL! non-random sampling requires batch_size=1, otherwise you will iterate batch_size times over the dataset
        :param pseudo_3d_slices: 7 = 3 below and 3 above the center slice
        """
        super(DataLoaderAugmentValidation, self).__init__(data, batch_size, None)
        if pad_kwargs_data is None:
            pad_kwargs_data = OrderedDict()
        self.pad_kwargs_data = pad_kwargs_data
        self.pad_mode = pad_mode
        self.is_val = is_val
        self.crop_size = crop_size
        self.pseudo_3d_slices = pseudo_3d_slices
        self.oversample_foreground_percent = oversample_foreground_percent
        self.final_patch_size = final_patch_size
        self.patch_size = patch_size
        self.list_of_keys = list(self._data.keys())
        self.need_to_pad = np.array(patch_size) - np.array(final_patch_size)
        self.memmap_mode = memmap_mode
        self.video_length = video_length
        self.percent = None
        if pad_sides is not None:
            if not isinstance(pad_sides, np.ndarray):
                pad_sides = np.array(pad_sides)
            self.need_to_pad += pad_sides
        self.pad_sides = pad_sides
        self.data_shape, self.seg_shape = self.determine_shapes()
        self.do_data_aug = do_data_aug

        self.preprocessing_transform = self.set_up_preprocessing_pipeline()
        self.pixel_transform, self.spatial_transform = self.set_up_augmentation_pipeline()

        self.processor = processor
        

    def determine_shapes(self):
        num_seg = 1

        k = list(self._data.keys())[0]
        if isfile(self._data[k]['data_file'][:-4] + ".npy"):
            case_all_data = np.load(self._data[k]['data_file'][:-4] + ".npy", self.memmap_mode)
        else:
            case_all_data = np.load(self._data[k]['data_file'])['data']
        num_color_channels = case_all_data.shape[0] - num_seg
        data_shape = (self.batch_size, num_color_channels, *self.patch_size)
        seg_shape = (self.batch_size, num_seg, *self.patch_size)
        return data_shape, seg_shape

    def get_do_oversample(self, batch_idx):
        return not batch_idx < round(self.batch_size * (1 - self.oversample_foreground_percent))
    
    def select_deep_supervision(self, x):
        if self.deep_supervision:
            return x[0]
        else:
            return x
        

    def my_crop(self, labeled, seg, unlabeled):

        xm = labeled.shape[-1] / 2
        ym = labeled.shape[-2] / 2

        x1 = int(round(xm - self.final_patch_size[1] / 2))
        x2 = int(round(xm + self.final_patch_size[1] / 2))
        y1 = int(round(ym - self.final_patch_size[0] / 2))
        y2 = int(round(ym + self.final_patch_size[0] / 2))

        labeled = labeled[:, :, y1:y2, x1:x2]
        seg = seg[:, :, y1:y2, x1:x2]

        xm = unlabeled.shape[-1] / 2
        ym = unlabeled.shape[-2] / 2

        x1 = int(round(xm - self.final_patch_size[1] / 2))
        x2 = int(round(xm + self.final_patch_size[1] / 2))
        y1 = int(round(ym - self.final_patch_size[0] / 2))
        y2 = int(round(ym + self.final_patch_size[0] / 2))

        unlabeled = unlabeled[:, :, :, y1:y2, x1:x2]

        return labeled, seg, unlabeled
    
    def preprocess(self, image, mask):
        if mask is not None:
            data = {'image': image, 'mask': mask}
            transformed = self.preprocessing_transform(data)
            image = transformed['image']
            mask = transformed['mask']
        else:
            data = {'image': image}
            transformed = self.preprocessing_transform(data)
            image = transformed['image']

        return image, mask
    
    
    def augment(self, image, mask, pixel_transform, spatial_transform):
        
        temp = torch.clone(image)
        padding_mask = image == 0

        if mask is not None:
            data = {'image': image, 'mask': mask}
            pixel_transformed = pixel_transform(data)
            image = pixel_transformed['image']
            mask = pixel_transformed['mask']
            image[padding_mask] = 0
            data = {'image': image, 'mask': mask}
            spatial_transformed = spatial_transform(data)
            image = spatial_transformed['image']
            mask = spatial_transformed['mask']

            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(1, 3)
            #ax[0].imshow(temp[0].cpu(), cmap='gray')
            #ax[1].imshow(image[0].cpu(), cmap='gray')
            #ax[2].imshow(mask[0].cpu(), cmap='gray')
            #plt.show()
        else:
            data = {'image': image}
            pixel_transformed = pixel_transform(data)
            image = pixel_transformed['image']
            image[padding_mask] = 0
            data = {'image': image}
            spatial_transformed = spatial_transform(data)
            image = spatial_transformed['image']

            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(1, 2)
            #ax[0].imshow(temp[0].cpu(), cmap='gray')
            #ax[1].imshow(image[0].cpu(), cmap='gray')
            #plt.show()

        return image, mask
    
    def set_up_preprocessing_pipeline(self):
        preprocess = Compose([
                            T.SpatialPadd(keys=['image', 'mask'], spatial_size=self.final_patch_size, allow_missing_keys=True),
                            T.CenterSpatialCropd(keys=['image', 'mask'], roi_size=self.final_patch_size, allow_missing_keys=True),
                            T.ToTensor(dtype=torch.float32, device='cuda:0')
                            ])

        return preprocess
    
    def set_up_augmentation_pipeline(self):
        spatial_transform = [
                                    T.RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=0, allow_missing_keys=True), 
                                    T.RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=1, allow_missing_keys=True),
                                    T.RandRotated(keys=['image', 'mask'], prob=0.5, range_x=np.pi, mode=["bilinear", "nearest"], padding_mode="zeros", allow_missing_keys=True),
                                    T.RandZoomd(keys=['image', 'mask'], prob=0.5, min_zoom=0.5, max_zoom=1.5, mode=["bilinear", "nearest"], padding_mode="constant", allow_missing_keys=True),
                                    T.RandAffined(keys=['image', 'mask'], prob=0.5, translate_range=(-26, 26), mode=["bilinear", "nearest"], padding_mode="zeros", allow_missing_keys=True)
                            ]

        pixel_transform = [
                                        T.RandRicianNoised(keys=['image'], prob=0.5, std=0.075),
                                        T.RandGibbsNoised(keys=['image'], prob=0.5, alpha=[0.45, 0.75]),
                                        T.RandAdjustContrastd(keys=['image'], prob=0.5, gamma=(0.7, 1.5), allow_missing_keys=True),
                                        T.RandGaussianNoised(keys=['image'], prob=0.5, std=0.04, allow_missing_keys=True),
                                        T.RandGaussianSharpend(keys=['image'], prob=0.5, sigma1_x=(0.1, 0.2), sigma1_y=(0.1, 0.2), sigma2_x=(0.2, 0.4), sigma2_y=(0.2, 0.4), alpha=(2.0, 3.0), allow_missing_keys=True)
                                ]

        return pixel_transform, spatial_transform
    

    def generate_train_batch(self):
        #eval_idx = None
        selected_keys = np.random.choice(self.list_of_keys, self.batch_size, True, None)
        list_of_frames = []
        for i in range(len(selected_keys)):
            patient_id = selected_keys[i][:10]
            filtered = [x for x in self.list_of_keys if x[:10] in patient_id]
            list_of_frames.append(filtered)
        
        target_list = []
        unlabeled_list = []
        padding_need_list = []
        target_mask_list = []
        database_list = []

        case_properties = []
        for j, frames in enumerate(list_of_frames):
            labeled_frame = frames[0]
            if 'properties' in self._data[labeled_frame].keys():
                properties = self._data[labeled_frame]['properties']
            else:
                properties = load_pickle(self._data[labeled_frame]['properties_file'])
            case_properties.append(properties)

            database_list.append(properties['Database'])

            #print(properties['ed_number'])
            #print(np.rint(np.array(properties['ed_number'])))
            #print('******************')

            ed_idx = np.rint(np.array(properties['ed_number'])).astype(np.uint8) % len(frames)
            es_idx = np.rint(np.array(properties['es_number'])).astype(np.uint8) % len(frames)

            depth = properties['size_after_resampling'][0]
            depth_idx = np.random.choice(depth)

            global_labeled_idx = np.array([ed_idx, es_idx])

            frames = sorted(frames, key=lambda x: int(x[16:18]))
            frames = np.array(frames)

            possible_indices = np.arange(len(frames))

            before_where = np.argwhere(possible_indices < ed_idx).reshape(-1,)
            after_where = np.argwhere(possible_indices >= ed_idx).reshape(-1,)

            before_indices = possible_indices[before_where]
            after_indices = possible_indices[after_where]

            possible_indices = np.concatenate([after_indices, before_indices])

            stop_idx = np.argwhere(possible_indices == es_idx)[0][0]
            chunk1 = possible_indices[:stop_idx + 1]
            chunk2 = possible_indices[stop_idx:]
            chunk2 = np.concatenate([np.array([possible_indices[0]]), chunk2[::-1]])

            r = np.random.randint(0, 2)
            if r == 0:
                possible_indices = chunk1
            elif r == 1:
                possible_indices = chunk2
            
            assert possible_indices[0] == ed_idx
            assert possible_indices[-1] == es_idx

            target_mask = np.full_like(possible_indices, fill_value=False).astype(bool)
            target_mask[0] = True
            target_mask[-1] = True
            target_mask = torch.from_numpy(target_mask).to('cuda:0')

            video = frames[possible_indices]

            labeled_idx = np.where(np.isin(possible_indices, global_labeled_idx))[0]

            unlabeled = torch.full(size=(len(possible_indices), 1, *self.final_patch_size), fill_value=torch.nan, device='cuda:0')
            seg = torch.full(size=(len(possible_indices), 1, *self.final_patch_size), fill_value=torch.nan, device='cuda:0')
            cropped_unlabeled = torch.full(size=(len(possible_indices), 1, self.crop_size, self.crop_size), fill_value=torch.nan, device='cuda:0')
            cropped_seg = torch.full(size=(len(possible_indices), 1, self.crop_size, self.crop_size), fill_value=torch.nan, device='cuda:0')

            for frame_idx, t in enumerate(video):

                if '_u' in t:
                    if not isfile(self.un_data[t]['data_file'][:-4] + ".npy"):
                        # lets hope you know what you're doing
                        case_all_data = np.load(self.un_data[t]['data_file'][:-4] + ".npz")['data']
                    else:
                        case_all_data = np.load(self.un_data[t]['data_file'][:-4] + ".npy", self.memmap_mode)
                else:
                    if not isfile(self._data[t]['data_file'][:-4] + ".npy"):
                        # lets hope you know what you're doing
                        case_all_data = np.load(self._data[t]['data_file'][:-4] + ".npz")['data']
                    else:
                        case_all_data = np.load(self._data[t]['data_file'][:-4] + ".npy", self.memmap_mode)

                # this is for when there is just a 2d slice in case_all_data (2d support)
                if len(case_all_data.shape) == 3:
                    case_all_data = case_all_data[:, None]
                
                assert case_all_data.shape[1] == depth

                # now crop case_all_data to contain just the slice of interest. If we want additional slice above and
                # below the current slice, here is where we get them. We stack those as additional color channels
                slice_data = case_all_data[:, depth_idx]

                # case all data should now be (c, x, y)
                assert len(slice_data.shape) == 3

                if '_u' not in t:
                    image = slice_data[:-1].copy()
                    mask = slice_data[-1:].copy()
                    mask[mask < 0] = 0

                    image, mask = self.preprocess(image, mask)

                    if target_mask[frame_idx] == True:
                        seg[frame_idx] = mask
                        assert frame_idx in labeled_idx
                else:
                    image = slice_data.copy()

                    image, _ = self.preprocess(image, None)
                
                unlabeled[frame_idx] = image

            with torch.no_grad():
                mean_centroid, _ = self.processor.preprocess_no_registration(data=torch.clone(unlabeled)) # T, C(1), H, W

                cropped_unlabeled, padding_need = self.processor.crop_and_pad(data=unlabeled, mean_centroid=mean_centroid)
                padding_need_list.append(padding_need)

                cropped_seg, _ = self.processor.crop_and_pad(data=seg, mean_centroid=mean_centroid)

            cropped_unlabeled = (cropped_unlabeled - cropped_unlabeled.min()) / (cropped_unlabeled.max() - cropped_unlabeled.min() + 1e-8)

            assert torch.count_nonzero(torch.any(~torch.isnan(cropped_seg.reshape(len(video), -1)), dim=-1)) == 2
            assert target_mask[0] == True
            assert target_mask[-1] == True
            assert torch.count_nonzero(target_mask) == 2
                    
            cropped_unlabeled = NormalizeIntensity()(cropped_unlabeled)

            unlabeled_list.append(cropped_unlabeled)
            target_list.append(cropped_seg)
            target_mask_list.append(target_mask)

        unlabeled = torch.stack(unlabeled_list, dim=1) # T, B, 1, H, W
        target = torch.stack(target_list, dim=1) # T, B, 1, H, W
        target_mask = torch.stack(target_mask_list, dim=1) # T, B
        padding_need = torch.stack(padding_need_list, dim=0) # B, 4

        #print(seg.shape)
        #print(np.unique(seg))
        #matplotlib.use('QtAgg')
        #fig, ax = plt.subplots(1, 2)
        #ax[0].imshow(labeled[3, 0], cmap='gray')
        #ax[1].imshow(seg[3, 0], cmap='gray')
        #plt.show()

        return {'unlabeled':unlabeled, 
                'target': target,
                'in_between': None,
                'database': database_list,
                'padding_need': padding_need,
                'properties': case_properties,
                'target_mask': target_mask}
    





class DataLoaderAugmentValidationPreTrained(SlimDataLoaderBase):
    def __init__(self, data, patch_size, final_patch_size, batch_size, unlabeled_dataset, video_length, processor, crop_size, is_val, do_data_aug, oversample_foreground_percent=0.0,
                 memmap_mode="r", pseudo_3d_slices=1, pad_mode="edge",
                 pad_kwargs_data=None, pad_sides=None):
        """
        This is the basic data loader for 2D networks. It uses preprocessed data as produced by my (Fabian) preprocessing.
        You can load the data with load_dataset(folder) where folder is the folder where the npz files are located. If there
        are only npz files present in that folder, the data loader will unpack them on the fly. This may take a while
        and increase CPU usage. Therefore, I advise you to call unpack_dataset(folder) first, which will unpack all npz
        to npy. Don't forget to call delete_npy(folder) after you are done with training?
        Why all the hassle? Well the decathlon dataset is huge. Using npy for everything will consume >1 TB and that is uncool
        given that I (Fabian) will have to store that permanently on /datasets and my local computer. With this strategy all
        data is stored in a compressed format (factor 10 smaller) and only unpacked when needed.
        :param data: get this with load_dataset(folder, stage=0). Plug the return value in here and you are g2g (good to go)
        :param patch_size: what patch size will this data loader return? it is common practice to first load larger
        patches so that a central crop after data augmentation can be done to reduce border artifacts. If unsure, use
        get_patch_size() from data_augmentation.default_data_augmentation
        :param final_patch_size: what will the patch finally be cropped to (after data augmentation)? this is the patch
        size that goes into your network. We need this here because we will pad patients in here so that patches at the
        border of patients are sampled properly
        :param batch_size:
        :param num_batches: how many batches will the data loader produce before stopping? None=endless
        :param seed:
        :param stage: ignore this (Fabian only)
        :param transpose: ignore this
        :param random: sample randomly; CAREFUL! non-random sampling requires batch_size=1, otherwise you will iterate batch_size times over the dataset
        :param pseudo_3d_slices: 7 = 3 below and 3 above the center slice
        """
        super(DataLoaderAugmentValidationPreTrained, self).__init__(data, batch_size, None)
        if pad_kwargs_data is None:
            pad_kwargs_data = OrderedDict()
        self.pad_kwargs_data = pad_kwargs_data
        self.pad_mode = pad_mode
        self.is_val = is_val
        self.crop_size = crop_size
        self.pseudo_3d_slices = pseudo_3d_slices
        self.oversample_foreground_percent = oversample_foreground_percent
        self.final_patch_size = final_patch_size
        self.patch_size = patch_size
        self.list_of_keys = list(self._data.keys())
        self.un_list_of_keys = list(unlabeled_dataset.keys())
        self.need_to_pad = np.array(patch_size) - np.array(final_patch_size)
        self.memmap_mode = memmap_mode
        self.video_length = video_length
        self.percent = None
        if pad_sides is not None:
            if not isinstance(pad_sides, np.ndarray):
                pad_sides = np.array(pad_sides)
            self.need_to_pad += pad_sides
        self.pad_sides = pad_sides
        self.data_shape, self.seg_shape = self.determine_shapes()
        self.do_data_aug = do_data_aug
        self.un_data = unlabeled_dataset

        self.preprocessing_transform = self.set_up_preprocessing_pipeline()
        self.pixel_transform, self.spatial_transform = self.set_up_augmentation_pipeline()

        self.processor = processor
        

    def determine_shapes(self):
        num_seg = 1

        k = list(self._data.keys())[0]
        if isfile(self._data[k]['data_file'][:-4] + ".npy"):
            case_all_data = np.load(self._data[k]['data_file'][:-4] + ".npy", self.memmap_mode)
        else:
            case_all_data = np.load(self._data[k]['data_file'])['data']
        num_color_channels = case_all_data.shape[0] - num_seg
        data_shape = (self.batch_size, num_color_channels, *self.patch_size)
        seg_shape = (self.batch_size, num_seg, *self.patch_size)
        return data_shape, seg_shape

    def get_do_oversample(self, batch_idx):
        return not batch_idx < round(self.batch_size * (1 - self.oversample_foreground_percent))
    
    def select_deep_supervision(self, x):
        if self.deep_supervision:
            return x[0]
        else:
            return x
        

    def my_crop(self, labeled, seg, unlabeled):

        xm = labeled.shape[-1] / 2
        ym = labeled.shape[-2] / 2

        x1 = int(round(xm - self.final_patch_size[1] / 2))
        x2 = int(round(xm + self.final_patch_size[1] / 2))
        y1 = int(round(ym - self.final_patch_size[0] / 2))
        y2 = int(round(ym + self.final_patch_size[0] / 2))

        labeled = labeled[:, :, y1:y2, x1:x2]
        seg = seg[:, :, y1:y2, x1:x2]

        xm = unlabeled.shape[-1] / 2
        ym = unlabeled.shape[-2] / 2

        x1 = int(round(xm - self.final_patch_size[1] / 2))
        x2 = int(round(xm + self.final_patch_size[1] / 2))
        y1 = int(round(ym - self.final_patch_size[0] / 2))
        y2 = int(round(ym + self.final_patch_size[0] / 2))

        unlabeled = unlabeled[:, :, :, y1:y2, x1:x2]

        return labeled, seg, unlabeled
    
    def preprocess(self, image, mask):
        if mask is not None:
            data = {'image': image, 'mask': mask}
            transformed = self.preprocessing_transform(data)
            image = transformed['image']
            mask = transformed['mask']
        else:
            data = {'image': image}
            transformed = self.preprocessing_transform(data)
            image = transformed['image']

        return image, mask
    
    
    def augment(self, image, mask, pixel_transform, spatial_transform):
        
        temp = torch.clone(image)
        padding_mask = image == 0

        if mask is not None:
            data = {'image': image, 'mask': mask}
            pixel_transformed = pixel_transform(data)
            image = pixel_transformed['image']
            mask = pixel_transformed['mask']
            image[padding_mask] = 0
            data = {'image': image, 'mask': mask}
            spatial_transformed = spatial_transform(data)
            image = spatial_transformed['image']
            mask = spatial_transformed['mask']

            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(1, 3)
            #ax[0].imshow(temp[0].cpu(), cmap='gray')
            #ax[1].imshow(image[0].cpu(), cmap='gray')
            #ax[2].imshow(mask[0].cpu(), cmap='gray')
            #plt.show()
        else:
            data = {'image': image}
            pixel_transformed = pixel_transform(data)
            image = pixel_transformed['image']
            image[padding_mask] = 0
            data = {'image': image}
            spatial_transformed = spatial_transform(data)
            image = spatial_transformed['image']

            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(1, 2)
            #ax[0].imshow(temp[0].cpu(), cmap='gray')
            #ax[1].imshow(image[0].cpu(), cmap='gray')
            #plt.show()

        return image, mask
    
    def set_up_preprocessing_pipeline(self):
        preprocess = Compose([
                            T.SpatialPadd(keys=['image', 'mask'], spatial_size=self.final_patch_size, allow_missing_keys=True),
                            T.CenterSpatialCropd(keys=['image', 'mask'], roi_size=self.final_patch_size, allow_missing_keys=True),
                            T.ToTensor(dtype=torch.float32, device='cuda:0')
                            ])

        return preprocess
    
    def set_up_augmentation_pipeline(self):
        spatial_transform = [
                                    T.RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=0, allow_missing_keys=True), 
                                    T.RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=1, allow_missing_keys=True),
                                    T.RandRotated(keys=['image', 'mask'], prob=0.5, range_x=np.pi, mode=["bilinear", "nearest"], padding_mode="zeros", allow_missing_keys=True),
                                    T.RandZoomd(keys=['image', 'mask'], prob=0.5, min_zoom=0.5, max_zoom=1.5, mode=["bilinear", "nearest"], padding_mode="constant", allow_missing_keys=True),
                                    T.RandAffined(keys=['image', 'mask'], prob=0.5, translate_range=(-26, 26), mode=["bilinear", "nearest"], padding_mode="zeros", allow_missing_keys=True)
                            ]

        pixel_transform = [
                                        T.RandRicianNoised(keys=['image'], prob=0.5, std=0.075),
                                        T.RandGibbsNoised(keys=['image'], prob=0.5, alpha=[0.45, 0.75]),
                                        T.RandAdjustContrastd(keys=['image'], prob=0.5, gamma=(0.7, 1.5), allow_missing_keys=True),
                                        T.RandGaussianNoised(keys=['image'], prob=0.5, std=0.04, allow_missing_keys=True),
                                        T.RandGaussianSharpend(keys=['image'], prob=0.5, sigma1_x=(0.1, 0.2), sigma1_y=(0.1, 0.2), sigma2_x=(0.2, 0.4), sigma2_y=(0.2, 0.4), alpha=(2.0, 3.0), allow_missing_keys=True)
                                ]

        return pixel_transform, spatial_transform
    

    def generate_train_batch(self):
        #eval_idx = None
        selected_keys = np.random.choice(self.list_of_keys, self.batch_size, True, None)
        list_of_frames = []
        for i in range(len(selected_keys)):
            patient_id = selected_keys[i][:10]
            l_filtered = [x for x in self.list_of_keys if x[:10] in patient_id]
            un_filtered = [x for x in self.un_list_of_keys if x[:10] in patient_id]
            filtered = l_filtered + un_filtered
            list_of_frames.append(filtered)
        
        target_list = []
        unlabeled_list = []
        padding_need_list = []
        target_mask_list = []
        database_list = []

        case_properties = []
        for j, frames in enumerate(list_of_frames):
            labeled_frame = frames[0]
            if 'properties' in self._data[labeled_frame].keys():
                properties = self._data[labeled_frame]['properties']
            else:
                properties = load_pickle(self._data[labeled_frame]['properties_file'])
            case_properties.append(properties)


            depth = properties['size_after_resampling'][0]
            depth_array = np.arange(depth)
            depth_idx = np.random.choice(depth_array[2:-2])

            frames = sorted(frames, key=lambda x: int(x[16:18]))
            frames = np.array(frames)
            global_labeled_idx = np.where(~np.char.endswith(frames, '_u'))[0]
            ed_idx = global_labeled_idx[0]
            es_idx = global_labeled_idx[1]
            assert len(global_labeled_idx) == 2

            global_labeled_idx = np.array([ed_idx, es_idx])

            possible_indices = np.arange(len(frames))

            before_where = np.argwhere(possible_indices < ed_idx).reshape(-1,)
            after_where = np.argwhere(possible_indices >= ed_idx).reshape(-1,)

            before_indices = possible_indices[before_where]
            after_indices = possible_indices[after_where]

            possible_indices = np.concatenate([after_indices, before_indices])

            stop_idx = np.argwhere(possible_indices == es_idx)[0][0]
            chunk1 = possible_indices[:stop_idx + 1]
            chunk2 = possible_indices[stop_idx:]
            chunk2 = np.concatenate([np.array([possible_indices[0]]), chunk2[::-1]])

            r = np.random.randint(0, 2)
            if r == 0:
                possible_indices = chunk1
            elif r == 1:
                possible_indices = chunk2
            
            assert possible_indices[0] == ed_idx
            assert possible_indices[-1] == es_idx

            target_mask = np.full_like(possible_indices, fill_value=False).astype(bool)
            target_mask[0] = True
            target_mask[-1] = True
            target_mask = torch.from_numpy(target_mask).to('cuda:0')

            video = frames[possible_indices]

            labeled_idx = np.where(np.isin(possible_indices, global_labeled_idx))[0]

            unlabeled = torch.full(size=(len(possible_indices), 1, *self.final_patch_size), fill_value=torch.nan, device='cuda:0')
            seg = torch.full(size=(len(possible_indices), 1, *self.final_patch_size), fill_value=torch.nan, device='cuda:0')
            cropped_unlabeled = torch.full(size=(len(possible_indices), 1, self.crop_size, self.crop_size), fill_value=torch.nan, device='cuda:0')
            cropped_seg = torch.full(size=(len(possible_indices), 1, self.crop_size, self.crop_size), fill_value=torch.nan, device='cuda:0')

            for frame_idx, t in enumerate(video):

                if '_u' in t:
                    if not isfile(self.un_data[t]['data_file'][:-4] + ".npy"):
                        # lets hope you know what you're doing
                        case_all_data = np.load(self.un_data[t]['data_file'][:-4] + ".npz")['data']
                    else:
                        case_all_data = np.load(self.un_data[t]['data_file'][:-4] + ".npy", self.memmap_mode)
                else:
                    if not isfile(self._data[t]['data_file'][:-4] + ".npy"):
                        # lets hope you know what you're doing
                        case_all_data = np.load(self._data[t]['data_file'][:-4] + ".npz")['data']
                    else:
                        case_all_data = np.load(self._data[t]['data_file'][:-4] + ".npy", self.memmap_mode)

                # this is for when there is just a 2d slice in case_all_data (2d support)
                if len(case_all_data.shape) == 3:
                    case_all_data = case_all_data[:, None]
                
                assert case_all_data.shape[1] == depth

                # now crop case_all_data to contain just the slice of interest. If we want additional slice above and
                # below the current slice, here is where we get them. We stack those as additional color channels
                slice_data = case_all_data[:, depth_idx]

                # case all data should now be (c, x, y)
                assert len(slice_data.shape) == 3

                if '_u' not in t:
                    image = slice_data[:-1].copy()
                    mask = slice_data[-1:].copy()
                    mask[mask < 0] = 0

                    image, mask = self.preprocess(image, mask)

                    if target_mask[frame_idx] == True:
                        seg[frame_idx] = mask
                        assert frame_idx in labeled_idx
                else:
                    image = slice_data.copy()

                    image, _ = self.preprocess(image, None)
                
                unlabeled[frame_idx] = image

            with torch.no_grad():
                mean_centroid, _ = self.processor.preprocess_no_registration(data=torch.clone(unlabeled)) # T, C(1), H, W

                cropped_unlabeled, padding_need = self.processor.crop_and_pad(data=unlabeled, mean_centroid=mean_centroid)
                padding_need_list.append(padding_need)

                cropped_seg, _ = self.processor.crop_and_pad(data=seg, mean_centroid=mean_centroid)

            cropped_unlabeled = (cropped_unlabeled - cropped_unlabeled.min()) / (cropped_unlabeled.max() - cropped_unlabeled.min() + 1e-8)

            assert torch.count_nonzero(torch.any(~torch.isnan(cropped_seg.reshape(len(video), -1)), dim=-1)) == 2
            assert target_mask[0] == True
            assert target_mask[-1] == True
            assert torch.count_nonzero(target_mask) == 2
                    
            cropped_unlabeled = NormalizeIntensity()(cropped_unlabeled)

            unlabeled_list.append(cropped_unlabeled)
            target_list.append(cropped_seg)
            target_mask_list.append(target_mask)

        unlabeled = torch.stack(unlabeled_list, dim=1) # T, B, 1, H, W
        target = torch.stack(target_list, dim=1) # T, B, 1, H, W
        target_mask = torch.stack(target_mask_list, dim=1) # T, B
        padding_need = torch.stack(padding_need_list, dim=0) # B, 4

        #print(seg.shape)
        #print(np.unique(seg))
        #matplotlib.use('QtAgg')
        #fig, ax = plt.subplots(1, 2)
        #ax[0].imshow(labeled[3, 0], cmap='gray')
        #ax[1].imshow(seg[3, 0], cmap='gray')
        #plt.show()

        return {'unlabeled':unlabeled, 
                'target': target,
                'in_between': None,
                'database': database_list,
                'padding_need': padding_need,
                'properties': case_properties,
                'target_mask': target_mask}




class DataLoaderAugment2D(SlimDataLoaderBase):
    def __init__(self, data, patch_size, final_patch_size, batch_size, video_length, processor, crop_size, is_val, do_data_aug, oversample_foreground_percent=0.0,
                 memmap_mode="r", pseudo_3d_slices=1, pad_mode="edge",
                 pad_kwargs_data=None, pad_sides=None, filter_phase=False):
        """
        This is the basic data loader for 2D networks. It uses preprocessed data as produced by my (Fabian) preprocessing.
        You can load the data with load_dataset(folder) where folder is the folder where the npz files are located. If there
        are only npz files present in that folder, the data loader will unpack them on the fly. This may take a while
        and increase CPU usage. Therefore, I advise you to call unpack_dataset(folder) first, which will unpack all npz
        to npy. Don't forget to call delete_npy(folder) after you are done with training?
        Why all the hassle? Well the decathlon dataset is huge. Using npy for everything will consume >1 TB and that is uncool
        given that I (Fabian) will have to store that permanently on /datasets and my local computer. With this strategy all
        data is stored in a compressed format (factor 10 smaller) and only unpacked when needed.
        :param data: get this with load_dataset(folder, stage=0). Plug the return value in here and you are g2g (good to go)
        :param patch_size: what patch size will this data loader return? it is common practice to first load larger
        patches so that a central crop after data augmentation can be done to reduce border artifacts. If unsure, use
        get_patch_size() from data_augmentation.default_data_augmentation
        :param final_patch_size: what will the patch finally be cropped to (after data augmentation)? this is the patch
        size that goes into your network. We need this here because we will pad patients in here so that patches at the
        border of patients are sampled properly
        :param batch_size:
        :param num_batches: how many batches will the data loader produce before stopping? None=endless
        :param seed:
        :param stage: ignore this (Fabian only)
        :param transpose: ignore this
        :param random: sample randomly; CAREFUL! non-random sampling requires batch_size=1, otherwise you will iterate batch_size times over the dataset
        :param pseudo_3d_slices: 7 = 3 below and 3 above the center slice
        """
        super(DataLoaderAugment2D, self).__init__(data, batch_size, None)
        if pad_kwargs_data is None:
            pad_kwargs_data = OrderedDict()
        self.pad_kwargs_data = pad_kwargs_data
        self.pad_mode = pad_mode
        self.is_val = is_val
        self.crop_size = crop_size
        self.pseudo_3d_slices = pseudo_3d_slices
        self.oversample_foreground_percent = oversample_foreground_percent
        self.final_patch_size = final_patch_size
        self.patch_size = patch_size
        self.list_of_keys = list(self._data.keys())
        self.need_to_pad = np.array(patch_size) - np.array(final_patch_size)
        self.memmap_mode = memmap_mode
        self.video_length = video_length
        self.percent = None
        if pad_sides is not None:
            if not isinstance(pad_sides, np.ndarray):
                pad_sides = np.array(pad_sides)
            self.need_to_pad += pad_sides
        self.pad_sides = pad_sides
        self.data_shape, self.seg_shape = self.determine_shapes()
        self.do_data_aug = do_data_aug

        self.preprocessing_transform = self.set_up_preprocessing_pipeline()
        self.pixel_transform, self.spatial_transform = self.set_up_augmentation_pipeline()

        self.processor = processor

        if filter_phase:
            self.list_of_keys = self.filter_phase()
    

    def filter_phase(self):
        list_of_keys_filtered = []
        patient_list = []
        for key in self.list_of_keys:
            patient_list.append(key.split('_')[0])
            if 'properties' in self._data[key].keys():
                properties = self._data[key]['properties']
            else:
                properties = load_pickle(self._data[key]['properties_file'])
            ed_idx = np.rint(properties['ed_number']).astype(int)
            es_idx = np.rint(properties['es_number']).astype(int)
            frame_nb = int(key.split('frame')[-1])
            if frame_nb == ed_idx + 1 or frame_nb == es_idx + 1:
                list_of_keys_filtered.append(key)
        
        patient_list = sorted(list(set(patient_list)), key=lambda x:int(x[-3:]))
        
        assert len(list_of_keys_filtered) == len(patient_list) * 2
        return list_of_keys_filtered
        

    def determine_shapes(self):
        num_seg = 1

        k = list(self._data.keys())[0]
        if isfile(self._data[k]['data_file'][:-4] + ".npy"):
            case_all_data = np.load(self._data[k]['data_file'][:-4] + ".npy", self.memmap_mode)
        else:
            case_all_data = np.load(self._data[k]['data_file'])['data']
        num_color_channels = case_all_data.shape[0] - num_seg
        data_shape = (self.batch_size, num_color_channels, *self.patch_size)
        seg_shape = (self.batch_size, num_seg, *self.patch_size)
        return data_shape, seg_shape

    def get_do_oversample(self, batch_idx):
        return not batch_idx < round(self.batch_size * (1 - self.oversample_foreground_percent))
    
    def select_deep_supervision(self, x):
        if self.deep_supervision:
            return x[0]
        else:
            return x
        

    def my_crop(self, labeled, seg, unlabeled):

        xm = labeled.shape[-1] / 2
        ym = labeled.shape[-2] / 2

        x1 = int(round(xm - self.final_patch_size[1] / 2))
        x2 = int(round(xm + self.final_patch_size[1] / 2))
        y1 = int(round(ym - self.final_patch_size[0] / 2))
        y2 = int(round(ym + self.final_patch_size[0] / 2))

        labeled = labeled[:, :, y1:y2, x1:x2]
        seg = seg[:, :, y1:y2, x1:x2]

        xm = unlabeled.shape[-1] / 2
        ym = unlabeled.shape[-2] / 2

        x1 = int(round(xm - self.final_patch_size[1] / 2))
        x2 = int(round(xm + self.final_patch_size[1] / 2))
        y1 = int(round(ym - self.final_patch_size[0] / 2))
        y2 = int(round(ym + self.final_patch_size[0] / 2))

        unlabeled = unlabeled[:, :, :, y1:y2, x1:x2]

        return labeled, seg, unlabeled
    
    def preprocess(self, image, mask):
        if mask is not None:
            data = {'image': image, 'mask': mask}
            transformed = self.preprocessing_transform(data)
            image = transformed['image']
            mask = transformed['mask']
        else:
            data = {'image': image}
            transformed = self.preprocessing_transform(data)
            image = transformed['image']

        return image, mask
    
    
    def augment(self, image, mask, pixel_transform, spatial_transform):
        
        temp = torch.clone(image)
        padding_mask = image == 0

        if mask is not None:
            data = {'image': image, 'mask': mask}
            pixel_transformed = pixel_transform(data)
            image = pixel_transformed['image']
            mask = pixel_transformed['mask']
            image[padding_mask] = 0
            data = {'image': image, 'mask': mask}
            spatial_transformed = spatial_transform(data)
            image = spatial_transformed['image']
            mask = spatial_transformed['mask']

            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(1, 3)
            #ax[0].imshow(temp[0].cpu(), cmap='gray')
            #ax[1].imshow(image[0].cpu(), cmap='gray')
            #ax[2].imshow(mask[0].cpu(), cmap='gray')
            #plt.show()
        else:
            data = {'image': image}
            pixel_transformed = pixel_transform(data)
            image = pixel_transformed['image']
            image[padding_mask] = 0
            data = {'image': image}
            spatial_transformed = spatial_transform(data)
            image = spatial_transformed['image']

            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(1, 2)
            #ax[0].imshow(temp[0].cpu(), cmap='gray')
            #ax[1].imshow(image[0].cpu(), cmap='gray')
            #plt.show()

        return image, mask
    
    def set_up_preprocessing_pipeline(self):
        preprocess = Compose([
                            T.SpatialPadd(keys=['image', 'mask'], spatial_size=self.final_patch_size, allow_missing_keys=True),
                            T.CenterSpatialCropd(keys=['image', 'mask'], roi_size=self.final_patch_size, allow_missing_keys=True),
                            T.ToTensor(dtype=torch.float32, device='cuda:0')
                            ])

        return preprocess
    
    def set_up_augmentation_pipeline(self):
        spatial_transform = [
                                    T.RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=0, allow_missing_keys=True), 
                                    T.RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=1, allow_missing_keys=True),
                                    T.RandRotated(keys=['image', 'mask'], prob=0.5, range_x=np.pi, mode=["bilinear", "nearest"], padding_mode="zeros", allow_missing_keys=True),
                                    T.RandZoomd(keys=['image', 'mask'], prob=0.5, min_zoom=0.5, max_zoom=1.5, mode=["bilinear", "nearest"], padding_mode="constant", allow_missing_keys=True),
                                    T.RandAffined(keys=['image', 'mask'], prob=0.5, translate_range=(-26, 26), mode=["bilinear", "nearest"], padding_mode="zeros", allow_missing_keys=True)
                            ]

        pixel_transform = [
                                        T.RandRicianNoised(keys=['image'], prob=0.5, std=0.075),
                                        T.RandGibbsNoised(keys=['image'], prob=0.5, alpha=[0.45, 0.75]),
                                        T.RandAdjustContrastd(keys=['image'], prob=0.5, gamma=(0.7, 1.5), allow_missing_keys=True),
                                        T.RandGaussianNoised(keys=['image'], prob=0.5, std=0.04, allow_missing_keys=True),
                                        T.RandGaussianSharpend(keys=['image'], prob=0.5, sigma1_x=(0.1, 0.2), sigma1_y=(0.1, 0.2), sigma2_x=(0.2, 0.4), sigma2_y=(0.2, 0.4), alpha=(2.0, 3.0), allow_missing_keys=True)
                                ]

        return pixel_transform, spatial_transform
    

    def generate_train_batch(self):
        #eval_idx = None
        selected_keys = np.random.choice(self.list_of_keys, self.batch_size, True, None)
        list_of_frames = []
        for i in range(len(selected_keys)):
            patient_id = selected_keys[i][:10]
            filtered = [x for x in self.list_of_keys if x[:10] in patient_id]
            list_of_frames.append(filtered)
        
        target_list = []
        unlabeled_list = []
        padding_need_list = []
        target_mask_list = []

        case_properties = []
        for j, frames in enumerate(list_of_frames):
            labeled_frame = frames[0]
            if 'properties' in self._data[labeled_frame].keys():
                properties = self._data[labeled_frame]['properties']
            else:
                properties = load_pickle(self._data[labeled_frame]['properties_file'])
            case_properties.append(properties)

            #print(properties['ed_number'])
            #print(np.rint(np.array(properties['ed_number'])))
            #print('******************')

            ed_idx = np.rint(np.array(properties['ed_number'])).astype(np.uint8) % len(frames)
            es_idx = np.rint(np.array(properties['es_number'])).astype(np.uint8) % len(frames)

            depth = properties['size_after_resampling'][0]
            depth_idx = np.random.choice(depth)

            global_labeled_idx = np.array([ed_idx, es_idx])

            frames = sorted(frames, key=lambda x: int(x[16:18]))
            frames = np.array(frames)

            unlabeled = torch.full(size=(self.video_length, 1, *self.final_patch_size), fill_value=torch.nan, device='cuda:0')
            seg = torch.full(size=(self.video_length, 1, *self.final_patch_size), fill_value=torch.nan, device='cuda:0')
            cropped_unlabeled = torch.full(size=(self.video_length, 1, self.crop_size, self.crop_size), fill_value=torch.nan, device='cuda:0')
            cropped_seg = torch.full(size=(self.video_length, 1, self.crop_size, self.crop_size), fill_value=torch.nan, device='cuda:0')

            possible_indices = np.arange(len(frames))

            frame_indices = np.random.choice(possible_indices)

            video = [frames[frame_indices]]

            for frame_idx, t in enumerate(video):

                if '_u' in t:
                    if not isfile(self.un_data[t]['data_file'][:-4] + ".npy"):
                        # lets hope you know what you're doing
                        case_all_data = np.load(self.un_data[t]['data_file'][:-4] + ".npz")['data']
                    else:
                        case_all_data = np.load(self.un_data[t]['data_file'][:-4] + ".npy", self.memmap_mode)
                else:
                    if not isfile(self._data[t]['data_file'][:-4] + ".npy"):
                        # lets hope you know what you're doing
                        case_all_data = np.load(self._data[t]['data_file'][:-4] + ".npz")['data']
                    else:
                        case_all_data = np.load(self._data[t]['data_file'][:-4] + ".npy", self.memmap_mode)

                # this is for when there is just a 2d slice in case_all_data (2d support)
                if len(case_all_data.shape) == 3:
                    case_all_data = case_all_data[:, None]
                
                assert case_all_data.shape[1] == depth

                # now crop case_all_data to contain just the slice of interest. If we want additional slice above and
                # below the current slice, here is where we get them. We stack those as additional color channels
                slice_data = case_all_data[:, depth_idx]

                # case all data should now be (c, x, y)
                assert len(slice_data.shape) == 3

                if '_u' not in t:
                    image = slice_data[:-1].copy()
                    mask = slice_data[-1:].copy()
                    mask[mask < 0] = 0

                    image, mask = self.preprocess(image, mask)

                    seg[frame_idx] = mask
                else:
                    image = slice_data.copy()

                    image, _ = self.preprocess(image, None)
                
                unlabeled[frame_idx] = image

            with torch.no_grad():
                mean_centroid, _ = self.processor.preprocess_no_registration(data=torch.clone(unlabeled)) # T, C(1), H, W

                cropped_unlabeled, padding_need = self.processor.crop_and_pad(data=unlabeled, mean_centroid=mean_centroid)
                padding_need_list.append(padding_need)

                cropped_seg, _ = self.processor.crop_and_pad(data=seg, mean_centroid=mean_centroid)

            cropped_unlabeled = (cropped_unlabeled - cropped_unlabeled.min()) / (cropped_unlabeled.max() - cropped_unlabeled.min() + 1e-8)

            assert torch.count_nonzero(torch.any(~torch.isnan(cropped_seg.reshape(self.video_length, -1)), dim=-1)) == 1

            if not self.is_val and self.do_data_aug:
                temp_before_augment = torch.clone(cropped_unlabeled).to('cpu')

                current_pixel_transform = np.random.choice(self.pixel_transform)
                current_spatial_transform = np.random.choice(self.spatial_transform)

                #print(current_pixel_transform)
                
                seed = random.randint(0, 2**32-1)

                for t in range(len(cropped_unlabeled)):

                    current_pixel_transform = current_pixel_transform.set_random_state(seed=seed)
                    current_spatial_transform = current_spatial_transform.set_random_state(seed=seed)

                    cropped_unlabeled[t], cropped_seg[t] = self.augment(image=cropped_unlabeled[t], mask=cropped_seg[t], pixel_transform=current_pixel_transform, spatial_transform=current_spatial_transform)

            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(3, len(cropped_unlabeled))
            #for t in range(len(cropped_unlabeled)):
            #    ax[0, t].imshow(temp_before_augment[t, 0].cpu(), cmap='gray')
            #    ax[1, t].imshow(cropped_unlabeled[t, 0].cpu(), cmap='gray')
            #    ax[2, t].imshow(cropped_seg[t, 0].cpu(), cmap='gray')
            #plt.show()
                    
            cropped_unlabeled = NormalizeIntensity()(cropped_unlabeled)

            unlabeled_list.append(cropped_unlabeled)
            target_list.append(cropped_seg)

        unlabeled = torch.stack(unlabeled_list, dim=1)[0] # B, 1, H, W
        target = torch.stack(target_list, dim=1)[0] # B, 1, H, W
        padding_need = torch.stack(padding_need_list, dim=0) # B, 4

        keys = selected_keys

        #print(seg.shape)
        #print(np.unique(seg))
        #matplotlib.use('QtAgg')
        #fig, ax = plt.subplots(1, 2)
        #ax[0].imshow(labeled[3, 0], cmap='gray')
        #ax[1].imshow(seg[3, 0], cmap='gray')
        #plt.show()

        return {'unlabeled':unlabeled, 
                'target': target,
                'padding_need': padding_need,
                'properties': case_properties}
    



class DataLoaderFlowTrain5LibProgressive(SlimDataLoaderBase):
    def __init__(self, data, patch_size, final_patch_size, batch_size, video_length, processor, crop_size, is_val, do_data_aug, oversample_foreground_percent=0.0,
                 memmap_mode="r", pseudo_3d_slices=1, pad_mode="edge",
                 pad_kwargs_data=None, pad_sides=None):
        """
        This is the basic data loader for 2D networks. It uses preprocessed data as produced by my (Fabian) preprocessing.
        You can load the data with load_dataset(folder) where folder is the folder where the npz files are located. If there
        are only npz files present in that folder, the data loader will unpack them on the fly. This may take a while
        and increase CPU usage. Therefore, I advise you to call unpack_dataset(folder) first, which will unpack all npz
        to npy. Don't forget to call delete_npy(folder) after you are done with training?
        Why all the hassle? Well the decathlon dataset is huge. Using npy for everything will consume >1 TB and that is uncool
        given that I (Fabian) will have to store that permanently on /datasets and my local computer. With this strategy all
        data is stored in a compressed format (factor 10 smaller) and only unpacked when needed.
        :param data: get this with load_dataset(folder, stage=0). Plug the return value in here and you are g2g (good to go)
        :param patch_size: what patch size will this data loader return? it is common practice to first load larger
        patches so that a central crop after data augmentation can be done to reduce border artifacts. If unsure, use
        get_patch_size() from data_augmentation.default_data_augmentation
        :param final_patch_size: what will the patch finally be cropped to (after data augmentation)? this is the patch
        size that goes into your network. We need this here because we will pad patients in here so that patches at the
        border of patients are sampled properly
        :param batch_size:
        :param num_batches: how many batches will the data loader produce before stopping? None=endless
        :param seed:
        :param stage: ignore this (Fabian only)
        :param transpose: ignore this
        :param random: sample randomly; CAREFUL! non-random sampling requires batch_size=1, otherwise you will iterate batch_size times over the dataset
        :param pseudo_3d_slices: 7 = 3 below and 3 above the center slice
        """
        super(DataLoaderFlowTrain5LibProgressive, self).__init__(data, batch_size, None)
        if pad_kwargs_data is None:
            pad_kwargs_data = OrderedDict()
        self.pad_kwargs_data = pad_kwargs_data
        self.pad_mode = pad_mode
        self.is_val = is_val
        self.crop_size = crop_size
        self.pseudo_3d_slices = pseudo_3d_slices
        self.oversample_foreground_percent = oversample_foreground_percent
        self.final_patch_size = final_patch_size
        self.patch_size = patch_size
        self.list_of_keys = list(self._data.keys())
        self.need_to_pad = np.array(patch_size) - np.array(final_patch_size)
        self.memmap_mode = memmap_mode
        self.video_length = video_length
        self.percent = None
        if pad_sides is not None:
            if not isinstance(pad_sides, np.ndarray):
                pad_sides = np.array(pad_sides)
            self.need_to_pad += pad_sides
        self.pad_sides = pad_sides
        self.data_shape, self.seg_shape = self.determine_shapes()
        self.do_data_aug = do_data_aug

        self.preprocessing_transform = self.set_up_preprocessing_pipeline()
        self.pixel_transform, self.spatial_transform = self.set_up_augmentation_pipeline()

        self.processor = processor
        

    def determine_shapes(self):
        num_seg = 1

        k = list(self._data.keys())[0]
        if isfile(self._data[k]['data_file'][:-4] + ".npy"):
            case_all_data = np.load(self._data[k]['data_file'][:-4] + ".npy", self.memmap_mode)
        else:
            case_all_data = np.load(self._data[k]['data_file'])['data']
        num_color_channels = case_all_data.shape[0] - num_seg
        data_shape = (self.batch_size, num_color_channels, *self.patch_size)
        seg_shape = (self.batch_size, num_seg, *self.patch_size)
        return data_shape, seg_shape

    def get_do_oversample(self, batch_idx):
        return not batch_idx < round(self.batch_size * (1 - self.oversample_foreground_percent))
    
    def select_deep_supervision(self, x):
        if self.deep_supervision:
            return x[0]
        else:
            return x
        

    def my_crop(self, labeled, seg, unlabeled):

        xm = labeled.shape[-1] / 2
        ym = labeled.shape[-2] / 2

        x1 = int(round(xm - self.final_patch_size[1] / 2))
        x2 = int(round(xm + self.final_patch_size[1] / 2))
        y1 = int(round(ym - self.final_patch_size[0] / 2))
        y2 = int(round(ym + self.final_patch_size[0] / 2))

        labeled = labeled[:, :, y1:y2, x1:x2]
        seg = seg[:, :, y1:y2, x1:x2]

        xm = unlabeled.shape[-1] / 2
        ym = unlabeled.shape[-2] / 2

        x1 = int(round(xm - self.final_patch_size[1] / 2))
        x2 = int(round(xm + self.final_patch_size[1] / 2))
        y1 = int(round(ym - self.final_patch_size[0] / 2))
        y2 = int(round(ym + self.final_patch_size[0] / 2))

        unlabeled = unlabeled[:, :, :, y1:y2, x1:x2]

        return labeled, seg, unlabeled
    
    def preprocess(self, image, mask):
        if mask is not None:
            data = {'image': image, 'mask': mask}
            transformed = self.preprocessing_transform(data)
            image = transformed['image']
            mask = transformed['mask']
        else:
            data = {'image': image}
            transformed = self.preprocessing_transform(data)
            image = transformed['image']

        return image, mask
    
    
    def augment(self, image, mask, seed):
        self.pixel_transform = self.pixel_transform.set_random_state(seed=seed)
        self.spatial_transform = self.spatial_transform.set_random_state(seed=seed)
        
        temp = torch.clone(image)
        padding_mask = image == 0

        if mask is not None:
            data = {'image': image, 'mask': mask}
            pixel_transformed = self.pixel_transform(data)
            image = pixel_transformed['image']
            mask = pixel_transformed['mask']
            image[padding_mask] = 0
            data = {'image': image, 'mask': mask}
            spatial_transformed = self.spatial_transform(data)
            image = spatial_transformed['image']
            mask = spatial_transformed['mask']

            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(1, 3)
            #ax[0].imshow(temp[0].cpu(), cmap='gray')
            #ax[1].imshow(image[0].cpu(), cmap='gray')
            #ax[2].imshow(mask[0].cpu(), cmap='gray')
            #plt.show()
        else:
            data = {'image': image}
            pixel_transformed = self.pixel_transform(data)
            image = pixel_transformed['image']
            image[padding_mask] = 0
            data = {'image': image}
            spatial_transformed = self.spatial_transform(data)
            image = spatial_transformed['image']

            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(1, 2)
            #ax[0].imshow(temp[0].cpu(), cmap='gray')
            #ax[1].imshow(image[0].cpu(), cmap='gray')
            #plt.show()

        return image, mask
    
    def set_up_preprocessing_pipeline(self):
        preprocess = Compose([
                            T.SpatialPadd(keys=['image', 'mask'], spatial_size=self.final_patch_size, allow_missing_keys=True),
                            T.CenterSpatialCropd(keys=['image', 'mask'], roi_size=self.final_patch_size, allow_missing_keys=True),
                            T.ToTensor(dtype=torch.float32, device='cuda:0')
                            ])

        return preprocess
    

    def set_up_augmentation_pipeline(self):
        spatial_transform = Compose([
                                    T.RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=0, allow_missing_keys=True),
                                    T.RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=1, allow_missing_keys=True),
                                    T.RandRotated(keys=['image', 'mask'], prob=0.2, range_x=np.pi, mode=["bilinear", "nearest"], padding_mode="zeros", allow_missing_keys=True),
                                    T.RandZoomd(keys=['image', 'mask'], prob=0.2, min_zoom=0.5, max_zoom=1.5, mode=["bilinear", "nearest"], padding_mode="constant", allow_missing_keys=True),
                                    T.RandAffined(keys=['image', 'mask'], prob=0.2, translate_range=(-26, 26), mode=["bilinear", "nearest"], padding_mode="zeros", allow_missing_keys=True)
                                    ])

        pixel_transform = Compose([
                                #RandInvertd(keys=['image'], prob=0.5),
                                T.RandAdjustContrastd(keys=['image'], prob=0.2, gamma=(0.7, 1.5), allow_missing_keys=True),
                                T.RandGaussianNoised(keys=['image'], prob=0.2, std=0.04, allow_missing_keys=True),
                                T.RandScaleIntensityd(keys=['image'], prob=0.2, factors=0.2, allow_missing_keys=True),
                                T.RandGaussianSmoothd(keys=['image'], prob=0.2, sigma_x=(0.25, 0.5), sigma_y=(0.25, 0.5), allow_missing_keys=True),
                                T.RandGaussianSharpend(keys=['image'], prob=0.2, sigma1_x=(0.1, 0.2), sigma1_y=(0.1, 0.2), sigma2_x=(0.2, 0.4), sigma2_y=(0.2, 0.4), alpha=(2.0, 3.0), allow_missing_keys=True)
                                ])

        return pixel_transform, spatial_transform
    

    def generate_train_batch(self):
        #eval_idx = None
        selected_keys = np.random.choice(self.list_of_keys, self.batch_size, True, None)
        list_of_frames = []
        for i in range(len(selected_keys)):
            patient_id = selected_keys[i][:10]
            filtered = [x for x in self.list_of_keys if x[:10] in patient_id]
            list_of_frames.append(filtered)
        
        target_list = []
        unlabeled_list = []
        padding_need_list = []
        target_mask_list = []

        case_properties = []
        for j, frames in enumerate(list_of_frames):
            labeled_frame = frames[0]
            if 'properties' in self._data[labeled_frame].keys():
                properties = self._data[labeled_frame]['properties']
            else:
                properties = load_pickle(self._data[labeled_frame]['properties_file'])
            case_properties.append(properties)

            #print(properties['ed_number'])
            #print(np.rint(np.array(properties['ed_number'])))
            #print('******************')

            ed_idx = np.rint(np.array(properties['ed_number'])).astype(np.uint8) % len(frames)
            es_idx = np.rint(np.array(properties['es_number'])).astype(np.uint8) % len(frames)

            depth = properties['size_after_resampling'][0]
            depth_idx = np.random.choice(depth)

            global_labeled_idx = np.array([ed_idx, es_idx])

            frames = sorted(frames, key=lambda x: int(x[16:18]))
            frames = np.array(frames)

            unlabeled = torch.full(size=(self.video_length, 1, *self.final_patch_size), fill_value=torch.nan, device='cuda:0')
            seg = torch.full(size=(self.video_length, 1, *self.final_patch_size), fill_value=torch.nan, device='cuda:0')
            cropped_unlabeled = torch.full(size=(self.video_length, 1, self.crop_size, self.crop_size), fill_value=torch.nan, device='cuda:0')
            cropped_seg = torch.full(size=(self.video_length, 1, self.crop_size, self.crop_size), fill_value=torch.nan, device='cuda:0')

            possible_indices = np.arange(len(frames))

            before_where = np.argwhere(possible_indices < ed_idx).reshape(-1,)
            after_where = np.argwhere(possible_indices >= ed_idx).reshape(-1,)

            before_indices = possible_indices[before_where]
            after_indices = possible_indices[after_where]

            possible_indices = np.concatenate([after_indices, before_indices])

            stop_idx = np.argwhere(possible_indices == es_idx)[0][0]
            chunk1 = possible_indices[:stop_idx + 1]
            chunk2 = possible_indices[stop_idx:]
            chunk2 = np.concatenate([np.array([possible_indices[0]]), chunk2[::-1]])

            r = np.random.randint(0, 2)
            if r == 0:
                possible_indices = chunk1
            elif r == 1:
                possible_indices = chunk2
            
            assert possible_indices[0] == ed_idx
            assert possible_indices[-1] == es_idx

            random_indices = np.random.choice(np.arange(0, len(possible_indices)), size=self.video_length - 2)
            target_mask = np.concatenate([np.array([True]), np.full_like(random_indices, fill_value=False), np.array([True])]).astype(bool)
            random_indices = np.concatenate([np.array([0]), random_indices, np.array([len(possible_indices) - 1])])
            sorted_indices = np.argsort(random_indices)

            random_indices = random_indices[sorted_indices]
            target_mask = target_mask[sorted_indices]
            target_mask = torch.from_numpy(target_mask).to('cuda:0')

            frame_indices = possible_indices[random_indices]

            assert len(frame_indices) == self.video_length
            video = frames[frame_indices]


            labeled_idx = np.where(np.isin(frame_indices, global_labeled_idx))[0]

            for frame_idx, t in enumerate(video):

                if '_u' in t:
                    if not isfile(self.un_data[t]['data_file'][:-4] + ".npy"):
                        # lets hope you know what you're doing
                        case_all_data = np.load(self.un_data[t]['data_file'][:-4] + ".npz")['data']
                    else:
                        case_all_data = np.load(self.un_data[t]['data_file'][:-4] + ".npy", self.memmap_mode)
                else:
                    if not isfile(self._data[t]['data_file'][:-4] + ".npy"):
                        # lets hope you know what you're doing
                        case_all_data = np.load(self._data[t]['data_file'][:-4] + ".npz")['data']
                    else:
                        case_all_data = np.load(self._data[t]['data_file'][:-4] + ".npy", self.memmap_mode)

                # this is for when there is just a 2d slice in case_all_data (2d support)
                if len(case_all_data.shape) == 3:
                    case_all_data = case_all_data[:, None]
                
                assert case_all_data.shape[1] == depth

                # now crop case_all_data to contain just the slice of interest. If we want additional slice above and
                # below the current slice, here is where we get them. We stack those as additional color channels
                slice_data = case_all_data[:, depth_idx]

                # case all data should now be (c, x, y)
                assert len(slice_data.shape) == 3

                if '_u' not in t:
                    image = slice_data[:-1].copy()
                    mask = slice_data[-1:].copy()
                    mask[mask < 0] = 0

                    image, mask = self.preprocess(image, mask)

                    if target_mask[frame_idx] == True:
                        seg[frame_idx] = mask
                        assert frame_idx in labeled_idx
                else:
                    image = slice_data.copy()

                    image, _ = self.preprocess(image, None)
                
                unlabeled[frame_idx] = image

            with torch.no_grad():
                mean_centroid, _ = self.processor.preprocess_no_registration(data=torch.clone(unlabeled)) # T, C(1), H, W

                cropped_unlabeled, padding_need = self.processor.crop_and_pad(data=unlabeled, mean_centroid=mean_centroid)
                padding_need_list.append(padding_need)

                cropped_seg, _ = self.processor.crop_and_pad(data=seg, mean_centroid=mean_centroid)

            cropped_unlabeled = (cropped_unlabeled - cropped_unlabeled.min()) / (cropped_unlabeled.max() - cropped_unlabeled.min() + 1e-8)

            assert torch.count_nonzero(torch.any(~torch.isnan(cropped_seg.reshape(self.video_length, -1)), dim=-1)) == 2
            assert target_mask[0] == True
            assert target_mask[-1] == True
            assert torch.count_nonzero(target_mask) == 2

            if not self.is_val and self.do_data_aug:
                temp_before_augment = torch.clone(cropped_unlabeled).to('cpu')
                
                seed = random.randint(0, 2**32-1)

                for t in range(len(cropped_unlabeled)):
                    cropped_unlabeled[t], cropped_seg[t] = self.augment(image=cropped_unlabeled[t], mask=cropped_seg[t], seed=seed)

            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(3, len(cropped_unlabeled))
            #for t in range(len(cropped_unlabeled)):
            #    ax[0, t].imshow(temp_before_augment[t, 0].cpu(), cmap='gray')
            #    ax[1, t].imshow(cropped_unlabeled[t, 0].cpu(), cmap='gray')
            #    ax[2, t].imshow(cropped_seg[t, 0].cpu(), cmap='gray')
            #plt.show()
                    
            cropped_unlabeled = NormalizeIntensity()(cropped_unlabeled)

            unlabeled_list.append(cropped_unlabeled)
            target_list.append(cropped_seg)
            target_mask_list.append(target_mask)

        unlabeled = torch.stack(unlabeled_list, dim=1) # T, B, 1, H, W
        target = torch.stack(target_list, dim=1) # T, B, 1, H, W
        target_mask = torch.stack(target_mask_list, dim=1) # T, B
        padding_need = torch.stack(padding_need_list, dim=0) # B, 4

        keys = selected_keys

        #print(seg.shape)
        #print(np.unique(seg))
        #matplotlib.use('QtAgg')
        #fig, ax = plt.subplots(1, 2)
        #ax[0].imshow(labeled[3, 0], cmap='gray')
        #ax[1].imshow(seg[3, 0], cmap='gray')
        #plt.show()

        return {'unlabeled':unlabeled, 
                'target': target,
                'in_between': None,
                'padding_need': padding_need,
                'properties': case_properties,
                'target_mask': target_mask}
    
    


class DataLoaderFlowTrain5LibProgressiveSupervised(SlimDataLoaderBase):
    def __init__(self, data, patch_size, final_patch_size, batch_size, video_length, processor, crop_size, is_val, do_data_aug, oversample_foreground_percent=0.0,
                 memmap_mode="r", pseudo_3d_slices=1, pad_mode="edge",
                 pad_kwargs_data=None, pad_sides=None):
        """
        This is the basic data loader for 2D networks. It uses preprocessed data as produced by my (Fabian) preprocessing.
        You can load the data with load_dataset(folder) where folder is the folder where the npz files are located. If there
        are only npz files present in that folder, the data loader will unpack them on the fly. This may take a while
        and increase CPU usage. Therefore, I advise you to call unpack_dataset(folder) first, which will unpack all npz
        to npy. Don't forget to call delete_npy(folder) after you are done with training?
        Why all the hassle? Well the decathlon dataset is huge. Using npy for everything will consume >1 TB and that is uncool
        given that I (Fabian) will have to store that permanently on /datasets and my local computer. With this strategy all
        data is stored in a compressed format (factor 10 smaller) and only unpacked when needed.
        :param data: get this with load_dataset(folder, stage=0). Plug the return value in here and you are g2g (good to go)
        :param patch_size: what patch size will this data loader return? it is common practice to first load larger
        patches so that a central crop after data augmentation can be done to reduce border artifacts. If unsure, use
        get_patch_size() from data_augmentation.default_data_augmentation
        :param final_patch_size: what will the patch finally be cropped to (after data augmentation)? this is the patch
        size that goes into your network. We need this here because we will pad patients in here so that patches at the
        border of patients are sampled properly
        :param batch_size:
        :param num_batches: how many batches will the data loader produce before stopping? None=endless
        :param seed:
        :param stage: ignore this (Fabian only)
        :param transpose: ignore this
        :param random: sample randomly; CAREFUL! non-random sampling requires batch_size=1, otherwise you will iterate batch_size times over the dataset
        :param pseudo_3d_slices: 7 = 3 below and 3 above the center slice
        """
        super(DataLoaderFlowTrain5LibProgressiveSupervised, self).__init__(data, batch_size, None)
        if pad_kwargs_data is None:
            pad_kwargs_data = OrderedDict()
        self.pad_kwargs_data = pad_kwargs_data
        self.pad_mode = pad_mode
        self.is_val = is_val
        self.crop_size = crop_size
        self.pseudo_3d_slices = pseudo_3d_slices
        self.oversample_foreground_percent = oversample_foreground_percent
        self.final_patch_size = final_patch_size
        self.patch_size = patch_size
        self.list_of_keys = list(self._data.keys())
        self.need_to_pad = np.array(patch_size) - np.array(final_patch_size)
        self.memmap_mode = memmap_mode
        self.video_length = video_length
        self.percent = None
        if pad_sides is not None:
            if not isinstance(pad_sides, np.ndarray):
                pad_sides = np.array(pad_sides)
            self.need_to_pad += pad_sides
        self.pad_sides = pad_sides
        self.data_shape, self.seg_shape = self.determine_shapes()
        self.do_data_aug = do_data_aug

        self.preprocessing_transform = self.set_up_preprocessing_pipeline()
        self.pixel_transform, self.spatial_transform = self.set_up_augmentation_pipeline()

        self.processor = processor
        

    def determine_shapes(self):
        num_seg = 1

        k = list(self._data.keys())[0]
        if isfile(self._data[k]['data_file'][:-4] + ".npy"):
            case_all_data = np.load(self._data[k]['data_file'][:-4] + ".npy", self.memmap_mode)
        else:
            case_all_data = np.load(self._data[k]['data_file'])['data']
        num_color_channels = case_all_data.shape[0] - num_seg
        data_shape = (self.batch_size, num_color_channels, *self.patch_size)
        seg_shape = (self.batch_size, num_seg, *self.patch_size)
        return data_shape, seg_shape

    def get_do_oversample(self, batch_idx):
        return not batch_idx < round(self.batch_size * (1 - self.oversample_foreground_percent))
    
    def select_deep_supervision(self, x):
        if self.deep_supervision:
            return x[0]
        else:
            return x
        

    def my_crop(self, labeled, seg, unlabeled):

        xm = labeled.shape[-1] / 2
        ym = labeled.shape[-2] / 2

        x1 = int(round(xm - self.final_patch_size[1] / 2))
        x2 = int(round(xm + self.final_patch_size[1] / 2))
        y1 = int(round(ym - self.final_patch_size[0] / 2))
        y2 = int(round(ym + self.final_patch_size[0] / 2))

        labeled = labeled[:, :, y1:y2, x1:x2]
        seg = seg[:, :, y1:y2, x1:x2]

        xm = unlabeled.shape[-1] / 2
        ym = unlabeled.shape[-2] / 2

        x1 = int(round(xm - self.final_patch_size[1] / 2))
        x2 = int(round(xm + self.final_patch_size[1] / 2))
        y1 = int(round(ym - self.final_patch_size[0] / 2))
        y2 = int(round(ym + self.final_patch_size[0] / 2))

        unlabeled = unlabeled[:, :, :, y1:y2, x1:x2]

        return labeled, seg, unlabeled
    
    def preprocess(self, image, mask):
        if mask is not None:
            data = {'image': image, 'mask': mask}
            transformed = self.preprocessing_transform(data)
            image = transformed['image']
            mask = transformed['mask']
        else:
            data = {'image': image}
            transformed = self.preprocessing_transform(data)
            image = transformed['image']

        return image, mask
    
    
    def augment(self, image, mask, seed):
        self.pixel_transform = self.pixel_transform.set_random_state(seed=seed)
        self.spatial_transform = self.spatial_transform.set_random_state(seed=seed)
        
        temp = torch.clone(image)
        padding_mask = image == 0

        if mask is not None:
            data = {'image': image, 'mask': mask}
            pixel_transformed = self.pixel_transform(data)
            image = pixel_transformed['image']
            mask = pixel_transformed['mask']
            image[padding_mask] = 0
            data = {'image': image, 'mask': mask}
            spatial_transformed = self.spatial_transform(data)
            image = spatial_transformed['image']
            mask = spatial_transformed['mask']

            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(1, 3)
            #ax[0].imshow(temp[0].cpu(), cmap='gray')
            #ax[1].imshow(image[0].cpu(), cmap='gray')
            #ax[2].imshow(mask[0].cpu(), cmap='gray')
            #plt.show()
        else:
            data = {'image': image}
            pixel_transformed = self.pixel_transform(data)
            image = pixel_transformed['image']
            image[padding_mask] = 0
            data = {'image': image}
            spatial_transformed = self.spatial_transform(data)
            image = spatial_transformed['image']

            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(1, 2)
            #ax[0].imshow(temp[0].cpu(), cmap='gray')
            #ax[1].imshow(image[0].cpu(), cmap='gray')
            #plt.show()

        return image, mask
    
    def set_up_preprocessing_pipeline(self):
        preprocess = Compose([
                            T.SpatialPadd(keys=['image', 'mask'], spatial_size=self.final_patch_size, allow_missing_keys=True),
                            T.CenterSpatialCropd(keys=['image', 'mask'], roi_size=self.final_patch_size, allow_missing_keys=True),
                            T.ToTensor(dtype=torch.float32, device='cuda:0')
                            ])

        return preprocess
    
    def set_up_augmentation_pipeline(self):
        spatial_transform = Compose([
                                    T.RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=0, allow_missing_keys=True),
                                    T.RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=1, allow_missing_keys=True),
                                    T.RandRotated(keys=['image', 'mask'], prob=0.2, range_x=np.pi, mode=["bilinear", "nearest"], padding_mode="zeros", allow_missing_keys=True),
                                    T.RandZoomd(keys=['image', 'mask'], prob=0.2, min_zoom=0.5, max_zoom=1.5, mode=["bilinear", "nearest"], padding_mode="constant", allow_missing_keys=True),
                                    T.RandAffined(keys=['image', 'mask'], prob=0.2, translate_range=(-26, 26), mode=["bilinear", "nearest"], padding_mode="zeros", allow_missing_keys=True)
                                    ])

        pixel_transform = Compose([
                                #RandInvertd(keys=['image'], prob=0.5),
                                T.RandAdjustContrastd(keys=['image'], prob=0.2, gamma=(0.7, 1.5), allow_missing_keys=True),
                                T.RandGaussianNoised(keys=['image'], prob=0.2, std=0.04, allow_missing_keys=True),
                                T.RandScaleIntensityd(keys=['image'], prob=0.2, factors=0.2, allow_missing_keys=True),
                                T.RandGaussianSmoothd(keys=['image'], prob=0.2, sigma_x=(0.25, 0.5), sigma_y=(0.25, 0.5), allow_missing_keys=True),
                                T.RandGaussianSharpend(keys=['image'], prob=0.2, sigma1_x=(0.1, 0.2), sigma1_y=(0.1, 0.2), sigma2_x=(0.2, 0.4), sigma2_y=(0.2, 0.4), alpha=(2.0, 3.0), allow_missing_keys=True)
                                ])

        return pixel_transform, spatial_transform

    def generate_train_batch(self):
        #eval_idx = None
        selected_keys = np.random.choice(self.list_of_keys, self.batch_size, True, None)
        list_of_frames = []
        for i in range(len(selected_keys)):
            patient_id = selected_keys[i][:10]
            filtered = [x for x in self.list_of_keys if x[:10] in patient_id]
            list_of_frames.append(filtered)
        
        target_list = []
        unlabeled_list = []
        padding_need_list = []
        target_mask_list = []

        case_properties = []
        for j, frames in enumerate(list_of_frames):
            labeled_frame = frames[0]
            if 'properties' in self._data[labeled_frame].keys():
                properties = self._data[labeled_frame]['properties']
            else:
                properties = load_pickle(self._data[labeled_frame]['properties_file'])
            case_properties.append(properties)

            #print(properties['ed_number'])
            #print(np.rint(np.array(properties['ed_number'])))
            #print('******************')

            ed_idx = np.rint(np.array(properties['ed_number'])).astype(np.uint8) % len(frames)
            es_idx = np.rint(np.array(properties['es_number'])).astype(np.uint8) % len(frames)

            depth = properties['size_after_resampling'][0]
            depth_idx = np.random.choice(depth)

            global_labeled_idx = np.array([ed_idx, es_idx])

            frames = sorted(frames, key=lambda x: int(x[16:18]))
            frames = np.array(frames)

            unlabeled = torch.full(size=(self.video_length, 1, *self.final_patch_size), fill_value=torch.nan, device='cuda:0')
            seg = torch.full(size=(self.video_length, 1, *self.final_patch_size), fill_value=torch.nan, device='cuda:0')
            cropped_unlabeled = torch.full(size=(self.video_length, 1, self.crop_size, self.crop_size), fill_value=torch.nan, device='cuda:0')
            cropped_seg = torch.full(size=(self.video_length, 1, self.crop_size, self.crop_size), fill_value=torch.nan, device='cuda:0')

            possible_indices = np.arange(len(frames))

            before_where = np.argwhere(possible_indices < ed_idx).reshape(-1,)
            after_where = np.argwhere(possible_indices >= ed_idx).reshape(-1,)

            before_indices = possible_indices[before_where]
            after_indices = possible_indices[after_where]

            possible_indices = np.concatenate([after_indices, before_indices])

            stop_idx = np.argwhere(possible_indices == es_idx)[0][0]
            chunk1 = possible_indices[:stop_idx + 1]
            chunk2 = possible_indices[stop_idx:]
            chunk2 = np.concatenate([np.array([possible_indices[0]]), chunk2[::-1]])

            r = np.random.randint(0, 2)
            if r == 0:
                possible_indices = chunk1
            elif r == 1:
                possible_indices = chunk2
            
            assert possible_indices[0] == ed_idx
            assert possible_indices[-1] == es_idx

            random_indices = np.random.choice(np.arange(0, len(possible_indices)), size=self.video_length - 1)
            random_indices = np.concatenate([np.array([0]), random_indices])
            target_mask = np.full_like(random_indices, fill_value=True).astype(bool)
            sorted_indices = np.argsort(random_indices)

            random_indices = random_indices[sorted_indices]
            target_mask = target_mask[sorted_indices]
            target_mask = torch.from_numpy(target_mask).to('cuda:0')

            frame_indices = possible_indices[random_indices]

            assert len(frame_indices) == self.video_length
            video = frames[frame_indices]

            for frame_idx, t in enumerate(video):

                if '_u' in t:
                    if not isfile(self.un_data[t]['data_file'][:-4] + ".npy"):
                        # lets hope you know what you're doing
                        case_all_data = np.load(self.un_data[t]['data_file'][:-4] + ".npz")['data']
                    else:
                        case_all_data = np.load(self.un_data[t]['data_file'][:-4] + ".npy", self.memmap_mode)
                else:
                    if not isfile(self._data[t]['data_file'][:-4] + ".npy"):
                        # lets hope you know what you're doing
                        case_all_data = np.load(self._data[t]['data_file'][:-4] + ".npz")['data']
                    else:
                        case_all_data = np.load(self._data[t]['data_file'][:-4] + ".npy", self.memmap_mode)

                # this is for when there is just a 2d slice in case_all_data (2d support)
                if len(case_all_data.shape) == 3:
                    case_all_data = case_all_data[:, None]
                
                assert case_all_data.shape[1] == depth

                # now crop case_all_data to contain just the slice of interest. If we want additional slice above and
                # below the current slice, here is where we get them. We stack those as additional color channels
                slice_data = case_all_data[:, depth_idx]

                # case all data should now be (c, x, y)
                assert len(slice_data.shape) == 3

                if '_u' not in t:
                    image = slice_data[:-1].copy()
                    mask = slice_data[-1:].copy()
                    mask[mask < 0] = 0

                    image, mask = self.preprocess(image, mask)

                    if target_mask[frame_idx] == True:
                        seg[frame_idx] = mask
                else:
                    image = slice_data.copy()

                    image, _ = self.preprocess(image, None)
                
                unlabeled[frame_idx] = image

            with torch.no_grad():
                mean_centroid, _ = self.processor.preprocess_no_registration(data=torch.clone(unlabeled)) # T, C(1), H, W

                cropped_unlabeled, padding_need = self.processor.crop_and_pad(data=unlabeled, mean_centroid=mean_centroid)
                padding_need_list.append(padding_need)

                cropped_seg, _ = self.processor.crop_and_pad(data=seg, mean_centroid=mean_centroid)

            cropped_unlabeled = (cropped_unlabeled - cropped_unlabeled.min()) / (cropped_unlabeled.max() - cropped_unlabeled.min() + 1e-8)

            assert torch.count_nonzero(torch.any(~torch.isnan(cropped_seg.reshape(self.video_length, -1)), dim=-1)) == len(video)
            assert torch.count_nonzero(target_mask) == len(video)

            if not self.is_val and self.do_data_aug:
                temp_before_augment = torch.clone(cropped_unlabeled).to('cpu')
                
                seed = random.randint(0, 2**32-1)

                for t in range(len(cropped_unlabeled)):
                    cropped_unlabeled[t], cropped_seg[t] = self.augment(image=cropped_unlabeled[t], mask=cropped_seg[t], seed=seed)

            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(3, len(cropped_unlabeled))
            #for t in range(len(cropped_unlabeled)):
            #    ax[0, t].imshow(temp_before_augment[t, 0].cpu(), cmap='gray')
            #    ax[1, t].imshow(cropped_unlabeled[t, 0].cpu(), cmap='gray')
            #    ax[2, t].imshow(cropped_seg[t, 0].cpu(), cmap='gray')
            #plt.show()
                    
            cropped_unlabeled = NormalizeIntensity()(cropped_unlabeled)

            unlabeled_list.append(cropped_unlabeled)
            target_list.append(cropped_seg)
            target_mask_list.append(target_mask)

        unlabeled = torch.stack(unlabeled_list, dim=1) # T, B, 1, H, W
        target = torch.stack(target_list, dim=1) # T, B, 1, H, W
        target_mask = torch.stack(target_mask_list, dim=1) # T, B
        padding_need = torch.stack(padding_need_list, dim=0) # B, 4

        keys = selected_keys

        #print(seg.shape)
        #print(np.unique(seg))
        #matplotlib.use('QtAgg')
        #fig, ax = plt.subplots(1, 2)
        #ax[0].imshow(labeled[3, 0], cmap='gray')
        #ax[1].imshow(seg[3, 0], cmap='gray')
        #plt.show()

        return {'unlabeled':unlabeled, 
                'target': target,
                'in_between': None,
                'padding_need': padding_need,
                'properties': case_properties,
                'target_mask': target_mask}


class DataLoaderFlowTrain5LibProgressiveNoSorting(SlimDataLoaderBase):
    def __init__(self, data, patch_size, final_patch_size, batch_size, video_length, processor, crop_size, is_val, do_data_aug, oversample_foreground_percent=0.0,
                 memmap_mode="r", pseudo_3d_slices=1, pad_mode="edge",
                 pad_kwargs_data=None, pad_sides=None):
        """
        This is the basic data loader for 2D networks. It uses preprocessed data as produced by my (Fabian) preprocessing.
        You can load the data with load_dataset(folder) where folder is the folder where the npz files are located. If there
        are only npz files present in that folder, the data loader will unpack them on the fly. This may take a while
        and increase CPU usage. Therefore, I advise you to call unpack_dataset(folder) first, which will unpack all npz
        to npy. Don't forget to call delete_npy(folder) after you are done with training?
        Why all the hassle? Well the decathlon dataset is huge. Using npy for everything will consume >1 TB and that is uncool
        given that I (Fabian) will have to store that permanently on /datasets and my local computer. With this strategy all
        data is stored in a compressed format (factor 10 smaller) and only unpacked when needed.
        :param data: get this with load_dataset(folder, stage=0). Plug the return value in here and you are g2g (good to go)
        :param patch_size: what patch size will this data loader return? it is common practice to first load larger
        patches so that a central crop after data augmentation can be done to reduce border artifacts. If unsure, use
        get_patch_size() from data_augmentation.default_data_augmentation
        :param final_patch_size: what will the patch finally be cropped to (after data augmentation)? this is the patch
        size that goes into your network. We need this here because we will pad patients in here so that patches at the
        border of patients are sampled properly
        :param batch_size:
        :param num_batches: how many batches will the data loader produce before stopping? None=endless
        :param seed:
        :param stage: ignore this (Fabian only)
        :param transpose: ignore this
        :param random: sample randomly; CAREFUL! non-random sampling requires batch_size=1, otherwise you will iterate batch_size times over the dataset
        :param pseudo_3d_slices: 7 = 3 below and 3 above the center slice
        """
        super(DataLoaderFlowTrain5LibProgressiveNoSorting, self).__init__(data, batch_size, None)
        if pad_kwargs_data is None:
            pad_kwargs_data = OrderedDict()
        self.pad_kwargs_data = pad_kwargs_data
        self.pad_mode = pad_mode
        self.is_val = is_val
        self.crop_size = crop_size
        self.pseudo_3d_slices = pseudo_3d_slices
        self.oversample_foreground_percent = oversample_foreground_percent
        self.final_patch_size = final_patch_size
        self.patch_size = patch_size
        self.list_of_keys = list(self._data.keys())
        self.need_to_pad = np.array(patch_size) - np.array(final_patch_size)
        self.memmap_mode = memmap_mode
        self.video_length = video_length
        self.percent = None
        if pad_sides is not None:
            if not isinstance(pad_sides, np.ndarray):
                pad_sides = np.array(pad_sides)
            self.need_to_pad += pad_sides
        self.pad_sides = pad_sides
        self.data_shape, self.seg_shape = self.determine_shapes()
        self.do_data_aug = do_data_aug

        self.preprocessing_transform = self.set_up_preprocessing_pipeline()
        self.pixel_transform, self.spatial_transform = self.set_up_augmentation_pipeline()

        self.processor = processor
        

    def determine_shapes(self):
        num_seg = 1

        k = list(self._data.keys())[0]
        if isfile(self._data[k]['data_file'][:-4] + ".npy"):
            case_all_data = np.load(self._data[k]['data_file'][:-4] + ".npy", self.memmap_mode)
        else:
            case_all_data = np.load(self._data[k]['data_file'])['data']
        num_color_channels = case_all_data.shape[0] - num_seg
        data_shape = (self.batch_size, num_color_channels, *self.patch_size)
        seg_shape = (self.batch_size, num_seg, *self.patch_size)
        return data_shape, seg_shape

    def get_do_oversample(self, batch_idx):
        return not batch_idx < round(self.batch_size * (1 - self.oversample_foreground_percent))
    
    def select_deep_supervision(self, x):
        if self.deep_supervision:
            return x[0]
        else:
            return x
        

    def my_crop(self, labeled, seg, unlabeled):

        xm = labeled.shape[-1] / 2
        ym = labeled.shape[-2] / 2

        x1 = int(round(xm - self.final_patch_size[1] / 2))
        x2 = int(round(xm + self.final_patch_size[1] / 2))
        y1 = int(round(ym - self.final_patch_size[0] / 2))
        y2 = int(round(ym + self.final_patch_size[0] / 2))

        labeled = labeled[:, :, y1:y2, x1:x2]
        seg = seg[:, :, y1:y2, x1:x2]

        xm = unlabeled.shape[-1] / 2
        ym = unlabeled.shape[-2] / 2

        x1 = int(round(xm - self.final_patch_size[1] / 2))
        x2 = int(round(xm + self.final_patch_size[1] / 2))
        y1 = int(round(ym - self.final_patch_size[0] / 2))
        y2 = int(round(ym + self.final_patch_size[0] / 2))

        unlabeled = unlabeled[:, :, :, y1:y2, x1:x2]

        return labeled, seg, unlabeled
    
    def preprocess(self, image, mask):
        if mask is not None:
            data = {'image': image, 'mask': mask}
            transformed = self.preprocessing_transform(data)
            image = transformed['image']
            mask = transformed['mask']
        else:
            data = {'image': image}
            transformed = self.preprocessing_transform(data)
            image = transformed['image']

        return image, mask
    
    
    def augment(self, image, mask, seed):
        self.pixel_transform = self.pixel_transform.set_random_state(seed=seed)
        self.spatial_transform = self.spatial_transform.set_random_state(seed=seed)
        
        temp = torch.clone(image)
        padding_mask = image == 0

        if mask is not None:
            data = {'image': image, 'mask': mask}
            pixel_transformed = self.pixel_transform(data)
            image = pixel_transformed['image']
            mask = pixel_transformed['mask']
            image[padding_mask] = 0
            data = {'image': image, 'mask': mask}
            spatial_transformed = self.spatial_transform(data)
            image = spatial_transformed['image']
            mask = spatial_transformed['mask']

            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(1, 3)
            #ax[0].imshow(temp[0].cpu(), cmap='gray')
            #ax[1].imshow(image[0].cpu(), cmap='gray')
            #ax[2].imshow(mask[0].cpu(), cmap='gray')
            #plt.show()
        else:
            data = {'image': image}
            pixel_transformed = self.pixel_transform(data)
            image = pixel_transformed['image']
            image[padding_mask] = 0
            data = {'image': image}
            spatial_transformed = self.spatial_transform(data)
            image = spatial_transformed['image']

            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(1, 2)
            #ax[0].imshow(temp[0].cpu(), cmap='gray')
            #ax[1].imshow(image[0].cpu(), cmap='gray')
            #plt.show()

        return image, mask
    
    def set_up_preprocessing_pipeline(self):
        preprocess = Compose([
                            T.SpatialPadd(keys=['image', 'mask'], spatial_size=self.final_patch_size, allow_missing_keys=True),
                            T.CenterSpatialCropd(keys=['image', 'mask'], roi_size=self.final_patch_size, allow_missing_keys=True),
                            T.ToTensor(dtype=torch.float32, device='cuda:0')
                            ])

        return preprocess
    
    def set_up_augmentation_pipeline(self):
        spatial_transform = Compose([
                                    T.RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=0, allow_missing_keys=True),
                                    T.RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=1, allow_missing_keys=True),
                                    T.RandRotated(keys=['image', 'mask'], prob=0.2, range_x=np.pi, mode=["bilinear", "nearest"], padding_mode="zeros", allow_missing_keys=True),
                                    T.RandZoomd(keys=['image', 'mask'], prob=0.2, min_zoom=0.5, max_zoom=1.5, mode=["bilinear", "nearest"], padding_mode="constant", allow_missing_keys=True),
                                    T.RandAffined(keys=['image', 'mask'], prob=0.2, translate_range=(-26, 26), mode=["bilinear", "nearest"], padding_mode="zeros", allow_missing_keys=True)
                                    ])

        pixel_transform = Compose([
                                #RandInvertd(keys=['image'], prob=0.5),
                                T.RandAdjustContrastd(keys=['image'], prob=0.2, gamma=(0.7, 1.5), allow_missing_keys=True),
                                T.RandGaussianNoised(keys=['image'], prob=0.2, std=0.04, allow_missing_keys=True),
                                T.RandScaleIntensityd(keys=['image'], prob=0.2, factors=0.2, allow_missing_keys=True),
                                T.RandGaussianSmoothd(keys=['image'], prob=0.2, sigma_x=(0.25, 0.5), sigma_y=(0.25, 0.5), allow_missing_keys=True),
                                T.RandGaussianSharpend(keys=['image'], prob=0.2, sigma1_x=(0.1, 0.2), sigma1_y=(0.1, 0.2), sigma2_x=(0.2, 0.4), sigma2_y=(0.2, 0.4), alpha=(2.0, 3.0), allow_missing_keys=True)
                                ])

        return pixel_transform, spatial_transform

    def generate_train_batch(self):
        #eval_idx = None
        selected_keys = np.random.choice(self.list_of_keys, self.batch_size, True, None)
        list_of_frames = []
        for i in range(len(selected_keys)):
            patient_id = selected_keys[i][:10]
            filtered = [x for x in self.list_of_keys if x[:10] in patient_id]
            list_of_frames.append(filtered)
        
        target_list = []
        unlabeled_list = []
        padding_need_list = []
        target_mask_list = []

        case_properties = []
        for j, frames in enumerate(list_of_frames):
            labeled_frame = frames[0]
            if 'properties' in self._data[labeled_frame].keys():
                properties = self._data[labeled_frame]['properties']
            else:
                properties = load_pickle(self._data[labeled_frame]['properties_file'])
            case_properties.append(properties)

            #print(properties['ed_number'])
            #print(np.rint(np.array(properties['ed_number'])))
            #print('******************')

            ed_idx = 0
            #es_idx = len(frames) // 2
            es_idx = np.rint(np.array(properties['es_number'])).astype(np.uint8) % len(frames)

            depth = properties['size_after_resampling'][0]
            depth_idx = np.random.choice(depth)

            global_labeled_idx = np.array([ed_idx, es_idx])

            frames = sorted(frames, key=lambda x: int(x[16:18]))
            frames = np.array(frames)

            unlabeled = torch.full(size=(self.video_length, 1, *self.final_patch_size), fill_value=torch.nan, device='cuda:0')
            seg = torch.full(size=(self.video_length, 1, *self.final_patch_size), fill_value=torch.nan, device='cuda:0')
            cropped_unlabeled = torch.full(size=(self.video_length, 1, self.crop_size, self.crop_size), fill_value=torch.nan, device='cuda:0')
            cropped_seg = torch.full(size=(self.video_length, 1, self.crop_size, self.crop_size), fill_value=torch.nan, device='cuda:0')

            possible_indices = np.arange(len(frames))

            before_where = np.argwhere(possible_indices < ed_idx).reshape(-1,)
            after_where = np.argwhere(possible_indices >= ed_idx).reshape(-1,)

            before_indices = possible_indices[before_where]
            after_indices = possible_indices[after_where]

            possible_indices = np.concatenate([after_indices, before_indices])

            stop_idx = np.argwhere(possible_indices == es_idx)[0][0]
            chunk1 = possible_indices[:stop_idx + 1]
            chunk2 = possible_indices[stop_idx:]
            chunk2 = np.concatenate([np.array([possible_indices[0]]), chunk2[::-1]])

            r = np.random.randint(0, 2)
            if r == 0:
                possible_indices = chunk1
            elif r == 1:
                possible_indices = chunk2
            
            assert possible_indices[0] == ed_idx
            assert possible_indices[-1] == es_idx

            random_indices = np.random.choice(np.arange(0, len(possible_indices)), size=self.video_length - 2)
            target_mask = np.concatenate([np.array([True]), np.full_like(random_indices, fill_value=False), np.array([True])]).astype(bool)
            random_indices = np.concatenate([np.array([0]), random_indices, np.array([len(possible_indices) - 1])])
            sorted_indices = np.argsort(random_indices)

            random_indices = random_indices[sorted_indices]
            target_mask = target_mask[sorted_indices]
            target_mask = torch.from_numpy(target_mask).to('cuda:0')

            frame_indices = possible_indices[random_indices]

            assert len(frame_indices) == self.video_length
            video = frames[frame_indices]


            labeled_idx = np.where(np.isin(frame_indices, global_labeled_idx))[0]

            for frame_idx, t in enumerate(video):

                if '_u' in t:
                    if not isfile(self.un_data[t]['data_file'][:-4] + ".npy"):
                        # lets hope you know what you're doing
                        case_all_data = np.load(self.un_data[t]['data_file'][:-4] + ".npz")['data']
                    else:
                        case_all_data = np.load(self.un_data[t]['data_file'][:-4] + ".npy", self.memmap_mode)
                else:
                    if not isfile(self._data[t]['data_file'][:-4] + ".npy"):
                        # lets hope you know what you're doing
                        case_all_data = np.load(self._data[t]['data_file'][:-4] + ".npz")['data']
                    else:
                        case_all_data = np.load(self._data[t]['data_file'][:-4] + ".npy", self.memmap_mode)

                # this is for when there is just a 2d slice in case_all_data (2d support)
                if len(case_all_data.shape) == 3:
                    case_all_data = case_all_data[:, None]
                
                assert case_all_data.shape[1] == depth

                # now crop case_all_data to contain just the slice of interest. If we want additional slice above and
                # below the current slice, here is where we get them. We stack those as additional color channels
                slice_data = case_all_data[:, depth_idx]

                # case all data should now be (c, x, y)
                assert len(slice_data.shape) == 3

                if '_u' not in t:
                    image = slice_data[:-1].copy()
                    mask = slice_data[-1:].copy()
                    mask[mask < 0] = 0

                    image, mask = self.preprocess(image, mask)

                    if target_mask[frame_idx] == True:
                        seg[frame_idx] = mask
                        assert frame_idx in labeled_idx
                else:
                    image = slice_data.copy()

                    image, _ = self.preprocess(image, None)
                
                unlabeled[frame_idx] = image

            with torch.no_grad():
                mean_centroid, _ = self.processor.preprocess_no_registration(data=torch.clone(unlabeled)) # T, C(1), H, W

                cropped_unlabeled, padding_need = self.processor.crop_and_pad(data=unlabeled, mean_centroid=mean_centroid)
                padding_need_list.append(padding_need)

                cropped_seg, _ = self.processor.crop_and_pad(data=seg, mean_centroid=mean_centroid)

            cropped_unlabeled = (cropped_unlabeled - cropped_unlabeled.min()) / (cropped_unlabeled.max() - cropped_unlabeled.min() + 1e-8)

            assert torch.count_nonzero(torch.any(~torch.isnan(cropped_seg.reshape(self.video_length, -1)), dim=-1)) == 2
            assert target_mask[0] == True
            assert target_mask[-1] == True
            assert torch.count_nonzero(target_mask) == 2

            if not self.is_val and self.do_data_aug:
                temp_before_augment = torch.clone(cropped_unlabeled).to('cpu')
                
                seed = random.randint(0, 2**32-1)

                for t in range(len(cropped_unlabeled)):
                    cropped_unlabeled[t], cropped_seg[t] = self.augment(image=cropped_unlabeled[t], mask=cropped_seg[t], seed=seed)

            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(3, len(cropped_unlabeled))
            #for t in range(len(cropped_unlabeled)):
            #    ax[0, t].imshow(temp_before_augment[t, 0].cpu(), cmap='gray')
            #    ax[1, t].imshow(cropped_unlabeled[t, 0].cpu(), cmap='gray')
            #    ax[2, t].imshow(cropped_seg[t, 0].cpu(), cmap='gray')
            #plt.show()
                    
            cropped_unlabeled = NormalizeIntensity()(cropped_unlabeled)

            unlabeled_list.append(cropped_unlabeled)
            target_list.append(cropped_seg)
            target_mask_list.append(target_mask)

        unlabeled = torch.stack(unlabeled_list, dim=1) # T, B, 1, H, W
        target = torch.stack(target_list, dim=1) # T, B, 1, H, W
        target_mask = torch.stack(target_mask_list, dim=1) # T, B
        padding_need = torch.stack(padding_need_list, dim=0) # B, 4

        keys = selected_keys

        #print(seg.shape)
        #print(np.unique(seg))
        #matplotlib.use('QtAgg')
        #fig, ax = plt.subplots(1, 2)
        #ax[0].imshow(labeled[3, 0], cmap='gray')
        #ax[1].imshow(seg[3, 0], cmap='gray')
        #plt.show()

        return {'unlabeled':unlabeled, 
                'target': target,
                'in_between': None,
                'padding_need': padding_need,
                'properties': case_properties,
                'target_mask': target_mask}
    




class DataLoaderFlowTrainPrediction(SlimDataLoaderBase):
    def __init__(self, data, patch_size, final_patch_size, batch_size, video_length, processor, crop_size, is_val, do_data_aug, oversample_foreground_percent=0.0,
                 memmap_mode="r", pseudo_3d_slices=1, pad_mode="edge",
                 pad_kwargs_data=None, pad_sides=None):
        """
        This is the basic data loader for 2D networks. It uses preprocessed data as produced by my (Fabian) preprocessing.
        You can load the data with load_dataset(folder) where folder is the folder where the npz files are located. If there
        are only npz files present in that folder, the data loader will unpack them on the fly. This may take a while
        and increase CPU usage. Therefore, I advise you to call unpack_dataset(folder) first, which will unpack all npz
        to npy. Don't forget to call delete_npy(folder) after you are done with training?
        Why all the hassle? Well the decathlon dataset is huge. Using npy for everything will consume >1 TB and that is uncool
        given that I (Fabian) will have to store that permanently on /datasets and my local computer. With this strategy all
        data is stored in a compressed format (factor 10 smaller) and only unpacked when needed.
        :param data: get this with load_dataset(folder, stage=0). Plug the return value in here and you are g2g (good to go)
        :param patch_size: what patch size will this data loader return? it is common practice to first load larger
        patches so that a central crop after data augmentation can be done to reduce border artifacts. If unsure, use
        get_patch_size() from data_augmentation.default_data_augmentation
        :param final_patch_size: what will the patch finally be cropped to (after data augmentation)? this is the patch
        size that goes into your network. We need this here because we will pad patients in here so that patches at the
        border of patients are sampled properly
        :param batch_size:
        :param num_batches: how many batches will the data loader produce before stopping? None=endless
        :param seed:
        :param stage: ignore this (Fabian only)
        :param transpose: ignore this
        :param random: sample randomly; CAREFUL! non-random sampling requires batch_size=1, otherwise you will iterate batch_size times over the dataset
        :param pseudo_3d_slices: 7 = 3 below and 3 above the center slice
        """
        super(DataLoaderFlowTrainPrediction, self).__init__(data, batch_size, None)
        if pad_kwargs_data is None:
            pad_kwargs_data = OrderedDict()
        self.pad_kwargs_data = pad_kwargs_data
        self.pad_mode = pad_mode
        self.is_val = is_val
        self.crop_size = crop_size
        self.pseudo_3d_slices = pseudo_3d_slices
        self.oversample_foreground_percent = oversample_foreground_percent
        self.final_patch_size = final_patch_size
        self.patch_size = patch_size
        self.list_of_keys = list(self._data.keys())
        self.need_to_pad = np.array(patch_size) - np.array(final_patch_size)
        self.memmap_mode = memmap_mode
        self.video_length = video_length
        self.percent = None
        if pad_sides is not None:
            if not isinstance(pad_sides, np.ndarray):
                pad_sides = np.array(pad_sides)
            self.need_to_pad += pad_sides
        self.pad_sides = pad_sides
        self.data_shape, self.seg_shape = self.determine_shapes()
        self.do_data_aug = do_data_aug

        self.preprocessing_transform = self.set_up_preprocessing_pipeline()
        self.pixel_transform, self.spatial_transform = self.set_up_augmentation_pipeline()

        self.processor = processor
        

    def determine_shapes(self):
        num_seg = 1

        k = list(self._data.keys())[0]
        if isfile(self._data[k]['data_file'][:-4] + ".npy"):
            case_all_data = np.load(self._data[k]['data_file'][:-4] + ".npy", self.memmap_mode)
        else:
            case_all_data = np.load(self._data[k]['data_file'])['data']
        num_color_channels = case_all_data.shape[0] - num_seg
        data_shape = (self.batch_size, num_color_channels, *self.patch_size)
        seg_shape = (self.batch_size, num_seg, *self.patch_size)
        return data_shape, seg_shape

    def get_do_oversample(self, batch_idx):
        return not batch_idx < round(self.batch_size * (1 - self.oversample_foreground_percent))
    
    def select_deep_supervision(self, x):
        if self.deep_supervision:
            return x[0]
        else:
            return x
        

    def my_crop(self, labeled, seg, unlabeled):

        xm = labeled.shape[-1] / 2
        ym = labeled.shape[-2] / 2

        x1 = int(round(xm - self.final_patch_size[1] / 2))
        x2 = int(round(xm + self.final_patch_size[1] / 2))
        y1 = int(round(ym - self.final_patch_size[0] / 2))
        y2 = int(round(ym + self.final_patch_size[0] / 2))

        labeled = labeled[:, :, y1:y2, x1:x2]
        seg = seg[:, :, y1:y2, x1:x2]

        xm = unlabeled.shape[-1] / 2
        ym = unlabeled.shape[-2] / 2

        x1 = int(round(xm - self.final_patch_size[1] / 2))
        x2 = int(round(xm + self.final_patch_size[1] / 2))
        y1 = int(round(ym - self.final_patch_size[0] / 2))
        y2 = int(round(ym + self.final_patch_size[0] / 2))

        unlabeled = unlabeled[:, :, :, y1:y2, x1:x2]

        return labeled, seg, unlabeled
    
    def preprocess(self, image, mask):
        if mask is not None:
            data = {'image': image, 'mask': mask}
            transformed = self.preprocessing_transform(data)
            image = transformed['image']
            mask = transformed['mask']
        else:
            data = {'image': image}
            transformed = self.preprocessing_transform(data)
            image = transformed['image']

        return image, mask
    
    
    def augment(self, image, mask, seed):
        self.pixel_transform = self.pixel_transform.set_random_state(seed=seed)
        self.spatial_transform = self.spatial_transform.set_random_state(seed=seed)
        
        temp = torch.clone(image)
        padding_mask = image == 0

        if mask is not None:
            data = {'image': image, 'mask': mask}
            pixel_transformed = self.pixel_transform(data)
            image = pixel_transformed['image']
            mask = pixel_transformed['mask']
            image[padding_mask] = 0
            data = {'image': image, 'mask': mask}
            spatial_transformed = self.spatial_transform(data)
            image = spatial_transformed['image']
            mask = spatial_transformed['mask']

            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(1, 3)
            #ax[0].imshow(temp[0].cpu(), cmap='gray')
            #ax[1].imshow(image[0].cpu(), cmap='gray')
            #ax[2].imshow(mask[0].cpu(), cmap='gray')
            #plt.show()
        else:
            data = {'image': image}
            pixel_transformed = self.pixel_transform(data)
            image = pixel_transformed['image']
            image[padding_mask] = 0
            data = {'image': image}
            spatial_transformed = self.spatial_transform(data)
            image = spatial_transformed['image']

            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(1, 2)
            #ax[0].imshow(temp[0].cpu(), cmap='gray')
            #ax[1].imshow(image[0].cpu(), cmap='gray')
            #plt.show()

        return image, mask
    
    def set_up_preprocessing_pipeline(self):
        preprocess = Compose([
                            T.SpatialPadd(keys=['image', 'mask'], spatial_size=self.final_patch_size, allow_missing_keys=True),
                            T.CenterSpatialCropd(keys=['image', 'mask'], roi_size=self.final_patch_size, allow_missing_keys=True),
                            T.ToTensor(dtype=torch.float32, device='cuda:0')
                            ])

        return preprocess
    
    def set_up_augmentation_pipeline(self):
        spatial_transform = Compose([
                                    T.RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=0, allow_missing_keys=True),
                                    T.RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=1, allow_missing_keys=True),
                                    T.RandRotated(keys=['image', 'mask'], prob=0.2, range_x=np.pi, mode=["bilinear", "nearest"], padding_mode="zeros", allow_missing_keys=True),
                                    T.RandZoomd(keys=['image', 'mask'], prob=0.2, min_zoom=0.5, max_zoom=1.5, mode=["bilinear", "nearest"], padding_mode="constant", allow_missing_keys=True),
                                    T.RandAffined(keys=['image', 'mask'], prob=0.2, translate_range=(-26, 26), mode=["bilinear", "nearest"], padding_mode="zeros", allow_missing_keys=True)
                                    ])

        pixel_transform = Compose([
                                #RandInvertd(keys=['image'], prob=0.5),
                                T.RandAdjustContrastd(keys=['image'], prob=0.2, gamma=(0.7, 1.5), allow_missing_keys=True),
                                T.RandGaussianNoised(keys=['image'], prob=0.2, std=0.04, allow_missing_keys=True),
                                T.RandScaleIntensityd(keys=['image'], prob=0.2, factors=0.2, allow_missing_keys=True),
                                T.RandGaussianSmoothd(keys=['image'], prob=0.2, sigma_x=(0.25, 0.5), sigma_y=(0.25, 0.5), allow_missing_keys=True),
                                T.RandGaussianSharpend(keys=['image'], prob=0.2, sigma1_x=(0.1, 0.2), sigma1_y=(0.1, 0.2), sigma2_x=(0.2, 0.4), sigma2_y=(0.2, 0.4), alpha=(2.0, 3.0), allow_missing_keys=True)
                                ])

        return pixel_transform, spatial_transform

    def generate_train_batch(self):
        #eval_idx = None
        selected_keys = np.random.choice(self.list_of_keys, self.batch_size, True, None)
        list_of_frames = []
        for i in range(len(selected_keys)):
            patient_id = selected_keys[i][:10]
            filtered = [x for x in self.list_of_keys if x[:10] in patient_id]
            list_of_frames.append(filtered)
        
        target_list = []
        unlabeled_list = []
        padding_need_list = []
        target_mask_list = []
        local_distances_list = []

        case_properties = []
        for j, frames in enumerate(list_of_frames):
            labeled_frame = frames[0]
            if 'properties' in self._data[labeled_frame].keys():
                properties = self._data[labeled_frame]['properties']
            else:
                properties = load_pickle(self._data[labeled_frame]['properties_file'])
            case_properties.append(properties)

            #print(properties['ed_number'])
            #print(np.rint(np.array(properties['ed_number'])))
            #print('******************')

            ed_idx = np.rint(np.array(properties['ed_number'])).astype(np.uint8) % len(frames)
            es_idx = np.rint(np.array(properties['es_number'])).astype(np.uint8) % len(frames)

            depth = properties['size_after_resampling'][0]
            depth_idx = np.random.choice(depth)

            global_labeled_idx = np.array([ed_idx, es_idx])

            frames = sorted(frames, key=lambda x: int(x[16:18]))
            frames = np.array(frames)

            unlabeled = torch.full(size=(self.video_length, 1, *self.final_patch_size), fill_value=torch.nan, device='cuda:0')
            seg = torch.full(size=(self.video_length, 1, *self.final_patch_size), fill_value=torch.nan, device='cuda:0')
            cropped_unlabeled = torch.full(size=(self.video_length, 1, self.crop_size, self.crop_size), fill_value=torch.nan, device='cuda:0')
            cropped_seg = torch.full(size=(self.video_length, 1, self.crop_size, self.crop_size), fill_value=torch.nan, device='cuda:0')

            possible_indices = np.arange(len(frames))

            before_where = np.argwhere(possible_indices < ed_idx).reshape(-1,)
            after_where = np.argwhere(possible_indices >= ed_idx).reshape(-1,)

            before_indices = possible_indices[before_where]
            after_indices = possible_indices[after_where]

            possible_indices = np.concatenate([after_indices, before_indices])

            sorted_frames = frames[possible_indices]
            indices = np.arange(len(sorted_frames))
            sorted_frame_indices = [int(x.split('frame')[-1]) for x in sorted_frames.tolist()]
            assert sorted_frame_indices[0] == ed_idx + 1
            stop_idx = np.argwhere(np.array(sorted_frame_indices) == es_idx + 1)[0][0]

            chunk1 = sorted_frames[:stop_idx + 1]
            indices1 = indices[:stop_idx + 1]
            chunk2 = sorted_frames[stop_idx:]
            indices2 = indices[stop_idx:]
            chunk2 = np.concatenate([np.array(sorted_frames[0]).reshape(1,), chunk2[::-1]])
            indices2 = np.concatenate([np.array(indices[0]).reshape(1,), indices2[::-1]])

            r = np.random.randint(0, 2)
            if r == 0:
                possible_frames = chunk1
                possible_indices = indices1
            elif r == 1:
                possible_frames = chunk2
                possible_indices = indices2
            
            assert int(possible_frames[0].split('frame')[-1]) == ed_idx + 1
            assert int(possible_frames[-1].split('frame')[-1]) == es_idx + 1

            random_indices = np.random.choice(np.arange(0, len(possible_frames)), size=self.video_length - 2)
            target_mask = np.concatenate([np.array([True]), np.full_like(random_indices, fill_value=False), np.array([True])]).astype(bool)
            random_indices = np.concatenate([np.array([0]), random_indices, np.array([len(possible_frames) - 1])])
            sorted_indices = np.argsort(random_indices)

            random_indices = random_indices[sorted_indices]
            target_mask = target_mask[sorted_indices]
            target_mask = torch.from_numpy(target_mask).to('cuda:0')

            #print(random_indices)

            diff = np.concatenate([np.diff(random_indices), np.array([0])])
            diff = diff / len(frames)
            local_distances = torch.from_numpy(diff).float().to('cuda:0')
            assert len(local_distances) == len(random_indices)
            local_distances_list.append(local_distances)

            video = possible_frames[random_indices]

            assert len(video) == self.video_length

            labeled_idx = [0, len(video) - 1]

            for frame_idx, t in enumerate(video):

                if '_u' in t:
                    if not isfile(self.un_data[t]['data_file'][:-4] + ".npy"):
                        # lets hope you know what you're doing
                        case_all_data = np.load(self.un_data[t]['data_file'][:-4] + ".npz")['data']
                    else:
                        case_all_data = np.load(self.un_data[t]['data_file'][:-4] + ".npy", self.memmap_mode)
                else:
                    if not isfile(self._data[t]['data_file'][:-4] + ".npy"):
                        # lets hope you know what you're doing
                        case_all_data = np.load(self._data[t]['data_file'][:-4] + ".npz")['data']
                    else:
                        case_all_data = np.load(self._data[t]['data_file'][:-4] + ".npy", self.memmap_mode)

                # this is for when there is just a 2d slice in case_all_data (2d support)
                if len(case_all_data.shape) == 3:
                    case_all_data = case_all_data[:, None]
                
                assert case_all_data.shape[1] == depth

                # now crop case_all_data to contain just the slice of interest. If we want additional slice above and
                # below the current slice, here is where we get them. We stack those as additional color channels
                slice_data = case_all_data[:, depth_idx]

                # case all data should now be (c, x, y)
                assert len(slice_data.shape) == 3

                if '_u' not in t:
                    image = slice_data[:-1].copy()
                    mask = slice_data[-1:].copy()
                    mask[mask < 0] = 0

                    image, mask = self.preprocess(image, mask)

                    if target_mask[frame_idx] == True:
                        seg[frame_idx] = mask
                        assert frame_idx in labeled_idx
                else:
                    image = slice_data.copy()

                    image, _ = self.preprocess(image, None)
                
                unlabeled[frame_idx] = image

            with torch.no_grad():
                mean_centroid, _ = self.processor.preprocess_no_registration(data=torch.clone(unlabeled)) # T, C(1), H, W

                cropped_unlabeled, padding_need = self.processor.crop_and_pad(data=unlabeled, mean_centroid=mean_centroid)
                padding_need_list.append(padding_need)

                cropped_seg, _ = self.processor.crop_and_pad(data=seg, mean_centroid=mean_centroid)

            cropped_unlabeled = (cropped_unlabeled - cropped_unlabeled.min()) / (cropped_unlabeled.max() - cropped_unlabeled.min() + 1e-8)

            assert torch.count_nonzero(torch.any(~torch.isnan(cropped_seg.reshape(self.video_length, -1)), dim=-1)) == 2
            assert target_mask[0] == True
            assert target_mask[-1] == True
            assert torch.count_nonzero(target_mask) == 2

            if not self.is_val and self.do_data_aug:
                temp_before_augment = torch.clone(cropped_unlabeled).to('cpu')
                
                seed = random.randint(0, 2**32-1)

                for t in range(len(cropped_unlabeled)):
                    cropped_unlabeled[t], cropped_seg[t] = self.augment(image=cropped_unlabeled[t], mask=cropped_seg[t], seed=seed)

            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(3, len(cropped_unlabeled))
            #for t in range(len(cropped_unlabeled)):
            #    ax[0, t].imshow(temp_before_augment[t, 0].cpu(), cmap='gray')
            #    ax[1, t].imshow(cropped_unlabeled[t, 0].cpu(), cmap='gray')
            #    ax[2, t].imshow(cropped_seg[t, 0].cpu(), cmap='gray')
            #plt.show()
                    
            cropped_unlabeled = NormalizeIntensity()(cropped_unlabeled)

            unlabeled_list.append(cropped_unlabeled)
            target_list.append(cropped_seg)
            target_mask_list.append(target_mask)

        unlabeled = torch.stack(unlabeled_list, dim=1) # T, B, 1, H, W
        target = torch.stack(target_list, dim=1) # T, B, 1, H, W
        target_mask = torch.stack(target_mask_list, dim=1) # T, B
        padding_need = torch.stack(padding_need_list, dim=0) # B, 4
        local_distances = torch.stack(local_distances_list, dim=1) # T-1, B

        keys = selected_keys

        #print(seg.shape)
        #print(np.unique(seg))
        #matplotlib.use('QtAgg')
        #fig, ax = plt.subplots(1, 2)
        #ax[0].imshow(labeled[3, 0], cmap='gray')
        #ax[1].imshow(seg[3, 0], cmap='gray')
        #plt.show()

        return {'unlabeled':unlabeled, 
                'target': target,
                'distances': local_distances,
                'in_between': None,
                'padding_need': padding_need,
                'properties': case_properties,
                'target_mask': target_mask}





class DataLoaderAdjacentPair(SlimDataLoaderBase):
    def __init__(self, data, patch_size, final_patch_size, batch_size, video_length, processor, crop_size, is_val, do_data_aug, oversample_foreground_percent=0.0,
                 memmap_mode="r", pseudo_3d_slices=1, pad_mode="edge",
                 pad_kwargs_data=None, pad_sides=None):
        """
        This is the basic data loader for 2D networks. It uses preprocessed data as produced by my (Fabian) preprocessing.
        You can load the data with load_dataset(folder) where folder is the folder where the npz files are located. If there
        are only npz files present in that folder, the data loader will unpack them on the fly. This may take a while
        and increase CPU usage. Therefore, I advise you to call unpack_dataset(folder) first, which will unpack all npz
        to npy. Don't forget to call delete_npy(folder) after you are done with training?
        Why all the hassle? Well the decathlon dataset is huge. Using npy for everything will consume >1 TB and that is uncool
        given that I (Fabian) will have to store that permanently on /datasets and my local computer. With this strategy all
        data is stored in a compressed format (factor 10 smaller) and only unpacked when needed.
        :param data: get this with load_dataset(folder, stage=0). Plug the return value in here and you are g2g (good to go)
        :param patch_size: what patch size will this data loader return? it is common practice to first load larger
        patches so that a central crop after data augmentation can be done to reduce border artifacts. If unsure, use
        get_patch_size() from data_augmentation.default_data_augmentation
        :param final_patch_size: what will the patch finally be cropped to (after data augmentation)? this is the patch
        size that goes into your network. We need this here because we will pad patients in here so that patches at the
        border of patients are sampled properly
        :param batch_size:
        :param num_batches: how many batches will the data loader produce before stopping? None=endless
        :param seed:
        :param stage: ignore this (Fabian only)
        :param transpose: ignore this
        :param random: sample randomly; CAREFUL! non-random sampling requires batch_size=1, otherwise you will iterate batch_size times over the dataset
        :param pseudo_3d_slices: 7 = 3 below and 3 above the center slice
        """
        super(DataLoaderAdjacentPair, self).__init__(data, batch_size, None)
        if pad_kwargs_data is None:
            pad_kwargs_data = OrderedDict()
        self.pad_kwargs_data = pad_kwargs_data
        self.pad_mode = pad_mode
        self.is_val = is_val
        self.crop_size = crop_size
        self.pseudo_3d_slices = pseudo_3d_slices
        self.oversample_foreground_percent = oversample_foreground_percent
        self.final_patch_size = final_patch_size
        self.patch_size = patch_size
        self.list_of_keys = list(self._data.keys())
        self.need_to_pad = np.array(patch_size) - np.array(final_patch_size)
        self.memmap_mode = memmap_mode
        self.video_length = video_length
        self.percent = None
        if pad_sides is not None:
            if not isinstance(pad_sides, np.ndarray):
                pad_sides = np.array(pad_sides)
            self.need_to_pad += pad_sides
        self.pad_sides = pad_sides
        self.data_shape, self.seg_shape = self.determine_shapes()
        self.do_data_aug = do_data_aug

        self.preprocessing_transform = self.set_up_preprocessing_pipeline()
        self.pixel_transform, self.spatial_transform = self.set_up_augmentation_pipeline()

        self.processor = processor
        

    def determine_shapes(self):
        num_seg = 1

        k = list(self._data.keys())[0]
        if isfile(self._data[k]['data_file'][:-4] + ".npy"):
            case_all_data = np.load(self._data[k]['data_file'][:-4] + ".npy", self.memmap_mode)
        else:
            case_all_data = np.load(self._data[k]['data_file'])['data']
        num_color_channels = case_all_data.shape[0] - num_seg
        data_shape = (self.batch_size, num_color_channels, *self.patch_size)
        seg_shape = (self.batch_size, num_seg, *self.patch_size)
        return data_shape, seg_shape

    def get_do_oversample(self, batch_idx):
        return not batch_idx < round(self.batch_size * (1 - self.oversample_foreground_percent))
    
    def select_deep_supervision(self, x):
        if self.deep_supervision:
            return x[0]
        else:
            return x
        

    def my_crop(self, labeled, seg, unlabeled):

        xm = labeled.shape[-1] / 2
        ym = labeled.shape[-2] / 2

        x1 = int(round(xm - self.final_patch_size[1] / 2))
        x2 = int(round(xm + self.final_patch_size[1] / 2))
        y1 = int(round(ym - self.final_patch_size[0] / 2))
        y2 = int(round(ym + self.final_patch_size[0] / 2))

        labeled = labeled[:, :, y1:y2, x1:x2]
        seg = seg[:, :, y1:y2, x1:x2]

        xm = unlabeled.shape[-1] / 2
        ym = unlabeled.shape[-2] / 2

        x1 = int(round(xm - self.final_patch_size[1] / 2))
        x2 = int(round(xm + self.final_patch_size[1] / 2))
        y1 = int(round(ym - self.final_patch_size[0] / 2))
        y2 = int(round(ym + self.final_patch_size[0] / 2))

        unlabeled = unlabeled[:, :, :, y1:y2, x1:x2]

        return labeled, seg, unlabeled
    
    def preprocess(self, image, mask):
        if mask is not None:
            data = {'image': image, 'mask': mask}
            transformed = self.preprocessing_transform(data)
            image = transformed['image']
            mask = transformed['mask']
        else:
            data = {'image': image}
            transformed = self.preprocessing_transform(data)
            image = transformed['image']

        return image, mask
    
    
    def augment(self, image, mask, seed):
        self.pixel_transform = self.pixel_transform.set_random_state(seed=seed)
        self.spatial_transform = self.spatial_transform.set_random_state(seed=seed)
        
        temp = torch.clone(image)
        padding_mask = image == 0

        if mask is not None:
            data = {'image': image, 'mask': mask}
            pixel_transformed = self.pixel_transform(data)
            image = pixel_transformed['image']
            mask = pixel_transformed['mask']
            image[padding_mask] = 0
            data = {'image': image, 'mask': mask}
            spatial_transformed = self.spatial_transform(data)
            image = spatial_transformed['image']
            mask = spatial_transformed['mask']

            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(1, 3)
            #ax[0].imshow(temp[0].cpu(), cmap='gray')
            #ax[1].imshow(image[0].cpu(), cmap='gray')
            #ax[2].imshow(mask[0].cpu(), cmap='gray')
            #plt.show()
        else:
            data = {'image': image}
            pixel_transformed = self.pixel_transform(data)
            image = pixel_transformed['image']
            image[padding_mask] = 0
            data = {'image': image}
            spatial_transformed = self.spatial_transform(data)
            image = spatial_transformed['image']

            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(1, 2)
            #ax[0].imshow(temp[0].cpu(), cmap='gray')
            #ax[1].imshow(image[0].cpu(), cmap='gray')
            #plt.show()

        return image, mask
    
    def set_up_preprocessing_pipeline(self):
        preprocess = Compose([
                            T.SpatialPadd(keys=['image', 'mask'], spatial_size=self.final_patch_size, allow_missing_keys=True),
                            T.CenterSpatialCropd(keys=['image', 'mask'], roi_size=self.final_patch_size, allow_missing_keys=True),
                            T.ToTensor(dtype=torch.float32, device='cuda:0')
                            ])

        return preprocess
    
    def set_up_augmentation_pipeline(self):
        spatial_transform = Compose([
                                    T.RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=0, allow_missing_keys=True),
                                    T.RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=1, allow_missing_keys=True),
                                    T.RandRotated(keys=['image', 'mask'], prob=0.2, range_x=np.pi, mode=["bilinear", "nearest"], padding_mode="zeros", allow_missing_keys=True),
                                    T.RandZoomd(keys=['image', 'mask'], prob=0.2, min_zoom=0.5, max_zoom=1.5, mode=["bilinear", "nearest"], padding_mode="constant", allow_missing_keys=True),
                                    T.RandAffined(keys=['image', 'mask'], prob=0.2, translate_range=(-26, 26), mode=["bilinear", "nearest"], padding_mode="zeros", allow_missing_keys=True)
                                    ])

        pixel_transform = Compose([
                                #RandInvertd(keys=['image'], prob=0.5),
                                T.RandAdjustContrastd(keys=['image'], prob=0.2, gamma=(0.7, 1.5), allow_missing_keys=True),
                                T.RandGaussianNoised(keys=['image'], prob=0.2, std=0.04, allow_missing_keys=True),
                                T.RandScaleIntensityd(keys=['image'], prob=0.2, factors=0.2, allow_missing_keys=True),
                                T.RandGaussianSmoothd(keys=['image'], prob=0.2, sigma_x=(0.25, 0.5), sigma_y=(0.25, 0.5), allow_missing_keys=True),
                                T.RandGaussianSharpend(keys=['image'], prob=0.2, sigma1_x=(0.1, 0.2), sigma1_y=(0.1, 0.2), sigma2_x=(0.2, 0.4), sigma2_y=(0.2, 0.4), alpha=(2.0, 3.0), allow_missing_keys=True)
                                ])

        return pixel_transform, spatial_transform

    def generate_train_batch(self):
        #eval_idx = None
        selected_keys = np.random.choice(self.list_of_keys, self.batch_size, True, None)
        list_of_frames = []
        for i in range(len(selected_keys)):
            patient_id = selected_keys[i][:10]
            filtered = [x for x in self.list_of_keys if x[:10] in patient_id]
            list_of_frames.append(filtered)
        
        target_list = []
        unlabeled_list = []
        padding_need_list = []
        target_mask_list = []
        local_distances_list = []

        case_properties = []
        for j, frames in enumerate(list_of_frames):
            labeled_frame = frames[0]
            if 'properties' in self._data[labeled_frame].keys():
                properties = self._data[labeled_frame]['properties']
            else:
                properties = load_pickle(self._data[labeled_frame]['properties_file'])
            case_properties.append(properties)

            #print(properties['ed_number'])
            #print(np.rint(np.array(properties['ed_number'])))
            #print('******************')

            ed_idx = np.rint(np.array(properties['ed_number'])).astype(np.uint8) % len(frames)
            es_idx = np.rint(np.array(properties['es_number'])).astype(np.uint8) % len(frames)

            depth = properties['size_after_resampling'][0]
            depth_idx = np.random.choice(depth)

            global_labeled_idx = np.array([ed_idx, es_idx])

            frames = sorted(frames, key=lambda x: int(x[16:18]))
            frames = np.array(frames)

            unlabeled = torch.full(size=(3, 1, *self.final_patch_size), fill_value=torch.nan, device='cuda:0')
            seg = torch.full(size=(3, 1, *self.final_patch_size), fill_value=torch.nan, device='cuda:0')
            cropped_unlabeled = torch.full(size=(3, 1, self.crop_size, self.crop_size), fill_value=torch.nan, device='cuda:0')
            cropped_seg = torch.full(size=(3, 1, self.crop_size, self.crop_size), fill_value=torch.nan, device='cuda:0')

            r = np.random.randint(0, len(frames) - 2)

            frame1 = frames[r]
            frame2 = frames[r + 1]
            frame3 = frames[r + 2]
            video = [frame1, frame2, frame3]

            labeled_idx = [0, len(video) - 1]

            for frame_idx, t in enumerate(video):

                if '_u' in t:
                    if not isfile(self.un_data[t]['data_file'][:-4] + ".npy"):
                        # lets hope you know what you're doing
                        case_all_data = np.load(self.un_data[t]['data_file'][:-4] + ".npz")['data']
                    else:
                        case_all_data = np.load(self.un_data[t]['data_file'][:-4] + ".npy", self.memmap_mode)
                else:
                    if not isfile(self._data[t]['data_file'][:-4] + ".npy"):
                        # lets hope you know what you're doing
                        case_all_data = np.load(self._data[t]['data_file'][:-4] + ".npz")['data']
                    else:
                        case_all_data = np.load(self._data[t]['data_file'][:-4] + ".npy", self.memmap_mode)

                # this is for when there is just a 2d slice in case_all_data (2d support)
                if len(case_all_data.shape) == 3:
                    case_all_data = case_all_data[:, None]
                
                assert case_all_data.shape[1] == depth

                # now crop case_all_data to contain just the slice of interest. If we want additional slice above and
                # below the current slice, here is where we get them. We stack those as additional color channels
                slice_data = case_all_data[:, depth_idx]

                # case all data should now be (c, x, y)
                assert len(slice_data.shape) == 3

                if '_u' not in t:
                    image = slice_data[:-1].copy()
                    mask = slice_data[-1:].copy()
                    mask[mask < 0] = 0

                    image, mask = self.preprocess(image, mask)

                else:
                    image = slice_data.copy()

                    image, _ = self.preprocess(image, None)
                
                unlabeled[frame_idx] = image

            with torch.no_grad():
                mean_centroid, _ = self.processor.preprocess_no_registration(data=torch.clone(unlabeled)) # T, C(1), H, W

                cropped_unlabeled, padding_need = self.processor.crop_and_pad(data=unlabeled, mean_centroid=mean_centroid)
                padding_need_list.append(padding_need)

                cropped_seg, _ = self.processor.crop_and_pad(data=seg, mean_centroid=mean_centroid)

            cropped_unlabeled = (cropped_unlabeled - cropped_unlabeled.min()) / (cropped_unlabeled.max() - cropped_unlabeled.min() + 1e-8)

            if not self.is_val and self.do_data_aug:
                temp_before_augment = torch.clone(cropped_unlabeled).to('cpu')
                
                seed = random.randint(0, 2**32-1)

                for t in range(len(cropped_unlabeled)):
                    cropped_unlabeled[t], cropped_seg[t] = self.augment(image=cropped_unlabeled[t], mask=cropped_seg[t], seed=seed)

            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(3, len(cropped_unlabeled))
            #for t in range(len(cropped_unlabeled)):
            #    ax[0, t].imshow(temp_before_augment[t, 0].cpu(), cmap='gray')
            #    ax[1, t].imshow(cropped_unlabeled[t, 0].cpu(), cmap='gray')
            #    ax[2, t].imshow(cropped_seg[t, 0].cpu(), cmap='gray')
            #plt.show()
                    
            cropped_unlabeled = NormalizeIntensity()(cropped_unlabeled)

            unlabeled_list.append(cropped_unlabeled)

        unlabeled = torch.stack(unlabeled_list, dim=1) # T, B, 1, H, W
        padding_need = torch.stack(padding_need_list, dim=0) # B, 4

        keys = selected_keys

        #print(seg.shape)
        #print(np.unique(seg))
        #matplotlib.use('QtAgg')
        #fig, ax = plt.subplots(1, 2)
        #ax[0].imshow(labeled[3, 0], cmap='gray')
        #ax[1].imshow(seg[3, 0], cmap='gray')
        #plt.show()

        return {'x':unlabeled,
                'padding_need': padding_need,
                'properties': case_properties}




class DataLoaderFlowTrain3D(SlimDataLoaderBase):
    def __init__(self, data, patch_size, final_patch_size, batch_size, video_length, processor, crop_size, is_val, do_data_aug, oversample_foreground_percent=0.0,
                 memmap_mode="r", pseudo_3d_slices=1, pad_mode="edge",
                 pad_kwargs_data=None, pad_sides=None):
        """
        This is the basic data loader for 2D networks. It uses preprocessed data as produced by my (Fabian) preprocessing.
        You can load the data with load_dataset(folder) where folder is the folder where the npz files are located. If there
        are only npz files present in that folder, the data loader will unpack them on the fly. This may take a while
        and increase CPU usage. Therefore, I advise you to call unpack_dataset(folder) first, which will unpack all npz
        to npy. Don't forget to call delete_npy(folder) after you are done with training?
        Why all the hassle? Well the decathlon dataset is huge. Using npy for everything will consume >1 TB and that is uncool
        given that I (Fabian) will have to store that permanently on /datasets and my local computer. With this strategy all
        data is stored in a compressed format (factor 10 smaller) and only unpacked when needed.
        :param data: get this with load_dataset(folder, stage=0). Plug the return value in here and you are g2g (good to go)
        :param patch_size: what patch size will this data loader return? it is common practice to first load larger
        patches so that a central crop after data augmentation can be done to reduce border artifacts. If unsure, use
        get_patch_size() from data_augmentation.default_data_augmentation
        :param final_patch_size: what will the patch finally be cropped to (after data augmentation)? this is the patch
        size that goes into your network. We need this here because we will pad patients in here so that patches at the
        border of patients are sampled properly
        :param batch_size:
        :param num_batches: how many batches will the data loader produce before stopping? None=endless
        :param seed:
        :param stage: ignore this (Fabian only)
        :param transpose: ignore this
        :param random: sample randomly; CAREFUL! non-random sampling requires batch_size=1, otherwise you will iterate batch_size times over the dataset
        :param pseudo_3d_slices: 7 = 3 below and 3 above the center slice
        """
        super(DataLoaderFlowTrain3D, self).__init__(data, batch_size, None)
        if pad_kwargs_data is None:
            pad_kwargs_data = OrderedDict()
        self.pad_kwargs_data = pad_kwargs_data
        self.pad_mode = pad_mode
        self.is_val = is_val
        self.crop_size = crop_size
        self.pseudo_3d_slices = pseudo_3d_slices
        self.oversample_foreground_percent = oversample_foreground_percent
        self.final_patch_size = final_patch_size
        self.patch_size = patch_size
        self.list_of_keys = list(self._data.keys())
        self.need_to_pad = np.array(patch_size) - np.array(final_patch_size)
        self.memmap_mode = memmap_mode
        self.video_length = video_length
        self.percent = None
        if pad_sides is not None:
            if not isinstance(pad_sides, np.ndarray):
                pad_sides = np.array(pad_sides)
            self.need_to_pad += pad_sides
        self.pad_sides = pad_sides
        self.data_shape, self.seg_shape = self.determine_shapes()
        self.do_data_aug = do_data_aug

        self.preprocessing_transform = self.set_up_preprocessing_pipeline()
        self.pixel_transform, self.spatial_transform = self.set_up_augmentation_pipeline()

        self.processor = processor
        

    def determine_shapes(self):
        num_seg = 1

        k = list(self._data.keys())[0]
        if isfile(self._data[k]['data_file'][:-4] + ".npy"):
            case_all_data = np.load(self._data[k]['data_file'][:-4] + ".npy", self.memmap_mode)
        else:
            case_all_data = np.load(self._data[k]['data_file'])['data']
        num_color_channels = case_all_data.shape[0] - num_seg
        data_shape = (self.batch_size, num_color_channels, *self.patch_size)
        seg_shape = (self.batch_size, num_seg, *self.patch_size)
        return data_shape, seg_shape

    def get_do_oversample(self, batch_idx):
        return not batch_idx < round(self.batch_size * (1 - self.oversample_foreground_percent))
    
    def select_deep_supervision(self, x):
        if self.deep_supervision:
            return x[0]
        else:
            return x
        

    def my_crop(self, labeled, seg, unlabeled):

        xm = labeled.shape[-1] / 2
        ym = labeled.shape[-2] / 2

        x1 = int(round(xm - self.final_patch_size[1] / 2))
        x2 = int(round(xm + self.final_patch_size[1] / 2))
        y1 = int(round(ym - self.final_patch_size[0] / 2))
        y2 = int(round(ym + self.final_patch_size[0] / 2))

        labeled = labeled[:, :, y1:y2, x1:x2]
        seg = seg[:, :, y1:y2, x1:x2]

        xm = unlabeled.shape[-1] / 2
        ym = unlabeled.shape[-2] / 2

        x1 = int(round(xm - self.final_patch_size[1] / 2))
        x2 = int(round(xm + self.final_patch_size[1] / 2))
        y1 = int(round(ym - self.final_patch_size[0] / 2))
        y2 = int(round(ym + self.final_patch_size[0] / 2))

        unlabeled = unlabeled[:, :, :, y1:y2, x1:x2]

        return labeled, seg, unlabeled
    
    def preprocess(self, image, mask):
        if mask is not None:
            data = {'image': image, 'mask': mask}
            transformed = self.preprocessing_transform(data)
            image = transformed['image']
            mask = transformed['mask']
        else:
            data = {'image': image}
            transformed = self.preprocessing_transform(data)
            image = transformed['image']

        return image, mask
    
    
    def augment(self, image, mask, seed):
        self.pixel_transform = self.pixel_transform.set_random_state(seed=seed)
        self.spatial_transform = self.spatial_transform.set_random_state(seed=seed)
        
        temp = torch.clone(image)
        padding_mask = image == 0

        if mask is not None:
            data = {'image': image, 'mask': mask}
            pixel_transformed = self.pixel_transform(data)
            image = pixel_transformed['image']
            mask = pixel_transformed['mask']
            image[padding_mask] = 0
            data = {'image': image, 'mask': mask}
            spatial_transformed = self.spatial_transform(data)
            image = spatial_transformed['image']
            mask = spatial_transformed['mask']

            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(1, 3)
            #ax[0].imshow(temp[0].cpu(), cmap='gray')
            #ax[1].imshow(image[0].cpu(), cmap='gray')
            #ax[2].imshow(mask[0].cpu(), cmap='gray')
            #plt.show()
        else:
            data = {'image': image}
            pixel_transformed = self.pixel_transform(data)
            image = pixel_transformed['image']
            image[padding_mask] = 0
            data = {'image': image}
            spatial_transformed = self.spatial_transform(data)
            image = spatial_transformed['image']

            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(1, 2)
            #ax[0].imshow(temp[0].cpu(), cmap='gray')
            #ax[1].imshow(image[0].cpu(), cmap='gray')
            #plt.show()

        return image, mask
    
    def set_up_preprocessing_pipeline(self):
        preprocess = Compose([
                            T.SpatialPadd(keys=['image', 'mask'], spatial_size=self.final_patch_size, allow_missing_keys=True),
                            T.CenterSpatialCropd(keys=['image', 'mask'], roi_size=self.final_patch_size, allow_missing_keys=True),
                            T.ToTensor(dtype=torch.float32, device='cuda:0')
                            ])

        return preprocess
    
    def set_up_augmentation_pipeline(self):
        spatial_transform = Compose([
                                    T.RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=0, allow_missing_keys=True),
                                    T.RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=1, allow_missing_keys=True),
                                    T.RandRotated(keys=['image', 'mask'], prob=0.2, range_x=np.pi, mode=["bilinear", "nearest"], padding_mode="zeros", allow_missing_keys=True),
                                    T.RandZoomd(keys=['image', 'mask'], prob=0.2, min_zoom=0.5, max_zoom=1.5, mode=["bilinear", "nearest"], padding_mode="constant", allow_missing_keys=True),
                                    T.RandAffined(keys=['image', 'mask'], prob=0.2, translate_range=(-26, 26), mode=["bilinear", "nearest"], padding_mode="zeros", allow_missing_keys=True)
                                    ])

        pixel_transform = Compose([
                                #RandInvertd(keys=['image'], prob=0.5),
                                T.RandAdjustContrastd(keys=['image'], prob=0.2, gamma=(0.7, 1.5), allow_missing_keys=True),
                                T.RandGaussianNoised(keys=['image'], prob=0.2, std=0.04, allow_missing_keys=True),
                                T.RandScaleIntensityd(keys=['image'], prob=0.2, factors=0.2, allow_missing_keys=True),
                                T.RandGaussianSmoothd(keys=['image'], prob=0.2, sigma_x=(0.25, 0.5), sigma_y=(0.25, 0.5), allow_missing_keys=True),
                                T.RandGaussianSharpend(keys=['image'], prob=0.2, sigma1_x=(0.1, 0.2), sigma1_y=(0.1, 0.2), sigma2_x=(0.2, 0.4), sigma2_y=(0.2, 0.4), alpha=(2.0, 3.0), allow_missing_keys=True)
                                ])

        return pixel_transform, spatial_transform

    def generate_train_batch(self):
        #eval_idx = None
        selected_keys = np.random.choice(self.list_of_keys, self.batch_size, True, None)
        list_of_frames = []
        for i in range(len(selected_keys)):
            patient_id = selected_keys[i][:10]
            filtered = [x for x in self.list_of_keys if x[:10] in patient_id]
            list_of_frames.append(filtered)
        
        target_list = []
        unlabeled_list = []
        padding_need_list = []
        target_mask_list = []
        local_distances_list = []

        case_properties = []
        for j, frames in enumerate(list_of_frames):
            labeled_frame = frames[0]
            if 'properties' in self._data[labeled_frame].keys():
                properties = self._data[labeled_frame]['properties']
            else:
                properties = load_pickle(self._data[labeled_frame]['properties_file'])
            case_properties.append(properties)

            #print(properties['ed_number'])
            #print(np.rint(np.array(properties['ed_number'])))
            #print('******************')

            ed_idx = np.rint(np.array(properties['ed_number'])).astype(np.uint8) % len(frames)
            es_idx = np.rint(np.array(properties['es_number'])).astype(np.uint8) % len(frames)

            depth = properties['size_after_resampling'][0]
            depth_idx = np.random.choice(depth)

            global_labeled_idx = np.array([ed_idx, es_idx])

            frames = sorted(frames, key=lambda x: int(x[16:18]))
            frames = np.array(frames)

            unlabeled = torch.full(size=(self.video_length, 1, *self.final_patch_size), fill_value=torch.nan, device='cuda:0')
            seg = torch.full(size=(self.video_length, 1, *self.final_patch_size), fill_value=torch.nan, device='cuda:0')
            cropped_unlabeled = torch.full(size=(self.video_length, 1, self.crop_size, self.crop_size), fill_value=torch.nan, device='cuda:0')
            cropped_seg = torch.full(size=(self.video_length, 1, self.crop_size, self.crop_size), fill_value=torch.nan, device='cuda:0')

            possible_indices = np.arange(len(frames))

            before_where = np.argwhere(possible_indices < ed_idx).reshape(-1,)
            after_where = np.argwhere(possible_indices >= ed_idx).reshape(-1,)

            before_indices = possible_indices[before_where]
            after_indices = possible_indices[after_where]

            possible_indices = np.concatenate([after_indices, before_indices])

            stop_idx = np.argwhere(possible_indices == es_idx)[0][0]
            chunk1 = possible_indices[:stop_idx + 1]
            chunk2 = possible_indices[stop_idx:]
            chunk2 = np.concatenate([np.array([possible_indices[0]]), chunk2[::-1]])

            r = np.random.randint(0, 2)
            if r == 0:
                possible_indices = chunk1
            elif r == 1:
                possible_indices = chunk2
            
            assert possible_indices[0] == ed_idx
            assert possible_indices[-1] == es_idx

            random_indices = np.random.choice(np.arange(0, len(possible_indices)), size=self.video_length - 2)
            target_mask = np.concatenate([np.array([True]), np.full_like(random_indices, fill_value=False), np.array([True])]).astype(bool)
            random_indices = np.concatenate([np.array([0]), random_indices, np.array([len(possible_indices) - 1])])
            sorted_indices = np.argsort(random_indices)

            random_indices = random_indices[sorted_indices]
            target_mask = target_mask[sorted_indices]
            target_mask = torch.from_numpy(target_mask).to('cuda:0')

            diff = np.diff(random_indices)
            diff = diff / len(frames)
            local_distances = torch.from_numpy(diff).float().to('cuda:0')
            assert len(local_distances) == len(random_indices) - 1
            local_distances_list.append(local_distances)

            frame_indices = possible_indices[random_indices]

            assert len(frame_indices) == self.video_length
            video = frames[frame_indices]


            labeled_idx = np.where(np.isin(frame_indices, global_labeled_idx))[0]

            for frame_idx, t in enumerate(video):

                if '_u' in t:
                    if not isfile(self.un_data[t]['data_file'][:-4] + ".npy"):
                        # lets hope you know what you're doing
                        case_all_data = np.load(self.un_data[t]['data_file'][:-4] + ".npz")['data']
                    else:
                        case_all_data = np.load(self.un_data[t]['data_file'][:-4] + ".npy", self.memmap_mode)
                else:
                    if not isfile(self._data[t]['data_file'][:-4] + ".npy"):
                        # lets hope you know what you're doing
                        case_all_data = np.load(self._data[t]['data_file'][:-4] + ".npz")['data']
                    else:
                        case_all_data = np.load(self._data[t]['data_file'][:-4] + ".npy", self.memmap_mode)

                # this is for when there is just a 2d slice in case_all_data (2d support)
                if len(case_all_data.shape) == 3:
                    case_all_data = case_all_data[:, None]
                
                assert case_all_data.shape[1] == depth

                # now crop case_all_data to contain just the slice of interest. If we want additional slice above and
                # below the current slice, here is where we get them. We stack those as additional color channels
                slice_data = case_all_data[:, depth_idx]

                # case all data should now be (c, x, y)
                assert len(slice_data.shape) == 3

                if '_u' not in t:
                    image = slice_data[:-1].copy()
                    mask = slice_data[-1:].copy()
                    mask[mask < 0] = 0

                    image, mask = self.preprocess(image, mask)

                    if target_mask[frame_idx] == True:
                        seg[frame_idx] = mask
                        assert frame_idx in labeled_idx
                else:
                    image = slice_data.copy()

                    image, _ = self.preprocess(image, None)
                
                unlabeled[frame_idx] = image

            with torch.no_grad():
                mean_centroid, _ = self.processor.preprocess_no_registration(data=torch.clone(unlabeled)) # T, C(1), H, W

                cropped_unlabeled, padding_need = self.processor.crop_and_pad(data=unlabeled, mean_centroid=mean_centroid)
                padding_need_list.append(padding_need)

                cropped_seg, _ = self.processor.crop_and_pad(data=seg, mean_centroid=mean_centroid)

            cropped_unlabeled = (cropped_unlabeled - cropped_unlabeled.min()) / (cropped_unlabeled.max() - cropped_unlabeled.min() + 1e-8)

            assert torch.count_nonzero(torch.any(~torch.isnan(cropped_seg.reshape(self.video_length, -1)), dim=-1)) == 2
            assert target_mask[0] == True
            assert target_mask[-1] == True
            assert torch.count_nonzero(target_mask) == 2

            if not self.is_val and self.do_data_aug:
                temp_before_augment = torch.clone(cropped_unlabeled).to('cpu')
                
                seed = random.randint(0, 2**32-1)

                for t in range(len(cropped_unlabeled)):
                    cropped_unlabeled[t], cropped_seg[t] = self.augment(image=cropped_unlabeled[t], mask=cropped_seg[t], seed=seed)

            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(3, len(cropped_unlabeled))
            #for t in range(len(cropped_unlabeled)):
            #    ax[0, t].imshow(temp_before_augment[t, 0].cpu(), cmap='gray')
            #    ax[1, t].imshow(cropped_unlabeled[t, 0].cpu(), cmap='gray')
            #    ax[2, t].imshow(cropped_seg[t, 0].cpu(), cmap='gray')
            #plt.show()
                    
            cropped_unlabeled = NormalizeIntensity()(cropped_unlabeled)

            unlabeled_list.append(cropped_unlabeled)
            target_list.append(cropped_seg)
            target_mask_list.append(target_mask)

        unlabeled = torch.stack(unlabeled_list, dim=1) # T, B, 1, H, W
        target = torch.stack(target_list, dim=1) # T, B, 1, H, W
        target_mask = torch.stack(target_mask_list, dim=1) # T, B
        padding_need = torch.stack(padding_need_list, dim=0) # B, 4
        local_distances = torch.stack(local_distances_list, dim=1) # T-1, B

        keys = selected_keys

        #print(seg.shape)
        #print(np.unique(seg))
        #matplotlib.use('QtAgg')
        #fig, ax = plt.subplots(1, 2)
        #ax[0].imshow(labeled[3, 0], cmap='gray')
        #ax[1].imshow(seg[3, 0], cmap='gray')
        #plt.show()

        return {'unlabeled':unlabeled, 
                'target': target,
                'distances': local_distances,
                'in_between': None,
                'padding_need': padding_need,
                'properties': case_properties,
                'target_mask': target_mask}





class DataLoaderFlowTrain3DRegular(SlimDataLoaderBase):
    def __init__(self, data, patch_size, final_patch_size, batch_size, video_length, processor, crop_size, is_val, do_data_aug, oversample_foreground_percent=0.0,
                 memmap_mode="r", pseudo_3d_slices=1, pad_mode="edge",
                 pad_kwargs_data=None, pad_sides=None):
        """
        This is the basic data loader for 2D networks. It uses preprocessed data as produced by my (Fabian) preprocessing.
        You can load the data with load_dataset(folder) where folder is the folder where the npz files are located. If there
        are only npz files present in that folder, the data loader will unpack them on the fly. This may take a while
        and increase CPU usage. Therefore, I advise you to call unpack_dataset(folder) first, which will unpack all npz
        to npy. Don't forget to call delete_npy(folder) after you are done with training?
        Why all the hassle? Well the decathlon dataset is huge. Using npy for everything will consume >1 TB and that is uncool
        given that I (Fabian) will have to store that permanently on /datasets and my local computer. With this strategy all
        data is stored in a compressed format (factor 10 smaller) and only unpacked when needed.
        :param data: get this with load_dataset(folder, stage=0). Plug the return value in here and you are g2g (good to go)
        :param patch_size: what patch size will this data loader return? it is common practice to first load larger
        patches so that a central crop after data augmentation can be done to reduce border artifacts. If unsure, use
        get_patch_size() from data_augmentation.default_data_augmentation
        :param final_patch_size: what will the patch finally be cropped to (after data augmentation)? this is the patch
        size that goes into your network. We need this here because we will pad patients in here so that patches at the
        border of patients are sampled properly
        :param batch_size:
        :param num_batches: how many batches will the data loader produce before stopping? None=endless
        :param seed:
        :param stage: ignore this (Fabian only)
        :param transpose: ignore this
        :param random: sample randomly; CAREFUL! non-random sampling requires batch_size=1, otherwise you will iterate batch_size times over the dataset
        :param pseudo_3d_slices: 7 = 3 below and 3 above the center slice
        """
        super(DataLoaderFlowTrain3DRegular, self).__init__(data, batch_size, None)
        if pad_kwargs_data is None:
            pad_kwargs_data = OrderedDict()
        self.pad_kwargs_data = pad_kwargs_data
        self.pad_mode = pad_mode
        self.is_val = is_val
        self.crop_size = crop_size
        self.pseudo_3d_slices = pseudo_3d_slices
        self.oversample_foreground_percent = oversample_foreground_percent
        self.final_patch_size = final_patch_size
        self.patch_size = patch_size
        self.list_of_keys = list(self._data.keys())
        self.need_to_pad = np.array(patch_size) - np.array(final_patch_size)
        self.memmap_mode = memmap_mode
        self.video_length = video_length
        self.percent = None
        if pad_sides is not None:
            if not isinstance(pad_sides, np.ndarray):
                pad_sides = np.array(pad_sides)
            self.need_to_pad += pad_sides
        self.pad_sides = pad_sides
        self.data_shape, self.seg_shape = self.determine_shapes()
        self.do_data_aug = do_data_aug

        self.preprocessing_transform = self.set_up_preprocessing_pipeline()
        self.pixel_transform, self.spatial_transform = self.set_up_augmentation_pipeline()

        self.processor = processor
        

    def determine_shapes(self):
        num_seg = 1

        k = list(self._data.keys())[0]
        if isfile(self._data[k]['data_file'][:-4] + ".npy"):
            case_all_data = np.load(self._data[k]['data_file'][:-4] + ".npy", self.memmap_mode)
        else:
            case_all_data = np.load(self._data[k]['data_file'])['data']
        num_color_channels = case_all_data.shape[0] - num_seg
        data_shape = (self.batch_size, num_color_channels, *self.patch_size)
        seg_shape = (self.batch_size, num_seg, *self.patch_size)
        return data_shape, seg_shape

    def get_do_oversample(self, batch_idx):
        return not batch_idx < round(self.batch_size * (1 - self.oversample_foreground_percent))
    
    def select_deep_supervision(self, x):
        if self.deep_supervision:
            return x[0]
        else:
            return x
        

    def my_crop(self, labeled, seg, unlabeled):

        xm = labeled.shape[-1] / 2
        ym = labeled.shape[-2] / 2

        x1 = int(round(xm - self.final_patch_size[1] / 2))
        x2 = int(round(xm + self.final_patch_size[1] / 2))
        y1 = int(round(ym - self.final_patch_size[0] / 2))
        y2 = int(round(ym + self.final_patch_size[0] / 2))

        labeled = labeled[:, :, y1:y2, x1:x2]
        seg = seg[:, :, y1:y2, x1:x2]

        xm = unlabeled.shape[-1] / 2
        ym = unlabeled.shape[-2] / 2

        x1 = int(round(xm - self.final_patch_size[1] / 2))
        x2 = int(round(xm + self.final_patch_size[1] / 2))
        y1 = int(round(ym - self.final_patch_size[0] / 2))
        y2 = int(round(ym + self.final_patch_size[0] / 2))

        unlabeled = unlabeled[:, :, :, y1:y2, x1:x2]

        return labeled, seg, unlabeled
    
    def preprocess(self, image, mask):
        if mask is not None:
            data = {'image': image, 'mask': mask}
            transformed = self.preprocessing_transform(data)
            image = transformed['image']
            mask = transformed['mask']
        else:
            data = {'image': image}
            transformed = self.preprocessing_transform(data)
            image = transformed['image']

        return image, mask
    
    
    def augment(self, image, mask, seed):
        self.pixel_transform = self.pixel_transform.set_random_state(seed=seed)
        self.spatial_transform = self.spatial_transform.set_random_state(seed=seed)
        
        temp = torch.clone(image)
        padding_mask = image == 0

        if mask is not None:
            data = {'image': image, 'mask': mask}
            pixel_transformed = self.pixel_transform(data)
            image = pixel_transformed['image']
            mask = pixel_transformed['mask']
            image[padding_mask] = 0
            data = {'image': image, 'mask': mask}
            spatial_transformed = self.spatial_transform(data)
            image = spatial_transformed['image']
            mask = spatial_transformed['mask']

            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(1, 3)
            #ax[0].imshow(temp[0].cpu(), cmap='gray')
            #ax[1].imshow(image[0].cpu(), cmap='gray')
            #ax[2].imshow(mask[0].cpu(), cmap='gray')
            #plt.show()
        else:
            data = {'image': image}
            pixel_transformed = self.pixel_transform(data)
            image = pixel_transformed['image']
            image[padding_mask] = 0
            data = {'image': image}
            spatial_transformed = self.spatial_transform(data)
            image = spatial_transformed['image']

            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(1, 2)
            #ax[0].imshow(temp[0].cpu(), cmap='gray')
            #ax[1].imshow(image[0].cpu(), cmap='gray')
            #plt.show()

        return image, mask
    
    def set_up_preprocessing_pipeline(self):
        preprocess = Compose([
                            T.SpatialPadd(keys=['image', 'mask'], spatial_size=self.final_patch_size, allow_missing_keys=True),
                            T.CenterSpatialCropd(keys=['image', 'mask'], roi_size=self.final_patch_size, allow_missing_keys=True),
                            T.ToTensor(dtype=torch.float32, device='cuda:0')
                            ])

        return preprocess
    
    def set_up_augmentation_pipeline(self):
        spatial_transform = Compose([
                                    T.RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=0, allow_missing_keys=True),
                                    T.RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=1, allow_missing_keys=True),
                                    T.RandRotated(keys=['image', 'mask'], prob=0.2, range_x=np.pi, mode=["bilinear", "nearest"], padding_mode="zeros", allow_missing_keys=True),
                                    T.RandZoomd(keys=['image', 'mask'], prob=0.2, min_zoom=0.5, max_zoom=1.5, mode=["bilinear", "nearest"], padding_mode="constant", allow_missing_keys=True),
                                    T.RandAffined(keys=['image', 'mask'], prob=0.2, translate_range=(-26, 26), mode=["bilinear", "nearest"], padding_mode="zeros", allow_missing_keys=True)
                                    ])

        pixel_transform = Compose([
                                #RandInvertd(keys=['image'], prob=0.5),
                                T.RandAdjustContrastd(keys=['image'], prob=0.2, gamma=(0.7, 1.5), allow_missing_keys=True),
                                T.RandGaussianNoised(keys=['image'], prob=0.2, std=0.04, allow_missing_keys=True),
                                T.RandScaleIntensityd(keys=['image'], prob=0.2, factors=0.2, allow_missing_keys=True),
                                T.RandGaussianSmoothd(keys=['image'], prob=0.2, sigma_x=(0.25, 0.5), sigma_y=(0.25, 0.5), allow_missing_keys=True),
                                T.RandGaussianSharpend(keys=['image'], prob=0.2, sigma1_x=(0.1, 0.2), sigma1_y=(0.1, 0.2), sigma2_x=(0.2, 0.4), sigma2_y=(0.2, 0.4), alpha=(2.0, 3.0), allow_missing_keys=True)
                                ])

        return pixel_transform, spatial_transform

    def generate_train_batch(self):
        #eval_idx = None
        selected_keys = np.random.choice(self.list_of_keys, self.batch_size, True, None)
        list_of_frames = []
        for i in range(len(selected_keys)):
            patient_id = selected_keys[i][:10]
            filtered = [x for x in self.list_of_keys if x[:10] in patient_id]
            list_of_frames.append(filtered)
        
        target_list = []
        unlabeled_list = []
        padding_need_list = []
        target_mask_list = []

        case_properties = []
        for j, frames in enumerate(list_of_frames):
            labeled_frame = frames[0]
            if 'properties' in self._data[labeled_frame].keys():
                properties = self._data[labeled_frame]['properties']
            else:
                properties = load_pickle(self._data[labeled_frame]['properties_file'])
            case_properties.append(properties)

            #print(properties['ed_number'])
            #print(np.rint(np.array(properties['ed_number'])))
            #print('******************')

            ed_idx = np.rint(np.array(properties['ed_number'])).astype(np.uint8) % len(frames)
            es_idx = np.rint(np.array(properties['es_number'])).astype(np.uint8) % len(frames)

            depth = properties['size_after_resampling'][0]
            depth_idx = np.random.choice(depth)

            global_labeled_idx = np.array([ed_idx, es_idx])

            frames = sorted(frames, key=lambda x: int(x[16:18]))
            frames = np.array(frames)

            unlabeled = torch.full(size=(self.video_length, 1, *self.final_patch_size), fill_value=torch.nan, device='cuda:0')
            seg = torch.full(size=(self.video_length, 1, *self.final_patch_size), fill_value=torch.nan, device='cuda:0')
            cropped_unlabeled = torch.full(size=(self.video_length, 1, self.crop_size, self.crop_size), fill_value=torch.nan, device='cuda:0')
            cropped_seg = torch.full(size=(self.video_length, 1, self.crop_size, self.crop_size), fill_value=torch.nan, device='cuda:0')

            possible_indices = np.arange(len(frames))

            before_where = np.argwhere(possible_indices < ed_idx).reshape(-1,)
            after_where = np.argwhere(possible_indices >= ed_idx).reshape(-1,)

            before_indices = possible_indices[before_where]
            after_indices = possible_indices[after_where]

            possible_indices = np.concatenate([after_indices, before_indices])

            stop_idx = np.argwhere(possible_indices == es_idx)[0][0]
            chunk1 = possible_indices[:stop_idx + 1]
            chunk2 = possible_indices[stop_idx:]
            chunk2 = np.concatenate([np.array([possible_indices[0]]), chunk2[::-1]])

            r = np.random.randint(0, 2)
            if r == 0:
                possible_indices = chunk1
            elif r == 1:
                possible_indices = chunk2
            
            assert possible_indices[0] == ed_idx
            assert possible_indices[-1] == es_idx

            possible_indices = np.pad(possible_indices, pad_width=(1000, 1000), mode='wrap')
            start = np.random.randint(0, len(possible_indices) - self.video_length)
            random_indices = possible_indices[start:start + self.video_length]

            target_mask = np.full_like(random_indices, fill_value=False)
            sorted_indices = np.argsort(random_indices)

            random_indices = random_indices[sorted_indices]
            target_mask = torch.from_numpy(target_mask).to('cuda:0')

            frame_indices = possible_indices[random_indices]

            assert len(frame_indices) == self.video_length
            video = frames[frame_indices]


            labeled_idx = np.where(np.isin(frame_indices, global_labeled_idx))[0]

            for frame_idx, t in enumerate(video):

                if '_u' in t:
                    if not isfile(self.un_data[t]['data_file'][:-4] + ".npy"):
                        # lets hope you know what you're doing
                        case_all_data = np.load(self.un_data[t]['data_file'][:-4] + ".npz")['data']
                    else:
                        case_all_data = np.load(self.un_data[t]['data_file'][:-4] + ".npy", self.memmap_mode)
                else:
                    if not isfile(self._data[t]['data_file'][:-4] + ".npy"):
                        # lets hope you know what you're doing
                        case_all_data = np.load(self._data[t]['data_file'][:-4] + ".npz")['data']
                    else:
                        case_all_data = np.load(self._data[t]['data_file'][:-4] + ".npy", self.memmap_mode)

                # this is for when there is just a 2d slice in case_all_data (2d support)
                if len(case_all_data.shape) == 3:
                    case_all_data = case_all_data[:, None]
                
                assert case_all_data.shape[1] == depth

                # now crop case_all_data to contain just the slice of interest. If we want additional slice above and
                # below the current slice, here is where we get them. We stack those as additional color channels
                slice_data = case_all_data[:, depth_idx]

                # case all data should now be (c, x, y)
                assert len(slice_data.shape) == 3

                if '_u' not in t:
                    image = slice_data[:-1].copy()
                    mask = slice_data[-1:].copy()
                    mask[mask < 0] = 0

                    image, mask = self.preprocess(image, mask)

                    if target_mask[frame_idx] == True:
                        seg[frame_idx] = mask
                        assert frame_idx in labeled_idx
                else:
                    image = slice_data.copy()

                    image, _ = self.preprocess(image, None)
                
                unlabeled[frame_idx] = image

            with torch.no_grad():
                mean_centroid, _ = self.processor.preprocess_no_registration(data=torch.clone(unlabeled)) # T, C(1), H, W

                cropped_unlabeled, padding_need = self.processor.crop_and_pad(data=unlabeled, mean_centroid=mean_centroid)
                padding_need_list.append(padding_need)

                cropped_seg, _ = self.processor.crop_and_pad(data=seg, mean_centroid=mean_centroid)

            cropped_unlabeled = (cropped_unlabeled - cropped_unlabeled.min()) / (cropped_unlabeled.max() - cropped_unlabeled.min() + 1e-8)

            if not self.is_val and self.do_data_aug:
                temp_before_augment = torch.clone(cropped_unlabeled).to('cpu')
                
                seed = random.randint(0, 2**32-1)

                for t in range(len(cropped_unlabeled)):
                    cropped_unlabeled[t], cropped_seg[t] = self.augment(image=cropped_unlabeled[t], mask=cropped_seg[t], seed=seed)

            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(3, len(cropped_unlabeled))
            #for t in range(len(cropped_unlabeled)):
            #    ax[0, t].imshow(temp_before_augment[t, 0].cpu(), cmap='gray')
            #    ax[1, t].imshow(cropped_unlabeled[t, 0].cpu(), cmap='gray')
            #    ax[2, t].imshow(cropped_seg[t, 0].cpu(), cmap='gray')
            #plt.show()
                    
            cropped_unlabeled = NormalizeIntensity()(cropped_unlabeled)

            unlabeled_list.append(cropped_unlabeled)
            target_list.append(cropped_seg)
            target_mask_list.append(target_mask)

        unlabeled = torch.stack(unlabeled_list, dim=1) # T, B, 1, H, W
        target = torch.stack(target_list, dim=1) # T, B, 1, H, W
        target_mask = torch.stack(target_mask_list, dim=1) # T, B
        padding_need = torch.stack(padding_need_list, dim=0) # B, 4

        keys = selected_keys

        #print(seg.shape)
        #print(np.unique(seg))
        #matplotlib.use('QtAgg')
        #fig, ax = plt.subplots(1, 2)
        #ax[0].imshow(labeled[3, 0], cmap='gray')
        #ax[1].imshow(seg[3, 0], cmap='gray')
        #plt.show()

        return {'unlabeled':unlabeled, 
                'target': target,
                'in_between': None,
                'padding_need': padding_need,
                'properties': case_properties,
                'target_mask': target_mask}
    



class DataLoaderFlowTrain5LibRegular(SlimDataLoaderBase):
    def __init__(self, data, patch_size, final_patch_size, batch_size, video_length, processor, crop_size, is_val, do_data_aug, oversample_foreground_percent=0.0,
                 memmap_mode="r", pseudo_3d_slices=1, pad_mode="edge",
                 pad_kwargs_data=None, pad_sides=None):
        """
        This is the basic data loader for 2D networks. It uses preprocessed data as produced by my (Fabian) preprocessing.
        You can load the data with load_dataset(folder) where folder is the folder where the npz files are located. If there
        are only npz files present in that folder, the data loader will unpack them on the fly. This may take a while
        and increase CPU usage. Therefore, I advise you to call unpack_dataset(folder) first, which will unpack all npz
        to npy. Don't forget to call delete_npy(folder) after you are done with training?
        Why all the hassle? Well the decathlon dataset is huge. Using npy for everything will consume >1 TB and that is uncool
        given that I (Fabian) will have to store that permanently on /datasets and my local computer. With this strategy all
        data is stored in a compressed format (factor 10 smaller) and only unpacked when needed.
        :param data: get this with load_dataset(folder, stage=0). Plug the return value in here and you are g2g (good to go)
        :param patch_size: what patch size will this data loader return? it is common practice to first load larger
        patches so that a central crop after data augmentation can be done to reduce border artifacts. If unsure, use
        get_patch_size() from data_augmentation.default_data_augmentation
        :param final_patch_size: what will the patch finally be cropped to (after data augmentation)? this is the patch
        size that goes into your network. We need this here because we will pad patients in here so that patches at the
        border of patients are sampled properly
        :param batch_size:
        :param num_batches: how many batches will the data loader produce before stopping? None=endless
        :param seed:
        :param stage: ignore this (Fabian only)
        :param transpose: ignore this
        :param random: sample randomly; CAREFUL! non-random sampling requires batch_size=1, otherwise you will iterate batch_size times over the dataset
        :param pseudo_3d_slices: 7 = 3 below and 3 above the center slice
        """
        super(DataLoaderFlowTrain5LibRegular, self).__init__(data, batch_size, None)
        if pad_kwargs_data is None:
            pad_kwargs_data = OrderedDict()
        self.pad_kwargs_data = pad_kwargs_data
        self.pad_mode = pad_mode
        self.is_val = is_val
        self.crop_size = crop_size
        self.pseudo_3d_slices = pseudo_3d_slices
        self.oversample_foreground_percent = oversample_foreground_percent
        self.final_patch_size = final_patch_size
        self.patch_size = patch_size
        self.list_of_keys = list(self._data.keys())
        self.need_to_pad = np.array(patch_size) - np.array(final_patch_size)
        self.memmap_mode = memmap_mode
        self.video_length = video_length
        self.percent = None
        if pad_sides is not None:
            if not isinstance(pad_sides, np.ndarray):
                pad_sides = np.array(pad_sides)
            self.need_to_pad += pad_sides
        self.pad_sides = pad_sides
        self.data_shape, self.seg_shape = self.determine_shapes()
        self.do_data_aug = do_data_aug

        self.preprocessing_transform = self.set_up_preprocessing_pipeline()
        self.pixel_transform, self.spatial_transform = self.set_up_augmentation_pipeline()

        self.processor = processor
        

    def determine_shapes(self):
        num_seg = 1

        k = list(self._data.keys())[0]
        if isfile(self._data[k]['data_file'][:-4] + ".npy"):
            case_all_data = np.load(self._data[k]['data_file'][:-4] + ".npy", self.memmap_mode)
        else:
            case_all_data = np.load(self._data[k]['data_file'])['data']
        num_color_channels = case_all_data.shape[0] - num_seg
        data_shape = (self.batch_size, num_color_channels, *self.patch_size)
        seg_shape = (self.batch_size, num_seg, *self.patch_size)
        return data_shape, seg_shape

    def get_do_oversample(self, batch_idx):
        return not batch_idx < round(self.batch_size * (1 - self.oversample_foreground_percent))
    
    def select_deep_supervision(self, x):
        if self.deep_supervision:
            return x[0]
        else:
            return x
        

    def my_crop(self, labeled, seg, unlabeled):

        xm = labeled.shape[-1] / 2
        ym = labeled.shape[-2] / 2

        x1 = int(round(xm - self.final_patch_size[1] / 2))
        x2 = int(round(xm + self.final_patch_size[1] / 2))
        y1 = int(round(ym - self.final_patch_size[0] / 2))
        y2 = int(round(ym + self.final_patch_size[0] / 2))

        labeled = labeled[:, :, y1:y2, x1:x2]
        seg = seg[:, :, y1:y2, x1:x2]

        xm = unlabeled.shape[-1] / 2
        ym = unlabeled.shape[-2] / 2

        x1 = int(round(xm - self.final_patch_size[1] / 2))
        x2 = int(round(xm + self.final_patch_size[1] / 2))
        y1 = int(round(ym - self.final_patch_size[0] / 2))
        y2 = int(round(ym + self.final_patch_size[0] / 2))

        unlabeled = unlabeled[:, :, :, y1:y2, x1:x2]

        return labeled, seg, unlabeled
    
    def preprocess(self, image, mask):
        if mask is not None:
            data = {'image': image, 'mask': mask}
            transformed = self.preprocessing_transform(data)
            image = transformed['image']
            mask = transformed['mask']
        else:
            data = {'image': image}
            transformed = self.preprocessing_transform(data)
            image = transformed['image']

        return image, mask
    
    
    def augment(self, image, mask, pixel_transform, spatial_transform):
        
        temp = torch.clone(image)
        padding_mask = image == 0

        if mask is not None:
            data = {'image': image, 'mask': mask}
            pixel_transformed = pixel_transform(data)
            image = pixel_transformed['image']
            mask = pixel_transformed['mask']
            image[padding_mask] = 0
            data = {'image': image, 'mask': mask}
            spatial_transformed = spatial_transform(data)
            image = spatial_transformed['image']
            mask = spatial_transformed['mask']

            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(1, 3)
            #ax[0].imshow(temp[0].cpu(), cmap='gray')
            #ax[1].imshow(image[0].cpu(), cmap='gray')
            #ax[2].imshow(mask[0].cpu(), cmap='gray')
            #plt.show()
        else:
            data = {'image': image}
            pixel_transformed = pixel_transform(data)
            image = pixel_transformed['image']
            image[padding_mask] = 0
            data = {'image': image}
            spatial_transformed = spatial_transform(data)
            image = spatial_transformed['image']

            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(1, 2)
            #ax[0].imshow(temp[0].cpu(), cmap='gray')
            #ax[1].imshow(image[0].cpu(), cmap='gray')
            #plt.show()

        return image, mask
    
    def set_up_preprocessing_pipeline(self):
        preprocess = Compose([
                            T.SpatialPadd(keys=['image', 'mask'], spatial_size=self.final_patch_size, allow_missing_keys=True),
                            T.CenterSpatialCropd(keys=['image', 'mask'], roi_size=self.final_patch_size, allow_missing_keys=True),
                            T.ToTensor(dtype=torch.float32, device='cuda:0')
                            ])

        return preprocess
    
    def set_up_augmentation_pipeline(self):
        spatial_transform = [
                                    T.RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=0, allow_missing_keys=True), 
                                    T.RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=1, allow_missing_keys=True),
                                    T.RandRotated(keys=['image', 'mask'], prob=0.5, range_x=np.pi, mode=["bilinear", "nearest"], padding_mode="zeros", allow_missing_keys=True),
                                    T.RandZoomd(keys=['image', 'mask'], prob=0.5, min_zoom=0.5, max_zoom=1.5, mode=["bilinear", "nearest"], padding_mode="constant", allow_missing_keys=True),
                                    T.RandAffined(keys=['image', 'mask'], prob=0.5, translate_range=(-26, 26), mode=["bilinear", "nearest"], padding_mode="zeros", allow_missing_keys=True)
                            ]

        pixel_transform = [
                                        T.RandRicianNoised(keys=['image'], prob=0.5, std=0.075),
                                        T.RandGibbsNoised(keys=['image'], prob=0.5, alpha=[0.45, 0.75]),
                                        T.RandAdjustContrastd(keys=['image'], prob=0.5, gamma=(0.7, 1.5), allow_missing_keys=True),
                                        T.RandGaussianNoised(keys=['image'], prob=0.5, std=0.04, allow_missing_keys=True),
                                        T.RandGaussianSharpend(keys=['image'], prob=0.5, sigma1_x=(0.1, 0.2), sigma1_y=(0.1, 0.2), sigma2_x=(0.2, 0.4), sigma2_y=(0.2, 0.4), alpha=(2.0, 3.0), allow_missing_keys=True)
                                ]

        return pixel_transform, spatial_transform

    def generate_train_batch(self):
        #eval_idx = None
        selected_keys = np.random.choice(self.list_of_keys, self.batch_size, True, None)
        list_of_frames = []
        for i in range(len(selected_keys)):
            patient_id = selected_keys[i][:10]
            filtered = [x for x in self.list_of_keys if x[:10] in patient_id]
            list_of_frames.append(filtered)
        
        target_list = []
        unlabeled_list = []
        padding_need_list = []
        target_mask_list = []

        case_properties = []
        for j, frames in enumerate(list_of_frames):
            labeled_frame = frames[0]
            if 'properties' in self._data[labeled_frame].keys():
                properties = self._data[labeled_frame]['properties']
            else:
                properties = load_pickle(self._data[labeled_frame]['properties_file'])
            case_properties.append(properties)

            #print(properties['ed_number'])
            #print(np.rint(np.array(properties['ed_number'])))
            #print('******************')

            ed_idx = np.rint(np.array(properties['ed_number'])).astype(np.uint8) % len(frames)
            es_idx = np.rint(np.array(properties['es_number'])).astype(np.uint8) % len(frames)

            depth = properties['size_after_resampling'][0]
            depth_idx = np.random.choice(depth)

            global_labeled_idx = np.array([ed_idx, es_idx])

            frames = sorted(frames, key=lambda x: int(x[16:18]))
            frames = np.array(frames)

            unlabeled = torch.full(size=(self.video_length, 1, *self.final_patch_size), fill_value=torch.nan, device='cuda:0')
            seg = torch.full(size=(self.video_length, 1, *self.final_patch_size), fill_value=torch.nan, device='cuda:0')
            cropped_unlabeled = torch.full(size=(self.video_length, 1, self.crop_size, self.crop_size), fill_value=torch.nan, device='cuda:0')
            cropped_seg = torch.full(size=(self.video_length, 1, self.crop_size, self.crop_size), fill_value=torch.nan, device='cuda:0')

            possible_indices = np.arange(len(frames))

            before_where = np.argwhere(possible_indices < ed_idx).reshape(-1,)
            after_where = np.argwhere(possible_indices >= ed_idx).reshape(-1,)

            before_indices = possible_indices[before_where]
            after_indices = possible_indices[after_where]

            possible_indices = np.concatenate([after_indices, before_indices])

            possible_indices = np.pad(possible_indices, pad_width=(1000, 1000), mode='wrap')
            start = np.random.randint(0, len(possible_indices) - self.video_length)
            random_indices = possible_indices[start:start + self.video_length]

            target_mask = np.full_like(random_indices, fill_value=False)
            sorted_indices = np.argsort(random_indices)

            random_indices = random_indices[sorted_indices]
            target_mask = torch.from_numpy(target_mask).to('cuda:0')

            frame_indices = possible_indices[random_indices]

            assert len(frame_indices) == self.video_length
            video = frames[frame_indices]


            labeled_idx = np.where(np.isin(frame_indices, global_labeled_idx))[0]

            for frame_idx, t in enumerate(video):

                if '_u' in t:
                    if not isfile(self.un_data[t]['data_file'][:-4] + ".npy"):
                        # lets hope you know what you're doing
                        case_all_data = np.load(self.un_data[t]['data_file'][:-4] + ".npz")['data']
                    else:
                        case_all_data = np.load(self.un_data[t]['data_file'][:-4] + ".npy", self.memmap_mode)
                else:
                    if not isfile(self._data[t]['data_file'][:-4] + ".npy"):
                        # lets hope you know what you're doing
                        case_all_data = np.load(self._data[t]['data_file'][:-4] + ".npz")['data']
                    else:
                        case_all_data = np.load(self._data[t]['data_file'][:-4] + ".npy", self.memmap_mode)

                # this is for when there is just a 2d slice in case_all_data (2d support)
                if len(case_all_data.shape) == 3:
                    case_all_data = case_all_data[:, None]
                
                assert case_all_data.shape[1] == depth

                # now crop case_all_data to contain just the slice of interest. If we want additional slice above and
                # below the current slice, here is where we get them. We stack those as additional color channels
                slice_data = case_all_data[:, depth_idx]

                # case all data should now be (c, x, y)
                assert len(slice_data.shape) == 3

                if '_u' not in t:
                    image = slice_data[:-1].copy()
                    mask = slice_data[-1:].copy()
                    mask[mask < 0] = 0

                    image, mask = self.preprocess(image, mask)

                    if target_mask[frame_idx] == True:
                        seg[frame_idx] = mask
                        assert frame_idx in labeled_idx
                else:
                    image = slice_data.copy()

                    image, _ = self.preprocess(image, None)
                
                unlabeled[frame_idx] = image

            with torch.no_grad():
                mean_centroid, _ = self.processor.preprocess_no_registration(data=torch.clone(unlabeled)) # T, C(1), H, W

                cropped_unlabeled, padding_need = self.processor.crop_and_pad(data=unlabeled, mean_centroid=mean_centroid)
                padding_need_list.append(padding_need)

                cropped_seg, _ = self.processor.crop_and_pad(data=seg, mean_centroid=mean_centroid)

            cropped_unlabeled = (cropped_unlabeled - cropped_unlabeled.min()) / (cropped_unlabeled.max() - cropped_unlabeled.min() + 1e-8)

            assert torch.count_nonzero(torch.any(~torch.isnan(cropped_seg.reshape(self.video_length, -1)), dim=-1)) == 0
            assert torch.count_nonzero(target_mask) == 0

            if not self.is_val and self.do_data_aug:
                temp_before_augment = torch.clone(cropped_unlabeled).to('cpu')

                current_pixel_transform = np.random.choice(self.pixel_transform)
                current_spatial_transform = np.random.choice(self.spatial_transform)

                #print(current_pixel_transform)
                
                seed = random.randint(0, 2**32-1)

                for t in range(len(cropped_unlabeled)):

                    current_pixel_transform = current_pixel_transform.set_random_state(seed=seed)
                    current_spatial_transform = current_spatial_transform.set_random_state(seed=seed)

                    cropped_unlabeled[t], cropped_seg[t] = self.augment(image=cropped_unlabeled[t], mask=cropped_seg[t], pixel_transform=current_pixel_transform, spatial_transform=current_spatial_transform)

                #matplotlib.use('QtAgg')
                #fig, ax = plt.subplots(3, len(cropped_unlabeled))
                #for t in range(len(cropped_unlabeled)):
                #    ax[0, t].imshow(temp_before_augment[t, 0].cpu(), cmap='gray')
                #    ax[1, t].imshow(cropped_unlabeled[t, 0].cpu(), cmap='gray')
                #    ax[2, t].imshow(cropped_seg[t, 0].cpu(), cmap='gray')
                #plt.show()
                    
            cropped_unlabeled = NormalizeIntensity()(cropped_unlabeled)

            unlabeled_list.append(cropped_unlabeled)
            target_list.append(cropped_seg)
            target_mask_list.append(target_mask)

        unlabeled = torch.stack(unlabeled_list, dim=1) # T, B, 1, H, W
        target = torch.stack(target_list, dim=1) # T, B, 1, H, W
        target_mask = torch.stack(target_mask_list, dim=1) # T, B
        padding_need = torch.stack(padding_need_list, dim=0) # B, 4

        keys = selected_keys

        #print(seg.shape)
        #print(np.unique(seg))
        #matplotlib.use('QtAgg')
        #fig, ax = plt.subplots(1, 2)
        #ax[0].imshow(labeled[3, 0], cmap='gray')
        #ax[1].imshow(seg[3, 0], cmap='gray')
        #plt.show()

        return {'unlabeled':unlabeled, 
                'target': target,
                'in_between': None,
                'padding_need': padding_need,
                'properties': case_properties,
                'target_mask': target_mask}
    



class DataLoaderFlowLibProgressiveAllDataFirst(SlimDataLoaderBase):
    def __init__(self, data, patch_size, final_patch_size, batch_size, video_length, processor, crop_size, is_val, do_data_aug, oversample_foreground_percent=0.0,
                 memmap_mode="r", pseudo_3d_slices=1, pad_mode="edge",
                 pad_kwargs_data=None, pad_sides=None):
        """
        This is the basic data loader for 2D networks. It uses preprocessed data as produced by my (Fabian) preprocessing.
        You can load the data with load_dataset(folder) where folder is the folder where the npz files are located. If there
        are only npz files present in that folder, the data loader will unpack them on the fly. This may take a while
        and increase CPU usage. Therefore, I advise you to call unpack_dataset(folder) first, which will unpack all npz
        to npy. Don't forget to call delete_npy(folder) after you are done with training?
        Why all the hassle? Well the decathlon dataset is huge. Using npy for everything will consume >1 TB and that is uncool
        given that I (Fabian) will have to store that permanently on /datasets and my local computer. With this strategy all
        data is stored in a compressed format (factor 10 smaller) and only unpacked when needed.
        :param data: get this with load_dataset(folder, stage=0). Plug the return value in here and you are g2g (good to go)
        :param patch_size: what patch size will this data loader return? it is common practice to first load larger
        patches so that a central crop after data augmentation can be done to reduce border artifacts. If unsure, use
        get_patch_size() from data_augmentation.default_data_augmentation
        :param final_patch_size: what will the patch finally be cropped to (after data augmentation)? this is the patch
        size that goes into your network. We need this here because we will pad patients in here so that patches at the
        border of patients are sampled properly
        :param batch_size:
        :param num_batches: how many batches will the data loader produce before stopping? None=endless
        :param seed:
        :param stage: ignore this (Fabian only)
        :param transpose: ignore this
        :param random: sample randomly; CAREFUL! non-random sampling requires batch_size=1, otherwise you will iterate batch_size times over the dataset
        :param pseudo_3d_slices: 7 = 3 below and 3 above the center slice
        """
        super(DataLoaderFlowLibProgressiveAllDataFirst, self).__init__(data, batch_size, None)
        if pad_kwargs_data is None:
            pad_kwargs_data = OrderedDict()
        self.pad_kwargs_data = pad_kwargs_data
        self.pad_mode = pad_mode
        self.is_val = is_val
        self.crop_size = crop_size
        self.pseudo_3d_slices = pseudo_3d_slices
        self.oversample_foreground_percent = oversample_foreground_percent
        self.final_patch_size = final_patch_size
        self.patch_size = patch_size
        self.list_of_keys = list(self._data.keys())
        self.need_to_pad = np.array(patch_size) - np.array(final_patch_size)
        self.memmap_mode = memmap_mode
        self.video_length = video_length
        self.percent = None
        if pad_sides is not None:
            if not isinstance(pad_sides, np.ndarray):
                pad_sides = np.array(pad_sides)
            self.need_to_pad += pad_sides
        self.pad_sides = pad_sides
        self.data_shape, self.seg_shape = self.determine_shapes()
        self.do_data_aug = do_data_aug

        self.preprocessing_transform = self.set_up_preprocessing_pipeline()
        self.pixel_transform, self.spatial_transform = self.set_up_augmentation_pipeline()

        self.processor = processor
        

    def determine_shapes(self):
        num_seg = 1

        k = list(self._data.keys())[0]
        if isfile(self._data[k]['data_file'][:-4] + ".npy"):
            case_all_data = np.load(self._data[k]['data_file'][:-4] + ".npy", self.memmap_mode)
        else:
            case_all_data = np.load(self._data[k]['data_file'])['data']
        num_color_channels = case_all_data.shape[0] - num_seg
        data_shape = (self.batch_size, num_color_channels, *self.patch_size)
        seg_shape = (self.batch_size, num_seg, *self.patch_size)
        return data_shape, seg_shape

    def get_do_oversample(self, batch_idx):
        return not batch_idx < round(self.batch_size * (1 - self.oversample_foreground_percent))
    
    def select_deep_supervision(self, x):
        if self.deep_supervision:
            return x[0]
        else:
            return x
        

    def my_crop(self, labeled, seg, unlabeled):

        xm = labeled.shape[-1] / 2
        ym = labeled.shape[-2] / 2

        x1 = int(round(xm - self.final_patch_size[1] / 2))
        x2 = int(round(xm + self.final_patch_size[1] / 2))
        y1 = int(round(ym - self.final_patch_size[0] / 2))
        y2 = int(round(ym + self.final_patch_size[0] / 2))

        labeled = labeled[:, :, y1:y2, x1:x2]
        seg = seg[:, :, y1:y2, x1:x2]

        xm = unlabeled.shape[-1] / 2
        ym = unlabeled.shape[-2] / 2

        x1 = int(round(xm - self.final_patch_size[1] / 2))
        x2 = int(round(xm + self.final_patch_size[1] / 2))
        y1 = int(round(ym - self.final_patch_size[0] / 2))
        y2 = int(round(ym + self.final_patch_size[0] / 2))

        unlabeled = unlabeled[:, :, :, y1:y2, x1:x2]

        return labeled, seg, unlabeled
    
    def preprocess(self, image, mask):
        if mask is not None:
            data = {'image': image, 'mask': mask}
            transformed = self.preprocessing_transform(data)
            image = transformed['image']
            mask = transformed['mask']
        else:
            data = {'image': image}
            transformed = self.preprocessing_transform(data)
            image = transformed['image']

        return image, mask
    
    
    def augment(self, image, mask, seed):
        self.pixel_transform = self.pixel_transform.set_random_state(seed=seed)
        self.spatial_transform = self.spatial_transform.set_random_state(seed=seed)
        
        temp = torch.clone(image)
        padding_mask = image == 0

        if mask is not None:
            data = {'image': image, 'mask': mask}
            pixel_transformed = self.pixel_transform(data)
            image = pixel_transformed['image']
            mask = pixel_transformed['mask']
            image[padding_mask] = 0
            data = {'image': image, 'mask': mask}
            spatial_transformed = self.spatial_transform(data)
            image = spatial_transformed['image']
            mask = spatial_transformed['mask']

            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(1, 3)
            #ax[0].imshow(temp[0].cpu(), cmap='gray')
            #ax[1].imshow(image[0].cpu(), cmap='gray')
            #ax[2].imshow(mask[0].cpu(), cmap='gray')
            #plt.show()
        else:
            data = {'image': image}
            pixel_transformed = self.pixel_transform(data)
            image = pixel_transformed['image']
            image[padding_mask] = 0
            data = {'image': image}
            spatial_transformed = self.spatial_transform(data)
            image = spatial_transformed['image']

            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(1, 2)
            #ax[0].imshow(temp[0].cpu(), cmap='gray')
            #ax[1].imshow(image[0].cpu(), cmap='gray')
            #plt.show()

        return image, mask
    
    def set_up_preprocessing_pipeline(self):
        preprocess = Compose([
                            T.SpatialPadd(keys=['image', 'mask'], spatial_size=self.final_patch_size, allow_missing_keys=True),
                            T.CenterSpatialCropd(keys=['image', 'mask'], roi_size=self.final_patch_size, allow_missing_keys=True),
                            T.ToTensor(dtype=torch.float32, device='cuda:0')
                            ])

        return preprocess
    
    def set_up_augmentation_pipeline(self):
        spatial_transform = Compose([
                                    T.RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=0, allow_missing_keys=True),
                                    T.RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=1, allow_missing_keys=True),
                                    T.RandRotated(keys=['image', 'mask'], prob=0.2, range_x=np.pi, mode=["bilinear", "nearest"], padding_mode="zeros", allow_missing_keys=True),
                                    T.RandZoomd(keys=['image', 'mask'], prob=0.2, min_zoom=0.5, max_zoom=1.5, mode=["bilinear", "nearest"], padding_mode="constant", allow_missing_keys=True),
                                    T.RandAffined(keys=['image', 'mask'], prob=0.2, translate_range=(-26, 26), mode=["bilinear", "nearest"], padding_mode="zeros", allow_missing_keys=True)
                                    ])

        pixel_transform = Compose([
                                #RandInvertd(keys=['image'], prob=0.5),
                                T.RandAdjustContrastd(keys=['image'], prob=0.2, gamma=(0.7, 1.5), allow_missing_keys=True),
                                T.RandGaussianNoised(keys=['image'], prob=0.2, std=0.04, allow_missing_keys=True),
                                T.RandScaleIntensityd(keys=['image'], prob=0.2, factors=0.2, allow_missing_keys=True),
                                T.RandGaussianSmoothd(keys=['image'], prob=0.2, sigma_x=(0.25, 0.5), sigma_y=(0.25, 0.5), allow_missing_keys=True),
                                T.RandGaussianSharpend(keys=['image'], prob=0.2, sigma1_x=(0.1, 0.2), sigma1_y=(0.1, 0.2), sigma2_x=(0.2, 0.4), sigma2_y=(0.2, 0.4), alpha=(2.0, 3.0), allow_missing_keys=True)
                                ])

        return pixel_transform, spatial_transform

    def generate_train_batch(self):
        #eval_idx = None
        selected_keys = np.random.choice(self.list_of_keys, self.batch_size, True, None)
        list_of_frames = []
        for i in range(len(selected_keys)):
            patient_id = selected_keys[i][:10]
            filtered = [x for x in self.list_of_keys if x[:10] in patient_id]
            list_of_frames.append(filtered)
        
        target_list = []
        unlabeled_list = []
        padding_need_list = []
        target_mask_list = []

        case_properties = []
        for j, frames in enumerate(list_of_frames):
            labeled_frame = frames[0]
            if 'properties' in self._data[labeled_frame].keys():
                properties = self._data[labeled_frame]['properties']
            else:
                properties = load_pickle(self._data[labeled_frame]['properties_file'])
            case_properties.append(properties)

            #print(properties['ed_number'])
            #print(np.rint(np.array(properties['ed_number'])))
            #print('******************')

            ed_idx = np.rint(np.array(properties['ed_number'])).astype(np.uint8) % len(frames)
            es_idx = np.rint(np.array(properties['es_number'])).astype(np.uint8) % len(frames)

            depth = properties['size_after_resampling'][0]
            depth_idx = np.random.choice(depth)

            global_labeled_idx = np.array([ed_idx, es_idx])

            frames = sorted(frames, key=lambda x: int(x[16:18]))
            frames = np.array(frames)

            unlabeled = torch.full(size=(self.video_length, 1, *self.final_patch_size), fill_value=torch.nan, device='cuda:0')
            seg = torch.full(size=(self.video_length, 1, *self.final_patch_size), fill_value=torch.nan, device='cuda:0')
            cropped_unlabeled = torch.full(size=(self.video_length, 1, self.crop_size, self.crop_size), fill_value=torch.nan, device='cuda:0')
            cropped_seg = torch.full(size=(self.video_length, 1, self.crop_size, self.crop_size), fill_value=torch.nan, device='cuda:0')

            possible_indices = np.arange(len(frames))

            before_where = np.argwhere(possible_indices < ed_idx).reshape(-1,)
            after_where = np.argwhere(possible_indices >= ed_idx).reshape(-1,)

            before_indices = possible_indices[before_where]
            after_indices = possible_indices[after_where]

            possible_indices = np.concatenate([after_indices, before_indices])

            stop_idx = np.argwhere(possible_indices == es_idx)[0][0]
            chunk1 = possible_indices[:stop_idx + 1]
            chunk2 = possible_indices[stop_idx:]
            chunk2 = np.concatenate([np.array([possible_indices[0]]), chunk2[::-1]])

            r = np.random.randint(0, 2)
            if r == 0:
                possible_indices = chunk1
            elif r == 1:
                possible_indices = chunk2
            
            assert possible_indices[0] == ed_idx
            assert possible_indices[-1] == es_idx

            random_indices = np.random.choice(np.arange(1, len(possible_indices)), size=1)
            target_mask = np.concatenate([np.array([True]), np.full_like(random_indices, fill_value=False)]).astype(bool)
            random_indices = np.concatenate([np.array([0]), random_indices])
            sorted_indices = np.argsort(random_indices)

            random_indices = random_indices[sorted_indices]
            target_mask = target_mask[sorted_indices]
            target_mask = torch.from_numpy(target_mask).to('cuda:0')

            frame_indices = possible_indices[random_indices]

            assert len(frame_indices) == self.video_length
            video = frames[frame_indices]


            labeled_idx = np.where(np.isin(frame_indices, global_labeled_idx))[0]

            for frame_idx, t in enumerate(video):

                if '_u' in t:
                    if not isfile(self.un_data[t]['data_file'][:-4] + ".npy"):
                        # lets hope you know what you're doing
                        case_all_data = np.load(self.un_data[t]['data_file'][:-4] + ".npz")['data']
                    else:
                        case_all_data = np.load(self.un_data[t]['data_file'][:-4] + ".npy", self.memmap_mode)
                else:
                    if not isfile(self._data[t]['data_file'][:-4] + ".npy"):
                        # lets hope you know what you're doing
                        case_all_data = np.load(self._data[t]['data_file'][:-4] + ".npz")['data']
                    else:
                        case_all_data = np.load(self._data[t]['data_file'][:-4] + ".npy", self.memmap_mode)

                # this is for when there is just a 2d slice in case_all_data (2d support)
                if len(case_all_data.shape) == 3:
                    case_all_data = case_all_data[:, None]
                
                assert case_all_data.shape[1] == depth

                # now crop case_all_data to contain just the slice of interest. If we want additional slice above and
                # below the current slice, here is where we get them. We stack those as additional color channels
                slice_data = case_all_data[:, depth_idx]

                # case all data should now be (c, x, y)
                assert len(slice_data.shape) == 3

                if '_u' not in t:
                    image = slice_data[:-1].copy()
                    mask = slice_data[-1:].copy()
                    mask[mask < 0] = 0

                    image, mask = self.preprocess(image, mask)

                    if target_mask[frame_idx] == True:
                        seg[frame_idx] = mask
                        assert frame_idx in labeled_idx
                else:
                    image = slice_data.copy()

                    image, _ = self.preprocess(image, None)
                
                unlabeled[frame_idx] = image

            with torch.no_grad():
                mean_centroid, _ = self.processor.preprocess_no_registration(data=torch.clone(unlabeled)) # T, C(1), H, W

                cropped_unlabeled, padding_need = self.processor.crop_and_pad(data=unlabeled, mean_centroid=mean_centroid)
                padding_need_list.append(padding_need)

                cropped_seg, _ = self.processor.crop_and_pad(data=seg, mean_centroid=mean_centroid)

            cropped_unlabeled = (cropped_unlabeled - cropped_unlabeled.min()) / (cropped_unlabeled.max() - cropped_unlabeled.min() + 1e-8)

            assert torch.count_nonzero(torch.any(~torch.isnan(cropped_seg.reshape(self.video_length, -1)), dim=-1)) == 1
            assert target_mask[0] == True
            assert torch.count_nonzero(target_mask) == 1

            if not self.is_val and self.do_data_aug:
                temp_before_augment = torch.clone(cropped_unlabeled).to('cpu')
                
                seed = random.randint(0, 2**32-1)

                for t in range(len(cropped_unlabeled)):
                    cropped_unlabeled[t], cropped_seg[t] = self.augment(image=cropped_unlabeled[t], mask=cropped_seg[t], seed=seed)

            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(3, len(cropped_unlabeled))
            #for t in range(len(cropped_unlabeled)):
            #    ax[0, t].imshow(temp_before_augment[t, 0].cpu(), cmap='gray')
            #    ax[1, t].imshow(cropped_unlabeled[t, 0].cpu(), cmap='gray')
            #    ax[2, t].imshow(cropped_seg[t, 0].cpu(), cmap='gray')
            #plt.show()
                    
            cropped_unlabeled = NormalizeIntensity()(cropped_unlabeled)

            unlabeled_list.append(cropped_unlabeled)
            target_list.append(cropped_seg)
            target_mask_list.append(target_mask)

        unlabeled = torch.stack(unlabeled_list, dim=1) # T, B, 1, H, W
        target = torch.stack(target_list, dim=1) # T, B, 1, H, W
        target_mask = torch.stack(target_mask_list, dim=1) # T, B
        padding_need = torch.stack(padding_need_list, dim=0) # B, 4

        keys = selected_keys

        #print(seg.shape)
        #print(np.unique(seg))
        #matplotlib.use('QtAgg')
        #fig, ax = plt.subplots(1, 2)
        #ax[0].imshow(labeled[3, 0], cmap='gray')
        #ax[1].imshow(seg[3, 0], cmap='gray')
        #plt.show()

        return {'unlabeled':unlabeled, 
                'target': target,
                'in_between': None,
                'padding_need': padding_need,
                'properties': case_properties,
                'target_mask': target_mask}
    


class DataLoaderFlowLibProgressiveAllDataAdjacent(SlimDataLoaderBase):
    def __init__(self, data, patch_size, final_patch_size, batch_size, video_length, processor, crop_size, is_val, do_data_aug, oversample_foreground_percent=0.0,
                 memmap_mode="r", pseudo_3d_slices=1, pad_mode="edge",
                 pad_kwargs_data=None, pad_sides=None):
        """
        This is the basic data loader for 2D networks. It uses preprocessed data as produced by my (Fabian) preprocessing.
        You can load the data with load_dataset(folder) where folder is the folder where the npz files are located. If there
        are only npz files present in that folder, the data loader will unpack them on the fly. This may take a while
        and increase CPU usage. Therefore, I advise you to call unpack_dataset(folder) first, which will unpack all npz
        to npy. Don't forget to call delete_npy(folder) after you are done with training?
        Why all the hassle? Well the decathlon dataset is huge. Using npy for everything will consume >1 TB and that is uncool
        given that I (Fabian) will have to store that permanently on /datasets and my local computer. With this strategy all
        data is stored in a compressed format (factor 10 smaller) and only unpacked when needed.
        :param data: get this with load_dataset(folder, stage=0). Plug the return value in here and you are g2g (good to go)
        :param patch_size: what patch size will this data loader return? it is common practice to first load larger
        patches so that a central crop after data augmentation can be done to reduce border artifacts. If unsure, use
        get_patch_size() from data_augmentation.default_data_augmentation
        :param final_patch_size: what will the patch finally be cropped to (after data augmentation)? this is the patch
        size that goes into your network. We need this here because we will pad patients in here so that patches at the
        border of patients are sampled properly
        :param batch_size:
        :param num_batches: how many batches will the data loader produce before stopping? None=endless
        :param seed:
        :param stage: ignore this (Fabian only)
        :param transpose: ignore this
        :param random: sample randomly; CAREFUL! non-random sampling requires batch_size=1, otherwise you will iterate batch_size times over the dataset
        :param pseudo_3d_slices: 7 = 3 below and 3 above the center slice
        """
        super(DataLoaderFlowLibProgressiveAllDataAdjacent, self).__init__(data, batch_size, None)
        if pad_kwargs_data is None:
            pad_kwargs_data = OrderedDict()
        self.pad_kwargs_data = pad_kwargs_data
        self.pad_mode = pad_mode
        self.is_val = is_val
        self.crop_size = crop_size
        self.pseudo_3d_slices = pseudo_3d_slices
        self.oversample_foreground_percent = oversample_foreground_percent
        self.final_patch_size = final_patch_size
        self.patch_size = patch_size
        self.list_of_keys = list(self._data.keys())
        self.need_to_pad = np.array(patch_size) - np.array(final_patch_size)
        self.memmap_mode = memmap_mode
        self.video_length = video_length
        self.percent = None
        if pad_sides is not None:
            if not isinstance(pad_sides, np.ndarray):
                pad_sides = np.array(pad_sides)
            self.need_to_pad += pad_sides
        self.pad_sides = pad_sides
        self.data_shape, self.seg_shape = self.determine_shapes()
        self.do_data_aug = do_data_aug

        self.preprocessing_transform = self.set_up_preprocessing_pipeline()
        self.pixel_transform, self.spatial_transform = self.set_up_augmentation_pipeline()

        self.processor = processor
        

    def determine_shapes(self):
        num_seg = 1

        k = list(self._data.keys())[0]
        if isfile(self._data[k]['data_file'][:-4] + ".npy"):
            case_all_data = np.load(self._data[k]['data_file'][:-4] + ".npy", self.memmap_mode)
        else:
            case_all_data = np.load(self._data[k]['data_file'])['data']
        num_color_channels = case_all_data.shape[0] - num_seg
        data_shape = (self.batch_size, num_color_channels, *self.patch_size)
        seg_shape = (self.batch_size, num_seg, *self.patch_size)
        return data_shape, seg_shape

    def get_do_oversample(self, batch_idx):
        return not batch_idx < round(self.batch_size * (1 - self.oversample_foreground_percent))
    
    def select_deep_supervision(self, x):
        if self.deep_supervision:
            return x[0]
        else:
            return x
        

    def my_crop(self, labeled, seg, unlabeled):

        xm = labeled.shape[-1] / 2
        ym = labeled.shape[-2] / 2

        x1 = int(round(xm - self.final_patch_size[1] / 2))
        x2 = int(round(xm + self.final_patch_size[1] / 2))
        y1 = int(round(ym - self.final_patch_size[0] / 2))
        y2 = int(round(ym + self.final_patch_size[0] / 2))

        labeled = labeled[:, :, y1:y2, x1:x2]
        seg = seg[:, :, y1:y2, x1:x2]

        xm = unlabeled.shape[-1] / 2
        ym = unlabeled.shape[-2] / 2

        x1 = int(round(xm - self.final_patch_size[1] / 2))
        x2 = int(round(xm + self.final_patch_size[1] / 2))
        y1 = int(round(ym - self.final_patch_size[0] / 2))
        y2 = int(round(ym + self.final_patch_size[0] / 2))

        unlabeled = unlabeled[:, :, :, y1:y2, x1:x2]

        return labeled, seg, unlabeled
    
    def preprocess(self, image, mask):
        if mask is not None:
            data = {'image': image, 'mask': mask}
            transformed = self.preprocessing_transform(data)
            image = transformed['image']
            mask = transformed['mask']
        else:
            data = {'image': image}
            transformed = self.preprocessing_transform(data)
            image = transformed['image']

        return image, mask
    
    
    def augment(self, image, mask, seed):
        self.pixel_transform = self.pixel_transform.set_random_state(seed=seed)
        self.spatial_transform = self.spatial_transform.set_random_state(seed=seed)
        
        temp = torch.clone(image)
        padding_mask = image == 0

        if mask is not None:
            data = {'image': image, 'mask': mask}
            pixel_transformed = self.pixel_transform(data)
            image = pixel_transformed['image']
            mask = pixel_transformed['mask']
            image[padding_mask] = 0
            data = {'image': image, 'mask': mask}
            spatial_transformed = self.spatial_transform(data)
            image = spatial_transformed['image']
            mask = spatial_transformed['mask']

            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(1, 3)
            #ax[0].imshow(temp[0].cpu(), cmap='gray')
            #ax[1].imshow(image[0].cpu(), cmap='gray')
            #ax[2].imshow(mask[0].cpu(), cmap='gray')
            #plt.show()
        else:
            data = {'image': image}
            pixel_transformed = self.pixel_transform(data)
            image = pixel_transformed['image']
            image[padding_mask] = 0
            data = {'image': image}
            spatial_transformed = self.spatial_transform(data)
            image = spatial_transformed['image']

            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(1, 2)
            #ax[0].imshow(temp[0].cpu(), cmap='gray')
            #ax[1].imshow(image[0].cpu(), cmap='gray')
            #plt.show()

        return image, mask
    
    def set_up_preprocessing_pipeline(self):
        preprocess = Compose([
                            T.SpatialPadd(keys=['image', 'mask'], spatial_size=self.final_patch_size, allow_missing_keys=True),
                            T.CenterSpatialCropd(keys=['image', 'mask'], roi_size=self.final_patch_size, allow_missing_keys=True),
                            T.ToTensor(dtype=torch.float32, device='cuda:0')
                            ])

        return preprocess
    
    def set_up_augmentation_pipeline(self):
        spatial_transform = Compose([
                                    T.RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=0, allow_missing_keys=True),
                                    T.RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=1, allow_missing_keys=True),
                                    T.RandRotated(keys=['image', 'mask'], prob=0.2, range_x=np.pi, mode=["bilinear", "nearest"], padding_mode="zeros", allow_missing_keys=True),
                                    T.RandZoomd(keys=['image', 'mask'], prob=0.2, min_zoom=0.5, max_zoom=1.5, mode=["bilinear", "nearest"], padding_mode="constant", allow_missing_keys=True),
                                    T.RandAffined(keys=['image', 'mask'], prob=0.2, translate_range=(-26, 26), mode=["bilinear", "nearest"], padding_mode="zeros", allow_missing_keys=True)
                                    ])

        pixel_transform = Compose([
                                #RandInvertd(keys=['image'], prob=0.5),
                                T.RandAdjustContrastd(keys=['image'], prob=0.2, gamma=(0.7, 1.5), allow_missing_keys=True),
                                T.RandGaussianNoised(keys=['image'], prob=0.2, std=0.04, allow_missing_keys=True),
                                T.RandScaleIntensityd(keys=['image'], prob=0.2, factors=0.2, allow_missing_keys=True),
                                T.RandGaussianSmoothd(keys=['image'], prob=0.2, sigma_x=(0.25, 0.5), sigma_y=(0.25, 0.5), allow_missing_keys=True),
                                T.RandGaussianSharpend(keys=['image'], prob=0.2, sigma1_x=(0.1, 0.2), sigma1_y=(0.1, 0.2), sigma2_x=(0.2, 0.4), sigma2_y=(0.2, 0.4), alpha=(2.0, 3.0), allow_missing_keys=True)
                                ])

        return pixel_transform, spatial_transform

    def generate_train_batch(self):
        #eval_idx = None
        selected_keys = np.random.choice(self.list_of_keys, self.batch_size, True, None)
        list_of_frames = []
        for i in range(len(selected_keys)):
            patient_id = selected_keys[i][:10]
            filtered = [x for x in self.list_of_keys if x[:10] in patient_id]
            list_of_frames.append(filtered)
        
        target_list = []
        unlabeled_list = []
        padding_need_list = []
        target_mask_list = []

        case_properties = []
        for j, frames in enumerate(list_of_frames):
            labeled_frame = frames[0]
            if 'properties' in self._data[labeled_frame].keys():
                properties = self._data[labeled_frame]['properties']
            else:
                properties = load_pickle(self._data[labeled_frame]['properties_file'])
            case_properties.append(properties)

            #print(properties['ed_number'])
            #print(np.rint(np.array(properties['ed_number'])))
            #print('******************')

            ed_idx = np.rint(np.array(properties['ed_number'])).astype(np.uint8) % len(frames)
            es_idx = np.rint(np.array(properties['es_number'])).astype(np.uint8) % len(frames)

            depth = properties['size_after_resampling'][0]
            depth_idx = np.random.choice(depth)

            global_labeled_idx = np.array([ed_idx, es_idx])

            frames = sorted(frames, key=lambda x: int(x[16:18]))
            frames = np.array(frames)

            unlabeled = torch.full(size=(self.video_length, 1, *self.final_patch_size), fill_value=torch.nan, device='cuda:0')
            seg = torch.full(size=(self.video_length, 1, *self.final_patch_size), fill_value=torch.nan, device='cuda:0')
            cropped_unlabeled = torch.full(size=(self.video_length, 1, self.crop_size, self.crop_size), fill_value=torch.nan, device='cuda:0')
            cropped_seg = torch.full(size=(self.video_length, 1, self.crop_size, self.crop_size), fill_value=torch.nan, device='cuda:0')

            possible_indices = np.arange(len(frames))

            before_where = np.argwhere(possible_indices < ed_idx).reshape(-1,)
            after_where = np.argwhere(possible_indices >= ed_idx).reshape(-1,)

            before_indices = possible_indices[before_where]
            after_indices = possible_indices[after_where]

            possible_indices = np.concatenate([after_indices, before_indices])
            
            assert possible_indices[0] == ed_idx

            random_indices = np.random.choice(np.arange(0, len(possible_indices) - 1), size=1)
            random_indices = np.concatenate([random_indices, np.array(random_indices + 1)], axis=0)
            target_mask = np.full_like(random_indices, fill_value=False).astype(bool)
            sorted_indices = np.argsort(random_indices)

            random_indices = random_indices[sorted_indices]
            target_mask = target_mask[sorted_indices]
            target_mask = torch.from_numpy(target_mask).to('cuda:0')

            frame_indices = possible_indices[random_indices]

            assert len(frame_indices) == self.video_length
            video = frames[frame_indices]


            labeled_idx = np.where(np.isin(frame_indices, global_labeled_idx))[0]

            for frame_idx, t in enumerate(video):

                if '_u' in t:
                    if not isfile(self.un_data[t]['data_file'][:-4] + ".npy"):
                        # lets hope you know what you're doing
                        case_all_data = np.load(self.un_data[t]['data_file'][:-4] + ".npz")['data']
                    else:
                        case_all_data = np.load(self.un_data[t]['data_file'][:-4] + ".npy", self.memmap_mode)
                else:
                    if not isfile(self._data[t]['data_file'][:-4] + ".npy"):
                        # lets hope you know what you're doing
                        case_all_data = np.load(self._data[t]['data_file'][:-4] + ".npz")['data']
                    else:
                        case_all_data = np.load(self._data[t]['data_file'][:-4] + ".npy", self.memmap_mode)

                # this is for when there is just a 2d slice in case_all_data (2d support)
                if len(case_all_data.shape) == 3:
                    case_all_data = case_all_data[:, None]
                
                assert case_all_data.shape[1] == depth

                # now crop case_all_data to contain just the slice of interest. If we want additional slice above and
                # below the current slice, here is where we get them. We stack those as additional color channels
                slice_data = case_all_data[:, depth_idx]

                # case all data should now be (c, x, y)
                assert len(slice_data.shape) == 3

                if '_u' not in t:
                    image = slice_data[:-1].copy()
                    mask = slice_data[-1:].copy()
                    mask[mask < 0] = 0

                    image, mask = self.preprocess(image, mask)

                    if target_mask[frame_idx] == True:
                        seg[frame_idx] = mask
                        assert frame_idx in labeled_idx
                else:
                    image = slice_data.copy()

                    image, _ = self.preprocess(image, None)
                
                unlabeled[frame_idx] = image

            with torch.no_grad():
                mean_centroid, _ = self.processor.preprocess_no_registration(data=torch.clone(unlabeled)) # T, C(1), H, W

                cropped_unlabeled, padding_need = self.processor.crop_and_pad(data=unlabeled, mean_centroid=mean_centroid)
                padding_need_list.append(padding_need)

                cropped_seg, _ = self.processor.crop_and_pad(data=seg, mean_centroid=mean_centroid)

            cropped_unlabeled = (cropped_unlabeled - cropped_unlabeled.min()) / (cropped_unlabeled.max() - cropped_unlabeled.min() + 1e-8)

            if not self.is_val and self.do_data_aug:
                temp_before_augment = torch.clone(cropped_unlabeled).to('cpu')
                
                seed = random.randint(0, 2**32-1)

                for t in range(len(cropped_unlabeled)):
                    cropped_unlabeled[t], cropped_seg[t] = self.augment(image=cropped_unlabeled[t], mask=cropped_seg[t], seed=seed)

            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(3, len(cropped_unlabeled))
            #for t in range(len(cropped_unlabeled)):
            #    ax[0, t].imshow(temp_before_augment[t, 0].cpu(), cmap='gray')
            #    ax[1, t].imshow(cropped_unlabeled[t, 0].cpu(), cmap='gray')
            #    ax[2, t].imshow(cropped_seg[t, 0].cpu(), cmap='gray')
            #plt.show()
                    
            cropped_unlabeled = NormalizeIntensity()(cropped_unlabeled)

            unlabeled_list.append(cropped_unlabeled)
            target_list.append(cropped_seg)
            target_mask_list.append(target_mask)

        unlabeled = torch.stack(unlabeled_list, dim=1) # T, B, 1, H, W
        target = torch.stack(target_list, dim=1) # T, B, 1, H, W
        target_mask = torch.stack(target_mask_list, dim=1) # T, B
        padding_need = torch.stack(padding_need_list, dim=0) # B, 4

        keys = selected_keys

        #print(seg.shape)
        #print(np.unique(seg))
        #matplotlib.use('QtAgg')
        #fig, ax = plt.subplots(1, 2)
        #ax[0].imshow(labeled[3, 0], cmap='gray')
        #ax[1].imshow(seg[3, 0], cmap='gray')
        #plt.show()

        return {'unlabeled':unlabeled, 
                'target': target,
                'in_between': None,
                'padding_need': padding_need,
                'properties': case_properties,
                'target_mask': target_mask}




class DataLoaderFlowTrain5LibAllData(SlimDataLoaderBase):
    def __init__(self, data, patch_size, final_patch_size, batch_size, video_length, processor, crop_size, is_val, oversample_foreground_percent=0.0,
                 memmap_mode="r", pseudo_3d_slices=1, pad_mode="edge",
                 pad_kwargs_data=None, pad_sides=None):
        """
        This is the basic data loader for 2D networks. It uses preprocessed data as produced by my (Fabian) preprocessing.
        You can load the data with load_dataset(folder) where folder is the folder where the npz files are located. If there
        are only npz files present in that folder, the data loader will unpack them on the fly. This may take a while
        and increase CPU usage. Therefore, I advise you to call unpack_dataset(folder) first, which will unpack all npz
        to npy. Don't forget to call delete_npy(folder) after you are done with training?
        Why all the hassle? Well the decathlon dataset is huge. Using npy for everything will consume >1 TB and that is uncool
        given that I (Fabian) will have to store that permanently on /datasets and my local computer. With this strategy all
        data is stored in a compressed format (factor 10 smaller) and only unpacked when needed.
        :param data: get this with load_dataset(folder, stage=0). Plug the return value in here and you are g2g (good to go)
        :param patch_size: what patch size will this data loader return? it is common practice to first load larger
        patches so that a central crop after data augmentation can be done to reduce border artifacts. If unsure, use
        get_patch_size() from data_augmentation.default_data_augmentation
        :param final_patch_size: what will the patch finally be cropped to (after data augmentation)? this is the patch
        size that goes into your network. We need this here because we will pad patients in here so that patches at the
        border of patients are sampled properly
        :param batch_size:
        :param num_batches: how many batches will the data loader produce before stopping? None=endless
        :param seed:
        :param stage: ignore this (Fabian only)
        :param transpose: ignore this
        :param random: sample randomly; CAREFUL! non-random sampling requires batch_size=1, otherwise you will iterate batch_size times over the dataset
        :param pseudo_3d_slices: 7 = 3 below and 3 above the center slice
        """
        super(DataLoaderFlowTrain5LibAllData, self).__init__(data, batch_size, None)
        if pad_kwargs_data is None:
            pad_kwargs_data = OrderedDict()
        self.pad_kwargs_data = pad_kwargs_data
        self.pad_mode = pad_mode
        self.is_val = is_val
        self.crop_size = crop_size
        self.pseudo_3d_slices = pseudo_3d_slices
        self.oversample_foreground_percent = oversample_foreground_percent
        self.final_patch_size = final_patch_size
        self.patch_size = patch_size
        self.list_of_keys = list(self._data.keys())
        self.need_to_pad = np.array(patch_size) - np.array(final_patch_size)
        self.memmap_mode = memmap_mode
        self.video_length = video_length
        self.percent = None
        if pad_sides is not None:
            if not isinstance(pad_sides, np.ndarray):
                pad_sides = np.array(pad_sides)
            self.need_to_pad += pad_sides
        self.pad_sides = pad_sides
        self.data_shape, self.seg_shape = self.determine_shapes()

        self.preprocessing_transform = self.set_up_preprocessing_pipeline()
        self.pixel_transform, self.spatial_transform = self.set_up_augmentation_pipeline()

        self.processor = processor

    def determine_shapes(self):
        num_seg = 1

        k = list(self._data.keys())[0]
        if isfile(self._data[k]['data_file'][:-4] + ".npy"):
            case_all_data = np.load(self._data[k]['data_file'][:-4] + ".npy", self.memmap_mode)
        else:
            case_all_data = np.load(self._data[k]['data_file'])['data']
        num_color_channels = case_all_data.shape[0] - num_seg
        data_shape = (self.batch_size, num_color_channels, *self.patch_size)
        seg_shape = (self.batch_size, num_seg, *self.patch_size)
        return data_shape, seg_shape

    def get_do_oversample(self, batch_idx):
        return not batch_idx < round(self.batch_size * (1 - self.oversample_foreground_percent))
    
    def select_deep_supervision(self, x):
        if self.deep_supervision:
            return x[0]
        else:
            return x
        

    def my_crop(self, labeled, seg, unlabeled):

        xm = labeled.shape[-1] / 2
        ym = labeled.shape[-2] / 2

        x1 = int(round(xm - self.final_patch_size[1] / 2))
        x2 = int(round(xm + self.final_patch_size[1] / 2))
        y1 = int(round(ym - self.final_patch_size[0] / 2))
        y2 = int(round(ym + self.final_patch_size[0] / 2))

        labeled = labeled[:, :, y1:y2, x1:x2]
        seg = seg[:, :, y1:y2, x1:x2]

        xm = unlabeled.shape[-1] / 2
        ym = unlabeled.shape[-2] / 2

        x1 = int(round(xm - self.final_patch_size[1] / 2))
        x2 = int(round(xm + self.final_patch_size[1] / 2))
        y1 = int(round(ym - self.final_patch_size[0] / 2))
        y2 = int(round(ym + self.final_patch_size[0] / 2))

        unlabeled = unlabeled[:, :, :, y1:y2, x1:x2]

        return labeled, seg, unlabeled
    
    def preprocess(self, image, mask):
        if mask is not None:
            data = {'image': image, 'mask': mask}
            transformed = self.preprocessing_transform(data)
            image = transformed['image']
            mask = transformed['mask']
        else:
            data = {'image': image}
            transformed = self.preprocessing_transform(data)
            image = transformed['image']

        return image, mask
    
    
    def augment(self, image, mask, seed):
        self.pixel_transform = self.pixel_transform.set_random_state(seed=seed)
        self.spatial_transform = self.spatial_transform.set_random_state(seed=seed)
        
        temp = torch.clone(image)
        padding_mask = image == 0

        if mask is not None:
            data = {'image': image, 'mask': mask}
            pixel_transformed = self.pixel_transform(data)
            image = pixel_transformed['image']
            mask = pixel_transformed['mask']
            image[padding_mask] = 0
            data = {'image': image, 'mask': mask}
            spatial_transformed = self.spatial_transform(data)
            image = spatial_transformed['image']
            mask = spatial_transformed['mask']

            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(1, 3)
            #ax[0].imshow(temp[0].cpu(), cmap='gray')
            #ax[1].imshow(image[0].cpu(), cmap='gray')
            #ax[2].imshow(mask[0].cpu(), cmap='gray')
            #plt.show()
        else:
            data = {'image': image}
            pixel_transformed = self.pixel_transform(data)
            image = pixel_transformed['image']
            image[padding_mask] = 0
            data = {'image': image}
            spatial_transformed = self.spatial_transform(data)
            image = spatial_transformed['image']

            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(1, 2)
            #ax[0].imshow(temp[0].cpu(), cmap='gray')
            #ax[1].imshow(image[0].cpu(), cmap='gray')
            #plt.show()

        return image, mask
    
    def set_up_preprocessing_pipeline(self):
        preprocess = Compose([
                            T.SpatialPadd(keys=['image', 'mask'], spatial_size=self.final_patch_size, allow_missing_keys=True),
                            T.CenterSpatialCropd(keys=['image', 'mask'], roi_size=self.final_patch_size, allow_missing_keys=True),
                            T.ToTensor(dtype=torch.float32, device='cuda:0')
                            ])

        return preprocess
    
    def set_up_augmentation_pipeline(self):
        spatial_transform = Compose([
                                    T.RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=0, allow_missing_keys=True),
                                    T.RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=1, allow_missing_keys=True),
                                    T.RandRotated(keys=['image', 'mask'], prob=0.2, range_x=np.pi, mode=["bilinear", "nearest"], padding_mode="zeros", allow_missing_keys=True),
                                    T.RandZoomd(keys=['image', 'mask'], prob=0.2, min_zoom=0.5, max_zoom=1.5, mode=["bilinear", "nearest"], padding_mode="constant", allow_missing_keys=True),
                                    T.RandAffined(keys=['image', 'mask'], prob=0.2, translate_range=(-26, 26), mode=["bilinear", "nearest"], padding_mode="zeros", allow_missing_keys=True)
                                    ])

        pixel_transform = Compose([
                                #RandInvertd(keys=['image'], prob=0.5),
                                T.RandAdjustContrastd(keys=['image'], prob=0.2, gamma=(0.7, 1.5), allow_missing_keys=True),
                                T.RandGaussianNoised(keys=['image'], prob=0.2, std=0.04, allow_missing_keys=True),
                                T.RandScaleIntensityd(keys=['image'], prob=0.2, factors=0.2, allow_missing_keys=True),
                                T.RandGaussianSmoothd(keys=['image'], prob=0.2, sigma_x=(0.25, 0.5), sigma_y=(0.25, 0.5), allow_missing_keys=True),
                                T.RandGaussianSharpend(keys=['image'], prob=0.2, sigma1_x=(0.1, 0.2), sigma1_y=(0.1, 0.2), sigma2_x=(0.2, 0.4), sigma2_y=(0.2, 0.4), alpha=(2.0, 3.0), allow_missing_keys=True)
                                ])

        return pixel_transform, spatial_transform

    def generate_train_batch(self):
        #eval_idx = None
        selected_keys = np.random.choice(self.list_of_keys, self.batch_size, True, None)
        list_of_frames = []
        for i in range(len(selected_keys)):
            patient_id = selected_keys[i][:10]
            filtered = [x for x in self.list_of_keys if x[:10] in patient_id]
            list_of_frames.append(filtered)
        
        target_list = []
        unlabeled_list = []
        padding_need_list = []
        target_mask_list = []
        distances_list = []

        case_properties = []
        for j, frames in enumerate(list_of_frames):
            labeled_frame = frames[0]
            if 'properties' in self._data[labeled_frame].keys():
                properties = self._data[labeled_frame]['properties']
            else:
                properties = load_pickle(self._data[labeled_frame]['properties_file'])
            case_properties.append(properties)

            #print(properties['ed_number'])
            #print(np.rint(np.array(properties['ed_number'])))
            #print('******************')

            ed_idx = np.rint(np.array(properties['ed_number'])).astype(np.uint8) % len(frames)

            depth = properties['size_after_resampling'][0]
            depth_idx = np.random.choice(depth)

            global_labeled_idx = np.array([ed_idx])

            frames = sorted(frames, key=lambda x: int(x[16:18]))
            frames = np.array(frames)

            unlabeled = torch.full(size=(self.video_length, 1, *self.final_patch_size), fill_value=torch.nan, device='cuda:0')
            seg = torch.full(size=(self.video_length, 1, *self.final_patch_size), fill_value=torch.nan, device='cuda:0')
            cropped_unlabeled = torch.full(size=(self.video_length, 1, self.crop_size, self.crop_size), fill_value=torch.nan, device='cuda:0')
            cropped_seg = torch.full(size=(self.video_length, 1, self.crop_size, self.crop_size), fill_value=torch.nan, device='cuda:0')

            possible_indices = np.arange(0, len(frames))

            random_indices = np.random.choice(possible_indices, size=self.video_length - 1)
            frame_indices = np.concatenate([global_labeled_idx.reshape((-1,)), random_indices])
            sorted_indices = np.argsort(frame_indices)
            frame_indices = frame_indices[sorted_indices]
            target_mask = torch.tensor([True, True], device='cuda:0').reshape((-1,))

            before_where = np.argwhere(frame_indices < ed_idx).reshape(-1,)
            after_where = np.argwhere(frame_indices >= ed_idx).reshape(-1,)

            before_indices = frame_indices[before_where]
            after_indices = frame_indices[after_where]

            before_mask = target_mask[before_where]
            after_mask = target_mask[after_where]

            frame_indices = np.concatenate([after_indices, before_indices])
            target_mask = torch.cat([after_mask, before_mask])
            assert frame_indices[0] == ed_idx

            assert len(frame_indices) == self.video_length
            distances_list.append(np.nan)
            video = frames[frame_indices]

            for frame_idx, t in enumerate(video):

                if '_u' in t:
                    if not isfile(self.un_data[t]['data_file'][:-4] + ".npy"):
                        # lets hope you know what you're doing
                        case_all_data = np.load(self.un_data[t]['data_file'][:-4] + ".npz")['data']
                    else:
                        case_all_data = np.load(self.un_data[t]['data_file'][:-4] + ".npy", self.memmap_mode)
                else:
                    if not isfile(self._data[t]['data_file'][:-4] + ".npy"):
                        # lets hope you know what you're doing
                        case_all_data = np.load(self._data[t]['data_file'][:-4] + ".npz")['data']
                    else:
                        case_all_data = np.load(self._data[t]['data_file'][:-4] + ".npy", self.memmap_mode)

                # this is for when there is just a 2d slice in case_all_data (2d support)
                if len(case_all_data.shape) == 3:
                    case_all_data = case_all_data[:, None]
                
                assert case_all_data.shape[1] == depth

                # now crop case_all_data to contain just the slice of interest. If we want additional slice above and
                # below the current slice, here is where we get them. We stack those as additional color channels
                slice_data = case_all_data[:, depth_idx]

                # case all data should now be (c, x, y)
                assert len(slice_data.shape) == 3

                if '_u' not in t:
                    image = slice_data[:-1].copy()
                    mask = slice_data[-1:].copy()
                    mask[mask < 0] = 0

                    image, mask = self.preprocess(image, mask)

                    seg[frame_idx] = mask
                else:
                    image = slice_data.copy()

                    image, _ = self.preprocess(image, None)

                unlabeled[frame_idx] = image

            with torch.no_grad():
                mean_centroid, _ = self.processor.preprocess_no_registration(data=torch.clone(unlabeled)) # T, C(1), H, W

                cropped_unlabeled, padding_need = self.processor.crop_and_pad(data=unlabeled, mean_centroid=mean_centroid)
                padding_need_list.append(padding_need)

                cropped_seg, _ = self.processor.crop_and_pad(data=seg, mean_centroid=mean_centroid)

            cropped_unlabeled = (cropped_unlabeled - cropped_unlabeled.min()) / (cropped_unlabeled.max() - cropped_unlabeled.min() + 1e-8)

            assert torch.count_nonzero(torch.any(~torch.isnan(cropped_seg.reshape(self.video_length, -1)), dim=-1)) == 2
            assert target_mask[0] == True
            assert torch.count_nonzero(target_mask) == 2

            if not self.is_val:
                temp_before_augment = torch.clone(cropped_unlabeled).to('cpu')
                
                seed = random.randint(0, 2**32-1)

                for t in range(len(cropped_unlabeled)):
                    cropped_unlabeled[t], cropped_seg[t] = self.augment(image=cropped_unlabeled[t], mask=cropped_seg[t], seed=seed)

            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(3, len(cropped_unlabeled))
            #for t in range(len(cropped_unlabeled)):
            #    ax[0, t].imshow(temp_before_augment[t, 0].cpu(), cmap='gray')
            #    ax[1, t].imshow(cropped_unlabeled[t, 0].cpu(), cmap='gray')
            #    ax[2, t].imshow(cropped_seg[t, 0].cpu(), cmap='gray')
            #plt.show()
                    
            cropped_unlabeled = NormalizeIntensity()(cropped_unlabeled)

            unlabeled_list.append(cropped_unlabeled)
            target_list.append(cropped_seg)
            target_mask_list.append(target_mask)

        unlabeled = torch.stack(unlabeled_list, dim=1) # T, B, 1, H, W
        target = torch.stack(target_list, dim=1) # T, B, 1, H, W
        target_mask = torch.stack(target_mask_list, dim=1) # T, B
        padding_need = torch.stack(padding_need_list, dim=0) # B, 4
        distances = torch.stack(distances_list, dim=1).float().to('cuda:0') # T - 1, B

        keys = selected_keys

        #print(seg.shape)
        #print(np.unique(seg))
        #matplotlib.use('QtAgg')
        #fig, ax = plt.subplots(1, 2)
        #ax[0].imshow(labeled[3, 0], cmap='gray')
        #ax[1].imshow(seg[3, 0], cmap='gray')
        #plt.show()

        return {'unlabeled':unlabeled, 
                'target': target,
                'padding_need': padding_need,
                'properties': case_properties, 
                "distances": distances,
                'target_mask': target_mask}



class SinusoidalPositionEmbeddings(nn.Module):
    """Positional embeddings"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings.to('cuda:0')


class DataLoaderFlowTrainVariableLength(SlimDataLoaderBase):
    def __init__(self, data, patch_size, final_patch_size, batch_size, unlabeled_dataset, video_length, processor, crop_size, embedding_dim, is_val, oversample_foreground_percent=0.0,
                 memmap_mode="r", pseudo_3d_slices=1, pad_mode="edge",
                 pad_kwargs_data=None, pad_sides=None):
        """
        This is the basic data loader for 2D networks. It uses preprocessed data as produced by my (Fabian) preprocessing.
        You can load the data with load_dataset(folder) where folder is the folder where the npz files are located. If there
        are only npz files present in that folder, the data loader will unpack them on the fly. This may take a while
        and increase CPU usage. Therefore, I advise you to call unpack_dataset(folder) first, which will unpack all npz
        to npy. Don't forget to call delete_npy(folder) after you are done with training?
        Why all the hassle? Well the decathlon dataset is huge. Using npy for everything will consume >1 TB and that is uncool
        given that I (Fabian) will have to store that permanently on /datasets and my local computer. With this strategy all
        data is stored in a compressed format (factor 10 smaller) and only unpacked when needed.
        :param data: get this with load_dataset(folder, stage=0). Plug the return value in here and you are g2g (good to go)
        :param patch_size: what patch size will this data loader return? it is common practice to first load larger
        patches so that a central crop after data augmentation can be done to reduce border artifacts. If unsure, use
        get_patch_size() from data_augmentation.default_data_augmentation
        :param final_patch_size: what will the patch finally be cropped to (after data augmentation)? this is the patch
        size that goes into your network. We need this here because we will pad patients in here so that patches at the
        border of patients are sampled properly
        :param batch_size:
        :param num_batches: how many batches will the data loader produce before stopping? None=endless
        :param seed:
        :param stage: ignore this (Fabian only)
        :param transpose: ignore this
        :param random: sample randomly; CAREFUL! non-random sampling requires batch_size=1, otherwise you will iterate batch_size times over the dataset
        :param pseudo_3d_slices: 7 = 3 below and 3 above the center slice
        """
        super(DataLoaderFlowTrainVariableLength, self).__init__(data, batch_size, None)
        if pad_kwargs_data is None:
            pad_kwargs_data = OrderedDict()
        self.pad_kwargs_data = pad_kwargs_data
        self.pad_mode = pad_mode
        self.is_val = is_val
        self.crop_size = crop_size
        self.pseudo_3d_slices = pseudo_3d_slices
        self.oversample_foreground_percent = oversample_foreground_percent
        self.final_patch_size = final_patch_size
        self.patch_size = patch_size
        self.list_of_keys = list(self._data.keys())
        self.un_list_of_keys = list(unlabeled_dataset.keys())
        self.need_to_pad = np.array(patch_size) - np.array(final_patch_size)
        self.memmap_mode = memmap_mode
        self.video_length = video_length
        self.percent = None
        if pad_sides is not None:
            if not isinstance(pad_sides, np.ndarray):
                pad_sides = np.array(pad_sides)
            self.need_to_pad += pad_sides
        self.pad_sides = pad_sides
        self.data_shape, self.seg_shape = self.determine_shapes()
        self.un_data = unlabeled_dataset

        self.preprocessing_transform = self.set_up_preprocessing_pipeline()
        self.pixel_transform, self.spatial_transform = self.set_up_augmentation_pipeline()

        self.processor = processor
        self.embedder = SinusoidalPositionEmbeddings(embedding_dim)

    def determine_shapes(self):
        num_seg = 1

        k = list(self._data.keys())[0]
        if isfile(self._data[k]['data_file'][:-4] + ".npy"):
            case_all_data = np.load(self._data[k]['data_file'][:-4] + ".npy", self.memmap_mode)
        else:
            case_all_data = np.load(self._data[k]['data_file'])['data']
        num_color_channels = case_all_data.shape[0] - num_seg
        data_shape = (self.batch_size, num_color_channels, *self.patch_size)
        seg_shape = (self.batch_size, num_seg, *self.patch_size)
        return data_shape, seg_shape

    def get_do_oversample(self, batch_idx):
        return not batch_idx < round(self.batch_size * (1 - self.oversample_foreground_percent))
    
    def select_deep_supervision(self, x):
        if self.deep_supervision:
            return x[0]
        else:
            return x
        

    def my_crop(self, labeled, seg, unlabeled):

        xm = labeled.shape[-1] / 2
        ym = labeled.shape[-2] / 2

        x1 = int(round(xm - self.final_patch_size[1] / 2))
        x2 = int(round(xm + self.final_patch_size[1] / 2))
        y1 = int(round(ym - self.final_patch_size[0] / 2))
        y2 = int(round(ym + self.final_patch_size[0] / 2))

        labeled = labeled[:, :, y1:y2, x1:x2]
        seg = seg[:, :, y1:y2, x1:x2]

        xm = unlabeled.shape[-1] / 2
        ym = unlabeled.shape[-2] / 2

        x1 = int(round(xm - self.final_patch_size[1] / 2))
        x2 = int(round(xm + self.final_patch_size[1] / 2))
        y1 = int(round(ym - self.final_patch_size[0] / 2))
        y2 = int(round(ym + self.final_patch_size[0] / 2))

        unlabeled = unlabeled[:, :, :, y1:y2, x1:x2]

        return labeled, seg, unlabeled
    
    def preprocess(self, image, mask):
        if mask is not None:
            data = {'image': image, 'mask': mask}
            transformed = self.preprocessing_transform(data)
            image = transformed['image']
            mask = transformed['mask']
        else:
            data = {'image': image}
            transformed = self.preprocessing_transform(data)
            image = transformed['image']

        return image, mask
    
    
    def augment(self, image, mask, seed):
        self.pixel_transform = self.pixel_transform.set_random_state(seed=seed)
        self.spatial_transform = self.spatial_transform.set_random_state(seed=seed)
        
        temp = torch.clone(image)
        padding_mask = image == 0

        if mask is not None:
            data = {'image': image, 'mask': mask}
            pixel_transformed = self.pixel_transform(data)
            image = pixel_transformed['image']
            mask = pixel_transformed['mask']
            image[padding_mask] = 0
            data = {'image': image, 'mask': mask}
            spatial_transformed = self.spatial_transform(data)
            image = spatial_transformed['image']
            mask = spatial_transformed['mask']

            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(1, 3)
            #ax[0].imshow(temp[0].cpu(), cmap='gray')
            #ax[1].imshow(image[0].cpu(), cmap='gray')
            #ax[2].imshow(mask[0].cpu(), cmap='gray')
            #plt.show()
        else:
            data = {'image': image}
            pixel_transformed = self.pixel_transform(data)
            image = pixel_transformed['image']
            image[padding_mask] = 0
            data = {'image': image}
            spatial_transformed = self.spatial_transform(data)
            image = spatial_transformed['image']

            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(1, 2)
            #ax[0].imshow(temp[0].cpu(), cmap='gray')
            #ax[1].imshow(image[0].cpu(), cmap='gray')
            #plt.show()

        return image, mask
    
    def set_up_preprocessing_pipeline(self):
        preprocess = Compose([
                            T.SpatialPadd(keys=['image', 'mask'], spatial_size=self.final_patch_size, allow_missing_keys=True),
                            T.CenterSpatialCropd(keys=['image', 'mask'], roi_size=self.final_patch_size, allow_missing_keys=True),
                            T.ToTensor(dtype=torch.float32, device='cuda:0')
                            ])

        return preprocess
    
    def set_up_augmentation_pipeline(self):
        spatial_transform = Compose([
                                    T.RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=0, allow_missing_keys=True),
                                    T.RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=1, allow_missing_keys=True),
                                    T.RandRotated(keys=['image', 'mask'], prob=0.2, range_x=np.pi, mode=["bilinear", "nearest"], padding_mode="zeros", allow_missing_keys=True),
                                    T.RandZoomd(keys=['image', 'mask'], prob=0.2, min_zoom=0.5, max_zoom=1.5, mode=["bilinear", "nearest"], padding_mode="constant", allow_missing_keys=True),
                                    T.RandAffined(keys=['image', 'mask'], prob=0.2, translate_range=(-26, 26), mode=["bilinear", "nearest"], padding_mode="zeros", allow_missing_keys=True)
                                    ])

        pixel_transform = Compose([
                                #RandInvertd(keys=['image'], prob=0.5),
                                T.RandAdjustContrastd(keys=['image'], prob=0.2, gamma=(0.7, 1.5), allow_missing_keys=True),
                                T.RandGaussianNoised(keys=['image'], prob=0.2, std=0.04, allow_missing_keys=True),
                                T.RandScaleIntensityd(keys=['image'], prob=0.2, factors=0.2, allow_missing_keys=True),
                                T.RandGaussianSmoothd(keys=['image'], prob=0.2, sigma_x=(0.25, 0.5), sigma_y=(0.25, 0.5), allow_missing_keys=True),
                                T.RandGaussianSharpend(keys=['image'], prob=0.2, sigma1_x=(0.1, 0.2), sigma1_y=(0.1, 0.2), sigma2_x=(0.2, 0.4), sigma2_y=(0.2, 0.4), alpha=(2.0, 3.0), allow_missing_keys=True)
                                ])

        return pixel_transform, spatial_transform

    def generate_train_batch(self):
        #eval_idx = None
        selected_keys = np.random.choice(self.list_of_keys, self.batch_size, True, None)
        list_of_frames = []
        for i in range(len(selected_keys)):
            patient_id = selected_keys[i][:10]
            l_filtered = [x for x in self.list_of_keys if x[:10] in patient_id]
            un_filtered = [x for x in self.un_list_of_keys if x[:10] in patient_id]
            filtered = l_filtered + un_filtered
            list_of_frames.append(filtered)
        
        target_list = []
        unlabeled_list = []
        padding_need_list = []
        target_mask_list = []
        embedding_list = []

        case_properties = []
        for j, frames in enumerate(list_of_frames):
            labeled_frame = frames[0]
            if 'properties' in self._data[labeled_frame].keys():
                properties = self._data[labeled_frame]['properties']
            else:
                properties = load_pickle(self._data[labeled_frame]['properties_file'])
            case_properties.append(properties)

            depth = properties['size_after_resampling'][0]
            depth_idx = np.random.choice(depth)

            frames = sorted(frames, key=lambda x: int(x[16:18]))
            frames = np.array(frames)
            global_labeled_idx = np.where(~np.char.endswith(frames, '_u'))[0]
            assert len(global_labeled_idx) == 2

            current_video_length = np.random.randint(2, self.video_length + 1)
            current_embedding = self.embedder(torch.LongTensor([current_video_length]))[0]
            embedding_list.append(current_embedding)

            unlabeled = torch.full(size=(current_video_length, 1, *self.final_patch_size), fill_value=torch.nan, device='cuda:0')
            seg = torch.full(size=(current_video_length, 1, *self.final_patch_size), fill_value=torch.nan, device='cuda:0')
            cropped_unlabeled = torch.full(size=(current_video_length, 1, self.crop_size, self.crop_size), fill_value=torch.nan, device='cuda:0')
            cropped_seg = torch.full(size=(current_video_length, 1, self.crop_size, self.crop_size), fill_value=torch.nan, device='cuda:0')
            target_mask = torch.zeros(size=(current_video_length - 2,), dtype=bool, device='cuda:0')

            possible_indices = np.arange(0, len(frames))
            possible_indices = set(global_labeled_idx).symmetric_difference(possible_indices)
            possible_indices = np.array(list(possible_indices))

            random_indices = np.random.choice(possible_indices, size=current_video_length - 2)
            frame_indices = np.concatenate([global_labeled_idx.reshape((-1,)), random_indices])
            sorted_indices = np.argsort(frame_indices)
            frame_indices = frame_indices[sorted_indices]
            target_mask = torch.cat([torch.tensor([True, True], device='cuda:0').reshape((-1,)), target_mask])
            target_mask = target_mask[sorted_indices]

            before_where = np.argwhere(frame_indices < global_labeled_idx[0]).reshape(-1,)
            after_where = np.argwhere(frame_indices >= global_labeled_idx[0]).reshape(-1,)

            before_indices = frame_indices[before_where]
            after_indices = frame_indices[after_where]

            before_mask = target_mask[before_where]
            after_mask = target_mask[after_where]

            frame_indices = np.concatenate([after_indices, before_indices])
            target_mask = torch.cat([after_mask, before_mask])
            assert frame_indices[0] == global_labeled_idx[0]

            assert len(frame_indices) == current_video_length
            video = frames[frame_indices]

            labeled_idx = np.where(~np.char.endswith(video, '_u'))[0]

            for frame_idx, t in enumerate(video):

                if '_u' in t:
                    if not isfile(self.un_data[t]['data_file'][:-4] + ".npy"):
                        # lets hope you know what you're doing
                        case_all_data = np.load(self.un_data[t]['data_file'][:-4] + ".npz")['data']
                    else:
                        case_all_data = np.load(self.un_data[t]['data_file'][:-4] + ".npy", self.memmap_mode)
                else:
                    if not isfile(self._data[t]['data_file'][:-4] + ".npy"):
                        # lets hope you know what you're doing
                        case_all_data = np.load(self._data[t]['data_file'][:-4] + ".npz")['data']
                    else:
                        case_all_data = np.load(self._data[t]['data_file'][:-4] + ".npy", self.memmap_mode)

                # this is for when there is just a 2d slice in case_all_data (2d support)
                if len(case_all_data.shape) == 3:
                    case_all_data = case_all_data[:, None]
                
                assert case_all_data.shape[1] == depth

                # now crop case_all_data to contain just the slice of interest. If we want additional slice above and
                # below the current slice, here is where we get them. We stack those as additional color channels
                slice_data = case_all_data[:, depth_idx]

                # case all data should now be (c, x, y)
                assert len(slice_data.shape) == 3

                if '_u' not in t:
                    image = slice_data[:-1].copy()
                    mask = slice_data[-1:].copy()
                    mask[mask < 0] = 0

                    image, mask = self.preprocess(image, mask)

                    if target_mask[frame_idx] == True:
                        seg[frame_idx] = mask
                        assert frame_idx in labeled_idx
                else:
                    image = slice_data.copy()

                    image, _ = self.preprocess(image, None)

                unlabeled[frame_idx] = image

            with torch.no_grad():
                mean_centroid, _ = self.processor.preprocess_no_registration(data=torch.clone(unlabeled)) # T, C(1), H, W

                cropped_unlabeled, padding_need = self.processor.crop_and_pad(data=unlabeled, mean_centroid=mean_centroid)
                padding_need_list.append(padding_need)

                cropped_seg, _ = self.processor.crop_and_pad(data=seg, mean_centroid=mean_centroid)

            cropped_unlabeled = (cropped_unlabeled - cropped_unlabeled.min()) / (cropped_unlabeled.max() - cropped_unlabeled.min() + 1e-8)

            assert torch.count_nonzero(torch.any(~torch.isnan(cropped_seg.reshape(current_video_length, -1)), dim=-1)) == 2
            assert target_mask[0] == True

            if not self.is_val:
                temp_before_augment = torch.clone(cropped_unlabeled).to('cpu')
                
                seed = random.randint(0, 2**32-1)

                for t in range(len(cropped_unlabeled)):
                    cropped_unlabeled[t], cropped_seg[t] = self.augment(image=cropped_unlabeled[t], mask=cropped_seg[t], seed=seed)

            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(3, len(cropped_unlabeled))
            #for t in range(len(cropped_unlabeled)):
            #    ax[0, t].imshow(temp_before_augment[t, 0].cpu(), cmap='gray')
            #    ax[1, t].imshow(cropped_unlabeled[t, 0].cpu(), cmap='gray')
            #    ax[2, t].imshow(cropped_seg[t, 0].cpu(), cmap='gray')
            #plt.show()
                    
            cropped_unlabeled = NormalizeIntensity()(cropped_unlabeled)

            unlabeled_list.append(cropped_unlabeled)
            target_list.append(cropped_seg)
            target_mask_list.append(target_mask)

        unlabeled = torch.stack(unlabeled_list, dim=1) # T, B, 1, H, W
        target = torch.stack(target_list, dim=1) # T, B, 1, H, W
        target_mask = torch.stack(target_mask_list, dim=1) # T, B
        padding_need = torch.stack(padding_need_list, dim=0) # B, 4
        embedding = torch.stack(embedding_list, dim=0) # B, embedding_dim

        keys = selected_keys

        #print(seg.shape)
        #print(np.unique(seg))
        #matplotlib.use('QtAgg')
        #fig, ax = plt.subplots(1, 2)
        #ax[0].imshow(labeled[3, 0], cmap='gray')
        #ax[1].imshow(seg[3, 0], cmap='gray')
        #plt.show()

        return {'unlabeled':unlabeled, 
                'target': target,
                'embedding': embedding,
                'padding_need': padding_need,
                'properties': case_properties, 
                "keys": keys,
                'target_mask': target_mask}
    


class DataLoaderFlowTrainPredictionVal(SlimDataLoaderBase):
    def __init__(self, data, patch_size, final_patch_size, batch_size, unlabeled_dataset, video_length, processor, crop_size, is_val, oversample_foreground_percent=0.0,
                 memmap_mode="r", pseudo_3d_slices=1, pad_mode="edge",
                 pad_kwargs_data=None, pad_sides=None):
        """
        This is the basic data loader for 2D networks. It uses preprocessed data as produced by my (Fabian) preprocessing.
        You can load the data with load_dataset(folder) where folder is the folder where the npz files are located. If there
        are only npz files present in that folder, the data loader will unpack them on the fly. This may take a while
        and increase CPU usage. Therefore, I advise you to call unpack_dataset(folder) first, which will unpack all npz
        to npy. Don't forget to call delete_npy(folder) after you are done with training?
        Why all the hassle? Well the decathlon dataset is huge. Using npy for everything will consume >1 TB and that is uncool
        given that I (Fabian) will have to store that permanently on /datasets and my local computer. With this strategy all
        data is stored in a compressed format (factor 10 smaller) and only unpacked when needed.
        :param data: get this with load_dataset(folder, stage=0). Plug the return value in here and you are g2g (good to go)
        :param patch_size: what patch size will this data loader return? it is common practice to first load larger
        patches so that a central crop after data augmentation can be done to reduce border artifacts. If unsure, use
        get_patch_size() from data_augmentation.default_data_augmentation
        :param final_patch_size: what will the patch finally be cropped to (after data augmentation)? this is the patch
        size that goes into your network. We need this here because we will pad patients in here so that patches at the
        border of patients are sampled properly
        :param batch_size:
        :param num_batches: how many batches will the data loader produce before stopping? None=endless
        :param seed:
        :param stage: ignore this (Fabian only)
        :param transpose: ignore this
        :param random: sample randomly; CAREFUL! non-random sampling requires batch_size=1, otherwise you will iterate batch_size times over the dataset
        :param pseudo_3d_slices: 7 = 3 below and 3 above the center slice
        """
        super(DataLoaderFlowTrainPredictionVal, self).__init__(data, batch_size, None)
        if pad_kwargs_data is None:
            pad_kwargs_data = OrderedDict()
        self.pad_kwargs_data = pad_kwargs_data
        self.pad_mode = pad_mode
        self.is_val = is_val
        self.crop_size = crop_size
        self.pseudo_3d_slices = pseudo_3d_slices
        self.oversample_foreground_percent = oversample_foreground_percent
        self.final_patch_size = final_patch_size
        self.patch_size = patch_size
        self.list_of_keys = list(self._data.keys())
        self.un_list_of_keys = list(unlabeled_dataset.keys())
        self.need_to_pad = np.array(patch_size) - np.array(final_patch_size)
        self.memmap_mode = memmap_mode
        self.video_length = video_length
        self.percent = None
        if pad_sides is not None:
            if not isinstance(pad_sides, np.ndarray):
                pad_sides = np.array(pad_sides)
            self.need_to_pad += pad_sides
        self.pad_sides = pad_sides
        self.data_shape, self.seg_shape = self.determine_shapes()
        self.un_data = unlabeled_dataset

        self.preprocessing_transform = self.set_up_preprocessing_pipeline()
        self.pixel_transform, self.spatial_transform = self.set_up_augmentation_pipeline()

        self.processor = processor

    def determine_shapes(self):
        num_seg = 1

        k = list(self._data.keys())[0]
        if isfile(self._data[k]['data_file'][:-4] + ".npy"):
            case_all_data = np.load(self._data[k]['data_file'][:-4] + ".npy", self.memmap_mode)
        else:
            case_all_data = np.load(self._data[k]['data_file'])['data']
        num_color_channels = case_all_data.shape[0] - num_seg
        data_shape = (self.batch_size, num_color_channels, *self.patch_size)
        seg_shape = (self.batch_size, num_seg, *self.patch_size)
        return data_shape, seg_shape

    def get_do_oversample(self, batch_idx):
        return not batch_idx < round(self.batch_size * (1 - self.oversample_foreground_percent))
    
    def select_deep_supervision(self, x):
        if self.deep_supervision:
            return x[0]
        else:
            return x
        

    def my_crop(self, labeled, seg, unlabeled):

        xm = labeled.shape[-1] / 2
        ym = labeled.shape[-2] / 2

        x1 = int(round(xm - self.final_patch_size[1] / 2))
        x2 = int(round(xm + self.final_patch_size[1] / 2))
        y1 = int(round(ym - self.final_patch_size[0] / 2))
        y2 = int(round(ym + self.final_patch_size[0] / 2))

        labeled = labeled[:, :, y1:y2, x1:x2]
        seg = seg[:, :, y1:y2, x1:x2]

        xm = unlabeled.shape[-1] / 2
        ym = unlabeled.shape[-2] / 2

        x1 = int(round(xm - self.final_patch_size[1] / 2))
        x2 = int(round(xm + self.final_patch_size[1] / 2))
        y1 = int(round(ym - self.final_patch_size[0] / 2))
        y2 = int(round(ym + self.final_patch_size[0] / 2))

        unlabeled = unlabeled[:, :, :, y1:y2, x1:x2]

        return labeled, seg, unlabeled
    
    def preprocess(self, image, mask):
        if mask is not None:
            data = {'image': image, 'mask': mask}
            transformed = self.preprocessing_transform(data)
            image = transformed['image']
            mask = transformed['mask']
        else:
            data = {'image': image}
            transformed = self.preprocessing_transform(data)
            image = transformed['image']

        return image, mask
    
    
    def augment(self, image, mask, seed):
        self.pixel_transform = self.pixel_transform.set_random_state(seed=seed)
        self.spatial_transform = self.spatial_transform.set_random_state(seed=seed)
        
        temp = torch.clone(image)
        padding_mask = image == 0

        if mask is not None:
            data = {'image': image, 'mask': mask}
            pixel_transformed = self.pixel_transform(data)
            image = pixel_transformed['image']
            mask = pixel_transformed['mask']
            image[padding_mask] = 0
            data = {'image': image, 'mask': mask}
            spatial_transformed = self.spatial_transform(data)
            image = spatial_transformed['image']
            mask = spatial_transformed['mask']

            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(1, 3)
            #ax[0].imshow(temp[0].cpu(), cmap='gray')
            #ax[1].imshow(image[0].cpu(), cmap='gray')
            #ax[2].imshow(mask[0].cpu(), cmap='gray')
            #plt.show()
        else:
            data = {'image': image}
            pixel_transformed = self.pixel_transform(data)
            image = pixel_transformed['image']
            image[padding_mask] = 0
            data = {'image': image}
            spatial_transformed = self.spatial_transform(data)
            image = spatial_transformed['image']

            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(1, 2)
            #ax[0].imshow(temp[0].cpu(), cmap='gray')
            #ax[1].imshow(image[0].cpu(), cmap='gray')
            #plt.show()

        return image, mask
    
    def set_up_preprocessing_pipeline(self):
        preprocess = Compose([
                            T.SpatialPadd(keys=['image', 'mask'], spatial_size=self.final_patch_size, allow_missing_keys=True),
                            T.CenterSpatialCropd(keys=['image', 'mask'], roi_size=self.final_patch_size, allow_missing_keys=True),
                            T.ToTensor(dtype=torch.float32, device='cuda:0')
                            ])

        return preprocess
    
    def set_up_augmentation_pipeline(self):
        spatial_transform = Compose([
                                    T.RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=0, allow_missing_keys=True),
                                    T.RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=1, allow_missing_keys=True),
                                    T.RandRotated(keys=['image', 'mask'], prob=0.2, range_x=np.pi, mode=["bilinear", "nearest"], padding_mode="zeros", allow_missing_keys=True),
                                    T.RandZoomd(keys=['image', 'mask'], prob=0.2, min_zoom=0.5, max_zoom=1.5, mode=["bilinear", "nearest"], padding_mode="constant", allow_missing_keys=True),
                                    T.RandAffined(keys=['image', 'mask'], prob=0.2, translate_range=(-26, 26), mode=["bilinear", "nearest"], padding_mode="zeros", allow_missing_keys=True)
                                    ])

        pixel_transform = Compose([
                                #RandInvertd(keys=['image'], prob=0.5),
                                T.RandAdjustContrastd(keys=['image'], prob=0.2, gamma=(0.7, 1.5), allow_missing_keys=True),
                                T.RandGaussianNoised(keys=['image'], prob=0.2, std=0.04, allow_missing_keys=True),
                                T.RandScaleIntensityd(keys=['image'], prob=0.2, factors=0.2, allow_missing_keys=True),
                                T.RandGaussianSmoothd(keys=['image'], prob=0.2, sigma_x=(0.25, 0.5), sigma_y=(0.25, 0.5), allow_missing_keys=True),
                                T.RandGaussianSharpend(keys=['image'], prob=0.2, sigma1_x=(0.1, 0.2), sigma1_y=(0.1, 0.2), sigma2_x=(0.2, 0.4), sigma2_y=(0.2, 0.4), alpha=(2.0, 3.0), allow_missing_keys=True)
                                ])

        return pixel_transform, spatial_transform

    def generate_train_batch(self):
        selected_keys = np.random.choice(self.list_of_keys, self.batch_size, True, None)
        #selected_keys = np.random.choice(self.list_of_keys, 1, True, None)
        list_of_frames = []
        for i in range(len(selected_keys)):
            patient_id = selected_keys[i][:10]
            l_filtered = [x for x in self.list_of_keys if x[:10] in patient_id]
            un_filtered = [x for x in self.un_list_of_keys if x[:10] in patient_id]
            filtered = l_filtered + un_filtered
            list_of_frames.append(filtered)
        
        target_list = []
        unlabeled_list = []
        padding_need_list = []
        target_mask_list = []
        distances_list = []

        case_properties = []
        for j, frames in enumerate(list_of_frames):
            labeled_frame = frames[0]
            if 'properties' in self._data[labeled_frame].keys():
                properties = self._data[labeled_frame]['properties']
            else:
                properties = load_pickle(self._data[labeled_frame]['properties_file'])
            case_properties.append(properties)

            depth = properties['size_after_resampling'][0]
            depth_idx = np.random.choice(depth)

            frames = sorted(frames, key=lambda x: int(x[16:18]))
            frames = np.array(frames)
            global_labeled_idx = np.where(~np.char.endswith(frames, '_u'))[0]
            assert len(global_labeled_idx) == 2

            
            target_mask = torch.zeros(size=(len(frames),), dtype=bool, device='cuda:0')

            frame_indices = np.arange(len(frames))
            before_where = np.argwhere(frame_indices < global_labeled_idx[0]).reshape(-1,)
            after_where = np.argwhere(frame_indices >= global_labeled_idx[0]).reshape(-1,)

            before_indices = frame_indices[before_where]
            after_indices = frame_indices[after_where]

            before_mask = target_mask[before_where]
            after_mask = target_mask[after_where]

            frame_indices = np.concatenate([after_indices, before_indices])
            stop_idx = np.where(frame_indices == global_labeled_idx[1])[0][0]
            frame_indices = frame_indices[:stop_idx + 1]
            target_mask = torch.cat([after_mask, before_mask])
            target_mask = target_mask[:stop_idx + 1]
            assert frame_indices[0] == global_labeled_idx[0]

            distances_list.append(np.nan)
            video = frames[frame_indices]

            unlabeled = torch.full(size=(len(video), 1, *self.final_patch_size), fill_value=torch.nan, device='cuda:0')
            seg = torch.full(size=(len(video), 1, *self.final_patch_size), fill_value=torch.nan, device='cuda:0')
            cropped_unlabeled = torch.full(size=(len(video), 1, self.crop_size, self.crop_size), fill_value=torch.nan, device='cuda:0')
            cropped_seg = torch.full(size=(len(video), 1, self.crop_size, self.crop_size), fill_value=torch.nan, device='cuda:0')

            labeled_idx = np.where(~np.char.endswith(video, '_u'))[0]

            for frame_idx, t in enumerate(video):

                if '_u' in t:
                    if not isfile(self.un_data[t]['data_file'][:-4] + ".npy"):
                        # lets hope you know what you're doing
                        case_all_data = np.load(self.un_data[t]['data_file'][:-4] + ".npz")['data']
                    else:
                        case_all_data = np.load(self.un_data[t]['data_file'][:-4] + ".npy", self.memmap_mode)
                else:
                    if not isfile(self._data[t]['data_file'][:-4] + ".npy"):
                        # lets hope you know what you're doing
                        case_all_data = np.load(self._data[t]['data_file'][:-4] + ".npz")['data']
                    else:
                        case_all_data = np.load(self._data[t]['data_file'][:-4] + ".npy", self.memmap_mode)

                # this is for when there is just a 2d slice in case_all_data (2d support)
                if len(case_all_data.shape) == 3:
                    case_all_data = case_all_data[:, None]
                
                assert case_all_data.shape[1] == depth

                # now crop case_all_data to contain just the slice of interest. If we want additional slice above and
                # below the current slice, here is where we get them. We stack those as additional color channels
                slice_data = case_all_data[:, depth_idx]

                # case all data should now be (c, x, y)
                assert len(slice_data.shape) == 3

                if '_u' not in t:
                    image = slice_data[:-1].copy()
                    mask = slice_data[-1:].copy()
                    mask[mask < 0] = 0

                    image, mask = self.preprocess(image, mask)

                    seg[frame_idx] = mask
                    target_mask[frame_idx] = True
                else:
                    image = slice_data.copy()

                    image, _ = self.preprocess(image, None)

                unlabeled[frame_idx] = image

            with torch.no_grad():
                mean_centroid, _ = self.processor.preprocess_no_registration(data=torch.clone(unlabeled)) # T, C(1), H, W

                cropped_unlabeled, padding_need = self.processor.crop_and_pad(data=unlabeled, mean_centroid=mean_centroid)
                padding_need_list.append(padding_need)

                cropped_seg, _ = self.processor.crop_and_pad(data=seg, mean_centroid=mean_centroid)

            cropped_unlabeled = (cropped_unlabeled - cropped_unlabeled.min()) / (cropped_unlabeled.max() - cropped_unlabeled.min() + 1e-8)

            assert target_mask[0] == True
            assert torch.count_nonzero(target_mask) == 2
            assert torch.count_nonzero(torch.any(~torch.isnan(cropped_seg.reshape(len(video), -1)), dim=-1)) == 2, torch.count_nonzero(torch.any(~torch.isnan(cropped_seg.reshape(len(video), -1)), dim=-1))

            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(3, len(cropped_unlabeled))
            #for t in range(len(cropped_unlabeled)):
            #    ax[0, t].imshow(temp_before_augment[t, 0].cpu(), cmap='gray')
            #    ax[1, t].imshow(cropped_unlabeled[t, 0].cpu(), cmap='gray')
            #    ax[2, t].imshow(cropped_seg[t, 0].cpu(), cmap='gray')
            #plt.show()
                    
            cropped_unlabeled = NormalizeIntensity()(cropped_unlabeled)

            unlabeled_list.append(cropped_unlabeled)
            target_list.append(cropped_seg)
            target_mask_list.append(target_mask)

        max_length = max([unlabeled_list[i].shape[0] for i in range(len(unlabeled_list))])
        unlabeled_list = [torch.nn.functional.pad(unlabeled_list[i], (0, 0, 0, 0, 0, 0, 0, max_length - len(unlabeled_list[i]))) for i in range(len(unlabeled_list))]
        target_list = [torch.nn.functional.pad(target_list[i], (0, 0, 0, 0, 0, 0, 0, max_length - len(target_list[i]))) for i in range(len(target_list))]
        target_mask_list = [torch.nn.functional.pad(target_mask_list[i], (0, max_length - len(target_mask_list[i]))) for i in range(len(target_mask_list))]

        unlabeled = torch.stack(unlabeled_list, dim=1) # T, B, 1, H, W
        target = torch.stack(target_list, dim=1) # T, B, 1, H, W
        target_mask = torch.stack(target_mask_list, dim=1) # T, B
        padding_need = torch.stack(padding_need_list, dim=0) # B, 4
        distances = torch.stack(distances_list, dim=1).float().to('cuda:0') # T - 1, B

        #matplotlib.use('QtAgg')
        #fig, ax = plt.subplots(1, len(target))
        #for t in range(len(target)):
        #    ax[t].imshow(target[t, 0, 0].cpu(), cmap='gray')
        #plt.show()

        keys = selected_keys

        #print(seg.shape)
        #print(np.unique(seg))
        #matplotlib.use('QtAgg')
        #fig, ax = plt.subplots(1, 2)
        #ax[0].imshow(labeled[3, 0], cmap='gray')
        #ax[1].imshow(seg[3, 0], cmap='gray')
        #plt.show()

        return {'unlabeled':unlabeled, 
                'target': target,
                'padding_need': padding_need,
                'properties': case_properties, 
                "distances": distances,
                'target_mask': target_mask}
    



class DataLoaderFlowTrainPredictionValLib(SlimDataLoaderBase):
    def __init__(self, data, patch_size, final_patch_size, batch_size, video_length, processor, crop_size, is_val, oversample_foreground_percent=0.0,
                 memmap_mode="r", pseudo_3d_slices=1, pad_mode="edge",
                 pad_kwargs_data=None, pad_sides=None):
        """
        This is the basic data loader for 2D networks. It uses preprocessed data as produced by my (Fabian) preprocessing.
        You can load the data with load_dataset(folder) where folder is the folder where the npz files are located. If there
        are only npz files present in that folder, the data loader will unpack them on the fly. This may take a while
        and increase CPU usage. Therefore, I advise you to call unpack_dataset(folder) first, which will unpack all npz
        to npy. Don't forget to call delete_npy(folder) after you are done with training?
        Why all the hassle? Well the decathlon dataset is huge. Using npy for everything will consume >1 TB and that is uncool
        given that I (Fabian) will have to store that permanently on /datasets and my local computer. With this strategy all
        data is stored in a compressed format (factor 10 smaller) and only unpacked when needed.
        :param data: get this with load_dataset(folder, stage=0). Plug the return value in here and you are g2g (good to go)
        :param patch_size: what patch size will this data loader return? it is common practice to first load larger
        patches so that a central crop after data augmentation can be done to reduce border artifacts. If unsure, use
        get_patch_size() from data_augmentation.default_data_augmentation
        :param final_patch_size: what will the patch finally be cropped to (after data augmentation)? this is the patch
        size that goes into your network. We need this here because we will pad patients in here so that patches at the
        border of patients are sampled properly
        :param batch_size:
        :param num_batches: how many batches will the data loader produce before stopping? None=endless
        :param seed:
        :param stage: ignore this (Fabian only)
        :param transpose: ignore this
        :param random: sample randomly; CAREFUL! non-random sampling requires batch_size=1, otherwise you will iterate batch_size times over the dataset
        :param pseudo_3d_slices: 7 = 3 below and 3 above the center slice
        """
        super(DataLoaderFlowTrainPredictionValLib, self).__init__(data, batch_size, None)
        if pad_kwargs_data is None:
            pad_kwargs_data = OrderedDict()
        self.pad_kwargs_data = pad_kwargs_data
        self.pad_mode = pad_mode
        self.is_val = is_val
        self.crop_size = crop_size
        self.pseudo_3d_slices = pseudo_3d_slices
        self.oversample_foreground_percent = oversample_foreground_percent
        self.final_patch_size = final_patch_size
        self.patch_size = patch_size
        self.list_of_keys = list(self._data.keys())
        self.need_to_pad = np.array(patch_size) - np.array(final_patch_size)
        self.memmap_mode = memmap_mode
        self.video_length = video_length
        self.percent = None
        if pad_sides is not None:
            if not isinstance(pad_sides, np.ndarray):
                pad_sides = np.array(pad_sides)
            self.need_to_pad += pad_sides
        self.pad_sides = pad_sides
        self.data_shape, self.seg_shape = self.determine_shapes()

        self.preprocessing_transform = self.set_up_preprocessing_pipeline()
        self.pixel_transform, self.spatial_transform = self.set_up_augmentation_pipeline()

        self.processor = processor

    def determine_shapes(self):
        num_seg = 1

        k = list(self._data.keys())[0]
        if isfile(self._data[k]['data_file'][:-4] + ".npy"):
            case_all_data = np.load(self._data[k]['data_file'][:-4] + ".npy", self.memmap_mode)
        else:
            case_all_data = np.load(self._data[k]['data_file'])['data']
        num_color_channels = case_all_data.shape[0] - num_seg
        data_shape = (self.batch_size, num_color_channels, *self.patch_size)
        seg_shape = (self.batch_size, num_seg, *self.patch_size)
        return data_shape, seg_shape

    def get_do_oversample(self, batch_idx):
        return not batch_idx < round(self.batch_size * (1 - self.oversample_foreground_percent))
    
    def select_deep_supervision(self, x):
        if self.deep_supervision:
            return x[0]
        else:
            return x
        

    def my_crop(self, labeled, seg, unlabeled):

        xm = labeled.shape[-1] / 2
        ym = labeled.shape[-2] / 2

        x1 = int(round(xm - self.final_patch_size[1] / 2))
        x2 = int(round(xm + self.final_patch_size[1] / 2))
        y1 = int(round(ym - self.final_patch_size[0] / 2))
        y2 = int(round(ym + self.final_patch_size[0] / 2))

        labeled = labeled[:, :, y1:y2, x1:x2]
        seg = seg[:, :, y1:y2, x1:x2]

        xm = unlabeled.shape[-1] / 2
        ym = unlabeled.shape[-2] / 2

        x1 = int(round(xm - self.final_patch_size[1] / 2))
        x2 = int(round(xm + self.final_patch_size[1] / 2))
        y1 = int(round(ym - self.final_patch_size[0] / 2))
        y2 = int(round(ym + self.final_patch_size[0] / 2))

        unlabeled = unlabeled[:, :, :, y1:y2, x1:x2]

        return labeled, seg, unlabeled
    
    def preprocess(self, image, mask):
        if mask is not None:
            data = {'image': image, 'mask': mask}
            transformed = self.preprocessing_transform(data)
            image = transformed['image']
            mask = transformed['mask']
        else:
            data = {'image': image}
            transformed = self.preprocessing_transform(data)
            image = transformed['image']

        return image, mask
    
    
    def augment(self, image, mask, seed):
        self.pixel_transform = self.pixel_transform.set_random_state(seed=seed)
        self.spatial_transform = self.spatial_transform.set_random_state(seed=seed)
        
        temp = torch.clone(image)
        padding_mask = image == 0

        if mask is not None:
            data = {'image': image, 'mask': mask}
            pixel_transformed = self.pixel_transform(data)
            image = pixel_transformed['image']
            mask = pixel_transformed['mask']
            image[padding_mask] = 0
            data = {'image': image, 'mask': mask}
            spatial_transformed = self.spatial_transform(data)
            image = spatial_transformed['image']
            mask = spatial_transformed['mask']

            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(1, 3)
            #ax[0].imshow(temp[0].cpu(), cmap='gray')
            #ax[1].imshow(image[0].cpu(), cmap='gray')
            #ax[2].imshow(mask[0].cpu(), cmap='gray')
            #plt.show()
        else:
            data = {'image': image}
            pixel_transformed = self.pixel_transform(data)
            image = pixel_transformed['image']
            image[padding_mask] = 0
            data = {'image': image}
            spatial_transformed = self.spatial_transform(data)
            image = spatial_transformed['image']

            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(1, 2)
            #ax[0].imshow(temp[0].cpu(), cmap='gray')
            #ax[1].imshow(image[0].cpu(), cmap='gray')
            #plt.show()

        return image, mask
    
    def set_up_preprocessing_pipeline(self):
        preprocess = Compose([
                            T.SpatialPadd(keys=['image', 'mask'], spatial_size=self.final_patch_size, allow_missing_keys=True),
                            T.CenterSpatialCropd(keys=['image', 'mask'], roi_size=self.final_patch_size, allow_missing_keys=True),
                            T.ToTensor(dtype=torch.float32, device='cuda:0')
                            ])

        return preprocess
    
    def set_up_augmentation_pipeline(self):
        spatial_transform = Compose([
                                    T.RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=0, allow_missing_keys=True),
                                    T.RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=1, allow_missing_keys=True),
                                    T.RandRotated(keys=['image', 'mask'], prob=0.2, range_x=np.pi, mode=["bilinear", "nearest"], padding_mode="zeros", allow_missing_keys=True),
                                    T.RandZoomd(keys=['image', 'mask'], prob=0.2, min_zoom=0.5, max_zoom=1.5, mode=["bilinear", "nearest"], padding_mode="constant", allow_missing_keys=True),
                                    T.RandAffined(keys=['image', 'mask'], prob=0.2, translate_range=(-26, 26), mode=["bilinear", "nearest"], padding_mode="zeros", allow_missing_keys=True)
                                    ])

        pixel_transform = Compose([
                                #RandInvertd(keys=['image'], prob=0.5),
                                T.RandAdjustContrastd(keys=['image'], prob=0.2, gamma=(0.7, 1.5), allow_missing_keys=True),
                                T.RandGaussianNoised(keys=['image'], prob=0.2, std=0.04, allow_missing_keys=True),
                                T.RandScaleIntensityd(keys=['image'], prob=0.2, factors=0.2, allow_missing_keys=True),
                                T.RandGaussianSmoothd(keys=['image'], prob=0.2, sigma_x=(0.25, 0.5), sigma_y=(0.25, 0.5), allow_missing_keys=True),
                                T.RandGaussianSharpend(keys=['image'], prob=0.2, sigma1_x=(0.1, 0.2), sigma1_y=(0.1, 0.2), sigma2_x=(0.2, 0.4), sigma2_y=(0.2, 0.4), alpha=(2.0, 3.0), allow_missing_keys=True)
                                ])

        return pixel_transform, spatial_transform

    def generate_train_batch(self):
        selected_keys = np.random.choice(self.list_of_keys, self.batch_size, True, None)
        #selected_keys = np.random.choice(self.list_of_keys, 1, True, None)
        list_of_frames = []
        for i in range(len(selected_keys)):
            patient_id = selected_keys[i][:10]
            filtered = [x for x in self.list_of_keys if x[:10] in patient_id]
            list_of_frames.append(filtered)
        
        target_list = []
        unlabeled_list = []
        padding_need_list = []
        target_mask_list = []

        case_properties = []
        for j, frames in enumerate(list_of_frames):
            labeled_frame = frames[0]
            if 'properties' in self._data[labeled_frame].keys():
                properties = self._data[labeled_frame]['properties']
            else:
                properties = load_pickle(self._data[labeled_frame]['properties_file'])
            case_properties.append(properties)

            ed_idx = np.rint(np.array(properties['ed_number'])).astype(np.uint8)
            es_idx = np.rint(np.array(properties['es_number'])).astype(np.uint8)

            depth = properties['size_after_resampling'][0]
            depth_idx = np.random.choice(depth)

            global_labeled_idx = np.array([ed_idx, es_idx])

            frames = sorted(frames, key=lambda x: int(x[16:18]))
            frames = np.array(frames)

            unlabeled = torch.full(size=(len(frames), 1, *self.final_patch_size), fill_value=torch.nan, device='cuda:0')
            seg = torch.full(size=(len(frames), 1, *self.final_patch_size), fill_value=torch.nan, device='cuda:0')
            cropped_unlabeled = torch.full(size=(len(frames), 1, self.crop_size, self.crop_size), fill_value=torch.nan, device='cuda:0')
            cropped_seg = torch.full(size=(len(frames), 1, self.crop_size, self.crop_size), fill_value=torch.nan, device='cuda:0')
            target_mask = torch.zeros(size=(len(frames),), dtype=bool, device='cuda:0')

            frame_indices = np.arange(len(frames))
            before_where = np.argwhere(frame_indices < ed_idx).reshape(-1,)
            after_where = np.argwhere(frame_indices >= ed_idx).reshape(-1,)

            before_indices = frame_indices[before_where]
            after_indices = frame_indices[after_where]

            before_mask = target_mask[before_where]
            after_mask = target_mask[after_where]

            frame_indices = np.concatenate([after_indices, before_indices])
            target_mask = torch.cat([after_mask, before_mask])
            assert frame_indices[0] == ed_idx

            assert len(frame_indices) == len(frames)
            video = frames[frame_indices]

            labeled_idx = np.where(np.isin(frame_indices, global_labeled_idx))[0]

            for frame_idx, t in enumerate(video):

                if '_u' in t:
                    if not isfile(self.un_data[t]['data_file'][:-4] + ".npy"):
                        # lets hope you know what you're doing
                        case_all_data = np.load(self.un_data[t]['data_file'][:-4] + ".npz")['data']
                    else:
                        case_all_data = np.load(self.un_data[t]['data_file'][:-4] + ".npy", self.memmap_mode)
                else:
                    if not isfile(self._data[t]['data_file'][:-4] + ".npy"):
                        # lets hope you know what you're doing
                        case_all_data = np.load(self._data[t]['data_file'][:-4] + ".npz")['data']
                    else:
                        case_all_data = np.load(self._data[t]['data_file'][:-4] + ".npy", self.memmap_mode)

                # this is for when there is just a 2d slice in case_all_data (2d support)
                if len(case_all_data.shape) == 3:
                    case_all_data = case_all_data[:, None]
                
                assert case_all_data.shape[1] == depth

                # now crop case_all_data to contain just the slice of interest. If we want additional slice above and
                # below the current slice, here is where we get them. We stack those as additional color channels
                slice_data = case_all_data[:, depth_idx]

                # case all data should now be (c, x, y)
                assert len(slice_data.shape) == 3

                if '_u' not in t:
                    image = slice_data[:-1].copy()
                    mask = slice_data[-1:].copy()
                    mask[mask < 0] = 0

                    image, mask = self.preprocess(image, mask)

                    seg[frame_idx] = mask
                    target_mask[frame_idx] = True
                else:
                    image = slice_data.copy()

                    image, _ = self.preprocess(image, None)

                unlabeled[frame_idx] = image

            with torch.no_grad():
                mean_centroid, _ = self.processor.preprocess_no_registration(data=torch.clone(unlabeled)) # T, C(1), H, W

                cropped_unlabeled, padding_need = self.processor.crop_and_pad(data=unlabeled, mean_centroid=mean_centroid)
                padding_need_list.append(padding_need)

                cropped_seg, _ = self.processor.crop_and_pad(data=seg, mean_centroid=mean_centroid)

            cropped_unlabeled = (cropped_unlabeled - cropped_unlabeled.min()) / (cropped_unlabeled.max() - cropped_unlabeled.min() + 1e-8)

            assert target_mask[0] == True
            assert torch.count_nonzero(target_mask) == len(frames)
            assert torch.count_nonzero(torch.any(~torch.isnan(cropped_seg.reshape(len(frames), -1)), dim=-1)) == len(frames), torch.count_nonzero(torch.any(~torch.isnan(cropped_seg.reshape(len(frames), -1)), dim=-1))

            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(3, len(cropped_unlabeled))
            #for t in range(len(cropped_unlabeled)):
            #    ax[0, t].imshow(temp_before_augment[t, 0].cpu(), cmap='gray')
            #    ax[1, t].imshow(cropped_unlabeled[t, 0].cpu(), cmap='gray')
            #    ax[2, t].imshow(cropped_seg[t, 0].cpu(), cmap='gray')
            #plt.show()
                    
            cropped_unlabeled = NormalizeIntensity()(cropped_unlabeled)

            unlabeled_list.append(cropped_unlabeled)
            target_list.append(cropped_seg)
            target_mask_list.append(target_mask)

        max_length = max([unlabeled_list[i].shape[0] for i in range(len(unlabeled_list))])
        unlabeled_list = [torch.nn.functional.pad(unlabeled_list[i], (0, 0, 0, 0, 0, 0, 0, max_length - len(unlabeled_list[i]))) for i in range(len(unlabeled_list))]
        target_list = [torch.nn.functional.pad(target_list[i], (0, 0, 0, 0, 0, 0, 0, max_length - len(target_list[i]))) for i in range(len(target_list))]
        target_mask_list = [torch.nn.functional.pad(target_mask_list[i], (0, max_length - len(target_mask_list[i]))) for i in range(len(target_mask_list))]

        unlabeled = torch.stack(unlabeled_list, dim=1) # T, B, 1, H, W
        target = torch.stack(target_list, dim=1) # T, B, 1, H, W
        target_mask = torch.stack(target_mask_list, dim=1) # T, B
        padding_need = torch.stack(padding_need_list, dim=0) # B, 4

        #matplotlib.use('QtAgg')
        #fig, ax = plt.subplots(1, len(target))
        #for t in range(len(target)):
        #    ax[t].imshow(target[t, 0, 0].cpu(), cmap='gray')
        #plt.show()

        keys = selected_keys

        #print(seg.shape)
        #print(np.unique(seg))
        #matplotlib.use('QtAgg')
        #fig, ax = plt.subplots(1, 2)
        #ax[0].imshow(labeled[3, 0], cmap='gray')
        #ax[1].imshow(seg[3, 0], cmap='gray')
        #plt.show()

        return {'unlabeled':unlabeled, 
                'target': target,
                'padding_need': padding_need,
                'properties': case_properties, 
                "keys": keys,
                'target_mask': target_mask}
    



class DataLoaderFlowTrainValLib(SlimDataLoaderBase):
    def __init__(self, data, patch_size, final_patch_size, batch_size, video_length, processor, crop_size, is_val, oversample_foreground_percent=0.0,
                 memmap_mode="r", pseudo_3d_slices=1, pad_mode="edge",
                 pad_kwargs_data=None, pad_sides=None):
        """
        This is the basic data loader for 2D networks. It uses preprocessed data as produced by my (Fabian) preprocessing.
        You can load the data with load_dataset(folder) where folder is the folder where the npz files are located. If there
        are only npz files present in that folder, the data loader will unpack them on the fly. This may take a while
        and increase CPU usage. Therefore, I advise you to call unpack_dataset(folder) first, which will unpack all npz
        to npy. Don't forget to call delete_npy(folder) after you are done with training?
        Why all the hassle? Well the decathlon dataset is huge. Using npy for everything will consume >1 TB and that is uncool
        given that I (Fabian) will have to store that permanently on /datasets and my local computer. With this strategy all
        data is stored in a compressed format (factor 10 smaller) and only unpacked when needed.
        :param data: get this with load_dataset(folder, stage=0). Plug the return value in here and you are g2g (good to go)
        :param patch_size: what patch size will this data loader return? it is common practice to first load larger
        patches so that a central crop after data augmentation can be done to reduce border artifacts. If unsure, use
        get_patch_size() from data_augmentation.default_data_augmentation
        :param final_patch_size: what will the patch finally be cropped to (after data augmentation)? this is the patch
        size that goes into your network. We need this here because we will pad patients in here so that patches at the
        border of patients are sampled properly
        :param batch_size:
        :param num_batches: how many batches will the data loader produce before stopping? None=endless
        :param seed:
        :param stage: ignore this (Fabian only)
        :param transpose: ignore this
        :param random: sample randomly; CAREFUL! non-random sampling requires batch_size=1, otherwise you will iterate batch_size times over the dataset
        :param pseudo_3d_slices: 7 = 3 below and 3 above the center slice
        """
        super(DataLoaderFlowTrainValLib, self).__init__(data, batch_size, None)
        if pad_kwargs_data is None:
            pad_kwargs_data = OrderedDict()
        self.pad_kwargs_data = pad_kwargs_data
        self.pad_mode = pad_mode
        self.is_val = is_val
        self.crop_size = crop_size
        self.pseudo_3d_slices = pseudo_3d_slices
        self.oversample_foreground_percent = oversample_foreground_percent
        self.final_patch_size = final_patch_size
        self.patch_size = patch_size
        self.list_of_keys = list(self._data.keys())
        self.need_to_pad = np.array(patch_size) - np.array(final_patch_size)
        self.memmap_mode = memmap_mode
        self.video_length = video_length
        self.percent = None
        if pad_sides is not None:
            if not isinstance(pad_sides, np.ndarray):
                pad_sides = np.array(pad_sides)
            self.need_to_pad += pad_sides
        self.pad_sides = pad_sides
        self.data_shape, self.seg_shape = self.determine_shapes()

        self.preprocessing_transform = self.set_up_preprocessing_pipeline()
        self.pixel_transform, self.spatial_transform = self.set_up_augmentation_pipeline()

        self.processor = processor

    def determine_shapes(self):
        num_seg = 1

        k = list(self._data.keys())[0]
        if isfile(self._data[k]['data_file'][:-4] + ".npy"):
            case_all_data = np.load(self._data[k]['data_file'][:-4] + ".npy", self.memmap_mode)
        else:
            case_all_data = np.load(self._data[k]['data_file'])['data']
        num_color_channels = case_all_data.shape[0] - num_seg
        data_shape = (self.batch_size, num_color_channels, *self.patch_size)
        seg_shape = (self.batch_size, num_seg, *self.patch_size)
        return data_shape, seg_shape

    def get_do_oversample(self, batch_idx):
        return not batch_idx < round(self.batch_size * (1 - self.oversample_foreground_percent))
    
    def select_deep_supervision(self, x):
        if self.deep_supervision:
            return x[0]
        else:
            return x
        

    def my_crop(self, labeled, seg, unlabeled):

        xm = labeled.shape[-1] / 2
        ym = labeled.shape[-2] / 2

        x1 = int(round(xm - self.final_patch_size[1] / 2))
        x2 = int(round(xm + self.final_patch_size[1] / 2))
        y1 = int(round(ym - self.final_patch_size[0] / 2))
        y2 = int(round(ym + self.final_patch_size[0] / 2))

        labeled = labeled[:, :, y1:y2, x1:x2]
        seg = seg[:, :, y1:y2, x1:x2]

        xm = unlabeled.shape[-1] / 2
        ym = unlabeled.shape[-2] / 2

        x1 = int(round(xm - self.final_patch_size[1] / 2))
        x2 = int(round(xm + self.final_patch_size[1] / 2))
        y1 = int(round(ym - self.final_patch_size[0] / 2))
        y2 = int(round(ym + self.final_patch_size[0] / 2))

        unlabeled = unlabeled[:, :, :, y1:y2, x1:x2]

        return labeled, seg, unlabeled
    
    def preprocess(self, image, mask):
        if mask is not None:
            data = {'image': image, 'mask': mask}
            transformed = self.preprocessing_transform(data)
            image = transformed['image']
            mask = transformed['mask']
        else:
            data = {'image': image}
            transformed = self.preprocessing_transform(data)
            image = transformed['image']

        return image, mask
    
    
    def augment(self, image, mask, seed):
        self.pixel_transform = self.pixel_transform.set_random_state(seed=seed)
        self.spatial_transform = self.spatial_transform.set_random_state(seed=seed)
        
        temp = torch.clone(image)
        padding_mask = image == 0

        if mask is not None:
            data = {'image': image, 'mask': mask}
            pixel_transformed = self.pixel_transform(data)
            image = pixel_transformed['image']
            mask = pixel_transformed['mask']
            image[padding_mask] = 0
            data = {'image': image, 'mask': mask}
            spatial_transformed = self.spatial_transform(data)
            image = spatial_transformed['image']
            mask = spatial_transformed['mask']

            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(1, 3)
            #ax[0].imshow(temp[0].cpu(), cmap='gray')
            #ax[1].imshow(image[0].cpu(), cmap='gray')
            #ax[2].imshow(mask[0].cpu(), cmap='gray')
            #plt.show()
        else:
            data = {'image': image}
            pixel_transformed = self.pixel_transform(data)
            image = pixel_transformed['image']
            image[padding_mask] = 0
            data = {'image': image}
            spatial_transformed = self.spatial_transform(data)
            image = spatial_transformed['image']

            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(1, 2)
            #ax[0].imshow(temp[0].cpu(), cmap='gray')
            #ax[1].imshow(image[0].cpu(), cmap='gray')
            #plt.show()

        return image, mask
    
    def set_up_preprocessing_pipeline(self):
        preprocess = Compose([
                            T.SpatialPadd(keys=['image', 'mask'], spatial_size=self.final_patch_size, allow_missing_keys=True),
                            T.CenterSpatialCropd(keys=['image', 'mask'], roi_size=self.final_patch_size, allow_missing_keys=True),
                            T.ToTensor(dtype=torch.float32, device='cuda:0')
                            ])

        return preprocess
    
    def set_up_augmentation_pipeline(self):
        spatial_transform = Compose([
                                    T.RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=0, allow_missing_keys=True),
                                    T.RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=1, allow_missing_keys=True),
                                    T.RandRotated(keys=['image', 'mask'], prob=0.2, range_x=np.pi, mode=["bilinear", "nearest"], padding_mode="zeros", allow_missing_keys=True),
                                    T.RandZoomd(keys=['image', 'mask'], prob=0.2, min_zoom=0.5, max_zoom=1.5, mode=["bilinear", "nearest"], padding_mode="constant", allow_missing_keys=True),
                                    T.RandAffined(keys=['image', 'mask'], prob=0.2, translate_range=(-26, 26), mode=["bilinear", "nearest"], padding_mode="zeros", allow_missing_keys=True)
                                    ])

        pixel_transform = Compose([
                                #RandInvertd(keys=['image'], prob=0.5),
                                T.RandAdjustContrastd(keys=['image'], prob=0.2, gamma=(0.7, 1.5), allow_missing_keys=True),
                                T.RandGaussianNoised(keys=['image'], prob=0.2, std=0.04, allow_missing_keys=True),
                                T.RandScaleIntensityd(keys=['image'], prob=0.2, factors=0.2, allow_missing_keys=True),
                                T.RandGaussianSmoothd(keys=['image'], prob=0.2, sigma_x=(0.25, 0.5), sigma_y=(0.25, 0.5), allow_missing_keys=True),
                                T.RandGaussianSharpend(keys=['image'], prob=0.2, sigma1_x=(0.1, 0.2), sigma1_y=(0.1, 0.2), sigma2_x=(0.2, 0.4), sigma2_y=(0.2, 0.4), alpha=(2.0, 3.0), allow_missing_keys=True)
                                ])

        return pixel_transform, spatial_transform

    def generate_train_batch(self):
        selected_keys = np.random.choice(self.list_of_keys, self.batch_size, True, None)
        #selected_keys = np.random.choice(self.list_of_keys, 1, True, None)
        list_of_frames = []
        for i in range(len(selected_keys)):
            patient_id = selected_keys[i][:10]
            l_filtered = [x for x in self.list_of_keys if x[:10] in patient_id]
            un_filtered = [x for x in self.un_list_of_keys if x[:10] in patient_id]
            filtered = l_filtered + un_filtered
            list_of_frames.append(filtered)
        
        target_list = []
        unlabeled_list = []
        padding_need_list = []
        target_mask_list = []

        case_properties = []
        for j, frames in enumerate(list_of_frames):
            labeled_frame = frames[0]
            if 'properties' in self._data[labeled_frame].keys():
                properties = self._data[labeled_frame]['properties']
            else:
                properties = load_pickle(self._data[labeled_frame]['properties_file'])
            case_properties.append(properties)

            depth = properties['size_after_resampling'][0]
            depth_idx = np.random.choice(depth)

            frames = sorted(frames, key=lambda x: int(x[16:18]))
            frames = np.array(frames)
            global_labeled_idx = np.where(~np.char.endswith(frames, '_u'))[0]
            assert len(global_labeled_idx) == 2

            unlabeled = torch.full(size=(len(frames), 1, *self.final_patch_size), fill_value=torch.nan, device='cuda:0')
            seg = torch.full(size=(len(frames), 1, *self.final_patch_size), fill_value=torch.nan, device='cuda:0')
            cropped_unlabeled = torch.full(size=(len(frames), 1, self.crop_size, self.crop_size), fill_value=torch.nan, device='cuda:0')
            cropped_seg = torch.full(size=(len(frames), 1, self.crop_size, self.crop_size), fill_value=torch.nan, device='cuda:0')
            target_mask = torch.zeros(size=(len(frames),), dtype=bool, device='cuda:0')

            frame_indices = np.arange(len(frames))
            before_where = np.argwhere(frame_indices < global_labeled_idx[0]).reshape(-1,)
            after_where = np.argwhere(frame_indices >= global_labeled_idx[0]).reshape(-1,)

            before_indices = frame_indices[before_where]
            after_indices = frame_indices[after_where]

            before_mask = target_mask[before_where]
            after_mask = target_mask[after_where]

            frame_indices = np.concatenate([after_indices, before_indices])
            target_mask = torch.cat([after_mask, before_mask])
            assert frame_indices[0] == global_labeled_idx[0]

            assert len(frame_indices) == len(frames)
            video = frames[frame_indices]

            labeled_idx = np.where(~np.char.endswith(video, '_u'))[0]

            for frame_idx, t in enumerate(video):

                if '_u' in t:
                    if not isfile(self.un_data[t]['data_file'][:-4] + ".npy"):
                        # lets hope you know what you're doing
                        case_all_data = np.load(self.un_data[t]['data_file'][:-4] + ".npz")['data']
                    else:
                        case_all_data = np.load(self.un_data[t]['data_file'][:-4] + ".npy", self.memmap_mode)
                else:
                    if not isfile(self._data[t]['data_file'][:-4] + ".npy"):
                        # lets hope you know what you're doing
                        case_all_data = np.load(self._data[t]['data_file'][:-4] + ".npz")['data']
                    else:
                        case_all_data = np.load(self._data[t]['data_file'][:-4] + ".npy", self.memmap_mode)

                # this is for when there is just a 2d slice in case_all_data (2d support)
                if len(case_all_data.shape) == 3:
                    case_all_data = case_all_data[:, None]
                
                assert case_all_data.shape[1] == depth

                # now crop case_all_data to contain just the slice of interest. If we want additional slice above and
                # below the current slice, here is where we get them. We stack those as additional color channels
                slice_data = case_all_data[:, depth_idx]

                # case all data should now be (c, x, y)
                assert len(slice_data.shape) == 3

                if '_u' not in t:
                    image = slice_data[:-1].copy()
                    mask = slice_data[-1:].copy()
                    mask[mask < 0] = 0

                    image, mask = self.preprocess(image, mask)

                    seg[frame_idx] = mask
                    target_mask[frame_idx] = True
                else:
                    image = slice_data.copy()

                    image, _ = self.preprocess(image, None)

                unlabeled[frame_idx] = image

            with torch.no_grad():
                mean_centroid, _ = self.processor.preprocess_no_registration(data=torch.clone(unlabeled)) # T, C(1), H, W

                cropped_unlabeled, padding_need = self.processor.crop_and_pad(data=unlabeled, mean_centroid=mean_centroid)
                padding_need_list.append(padding_need)

                cropped_seg, _ = self.processor.crop_and_pad(data=seg, mean_centroid=mean_centroid)

            cropped_unlabeled = (cropped_unlabeled - cropped_unlabeled.min()) / (cropped_unlabeled.max() - cropped_unlabeled.min() + 1e-8)

            assert target_mask[0] == True
            assert torch.count_nonzero(target_mask) == 2
            assert torch.count_nonzero(torch.any(~torch.isnan(cropped_seg.reshape(len(frames), -1)), dim=-1)) == 2, torch.count_nonzero(torch.any(~torch.isnan(cropped_seg.reshape(len(frames), -1)), dim=-1))

            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(3, len(cropped_unlabeled))
            #for t in range(len(cropped_unlabeled)):
            #    ax[0, t].imshow(temp_before_augment[t, 0].cpu(), cmap='gray')
            #    ax[1, t].imshow(cropped_unlabeled[t, 0].cpu(), cmap='gray')
            #    ax[2, t].imshow(cropped_seg[t, 0].cpu(), cmap='gray')
            #plt.show()
                    
            cropped_unlabeled = NormalizeIntensity()(cropped_unlabeled)

            unlabeled_list.append(cropped_unlabeled)
            target_list.append(cropped_seg)
            target_mask_list.append(target_mask)

        max_length = max([unlabeled_list[i].shape[0] for i in range(len(unlabeled_list))])
        unlabeled_list = [torch.nn.functional.pad(unlabeled_list[i], (0, 0, 0, 0, 0, 0, 0, max_length - len(unlabeled_list[i]))) for i in range(len(unlabeled_list))]
        target_list = [torch.nn.functional.pad(target_list[i], (0, 0, 0, 0, 0, 0, 0, max_length - len(target_list[i]))) for i in range(len(target_list))]
        target_mask_list = [torch.nn.functional.pad(target_mask_list[i], (0, max_length - len(target_mask_list[i]))) for i in range(len(target_mask_list))]

        unlabeled = torch.stack(unlabeled_list, dim=1) # T, B, 1, H, W
        target = torch.stack(target_list, dim=1) # T, B, 1, H, W
        target_mask = torch.stack(target_mask_list, dim=1) # T, B
        padding_need = torch.stack(padding_need_list, dim=0) # B, 4

        #matplotlib.use('QtAgg')
        #fig, ax = plt.subplots(1, len(target))
        #for t in range(len(target)):
        #    ax[t].imshow(target[t, 0, 0].cpu(), cmap='gray')
        #plt.show()

        keys = selected_keys

        #print(seg.shape)
        #print(np.unique(seg))
        #matplotlib.use('QtAgg')
        #fig, ax = plt.subplots(1, 2)
        #ax[0].imshow(labeled[3, 0], cmap='gray')
        #ax[1].imshow(seg[3, 0], cmap='gray')
        #plt.show()

        return {'unlabeled':unlabeled, 
                'target': target,
                'padding_need': padding_need,
                'properties': case_properties, 
                "keys": keys,
                'target_mask': target_mask}


class DataLoaderFlowTrain(SlimDataLoaderBase):
    def __init__(self, data, patch_size, final_patch_size, batch_size, unlabeled_dataset, video_length, random_step, one_to_all, processor, crop_size, oversample_foreground_percent=0.0,
                 memmap_mode="r", pseudo_3d_slices=1, pad_mode="edge",
                 pad_kwargs_data=None, pad_sides=None):
        """
        This is the basic data loader for 2D networks. It uses preprocessed data as produced by my (Fabian) preprocessing.
        You can load the data with load_dataset(folder) where folder is the folder where the npz files are located. If there
        are only npz files present in that folder, the data loader will unpack them on the fly. This may take a while
        and increase CPU usage. Therefore, I advise you to call unpack_dataset(folder) first, which will unpack all npz
        to npy. Don't forget to call delete_npy(folder) after you are done with training?
        Why all the hassle? Well the decathlon dataset is huge. Using npy for everything will consume >1 TB and that is uncool
        given that I (Fabian) will have to store that permanently on /datasets and my local computer. With this strategy all
        data is stored in a compressed format (factor 10 smaller) and only unpacked when needed.
        :param data: get this with load_dataset(folder, stage=0). Plug the return value in here and you are g2g (good to go)
        :param patch_size: what patch size will this data loader return? it is common practice to first load larger
        patches so that a central crop after data augmentation can be done to reduce border artifacts. If unsure, use
        get_patch_size() from data_augmentation.default_data_augmentation
        :param final_patch_size: what will the patch finally be cropped to (after data augmentation)? this is the patch
        size that goes into your network. We need this here because we will pad patients in here so that patches at the
        border of patients are sampled properly
        :param batch_size:
        :param num_batches: how many batches will the data loader produce before stopping? None=endless
        :param seed:
        :param stage: ignore this (Fabian only)
        :param transpose: ignore this
        :param random: sample randomly; CAREFUL! non-random sampling requires batch_size=1, otherwise you will iterate batch_size times over the dataset
        :param pseudo_3d_slices: 7 = 3 below and 3 above the center slice
        """
        super(DataLoaderFlowTrain, self).__init__(data, batch_size, None)
        if pad_kwargs_data is None:
            pad_kwargs_data = OrderedDict()
        self.pad_kwargs_data = pad_kwargs_data
        self.pad_mode = pad_mode
        self.crop_size = crop_size
        self.pseudo_3d_slices = pseudo_3d_slices
        self.oversample_foreground_percent = oversample_foreground_percent
        self.final_patch_size = final_patch_size
        self.patch_size = patch_size
        self.list_of_keys = list(self._data.keys())
        self.un_list_of_keys = list(unlabeled_dataset.keys())
        self.need_to_pad = np.array(patch_size) - np.array(final_patch_size)
        self.memmap_mode = memmap_mode
        self.video_length = video_length
        self.random_step = random_step
        self.percent = None
        self.one_to_all = one_to_all
        if pad_sides is not None:
            if not isinstance(pad_sides, np.ndarray):
                pad_sides = np.array(pad_sides)
            self.need_to_pad += pad_sides
        self.pad_sides = pad_sides
        self.data_shape, self.seg_shape = self.determine_shapes()
        self.un_data = unlabeled_dataset

        self.preprocessing_transform = self.set_up_preprocessing_pipeline()
        self.pixel_transform, self.spatial_transform = self.set_up_augmentation_pipeline()

        self.processor = processor

    def determine_shapes(self):
        num_seg = 1

        k = list(self._data.keys())[0]
        if isfile(self._data[k]['data_file'][:-4] + ".npy"):
            case_all_data = np.load(self._data[k]['data_file'][:-4] + ".npy", self.memmap_mode)
        else:
            case_all_data = np.load(self._data[k]['data_file'])['data']
        num_color_channels = case_all_data.shape[0] - num_seg
        data_shape = (self.batch_size, num_color_channels, *self.patch_size)
        seg_shape = (self.batch_size, num_seg, *self.patch_size)
        return data_shape, seg_shape

    def get_do_oversample(self, batch_idx):
        return not batch_idx < round(self.batch_size * (1 - self.oversample_foreground_percent))
    
    def select_deep_supervision(self, x):
        if self.deep_supervision:
            return x[0]
        else:
            return x
        

    def my_crop(self, labeled, seg, unlabeled):

        xm = labeled.shape[-1] / 2
        ym = labeled.shape[-2] / 2

        x1 = int(round(xm - self.final_patch_size[1] / 2))
        x2 = int(round(xm + self.final_patch_size[1] / 2))
        y1 = int(round(ym - self.final_patch_size[0] / 2))
        y2 = int(round(ym + self.final_patch_size[0] / 2))

        labeled = labeled[:, :, y1:y2, x1:x2]
        seg = seg[:, :, y1:y2, x1:x2]

        xm = unlabeled.shape[-1] / 2
        ym = unlabeled.shape[-2] / 2

        x1 = int(round(xm - self.final_patch_size[1] / 2))
        x2 = int(round(xm + self.final_patch_size[1] / 2))
        y1 = int(round(ym - self.final_patch_size[0] / 2))
        y2 = int(round(ym + self.final_patch_size[0] / 2))

        unlabeled = unlabeled[:, :, :, y1:y2, x1:x2]

        return labeled, seg, unlabeled
    
    def preprocess(self, image, mask):
        if mask is not None:
            data = {'image': image, 'mask': mask}
            transformed = self.preprocessing_transform(data)
            image = transformed['image']
            mask = transformed['mask']
        else:
            data = {'image': image}
            transformed = self.preprocessing_transform(data)
            image = transformed['image']

        return image, mask
    
    
    def augment(self, image, mask, seed):
        self.pixel_transform = self.pixel_transform.set_random_state(seed=seed)
        self.spatial_transform = self.spatial_transform.set_random_state(seed=seed)
        
        temp = torch.clone(image)
        padding_mask = image == 0

        if mask is not None:
            data = {'image': image, 'mask': mask}
            pixel_transformed = self.pixel_transform(data)
            image = pixel_transformed['image']
            mask = pixel_transformed['mask']
            image[padding_mask] = 0
            data = {'image': image, 'mask': mask}
            spatial_transformed = self.spatial_transform(data)
            image = spatial_transformed['image']
            mask = spatial_transformed['mask']

            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(1, 3)
            #ax[0].imshow(temp[0].cpu(), cmap='gray')
            #ax[1].imshow(image[0].cpu(), cmap='gray')
            #ax[2].imshow(mask[0].cpu(), cmap='gray')
            #plt.show()
        else:
            data = {'image': image}
            pixel_transformed = self.pixel_transform(data)
            image = pixel_transformed['image']
            image[padding_mask] = 0
            data = {'image': image}
            spatial_transformed = self.spatial_transform(data)
            image = spatial_transformed['image']

            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(1, 2)
            #ax[0].imshow(temp[0].cpu(), cmap='gray')
            #ax[1].imshow(image[0].cpu(), cmap='gray')
            #plt.show()

        return image, mask
    
    def set_up_preprocessing_pipeline(self):
        preprocess = Compose([
                            T.SpatialPadd(keys=['image', 'mask'], spatial_size=self.final_patch_size, allow_missing_keys=True),
                            T.CenterSpatialCropd(keys=['image', 'mask'], roi_size=self.final_patch_size, allow_missing_keys=True),
                            T.ToTensor(dtype=torch.float32, device='cuda:0')
                            ])

        return preprocess
    
    def set_up_augmentation_pipeline(self):
        spatial_transform = Compose([
                                    T.RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=0, allow_missing_keys=True),
                                    T.RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=1, allow_missing_keys=True),
                                    T.RandRotated(keys=['image', 'mask'], prob=0.2, range_x=np.pi, mode=["bilinear", "nearest"], padding_mode="zeros", allow_missing_keys=True),
                                    T.RandZoomd(keys=['image', 'mask'], prob=0.2, min_zoom=0.5, max_zoom=1.5, mode=["bilinear", "nearest"], padding_mode="constant", allow_missing_keys=True),
                                    T.RandAffined(keys=['image', 'mask'], prob=0.2, translate_range=(-26, 26), mode=["bilinear", "nearest"], padding_mode="zeros", allow_missing_keys=True)
                                    ])

        pixel_transform = Compose([
                                #RandInvertd(keys=['image'], prob=0.5),
                                T.RandAdjustContrastd(keys=['image'], prob=0.2, gamma=(0.7, 1.5), allow_missing_keys=True),
                                T.RandGaussianNoised(keys=['image'], prob=0.2, std=0.04, allow_missing_keys=True),
                                T.RandScaleIntensityd(keys=['image'], prob=0.2, factors=0.2, allow_missing_keys=True),
                                T.RandGaussianSmoothd(keys=['image'], prob=0.2, sigma_x=(0.25, 0.5), sigma_y=(0.25, 0.5), allow_missing_keys=True),
                                T.RandGaussianSharpend(keys=['image'], prob=0.2, sigma1_x=(0.1, 0.2), sigma1_y=(0.1, 0.2), sigma2_x=(0.2, 0.4), sigma2_y=(0.2, 0.4), alpha=(2.0, 3.0), allow_missing_keys=True)
                                ])

        return pixel_transform, spatial_transform

    def generate_train_batch(self):
        #eval_idx = None
        selected_keys = np.random.choice(self.list_of_keys, self.batch_size, True, None)
        list_of_frames = []
        for i in range(len(selected_keys)):
            patient_id = selected_keys[i][:10]
            l_filtered = [x for x in self.list_of_keys if x[:10] in patient_id]
            un_filtered = [x for x in self.un_list_of_keys if x[:10] in patient_id]
            filtered = l_filtered + un_filtered
            list_of_frames.append(filtered)
        
        target_list = []
        unlabeled_list = []
        padding_need_list = []
        target_mask_list = []

        case_properties = []
        for j, frames in enumerate(list_of_frames):
            labeled_frame = frames[0]
            if 'properties' in self._data[labeled_frame].keys():
                properties = self._data[labeled_frame]['properties']
            else:
                properties = load_pickle(self._data[labeled_frame]['properties_file'])
            case_properties.append(properties)

            depth = properties['size_after_resampling'][0]
            depth_idx = np.random.choice(depth)

            frames = sorted(frames, key=lambda x: int(x[16:18]))
            frames = np.array(frames)
            global_labeled_idx = np.where(~np.char.endswith(frames, '_u'))[0]

            unlabeled = torch.full(size=(self.video_length, 1, *self.final_patch_size), fill_value=torch.nan, device='cuda:0')
            seg = torch.full(size=(self.video_length, 1, *self.final_patch_size), fill_value=torch.nan, device='cuda:0')
            cropped_unlabeled = torch.full(size=(self.video_length, 1, self.crop_size, self.crop_size), fill_value=torch.nan, device='cuda:0')
            cropped_seg = torch.full(size=(self.video_length, 1, self.crop_size, self.crop_size), fill_value=torch.nan, device='cuda:0')
            target_mask = torch.zeros(size=(self.video_length,), dtype=bool, device='cuda:0')

            values = np.arange(len(frames))
            values = np.pad(values, (10000, 10000), mode='wrap')
            #print(values[:50])
            #labeled_indices = np.nonzero(np.isin(values, labeled_idx))
            #print(labeled_indices)

            if self.random_step:
                step = np.random.randint(1, 3)
                values = values[::step]

            windows = sliding_window_view(values, self.video_length)
            #print(np.isin(windows[:, 0], global_labeled_idx))
            if self.one_to_all:
                mask = np.isin(windows[:, 0], global_labeled_idx)
            else:
                mask = np.isin(windows, global_labeled_idx)
                mask = np.any(mask, axis=1)
            windows = windows[mask]
            window_idx = np.random.choice(len(windows))
            frame_indices = windows[window_idx]
            assert len(frame_indices) == self.video_length
            video = frames[frame_indices]

            labeled_idx = np.where(~np.char.endswith(video, '_u'))[0]
            compteur = 0

            for frame_idx, t in enumerate(video):

                if '_u' in t:
                    if not isfile(self.un_data[t]['data_file'][:-4] + ".npy"):
                        # lets hope you know what you're doing
                        case_all_data = np.load(self.un_data[t]['data_file'][:-4] + ".npz")['data']
                    else:
                        case_all_data = np.load(self.un_data[t]['data_file'][:-4] + ".npy", self.memmap_mode)
                else:
                    if not isfile(self._data[t]['data_file'][:-4] + ".npy"):
                        # lets hope you know what you're doing
                        case_all_data = np.load(self._data[t]['data_file'][:-4] + ".npz")['data']
                    else:
                        case_all_data = np.load(self._data[t]['data_file'][:-4] + ".npy", self.memmap_mode)

                # this is for when there is just a 2d slice in case_all_data (2d support)
                if len(case_all_data.shape) == 3:
                    case_all_data = case_all_data[:, None]
                
                assert case_all_data.shape[1] == depth

                # now crop case_all_data to contain just the slice of interest. If we want additional slice above and
                # below the current slice, here is where we get them. We stack those as additional color channels
                slice_data = case_all_data[:, depth_idx]

                # case all data should now be (c, x, y)
                assert len(slice_data.shape) == 3

                if '_u' not in t:
                    image = slice_data[:-1].copy()
                    mask = slice_data[-1:].copy()
                    mask[mask < 0] = 0

                    image, mask = self.preprocess(image, mask)

                    if compteur == 0:
                        seg[frame_idx] = mask
                        target_mask[frame_idx] = True
                        assert frame_idx in labeled_idx
                        compteur += 1
                else:
                    image = slice_data.copy()

                    image, _ = self.preprocess(image, None)

                unlabeled[frame_idx] = image

            with torch.no_grad():
                mean_centroid, _ = self.processor.preprocess_no_registration(data=torch.clone(unlabeled)) # T, C(1), H, W

                cropped_unlabeled, padding_need = self.processor.crop_and_pad(data=unlabeled, mean_centroid=mean_centroid)
                padding_need_list.append(padding_need)

                cropped_seg, _ = self.processor.crop_and_pad(data=seg, mean_centroid=mean_centroid)

            cropped_unlabeled = (cropped_unlabeled - cropped_unlabeled.min()) / (cropped_unlabeled.max() - cropped_unlabeled.min() + 1e-8)

            assert torch.count_nonzero(torch.any(~torch.isnan(cropped_seg.reshape(self.video_length, -1)), dim=-1)) == 1

            assert torch.any(cropped_unlabeled[0] != cropped_unlabeled[1])

            temp_before_augment = torch.clone(cropped_unlabeled).to('cpu')
            
            seed = random.randint(0, 2**32-1)
            #cropped_labeled[j], cropped_seg[j] = self.augment(image=cropped_labeled[j], mask=cropped_seg[j], seed=seed)

            for t in range(len(cropped_unlabeled)):
                cropped_unlabeled[t], cropped_seg[t] = self.augment(image=cropped_unlabeled[t], mask=cropped_seg[t], seed=seed)

            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(3, len(cropped_unlabeled))
            #for t in range(len(cropped_unlabeled)):
            #    ax[0, t].imshow(temp_before_augment[t, 0].cpu(), cmap='gray')
            #    ax[1, t].imshow(cropped_unlabeled[t, 0].cpu(), cmap='gray')
            #    ax[2, t].imshow(cropped_seg[t, 0].cpu(), cmap='gray')
            #plt.show()
                    
            cropped_unlabeled = NormalizeIntensity()(cropped_unlabeled)

            if self.one_to_all:
                assert target_mask[0] == True

            unlabeled_list.append(cropped_unlabeled)
            target_list.append(cropped_seg)
            target_mask_list.append(target_mask)

        unlabeled = torch.stack(unlabeled_list, dim=1) # T, B, 1, H, W
        target = torch.stack(target_list, dim=1) # T, B, 1, H, W
        target_mask = torch.stack(target_mask_list, dim=1) # T, B
        padding_need = torch.stack(padding_need_list, dim=0) # B, 4

        keys = selected_keys

        #print(seg.shape)
        #print(np.unique(seg))
        #matplotlib.use('QtAgg')
        #fig, ax = plt.subplots(1, 2)
        #ax[0].imshow(labeled[3, 0], cmap='gray')
        #ax[1].imshow(seg[3, 0], cmap='gray')
        #plt.show()

        return {'unlabeled':unlabeled, 
                'target': target,
                'padding_need': padding_need,
                'properties': case_properties, 
                "keys": keys,
                'target_mask': target_mask}
    


class DataLoaderFlowValidationOneStep(SlimDataLoaderBase):
    def __init__(self, data, patch_size, final_patch_size, batch_size, unlabeled_dataset, video_length, step, force_one_label, processor, crop_size, oversample_foreground_percent=0.0,
                 memmap_mode="r", pseudo_3d_slices=1, pad_mode="edge",
                 pad_kwargs_data=None, pad_sides=None):
        """
        This is the basic data loader for 2D networks. It uses preprocessed data as produced by my (Fabian) preprocessing.
        You can load the data with load_dataset(folder) where folder is the folder where the npz files are located. If there
        are only npz files present in that folder, the data loader will unpack them on the fly. This may take a while
        and increase CPU usage. Therefore, I advise you to call unpack_dataset(folder) first, which will unpack all npz
        to npy. Don't forget to call delete_npy(folder) after you are done with training?
        Why all the hassle? Well the decathlon dataset is huge. Using npy for everything will consume >1 TB and that is uncool
        given that I (Fabian) will have to store that permanently on /datasets and my local computer. With this strategy all
        data is stored in a compressed format (factor 10 smaller) and only unpacked when needed.
        :param data: get this with load_dataset(folder, stage=0). Plug the return value in here and you are g2g (good to go)
        :param patch_size: what patch size will this data loader return? it is common practice to first load larger
        patches so that a central crop after data augmentation can be done to reduce border artifacts. If unsure, use
        get_patch_size() from data_augmentation.default_data_augmentation
        :param final_patch_size: what will the patch finally be cropped to (after data augmentation)? this is the patch
        size that goes into your network. We need this here because we will pad patients in here so that patches at the
        border of patients are sampled properly
        :param batch_size:
        :param num_batches: how many batches will the data loader produce before stopping? None=endless
        :param seed:
        :param stage: ignore this (Fabian only)
        :param transpose: ignore this
        :param random: sample randomly; CAREFUL! non-random sampling requires batch_size=1, otherwise you will iterate batch_size times over the dataset
        :param pseudo_3d_slices: 7 = 3 below and 3 above the center slice
        """
        super(DataLoaderFlowValidationOneStep, self).__init__(data, batch_size, None)
        if pad_kwargs_data is None:
            pad_kwargs_data = OrderedDict()
        self.pad_kwargs_data = pad_kwargs_data
        self.pad_mode = pad_mode
        self.crop_size = crop_size
        self.pseudo_3d_slices = pseudo_3d_slices
        self.oversample_foreground_percent = oversample_foreground_percent
        self.final_patch_size = final_patch_size
        self.patch_size = patch_size
        self.list_of_keys = list(self._data.keys())
        self.un_list_of_keys = list(unlabeled_dataset.keys())
        self.need_to_pad = np.array(patch_size) - np.array(final_patch_size)
        self.memmap_mode = memmap_mode
        self.video_length = video_length
        self.step = step
        self.percent = None
        self.force_one_label = force_one_label
        if pad_sides is not None:
            if not isinstance(pad_sides, np.ndarray):
                pad_sides = np.array(pad_sides)
            self.need_to_pad += pad_sides
        self.pad_sides = pad_sides
        self.data_shape, self.seg_shape = self.determine_shapes()
        self.un_data = unlabeled_dataset

        self.preprocessing_transform = self.set_up_preprocessing_pipeline()
        self.pixel_transform, self.spatial_transform = self.set_up_augmentation_pipeline()

        self.processor = processor

    def determine_shapes(self):
        num_seg = 1

        k = list(self._data.keys())[0]
        if isfile(self._data[k]['data_file'][:-4] + ".npy"):
            case_all_data = np.load(self._data[k]['data_file'][:-4] + ".npy", self.memmap_mode)
        else:
            case_all_data = np.load(self._data[k]['data_file'])['data']
        num_color_channels = case_all_data.shape[0] - num_seg
        data_shape = (self.batch_size, num_color_channels, *self.patch_size)
        seg_shape = (self.batch_size, num_seg, *self.patch_size)
        return data_shape, seg_shape

    def get_do_oversample(self, batch_idx):
        return not batch_idx < round(self.batch_size * (1 - self.oversample_foreground_percent))
    
    def select_deep_supervision(self, x):
        if self.deep_supervision:
            return x[0]
        else:
            return x
        

    def my_crop(self, labeled, seg, unlabeled):

        xm = labeled.shape[-1] / 2
        ym = labeled.shape[-2] / 2

        x1 = int(round(xm - self.final_patch_size[1] / 2))
        x2 = int(round(xm + self.final_patch_size[1] / 2))
        y1 = int(round(ym - self.final_patch_size[0] / 2))
        y2 = int(round(ym + self.final_patch_size[0] / 2))

        labeled = labeled[:, :, y1:y2, x1:x2]
        seg = seg[:, :, y1:y2, x1:x2]

        xm = unlabeled.shape[-1] / 2
        ym = unlabeled.shape[-2] / 2

        x1 = int(round(xm - self.final_patch_size[1] / 2))
        x2 = int(round(xm + self.final_patch_size[1] / 2))
        y1 = int(round(ym - self.final_patch_size[0] / 2))
        y2 = int(round(ym + self.final_patch_size[0] / 2))

        unlabeled = unlabeled[:, :, :, y1:y2, x1:x2]

        return labeled, seg, unlabeled
    
    def preprocess(self, image, mask):
        if mask is not None:
            data = {'image': image, 'mask': mask}
            transformed = self.preprocessing_transform(data)
            image = transformed['image']
            mask = transformed['mask']
        else:
            data = {'image': image}
            transformed = self.preprocessing_transform(data)
            image = transformed['image']

        return image, mask
    
    
    def augment(self, image, mask, seed):
        self.pixel_transform = self.pixel_transform.set_random_state(seed=seed)
        self.spatial_transform = self.spatial_transform.set_random_state(seed=seed)
        
        temp = torch.clone(image)
        padding_mask = image == 0

        if mask is not None:
            data = {'image': image, 'mask': mask}
            pixel_transformed = self.pixel_transform(data)
            image = pixel_transformed['image']
            mask = pixel_transformed['mask']
            image[padding_mask] = 0
            data = {'image': image, 'mask': mask}
            spatial_transformed = self.spatial_transform(data)
            image = spatial_transformed['image']
            mask = spatial_transformed['mask']

            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(1, 3)
            #ax[0].imshow(temp[0].cpu(), cmap='gray')
            #ax[1].imshow(image[0].cpu(), cmap='gray')
            #ax[2].imshow(mask[0].cpu(), cmap='gray')
            #plt.show()
        else:
            data = {'image': image}
            pixel_transformed = self.pixel_transform(data)
            image = pixel_transformed['image']
            image[padding_mask] = 0
            data = {'image': image}
            spatial_transformed = self.spatial_transform(data)
            image = spatial_transformed['image']

            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(1, 2)
            #ax[0].imshow(temp[0].cpu(), cmap='gray')
            #ax[1].imshow(image[0].cpu(), cmap='gray')
            #plt.show()

        return image, mask
    
    def set_up_preprocessing_pipeline(self):
        preprocess = Compose([
                            T.SpatialPadd(keys=['image', 'mask'], spatial_size=self.final_patch_size, allow_missing_keys=True),
                            T.CenterSpatialCropd(keys=['image', 'mask'], roi_size=self.final_patch_size, allow_missing_keys=True),
                            T.ToTensor(dtype=torch.float32, device='cuda:0')
                            ])

        return preprocess
    
    def set_up_augmentation_pipeline(self):
        spatial_transform = Compose([
                                    T.RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=0, allow_missing_keys=True),
                                    T.RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=1, allow_missing_keys=True),
                                    T.RandRotated(keys=['image', 'mask'], prob=0.2, range_x=np.pi, mode=["bilinear", "nearest"], padding_mode="zeros", allow_missing_keys=True),
                                    T.RandZoomd(keys=['image', 'mask'], prob=0.2, min_zoom=0.5, max_zoom=1.5, mode=["bilinear", "nearest"], padding_mode="constant", allow_missing_keys=True),
                                    T.RandAffined(keys=['image', 'mask'], prob=0.2, translate_range=(-26, 26), mode=["bilinear", "nearest"], padding_mode="zeros", allow_missing_keys=True)
                                    ])

        pixel_transform = Compose([
                                #RandInvertd(keys=['image'], prob=0.5),
                                T.RandAdjustContrastd(keys=['image'], prob=0.2, gamma=(0.7, 1.5), allow_missing_keys=True),
                                T.RandGaussianNoised(keys=['image'], prob=0.2, std=0.04, allow_missing_keys=True),
                                T.RandScaleIntensityd(keys=['image'], prob=0.2, factors=0.2, allow_missing_keys=True),
                                T.RandGaussianSmoothd(keys=['image'], prob=0.2, sigma_x=(0.25, 0.5), sigma_y=(0.25, 0.5), allow_missing_keys=True),
                                T.RandGaussianSharpend(keys=['image'], prob=0.2, sigma1_x=(0.1, 0.2), sigma1_y=(0.1, 0.2), sigma2_x=(0.2, 0.4), sigma2_y=(0.2, 0.4), alpha=(2.0, 3.0), allow_missing_keys=True)
                                ])

        return pixel_transform, spatial_transform

    def generate_train_batch(self):
        #eval_idx = None
        selected_keys = np.random.choice(self.list_of_keys, self.batch_size, True, None)
        list_of_frames = []
        for i in range(len(selected_keys)):
            patient_id = selected_keys[i][:10]
            l_filtered = [x for x in self.list_of_keys if x[:10] in patient_id]
            un_filtered = [x for x in self.un_list_of_keys if x[:10] in patient_id]
            filtered = l_filtered + un_filtered
            list_of_frames.append(filtered)

        target_list = []
        unlabeled_list = []
        padding_need_list = []
        target_mask_list = []
        padding_mask_list = []

        max_length = len(max(list_of_frames, key=lambda x:len(x)))

        case_properties = []
        for j, frames in enumerate(list_of_frames):
            labeled_frame = frames[0]
            if 'properties' in self._data[labeled_frame].keys():
                properties = self._data[labeled_frame]['properties']
            else:
                properties = load_pickle(self._data[labeled_frame]['properties_file'])
            case_properties.append(properties)

            depth = properties['size_after_resampling'][0]
            depth_idx = np.random.choice(depth)

            frames = sorted(frames, key=lambda x: int(x[16:18]))
            frames = np.array(frames)
            global_labeled_idx = np.where(~np.char.endswith(frames, '_u'))[0]

            to_pad = max_length - len(frames)

            unlabeled = torch.full(size=(len(frames), 1, *self.final_patch_size), fill_value=torch.nan, device='cuda:0')
            seg = torch.full(size=(len(frames), 1, *self.final_patch_size), fill_value=torch.nan, device='cuda:0')
            cropped_unlabeled = torch.full(size=(len(frames), 1, self.crop_size, self.crop_size), fill_value=torch.nan, device='cuda:0')
            cropped_seg = torch.full(size=(len(frames), 1, self.crop_size, self.crop_size), fill_value=torch.nan, device='cuda:0')
            target_mask = torch.zeros(size=(len(frames),), dtype=bool, device='cuda:0')
            padding_mask = torch.ones(size=(len(frames),), dtype=bool, device='cuda:0')

            for frame_idx, t in enumerate(frames):
                if '_u' in t:
                    if not isfile(self.un_data[t]['data_file'][:-4] + ".npy"):
                        # lets hope you know what you're doing
                        case_all_data = np.load(self.un_data[t]['data_file'][:-4] + ".npz")['data']
                    else:
                        case_all_data = np.load(self.un_data[t]['data_file'][:-4] + ".npy", self.memmap_mode)
                else:
                    if not isfile(self._data[t]['data_file'][:-4] + ".npy"):
                        # lets hope you know what you're doing
                        case_all_data = np.load(self._data[t]['data_file'][:-4] + ".npz")['data']
                    else:
                        case_all_data = np.load(self._data[t]['data_file'][:-4] + ".npy", self.memmap_mode)

                # this is for when there is just a 2d slice in case_all_data (2d support)
                if len(case_all_data.shape) == 3:
                    case_all_data = case_all_data[:, None]
                
                assert case_all_data.shape[1] == depth

                # now crop case_all_data to contain just the slice of interest. If we want additional slice above and
                # below the current slice, here is where we get them. We stack those as additional color channels
                slice_data = case_all_data[:, depth_idx]

                # case all data should now be (c, x, y)
                assert len(slice_data.shape) == 3

                if '_u' not in t:
                    image = slice_data[:-1].copy()
                    mask = slice_data[-1:].copy()
                    mask[mask < 0] = 0

                    image, mask = self.preprocess(image, mask)

                    seg[frame_idx] = mask
                    target_mask[frame_idx] = True
                    assert frame_idx in global_labeled_idx

                    #local_idx = frame_idx + (video_idx * self.video_length)
                    #global_idx = local_idx - video_idx 
                    #if global_idx in labeled_idx:
                    #    labeled_indices_video_list.append(local_idx)
                else:
                    image = slice_data.copy()

                    image, _ = self.preprocess(image, None)

                unlabeled[frame_idx] = image

            with torch.no_grad():
                mean_centroid, _ = self.processor.preprocess_no_registration(data=torch.clone(unlabeled)) # T, C(1), H, W

                cropped_unlabeled, padding_need = self.processor.crop_and_pad(data=unlabeled, mean_centroid=mean_centroid)
                padding_need_list.append(padding_need)

                cropped_seg, _ = self.processor.crop_and_pad(data=seg, mean_centroid=mean_centroid)

            assert torch.any(cropped_unlabeled[0] != cropped_unlabeled[1]) 

            cropped_unlabeled = NormalizeIntensity()(cropped_unlabeled)

            for p in range(to_pad // len(frames) + 1):
                current_pad = to_pad - (p * len(frames))
                cropped_unlabeled = torch.cat([cropped_unlabeled, cropped_unlabeled[:current_pad]], dim=0)
                cropped_seg = torch.cat([cropped_seg, cropped_seg[:current_pad]], dim=0)
                target_mask = torch.cat([target_mask, target_mask[:current_pad]], dim=0)
            padding_mask = torch.nn.functional.pad(padding_mask, pad=(0, to_pad), mode='constant')

            unlabeled_list.append(cropped_unlabeled)
            target_list.append(cropped_seg)
            target_mask_list.append(target_mask)
            padding_mask_list.append(padding_mask)
        
        unlabeled = torch.stack(unlabeled_list, dim=1) # T_max, B, 1, H, W
        target = torch.stack(target_list, dim=1) # T_max, B, 1, H, W
        target_mask = torch.stack(target_mask_list, dim=1) # T_max, B
        padding_mask = torch.stack(padding_mask_list, dim=1) # T_max, B
        padding_need = torch.stack(padding_need_list, dim=0) # B, 4

        unlabeled = unlabeled[None]
        target = target[None]
        target_mask = target_mask[None]

        #print(seg.shape)
        #print(np.unique(seg))
        #matplotlib.use('QtAgg')
        #fig, ax = plt.subplots(1, 2)
        #ax[0].imshow(labeled[3, 0], cmap='gray')
        #ax[1].imshow(seg[3, 0], cmap='gray')
        #plt.show()

        return {'unlabeled':unlabeled, 
                'target': target,
                'padding_need': padding_need,
                'padding_mask': padding_mask,
                'properties': case_properties,
                'target_mask': target_mask}
    
    

class DataLoader2DBinary(SlimDataLoaderBase):
    def __init__(self, data, patch_size, final_patch_size, batch_size, deep_supervision, isval, oversample_foreground_percent=0.0,
                 memmap_mode="r", pseudo_3d_slices=1, pad_mode="edge",
                 pad_kwargs_data=None, pad_sides=None):
        """
        This is the basic data loader for 2D networks. It uses preprocessed data as produced by my (Fabian) preprocessing.
        You can load the data with load_dataset(folder) where folder is the folder where the npz files are located. If there
        are only npz files present in that folder, the data loader will unpack them on the fly. This may take a while
        and increase CPU usage. Therefore, I advise you to call unpack_dataset(folder) first, which will unpack all npz
        to npy. Don't forget to call delete_npy(folder) after you are done with training?
        Why all the hassle? Well the decathlon dataset is huge. Using npy for everything will consume >1 TB and that is uncool
        given that I (Fabian) will have to store that permanently on /datasets and my local computer. With this strategy all
        data is stored in a compressed format (factor 10 smaller) and only unpacked when needed.
        :param data: get this with load_dataset(folder, stage=0). Plug the return value in here and you are g2g (good to go)
        :param patch_size: what patch size will this data loader return? it is common practice to first load larger
        patches so that a central crop after data augmentation can be done to reduce border artifacts. If unsure, use
        get_patch_size() from data_augmentation.default_data_augmentation
        :param final_patch_size: what will the patch finally be cropped to (after data augmentation)? this is the patch
        size that goes into your network. We need this here because we will pad patients in here so that patches at the
        border of patients are sampled properly
        :param batch_size:
        :param num_batches: how many batches will the data loader produce before stopping? None=endless
        :param seed:
        :param stage: ignore this (Fabian only)
        :param transpose: ignore this
        :param random: sample randomly; CAREFUL! non-random sampling requires batch_size=1, otherwise you will iterate batch_size times over the dataset
        :param pseudo_3d_slices: 7 = 3 below and 3 above the center slice
        """
        super(DataLoader2DBinary, self).__init__(data, batch_size, None)
        if pad_kwargs_data is None:
            pad_kwargs_data = OrderedDict()
        self.pad_kwargs_data = pad_kwargs_data
        self.pad_mode = pad_mode
        self.isval = isval
        self.pseudo_3d_slices = pseudo_3d_slices
        self.oversample_foreground_percent = oversample_foreground_percent
        self.final_patch_size = final_patch_size
        self.patch_size = patch_size
        self.list_of_keys = list(self._data.keys())
        self.need_to_pad = np.array(patch_size) - np.array(final_patch_size)
        self.memmap_mode = memmap_mode
        if pad_sides is not None:
            if not isinstance(pad_sides, np.ndarray):
                pad_sides = np.array(pad_sides)
            self.need_to_pad += pad_sides
        self.pad_sides = pad_sides
        self.data_shape, self.seg_shape = self.determine_shapes()

        self.preprocessing_transform = self.set_up_preprocessing_pipeline()
        if isval:
            self.pixel_transform = self.spatial_transform = None
        else:
            self.pixel_transform, self.spatial_transform = self.set_up_augmentation_pipeline()
        self.deep_supervision = deep_supervision

    def determine_shapes(self):
        num_seg = 1

        k = list(self._data.keys())[0]
        if isfile(self._data[k]['data_file'][:-4] + ".npy"):
            case_all_data = np.load(self._data[k]['data_file'][:-4] + ".npy", self.memmap_mode)
        else:
            case_all_data = np.load(self._data[k]['data_file'])['data']
        num_color_channels = case_all_data.shape[0] - num_seg
        data_shape = (self.batch_size, num_color_channels, *self.patch_size)
        seg_shape = (self.batch_size, num_seg, *self.patch_size)
        return data_shape, seg_shape

    def get_do_oversample(self, batch_idx):
        return not batch_idx < round(self.batch_size * (1 - self.oversample_foreground_percent))
    
    def preprocess(self, image, mask):
        data = {'image': image, 'mask': mask}
        transformed = self.preprocessing_transform(data)
        image = transformed['image']
        mask = transformed['mask']

        return image, mask
    
    def augment(self, image, mask):
        temp = np.copy(image)
        padding_mask = image == 0

        data = {'image': image, 'mask': mask}
        pixel_transformed = self.pixel_transform(data)
        image = pixel_transformed['image']
        mask = pixel_transformed['mask']
        image[padding_mask] = 0

        #matplotlib.use('QtAgg')
        #fig, ax = plt.subplots(1, 3)
        #ax[0].imshow(temp[0], cmap='gray')
        #ax[1].imshow(image[0], cmap='gray')
        #ax[2].imshow(mask[0], cmap='gray')
        #plt.show()
        
        data = {'image': image, 'mask': mask}
        spatial_transformed = self.spatial_transform(data)
        image = spatial_transformed['image']
        mask = spatial_transformed['mask']

        #matplotlib.use('QtAgg')
        #fig, ax = plt.subplots(1, 3)
        #ax[0].imshow(temp[0], cmap='gray')
        #ax[1].imshow(image[0], cmap='gray')
        #ax[2].imshow(mask[0], cmap='gray')
        #plt.show()

        return image, mask

    def get_deep_supervision_scales(self, image, mask):
        resized_image_list = []
        resized_mask_list = []
        for i in range(3):
            resized_image = cv.resize(image, (self.final_patch_size[0] // 2**i, self.final_patch_size[1] // 2**i), interpolation=cv.INTER_LINEAR)
            resized_mask = cv.resize(mask, (self.final_patch_size[0] // 2**i, self.final_patch_size[1] // 2**i), interpolation=cv.INTER_NEAREST)
            resized_image_list.append(resized_image)
            resized_mask_list.append(resized_mask)

        return resized_image_list, resized_mask_list

    def set_up_preprocessing_pipeline(self):
        preprocess = Compose([
                            T.SpatialPadd(keys=['image', 'mask'], spatial_size=self.final_patch_size),
                            T.CenterSpatialCropd(keys=['image', 'mask'], roi_size=self.final_patch_size)
                            ])

        return preprocess
    
    def set_up_augmentation_pipeline(self):
        spatial_transform = Compose([
                                    T.RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=0),
                                    T.RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=1),
                                    T.RandRotated(keys=['image', 'mask'], prob=0.2, range_x=np.pi, mode=["bilinear", "nearest"], padding_mode="zeros"),
                                    T.RandZoomd(keys=['image', 'mask'], prob=0.2, min_zoom=0.5, max_zoom=1.5, mode=["bilinear", "nearest"], padding_mode="constant"),
                                    T.RandAffined(keys=['image', 'mask'], prob=0.2, translate_range=(-45, 45), mode=["bilinear", "nearest"], padding_mode="zeros")
                                    ])

        pixel_transform = Compose([
                                    T.RandGaussianNoised(keys=['image'], prob=0.2, std=0.04),
                                    T.RandScaleIntensityd(keys=['image'], prob=0.2, factors=0.2),
                                    T.RandAdjustContrastd(keys=['image'], prob=0.2, gamma=(0.7, 1.5)),
                                    T.RandGaussianSmoothd(keys=['image'], prob=0.2, sigma_x=(0.5, 1.0), sigma_y=(0.5, 1.0)),
                                    T.RandGaussianSharpend(keys=['image'], prob=0.2, sigma1_x=(0.2, 0.3), sigma1_y=(0.2, 0.3), sigma2_x=0.2, sigma2_y=0.2, alpha=(20.0, 20.0))
                                    ])

        return pixel_transform, spatial_transform

    def generate_train_batch(self):
        selected_keys = np.random.choice(self.list_of_keys, self.batch_size, True, None)

        if self.deep_supervision:
            B, C, H, W = (self.batch_size, 1, *self.final_patch_size)
            data = [np.zeros((B, C, H // 2**i, W // 2**i), dtype=np.float32) for i in range(3)]
            seg = [np.zeros((B, C, H // 2**i, W // 2**i), dtype=np.float32) for i in range(3)]
        else:
            B, C, H, W = (self.batch_size, 1, *self.final_patch_size)
            data = [np.zeros((B, C, H // 2**i, W // 2**i), dtype=np.float32) for i in range(1)]
            seg = [np.zeros((B, C, H // 2**i, W // 2**i), dtype=np.float32) for i in range(1)]

        case_properties = []
        for j, i in enumerate(selected_keys):
            if 'properties' in self._data[i].keys():
                properties = self._data[i]['properties']
            else:
                properties = load_pickle(self._data[i]['properties_file'])
            case_properties.append(properties)

            if not isfile(self._data[i]['data_file'][:-4] + ".npy"):
                # lets hope you know what you're doing
                case_all_data = np.load(self._data[i]['data_file'][:-4] + ".npz")['data']
            else:
                case_all_data = np.load(self._data[i]['data_file'][:-4] + ".npy", self.memmap_mode)

            # this is for when there is just a 2d slice in case_all_data (2d support)
            if len(case_all_data.shape) == 3:
                case_all_data = case_all_data[:, None]

            # first select a slice. This can be either random (no force fg) or guaranteed to contain some class
            random_slice = np.random.choice(case_all_data.shape[1])

            # now crop case_all_data to contain just the slice of interest. If we want additional slice above and
            # below the current slice, here is where we get them. We stack those as additional color channels
            case_all_data = case_all_data[:, random_slice]

            # case all data should now be (c, x, y)
            assert len(case_all_data.shape) == 3
            
            assert np.min(case_all_data[:-1]) >= 0.0
            image = case_all_data[:-1].copy()
            mask = case_all_data[-1:].copy()
            mask[mask < 0] = 0
            mask[mask > 0] = 1

            image = (image - image.min()) / (image.max() - image.min() + 1e-8)
            image, mask = self.preprocess(image, mask)
            if not self.isval:
                image, mask = self.augment(image, mask)

            image = NormalizeIntensity()(image)

            assert np.all(np.isin(mask, [0, 1]))

            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(1, 2)
            #ax[0].imshow(image[0], cmap='gray')
            #ax[1].imshow(mask[0], cmap='gray')
            #plt.show()

            if self.deep_supervision:
                image_list, mask_list = self.get_deep_supervision_scales(image[0], mask[0])
                for i in range(3):
                    data[i][j] = image_list[i][None]
                    seg[i][j] = mask_list[i][None]
            else:
                data[j] = image[None]
                seg[j] = mask[None]

        keys = selected_keys

        #print(seg.shape)
        #print(np.unique(seg))
        #matplotlib.use('QtAgg')
        #fig, ax = plt.subplots(1, 2)
        #ax[0].imshow(labeled[3, 0], cmap='gray')
        #ax[1].imshow(seg[3, 0], cmap='gray')
        #plt.show()

        return {'data': data, 'target': seg, 'properties': case_properties, "keys": keys}




class DataLoaderStableDiffusion(SlimDataLoaderBase):
    def __init__(self, data, patch_size, final_patch_size, batch_size, unlabeled_dataset, video_length, processor, crop_size, is_val, oversample_foreground_percent=0.0,
                 memmap_mode="r", pseudo_3d_slices=1, pad_mode="edge",
                 pad_kwargs_data=None, pad_sides=None):
        """
        This is the basic data loader for 2D networks. It uses preprocessed data as produced by my (Fabian) preprocessing.
        You can load the data with load_dataset(folder) where folder is the folder where the npz files are located. If there
        are only npz files present in that folder, the data loader will unpack them on the fly. This may take a while
        and increase CPU usage. Therefore, I advise you to call unpack_dataset(folder) first, which will unpack all npz
        to npy. Don't forget to call delete_npy(folder) after you are done with training?
        Why all the hassle? Well the decathlon dataset is huge. Using npy for everything will consume >1 TB and that is uncool
        given that I (Fabian) will have to store that permanently on /datasets and my local computer. With this strategy all
        data is stored in a compressed format (factor 10 smaller) and only unpacked when needed.
        :param data: get this with load_dataset(folder, stage=0). Plug the return value in here and you are g2g (good to go)
        :param patch_size: what patch size will this data loader return? it is common practice to first load larger
        patches so that a central crop after data augmentation can be done to reduce border artifacts. If unsure, use
        get_patch_size() from data_augmentation.default_data_augmentation
        :param final_patch_size: what will the patch finally be cropped to (after data augmentation)? this is the patch
        size that goes into your network. We need this here because we will pad patients in here so that patches at the
        border of patients are sampled properly
        :param batch_size:
        :param num_batches: how many batches will the data loader produce before stopping? None=endless
        :param seed:
        :param stage: ignore this (Fabian only)
        :param transpose: ignore this
        :param random: sample randomly; CAREFUL! non-random sampling requires batch_size=1, otherwise you will iterate batch_size times over the dataset
        :param pseudo_3d_slices: 7 = 3 below and 3 above the center slice
        """
        super(DataLoaderStableDiffusion, self).__init__(data, batch_size, None)
        if pad_kwargs_data is None:
            pad_kwargs_data = OrderedDict()
        self.pad_kwargs_data = pad_kwargs_data
        self.pad_mode = pad_mode
        self.is_val = is_val
        self.crop_size = crop_size
        self.pseudo_3d_slices = pseudo_3d_slices
        self.oversample_foreground_percent = oversample_foreground_percent
        self.final_patch_size = final_patch_size
        self.patch_size = patch_size
        self.list_of_keys = list(self._data.keys())
        self.un_list_of_keys = list(unlabeled_dataset.keys())
        self.need_to_pad = np.array(patch_size) - np.array(final_patch_size)
        self.memmap_mode = memmap_mode
        self.video_length = video_length
        self.percent = None
        if pad_sides is not None:
            if not isinstance(pad_sides, np.ndarray):
                pad_sides = np.array(pad_sides)
            self.need_to_pad += pad_sides
        self.pad_sides = pad_sides
        self.data_shape, self.seg_shape = self.determine_shapes()
        self.un_data = unlabeled_dataset

        self.preprocessing_transform = self.set_up_preprocessing_pipeline()
        self.pixel_transform, self.spatial_transform = self.set_up_augmentation_pipeline()

        self.processor = processor

    def determine_shapes(self):
        num_seg = 1

        k = list(self._data.keys())[0]
        if isfile(self._data[k]['data_file'][:-4] + ".npy"):
            case_all_data = np.load(self._data[k]['data_file'][:-4] + ".npy", self.memmap_mode)
        else:
            case_all_data = np.load(self._data[k]['data_file'])['data']
        num_color_channels = case_all_data.shape[0] - num_seg
        data_shape = (self.batch_size, num_color_channels, *self.patch_size)
        seg_shape = (self.batch_size, num_seg, *self.patch_size)
        return data_shape, seg_shape

    def get_do_oversample(self, batch_idx):
        return not batch_idx < round(self.batch_size * (1 - self.oversample_foreground_percent))
    
    def select_deep_supervision(self, x):
        if self.deep_supervision:
            return x[0]
        else:
            return x
        

    def my_crop(self, labeled, seg, unlabeled):

        xm = labeled.shape[-1] / 2
        ym = labeled.shape[-2] / 2

        x1 = int(round(xm - self.final_patch_size[1] / 2))
        x2 = int(round(xm + self.final_patch_size[1] / 2))
        y1 = int(round(ym - self.final_patch_size[0] / 2))
        y2 = int(round(ym + self.final_patch_size[0] / 2))

        labeled = labeled[:, :, y1:y2, x1:x2]
        seg = seg[:, :, y1:y2, x1:x2]

        xm = unlabeled.shape[-1] / 2
        ym = unlabeled.shape[-2] / 2

        x1 = int(round(xm - self.final_patch_size[1] / 2))
        x2 = int(round(xm + self.final_patch_size[1] / 2))
        y1 = int(round(ym - self.final_patch_size[0] / 2))
        y2 = int(round(ym + self.final_patch_size[0] / 2))

        unlabeled = unlabeled[:, :, :, y1:y2, x1:x2]

        return labeled, seg, unlabeled
    
    def preprocess(self, image):
        data = {'image': image}
        transformed = self.preprocessing_transform(data)
        image = transformed['image']

        return image
    
    
    def augment(self, image, seed):
        self.pixel_transform = self.pixel_transform.set_random_state(seed=seed)
        self.spatial_transform = self.spatial_transform.set_random_state(seed=seed)
        
        temp = torch.clone(image)
        padding_mask = image == 0

        data = {'image': image}
        pixel_transformed = self.pixel_transform(data)
        image = pixel_transformed['image']
        image[padding_mask] = 0
        data = {'image': image}
        spatial_transformed = self.spatial_transform(data)
        image = spatial_transformed['image']

        #matplotlib.use('QtAgg')
        #fig, ax = plt.subplots(1, 2)
        #ax[0].imshow(temp[0].cpu(), cmap='gray')
        #ax[1].imshow(image[0].cpu(), cmap='gray')
        #plt.show()

        return image
    
    def set_up_preprocessing_pipeline(self):
        preprocess = Compose([
                            T.SpatialPadd(keys=['image'], spatial_size=self.final_patch_size, allow_missing_keys=True),
                            T.CenterSpatialCropd(keys=['image'], roi_size=self.final_patch_size, allow_missing_keys=True),
                            T.ToTensor(dtype=torch.float32, device='cuda:0')
                            ])

        return preprocess
    
    def set_up_augmentation_pipeline(self):
        spatial_transform = Compose([
                                    T.RandFlipd(keys=['image'], prob=0.5, spatial_axis=0, allow_missing_keys=True),
                                    T.RandFlipd(keys=['image'], prob=0.5, spatial_axis=1, allow_missing_keys=True),
                                    T.RandRotated(keys=['image'], prob=0.0, range_x=np.pi, mode=["bilinear"], padding_mode="zeros", allow_missing_keys=True),
                                    T.RandZoomd(keys=['image'], prob=0.0, min_zoom=0.5, max_zoom=1.5, mode=["bilinear"], padding_mode="constant", allow_missing_keys=True),
                                    T.RandAffined(keys=['image'], prob=0.0, translate_range=(-26, 26), mode=["bilinear"], padding_mode="zeros", allow_missing_keys=True)
                                    ])

        pixel_transform = Compose([
                                #RandInvertd(keys=['image'], prob=0.5),
                                T.RandAdjustContrastd(keys=['image'], prob=0.0, gamma=(0.7, 1.5), allow_missing_keys=True),
                                T.RandGaussianNoised(keys=['image'], prob=0.0, std=0.04, allow_missing_keys=True),
                                T.RandScaleIntensityd(keys=['image'], prob=0.0, factors=0.2, allow_missing_keys=True),
                                T.RandGaussianSmoothd(keys=['image'], prob=0.0, sigma_x=(0.25, 0.5), sigma_y=(0.25, 0.5), allow_missing_keys=True),
                                T.RandGaussianSharpend(keys=['image'], prob=0.0, sigma1_x=(0.1, 0.2), sigma1_y=(0.1, 0.2), sigma2_x=(0.2, 0.4), sigma2_y=(0.2, 0.4), alpha=(2.0, 3.0), allow_missing_keys=True)
                                ])

        return pixel_transform, spatial_transform

    def generate_train_batch(self):
        #eval_idx = None
        selected_keys = np.random.choice(self.list_of_keys, self.batch_size, True, None)
        list_of_frames = []
        for i in range(len(selected_keys)):
            patient_id = selected_keys[i][:10]
            l_filtered = [x for x in self.list_of_keys if x[:10] in patient_id]
            un_filtered = [x for x in self.un_list_of_keys if x[:10] in patient_id]
            filtered = l_filtered + un_filtered
            list_of_frames.append(filtered)
        
        target_list = []
        unlabeled_list = []
        padding_need_list = []
        target_mask_list = []

        case_properties = []
        for j, frames in enumerate(list_of_frames):
            labeled_frame = frames[0]
            if 'properties' in self._data[labeled_frame].keys():
                properties = self._data[labeled_frame]['properties']
            else:
                properties = load_pickle(self._data[labeled_frame]['properties_file'])
            case_properties.append(properties)

            depth = properties['size_after_resampling'][0]
            depth_idx = np.random.choice(depth)

            frames = sorted(frames, key=lambda x: int(x[16:18]))
            frames = np.array(frames)
            global_labeled_idx = np.where(~np.char.endswith(frames, '_u'))[0]
            assert len(global_labeled_idx) == 2

            unlabeled = torch.full(size=(1, 1, *self.final_patch_size), fill_value=torch.nan, device='cuda:0')
            cropped_unlabeled = torch.full(size=(1, 1, self.crop_size, self.crop_size), fill_value=torch.nan, device='cuda:0')

            frame_indices = torch.randint(low=0, high=len(frames), size=(1,))
            video = [frames[frame_indices]]

            for frame_idx, t in enumerate(video):

                if '_u' in t:
                    if not isfile(self.un_data[t]['data_file'][:-4] + ".npy"):
                        # lets hope you know what you're doing
                        case_all_data = np.load(self.un_data[t]['data_file'][:-4] + ".npz")['data']
                    else:
                        case_all_data = np.load(self.un_data[t]['data_file'][:-4] + ".npy", self.memmap_mode)
                else:
                    if not isfile(self._data[t]['data_file'][:-4] + ".npy"):
                        # lets hope you know what you're doing
                        case_all_data = np.load(self._data[t]['data_file'][:-4] + ".npz")['data']
                    else:
                        case_all_data = np.load(self._data[t]['data_file'][:-4] + ".npy", self.memmap_mode)

                # this is for when there is just a 2d slice in case_all_data (2d support)
                if len(case_all_data.shape) == 3:
                    case_all_data = case_all_data[:, None]
                
                assert case_all_data.shape[1] == depth

                # now crop case_all_data to contain just the slice of interest. If we want additional slice above and
                # below the current slice, here is where we get them. We stack those as additional color channels
                slice_data = case_all_data[:, depth_idx]

                # case all data should now be (c, x, y)
                assert len(slice_data.shape) == 3

                if '_u' not in t:
                    image = slice_data[:-1].copy()
                    mask = slice_data[-1:].copy()
                    mask[mask < 0] = 0

                    image = self.preprocess(image)
                else:
                    image = slice_data.copy()

                    image = self.preprocess(image)

                unlabeled[frame_idx] = image

            with torch.no_grad():
                mean_centroid, _ = self.processor.preprocess_no_registration(data=torch.clone(unlabeled)) # T, C(1), H, W

                cropped_unlabeled, padding_need = self.processor.crop_and_pad(data=unlabeled, mean_centroid=mean_centroid)
                padding_need_list.append(padding_need)

            #cropped_unlabeled = (cropped_unlabeled - cropped_unlabeled.min()) / (cropped_unlabeled.max() - cropped_unlabeled.min() + 1e-8)
            #cropped_unlabeled = cropped_unlabeled + 1e-8
            #temp_before_augment = torch.clone(cropped_unlabeled).to('cpu')
            #seed = random.randint(0, 2**32-1)
            #for t in range(len(cropped_unlabeled)):
            #    cropped_unlabeled[t] = self.augment(image=cropped_unlabeled[t], seed=seed)


            #subtrahend = (cropped_unlabeled.max() + cropped_unlabeled.min()).item()
            subtrahend = cropped_unlabeled.min().item()
            divisor = (cropped_unlabeled.max() - cropped_unlabeled.min() + 1e-8).item()

            cropped_unlabeled = NormalizeIntensity(subtrahend=subtrahend, 
                                                divisor=divisor, 
                                                )(cropped_unlabeled)
                    
            #cropped_unlabeled = NormalizeIntensity(subtrahend=subtrahend, 
            #                                    divisor=divisor, 
            #                                    )(2 * cropped_unlabeled)
            
            cropped_unlabeled = torch.clamp(cropped_unlabeled, 0, 1)
            
            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(1, 1)
            #print(cropped_unlabeled.min())
            #print(cropped_unlabeled.max())
            #ax.imshow(cropped_unlabeled[0, 0].cpu(), cmap='gray')
            #plt.show()
            
            #assert cropped_unlabeled.min() >= 0 and cropped_unlabeled.max() <= 1

            unlabeled_list.append(cropped_unlabeled)

        unlabeled = torch.stack(unlabeled_list, dim=1) # T, B, 1, H, W
        padding_need = torch.stack(padding_need_list, dim=0) # B, 4

        keys = selected_keys

        #print(seg.shape)
        #print(np.unique(seg))
        #matplotlib.use('QtAgg')
        #fig, ax = plt.subplots(1, 2)
        #ax[0].imshow(labeled[3, 0], cmap='gray')
        #ax[1].imshow(seg[3, 0], cmap='gray')
        #plt.show()

        return {'unlabeled':unlabeled.squeeze(0),
                'padding_need': padding_need,
                'properties': case_properties, 
                "keys": keys}
    
    

if __name__ == "__main__":
    t = "Task002_Heart"
    p = join(preprocessing_output_dir, t, "stage1")
    dataset = load_dataset(p)
    with open(join(join(preprocessing_output_dir, t), "plans_stage1.pkl"), 'rb') as f:
        plans = pickle.load(f)
    unpack_dataset(p)
    dl = DataLoader3D(dataset, (32, 32, 32), (32, 32, 32), 2, oversample_foreground_percent=0.33)
    dl = DataLoader3D(dataset, np.array(plans['patch_size']).astype(int), np.array(plans['patch_size']).astype(int), 2,
                      oversample_foreground_percent=0.33)
    dl2d = DataLoader2D(dataset, (64, 64), np.array(plans['patch_size']).astype(int)[1:], 12,
                        oversample_foreground_percent=0.33)
