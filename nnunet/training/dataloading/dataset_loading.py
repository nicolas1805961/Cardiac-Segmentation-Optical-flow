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

from collections import OrderedDict
import numpy as np
import sys
from numpy.random import RandomState
from multiprocessing import Pool
from math import ceil, floor
from scipy.signal import savgol_filter
from numpy.lib.stride_tricks import sliding_window_view

from batchgenerators.dataloading.data_loader import SlimDataLoaderBase

from nnunet.configuration import default_num_threads
from nnunet.paths import preprocessing_output_dir
from batchgenerators.utilities.file_and_folder_operations import *
import matplotlib
import matplotlib.pyplot as plt
from copy import copy

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
