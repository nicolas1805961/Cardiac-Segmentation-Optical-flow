from batchgenerators.transforms.abstract_transforms import AbstractTransform
import numpy as np

from builtins import range

from batchgenerators.augmentations.spatial_transformations import augment_spatial_2, \
    augment_channel_translation, \
    augment_transpose_axes, augment_zoom, augment_resize, augment_rot90

from batchgenerators.augmentations.utils import create_zero_centered_coordinate_mesh, elastic_deform_coordinates, \
    interpolate_img, \
    rotate_coords_2d, rotate_coords_3d, scale_coords, resize_segmentation, resize_multichannel_image, \
    elastic_deform_coordinates_2

from batchgenerators.augmentations.crop_and_pad_augmentations import random_crop as random_crop_aug
from batchgenerators.augmentations.crop_and_pad_augmentations import center_crop as center_crop_aug
from batchgenerators.augmentations.crop_and_pad_augmentations import get_lbs_for_random_crop, get_lbs_for_center_crop

import random

import numpy as np
from numpy.random import RandomState
from batchgenerators.augmentations.utils import get_range_val, mask_random_squares
from builtins import range
from scipy.ndimage import gaussian_filter
from typing import Union, Tuple, Callable
from skimage.transform import resize

from nnunet.training.data_augmentation.downsampling import downsample_seg_for_ds_transform3, downsample_x_for_ds_transform2, downsample_seg_for_ds_transform2



class DownsampleSegForDSTransform2(AbstractTransform):
    '''
    data_dict['output_key'] will be a list of segmentations scaled according to ds_scales
    '''
    def __init__(self, ds_scales=(1, 0.5, 0.25), order=0, input_key="seg", output_key="seg", axes=None):
        self.axes = axes
        self.output_key = output_key
        self.input_key = input_key
        self.order = order
        self.ds_scales = ds_scales

    def __call__(self, **data_dict):
        for i, seg in enumerate(data_dict[self.input_key]):
            data_dict[self.output_key][i] = downsample_seg_for_ds_transform2(seg, self.ds_scales,
                                                                      self.order, self.axes)
        return data_dict


class DownsampleX(AbstractTransform):
    '''
    data_dict['output_key'] will be a list of segmentations scaled according to ds_scales
    '''
    def __init__(self, ds_scales, input_key="data", output_key="data", order=3):
        self.output_key = output_key
        self.input_key = input_key
        self.ds_scales = ds_scales
        self.order = order

    def __call__(self, **data_dict):
        for i, data in enumerate(data_dict[self.input_key]):
            data_dict[self.output_key][i] = downsample_x_for_ds_transform2(data, self.ds_scales, self.order)
        return data_dict


class DownsampleSegForDSTransform3(AbstractTransform):
    '''
    returns one hot encodings of the segmentation maps if downsampling has occured (no one hot for highest resolution)
    downsampled segmentations are smooth, not 0/1

    returns torch tensors, not numpy arrays!

    always uses seg channel 0!!

    you should always give classes! Otherwise weird stuff may happen
    '''
    def __init__(self, ds_scales=(1, 0.5, 0.25), input_key="seg", output_key="seg", classes=None):
        self.classes = classes
        self.output_key = output_key
        self.input_key = input_key
        self.ds_scales = ds_scales

    def __call__(self, **data_dict):
        for i, seg in enumerate(data_dict[self.input_key]):
            data_dict[self.output_key][i] = downsample_seg_for_ds_transform3(seg[:, 0], self.ds_scales, self.classes)
        return data_dict


class RemoveLabelTransform(AbstractTransform):
    '''
    Replaces all pixels in data_dict[input_key] that have value remove_label with replace_with and saves the result to
    data_dict[output_key]
    '''

    def __init__(self, remove_label, replace_with=0, input_key="seg", output_key="seg"):
        self.output_key = output_key
        self.input_key = input_key
        self.replace_with = replace_with
        self.remove_label = remove_label

    def __call__(self, **data_dict):
        for i, seg, in enumerate(data_dict[self.input_key]):
            seg[seg == self.remove_label] = self.replace_with
            data_dict[self.output_key][i] = seg
        return data_dict


class Convert3DTo2DTransform(AbstractTransform):
    def __init__(self):
        pass

    def convert_3d_to_2d_generator(self, data_dict):
        for i, data in enumerate(data_dict['data']):
            shp = data.shape
            data_dict['data'][i] = data.reshape((shp[0], shp[1] * shp[2], shp[3], shp[4]))
            data_dict['orig_shape_data'] = shp
        if data_dict['seg'] is not None:
            for i, seg in enumerate(data_dict['seg']):
                shp = data_dict['seg'].shape
                data_dict['seg'][i] = seg.reshape((shp[0], shp[1] * shp[2], shp[3], shp[4]))
                data_dict['orig_shape_seg'] = shp
        return data_dict

    def __call__(self, **data_dict):
        return self.convert_3d_to_2d_generator(data_dict)


class SegChannelSelectionTransform(AbstractTransform):
    """Segmentations may have more than one channel. This transform selects segmentation channels

    Args:
        channels (list of int): List of channels to be kept.

    """

    def __init__(self, channels, keep_discarded_seg=False, label_key="seg"):
        self.label_key = label_key
        self.channels = channels
        self.keep_discarded = keep_discarded_seg

    def __call__(self, **data_dict):
        seg_list = data_dict.get(self.label_key)

        if seg_list is None:
            warn("You used SegChannelSelectionTransform but there is no 'seg' key in your data_dict, returning "
                 "data_dict unmodified", Warning)
        else:
            for j, seg in enumerate(seg_list):
                if self.keep_discarded:
                    discarded_seg_idx = [i for i in range(len(seg[0])) if i not in self.channels]
                    data_dict['discarded_seg'] = seg[:, discarded_seg_idx]
                data_dict[self.label_key][j] = seg[:, self.channels]
        return data_dict


def crop(data, middle, seg=None, crop_size=128, margins=(0, 0, 0), crop_type="center",
         pad_mode='constant', pad_kwargs={'constant_values': 0},
         pad_mode_seg='constant', pad_kwargs_seg={'constant_values': 0}):
    """
    crops data and seg (seg may be None) to crop_size. Whether this will be achieved via center or random crop is
    determined by crop_type. Margin will be respected only for random_crop and will prevent the crops form being closer
    than margin to the respective image border. crop_size can be larger than data_shape - margin -> data/seg will be
    padded with zeros in that case. margins can be negative -> results in padding of data/seg followed by cropping with
    margin=0 for the appropriate axes

    :param data: b, c, x, y(, z)
    :param seg:
    :param crop_size:
    :param margins: distance from each border, can be int or list/tuple of ints (one element for each dimension).
    Can be negative (data/seg will be padded if needed)
    :param crop_type: random or center
    :return:
    """
    if not isinstance(data, (list, tuple, np.ndarray)):
        raise TypeError("data has to be either a numpy array or a list")

    data_shape = tuple([len(data)] + list(data[0].shape))
    data_dtype = data[0].dtype
    dim = len(data_shape) - 2

    if seg is not None:
        seg_shape = tuple([len(seg)] + list(seg[0].shape))
        seg_dtype = seg[0].dtype

        if not isinstance(seg, (list, tuple, np.ndarray)):
            raise TypeError("data has to be either a numpy array or a list")

        assert all([i == j for i, j in zip(seg_shape[2:], data_shape[2:])]), "data and seg must have the same spatial " \
                                                                             "dimensions. Data: %s, seg: %s" % \
                                                                             (str(data_shape), str(seg_shape))

    if type(crop_size) not in (tuple, list, np.ndarray):
        crop_size = [crop_size] * dim
    else:
        assert len(crop_size) == len(
            data_shape) - 2, "If you provide a list/tuple as center crop make sure it has the same dimension as your " \
                             "data (2d/3d)"

    if not isinstance(margins, (np.ndarray, tuple, list)):
        margins = [margins] * dim

    data_return = np.zeros([data_shape[0], data_shape[1]] + list(crop_size), dtype=data_dtype)
    middle_return = np.zeros([data_shape[0], data_shape[1]] + list(crop_size), dtype=data_dtype)
    if seg is not None:
        seg_return = np.zeros([seg_shape[0], seg_shape[1]] + list(crop_size), dtype=seg_dtype)
    else:
        seg_return = None

    for b in range(data_shape[0]):
        data_shape_here = [data_shape[0]] + list(data[b].shape)
        if seg is not None:
            seg_shape_here = [seg_shape[0]] + list(seg[b].shape)

        if crop_type == "center":
            lbs = get_lbs_for_center_crop(crop_size, data_shape_here)
        elif crop_type == "random":
            lbs = get_lbs_for_random_crop(crop_size, data_shape_here, margins)
        else:
            raise NotImplementedError("crop_type must be either center or random")

        need_to_pad = [[0, 0]] + [[abs(min(0, lbs[d])),
                                   abs(min(0, data_shape_here[d + 2] - (lbs[d] + crop_size[d])))]
                                  for d in range(dim)]

        # we should crop first, then pad -> reduces i/o for memmaps, reduces RAM usage and improves speed
        ubs = [min(lbs[d] + crop_size[d], data_shape_here[d+2]) for d in range(dim)]
        lbs = [max(0, lbs[d]) for d in range(dim)]

        slicer_data = [slice(0, data_shape_here[1])] + [slice(lbs[d], ubs[d]) for d in range(dim)]
        data_cropped = data[b][tuple(slicer_data)]
        middle_cropped = middle[b][tuple(slicer_data)]

        if seg_return is not None:
            slicer_seg = [slice(0, seg_shape_here[1])] + [slice(lbs[d], ubs[d]) for d in range(dim)]
            seg_cropped = seg[b][tuple(slicer_seg)]

        if any([i > 0 for j in need_to_pad for i in j]):
            data_return[b] = np.pad(data_cropped, need_to_pad, pad_mode, **pad_kwargs)
            middle_return[b] = np.pad(middle_cropped, need_to_pad, pad_mode, **pad_kwargs)
            if seg_return is not None:
                seg_return[b] = np.pad(seg_cropped, need_to_pad, pad_mode_seg, **pad_kwargs_seg)
        else:
            data_return[b] = data_cropped
            middle_return[b] = middle_cropped
            if seg_return is not None:
                seg_return[b] = seg_cropped

    return data_return, seg_return, middle_cropped

#def augment_spatial(data, seg, middle, middle_seg, inter_seg, patch_size, patch_center_dist_from_border=30,
#                    do_elastic_deform=True, alpha=(0., 1000.), sigma=(10., 13.),
#                    do_rotation=True, angle_x=(0, 2 * np.pi), angle_y=(0, 2 * np.pi), angle_z=(0, 2 * np.pi),
#                    do_scale=True, scale=(0.75, 1.25), border_mode_data='nearest', border_cval_data=0, order_data=3,
#                    border_mode_seg='constant', border_cval_seg=0, order_seg=0, random_crop=True, p_el_per_sample=1,
#                    p_scale_per_sample=1, p_rot_per_sample=1, independent_scale_for_each_axis=False,
#                    p_rot_per_axis: float = 1, p_independent_scale_per_axis: int = 1):
#    dim = len(patch_size)
#    seg_result = None
#    if seg is not None:
#        if dim == 2:
#            seg_result = np.zeros((seg.shape[0], seg.shape[1], patch_size[0], patch_size[1]), dtype=np.float32)
#        else:
#            seg_result = np.zeros((seg.shape[0], seg.shape[1], patch_size[0], patch_size[1], patch_size[2]),
#                                  dtype=np.float32)
#
#    if dim == 2:
#        data_result = np.zeros((data.shape[0], data.shape[1], patch_size[0], patch_size[1]), dtype=np.float32)
#    else:
#        data_result = np.zeros((data.shape[0], data.shape[1], patch_size[0], patch_size[1], patch_size[2]),
#                               dtype=np.float32)
#    
#    if dim == 2:
#        middle_result = np.zeros((middle.shape[0], middle.shape[1], patch_size[0], patch_size[1]), dtype=np.float32)
#    else:
#        middle_result = np.zeros((middle.shape[0], middle.shape[1], patch_size[0], patch_size[1], patch_size[2]),
#                               dtype=np.float32)
#    if middle_seg is not None:
#        if dim == 2:
#            middle_seg_result = np.zeros((middle_seg.shape[0], middle_seg.shape[1], patch_size[0], patch_size[1]), dtype=np.float32)
#        else:
#            middle_seg_result = np.zeros((middle_seg.shape[0], middle_seg.shape[1], patch_size[0], patch_size[1], patch_size[2]),
#                                dtype=np.float32)
#    else:
#        middle_seg_result = None
#    
#    if inter_seg is not None:
#        if dim == 2:
#            inter_seg_result = np.zeros((inter_seg.shape[0], inter_seg.shape[1], patch_size[0], patch_size[1]), dtype=np.float32)
#        else:
#            inter_seg_result = np.zeros((inter_seg.shape[0], inter_seg.shape[1], patch_size[0], patch_size[1], patch_size[2]),
#                                dtype=np.float32)
#    else:
#        inter_seg_result = None
#
#    if not isinstance(patch_center_dist_from_border, (list, tuple, np.ndarray)):
#        patch_center_dist_from_border = dim * [patch_center_dist_from_border]
#
#    for sample_id in range(data.shape[0]):
#        coords = create_zero_centered_coordinate_mesh(patch_size)
#        modified_coords = False
#
#        if do_elastic_deform and np.random.uniform() < p_el_per_sample:
#            a = np.random.uniform(alpha[0], alpha[1])
#            s = np.random.uniform(sigma[0], sigma[1])
#            coords = elastic_deform_coordinates(coords, a, s)
#            modified_coords = True
#
#        if do_rotation and np.random.uniform() < p_rot_per_sample:
#
#            if np.random.uniform() <= p_rot_per_axis:
#                a_x = np.random.uniform(angle_x[0], angle_x[1])
#            else:
#                a_x = 0
#
#            if dim == 3:
#                if np.random.uniform() <= p_rot_per_axis:
#                    a_y = np.random.uniform(angle_y[0], angle_y[1])
#                else:
#                    a_y = 0
#
#                if np.random.uniform() <= p_rot_per_axis:
#                    a_z = np.random.uniform(angle_z[0], angle_z[1])
#                else:
#                    a_z = 0
#
#                coords = rotate_coords_3d(coords, a_x, a_y, a_z)
#            else:
#                coords = rotate_coords_2d(coords, a_x)
#            modified_coords = True
#
#        if do_scale and np.random.uniform() < p_scale_per_sample:
#            if independent_scale_for_each_axis and np.random.uniform() < p_independent_scale_per_axis:
#                sc = []
#                for _ in range(dim):
#                    if np.random.random() < 0.5 and scale[0] < 1:
#                        sc.append(np.random.uniform(scale[0], 1))
#                    else:
#                        sc.append(np.random.uniform(max(scale[0], 1), scale[1]))
#            else:
#                if np.random.random() < 0.5 and scale[0] < 1:
#                    sc = np.random.uniform(scale[0], 1)
#                else:
#                    sc = np.random.uniform(max(scale[0], 1), scale[1])
#
#            coords = scale_coords(coords, sc)
#            modified_coords = True
#
#        # now find a nice center location 
#        if modified_coords:
#            for d in range(dim):
#                if random_crop:
#                    ctr = np.random.uniform(patch_center_dist_from_border[d],
#                                            data.shape[d + 2] - patch_center_dist_from_border[d])
#                else:
#                    ctr = data.shape[d + 2] / 2. - 0.5
#                coords[d] += ctr
#            for channel_id in range(data.shape[1]):
#                data_result[sample_id, channel_id] = interpolate_img(data[sample_id, channel_id], coords, order_data,
#                                                                     border_mode_data, cval=border_cval_data)
#            for channel_id in range(middle.shape[1]):
#                middle_result[sample_id, channel_id] = interpolate_img(middle[sample_id, channel_id], coords, order_data,
#                                                                     border_mode_data, cval=border_cval_data)
#            if seg is not None:
#                for channel_id in range(seg.shape[1]):
#                    seg_result[sample_id, channel_id] = interpolate_img(seg[sample_id, channel_id], coords, order_seg,
#                                                                        border_mode_seg, cval=border_cval_seg,
#                                                                        is_seg=True)
#                if middle_seg is not None:
#                    for channel_id in range(middle_seg.shape[1]):
#                        middle_seg_result[sample_id, channel_id] = interpolate_img(middle_seg[sample_id, channel_id], coords, order_seg,
#                                                                            border_mode_seg, cval=border_cval_seg,
#                                                                            is_seg=True)
#                if inter_seg is not None:
#                    for channel_id in range(inter_seg.shape[1]):
#                        inter_seg_result[sample_id, channel_id] = interpolate_img(inter_seg[sample_id, channel_id], coords, order_seg,
#                                                                            border_mode_seg, cval=border_cval_seg,
#                                                                            is_seg=True)
#        else:
#            if seg is None:
#                s = None
#                middle_s = None
#                inter_s = None
#            else:
#                s = seg[sample_id:sample_id + 1]
#                if middle_seg is not None:
#                    middle_s = middle_seg[sample_id:sample_id + 1]
#                else:
#                    middle_s = None
#                if inter_seg is not None:
#                    inter_s = inter_seg[sample_id:sample_id + 1]
#                else:
#                    inter_s = None
#            if random_crop:
#                margin = [patch_center_dist_from_border[d] - patch_size[d] // 2 for d in range(dim)]
#                d, s = random_crop_aug(data[sample_id:sample_id + 1], s, patch_size, margin)
#                d_middle, middle_s = random_crop_aug(middle[sample_id:sample_id + 1], middle_s, patch_size, margin)
#                if inter_seg is not None:
#                    _, inter_s = random_crop_aug(data[sample_id:sample_id + 1], inter_s, patch_size, margin)
#            else:
#                d, s = center_crop_aug(data[sample_id:sample_id + 1], patch_size, s)
#                d_middle, middle_s = center_crop_aug(middle[sample_id:sample_id + 1], patch_size, middle_s)
#                if inter_seg is not None:
#                    _, inter_s = center_crop_aug(data[sample_id:sample_id + 1], patch_size, inter_s)
#
#            data_result[sample_id] = d[0]
#            middle_result[sample_id] = d_middle[0]
#            if seg is not None:
#                seg_result[sample_id] = s[0]
#                if middle_seg is not None:
#                    middle_seg_result[sample_id] = middle_s[0]
#                if inter_seg is not None:
#                    inter_seg_result[sample_id] = inter_s[0]
#
#    return data_result, seg_result, middle_result, middle_seg_result, inter_seg_result


def augment_spatial(data, seg, patch_size, random_state, patch_center_dist_from_border=30,
                    do_elastic_deform=True, alpha=(0., 1000.), sigma=(10., 13.),
                    do_rotation=True, angle_x=(0, 2 * np.pi), angle_y=(0, 2 * np.pi), angle_z=(0, 2 * np.pi),
                    do_scale=True, scale=(0.75, 1.25), border_mode_data='nearest', border_cval_data=0, order_data=3,
                    border_mode_seg='constant', border_cval_seg=0, order_seg=0, random_crop=True, p_el_per_sample=1,
                    p_scale_per_sample=1, p_rot_per_sample=1, independent_scale_for_each_axis=False,
                    p_rot_per_axis: float = 1, p_independent_scale_per_axis: int = 1):
    dim = len(patch_size)
    seg_result = None
    if seg is not None:
        if dim == 2:
            seg_result = np.zeros((seg.shape[0], seg.shape[1], patch_size[0], patch_size[1]), dtype=np.float32)
        else:
            seg_result = np.zeros((seg.shape[0], seg.shape[1], patch_size[0], patch_size[1], patch_size[2]),
                                  dtype=np.float32)

    if dim == 2:
        data_result = np.zeros((data.shape[0], data.shape[1], patch_size[0], patch_size[1]), dtype=np.float32)
    else:
        data_result = np.zeros((data.shape[0], data.shape[1], patch_size[0], patch_size[1], patch_size[2]),
                               dtype=np.float32)

    if not isinstance(patch_center_dist_from_border, (list, tuple, np.ndarray)):
        patch_center_dist_from_border = dim * [patch_center_dist_from_border]

    for sample_id in range(data.shape[0]):
        coords = create_zero_centered_coordinate_mesh(patch_size)
        modified_coords = False

        if do_elastic_deform and random_state.uniform() < p_el_per_sample:
            a = random_state.uniform(alpha[0], alpha[1])
            s = random_state.uniform(sigma[0], sigma[1])
            coords = elastic_deform_coordinates(coords, a, s)
            modified_coords = True

        if do_rotation and random_state.uniform() < p_rot_per_sample:

            if random_state.uniform() <= p_rot_per_axis:
                a_x = random_state.uniform(angle_x[0], angle_x[1])
            else:
                a_x = 0

            if dim == 3:
                if random_state.uniform() <= p_rot_per_axis:
                    a_y = random_state.uniform(angle_y[0], angle_y[1])
                else:
                    a_y = 0

                if random_state.uniform() <= p_rot_per_axis:
                    a_z = random_state.uniform(angle_z[0], angle_z[1])
                else:
                    a_z = 0

                coords = rotate_coords_3d(coords, a_x, a_y, a_z)
            else:
                coords = rotate_coords_2d(coords, a_x)
            modified_coords = True

        if do_scale and random_state.uniform() < p_scale_per_sample:
            if independent_scale_for_each_axis and random_state.uniform() < p_independent_scale_per_axis:
                sc = []
                for _ in range(dim):
                    if random_state.random() < 0.5 and scale[0] < 1:
                        sc.append(random_state.uniform(scale[0], 1))
                    else:
                        sc.append(random_state.uniform(max(scale[0], 1), scale[1]))
            else:
                if random_state.random() < 0.5 and scale[0] < 1:
                    sc = random_state.uniform(scale[0], 1)
                else:
                    sc = random_state.uniform(max(scale[0], 1), scale[1])

            coords = scale_coords(coords, sc)
            modified_coords = True

        # now find a nice center location 
        if modified_coords:
            for d in range(dim):
                if random_crop:
                    ctr = random_state.uniform(patch_center_dist_from_border[d],
                                            data.shape[d + 2] - patch_center_dist_from_border[d])
                else:
                    ctr = data.shape[d + 2] / 2. - 0.5
                coords[d] += ctr
            for channel_id in range(data.shape[1]):
                data_result[sample_id, channel_id] = interpolate_img(data[sample_id, channel_id], coords, order_data,
                                                                     border_mode_data, cval=border_cval_data)
            if seg is not None:
                for channel_id in range(seg.shape[1]):
                    seg_result[sample_id, channel_id] = interpolate_img(seg[sample_id, channel_id], coords, order_seg,
                                                                        border_mode_seg, cval=border_cval_seg,
                                                                        is_seg=True)
        else:
            if seg is None:
                s = None
            else:
                s = seg[sample_id:sample_id + 1]
            if random_crop:
                margin = [patch_center_dist_from_border[d] - patch_size[d] // 2 for d in range(dim)]
                d, s = random_crop_aug(data[sample_id:sample_id + 1], s, patch_size, margin)
            else:
                d, s = center_crop_aug(data[sample_id:sample_id + 1], patch_size, s)
            data_result[sample_id] = d[0]
            if seg is not None:
                seg_result[sample_id] = s[0]
    return data_result, seg_result


class SpatialTransform(AbstractTransform):
    """The ultimate spatial transform generator. Rotation, deformation, scaling, cropping: It has all you ever dreamed
    of. Computational time scales only with patch_size, not with input patch size or type of augmentations used.
    Internally, this transform will use a coordinate grid of shape patch_size to which the transformations are
    applied (very fast). Interpolation on the image data will only be done at the very end

    Args:
        patch_size (tuple/list/ndarray of int): Output patch size

        patch_center_dist_from_border (tuple/list/ndarray of int, or int): How far should the center pixel of the
        extracted patch be from the image border? Recommended to use patch_size//2.
        This only applies when random_crop=True

        do_elastic_deform (bool): Whether or not to apply elastic deformation

        alpha (tuple of float): magnitude of the elastic deformation; randomly sampled from interval

        sigma (tuple of float): scale of the elastic deformation (small = local, large = global); randomly sampled
        from interval

        do_rotation (bool): Whether or not to apply rotation

        angle_x, angle_y, angle_z (tuple of float): angle in rad; randomly sampled from interval. Always double check
        whether axes are correct!

        do_scale (bool): Whether or not to apply scaling

        scale (tuple of float): scale range ; scale is randomly sampled from interval

        border_mode_data: How to treat border pixels in data? see scipy.ndimage.map_coordinates

        border_cval_data: If border_mode_data=constant, what value to use?

        order_data: Order of interpolation for data. see scipy.ndimage.map_coordinates

        border_mode_seg: How to treat border pixels in seg? see scipy.ndimage.map_coordinates

        border_cval_seg: If border_mode_seg=constant, what value to use?

        order_seg: Order of interpolation for seg. see scipy.ndimage.map_coordinates. Strongly recommended to use 0!
        If !=0 then you will have to round to int and also beware of interpolation artifacts if you have more then
        labels 0 and 1. (for example if you have [0, 0, 0, 2, 2, 1, 0] the neighboring [0, 0, 2] bay result in [0, 1, 2])

        random_crop: True: do a random crop of size patch_size and minimal distance to border of
        patch_center_dist_from_border. False: do a center crop of size patch_size

        independent_scale_for_each_axis: If True, a scale factor will be chosen independently for each axis.
    """

    def __init__(self, patch_size, patch_center_dist_from_border=30,
                 do_elastic_deform=True, alpha=(0., 1000.), sigma=(10., 13.),
                 do_rotation=True, angle_x=(0, 2 * np.pi), angle_y=(0, 2 * np.pi), angle_z=(0, 2 * np.pi),
                 do_scale=True, scale=(0.75, 1.25), border_mode_data='nearest', border_cval_data=0, order_data=3,
                 border_mode_seg='constant', border_cval_seg=0, order_seg=0, random_crop=True, data_key="data",
                 label_key="seg", p_el_per_sample=1, p_scale_per_sample=1, p_rot_per_sample=1,
                 independent_scale_for_each_axis=False, p_rot_per_axis:float=1, p_independent_scale_per_axis: int=1):
        self.independent_scale_for_each_axis = independent_scale_for_each_axis
        self.p_rot_per_sample = p_rot_per_sample
        self.p_scale_per_sample = p_scale_per_sample
        self.p_el_per_sample = p_el_per_sample
        self.data_key = data_key
        self.label_key = label_key
        self.patch_size = patch_size
        self.patch_center_dist_from_border = patch_center_dist_from_border
        self.do_elastic_deform = do_elastic_deform
        self.alpha = alpha
        self.sigma = sigma
        self.do_rotation = do_rotation
        self.angle_x = angle_x
        self.angle_y = angle_y
        self.angle_z = angle_z
        self.do_scale = do_scale
        self.scale = scale
        self.border_mode_data = border_mode_data
        self.border_cval_data = border_cval_data
        self.order_data = order_data
        self.border_mode_seg = border_mode_seg
        self.border_cval_seg = border_cval_seg
        self.order_seg = order_seg
        self.random_crop = random_crop
        self.p_rot_per_axis = p_rot_per_axis
        self.p_independent_scale_per_axis = p_independent_scale_per_axis

    def __call__(self, **data_dict):
        data = data_dict.get(self.data_key)
        seg = data_dict.get(self.label_key)

        if self.patch_size is None:
            if len(data.shape) == 4:
                patch_size = (data.shape[2], data.shape[3])
            elif len(data.shape) == 5:
                patch_size = (data.shape[2], data.shape[3], data.shape[4])
            else:
                raise ValueError("only support 2D/3D batch data.")
        else:
            patch_size = self.patch_size

        seed = np.random.randint(0, 4294967296, dtype=np.int64)
        for i in range(max(len(data), len(seg))):
            d = data[i]
            s = seg[i] if i < len(seg) else None

            prng = RandomState(seed)
            ret_val = augment_spatial(d, s, patch_size=patch_size, random_state=prng,
                                    patch_center_dist_from_border=self.patch_center_dist_from_border,
                                    do_elastic_deform=self.do_elastic_deform, alpha=self.alpha, sigma=self.sigma,
                                    do_rotation=self.do_rotation, angle_x=self.angle_x, angle_y=self.angle_y,
                                    angle_z=self.angle_z, do_scale=self.do_scale, scale=self.scale,
                                    border_mode_data=self.border_mode_data,
                                    border_cval_data=self.border_cval_data, order_data=self.order_data,
                                    border_mode_seg=self.border_mode_seg, border_cval_seg=self.border_cval_seg,
                                    order_seg=self.order_seg, random_crop=self.random_crop,
                                    p_el_per_sample=self.p_el_per_sample, p_scale_per_sample=self.p_scale_per_sample,
                                    p_rot_per_sample=self.p_rot_per_sample,
                                    independent_scale_for_each_axis=self.independent_scale_for_each_axis,
                                    p_rot_per_axis=self.p_rot_per_axis, 
                                    p_independent_scale_per_axis=self.p_independent_scale_per_axis)
        
            data_dict[self.data_key][i] = ret_val[0]
            if s is not None:
                data_dict[self.label_key][i] = ret_val[1]

        #assert np.any(data_dict['data'] != data_dict['middle'])

        return data_dict


class DataChannelSelectionTransform(AbstractTransform):
    """Selects color channels from the batch and discards the others.

    Args:
        channels (list of int): List of channels to be kept.

    """

    def __init__(self, channels, data_key="data"):
        self.data_key = data_key
        self.channels = channels

    def __call__(self, **data_dict):
        data_list = data_dict[self.data_key]
        for i, data in enumerate(data_list):
            data_dict[self.data_key][i] = data[:, self.channels]
        return data_dict


def augment_gaussian_noise(data_sample: np.ndarray, random_state, noise_variance: Tuple[float, float] = (0, 0.1),
                           p_per_channel: float = 1, per_channel: bool = False) -> np.ndarray:
    if not per_channel:
        variance = noise_variance[0] if noise_variance[0] == noise_variance[1] else \
            random_state.uniform(noise_variance[0], noise_variance[1])
    else:
        variance = None
    for c in range(data_sample.shape[0]):
        if random_state.uniform() < p_per_channel:
            # lol good luck reading this
            variance_here = variance if variance is not None else \
                noise_variance[0] if noise_variance[0] == noise_variance[1] else \
                    random.uniform(noise_variance[0], noise_variance[1])
            # bug fixed: https://github.com/MIC-DKFZ/batchgenerators/issues/86
            data_sample[c] = data_sample[c] + random_state.normal(0.0, variance_here, size=data_sample[c].shape)
    return data_sample


class GaussianNoiseTransform(AbstractTransform):
    def __init__(self, noise_variance=(0, 0.1), p_per_sample=1, p_per_channel: float = 1,
                 per_channel: bool = False, data_key="data"):
        """
        Adds additive Gaussian Noise

        :param noise_variance: variance is uniformly sampled from that range
        :param p_per_sample:
        :param p_per_channel:
        :param per_channel: if True, each channel will get its own variance sampled from noise_variance
        :param data_key:

        CAREFUL: This transform will modify the value range of your data!
        """
        self.p_per_sample = p_per_sample
        self.data_key = data_key
        self.noise_variance = noise_variance
        self.p_per_channel = p_per_channel
        self.per_channel = per_channel

    def __call__(self, **data_dict):
        seed = np.random.randint(0, 4294967296, dtype=np.int64)
        for i, data in enumerate(data_dict[self.data_key]):
            prng = RandomState(seed)
            for b in range(len(data)):
                if prng.uniform() < self.p_per_sample:
                    data_dict[self.data_key][i][b] = augment_gaussian_noise(data[b], prng, self.noise_variance,
                                                                        self.p_per_channel, self.per_channel)
        return data_dict


def augment_gaussian_blur(data_sample: np.ndarray, random_state, sigma_range: Tuple[float, float], per_channel: bool = True,
                          p_per_channel: float = 1, different_sigma_per_axis: bool = False,
                          p_isotropic: float = 0) -> np.ndarray:
    if not per_channel:
        # Godzilla Had a Stroke Trying to Read This and F***ing Died
        # https://i.kym-cdn.com/entries/icons/original/000/034/623/Untitled-3.png
        sigma = get_range_val(sigma_range) if ((not different_sigma_per_axis) or
                                               ((random_state.uniform() < p_isotropic) and
                                                different_sigma_per_axis)) \
            else [get_range_val(sigma_range) for _ in data_sample.shape[1:]]
    else:
        sigma = None
    for c in range(data_sample.shape[0]):
        if random_state.uniform() <= p_per_channel:
            if per_channel:
                sigma = get_range_val(sigma_range) if ((not different_sigma_per_axis) or
                                                       ((random_state.uniform() < p_isotropic) and
                                                        different_sigma_per_axis)) \
                    else [get_range_val(sigma_range) for _ in data_sample.shape[1:]]
            data_sample[c] = gaussian_filter(data_sample[c], sigma, order=0)
    return data_sample


class GaussianBlurTransform(AbstractTransform):
    def __init__(self, blur_sigma: Tuple[float, float] = (1, 5), different_sigma_per_channel: bool = True,
                 different_sigma_per_axis: bool = False, p_isotropic: float = 0, p_per_channel: float = 1,
                 p_per_sample: float = 1, data_key: str = "data"):
        """

        :param blur_sigma:
        :param data_key:
        :param different_sigma_per_axis: if True, anisotropic kernels are possible
        :param p_isotropic: only applies if different_sigma_per_axis=True, p_isotropic is the proportion of isotropic
        kernels, the rest gets random sigma per axis
        :param different_sigma_per_channel: whether to sample a sigma for each channel or all channels at once
        :param p_per_channel: probability of applying gaussian blur for each channel. Default = 1 (all channels are
        blurred with prob 1)
        """
        self.p_per_sample = p_per_sample
        self.different_sigma_per_channel = different_sigma_per_channel
        self.p_per_channel = p_per_channel
        self.data_key = data_key
        self.blur_sigma = blur_sigma
        self.different_sigma_per_axis = different_sigma_per_axis
        self.p_isotropic = p_isotropic

    def __call__(self, **data_dict):
        seed = np.random.randint(0, 4294967296, dtype=np.int64)
        for i, data in enumerate(data_dict[self.data_key]):
            prng = RandomState(seed)
            for b in range(len(data)):
                if prng.uniform() < self.p_per_sample:
                    data_dict[self.data_key][i][b] = augment_gaussian_blur(data[b], prng, self.blur_sigma,
                                                                    self.different_sigma_per_channel,
                                                                    self.p_per_channel,
                                                                    different_sigma_per_axis=self.different_sigma_per_axis,
                                                                    p_isotropic=self.p_isotropic)
        return data_dict


def augment_brightness_multiplicative(data_sample, random_state, multiplier_range=(0.5, 2), per_channel=True):
    multiplier = random_state.uniform(multiplier_range[0], multiplier_range[1])
    if not per_channel:
        data_sample *= multiplier
    else:
        for c in range(data_sample.shape[0]):
            multiplier = random_state.uniform(multiplier_range[0], multiplier_range[1])
            data_sample[c] *= multiplier
    return data_sample


class BrightnessMultiplicativeTransform(AbstractTransform):
    def __init__(self, multiplier_range=(0.5, 2), per_channel=True, data_key="data", p_per_sample=1):
        """
        Augments the brightness of data. Multiplicative brightness is sampled from multiplier_range
        :param multiplier_range: range to uniformly sample the brightness modifier from
        :param per_channel:  whether to use the same brightness modifier for all color channels or a separate one for
        each channel
        :param data_key:
        :param p_per_sample:
        """
        self.p_per_sample = p_per_sample
        self.data_key = data_key
        self.multiplier_range = multiplier_range
        self.per_channel = per_channel

    def __call__(self, **data_dict):
        seed = np.random.randint(0, 4294967296, dtype=np.int64)
        for i, data in enumerate(data_dict[self.data_key]):
            prng = RandomState(seed)
            for b in range(len(data)):
                if prng.uniform() < self.p_per_sample:
                    data_dict[self.data_key][i][b] = augment_brightness_multiplicative(data[b], prng,
                                                                                self.multiplier_range,
                                                                                self.per_channel)
        return data_dict


def augment_brightness_additive(data_sample, random_state, mu:float, sigma:float , per_channel:bool=True, p_per_channel:float=1.):
    """
    data_sample must have shape (c, x, y(, z)))
    :param data_sample: 
    :param mu: 
    :param sigma: 
    :param per_channel: 
    :param p_per_channel: 
    :return: 
    """
    if not per_channel:
        rnd_nb = random_state.normal(mu, sigma)
        for c in range(data_sample.shape[0]):
            if random_state.uniform() <= p_per_channel:
                data_sample[c] += rnd_nb
    else:
        for c in range(data_sample.shape[0]):
            if random_state.uniform() <= p_per_channel:
                rnd_nb = random_state.normal(mu, sigma)
                data_sample[c] += rnd_nb
    return data_sample


class BrightnessTransform(AbstractTransform):
    def __init__(self, mu, sigma, per_channel=True, data_key="data", p_per_sample=1, p_per_channel=1):
        """
        Augments the brightness of data. Additive brightness is sampled from Gaussian distribution with mu and sigma
        :param mu: mean of the Gaussian distribution to sample the added brightness from
        :param sigma: standard deviation of the Gaussian distribution to sample the added brightness from
        :param per_channel: whether to use the same brightness modifier for all color channels or a separate one for
        each channel
        :param data_key:
        :param p_per_sample:
        """
        self.p_per_sample = p_per_sample
        self.data_key = data_key
        self.mu = mu
        self.sigma = sigma
        self.per_channel = per_channel
        self.p_per_channel = p_per_channel

    def __call__(self, **data_dict):
        seed = np.random.randint(0, 4294967296, dtype=np.int64)
        for i, data in enumerate(data_dict[self.data_key]):
            prng = RandomState(seed)
            for b in range(data.shape[0]):
                if prng.uniform() < self.p_per_sample:
                    data_dict[self.data_key][i][b] = augment_brightness_additive(data[b], prng, self.mu, self.sigma, self.per_channel,
                                                        p_per_channel=self.p_per_channel)

        return data_dict


def augment_contrast(data_sample: np.ndarray,
                     random_state,
                     contrast_range: Union[Tuple[float, float], Callable[[], float]] = (0.75, 1.25),
                     preserve_range: bool = True,
                     per_channel: bool = True,
                     p_per_channel: float = 1) -> np.ndarray:
    if not per_channel:
        if callable(contrast_range):
            factor = contrast_range()
        else:
            if random_state.random() < 0.5 and contrast_range[0] < 1:
                factor = random_state.uniform(contrast_range[0], 1)
            else:
                factor = random_state.uniform(max(contrast_range[0], 1), contrast_range[1])

        for c in range(data_sample.shape[0]):
            if random_state.uniform() < p_per_channel:
                mn = data_sample[c].mean()
                if preserve_range:
                    minm = data_sample[c].min()
                    maxm = data_sample[c].max()

                data_sample[c] = (data_sample[c] - mn) * factor + mn

                if preserve_range:
                    data_sample[c][data_sample[c] < minm] = minm
                    data_sample[c][data_sample[c] > maxm] = maxm
    else:
        for c in range(data_sample.shape[0]):
            if random_state.uniform() < p_per_channel:
                if callable(contrast_range):
                    factor = contrast_range()
                else:
                    if random_state.random() < 0.5 and contrast_range[0] < 1:
                        factor = random_state.uniform(contrast_range[0], 1)
                    else:
                        factor = random_state.uniform(max(contrast_range[0], 1), contrast_range[1])

                mn = data_sample[c].mean()
                if preserve_range:
                    minm = data_sample[c].min()
                    maxm = data_sample[c].max()

                data_sample[c] = (data_sample[c] - mn) * factor + mn

                if preserve_range:
                    data_sample[c][data_sample[c] < minm] = minm
                    data_sample[c][data_sample[c] > maxm] = maxm
    return data_sample


class ContrastAugmentationTransform(AbstractTransform):
    def __init__(self,
                 contrast_range: Union[Tuple[float, float], Callable[[], float]] = (0.75, 1.25),
                 preserve_range: bool = True,
                 per_channel: bool = True,
                 data_key: str = "data",
                 p_per_sample: float = 1,
                 p_per_channel: float = 1):
        """
        Augments the contrast of data
        :param contrast_range:
            (float, float): range from which to sample a random contrast that is applied to the data. If
                            one value is smaller and one is larger than 1, half of the contrast modifiers will be >1
                            and the other half <1 (in the inverval that was specified)
            callable      : must be contrast_range() -> float
        :param preserve_range: if True then the intensity values after contrast augmentation will be cropped to min and
        max values of the data before augmentation.
        :param per_channel: whether to use the same contrast modifier for all color channels or a separate one for each
        channel
        :param data_key:
        :param p_per_sample:
        """
        self.p_per_sample = p_per_sample
        self.data_key = data_key
        self.contrast_range = contrast_range
        self.preserve_range = preserve_range
        self.per_channel = per_channel
        self.p_per_channel = p_per_channel

    def __call__(self, **data_dict):
        seed = np.random.randint(0, 4294967296, dtype=np.int64)
        for i, data in enumerate(data_dict[self.data_key]):
            prng = RandomState(seed)
            for b in range(len(data)):
                if prng.uniform() < self.p_per_sample:
                    data_dict[self.data_key][i][b] = augment_contrast(data[b],
                                                                    prng,
                                                               contrast_range=self.contrast_range,
                                                               preserve_range=self.preserve_range,
                                                               per_channel=self.per_channel,
                                                               p_per_channel=self.p_per_channel)
        return data_dict


def uniform(low, high, random_state, size=None):
    """
    wrapper for np.random.uniform to allow it to handle low=high
    :param low:
    :param high:
    :return:
    """
    if low == high:
        if size is None:
            return low
        else:
            return np.ones(size) * low
    else:
        return random_state.uniform(low, high, size)


def augment_linear_downsampling_scipy(data_sample, random_state, zoom_range=(0.5, 1), per_channel=True, p_per_channel=1,
                                      channels=None, order_downsample=1, order_upsample=0, ignore_axes=None):
    '''
    Downsamples each sample (linearly) by a random factor and upsamples to original resolution again (nearest neighbor)

    Info:
    * Uses scipy zoom for resampling. A bit faster than nilearn.
    * Resamples all dimensions (channels, x, y, z) with same downsampling factor (like isotropic=True from
    linear_downsampling_generator_nilearn)

    Args:
        zoom_range: can be either tuple/list/np.ndarray or tuple of tuple. If tuple/list/np.ndarray, then the zoom
        factor will be sampled from zoom_range[0], zoom_range[1] (zoom < 0 = downsampling!). If tuple of tuple then
        each inner tuple will give a sampling interval for each axis (allows for different range of zoom values for
        each axis

        p_per_channel: probability for downsampling/upsampling a channel

        per_channel (bool): whether to draw a new zoom_factor for each channel or keep one for all channels

        channels (list, tuple): if None then all channels can be augmented. If list then only the channel indices can
        be augmented (but may not always be depending on p_per_channel)

        order_downsample:

        order_upsample:

        ignore_axes: tuple/list

    '''
    if not isinstance(zoom_range, (list, tuple, np.ndarray)):
        zoom_range = [zoom_range]

    shp = np.array(data_sample.shape[1:])
    dim = len(shp)

    if not per_channel:
        if isinstance(zoom_range[0], (tuple, list, np.ndarray)):
            assert len(zoom_range) == dim
            zoom = np.array([uniform(i[0], i[1], random_state) for i in zoom_range])
        else:
            zoom = uniform(zoom_range[0], zoom_range[1], random_state)

        target_shape = np.round(shp * zoom).astype(int)

        if ignore_axes is not None:
            for i in ignore_axes:
                target_shape[i] = shp[i]

    if channels is None:
        channels = list(range(data_sample.shape[0]))

    for c in channels:
        if random_state.uniform() < p_per_channel:
            if per_channel:
                if isinstance(zoom_range[0], (tuple, list, np.ndarray)):
                    assert len(zoom_range) == dim
                    zoom = np.array([uniform(i[0], i[1], random_state) for i in zoom_range])
                else:
                    zoom = uniform(zoom_range[0], zoom_range[1], random_state)

                target_shape = np.round(shp * zoom).astype(int)
                if ignore_axes is not None:
                    for i in ignore_axes:
                        target_shape[i] = shp[i]

            downsampled = resize(data_sample[c].astype(float), target_shape, order=order_downsample, mode='edge',
                                 anti_aliasing=False)
            data_sample[c] = resize(downsampled, shp, order=order_upsample, mode='edge',
                                    anti_aliasing=False)

    return data_sample


class SimulateLowResolutionTransform(AbstractTransform):
    """Downsamples each sample (linearly) by a random factor and upsamples to original resolution again
    (nearest neighbor)

    Info:
    * Uses scipy zoom for resampling.
    * Resamples all dimensions (channels, x, y, z) with same downsampling factor (like isotropic=True from
    linear_downsampling_generator_nilearn)

    Args:
        zoom_range: can be either tuple/list/np.ndarray or tuple of tuple. If tuple/list/np.ndarray, then the zoom
        factor will be sampled from zoom_range[0], zoom_range[1] (zoom < 0 = downsampling!). If tuple of tuple then
        each inner tuple will give a sampling interval for each axis (allows for different range of zoom values for
        each axis

        p_per_channel:

        per_channel (bool): whether to draw a new zoom_factor for each channel or keep one for all channels

        channels (list, tuple): if None then all channels can be augmented. If list then only the channel indices can
        be augmented (but may not always be depending on p_per_channel)

        order_downsample:

        order_upsample:
    """

    def __init__(self, zoom_range=(0.5, 1), per_channel=False, p_per_channel=1,
                 channels=None, order_downsample=1, order_upsample=0, data_key="data", p_per_sample=1,
                 ignore_axes=None):
        self.order_upsample = order_upsample
        self.order_downsample = order_downsample
        self.channels = channels
        self.per_channel = per_channel
        self.p_per_channel = p_per_channel
        self.p_per_sample = p_per_sample
        self.data_key = data_key
        self.zoom_range = zoom_range
        self.ignore_axes = ignore_axes

    def __call__(self, **data_dict):
        seed = np.random.randint(0, 4294967296, dtype=np.int64)
        for i, data in enumerate(data_dict[self.data_key]):
            prng = RandomState(seed)
            for b in range(len(data)):
                if prng.uniform() < self.p_per_sample:
                    data_dict[self.data_key][i][b] = augment_linear_downsampling_scipy(data[b], prng,
                                                                                zoom_range=self.zoom_range,
                                                                                per_channel=self.per_channel,
                                                                                p_per_channel=self.p_per_channel,
                                                                                channels=self.channels,
                                                                                order_downsample=self.order_downsample,
                                                                                order_upsample=self.order_upsample,
                                                                                ignore_axes=self.ignore_axes)
        return data_dict


def augment_gamma(data_sample, random_state, gamma_range=(0.5, 2), invert_image=False, epsilon=1e-7, per_channel=False,
                  retain_stats: Union[bool, Callable[[], bool]] = False):
    if invert_image:
        data_sample = - data_sample

    if not per_channel:
        retain_stats_here = retain_stats() if callable(retain_stats) else retain_stats
        if retain_stats_here:
            mn = data_sample.mean()
            sd = data_sample.std()
        if random_state.random() < 0.5 and gamma_range[0] < 1:
            gamma = random_state.uniform(gamma_range[0], 1)
        else:
            gamma = random_state.uniform(max(gamma_range[0], 1), gamma_range[1])
        minm = data_sample.min()
        rnge = data_sample.max() - minm
        data_sample = np.power(((data_sample - minm) / float(rnge + epsilon)), gamma) * rnge + minm
        if retain_stats_here:
            data_sample = data_sample - data_sample.mean()
            data_sample = data_sample / (data_sample.std() + 1e-8) * sd
            data_sample = data_sample + mn
    else:
        for c in range(data_sample.shape[0]):
            retain_stats_here = retain_stats() if callable(retain_stats) else retain_stats
            if retain_stats_here:
                mn = data_sample[c].mean()
                sd = data_sample[c].std()
            if random_state.random() < 0.5 and gamma_range[0] < 1:
                gamma = random_state.uniform(gamma_range[0], 1)
            else:
                gamma = random_state.uniform(max(gamma_range[0], 1), gamma_range[1])
            minm = data_sample[c].min()
            rnge = data_sample[c].max() - minm
            data_sample[c] = np.power(((data_sample[c] - minm) / float(rnge + epsilon)), gamma) * float(rnge + epsilon) + minm
            if retain_stats_here:
                data_sample[c] = data_sample[c] - data_sample[c].mean()
                data_sample[c] = data_sample[c] / (data_sample[c].std() + 1e-8) * sd
                data_sample[c] = data_sample[c] + mn
    if invert_image:
        data_sample = - data_sample
    return data_sample


class GammaTransform(AbstractTransform):
    def __init__(self, gamma_range=(0.5, 2), invert_image=False, per_channel=False, data_key="data",
                 retain_stats: Union[bool, Callable[[], bool]] = False, p_per_sample=1):
        """
        Augments by changing 'gamma' of the image (same as gamma correction in photos or computer monitors

        :param gamma_range: range to sample gamma from. If one value is smaller than 1 and the other one is
        larger then half the samples will have gamma <1 and the other >1 (in the inverval that was specified).
        Tuple of float. If one value is < 1 and the other > 1 then half the images will be augmented with gamma values
        smaller than 1 and the other half with > 1
        :param invert_image: whether to invert the image before applying gamma augmentation
        :param per_channel:
        :param data_key:
        :param retain_stats: Gamma transformation will alter the mean and std of the data in the patch. If retain_stats=True,
        the data will be transformed to match the mean and standard deviation before gamma augmentation. retain_stats
        can also be callable (signature retain_stats() -> bool)
        :param p_per_sample:
        """
        self.p_per_sample = p_per_sample
        self.retain_stats = retain_stats
        self.per_channel = per_channel
        self.data_key = data_key
        self.gamma_range = gamma_range
        self.invert_image = invert_image

    def __call__(self, **data_dict):
        seed = np.random.randint(0, 4294967296, dtype=np.int64)
        for i, data in enumerate(data_dict[self.data_key]):
            prng = RandomState(seed)
            for b in range(len(data)):
                if prng.uniform() < self.p_per_sample:
                    data_dict[self.data_key][i][b] = augment_gamma(data[b], prng, self.gamma_range,
                                                            self.invert_image,
                                                            per_channel=self.per_channel,
                                                            retain_stats=self.retain_stats)
        return data_dict


#def augment_mirroring(sample_data, middle_data, sample_seg=None, sample_middle_seg=None, sample_inter_seg=None, axes=(0, 1, 2)):
#    if (len(sample_data.shape) != 3) and (len(sample_data.shape) != 4):
#        raise Exception(
#            "Invalid dimension for sample_data and sample_seg. sample_data and sample_seg should be either "
#            "[channels, x, y] or [channels, x, y, z]")
#    if 0 in axes and np.random.uniform() < 0.5:
#        sample_data[:, :] = sample_data[:, ::-1]
#        middle_data[:, :] = middle_data[:, ::-1]
#        if sample_seg is not None:
#            sample_seg[:, :] = sample_seg[:, ::-1]
#            if sample_middle_seg is not None:
#                sample_middle_seg[:, :] = sample_middle_seg[:, ::-1]
#            if sample_inter_seg is not None:
#                sample_inter_seg[:, :] = sample_inter_seg[:, ::-1]
#    if 1 in axes and np.random.uniform() < 0.5:
#        sample_data[:, :, :] = sample_data[:, :, ::-1]
#        middle_data[:, :, :] = middle_data[:, :, ::-1]
#        if sample_seg is not None:
#            sample_seg[:, :, :] = sample_seg[:, :, ::-1]
#            if sample_middle_seg is not None:
#                sample_middle_seg[:, :, :] = sample_middle_seg[:, :, ::-1]
#            if sample_inter_seg is not None:
#                sample_inter_seg[:, :, :] = sample_inter_seg[:, :, ::-1]
#    if 2 in axes and len(sample_data.shape) == 4:
#        if np.random.uniform() < 0.5:
#            sample_data[:, :, :, :] = sample_data[:, :, :, ::-1]
#            middle_data[:, :, :, :] = middle_data[:, :, :, ::-1]
#            if sample_seg is not None:
#                sample_seg[:, :, :, :] = sample_seg[:, :, :, ::-1]
#                if sample_middle_seg is not None:
#                    sample_middle_seg[:, :, :, :] = sample_middle_seg[:, :, :, ::-1]
#                if sample_inter_seg is not None:
#                    sample_inter_seg[:, :, :, :] = sample_inter_seg[:, :, :, ::-1]
#    return sample_data, sample_seg, middle_data, sample_middle_seg, sample_inter_seg


def augment_mirroring(sample_data, random_state, sample_seg=None, axes=(0, 1, 2)):
    if (len(sample_data.shape) != 3) and (len(sample_data.shape) != 4):
        raise Exception(
            "Invalid dimension for sample_data and sample_seg. sample_data and sample_seg should be either "
            "[channels, x, y] or [channels, x, y, z]")
    if 0 in axes and random_state.uniform() < 0.5:
        sample_data[:, :] = sample_data[:, ::-1]
        if sample_seg is not None:
            sample_seg[:, :] = sample_seg[:, ::-1]
    if 1 in axes and random_state.uniform() < 0.5:
        sample_data[:, :, :] = sample_data[:, :, ::-1]
        if sample_seg is not None:
            sample_seg[:, :, :] = sample_seg[:, :, ::-1]
    if 2 in axes and len(sample_data.shape) == 4:
        if random_state.uniform() < 0.5:
            sample_data[:, :, :, :] = sample_data[:, :, :, ::-1]
            if sample_seg is not None:
                sample_seg[:, :, :, :] = sample_seg[:, :, :, ::-1]
    return sample_data, sample_seg



class MirrorTransform(AbstractTransform):
    """ Randomly mirrors data along specified axes. Mirroring is evenly distributed. Probability of mirroring along
    each axis is 0.5

    Args:
        axes (tuple of int): axes along which to mirror

    """

    def __init__(self, axes=(0, 1, 2), data_key="data", label_key="seg", p_per_sample=1):
        self.p_per_sample = p_per_sample
        self.data_key = data_key
        self.label_key = label_key
        self.axes = axes
        if max(axes) > 2:
            raise ValueError("MirrorTransform now takes the axes as the spatial dimensions. What previously was "
                             "axes=(2, 3, 4) to mirror along all spatial dimensions of a 5d tensor (b, c, x, y, z) "
                             "is now axes=(0, 1, 2). Please adapt your scripts accordingly.")

    def __call__(self, **data_dict):
        data = data_dict.get(self.data_key)
        seg = data_dict.get(self.label_key)

        seed = np.random.randint(0, 4294967296, dtype=np.int64)

        for i in range(max(len(data), len(seg))):
            d = data[i]
            s = seg[i] if i < len(seg) else None
            prng = RandomState(seed)

            for b in range(len(d)):
                if prng.uniform() < self.p_per_sample:
                    sample_seg = None
                    if s is not None:
                        sample_seg = s[b]
                    ret_val = augment_mirroring(d[b], prng, sample_seg, axes=self.axes)
                    d[b] = ret_val[0]
                    if s is not None:
                        s[b] = ret_val[1]

            data_dict[self.data_key][i] = d
            if s is not None:
                data_dict[self.label_key][i] = s

        return data_dict