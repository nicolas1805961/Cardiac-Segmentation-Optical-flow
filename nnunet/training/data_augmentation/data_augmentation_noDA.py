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

from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.channel_selection_transforms import DataChannelSelectionTransform, \
    SegChannelSelectionTransform
from batchgenerators.transforms.utility_transforms import RemoveLabelTransform, RenameTransform, NumpyToTensor

from nnunet.training.data_augmentation.custom_transforms import ConvertSegmentationToRegionsTransform
from nnunet.training.data_augmentation.default_data_augmentation import default_3D_augmentation_params
from nnunet.training.data_augmentation.downsampling import DownsampleSegForDSTransform3, DownsampleSegForDSTransform2, DownsampleX

from nnunet.training.data_augmentation.custom_transforms import Convert3DTo2DTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter

from copy import copy

from nnunet.training.data_augmentation.distance_map import DistanceMap

try:
    from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
except ImportError as ie:
    NonDetMultiThreadedAugmenter = None


def get_no_augmentation(dataloader_train, dataloader_val, params=default_3D_augmentation_params,
                        deep_supervision_scales=None, soft_ds=False,
                        classes=None, pin_memory=True, regions=None):
    """
    use this instead of get_default_augmentation (drop in replacement) to turn off all data augmentation
    """
    tr_transforms = []

    if params.get("selected_data_channels") is not None:
        tr_transforms.append(DataChannelSelectionTransform(params.get("selected_data_channels")))

    if params.get("selected_seg_channels") is not None:
        tr_transforms.append(SegChannelSelectionTransform(params.get("selected_seg_channels")))

    tr_transforms.append(RemoveLabelTransform(-1, 0))

    tr_transforms.append(RenameTransform('seg', 'target', True))

    if regions is not None:
        tr_transforms.append(ConvertSegmentationToRegionsTransform(regions, 'target', 'target'))

    if deep_supervision_scales is not None:
        if soft_ds:
            assert classes is not None
            tr_transforms.append(DownsampleSegForDSTransform3(deep_supervision_scales, 'target', 'target', classes))
        else:
            tr_transforms.append(DownsampleSegForDSTransform2(deep_supervision_scales, 0, input_key='target',
                                                              output_key='target'))

    tr_transforms.append(NumpyToTensor(['data', 'target'], 'float'))

    tr_transforms = Compose(tr_transforms)

    batchgenerator_train = MultiThreadedAugmenter(dataloader_train, tr_transforms, params.get('num_threads'),
                                                  params.get("num_cached_per_thread"),
                                                  seeds=range(params.get('num_threads')), pin_memory=pin_memory)
    batchgenerator_train.restart()

    val_transforms = []
    val_transforms.append(RemoveLabelTransform(-1, 0))
    if params.get("selected_data_channels") is not None:
        val_transforms.append(DataChannelSelectionTransform(params.get("selected_data_channels")))
    if params.get("selected_seg_channels") is not None:
        val_transforms.append(SegChannelSelectionTransform(params.get("selected_seg_channels")))

    val_transforms.append(RenameTransform('seg', 'target', True))

    if regions is not None:
        val_transforms.append(ConvertSegmentationToRegionsTransform(regions, 'target', 'target'))

    if deep_supervision_scales is not None:
        if soft_ds:
            assert classes is not None
            val_transforms.append(DownsampleSegForDSTransform3(deep_supervision_scales, 'target', 'target', classes))
        else:
            val_transforms.append(DownsampleSegForDSTransform2(deep_supervision_scales, 0, input_key='target',
                                                               output_key='target'))

    val_transforms.append(NumpyToTensor(['data', 'target'], 'float'))
    val_transforms = Compose(val_transforms)

    batchgenerator_val = MultiThreadedAugmenter(dataloader_val, val_transforms, max(params.get('num_threads') // 2, 1),
                                                params.get("num_cached_per_thread"),
                                                seeds=range(max(params.get('num_threads') // 2, 1)),
                                                pin_memory=pin_memory)
    batchgenerator_val.restart()
    return batchgenerator_train, batchgenerator_val


def get_no_augmentation_mtl(dataloader_train, dataloader_val, dataloader_un_tr, dataloader_un_val, patch_size, directional_field, params=default_3D_augmentation_params,
                        deep_supervision_scales=None, soft_ds=False,
                        classes=None, regions=None,
                        border_val_seg=-1, order_seg=1, order_data=3):
    """
    use this instead of get_default_augmentation (drop in replacement) to turn off all data augmentation
    """
    tr_transforms = []
    un_transforms = []

    if params.get("selected_data_channels") is not None:
        tr_transforms.append(DataChannelSelectionTransform(params.get("selected_data_channels")))
        un_transforms.append(DataChannelSelectionTransform(params.get("selected_data_channels")))

    if params.get("selected_seg_channels") is not None:
        tr_transforms.append(SegChannelSelectionTransform(params.get("selected_seg_channels")))

    # don't do color augmentations while in 2d mode with 3d data because the color channel is overloaded!!
    if params.get("dummy_2D") is not None and params.get("dummy_2D"):
        ignore_axes = (0,)
        tr_transforms.append(Convert3DTo2DTransform())
        un_transforms.append(Convert3DTo2DTransform())
        patch_size_spatial = patch_size[1:]
    else:
        patch_size_spatial = patch_size
        ignore_axes = None

    tr_transforms.append(SpatialTransform(
        patch_size_spatial, patch_center_dist_from_border=None,
        do_elastic_deform=False, alpha=params.get("elastic_deform_alpha"),
        sigma=params.get("elastic_deform_sigma"),
        do_rotation=False, angle_x=params.get("rotation_x"), angle_y=params.get("rotation_y"),
        angle_z=params.get("rotation_z"), p_rot_per_axis=params.get("rotation_p_per_axis"),
        do_scale=False, scale=params.get("scale_range"),
        border_mode_data=params.get("border_mode_data"), border_cval_data=0, order_data=order_data,
        border_mode_seg="constant", border_cval_seg=border_val_seg,
        order_seg=order_seg, random_crop=False, p_el_per_sample=params.get("p_eldef"),
        p_scale_per_sample=params.get("p_scale"), p_rot_per_sample=params.get("p_rot"),
        independent_scale_for_each_axis=params.get("independent_scale_factor_for_each_axis")
    ))

    un_transforms.append(SpatialTransform(
        patch_size_spatial, patch_center_dist_from_border=None,
        do_elastic_deform=False, alpha=params.get("elastic_deform_alpha"),
        sigma=params.get("elastic_deform_sigma"),
        do_rotation=False, angle_x=params.get("rotation_x"), angle_y=params.get("rotation_y"),
        angle_z=params.get("rotation_z"), p_rot_per_axis=params.get("rotation_p_per_axis"),
        do_scale=False, scale=params.get("scale_range"),
        border_mode_data=params.get("border_mode_data"), border_cval_data=0, order_data=order_data,
        border_mode_seg="constant", border_cval_seg=border_val_seg,
        order_seg=order_seg, random_crop=False, p_el_per_sample=params.get("p_eldef"),
        p_scale_per_sample=params.get("p_scale"), p_rot_per_sample=params.get("p_rot"),
        independent_scale_for_each_axis=params.get("independent_scale_factor_for_each_axis")
    ))

    tr_transforms.append(RemoveLabelTransform(-1, 0))

    tr_transforms.append(RenameTransform('seg', 'target', True))

    if regions is not None:
        tr_transforms.append(ConvertSegmentationToRegionsTransform(regions, 'target', 'target'))

    if directional_field:
        tr_transforms.append(DistanceMap(input_key='target', output_key='directional_field'))

    if deep_supervision_scales is not None:
        scales = copy(deep_supervision_scales)
        if directional_field:
            scales.insert(1, [1.0, 1.0])
        if soft_ds:
            assert classes is not None
            tr_transforms.append(DownsampleSegForDSTransform3(scales, 'target', 'target', classes))
            tr_transforms.append(DownsampleX(deep_supervision_scales, input_key='data', output_key='data', order=3))
            un_transforms.append(DownsampleX(deep_supervision_scales, input_key='data', output_key='data', order=3))
        else:
            tr_transforms.append(DownsampleSegForDSTransform2(scales, 0, input_key='target', output_key='target'))
            tr_transforms.append(DownsampleX(deep_supervision_scales, input_key='data', output_key='data', order=3))
            un_transforms.append(DownsampleX(deep_supervision_scales, input_key='data', output_key='data', order=3))

    tr_transforms.append(NumpyToTensor(['data', 'target'], 'float'))
    un_transforms.append(NumpyToTensor(['data'], 'float'))

    tr_transforms = Compose(tr_transforms)
    un_transforms = Compose(un_transforms)

    batchgenerator_train = SingleThreadedAugmenter(dataloader_train, tr_transforms)

    if dataloader_un_tr is not None:
        batchgenerator_train_unlabeled = SingleThreadedAugmenter(dataloader_un_tr, un_transforms)
        batchgenerator_val_unlabeled = SingleThreadedAugmenter(dataloader_un_val, un_transforms)
    else:
        batchgenerator_train_unlabeled = None
        batchgenerator_val_unlabeled = None

    val_transforms = []
    val_transforms.append(RemoveLabelTransform(-1, 0))
    if params.get("selected_data_channels") is not None:
        val_transforms.append(DataChannelSelectionTransform(params.get("selected_data_channels")))
    if params.get("selected_seg_channels") is not None:
        val_transforms.append(SegChannelSelectionTransform(params.get("selected_seg_channels")))

    val_transforms.append(RenameTransform('seg', 'target', True))

    if regions is not None:
        val_transforms.append(ConvertSegmentationToRegionsTransform(regions, 'target', 'target'))

    if directional_field:
        val_transforms.append(DistanceMap(input_key='target', output_key='directional_field'))

    if deep_supervision_scales is not None:
        if directional_field:
            scales = copy(deep_supervision_scales)
            scales.insert(1, [1.0, 1.0])
        if soft_ds:
            assert classes is not None
            val_transforms.append(DownsampleSegForDSTransform3(scales, 'target', 'target', classes))
            val_transforms.append(DownsampleX(deep_supervision_scales, input_key='data', output_key='data', order=3))
        else:
            val_transforms.append(DownsampleSegForDSTransform2(scales, 0, input_key='target', output_key='target'))
            val_transforms.append(DownsampleX(deep_supervision_scales, input_key='data', output_key='data', order=3))

    val_transforms.append(NumpyToTensor(['data', 'target'], 'float'))
    val_transforms = Compose(val_transforms)

    batchgenerator_val = SingleThreadedAugmenter(dataloader_val, val_transforms)
    return batchgenerator_train, batchgenerator_val, batchgenerator_train_unlabeled, batchgenerator_val_unlabeled

