from .custom_transform_middle import *
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.utility_transforms import RenameTransform, NumpyToTensor
from nnunet.training.data_augmentation.custom_transforms import ConvertSegmentationToRegionsTransform
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from nnunet.training.data_augmentation.default_data_augmentation import default_3D_augmentation_params

def get_no_augmentation_video(dataloader_train, dataloader_val, params=default_3D_augmentation_params,
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

    batchgenerator_train = SingleThreadedAugmenter(dataloader_train, tr_transforms)

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

    batchgenerator_val = SingleThreadedAugmenter(dataloader_val, val_transforms)
    return batchgenerator_train, batchgenerator_val