from copy import deepcopy

from nnunet.experiment_planning.common_utils import get_pool_and_conv_props
from nnunet.experiment_planning.experiment_planner_baseline_2DUNet import ExperimentPlanner2D
from nnunet.network_architecture.generic_UNet import Generic_UNet
from nnunet.paths import *
import numpy as np

from pathlib import Path
import yaml


def read_config(filename):
    with open(filename) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    if config['nb_nets'] > 1:
        assert config['small'] == False and config['middle'] == False and config['big'] == False
    if config['bottleneck'] == 'swin' or config['bottleneck'] == 'vit':
        assert config['nb_frames'] == 1, "bottleneck mode 'swin' and 'vit' require nb_frames to be 1"
        assert len(config['patch_size']) == 2, "bottleneck mode 'swin' and 'vit' require len(patch_size) to be 2"
    elif config['bottleneck'] == 'swin_3d' or config['bottleneck'] == 'vit_3d' or config['bottleneck'] == 'factorized':
        assert config['nb_frames'] > 1, "bottleneck mode 'swin_3d', 'vit_3d' and 'factorized' require nb_frames to be more than 1"
    if config['bottleneck'] == 'factorized':
        assert len(config['patch_size']) == 2, "bottleneck mode 'factorized' require len(patch_size) to be 2"
    if filename == 'lib_config.yaml':
        assert config['semi_supervised'] == False, "can not run in a semi supervised manner with the lib dataset"
    if config['semi_supervised'] == True:
        assert config['use_spatial_transformer'] == False, "Semi supervised model can not be used with spatial transformer"
    assert len(config['transformer_depth']) == len(config['num_heads']), "transformer_depth and num_heads must have the same size"
    if not config['reconstruction']:
        assert config['reconstruction_skip'] is False, "Cannot use skip connection between decoder and reconstruction decoder if 'reconstruction' is False"
    if config['uncertainty_weighting'] or config['dynamic_weight_averaging']:
        assert config['reconstruction'], "Need reconstruction for uncertainty weighting or dynamic weight averaging"
    if config['similarity']:
        assert config['reconstruction'], "Similarity can not be true without reconstruction"
    if config['adversarial_loss']:
        assert config['reconstruction'], "Reconstruction needs to be activated to use adversarial losses"
    return config


class CustomExperimentPlanner(ExperimentPlanner2D):
    def __init__(self, folder_with_cropped_data, preprocessed_output_folder):
        super(CustomExperimentPlanner, self).__init__(folder_with_cropped_data, preprocessed_output_folder)
        self.data_identifier = "custom_experiment_planner"
        self.plans_fname = join(self.preprocessed_output_folder,
                                "custom_experiment_planner_plans_2D.pkl")
        self.unet_base_num_features = 32
        #self.preprocessor_name = "CustomPreprocessorFor2D"
        self.config = read_config(os.path.join(Path.cwd(), 'adversarial_acdc.yaml'))

    def get_properties_for_stage(self, current_spacing, original_spacing, original_shape, num_cases,
                                 num_modalities, num_classes):

        new_median_shape = np.round(original_spacing / current_spacing * original_shape).astype(int)

        dataset_num_voxels = np.prod(new_median_shape, dtype=np.int64) * num_cases
        input_patch_size = new_median_shape[1:]

        network_num_pool_per_axis, pool_op_kernel_sizes, conv_kernel_sizes, new_shp, \
        shape_must_be_divisible_by = get_pool_and_conv_props(current_spacing[1:], input_patch_size,
                                                             self.unet_featuremap_min_edge_length,
                                                             self.unet_max_numpool)

        # we pretend to use 30 feature maps. This will yield the same configuration as in V1. The larger memory
        # footpring of 32 vs 30 is mor ethan offset by the fp16 training. We make fp16 training default
        # Reason for 32 vs 30 feature maps is that 32 is faster in fp16 training (because multiple of 8)
        ref = Generic_UNet.use_this_for_batch_size_computation_2D * Generic_UNet.DEFAULT_BATCH_SIZE_2D / 2  # for batch size 2
        here = Generic_UNet.compute_approx_vram_consumption(new_shp,
                                                            network_num_pool_per_axis,
                                                            30,
                                                            self.unet_max_num_filters,
                                                            num_modalities, num_classes,
                                                            pool_op_kernel_sizes,
                                                            conv_per_stage=self.conv_per_stage)
        while here > ref:
            axis_to_be_reduced = np.argsort(new_shp / new_median_shape[1:])[-1]

            tmp = deepcopy(new_shp)
            tmp[axis_to_be_reduced] -= shape_must_be_divisible_by[axis_to_be_reduced]
            _, _, _, _, shape_must_be_divisible_by_new = \
                get_pool_and_conv_props(current_spacing[1:], tmp, self.unet_featuremap_min_edge_length,
                                        self.unet_max_numpool)
            new_shp[axis_to_be_reduced] -= shape_must_be_divisible_by_new[axis_to_be_reduced]

            # we have to recompute numpool now:
            network_num_pool_per_axis, pool_op_kernel_sizes, conv_kernel_sizes, new_shp, \
            shape_must_be_divisible_by = get_pool_and_conv_props(current_spacing[1:], new_shp,
                                                                 self.unet_featuremap_min_edge_length,
                                                                 self.unet_max_numpool)

            here = Generic_UNet.compute_approx_vram_consumption(new_shp, network_num_pool_per_axis,
                                                                self.unet_base_num_features,
                                                                self.unet_max_num_filters, num_modalities,
                                                                num_classes, pool_op_kernel_sizes,
                                                                conv_per_stage=self.conv_per_stage)
            # print(new_shp)

        batch_size = int(np.floor(ref / here) * 2)
        input_patch_size = new_shp

        if batch_size < self.unet_min_batch_size:
            raise RuntimeError("This should not happen")

        # check if batch size is too large (more than 5 % of dataset)
        max_batch_size = np.round(self.batch_size_covers_max_percent_of_dataset * dataset_num_voxels /
                                  np.prod(input_patch_size, dtype=np.int64)).astype(int)
        batch_size = max(1, min(batch_size, max_batch_size))

        plan = {
            'batch_size': self.config['batch_size'],
            'num_pool_per_axis': [len(self.config['in_encoder_dims']), len(self.config['in_encoder_dims'])],
            'patch_size': np.array([self.config['image_size'], self.config['image_size']]),
            'median_patient_size_in_voxels': new_median_shape,
            'current_spacing': current_spacing,
            'original_spacing': original_spacing,
            'pool_op_kernel_sizes': [[2, 2], [2, 2], [2, 2]],
            'conv_kernel_sizes': conv_kernel_sizes,
            'do_dummy_2D_data_aug': False
        }
        return plan