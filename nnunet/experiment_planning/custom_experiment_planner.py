from copy import deepcopy

from nnunet.experiment_planning.common_utils import get_pool_and_conv_props
from nnunet.experiment_planning.experiment_planner_baseline_2DUNet import ExperimentPlanner2D
from nnunet.network_architecture.generic_UNet import Generic_UNet
from nnunet.paths import *
import numpy as np

from nnunet.training.model_restore import recursive_find_python_class
from nnunet.configuration import default_num_threads
import nnunet

from pathlib import Path
import yaml


def read_config(filename):
    with open(filename) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
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
        self.patch_size = 320 if '026' in self.preprocessed_output_folder else 224

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

        print(f'Advised patch size is: {input_patch_size}')
        print(f'Advised network_num_pool_per_axis is: {network_num_pool_per_axis}')
        print(f'Advised pool_op_kernel_sizes is: {pool_op_kernel_sizes}')
        print(f'Advised conv_kernel_sizes is: {conv_kernel_sizes}')

        plan = {
            'batch_size': self.config['batch_size'],
            'num_pool_per_axis': [len(self.config['in_encoder_dims']), len(self.config['in_encoder_dims'])],
            'patch_size': np.array([self.patch_size, self.patch_size]),
            'median_patient_size_in_voxels': new_median_shape,
            'current_spacing': current_spacing,
            'original_spacing': original_spacing,
            'pool_op_kernel_sizes': [[2, 2], [2, 2], [2, 2]],
            'conv_kernel_sizes': conv_kernel_sizes,
            'do_dummy_2D_data_aug': False
        }
        return plan



class CustomExperimentPlannerUnlabeled(ExperimentPlanner2D):
    def __init__(self, folder_with_cropped_data, preprocessed_output_folder):
        super(CustomExperimentPlannerUnlabeled, self).__init__(folder_with_cropped_data, preprocessed_output_folder)
        self.data_identifier = "custom_experiment_planner_unlabeled"
        self.plans_fname = join(self.preprocessed_output_folder,
                                "custom_experiment_planner_unlabeled_plans_2D.pkl")
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
    
    def run_preprocessing(self, num_threads):
        normalization_schemes = self.plans['normalization_schemes']
        use_nonzero_mask_for_normalization = self.plans['use_mask_for_norm']
        intensityproperties = self.plans['dataset_properties']['intensityproperties']
        preprocessor_class = recursive_find_python_class([join(nnunet.__path__[0], "preprocessing")],
                                                         self.preprocessor_name, current_module="nnunet.preprocessing")
        assert preprocessor_class is not None
        preprocessor = preprocessor_class(normalization_schemes, use_nonzero_mask_for_normalization,
                                         self.transpose_forward,
                                          intensityproperties)
        target_spacings = [i["current_spacing"] for i in self.plans_per_stage.values()]
        if self.plans['num_stages'] > 1 and not isinstance(num_threads, (list, tuple)):
            num_threads = (default_num_threads, num_threads)
        elif self.plans['num_stages'] == 1 and isinstance(num_threads, (list, tuple)):
            num_threads = num_threads[-1]
        preprocessor.run_unlabeled(target_spacings, self.folder_with_cropped_data, self.preprocessed_output_folder,
                         self.plans['data_identifier'], num_threads)