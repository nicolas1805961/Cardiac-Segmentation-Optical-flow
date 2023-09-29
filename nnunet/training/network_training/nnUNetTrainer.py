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

import uuid
import random
import string
from tqdm import tqdm
import psutil
import shutil
from collections import OrderedDict
from multiprocessing import Pool
from time import sleep
from typing import Tuple, List
import matplotlib.pyplot as plt
import SimpleITK as sitk
from numpy.lib.stride_tricks import sliding_window_view
from math import ceil
from glob import glob
import nibabel as nib



import matplotlib
import numpy as np
import torch
from batchgenerators.utilities.file_and_folder_operations import *
from torch import nn
from torch.optim import lr_scheduler

import cv2 as cv
import nnunet
from nnunet.configuration import default_num_threads
from nnunet.evaluation.evaluator import NiftiEvaluator, aggregate_scores
from nnunet.inference.segmentation_export import save_segmentation_nifti_from_softmax
from nnunet.network_architecture.generic_UNet import Generic_UNet
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.network_architecture.neural_network import SegmentationNetwork
from nnunet.postprocessing.connected_components import determine_postprocessing, determine_postprocessing_no_metric
from nnunet.training.data_augmentation.default_data_augmentation import default_3D_augmentation_params, \
    default_2D_augmentation_params, get_default_augmentation, get_patch_size
from nnunet.training.dataloading.dataset_loading import load_dataset, DataLoader3D, DataLoader2D, unpack_dataset
from nnunet.training.loss_functions.dice_loss import DC_and_CE_loss
from nnunet.training.network_training.network_trainer import NetworkTrainer
from nnunet.utilities.nd_softmax import softmax_helper
from nnunet.utilities.tensor_utilities import sum_tensor

matplotlib.use("agg")


def strain_compute_metric(patient_name, all_files, gt_folder_name):
    results = {'all': []}
    phase_list = [x for x in all_files if patient_name in x]
    phase_list = np.array(sorted(phase_list))
    video = []
    for phase in phase_list:
        arr = np.load(phase) # T
        video.append(arr)
    video = np.stack(video, axis=0) # D, T
    for d in range(len(video)):
        slice_nb = str(d + 1).zfill(2)
        filename = patient_name + '_slice' + slice_nb + '.npy'
        gt_path = os.path.join(gt_folder_name, 'strain', 'LV', 'tangential', filename)
        lv_tangential_strain = video[d]
        lv_tangential_strain_gt = np.load(gt_path)[0]
        current_res = {'reference': gt_path, 'test': phase_list[0][:-12]}
        current_res['lv_tangential'] = (lv_tangential_strain - lv_tangential_strain_gt).tolist()
        results['all'].append(current_res)
    return results



def save_strain_bottleneck(save_path, patient_id, lv_tangential_strain):
    lv_tangential_strain = lv_tangential_strain.transpose((1, 0)) # D, T
    for d in range(len(lv_tangential_strain)):
        current_strain = lv_tangential_strain[d] * 100
        slice_nb = str(d + 1).zfill(2)
        filename = patient_id + '_slice' + slice_nb + '.npy'
        np.save(os.path.join(save_path, filename), current_strain)


def move_gt_files(input_file_path, output_dir):
    success = False
    attempts = 0
    e = None
    while not success and attempts < 10:
        try:
            shutil.copy(input_file_path, output_dir)
            success = True
        except OSError as e:
            attempts += 1
            sleep(1)
    if not success:
        print("Could not copy gt nifti file %s into folder %s" % (input_file_path, output_dir))
        if e is not None:
            raise e


def save_strain_compute_metric(patient_name, all_files, gt_folder_name, search_path):
    results = {'all': []}
    phase_list = [x for x in all_files if patient_name in x]
    phase_list = np.array(sorted(phase_list, key=lambda x: int(os.path.basename(x).split('frame')[-1][:2])))
    video = []
    for phase in phase_list:
        data = nib.load(phase)
        arr = data.get_fdata() # H, W, D
        arr = arr.transpose((2, 0, 1)) # D, H, W
        video.append(arr)
    video = np.stack(video, axis=1) # D, T, H, W
    for d in range(len(video)):
        slice_nb = str(d + 1).zfill(2)
        filename = patient_name + '_slice' + slice_nb + '.npy'
        gt_path = os.path.join(gt_folder_name, 'strain', 'LV', 'tangential', filename)
        rv_tangential_strain, lv_tangential_strain = get_strain(video[d])
        rv_tangential_strain = rv_tangential_strain * 100
        lv_tangential_strain = lv_tangential_strain * 100
        lv_tangential_strain_gt = np.load(gt_path)[0]
        current_res = {'reference': gt_path, 'test': phase_list[0][:-15]}
        current_res['lv_tangential'] = (lv_tangential_strain - lv_tangential_strain_gt).tolist()
        results['all'].append(current_res)
        np.save(os.path.join(search_path, 'Strain', 'LV', 'Tangential', filename), lv_tangential_strain)
    return results


def compute_contour_metric(patient_name, all_files, gt_folder_name):
    results = {'all': []}
    phase_list = [x for x in all_files if patient_name in x]
    phase_list = np.array(sorted(phase_list, key=lambda x: int(os.path.basename(x).split('frame')[-1][:2])))
    video = []
    for phase in phase_list:
        data = np.load(phase)
        arr = data['flow'] # H, W, D, C
        arr = arr.transpose((2, 3, 0, 1)) # D, C, H, W
        video.append(arr)
    flow = np.stack(video, axis=1) # D, T, C, H, W
    slice_error_list = []
    for d in range(len(flow)):
        current_slice_flow = flow[d] # T, 2, H, W
        slice_nb = str(d + 1).zfill(2)
        filename = patient_name + '_slice' + slice_nb + '.npy'
        gt_path_lv = os.path.join(gt_folder_name, 'contour', 'LV', filename)
        gt_path_rv = os.path.join(gt_folder_name, 'contour', 'RV', filename)
        gt_lv_contour = np.load(gt_path_lv).transpose((2, 1, 0)) # T, P1, 4
        gt_rv_contour = np.load(gt_path_rv).transpose((2, 1, 0)) # T, P2, 2
        gt_endo_contour = gt_lv_contour[:, :, :2]
        gt_epi_contour = gt_lv_contour[:, :, 2:]
        split_index = np.cumsum([gt_endo_contour.shape[1], gt_epi_contour.shape[1]])
        contours = np.concatenate([gt_endo_contour, gt_epi_contour, gt_rv_contour], axis=1) # T, P, 2

        current_slice_flow = np.flip(current_slice_flow, axis=0)
        contours = np.flip(contours, axis=0)

        temporal_error_list = []
        for t in range(len(current_slice_flow) - 1):
            current_contours = contours[t] # P, 2
            next_contours = contours[t + 1] # P, 2
            gt_delta = next_contours - current_contours
            current_frame_flow = current_slice_flow[t] # 2, H, W
            current_frame_flow = current_frame_flow.transpose((1, 2, 0)) # H, W, 2
            y = np.rint(current_contours[:, 0]).astype(int)
            x = np.rint(current_contours[:, 1]).astype(int)

            delta_pred = current_frame_flow[y, x, :] # P, 2
            error = np.abs(gt_delta - delta_pred).mean(-1) # P,
            temporal_error_list.append(error) 

        temporal_error_list = np.stack(temporal_error_list, axis=0) # T, P
        temporal_error_list = np.flip(temporal_error_list, axis=0)

        error = np.split(temporal_error_list, indices_or_sections=split_index, axis=1) # [(T, P1) , (T, P2), (T, P3)]
        error = np.stack([x.mean(-1) for x in error], axis=-1) # T, 3

        slice_error_list.append(error)
    slice_error_list = np.stack(slice_error_list, axis=0) # D, T, 3

    for i in range(slice_error_list.shape[1]):
        current_res = slice_error_list[:, i] # D, 3
        current_res_info = {'patient_name': patient_name, 
                            'reference': gt_path_lv[:-12], 
                            'test': phase_list[i + 1],
                            'ENDO_mae': current_res[:, 0].tolist(),
                            'EPI_mae': current_res[:, 1].tolist(),
                            'RV_mae': current_res[:, 2].tolist()}
        results['all'].append(current_res_info)
    return results



class nnUNetTrainer(NetworkTrainer):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        """
        :param deterministic:
        :param fold: can be either [0 ... 5) for cross-validation, 'all' to train on all available training data or
        None if you wish to load some checkpoint and do inference only
        :param plans_file: the pkl file generated by preprocessing. This file will determine all design choices
        :param subfolder_with_preprocessed_data: must be a subfolder of dataset_directory (just the name of the folder,
        not the entire path). This is where the preprocessed data lies that will be used for network training. We made
        this explicitly available so that differently preprocessed data can coexist and the user can choose what to use.
        Can be None if you are doing inference only.
        :param output_folder: where to store parameters, plot progress and to the validation
        :param dataset_directory: the parent directory in which the preprocessed Task data is stored. This is required
        because the split information is stored in this directory. For running prediction only this input is not
        required and may be set to None
        :param batch_dice: compute dice loss for each sample and average over all samples in the batch or pretend the
        batch is a pseudo volume?
        :param stage: The plans file may contain several stages (used for lowres / highres / pyramid). Stage must be
        specified for training:
        if stage 1 exists then stage 1 is the high resolution stage, otherwise it's 0
        :param unpack_data: if False, npz preprocessed data will not be unpacked to npy. This consumes less space but
        is considerably slower! Running unpack_data=False with 2d should never be done!

        IMPORTANT: If you inherit from nnUNetTrainer and the init args change then you need to redefine self.init_args
        in your init accordingly. Otherwise checkpoints won't load properly!
        """
        super(nnUNetTrainer, self).__init__(deterministic, fp16)
        self.unpack_data = unpack_data
        self.init_args = (plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                          deterministic, fp16)
        # set through arguments from init
        self.stage = stage
        self.experiment_name = self.__class__.__name__
        self.plans_file = plans_file
        self.output_folder = output_folder
        self.dataset_directory = dataset_directory
        self.output_folder_base = self.output_folder
        self.fold = fold

        self.plans = None

        # if we are running inference only then the self.dataset_directory is set (due to checkpoint loading) but it
        # irrelevant
        if self.dataset_directory is not None and isdir(self.dataset_directory):
            self.gt_niftis_folder = join(self.dataset_directory, "gt_segmentations")
        else:
            self.gt_niftis_folder = None

        self.folder_with_preprocessed_data = None

        # set in self.initialize()

        self.dl_tr = self.dl_val = None
        self.num_input_channels = self.num_classes = self.net_pool_per_axis = self.patch_size = self.batch_size = \
            self.threeD = self.base_num_features = self.intensity_properties = self.normalization_schemes = \
            self.net_num_pool_op_kernel_sizes = self.net_conv_kernel_sizes = None  # loaded automatically from plans_file
        self.basic_generator_patch_size = self.data_aug_params = self.transpose_forward = self.transpose_backward = None

        self.batch_dice = batch_dice
        self.loss = DC_and_CE_loss({'batch_dice': self.batch_dice, 'smooth': 1e-5, 'do_bg': False}, {})

        self.online_eval_foreground_dc = []
        self.online_eval_tp = []
        self.online_eval_fp = []
        self.online_eval_fn = []

        self.classes = self.do_dummy_2D_aug = self.use_mask_for_norm = self.only_keep_largest_connected_component = \
            self.min_region_size_per_class = self.min_size_per_class = None

        self.inference_pad_border_mode = "constant"
        self.inference_pad_kwargs = {'constant_values': 0}

        self.update_fold(fold)
        self.pad_all_sides = None

        self.lr_scheduler_eps = 1e-3
        self.lr_scheduler_patience = 30
        self.initial_lr = 3e-4
        self.weight_decay = 3e-5

        self.oversample_foreground_percent = 0.33

        self.conv_per_stage = None
        self.regions_class_order = None

    def update_fold(self, fold):
        """
        used to swap between folds for inference (ensemble of models from cross-validation)
        DO NOT USE DURING TRAINING AS THIS WILL NOT UPDATE THE DATASET SPLIT AND THE DATA AUGMENTATION GENERATORS
        :param fold:
        :return:
        """
        if fold is not None:
            if isinstance(fold, str):
                assert fold == "all", "if self.fold is a string then it must be \'all\'"
                if self.output_folder.endswith("%s" % str(self.fold)):
                    self.output_folder = self.output_folder_base
                self.output_folder = join(self.output_folder, "%s" % str(fold))
            else:
                if self.output_folder.endswith("fold_%s" % str(self.fold)):
                    self.output_folder = self.output_folder_base
                self.output_folder = join(self.output_folder, "fold_%s" % str(fold))
            self.fold = fold

    def setup_DA_params(self):
        if self.threeD:
            self.data_aug_params = default_3D_augmentation_params
            if self.do_dummy_2D_aug:
                self.data_aug_params["dummy_2D"] = True
                self.print_to_log_file("Using dummy2d data augmentation")
                self.data_aug_params["elastic_deform_alpha"] = \
                    default_2D_augmentation_params["elastic_deform_alpha"]
                self.data_aug_params["elastic_deform_sigma"] = \
                    default_2D_augmentation_params["elastic_deform_sigma"]
                self.data_aug_params["rotation_x"] = default_2D_augmentation_params["rotation_x"]
        else:
            self.do_dummy_2D_aug = False
            if max(self.patch_size) / min(self.patch_size) > 1.5:
                default_2D_augmentation_params['rotation_x'] = (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi)
            self.data_aug_params = default_2D_augmentation_params
        self.data_aug_params["mask_was_used_for_normalization"] = self.use_mask_for_norm

        if self.do_dummy_2D_aug:
            self.basic_generator_patch_size = get_patch_size(self.patch_size[1:],
                                                             self.data_aug_params['rotation_x'],
                                                             self.data_aug_params['rotation_y'],
                                                             self.data_aug_params['rotation_z'],
                                                             self.data_aug_params['scale_range'])
            self.basic_generator_patch_size = np.array([self.patch_size[0]] + list(self.basic_generator_patch_size))
        else:
            self.basic_generator_patch_size = get_patch_size(self.patch_size, self.data_aug_params['rotation_x'],
                                                             self.data_aug_params['rotation_y'],
                                                             self.data_aug_params['rotation_z'],
                                                             self.data_aug_params['scale_range'])

        self.data_aug_params['selected_seg_channels'] = [0]
        self.data_aug_params['patch_size_for_spatialtransform'] = self.patch_size

    def initialize(self, training=True, force_load_plans=False):
        """
        For prediction of test cases just set training=False, this will prevent loading of training data and
        training batchgenerator initialization
        :param training:
        :return:
        """

        maybe_mkdir_p(self.output_folder)

        if force_load_plans or (self.plans is None):
            self.load_plans_file()

        self.process_plans(self.plans)

        self.setup_DA_params()

        if training:
            self.folder_with_preprocessed_data = join(self.dataset_directory, self.plans['data_identifier'] +
                                                      "_stage%d" % self.stage)

            self.dl_tr, self.dl_val = self.get_basic_generators()
            if self.unpack_data:
                self.print_to_log_file("unpacking dataset")
                unpack_dataset(self.folder_with_preprocessed_data)
                self.print_to_log_file("done")
            else:
                self.print_to_log_file(
                    "INFO: Not unpacking data! Training may be slow due to that. Pray you are not using 2d or you "
                    "will wait all winter for your model to finish!")
            self.tr_gen, self.val_gen = get_default_augmentation(self.dl_tr, self.dl_val,
                                                                 self.data_aug_params[
                                                                     'patch_size_for_spatialtransform'],
                                                                 self.data_aug_params)
            self.print_to_log_file("TRAINING KEYS:\n %s" % (str(self.dataset_tr.keys())),
                                   also_print_to_console=False)
            self.print_to_log_file("VALIDATION KEYS:\n %s" % (str(self.dataset_val.keys())),
                                   also_print_to_console=False)
        else:
            pass
        self.initialize_network()
        self.initialize_optimizer_and_scheduler()
        # assert isinstance(self.network, (SegmentationNetwork, nn.DataParallel))
        self.was_initialized = True

    def initialize_network(self):
        """
        This is specific to the U-Net and must be adapted for other network architectures
        :return:
        """
        # self.print_to_log_file(self.net_num_pool_op_kernel_sizes)
        # self.print_to_log_file(self.net_conv_kernel_sizes)

        net_numpool = len(self.net_num_pool_op_kernel_sizes)

        if self.threeD:
            conv_op = nn.Conv3d
            dropout_op = nn.Dropout3d
            norm_op = nn.InstanceNorm3d
        else:
            conv_op = nn.Conv2d
            dropout_op = nn.Dropout2d
            norm_op = nn.InstanceNorm2d

        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        net_nonlin = nn.LeakyReLU
        net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        self.network = Generic_UNet(self.num_input_channels, self.base_num_features, self.num_classes, net_numpool,
                                    self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                    dropout_op_kwargs,
                                    net_nonlin, net_nonlin_kwargs, False, False, lambda x: x, InitWeights_He(1e-2),
                                    self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True)
        self.network.inference_apply_nonlin = softmax_helper

        if torch.cuda.is_available():
            self.network.cuda()

    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, "self.initialize_network must be called first"
        self.optimizer = torch.optim.Adam(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                          amsgrad=True)
        self.lr_scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.2,
                                                           patience=self.lr_scheduler_patience,
                                                           verbose=True, threshold=self.lr_scheduler_eps,
                                                           threshold_mode="abs")

    def plot_network_architecture(self):
        try:
            from batchgenerators.utilities.file_and_folder_operations import join
            import hiddenlayer as hl
            if torch.cuda.is_available():
                g = hl.build_graph(self.network, torch.rand((1, self.num_input_channels, *self.patch_size)).cuda(),
                                   transforms=None)
            else:
                g = hl.build_graph(self.network, torch.rand((1, self.num_input_channels, *self.patch_size)),
                                   transforms=None)
            g.save(join(self.output_folder, "network_architecture.pdf"))
            del g
        except Exception as e:
            self.print_to_log_file("Unable to plot network architecture:")
            self.print_to_log_file(e)

            self.print_to_log_file("\nprinting the network instead:\n")
            self.print_to_log_file(self.network)
            self.print_to_log_file("\n")
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def save_debug_information(self):
        # saving some debug information
        dct = OrderedDict()
        for k in self.__dir__():
            if not k.startswith("__"):
                if not callable(getattr(self, k)):
                    dct[k] = str(getattr(self, k))
        del dct['plans']
        del dct['intensity_properties']
        del dct['dataset']
        del dct['dataset_tr']
        del dct['dataset_val']
        save_json(dct, join(self.output_folder, "debug.json"))

        import shutil

        shutil.copy(self.plans_file, join(self.output_folder_base, "plans.pkl"))

    def run_training(self):
        self.save_debug_information()
        super(nnUNetTrainer, self).run_training()

    def load_plans_file(self):
        """
        This is what actually configures the entire experiment. The plans file is generated by experiment planning
        :return:
        """
        self.plans = load_pickle(self.plans_file)

    def process_plans(self, plans):
        if self.stage is None:
            assert len(list(plans['plans_per_stage'].keys())) == 1, \
                "If self.stage is None then there can be only one stage in the plans file. That seems to not be the " \
                "case. Please specify which stage of the cascade must be trained"
            self.stage = list(plans['plans_per_stage'].keys())[0]
        self.plans = plans

        stage_plans = self.plans['plans_per_stage'][self.stage]
        self.batch_size = stage_plans['batch_size']
        self.net_pool_per_axis = stage_plans['num_pool_per_axis']
        self.patch_size = np.array(stage_plans['patch_size']).astype(int)
        self.do_dummy_2D_aug = stage_plans['do_dummy_2D_data_aug']

        if 'pool_op_kernel_sizes' not in stage_plans.keys():
            assert 'num_pool_per_axis' in stage_plans.keys()
            self.print_to_log_file("WARNING! old plans file with missing pool_op_kernel_sizes. Attempting to fix it...")
            self.net_num_pool_op_kernel_sizes = []
            for i in range(max(self.net_pool_per_axis)):
                curr = []
                for j in self.net_pool_per_axis:
                    if (max(self.net_pool_per_axis) - j) <= i:
                        curr.append(2)
                    else:
                        curr.append(1)
                self.net_num_pool_op_kernel_sizes.append(curr)
        else:
            self.net_num_pool_op_kernel_sizes = stage_plans['pool_op_kernel_sizes']

        if 'conv_kernel_sizes' not in stage_plans.keys():
            self.print_to_log_file("WARNING! old plans file with missing conv_kernel_sizes. Attempting to fix it...")
            self.net_conv_kernel_sizes = [[3] * len(self.net_pool_per_axis)] * (max(self.net_pool_per_axis) + 1)
        else:
            self.net_conv_kernel_sizes = stage_plans['conv_kernel_sizes']

        self.pad_all_sides = None  # self.patch_size
        self.intensity_properties = plans['dataset_properties']['intensityproperties']
        self.normalization_schemes = plans['normalization_schemes']
        self.base_num_features = plans['base_num_features']
        self.num_input_channels = plans['num_modalities']
        if self.binary:
            self.num_classes = 2  # background is no longer in num_classes
        else:
            self.num_classes = plans['num_classes'] + 1  # background is no longer in num_classes
        self.classes = plans['all_classes']
        self.use_mask_for_norm = plans['use_mask_for_norm']
        self.only_keep_largest_connected_component = plans['keep_only_largest_region']
        self.min_region_size_per_class = plans['min_region_size_per_class']
        self.min_size_per_class = None  # DONT USE THIS. plans['min_size_per_class']

        if plans.get('transpose_forward') is None or plans.get('transpose_backward') is None:
            print("WARNING! You seem to have data that was preprocessed with a previous version of nnU-Net. "
                  "You should rerun preprocessing. We will proceed and assume that both transpose_foward "
                  "and transpose_backward are [0, 1, 2]. If that is not correct then weird things will happen!")
            plans['transpose_forward'] = [0, 1, 2]
            plans['transpose_backward'] = [0, 1, 2]
        self.transpose_forward = plans['transpose_forward']
        self.transpose_backward = plans['transpose_backward']

        if len(self.patch_size) == 2:
            self.threeD = False
        elif len(self.patch_size) == 3:
            self.threeD = True
        else:
            raise RuntimeError("invalid patch size in plans file: %s" % str(self.patch_size))

        if "conv_per_stage" in plans.keys():  # this ha sbeen added to the plans only recently
            self.conv_per_stage = plans['conv_per_stage']
        else:
            self.conv_per_stage = 2

    def load_dataset(self):
        self.dataset = load_dataset(self.folder_with_preprocessed_data)

    def get_basic_generators(self):
        self.load_dataset()
        self.do_split()

        if self.threeD:
            dl_tr = DataLoader3D(self.dataset_tr, self.basic_generator_patch_size, self.patch_size, self.batch_size,
                                 False, oversample_foreground_percent=self.oversample_foreground_percent,
                                 pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
            dl_val = DataLoader3D(self.dataset_val, self.patch_size, self.patch_size, self.batch_size, False,
                                  oversample_foreground_percent=self.oversample_foreground_percent,
                                  pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
        else:
            dl_tr = DataLoader2D(self.dataset_tr, self.basic_generator_patch_size, self.patch_size, self.batch_size,
                                 oversample_foreground_percent=self.oversample_foreground_percent,
                                 pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
            dl_val = DataLoader2D(self.dataset_val, self.patch_size, self.patch_size, self.batch_size,
                                  oversample_foreground_percent=self.oversample_foreground_percent,
                                  pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
        return dl_tr, dl_val

    def preprocess_patient(self, input_files):
        """
        Used to predict new unseen data. Not used for the preprocessing of the training/test data
        :param input_files:
        :return:
        """
        from nnunet.training.model_restore import recursive_find_python_class
        preprocessor_name = self.plans.get('preprocessor_name')
        if preprocessor_name is None:
            if self.threeD:
                preprocessor_name = "GenericPreprocessor"
            else:
                preprocessor_name = "PreprocessorFor2D"

        print("using preprocessor", preprocessor_name)
        preprocessor_class = recursive_find_python_class([join(nnunet.__path__[0], "preprocessing")],
                                                         preprocessor_name,
                                                         current_module="nnunet.preprocessing")
        assert preprocessor_class is not None, "Could not find preprocessor %s in nnunet.preprocessing" % \
                                               preprocessor_name
        preprocessor = preprocessor_class(self.normalization_schemes, self.use_mask_for_norm,
                                          self.transpose_forward, self.intensity_properties)

        d, s, properties = preprocessor.preprocess_test_case(input_files,
                                                             self.plans['plans_per_stage'][self.stage][
                                                                 'current_spacing'])
        return d, s, properties

    def preprocess_predict_nifti(self, input_files: List[str], output_file: str = None,
                                 softmax_ouput_file: str = None, mixed_precision: bool = True) -> None:
        """
        Use this to predict new data
        :param input_files:
        :param output_file:
        :param softmax_ouput_file:
        :param mixed_precision:
        :return:
        """
        print("preprocessing...")
        d, s, properties = self.preprocess_patient(input_files)
        print("predicting...")
        pred = self.predict_preprocessed_data_return_seg_and_softmax(d, do_mirroring=self.data_aug_params["do_mirror"],
                                                                     mirror_axes=self.data_aug_params['mirror_axes'],
                                                                     use_sliding_window=True, step_size=0.5,
                                                                     use_gaussian=True, pad_border_mode='constant',
                                                                     pad_kwargs={'constant_values': 0},
                                                                     verbose=True, all_in_gpu=False,
                                                                     mixed_precision=mixed_precision)[1]
        pred = pred.transpose([0] + [i + 1 for i in self.transpose_backward])

        if 'segmentation_export_params' in self.plans.keys():
            force_separate_z = self.plans['segmentation_export_params']['force_separate_z']
            interpolation_order = self.plans['segmentation_export_params']['interpolation_order']
            interpolation_order_z = self.plans['segmentation_export_params']['interpolation_order_z']
        else:
            force_separate_z = None
            interpolation_order = 1
            interpolation_order_z = 0

        print("resampling to original spacing and nifti export...")
        save_segmentation_nifti_from_softmax(pred, output_file, properties, interpolation_order,
                                             self.regions_class_order, None, None, softmax_ouput_file,
                                             None, force_separate_z=force_separate_z,
                                             interpolation_order_z=interpolation_order_z)
        print("done")

    def predict_preprocessed_data_return_seg_and_softmax(self, data: np.ndarray, do_mirroring: bool = True,
                                                         mirror_axes: Tuple[int] = None,
                                                         use_sliding_window: bool = True, step_size: float = 0.5,
                                                         use_gaussian: bool = True, pad_border_mode: str = 'constant',
                                                         pad_kwargs: dict = None, all_in_gpu: bool = False,
                                                         verbose: bool = True, mixed_precision: bool = True,
                                                         get_flops=False, binary=False) -> Tuple[np.ndarray, np.ndarray]:
        """
        :param data:
        :param do_mirroring:
        :param mirror_axes:
        :param use_sliding_window:
        :param step_size:
        :param use_gaussian:
        :param pad_border_mode:
        :param pad_kwargs:
        :param all_in_gpu:
        :param verbose:
        :return:
        """
        if pad_border_mode == 'constant' and pad_kwargs is None:
            pad_kwargs = {'constant_values': 0}

        if do_mirroring and mirror_axes is None:
            mirror_axes = self.data_aug_params['mirror_axes']

        if do_mirroring:
            assert self.data_aug_params["do_mirror"], "Cannot do mirroring as test time augmentation when training " \
                                                      "was done without mirroring"

        valid = list((SegmentationNetwork, nn.DataParallel))
        assert isinstance(self.network, tuple(valid))

        current_mode = self.network.training
        self.network.eval()
        ret = self.network.predict_3D(data, do_mirroring=do_mirroring, mirror_axes=mirror_axes,
                                      use_sliding_window=use_sliding_window, step_size=step_size,
                                      patch_size=self.patch_size, regions_class_order=self.regions_class_order,
                                      use_gaussian=use_gaussian, pad_border_mode=pad_border_mode,
                                      pad_kwargs=pad_kwargs, all_in_gpu=all_in_gpu, verbose=verbose,
                                      mixed_precision=mixed_precision, get_flops=get_flops, binary=binary)
        self.network.train(current_mode)
        return ret


    def predict_preprocessed_data_return_seg_and_softmax_flow(self, unlabeled, target, target_mask, processor, do_mirroring: bool = True,
                                                         mirror_axes: Tuple[int] = None,
                                                         use_sliding_window: bool = True, step_size: float = 0.5,
                                                         use_gaussian: bool = True, pad_border_mode: str = 'constant',
                                                         pad_kwargs: dict = None, all_in_gpu: bool = False,
                                                         verbose: bool = True, mixed_precision: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        :param data:
        :param do_mirroring:
        :param mirror_axes:
        :param use_sliding_window:
        :param step_size:
        :param use_gaussian:
        :param pad_border_mode:
        :param pad_kwargs:
        :param all_in_gpu:
        :param verbose:
        :return:
        """

        # P, T, 1, D, H, W

        if pad_border_mode == 'constant' and pad_kwargs is None:
            pad_kwargs = {'constant_values': 0}

        if do_mirroring and mirror_axes is None:
            mirror_axes = self.data_aug_params['mirror_axes']

        if do_mirroring:
            assert self.data_aug_params["do_mirror"], "Cannot do mirroring as test time augmentation when training " \
                                                      "was done without mirroring"

        valid = list((SegmentationNetwork, nn.DataParallel))
        assert isinstance(self.network, tuple(valid))

        current_mode = self.network.training
        self.network.eval()
        ret = self.network.predict_3D_flow(unlabeled=unlabeled, target=target, target_mask=target_mask, processor=processor, do_mirroring=do_mirroring, mirror_axes=mirror_axes,
                                      use_sliding_window=use_sliding_window, step_size=step_size,
                                      patch_size=self.patch_size, regions_class_order=self.regions_class_order,
                                      use_gaussian=use_gaussian, pad_border_mode=pad_border_mode,
                                      pad_kwargs=pad_kwargs, all_in_gpu=all_in_gpu, verbose=verbose,
                                      mixed_precision=mixed_precision)
        self.network.train(current_mode)
        return ret
    
    def create_metadata_dict(self, properties):
        key_list = ['center', 'manufacturer', 'phase', 'strength', 'slice thickness', 'spacing between slices']
        out = {key: properties[key] for key in key_list if key in properties}
        return out

    def validate(self, log_function, do_mirroring: bool = True, use_sliding_window: bool = True, step_size: float = 0.5,
                 save_softmax: bool = True, use_gaussian: bool = True, overwrite: bool = True,
                 validation_folder_name: str = 'validation_raw', debug: bool = True, all_in_gpu: bool = False,
                 segmentation_export_kwargs: dict = None, run_postprocessing_on_folds: bool = True, output_folder=None, binary=False):
        """
        if debug=True then the temporary files generated for postprocessing determination will be kept
        """

        current_mode = self.network.training
        self.network.eval()

        assert self.was_initialized, "must initialize, ideally with checkpoint (or train first)"
        if self.dataset_val is None:
            self.load_dataset()
            self.do_split()

        if segmentation_export_kwargs is None:
            if 'segmentation_export_params' in self.plans.keys():
                force_separate_z = self.plans['segmentation_export_params']['force_separate_z']
                interpolation_order = self.plans['segmentation_export_params']['interpolation_order']
                interpolation_order_z = self.plans['segmentation_export_params']['interpolation_order_z']
            else:
                force_separate_z = None
                interpolation_order = 1
                interpolation_order_z = 0
        else:
            force_separate_z = segmentation_export_kwargs['force_separate_z']
            interpolation_order = segmentation_export_kwargs['interpolation_order']
            interpolation_order_z = segmentation_export_kwargs['interpolation_order_z']

        # predictions as they come from the network go here
        if output_folder is not None:
            self.output_folder = output_folder
        output_folder = join(self.output_folder, validation_folder_name)
        maybe_mkdir_p(output_folder)
        # this is for debug purposes
        my_input_args = {'do_mirroring': do_mirroring,
                         'use_sliding_window': use_sliding_window,
                         'step_size': step_size,
                         'save_softmax': save_softmax,
                         'use_gaussian': use_gaussian,
                         'overwrite': overwrite,
                         'validation_folder_name': validation_folder_name,
                         'debug': debug,
                         'all_in_gpu': all_in_gpu,
                         'segmentation_export_kwargs': segmentation_export_kwargs,
                         }
        save_json(my_input_args, join(output_folder, "validation_args.json"))

        if binary:
            do_mirroring = True

        if do_mirroring:
            if not self.data_aug_params['do_mirror']:
                raise RuntimeError("We did not train with mirroring so you cannot do inference with mirroring enabled")
            mirror_axes = self.data_aug_params['mirror_axes']
        else:
            mirror_axes = ()

        pred_gt_tuples = []

        export_pool = Pool(default_num_threads)
        results = []

        metadata_list = []
        for idx, k in enumerate(self.dataset_val.keys()):
            properties = load_pickle(self.dataset[k]['properties_file'])
            metadata_list.append(self.create_metadata_dict(properties))
            fname = properties['list_of_data_files'][0].split(os.sep)[-1][:-12]
            if overwrite or (not isfile(join(output_folder, fname + ".nii.gz"))) or \
                    (save_softmax and not isfile(join(output_folder, fname + ".npz"))):
                data = np.load(self.dataset[k]['data_file'])['data']

                #print(k, data.shape)
                self.print_to_log_file(k, data.shape)
                data[-1][data[-1] == -1] = 0

                #get_flops = idx == 0
                get_flops = False

                out = self.predict_preprocessed_data_return_seg_and_softmax(data[:-1],
                                                                            do_mirroring=do_mirroring,
                                                                            mirror_axes=mirror_axes,
                                                                            use_sliding_window=use_sliding_window,
                                                                            step_size=step_size,
                                                                            use_gaussian=use_gaussian,
                                                                            all_in_gpu=all_in_gpu,
                                                                            mixed_precision=self.fp16,
                                                                            get_flops=get_flops,
                                                                            binary=binary)

                softmax_pred = out[1]
                
                if get_flops and idx == 0:
                    flop_dict = out[2]
                    inference_time = out[3] / 1000
                    gflops = 0
                    for flop_key in flop_dict.keys():
                        gflops += flop_dict[flop_key]
                    with open(join(self.output_folder, "computational_time"), 'w') as fd:
                        fd.writelines([str(gflops), '\n', str(inference_time)])

                softmax_pred = softmax_pred.transpose([0] + [i + 1 for i in self.transpose_backward])

                if save_softmax:
                    softmax_fname = join(output_folder, fname + ".npz")
                else:
                    softmax_fname = None

                """There is a problem with python process communication that prevents us from communicating objects
                larger than 2 GB between processes (basically when the length of the pickle string that will be sent is
                communicated by the multiprocessing.Pipe object then the placeholder (I think) does not allow for long
                enough strings (lol). This could be fixed by changing i to l (for long) but that would require manually
                patching system python code. We circumvent that problem here by saving softmax_pred to a npy file that will
                then be read (and finally deleted) by the Process. save_segmentation_nifti_from_softmax can take either
                filename or np.ndarray and will handle this automatically"""
                if np.prod(softmax_pred.shape) > (2e9 / 4 * 0.85):  # *0.85 just to be save
                    np.save(join(output_folder, fname + ".npy"), softmax_pred)
                    softmax_pred = join(output_folder, fname + ".npy")

                results.append(export_pool.starmap_async(save_segmentation_nifti_from_softmax,
                                                         ((softmax_pred, join(output_folder, fname + ".nii.gz"),
                                                           properties, interpolation_order, self.regions_class_order,
                                                           None, None,
                                                           softmax_fname, None, force_separate_z,
                                                           interpolation_order_z),
                                                          )
                                                         )
                               )

            pred_gt_tuples.append([join(output_folder, fname + ".nii.gz"),
                                   join(self.gt_niftis_folder, fname + ".nii.gz")])

        _ = [i.get() for i in results]
        self.print_to_log_file("finished prediction")

        # evaluate raw predictions
        self.print_to_log_file("evaluation of raw predictions")
        task = self.dataset_directory.split(os.sep)[-1]
        job_name = self.experiment_name
        _ = aggregate_scores(pred_gt_tuples, labels=list(range(self.num_classes)),
                             json_output_file=join(output_folder, "summary.json"),
                             json_name=job_name + " val tiled %s" % (str(use_sliding_window)),
                             json_author="Fabian",
                             json_task=task, num_threads=default_num_threads,
                             advanced=True,
                             metadata_list=metadata_list,
                             binary=binary)

        if run_postprocessing_on_folds:
            # in the old nnunet we would stop here. Now we add a postprocessing. This postprocessing can remove everything
            # except the largest connected component for each class. To see if this improves results, we do this for all
            # classes and then rerun the evaluation. Those classes for which this resulted in an improved dice score will
            # have this applied during inference as well
            self.print_to_log_file("determining postprocessing")
            determine_postprocessing(self.output_folder, self.gt_niftis_folder, validation_folder_name,
                                     final_subf_name=validation_folder_name + "_postprocessed", debug=debug, log_function=log_function,
                                     metadata_list=metadata_list, binary=binary)
            # after this the final predictions for the vlaidation set can be found in validation_folder_name_base + "_postprocessed"
            # They are always in that folder, even if no postprocessing as applied!

        # detemining postprocesing on a per-fold basis may be OK for this fold but what if another fold finds another
        # postprocesing to be better? In this case we need to consolidate. At the time the consolidation is going to be
        # done we won't know what self.gt_niftis_folder was, so now we copy all the niftis into a separate folder to
        # be used later
        gt_nifti_folder = join(self.output_folder_base, "gt_niftis")
        maybe_mkdir_p(gt_nifti_folder)
        for f in subfiles(self.gt_niftis_folder, suffix=".nii.gz"):
            success = False
            attempts = 0
            e = None
            while not success and attempts < 10:
                try:
                    shutil.copy(f, gt_nifti_folder)
                    success = True
                except OSError as e:
                    attempts += 1
                    sleep(1)
            if not success:
                print("Could not copy gt nifti file %s into folder %s" % (f, gt_nifti_folder))
                if e is not None:
                    raise e

        self.network.train(current_mode)
    

    def validate_video(self, processor, log_function, step, do_mirroring: bool = True, use_sliding_window: bool = True, step_size: float = 0.5,
                 save_softmax: bool = True, use_gaussian: bool = True, overwrite: bool = True,
                 validation_folder_name: str = 'validation_raw', debug: bool = True, all_in_gpu: bool = False,
                 segmentation_export_kwargs: dict = None, run_postprocessing_on_folds: bool = True, output_folder=None):

        """
        if debug=True then the temporary files generated for postprocessing determination will be kept
        """
        current_mode = self.network.training
        self.network.eval()

        assert self.was_initialized, "must initialize, ideally with checkpoint (or train first)"
        if self.dataset_val is None:
            self.load_dataset()
            self.do_split()

        if segmentation_export_kwargs is None:
            if 'segmentation_export_params' in self.plans.keys():
                force_separate_z = self.plans['segmentation_export_params']['force_separate_z']
                interpolation_order = self.plans['segmentation_export_params']['interpolation_order']
                interpolation_order_z = self.plans['segmentation_export_params']['interpolation_order_z']
            else:
                force_separate_z = None
                interpolation_order = 1
                interpolation_order_z = 0
        else:
            force_separate_z = segmentation_export_kwargs['force_separate_z']
            interpolation_order = segmentation_export_kwargs['interpolation_order']
            interpolation_order_z = segmentation_export_kwargs['interpolation_order_z']

        # predictions as they come from the network go here
        if output_folder is not None:
            self.output_folder = output_folder
        output_folder = join(self.output_folder, validation_folder_name)
        maybe_mkdir_p(output_folder)
        # this is for debug purposes
        my_input_args = {'do_mirroring': do_mirroring,
                         'use_sliding_window': use_sliding_window,
                         'step_size': step_size,
                         'save_softmax': save_softmax,
                         'use_gaussian': use_gaussian,
                         'overwrite': overwrite,
                         'validation_folder_name': validation_folder_name,
                         'debug': debug,
                         'all_in_gpu': all_in_gpu,
                         'segmentation_export_kwargs': segmentation_export_kwargs,
                         }
        save_json(my_input_args, join(output_folder, "validation_args.json"))

        if do_mirroring:
            if not self.data_aug_params['do_mirror']:
                raise RuntimeError("We did not train with mirroring so you cannot do inference with mirroring enabled")
            mirror_axes = self.data_aug_params['mirror_axes']
        else:
            mirror_axes = ()

        pred_gt_tuples = []

        export_pool = Pool(default_num_threads)
        results = []

        list_of_keys = list(self.dataset_val.keys())
        un_list_of_keys = list(self.dataset_un_val.keys())
        metadata_list = []
        for k in list_of_keys:
            properties = load_pickle(self.dataset[k]['properties_file'])
            metadata_list.append(self.create_metadata_dict(properties))
            fname = properties['list_of_data_files'][0].split(os.sep)[-1][:-12]

            patient_id = k[:10]
            l_filtered = [x for x in list_of_keys if x[:10] in patient_id]
            un_filtered = [x for x in un_list_of_keys if x[:10] in patient_id]
            filtered = l_filtered + un_filtered
            filtered = sorted(filtered, key=lambda x: int(x[16:18]))
            filtered = np.array(filtered)
            labeled_idx = np.where(filtered == k)[0][0]

            if step > 1:
                if labeled_idx % 2 == 0:
                    filtered = filtered[0::step]
                else:
                    filtered = filtered[1::step]
                labeled_idx = np.where(filtered == k)[0][0]
            

            values = np.arange(len(filtered))
            before = self.video_length // 2
            after = before + (self.video_length % 2)
            values = np.pad(values, (before, after), mode='wrap')
            mask = np.isin(values, labeled_idx)
            possible_indices = np.argwhere(mask)
            possible_indices = possible_indices[np.logical_and(possible_indices >= before, possible_indices <= len(values) - after)]
            m = np.random.choice(possible_indices)
            start = m - before
            end = m + after
            assert start >= 0
            assert end <= len(values)
            #start = min(max(labeled_idx - half_window, 0), len(filtered) - self.video_length)
            #frame_indices = values[start:start + self.video_length]
            frame_indices = values[start:end]
            assert len(frame_indices) == self.video_length
            video = filtered[frame_indices]

            data = np.zeros(shape=(self.video_length, 1) + properties['size_after_resampling'])

            labeled_idx = None
            if overwrite or (not isfile(join(output_folder, fname + ".nii.gz"))) or (save_softmax and not isfile(join(output_folder, fname + ".npz"))):
                for idx, frame in enumerate(video):
                    video_idx = idx
                    if frame == k:
                        labeled_idx = video_idx
                    if '_u' in frame:
                        data[video_idx] = np.load(self.dataset_un_val[frame]['data_file'])['data']
                    else:
                        current_data = np.load(self.dataset_val[frame]['data_file'])['data']
                        current_data[-1][current_data[-1] == -1] = 0
                        data[video_idx] = current_data[:-1]
                    #self.print_to_log_file(k, data.shape)

                #matplotlib.use('QtAgg')
                #print(labeled_idx)
                #print(video_padding)
                #fig, ax = plt.subplots(1, self.video_length)
                #for i in range(self.video_length):
                #    ax[i].imshow(data[i, 0, 5], cmap='gray')
                #plt.show()

                softmax_pred = self.predict_preprocessed_data_return_seg_and_softmax_video(data=data,
                                                                                     idx=labeled_idx,
                                                                                     processor=processor,
                                                                                     do_mirroring=do_mirroring,
                                                                                     mirror_axes=mirror_axes,
                                                                                     use_sliding_window=use_sliding_window,
                                                                                     step_size=step_size,
                                                                                     use_gaussian=use_gaussian,
                                                                                     all_in_gpu=all_in_gpu,
                                                                                     mixed_precision=self.fp16)[1]

                softmax_pred = softmax_pred.transpose([0] + [i + 1 for i in self.transpose_backward])

                if save_softmax:
                    softmax_fname = join(output_folder, fname + ".npz")
                else:
                    softmax_fname = None

                """There is a problem with python process communication that prevents us from communicating objects
                larger than 2 GB between processes (basically when the length of the pickle string that will be sent is
                communicated by the multiprocessing.Pipe object then the placeholder (I think) does not allow for long
                enough strings (lol). This could be fixed by changing i to l (for long) but that would require manually
                patching system python code. We circumvent that problem here by saving softmax_pred to a npy file that will
                then be read (and finally deleted) by the Process. save_segmentation_nifti_from_softmax can take either
                filename or np.ndarray and will handle this automatically"""
                if np.prod(softmax_pred.shape) > (2e9 / 4 * 0.85):  # *0.85 just to be save
                    np.save(join(output_folder, fname + ".npy"), softmax_pred)
                    softmax_pred = join(output_folder, fname + ".npy")

                results.append(export_pool.starmap_async(save_segmentation_nifti_from_softmax,
                                                         ((softmax_pred, join(output_folder, fname + ".nii.gz"),
                                                           properties, interpolation_order, self.regions_class_order,
                                                           None, None,
                                                           softmax_fname, None, force_separate_z,
                                                           interpolation_order_z),
                                                          )
                                                         )
                               )

            pred_gt_tuples.append([join(output_folder, fname + ".nii.gz"),
                                   join(self.gt_niftis_folder, fname + ".nii.gz")])

        _ = [i.get() for i in results]
        self.print_to_log_file("finished prediction")

        # evaluate raw predictions
        self.print_to_log_file("evaluation of raw predictions")
        task = self.dataset_directory.split(os.sep)[-1]
        job_name = self.experiment_name
        _ = aggregate_scores(pred_gt_tuples, labels=list(range(self.num_classes)),
                             json_output_file=join(output_folder, "summary.json"),
                             json_name=job_name + " val tiled %s" % (str(use_sliding_window)),
                             json_author="Fabian",
                             json_task=task, num_threads=default_num_threads,
                             advanced=True,
                             metadata_list=metadata_list)

        if run_postprocessing_on_folds:
            # in the old nnunet we would stop here. Now we add a postprocessing. This postprocessing can remove everything
            # except the largest connected component for each class. To see if this improves results, we do this for all
            # classes and then rerun the evaluation. Those classes for which this resulted in an improved dice score will
            # have this applied during inference as well
            self.print_to_log_file("determining postprocessing")
            determine_postprocessing(self.output_folder, self.gt_niftis_folder, validation_folder_name,
                                     final_subf_name=validation_folder_name + "_postprocessed", debug=debug, log_function=log_function,
                                     metadata_list=metadata_list)
            # after this the final predictions for the vlaidation set can be found in validation_folder_name_base + "_postprocessed"
            # They are always in that folder, even if no postprocessing as applied!

        # detemining postprocesing on a per-fold basis may be OK for this fold but what if another fold finds another
        # postprocesing to be better? In this case we need to consolidate. At the time the consolidation is going to be
        # done we won't know what self.gt_niftis_folder was, so now we copy all the niftis into a separate folder to
        # be used later
        gt_nifti_folder = join(self.output_folder_base, "gt_niftis")
        maybe_mkdir_p(gt_nifti_folder)
        for f in subfiles(self.gt_niftis_folder, suffix=".nii.gz"):
            success = False
            attempts = 0
            e = None
            while not success and attempts < 10:
                try:
                    shutil.copy(f, gt_nifti_folder)
                    success = True
                except OSError as e:
                    attempts += 1
                    sleep(1)
            if not success:
                print("Could not copy gt nifti file %s into folder %s" % (f, gt_nifti_folder))
                if e is not None:
                    raise e

        self.network.train(current_mode)
    

    def validate_flow(self, processor, log_function, step, do_mirroring: bool = True, use_sliding_window: bool = True, step_size: float = 0.5,
                 save_softmax: bool = True, use_gaussian: bool = True, overwrite: bool = True,
                 validation_folder_name: str = 'validation_raw', debug: bool = True, all_in_gpu: bool = False,
                 segmentation_export_kwargs: dict = None, run_postprocessing_on_folds: bool = True, output_folder=None, save_flow=True):

        """
        if debug=True then the temporary files generated for postprocessing determination will be kept
        """
        current_mode = self.network.training
        self.network.eval()

        assert self.was_initialized, "must initialize, ideally with checkpoint (or train first)"
        if self.dataset_val is None:
            self.load_dataset()
            self.do_split()

        if segmentation_export_kwargs is None:
            if 'segmentation_export_params' in self.plans.keys():
                force_separate_z = self.plans['segmentation_export_params']['force_separate_z']
                interpolation_order = self.plans['segmentation_export_params']['interpolation_order']
                interpolation_order_z = self.plans['segmentation_export_params']['interpolation_order_z']
            else:
                force_separate_z = None
                interpolation_order = 1
                interpolation_order_z = 0
        else:
            force_separate_z = segmentation_export_kwargs['force_separate_z']
            interpolation_order = segmentation_export_kwargs['interpolation_order']
            interpolation_order_z = segmentation_export_kwargs['interpolation_order_z']

        # predictions as they come from the network go here
        if output_folder is not None:
            self.output_folder = output_folder
        output_folder = join(self.output_folder, validation_folder_name)
        maybe_mkdir_p(output_folder)
        # this is for debug purposes
        my_input_args = {'do_mirroring': do_mirroring,
                         'use_sliding_window': use_sliding_window,
                         'step_size': step_size,
                         'save_softmax': save_softmax,
                         'use_gaussian': use_gaussian,
                         'overwrite': overwrite,
                         'validation_folder_name': validation_folder_name,
                         'debug': debug,
                         'all_in_gpu': all_in_gpu,
                         'segmentation_export_kwargs': segmentation_export_kwargs,
                         }
        save_json(my_input_args, join(output_folder, "validation_args.json"))

        if do_mirroring:
            if not self.data_aug_params['do_mirror']:
                raise RuntimeError("We did not train with mirroring so you cannot do inference with mirroring enabled")
            mirror_axes = self.data_aug_params['mirror_axes']
        else:
            mirror_axes = ()

        pred_gt_tuples = []
        pred_gt_tuples_register = []

        export_pool = Pool(default_num_threads)
        results = []

        if save_flow:
            newpath_flow = join(output_folder, 'flow')
            if not os.path.exists(newpath_flow):
                os.makedirs(newpath_flow)

            newpath_registered = join(output_folder, 'registered')
            if not os.path.exists(newpath_registered):
                os.makedirs(newpath_registered)

        list_of_keys = list(self.dataset_val.keys())
        un_list_of_keys = list(self.dataset_un_val.keys())
        patient_id_list = np.unique([x[:10] for x in list_of_keys])
        metadata_list = []
        metadata_list_registered = []
        for patient_id in tqdm(patient_id_list):
            all_keys = list_of_keys + un_list_of_keys
            phase_list = [x for x in all_keys if patient_id in x]
            phase_list = np.array(sorted(phase_list, key=lambda x: int(x[16:18])))
            phase_list = np.array(phase_list)
            global_labeled_idx = np.where(~np.char.endswith(phase_list, '_u'))[0]

            if '_u' in phase_list[0]:
                properties = load_pickle(self.unlabeled_dataset[phase_list[0]]['properties_file'])
            else:
                properties = load_pickle(self.dataset[phase_list[0]]['properties_file'])

            unlabeled = np.full(shape=((len(phase_list), 1) + properties['size_after_resampling']), fill_value=np.nan)
            target = np.full(shape=((len(phase_list), 1) + properties['size_after_resampling']), fill_value=np.nan)
            target_mask = np.zeros(shape=(len(phase_list),), dtype=bool)
            padding_mask = np.ones(shape=(len(phase_list),), dtype=bool)

            if overwrite or (not isfile(join(output_folder, phase_list[0] + ".nii.gz"))) or (save_softmax and not isfile(join(output_folder, phase_list[0] + ".npz"))):
                for idx, frame in enumerate(phase_list):
                    #if frame == k:
                    #    labeled_idx = idx
                    if '_u' in frame:
                        current_data = np.load(self.dataset_un_val[frame]['data_file'])['data']
                        current_data = current_data + 1e-8
                        unlabeled[idx] = current_data
                    else:
                        current_data = np.load(self.dataset_val[frame]['data_file'])['data']
                        #current_data[-1][current_data[-1] == -1] = 0
                        target[idx] = current_data[1]
                        target[idx][target[idx] == -1] = 0
                        unlabeled[idx] = current_data[0] + 1e-8
                        target_mask[idx] = True
                    #self.print_to_log_file(k, data.shape)
                
                values = np.arange(len(phase_list))
                #if self.step > 1:
                #    values = values[::self.step]
                windows = [values[i : i + self.video_length] for i in range(0, len(values), self.video_length - 1)]
                padding_mask = np.concatenate([padding_mask, np.zeros(shape=(self.video_length - len(windows[-1]),), dtype=bool)], axis=0)
                target_mask = np.concatenate([target_mask, np.zeros(shape=(self.video_length - len(windows[-1]),), dtype=bool)], axis=0)
                windows[-1] = np.concatenate([windows[-1], np.arange(self.video_length - len(windows[-1]))])
                unlabeled = [unlabeled[x] for x in windows]
                unlabeled = np.stack(unlabeled, axis=0) # P, T, 1, D, H, W


                ret = self.predict_preprocessed_data_return_seg_and_softmax_flow(unlabeled=unlabeled,
                                                                                target_mask=target_mask,
                                                                                padding_mask=padding_mask,
                                                                                target=target,
                                                                                processor=processor,
                                                                                do_mirroring=do_mirroring,
                                                                                mirror_axes=mirror_axes,
                                                                                use_sliding_window=use_sliding_window,
                                                                                step_size=step_size,
                                                                                use_gaussian=use_gaussian,
                                                                                all_in_gpu=all_in_gpu,
                                                                                mixed_precision=self.fp16,
                                                                                verbose=False)
                predicted_segmentation = ret[0] # T, depth, H, W
                softmax_pred = ret[1] # T, C, depth, H, W
                flow_pred = ret[2] # T, C, depth, H, W
                registered_pred = ret[3] # C, depth, H, W

                #matplotlib.use('QtAgg')
                #fig, ax = plt.subplots(1, 7)
                #plot_idx = 0
                #for a in range(8, 15):
                #    current_img = softmax_pred[a, :, 0]
                #    current_img = current_img.argmax(0)
                #    ax[plot_idx].imshow(current_img, cmap='gray')
                #    plot_idx += 1
                #plt.show()
                #plt.waitforbuttonpress()
                #plt.close(fig)

                #stop_indices = [len(x) for x in windows]
                #videos = [phase_list[x] for x in windows]

                #matplotlib.use('QtAgg')
                #print(labeled_idx)
                #print(video_padding)
                #fig, ax = plt.subplots(1, self.video_length)
                #for i in range(self.video_length):
                #    ax[i].imshow(data[i, 0, 5], cmap='gray')
                #plt.show()

                assert len(softmax_pred) == len(flow_pred) == len(phase_list)

                and_mask = target_mask[padding_mask]
                gt_indices = np.where(and_mask)[0]
                assert len(gt_indices) == 2

                for t in range(len(softmax_pred)):
                    if '_u' in phase_list[t]:
                        properties = load_pickle(self.unlabeled_dataset[phase_list[t]]['properties_file'])
                    else:
                        properties = load_pickle(self.dataset[phase_list[t]]['properties_file'])
                    fname = properties['list_of_data_files'][0].split(os.sep)[-1][:-12]

                    current_softmax_pred = softmax_pred[t]
                    current_softmax_pred = current_softmax_pred.transpose([0] + [i + 1 for i in self.transpose_backward])

                    if save_softmax:
                        softmax_fname = join(output_folder, fname + ".npz")
                    else:
                        softmax_fname = None

                    if save_flow and t > 0:
                        splitted = fname.split('frame')
                        from_nb = int(splitted[-1].split('_')[0].split('.')[0]) - 1
                        to_nb = from_nb + 1
                        flow_name = splitted[0] + 'frame' + str(from_nb).zfill(2) + '_to_' + str(to_nb).zfill(2)
                        flow_path = join(output_folder, 'flow', flow_name + ".nii.gz")
                        current_flow = flow_pred[t]
                        current_flow = current_flow.transpose([0] + [i + 1 for i in self.transpose_backward])
                    else:
                        flow_path = current_flow = None

                    if t == gt_indices[0]:
                        registered_path = join(output_folder, 'registered', fname + ".nii.gz")
                        current_registered = registered_pred
                        current_registered = current_registered.transpose([0] + [i + 1 for i in self.transpose_backward])
                    else:
                        registered_path = current_registered = None

                    """There is a problem with python process communication that prevents us from communicating objects
                    larger than 2 GB between processes (basically when the length of the pickle string that will be sent is
                    communicated by the multiprocessing.Pipe object then the placeholder (I think) does not allow for long
                    enough strings (lol). This could be fixed by changing i to l (for long) but that would require manually
                    patching system python code. We circumvent that problem here by saving softmax_pred to a npy file that will
                    then be read (and finally deleted) by the Process. save_segmentation_nifti_from_softmax can take either
                    filename or np.ndarray and will handle this automatically"""
                    if np.prod(current_softmax_pred.shape) > (2e9 / 4 * 0.85):  # *0.85 just to be save
                        np.save(join(output_folder, fname + ".npy"), current_softmax_pred)
                        current_softmax_pred = join(output_folder, fname + ".npy")

                    results.append(export_pool.starmap_async(save_segmentation_nifti_from_softmax,
                                                            ((current_softmax_pred, join(output_folder, fname + ".nii.gz"),
                                                            properties, interpolation_order, self.regions_class_order,
                                                            None, None,
                                                            softmax_fname, None, force_separate_z,
                                                            interpolation_order_z, False, current_flow, flow_path,
                                                            current_registered, registered_path),
                                                            )
                                                            )
                                )
                    
                    
                    if t in gt_indices:
                        metadata_list.append(self.create_metadata_dict(properties))
                        pred_gt_tuples.append([join(output_folder, fname + ".nii.gz"),
                                                join(self.gt_niftis_folder, fname + ".nii.gz")])
                    if t == gt_indices[0]:
                        metadata_list_registered.append(self.create_metadata_dict(properties))
                        pred_gt_tuples_register.append([registered_path,
                                                join(self.gt_niftis_folder, fname + ".nii.gz")])

        _ = [i.get() for i in results]
        self.print_to_log_file("finished prediction")

        # evaluate raw predictions
        self.print_to_log_file("evaluation of raw predictions")
        task = self.dataset_directory.split(os.sep)[-1]
        job_name = self.experiment_name
        _ = aggregate_scores(pred_gt_tuples_register, labels=list(range(self.num_classes)),
                             json_output_file=join(output_folder, 'registered', "summary.json"),
                             json_name=job_name + " val tiled %s" % (str(use_sliding_window)),
                             json_author="Fabian",
                             json_task=task, num_threads=default_num_threads,
                             advanced=True,
                             metadata_list=metadata_list_registered)
        _ = aggregate_scores(pred_gt_tuples, labels=list(range(self.num_classes)),
                             json_output_file=join(output_folder, "summary.json"),
                             json_name=job_name + " val tiled %s" % (str(use_sliding_window)),
                             json_author="Fabian",
                             json_task=task, num_threads=default_num_threads,
                             advanced=True,
                             metadata_list=metadata_list)

        if run_postprocessing_on_folds:
            # in the old nnunet we would stop here. Now we add a postprocessing. This postprocessing can remove everything
            # except the largest connected component for each class. To see if this improves results, we do this for all
            # classes and then rerun the evaluation. Those classes for which this resulted in an improved dice score will
            # have this applied during inference as well
            self.print_to_log_file("determining postprocessing")
            determine_postprocessing(self.output_folder, self.gt_niftis_folder, validation_folder_name,
                                     final_subf_name=validation_folder_name + "_postprocessed", debug=debug, log_function=log_function,
                                     metadata_list=metadata_list)
            # after this the final predictions for the vlaidation set can be found in validation_folder_name_base + "_postprocessed"
            # They are always in that folder, even if no postprocessing as applied!

        # detemining postprocesing on a per-fold basis may be OK for this fold but what if another fold finds another
        # postprocesing to be better? In this case we need to consolidate. At the time the consolidation is going to be
        # done we won't know what self.gt_niftis_folder was, so now we copy all the niftis into a separate folder to
        # be used later
        gt_nifti_folder = join(self.output_folder_base, "gt_niftis")
        maybe_mkdir_p(gt_nifti_folder)
        for f in subfiles(self.gt_niftis_folder, suffix=".nii.gz"):
            success = False
            attempts = 0
            e = None
            while not success and attempts < 10:
                try:
                    shutil.copy(f, gt_nifti_folder)
                    success = True
                except OSError as e:
                    attempts += 1
                    sleep(1)
            if not success:
                print("Could not copy gt nifti file %s into folder %s" % (f, gt_nifti_folder))
                if e is not None:
                    raise e

        self.network.train(current_mode)


    def validate_flow_sliding_window(self, processor, log_function, step, do_mirroring: bool = True, use_sliding_window: bool = True, step_size: float = 0.5,
                 save_softmax: bool = True, use_gaussian: bool = True, overwrite: bool = True,
                 validation_folder_name: str = 'validation_raw', debug: bool = True, all_in_gpu: bool = False,
                 segmentation_export_kwargs: dict = None, run_postprocessing_on_folds: bool = True, output_folder=None, save_flow=True):

        """
        if debug=True then the temporary files generated for postprocessing determination will be kept
        """
        current_mode = self.network.training
        self.network.eval()

        assert self.was_initialized, "must initialize, ideally with checkpoint (or train first)"
        if self.dataset_val is None:
            self.load_dataset()
            self.do_split()

        if segmentation_export_kwargs is None:
            if 'segmentation_export_params' in self.plans.keys():
                force_separate_z = self.plans['segmentation_export_params']['force_separate_z']
                interpolation_order = self.plans['segmentation_export_params']['interpolation_order']
                interpolation_order_z = self.plans['segmentation_export_params']['interpolation_order_z']
            else:
                force_separate_z = None
                interpolation_order = 1
                interpolation_order_z = 0
        else:
            force_separate_z = segmentation_export_kwargs['force_separate_z']
            interpolation_order = segmentation_export_kwargs['interpolation_order']
            interpolation_order_z = segmentation_export_kwargs['interpolation_order_z']

        # predictions as they come from the network go here
        if output_folder is not None:
            self.output_folder = output_folder
        output_folder = join(self.output_folder, validation_folder_name)
        maybe_mkdir_p(output_folder)
        # this is for debug purposes
        my_input_args = {'do_mirroring': do_mirroring,
                         'use_sliding_window': use_sliding_window,
                         'step_size': step_size,
                         'save_softmax': save_softmax,
                         'use_gaussian': use_gaussian,
                         'overwrite': overwrite,
                         'validation_folder_name': validation_folder_name,
                         'debug': debug,
                         'all_in_gpu': all_in_gpu,
                         'segmentation_export_kwargs': segmentation_export_kwargs,
                         }
        save_json(my_input_args, join(output_folder, "validation_args.json"))

        if do_mirroring:
            if not self.data_aug_params['do_mirror']:
                raise RuntimeError("We did not train with mirroring so you cannot do inference with mirroring enabled")
            mirror_axes = self.data_aug_params['mirror_axes']
        else:
            mirror_axes = ()

        pred_gt_tuples = []
        pred_gt_tuples_register = []

        export_pool = Pool(default_num_threads)
        results = []

        if save_flow:
            newpath_flow = join(output_folder, 'flow')
            if not os.path.exists(newpath_flow):
                os.makedirs(newpath_flow)

            newpath_registered = join(output_folder, 'registered')
            if not os.path.exists(newpath_registered):
                os.makedirs(newpath_registered)

        list_of_keys = list(self.dataset_val.keys())
        un_list_of_keys = list(self.dataset_un_val.keys())
        patient_id_list = np.unique([x[:10] for x in list_of_keys])
        metadata_list = []
        metadata_list_registered = []
        for patient_id in tqdm(patient_id_list):
            all_keys = list_of_keys + un_list_of_keys
            phase_list = [x for x in all_keys if patient_id in x]
            phase_list = np.array(sorted(phase_list, key=lambda x: int(x[16:18])))
            phase_list = np.array(phase_list)
            global_labeled_idx = np.where(~np.char.endswith(phase_list, '_u'))[0]

            if '_u' in phase_list[0]:
                properties = load_pickle(self.unlabeled_dataset[phase_list[0]]['properties_file'])
            else:
                properties = load_pickle(self.dataset[phase_list[0]]['properties_file'])

            unlabeled = np.full(shape=((len(phase_list), 1) + properties['size_after_resampling']), fill_value=np.nan)
            target = np.full(shape=((len(phase_list), 1) + properties['size_after_resampling']), fill_value=np.nan)
            target_mask = np.zeros(shape=(len(phase_list),), dtype=bool)
            padding_mask = np.ones(shape=(len(phase_list),), dtype=bool)

            if overwrite or (not isfile(join(output_folder, phase_list[0] + ".nii.gz"))) or (save_softmax and not isfile(join(output_folder, phase_list[0] + ".npz"))):
                for idx, frame in enumerate(phase_list):
                    #if frame == k:
                    #    labeled_idx = idx
                    if '_u' in frame:
                        current_data = np.load(self.dataset_un_val[frame]['data_file'])['data']
                        current_data = current_data + 1e-8
                        unlabeled[idx] = current_data
                    else:
                        current_data = np.load(self.dataset_val[frame]['data_file'])['data']
                        #current_data[-1][current_data[-1] == -1] = 0
                        target[idx] = current_data[1]
                        target[idx][target[idx] == -1] = 0
                        unlabeled[idx] = current_data[0] + 1e-8
                        target_mask[idx] = True
                    #self.print_to_log_file(k, data.shape)

                values = np.arange(len(phase_list))
                assert self.video_length % 2 == 1
                values = np.pad(values, (self.video_length // 2, self.video_length // 2), mode='wrap')

                if self.step > 1:
                    values = values[::self.step]

                windows = sliding_window_view(values, self.video_length)
                unlabeled = [unlabeled[x] for x in windows]
                unlabeled = np.stack(unlabeled, axis=0) # P, T, 1, D, H, W
                assert len(unlabeled) == len(phase_list)

                ret = self.predict_preprocessed_data_return_seg_and_softmax_flow(unlabeled=unlabeled,
                                                                                target_mask=target_mask,
                                                                                padding_mask=padding_mask,
                                                                                target=target,
                                                                                processor=processor,
                                                                                do_mirroring=do_mirroring,
                                                                                mirror_axes=mirror_axes,
                                                                                use_sliding_window=use_sliding_window,
                                                                                step_size=step_size,
                                                                                use_gaussian=use_gaussian,
                                                                                all_in_gpu=all_in_gpu,
                                                                                mixed_precision=self.fp16,
                                                                                verbose=False)
                predicted_segmentation = ret[0] # T, depth, H, W
                softmax_pred = ret[1] # T, C, depth, H, W
                flow_pred = ret[2] # T, C, depth, H, W
                registered_pred = ret[3] # C, depth, H, W

                #matplotlib.use('QtAgg')
                #fig, ax = plt.subplots(1, 7)
                #plot_idx = 0
                #for a in range(8, 15):
                #    current_img = softmax_pred[a, :, 0]
                #    current_img = current_img.argmax(0)
                #    ax[plot_idx].imshow(current_img, cmap='gray')
                #    plot_idx += 1
                #plt.show()
                #plt.waitforbuttonpress()
                #plt.close(fig)

                #stop_indices = [len(x) for x in windows]
                #videos = [phase_list[x] for x in windows]

                #matplotlib.use('QtAgg')
                #print(labeled_idx)
                #print(video_padding)
                #fig, ax = plt.subplots(1, self.video_length)
                #for i in range(self.video_length):
                #    ax[i].imshow(data[i, 0, 5], cmap='gray')
                #plt.show()

                assert len(softmax_pred) == len(flow_pred) == len(phase_list)

                and_mask = target_mask[padding_mask]
                gt_indices = np.where(and_mask)[0]
                assert len(gt_indices) == 2

                for t in range(len(softmax_pred)):
                    if '_u' in phase_list[t]:
                        properties = load_pickle(self.unlabeled_dataset[phase_list[t]]['properties_file'])
                    else:
                        properties = load_pickle(self.dataset[phase_list[t]]['properties_file'])
                    fname = properties['list_of_data_files'][0].split(os.sep)[-1][:-12]

                    current_softmax_pred = softmax_pred[t]
                    current_softmax_pred = current_softmax_pred.transpose([0] + [i + 1 for i in self.transpose_backward])

                    if save_softmax:
                        softmax_fname = join(output_folder, fname + ".npz")
                    else:
                        softmax_fname = None

                    if save_flow and t > 0:
                        splitted = fname.split('frame')
                        from_nb = int(splitted[-1].split('_')[0].split('.')[0]) - 1
                        to_nb = from_nb + 1
                        flow_name = splitted[0] + 'frame' + str(from_nb).zfill(2) + '_to_' + str(to_nb).zfill(2)
                        flow_path = join(output_folder, 'flow', flow_name + ".nii.gz")
                        current_flow = flow_pred[t]
                        current_flow = current_flow.transpose([0] + [i + 1 for i in self.transpose_backward])
                    else:
                        flow_path = current_flow = None

                    if t == gt_indices[0]:
                        registered_path = join(output_folder, 'registered', fname + ".nii.gz")
                        current_registered = registered_pred
                        current_registered = current_registered.transpose([0] + [i + 1 for i in self.transpose_backward])
                    else:
                        registered_path = current_registered = None

                    """There is a problem with python process communication that prevents us from communicating objects
                    larger than 2 GB between processes (basically when the length of the pickle string that will be sent is
                    communicated by the multiprocessing.Pipe object then the placeholder (I think) does not allow for long
                    enough strings (lol). This could be fixed by changing i to l (for long) but that would require manually
                    patching system python code. We circumvent that problem here by saving softmax_pred to a npy file that will
                    then be read (and finally deleted) by the Process. save_segmentation_nifti_from_softmax can take either
                    filename or np.ndarray and will handle this automatically"""
                    if np.prod(current_softmax_pred.shape) > (2e9 / 4 * 0.85):  # *0.85 just to be save
                        np.save(join(output_folder, fname + ".npy"), current_softmax_pred)
                        current_softmax_pred = join(output_folder, fname + ".npy")

                    results.append(export_pool.starmap_async(save_segmentation_nifti_from_softmax,
                                                            ((current_softmax_pred, join(output_folder, fname + ".nii.gz"),
                                                            properties, interpolation_order, self.regions_class_order,
                                                            None, None,
                                                            softmax_fname, None, force_separate_z,
                                                            interpolation_order_z, False, current_flow, flow_path,
                                                            current_registered, registered_path),
                                                            )
                                                            )
                                )
                    
                    
                    if t in gt_indices:
                        metadata_list.append(self.create_metadata_dict(properties))
                        pred_gt_tuples.append([join(output_folder, fname + ".nii.gz"),
                                                join(self.gt_niftis_folder, fname + ".nii.gz")])
                    if t == gt_indices[0]:
                        metadata_list_registered.append(self.create_metadata_dict(properties))
                        pred_gt_tuples_register.append([registered_path,
                                                join(self.gt_niftis_folder, fname + ".nii.gz")])

        _ = [i.get() for i in results]
        self.print_to_log_file("finished prediction")

        # evaluate raw predictions
        self.print_to_log_file("evaluation of raw predictions")
        task = self.dataset_directory.split(os.sep)[-1]
        job_name = self.experiment_name
        _ = aggregate_scores(pred_gt_tuples_register, labels=list(range(self.num_classes)),
                             json_output_file=join(output_folder, 'registered', "summary.json"),
                             json_name=job_name + " val tiled %s" % (str(use_sliding_window)),
                             json_author="Fabian",
                             json_task=task, num_threads=default_num_threads,
                             advanced=True,
                             metadata_list=metadata_list_registered)
        _ = aggregate_scores(pred_gt_tuples, labels=list(range(self.num_classes)),
                             json_output_file=join(output_folder, "summary.json"),
                             json_name=job_name + " val tiled %s" % (str(use_sliding_window)),
                             json_author="Fabian",
                             json_task=task, num_threads=default_num_threads,
                             advanced=True,
                             metadata_list=metadata_list)

        if run_postprocessing_on_folds:
            # in the old nnunet we would stop here. Now we add a postprocessing. This postprocessing can remove everything
            # except the largest connected component for each class. To see if this improves results, we do this for all
            # classes and then rerun the evaluation. Those classes for which this resulted in an improved dice score will
            # have this applied during inference as well
            self.print_to_log_file("determining postprocessing")
            determine_postprocessing(self.output_folder, self.gt_niftis_folder, validation_folder_name,
                                     final_subf_name=validation_folder_name + "_postprocessed", debug=debug, log_function=log_function,
                                     metadata_list=metadata_list)
            # after this the final predictions for the vlaidation set can be found in validation_folder_name_base + "_postprocessed"
            # They are always in that folder, even if no postprocessing as applied!

        # detemining postprocesing on a per-fold basis may be OK for this fold but what if another fold finds another
        # postprocesing to be better? In this case we need to consolidate. At the time the consolidation is going to be
        # done we won't know what self.gt_niftis_folder was, so now we copy all the niftis into a separate folder to
        # be used later
        gt_nifti_folder = join(self.output_folder_base, "gt_niftis")
        maybe_mkdir_p(gt_nifti_folder)
        for f in subfiles(self.gt_niftis_folder, suffix=".nii.gz"):
            success = False
            attempts = 0
            e = None
            while not success and attempts < 10:
                try:
                    shutil.copy(f, gt_nifti_folder)
                    success = True
                except OSError as e:
                    attempts += 1
                    sleep(1)
            if not success:
                print("Could not copy gt nifti file %s into folder %s" % (f, gt_nifti_folder))
                if e is not None:
                    raise e

        self.network.train(current_mode)
    

    def validate_stable_diffusion(self, images, do_mirroring: bool = True, use_sliding_window: bool = True, step_size: float = 0.5,
                 save_softmax: bool = True, use_gaussian: bool = True, overwrite: bool = True,
                 validation_folder_name: str = 'validation_raw', debug: bool = True, all_in_gpu: bool = False,
                 segmentation_export_kwargs: dict = None, output_folder=None):

        """
        if debug=True then the temporary files generated for postprocessing determination will be kept
        """
        current_mode = self.network.training
        self.network.eval()

        assert self.was_initialized, "must initialize, ideally with checkpoint (or train first)"
        if self.dataset_val is None:
            self.load_dataset()
            self.do_split()

        # predictions as they come from the network go here
        if output_folder is not None:
            self.output_folder = output_folder
        #output_folder = join(self.output_folder, validation_folder_name)
        #maybe_mkdir_p(output_folder)
        # this is for debug purposes
        my_input_args = {'do_mirroring': do_mirroring,
                         'use_sliding_window': use_sliding_window,
                         'step_size': step_size,
                         'save_softmax': save_softmax,
                         'use_gaussian': use_gaussian,
                         'overwrite': overwrite,
                         'validation_folder_name': validation_folder_name,
                         'debug': debug,
                         'all_in_gpu': all_in_gpu,
                         'segmentation_export_kwargs': segmentation_export_kwargs,
                         }
        save_json(my_input_args, join(self.output_folder, "validation_args.json"))

        newpath_predictions = join(self.output_folder, 'Predictions', validation_folder_name)
        if not os.path.exists(newpath_predictions):
            os.makedirs(newpath_predictions)

        for i in range(len(images)):
            current_image = images[i]
            filename = uuid.uuid4().hex[:20].upper()
            prediction_path = join(newpath_predictions, filename + ".png")
            cv.imwrite(prediction_path, current_image)

        self.network.train(current_mode)

    


    def validate_flow_one_step_simple(self, processor, log_function, step, do_mirroring: bool = True, use_sliding_window: bool = True, step_size: float = 0.5,
                 save_softmax: bool = True, use_gaussian: bool = True, overwrite: bool = True,
                 validation_folder_name: str = 'validation_raw', debug: bool = True, all_in_gpu: bool = False,
                 segmentation_export_kwargs: dict = None, run_postprocessing_on_folds: bool = True, output_folder=None, save_flow=True):

        """
        if debug=True then the temporary files generated for postprocessing determination will be kept
        """
        current_mode = self.network.training
        self.network.eval()

        assert self.was_initialized, "must initialize, ideally with checkpoint (or train first)"
        if self.dataset_val is None:
            self.load_dataset()
            self.do_split()

        if segmentation_export_kwargs is None:
            if 'segmentation_export_params' in self.plans.keys():
                force_separate_z = self.plans['segmentation_export_params']['force_separate_z']
                interpolation_order = self.plans['segmentation_export_params']['interpolation_order']
                interpolation_order_z = self.plans['segmentation_export_params']['interpolation_order_z']
            else:
                force_separate_z = None
                interpolation_order = 1
                interpolation_order_z = 0
        else:
            force_separate_z = segmentation_export_kwargs['force_separate_z']
            interpolation_order = segmentation_export_kwargs['interpolation_order']
            interpolation_order_z = segmentation_export_kwargs['interpolation_order_z']

        # predictions as they come from the network go here
        if output_folder is not None:
            self.output_folder = output_folder
        #output_folder = join(self.output_folder, validation_folder_name)
        #maybe_mkdir_p(output_folder)
        # this is for debug purposes
        my_input_args = {'do_mirroring': do_mirroring,
                         'use_sliding_window': use_sliding_window,
                         'step_size': step_size,
                         'save_softmax': save_softmax,
                         'use_gaussian': use_gaussian,
                         'overwrite': overwrite,
                         'validation_folder_name': validation_folder_name,
                         'debug': debug,
                         'all_in_gpu': all_in_gpu,
                         'segmentation_export_kwargs': segmentation_export_kwargs,
                         }
        save_json(my_input_args, join(self.output_folder, "validation_args.json"))

        if do_mirroring:
            if not self.data_aug_params['do_mirror']:
                raise RuntimeError("We did not train with mirroring so you cannot do inference with mirroring enabled")
            mirror_axes = self.data_aug_params['mirror_axes']
        else:
            mirror_axes = ()

        pred_gt_tuples = []
        pred_gt_tuples_register = []

        export_pool = Pool(default_num_threads)
        results = []

        newpath_flow = join(self.output_folder, 'Flow', validation_folder_name)
        if not os.path.exists(newpath_flow):
            os.makedirs(newpath_flow)

        newpath_registered = join(self.output_folder, 'Registered', validation_folder_name)
        if not os.path.exists(newpath_registered):
            os.makedirs(newpath_registered)
        
        newpath_seg = join(self.output_folder, 'Segmentation', validation_folder_name)
        if not os.path.exists(newpath_seg):
            os.makedirs(newpath_seg)
        
        to_validate_registered_list = []

        list_of_keys = list(self.dataset_val.keys())
        un_list_of_keys = list(self.dataset_un_val.keys())
        patient_id_list = np.unique([x[:10] for x in list_of_keys])
        metadata_list = []
        metadata_list_registered = []
        for patient_id in tqdm(patient_id_list):
            all_keys = list_of_keys + un_list_of_keys
            phase_list = [x for x in all_keys if patient_id in x]
            phase_list = np.array(sorted(phase_list, key=lambda x: int(x[16:18])))
            phase_list = np.array(phase_list)
            global_labeled_idx = np.where(~np.char.endswith(phase_list, '_u'))[0]

            frame_indices = np.arange(len(phase_list))

            before_where = np.argwhere(frame_indices < global_labeled_idx[0]).reshape(-1,)
            after_where = np.argwhere(frame_indices >= global_labeled_idx[0]).reshape(-1,)
            assert after_where[0] == global_labeled_idx[0]

            before_str = phase_list[before_where]
            after_str = phase_list[after_where]

            all_where = np.concatenate([after_where, before_where])
            phase_list = np.concatenate([after_str, before_str])

            if '_u' in phase_list[0]:
                properties = load_pickle(self.unlabeled_dataset[phase_list[0]]['properties_file'])
            else:
                properties = load_pickle(self.dataset[phase_list[0]]['properties_file'])

            unlabeled = np.full(shape=((len(phase_list), 1) + properties['size_after_resampling']), fill_value=np.nan)
            target = np.full(shape=((len(phase_list), 1) + properties['size_after_resampling']), fill_value=np.nan)
            target_mask = np.zeros(shape=(len(phase_list),), dtype=bool)

            if overwrite or (not isfile(join(output_folder, phase_list[0] + ".nii.gz"))) or (save_softmax and not isfile(join(output_folder, phase_list[0] + ".npz"))):
                for idx, frame in enumerate(phase_list):
                    #if frame == k:
                    #    labeled_idx = idx
                    if '_u' in frame:
                        current_data = np.load(self.dataset_un_val[frame]['data_file'])['data']
                        current_data = current_data + 1e-8
                        unlabeled[idx] = current_data
                    else:
                        current_data = np.load(self.dataset_val[frame]['data_file'])['data']
                        #current_data[-1][current_data[-1] == -1] = 0
                        target[idx] = current_data[1]
                        target[idx][target[idx] == -1] = 0
                        unlabeled[idx] = current_data[0] + 1e-8
                        target_mask[idx] = True
                    #self.print_to_log_file(k, data.shape)
                
                #if self.step > 1:
                #    values = values[::self.step]
                assert target_mask[0] == True
                ret = self.predict_preprocessed_data_return_seg_and_softmax_flow(unlabeled=unlabeled,
                                                                                target=target,
                                                                                target_mask=target_mask,
                                                                                processor=processor,
                                                                                do_mirroring=do_mirroring,
                                                                                mirror_axes=mirror_axes,
                                                                                use_sliding_window=use_sliding_window,
                                                                                step_size=step_size,
                                                                                use_gaussian=use_gaussian,
                                                                                all_in_gpu=all_in_gpu,
                                                                                mixed_precision=self.fp16,
                                                                                verbose=False)
                flow_pred = ret[0] # T, C, depth, H, W
                registered_pred = ret[1] # T, C, depth, H, W

                assert len(flow_pred) == len(registered_pred)

                sorted_where = np.argsort(all_where)
                flow_pred = flow_pred[sorted_where]
                registered_pred = registered_pred[sorted_where]
                target_mask = target_mask[sorted_where]
                phase_list = phase_list[sorted_where]

                softmax_pred = np.zeros_like(registered_pred)

                #matplotlib.use('QtAgg')
                #fig, ax = plt.subplots(1, 7)
                #plot_idx = 0
                #for a in range(8, 15):
                #    current_img = softmax_pred[a, :, 0]
                #    current_img = current_img.argmax(0)
                #    ax[plot_idx].imshow(current_img, cmap='gray')
                #    plot_idx += 1
                #plt.show()
                #plt.waitforbuttonpress()
                #plt.close(fig)

                #stop_indices = [len(x) for x in windows]
                #videos = [phase_list[x] for x in windows]

                #matplotlib.use('QtAgg')
                #print(labeled_idx)
                #print(video_padding)
                #fig, ax = plt.subplots(1, self.video_length)
                #for i in range(self.video_length):
                #    ax[i].imshow(data[i, 0, 5], cmap='gray')
                #plt.show()

                assert len(flow_pred) == len(phase_list)

                and_mask = target_mask
                gt_indices = np.where(and_mask)[0]
                assert len(gt_indices) == 2

                for t in range(len(phase_list)):
                    if '_u' in phase_list[t]:
                        properties = load_pickle(self.unlabeled_dataset[phase_list[t]]['properties_file'])
                    else:
                        properties = load_pickle(self.dataset[phase_list[t]]['properties_file'])
                    fname = properties['list_of_data_files'][0].split(os.sep)[-1][:-12]

                    current_softmax_pred = softmax_pred[t]
                    current_softmax_pred = current_softmax_pred.transpose([0] + [i + 1 for i in self.transpose_backward])

                    if save_softmax:
                        softmax_fname = join(output_folder, fname + ".npz")
                    else:
                        softmax_fname = None

                    splitted = fname.split('frame')
                    to_nb = int(splitted[-1].split('_')[0].split('.')[0])
                    flow_name = splitted[0] + 'frame' + '01'.zfill(2) + '_to_' + str(to_nb).zfill(2)
                    flow_path = join(newpath_flow, flow_name + ".nii.gz")
                    current_flow = flow_pred[t]
                    current_flow = current_flow.transpose([0] + [i + 1 for i in self.transpose_backward])

                    registered_path = join(newpath_registered, fname + ".nii.gz")
                    current_registered = registered_pred[t]
                    current_registered = current_registered.transpose([0] + [i + 1 for i in self.transpose_backward])

                    """There is a problem with python process communication that prevents us from communicating objects
                    larger than 2 GB between processes (basically when the length of the pickle string that will be sent is
                    communicated by the multiprocessing.Pipe object then the placeholder (I think) does not allow for long
                    enough strings (lol). This could be fixed by changing i to l (for long) but that would require manually
                    patching system python code. We circumvent that problem here by saving softmax_pred to a npy file that will
                    then be read (and finally deleted) by the Process. save_segmentation_nifti_from_softmax can take either
                    filename or np.ndarray and will handle this automatically"""
                    if np.prod(current_softmax_pred.shape) > (2e9 / 4 * 0.85):  # *0.85 just to be save
                        np.save(join(newpath_seg, fname + ".npy"), current_softmax_pred)
                        current_softmax_pred = join(newpath_seg, fname + ".npy")

                    results.append(export_pool.starmap_async(save_segmentation_nifti_from_softmax,
                                                            ((current_softmax_pred, join(newpath_seg, fname + ".nii.gz"),
                                                            properties, interpolation_order, self.regions_class_order,
                                                            None, None,
                                                            softmax_fname, None, force_separate_z,
                                                            interpolation_order_z, False, current_flow, flow_path,
                                                            current_registered, registered_path),
                                                            )
                                                            )
                                )
                    
                    
                    if t == gt_indices[1]:
                        to_validate_registered_list.append(fname + ".nii.gz")
                        metadata_list_registered.append(self.create_metadata_dict(properties))
                        pred_gt_tuples_register.append([registered_path,
                                                join(self.gt_niftis_folder, fname + ".nii.gz")])

        _ = [i.get() for i in results]
        self.print_to_log_file("finished prediction")

        # evaluate raw predictions
        self.print_to_log_file("evaluation of raw predictions")
        task = self.dataset_directory.split(os.sep)[-1]
        job_name = self.experiment_name
        _ = aggregate_scores(pred_gt_tuples_register, labels=list(range(self.num_classes)),
                             json_output_file=join(newpath_registered, "summary.json"),
                             json_name=job_name + " val tiled %s" % (str(use_sliding_window)),
                             json_author="Fabian",
                             json_task=task, num_threads=default_num_threads,
                             advanced=True,
                             metadata_list=metadata_list_registered)

        if run_postprocessing_on_folds:
            # in the old nnunet we would stop here. Now we add a postprocessing. This postprocessing can remove everything
            # except the largest connected component for each class. To see if this improves results, we do this for all
            # classes and then rerun the evaluation. Those classes for which this resulted in an improved dice score will
            # have this applied during inference as well
            
            base_registered = join(self.output_folder, 'Registered')
            determine_postprocessing(base_registered, self.gt_niftis_folder, validation_folder_name,
                                     final_subf_name=validation_folder_name + "_postprocessed", debug=debug, log_function=log_function,
                                     metadata_list=metadata_list_registered, to_validate_list=to_validate_registered_list)
            # after this the final predictions for the vlaidation set can be found in validation_folder_name_base + "_postprocessed"
            # They are always in that folder, even if no postprocessing as applied!

        # detemining postprocesing on a per-fold basis may be OK for this fold but what if another fold finds another
        # postprocesing to be better? In this case we need to consolidate. At the time the consolidation is going to be
        # done we won't know what self.gt_niftis_folder was, so now we copy all the niftis into a separate folder to
        # be used later
        gt_nifti_folder = join(self.output_folder_base, "gt_niftis")
        maybe_mkdir_p(gt_nifti_folder)
        for f in subfiles(self.gt_niftis_folder, suffix=".nii.gz"):
            success = False
            attempts = 0
            e = None
            while not success and attempts < 10:
                try:
                    shutil.copy(f, gt_nifti_folder)
                    success = True
                except OSError as e:
                    attempts += 1
                    sleep(1)
            if not success:
                print("Could not copy gt nifti file %s into folder %s" % (f, gt_nifti_folder))
                if e is not None:
                    raise e

        self.network.train(current_mode)




    
    def validate_flow_one_step(self, processor, log_function, step, do_mirroring: bool = True, use_sliding_window: bool = True, step_size: float = 0.5,
                 save_softmax: bool = True, use_gaussian: bool = True, overwrite: bool = True,
                 validation_folder_name: str = 'validation_raw', debug: bool = True, all_in_gpu: bool = False,
                 segmentation_export_kwargs: dict = None, run_postprocessing_on_folds: bool = True, output_folder=None, save_flow=True):

        """
        if debug=True then the temporary files generated for postprocessing determination will be kept
        """
        current_mode = self.network.training
        self.network.eval()

        assert self.was_initialized, "must initialize, ideally with checkpoint (or train first)"
        if self.dataset_val is None:
            self.load_dataset()
            self.do_split()

        if segmentation_export_kwargs is None:
            if 'segmentation_export_params' in self.plans.keys():
                force_separate_z = self.plans['segmentation_export_params']['force_separate_z']
                interpolation_order = self.plans['segmentation_export_params']['interpolation_order']
                interpolation_order_z = self.plans['segmentation_export_params']['interpolation_order_z']
            else:
                force_separate_z = None
                interpolation_order = 1
                interpolation_order_z = 0
        else:
            force_separate_z = segmentation_export_kwargs['force_separate_z']
            interpolation_order = segmentation_export_kwargs['interpolation_order']
            interpolation_order_z = segmentation_export_kwargs['interpolation_order_z']

        # predictions as they come from the network go here
        if output_folder is not None:
            self.output_folder = output_folder
        #output_folder = join(self.output_folder, validation_folder_name)
        #maybe_mkdir_p(output_folder)
        # this is for debug purposes
        my_input_args = {'do_mirroring': do_mirroring,
                         'use_sliding_window': use_sliding_window,
                         'step_size': step_size,
                         'save_softmax': save_softmax,
                         'use_gaussian': use_gaussian,
                         'overwrite': overwrite,
                         'validation_folder_name': validation_folder_name,
                         'debug': debug,
                         'all_in_gpu': all_in_gpu,
                         'segmentation_export_kwargs': segmentation_export_kwargs,
                         }
        save_json(my_input_args, join(self.output_folder, "validation_args.json"))

        if do_mirroring:
            if not self.data_aug_params['do_mirror']:
                raise RuntimeError("We did not train with mirroring so you cannot do inference with mirroring enabled")
            mirror_axes = self.data_aug_params['mirror_axes']
        else:
            mirror_axes = ()

        pred_gt_tuples = []
        pred_gt_tuples_register = []

        export_pool = Pool(default_num_threads)
        results = []

        newpath_flow = join(self.output_folder, 'Flow', validation_folder_name)
        if not os.path.exists(newpath_flow):
            os.makedirs(newpath_flow)

        newpath_registered = join(self.output_folder, 'Registered', validation_folder_name)
        if not os.path.exists(newpath_registered):
            os.makedirs(newpath_registered)

        newpath_seg = join(self.output_folder, 'Segmentation', validation_folder_name)
        if not os.path.exists(newpath_seg):
            os.makedirs(newpath_seg)
        
        to_validate_registered_list = []

        list_of_keys = list(self.dataset_val.keys())
        un_list_of_keys = list(self.dataset_un_val.keys())
        patient_id_list = np.unique([x[:10] for x in list_of_keys])
        metadata_list = []
        metadata_list_registered = []
        for patient_id in tqdm(patient_id_list):
            all_keys = list_of_keys + un_list_of_keys
            phase_list = [x for x in all_keys if patient_id in x]
            phase_list = np.array(sorted(phase_list, key=lambda x: int(x[16:18])))
            phase_list = np.array(phase_list)
            global_labeled_idx = np.where(~np.char.endswith(phase_list, '_u'))[0]

            frame_indices = np.arange(len(phase_list))

            before_where = np.argwhere(frame_indices < global_labeled_idx[0]).reshape(-1,)
            after_where = np.argwhere(frame_indices >= global_labeled_idx[0]).reshape(-1,)
            assert after_where[0] == global_labeled_idx[0]

            before_str = phase_list[before_where]
            after_str = phase_list[after_where]

            all_where = np.concatenate([after_where, before_where])
            phase_list = np.concatenate([after_str, before_str])

            if '_u' in phase_list[0]:
                properties = load_pickle(self.unlabeled_dataset[phase_list[0]]['properties_file'])
            else:
                properties = load_pickle(self.dataset[phase_list[0]]['properties_file'])

            unlabeled = np.full(shape=((len(phase_list), 1) + properties['size_after_resampling']), fill_value=np.nan)
            target = np.full(shape=((len(phase_list), 1) + properties['size_after_resampling']), fill_value=np.nan)
            target_mask = np.zeros(shape=(len(phase_list),), dtype=bool)

            if overwrite or (not isfile(join(output_folder, phase_list[0] + ".nii.gz"))) or (save_softmax and not isfile(join(output_folder, phase_list[0] + ".npz"))):
                for idx, frame in enumerate(phase_list):
                    #if frame == k:
                    #    labeled_idx = idx
                    if '_u' in frame:
                        current_data = np.load(self.dataset_un_val[frame]['data_file'])['data']
                        current_data = current_data + 1e-8
                        unlabeled[idx] = current_data
                    else:
                        current_data = np.load(self.dataset_val[frame]['data_file'])['data']
                        #current_data[-1][current_data[-1] == -1] = 0
                        target[idx] = current_data[1]
                        target[idx][target[idx] == -1] = 0
                        unlabeled[idx] = current_data[0] + 1e-8
                        target_mask[idx] = True
                    #self.print_to_log_file(k, data.shape)
                
                #if self.step > 1:
                #    values = values[::self.step]
                assert target_mask[0] == True
                ret = self.predict_preprocessed_data_return_seg_and_softmax_flow(unlabeled=unlabeled,
                                                                                target=target,
                                                                                target_mask=target_mask,
                                                                                processor=processor,
                                                                                do_mirroring=do_mirroring,
                                                                                mirror_axes=mirror_axes,
                                                                                use_sliding_window=use_sliding_window,
                                                                                step_size=step_size,
                                                                                use_gaussian=use_gaussian,
                                                                                all_in_gpu=all_in_gpu,
                                                                                mixed_precision=self.fp16,
                                                                                verbose=False)
                predicted_segmentation = ret[0] # T, depth, H, W
                softmax_pred = ret[1] # T, C, depth, H, W
                flow_pred = ret[2] # T, C, depth, H, W
                registered_pred = ret[3] # T, C, depth, H, W

                assert len(softmax_pred) == len(flow_pred) == len(registered_pred)

                sorted_where = np.argsort(all_where)
                predicted_segmentation = predicted_segmentation[sorted_where]
                softmax_pred = softmax_pred[sorted_where]
                flow_pred = flow_pred[sorted_where]
                registered_pred = registered_pred[sorted_where]
                target_mask = target_mask[sorted_where]
                phase_list = phase_list[sorted_where]

                #matplotlib.use('QtAgg')
                #fig, ax = plt.subplots(1, 7)
                #plot_idx = 0
                #for a in range(8, 15):
                #    current_img = softmax_pred[a, :, 0]
                #    current_img = current_img.argmax(0)
                #    ax[plot_idx].imshow(current_img, cmap='gray')
                #    plot_idx += 1
                #plt.show()
                #plt.waitforbuttonpress()
                #plt.close(fig)

                #stop_indices = [len(x) for x in windows]
                #videos = [phase_list[x] for x in windows]

                #matplotlib.use('QtAgg')
                #print(labeled_idx)
                #print(video_padding)
                #fig, ax = plt.subplots(1, self.video_length)
                #for i in range(self.video_length):
                #    ax[i].imshow(data[i, 0, 5], cmap='gray')
                #plt.show()

                assert len(softmax_pred) == len(flow_pred) == len(phase_list)

                and_mask = target_mask
                gt_indices = np.where(and_mask)[0]
                assert len(gt_indices) == 2

                for t in range(len(softmax_pred)):
                    if '_u' in phase_list[t]:
                        properties = load_pickle(self.unlabeled_dataset[phase_list[t]]['properties_file'])
                    else:
                        properties = load_pickle(self.dataset[phase_list[t]]['properties_file'])
                    fname = properties['list_of_data_files'][0].split(os.sep)[-1][:-12]

                    current_softmax_pred = softmax_pred[t]
                    current_softmax_pred = current_softmax_pred.transpose([0] + [i + 1 for i in self.transpose_backward])

                    if save_softmax:
                        softmax_fname = join(output_folder, fname + ".npz")
                    else:
                        softmax_fname = None

                    splitted = fname.split('frame')
                    to_nb = int(splitted[-1].split('_')[0].split('.')[0])
                    flow_name = splitted[0] + 'frame' + '01'.zfill(2) + '_to_' + str(to_nb).zfill(2)
                    flow_path = join(newpath_flow, flow_name + ".nii.gz")
                    current_flow = flow_pred[t]
                    current_flow = current_flow.transpose([0] + [i + 1 for i in self.transpose_backward])

                    registered_path = join(newpath_registered, fname + ".nii.gz")
                    current_registered = registered_pred[t]
                    current_registered = current_registered.transpose([0] + [i + 1 for i in self.transpose_backward])

                    """There is a problem with python process communication that prevents us from communicating objects
                    larger than 2 GB between processes (basically when the length of the pickle string that will be sent is
                    communicated by the multiprocessing.Pipe object then the placeholder (I think) does not allow for long
                    enough strings (lol). This could be fixed by changing i to l (for long) but that would require manually
                    patching system python code. We circumvent that problem here by saving softmax_pred to a npy file that will
                    then be read (and finally deleted) by the Process. save_segmentation_nifti_from_softmax can take either
                    filename or np.ndarray and will handle this automatically"""
                    if np.prod(current_softmax_pred.shape) > (2e9 / 4 * 0.85):  # *0.85 just to be save
                        np.save(join(newpath_seg, fname + ".npy"), current_softmax_pred)
                        current_softmax_pred = join(newpath_seg, fname + ".npy")

                    results.append(export_pool.starmap_async(save_segmentation_nifti_from_softmax,
                                                            ((current_softmax_pred, join(newpath_seg, fname + ".nii.gz"),
                                                            properties, interpolation_order, self.regions_class_order,
                                                            None, None,
                                                            softmax_fname, None, force_separate_z,
                                                            interpolation_order_z, False, current_flow, flow_path,
                                                            current_registered, registered_path),
                                                            )
                                                            )
                                )
                    
                    
                    if t in gt_indices:
                        metadata_list.append(self.create_metadata_dict(properties))
                        pred_gt_tuples.append([join(newpath_seg, fname + ".nii.gz"),
                                                join(self.gt_niftis_folder, fname + ".nii.gz")])
                    if t == gt_indices[1]:
                        to_validate_registered_list.append(fname + ".nii.gz")
                        metadata_list_registered.append(self.create_metadata_dict(properties))
                        pred_gt_tuples_register.append([registered_path,
                                                join(self.gt_niftis_folder, fname + ".nii.gz")])

        _ = [i.get() for i in results]
        self.print_to_log_file("finished prediction")

        # evaluate raw predictions
        self.print_to_log_file("evaluation of raw predictions")
        task = self.dataset_directory.split(os.sep)[-1]
        job_name = self.experiment_name
        _ = aggregate_scores(pred_gt_tuples_register, labels=list(range(self.num_classes)),
                             json_output_file=join(newpath_registered, "summary.json"),
                             json_name=job_name + " val tiled %s" % (str(use_sliding_window)),
                             json_author="Fabian",
                             json_task=task, num_threads=default_num_threads,
                             advanced=True,
                             metadata_list=metadata_list_registered)
        _ = aggregate_scores(pred_gt_tuples, labels=list(range(self.num_classes)),
                             json_output_file=join(newpath_seg, "summary.json"),
                             json_name=job_name + " val tiled %s" % (str(use_sliding_window)),
                             json_author="Fabian",
                             json_task=task, num_threads=default_num_threads,
                             advanced=True,
                             metadata_list=metadata_list)

        if run_postprocessing_on_folds:
            # in the old nnunet we would stop here. Now we add a postprocessing. This postprocessing can remove everything
            # except the largest connected component for each class. To see if this improves results, we do this for all
            # classes and then rerun the evaluation. Those classes for which this resulted in an improved dice score will
            # have this applied during inference as well
            self.print_to_log_file("determining postprocessing")
            base_segmentation = join(self.output_folder, 'Segmentation')
            determine_postprocessing(base_segmentation, self.gt_niftis_folder, validation_folder_name,
                                     final_subf_name=validation_folder_name + "_postprocessed", debug=debug, log_function=log_function,
                                     metadata_list=metadata_list, to_validate_list=None)
            
            base_registered = join(self.output_folder, 'Registered')
            determine_postprocessing(base_registered, self.gt_niftis_folder, validation_folder_name,
                                     final_subf_name=validation_folder_name + "_postprocessed", debug=debug, log_function=log_function,
                                     metadata_list=metadata_list_registered, to_validate_list=to_validate_registered_list)
            # after this the final predictions for the vlaidation set can be found in validation_folder_name_base + "_postprocessed"
            # They are always in that folder, even if no postprocessing as applied!

        # detemining postprocesing on a per-fold basis may be OK for this fold but what if another fold finds another
        # postprocesing to be better? In this case we need to consolidate. At the time the consolidation is going to be
        # done we won't know what self.gt_niftis_folder was, so now we copy all the niftis into a separate folder to
        # be used later
        gt_nifti_folder = join(self.output_folder_base, "gt_niftis")
        maybe_mkdir_p(gt_nifti_folder)
        for f in subfiles(self.gt_niftis_folder, suffix=".nii.gz"):
            success = False
            attempts = 0
            e = None
            while not success and attempts < 10:
                try:
                    shutil.copy(f, gt_nifti_folder)
                    success = True
                except OSError as e:
                    attempts += 1
                    sleep(1)
            if not success:
                print("Could not copy gt nifti file %s into folder %s" % (f, gt_nifti_folder))
                if e is not None:
                    raise e

        self.network.train(current_mode)

    
    #def save_strain_compute_metric(self, search_path):
    #    newpath_strain_lv_tangential = join(search_path, 'Strain', 'LV', 'Tangential')
    #    if not os.path.exists(newpath_strain_lv_tangential):
    #        os.makedirs(newpath_strain_lv_tangential)
#
    #    results = {"all": [], "mean_lv_tangential": None}
    #    gt_folder_name = os.path.dirname(self.gt_niftis_folder)
    #    all_files = glob(os.path.join(search_path, '*.gz'))
    #    patient_names = list(set([os.path.basename(x).split('_')[0] for x in all_files]))
    #    for patient_name in patient_names:
    #        phase_list = [x for x in all_files if patient_name in x]
    #        phase_list = np.array(sorted(phase_list, key=lambda x: int(os.path.basename(x).split('frame')[-1][:2])))
    #        video = []
    #        for phase in phase_list:
    #            data = nib.load(phase)
    #            arr = data.get_fdata() # H, W, D
    #            arr = arr.transpose((2, 0, 1)) # D, H, W
    #            video.append(arr)
    #        video = np.stack(video, axis=1) # D, T, H, W
    #        for d in range(len(video)):
    #            slice_nb = str(d + 1).zfill(2)
    #            filename = patient_name + '_slice' + slice_nb + '.npy'
    #            gt_path = os.path.join(gt_folder_name, 'strain', 'LV', 'tangential', filename)
    #            rv_tangential_strain, lv_tangential_strain = self.get_strain(video[d])
    #            rv_tangential_strain = rv_tangential_strain * 100
    #            lv_tangential_strain = lv_tangential_strain * 100
    #            lv_tangential_strain_gt = np.load(gt_path)[0]
    #            current_res = {'reference': gt_path, 'test': phase_list[0][:-15]}
    #            current_res['lv_tangential'] = (lv_tangential_strain - lv_tangential_strain_gt).tolist()
    #            results['all'].append(current_res)
    #            np.save(os.path.join(search_path, 'Strain', 'LV', 'Tangential', filename), lv_tangential_strain)
#
    #    results['mean_lv_tangential'] = np.concatenate([np.array(x['lv_tangential']) for x in results['all']]).mean()
    #    save_json(results, os.path.join(search_path, 'strain_summary.json'))
    


    #def save_strain_compute_metric_bottleneck(self, save_path, patient_id, strain_results, lv_tangential_strain):
    #    lv_tangential_strain = lv_tangential_strain.transpose((1, 0)) # D, T
    #    gt_folder_name = os.path.dirname(self.gt_niftis_folder)
    #    for d in range(len(lv_tangential_strain)):
    #        current_strain = lv_tangential_strain[d] * 100
    #        slice_nb = str(d + 1).zfill(2)
    #        filename = patient_id + '_slice' + slice_nb + '.npy'
    #        gt_path = os.path.join(gt_folder_name, 'strain', 'LV', 'tangential', filename)
    #        lv_tangential_strain_gt = np.load(gt_path)[0]
    #        current_res = {'reference': gt_path, 'test': filename}
#
    #        #a, b = self.get_strain(target[:, 0, d, :, :])
    #        current_res['lv_tangential'] = (current_strain - lv_tangential_strain_gt).tolist()
    #        strain_results['all'].append(current_res)
    #        np.save(os.path.join(save_path, filename), current_strain)
    #    return strain_results

    
    def compute_contour_metric_threads(self, search_path):
        p = Pool(default_num_threads)

        results = {"all": [], "mean": None}
        all_files = glob(os.path.join(search_path, '*.npz'))
        patient_names = list(set([os.path.basename(x).split('_')[0] for x in all_files]))
        gt_folder_name = os.path.dirname(self.gt_niftis_folder)

        all_res = p.starmap(compute_contour_metric, zip(patient_names, [all_files]*len(patient_names), [gt_folder_name]*len(patient_names)))
        p.close()
        p.join()

        for i in range(len(all_res)):
            results['all'].extend(all_res[i]['all'])

        current_res_info = {'ENDO_mae': np.concatenate([np.array(x['ENDO_mae']) for x in results['all']]).mean(),
                            'EPI_mae': np.concatenate([np.array(x['EPI_mae']) for x in results['all']]).mean(),
                            'RV_mae': np.concatenate([np.array(x['RV_mae']) for x in results['all']]).mean()}
        results['mean'] = current_res_info
        save_json(results, os.path.join(search_path, 'summary.json'))




    def save_strain_compute_metric_threads(self, search_path):
        p = Pool(default_num_threads)

        newpath_strain_lv_tangential = join(search_path, 'Strain', 'LV', 'Tangential')
        if not os.path.exists(newpath_strain_lv_tangential):
            os.makedirs(newpath_strain_lv_tangential)

        results = {"all": [], "mean_lv_tangential": None}
        gt_folder_name = os.path.dirname(self.gt_niftis_folder)
        all_files = glob(os.path.join(search_path, '*.gz'))
        patient_names = list(set([os.path.basename(x).split('_')[0] for x in all_files]))

        all_res = p.starmap(save_strain_compute_metric, zip(patient_names, [all_files]*len(patient_names), [gt_folder_name]*len(patient_names), [search_path]*len(patient_names)))
        p.close()
        p.join()

        for i in range(len(all_res)):
            results['all'].extend(all_res[i]['all'])

        results['mean_lv_tangential'] = np.concatenate([np.array(x['lv_tangential']) for x in results['all']]).mean()
        save_json(results, os.path.join(search_path, 'strain_summary.json'))



    def strain_compute_metric_bottleneck_threads(self, search_path):
        p = Pool(default_num_threads)

        results = {"all": [], "mean_lv_tangential": None}
        gt_folder_name = os.path.dirname(self.gt_niftis_folder)
        all_files = glob(os.path.join(search_path, '*.npy'))
        patient_names = list(set([os.path.basename(x).split('_')[0] for x in all_files]))

        all_res = p.starmap(strain_compute_metric, zip(patient_names, [all_files]*len(patient_names), [gt_folder_name]*len(patient_names)))
        p.close()
        p.join()

        for i in range(len(all_res)):
            results['all'].extend(all_res[i]['all'])

        results['mean_lv_tangential'] = np.concatenate([np.array(x['lv_tangential']) for x in results['all']]).mean()
        save_json(results, os.path.join(search_path, 'strain_summary.json'))


    def move_gt_files_threads(self):
        p = Pool(default_num_threads)

        gt_nifti_folder = join(self.output_folder_base, "gt_niftis")
        maybe_mkdir_p(gt_nifti_folder)
        path_list = subfiles(self.gt_niftis_folder, suffix=".nii.gz")

        p.starmap(move_gt_files, zip(path_list, [gt_nifti_folder]*len(path_list)))
        p.close()
        p.join()


    #def compute_contour_metric(self, search_path):
    #    results = {"all": [], "mean": None}
    #    all_files = glob(os.path.join(search_path, '*.npz'))
    #    patient_names = list(set([os.path.basename(x).split('_')[0] for x in all_files]))
    #    gt_folder_name = os.path.dirname(self.gt_niftis_folder)
    #    for patient_name in patient_names:
    #        phase_list = [x for x in all_files if patient_name in x]
    #        phase_list = np.array(sorted(phase_list, key=lambda x: int(os.path.basename(x).split('frame')[-1][:2])))
    #        video = []
    #        for phase in phase_list:
    #            data = np.load(phase)
    #            arr = data['flow'] # H, W, D, C
    #            arr = arr.transpose((2, 3, 0, 1)) # D, C, H, W
    #            video.append(arr)
    #        flow = np.stack(video, axis=1) # D, T, C, H, W
    #        slice_error_list = []
    #        for d in range(len(flow)):
    #            current_slice_flow = flow[d] # T, 2, H, W
    #            slice_nb = str(d + 1).zfill(2)
    #            filename = patient_name + '_slice' + slice_nb + '.npy'
    #            gt_path_lv = os.path.join(gt_folder_name, 'contour', 'LV', filename)
    #            gt_path_rv = os.path.join(gt_folder_name, 'contour', 'RV', filename)
    #            gt_lv_contour = np.load(gt_path_lv).transpose((2, 1, 0)) # T, P1, 4
    #            gt_rv_contour = np.load(gt_path_rv).transpose((2, 1, 0)) # T, P2, 2
    #            gt_endo_contour = gt_lv_contour[:, :, :2]
    #            gt_epi_contour = gt_lv_contour[:, :, 2:]
    #            split_index = np.cumsum([gt_endo_contour.shape[1], gt_epi_contour.shape[1]])
    #            contours = np.concatenate([gt_endo_contour, gt_epi_contour, gt_rv_contour], axis=1) # T, P, 2
#
    #            current_slice_flow = np.flip(current_slice_flow, axis=0)
    #            contours = np.flip(contours, axis=0)
#
    #            temporal_error_list = []
    #            for t in range(len(current_slice_flow) - 1):
    #                current_contours = contours[t] # P, 2
    #                next_contours = contours[t + 1] # P, 2
    #                gt_delta = next_contours - current_contours
    #                current_frame_flow = current_slice_flow[t] # 2, H, W
    #                current_frame_flow = current_frame_flow.transpose((1, 2, 0)) # H, W, 2
    #                y = np.rint(current_contours[:, 0]).astype(int)
    #                x = np.rint(current_contours[:, 1]).astype(int)
#
    #                delta_pred = current_frame_flow[y, x, :] # P, 2
    #                error = np.abs(gt_delta - delta_pred).mean(-1) # P,
    #                temporal_error_list.append(error) 
#
    #            temporal_error_list = np.stack(temporal_error_list, axis=0) # T, P
    #            temporal_error_list = np.flip(temporal_error_list, axis=0)
#
    #            error = np.split(temporal_error_list, indices_or_sections=split_index, axis=1) # [(T, P1) , (T, P2), (T, P3)]
    #            error = np.stack([x.mean(-1) for x in error], axis=-1) # T, 3
#
    #            slice_error_list.append(error)
    #        slice_error_list = np.stack(slice_error_list, axis=0) # D, T, 3
#
    #        for i in range(slice_error_list.shape[1]):
    #            current_res = slice_error_list[:, i] # D, 3
    #            current_res_info = {'patient_name': patient_name, 
    #                                'reference': gt_path_lv[:-12], 
    #                                'test': phase_list[i + 1],
    #                                'ENDO_mae': current_res[:, 0].tolist(),
    #                                'EPI_mae': current_res[:, 1].tolist(),
    #                                'RV_mae': current_res[:, 2].tolist()}
    #            results['all'].append(current_res_info)
#
    #    current_res_info = {'ENDO_mae': np.concatenate([np.array(x['ENDO_mae']) for x in results['all']]).mean(),
    #                        'EPI_mae': np.concatenate([np.array(x['EPI_mae']) for x in results['all']]).mean(),
    #                        'RV_mae': np.concatenate([np.array(x['RV_mae']) for x in results['all']]).mean()}
    #    results['mean'] = current_res_info
    #    save_json(results, os.path.join(search_path, 'summary.json'))


    def validate_flow_one_step_lib(self, processor, log_function, step, do_mirroring: bool = True, use_sliding_window: bool = True, step_size: float = 0.5,
                 save_softmax: bool = True, use_gaussian: bool = True, overwrite: bool = True,
                 validation_folder_name: str = 'validation_raw', debug: bool = True, all_in_gpu: bool = False,
                 segmentation_export_kwargs: dict = None, run_postprocessing_on_folds: bool = True, output_folder=None, save_flow=True):

        """
        if debug=True then the temporary files generated for postprocessing determination will be kept
        """
        current_mode = self.network.training
        self.network.eval()

        assert self.was_initialized, "must initialize, ideally with checkpoint (or train first)"
        if self.dataset_val is None:
            self.load_dataset()
            self.do_split()

        if segmentation_export_kwargs is None:
            if 'segmentation_export_params' in self.plans.keys():
                force_separate_z = self.plans['segmentation_export_params']['force_separate_z']
                interpolation_order = self.plans['segmentation_export_params']['interpolation_order']
                interpolation_order_z = self.plans['segmentation_export_params']['interpolation_order_z']
            else:
                force_separate_z = None
                interpolation_order = 1
                interpolation_order_z = 0
        else:
            force_separate_z = segmentation_export_kwargs['force_separate_z']
            interpolation_order = segmentation_export_kwargs['interpolation_order']
            interpolation_order_z = segmentation_export_kwargs['interpolation_order_z']

        # predictions as they come from the network go here
        if output_folder is not None:
            self.output_folder = output_folder
        #output_folder = join(self.output_folder, validation_folder_name)
        #maybe_mkdir_p(output_folder)
        # this is for debug purposes
        my_input_args = {'do_mirroring': do_mirroring,
                         'use_sliding_window': use_sliding_window,
                         'step_size': step_size,
                         'save_softmax': save_softmax,
                         'use_gaussian': use_gaussian,
                         'overwrite': overwrite,
                         'validation_folder_name': validation_folder_name,
                         'debug': debug,
                         'all_in_gpu': all_in_gpu,
                         'segmentation_export_kwargs': segmentation_export_kwargs,
                         }
        save_json(my_input_args, join(self.output_folder, "validation_args.json"))

        if do_mirroring:
            if not self.data_aug_params['do_mirror']:
                raise RuntimeError("We did not train with mirroring so you cannot do inference with mirroring enabled")
            mirror_axes = self.data_aug_params['mirror_axes']
        else:
            mirror_axes = ()

        pred_gt_tuples = []
        pred_gt_tuples_register = []

        export_pool = Pool(default_num_threads)
        results = []
        results_strain_list = []

        newpath_flow = join(self.output_folder, 'Flow', validation_folder_name)
        if not os.path.exists(newpath_flow):
            os.makedirs(newpath_flow)

        newpath_registered = join(self.output_folder, 'Registered', validation_folder_name)
        if not os.path.exists(newpath_registered):
            os.makedirs(newpath_registered)

        newpath_seg = join(self.output_folder, 'Segmentation', validation_folder_name)
        if not os.path.exists(newpath_seg):
            os.makedirs(newpath_seg)

        newpath_strain = join(self.output_folder, 'Strain')
        newpath_strain_lv_tangential = join(newpath_strain, 'LV', 'Tangential')
        if not os.path.exists(newpath_strain_lv_tangential):
            os.makedirs(newpath_strain_lv_tangential)

        #newpath_strain_radial = join(self.output_folder, 'Strain', 'Radial')
        #if not os.path.exists(newpath_strain_radial):
        #    os.makedirs(newpath_strain_radial)
        
        to_validate_registered_list = []

        list_of_keys = list(self.dataset_val.keys())
        patient_id_list = np.unique([x[:10] for x in list_of_keys])
        metadata_list = []
        metadata_list_registered = []
        #strain_results = {"all": [], "mean_lv_tangential": None}
        for patient_id in tqdm(patient_id_list):
            phase_list = [x for x in list_of_keys if patient_id in x]
            phase_list = np.array(sorted(phase_list, key=lambda x: int(x[16:18])))
            phase_list = np.array(phase_list)
            #print(phase_list[0])
            #print(self.dataset)
            #print(self.dataset[phase_list[0]])
            properties = load_pickle(self.dataset[phase_list[0]]['properties_file'])

            #ed_indices = np.array(properties['ed_number']) - 1
            #es_indices = np.array(properties['es_number']) - 1

            unlabeled = np.full(shape=((len(phase_list), 1) + properties['size_after_resampling']), fill_value=np.nan)
            target = np.full(shape=((len(phase_list), 1) + properties['size_after_resampling']), fill_value=np.nan)

            if overwrite or (not isfile(join(output_folder, phase_list[0] + ".nii.gz"))) or (save_softmax and not isfile(join(output_folder, phase_list[0] + ".npz"))):
                for idx, frame in enumerate(phase_list):
                    current_data = np.load(self.dataset_val[frame]['data_file'])['data']
                    target[idx] = current_data[1]
                    target[idx][target[idx] == -1] = 0
                    unlabeled[idx] = current_data[0] + 1e-8

                #all_where_list = []
                #print(ed_indices)
                #print(unlabeled.shape[2])
                #print(phase_list)
                #assert len(ed_indices) == unlabeled.shape[2]
                #for d in range(unlabeled.shape[2]):
                #    frame_indices = np.arange(len(phase_list))
#
                #    before_where = np.argwhere(frame_indices < ed_indices[d]).reshape(-1,)
                #    after_where = np.argwhere(frame_indices >= ed_indices[d]).reshape(-1,)
#
                #    all_where = np.concatenate([after_where, before_where])
                #    all_where_list.append(all_where)
#
                #    frame_indices = frame_indices[all_where]
                #    unlabeled[:, :, d] = unlabeled[frame_indices, :, d]


                #matplotlib.use('QtAgg')
                #print(ed_indices)
                #print(es_indices)
                #fig, ax = plt.subplots(1, int(len(unlabeled) / 4))
                #for i in range(0, len(unlabeled), 4):
                #    ax[int(i / 4)].imshow(unlabeled[i, 0, 1], cmap='gray')
                #plt.show()
                
                
                ret = self.predict_preprocessed_data_return_seg_and_softmax_flow(unlabeled=unlabeled,
                                                                                target=target,
                                                                                target_mask=None,
                                                                                processor=processor,
                                                                                do_mirroring=do_mirroring,
                                                                                mirror_axes=mirror_axes,
                                                                                use_sliding_window=use_sliding_window,
                                                                                step_size=step_size,
                                                                                use_gaussian=use_gaussian,
                                                                                all_in_gpu=all_in_gpu,
                                                                                mixed_precision=self.fp16,
                                                                                verbose=False)
                predicted_segmentation = ret[0] # T, depth, H, W
                softmax_pred = ret[1] # T, C, depth, H, W
                flow_pred = ret[2] # T, C, depth, H, W
                registered_pred = ret[3] # T, C, depth, H, W
                lv_strain_bottleneck_pred = ret[4] # T, depth
                rv_strain_bottleneck_pred = ret[5] # T, depth

                assert len(softmax_pred) == len(flow_pred) == len(registered_pred) == len(lv_strain_bottleneck_pred)

                #for d, all_where in enumerate(all_where_list):
                #    sorted_where = np.argsort(all_where)
                #    print(all_where)
                #    print(sorted_where)
                #    softmax_pred[:, :, d] = softmax_pred[sorted_where, :, d]
                #    flow_pred[:, :, d] = flow_pred[sorted_where, :, d]
                #    registered_pred[:, :, d] = registered_pred[sorted_where, :, d]
                #    lv_strain_pred[:, d] = lv_strain_pred[sorted_where, d]
                #    rv_strain_pred[:, d] = rv_strain_pred[sorted_where, d]
                #    phase_list[:, :, d] = phase_list[sorted_where, :, d]

                assert len(softmax_pred) == len(flow_pred) == len(phase_list)

                results_strain_list.append(export_pool.starmap_async(save_strain_bottleneck, ((newpath_strain_lv_tangential, 
                                                                            patient_id,
                                                                            lv_strain_bottleneck_pred),)))

                for t in range(len(softmax_pred)):
                    properties = load_pickle(self.dataset[phase_list[t]]['properties_file'])
                    fname = properties['list_of_data_files'][0].split(os.sep)[-1][:-12]

                    current_softmax_pred = softmax_pred[t]
                    current_softmax_pred = current_softmax_pred.transpose([0] + [i + 1 for i in self.transpose_backward])

                    if save_softmax:
                        softmax_fname = join(output_folder, fname + ".npz")
                    else:
                        softmax_fname = None

                    splitted = fname.split('frame')
                    from_nb = t - 1 if t > 0 else 0
                    to_nb = t
                    flow_name = splitted[0] + 'frame' + str(from_nb + 1).zfill(2) + '_to_' + str(to_nb + 1).zfill(2)
                    flow_path = join(newpath_flow, flow_name + ".npz")
                    current_flow = flow_pred[t]
                    current_flow = current_flow.transpose([0] + [i + 1 for i in self.transpose_backward])

                    registered_path = join(newpath_registered, fname + ".nii.gz")
                    current_registered = registered_pred[t]
                    current_registered = current_registered.transpose([0] + [i + 1 for i in self.transpose_backward])

                    """There is a problem with python process communication that prevents us from communicating objects
                    larger than 2 GB between processes (basically when the length of the pickle string that will be sent is
                    communicated by the multiprocessing.Pipe object then the placeholder (I think) does not allow for long
                    enough strings (lol). This could be fixed by changing i to l (for long) but that would require manually
                    patching system python code. We circumvent that problem here by saving softmax_pred to a npy file that will
                    then be read (and finally deleted) by the Process. save_segmentation_nifti_from_softmax can take either
                    filename or np.ndarray and will handle this automatically"""
                    if np.prod(current_softmax_pred.shape) > (2e9 / 4 * 0.85):  # *0.85 just to be save
                        np.save(join(newpath_seg, fname + ".npy"), current_softmax_pred)
                        current_softmax_pred = join(newpath_seg, fname + ".npy")

                    results.append(export_pool.starmap_async(save_segmentation_nifti_from_softmax,
                                                            ((current_softmax_pred, join(newpath_seg, fname + ".nii.gz"),
                                                            properties, interpolation_order, self.regions_class_order,
                                                            None, None,
                                                            softmax_fname, None, force_separate_z,
                                                            interpolation_order_z, False, current_flow, flow_path,
                                                            current_registered, registered_path),
                                                            )
                                                            )
                                )
                    
                    
                    metadata_list.append(self.create_metadata_dict(properties))
                    pred_gt_tuples.append([join(newpath_seg, fname + ".nii.gz"),
                                            join(self.gt_niftis_folder, fname + ".nii.gz")])
                    if t > 0:
                        to_validate_registered_list.append(fname + ".nii.gz")
                        metadata_list_registered.append(self.create_metadata_dict(properties))
                        pred_gt_tuples_register.append([registered_path,
                                                join(self.gt_niftis_folder, fname + ".nii.gz")])

        _ = [i.get() for i in results]
        _ = [i.get() for i in results_strain_list]
        self.print_to_log_file("finished prediction")
        #strain_results['mean_lv_tangential'] = np.concatenate([np.array(x['lv_tangential']) for x in strain_results['all']]).mean()
        #save_json(strain_results, os.path.join(newpath_strain, 'summary.json'))

        # evaluate raw predictions
        #self.print_to_log_file("evaluation of raw predictions")
        #task = self.dataset_directory.split(os.sep)[-1]
        #job_name = self.experiment_name
        #_ = aggregate_scores(pred_gt_tuples_register, labels=list(range(self.num_classes)),
        #                     json_output_file=join(newpath_registered, "summary.json"),
        #                     json_name=job_name + " val tiled %s" % (str(use_sliding_window)),
        #                     json_author="Fabian",
        #                     json_task=task, num_threads=default_num_threads,
        #                     advanced=True,
        #                     metadata_list=metadata_list_registered)
        #_ = aggregate_scores(pred_gt_tuples, labels=list(range(self.num_classes)),
        #                     json_output_file=join(newpath_seg, "summary.json"),
        #                     json_name=job_name + " val tiled %s" % (str(use_sliding_window)),
        #                     json_author="Fabian",
        #                     json_task=task, num_threads=default_num_threads,
        #                     advanced=True,
        #                     metadata_list=metadata_list)

        if run_postprocessing_on_folds:
            # in the old nnunet we would stop here. Now we add a postprocessing. This postprocessing can remove everything
            # except the largest connected component for each class. To see if this improves results, we do this for all
            # classes and then rerun the evaluation. Those classes for which this resulted in an improved dice score will
            # have this applied during inference as well
            self.print_to_log_file("determining postprocessing")
            base_segmentation = join(self.output_folder, 'Segmentation')
            determine_postprocessing_no_metric(base_segmentation, self.gt_niftis_folder, validation_folder_name,
                                     final_subf_name=validation_folder_name + "_postprocessed", debug=debug, log_function=log_function,
                                     metadata_list=metadata_list, to_validate_list=None)
            
            base_registered = join(self.output_folder, 'Registered')
            determine_postprocessing_no_metric(base_registered, self.gt_niftis_folder, validation_folder_name,
                                     final_subf_name=validation_folder_name + "_postprocessed", debug=debug, log_function=log_function,
                                     metadata_list=metadata_list_registered, to_validate_list=to_validate_registered_list)
            # after this the final predictions for the vlaidation set can be found in validation_folder_name_base + "_postprocessed"
            # They are always in that folder, even if no postprocessing as applied!


        # detemining postprocesing on a per-fold basis may be OK for this fold but what if another fold finds another
        # postprocesing to be better? In this case we need to consolidate. At the time the consolidation is going to be
        # done we won't know what self.gt_niftis_folder was, so now we copy all the niftis into a separate folder to
        # be used later
        self.print_to_log_file("Moving ground truth files")
        self.move_gt_files_threads()
        #gt_nifti_folder = join(self.output_folder_base, "gt_niftis")
        #maybe_mkdir_p(gt_nifti_folder)
        #for f in subfiles(self.gt_niftis_folder, suffix=".nii.gz"):
        #    success = False
        #    attempts = 0
        #    e = None
        #    while not success and attempts < 10:
        #        try:
        #            shutil.copy(f, gt_nifti_folder)
        #            success = True
        #        except OSError as e:
        #            attempts += 1
        #            sleep(1)
        #    if not success:
        #        print("Could not copy gt nifti file %s into folder %s" % (f, gt_nifti_folder))
        #        if e is not None:
        #            raise e

        self.network.train(current_mode)
    



    def validate_flow_one_step_recursive(self, processor, log_function, step, do_mirroring: bool = True, use_sliding_window: bool = True, step_size: float = 0.5,
                 save_softmax: bool = True, use_gaussian: bool = True, overwrite: bool = True,
                 validation_folder_name: str = 'validation_raw', debug: bool = True, all_in_gpu: bool = False,
                 segmentation_export_kwargs: dict = None, run_postprocessing_on_folds: bool = True, output_folder=None, save_flow=True):

        """
        if debug=True then the temporary files generated for postprocessing determination will be kept
        """
        current_mode = self.network.training
        self.network.eval()

        assert self.was_initialized, "must initialize, ideally with checkpoint (or train first)"
        if self.dataset_val is None:
            self.load_dataset()
            self.do_split()

        if segmentation_export_kwargs is None:
            if 'segmentation_export_params' in self.plans.keys():
                force_separate_z = self.plans['segmentation_export_params']['force_separate_z']
                interpolation_order = self.plans['segmentation_export_params']['interpolation_order']
                interpolation_order_z = self.plans['segmentation_export_params']['interpolation_order_z']
            else:
                force_separate_z = None
                interpolation_order = 1
                interpolation_order_z = 0
        else:
            force_separate_z = segmentation_export_kwargs['force_separate_z']
            interpolation_order = segmentation_export_kwargs['interpolation_order']
            interpolation_order_z = segmentation_export_kwargs['interpolation_order_z']

        # predictions as they come from the network go here
        if output_folder is not None:
            self.output_folder = output_folder
        #output_folder = join(self.output_folder, validation_folder_name)
        #maybe_mkdir_p(output_folder)
        # this is for debug purposes
        my_input_args = {'do_mirroring': do_mirroring,
                         'use_sliding_window': use_sliding_window,
                         'step_size': step_size,
                         'save_softmax': save_softmax,
                         'use_gaussian': use_gaussian,
                         'overwrite': overwrite,
                         'validation_folder_name': validation_folder_name,
                         'debug': debug,
                         'all_in_gpu': all_in_gpu,
                         'segmentation_export_kwargs': segmentation_export_kwargs,
                         }
        save_json(my_input_args, join(self.output_folder, "validation_args.json"))

        if do_mirroring:
            if not self.data_aug_params['do_mirror']:
                raise RuntimeError("We did not train with mirroring so you cannot do inference with mirroring enabled")
            mirror_axes = self.data_aug_params['mirror_axes']
        else:
            mirror_axes = ()

        pred_gt_tuples = []
        pred_gt_tuples_register_local = []
        pred_gt_tuples_register_global = []

        export_pool = Pool(default_num_threads)
        results = []

        newpath_flow = join(self.output_folder, 'Flow', validation_folder_name)
        if not os.path.exists(newpath_flow):
            os.makedirs(newpath_flow)

        newpath_registered_local = join(self.output_folder, 'Registered_local', validation_folder_name)
        if not os.path.exists(newpath_registered_local):
            os.makedirs(newpath_registered_local)

        newpath_registered_global = join(self.output_folder, 'Registered_global', validation_folder_name)
        if not os.path.exists(newpath_registered_global):
            os.makedirs(newpath_registered_global)

        newpath_seg = join(self.output_folder, 'Segmentation', validation_folder_name)
        if not os.path.exists(newpath_seg):
            os.makedirs(newpath_seg)
        
        to_validate_registered_list = []

        list_of_keys = list(self.dataset_val.keys())
        un_list_of_keys = list(self.dataset_un_val.keys())
        patient_id_list = np.unique([x[:10] for x in list_of_keys])
        metadata_list = []
        metadata_list_registered = []
        for patient_id in tqdm(patient_id_list):
            all_keys = list_of_keys + un_list_of_keys
            phase_list = [x for x in all_keys if patient_id in x]
            phase_list = np.array(sorted(phase_list, key=lambda x: int(x[16:18])))
            phase_list = np.array(phase_list)
            global_labeled_idx = np.where(~np.char.endswith(phase_list, '_u'))[0]

            frame_indices = np.arange(len(phase_list))

            before_where = np.argwhere(frame_indices < global_labeled_idx[0]).reshape(-1,)
            after_where = np.argwhere(frame_indices >= global_labeled_idx[0]).reshape(-1,)
            assert after_where[0] == global_labeled_idx[0]

            before_str = phase_list[before_where]
            after_str = phase_list[after_where]

            all_where = np.concatenate([after_where, before_where])
            phase_list = np.concatenate([after_str, before_str])

            if '_u' in phase_list[0]:
                properties = load_pickle(self.unlabeled_dataset[phase_list[0]]['properties_file'])
            else:
                properties = load_pickle(self.dataset[phase_list[0]]['properties_file'])

            unlabeled = np.full(shape=((len(phase_list), 1) + properties['size_after_resampling']), fill_value=np.nan)
            target = np.full(shape=((len(phase_list), 1) + properties['size_after_resampling']), fill_value=np.nan)
            target_mask = np.zeros(shape=(len(phase_list),), dtype=bool)

            if overwrite or (not isfile(join(output_folder, phase_list[0] + ".nii.gz"))) or (save_softmax and not isfile(join(output_folder, phase_list[0] + ".npz"))):
                for idx, frame in enumerate(phase_list):
                    #if frame == k:
                    #    labeled_idx = idx
                    if '_u' in frame:
                        current_data = np.load(self.dataset_un_val[frame]['data_file'])['data']
                        current_data = current_data + 1e-8
                        unlabeled[idx] = current_data
                    else:
                        current_data = np.load(self.dataset_val[frame]['data_file'])['data']
                        #current_data[-1][current_data[-1] == -1] = 0
                        target[idx] = current_data[1]
                        target[idx][target[idx] == -1] = 0
                        unlabeled[idx] = current_data[0] + 1e-8
                        target_mask[idx] = True
                    #self.print_to_log_file(k, data.shape)
                
                #if self.step > 1:
                #    values = values[::self.step]
                assert target_mask[0] == True
                ret = self.predict_preprocessed_data_return_seg_and_softmax_flow(unlabeled=unlabeled,
                                                                                target=target,
                                                                                target_mask=target_mask,
                                                                                processor=processor,
                                                                                do_mirroring=do_mirroring,
                                                                                mirror_axes=mirror_axes,
                                                                                use_sliding_window=use_sliding_window,
                                                                                step_size=step_size,
                                                                                use_gaussian=use_gaussian,
                                                                                all_in_gpu=all_in_gpu,
                                                                                mixed_precision=self.fp16,
                                                                                verbose=False)
                predicted_segmentation = ret[0] # T, depth, H, W
                softmax_pred = ret[1] # T, C, depth, H, W
                flow_pred = ret[2] # T, C, depth, H, W
                registered_pred_local = ret[3] # T, C, depth, H, W
                registered_pred_global = ret[4] # T, C, depth, H, W

                assert len(softmax_pred) == len(flow_pred) == len(registered_pred_local) == len(registered_pred_global)

                sorted_where = np.argsort(all_where)
                predicted_segmentation = predicted_segmentation[sorted_where]
                softmax_pred = softmax_pred[sorted_where]
                flow_pred = flow_pred[sorted_where]
                registered_pred_local = registered_pred_local[sorted_where]
                registered_pred_global = registered_pred_global[sorted_where]
                target_mask = target_mask[sorted_where]
                phase_list = phase_list[sorted_where]

                #matplotlib.use('QtAgg')
                #fig, ax = plt.subplots(1, 7)
                #plot_idx = 0
                #for a in range(8, 15):
                #    current_img = softmax_pred[a, :, 0]
                #    current_img = current_img.argmax(0)
                #    ax[plot_idx].imshow(current_img, cmap='gray')
                #    plot_idx += 1
                #plt.show()
                #plt.waitforbuttonpress()
                #plt.close(fig)

                #stop_indices = [len(x) for x in windows]
                #videos = [phase_list[x] for x in windows]

                #matplotlib.use('QtAgg')
                #print(labeled_idx)
                #print(video_padding)
                #fig, ax = plt.subplots(1, self.video_length)
                #for i in range(self.video_length):
                #    ax[i].imshow(data[i, 0, 5], cmap='gray')
                #plt.show()

                assert len(softmax_pred) == len(flow_pred) == len(phase_list)

                and_mask = target_mask
                gt_indices = np.where(and_mask)[0]
                assert len(gt_indices) == 2

                for t in range(len(softmax_pred)):
                    if '_u' in phase_list[t]:
                        properties = load_pickle(self.unlabeled_dataset[phase_list[t]]['properties_file'])
                    else:
                        properties = load_pickle(self.dataset[phase_list[t]]['properties_file'])
                    fname = properties['list_of_data_files'][0].split(os.sep)[-1][:-12]

                    current_softmax_pred = softmax_pred[t]
                    current_softmax_pred = current_softmax_pred.transpose([0] + [i + 1 for i in self.transpose_backward])

                    if save_softmax:
                        softmax_fname = join(output_folder, fname + ".npz")
                    else:
                        softmax_fname = None

                    splitted = fname.split('frame')
                    to_nb = int(splitted[-1].split('_')[0].split('.')[0])
                    flow_name = splitted[0] + 'frame' + '01'.zfill(2) + '_to_' + str(to_nb).zfill(2)
                    flow_path = join(newpath_flow, flow_name + ".nii.gz")
                    current_flow = flow_pred[t]
                    current_flow = current_flow.transpose([0] + [i + 1 for i in self.transpose_backward])

                    registered_path_local = join(newpath_registered_local, fname + ".nii.gz")
                    current_registered_local = registered_pred_local[t]
                    current_registered_local = current_registered_local.transpose([0] + [i + 1 for i in self.transpose_backward])

                    registered_path_global = join(newpath_registered_global, fname + ".nii.gz")
                    current_registered_global = registered_pred_global[t]
                    current_registered_global = current_registered_global.transpose([0] + [i + 1 for i in self.transpose_backward])

                    """There is a problem with python process communication that prevents us from communicating objects
                    larger than 2 GB between processes (basically when the length of the pickle string that will be sent is
                    communicated by the multiprocessing.Pipe object then the placeholder (I think) does not allow for long
                    enough strings (lol). This could be fixed by changing i to l (for long) but that would require manually
                    patching system python code. We circumvent that problem here by saving softmax_pred to a npy file that will
                    then be read (and finally deleted) by the Process. save_segmentation_nifti_from_softmax can take either
                    filename or np.ndarray and will handle this automatically"""
                    if np.prod(current_softmax_pred.shape) > (2e9 / 4 * 0.85):  # *0.85 just to be save
                        np.save(join(newpath_seg, fname + ".npy"), current_softmax_pred)
                        current_softmax_pred = join(newpath_seg, fname + ".npy")

                    results.append(export_pool.starmap_async(save_segmentation_nifti_from_softmax,
                                                            ((current_softmax_pred, join(newpath_seg, fname + ".nii.gz"),
                                                            properties, interpolation_order, self.regions_class_order,
                                                            None, None,
                                                            softmax_fname, None, force_separate_z,
                                                            interpolation_order_z, False, current_flow, flow_path,
                                                            current_registered_local, registered_path_local, current_registered_global, registered_path_global),
                                                            )
                                                            )
                                )
                    
                    
                    if t in gt_indices:
                        metadata_list.append(self.create_metadata_dict(properties))
                        pred_gt_tuples.append([join(newpath_seg, fname + ".nii.gz"),
                                                join(self.gt_niftis_folder, fname + ".nii.gz")])
                    if t == gt_indices[1]:
                        to_validate_registered_list.append(fname + ".nii.gz")
                        metadata_list_registered.append(self.create_metadata_dict(properties))
                        pred_gt_tuples_register_local.append([registered_path_local,
                                                join(self.gt_niftis_folder, fname + ".nii.gz")])
                        pred_gt_tuples_register_global.append([registered_path_global,
                                                join(self.gt_niftis_folder, fname + ".nii.gz")])

        _ = [i.get() for i in results]
        self.print_to_log_file("finished prediction")

        # evaluate raw predictions
        self.print_to_log_file("evaluation of raw predictions")
        task = self.dataset_directory.split(os.sep)[-1]
        job_name = self.experiment_name
        _ = aggregate_scores(pred_gt_tuples_register_local, labels=list(range(self.num_classes)),
                             json_output_file=join(newpath_registered_local, "summary.json"),
                             json_name=job_name + " val tiled %s" % (str(use_sliding_window)),
                             json_author="Fabian",
                             json_task=task, num_threads=default_num_threads,
                             advanced=True,
                             metadata_list=metadata_list_registered)
        _ = aggregate_scores(pred_gt_tuples_register_global, labels=list(range(self.num_classes)),
                             json_output_file=join(newpath_registered_global, "summary.json"),
                             json_name=job_name + " val tiled %s" % (str(use_sliding_window)),
                             json_author="Fabian",
                             json_task=task, num_threads=default_num_threads,
                             advanced=True,
                             metadata_list=metadata_list)
        _ = aggregate_scores(pred_gt_tuples, labels=list(range(self.num_classes)),
                             json_output_file=join(newpath_seg, "summary.json"),
                             json_name=job_name + " val tiled %s" % (str(use_sliding_window)),
                             json_author="Fabian",
                             json_task=task, num_threads=default_num_threads,
                             advanced=True,
                             metadata_list=metadata_list)

        if run_postprocessing_on_folds:
            # in the old nnunet we would stop here. Now we add a postprocessing. This postprocessing can remove everything
            # except the largest connected component for each class. To see if this improves results, we do this for all
            # classes and then rerun the evaluation. Those classes for which this resulted in an improved dice score will
            # have this applied during inference as well
            self.print_to_log_file("determining postprocessing")
            base_segmentation = join(self.output_folder, 'Segmentation')
            determine_postprocessing(base_segmentation, self.gt_niftis_folder, validation_folder_name,
                                     final_subf_name=validation_folder_name + "_postprocessed", debug=debug, log_function=log_function,
                                     metadata_list=metadata_list, to_validate_list=None)
            
            base_registered_local = join(self.output_folder, 'Registered_local')
            determine_postprocessing(base_registered_local, self.gt_niftis_folder, validation_folder_name,
                                     final_subf_name=validation_folder_name + "_postprocessed", debug=debug, log_function=log_function,
                                     metadata_list=metadata_list_registered, to_validate_list=to_validate_registered_list)
            
            base_registered_global = join(self.output_folder, 'Registered_global')
            determine_postprocessing(base_registered_global, self.gt_niftis_folder, validation_folder_name,
                                     final_subf_name=validation_folder_name + "_postprocessed", debug=debug, log_function=log_function,
                                     metadata_list=metadata_list_registered, to_validate_list=to_validate_registered_list)
            # after this the final predictions for the vlaidation set can be found in validation_folder_name_base + "_postprocessed"
            # They are always in that folder, even if no postprocessing as applied!

        # detemining postprocesing on a per-fold basis may be OK for this fold but what if another fold finds another
        # postprocesing to be better? In this case we need to consolidate. At the time the consolidation is going to be
        # done we won't know what self.gt_niftis_folder was, so now we copy all the niftis into a separate folder to
        # be used later
        gt_nifti_folder = join(self.output_folder_base, "gt_niftis")
        maybe_mkdir_p(gt_nifti_folder)
        for f in subfiles(self.gt_niftis_folder, suffix=".nii.gz"):
            success = False
            attempts = 0
            e = None
            while not success and attempts < 10:
                try:
                    shutil.copy(f, gt_nifti_folder)
                    success = True
                except OSError as e:
                    attempts += 1
                    sleep(1)
            if not success:
                print("Could not copy gt nifti file %s into folder %s" % (f, gt_nifti_folder))
                if e is not None:
                    raise e

        self.network.train(current_mode)

    


    def validate_flow_overlap(self, processor, log_function, step, do_mirroring: bool = True, use_sliding_window: bool = True, step_size: float = 0.5,
                 save_softmax: bool = True, use_gaussian: bool = True, overwrite: bool = True,
                 validation_folder_name: str = 'validation_raw', debug: bool = True, all_in_gpu: bool = False,
                 segmentation_export_kwargs: dict = None, run_postprocessing_on_folds: bool = True, output_folder=None, save_flow=True):

        """
        if debug=True then the temporary files generated for postprocessing determination will be kept
        """
        current_mode = self.network.training
        self.network.eval()

        assert self.was_initialized, "must initialize, ideally with checkpoint (or train first)"
        if self.dataset_val is None:
            self.load_dataset()
            self.do_split()

        if segmentation_export_kwargs is None:
            if 'segmentation_export_params' in self.plans.keys():
                force_separate_z = self.plans['segmentation_export_params']['force_separate_z']
                interpolation_order = self.plans['segmentation_export_params']['interpolation_order']
                interpolation_order_z = self.plans['segmentation_export_params']['interpolation_order_z']
            else:
                force_separate_z = None
                interpolation_order = 1
                interpolation_order_z = 0
        else:
            force_separate_z = segmentation_export_kwargs['force_separate_z']
            interpolation_order = segmentation_export_kwargs['interpolation_order']
            interpolation_order_z = segmentation_export_kwargs['interpolation_order_z']

        # predictions as they come from the network go here
        if output_folder is not None:
            self.output_folder = output_folder
        #output_folder = join(self.output_folder, validation_folder_name)
        #maybe_mkdir_p(output_folder)
        # this is for debug purposes
        my_input_args = {'do_mirroring': do_mirroring,
                         'use_sliding_window': use_sliding_window,
                         'step_size': step_size,
                         'save_softmax': save_softmax,
                         'use_gaussian': use_gaussian,
                         'overwrite': overwrite,
                         'validation_folder_name': validation_folder_name,
                         'debug': debug,
                         'all_in_gpu': all_in_gpu,
                         'segmentation_export_kwargs': segmentation_export_kwargs,
                         }
        save_json(my_input_args, join(self.output_folder, "validation_args.json"))

        if do_mirroring:
            if not self.data_aug_params['do_mirror']:
                raise RuntimeError("We did not train with mirroring so you cannot do inference with mirroring enabled")
            mirror_axes = self.data_aug_params['mirror_axes']
        else:
            mirror_axes = ()

        pred_gt_tuples = []
        pred_gt_tuples_register = []

        export_pool = Pool(default_num_threads)
        results = []

        newpath_flow = join(self.output_folder, 'Flow', validation_folder_name)
        if not os.path.exists(newpath_flow):
            os.makedirs(newpath_flow)

        newpath_registered = join(self.output_folder, 'Registered', validation_folder_name)
        if not os.path.exists(newpath_registered):
            os.makedirs(newpath_registered)

        newpath_seg = join(self.output_folder, 'Segmentation', validation_folder_name)
        if not os.path.exists(newpath_seg):
            os.makedirs(newpath_seg)
        
        to_validate_registered_list = []

        list_of_keys = list(self.dataset_val.keys())
        un_list_of_keys = list(self.dataset_un_val.keys())
        patient_id_list = np.unique([x[:10] for x in list_of_keys])
        metadata_list = []
        metadata_list_registered = []
        for patient_id in tqdm(patient_id_list):
            all_keys = list_of_keys + un_list_of_keys
            phase_list = [x for x in all_keys if patient_id in x]
            phase_list = np.array(sorted(phase_list, key=lambda x: int(x[16:18])))
            phase_list = np.array(phase_list)
            global_labeled_idx = np.where(~np.char.endswith(phase_list, '_u'))[0]

            frame_indices = np.arange(len(phase_list))

            before_where = np.argwhere(frame_indices < global_labeled_idx[0]).reshape(-1,)
            after_where = np.argwhere(frame_indices >= global_labeled_idx[0]).reshape(-1,)
            assert after_where[0] == global_labeled_idx[0]

            before_str = phase_list[before_where]
            after_str = phase_list[after_where]

            all_where = np.concatenate([after_where, before_where])
            phase_list = np.concatenate([after_str, before_str])

            if '_u' in phase_list[0]:
                properties = load_pickle(self.unlabeled_dataset[phase_list[0]]['properties_file'])
            else:
                properties = load_pickle(self.dataset[phase_list[0]]['properties_file'])

            unlabeled = np.full(shape=((len(phase_list), 1) + properties['size_after_resampling']), fill_value=np.nan)
            target = np.full(shape=((len(phase_list), 1) + properties['size_after_resampling']), fill_value=np.nan)
            target_mask = np.zeros(shape=(len(phase_list),), dtype=bool)

            if overwrite or (not isfile(join(output_folder, phase_list[0] + ".nii.gz"))) or (save_softmax and not isfile(join(output_folder, phase_list[0] + ".npz"))):
                for idx, frame in enumerate(phase_list):
                    #if frame == k:
                    #    labeled_idx = idx
                    if '_u' in frame:
                        current_data = np.load(self.dataset_un_val[frame]['data_file'])['data']
                        current_data = current_data + 1e-8
                        unlabeled[idx] = current_data
                    else:
                        current_data = np.load(self.dataset_val[frame]['data_file'])['data']
                        #current_data[-1][current_data[-1] == -1] = 0
                        target[idx] = current_data[1]
                        target[idx][target[idx] == -1] = 0
                        unlabeled[idx] = current_data[0] + 1e-8
                        target_mask[idx] = True
                    #self.print_to_log_file(k, data.shape)

                step = int(ceil((len(phase_list) - 1) / (self.video_length - 1)))
                values = np.arange(1, len(phase_list))
                windows = [values[i::step] for i in range(step)]
                output_dict = {'softmax': [], 'flow': [], 'registered': [], 'windows': []}
                merged_dict = {'softmax': np.full((len(phase_list), 4) + properties['size_after_resampling'], fill_value=np.nan),
                                'flow': np.full((len(phase_list), 2) + properties['size_after_resampling'], fill_value=np.nan),
                                'registered': np.full((len(phase_list), 1) + properties['size_after_resampling'], fill_value=np.nan)}
                for window_idx, window in enumerate(windows):
                    pad_idx = []
                    if len(window) < self.video_length - 1:
                        to_pad = (self.video_length - 1) - len(window)
                        pad_idx = [(window[-1] + (step * i)) % len(phase_list) for i in range(1, to_pad + 1)]
                    window = np.concatenate([[0], window, pad_idx]).astype(int)
                    assert len(window) == self.video_length
                    current_unlabeled = unlabeled[window]
                    current_target = target[window]

                    assert np.all(np.isfinite(current_target[0]))

                    ret = self.predict_preprocessed_data_return_seg_and_softmax_flow(unlabeled=current_unlabeled,
                                                                                target=current_target,
                                                                                processor=processor,
                                                                                do_mirroring=do_mirroring,
                                                                                mirror_axes=mirror_axes,
                                                                                use_sliding_window=use_sliding_window,
                                                                                step_size=step_size,
                                                                                use_gaussian=use_gaussian,
                                                                                all_in_gpu=all_in_gpu,
                                                                                mixed_precision=self.fp16,
                                                                                verbose=False)
                    predicted_segmentation = ret[0] # T, depth, H, W
                    softmax_pred = ret[1] # T, C, depth, H, W
                    flow_pred = ret[2] # T, C, depth, H, W
                    registered_pred = ret[3] # T, C, depth, H, W
                    if pad_idx != []:
                        softmax_pred = softmax_pred[:-to_pad]
                        flow_pred = flow_pred[:-to_pad]
                        registered_pred = registered_pred[:-to_pad]
                        window = window[:-to_pad]
                    if window_idx < len(windows) - 1:
                        softmax_pred = softmax_pred[1:]
                        flow_pred = flow_pred[1:]
                        registered_pred = registered_pred[1:]
                        window = window[1:]

                    assert len(window) == len(softmax_pred)
                    output_dict['softmax'].append(softmax_pred)
                    output_dict['flow'].append(flow_pred)
                    output_dict['registered'].append(registered_pred)
                    output_dict['windows'].append(window)
                
                assert len(merged_dict['softmax'][output_dict['windows'][0]]) == len(output_dict['softmax'][0])
                
                for idx, window in enumerate(output_dict['windows']):
                    merged_dict['softmax'][window] = output_dict['softmax'][idx]
                    merged_dict['flow'][window] = output_dict['flow'][idx]
                    merged_dict['registered'][window] = output_dict['registered'][idx]

                assert np.all(np.isfinite(merged_dict['softmax']))

                sorted_where = np.argsort(all_where)
                merged_dict['softmax'] = merged_dict['softmax'][sorted_where]
                merged_dict['flow'] = merged_dict['flow'][sorted_where]
                merged_dict['registered'] = merged_dict['registered'][sorted_where]

                #matplotlib.use('QtAgg')
                #fig, ax = plt.subplots(1, 7)
                #plot_idx = 0
                #for a in range(8, 15):
                #    current_img = softmax_pred[a, :, 0]
                #    current_img = current_img.argmax(0)
                #    ax[plot_idx].imshow(current_img, cmap='gray')
                #    plot_idx += 1
                #plt.show()
                #plt.waitforbuttonpress()
                #plt.close(fig)

                #stop_indices = [len(x) for x in windows]
                #videos = [phase_list[x] for x in windows]

                #matplotlib.use('QtAgg')
                #print(labeled_idx)
                #print(video_padding)
                #fig, ax = plt.subplots(1, self.video_length)
                #for i in range(self.video_length):
                #    ax[i].imshow(data[i, 0, 5], cmap='gray')
                #plt.show()

                and_mask = target_mask
                gt_indices = np.where(and_mask)[0]
                assert len(gt_indices) == 2

                for t in range(len(merged_dict['softmax'])):
                    if '_u' in phase_list[t]:
                        properties = load_pickle(self.unlabeled_dataset[phase_list[t]]['properties_file'])
                    else:
                        properties = load_pickle(self.dataset[phase_list[t]]['properties_file'])
                    fname = properties['list_of_data_files'][0].split(os.sep)[-1][:-12]

                    current_softmax_pred = merged_dict['softmax'][t]
                    current_softmax_pred = current_softmax_pred.transpose([0] + [i + 1 for i in self.transpose_backward])

                    if save_softmax:
                        softmax_fname = join(output_folder, fname + ".npz")
                    else:
                        softmax_fname = None

                    splitted = fname.split('frame')
                    to_nb = int(splitted[-1].split('_')[0].split('.')[0])
                    flow_name = splitted[0] + 'frame' + '01'.zfill(2) + '_to_' + str(to_nb).zfill(2)
                    flow_path = join(newpath_flow, flow_name + ".nii.gz")
                    current_flow = merged_dict['flow'][t]
                    current_flow = current_flow.transpose([0] + [i + 1 for i in self.transpose_backward])

                    registered_path = join(newpath_registered, fname + ".nii.gz")
                    current_registered = merged_dict['registered'][t]
                    current_registered = current_registered.transpose([0] + [i + 1 for i in self.transpose_backward])

                    """There is a problem with python process communication that prevents us from communicating objects
                    larger than 2 GB between processes (basically when the length of the pickle string that will be sent is
                    communicated by the multiprocessing.Pipe object then the placeholder (I think) does not allow for long
                    enough strings (lol). This could be fixed by changing i to l (for long) but that would require manually
                    patching system python code. We circumvent that problem here by saving softmax_pred to a npy file that will
                    then be read (and finally deleted) by the Process. save_segmentation_nifti_from_softmax can take either
                    filename or np.ndarray and will handle this automatically"""
                    if np.prod(current_softmax_pred.shape) > (2e9 / 4 * 0.85):  # *0.85 just to be save
                        np.save(join(newpath_seg, fname + ".npy"), current_softmax_pred)
                        current_softmax_pred = join(newpath_seg, fname + ".npy")

                    results.append(export_pool.starmap_async(save_segmentation_nifti_from_softmax,
                                                            ((current_softmax_pred, join(newpath_seg, fname + ".nii.gz"),
                                                            properties, interpolation_order, self.regions_class_order,
                                                            None, None,
                                                            softmax_fname, None, force_separate_z,
                                                            interpolation_order_z, False, current_flow, flow_path,
                                                            current_registered, registered_path),
                                                            )
                                                            )
                                )
                    
                    
                    if t in gt_indices:
                        metadata_list.append(self.create_metadata_dict(properties))
                        pred_gt_tuples.append([join(newpath_seg, fname + ".nii.gz"),
                                                join(self.gt_niftis_folder, fname + ".nii.gz")])
                    if t == gt_indices[1]:
                        to_validate_registered_list.append(fname + ".nii.gz")
                        metadata_list_registered.append(self.create_metadata_dict(properties))
                        pred_gt_tuples_register.append([registered_path,
                                                join(self.gt_niftis_folder, fname + ".nii.gz")])

        _ = [i.get() for i in results]
        self.print_to_log_file("finished prediction")

        # evaluate raw predictions
        self.print_to_log_file("evaluation of raw predictions")
        task = self.dataset_directory.split(os.sep)[-1]
        job_name = self.experiment_name
        _ = aggregate_scores(pred_gt_tuples_register, labels=list(range(self.num_classes)),
                             json_output_file=join(newpath_registered, "summary.json"),
                             json_name=job_name + " val tiled %s" % (str(use_sliding_window)),
                             json_author="Fabian",
                             json_task=task, num_threads=default_num_threads,
                             advanced=True,
                             metadata_list=metadata_list_registered)
        _ = aggregate_scores(pred_gt_tuples, labels=list(range(self.num_classes)),
                             json_output_file=join(newpath_seg, "summary.json"),
                             json_name=job_name + " val tiled %s" % (str(use_sliding_window)),
                             json_author="Fabian",
                             json_task=task, num_threads=default_num_threads,
                             advanced=True,
                             metadata_list=metadata_list)

        if run_postprocessing_on_folds:
            # in the old nnunet we would stop here. Now we add a postprocessing. This postprocessing can remove everything
            # except the largest connected component for each class. To see if this improves results, we do this for all
            # classes and then rerun the evaluation. Those classes for which this resulted in an improved dice score will
            # have this applied during inference as well
            self.print_to_log_file("determining postprocessing")
            base_segmentation = join(self.output_folder, 'Segmentation')
            determine_postprocessing(base_segmentation, self.gt_niftis_folder, validation_folder_name,
                                     final_subf_name=validation_folder_name + "_postprocessed", debug=debug, log_function=log_function,
                                     metadata_list=metadata_list, to_validate_list=None)
            
            base_registered = join(self.output_folder, 'Registered')
            determine_postprocessing(base_registered, self.gt_niftis_folder, validation_folder_name,
                                     final_subf_name=validation_folder_name + "_postprocessed", debug=debug, log_function=log_function,
                                     metadata_list=metadata_list_registered, to_validate_list=to_validate_registered_list)
            # after this the final predictions for the vlaidation set can be found in validation_folder_name_base + "_postprocessed"
            # They are always in that folder, even if no postprocessing as applied!

        # detemining postprocesing on a per-fold basis may be OK for this fold but what if another fold finds another
        # postprocesing to be better? In this case we need to consolidate. At the time the consolidation is going to be
        # done we won't know what self.gt_niftis_folder was, so now we copy all the niftis into a separate folder to
        # be used later
        gt_nifti_folder = join(self.output_folder_base, "gt_niftis")
        maybe_mkdir_p(gt_nifti_folder)
        for f in subfiles(self.gt_niftis_folder, suffix=".nii.gz"):
            success = False
            attempts = 0
            e = None
            while not success and attempts < 10:
                try:
                    shutil.copy(f, gt_nifti_folder)
                    success = True
                except OSError as e:
                    attempts += 1
                    sleep(1)
            if not success:
                print("Could not copy gt nifti file %s into folder %s" % (f, gt_nifti_folder))
                if e is not None:
                    raise e

        self.network.train(current_mode)


    def validate_flow_overlap_alt(self, processor, log_function, do_mirroring: bool = True, use_sliding_window: bool = True, step_size: float = 0.5,
                 save_softmax: bool = True, use_gaussian: bool = True, overwrite: bool = True,
                 validation_folder_name: str = 'validation_raw', debug: bool = True, all_in_gpu: bool = False,
                 segmentation_export_kwargs: dict = None, run_postprocessing_on_folds: bool = True, output_folder=None, save_flow=True):

        """
        if debug=True then the temporary files generated for postprocessing determination will be kept
        """
        current_mode = self.network.training
        self.network.eval()

        assert self.was_initialized, "must initialize, ideally with checkpoint (or train first)"
        if self.dataset_val is None:
            self.load_dataset()
            self.do_split()

        if segmentation_export_kwargs is None:
            if 'segmentation_export_params' in self.plans.keys():
                force_separate_z = self.plans['segmentation_export_params']['force_separate_z']
                interpolation_order = self.plans['segmentation_export_params']['interpolation_order']
                interpolation_order_z = self.plans['segmentation_export_params']['interpolation_order_z']
            else:
                force_separate_z = None
                interpolation_order = 1
                interpolation_order_z = 0
        else:
            force_separate_z = segmentation_export_kwargs['force_separate_z']
            interpolation_order = segmentation_export_kwargs['interpolation_order']
            interpolation_order_z = segmentation_export_kwargs['interpolation_order_z']

        # predictions as they come from the network go here
        if output_folder is not None:
            self.output_folder = output_folder
        #output_folder = join(self.output_folder, validation_folder_name)
        #maybe_mkdir_p(output_folder)
        # this is for debug purposes
        my_input_args = {'do_mirroring': do_mirroring,
                         'use_sliding_window': use_sliding_window,
                         'step_size': step_size,
                         'save_softmax': save_softmax,
                         'use_gaussian': use_gaussian,
                         'overwrite': overwrite,
                         'validation_folder_name': validation_folder_name,
                         'debug': debug,
                         'all_in_gpu': all_in_gpu,
                         'segmentation_export_kwargs': segmentation_export_kwargs,
                         }
        save_json(my_input_args, join(self.output_folder, "validation_args.json"))

        if do_mirroring:
            if not self.data_aug_params['do_mirror']:
                raise RuntimeError("We did not train with mirroring so you cannot do inference with mirroring enabled")
            mirror_axes = self.data_aug_params['mirror_axes']
        else:
            mirror_axes = ()

        pred_gt_tuples = []
        pred_gt_tuples_register = []

        export_pool = Pool(default_num_threads)
        results = []

        newpath_flow = join(self.output_folder, 'Flow', validation_folder_name)
        if not os.path.exists(newpath_flow):
            os.makedirs(newpath_flow)

        newpath_registered = join(self.output_folder, 'Registered', validation_folder_name)
        if not os.path.exists(newpath_registered):
            os.makedirs(newpath_registered)

        newpath_seg = join(self.output_folder, 'Segmentation', validation_folder_name)
        if not os.path.exists(newpath_seg):
            os.makedirs(newpath_seg)
        
        to_validate_registered_list = []

        list_of_keys = list(self.dataset_val.keys())
        un_list_of_keys = list(self.dataset_un_val.keys())
        patient_id_list = np.unique([x[:10] for x in list_of_keys])
        metadata_list = []
        metadata_list_registered = []
        for patient_id in tqdm(patient_id_list):
            all_keys = list_of_keys + un_list_of_keys
            phase_list = [x for x in all_keys if patient_id in x]
            phase_list = np.array(sorted(phase_list, key=lambda x: int(x[16:18])))
            phase_list = np.array(phase_list)
            global_labeled_idx = np.where(~np.char.endswith(phase_list, '_u'))[0]

            frame_indices = np.arange(len(phase_list))

            before_where = np.argwhere(frame_indices < global_labeled_idx[0]).reshape(-1,)
            after_where = np.argwhere(frame_indices >= global_labeled_idx[0]).reshape(-1,)
            assert after_where[0] == global_labeled_idx[0]

            before_str = phase_list[before_where]
            after_str = phase_list[after_where]

            all_where = np.concatenate([after_where, before_where])
            phase_list = np.concatenate([after_str, before_str])

            if '_u' in phase_list[0]:
                properties = load_pickle(self.unlabeled_dataset[phase_list[0]]['properties_file'])
            else:
                properties = load_pickle(self.dataset[phase_list[0]]['properties_file'])

            unlabeled = np.full(shape=((len(phase_list), 1) + properties['size_after_resampling']), fill_value=np.nan)
            target = np.full(shape=((len(phase_list), 1) + properties['size_after_resampling']), fill_value=np.nan)
            target_mask = np.zeros(shape=(len(phase_list),), dtype=bool)

            if overwrite or (not isfile(join(output_folder, phase_list[0] + ".nii.gz"))) or (save_softmax and not isfile(join(output_folder, phase_list[0] + ".npz"))):
                for idx, frame in enumerate(phase_list):
                    #if frame == k:
                    #    labeled_idx = idx
                    if '_u' in frame:
                        current_data = np.load(self.dataset_un_val[frame]['data_file'])['data']
                        current_data = current_data + 1e-8
                        unlabeled[idx] = current_data
                    else:
                        current_data = np.load(self.dataset_val[frame]['data_file'])['data']
                        #current_data[-1][current_data[-1] == -1] = 0
                        target[idx] = current_data[1]
                        target[idx][target[idx] == -1] = 0
                        unlabeled[idx] = current_data[0] + 1e-8
                        target_mask[idx] = True
                    #self.print_to_log_file(k, data.shape)

                step = int(ceil(len(phase_list) / self.video_length))
                values = np.arange(len(phase_list))
                values = np.pad(values, (0, 100), mode='wrap')
                windows = [values[i::step] for i in range(step)]
                windows = [windows[i][:self.video_length] for i in range(len(windows))]
                output_dict = {'softmax': [], 'flow': [], 'registered': [], 'windows': []}
                merged_dict = {'softmax': np.full((len(phase_list), 4) + properties['size_after_resampling'], fill_value=np.nan),
                                'flow': np.full((len(phase_list), 2) + properties['size_after_resampling'], fill_value=np.nan),
                                'registered': np.full((len(phase_list), 1) + properties['size_after_resampling'], fill_value=np.nan)}
                
                for window in windows:
                    assert len(window) == self.video_length
                    current_unlabeled = unlabeled[window]
                    current_target = target[window]

                    current_target_mask = np.zeros(shape=(len(window),), dtype=bool)
                    ed_index = np.where(window == 0)[0]
                    current_target_mask[ed_index] = True

                    #assert np.all(np.isfinite(current_target[0]))

                    ret = self.predict_preprocessed_data_return_seg_and_softmax_flow(unlabeled=current_unlabeled,
                                                                                target=current_target,
                                                                                target_mask=current_target_mask,
                                                                                processor=processor,
                                                                                do_mirroring=do_mirroring,
                                                                                mirror_axes=mirror_axes,
                                                                                use_sliding_window=use_sliding_window,
                                                                                step_size=step_size,
                                                                                use_gaussian=use_gaussian,
                                                                                all_in_gpu=all_in_gpu,
                                                                                mixed_precision=self.fp16,
                                                                                verbose=False)
                    predicted_segmentation = ret[0] # T, depth, H, W
                    softmax_pred = ret[1] # T, C, depth, H, W
                    flow_pred = ret[2] # T, C, depth, H, W
                    registered_pred = ret[3] # T, C, depth, H, W

                    start_padding_idx = np.where(window[:-1] > window[1:])[0]

                    if start_padding_idx:
                        softmax_pred = softmax_pred[:start_padding_idx + 1]
                        flow_pred = flow_pred[:start_padding_idx + 1]
                        registered_pred = registered_pred[:start_padding_idx + 1]
                        window = window[:start_padding_idx + 1]

                    assert len(window) == len(softmax_pred)
                    output_dict['softmax'].append(softmax_pred)
                    output_dict['flow'].append(flow_pred)
                    output_dict['registered'].append(registered_pred)
                    output_dict['windows'].append(window)
                
                assert len(merged_dict['softmax'][output_dict['windows'][0]]) == len(output_dict['softmax'][0])
                
                for idx, window in enumerate(output_dict['windows']):
                    merged_dict['softmax'][window] = output_dict['softmax'][idx]
                    merged_dict['flow'][window] = output_dict['flow'][idx]
                    merged_dict['registered'][window] = output_dict['registered'][idx]

                assert np.all(np.isfinite(merged_dict['softmax']))

                sorted_where = np.argsort(all_where)
                merged_dict['softmax'] = merged_dict['softmax'][sorted_where]
                merged_dict['flow'] = merged_dict['flow'][sorted_where]
                merged_dict['registered'] = merged_dict['registered'][sorted_where]
                target_mask = target_mask[sorted_where]
                phase_list = phase_list[sorted_where]

                #matplotlib.use('QtAgg')
                #fig, ax = plt.subplots(1, 7)
                #plot_idx = 0
                #for a in range(8, 15):
                #    current_img = softmax_pred[a, :, 0]
                #    current_img = current_img.argmax(0)
                #    ax[plot_idx].imshow(current_img, cmap='gray')
                #    plot_idx += 1
                #plt.show()
                #plt.waitforbuttonpress()
                #plt.close(fig)

                #stop_indices = [len(x) for x in windows]
                #videos = [phase_list[x] for x in windows]

                #matplotlib.use('QtAgg')
                #print(labeled_idx)
                #print(video_padding)
                #fig, ax = plt.subplots(1, self.video_length)
                #for i in range(self.video_length):
                #    ax[i].imshow(data[i, 0, 5], cmap='gray')
                #plt.show()

                and_mask = target_mask
                gt_indices = np.where(and_mask)[0]
                assert len(gt_indices) == 2

                for t in range(len(merged_dict['softmax'])):
                    if '_u' in phase_list[t]:
                        properties = load_pickle(self.unlabeled_dataset[phase_list[t]]['properties_file'])
                    else:
                        properties = load_pickle(self.dataset[phase_list[t]]['properties_file'])
                    fname = properties['list_of_data_files'][0].split(os.sep)[-1][:-12]

                    current_softmax_pred = merged_dict['softmax'][t]
                    current_softmax_pred = current_softmax_pred.transpose([0] + [i + 1 for i in self.transpose_backward])

                    if save_softmax:
                        softmax_fname = join(output_folder, fname + ".npz")
                    else:
                        softmax_fname = None

                    splitted = fname.split('frame')
                    to_nb = int(splitted[-1].split('_')[0].split('.')[0])
                    flow_name = splitted[0] + 'frame' + '01'.zfill(2) + '_to_' + str(to_nb).zfill(2)
                    flow_path = join(newpath_flow, flow_name + ".nii.gz")
                    current_flow = merged_dict['flow'][t]
                    current_flow = current_flow.transpose([0] + [i + 1 for i in self.transpose_backward])

                    registered_path = join(newpath_registered, fname + ".nii.gz")
                    current_registered = merged_dict['registered'][t]
                    current_registered = current_registered.transpose([0] + [i + 1 for i in self.transpose_backward])

                    """There is a problem with python process communication that prevents us from communicating objects
                    larger than 2 GB between processes (basically when the length of the pickle string that will be sent is
                    communicated by the multiprocessing.Pipe object then the placeholder (I think) does not allow for long
                    enough strings (lol). This could be fixed by changing i to l (for long) but that would require manually
                    patching system python code. We circumvent that problem here by saving softmax_pred to a npy file that will
                    then be read (and finally deleted) by the Process. save_segmentation_nifti_from_softmax can take either
                    filename or np.ndarray and will handle this automatically"""
                    if np.prod(current_softmax_pred.shape) > (2e9 / 4 * 0.85):  # *0.85 just to be save
                        np.save(join(newpath_seg, fname + ".npy"), current_softmax_pred)
                        current_softmax_pred = join(newpath_seg, fname + ".npy")

                    results.append(export_pool.starmap_async(save_segmentation_nifti_from_softmax,
                                                            ((current_softmax_pred, join(newpath_seg, fname + ".nii.gz"),
                                                            properties, interpolation_order, self.regions_class_order,
                                                            None, None,
                                                            softmax_fname, None, force_separate_z,
                                                            interpolation_order_z, False, current_flow, flow_path,
                                                            current_registered, registered_path),
                                                            )
                                                            )
                                )
                    
                    
                    if t in gt_indices:
                        metadata_list.append(self.create_metadata_dict(properties))
                        pred_gt_tuples.append([join(newpath_seg, fname + ".nii.gz"),
                                                join(self.gt_niftis_folder, fname + ".nii.gz")])
                    if t == gt_indices[1]:
                        to_validate_registered_list.append(fname + ".nii.gz")
                        metadata_list_registered.append(self.create_metadata_dict(properties))
                        pred_gt_tuples_register.append([registered_path,
                                                join(self.gt_niftis_folder, fname + ".nii.gz")])

        _ = [i.get() for i in results]
        self.print_to_log_file("finished prediction")

        # evaluate raw predictions
        self.print_to_log_file("evaluation of raw predictions")
        task = self.dataset_directory.split(os.sep)[-1]
        job_name = self.experiment_name
        _ = aggregate_scores(pred_gt_tuples_register, labels=list(range(self.num_classes)),
                             json_output_file=join(newpath_registered, "summary.json"),
                             json_name=job_name + " val tiled %s" % (str(use_sliding_window)),
                             json_author="Fabian",
                             json_task=task, num_threads=default_num_threads,
                             advanced=True,
                             metadata_list=metadata_list_registered)
        _ = aggregate_scores(pred_gt_tuples, labels=list(range(self.num_classes)),
                             json_output_file=join(newpath_seg, "summary.json"),
                             json_name=job_name + " val tiled %s" % (str(use_sliding_window)),
                             json_author="Fabian",
                             json_task=task, num_threads=default_num_threads,
                             advanced=True,
                             metadata_list=metadata_list)

        if run_postprocessing_on_folds:
            # in the old nnunet we would stop here. Now we add a postprocessing. This postprocessing can remove everything
            # except the largest connected component for each class. To see if this improves results, we do this for all
            # classes and then rerun the evaluation. Those classes for which this resulted in an improved dice score will
            # have this applied during inference as well
            self.print_to_log_file("determining postprocessing")
            base_segmentation = join(self.output_folder, 'Segmentation')
            determine_postprocessing(base_segmentation, self.gt_niftis_folder, validation_folder_name,
                                     final_subf_name=validation_folder_name + "_postprocessed", debug=debug, log_function=log_function,
                                     metadata_list=metadata_list, to_validate_list=None)
            
            base_registered = join(self.output_folder, 'Registered')
            determine_postprocessing(base_registered, self.gt_niftis_folder, validation_folder_name,
                                     final_subf_name=validation_folder_name + "_postprocessed", debug=debug, log_function=log_function,
                                     metadata_list=metadata_list_registered, to_validate_list=to_validate_registered_list)
            # after this the final predictions for the vlaidation set can be found in validation_folder_name_base + "_postprocessed"
            # They are always in that folder, even if no postprocessing as applied!

        # detemining postprocesing on a per-fold basis may be OK for this fold but what if another fold finds another
        # postprocesing to be better? In this case we need to consolidate. At the time the consolidation is going to be
        # done we won't know what self.gt_niftis_folder was, so now we copy all the niftis into a separate folder to
        # be used later
        gt_nifti_folder = join(self.output_folder_base, "gt_niftis")
        maybe_mkdir_p(gt_nifti_folder)
        for f in subfiles(self.gt_niftis_folder, suffix=".nii.gz"):
            success = False
            attempts = 0
            e = None
            while not success and attempts < 10:
                try:
                    shutil.copy(f, gt_nifti_folder)
                    success = True
                except OSError as e:
                    attempts += 1
                    sleep(1)
            if not success:
                print("Could not copy gt nifti file %s into folder %s" % (f, gt_nifti_folder))
                if e is not None:
                    raise e

        self.network.train(current_mode)


    def validate_flow_sequence(self, processor, log_function, step, do_mirroring: bool = True, use_sliding_window: bool = True, step_size: float = 0.5,
                 save_softmax: bool = True, use_gaussian: bool = True, overwrite: bool = True,
                 validation_folder_name: str = 'validation_raw', debug: bool = True, all_in_gpu: bool = False,
                 segmentation_export_kwargs: dict = None, run_postprocessing_on_folds: bool = True, output_folder=None, save_flow=True):

        """
        if debug=True then the temporary files generated for postprocessing determination will be kept
        """
        current_mode = self.network.training
        self.network.eval()

        assert self.was_initialized, "must initialize, ideally with checkpoint (or train first)"
        if self.dataset_val is None:
            self.load_dataset()
            self.do_split()

        if segmentation_export_kwargs is None:
            if 'segmentation_export_params' in self.plans.keys():
                force_separate_z = self.plans['segmentation_export_params']['force_separate_z']
                interpolation_order = self.plans['segmentation_export_params']['interpolation_order']
                interpolation_order_z = self.plans['segmentation_export_params']['interpolation_order_z']
            else:
                force_separate_z = None
                interpolation_order = 1
                interpolation_order_z = 0
        else:
            force_separate_z = segmentation_export_kwargs['force_separate_z']
            interpolation_order = segmentation_export_kwargs['interpolation_order']
            interpolation_order_z = segmentation_export_kwargs['interpolation_order_z']

        # predictions as they come from the network go here
        if output_folder is not None:
            self.output_folder = output_folder
        #output_folder = join(self.output_folder, validation_folder_name)
        #maybe_mkdir_p(output_folder)
        # this is for debug purposes
        my_input_args = {'do_mirroring': do_mirroring,
                         'use_sliding_window': use_sliding_window,
                         'step_size': step_size,
                         'save_softmax': save_softmax,
                         'use_gaussian': use_gaussian,
                         'overwrite': overwrite,
                         'validation_folder_name': validation_folder_name,
                         'debug': debug,
                         'all_in_gpu': all_in_gpu,
                         'segmentation_export_kwargs': segmentation_export_kwargs,
                         }
        save_json(my_input_args, join(self.output_folder, "validation_args.json"))

        if do_mirroring:
            if not self.data_aug_params['do_mirror']:
                raise RuntimeError("We did not train with mirroring so you cannot do inference with mirroring enabled")
            mirror_axes = self.data_aug_params['mirror_axes']
        else:
            mirror_axes = ()

        pred_gt_tuples = []
        pred_gt_tuples_register = []

        export_pool = Pool(default_num_threads)
        results = []

        newpath_flow = join(self.output_folder, 'Flow', validation_folder_name)
        if not os.path.exists(newpath_flow):
            os.makedirs(newpath_flow)

        newpath_registered = join(self.output_folder, 'Registered', validation_folder_name)
        if not os.path.exists(newpath_registered):
            os.makedirs(newpath_registered)

        newpath_seg = join(self.output_folder, 'Segmentation', validation_folder_name)
        if not os.path.exists(newpath_seg):
            os.makedirs(newpath_seg)
        
        to_validate_registered_list = []

        list_of_keys = list(self.dataset_val.keys())
        un_list_of_keys = list(self.dataset_un_val.keys())
        patient_id_list = np.unique([x[:10] for x in list_of_keys])
        metadata_list = []
        metadata_list_registered = []
        for patient_id in tqdm(patient_id_list):
            all_keys = list_of_keys + un_list_of_keys
            phase_list = [x for x in all_keys if patient_id in x]
            phase_list = np.array(sorted(phase_list, key=lambda x: int(x[16:18])))
            phase_list = np.array(phase_list)
            global_labeled_idx = np.where(~np.char.endswith(phase_list, '_u'))[0]

            frame_indices = np.arange(len(phase_list))

            before_where = np.argwhere(frame_indices < global_labeled_idx[0]).reshape(-1,)
            after_where = np.argwhere(frame_indices >= global_labeled_idx[0]).reshape(-1,)
            assert after_where[0] == global_labeled_idx[0]

            before_str = phase_list[before_where]
            after_str = phase_list[after_where]

            all_where = np.concatenate([after_where, before_where])
            phase_list = np.concatenate([after_str, before_str])

            if '_u' in phase_list[0]:
                properties = load_pickle(self.unlabeled_dataset[phase_list[0]]['properties_file'])
            else:
                properties = load_pickle(self.dataset[phase_list[0]]['properties_file'])

            unlabeled = np.full(shape=((len(phase_list), 1) + properties['size_after_resampling']), fill_value=np.nan)
            target = np.full(shape=((len(phase_list), 1) + properties['size_after_resampling']), fill_value=np.nan)
            target_mask = np.zeros(shape=(len(phase_list),), dtype=bool)

            if overwrite or (not isfile(join(output_folder, phase_list[0] + ".nii.gz"))) or (save_softmax and not isfile(join(output_folder, phase_list[0] + ".npz"))):
                for idx, frame in enumerate(phase_list):
                    #if frame == k:
                    #    labeled_idx = idx
                    if '_u' in frame:
                        current_data = np.load(self.dataset_un_val[frame]['data_file'])['data']
                        current_data = current_data + 1e-8
                        unlabeled[idx] = current_data
                    else:
                        current_data = np.load(self.dataset_val[frame]['data_file'])['data']
                        #current_data[-1][current_data[-1] == -1] = 0
                        target[idx] = current_data[1]
                        target[idx][target[idx] == -1] = 0
                        unlabeled[idx] = current_data[0] + 1e-8
                        target_mask[idx] = True
                    #self.print_to_log_file(k, data.shape)

                values = np.arange(len(phase_list))
                nb_sequences = int(ceil(len(phase_list) / (self.video_length - 1)))
                windows = [values[(self.video_length - 1) * i:(self.video_length - 1) * (i+1)] for i in range(nb_sequences)]
                windows = [np.concatenate([[0], windows[i]]) for i in range(len(windows))]
                output_dict = {'softmax': [], 'flow': [], 'registered': [], 'windows': []}
                merged_dict = {'softmax': np.full((len(phase_list), 4) + properties['size_after_resampling'], fill_value=np.nan),
                                'flow': np.full((len(phase_list), 2) + properties['size_after_resampling'], fill_value=np.nan),
                                'registered': np.full((len(phase_list), 1) + properties['size_after_resampling'], fill_value=np.nan)}
                for window_idx, window in enumerate(windows):
                    #pad_idx = []
                    to_pad_right = 0
                    if len(window) < self.video_length:
                        to_pad_right = self.video_length - len(window)
                        window = np.pad(window, pad_width=(0, to_pad_right), mode='edge').astype(int)
                    assert len(window) == self.video_length
                    current_unlabeled = unlabeled[window]
                    current_target = target[window]

                    #assert np.all(np.isfinite(current_target[0]))

                    ret = self.predict_preprocessed_data_return_seg_and_softmax_flow(unlabeled=current_unlabeled,
                                                                                target=current_target,
                                                                                processor=processor,
                                                                                do_mirroring=do_mirroring,
                                                                                mirror_axes=mirror_axes,
                                                                                use_sliding_window=use_sliding_window,
                                                                                step_size=step_size,
                                                                                use_gaussian=use_gaussian,
                                                                                all_in_gpu=all_in_gpu,
                                                                                mixed_precision=self.fp16,
                                                                                verbose=False)
                    predicted_segmentation = ret[0] # T, depth, H, W
                    softmax_pred = ret[1] # T, C, depth, H, W
                    flow_pred = ret[2] # T, C, depth, H, W
                    registered_pred = ret[3] # T, C, depth, H, W
                    if to_pad_right > 0:
                        softmax_pred = softmax_pred[:-to_pad_right]
                        flow_pred = flow_pred[:-to_pad_right]
                        registered_pred = registered_pred[:-to_pad_right]
                        window = window[:-to_pad_right]
                    softmax_pred = softmax_pred[1:]
                    flow_pred = flow_pred[1:]
                    registered_pred = registered_pred[1:]
                    window = window[1:]

                    assert len(window) == len(softmax_pred)
                    output_dict['softmax'].append(softmax_pred)
                    output_dict['flow'].append(flow_pred)
                    output_dict['registered'].append(registered_pred)
                    output_dict['windows'].append(window)
                
                assert len(merged_dict['softmax'][output_dict['windows'][0]]) == len(output_dict['softmax'][0])
                
                for idx, window in enumerate(output_dict['windows']):
                    merged_dict['softmax'][window] = output_dict['softmax'][idx]
                    merged_dict['flow'][window] = output_dict['flow'][idx]
                    merged_dict['registered'][window] = output_dict['registered'][idx]

                assert np.all(np.isfinite(merged_dict['softmax']))

                sorted_where = np.argsort(all_where)
                merged_dict['softmax'] = merged_dict['softmax'][sorted_where]
                merged_dict['flow'] = merged_dict['flow'][sorted_where]
                merged_dict['registered'] = merged_dict['registered'][sorted_where]
                target_mask = target_mask[sorted_where]
                phase_list = phase_list[sorted_where]

                #matplotlib.use('QtAgg')
                #fig, ax = plt.subplots(1, 7)
                #plot_idx = 0
                #for a in range(8, 15):
                #    current_img = softmax_pred[a, :, 0]
                #    current_img = current_img.argmax(0)
                #    ax[plot_idx].imshow(current_img, cmap='gray')
                #    plot_idx += 1
                #plt.show()
                #plt.waitforbuttonpress()
                #plt.close(fig)

                #stop_indices = [len(x) for x in windows]
                #videos = [phase_list[x] for x in windows]

                #matplotlib.use('QtAgg')
                #print(labeled_idx)
                #print(video_padding)
                #fig, ax = plt.subplots(1, self.video_length)
                #for i in range(self.video_length):
                #    ax[i].imshow(data[i, 0, 5], cmap='gray')
                #plt.show()

                and_mask = target_mask
                gt_indices = np.where(and_mask)[0]
                assert len(gt_indices) == 2

                for t in range(len(merged_dict['softmax'])):
                    if '_u' in phase_list[t]:
                        properties = load_pickle(self.unlabeled_dataset[phase_list[t]]['properties_file'])
                    else:
                        properties = load_pickle(self.dataset[phase_list[t]]['properties_file'])
                    fname = properties['list_of_data_files'][0].split(os.sep)[-1][:-12]

                    current_softmax_pred = merged_dict['softmax'][t]
                    current_softmax_pred = current_softmax_pred.transpose([0] + [i + 1 for i in self.transpose_backward])

                    if save_softmax:
                        softmax_fname = join(output_folder, fname + ".npz")
                    else:
                        softmax_fname = None

                    splitted = fname.split('frame')
                    to_nb = int(splitted[-1].split('_')[0].split('.')[0])
                    flow_name = splitted[0] + 'frame' + '01'.zfill(2) + '_to_' + str(to_nb).zfill(2)
                    flow_path = join(newpath_flow, flow_name + ".nii.gz")
                    current_flow = merged_dict['flow'][t]
                    current_flow = current_flow.transpose([0] + [i + 1 for i in self.transpose_backward])

                    registered_path = join(newpath_registered, fname + ".nii.gz")
                    current_registered = merged_dict['registered'][t]
                    current_registered = current_registered.transpose([0] + [i + 1 for i in self.transpose_backward])

                    """There is a problem with python process communication that prevents us from communicating objects
                    larger than 2 GB between processes (basically when the length of the pickle string that will be sent is
                    communicated by the multiprocessing.Pipe object then the placeholder (I think) does not allow for long
                    enough strings (lol). This could be fixed by changing i to l (for long) but that would require manually
                    patching system python code. We circumvent that problem here by saving softmax_pred to a npy file that will
                    then be read (and finally deleted) by the Process. save_segmentation_nifti_from_softmax can take either
                    filename or np.ndarray and will handle this automatically"""
                    if np.prod(current_softmax_pred.shape) > (2e9 / 4 * 0.85):  # *0.85 just to be save
                        np.save(join(newpath_seg, fname + ".npy"), current_softmax_pred)
                        current_softmax_pred = join(newpath_seg, fname + ".npy")

                    results.append(export_pool.starmap_async(save_segmentation_nifti_from_softmax,
                                                            ((current_softmax_pred, join(newpath_seg, fname + ".nii.gz"),
                                                            properties, interpolation_order, self.regions_class_order,
                                                            None, None,
                                                            softmax_fname, None, force_separate_z,
                                                            interpolation_order_z, False, current_flow, flow_path,
                                                            current_registered, registered_path),
                                                            )
                                                            )
                                )
                    
                    
                    if t in gt_indices:
                        metadata_list.append(self.create_metadata_dict(properties))
                        pred_gt_tuples.append([join(newpath_seg, fname + ".nii.gz"),
                                                join(self.gt_niftis_folder, fname + ".nii.gz")])
                    if t == gt_indices[1]:
                        to_validate_registered_list.append(fname + ".nii.gz")
                        metadata_list_registered.append(self.create_metadata_dict(properties))
                        pred_gt_tuples_register.append([registered_path,
                                                join(self.gt_niftis_folder, fname + ".nii.gz")])

        _ = [i.get() for i in results]
        self.print_to_log_file("finished prediction")

        # evaluate raw predictions
        self.print_to_log_file("evaluation of raw predictions")
        task = self.dataset_directory.split(os.sep)[-1]
        job_name = self.experiment_name
        _ = aggregate_scores(pred_gt_tuples_register, labels=list(range(self.num_classes)),
                             json_output_file=join(newpath_registered, "summary.json"),
                             json_name=job_name + " val tiled %s" % (str(use_sliding_window)),
                             json_author="Fabian",
                             json_task=task, num_threads=default_num_threads,
                             advanced=True,
                             metadata_list=metadata_list_registered)
        _ = aggregate_scores(pred_gt_tuples, labels=list(range(self.num_classes)),
                             json_output_file=join(newpath_seg, "summary.json"),
                             json_name=job_name + " val tiled %s" % (str(use_sliding_window)),
                             json_author="Fabian",
                             json_task=task, num_threads=default_num_threads,
                             advanced=True,
                             metadata_list=metadata_list)

        if run_postprocessing_on_folds:
            # in the old nnunet we would stop here. Now we add a postprocessing. This postprocessing can remove everything
            # except the largest connected component for each class. To see if this improves results, we do this for all
            # classes and then rerun the evaluation. Those classes for which this resulted in an improved dice score will
            # have this applied during inference as well
            self.print_to_log_file("determining postprocessing")
            base_segmentation = join(self.output_folder, 'Segmentation')
            determine_postprocessing(base_segmentation, self.gt_niftis_folder, validation_folder_name,
                                     final_subf_name=validation_folder_name + "_postprocessed", debug=debug, log_function=log_function,
                                     metadata_list=metadata_list, to_validate_list=None)
            
            base_registered = join(self.output_folder, 'Registered')
            determine_postprocessing(base_registered, self.gt_niftis_folder, validation_folder_name,
                                     final_subf_name=validation_folder_name + "_postprocessed", debug=debug, log_function=log_function,
                                     metadata_list=metadata_list_registered, to_validate_list=to_validate_registered_list)
            # after this the final predictions for the vlaidation set can be found in validation_folder_name_base + "_postprocessed"
            # They are always in that folder, even if no postprocessing as applied!

        # detemining postprocesing on a per-fold basis may be OK for this fold but what if another fold finds another
        # postprocesing to be better? In this case we need to consolidate. At the time the consolidation is going to be
        # done we won't know what self.gt_niftis_folder was, so now we copy all the niftis into a separate folder to
        # be used later
        gt_nifti_folder = join(self.output_folder_base, "gt_niftis")
        maybe_mkdir_p(gt_nifti_folder)
        for f in subfiles(self.gt_niftis_folder, suffix=".nii.gz"):
            success = False
            attempts = 0
            e = None
            while not success and attempts < 10:
                try:
                    shutil.copy(f, gt_nifti_folder)
                    success = True
                except OSError as e:
                    attempts += 1
                    sleep(1)
            if not success:
                print("Could not copy gt nifti file %s into folder %s" % (f, gt_nifti_folder))
                if e is not None:
                    raise e

        self.network.train(current_mode)


    def run_online_evaluation(self, output, target):
        with torch.no_grad():
            num_classes = output.shape[1]
            output_softmax = softmax_helper(output)
            output_seg = output_softmax.argmax(1)
            target = target[:, 0]
            axes = tuple(range(1, len(target.shape)))
            tp_hard = torch.zeros((target.shape[0], num_classes - 1)).to(output_seg.device.index)
            fp_hard = torch.zeros((target.shape[0], num_classes - 1)).to(output_seg.device.index)
            fn_hard = torch.zeros((target.shape[0], num_classes - 1)).to(output_seg.device.index)
            for c in range(1, num_classes):
                tp_hard[:, c - 1] = sum_tensor((output_seg == c).float() * (target == c).float(), axes=axes)
                fp_hard[:, c - 1] = sum_tensor((output_seg == c).float() * (target != c).float(), axes=axes)
                fn_hard[:, c - 1] = sum_tensor((output_seg != c).float() * (target == c).float(), axes=axes)

            tp_hard = tp_hard.sum(0, keepdim=False).detach().cpu().numpy()
            fp_hard = fp_hard.sum(0, keepdim=False).detach().cpu().numpy()
            fn_hard = fn_hard.sum(0, keepdim=False).detach().cpu().numpy()

            self.online_eval_foreground_dc.append(list((2 * tp_hard) / (2 * tp_hard + fp_hard + fn_hard + 1e-8)))
            self.online_eval_tp.append(list(tp_hard))
            self.online_eval_fp.append(list(fp_hard))
            self.online_eval_fn.append(list(fn_hard))

    def finish_online_evaluation(self):
        self.online_eval_tp = np.sum(self.online_eval_tp, 0)
        self.online_eval_fp = np.sum(self.online_eval_fp, 0)
        self.online_eval_fn = np.sum(self.online_eval_fn, 0)

        global_dc_per_class = [i for i in [2 * i / (2 * i + j + k) for i, j, k in
                                           zip(self.online_eval_tp, self.online_eval_fp, self.online_eval_fn)]
                               if not np.isnan(i)]
        self.all_val_eval_metrics.append(np.mean(global_dc_per_class))

        self.print_to_log_file("Average global foreground Dice:", [np.round(i, 4) for i in global_dc_per_class])
        self.print_to_log_file("(interpret this as an estimate for the Dice of the different classes. This is not "
                               "exact.)")

        self.online_eval_foreground_dc = []
        self.online_eval_tp = []
        self.online_eval_fp = []
        self.online_eval_fn = []

    def save_checkpoint(self, fname, save_optimizer=True):
        super(nnUNetTrainer, self).save_checkpoint(fname, save_optimizer)
        info = OrderedDict()
        info['init'] = self.init_args
        info['name'] = self.__class__.__name__
        info['class'] = str(self.__class__)
        info['plans'] = self.plans

        write_pickle(info, fname + ".pkl")





