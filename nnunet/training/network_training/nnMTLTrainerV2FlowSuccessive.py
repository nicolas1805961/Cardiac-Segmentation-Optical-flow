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

from nnunet.evaluation.evaluator import NiftiEvaluator, aggregate_scores
from nnunet.inference.segmentation_export import save_segmentation_nifti_from_softmax
from nnunet.postprocessing.connected_components import determine_postprocessing, determine_postprocessing_no_metric

from tqdm import tqdm
from monai.transforms import KeepLargestConnectedComponent
from kornia.morphology import erosion
from ruamel.yaml import YAML
from torch.autograd import Variable
from nnunet.training.network_training.data_augmentation import Augmenter
from torchvision.transforms.functional import gaussian_blur
from torchvision.ops import masks_to_boxes
import torch.nn.functional as F
import random
from collections import OrderedDict
from multiprocessing import Pool
from nnunet.configuration import default_num_threads
from pickle import NONE
from traceback import print_tb
from typing import Tuple
import matplotlib
from datetime import datetime
from sklearn.metrics import accuracy_score
from tqdm import trange
import torch.backends.cudnn as cudnn
from _warnings import warn
from sklearn.model_selection import train_test_split
from scipy.signal import savgol_filter
from torchvision.utils import flow_to_image
from kornia.filters import spatial_gradient3d, spatial_gradient


import psutil
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

import cv2 as cv
import sys
from matplotlib import cm
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
from copy import copy
from time import time, sleep, strftime
#import yaml
import numpy as np
import torch
from nnunet.torchinfo.torchinfo.torchinfo import summary
from nnunet.lib.ssim import ssim
from nnunet.utilities.tensor_utilities import sum_tensor
from nnunet.training.data_augmentation.data_augmentation_moreDA import get_moreDA_augmentation_mtl
from nnunet.training.data_augmentation.data_augmentation_mtl import get_moreDA_augmentation_middle, get_moreDA_augmentation_middle_unlabeled
from nnunet.training.loss_functions.deep_supervision import MultipleOutputLoss2
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
from nnunet.network_architecture.generic_UNet import Generic_UNet
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.network_architecture.neural_network import SegmentationNetwork
from nnunet.training.data_augmentation.default_data_augmentation import default_2D_augmentation_params, \
    get_patch_size, default_3D_augmentation_params
from nnunet.training.dataloading.dataset_loading import DataLoaderPreprocessedAdjacent, DataLoaderAugment, DataLoaderFlowTrain5LibProgressiveNoSorting, DataLoaderFlowTrainRecursiveVideoLib, DataLoaderFlowTrainPredictionVal, DataLoaderFlowTrainPredictionValLib, DataLoader2DMiddleUnlabeled, unpack_dataset, DataLoaderVideoUnlabeled
from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer
from nnunet.utilities.nd_softmax import softmax_helper
from sklearn.model_selection import KFold
from torch import nn
from torch.cuda.amp import autocast
from nnunet.training.learning_rate.poly_lr import poly_lr
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.lib.training_utils import build_flow_model_successive, build_2d_model, build_flow_model_video, build_discriminator, read_config_video, read_config, build_discriminator, build_discriminator, build_video_model
from nnunet.lib.loss import SpatialSmoothingLoss, TemporalSmoothingLoss, AverageDistanceLoss
from pathlib import Path
from monai.losses import DiceFocalLoss, DiceLoss
from torch.utils.tensorboard import SummaryWriter
from nnunet.training.loss_functions.dice_loss import DC_and_CE_loss, DC_and_focal_loss, DC_and_topk_loss, DC_and_CE_loss_Weighted, SoftDiceLoss
from nnunet.training.loss_functions.crossentropy import RobustCrossEntropyLoss, WeightedRobustCrossEntropyLoss
from nnunet.training.dataloading.dataset_loading import DataLoaderPreprocessedValidation, DataLoaderPreprocessed, DataLoaderFlowACDCProgressiveAllDataFirst, DataLoaderFlowTrain5LibRegular, DataLoaderFlowACDCProgressiveAllDataAdjacent, DataLoaderFlowLibProgressiveAllDataFirst, DataLoaderFlowLibProgressiveAllDataAdjacent, DataLoaderFlowTrainPredictionVal, DataLoaderFlowTrain5Progressive, DataLoaderPreprocessedSupervised, DataLoaderFlowTrain5Lib, DataLoaderFlowTrainRecursiveVideo, DataLoaderFlowValidationOneStep
from nnunet.training.dataloading.dataset_loading import load_dataset, load_unlabeled_dataset
from nnunet.network_architecture.Optical_flow_model_successive import ModelWrap
from nnunet.lib.utils import RFR, ConvBlocks2DGroup, Resblock, LayerNorm, RFR_1d, Resblock1D, ConvBlocks1D
from nnunet.lib.loss import SeparabilityLoss, ContrastiveLoss, NCC
from nnunet.training.data_augmentation.cutmix import cutmix, batched_rand_bbox
import shutil
from nnunet.visualization.visualization import Visualizer
from nnunet.training.network_training.processor import Processor
from nnunet.network_architecture.integration import SpatialTransformer
import kornia

class nnMTLTrainerV2FlowSuccessive(nnUNetTrainer):
    """
    Info for Fabian: same as internal nnUNetTrainerV2_2
    """

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False, config=None):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)

        self.task = os.path.basename(dataset_directory)

        self.cropper_config = read_config(os.path.join(Path.cwd(), 'adversarial_acdc.yaml'), False, False)
        self.inference = False
        self.config = config
        self.video_length = self.config['video_length']
        self.crop = self.config['crop']
        self.feature_extractor = self.config['feature_extractor']

        if any([x in self.task for x in ['31', '35']]):
            self.crop_size = 128
            self.image_size = 224
            self.window_size = 7
            self.cropper_weights_folder_path = 'binary'
        else:
            self.crop_size = 192
            self.image_size = 384
            self.window_size = 8
            self.cropper_weights_folder_path = 'binary_lib'
            
        self.force_one_label = self.config['force_one_label']
        self.binary = False
        self.inference_mode = self.config['inference_mode']
        if self.inference_mode == 'sliding_window':
            assert self.video_length % 2 == 1

        self.nb_inter_frame = self.config['nb_interp_frame']
        self.dataloader_not_random = self.config['dataloader_not_random']
        #self.gradient_kernel_size = self.config['gradient_kernel_size']

        self.max_num_epochs = self.config['max_num_epochs']
        self.log_images = self.config['log_images']
        self.initial_lr = self.config['initial_lr']
        self.weight_decay = self.config['weight_decay']
        self.deep_supervision_scales = None
        self.ds_loss_weights = None
        self.use_progress_bar=True
        self.motion_estimation = SpatialTransformer((self.crop_size, self.crop_size)).to('cuda:0')
        self.padding = self.config['padding']
        self.do_data_aug = self.config['do_data_aug']
        self.all_data_lib = self.config['all_data_lib']
        self.dataloader_modality = self.config['dataloader_modality']
        self.training_modality = self.config['training_modality']
        self.motion_from_ed = self.config['motion_from_ed']
        self.backward = self.config['backward']
        self.segmentation = self.config['segmentation']
        self.no_error = self.config['no_error']
        self.binary_distance_loss = self.config['binary_distance_loss']
        self.binary_distance_input = self.config['binary_distance_input']
        #self.logits_input = self.config['logits_input']
        #self.pretrained_2d_folder_path = self.config['pretrained_2d_folder_path']
        #self.pretrained_config = read_config(os.path.join(Path.cwd(), self.config['pretrained_2d_folder_path'], 'config.yaml'), False, False)

        if self.dataloader_modality == 'regular':
            assert self.config['global_motion_forward_loss_weight'] == 0.0

        #self.fine_tuning = True if self.video_length > 2 and not self.inference else False
        self.fine_tuning = False

        self.deep_supervision = self.config['deep_supervision']
        self.segmentation_loss_weight = self.config['segmentation_loss_weight']
        self.regularization_weight_xy_local = self.config['regularization_weight_xy_local']
        self.regularization_weight_xy_global = self.config['regularization_weight_xy_global']
        self.regularization_weight_z = self.config['regularization_weight_z']
        self.adversarial_weight = self.config['adversarial_weight']
        self.do_adv = self.config['do_adv']
        self.distance_map_power = self.config['distance_map_power']
        self.start_es = self.config['start_es']
        self.supervised = self.config['supervised']

        if self.video_length == 2:
            assert not self.supervised

        self.discriminator_lr = self.config['discriminator_lr']
        self.discriminator_decay = self.config['discriminator_decay']

        weights = np.array([1 / (2 ** i) for i in range(len(self.config['conv_depth']))])
        self.ds_weights = weights / weights.sum()

        self.iter_nb = 0
        self.epoch_iter_nb = 0

        self.val_loss = []
        self.train_loss = []
        
        self.image_flow_loss_weight_global = self.config['image_flow_loss_weight_global']
        self.image_flow_loss_weight_local = self.config['image_flow_loss_weight_local']
        self.global_motion_forward_loss_weight = self.config['global_motion_forward_loss_weight']

        #timestr = strftime("%Y-%m-%d_%HH%M_%S")
        timestr = datetime.now().strftime("%Y-%m-%d_%HH%M_%Ss_%f")
        self.log_dir = os.path.join(copy(self.output_folder), timestr)
        self.writer = SummaryWriter(log_dir=self.log_dir)

        self.one_to_all = self.config['one_to_all']
        self.all_to_all = self.config['all_to_all']

        if self.log_images:
            self.vis = Visualizer(unlabeled=False,
                                    adversarial_loss=False,
                                    registered_seg=False,
                                    writer=self.writer,
                                    crop_size=self.crop_size)
            
        self.output_folder = self.log_dir

        #if output_folder.count(os.sep) < 2:
        #    self.output_folder = output_folder
        #else:
        #    self.output_folder = self.log_dir

        self.setup_loss_functions()
        
        self.pin_memory = True

        #self.sim_l2_md_list = []
        self.table = self.initialize_table()

        self.loss_data = self.setup_loss_data()

        self.loss_function = self.choose_loss()

    def setup_loss_data(self):
        loss_data = {}
        #loss_data = {'segmentation': [self.segmentation_loss_weight, float('nan')]}
        
        if self.video_length > 2:
            loss_data['local_image_flow'] = [self.image_flow_loss_weight_local, float('nan')]
            loss_data['local_flow_regularization'] = [self.regularization_weight_xy_local, float('nan')]
            loss_data['cumulated_flow_regularization'] = [self.regularization_weight_xy_global, float('nan')]
            loss_data['intermediate'] = [self.image_flow_loss_weight_global, float('nan')]
        else:
            loss_data['global_flow_regularization'] = [self.regularization_weight_xy_global, float('nan')]
            loss_data['global_image_flow'] = [self.image_flow_loss_weight_global, float('nan')]
        
        if self.dataloader_modality == 'other':
            loss_data['global_motion_forward'] = [self.global_motion_forward_loss_weight, float('nan')]
            if self.supervised:
                loss_data['local_motion_forward'] = [self.global_motion_forward_loss_weight, float('nan')] 
        #loss_data['global_motion_forward_semi_supervised'] = [self.semi_supervised_forward_loss_weight, float('nan')]
        if self.do_adv:
            loss_data['adversarial'] = [self.adversarial_weight, float('nan')]
        return loss_data
    
    def setup_loss_functions(self):
        if self.config['loss'] == 'focal_and_dice2':
            self.segmentation_loss = DiceFocalLoss(include_background=False, focal_weight=loss_weights[1:] if not self.config['binary'] else None, softmax=True, to_onehot_y=True)
        elif self.config['loss'] == 'focal_and_dice':
            self.segmentation_loss = DC_and_focal_loss({'batch_dice': self.batch_dice, 'smooth': 1e-5, 'do_bg': False}, {'apply_nonlin': nn.Softmax(dim=1), 'alpha':None, 'gamma':2, 'smooth':1e-5})
        elif self.config['loss'] == 'topk_and_dice':
            self.segmentation_loss = DC_and_topk_loss({'batch_dice': self.batch_dice, 'smooth': 1e-5, 'do_bg': False}, {'k': 10}, ce_weight=0.5, dc_weight=0.5)
        elif self.config['loss'] == 'ce_and_dice':
            self.segmentation_loss = DC_and_CE_loss({'batch_dice': True, 'smooth': 1e-5, 'do_bg': False}, {})
        elif self.config['loss'] == 'ce':
            self.segmentation_loss = RobustCrossEntropyLoss(weight=loss_weights)
        elif self.config['loss'] == 'dice':
            self.segmentation_loss = SoftDiceLoss(apply_nonlin=softmax_helper, batch_dice=True, smooth=1e-5, do_bg=False)
        #self.image_flow_loss = ImageFlowLoss(alpha=1.0, beta=1.0)
        
        self.seg_flow_loss = self.segmentation_loss
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.adv_loss = nn.BCELoss()
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        if self.config['registration_loss'] == 'ncc':
            self.spatial_smoothing_loss = SpatialSmoothingLoss(reduction=None)
            self.temporal_smoothing_loss = TemporalSmoothingLoss(reduction=None)
            self.registration_loss = NCC(reduction=None)
        elif self.config['registration_loss'] == 'mse':
            self.registration_loss = nn.MSELoss()


    def initialize_table(self):
        table = {}
        table['seg'] = {'foreground_dc': [], 'tp': [], 'fp': [], 'fn': []}
        table['flow'] = {'foreground_dc': [], 'tp': [], 'fp': [], 'fn': []}
        return table

    def initialize(self, training=True, force_load_plans=False):
        """
        - replaced get_default_augmentation with get_moreDA_augmentation
        - enforce to only run this code once
        - loss function wrapper for deep supervision

        :param training:
        :param force_load_plans:
        :return:
        """
        if not self.was_initialized:
            maybe_mkdir_p(self.output_folder)

            if force_load_plans or (self.plans is None):
                self.load_plans_file()

            self.process_plans(self.plans)

            self.setup_DA_params()

            ################# Here we wrap the loss for deep supervision ############
            # we need to know the number of outputs of the network
            net_numpool = len(self.net_num_pool_op_kernel_sizes)

            # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
            # this gives higher resolution outputs more weight in the loss
            weights = np.array([1 / (2 ** i) for i in range(net_numpool)])

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            #mask = np.array([True] + [True if i < net_numpool - 1 else False for i in range(1, net_numpool)])
            #weights[~mask] = 0
            weights = weights / weights.sum()
            #self.ds_loss_weights = weights
            # now wrap the loss
            seg_weights = weights

            #if self.deep_supervision:
            #    self.segmentation_loss = MultipleOutputLoss2(self.segmentation_loss, seg_weights)
            #    self.image_flow_loss = MultipleOutputLoss2(self.image_flow_loss, seg_weights)
            #    self.seg_flow_loss = MultipleOutputLoss2(self.seg_flow_loss, seg_weights)

            ################# END ###################

            self.folder_with_preprocessed_data = join(self.dataset_directory, self.plans['data_identifier'] +
                                                      "_stage%d" % self.stage)
            self.initialize_network()
            self.initialize_optimizer_and_scheduler()
            self.dl_tr, self.dl_val, self.dl_un_tr, self.dl_un_val, self.dl_overfitting = self.get_basic_generators()

            if training:
                if self.unpack_data:
                    print("unpacking dataset")
                    unpack_dataset(self.folder_with_preprocessed_data)
                    print("done")
                else:
                    print(
                        "INFO: Not unpacking data! Training may be slow due to that. Pray you are not using 2d or you "
                        "will wait all winter for your model to finish!")
                
                self.print_to_log_file("TRAINING KEYS:\n %s" % (str(self.dataset_tr.keys())),
                                       also_print_to_console=False)
                self.print_to_log_file("VALIDATION KEYS:\n %s" % (str(self.dataset_val.keys())),
                                       also_print_to_console=False)
                if self.image_size == 224:
                    self.print_to_log_file("UNLABELED TRAINING KEYS:\n %s" % (str(self.dataset_un_tr.keys())),
                                       also_print_to_console=False)
                    self.print_to_log_file("UNLABELED VALIDATION KEYS:\n %s" % (str(self.dataset_un_val.keys())),
                                       also_print_to_console=False)
            else:
                pass


            assert isinstance(self.network, (SegmentationNetwork, nn.DataParallel))
        else:
            self.print_to_log_file('self.was_initialized is True, not running self.initialize again')
        self.was_initialized = True

    def choose_loss(self):
        if self.supervised:
            return self.compute_losses_recursive_2_supervised
        else:
            return self.compute_losses_recursive_2
    
    def count_parameters(self, config, models):
        if not self.inference:
            yaml = YAML()
            with open(os.path.join(self.log_dir, 'config.yaml'), 'wb') as f:
                yaml.dump(config, f)
                self.print_to_log_file(config, also_print_to_console=False)
        
        #for name, param in models['temporal_model'][0].named_parameters():
        #    if param.requires_grad:
        #        print(name)
        #        print(param.numel())

        params_sum = 0
        for k, v in models.items():
            nb_params =  sum(p.numel() for p in v[0].parameters() if p.requires_grad)
            self.print_to_log_file(f"{k} has {nb_params:,} parameters")
            if v[0].training:
                params_sum += nb_params

            model_stats = summary(v[0], input_data=v[1], 
                                col_names=["input_size", "output_size", "num_params", "mult_adds"], 
                                col_width=16,
                                verbose=0)
            model_stats.formatting.verbose = 1
            self.print_to_log_file(model_stats, also_print_to_console=False)
        
        self.print_to_log_file("The Whole model has", "{:,}".format(params_sum), "parameters")
    
    def get_conv_layer(self, config):
        if config['conv_layer'] == 'RFR':
            conv_layer = RFR
            conv_layer_1d = RFR_1d
        elif config['conv_layer'] == 'resblock':
            conv_layer = Resblock
            conv_layer_1d = Resblock1D
        else:
            conv_layer = ConvBlocks2DGroup
            conv_layer_1d = ConvBlocks1D
        return conv_layer, conv_layer_1d

    def freeze_batchnorm_layers(self, model):
        for module in model.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm3d)):
                if hasattr(module, 'weight'):
                    module.weight.requires_grad_(False)
                if hasattr(module, 'bias'):
                    module.bias.requires_grad_(False)
                module.eval()

    def load_video_weights(self, model, file_name):
        weight_path = os.path.join(self.video_weights_folder_path, file_name)
        loaded_state_dict = torch.load(weight_path)['state_dict']
        #print(loaded_state_dict.keys())
        current_model_dict = model.state_dict()
        #print(current_model_dict.keys())

        new_state_dict = copy(current_model_dict)
        for k, v in loaded_state_dict.items():
            for module_name, module in model.named_modules():
                if list(module.children()) == []:
                    if self.load_only_batchnorm:
                        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm3d)) and module_name in k:
                            new_state_dict[k] = v
                    else:
                        if module_name in k:
                            new_state_dict[k] = v
                            #if new_state_dict[k].shape != v.shape:
                            #    print(k)
                            #else:
                            #    new_state_dict[k] = v

        try:
            missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
        except Exception as e:
            self.print_to_log_file(e)

        self.freeze_batchnorm_layers(model)

    def initialize_network(self):
        """
        - momentum 0.99
        - SGD instead of Adam
        - self.lr_scheduler = None because we do poly_lr
        - deep supervision = True
        - i am sure I forgot something here

        Known issue: forgot to set neg_slope=0 in InitWeights_He; should not make a difference though
        :return:
        """
        models = {}

        num_classes = 3 if '029' in self.dataset_directory else 4

        wanted_norm = self.config['norm']
        if wanted_norm == 'batchnorm':
            norm_2d = nn.BatchNorm2d
            norm_1d = nn.BatchNorm1d
        elif wanted_norm == 'instancenorm':
            norm_2d = nn.InstanceNorm2d
            norm_1d = nn.InstanceNorm1d
        
        conv_layer_2d, conv_layer_1d = self.get_conv_layer(self.config)

        if self.do_adv:
            self.discriminator = build_discriminator(config=self.config, conv_layer=conv_layer_2d, norm=norm_2d, image_size=self.crop_size)
            if self.fine_tuning:
                self.print_to_log_file("Loading weights from pretrained discriminator")
                self.load_video_weights(self.discriminator, file_name='discriminator_final_checkpoint.model')
            discriminator_input = torch.randn(self.config['batch_size'], 4, self.crop_size, self.crop_size)
            models['discriminator'] = (self.discriminator, discriminator_input)

        in_shape_crop = torch.randn(self.config['batch_size'], 1, self.image_size, self.image_size)
        cropping_conv_layer, _ = self.get_conv_layer(self.cropper_config)
        cropping_network = build_2d_model(self.cropper_config, conv_layer=cropping_conv_layer, norm=getattr(torch.nn, self.cropper_config['norm']), log_function=self.print_to_log_file, image_size=self.image_size, window_size=self.window_size, middle=False, num_classes=2, processor=False)
        cropping_network.load_state_dict(torch.load(os.path.join(self.cropper_weights_folder_path, 'model_final_checkpoint.model'))['state_dict'], strict=True)
        cropping_network.eval()
        cropping_network.do_ds = False
        models['cropping_model'] = (cropping_network, in_shape_crop)


        network_1 = build_flow_model_successive(self.config, image_size=self.crop_size, log_function=self.print_to_log_file, nb_channels=1, backward=False, segmentation=self.segmentation)
        if self.video_length > 2:
            nb_channels = 6 if not self.no_error else 4
            network_2 = build_flow_model_successive(self.config, image_size=self.crop_size, log_function=self.print_to_log_file, nb_channels=nb_channels, backward=self.backward, segmentation=False)
        else:
            network_2 = None
        self.network = ModelWrap(model1=network_1, model2=network_2, do_ds=self.deep_supervision, motion_from_ed=self.motion_from_ed, backward=self.backward, segmentation=self.segmentation, no_error=self.no_error, use_label=False)

        unlabeled_input_data = torch.randn(self.video_length, self.config['batch_size'], 1, self.crop_size, self.crop_size)
        #in_shape_distance = torch.randn(self.video_length - 1, self.config['batch_size'])
        models['temporal_model'] = (self.network, unlabeled_input_data)

        #if self.fine_tuning:
        #    self.print_to_log_file("Loading weights from pretrained network")
        #    self.load_video_weights(self.network, file_name='model_final_checkpoint.model')
        
        

        self.processor = Processor(crop_size=self.crop_size, image_size=self.image_size, cropping_network=cropping_network)

        self.count_parameters(self.config, models)

        #nb_inputs = 2 if self.middle else 1
        #model_input_size = [(self.config['batch_size'], 1, self.image_size, self.image_size)] * nb_inputs

        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper

    def get_optimizer_scheduler(self, net, lr, decay):
        if self.feature_extractor:
            params_to_update = []
            for name, param in net.named_parameters():
                if param.requires_grad == True:
                    params_to_update.append(param)
        else:
            params_to_update = net.parameters()

        if self.config['optimizer'] == 'sgd':
            optimizer = torch.optim.SGD(params_to_update, lr, weight_decay=decay, momentum=0.99, nesterov=True)
        elif self.config['optimizer'] == 'adam':
            optimizer = torch.optim.AdamW(params_to_update, lr=lr, weight_decay=decay)

        if self.config['scheduler'] == 'cosine':
            lr_scheduler = CosineAnnealingLR(optimizer, T_max=self.max_num_epochs)
            #self.warmup = LinearLR(optimizer=self.optimizer, start_factor=0.1, end_factor=1, total_iters=self.num_batches_per_epoch)
            #self.lr_scheduler = SequentialLR(optimizer=self.optimizer, schedulers=[warmup, cosine_scheduler], milestones=[1])
        else:
            lr_scheduler = None

        return optimizer, lr_scheduler

    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, "self.initialize_network must be called first"
        self.optimizer, self.lr_scheduler = self.get_optimizer_scheduler(net=self.network, lr=self.initial_lr, decay=self.weight_decay)

        if self.do_adv:
            self.discriminator_optimizer, self.discriminator_scheduler = self.get_optimizer_scheduler(net=self.discriminator, lr=self.discriminator_lr, decay=self.discriminator_decay)

    def compute_dice(self, target, num_classes, output_seg, key):
        axes = tuple(range(1, len(target.shape)))
        tp_hard = torch.zeros((target.shape[0], num_classes - 1)).to(output_seg.device.index)
        fp_hard = torch.zeros((target.shape[0], num_classes - 1)).to(output_seg.device.index)
        fn_hard = torch.zeros((target.shape[0], num_classes - 1)).to(output_seg.device.index)
        for c in range(1, num_classes):
            tp_hard[:, c - 1] = sum_tensor((output_seg == c).float() * (target == c).float(), axes=axes)
            fp_hard[:, c - 1] = sum_tensor((output_seg == c).float() * (target != c).float(), axes=axes)
            fn_hard[:, c - 1] = sum_tensor((output_seg != c).float() * (target == c).float(), axes=axes)

        seg_dice = ((2 * tp_hard) / (2 * tp_hard + fp_hard + fn_hard + 1e-8))

        tp_hard = tp_hard.sum(0, keepdim=False).detach().cpu().numpy()
        fp_hard = fp_hard.sum(0, keepdim=False).detach().cpu().numpy()
        fn_hard = fn_hard.sum(0, keepdim=False).detach().cpu().numpy()

        self.table[key]['foreground_dc'].append(list((2 * tp_hard) / (2 * tp_hard + fp_hard + fn_hard + 1e-8)))
        self.table[key]['tp'].append(list(tp_hard))
        self.table[key]['fp'].append(list(fp_hard))
        self.table[key]['fn'].append(list(fn_hard))

        #self.online_eval_foreground_dc.append(list((2 * tp_hard) / (2 * tp_hard + fp_hard + fn_hard + 1e-8)))
        #self.online_eval_tp.append(list(tp_hard))
        #self.online_eval_fp.append(list(fp_hard))
        #self.online_eval_fn.append(list(fn_hard))
            
        return seg_dice

    def get_gradient_images(self, unlabeled, indices, task):
        t, b, c, y, x = indices
        with torch.enable_grad():
            self.network.zero_grad()
            unlabeled.requires_grad = True
            output = self.network(unlabeled)
            if self.deep_supervision:
                prediction = output[task][0]
            else:
                prediction = output[task]
            out = prediction[t, b, c, y, x]
            out.backward()

            gradient_image_unlabeled = torch.abs(unlabeled.grad)
            gradient_image_unlabeled = gradient_image_unlabeled[:, b, 0, :, :]

            unlabeled = unlabeled[:, b, 0]
            return gradient_image_unlabeled, unlabeled

    
    def start_online_evaluation(self, out,
                                    unlabeled,
                                    target,
                                    gradient_image_unlabeled,
                                    gradient_x_unlabeled,
                                    coords,
                                    seg_dice,
                                    target_mask):
        """
            gradient_image: T, H, W
            gradient_x: T, H, W
            """
        
        unlabeled = unlabeled.permute(1, 0, 2, 3, 4).contiguous() # B, T, 1, H, W
        target = target.permute(1, 0, 2, 3, 4).contiguous() # B, T, 1, H, W
        forward_flow = out['global_motion_forward'].permute(1, 0, 2, 3, 4).contiguous() # B, T, 2, H, W
        target_mask = target_mask.permute(1, 0).contiguous() # B, T
        seg_registered_forward = out['seg_registered_forward'].permute(1, 0, 2, 3).contiguous() # B, T, H, W

        with torch.no_grad():

            if self.log_images:
                for b in range(len(target)):
                    
                    motion_flow = flow_to_image(forward_flow[b]) # 3, H, W
                    #registered_seg = out['registered_seg'][:, b, :]
                    #registered_seg = torch.softmax(registered_seg, dim=1)
                    #registered_seg = torch.argmax(registered_seg, dim=1)
                    
                    #warping_distance = self.l1_loss(out['registered_input_u_to_l'][:, b, 0], labeled[None].repeat(self.video_length, 1, 1, 1, 1)[:, b, 0]).mean()

                    current_x = unlabeled[b, target_mask[b], 0][0]
                    current_target = target[b, target_mask[b], 0][0]

                    #if not self.one_to_all:
                    #    current_pred_registered_backward = pred_registered_backward[b, :]
                    #else:
                        #current_pred_registered_backward = seg_registered_forward[b, :]

                    current_seg_registered_forward = seg_registered_forward[b, :]

                    #self.vis.set_up_image_seg_best(seg_dice=seg_dice[b].mean(), gt=current_target, pred=current_pred, x=current_x)
                    #self.vis.set_up_image_seg_worst(seg_dice=seg_dice[b].mean(), gt=current_target, pred=current_pred, x=current_x)
                    self.vis.set_up_image_gradient(unlabeled_gradient=gradient_image_unlabeled,
                                                        unlabeled_x=gradient_x_unlabeled,
                                                        gradient_coords=coords)
                    #self.vis.set_up_image_gradient_seg(unlabeled_gradient=gradient_image_seg,
                    #                                    unlabeled_x=gradient_x_seg,
                    #                                    gradient_coords=coords_seg)
                    self.vis.set_up_image_flow(seg_dice=seg_dice[b].mean(), 
                                                moving=unlabeled[b, :-1, 0], 
                                                registered_input=out['registered_input_forward'][:, b, 0],
                                                registered_seg=current_seg_registered_forward, #current_seg_registered_forward,
                                                target=current_target,
                                                motion_flow=motion_flow)
                    #self.vis.set_up_image_seg_registered_long(seg_dice=seg_dice[b].mean(), 
                    #                                     seg_registered_long=out['seg_registered_forward_long'][b, 0],
                    #                                     target=current_target)
                    
                    self.vis.set_up_image_seg_sequence(seg_dice=seg_dice[b].mean(), gt=current_target, pred=current_seg_registered_forward, x=unlabeled[b, :, 0])
                    #self.vis.set_up_image_seg_sequence_mask(seg_dice=seg_dice[b].mean(), gt=current_target, pred=output_seg_all_mask[:, b], x=with_mask[:, b, 0])
                
    
    def get_dc_per_class(self, key):
        self.table[key]['tp'] = np.sum(self.table[key]['tp'], 0)
        self.table[key]['fp'] = np.sum(self.table[key]['fp'], 0)
        self.table[key]['fn'] = np.sum(self.table[key]['fn'], 0)

        global_dc_per_class = [i for i in [2 * i / (2 * i + j + k) for i, j, k in
                                           zip(self.table[key]['tp'], self.table[key]['fp'], self.table[key]['fn'])]
                               if not np.isnan(i)]
        
        return global_dc_per_class
    
    def log_dice(self, key):
        global_dc_per_class_seg = self.get_dc_per_class(key)
        self.writer.add_scalar('Epoch/' + key.title() + ' Dice', torch.tensor(global_dc_per_class_seg).mean().item(), self.epoch)
        self.all_val_eval_metrics.append(np.mean(global_dc_per_class_seg))
        self.print_to_log_file("(interpret this as an estimate for the " + key.title() + " Dice of the different classes. This is not exact.)")
        self.print_to_log_file("Average global " + key.title() + " foreground Dice:", [np.round(i, 4) for i in global_dc_per_class_seg])
        class_dice = {'RV': global_dc_per_class_seg[0], 'MYO': global_dc_per_class_seg[1], 'LV': global_dc_per_class_seg[2]}
        self.writer.add_scalars('Epoch/Class ' + key.title() + ' Dice', class_dice, self.epoch)   

    
    def log_losses(self):
        overfit_data = {'Train': torch.tensor(self.train_loss).mean().item(), 'Val': torch.tensor(self.val_loss).mean().item()}
        self.writer.add_scalars('Epoch/Train_vs_val_loss', overfit_data, self.epoch)
        self.val_loss = []
        self.train_loss = []

        
    def finish_online_evaluation(self):
        #self.online_eval_tp = np.sum(self.online_eval_tp, 0)
        #self.online_eval_fp = np.sum(self.online_eval_fp, 0)
        #self.online_eval_fn = np.sum(self.online_eval_fn, 0)

        #global_dc_per_class = [i for i in [2 * i / (2 * i + j + k) for i, j, k in
        #                                   zip(self.online_eval_tp, self.online_eval_fp, self.online_eval_fn)]
        #                       if not np.isnan(i)]
                
        self.log_dice('flow') 
        #self.log_dice('seg') 

        if self.log_images:
            cmap, norm = self.vis.get_custom_colormap()
            self.vis.log_sequence_seg_images(colormap_seg=cmap, norm=norm, epoch=self.epoch)
            #self.vis.log_sequence_seg_mask_images(colormap_seg=cmap, norm=norm, epoch=self.epoch)
            #self.vis.log_best_seg_images(colormap=cmap, norm=norm, epoch=self.epoch)
            #self.vis.log_worst_seg_images(colormap=cmap, norm=norm, epoch=self.epoch)
            self.vis.log_gradient_images(colormap=cm.plasma, epoch=self.epoch)
            #self.vis.log_gradient_images_seg(colormap=cm.plasma, epoch=self.epoch)
            #self.vis.log_seg_registered_long_images(colormap=cmap, norm=norm, epoch=self.epoch)
            #self.vis.log_worst_gradient_images(colormap=cm.plasma, epoch=self.epoch)
            #if not self.one_to_all:
            #    self.vis.log_long_registered_images(epoch=self.epoch)
            self.vis.log_motion_images(epoch=self.epoch, colormap_seg=cmap, norm=norm)
            #self.vis.log_weights_images(colormap=cm.plasma, epoch=self.epoch)
            #self.vis.log_slot_images(colormap=cmap, epoch=self.epoch)
            #self.vis.log_target_images(colormap=cm.plasma, epoch=self.epoch)
                #self.vis.log_deformable_attention_images(colormap=cm.plasma, epoch=self.epoch)
            #    self.vis.log_theta_images(epoch=self.epoch, area_size=self.area_size)

        #self.online_eval_foreground_dc = []
        #self.online_eval_tp = []
        #self.online_eval_fp = []
        #self.online_eval_fn = []
        if self.log_images:
            self.vis.reset()
        self.table = self.initialize_table()
        #self.sim_l2_md_list = []

    def sample_indices(self, cropped_target):
        T, B, C, H, W = cropped_target.shape
        if torch.count_nonzero(cropped_target) == 0:
            t = random.randint(0, T - 2)
            b = random.randint(0, B - 1)
            c = random.randint(0, C - 1)
            y = random.randint(0, H - 1)
            x = random.randint(0, W - 1)
        else:
            indices = torch.nonzero(cropped_target > 0)
            mask = indices[:, 0] < T - 1
            indices = indices[mask]
            high = max(0, indices.shape[0] - 1)
            idx = random.randint(0, high)
            indices = indices[idx]
            t = indices[0].item()
            b = indices[1].item()
            c = indices[2].item()
            y = indices[3].item()
            x = indices[4].item()
        return (t, b, c, y, x)

    
    def sample_indices_da(self, attention_weights):
        T, B, Z = attention_weights.shape[:3]
        t = random.randint(0, T - 1)
        b = random.randint(0, B - 1)
        z = random.randint(0, Z - 1)
        x = random.randint(0, self.area_size[-1] - 1)
        y = random.randint(0, self.area_size[-1] - 1)
        return (t, b, z, x, y)

    
    def setup_deformable_attention(self, sampling_locations, attention_weights, data, indices, theta_coords):
        # sampling_locations = T, B, n_zones, T, n_heads, area_size, area_size, n_points, 2
        # attention_weights = T, B, n_zones, T, n_heads, area_size, area_size, n_points
        # theta_coords = T, B, n_zones, 4

        t, b, z, x, y = indices

        sampling_locations = sampling_locations[t, b] # n_zones, T, n_heads, area_size, area_size, n_points, 2
        attention_weights = attention_weights[t, b] # n_zones, T, n_heads, area_size, area_size, n_points
        theta_coords = theta_coords[t, b] # n_zones, 4
        data = data[:, b] # T, 1, H, W

        sampling_locations = sampling_locations.permute(0, 1, 3, 4, 2, 5, 6)
        sampling_locations = torch.flatten(sampling_locations, start_dim=4, end_dim=5) # n_zones, T, area_size, area_size, -1, 2

        attention_weights = attention_weights.permute(0, 1, 3, 4, 2, 5)
        attention_weights = torch.flatten(attention_weights, start_dim=4, end_dim=5) # n_zones, T, area_size, area_size, -1

        return sampling_locations, attention_weights, data, theta_coords
    
    def run_online_evaluation(self, out, unlabeled, target, seg_dice, target_mask):
        """
        due to deep supervision the return value and the reference are now lists of tensors. We only need the full
        resolution output because this is what we are interested in in the end. The others are ignored
        :param output:
        :param target:
        :return:
        """
        gradient_image = None
        gradient_x = None
        coords = None
        coords_seg = None
        indices_da = None

        indices = self.sample_indices(target[target_mask][0][None].repeat(len(unlabeled), 1, 1, 1, 1))
        #indices_seg = self.sample_indices(target[target_mask][0][None].repeat(len(unlabeled), 1, 1, 1, 1))
        gradient_image_unlabeled, gradient_x_unlabeled = self.get_gradient_images(unlabeled, indices, 'global_motion_forward')
        #gradient_image_seg, gradient_x_seg = self.get_gradient_images(unlabeled, indices_seg, 'seg')
        coords = (indices[0], indices[4], indices[3]) # (t, x, y)
        #coords_seg = (indices_seg[0], indices_seg[4], indices_seg[3]) # (t, x, y)

        return self.start_online_evaluation(out=out,
                                            unlabeled=unlabeled,
                                            target=target,
                                            gradient_image_unlabeled=gradient_image_unlabeled,
                                            gradient_x_unlabeled=gradient_x_unlabeled,
                                            coords=coords,
                                            seg_dice=seg_dice,
                                            target_mask=target_mask)
    




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

        #newpath_strain_radial = join(self.output_folder, 'Strain', 'Radial')
        #if not os.path.exists(newpath_strain_radial):
        #    os.makedirs(newpath_strain_radial)
        
        to_validate_registered_list = []

        list_of_keys = list(self.dataset_val.keys())
        patient_id_list = np.unique([x[:10] for x in list_of_keys])
        metadata_list = []
        metadata_list_registered = []
        #strain_results = {"all": [], "mean_lv_tangential": None}
        for patient_id in tqdm(patient_id_list[:1]):
            phase_list = [x for x in list_of_keys if patient_id in x]
            phase_list = np.array(sorted(phase_list, key=lambda x: int(x[16:18])))
            phase_list = np.array(phase_list)
            #print(phase_list[0])
            #print(self.dataset)
            #print(self.dataset[phase_list[0]])
            properties = load_pickle(self.dataset[phase_list[0]]['properties_file'])

            ed_idx = np.rint(np.array(properties['ed_number'])).astype(np.uint8) % len(phase_list)
            es_idx = np.rint(np.array(properties['es_number'])).astype(np.uint8) % len(phase_list)

            unlabeled = np.full(shape=((len(phase_list), 1) + properties['size_after_resampling']), fill_value=np.nan)
            target = np.full(shape=((len(phase_list), 1) + properties['size_after_resampling']), fill_value=np.nan)

            if overwrite or (not isfile(join(output_folder, phase_list[0] + ".nii.gz"))) or (save_softmax and not isfile(join(output_folder, phase_list[0] + ".npz"))):
                for idx, frame in enumerate(phase_list):
                    current_data = np.load(self.dataset_val[frame]['data_file'])['data']
                    target[idx] = current_data[1]
                    target[idx][target[idx] == -1] = 0

                frame_indices = np.arange(len(phase_list))

                before_where = np.argwhere(frame_indices < ed_idx).reshape(-1,)
                after_where = np.argwhere(frame_indices >= ed_idx).reshape(-1,)

                all_where = np.concatenate([after_where, before_where])

                frame_indices = frame_indices[all_where]
                unlabeled = unlabeled[frame_indices]

                assert frame_indices[0] == ed_idx


                #matplotlib.use('QtAgg')
                #print(ed_idx)
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
                raw_flow = ret[4] # T, C, depth, H, W

                assert len(softmax_pred) == len(flow_pred) == len(registered_pred)

                sorted_where = np.argsort(all_where)
                softmax_pred = softmax_pred[sorted_where]
                flow_pred = flow_pred[sorted_where]
                registered_pred = registered_pred[sorted_where]
                raw_flow = raw_flow[sorted_where]

                assert len(softmax_pred) == len(flow_pred) == len(phase_list)

                for t in range(len(softmax_pred)):
                    properties = load_pickle(self.dataset[phase_list[t]]['properties_file'])
                    fname = properties['list_of_data_files'][0].split(os.sep)[-1][:-12]

                    current_softmax_pred = softmax_pred[t]
                    current_softmax_pred = current_softmax_pred.transpose([0] + [i + 1 for i in self.transpose_backward])

                    current_raw_flow = raw_flow[t]
                    current_raw_flow = current_raw_flow.transpose([0] + [i + 1 for i in self.transpose_backward])

                    if save_softmax:
                        softmax_fname = join(output_folder, fname + ".npz")
                    else:
                        softmax_fname = None

                    splitted = fname.split('frame')
                    from_nb = 0
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
                                                            interpolation_order_z),
                                                            )
                                                            )
                                )
                    
                    results.append(export_pool.starmap_async(save_segmentation_nifti_from_softmax,
                                                            ((current_softmax_pred, join(newpath_seg, fname + ".nii.gz"),
                                                            properties, interpolation_order, self.regions_class_order,
                                                            None, None,
                                                            softmax_fname, None, force_separate_z,
                                                            interpolation_order_z, False, current_flow, flow_path,
                                                            current_registered, registered_path, current_raw_flow),
                                                            )
                                                            )
                                )
                    
                    if t > 0:
                        to_validate_registered_list.append(fname + ".nii.gz")
                        metadata_list_registered.append(self.create_metadata_dict(properties))
                        pred_gt_tuples_register.append([registered_path,
                                                join(self.gt_niftis_folder, fname + ".nii.gz")])
                    
                    metadata_list.append(self.create_metadata_dict(properties))
                    pred_gt_tuples.append([join(newpath_seg, fname + ".nii.gz"),
                                            join(self.gt_niftis_folder, fname + ".nii.gz")])

        _ = [i.get() for i in results]
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
            
            #base_segmentation = join(self.output_folder, 'Segmentation')
            #determine_postprocessing(base_segmentation, self.gt_niftis_folder, validation_folder_name,
            #                         final_subf_name=validation_folder_name + "_postprocessed", debug=debug, log_function=log_function,
            #                         metadata_list=metadata_list, to_validate_list=None)
            #
            #base_registered = join(self.output_folder, 'Registered')
            #determine_postprocessing(base_registered, self.gt_niftis_folder, validation_folder_name,
            #                         final_subf_name=validation_folder_name + "_postprocessed", debug=debug, log_function=log_function,
            #                         metadata_list=metadata_list_registered, to_validate_list=to_validate_registered_list)
            
            base_segmentation = join(self.output_folder, 'Segmentation')
            determine_postprocessing_no_metric(base_segmentation, validation_folder_name,
                                     final_subf_name=validation_folder_name + "_postprocessed", debug=debug, log_function=log_function)
            
            base_registered = join(self.output_folder, 'Registered')
            determine_postprocessing_no_metric(base_registered, validation_folder_name,
                                     final_subf_name=validation_folder_name + "_postprocessed", debug=debug, log_function=log_function)
            
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




    

    def validate(self, do_mirroring: bool = True, use_sliding_window: bool = True,
                 step_size: float = 0.5, save_softmax: bool = True, use_gaussian: bool = True, overwrite: bool = True,
                 validation_folder_name: str = 'validation_raw', debug: bool = False, all_in_gpu: bool = False,
                 segmentation_export_kwargs: dict = None, run_postprocessing_on_folds: bool = True, output_folder=None):
        """
        We need to wrap this because we need to enforce self.network.do_ds = False for prediction
        """ 
        ds = self.network.do_ds
        self.network.do_ds = False
        
        if self.inference_mode == 'one_step':
            if '032' in self.task:
                ret = self.validate_flow_one_step_lib(processor=self.processor, log_function=self.print_to_log_file, step=1, do_mirroring=do_mirroring, use_sliding_window=use_sliding_window, step_size=step_size,
                            save_softmax=save_softmax, use_gaussian=use_gaussian,
                            overwrite=overwrite, validation_folder_name=validation_folder_name,
                            all_in_gpu=all_in_gpu, segmentation_export_kwargs=segmentation_export_kwargs,
                            run_postprocessing_on_folds=run_postprocessing_on_folds, debug=True, output_folder=output_folder, save_flow=True)
            else:
                ret = super().validate_flow_one_step(processor=self.processor, log_function=self.print_to_log_file, step=1, do_mirroring=do_mirroring, use_sliding_window=use_sliding_window, step_size=step_size,
                            save_softmax=save_softmax, use_gaussian=use_gaussian,
                            overwrite=overwrite, validation_folder_name=validation_folder_name,
                            all_in_gpu=all_in_gpu, segmentation_export_kwargs=segmentation_export_kwargs,
                            run_postprocessing_on_folds=run_postprocessing_on_folds, debug=True, output_folder=output_folder, save_flow=True)

        self.network.do_ds = ds
        return ret
    
        
    def get_only_labeled(self, output, target, labeled_binary):
        tuple_indices = torch.nonzero(labeled_binary, as_tuple=True)

        #output = torch.flatten(output, start_dim=0, end_dim=1)
        #target = torch.flatten(target, start_dim=0, end_dim=1)
        #labeled_binary = torch.flatten(labeled_binary, start_dim=0, end_dim=1)

        output_l = output[tuple_indices[0], tuple_indices[1]]
        target_l = target[tuple_indices[0], tuple_indices[1]]

        return output_l, target_l
    

    def get_perimeter(self, x):
        kernel = torch.ones(size=(3, 3), device=x.device)
        eroded = erosion(x, kernel)

        #matplotlib.use('QtAgg')
        #fig, ax = plt.subplots(1, 2)
        #ax[0].imshow(x[0, 0].detach().cpu(), cmap='gray')
        #ax[1].imshow((x - eroded)[0, 0].detach().cpu(), cmap='gray')
        #plt.show()

        perimeter = torch.sum(torch.abs(x - eroded))
        return perimeter
    

    def get_strain_curve(self, x):
        # T, B, 1, H, W
        endo_perim_list = []
        epi_perim_list = []
        for i in range(len(x)):
            current_frame = x[i]
            binarized_endo = (current_frame == 3).float()
            binarized_epi = torch.logical_or(current_frame == 2, binarized_endo).float()
            perim_endo = self.get_perimeter(binarized_endo)
            perim_epi = self.get_perimeter(binarized_epi)
            assert perim_endo >= 0
            assert perim_epi >= 0
            endo_perim_list.append(perim_endo)
            epi_perim_list.append(perim_epi)
        
        endo_strain = [(endo_perim_list[i] - endo_perim_list[0]) / (endo_perim_list[0] + 1e-8) for i in range(len(endo_perim_list))]
        epi_strain = [(epi_perim_list[i] - epi_perim_list[0]) / (epi_perim_list[0] + 1e-8) for i in range(len(epi_perim_list))]

        #endo_strain = [(endo_perim_list[i] - endo_perim_list[0]) for i in range(len(endo_perim_list))]
        #epi_strain = [(epi_perim_list[i] - epi_perim_list[0]) for i in range(len(epi_perim_list))]

        endo_strain = torch.tensor(endo_strain)
        epi_strain = torch.tensor(epi_strain)

        lv_strain = (endo_strain + epi_strain) / 2

        return lv_strain


    def get_curvature(self, x):
        # T, B, C, H, W
        #x = torch.softmax(x, dim=2) # T, B, C, H, W
        x = torch.argmax(x, dim=2, keepdim=True) # T, B, 1, H, W

        strain = self.get_strain_curve(x)
        second_order_derivative = torch.abs(strain[:-2] - 2* strain[1:-1] + strain[2:])
        assert torch.all(torch.isfinite(second_order_derivative)), strain
        return second_order_derivative
    
    def get_gradient(self, x):
        x = torch.stack(x, dim=2)
        gradient = spatial_gradient3d(x)
        return gradient[:, :, 2].pow(2).sum()
    

    def compute_losses_recursive(self, target_mask, unlabeled, out1, out2, target):
        out2['global_registered_input'] = []
        out1['local_registered_input'] = []

        t_indices, b_indices = torch.nonzero(target_mask, as_tuple=True)
        mask = t_indices != 0
        t_indices = t_indices[mask]
        b_indices = b_indices[mask]

        #seg_loss_l = self.segmentation_loss(out['seg'][t_indices - 1, b_indices], target[t_indices, b_indices])
        #self.loss_data['segmentation'][1] = seg_loss_l

        if self.dataloader_modality == 'other':

            initial_target = target[0]
            initial_target = torch.nn.functional.one_hot(initial_target[:, 0].long(), num_classes=4).permute(0, 3, 1, 2).contiguous().float()
            registered = self.motion_estimation(flow=out2['flow'], original=initial_target, mode='bilinear')
            #registered = self.motion_estimation(flow=out2['flow'][t_indices - 2, b_indices], original=initial_target, mode='bilinear')

            global_motion_loss = self.segmentation_loss(registered, target[-1])
            self.loss_data['global_motion_forward'][1] = global_motion_loss

        #semi_supervised_target = torch.softmax(out['seg'], dim=2)
        #semi_supervised_target = torch.argmax(semi_supervised_target, dim=2, keepdim=True)
#
        #semi_supervised_forward_loss = self.segmentation_loss(seg_registered_list_global_forward.permute(1, 2, 3, 4, 0), semi_supervised_target.permute(1, 2, 3, 4, 0))
        #self.loss_data['global_motion_forward_semi_supervised'][1] = semi_supervised_forward_loss

        for i in range(len(out1['flow'])):
            registered = self.motion_estimation(flow=out1['flow'][i], original=unlabeled[i])
            out1['local_registered_input'].append(registered)

        out1['local_registered_input'] = torch.stack(out1['local_registered_input'], dim=0)

        registered = self.motion_estimation(flow=out2['flow'], original=unlabeled[0])
        #for i in range(len(out2['flow'])):
        #    registered = self.motion_estimation(flow=out2['flow'][i], original=unlabeled[0])
        #    out2['global_registered_input'].append(registered)
        
        #out2['global_registered_input'] = torch.stack(out2['global_registered_input'], dim=0)
        out2['global_registered_input'] = registered

        global_flow_loss = self.mse_loss(out2['global_registered_input'], unlabeled[-1])
        local_flow_loss = self.mse_loss(out1['local_registered_input'], unlabeled[1:])
        global_flow_regularization_loss = self.image_flow_loss(flow=out2['flow'][None], iter_nb=self.iter_nb)
        local_flow_regularization_loss = self.image_flow_loss(flow=out1['flow'], iter_nb=self.iter_nb)
        self.loss_data['global_image_flow'][1] = global_flow_loss
        self.loss_data['local_image_flow'][1] = local_flow_loss
        self.loss_data['global_flow_regularization'][1] = global_flow_regularization_loss
        self.loss_data['local_flow_regularization'][1] = local_flow_regularization_loss


        if self.backward:
            backward_registered_list = []
            for i in range(len(out1['backward_flow'])):
                registered = self.motion_estimation(flow=out1['backward_flow'][i], original=unlabeled[i + 1])
                backward_registered_list.append(registered)

            backward_registered = torch.stack(backward_registered_list, dim=0)
            backward_local_flow_loss = self.mse_loss(backward_registered, unlabeled[:-1])
            self.loss_data['local_image_flow_backward'][1] = backward_local_flow_loss

        #global_sum = []
        #local_sum = []
        #for i in range(len(out['local_motion_forward'])):
        #    current_global_mean = torch.mean(torch.abs(out['global_motion_forward'][i]), dim=(2, 3)) # B, C
        #    current_local_mean = torch.mean(torch.abs(out['local_motion_forward'][:i+1]), dim=(3, 4)) # T, B, C
        #    global_sum.append(current_global_mean)
        #    local_sum.append(current_local_mean.sum(0))
        #global_sum = torch.stack(global_sum, dim=-1)
        #local_sum = torch.stack(local_sum, dim=-1)
#
        #magnitude_loss = self.mse_loss(global_sum, local_sum.detach())
        #self.loss_data['magnitude'][1] = magnitude_loss

        if self.do_adv:
            loss_fake, output_fake = self.get_fake_loss(out, label=1, detach=False)
            self.loss_data['adversarial'][1] = loss_fake.mean()

        return out1, out2




    def compute_losses_recursive_2(self, unlabeled, out1, out2, target, label_list):
        out2['global_registered_input'] = []
        out1['local_registered_input'] = []

        #seg_loss_l = self.segmentation_loss(out['seg'][t_indices - 1, b_indices], target[t_indices, b_indices])
        #self.loss_data['segmentation'][1] = seg_loss_l

        if self.dataloader_modality == 'other':

            initial_target = target[-1]
            initial_target = torch.nn.functional.one_hot(initial_target[:, 0].long(), num_classes=4).permute(0, 3, 1, 2).contiguous().float()
            registered = self.motion_estimation(flow=out2['flow'], original=initial_target, mode='bilinear')
            #registered = self.motion_estimation(flow=out2['flow'][t_indices - 2, b_indices], original=initial_target, mode='bilinear')

            global_motion_loss = self.segmentation_loss(registered, target[0])
            self.loss_data['global_motion_forward'][1] = global_motion_loss

        #semi_supervised_target = torch.softmax(out['seg'], dim=2)
        #semi_supervised_target = torch.argmax(semi_supervised_target, dim=2, keepdim=True)
#
        #semi_supervised_forward_loss = self.segmentation_loss(seg_registered_list_global_forward.permute(1, 2, 3, 4, 0), semi_supervised_target.permute(1, 2, 3, 4, 0))
        #self.loss_data['global_motion_forward_semi_supervised'][1] = semi_supervised_forward_loss

        if self.video_length > 2:
            for i in range(len(out1['flow'])):
                registered = self.motion_estimation(flow=out1['flow'][i], original=unlabeled[i + 1])
                out1['local_registered_input'].append(registered)

            out1['local_registered_input'] = torch.stack(out1['local_registered_input'], dim=0)

            local_flow_loss = self.registration_loss(out1['local_registered_input'], unlabeled[:-1])
            local_flow_regularization_loss = self.spatial_smoothing_loss(flow=out1['flow'])
            self.loss_data['local_image_flow'][1] = local_flow_loss
            self.loss_data['local_flow_regularization'][1] = local_flow_regularization_loss

            intermediate_register_list = []
            for i in range(len(out2['cumulated'])):
                registered = self.motion_estimation(flow=out2['cumulated'][i], original=unlabeled[i + 1])
                intermediate_register_list.append(registered)
            intermediate_registered = torch.stack(intermediate_register_list, dim=0)
            intermediate_loss = self.registration_loss(intermediate_registered, unlabeled[0][None].repeat(len(intermediate_registered), 1, 1, 1, 1))
            intermediate_loss_inside = intermediate_loss * label_list[0][None].repeat(len(intermediate_registered), 1, 1, 1, 1)
            self.loss_data['intermediate'][1] = intermediate_loss_inside

            cumulated_flow_regularization_loss = self.spatial_smoothing_loss(flow=out2['cumulated'])
            cumulated_flow_regularization_loss = cumulated_flow_regularization_loss * label_list[0][None].repeat(len(intermediate_registered), 1, 1, 1, 1)
            self.loss_data['cumulated_flow_regularization'][1] = cumulated_flow_regularization_loss
        
        else:
            registered = self.motion_estimation(flow=out2['flow'], original=unlabeled[-1])
            global_flow_loss = self.registration_loss(registered[None], unlabeled[0][None])
            global_flow_loss_inside = global_flow_loss * label_list[0][None]
            global_flow_regularization_loss = self.spatial_smoothing_loss(flow=out2['flow'][None])
            global_flow_regularization_loss_inside = global_flow_regularization_loss * label_list[0][None]
            self.loss_data['global_image_flow'][1] = global_flow_loss_inside
            self.loss_data['global_flow_regularization'][1] = global_flow_regularization_loss_inside

        return out1, out2
    



    def compute_losses_recursive_2_supervised(self, unlabeled, out1, out2, target, label_list):
        out2['global_registered_input'] = []
        out1['local_registered_input'] = []

        if self.video_length > 2:
            local_label_list = []
            for i in range(len(out1['flow'])):
                initial_target = target[i + 1]
                initial_target = torch.nn.functional.one_hot(initial_target[:, 0].long(), num_classes=4).permute(0, 3, 1, 2).contiguous().float()
                registered_label = self.motion_estimation(flow=out1['flow'][i], original=initial_target, mode='bilinear')
                local_label_list.append(registered_label)

                registered = self.motion_estimation(flow=out1['flow'][i], original=unlabeled[i + 1])
                out1['local_registered_input'].append(registered)

            out1['local_registered_input'] = torch.stack(out1['local_registered_input'], dim=0)
            local_label_list = torch.stack(local_label_list, dim=0)

            local_motion_loss = self.segmentation_loss(local_label_list.permute(1, 2, 3, 4, 0), target[:-1].permute(1, 2, 3, 4, 0))
            self.loss_data['local_motion_forward'][1] = local_motion_loss

            local_flow_loss = self.registration_loss(out1['local_registered_input'], unlabeled[:-1])
            local_flow_loss = local_flow_loss * label_list[:-1]

            local_flow_regularization_loss = self.spatial_smoothing_loss(flow=out1['flow'])
            local_flow_regularization_loss = local_flow_regularization_loss * label_list[:-1]

            self.loss_data['local_image_flow'][1] = local_flow_loss
            self.loss_data['local_flow_regularization'][1] = local_flow_regularization_loss

            intermediate_register_list = []
            global_label_list = []
            for i in range(len(out2['cumulated'])):
                initial_target = target[i + 1]
                initial_target = torch.nn.functional.one_hot(initial_target[:, 0].long(), num_classes=4).permute(0, 3, 1, 2).contiguous().float()
                registered_label = self.motion_estimation(flow=out2['cumulated'][i], original=initial_target, mode='bilinear')
                global_label_list.append(registered_label)

                registered = self.motion_estimation(flow=out2['cumulated'][i], original=unlabeled[i + 1])
                intermediate_register_list.append(registered)

            intermediate_registered = torch.stack(intermediate_register_list, dim=0)
            global_label_list = torch.stack(global_label_list, dim=0)

            global_motion_loss = self.segmentation_loss(global_label_list.permute(1, 2, 3, 4, 0), target[0][None].repeat(len(global_label_list), 1, 1, 1, 1).permute(1, 2, 3, 4, 0))
            self.loss_data['global_motion_forward'][1] = global_motion_loss

            intermediate_loss = self.registration_loss(intermediate_registered, unlabeled[0][None].repeat(len(intermediate_registered), 1, 1, 1, 1))
            intermediate_loss_inside = intermediate_loss * label_list[0][None].repeat(len(intermediate_registered), 1, 1, 1, 1)
            self.loss_data['intermediate'][1] = intermediate_loss_inside

            cumulated_flow_regularization_loss = self.spatial_smoothing_loss(flow=out2['cumulated'])
            cumulated_flow_regularization_loss = cumulated_flow_regularization_loss * label_list[0][None].repeat(len(intermediate_registered), 1, 1, 1, 1)
            self.loss_data['cumulated_flow_regularization'][1] = cumulated_flow_regularization_loss
        
        else:
            registered = self.motion_estimation(flow=out2['flow'], original=unlabeled[-1])
            global_flow_loss = self.registration_loss(registered[None], unlabeled[0][None])
            global_flow_loss_inside = global_flow_loss * label_list[0][None]
            global_flow_regularization_loss = self.spatial_smoothing_loss(flow=out2['flow'][None])
            global_flow_regularization_loss_inside = global_flow_regularization_loss * label_list[0][None]
            self.loss_data['global_image_flow'][1] = global_flow_loss_inside
            self.loss_data['global_flow_regularization'][1] = global_flow_regularization_loss_inside

        return out1, out2
    




    def compute_losses_label(self, target_mask, unlabeled, out1, out2, target, label_list):
        out2['global_registered_input'] = []
        out1['local_registered_input'] = []

        t_indices, b_indices = torch.nonzero(target_mask, as_tuple=True)
        mask = t_indices != self.video_length - 1
        t_indices = t_indices[mask]
        b_indices = b_indices[mask]

        #seg_loss_l = self.segmentation_loss(out['seg'][t_indices - 1, b_indices], target[t_indices, b_indices])
        #self.loss_data['segmentation'][1] = seg_loss_l

        if self.dataloader_modality == 'other':

            initial_target = target[-1]
            initial_target = torch.nn.functional.one_hot(initial_target[:, 0].long(), num_classes=4).permute(0, 3, 1, 2).contiguous().float()
            registered = self.motion_estimation(flow=out2['flow'], original=initial_target, mode='bilinear')
            #registered = self.motion_estimation(flow=out2['flow'][t_indices - 2, b_indices], original=initial_target, mode='bilinear')
            global_motion_loss = self.segmentation_loss(registered, target[0])

            self.loss_data['global_motion_forward'][1] = global_motion_loss

        #semi_supervised_target = torch.softmax(out['seg'], dim=2)
        #semi_supervised_target = torch.argmax(semi_supervised_target, dim=2, keepdim=True)
#
        #semi_supervised_forward_loss = self.segmentation_loss(seg_registered_list_global_forward.permute(1, 2, 3, 4, 0), semi_supervised_target.permute(1, 2, 3, 4, 0))
        #self.loss_data['global_motion_forward_semi_supervised'][1] = semi_supervised_forward_loss

        if self.video_length > 2:
            for i in range(len(out1['flow'])):
                registered = self.motion_estimation(flow=out1['flow'][i], original=unlabeled[i + 1])
                out1['local_registered_input'].append(registered)

            out1['local_registered_input'] = torch.stack(out1['local_registered_input'], dim=0)

            local_flow_loss = self.registration_loss(out1['local_registered_input'], unlabeled[:-1])
            local_flow_loss = local_flow_loss * label_list[:-1]
            self.loss_data['local_image_flow'][1] = local_flow_loss.mean()

            local_flow_regularization_loss = self.spatial_smoothing_loss(flow=out1['flow'])

            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(1, 3)
            #ax[0].imshow(local_flow_regularization_loss[-1, 0, 0].detach().cpu(), cmap='hot')
            #ax[1].imshow(label_list[-2, 0, 0].detach().cpu(), cmap='gray')
            #ax[2].imshow((local_flow_regularization_loss * label_list[:-1])[-1, 0, 0].detach().cpu(), cmap='hot')
            #plt.show()

            local_flow_regularization_loss = local_flow_regularization_loss * label_list[:-1]
            self.loss_data['local_flow_regularization'][1] = local_flow_regularization_loss

            intermediate_register_list = []
            for i in range(len(out2['cumulated'])):
                registered = self.motion_estimation(flow=out2['cumulated'][i], original=unlabeled[i + 1])
                intermediate_register_list.append(registered)
            intermediate_registered = torch.stack(intermediate_register_list, dim=0)
            intermediate_loss = self.registration_loss(intermediate_registered, unlabeled[0][None].repeat(len(intermediate_registered), 1, 1, 1, 1))
            intermediate_loss_inside = intermediate_loss * label_list[0][None].repeat(len(intermediate_registered), 1, 1, 1, 1)
            intermediate_loss_outside = intermediate_loss * (1 - label_list[0][None].repeat(len(intermediate_registered), 1, 1, 1, 1))

            self.loss_data['intermediate'][1] = intermediate_loss_inside.mean()
            self.loss_data['intermediate_outside'][1] = intermediate_loss_outside.mean()

            cumulated_flow_regularization_loss = self.spatial_smoothing_loss(flow=out2['cumulated'])
            cumulated_flow_regularization_loss = cumulated_flow_regularization_loss * label_list[0][None].repeat(len(intermediate_registered), 1, 1, 1, 1)
            self.loss_data['cumulated_flow_regularization'][1] = cumulated_flow_regularization_loss
        
        else:
            registered = self.motion_estimation(flow=out2['flow'], original=unlabeled[-1])
            global_flow_loss = self.registration_loss(registered, unlabeled[0])
            global_flow_regularization_loss = self.spatial_smoothing_loss(flow=out2['flow'][None])
            self.loss_data['global_image_flow'][1] = global_flow_loss
            self.loss_data['global_flow_regularization'][1] = global_flow_regularization_loss

        return out1, out2
    
    


    def compute_losses_recursive_ds(self, target_mask, unlabeled, out1, out2, target):
        out2['global_registered_input'] = []
        out1['local_registered_input'] = []

        t_indices, b_indices = torch.nonzero(target_mask, as_tuple=True)
        mask = t_indices != 0
        t_indices = t_indices[mask]
        b_indices = b_indices[mask]

        #seg_loss_l = self.segmentation_loss(out['seg'][t_indices - 1, b_indices], target[t_indices, b_indices])
        #self.loss_data['segmentation'][1] = seg_loss_l

        initial_target = target[0]
        initial_target = torch.nn.functional.one_hot(initial_target[:, 0].long(), num_classes=4).permute(0, 3, 1, 2).contiguous().float()
        
        global_motion_loss = 0
        global_flow_loss = 0
        global_flow_regularization_loss = 0
        local_flow_regularization_loss = 0
        for i in range(3):
            registered_img = self.motion_estimation(flow=out2['flow'][i], original=unlabeled[0])
            registered_seg = self.motion_estimation(flow=out2['flow'][i], original=initial_target, mode='bilinear')
            global_motion_loss += (self.ds_weights[i] * self.segmentation_loss(registered_seg, target[-1]))
            global_flow_loss += (self.ds_weights[i] * self.mse_loss(registered_img, unlabeled[-1]))
            global_flow_regularization_loss += (self.ds_weights[i] * self.image_flow_loss(flow=out2['flow'][i][None], iter_nb=self.iter_nb))
            local_flow_regularization_loss += (self.ds_weights[i] * self.image_flow_loss(flow=out1['flow'][i], iter_nb=self.iter_nb))
        self.loss_data['global_motion_forward'][1] = global_motion_loss
        self.loss_data['global_image_flow'][1] = global_flow_loss
        self.loss_data['global_flow_regularization'][1] = global_flow_regularization_loss
        self.loss_data['local_flow_regularization'][1] = local_flow_regularization_loss

        #semi_supervised_target = torch.softmax(out['seg'], dim=2)
        #semi_supervised_target = torch.argmax(semi_supervised_target, dim=2, keepdim=True)
#
        #semi_supervised_forward_loss = self.segmentation_loss(seg_registered_list_global_forward.permute(1, 2, 3, 4, 0), semi_supervised_target.permute(1, 2, 3, 4, 0))
        #self.loss_data['global_motion_forward_semi_supervised'][1] = semi_supervised_forward_loss

        local_flow_loss_list = []
        for i in range(len(out1['flow'][0])):
            local_flow_loss = 0
            for j in range(3):
                registered = self.motion_estimation(flow=out1['flow'][j][i], original=unlabeled[i])
                local_flow_loss += self.ds_weights[j] * self.mse_loss(registered, unlabeled[i + 1])
            local_flow_loss_list.append(local_flow_loss)
        self.loss_data['local_image_flow'][1] = torch.stack(local_flow_loss_list).mean()

        #global_sum = []
        #local_sum = []
        #for i in range(len(out['local_motion_forward'])):
        #    current_global_mean = torch.mean(torch.abs(out['global_motion_forward'][i]), dim=(2, 3)) # B, C
        #    current_local_mean = torch.mean(torch.abs(out['local_motion_forward'][:i+1]), dim=(3, 4)) # T, B, C
        #    global_sum.append(current_global_mean)
        #    local_sum.append(current_local_mean.sum(0))
        #global_sum = torch.stack(global_sum, dim=-1)
        #local_sum = torch.stack(local_sum, dim=-1)
#
        #magnitude_loss = self.mse_loss(global_sum, local_sum.detach())
        #self.loss_data['magnitude'][1] = magnitude_loss

        if self.do_adv:
            loss_fake, output_fake = self.get_fake_loss(out, label=1, detach=False)
            self.loss_data['adversarial'][1] = loss_fake.mean()

        return out1, out2
    


    def select_deep_supervision(self, x):
        if self.deep_supervision:
            return x[0]
        else:
            return x
        
    def log_weights(self, weights):
        self.writer.add_scalars('Iteration/weights', {str(i):weights[i] for i in range(len(weights))}, self.iter_nb)
        

    def run_iteration_train(self, data_generator):
        """
        gradient clipping improves training stability

        :param data_generator:
        :param do_backprop:
        :param run_online_evaluation:
        :return:
        """

        data_dict = next(data_generator)
        unlabeled = data_dict['unlabeled']
        target = data_dict['target']
        strain_mask = data_dict['strain_mask'].float()
        strain_mask_one_hot = data_dict['strain_mask_one_hot']

        #matplotlib.use('QtAgg')
        #fig, ax = plt.subplots(6, len(unlabeled))
        #for k in range(len(unlabeled)):
        #    ax[0, k].imshow(unlabeled[k, 0, 0].cpu(), cmap='gray')
        #    ax[1, k].imshow(target[k, 0, 0].cpu(), cmap='gray')
        #    ax[2, k].imshow(strain_mask_one_hot[k, 0, 0].cpu(), cmap='hot', vmin=0.0, vmax=1.0)
        #    ax[3, k].imshow(strain_mask_one_hot[k, 0, 1].cpu(), cmap='hot', vmin=0.0, vmax=1.0)
        #    ax[4, k].imshow(strain_mask_one_hot[k, 0, 2].cpu(), cmap='hot', vmin=0.0, vmax=1.0)
        #    ax[5, k].imshow(strain_mask[k, 0, 0].cpu(), cmap='hot', vmin=0.0, vmax=1.0)
        #plt.show()

        #matplotlib.use('QtAgg')
        #fig, ax = plt.subplots(2, 3)
        #ax[0, 0].imshow(unlabeled[0, 0, 0].cpu(), cmap='gray')
        #ax[0, 1].imshow(unlabeled[1, 0, 0].cpu(), cmap='gray')
        #ax[0, 2].imshow(labeled[0, 0].cpu(), cmap='gray')
        #ax[1, 0].imshow(unlabeled[0, 1, 0].cpu(), cmap='gray')
        #ax[1, 1].imshow(unlabeled[1, 1, 0].cpu(), cmap='gray')
        #ax[1, 2].imshow(labeled[1, 0].cpu(), cmap='gray')
        #plt.show()

        #labeled = maybe_to_torch(labeled)
        #unlabeled = maybe_to_torch(unlabeled)
        #target = maybe_to_torch(target)
        #if torch.cuda.is_available():
        #    labeled = to_cuda(labeled)
        #    unlabeled = to_cuda(unlabeled)
        #    target = to_cuda(target)

        #unlabeled.requires_grad = True
        out1, out2 = self.network(unlabeled)
        #unlabeled = unlabeled.permute(2, 0, 1, 3, 4).contiguous()
        self.optimizer.zero_grad()

        #self.writer.add_scalars('Iteration/attn_weights', {'frame_' + str(i):torch.abs(out['attn_weights'][i, 0]) for i in range(self.video_length)}, self.iter_nb)

        #self.log_weights(out['weights'])

        #with torch.enable_grad():
        #    labeled.requires_grad = True
        #    unlabeled.requires_grad = True
#
        #    t, b, c, y, x = self.sample_indices(target[None])
        #    out_gradient = out['motion_flow_u_to_l'][t, b, c, y, x]
        #    out_gradient.backward(retain_graph=True)
#
        #    gradient_unlabeled = torch.abs(unlabeled.grad).mean() * 1e6
        #    gradient_labeled = torch.abs(labeled.grad).mean() * 1e6
        #
        #    self.optimizer.zero_grad()

        #unlabeled = Variable(unlabeled, requires_grad=True)
        #unlabeled = unlabeled.cuda()
        #unlabeled.retain_grad()
#
        out1, out2 = self.loss_function(unlabeled=unlabeled, out1=out1, out2=out2, target=target, label_list=strain_mask)
        l = self.consolidate_only_one_loss_data(self.loss_data, log=True)

        #if self.iter_nb > 500:
        #    matplotlib.use('QtAgg')
        #    fig, ax = plt.subplots(1, 1)
        #    ax.imshow(out['motion_flow_u_to_l'][0, 0, 0].detach().cpu(), cmap='gray')
        #    plt.show()
        #del loss_data

        self.train_loss.append(l.mean().detach().cpu())
        l.mean().backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
        self.optimizer.step()

        if self.iter_nb == 0:
            for param_name, param in self.network.named_parameters():
                for module_name, module in self.network.named_modules():
                    if list(module.children()) == []:
                        if not isinstance(module, (nn.BatchNorm2d, nn.BatchNorm3d)) and module_name in param_name:
                            if param.grad is None:
                                print(param_name)
                                print(param.is_leaf)
                                print(param.requires_grad)
        
            #decoder_nb = 0
            #spatio_temporal_nb = 0
            #for param_name, param in self.network.named_parameters():
            #    if 'transformerDecode' in param_name:
            #        print(f'{param_name}: {param.numel()}, {param.requires_grad}')
            #        decoder_nb += param.numel()
            #    elif 'spatio_tempora' in param_name:
            #        print(f'{param_name}: {param.numel()}, {param.requires_grad}')
            #        spatio_temporal_nb += param.numel()
            #print(f'decoder_nb: {decoder_nb}')
            #print(f'spatio_temporal_nb: {spatio_temporal_nb}')

        self.iter_nb += 1

        return l.mean().detach().cpu().numpy()
    


    def run_iteration_val(self, data_generator):
        """
        gradient clipping improves training stability

        :param data_generator:
        :param do_backprop:
        :param run_online_evaluation:
        :return:
        """

        data_dict = next(data_generator)
        unlabeled = data_dict['unlabeled'] # T, B, 1, H, W
        target = data_dict['target'] # T, B, 1, H, W
        target_mask = data_dict['target_mask'] # T, B
        strain_mask_one_hot = data_dict['strain_mask_one_hot']

        with torch.no_grad():
            out1, out2 = self.network(unlabeled)


        seg_dice = self.get_stats(out=out2, 
                                  target=target,
                                  padding=data_dict['padding_need'])

        if self.log_images:

            out['seg'] = out['seg'][:3]
            out['seg_registered_forward'] = out['seg_registered_forward'][:3]
            out['registered_input_forward'] = out['registered_input_forward'][:3]
            out['global_motion_forward'] = out['global_motion_forward'][:3]
            unlabeled = unlabeled[:3]
            target = target[:3]
            target_mask = target_mask[:3]

            self.run_online_evaluation(out=out,
                                       unlabeled=unlabeled,
                                       target=target,
                                       seg_dice=seg_dice,
                                       target_mask=target_mask)



    def run_iteration_overfitting(self, data_generator):
        """
        gradient clipping improves training stability

        :param data_generator:
        :param do_backprop:
        :param run_online_evaluation:
        :return:
        """

        data_dict = next(data_generator)
        unlabeled = data_dict['unlabeled'] # T, B, 1, H, W
        target = data_dict['target'] # T, B, 1, H, W
        target_mask = data_dict['target_mask'] # T, B
        strain_mask = data_dict['strain_mask'].float()
        strain_mask_one_hot = data_dict['strain_mask_one_hot']

        with torch.no_grad():
            out1, out2 = self.network(unlabeled)

        #unlabeled = unlabeled.permute(2, 0, 1, 3, 4).contiguous()
        out1, out2 = self.loss_function(unlabeled=unlabeled, out1=out1, out2=out2, target=target, label_list=strain_mask)
        l = self.consolidate_only_one_loss_data(self.loss_data, log=False)
        self.val_loss.append(l.mean().detach().cpu())

        return l.mean().detach().cpu().numpy()
    

    """ def warp_linear(self, target, local_motions, global_motion):
        ed_target = target[0]
        es_target = target[-1]
        ed_target = torch.nn.functional.one_hot(ed_target[:, 0].long(), num_classes=4).permute(0, 3, 1, 2).contiguous().float()

        es_pred_global = self.motion_estimation(flow=global_motion, original=ed_target, mode='bilinear')
        
        original = ed_target
        for i in range(len(local_motions)):
            current_flow = local_motions[i]
            original = self.motion_estimation(flow=current_flow, original=original, mode='bilinear')
        es_pred_local = original
            
        registered_local = torch.argmax(es_pred_local, dim=1, keepdim=True) # B, 1, H, W
        registered_global = torch.argmax(es_pred_global, dim=1, keepdim=True) # B, 1, H, W

        flow_target = es_target[:, 0]
        registered_local = registered_local[:, 0]
        registered_global = registered_global[:, 0]
        assert torch.all(torch.isfinite(flow_target))
        return registered_local, registered_global, flow_target """
    


    def warp_linear(self, target, flow):
        if self.motion_from_ed:
            from_frame = target[:, -1]
            to_frame = target[:, 0]
        else:
            from_frame = target[:, 0]
            to_frame = target[:, -1]

        from_frame = torch.nn.functional.one_hot(from_frame[:, 0].long(), num_classes=4).permute(0, 3, 1, 2).contiguous().float()
        predicted = self.motion_estimation(flow=flow, original=from_frame, mode='bilinear')
            
        registered = torch.argmax(predicted, dim=1, keepdim=True) # B, 1, H, W

        flow_target = to_frame[:, 0]
        registered = registered[:, 0]
        assert torch.all(torch.isfinite(flow_target))
        return registered, flow_target

    
    def warp_linear_Lib(self, target, flow):
        registered_list = []
        flow_target_list = []
        ed_target = target[:, 0]
        ed_target = torch.nn.functional.one_hot(ed_target[:, 0].long(), num_classes=4).permute(0, 3, 1, 2).contiguous().float()
        for t in range(1, target.shape[1]):
            es_target = target[:, t]
            es_pred = self.motion_estimation(flow=flow[:, t - 1], original=ed_target, mode='bilinear')
            registered_list.append(es_pred)
            flow_target_list.append(es_target)
        registered = torch.stack(registered_list, dim=1) # B, T, 4, H, W
        registered = torch.argmax(registered, dim=2, keepdim=True) # B, T, 1, H, W
        flow_target = torch.stack(flow_target_list, dim=1) # B, T, 1, H, W
        assert torch.all(torch.isfinite(flow_target))
        return registered, flow_target
    

    def get_stats(self, out, target, padding):
        #step = int(len(out['forward_flow']) / len(target))
        #out['forward_flow'] = out['forward_flow'][::step]

        #seg = out['seg'].permute(1, 0, 2, 3, 4).contiguous()
        target = target.permute(1, 0, 2, 3, 4).contiguous()
        flow = out['flow'][0] if self.deep_supervision else out['flow']
        num_classes = 4

        #flow = torch.nn.functional.pad(flow, pad=(0, 0, 0, 0, 0, 0, 1, 0))

        registered, flow_target = self.warp_linear(target, flow) # B, H, W
        #if '31' in self.task:
        #    registered, flow_target = self.warp_linear(target, flow, and_mask) # B, H, W
        #elif '32' in self.task:
        #    registered, flow_target = self.warp_linear_Lib(target, flow) # B, H, W

        #registered_list = []
        #flow_target_list = []
        #for b in range(len(target)):
        #    indices = torch.nonzero(and_mask[b]) # 2, 1
        #    flow_target = target[b, indices[0].item()]
        #    current_motion = target[b, indices[1].item()][None]
        #    
        #    for t in range(indices[1].item(), indices[0].item(), -1):
        #        current_motion = self.motion_estimation(flow=flow[b, t][None], original=current_motion, mode='nearest')
        #    
        #    registered_list.append(current_motion[0])
        #    flow_target_list.append(flow_target)
        #registered = torch.stack(registered_list, dim=0) # B, 1, H, W
        #flow_target = torch.stack(flow_target_list, dim=0) # B, 1, H, W
#
        #flow_target = flow_target[:, 0]
        #registered = registered[:, 0]
        #assert torch.all(torch.isfinite(flow_target))

        flow_target = self.processor.uncrop_no_registration(flow_target[:, None, None], padding_need=padding) # B, T, 1, H, W
        registered = self.processor.uncrop_no_registration(registered[:, None, None], padding_need=padding) # B, T, 1, H, W
        flow_target = flow_target[:, 0, 0]
        registered = registered[:, 0, 0]
        seg_dice = self.compute_dice(flow_target.cpu(), num_classes, registered.cpu(), key='flow')

        #if '31' in self.task:
        #    flow_target = self.processor.uncrop_no_registration(flow_target[:, None, None], padding_need=padding) # B, T, 1, H, W
        #    registered = self.processor.uncrop_no_registration(registered[:, None, None], padding_need=padding) # B, T, 1, H, W
        #    flow_target = flow_target[:, 0, 0]
        #    registered = registered[:, 0, 0]
        #    seg_dice = self.compute_dice(flow_target.cpu(), num_classes, registered.cpu(), key='flow')
        #elif '32' in self.task:
        #    seg_dice = []
        #    for t in range(flow_target.shape[1]):
        #        current_seg_dice = self.compute_dice(flow_target[:, t, 0].cpu(), num_classes, registered[:, t, 0].cpu(), key='flow')
        #        seg_dice.append(current_seg_dice)
        #    seg_dice = torch.stack(seg_dice, dim=0).mean(0)
        
        #target = self.processor.uncrop_no_registration(target, padding_need=padding) # B, T, 1, H, W
        #assert list(target.shape[-2:]) == [self.image_size, self.image_size]
#
        #seg = torch.softmax(seg, dim=2)
        #seg = torch.argmax(seg, dim=2, keepdim=True) # B, T - 1, 1, H, W
        #seg = self.processor.uncrop_no_registration(seg, padding_need=padding)
#
        #and_mask = and_mask[:, 1:]
        #target = target[:, 1:]
#
        #seg = torch.stack([seg[b, and_mask[b]] for b in range(len(seg))], dim=0)
        #target = torch.stack([target[b, and_mask[b]] for b in range(len(target))], dim=0)
#
        #assert seg.shape[1] >= 1
#
        #target = target[:, :, 0]
        #seg = seg[:, :, 0]
        #assert torch.all(torch.isfinite(target))
        #self.compute_dice(target.cpu(), num_classes, seg.cpu(), key='seg')

        return seg_dice
    

    def consolidate_only_one_loss_data(self, loss_data, log, database=None):
        loss = 0.0
        for key, value in loss_data.items():
            if log:
                self.writer.add_scalar('Iteration/' + key + ' loss', value[1].mean(), self.iter_nb)
            loss += value[0] * value[1].mean()
            assert loss.numel() == 1
        if log:
            self.writer.add_scalars('Iteration/loss weights', {key:value[0] for key, value in loss_data.items()}, self.iter_nb)
            self.writer.add_scalar('Iteration/Training loss', loss.mean(), self.iter_nb)
        return loss

    def consolidate_loss_data(self, loss_data1, loss_data2, log, description=None, w1=1, w2=1):
        loss_data_consolidated = self.setup_loss_data()
        for (key1, value1), (key2, value2) in zip(loss_data1.items(), loss_data2.items()):
            assert key1 == key2
            loss_data_consolidated[key1][1] = w1 * value1[1] + w2 * value2[1]
            if log:
                self.writer.add_scalar('Iteration/' + description + key1 + ' loss', loss_data_consolidated[key1][1], self.iter_nb)
        return loss_data_consolidated
    
    def convert_loss_data_to_number(self, loss_data):
        loss = 0
        for key, value in loss_data.items():
            loss += value[0] * value[1]
        return loss
    

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

            #self.print_to_log_file("\nprinting the network instead:\n")
            #self.print_to_log_file(self.network)
            #self.print_to_log_file("\n")
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def get_mms_split_indices(self, all_keys_sorted):
        train_indices = []
        test_indices = []
        patient_dirs_test = subfolders(os.path.join(Path.cwd(), 'OpenDataset', 'Testing'))
        patient_dirs_val = subfolders(os.path.join(Path.cwd(), 'OpenDataset', 'Validation'))
        test_paths = patient_dirs_test + patient_dirs_val
        test_id = [x.split(os.sep)[-1] for x in test_paths]
        for idx, string_id in enumerate(all_keys_sorted):
            if any(string_id.split('_')[0] in x for x in test_id):
                test_indices.append(idx)
            else:
                train_indices.append(idx)
        return train_indices, test_indices
    
    def split_folders(self, all_keys_sorted):
        out = {'RACINE': [], 'cholcoeur': [], 'desktop': []}
        for k in all_keys_sorted:
            properties = load_pickle(self.dataset[k]['properties_file'])
            folder_name = properties['original_path'].split('\\')[1]
            out[folder_name].append(k)
        return out
    
    def split_factor(self, all_keys_sorted, factor):
        factor_list = self.get_factor_list(all_keys_sorted, factor)
        unique_subfactors = set(factor_list)
        out = {}
        for subfactor in unique_subfactors:
            out[subfactor] = []
        for name, factor_value in zip(all_keys_sorted, factor_list):
            out[factor_value].append(name)
        return out

    def get_factor_list(self, all_keys_sorted, factor):
        out = []
        for k in all_keys_sorted:
            properties = load_pickle(self.dataset[k]['properties_file'])
            current_subfactor = properties[factor]
            out.append(current_subfactor)
        return out

    def do_split(self):
        """
        The default split is a 5 fold CV on all available training cases. nnU-Net will create a split (it is seeded,
        so always the same) and save it as splits_final.pkl file in the preprocessed data directory.
        Sometimes you may want to create your own split for various reasons. For this you will need to create your own
        splits_final.pkl file. If this file is present, nnU-Net is going to use it and whatever splits are defined in
        it. You can create as many splits in this file as you want. Note that if you define only 4 splits (fold 0-3)
        and then set fold=4 when training (that would be the fifth split), nnU-Net will print a warning and proceed to
        use a random 80:20 data split.
        :return:
        """
        if self.fold == "all":
            # if fold==all then we use all images for training and validation
            tr_keys = val_keys = list(self.dataset.keys())
        else:
            splits_file = join(self.dataset_directory, "splits_final.pkl")

            # if the split file does not exist we need to create it
            if not isfile(splits_file):
                splits = []
                if '026' in self.dataset_directory:
                    self.print_to_log_file("Creating new M&Ms split...")
                    all_keys_sorted = np.sort(list(self.dataset.keys()))
                    splits.append(OrderedDict())
                    train_idx, test_idx = self.get_mms_split_indices(all_keys_sorted)
                    train_keys = np.array(all_keys_sorted)[train_idx]
                    test_keys = np.array(all_keys_sorted)[test_idx]
                    splits[-1]['train'] = train_keys
                    splits[-1]['val'] = test_keys
                elif '029' in self.dataset_directory:
                    print("Creating new 5-fold cross-validation split...")
                    all_keys_sorted = np.sort(list(self.dataset.keys()))
                    factor = 'manufacturer'
                    factor_dict = self.split_factor(all_keys_sorted, factor)
                    unique_subfactors = list(factor_dict.keys())
                    payload_dict = {'train': {x:[] for x in unique_subfactors}, 'val': {x:[] for x in unique_subfactors}}
                    for k in factor_dict.keys():
                        kf = KFold(n_splits=5, shuffle=True, random_state=12345)
                        patients = np.unique([i[:10] for i in factor_dict[k]])
                        for tr, val in kf.split(patients):
                            tr_patients = patients[tr]
                            payload_dict['train'][k].append(np.array([i for i in factor_dict[k] if i[:10] in tr_patients]))
                            val_patients = patients[val]
                            payload_dict['val'][k].append(np.array([i for i in factor_dict[k] if i[:10] in val_patients]))
                    splits = [OrderedDict() for _ in range(5)]
                    for j in range(5):
                        current_folder_data_train = payload_dict['train']
                        current_folder_data_val = payload_dict['val']
                        train_list = []
                        val_list = []
                        for k in current_folder_data_train.keys():
                            train_list.append(current_folder_data_train[k][j])
                            val_list.append(current_folder_data_val[k][j])
                        splits[j]['train'] = np.concatenate(train_list)
                        splits[j]['val'] = np.concatenate(val_list)
                elif '028' in self.dataset_directory:
                    self.print_to_log_file("Creating new 5-fold cross-validation split...")
                    all_keys_sorted = np.sort(list(self.dataset.keys()))
                    folder_dict = self.split_folders(all_keys_sorted)
                    payload_dict = {'train': {'RACINE': [], 'cholcoeur': [], 'desktop': []},
                                     'val': {'RACINE': [], 'cholcoeur': [], 'desktop': []}}
                    for k in folder_dict.keys():
                        kf = KFold(n_splits=5, shuffle=True, random_state=12345)
                        patients = np.unique([i[:10] for i in folder_dict[k]])
                        for tr, val in kf.split(patients):
                            tr_patients = patients[tr]
                            payload_dict['train'][k].append(np.array([i for i in folder_dict[k] if i[:10] in tr_patients]))
                            val_patients = patients[val]
                            payload_dict['val'][k].append(np.array([i for i in folder_dict[k] if i[:10] in val_patients]))
                    splits = [OrderedDict() for _ in range(5)]
                    for j in range(5):
                        current_folder_data_train = payload_dict['train']
                        current_folder_data_val = payload_dict['val']
                        train_list = []
                        val_list = []
                        for k in current_folder_data_train.keys():
                            train_list.append(current_folder_data_train[k][j])
                            val_list.append(current_folder_data_val[k][j])
                        splits[j]['train'] = np.concatenate(train_list)
                        splits[j]['val'] = np.concatenate(val_list)
                else:
                    self.print_to_log_file("Creating new 5-fold cross-validation split...")
                    all_keys_sorted = np.sort(list(self.dataset.keys()))
                    kfold = KFold(n_splits=5, shuffle=True, random_state=12345)
                    to_split = np.arange(len(all_keys_sorted) / 2)
                    for i, (train_idx, test_idx) in enumerate(kfold.split(to_split)):
                        train_idx = train_idx * 2
                        test_idx = test_idx * 2
                        train_idx = np.concatenate([train_idx, train_idx + 1])
                        test_idx = np.concatenate([test_idx, test_idx + 1])
                        train_idx = np.sort(train_idx)
                        test_idx = np.sort(test_idx)
                        assert len(train_idx) + len(test_idx) == len(all_keys_sorted)
                        train_keys = np.array(all_keys_sorted)[train_idx]
                        test_keys = np.array(all_keys_sorted)[test_idx]
                        splits.append(OrderedDict())
                        splits[-1]['train'] = train_keys
                        splits[-1]['val'] = test_keys
                save_pickle(splits, splits_file)
            else:
                self.print_to_log_file("Using splits from existing split file:", splits_file)
                splits = load_pickle(splits_file)
                self.print_to_log_file("The split file contains %d splits." % len(splits))
            
            #if self.unlabeled:
            #    un_splits_file = join(self.dataset_directory, "unlabeled_splits_final.pkl")
            #    if not isfile(un_splits_file):
            #        self.print_to_log_file("Creating new unlabeled train test split...")
            #        un_tr_keys, un_val_keys = train_test_split(list(self.unlabeled_dataset.keys()), test_size=int(len(all_keys_sorted) / 5), random_state=12345, shuffle=True)
            #        un_train_test_split = OrderedDict()
            #        un_train_test_split['un_train'] = un_tr_keys
            #        un_train_test_split['un_val'] = un_val_keys
            #        save_pickle(un_train_test_split, un_splits_file)
            #    else:
            #        self.print_to_log_file("Using splits from existing unlabeled train test split file:", un_splits_file)
            #        un_train_test_split = load_pickle(un_splits_file)
#
            #if self.unlabeled:
            #    un_tr_keys = un_train_test_split['un_train']
            #    un_val_keys = un_train_test_split['un_val']
            #    self.print_to_log_file("This unlabeled train test split has %d training and %d validation cases."
            #                           % (len(un_tr_keys), len(un_val_keys)))

            self.print_to_log_file("Desired fold for training: %d" % self.fold)
            if self.fold < len(splits):
                tr_keys = splits[self.fold]['train']
                val_keys = splits[self.fold]['val']
                self.print_to_log_file("This split has %d training and %d validation cases."
                                       % (len(tr_keys), len(val_keys)))
            else:
                self.print_to_log_file("INFO: You requested fold %d for training but splits "
                                       "contain only %d folds. I am now creating a "
                                       "random (but seeded) 80:20 split!" % (self.fold, len(splits)))
                # if we request a fold that is not in the split file, create a random 80:20 split
                rnd = np.random.RandomState(seed=12345 + self.fold)
                keys = np.sort(list(self.dataset.keys()))
                idx_tr = rnd.choice(len(keys), int(len(keys) * 0.8), replace=False)
                idx_val = [i for i in range(len(keys)) if i not in idx_tr]
                tr_keys = [keys[i] for i in idx_tr]
                val_keys = [keys[i] for i in idx_val]
                self.print_to_log_file("This random 80:20 split has %d training and %d validation cases."
                                       % (len(tr_keys), len(val_keys)))

        tr_keys.sort()
        val_keys.sort()
        self.dataset_tr = OrderedDict()
        for i in tr_keys:
            self.dataset_tr[i] = self.dataset[i]
        self.dataset_val = OrderedDict()
        for i in val_keys:
            self.dataset_val[i] = self.dataset[i]
        
        
        if self.image_size == 224:
            tr_patient_id = np.unique([x[:10] for x in tr_keys])
            val_patient_id = np.unique([x[:10] for x in val_keys])
            un_tr_keys = [x for x in self.unlabeled_dataset.keys() if x[:10] in tr_patient_id]
            un_val_keys = [x for x in self.unlabeled_dataset.keys() if x[:10] in val_patient_id]
            un_tr_keys.sort()
            un_val_keys.sort()
            self.dataset_un_tr = OrderedDict()
            for i in un_tr_keys:
                self.dataset_un_tr[i] = self.unlabeled_dataset[i]
            self.dataset_un_val = OrderedDict()
            for i in un_val_keys:
                self.dataset_un_val[i] = self.unlabeled_dataset[i]
        else:
            self.dataset_un_val = None
            self.dataset_un_tr = None

    def setup_DA_params(self):
        """
        - we increase roation angle from [-15, 15] to [-30, 30]
        - scale range is now (0.7, 1.4), was (0.85, 1.25)
        - we don't do elastic deformation anymore

        :return:
        """
        if self.deep_supervision:
            self.deep_supervision_scales = [[1, 1, 1]] + list(list(i) for i in 1 / np.cumprod(
                np.vstack(self.net_num_pool_op_kernel_sizes), axis=0))[:-1]

        if self.threeD:
            self.data_aug_params = default_3D_augmentation_params
            self.data_aug_params['rotation_x'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            self.data_aug_params['rotation_y'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            self.data_aug_params['rotation_z'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
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

        self.data_aug_params["scale_range"] = (0.7, 1.4)
        self.data_aug_params["noise_p"] = 0.1
        self.data_aug_params["blur_p"] = 0.2
        self.data_aug_params["mult_brightness_p"] = 0.15
        self.data_aug_params["constrast_p"] = 0.15
        self.data_aug_params["low_res_p"] = 0.25
        self.data_aug_params["inverted_gamma_p"] = 0.1
        self.data_aug_params["do_elastic"] = False
        self.data_aug_params['selected_seg_channels'] = [0]
        self.data_aug_params['patch_size_for_spatialtransform'] = self.patch_size

        self.data_aug_params["num_cached_per_thread"] = 2

    def maybe_update_lr(self, epoch=None):
        """
        if epoch is not None we overwrite epoch. Else we use epoch = self.epoch + 1

        (maybe_update_lr is called in on_epoch_end which is called before epoch is incremented.
        herefore we need to do +1 here)

        :param epoch:
        :return:
        """
        if epoch is None:
            ep = self.epoch + 1
            if self.config['scheduler'] == 'cosine':
                self.lr_scheduler.step()
                if self.do_adv:
                    self.discriminator_scheduler.step()
            else:
                self.optimizer.param_groups[0]['lr'] = poly_lr(ep, self.max_num_epochs, self.initial_lr, 0.9)
                if self.do_adv:
                    self.discriminator_optimizer.param_groups[0]['lr'] = poly_lr(ep, self.max_num_epochs, self.discriminator_lr, 0.9)
            if self.do_adv:
                lrs = {'Flow_lr': self.optimizer.param_groups[0]['lr'], 'Discriminator_lr': self.discriminator_optimizer.param_groups[0]['lr']}
                self.writer.add_scalars('Epoch/Learning rate', lrs, self.epoch)
            else:
                self.writer.add_scalar('Epoch/Learning rate', self.optimizer.param_groups[0]['lr'], self.epoch)
        else:
            ep = epoch
            if not self.config['scheduler'] == 'cosine':
                self.optimizer.param_groups[0]['lr'] = poly_lr(ep, self.max_num_epochs, self.initial_lr, 0.9)
                if self.do_adv:
                    self.discriminator_optimizer.param_groups[0]['lr'] = poly_lr(ep, self.max_num_epochs, self.discriminator_lr, 0.9)

        self.print_to_log_file("lr:", np.round(self.optimizer.param_groups[0]['lr'], decimals=6))

    def on_epoch_end(self):
        """
        overwrite patient-based early stopping. Always run to 1000 epochs
        :return:
        """
        super().on_epoch_end()
        continue_training = self.epoch < self.max_num_epochs

        #if self.config['progressive_similarity_growing']:
        #    epoch_threshold = self.max_num_epochs / 20
        #    if self.epoch + 1 <= epoch_threshold:
        #        a = self.config['similarity_weight'] / epoch_threshold
        #        self.loss_data['similarity'][0] = a * (self.epoch + 1)


        # it can rarely happen that the momentum of nnUNetTrainerV2 is too high for some dataset. If at epoch 100 the
        # estimated validation Dice is still 0 then we reduce the momentum from 0.99 to 0.95
        if self.epoch == 100:
            if self.all_val_eval_metrics[-1] == 0:
                self.optimizer.param_groups[0]["momentum"] = 0.95
                self.network.apply(InitWeights_He(1e-2))
                self.print_to_log_file("At epoch 100, the mean foreground Dice was 0. This can be caused by a too "
                                       "high momentum. High momentum (0.99) is good for datasets where it works, but "
                                       "sometimes causes issues such as this one. Momentum has now been reduced to "
                                       "0.95 and network weights have been reinitialized")
        return continue_training
    

    def maybe_save_checkpoint(self):
        """
        Saves a checkpoint every save_ever epochs.
        :return:
        """
        if self.save_intermediate_checkpoints and (self.epoch % self.save_every == (self.save_every - 1)):
            self.print_to_log_file("saving scheduled checkpoint file...")
            if not self.save_latest_only:
                self.save_checkpoint(join(self.output_folder, "model_ep_%03.0d.model" % (self.epoch + 1)))
                if self.do_adv:
                    self.save_checkpoint_discriminator(join(self.output_folder, "discriminator_ep_%03.0d.model" % (self.epoch + 1)))
            self.save_checkpoint(join(self.output_folder, "model_latest.model"))
            if self.do_adv:
                self.save_checkpoint_discriminator(join(self.output_folder, "discriminator_latest.model"))
            self.print_to_log_file("done")
    

    def run_training_flow(self):
        if not torch.cuda.is_available():
            self.print_to_log_file("WARNING!!! You are attempting to run training on a CPU (torch.cuda.is_available() is False). This can be VERY slow!")

        #_ = self.tr_gen.next()
        #_ = self.val_gen.next()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._maybe_init_amp()

        maybe_mkdir_p(self.output_folder)        
        self.plot_network_architecture()

        if cudnn.benchmark and cudnn.deterministic:
            warn("torch.backends.cudnn.deterministic is True indicating a deterministic training is desired. "
                 "But torch.backends.cudnn.benchmark is True as well and this will prevent deterministic training! "
                 "If you want deterministic then set benchmark=False")

        if not self.was_initialized:
            self.initialize(True)

        while self.epoch < self.max_num_epochs:
            self.print_to_log_file("\nepoch: ", self.epoch)
            epoch_start_time = time()
            train_losses_epoch = []

            # train one epoch
            self.network.train()
            if self.do_adv:
                self.discriminator.train()

            if self.fine_tuning:
                self.freeze_batchnorm_layers(self.network)

            if self.use_progress_bar:
                with trange(self.num_batches_per_epoch) as tbar:
                    for b in tbar:
                        self.epoch_iter_nb = b
                        tbar.set_description("Epoch {}/{}".format(self.epoch+1, self.max_num_epochs))

                        l = self.run_iteration_train(self.dl_tr)

                        tbar.set_postfix(loss=l)
                        train_losses_epoch.append(l)
            else:
                for _ in range(self.num_batches_per_epoch):
                    l = self.run_iteration_train(self.dl_tr)
                    train_losses_epoch.append(l)

            self.all_tr_losses.append(np.mean(train_losses_epoch))
            self.print_to_log_file("train loss : %.4f" % self.all_tr_losses[-1])

            if self.dataloader_modality == 'other':

                with torch.no_grad():
                    # validation with train=False
                    self.network.eval()
                    val_losses = []
                    if self.epoch % self.config['overfit_log'] == 0:
                        for b in range(self.num_val_batches_per_epoch):
                            l = self.run_iteration_overfitting(self.dl_overfitting)
                            val_losses.append(l)
                        self.all_val_losses.append(np.mean(val_losses))
                        self.print_to_log_file("validation loss: %.4f" % self.all_val_losses[-1])
                        self.log_losses()

                    if self.config['log_stats'] and self.epoch % self.config['epoch_log'] == 0:
                        #if any([x in self.task for x in ['31', '35']]):
                        for b in range(self.num_val_batches_per_epoch):
                            self.epoch_iter_nb = b
                            with torch.no_grad():
                                self.run_iteration_val(self.dl_val)
                        self.finish_online_evaluation()
                        max_memory_allocated = torch.cuda.max_memory_allocated(device=self.network.get_device())
                        self.print_to_log_file("Max GPU Memory allocated:", max_memory_allocated / 10e8, "Gb")
                        self.print_to_log_file("Max CPU Memory allocated:", psutil.Process(os.getpid()).memory_info().rss / 10e8, "Gb")
                    

                    if self.also_val_in_tr_mode:
                        self.network.train()
                        if self.do_adv:
                            self.discriminator.train()
                        if self.fine_tuning:
                            self.freeze_batchnorm_layers(self.network)
                        # validation with train=True
                        val_losses = []
                        if self.epoch % self.config['overfit_log'] == 0:
                            for b in range(self.num_val_batches_per_epoch):
                                l = self.run_iteration_overfitting(self.dl_overfitting)
                                val_losses.append(l)
                            self.all_val_losses_tr_mode.append(np.mean(val_losses))
                            self.print_to_log_file("validation loss (train=True): %.4f" % self.all_val_losses_tr_mode[-1])
                            self.log_losses()

                        if self.config['log_stats'] and self.epoch % self.config['epoch_log'] == 0:
                            if any([x in self.task for x in ['31', '35']]):
                                for b in range(self.num_val_batches_per_epoch):
                                    self.epoch_iter_nb = b
                                    self.run_iteration_val(self.dl_val)

                                self.finish_online_evaluation()
                            max_memory_allocated = torch.cuda.max_memory_allocated(device=self.network.get_device())
                            self.print_to_log_file("Max GPU Memory allocated:", max_memory_allocated / 10e8, "Gb")
                            self.print_to_log_file("Max CPU Memory allocated:", psutil.Process(os.getpid()).memory_info().rss / 10e8, "Gb")

            #self.update_train_loss_MA()  # needed for lr scheduler and stopping of training

            continue_training = True
            #continue_training = self.on_epoch_end()

            self.maybe_update_lr()
            self.maybe_save_checkpoint()

            epoch_end_time = time()

            if not continue_training:
                # allows for early stopping
                break

            self.epoch += 1
            self.print_to_log_file("This epoch took %f s\n" % (epoch_end_time - epoch_start_time))

        self.epoch -= 1  # if we don't do this we can get a problem with loading model_final_checkpoint.

        if self.save_final_checkpoint: self.save_checkpoint(join(self.output_folder, "model_final_checkpoint.model"))
        if self.do_adv and self.save_final_checkpoint:
            self.save_checkpoint_discriminator(join(self.output_folder, "discriminator_final_checkpoint.model"))
        # now we can delete latest as it will be identical with final
        if isfile(join(self.output_folder, "model_latest.model")):
            os.remove(join(self.output_folder, "model_latest.model"))
        if isfile(join(self.output_folder, "model_latest.model.pkl")):
            os.remove(join(self.output_folder, "model_latest.model.pkl"))

    
    """ def run_training_flow(self):
        if not torch.cuda.is_available():
            self.print_to_log_file("WARNING!!! You are attempting to run training on a CPU (torch.cuda.is_available() is False). This can be VERY slow!")

        #_ = self.tr_gen.next()
        #_ = self.val_gen.next()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._maybe_init_amp()

        maybe_mkdir_p(self.output_folder)        
        self.plot_network_architecture()

        if cudnn.benchmark and cudnn.deterministic:
            warn("torch.backends.cudnn.deterministic is True indicating a deterministic training is desired. "
                 "But torch.backends.cudnn.benchmark is True as well and this will prevent deterministic training! "
                 "If you want deterministic then set benchmark=False")

        if not self.was_initialized:
            self.initialize(True)

        while self.epoch < self.max_num_epochs:
            self.print_to_log_file("\nepoch: ", self.epoch)
            epoch_start_time = time()
            train_losses_epoch = []

            # train one epoch
            self.network.train()
            if self.do_adv:
                self.discriminator.train()

            if self.fine_tuning:
                self.freeze_batchnorm_layers(self.network)

            if self.use_progress_bar:
                with trange(self.num_batches_per_epoch) as tbar:
                    for b in tbar:
                        self.epoch_iter_nb = b
                        tbar.set_description("Epoch {}/{}".format(self.epoch+1, self.max_num_epochs))

                        l = self.run_iteration_train(self.dl_tr)

                        tbar.set_postfix(loss=l)
                        train_losses_epoch.append(l)
            else:
                for _ in range(self.num_batches_per_epoch):
                    l = self.run_iteration_train(self.dl_tr)
                    train_losses_epoch.append(l)

            self.all_tr_losses.append(np.mean(train_losses_epoch))
            self.print_to_log_file("train loss : %.4f" % self.all_tr_losses[-1])

            with torch.no_grad():
                # validation with train=False
                self.network.eval()
                val_losses = []
                if self.epoch % self.config['overfit_log'] == 0:
                    for b in range(self.num_val_batches_per_epoch):
                        l = self.run_iteration_overfitting(self.dl_overfitting)
                        val_losses.append(l)
                    self.all_val_losses.append(np.mean(val_losses))
                    self.print_to_log_file("validation loss: %.4f" % self.all_val_losses[-1])
                    self.log_losses()
                    
                    self.finish_online_evaluation()
                    max_memory_allocated = torch.cuda.max_memory_allocated(device=self.network.get_device())
                    self.print_to_log_file("Max GPU Memory allocated:", max_memory_allocated / 10e8, "Gb")
                    self.print_to_log_file("Max CPU Memory allocated:", psutil.Process(os.getpid()).memory_info().rss / 10e8, "Gb")
                

                if self.also_val_in_tr_mode:
                    self.network.train()
                    if self.do_adv:
                        self.discriminator.train()
                    if self.fine_tuning:
                        self.freeze_batchnorm_layers(self.network)
                    # validation with train=True
                    val_losses = []
                    if self.epoch % self.config['overfit_log'] == 0:
                        for b in range(self.num_val_batches_per_epoch):
                            l = self.run_iteration_overfitting(self.dl_overfitting)
                            val_losses.append(l)
                        self.all_val_losses_tr_mode.append(np.mean(val_losses))
                        self.print_to_log_file("validation loss (train=True): %.4f" % self.all_val_losses_tr_mode[-1])
                        self.log_losses()

                        self.finish_online_evaluation()
                        max_memory_allocated = torch.cuda.max_memory_allocated(device=self.network.get_device())
                        self.print_to_log_file("Max GPU Memory allocated:", max_memory_allocated / 10e8, "Gb")
                        self.print_to_log_file("Max CPU Memory allocated:", psutil.Process(os.getpid()).memory_info().rss / 10e8, "Gb")

            #self.update_train_loss_MA()  # needed for lr scheduler and stopping of training

            continue_training = True
            #continue_training = self.on_epoch_end()

            self.maybe_update_lr()
            self.maybe_save_checkpoint()

            epoch_end_time = time()

            if not continue_training:
                # allows for early stopping
                break

            self.epoch += 1
            self.print_to_log_file("This epoch took %f s\n" % (epoch_end_time - epoch_start_time))

        self.epoch -= 1  # if we don't do this we can get a problem with loading model_final_checkpoint.

        if self.save_final_checkpoint: self.save_checkpoint(join(self.output_folder, "model_final_checkpoint.model"))
        if self.do_adv and self.save_final_checkpoint:
            self.save_checkpoint_discriminator(join(self.output_folder, "discriminator_final_checkpoint.model"))
        # now we can delete latest as it will be identical with final
        if isfile(join(self.output_folder, "model_latest.model")):
            os.remove(join(self.output_folder, "model_latest.model"))
        if isfile(join(self.output_folder, "model_latest.model.pkl")):
            os.remove(join(self.output_folder, "model_latest.model.pkl")) """

    def run_training(self):
        """
        if we run with -c then we need to set the correct lr for the first epoch, otherwise it will run the first
        continued epoch with self.initial_lr

        we also need to make sure deep supervision in the network is enabled for training, thus the wrapper
        :return:
        """
        self.maybe_update_lr(self.epoch)  # if we dont overwrite epoch then self.epoch+1 is used which is not what we
        # want at the start of the training
        ds = self.network.do_ds
        #self.network.do_ds = True
        self.save_debug_information()
        ret = self.run_training_flow()
        self.network.do_ds = ds
        return ret

    
    def load_dataset(self):
        self.dataset = load_dataset(self.folder_with_preprocessed_data)
        self.unlabeled_dataset = load_unlabeled_dataset(self.folder_with_preprocessed_data)
    
    def get_basic_generators(self):
        self.load_dataset()
        self.do_split()
        dl_un_tr = None
        dl_un_val = None
        
        if any([x in self.task for x in ['32', '36', '45']]):

            if self.dataloader_modality == 'all_first':
                assert self.video_length == 2
                dataloader_class = DataLoaderFlowLibProgressiveAllDataFirst
            elif self.dataloader_modality == 'all_adjacent':
                assert self.video_length == 2
                dataloader_class = DataLoaderPreprocessedAdjacent
            elif self.dataloader_modality == 'regular':
                dataloader_class = DataLoaderFlowTrain5LibRegular
            elif self.dataloader_modality == 'other':
                dataloader_class = DataLoaderPreprocessed if not self.supervised else DataLoaderPreprocessedSupervised
                #dataloader_class = DataLoaderAugment
                #dataloader_class = DataLoaderFlowTrain5LibProgressive
                #dataloader_class = DataLoaderFlowTrain5LibProgressiveNoSorting

            dl_val = DataLoaderPreprocessedValidation(self.dataset_val, self.patch_size, self.patch_size, 1, do_data_aug=False, video_length=self.video_length,
                                    crop_size=self.crop_size, processor=self.processor, is_val=True, distance_map_power=self.distance_map_power, 
                                    binary_distance_input=self.binary_distance_input, binary_distance_loss=self.binary_distance_loss, start_es=self.start_es, oversample_foreground_percent=self.oversample_foreground_percent,
                                    pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
            
            dl_tr = dataloader_class(self.dataset_tr, self.basic_generator_patch_size, self.patch_size, self.batch_size, do_data_aug=self.do_data_aug, video_length=self.video_length,
                                            crop_size=self.crop_size, processor=self.processor, is_val=False, data_path=r'Lib_resampling_training_mask', distance_map_power=self.distance_map_power, binary_distance_input=self.binary_distance_input, binary_distance_loss=self.binary_distance_loss, start_es=self.start_es, oversample_foreground_percent=self.oversample_foreground_percent,
                                            pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
                
            dl_overfitting = dataloader_class(self.dataset_val, self.basic_generator_patch_size, self.patch_size, self.batch_size, do_data_aug=False, video_length=self.video_length,
                                            crop_size=self.crop_size, processor=self.processor, is_val=True, data_path=r'Lib_resampling_training_mask', distance_map_power=self.distance_map_power, binary_distance_input=self.binary_distance_input, binary_distance_loss=self.binary_distance_loss, start_es=self.start_es,  oversample_foreground_percent=self.oversample_foreground_percent,
                                            pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
        else:

            if self.dataloader_modality == 'all_first':
                assert self.video_length == 2
                dataloader_class = DataLoaderFlowACDCProgressiveAllDataFirst
            elif self.dataloader_modality == 'all_adjacent':
                assert self.video_length == 2
                dataloader_class = DataLoaderFlowACDCProgressiveAllDataAdjacent
            elif self.dataloader_modality == 'other':
                dataloader_class = DataLoaderFlowTrain5Progressive

            dl_val = dataloader_class(self.dataset_val, self.patch_size, self.patch_size, self.batch_size, unlabeled_dataset=self.dataset_un_val, video_length=self.video_length,
                                    crop_size=self.crop_size, processor=self.processor, is_val=True, oversample_foreground_percent=self.oversample_foreground_percent,
                                    pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
            
            dl_tr = dataloader_class(self.dataset_tr, self.basic_generator_patch_size, self.patch_size, self.batch_size, unlabeled_dataset=self.dataset_un_tr, video_length=self.video_length,
                                        crop_size=self.crop_size, processor=self.processor, is_val=False, oversample_foreground_percent=self.oversample_foreground_percent,
                                        pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
            
            dl_overfitting = dataloader_class(self.dataset_val, self.basic_generator_patch_size, self.patch_size, self.batch_size, unlabeled_dataset=self.dataset_un_val, video_length=self.video_length,
                                        crop_size=self.crop_size, processor=self.processor, is_val=True, oversample_foreground_percent=self.oversample_foreground_percent,
                                        pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
        
        return dl_tr, dl_val, dl_un_tr, dl_un_val, dl_overfitting
    
    def get_adversarial_loss(self, fake):
        label = torch.full((self.batch_size,), 1, dtype=torch.float, device=fake.device)
        output = self.discriminator(fake).view(-1)
        adversarial_loss = self.adv_loss(output, label)
        return adversarial_loss
    
    def get_discriminator_loss(self, x, label):
        label = torch.full((self.batch_size,), label, dtype=torch.float, device=x.device)
        output_real = self.discriminator(x).reshape(-1)
        loss_real = self.adv_loss(output_real, label).reshape(1)
        #loss_real.backward()
        return loss_real, output_real

    def get_fake_loss(self, out, label, detach):
        #if self.deep_supervision:
        #    one_hot_segs = torch.nn.functional.softmax(out['seg'][0], dim=2)
        #else:
        #    one_hot_segs = torch.nn.functional.softmax(out['seg'], dim=2)
        #one_hot_segs = torch.argmax(one_hot_segs, dim=2).long()
        one_hot_segs = torch.nn.functional.one_hot(out['seg_registered'][:, :, 0], num_classes=4).permute(0, 1, 4, 2, 3).float()

        loss_fake_list = []
        output_fake_list = []
        for i in range(len(one_hot_segs)):
            fake = one_hot_segs[i]
            if detach:
                fake = fake.detach()
            
            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(1, fake.shape[1])
            #for i in range(fake.shape[1]):
            #    ax[i].imshow(fake[0, i].detach().cpu(), cmap='gray')
            #plt.show()

            loss_fake, output_fake = self.get_discriminator_loss(fake, label=label)
            loss_fake_list.append(loss_fake)
            output_fake_list.append(output_fake)
        loss_fake = torch.cat(loss_fake_list).mean()
        output_fake = torch.cat(output_fake_list)
        return loss_fake, output_fake
    
    def train_discriminator(self, target, out, target_mask):
        self.discriminator.zero_grad()
        self.discriminator_optimizer.zero_grad()

        indices = torch.nonzero(target_mask, as_tuple=True)

        one_hot_target = target[indices[0], indices[1]]
        one_hot_target = torch.nn.functional.one_hot(one_hot_target.squeeze(1).long(), num_classes=4).permute(0, 3, 1, 2).float()

        #matplotlib.use('QtAgg')
        #fig, ax = plt.subplots(1, 4)
        #for o in range(4):
        #    ax[o].imshow(one_hot_target[0, o].detach().cpu(), cmap='gray')
        #plt.show()

        loss_real, output_real = self.get_discriminator_loss(one_hot_target, label=1)
        loss_real.backward()

        loss_fake, output_fake = self.get_fake_loss(out, label=0, detach=True)
        loss_fake.backward()
        
        discriminator_loss = loss_real + loss_fake

        self.writer.add_scalar('Discriminator/Real', output_real.mean(), self.iter_nb)
        self.writer.add_scalar('Discriminator/Fake', output_fake.mean(), self.iter_nb)
        self.writer.add_scalar('Discriminator/Loss', discriminator_loss, self.iter_nb)

        self.discriminator_optimizer.step()
