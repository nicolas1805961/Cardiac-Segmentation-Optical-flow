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


import psutil
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from nnunet.lib.loss import ImageFlowLoss
import cv2 as cv
import sys
from matplotlib import cm
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
from copy import copy
from time import time, sleep, strftime
import yaml
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
from nnunet.training.dataloading.dataset_loading import DataLoader2D, DataLoader2DMiddleUnlabeled, unpack_dataset, DataLoaderVideoUnlabeled
from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer
from nnunet.utilities.nd_softmax import softmax_helper
from sklearn.model_selection import KFold
from torch import nn
from torch.cuda.amp import autocast
from nnunet.training.learning_rate.poly_lr import poly_lr
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.lib.training_utils import build_2d_model, build_flow_model, build_discriminator, read_config, build_discriminator, build_discriminator, build_video_model
from nnunet.lib.loss import DirectionalFieldLoss, MaximizeDistanceLoss, AverageDistanceLoss
from pathlib import Path
from monai.losses import DiceFocalLoss, DiceLoss
from torch.utils.tensorboard import SummaryWriter
from nnunet.training.loss_functions.dice_loss import DC_and_CE_loss, DC_and_focal_loss, DC_and_topk_loss, DC_and_CE_loss_Weighted, SoftDiceLoss
from nnunet.training.loss_functions.crossentropy import RobustCrossEntropyLoss, WeightedRobustCrossEntropyLoss
from nnunet.training.dataloading.dataset_loading import DataLoaderFlow3
from nnunet.training.dataloading.dataset_loading import load_dataset, load_unlabeled_dataset
from nnunet.network_architecture.Optical_flow_model_3 import ModelWrap
from nnunet.lib.utils import RFR, ConvBlocks, Resblock, LayerNorm, RFR_1d, Resblock1D, ConvBlocks1D, MotionEstimation
from nnunet.lib.loss import SeparabilityLoss, ContrastiveLoss
from nnunet.training.data_augmentation.cutmix import cutmix, batched_rand_bbox
import shutil
from nnunet.visualization.visualization import Visualizer
from nnunet.training.network_training.processor import Processor

class nnMTLTrainerV2Flow3(nnUNetTrainer):
    """
    Info for Fabian: same as internal nnUNetTrainerV2_2
    """

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False, inference=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)

        self.cropper_config = read_config(os.path.join(Path.cwd(), 'adversarial_acdc.yaml'), False, False)
        self.config = read_config(os.path.join(Path.cwd(), 'video.yaml'), False, True)
        self.cropper_weights_folder_path = 'binary'
        self.video_length = self.config['video_length']
        self.crop = self.config['crop']
        self.feature_extractor = self.config['feature_extractor']
        self.crop_size = self.config['crop_size']
        self.force_one_label = self.config['force_one_label']
        self.video_weights_folder_path = self.config['video_weights_folder_path']
        self.area_size = self.config['area_size']
        self.step = self.config['step']
        self.binary = False
        
        self.image_size = self.config['patch_size'][0]
        self.window_size = 7 if self.image_size == 224 else 9 if self.image_size == 288 else None

        self.max_num_epochs = self.config['max_num_epochs']
        self.log_images = self.config['log_images']
        self.initial_lr = self.config['initial_lr']
        self.weight_decay = self.config['weight_decay']
        self.deep_supervision_scales = None
        self.ds_loss_weights = None
        self.use_progress_bar=True
        self.one_vs_all = True
        self.motion_estimation = MotionEstimation()

        #self.fine_tuning = True if self.video_length > 2 else False
        self.fine_tuning = False

        self.deep_supervision = self.config['deep_supervision']
        self.labeled_segmentation_weight = self.config['labeled_segmentation_weight']
        self.unlabeled_segmentation_weight = self.config['unlabeled_segmentation_weight']
        self.regularization_weight_xy = self.config['regularization_weight_xy']
        self.regularization_weight_z = self.config['regularization_weight_z']
        self.adversarial_weight = self.config['adversarial_weight']
        self.do_adv = self.config['do_adv']

        self.strong_network_lr = self.config['strong_network_lr']
        self.strong_network_decay = self.config['strong_network_decay']

        self.iter_nb = 0
        self.epoch_iter_nb = 0

        self.val_loss = []
        self.train_loss = []
        
        #loss_weights = torch.tensor(self.config['224_loss_weights'], device=self.config['device'])
        #self.segmentation_flow_loss_weight = self.config['segmentation_flow_loss_weight']
        self.image_flow_loss_weight = self.config['image_flow_loss_weight']
        #self.long_image_flow_loss_weight = self.config['long_image_flow_loss_weight']

        timestr = strftime("%Y-%m-%d_%HH%M")
        self.log_dir = os.path.join(copy(self.output_folder), timestr)
        self.writer = SummaryWriter(log_dir=self.log_dir)

        if self.log_images:
            self.vis = Visualizer(unlabeled=False,
                                    adversarial_loss=False,
                                    middle_unlabeled=False,
                                    middle=False,
                                    registered_seg=False,
                                    writer=self.writer,
                                    area_size=self.area_size[-1],
                                    crop_size=self.crop_size)
            
        if inference:
            self.output_folder = output_folder
        else:
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

    def setup_loss_data(self):
        loss_data = {'labeled_segmentation': [self.labeled_segmentation_weight, float('nan')]}
        loss_data['unlabeled_segmentation'] = [self.unlabeled_segmentation_weight, float('nan')]
        loss_data['forward_image_flow'] = [self.image_flow_loss_weight, float('nan')]
        loss_data['backward_image_flow'] = [self.image_flow_loss_weight, float('nan')]
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
        #self.image_flow_loss = ImageFlowLoss(alpha=1.0, beta=1.0)
        self.image_flow_loss = ImageFlowLoss(self.writer, w_xy=self.regularization_weight_xy, w_z=self.regularization_weight_z)
        self.seg_flow_loss = self.segmentation_loss
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.adv_loss = nn.BCELoss()


    def initialize_table(self):
        table = {}
        table['seg'] = {'foreground_dc': [], 'tp': [], 'fp': [], 'fn': []}
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

            if self.deep_supervision:
                self.segmentation_loss = MultipleOutputLoss2(self.segmentation_loss, seg_weights)
                self.image_flow_loss = MultipleOutputLoss2(self.image_flow_loss, seg_weights)
                self.seg_flow_loss = MultipleOutputLoss2(self.seg_flow_loss, seg_weights)

            ################# END ###################

            self.folder_with_preprocessed_data = join(self.dataset_directory, self.plans['data_identifier'] +
                                                      "_stage%d" % self.stage)
            self.initialize_network()
            self.initialize_optimizer_and_scheduler()
            self.dl_tr, self.dl_val, self.dl_un_tr, self.dl_un_val = self.get_basic_generators()

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


            assert isinstance(self.weak_network, (SegmentationNetwork, nn.DataParallel))
            assert isinstance(self.strong_network, (SegmentationNetwork, nn.DataParallel))
        else:
            self.print_to_log_file('self.was_initialized is True, not running self.initialize again')
        self.was_initialized = True

    
    def count_parameters(self, config, models):
        params_sum = 0
        self.print_to_log_file(yaml.safe_dump(config, default_flow_style=None, sort_keys=False), also_print_to_console=False)
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
            conv_layer = ConvBlocks
            conv_layer_1d = ConvBlocks1D
        return conv_layer, conv_layer_1d

    def freeze_batchnorm_layers(self, model):
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                if hasattr(module, 'weight'):
                    module.weight.requires_grad_(False)
                if hasattr(module, 'bias'):
                    module.bias.requires_grad_(False)
                module.eval()

    def load_video_weights(self, model):
        weight_path = os.path.join(self.video_weights_folder_path, 'model_final_checkpoint.model')
        loaded_state_dict = torch.load(weight_path)['state_dict']
        #print(loaded_state_dict.keys())
        current_model_dict = model.state_dict()
        #print(current_model_dict.keys())

        new_state_dict = copy(current_model_dict)
        for k, v in loaded_state_dict.items():
            for module_name, module in model.named_modules():
                if list(module.children()) == []:
                    #if isinstance(module, nn.BatchNorm2d) and module_name in k:
                    if module_name in k:
                        new_state_dict[k] = v

        missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)

        assert len(unexpected_keys) == 0, f'Unexpected_keys: {unexpected_keys}'

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
            norm_1d = nn.InstanceNorm1d
        elif wanted_norm == 'instancenorm':
            norm_2d = nn.InstanceNorm2d
            norm_1d = nn.InstanceNorm1d
        
        conv_layer, conv_layer_1d = self.get_conv_layer(self.config)

        in_shape_crop = torch.randn(self.config['batch_size'], 1, self.image_size, self.image_size)
        cropping_conv_layer, _ = self.get_conv_layer(self.cropper_config)
        cropping_network = build_2d_model(self.cropper_config, conv_layer=cropping_conv_layer, norm=getattr(torch.nn, self.cropper_config['norm']), log_function=self.print_to_log_file, image_size=self.image_size, window_size=self.window_size, middle=False, num_classes=2)
        cropping_network.load_state_dict(torch.load(os.path.join(self.cropper_weights_folder_path, 'model_final_checkpoint.model'))['state_dict'], strict=True)
        cropping_network.eval()
        cropping_network.do_ds = False
        models['cropping_model'] = (cropping_network, in_shape_crop)

        #end_idx = int((self.crop_size / 2**len(self.config['conv_depth']))**2)
        #in_shape_mask[:, :, :, :end_idx] = 1
        #in_shape_mask = in_shape_mask.view(self.config['batch_size'], self.video_length, 1, self.crop_size, self.crop_size)
        self.weak_network = build_flow_model_3(self.config, conv_layer=conv_layer, norm=norm_2d, image_size=self.crop_size, log_function=self.print_to_log_file)
        self.strong_network = build_flow_model_3(self.config, conv_layer=conv_layer, norm=norm_2d, image_size=self.crop_size, log_function=self.print_to_log_file)

        #for p1, p2 in zip(weak_network.parameters(), strong_network.parameters()):
        #    if p1.data.ne(p2.data).sum() > 0:
        #        print('Networks do not have same weights')
        #print('Networks have same weights')

        #self.network = ModelWrap(model1=weak_network, model2=strong_network)
        if self.fine_tuning:
            self.print_to_log_file("Loading weights from pretrained network")
            self.load_video_weights(self.weak_network)
            self.load_video_weights(self.strong_network)
        unlabeled_input_data = torch.randn(self.video_length - 1, self.config['batch_size'], 5, self.crop_size, self.crop_size)
        labeled_input_data = torch.randn(self.config['batch_size'], 5, self.crop_size, self.crop_size)
        models['weak network'] = (self.weak_network, [labeled_input_data, unlabeled_input_data])
        models['strong network'] = (self.strong_network, [labeled_input_data, unlabeled_input_data])

        self.processor = Processor(crop_size=self.crop_size, image_size=self.image_size, cropping_network=cropping_network, nb_layers=len(self.config['conv_depth']))

        #self.count_parameters(self.config, models)

        #nb_inputs = 2 if self.middle else 1
        #model_input_size = [(self.config['batch_size'], 1, self.image_size, self.image_size)] * nb_inputs

        if torch.cuda.is_available():
            self.weak_network.cuda()
            self.strong_network.cuda()
        self.weak_network.inference_apply_nonlin = softmax_helper
        self.strong_network.inference_apply_nonlin = softmax_helper

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
        assert self.weak_network is not None, "self.initialize_network must be called first"
        assert self.strong_network is not None, "self.initialize_network must be called first"
        self.optimizer, self.lr_scheduler = self.get_optimizer_scheduler(net=self.weak_network, lr=self.initial_lr, decay=self.weight_decay)

        self.strong_network_optimizer, self.strong_network_scheduler = self.get_optimizer_scheduler(net=self.strong_network, lr=self.strong_network_lr, decay=self.strong_network_decay)

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

    def get_gradient_images(self, labeled, unlabeled, indices):
        t, b, c, y, x = indices
        with torch.enable_grad():
            labeled_idx = -1
            labeled = unlabeled[labeled_idx]
            padding = torch.zeros_like(labeled)[None].repeat(len(unlabeled) - 1, 1, 4, 1, 1)
            unlabeled_in = unlabeled[:-1]
            unlabeled_in = torch.cat([unlabeled_in, padding], dim=2)
            labeled_in = torch.cat([labeled, padding[0]], dim=1)

            self.weak_network.zero_grad()
            unlabeled_in.requires_grad = True
            labeled_in.requires_grad = True
            output = self.weak_network(labeled_in, unlabeled_in)
            predictions = output['forward_flow']
            out = predictions[t, b, c, y, x]
            out.backward()

            gradient_image_unlabeled = torch.abs(unlabeled_in.grad)
            gradient_image_labeled = torch.abs(labeled_in.grad)
            gradient_image_unlabeled = gradient_image_unlabeled[:, b, 0, :, :]
            gradient_image_labeled = gradient_image_labeled[b, 0, :, :]

            unlabeled = unlabeled[:, b, 0]
            labeled = labeled[b, 0]
            return gradient_image_unlabeled, gradient_image_labeled, unlabeled_in[:, b, 0], labeled_in[b, 0]

    
    def start_online_evaluation(self, out,
                                    out_mask,
                                    unlabeled,
                                    target,
                                    gradient_image_unlabeled,
                                    gradient_image_labeled,
                                    gradient_x_unlabeled,
                                    gradient_x_labeled,
                                    coords,
                                    padding_need,
                                    labeled_idx):
        """
            gradient_image: T, H, W
            gradient_x: T, H, W
            """

        uncrop_target = self.processor.uncrop_no_registration(target[None], padding_need=padding_need)[0]
        assert list(uncrop_target.shape[-2:]) == [self.image_size, self.image_size]
        pred = out['labeled_seg']
        num_classes = pred.shape[1]
        output_softmax = torch.softmax(pred, dim=1)
        output_seg = output_softmax.argmax(1)

        #matplotlib.use('QtAgg')
        #fig, ax = plt.subplots(2, len(unlabeled))
        #ax[1, 0].imshow(target[0, 0].cpu(), cmap='gray')
        #for i in range(len(unlabeled)):
        #    ax[0, i].imshow(unlabeled[i, 0, 0].cpu(), cmap='gray')
        #plt.show()

        all_pred = out['unlabeled_seg']
        all_pred_mask = out_mask['unlabeled_seg']
        with torch.no_grad():
            output_softmax_all = torch.softmax(all_pred, dim=2)
            output_softmax_all_mask = torch.softmax(all_pred_mask, dim=2)
            output_seg_all = output_softmax_all.argmax(2)
            output_seg_all_mask = output_softmax_all_mask.argmax(2)

            #registered_seg = torch.softmax(out['registered_seg'], dim=2)
            #registered_seg = torch.argmax(registered_seg, dim=2) # T, B, H, W
            
            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(1, 2)
            #ax[0].imshow(target[0, 0].cpu(), cmap='gray')
            #ax[1].imshow(x[0, 0].cpu(), cmap='gray')
            #plt.show()


            output_seg = self.processor.uncrop_no_registration(output_seg[None, :, None], padding_need=padding_need)[0, :, 0]
            seg_dice = self.compute_dice(uncrop_target[:, 0], num_classes, output_seg, key='seg')

            if self.log_images:
                for b in range(len(target)):
                    
                    current_x = unlabeled[0, b, 0]
                    motion_flow = flow_to_image(out['forward_flow'][:, b]) # 3, H, W
                    #registered_seg = out['registered_seg'][:, b, :]
                    #registered_seg = torch.softmax(registered_seg, dim=1)
                    #registered_seg = torch.argmax(registered_seg, dim=1)
                    
                    #warping_distance = self.l1_loss(out['registered_input_u_to_l'][:, b, 0], labeled[None].repeat(self.video_length, 1, 1, 1, 1)[:, b, 0]).mean()

                    current_pred = torch.argmax(output_softmax[b], dim=0)
                    current_target = target[b, 0]
                    self.vis.set_up_image_seg_best(seg_dice=seg_dice[b].mean(), gt=current_target, pred=current_pred, x=current_x)
                    self.vis.set_up_image_seg_worst(seg_dice=seg_dice[b].mean(), gt=current_target, pred=current_pred, x=current_x)
                    self.vis.set_up_image_gradient(unlabeled_gradient=gradient_image_unlabeled,
                                                        labeled_gradient=gradient_image_labeled,
                                                        unlabeled_x=gradient_x_unlabeled,
                                                        labeled_x=gradient_x_labeled,
                                                        gradient_coords=coords)
                    self.vis.set_up_image_flow(seg_dice=seg_dice[b].mean(), 
                                                moving=unlabeled[-1, b, 0], 
                                                registered_input=out['registered_input_forward'][:, b, 0],
                                                registered_seg=current_target,
                                                target=current_target,
                                                fixed=unlabeled[:-1, b, 0],
                                                motion_flow=motion_flow)
                    #self.vis.set_up_long_registered_image(seg_dice=seg_dice[b].mean(),
                    #                                      moving=unlabeled[0, b, 0],
                    #                                      registered_input=out['registered_input_forward_long'][b, 0],
                    #                                      fixed=unlabeled[-1, b, 0])
                    
                    self.vis.set_up_image_seg_sequence(seg_dice=seg_dice[b].mean(), gt=current_target, pred=output_seg_all[:, b], x=unlabeled[:-1, b, 0])
                    self.vis.set_up_image_seg_sequence_mask(seg_dice=seg_dice[b].mean(), gt=current_target, pred=output_seg_all_mask[:, b], x=unlabeled[:-1, b, 0])
                    
                    #self.vis.set_up_image_weights(seg_dice=seg_dice[b].mean(), x=current_x, weights=out['weights'][b].mean(0))
                
    
    def get_dc_per_class(self, key):
        self.table[key]['tp'] = np.sum(self.table[key]['tp'], 0)
        self.table[key]['fp'] = np.sum(self.table[key]['fp'], 0)
        self.table[key]['fn'] = np.sum(self.table[key]['fn'], 0)

        global_dc_per_class = [i for i in [2 * i / (2 * i + j + k) for i, j, k in
                                           zip(self.table[key]['tp'], self.table[key]['fp'], self.table[key]['fn'])]
                               if not np.isnan(i)]
        
        return global_dc_per_class

        
    def finish_online_evaluation(self):
        #self.online_eval_tp = np.sum(self.online_eval_tp, 0)
        #self.online_eval_fp = np.sum(self.online_eval_fp, 0)
        #self.online_eval_fn = np.sum(self.online_eval_fn, 0)

        #global_dc_per_class = [i for i in [2 * i / (2 * i + j + k) for i, j, k in
        #                                   zip(self.online_eval_tp, self.online_eval_fp, self.online_eval_fn)]
        #                       if not np.isnan(i)]
            
        global_dc_per_class_seg = self.get_dc_per_class('seg')
        self.writer.add_scalar('Epoch/Dice', torch.tensor(global_dc_per_class_seg).mean().item(), self.epoch)
        self.all_val_eval_metrics.append(np.mean(global_dc_per_class_seg))
        self.print_to_log_file("(interpret this as an estimate for the Dice of the different classes. This is not exact.)")
        self.print_to_log_file("Average global foreground Dice:", [np.round(i, 4) for i in global_dc_per_class_seg])

        class_dice = {'RV': global_dc_per_class_seg[0], 'MYO': global_dc_per_class_seg[1], 'LV': global_dc_per_class_seg[2]}
        overfit_data = {'Train': torch.tensor(self.train_loss).mean().item(), 'Val': torch.tensor(self.val_loss).mean().item()}
        self.writer.add_scalars('Epoch/Train_vs_val_loss', overfit_data, self.epoch)
        self.writer.add_scalars('Epoch/Class dice', class_dice, self.epoch)            

        if self.log_images:
            cmap, norm = self.vis.get_custom_colormap()
            self.vis.log_sequence_seg_images(colormap_seg=cmap, norm=norm, epoch=self.epoch)
            self.vis.log_sequence_seg_mask_images(colormap_seg=cmap, norm=norm, epoch=self.epoch)
            self.vis.log_best_seg_images(colormap=cmap, norm=norm, epoch=self.epoch)
            self.vis.log_worst_seg_images(colormap=cmap, norm=norm, epoch=self.epoch)
            self.vis.log_gradient_images(colormap=cm.plasma, epoch=self.epoch)
            #self.vis.log_worst_gradient_images(colormap=cm.plasma, epoch=self.epoch)
            #self.vis.log_long_registered_images(epoch=self.epoch)
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
        self.val_loss = []
        self.train_loss = []
        #self.sim_l2_md_list = []

    def sample_indices(self, cropped_target):
        if torch.count_nonzero(cropped_target) == 0:
            T, B, C, H, W = cropped_target.shape
            t = random.randint(0, T - 1)
            b = random.randint(0, B - 1)
            c = random.randint(0, C - 1)
            y = random.randint(0, H - 1)
            x = random.randint(0, W - 1)
        else:
            indices = torch.nonzero(cropped_target > 0)
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
    
    def run_online_evaluation(self, out, out_mask, labeled, unlabeled, target, padding_need, labeled_idx):
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
        indices_da = None

        indices = self.sample_indices(target[None].repeat(self.video_length - 1, 1, 1, 1, 1))
        gradient_image_unlabeled, gradient_image_labeled, gradient_x_unlabeled, gradient_x_labeled = self.get_gradient_images(labeled, unlabeled, indices)
        coords = (indices[0], indices[4], indices[3]) # (t, x, y)

        return self.start_online_evaluation(out=out,
                                            out_mask=out_mask,
                                            unlabeled=unlabeled,
                                            target=target,
                                            gradient_image_unlabeled=gradient_image_unlabeled,
                                            gradient_image_labeled=gradient_image_labeled,
                                            gradient_x_unlabeled=gradient_x_unlabeled,
                                            gradient_x_labeled=gradient_x_labeled,
                                            coords=coords,
                                            padding_need=padding_need,
                                            labeled_idx=labeled_idx)

    def validate(self, do_mirroring: bool = True, use_sliding_window: bool = True,
                 step_size: float = 0.5, save_softmax: bool = True, use_gaussian: bool = True, overwrite: bool = True,
                 validation_folder_name: str = 'validation_raw', debug: bool = False, all_in_gpu: bool = False,
                 segmentation_export_kwargs: dict = None, run_postprocessing_on_folds: bool = True, output_folder=None):
        """
        We need to wrap this because we need to enforce self.network.do_ds = False for prediction
        """ 
        ds = self.network.do_ds
        self.network.do_ds = False
        ret = super().validate_flow(processor=self.processor, log_function=self.print_to_log_file, step=self.step, do_mirroring=do_mirroring, use_sliding_window=use_sliding_window, step_size=step_size,
                            save_softmax=save_softmax, use_gaussian=use_gaussian,
                            overwrite=overwrite, validation_folder_name=validation_folder_name,
                            all_in_gpu=all_in_gpu, segmentation_export_kwargs=segmentation_export_kwargs,
                            run_postprocessing_on_folds=run_postprocessing_on_folds, debug=True, output_folder=output_folder, save_flow=True)

        self.network.do_ds = ds
        return ret


    def predict_preprocessed_data_return_seg_and_softmax(self, data: np.ndarray, do_mirroring: bool = True,
                                                         mirror_axes: Tuple[int] = None,
                                                         use_sliding_window: bool = True, step_size: float = 0.5,
                                                         use_gaussian: bool = True, pad_border_mode: str = 'constant',
                                                         pad_kwargs: dict = None, all_in_gpu: bool = False,
                                                         verbose: bool = True, mixed_precision=True) -> Tuple[np.ndarray, np.ndarray]:
        """
        We need to wrap this because we need to enforce self.network.do_ds = False for prediction
        """
        ds = self.network.do_ds
        self.network.do_ds = False
        ret = super().predict_preprocessed_data_return_seg_and_softmax(data,
                                                                       do_mirroring=do_mirroring,
                                                                       mirror_axes=mirror_axes,
                                                                       use_sliding_window=use_sliding_window,
                                                                       step_size=step_size, use_gaussian=use_gaussian,
                                                                       pad_border_mode=pad_border_mode,
                                                                       pad_kwargs=pad_kwargs, all_in_gpu=all_in_gpu,
                                                                       verbose=verbose,
                                                                       mixed_precision=mixed_precision)
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
    
    def get_percent_same(self, unlabeled_seg, unlabeled_seg_mask):
        unlabeled_seg = torch.softmax(unlabeled_seg, dim=2)
        unlabeled_seg = torch.argmax(unlabeled_seg, dim=2, keepdim=True)
        unlabeled_seg = torch.nn.functional.one_hot(unlabeled_seg.squeeze(2).long(), num_classes=4).permute(0, 1, 4, 2, 3).float()
        
        unlabeled_seg_mask = torch.softmax(unlabeled_seg_mask, dim=2)
        unlabeled_seg_mask = torch.argmax(unlabeled_seg_mask, dim=2, keepdim=True)
        unlabeled_seg_mask = torch.nn.functional.one_hot(unlabeled_seg_mask.squeeze(2).long(), num_classes=4).permute(0, 1, 4, 2, 3).float()
        percent_same = torch.logical_xor(unlabeled_seg_mask, unlabeled_seg).sum() / torch.numel(unlabeled_seg_mask)

        self.writer.add_scalar('Iteration/percent_not_same', percent_same, self.iter_nb)

    def compute_losses(self, index, unlabeled, out, out_mask, target, labeled):

        #matplotlib.use('QtAgg')
        #fig, ax = plt.subplots(1, len(unlabeled))
        #for i in range(len(unlabeled)):
        #    ax[i].imshow(unlabeled[i, 0, 0].cpu(), cmap='gray')
        #plt.show()
#
        #matplotlib.use('QtAgg')
        #fig, ax = plt.subplots(1, 1)
        #ax.imshow(labeled[0, 0].cpu(), cmap='gray')
        #plt.show()
#
        #matplotlib.use('QtAgg')
        #fig, ax = plt.subplots(1, 1)
        #ax.imshow(target[0, 0].cpu(), cmap='gray')
        #plt.show()

        #regularization_weight = torch.sigmoid(self.network.regularization_loss_weight)

        #if not isval:
        #    t, b, c, y, x = self.sample_indices(target[None].repeat(self.video_length - 1, 1, 1, 1, 1)) # t, b, c, y, x
        #    predictions = out['forward_flow'][t, b, c, y, x]
        #    
        #    gradients = torch.autograd.grad(
        #        outputs=predictions,
        #        inputs=unlabeled,
        #        #grad_outputs=[torch.ones_like(out['forward_flow'][0, :, :, 0, 0])] * len(li),
        #        #create_graph=True,
        #        retain_graph=True,
        #    )[0]
#
        #    gradients = torch.abs(gradients)
#
        #    #for i in range(self.video_length):
        #    #    print(gradients[i].mean())
        #    #print('**********************************')
#
        #    #matplotlib.use('QtAgg')
        #    #fig, ax = plt.subplots(1, self.video_length)
        #    #for i in range(self.video_length):
        #    #    ax[i].imshow(gradients[i, 0, 0].cpu(), cmap='plasma')
        #    #plt.show()
#
        #    strong = gradients[t:t+2]
        #    weak = torch.cat([gradients[:t], gradients[t+2:]], dim=0)
    #
        #    #matplotlib.use('QtAgg')
        #    #fig, ax = plt.subplots(1, self.video_length - 2)
        #    #for i in range(self.video_length - 2):
        #    #    ax[i].imshow(weak[i, 0, 0].cpu(), cmap='plasma')
        #    #plt.show()
#
        #    strong = strong.permute(1, 2, 0, 3, 4).contiguous()
        #    weak = weak.permute(1, 2, 0, 3, 4).contiguous()
        #    strong = torch.flatten(strong, start_dim=-3)
        #    weak = torch.flatten(weak, start_dim=-3)
#
        #    gradient_l = self.gradient_loss(weak.mean(-1), strong.mean(-1))
        #    self.loss_data['gradient'][1] = gradient_l


        #matplotlib.use("QtAgg")
        #fig, ax = plt.subplots(2, len(output_l))
        #for i in range(len(output_l)):
        #    if len(output_l) == 1:
        #        ax[0].imshow(input_l[0, 0].cpu(), cmap='gray')   
        #        ax[1].imshow(target_l[0, 0].cpu(), cmap='gray')  
        #    else:
        #        ax[0, i].imshow(input_l[i, 0].cpu(), cmap='gray')   
        #        ax[1, i].imshow(target_l[i, 0].cpu(), cmap='gray')  
        #plt.show() 

        self.writer.add_scalar('Iteration/flow_magnitude', torch.abs(out['forward_flow']).mean(), self.iter_nb)

        B, C, H, W = target.shape

        #unlabeled = unlabeled.permute(1, 2, 3, 4, 0).contiguous()

        #if gradient_labeled is not None:
        #    gradient_loss = self.gradient_distance(gradient_labeled, gradient_unlabeled)
        #    self.loss_data['gradient_distance'][1] = gradient_loss

        #matplotlib.use('QtAgg')
        #fig, ax = plt.subplots(1, 2)
        #ax[0].imshow(unlabeled[index].cpu()[0, 0], cmap='gray')
        #ax[1].imshow(target.cpu()[0, 0], cmap='gray')
        #plt.show()

        labeled_seg = torch.softmax(out_mask['labeled_seg'], dim=1)
        labeled_seg = torch.argmax(labeled_seg, dim=1, keepdim=True)
        seg_loss_l = self.segmentation_loss(out['labeled_seg'], labeled_seg)
        self.loss_data['labeled_segmentation'][1] = seg_loss_l

        self.get_percent_same(out['unlabeled_seg'], out_mask['unlabeled_seg'])

        unlabeled_seg = torch.softmax(out_mask['unlabeled_seg'], dim=2)
        unlabeled_seg = torch.argmax(unlabeled_seg, dim=2, keepdim=True)

        assert len(unlabeled_seg) == len(unlabeled) - 1
        seg_loss_u_list = []
        for i in range(len(unlabeled) - 1):
            registered_input_forward = self.motion_estimation(flow=out['forward_flow'][i], original=labeled)
            out['registered_input_forward'].append(registered_input_forward)
            registered_input_forward_masked = self.motion_estimation(flow=out_mask['forward_flow'][i], original=labeled)
            out_mask['registered_input_forward'].append(registered_input_forward_masked)

            registered_input_backward = self.motion_estimation(flow=out['backward_flow'][i], original=unlabeled[i])
            out['registered_input_backward'].append(registered_input_backward)
            registered_input_backward_masked = self.motion_estimation(flow=out_mask['backward_flow'][i], original=unlabeled[i])
            out_mask['registered_input_backward'].append(registered_input_backward_masked)

            seg_loss_u = self.segmentation_loss(out['unlabeled_seg'][i], unlabeled_seg[i].detach())
            seg_loss_u_list.append(seg_loss_u)

        seg_loss_u_list = torch.stack(seg_loss_u_list)
        self.loss_data['unlabeled_segmentation'][1] = seg_loss_u_list.mean()

        out['registered_input_forward'] = torch.stack(out['registered_input_forward'], dim=0)
        out['registered_input_backward'] = torch.stack(out['registered_input_backward'], dim=0)
        out_mask['registered_input_forward'] = torch.stack(out_mask['registered_input_forward'], dim=0)
        out_mask['registered_input_backward'] = torch.stack(out_mask['registered_input_backward'], dim=0)

        forward_l = self.image_flow_loss(registered=out['registered_input_forward'], target=unlabeled[:-1], flow=out['forward_flow'])
        backward_l = self.image_flow_loss(registered=out['registered_input_backward'], target=labeled, flow=out['backward_flow'])
        #forward_u = self.image_flow_loss(registered=out_mask['registered_input_forward'], target=unlabeled[:-1], flow=out_mask['forward_flow'])
        #backward_u = self.image_flow_loss(registered=out_mask['registered_input_backward'], target=labeled, flow=out_mask['backward_flow'], iter_nb=self.iter_nb)
        #self.loss_data['forward_image_flow'][1] = (forward_l + forward_u) / 2
        self.loss_data['forward_image_flow'][1] = forward_l
        #self.loss_data['backward_image_flow'][1] = (backward_l + backward_u) / 2
        self.loss_data['backward_image_flow'][1] = backward_l

        #out['forward_flow'][:-1] - out['forward_flow'][1:]

        return out
    


    def compute_losses_strong(self, index, unlabeled, out_mask, target, labeled):

        seg_loss_l_masked = self.segmentation_loss(out_mask['labeled_seg'], target)

        self.writer.add_scalar('Iteration/strong_labeled_loss', seg_loss_l_masked, self.iter_nb)

        #seg_loss_u_list = []
        #for i in range(len(unlabeled) - 1):
        #    registered_input_forward_masked = self.motion_estimation(flow=out_mask['forward_flow'][i], original=labeled)
        #    out_mask['registered_input_forward'].append(registered_input_forward_masked)
#
        #    registered_input_backward_masked = self.motion_estimation(flow=out_mask['backward_flow'][i], original=unlabeled[i])
        #    out_mask['registered_input_backward'].append(registered_input_backward_masked)
#
        #seg_loss_u_list = torch.stack(seg_loss_u_list)
        #self.loss_data['unlabeled_segmentation'][1] = seg_loss_u_list.mean()
#
        #out_mask['registered_input_forward'] = torch.stack(out_mask['registered_input_forward'], dim=0)
        #out_mask['registered_input_backward'] = torch.stack(out_mask['registered_input_backward'], dim=0)
#
        #
        #forward_u = self.image_flow_loss(registered=out_mask['registered_input_forward'], target=unlabeled[:-1], flow=out_mask['forward_flow'])
        #backward_u = self.image_flow_loss(registered=out_mask['registered_input_backward'], target=labeled, flow=out_mask['backward_flow'], iter_nb=self.iter_nb)
        #self.loss_data['forward_image_flow'][1] = forward_u
        #self.loss_data['backward_image_flow'][1] = backward_u

        #out['forward_flow'][:-1] - out['forward_flow'][1:]

        return seg_loss_l_masked


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

        #labeled_idx = data_dict['labeled_idx']
        #assert labeled_idx == self.video_length - 1
        #labeled = unlabeled[labeled_idx]
        #mask = torch.nn.functional.one_hot(target.squeeze(1).long(), num_classes=4).permute(0, 3, 1, 2).float()
        #padding = torch.zeros_like(mask)[None].repeat(len(unlabeled) - 1, 1, 1, 1, 1)
        #unlabeled_in = unlabeled[:-1]
        #unlabeled_in = torch.cat([unlabeled_in, padding], dim=2)
        #labeled_in = torch.cat([labeled, padding[0]], dim=1)
        #mask_in = torch.cat([labeled, mask], dim=1)

        labeled_idx = data_dict['labeled_idx']
        labeled = unlabeled[labeled_idx]
        mask = torch.nn.functional.one_hot(target.squeeze(1).long(), num_classes=4).permute(0, 3, 1, 2).float()
        padding = torch.zeros_like(mask)[None].repeat(len(unlabeled) - 1, 1, 1, 1, 1)
        unlabeled_in = torch.cat([unlabeled[:labeled_idx], unlabeled[labeled_idx + 1:]], dim=0)
        unlabeled_in = torch.cat([unlabeled_in, padding], dim=2)
        labeled_in = torch.cat([labeled, padding[0]], dim=1)
        mask_in = torch.cat([labeled, mask], dim=1)

        #matplotlib.use('QtAgg')
        #fig, ax = plt.subplots(3, 5)
        #for i in range(5):
        #    ax[0, i].imshow(labeled_in[0, i].cpu(), cmap='gray')
        #    ax[1, i].imshow(mask_in[0, i].cpu(), cmap='gray')
        #    ax[2, i].imshow(unlabeled_in[0, 0, i].cpu(), cmap='gray')
        #plt.show()

        out = self.weak_network(labeled_in, unlabeled_in)
        out_mask = self.strong_network(mask_in, unlabeled_in)
        self.optimizer.zero_grad()
        self.strong_network_optimizer.zero_grad()

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
        out = self.compute_losses(index=labeled_idx, unlabeled=unlabeled, out=out, out_mask=out_mask, target=target, labeled=labeled)
        l = self.consolidate_only_one_loss_data(self.loss_data, log=True)

        #if self.iter_nb > 500:
        #    matplotlib.use('QtAgg')
        #    fig, ax = plt.subplots(1, 1)
        #    ax.imshow(out['motion_flow_u_to_l'][0, 0, 0].detach().cpu(), cmap='gray')
        #    plt.show()
        #del loss_data

        self.train_loss.append(l.mean().detach().cpu())
        l.mean().backward()
        torch.nn.utils.clip_grad_norm_(self.weak_network.parameters(), 12)
        self.optimizer.step()

        if self.iter_nb == 0:
            for param_name, param in self.weak_network.named_parameters():
                for module_name, module in self.weak_network.named_modules():
                    if list(module.children()) == []:
                        if not isinstance(module, nn.BatchNorm2d) and module_name in param_name:
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

        l_strong = self.compute_losses_strong(index=labeled_idx, unlabeled=unlabeled, out_mask=out_mask, target=target, labeled=labeled)

        l_strong.mean().backward()
        torch.nn.utils.clip_grad_norm_(self.strong_network.parameters(), 12)
        self.strong_network_optimizer.step()

        #if self.iter_nb > 500:
        #    matplotlib.use('QtAgg')
        #    fig, ax = plt.subplots(1, 1)
        #    ax.imshow(out['motion_flow_u_to_l'][0, 0, 0].detach().cpu(), cmap='gray')
        #    plt.show()
        #del loss_data

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
        unlabeled = data_dict['unlabeled']
        target = data_dict['target']

        #labeled = maybe_to_torch(labeled)
        #unlabeled = maybe_to_torch(unlabeled)
        #target = maybe_to_torch(target)
        #if torch.cuda.is_available():
        #    labeled = to_cuda(labeled)
        #    unlabeled = to_cuda(unlabeled)
        #    target = to_cuda(target)

        labeled_idx = data_dict['labeled_idx']
        assert labeled_idx == self.video_length - 1
        labeled = unlabeled[labeled_idx]
        mask = torch.nn.functional.one_hot(target.squeeze(1).long(), num_classes=4).permute(0, 3, 1, 2).float()
        padding = torch.zeros_like(mask)[None].repeat(len(unlabeled) - 1, 1, 1, 1, 1)
        unlabeled_in = unlabeled[:-1]
        unlabeled_in = torch.cat([unlabeled_in, padding], dim=2)
        labeled_in = torch.cat([labeled, padding[0]], dim=1)
        mask_in = torch.cat([labeled, mask], dim=1)

        #matplotlib.use('QtAgg')
        #fig, ax = plt.subplots(3, len(unlabeled))
        #ax[0, 0].imshow(labeled[0, 0].cpu(), cmap='gray')
        #ax[1, 0].imshow(target[0, 0].cpu(), cmap='gray')
        #for i in range(len(unlabeled)):
        #    ax[2, i].imshow(unlabeled[i, 0, 0].cpu(), cmap='gray')
        #plt.show()


        out = self.weak_network(labeled_in, unlabeled_in)
        out_mask = self.strong_network(mask_in, unlabeled_in)
        out = self.compute_losses(index=labeled_idx, unlabeled=unlabeled, out=out, out_mask=out_mask, target=target, labeled=labeled)
        l = self.consolidate_only_one_loss_data(self.loss_data, log=False)
        #del loss_data

        self.val_loss.append(l.mean().detach().cpu())

        if self.log_images:
            self.run_online_evaluation(out=out,
                                       out_mask=out_mask,
                                       labeled=labeled,
                                       unlabeled=unlabeled,
                                       target=target,
                                       padding_need=data_dict['padding_need'],
                                       labeled_idx=data_dict['labeled_idx'])

        return l.mean().detach().cpu().numpy()
    

    def consolidate_only_one_loss_data(self, loss_data, log):
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
                self.strong_network_scheduler.step()
            else:
                self.optimizer.param_groups[0]['lr'] = poly_lr(ep, self.max_num_epochs, self.initial_lr, 0.9)
                self.strong_network_optimizer.param_groups[0]['lr'] = poly_lr(ep, self.max_num_epochs, self.strong_network_lr, 0.9)
            lrs = {'Flow_lr': self.optimizer.param_groups[0]['lr'], 'Strong_network_lr': self.strong_network_optimizer.param_groups[0]['lr']}
            self.writer.add_scalars('Epoch/Learning rate', lrs, self.epoch)
            #self.writer.add_scalar('Epoch/Learning rate', self.optimizer.param_groups[0]['lr'], self.epoch)
        else:
            ep = epoch
            if not self.config['scheduler'] == 'cosine':
                self.optimizer.param_groups[0]['lr'] = poly_lr(ep, self.max_num_epochs, self.initial_lr, 0.9)
                self.strong_network_optimizer.param_groups[0]['lr'] = poly_lr(ep, self.max_num_epochs, self.strong_network_lr, 0.9)

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
            self.weak_network.train()
            self.strong_network.train()

            if self.fine_tuning:
                self.freeze_batchnorm_layers(self.weak_network)
                self.freeze_batchnorm_layers(self.strong_network)

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

            if self.epoch % self.config['epoch_log'] == 0:
                with torch.no_grad():
                    # validation with train=False
                    self.weak_network.eval()
                    self.strong_network.eval()
                    val_losses = []
                    for b in range(self.num_val_batches_per_epoch):
                        self.epoch_iter_nb = b
                        l = self.run_iteration_val(self.dl_val)
                        val_losses.append(l)
                    self.all_val_losses.append(np.mean(val_losses))
                    self.print_to_log_file("validation loss: %.4f" % self.all_val_losses[-1])

                    if self.also_val_in_tr_mode:
                        self.weak_network.train()
                        self.strong_network.train()
                        if self.fine_tuning:
                            self.freeze_batchnorm_layers(self.weak_network)
                            self.freeze_batchnorm_layers(self.strong_network)
                        # validation with train=True
                        val_losses = []
                        for b in range(self.num_val_batches_per_epoch):
                            self.epoch_iter_nb = b
                            l = self.run_iteration_val(self.dl_val)
                            val_losses.append(l)
                        self.all_val_losses_tr_mode.append(np.mean(val_losses))
                        self.print_to_log_file("validation loss (train=True): %.4f" % self.all_val_losses_tr_mode[-1])

                    self.finish_online_evaluation()
                    max_memory_allocated = torch.cuda.max_memory_allocated(device=self.weak_network.get_device())
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
        # now we can delete latest as it will be identical with final
        if isfile(join(self.output_folder, "model_latest.model")):
            os.remove(join(self.output_folder, "model_latest.model"))
        if isfile(join(self.output_folder, "model_latest.model.pkl")):
            os.remove(join(self.output_folder, "model_latest.model.pkl"))

    def run_training(self):
        """
        if we run with -c then we need to set the correct lr for the first epoch, otherwise it will run the first
        continued epoch with self.initial_lr

        we also need to make sure deep supervision in the network is enabled for training, thus the wrapper
        :return:
        """
        self.maybe_update_lr(self.epoch)  # if we dont overwrite epoch then self.epoch+1 is used which is not what we
        # want at the start of the training
        ds = self.weak_network.do_ds
        #self.network.do_ds = True
        self.save_debug_information()
        ret = self.run_training_flow()
        self.weak_network.do_ds = ds
        self.strong_network.do_ds = ds
        return ret

    
    def load_dataset(self):
        self.dataset = load_dataset(self.folder_with_preprocessed_data)
        self.unlabeled_dataset = load_unlabeled_dataset(self.folder_with_preprocessed_data)
    
    def get_basic_generators(self):
        self.load_dataset()
        self.do_split()
        dl_un_tr = None
        dl_un_val = None
        dl_tr = DataLoaderFlow3(self.dataset_tr, self.basic_generator_patch_size, self.patch_size, self.batch_size, is_val=False, unlabeled_dataset=self.dataset_un_tr, video_length=self.video_length, step=self.step, force_one_label=self.force_one_label,
                                    crop_size=self.crop_size, processor=self.processor, oversample_foreground_percent=self.oversample_foreground_percent,
                                    pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
        dl_val = DataLoaderFlow3(self.dataset_val, self.patch_size, self.patch_size, self.batch_size, is_val=True, unlabeled_dataset=self.dataset_un_val, video_length=self.video_length, step=self.step, force_one_label=self.force_one_label,
                                    crop_size=self.crop_size, processor=self.processor, oversample_foreground_percent=self.oversample_foreground_percent,
                                    pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
        return dl_tr, dl_val, dl_un_tr, dl_un_val
    
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
    

    def get_fake_loss_2(self, out, label, detach):
        one_hot_segs = torch.nn.functional.softmax(out['seg'], dim=2)
        one_hot_segs = torch.argmax(one_hot_segs, dim=2).long()
        one_hot_segs = torch.nn.functional.one_hot(one_hot_segs, num_classes=4).permute(0, 1, 4, 2, 3).float()
        video_pred_forward = out['registered_input_forward'].squeeze(2).permute(1, 0, 2, 3).contiguous()
        video_pred_backward = out['registered_input_backward'].squeeze(2).permute(1, 0, 2, 3).contiguous()
        middle = torch.stack([video_pred_backward[:, 1:], video_pred_forward[:, :-1]], dim=0).mean(0)
        video_pred = torch.cat([video_pred_backward[:, 0][:, None], middle, video_pred_forward[:, -1][:, None]], dim=1)
        loss_fake_list = []
        output_fake_list = []
        for i in range(len(one_hot_segs)):
            fake = torch.cat([one_hot_segs[i], video_pred], dim=1)
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

    def get_fake_loss(self, out, label, detach):
        one_hot_segs = torch.nn.functional.softmax(out['seg'], dim=2)
        one_hot_segs = torch.argmax(one_hot_segs, dim=2).long()
        one_hot_segs = torch.nn.functional.one_hot(one_hot_segs, num_classes=4).permute(0, 1, 4, 2, 3).float()
        video_pred_forward = out['registered_input_forward'].squeeze(2).permute(1, 0, 2, 3).contiguous()
        video_pred_backward = out['registered_input_backward'].squeeze(2).permute(1, 0, 2, 3).contiguous()
        middle = torch.stack([video_pred_backward[:, 1:], video_pred_forward[:, :-1]], dim=0).mean(0)
        video_pred = torch.cat([video_pred_backward[:, 0][:, None], middle, video_pred_forward[:, -1][:, None]], dim=1)
        loss_fake_list = []
        output_fake_list = []
        for i in range(len(one_hot_segs)):
            fake = torch.cat([one_hot_segs[i], video_pred], dim=1)
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
    
    def train_discriminator(self, target, video, out):
        self.discriminator.zero_grad()
        self.discriminator_optimizer.zero_grad()

        one_hot_target = torch.nn.functional.one_hot(target.squeeze(1).long(), num_classes=4).permute(0, 3, 1, 2).float()
        video = video.squeeze(2).permute(1, 0, 2, 3).contiguous()
        real = torch.cat([one_hot_target, video], dim=1)

        loss_real, output_real = self.get_discriminator_loss(real, label=1)
        loss_real.backward()

        loss_fake, output_fake = self.get_fake_loss(out, label=0, detach=True)
        loss_fake.backward()
        
        discriminator_loss = loss_real + loss_fake

        self.writer.add_scalar('Discriminator/Real', output_real.mean(), self.iter_nb)
        self.writer.add_scalar('Discriminator/Fake', output_fake.mean(), self.iter_nb)
        self.writer.add_scalar('Discriminator/Loss', discriminator_loss, self.iter_nb)

        self.discriminator_optimizer.step()

    def train_discriminator_2(self, target, video, out):
        self.discriminator.zero_grad()
        self.discriminator_optimizer.zero_grad()

        one_hot_target = torch.nn.functional.one_hot(target.squeeze(1).long(), num_classes=4).permute(0, 3, 1, 2).float()
        one_hot_target.repeat()
        video = video.squeeze(2).permute(1, 0, 2, 3).contiguous()
        real = torch.cat([one_hot_target, video], dim=1)

        loss_real, output_real = self.get_discriminator_loss(real, label=1)
        loss_real.backward()

        loss_fake, output_fake = self.get_fake_loss(out, label=0, detach=True)
        loss_fake.backward()
        
        discriminator_loss = loss_real + loss_fake

        self.writer.add_scalar('Discriminator/Real', output_real.mean(), self.iter_nb)
        self.writer.add_scalar('Discriminator/Fake', output_fake.mean(), self.iter_nb)
        self.writer.add_scalar('Discriminator/Loss', discriminator_loss, self.iter_nb)

        self.discriminator_optimizer.step()
