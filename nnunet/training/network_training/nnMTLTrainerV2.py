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

from nnunet.analysis import flop_count_operators
from tqdm import tqdm
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
from nnunet.training.network_training.data_augmentation import Augmenter


import psutil
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

import cv2 as cv
import sys
from matplotlib import cm
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
from nnunet.training.dataloading.dataset_loading import DataLoader2D, DataLoader2DMiddleUnlabeled, unpack_dataset
from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer
from nnunet.utilities.nd_softmax import softmax_helper
from sklearn.model_selection import KFold
from torch import nn
from torch.cuda.amp import autocast
from nnunet.training.learning_rate.poly_lr import poly_lr
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.lib.training_utils import build_2d_model, build_confidence_network, read_config, build_discriminator, build_discriminator
from nnunet.lib.loss import DirectionalFieldLoss, MaximizeDistanceLoss, AverageDistanceLoss
from pathlib import Path
from monai.losses import DiceFocalLoss, DiceLoss
from torch.utils.tensorboard import SummaryWriter
from nnunet.training.loss_functions.dice_loss import DC_and_CE_loss, DC_and_focal_loss, DC_and_topk_loss, DC_and_CE_loss_Weighted, SoftDiceLoss
from nnunet.training.loss_functions.crossentropy import RobustCrossEntropyLoss, WeightedRobustCrossEntropyLoss
from nnunet.training.dataloading.dataset_loading import DataLoader2DMiddle, DataLoader2DUnlabeled, DataLoader2DBinary
from nnunet.training.dataloading.dataset_loading import load_dataset, load_unlabeled_dataset
from nnunet.network_architecture.MTL_model import ModelWrap
from nnunet.lib.utils import RFR, ConvBlocks, Resblock, LayerNorm, RFR_1d, Resblock1D, ConvBlocks1D
from nnunet.lib.loss import SeparabilityLoss, ContrastiveLoss
from nnunet.training.data_augmentation.cutmix import cutmix, batched_rand_bbox
import shutil
from nnunet.visualization.visualization import Visualizer
from nnunet.training.network_training.processor import Processor

class nnMTLTrainerV2(nnUNetTrainer):
    """
    Info for Fabian: same as internal nnUNetTrainerV2_2
    """

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False, middle=False, video=False, inference=False, binary=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.middle = middle

        if self.middle:
            self.config = read_config(os.path.join(Path.cwd(), 'adversarial_acdc_middle.yaml'), middle, False)
        else:
            self.config = read_config(os.path.join(Path.cwd(), 'adversarial_acdc.yaml'), middle, False)
        
        self.image_size = self.config['patch_size'][0]
        self.window_size = 7 if self.image_size == 224 else 9 if self.image_size == 288 else None

        self.max_num_epochs = self.config['max_num_epochs']
        self.log_images = self.config['log_images']
        self.initial_lr = self.config['initial_lr']
        self.weight_decay = self.config['weight_decay']
        self.binary = binary
        self.deep_supervision_scales = None
        self.ds_loss_weights = None
        self.use_progress_bar=True
        self.middle_classification = self.config['middle_classification']
        self.directional_field = self.config['directional_field']
        self.reconstruction = self.config['reconstruction']
        self.separability = self.config['separability']
        self.unlabeled = self.config['unlabeled']
        if self.unlabeled:
            self.unlabeled_loss_weight = self.config['unlabeled_loss_weight']
        if self.middle:
            self.past_percent = {}
            self.cumulative_ema = 0
            self.mix_residual = self.config['mix_residual']
            self.registered_seg = self.config['registered_seg']
            self.alpha_ema = self.config['alpha_ema']
            self.one_vs_all = self.config['one_vs_all']
            self.t1 = self.config['t1'] * (self.max_num_epochs * self.num_batches_per_epoch)
            self.t2 = self.config['t2'] * (self.max_num_epochs * self.num_batches_per_epoch)
            self.max_unlabeled_weight = self.config['max_unlabeled_weight']
            self.middle_unlabeled = self.config['middle_unlabeled']
            self.v1 = self.config['v1']
        else:
            self.registered_x = False
            self.registered_seg = False
            self.one_vs_all = True
            self.interpolate = False
            self.maximize_distance = False
            self.middle_unlabeled = False

        self.deep_supervision = self.config['deep_supervision']
        self.classification = self.config['classification']
        self.adversarial_loss = self.config['adversarial_loss']

        self.iter_nb = 0

        self.val_loss = []
        self.train_loss = []
        
        loss_weights = torch.tensor(self.config['224_loss_weights'], device=self.config['device'])

        timestr = strftime("%Y-%m-%d_%HH%M")
        self.log_dir = os.path.join(copy(self.output_folder), timestr)
        self.writer = SummaryWriter(log_dir=self.log_dir)

        if self.log_images:
            self.vis = Visualizer(unlabeled=self.unlabeled,
                                    adversarial_loss=self.adversarial_loss,
                                    middle_unlabeled=self.middle_unlabeled,
                                    middle=self.middle,
                                    registered_seg=self.registered_seg,
                                    writer=self.writer)

        if inference:
            self.output_folder = output_folder
        else:
            self.output_folder = self.log_dir

        self.setup_loss_functions(loss_weights)
        
        self.pin_memory = True
        self.similarity_downscale = self.config['similarity_down_scale']

        if self.classification:
            self.classification_accuracy_list = []

        self.reconstruction_ssim_list = []
        self.df_ssim_list = []
        self.sim_l2_list = []
        self.corr_list = []
        self.aff_l2_list = []
        self.motion_ssim_list = []
        #self.sim_l2_md_list = []
        self.table = self.initialize_table()

        self.loss_data = self.setup_loss_data()

    def get_slice_percent(self):
        sorted_past_percent = dict(sorted(self.past_percent.items()))
        new_d = {}
        smooth = savgol_filter(list(sorted_past_percent.values()), len(sorted_past_percent), 3) # window size 50, polynomial order 3
        for k, v in zip(sorted_past_percent.keys(), smooth):
            new_d[k] = v
        self.percent = min(new_d, key= lambda x: new_d[x])
        self.writer.add_scalar('Epoch/Slice percent', self.percent, self.epoch)
        plt.close()
        plt.scatter(list(sorted_past_percent.keys()), list(sorted_past_percent.values()))
        plt.plot(list(new_d.keys()), list(new_d.values()), color='red')
        self.writer.add_figure('Epoch/Slice data', plt.gcf(), self.epoch)
        self.print_to_log_file(f"Slice percent: {self.percent}", also_print_to_console=True)

    def get_slice_percent_ova(self):
        self.percent = {}
        for i, (k, v) in enumerate(self.past_percent.items()):
            sorted_past_percent = dict(sorted(self.past_percent[k].items()))
            new_d = {}
            polyorder = 3 if len(sorted_past_percent) > 3 else len(sorted_past_percent) - 1
            smooth = savgol_filter(list(sorted_past_percent.values()), len(sorted_past_percent), polyorder) # window size 50, polynomial order 3
            for k2, v2 in zip(sorted_past_percent.keys(), smooth):
                new_d[k2] = v2
            percent = min(new_d, key= lambda x: new_d[x])
            self.percent[k] = percent

            if i == 0:
                self.writer.add_scalar(f'Epoch/Slice {i} percent', percent, self.epoch)
                plt.close()
                plt.scatter(list(sorted_past_percent.keys()), list(sorted_past_percent.values()))
                plt.plot(list(sorted_past_percent.keys()), list(new_d.values()), color='red')
                self.writer.add_figure('Epoch/Slice data', plt.gcf(), self.epoch)

        plt.close()
        sorted_percent = dict(sorted(self.percent.items()))
        smooth = savgol_filter(list(sorted_percent.values()), len(sorted_percent), 3)
        new_d = {}
        for k3, v3 in zip(sorted_percent.keys(), smooth):
            new_d[k3] = v3
        plt.scatter(list(sorted_percent.keys()), list(sorted_percent.values()))
        plt.plot(list(new_d.keys()), list(new_d.values()), color='red')
        self.writer.add_figure('Epoch/Slice percent', plt.gcf(), self.epoch)
        self.percent = new_d

    def setup_loss_data(self):
        loss_data = {'segmentation': [1.0, float('nan')]}

        if self.config['reconstruction']:
            loss_data['reconstruction'] = [self.config['reconstruction_loss_weight'], float('nan')]
            if self.vae:
                loss_data['vae'] = [self.config['vae_loss_weight'], float('nan')]
            elif self.vq_vae:
                loss_data['vq_vae'] = [self.config['vae_loss_weight'], float('nan')]
            if self.similarity:
                similarity_initial_weight = self.config['similarity_weight'] if not self.config['progressive_similarity_growing'] else 0
                loss_data['similarity'] = [similarity_initial_weight, float('nan')]

        if self.middle:
            #loss_data['heatmap'] = [self.config['heatmap_loss_weight'], float('nan')]
            #loss_data['contrastive'] = [self.config['contrastive_loss_weight'], float('nan')]
            if not self.middle_unlabeled:
                loss_data['similarity'] = [self.config['similarity_loss_weight'], float('nan')]
            if self.registered_seg:
                loss_data['motion_estimation_seg'] = [self.config['seg_motion_estimation_loss_weight'], float('nan')]
        
        if self.config['separability']:
            loss_data['separability'] = [self.config['separability_loss_weight'], float('nan')]

        if self.adversarial_loss:
            loss_data['adversarial'] = [self.config['adversarial_loss_weight'], float('nan')]
            if self.unlabeled:
                loss_data['confidence'] = [1.0, float('nan')]
            #else:
            #    loss_data['seg_adversarial'] = [self.config['adversarial_weight'], float('nan')]
            #    loss_data['rec_adversarial'] = [self.config['adversarial_weight'], float('nan')]
        
        if self.classification:
            loss_data['classification'] = [self.config['classification_weight'], float('nan')]
        
        if self.middle_classification:
            loss_data['middle_classification'] = [self.config['middle_classification_weight'], float('nan')]
        
        if self.directional_field:
            loss_data['directional_field'] = [self.config['directional_field_weight'], float('nan')]
        return loss_data
    
    def setup_loss_functions(self, loss_weights):
        if self.config['reconstruction']:
            self.mse_loss = nn.MSELoss()
            if self.similarity:
                self.similarity_loss = nn.L1Loss()
        if self.adversarial_loss:
            self.adv_loss = nn.BCELoss()
            self.discriminator_lr = self.config['discriminator_lr']
            self.discriminator_decay = self.config['discriminator_decay']
            self.r1_penalty_iteration = self.config['r1_penalty_iteration']

        if self.middle:

            if self.one_vs_all:
                self.mse_loss = nn.MSELoss()
                if self.registered_seg:
                    self.motion_estimation_loss_seg = DC_and_CE_loss({'batch_dice': True, 'smooth': 1e-5, 'do_bg': False}, {})
                #self.contrastive_loss = ContrastiveLoss(temp=self.temp)
            #else:
            #    self.motion_estimation_loss_x = nn.MSELoss(reduction='none')
            #    self.middle_seg_loss = DC_and_CE_loss({'batch_dice': self.batch_dice, 'smooth': 1e-5, 'do_bg': False}, {'reduction': 'none'})
            #    self.motion_estimation_loss_seg = DC_and_CE_loss({'batch_dice': self.batch_dice, 'smooth': 1e-5, 'do_bg': False}, {'reduction': 'none'})
            #    self.relation_loss = nn.MSELoss(reduction='none')
            #    if self.interpolate:
            #        self.inter_seg_loss = DC_and_CE_loss({'batch_dice': self.batch_dice, 'smooth': 1e-5, 'do_bg': False}, {'reduction': 'none'})
            
        
        if self.config['separability']:
            self.separability_loss = SeparabilityLoss()
        
        if self.classification:
            self.classification_loss = nn.CrossEntropyLoss()
        
        if self.middle_classification:
            self.middle_classification_loss = nn.BCELoss()
        
        if self.directional_field:
            self.directional_field_loss = DirectionalFieldLoss(weights=loss_weights, writer=self.writer)

        if self.unlabeled and self.adversarial_loss:
            self.confidence_loss = WeightedRobustCrossEntropyLoss(reduction='none')
            self.segmentation_loss = SoftDiceLoss(apply_nonlin=softmax_helper, batch_dice=self.batch_dice, do_bg=False, smooth=1e-5)
        else:
            if self.config['loss'] == 'focal_and_dice2':
                self.segmentation_loss = DiceFocalLoss(include_background=False, focal_weight=loss_weights[1:] if not self.config['binary'] else None, softmax=True, to_onehot_y=True)
            if self.config['loss'] == 'focal_and_dice':
                self.segmentation_loss = DC_and_focal_loss({'batch_dice': self.batch_dice, 'smooth': 1e-5, 'do_bg': False}, {'apply_nonlin': nn.Softmax(dim=1), 'alpha':0.5, 'gamma':2, 'smooth':1e-5})
            elif self.config['loss'] == 'topk_and_dice':
                self.segmentation_loss = DC_and_topk_loss({'batch_dice': self.batch_dice, 'smooth': 1e-5, 'do_bg': False}, {'k': 10}, ce_weight=0.5, dc_weight=0.5)
            elif self.config['loss'] == 'ce_and_dice':
                if self.middle:
                    self.segmentation_loss = DC_and_CE_loss({'batch_dice': False, 'smooth': 1e-5, 'do_bg': False}, {'reduction': 'none'})
                else:
                    self.segmentation_loss = DC_and_CE_loss({'batch_dice': True, 'smooth': 1e-5, 'do_bg': False}, {})
            elif self.config['loss'] == 'ce':
                self.segmentation_loss = RobustCrossEntropyLoss(weight=loss_weights)


    def initialize_table(self):

        table = {}
        table['seg'] = {'foreground_dc': [], 'tp': [], 'fp': [], 'fn': []}

        if self.middle and self.registered_seg:
            table['forward_motion'] = {'foreground_dc': [], 'tp': [], 'fp': [], 'fn': []}
            table['backward_motion'] = {'foreground_dc': [], 'tp': [], 'fp': [], 'fn': []}
    
        return table
    
    #def initialize_image_data(self):
    #    log_images_nb = 8
#
    #    eval_images_rec = {'input': None,
    #                        'reconstruction': None}
#
    #    eval_images_df = {'input': None,
    #                        'gt_df': None,
    #                        'pred_df': None}
#
    #    eval_images_sim = {'input': None,
    #                        'decoder_sm': None,
    #                        'reconstruction_sm': None}
    #    
    #    eval_images_seg = {'input': None,
    #                        'pred': None,
    #                        'gt': None}
    #    
    #    top = np.array([copy(eval_images_rec) for x in range(1)], dtype=object)
    #    bottom = np.array([copy(-1) for x in range(1)], dtype=float)
    #    rec_data = np.stack([top, bottom], axis=0)
#
    #    top = np.array([copy(eval_images_sim) for x in range(log_images_nb)], dtype=object)
    #    bottom = np.array([copy(float('inf')) for x in range(log_images_nb)], dtype=float)
    #    sim_data = np.stack([top, bottom], axis=0)
#
    #    top = np.array([copy(eval_images_df) for x in range(log_images_nb)], dtype=object)
    #    bottom = np.array([copy(-1) for x in range(log_images_nb)], dtype=float)
    #    df_data = np.stack([top, bottom], axis=0)
#
    #    top = np.array([copy(eval_images_seg) for x in range(log_images_nb)], dtype=object)
    #    bottom = np.array([copy(1) for x in range(log_images_nb)], dtype=float)
    #    seg_data = np.stack([top, bottom], axis=0)
    #    
    #    eval_images = {'rec': rec_data,
    #                    'sim': sim_data,
    #                    'df': df_data,
    #                    'seg': seg_data}
    #    
    #    return eval_images

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
            if self.directional_field:
                seg_weights = np.array([1 / (2 ** i) for i in range(net_numpool + 1)])
                seg_weights = seg_weights / seg_weights.sum()
            else:
                seg_weights = weights

            if self.deep_supervision:
                self.segmentation_loss = MultipleOutputLoss2(self.segmentation_loss, seg_weights)
                if self.middle:
                    if self.interpolate:
                        self.inter_seg_loss = MultipleOutputLoss2(self.inter_seg_loss, seg_weights)
                if self.unlabeled and self.adversarial_loss:
                    self.confidence_loss = MultipleOutputLoss2(self.confidence_loss, seg_weights)
            if self.reconstruction:
                if self.simple_decoder:
                    rec_weights = [1]
                    self.reconstruction_loss = MultipleOutputLoss2(self.mse_loss, rec_weights)
                else:
                    self.reconstruction_loss = MultipleOutputLoss2(self.mse_loss, weights)

            ################# END ###################

            self.folder_with_preprocessed_data = join(self.dataset_directory, self.plans['data_identifier'] +
                                                      "_stage%d" % self.stage)
            if training:
                self.dl_tr, self.dl_val, self.dl_un_tr, self.dl_un_val = self.get_basic_generators()
                if self.unpack_data:
                    print("unpacking dataset")
                    unpack_dataset(self.folder_with_preprocessed_data)
                    print("done")
                else:
                    print(
                        "INFO: Not unpacking data! Training may be slow due to that. Pray you are not using 2d or you "
                        "will wait all winter for your model to finish!")

                if not self.binary:
                    self.tr_gen, self.val_gen, self.tr_un_gen, self.val_un_gen = get_moreDA_augmentation_mtl(
                    self.dl_tr, self.dl_val, self.dl_un_tr, self.dl_un_val,
                    self.data_aug_params[
                        'patch_size_for_spatialtransform'],
                    params=self.data_aug_params,
                    deep_supervision_scales=self.deep_supervision_scales,
                    pin_memory=self.pin_memory,
                    use_nondetMultiThreadedAugmenter=False
                    )

                self.print_to_log_file("TRAINING KEYS:\n %s" % (str(self.dataset_tr.keys())),
                                       also_print_to_console=False)
                self.print_to_log_file("VALIDATION KEYS:\n %s" % (str(self.dataset_val.keys())),
                                       also_print_to_console=False)
                if self.unlabeled or self.middle_unlabeled:
                    self.print_to_log_file("UNLABELED TRAINING KEYS:\n %s" % (str(self.dataset_un_tr.keys())),
                                       also_print_to_console=False)
                    self.print_to_log_file("UNLABELED VALIDATION KEYS:\n %s" % (str(self.dataset_un_val.keys())),
                                       also_print_to_console=False)
            else:
                pass

            self.initialize_network()
            self.initialize_optimizer_and_scheduler()

            assert isinstance(self.network, (SegmentationNetwork, nn.DataParallel))
        else:
            self.print_to_log_file('self.was_initialized is True, not running self.initialize again')
        self.was_initialized = True
    
    #def print_to_log_file(self, *args, also_print_to_console=True, add_timestamp=True):
#
    #    timestamp = time()
    #    dt_object = datetime.fromtimestamp(timestamp)
#
    #    if add_timestamp:
    #        args = ("%s:" % dt_object, *args)
#
    #    if self.log_file is None:
    #        maybe_mkdir_p(self.log_dir)
    #        timestamp = datetime.now()
    #        self.log_file = join(self.log_dir, "training_log_%d_%d_%d_%02.0d_%02.0d_%02.0d.txt" %
    #                             (timestamp.year, timestamp.month, timestamp.day, timestamp.hour, timestamp.minute,
    #                              timestamp.second))
    #        with open(self.log_file, 'w') as f:
    #            f.write("Starting... \n")
    #    successful = False
    #    max_attempts = 5
    #    ctr = 0
    #    while not successful and ctr < max_attempts:
    #        try:
    #            with open(self.log_file, 'a+', encoding='utf-8') as f:
    #                for a in args:
    #                    f.write(str(a))
    #                    f.write(" ")
    #                f.write("\n")
    #            successful = True
    #        except IOError:
    #            print("%s: failed to log: " % datetime.fromtimestamp(timestamp), sys.exc_info())
    #            sleep(0.5)
    #            ctr += 1
    #    if also_print_to_console:
    #        print(*args)
    
    def count_parameters(self, config, models):
        params_sum = 0
        self.print_to_log_file(yaml.safe_dump(config, default_flow_style=None, sort_keys=False), also_print_to_console=False)
        for k, v in models.items():
            nb_params =  sum(p.numel() for p in v[0].parameters() if p.requires_grad)
            self.print_to_log_file(f"{k} has {nb_params:,} parameters")
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

        num_classes = 2 if self.binary else 4

        wanted_norm = self.config['norm']
        if wanted_norm == 'batchnorm':
            norm_2d = nn.BatchNorm2d
            norm_1d = nn.InstanceNorm1d
        elif wanted_norm == 'instancenorm':
            norm_2d = nn.InstanceNorm2d
            norm_1d = nn.InstanceNorm1d
        
        conv_layer, conv_layer_1d = self.get_conv_layer(self.config)
        if self.unlabeled:
            if self.adversarial_loss:
                self.discriminator = build_confidence_network(self.config, alpha=self.config['alpha_discriminator'], image_size=self.image_size, window_size=self.window_size)
                if torch.cuda.is_available():
                    self.discriminator.cuda()
                discriminator_input = torch.randn(self.config['batch_size'], 4, self.image_size, self.image_size)
                models['discriminator'] = (self.discriminator, discriminator_input)

            net1 = build_2d_model(self.config, conv_layer=conv_layer, norm=norm_2d, log_function=self.print_to_log_file, image_size=self.image_size, window_size=self.window_size)
            net2 = build_2d_model(self.config, conv_layer=conv_layer, norm=norm_2d, log_function=self.print_to_log_file, image_size=self.image_size, window_size=self.window_size)
            self.network = ModelWrap(net1, net2, do_ds=self.config['deep_supervision'])

            model_input_data = torch.randn(self.config['batch_size'], 1, self.image_size, self.image_size)
            model_input_data1 = [model_input_data, 1]
            models['model'] = (self.network, model_input_data1)
            self.count_parameters(self.config, models)
        else:
            if self.adversarial_loss:
                self.discriminator = build_discriminator(self.config, conv_layer=conv_layer, alpha=self.config['alpha_discriminator'], in_channel=4, image_size=self.image_size)
                if torch.cuda.is_available():
                    self.discriminator.cuda()
                discriminator_input = torch.randn(self.config['batch_size'], 4, self.image_size, self.image_size)
                models['discriminator'] = (self.discriminator, discriminator_input)

                #self.rec_discriminator = build_discriminator(self.config, alpha=0.125, use_swin_discriminator=False, in_channel=1)
                #if torch.cuda.is_available():
                #    self.rec_discriminator.cuda()
                #discriminator_input = torch.randn(self.config['batch_size'], 1, self.image_size, self.image_size)
                #models['rec_discriminator'] = (self.rec_discriminator, discriminator_input)

            in_shape = torch.randn(self.config['batch_size'], 1, self.image_size, self.image_size)
            self.network = build_2d_model(self.config, conv_layer=conv_layer, norm=getattr(torch.nn, self.config['norm']), log_function=self.print_to_log_file, image_size=self.image_size, window_size=self.window_size, middle=self.middle, num_classes=num_classes)
            if self.middle:
                model_input_data = {'l1': in_shape, 'l2': in_shape}
                if self.middle_unlabeled:
                    model_input_data['u1'] = in_shape
                    if self.v1:
                        model_input_data['u2'] = in_shape
                model_input_data = [model_input_data]
            else:
                model_input_data = in_shape
            models['model'] = (self.network, model_input_data)

            self.count_parameters(self.config, models)

        #nb_inputs = 2 if self.middle else 1
        #model_input_size = [(self.config['batch_size'], 1, self.image_size, self.image_size)] * nb_inputs

        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper

    def get_optimizer_scheduler(self, net, lr, decay):
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

        if self.adversarial_loss:
            self.discriminator_optimizer, self.discriminator_scheduler = self.get_optimizer_scheduler(net=self.discriminator, lr=self.discriminator_lr, decay=self.discriminator_decay)
            #if self.unlabeled:
            #else:
            #    self.seg_discriminator_optimizer, self.seg_discriminator_scheduler = self.get_optimizer_scheduler(net=self.seg_discriminator, lr=self.discriminator_lr, decay=self.discriminator_decay)
            #    self.rec_discriminator_optimizer, self.rec_discriminator_scheduler = self.get_optimizer_scheduler(net=self.rec_discriminator, lr=self.discriminator_lr, decay=self.discriminator_decay)

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


    def start_online_evaluation(self, pred, reconstructed, x, target, gt_df, classification_pred, classification_gt, confidence):

        with torch.no_grad():
            num_classes = pred.shape[1]
            output_softmax = softmax_helper(pred)
            output_seg = output_softmax.argmax(1)
            target = target[:, 0]
            seg_dice = self.compute_dice(target, num_classes, output_seg, key='seg')

            if self.log_images:
                if self.classification:
                    acc = accuracy_score(y_true=classification_gt.cpu(), y_pred=classification_pred.cpu())
                    self.classification_accuracy_list.append(acc)

                for t in range(len(target)):
                    current_x = x[t, 0]

                    #if self.affinity:
                    #    current_aff = aff[t]
                    #    current_seg_aff = seg_aff[t]
                    #    current_aff = self.get_similarity_ready(current_aff)
                    #    current_seg_aff = self.get_similarity_ready(current_seg_aff)
                    #    self.vis.set_up_image_aff(seg_dice=seg_dice[t].mean(), aff=current_aff, seg_aff=current_seg_aff, x=current_x)
                    #    aff_l2 = torch.linalg.norm(current_aff.type(torch.float32) - current_seg_aff.type(torch.float32), ord=2)
                    #    self.aff_l2_list.append(aff_l2)
                    #if self.reconstruction:
                    #    current_reconstructed = reconstructed[t, 0]
                    #    rec_ssim = ssim(current_x.type(torch.float32)[None, None, :, :], current_reconstructed.type(torch.float32)[None, None, :, :])
                    #    self.reconstruction_ssim_list.append(rec_ssim)
                    #    self.vis.set_up_image_rec(rec_ssim=rec_ssim, reconstructed=current_reconstructed, x=current_x)
#
                    #if self.directional_field:
                    #    current_pred_df = pred_df[t]
                    #    current_gt_df = gt_df[t]
                    #    df_ssim = ssim(current_gt_df.type(torch.float32)[None, :, :, :], current_pred_df.type(torch.float32)[None, :, :, :])
                    #    self.df_ssim_list.append(df_ssim)
                    #    current_gt_df = current_gt_df[0]
                    #    current_pred_df = current_pred_df[0]
                    #    self.vis.set_up_image_df(df_ssim=df_ssim, gt_df=current_gt_df, pred_df=current_pred_df, x=current_x)

                    current_pred = torch.argmax(output_softmax[t], dim=0)
                    if self.adversarial_loss and self.unlabeled:
                        self.vis.set_up_image_confidence(seg_dice=seg_dice[t].mean(), confidence=confidence[t, 0], pred=current_pred, x=current_x)

                    current_target = target[t]

                    self.vis.set_up_image_seg_best(seg_dice=seg_dice[t].mean(), gt=current_target, pred=current_pred, x=current_x)
                    self.vis.set_up_image_seg_worst(seg_dice=seg_dice[t].mean(), gt=current_target, pred=current_pred, x=current_x)
                    
                    #with autocast():

                #tp_hard = tp_hard.sum(0, keepdim=False).detach().cpu().numpy()
                #fp_hard = fp_hard.sum(0, keepdim=False).detach().cpu().numpy()
                #fn_hard = fn_hard.sum(0, keepdim=False).detach().cpu().numpy()
    #
                #self.online_eval_foreground_dc.append(list((2 * tp_hard) / (2 * tp_hard + fp_hard + fn_hard + 1e-8)))
                #self.online_eval_tp.append(list(tp_hard))
                #self.online_eval_fp.append(list(fp_hard))
                #self.online_eval_fn.append(list(fn_hard))

    
    def start_online_evaluation_middle(self, pred, x, target_list, forward_registered_seg, backward_registered_seg, forward_motion, backward_motion, pseudo_label, cross_sim, middle_pred):

        with torch.no_grad():
            num_classes = pred.shape[1]
            output_softmax = softmax_helper(pred)
            output_softmax_middle = softmax_helper(middle_pred)
            output_seg = output_softmax.argmax(1)
            output_middle_seg = output_softmax_middle.argmax(1)
            seg_dice = self.compute_dice(target_list[0][:, 0], num_classes, output_seg, key='seg')

            if self.registered_seg:
                forward_registered_output_softmax = softmax_helper(forward_registered_seg)
                backward_registered_output_softmax = softmax_helper(backward_registered_seg)
                output_registered_seg_forward = forward_registered_output_softmax.argmax(1)
                output_registered_seg_backward = backward_registered_output_softmax.argmax(1)
                _ = self.compute_dice(target_list[1][:, 0], num_classes, output_registered_seg_forward, key='forward_motion')
                _ = self.compute_dice(target_list[0][:, 0], num_classes, output_registered_seg_backward, key='backward_motion')

            if self.log_images:

                #w1 = self.vis.get_weights_ready(w1, self.similarity_downscale)
                #w2 = self.vis.get_weights_ready(w2, self.similarity_downscale)
                for t in range(len(target_list[0][:, 0])):
                    current_x = x[0][t, 0]
                    current_middle = x[1][t, 0]

                    #current_cross_sim = cross_sim[t]
                    #self.corr_list.append(current_cross_sim.mean())
                    #self.vis.set_up_image_corr(seg_dice=seg_dice[t].mean(), corr=current_cross_sim)
                    #self.vis.set_up_image_pseudo_label(seg_dice=seg_dice[t].mean(), pred=pseudo_label[t, 0], unlabeled_input=current_middle)

                    if self.registered_seg:
                        current_forward_motion = forward_motion[t, 0]
                        current_backward_motion = backward_motion[t, 0]
                        current_forward_registered_seg = output_registered_seg_forward[t]
                        current_backward_registered_seg = output_registered_seg_backward[t]
                        self.vis.set_up_image_motion(seg_dice=seg_dice[t].mean(), moving=output_seg[t], registered=current_forward_registered_seg, fixed=target_list[1][t, 0], motion=current_forward_motion, name='forward_motion')
                        self.vis.set_up_image_motion(seg_dice=seg_dice[t].mean(), moving=output_middle_seg[t], registered=current_backward_registered_seg, fixed=target_list[0][t, 0], motion=current_backward_motion, name='backward_motion')

                    current_pred = torch.argmax(output_softmax[t], dim=0)
                    current_target = target_list[0][t, 0]
                    self.vis.set_up_image_seg(seg_dice=seg_dice[t].mean(), gt=current_target, pred=current_pred, x=current_x)
    
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

        if self.binary:
            class_dice = {'Heart': global_dc_per_class_seg[0]}
        else:
            class_dice = {'RV': global_dc_per_class_seg[0], 'MYO': global_dc_per_class_seg[1], 'LV': global_dc_per_class_seg[2]}
        overfit_data = {'Train': torch.tensor(self.train_loss).mean().item(), 'Val': torch.tensor(self.val_loss).mean().item()}
        self.writer.add_scalars('Epoch/Train_vs_val_loss', overfit_data, self.epoch)
        self.writer.add_scalars('Epoch/Class dice', class_dice, self.epoch)            

        if self.log_images:
            cmap, norm = self.vis.get_custom_colormap()
        if self.classification:
            self.print_to_log_file("Classification accuracy:", torch.tensor(self.classification_accuracy_list).mean().item())
            self.writer.add_scalar('Epoch/Classification accuracy', torch.tensor(self.classification_accuracy_list).mean().item(), self.epoch)

        if self.middle:
            #self.vis.log_pseudo_label_images(colormap_seg=cmap, norm=norm, epoch=self.epoch)
            if self.registered_seg:
                global_dc_per_class_forward_motion = self.get_dc_per_class('forward_motion')
                global_dc_per_class_backward_motion = self.get_dc_per_class('backward_motion')
                motion_dice = [(x1 + x2) / 2 for (x1, x2) in zip(global_dc_per_class_forward_motion, global_dc_per_class_backward_motion)]
                data_dict = {'forward': torch.tensor(global_dc_per_class_forward_motion).mean().item(),
                            'backward': torch.tensor(global_dc_per_class_backward_motion).mean().item()}
                self.writer.add_scalars('Epoch/Motion estimation dice', data_dict, self.epoch)
                self.print_to_log_file("Registered seg dice:", [np.round(i, 4) for i in motion_dice])
                if self.log_images:
                    self.vis.log_motion_images(colormap_seg=cmap, colormap=cm.plasma, norm=norm, epoch=self.epoch, name='forward_motion')
                    self.vis.log_motion_images(colormap_seg=cmap, colormap=cm.plasma, norm=norm, epoch=self.epoch, name='backward_motion')

        #if self.reconstruction:
        #    self.print_to_log_file("Average reconstruction ssim:", torch.tensor(self.reconstruction_ssim_list).mean().item())
        #    self.writer.add_scalar('Epoch/Reconstruction ssim', torch.tensor(self.reconstruction_ssim_list).mean().item(), self.epoch)
        #    if self.log_images:
        #        self.vis.log_rec_images(epoch=self.epoch)
#
        #if self.directional_field:
        #    self.print_to_log_file("Average df ssim:", torch.tensor(self.df_ssim_list).mean().item())
        #    self.writer.add_scalar('Epoch/Directional field ssim', torch.tensor(self.df_ssim_list).mean().item(), self.epoch)
        #    if self.log_images:
        #        self.vis.log_df_images(colormap=cm.viridis, epoch=self.epoch)
#
        #if self.affinity:
        #    self.print_to_log_file("Average affinity L2 distance:", torch.tensor(self.aff_l2_list).mean().item())
        #    self.writer.add_scalar('Epoch/Affinity L2 distance', torch.tensor(self.aff_l2_list).mean().item(), self.epoch)
        #    if self.log_images:
        #        self.vis.log_aff_images(colormap=cm.plasma, epoch=self.epoch)

        if self.log_images:
            self.vis.log_best_seg_images(colormap=cmap, norm=norm, epoch=self.epoch)
            self.vis.log_worst_seg_images(colormap=cmap, norm=norm, epoch=self.epoch)
            if self.unlabeled and self.adversarial_loss:
                self.vis.log_confidence_images(colormap=cm.plasma, colormap_seg=cmap, norm=norm, epoch=self.epoch)

        #self.online_eval_foreground_dc = []
        #self.online_eval_tp = []
        #self.online_eval_fp = []
        #self.online_eval_fn = []
        if self.log_images:
            self.vis.reset()
        self.table = self.initialize_table()
        self.reconstruction_ssim_list = []
        self.df_ssim_list = []
        self.sim_l2_list = []
        self.corr_list = []
        self.val_loss = []
        self.motion_ssim_list = []
        self.train_loss = []
        #self.sim_l2_md_list = []

        if self.unlabeled:
            self.pseudo_online_eval_foreground_dc = []
            self.pseudo_online_eval_tp = []
            self.pseudo_online_eval_fp = []
            self.pseudo_online_eval_fn = []


    def run_online_evaluation(self, data, target, output, gt_df, classification_gt, confidence):
        """
        due to deep supervision the return value and the reference are now lists of tensors. We only need the full
        resolution output because this is what we are interested in in the end. The others are ignored
        :param output:
        :param target:
        :return:
        """

        pred = self.select_deep_supervision(output['pred'])

        if self.reconstruction:
            reconstructed = self.select_deep_supervision(output['reconstructed'])
        else:
            reconstructed = None
        
        if self.classification:
            classification_pred = output['classification']
            classification_pred = torch.nn.functional.softmax(classification_pred, dim=1)
            classification_pred = torch.argmax(classification_pred, dim=1)
        else:
            classification_pred = None

        x = self.select_deep_supervision(data)
        target = self.select_deep_supervision(target)
        return self.start_online_evaluation(pred=pred,
                                            reconstructed=reconstructed, 
                                            x=x, 
                                            target=target, 
                                            gt_df=gt_df,
                                            classification_pred=classification_pred,
                                            classification_gt=classification_gt,
                                            confidence=confidence)
        #return super().run_online_evaluation(output, target)


    def run_online_evaluation_middle(self, data, target, output):
        """
        due to deep supervision the return value and the reference are now lists of tensors. We only need the full
        resolution output because this is what we are interested in in the end. The others are ignored
        :param output:
        :param target:
        :return:
        """
        pseudo_label = None
        registered_seg = None

        x = data
        target = target
        pred = self.select_deep_supervision(output['pred_l_1'])
        forward_motion = output['forward_motion']
        backward_motion = output['backward_motion']
        middle_pred = self.select_deep_supervision(output['pred_l_2'])
        if self.registered_seg:
            forward_registered_seg = output['registered_seg_1']
            backward_registered_seg = output['registered_seg_2']
        cross_sim = output['cross_sim_l_1']

        return self.start_online_evaluation_middle(pred=pred,
                                                    x=x, 
                                                    target_list=target,
                                                    forward_motion=forward_motion,
                                                    backward_motion=backward_motion,
                                                    forward_registered_seg=forward_registered_seg,
                                                    backward_registered_seg=backward_registered_seg,
                                                    pseudo_label=pseudo_label,
                                                    cross_sim=cross_sim,
                                                    middle_pred=middle_pred)
    
    def get_throughput(self, optimal_batch_size, output_folder):
        self.network.do_ds = False
        model = self.network
        device = model.get_device()

        with torch.no_grad():
            dummy_input = torch.randn(1, 1,224,224, dtype=torch.float).to(device)
            out_flop = flop_count_operators(model, dummy_input) 
            dummy_input = torch.randn(optimal_batch_size, 1,224,224, dtype=torch.float).to(device)
            repetitions=100
            total_time = 0
            for rep in tqdm(range(repetitions)):
                starter = torch.cuda.Event(enable_timing=True)
                ender = torch.cuda.Event(enable_timing=True)
                starter.record()
                _ = model(dummy_input)
                ender.record()
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)/1000
                total_time += curr_time
        Throughput =   (repetitions*optimal_batch_size)/total_time
        gflops = 0
        for flop_key in out_flop.keys():
            gflops += out_flop[flop_key]
        with open(join(output_folder, "computational_time"), 'w') as fd:
            fd.writelines([f'Gflops: {gflops}', '\n', f'FPS: {Throughput}'])

    def validate(self, do_mirroring: bool = True, use_sliding_window: bool = True,
                 step_size: float = 0.5, save_softmax: bool = True, use_gaussian: bool = True, overwrite: bool = True,
                 validation_folder_name: str = 'validation_raw', debug: bool = False, all_in_gpu: bool = False,
                 segmentation_export_kwargs: dict = None, run_postprocessing_on_folds: bool = True, output_folder=None):
        """
        We need to wrap this because we need to enforce self.network.do_ds = False for prediction
        """ 
        ds = self.network.do_ds
        self.network.do_ds = False
        if self.middle_unlabeled and not self.v1:
            ret = super().validate_middle_unlabeled(log_function=self.print_to_log_file, do_mirroring=do_mirroring, use_sliding_window=use_sliding_window, step_size=step_size,
                               save_softmax=save_softmax, use_gaussian=use_gaussian,
                               overwrite=overwrite, validation_folder_name=validation_folder_name,
                               all_in_gpu=all_in_gpu, segmentation_export_kwargs=segmentation_export_kwargs,
                               run_postprocessing_on_folds=run_postprocessing_on_folds, debug=True)
        else:
            ret = super().validate(log_function=self.print_to_log_file, do_mirroring=do_mirroring, use_sliding_window=use_sliding_window, step_size=step_size,
                                save_softmax=save_softmax, use_gaussian=use_gaussian,
                                overwrite=overwrite, validation_folder_name=validation_folder_name,
                                all_in_gpu=all_in_gpu, segmentation_export_kwargs=segmentation_export_kwargs,
                                run_postprocessing_on_folds=run_postprocessing_on_folds,
                                output_folder=output_folder, debug=True)

        self.network.do_ds = ds
        return ret


    def predict_preprocessed_data_return_seg_and_softmax(self, data: np.ndarray, do_mirroring: bool = True,
                                                         mirror_axes: Tuple[int] = None,
                                                         use_sliding_window: bool = True, step_size: float = 0.5,
                                                         use_gaussian: bool = True, pad_border_mode: str = 'constant',
                                                         pad_kwargs: dict = None, all_in_gpu: bool = False,
                                                         verbose: bool = True, mixed_precision=True, get_flops=False) -> Tuple[np.ndarray, np.ndarray]:
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
                                                                       mixed_precision=mixed_precision,
                                                                       get_flops=get_flops)
        self.network.do_ds = ds
        return ret

    def compute_losses(self, x, output, target, do_backprop, gt_df=None, gt_classification=None):
        #loss_data = self.setup_loss_data()

        pred = output['pred']
        if self.reconstruction:
            if self.vae:
                self.loss_data['vae'][1] = output['vq_loss']
            elif self.vq_vae:
                self.loss_data['vq_vae'][1] = output['vq_loss']
            reconstructed = output['reconstructed']
            decoder_sm = output['decoder_sm']
            reconstruction_sm = output['reconstruction_sm']
            assert pred[0].shape[-2:] == target[0].shape[-2:] == reconstructed[0].shape[-2:] == x[0].shape[-2:]
            if not self.simple_decoder:
                assert len(pred) == len(target) == len(reconstructed) == len(x)
        else:
            assert pred[0].shape[-2:] == target[0].shape[-2:]
            assert len(pred) == len(target)

        if self.classification:
            classification_loss = self.classification_loss(output['classification'], gt_classification)
            self.loss_data['classification'][1] = classification_loss

        #print(x[0].dtype)
        #print(reconstructed[0].dtype)
        #print(reconstruction_sm.dtype)
        #print(decoder_sm.dtype)
        #print(output['directional_field'].dtype)

        #if self.directional_field:
        #    df_pred = output['directional_field']
        #    directional_field_loss = self.directional_field_loss(pred=df_pred, y_df=gt_df, y_seg=target[0].squeeze(1), iter_nb=self.iter_nb, do_backprop=do_backprop)
        #    self.loss_data['directional_field'][1] = directional_field_loss
        #
        #if self.affinity:
        #    self.loss_data['affinity'][1] = self.affinity_loss(output['affinity'], output['seg_affinity'])
        #if self.separability:
        #    self.loss_data['separability'][1] = self.separability_loss(output['separability'])

        
        sim_loss = 0
        rec_loss = 0
        if self.reconstruction:
            #a = (self.end_similarity - self.start_similarity) / self.total_nb_of_iterations
            #b = self.start_similarity
            #self.current_similarity_weight = a * iter_nb + b
            #self.loss_data['similarity'][0] = self.current_similarity_weight
            rec_loss = self.reconstruction_loss(reconstructed, x)
            self.loss_data['reconstruction'][1] = rec_loss

            if self.similarity:
                assert decoder_sm.shape == reconstruction_sm.shape
                sim_loss = self.similarity_loss(decoder_sm, reconstruction_sm)
                self.loss_data['similarity'][1] = sim_loss

        if self.adversarial_loss:
            if not self.unlabeled:
                if do_backprop:
                    self.discriminator.train()
                    #self.rec_discriminator.train()
                else:
                    self.discriminator.eval()
                    #self.rec_discriminator.eval()

                fake = torch.nn.functional.softmax(pred[0], dim=1)
                fake = torch.argmax(fake, dim=1).long()
                fake = torch.nn.functional.one_hot(fake, num_classes=4).permute(0, 3, 1, 2).float()
                fake = torch.cat([fake, reconstructed[0]], dim=1)
                real = torch.nn.functional.one_hot(target[0].long(), num_classes=4).permute(0, 4, 2, 3, 1).squeeze(-1).float()
                real = torch.cat([real, x[0]], dim=1)

                assert fake.shape == real.shape

                self.loss_data['adversarial'][1], discriminator_loss, output_real, output_fake = self.get_adversarial_loss(self.discriminator, 
                                                                        self.discriminator_optimizer,
                                                                        real=real, 
                                                                        fake=fake,
                                                                        do_backprop=do_backprop)

                
                #self.loss_data['rec_adversarial'][1], rec_discriminator_loss, rec_output_real, rec_output_fake = self.get_adversarial_loss(self.rec_discriminator, 
                #                                                        self.rec_discriminator_optimizer,
                #                                                        real=x[0], 
                #                                                        fake=reconstructed[0],
                #                                                        do_backprop=do_backprop)
                
                if do_backprop:
                    self.writer.add_scalar('Discriminator/Discriminator real', output_real.mean(), self.iter_nb)
                    self.writer.add_scalar('Discriminator/Discriminator fake', output_fake.mean(), self.iter_nb)
                    self.writer.add_scalar('Discriminator/Discriminator_loss', discriminator_loss, self.iter_nb)
                    #self.writer.add_scalar('Discriminator/Rec discriminator real', rec_output_real.mean(), self.iter_nb)
                    #self.writer.add_scalar('Discriminator/Rec discriminator fake', rec_output_fake.mean(), self.iter_nb)
                    #self.writer.add_scalar('Discriminator/rec_discriminator_loss', rec_discriminator_loss, self.iter_nb)
            else:
                self.loss_data['adversarial'][1] = 0
                self.loss_data['confidence'][1] = 0
        
        seg_loss = self.segmentation_loss(pred, target)
        assert seg_loss.numel() == 1
        self.loss_data['segmentation'][1] = seg_loss

    def get_unlabeled_loss_weight(self):
        if self.iter_nb < self.t1:
            return 0
        elif self.iter_nb >= self.t1 and self.iter_nb < self.t2:
            return self.max_unlabeled_weight * ((self.iter_nb - self.t1) / (self.t2 - self.t1))
        else:
            return self.max_unlabeled_weight
        
    def get_only_labeled(self, output, target_list, labeled_binary):
        output = torch.stack(output, dim=1)
        target_l = torch.stack(target_list, dim=1)
        labeled_binary = labeled_binary.permute(1, 0)

        output = torch.flatten(output, start_dim=0, end_dim=1)
        target_l = torch.flatten(target_l, start_dim=0, end_dim=1)
        labeled_binary = torch.flatten(labeled_binary, start_dim=0, end_dim=1)

        output_l = output[labeled_binary]
        target_l = target_l[labeled_binary]

        return output_l, target_l

    def compute_losses_middle(self, output, target_list):
        seg_loss_l_1 = self.segmentation_loss(output['pred_l_1'], target_list[0])
        seg_loss_l_2 = self.segmentation_loss(output['pred_l_2'], target_list[1])
        seg_loss_l = (seg_loss_l_1 + seg_loss_l_2) / 2
        self.loss_data['segmentation'][1] = seg_loss_l

        if self.registered_seg:
            motion_loss_seg_1 = self.motion_estimation_loss_seg(output['registered_seg_1'], self.select_deep_supervision(target_list[1]))
            motion_loss_seg_2 = self.motion_estimation_loss_seg(output['registered_seg_2'], self.select_deep_supervision(target_list[0]))
            self.loss_data['motion_estimation_seg'][1] = (motion_loss_seg_1 + motion_loss_seg_2) / 2

        if not self.middle_unlabeled:
            self.loss_data['similarity'][1] = (output['cross_sim_l_1'].mean() + output['cross_sim_l_2'].mean()) / 2
            #else:
            #    similarity_loss = 1 - output['cross_sim_1'].mean()
            #    self.loss_data['similarity'][1] = similarity_loss

        #if self.adversarial_loss:
        #    fake = torch.nn.functional.one_hot(output['pseudo_label'].squeeze(1).long(), num_classes=4).permute(0, 3, 1, 2).float()
        #    self.loss_data['adversarial'][1] = self.get_adversarial_loss(fake)
#
        #if self.middle_classification:
        #    gt1 = torch.full(size=(self.batch_size,), fill_value=0.0, device='cuda:0')
        #    gt2 = torch.full(size=(self.batch_size,), fill_value=1.0, device='cuda:0')
        #    mcl1 = self.middle_classification_loss(torch.sigmoid(output['classification_pred1']), gt1)
        #    mcl2 = self.middle_classification_loss(torch.sigmoid(output['classification_pred2']), gt2)
        #    self.loss_data['middle_classification'][1] = (mcl1 + mcl2) / 2


    def compute_losses_unlabeled(self, x, output, target, fake):
        loss_data = self.setup_loss_data()

        for t in target:
            assert t.shape[1] == 1

        pred = output['pred']
        if self.reconstruction:
            if self.vae:
                loss_data['vae'][1] = output['vq_loss']
            elif self.vq_vae:
                loss_data['vq_vae'][1] = output['vq_loss']
            reconstructed = output['reconstructed']
            decoder_sm = output['decoder_sm']
            reconstruction_sm = output['reconstruction_sm']
            assert pred[0].shape[-2:] == target[0].shape[-2:] == reconstructed[0].shape[-2:] == x[0].shape[-2:]
            assert len(pred) == len(target) == len(reconstructed) == len(x)
        else:
            assert pred[0].shape[-2:] == target[0].shape[-2:]
            assert len(pred) == len(target)

        #print(x[0].dtype)
        #print(reconstructed[0].dtype)
        #print(reconstruction_sm.dtype)
        #print(decoder_sm.dtype)
        #print(output['directional_field'].dtype)
        
        sim_loss = 0
        rec_loss = 0
        if self.reconstruction:

            #a = (self.end_similarity - self.start_similarity) / self.total_nb_of_iterations
            #b = self.start_similarity
            #self.current_similarity_weight = a * iter_nb + b
            #loss_data['similarity'][0] = self.current_similarity_weight
            rec_loss = self.reconstruction_loss(reconstructed, x)
            loss_data['reconstruction'][1] = rec_loss

            if self.similarity:
                assert decoder_sm.shape == reconstruction_sm.shape
                sim_loss = self.similarity_loss(decoder_sm, reconstruction_sm)
                loss_data['similarity'][1] = sim_loss

        if self.adversarial_loss:
            loss_data['adversarial'][1], confidence = self.get_adversarial_loss_unlabeled(self.discriminator, fake)
            seg_loss = self.confidence_loss(pred, target, confidence_weights=confidence)
            loss_data['confidence'][1] = seg_loss
            confidence = confidence[0].detach()
            loss_data['segmentation'][1] = 0
        else:
            seg_loss = self.segmentation_loss(pred, target)
            loss_data['segmentation'][1] = seg_loss
            confidence = None

        return loss_data

    def select_deep_supervision(self, x):
        if self.deep_supervision:
            return x[0]
        else:
            return x

    
    def run_iteration_middle(self, data_generator, do_backprop=True, run_online_evaluation=False):
        """
        gradient clipping improves training stability

        :param data_generator:
        :param do_backprop:
        :param run_online_evaluation:
        :return:
        """
        data_dict = next(data_generator)
        data_list = data_dict['data']
        target_list = data_dict['target']

        for i, data in enumerate(data_list):
            data = maybe_to_torch(data)
            if torch.cuda.is_available():
                data = to_cuda(data)
            data_list[i] = data
        
        for i, target in enumerate(target_list):
            target = maybe_to_torch(target)
            if torch.cuda.is_available():
                target = to_cuda(target)
            target_list[i] = target

        #matplotlib.use('QtAgg')
        #fig, ax = plt.subplots(len(data_list), 2)
        #for i in range(len(data_list)):
        #    ax[i, 0].imshow(data_list[i][0, 0].cpu(), cmap='gray')
        #    if i < 2:
        #        ax[i, 1].imshow(target_list[i][0, 0].cpu(), cmap='gray')
        #plt.show()

        self.optimizer.zero_grad()

        if self.adversarial_loss and do_backprop:
            self.discriminator.train()
        elif self.adversarial_loss:
            self.discriminator.eval()

        data_list = [self.select_deep_supervision(d) for d in data_list]
        network_input = {'labeled_data': [data_list[0], data_list[1]]}
        if self.middle_unlabeled:
            network_input = {'unlabeled_data': [data_list[2]]}
            if self.v1:
                network_input['unlabeled_data'].append(data_list[3])

        output = self.network(network_input)
        self.compute_losses_middle(output=output, target_list=target_list)

        #if self.middle and do_backprop and not self.middle_unlabeled:
        #    #self.cumulative_average = (self.loss_data['segmentation'][1].mean() + self.iter_nb * self.cumulative_average) / (self.iter_nb + 1)
        #    self.cumulative_ema = self.alpha_ema * self.loss_data['segmentation'][1].mean().detach().cpu() + (1 - self.alpha_ema) * self.cumulative_ema
        #    if self.one_vs_all:
        #        for i in range(len(data_dict['slice_distance'])):
        #            if data_dict['slice_distance'][i] not in self.past_percent:
        #                s_0 = 0
        #            else:
        #                s_0 = self.past_percent[data_dict['slice_distance'][i]]
        #            slice_loss = self.loss_data['segmentation'][1][i] / self.cumulative_ema
        #            self.past_percent[data_dict['slice_distance'][i]] = self.alpha_ema * slice_loss.detach().cpu() + (1 - self.alpha_ema) * s_0
        #    else:
        #        for i in range(len(data_dict['middle_slice_percent'])):
        #            if data_dict['middle_slice_percent'][i] not in self.past_percent:
        #                s_0 = 0
        #            else:
        #                s_0 = self.past_percent[data_dict['middle_slice_percent'][i]]
        #            slice_loss = self.loss_data['segmentation'][1][i]
        #            self.past_percent[data_dict['middle_slice_percent'][i]] = self.alpha_ema * slice_loss.detach().cpu() + (1 - self.alpha_ema) * s_0
        
        l = self.consolidate_only_one_loss_data(self.loss_data, log=do_backprop)
        #del loss_data

        if not do_backprop:
            self.val_loss.append(l.mean().detach().cpu())
        else:
            self.train_loss.append(l.mean().detach().cpu())
            l.mean().backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

            if self.adversarial_loss:
                self.train_discriminator(target, output2['pseudo_label'])

        if run_online_evaluation:
            target_list = [self.select_deep_supervision(t) for t in target_list]
            self.run_online_evaluation_middle(data=data_list, target=target_list, output=output)
        del data_list, target_list, network_input

        if do_backprop:
            self.iter_nb += 1

        return l.mean().detach().cpu().numpy()

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
        """
        gradient clipping improves training stability

        :param data_generator:
        :param do_backprop:
        :param run_online_evaluation:
        :return:
        """
        data_dict = next(data_generator)
        data = data_dict['data']
        target = data_dict['target']

        if self.classification:
            gt_classification = data_dict['classification']
            gt_classification = maybe_to_torch(gt_classification).long()
            if torch.cuda.is_available():
                gt_classification = to_cuda(gt_classification)
        else:
            gt_classification = None


        #matplotlib.use('QtAgg')
        #fig, ax = plt.subplots(1, 2)
        #ax[0].imshow(data[0][0, 0], cmap='gray')
        #ax[1].imshow(target[0][0, 0], cmap='gray')
        #plt.show()

        if self.directional_field:
            gt_df = data_dict['directional_field']
            gt_df = maybe_to_torch(gt_df)
            if torch.cuda.is_available():
                gt_df = to_cuda(gt_df)
        else:
            gt_df = None

        #matplotlib.use('QtAgg')
        #fig, ax = plt.subplots(1, 2)
        #ax[0].imshow(gt_df[0, 0])
        #ax[1].imshow(target[0][0, 0])
        #plt.show()

        data = maybe_to_torch(data)
        target = maybe_to_torch(target)

        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)

        self.optimizer.zero_grad()
    
        output = self.network(self.select_deep_supervision(data))
        self.compute_losses(x=data, output=output, target=target, gt_df=gt_df, do_backprop=do_backprop, gt_classification=gt_classification)
        
        l = self.consolidate_only_one_loss_data(self.loss_data, log=do_backprop)
        #del loss_data

        if not do_backprop:
            self.val_loss.append(l.mean().detach().cpu())
        else:
            self.train_loss.append(l.mean().detach().cpu())
            l.mean().backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

            if self.adversarial_loss:
                self.train_discriminator(target, output2['pseudo_label'])

        if run_online_evaluation:
            self.run_online_evaluation(data=data, target=target, output=output, gt_df=gt_df, classification_gt=gt_classification, confidence=None)
        del data, target

        if do_backprop:
            self.iter_nb += 1

        return l.mean().detach().cpu().numpy()
    

    def get_middle_ready(self, tr_data_dict, un_data_dict):
        if self.middle:
            tr_middle = tr_data_dict['middle']
            tr_middle = maybe_to_torch(tr_middle)
            if torch.cuda.is_available():
                tr_middle = to_cuda(tr_middle)
            
            un_middle = un_data_dict['middle']
            un_middle = maybe_to_torch(un_middle)
            if torch.cuda.is_available():
                un_middle = to_cuda(un_middle)
        else:
            tr_middle = None
            un_middle = None
        return tr_middle, un_middle
    

    def get_classification_ready(self, tr_data_dict, un_data_dict):
        if self.classification:
            tr_gt_classification = tr_data_dict['classification']
            tr_gt_classification = maybe_to_torch(tr_gt_classification).long()
            if torch.cuda.is_available():
                tr_gt_classification = to_cuda(tr_gt_classification)
            
            un_gt_classification = un_data_dict['classification']
            un_gt_classification = maybe_to_torch(un_gt_classification).long()
            if torch.cuda.is_available():
                un_gt_classification = to_cuda(un_gt_classification)
        else:
            tr_gt_classification = None
            un_gt_classification = None
        return tr_gt_classification, un_gt_classification
    

    def get_directional_field_ready(self, tr_data_dict, un_data_dict):
        if self.directional_field:
            tr_gt_df = tr_data_dict['directional_field']
            tr_gt_df = maybe_to_torch(tr_gt_df)
            if torch.cuda.is_available():
                tr_gt_df = to_cuda(tr_gt_df)

            un_gt_df = un_data_dict['directional_field']
            un_gt_df = maybe_to_torch(un_gt_df)
            if torch.cuda.is_available():
                un_gt_df = to_cuda(un_gt_df)
        else:
            tr_gt_df = None
            un_gt_df = None
        return tr_gt_df, un_gt_df
    
    def get_cps_loss_cutmix(self, data1, data2):
        mask_coords = batched_rand_bbox(data1[0].size())
        cutmixed_img = cutmix(data1, data2, mask_coords)
        with torch.no_grad():
            output1_1 = self.network(data1[0], 1)
            output1_2 = self.network(data2[0], 1)
            output1_1_pred = output1_1['pred']
            output1_2_pred = output1_2['pred']

            output2_1 = self.network(data1[0], 2)
            output2_2 = self.network(data2[0], 2)
            output2_1_pred = output2_1['pred']
            output2_2_pred = output2_2['pred']
        
        pseudo_mask2 = []
        pseudo_mask1 = []

        for i in range(len(output1_1_pred)):
            mask_coords = [x//(2**i) for x in mask_coords]

            softmaxed1_1 = torch.nn.functional.softmax(output1_1_pred[i], dim=1)
            argmaxed1_1 = torch.argmax(softmaxed1_1, dim=1, keepdim=True)
            softmaxed1_2 = torch.nn.functional.softmax(output1_2_pred[i], dim=1)
            argmaxed1_2 = torch.argmax(softmaxed1_2, dim=1, keepdim=True)
            pseudo_mask1.append(cutmix(argmaxed1_1, argmaxed1_2, mask_coords))

            softmaxed2_1 = torch.nn.functional.softmax(output2_1_pred[i], dim=1)
            argmaxed2_1 = torch.argmax(softmaxed2_1, dim=1, keepdim=True)
            softmaxed2_2 = torch.nn.functional.softmax(output2_2_pred[i], dim=1)
            argmaxed2_2 = torch.argmax(softmaxed2_2, dim=1, keepdim=True)
            pseudo_mask2.append(cutmix(argmaxed2_1, argmaxed2_2, mask_coords))
        
        fake1 = torch.nn.functional.one_hot(pseudo_mask1[0].squeeze(1), num_classes=4).permute(0, 3, 1, 2).float()
        fake2 = torch.nn.functional.one_hot(pseudo_mask2[0].squeeze(1), num_classes=4).permute(0, 3, 1, 2).float()
        #fake1 = torch.cat([fake1, cutmixed_img[0]], dim=1)
        #fake2 = torch.cat([fake2, cutmixed_img[0]], dim=1)

        output1 = self.network(cutmixed_img[0], 1)
        output2 = self.network(cutmixed_img[0], 2)

        loss_data1 = self.compute_losses_unlabeled(x=cutmixed_img, output=output1, target=pseudo_mask2, fake=fake1)
        loss_data2 = self.compute_losses_unlabeled(x=cutmixed_img, output=output2, target=pseudo_mask1, fake=fake2)

        loss_data = self.consolidate_loss_data(loss_data1, loss_data2, log=False)
        return loss_data, fake1.detach(), fake2.detach()

    def get_cps_loss(self, data):
        with torch.no_grad():
            output1 = self.network(data[0], 1)
            output2 = self.network(data[0], 2)

        output1_pred = output1['pred']
        output2_pred = output2['pred']
        
        pseudo_mask2 = []
        pseudo_mask1 = []

        for i in range(len(output1_pred)):
            softmaxed1 = torch.nn.functional.softmax(output1_pred[i], dim=1)
            pseudo_mask1.append(torch.argmax(softmaxed1, dim=1, keepdim=True))

            softmaxed2 = torch.nn.functional.softmax(output2_pred[i], dim=1)
            pseudo_mask2.append(torch.argmax(softmaxed2, dim=1, keepdim=True))

        fake1 = torch.nn.functional.one_hot(pseudo_mask1[0].squeeze(1), num_classes=4).permute(0, 3, 1, 2).float()
        fake2 = torch.nn.functional.one_hot(pseudo_mask2[0].squeeze(1), num_classes=4).permute(0, 3, 1, 2).float()
        #fake1 = torch.cat([fake1, data[0]], dim=1)
        #fake2 = torch.cat([fake2, data[0]], dim=1)

        loss_data1 = self.compute_losses_unlabeled(x=data, output=output1, target=pseudo_mask2, fake=fake1)
        loss_data2 = self.compute_losses_unlabeled(x=data, output=output2, target=pseudo_mask1, fake=fake2)

        loss_data = self.consolidate_loss_data(loss_data1, loss_data2, log=False)
        return loss_data, fake1.detach(), fake2.detach()
    
    def get_confidence(self, data):
        with torch.no_grad():
            output = self.network(data[0], 1)

            output_pred = output['pred']
            pseudo_mask = []

            for i in range(len(output_pred)):
                softmaxed1 = torch.nn.functional.softmax(output_pred[i], dim=1)
                pseudo_mask.append(torch.argmax(softmaxed1, dim=1, keepdim=True))

            fake = torch.nn.functional.one_hot(pseudo_mask[0].squeeze(1), num_classes=4).permute(0, 3, 1, 2).float()

            _, confidence = self.get_adversarial_loss_unlabeled(self.discriminator, fake)
            confidence = confidence[0].detach()

            return confidence

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
            
    def run_iteration_unlabeled(self, tr_generator, un_generator, do_backprop, run_online_evaluation=False):
        """
        gradient clipping improves training stability

        :param data_generator:
        :param do_backprop:
        :param run_online_evaluation:
        :return:
        """
        tr_data_dict1 = next(tr_generator)
        un_data_dict1 = next(un_generator)

        if self.cutmix:
            #tr_data_dict2 = next(tr_generator)
            #tr_data2 = tr_data_dict2['data']
            #tr_data2 = maybe_to_torch(tr_data2)
            un_data_dict2 = next(un_generator)
            un_data2 = un_data_dict2['data']
            un_data2 = maybe_to_torch(un_data2)
            if torch.cuda.is_available():
                #tr_data2 = to_cuda(tr_data2)
                un_data2 = to_cuda(un_data2)


        tr_data1 = tr_data_dict1['data']
        tr_target1 = tr_data_dict1['target']
        un_data1 = un_data_dict1['data']

        #matplotlib.use('QtAgg')
        #fig, ax = plt.subplots(1, 3)
        #ax[0].imshow(un_data[0][0, 0], cmap='gray')
        #ax[1].imshow(tr_data[0][0, 0], cmap='gray')
        #ax[2].imshow(tr_target1[0][0, 0], cmap='gray')
        #plt.show()

        tr_data1 = maybe_to_torch(tr_data1)
        un_data1 = maybe_to_torch(un_data1)
        tr_target1 = maybe_to_torch(tr_target1)

        if torch.cuda.is_available():
            tr_data1 = to_cuda(tr_data1)
            un_data1 = to_cuda(un_data1)
            tr_target1 = to_cuda(tr_target1)

        if self.adversarial_loss:
            if do_backprop:
                self.discriminator.train()
            else:
                self.discriminator.eval()

        self.optimizer.zero_grad()

        if self.cutmix:
            un_loss_data, un_fake1, un_fake2 = self.get_cps_loss_cutmix(un_data1, un_data2)
            #tr_loss_data, tr_fake1, tr_fake2, confidence = self.get_cps_loss_cutmix(tr_data1, tr_data2)
        else:
            un_loss_data, un_fake1, un_fake2 = self.get_cps_loss(un_data1)
            #tr_loss_data, tr_fake1, tr_fake2, confidence = self.get_cps_loss(tr_data1)
        #cps_loss_data = self.consolidate_loss_data(un_loss_data, tr_loss_data, log=False)
        self.writer.add_scalar('Iteration/Whole cps loss', self.convert_loss_data_to_number(un_loss_data), self.iter_nb)
        #del un_loss_data, tr_loss_data

        l_output1 = self.network(tr_data1[0], 1)
        l_output2 = self.network(tr_data1[0], 2)

        s_loss_data1 = self.compute_losses(x=tr_data1, output=l_output1, target=tr_target1, do_backprop=do_backprop)
        s_loss_data2 = self.compute_losses(x=tr_data1, output=l_output2, target=tr_target1, do_backprop=do_backprop)
        s_loss_data = self.consolidate_loss_data(s_loss_data1, s_loss_data2, log=False)
        self.writer.add_scalar('Iteration/Whole supervised loss', self.convert_loss_data_to_number(s_loss_data), self.iter_nb)
        del s_loss_data1, s_loss_data2

        l_loss_data = self.consolidate_loss_data(s_loss_data, un_loss_data, log=do_backprop, description='Whole ', w1=1, w2=self.unlabeled_loss_weight)
        self.writer.add_scalars('Iteration/supervised loss weights', {key:value[0] for key, value in l_loss_data.items()}, self.iter_nb)

        l = self.convert_loss_data_to_number(l_loss_data)

        if do_backprop:
            self.writer.add_scalar('Iteration/Training loss', l, self.iter_nb)
            self.val_loss.append(l.detach().cpu())
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

            if self.adversarial_loss:
                real = torch.nn.functional.one_hot(tr_target1[0].squeeze(1).long(), num_classes=4).permute(0, 3, 1, 2).float()
                #real = torch.cat([real, tr_data1[0]], dim=1)
                #fake = [un_fake1, un_fake2, tr_fake1, tr_fake2]
                fake = [un_fake1, un_fake2]
                discriminator_loss = self.learn_discriminator_unlabeled(self.discriminator, self.discriminator_optimizer, real, fake)
                self.writer.add_scalar('Discriminator/discriminator_loss', discriminator_loss, self.iter_nb)
        else:
            #if self.vq_vae:
            #    self.writer.add_scalar('Iteration/Perplexity', output['perplexity'], self.iter_nb)
            self.train_loss.append(l.detach().cpu())

        if run_online_evaluation:
            confidence = self.get_confidence(tr_data1)
            self.run_online_evaluation(data=tr_data1, target=tr_target1, output=l_output1, gt_df=None, classification_gt=None, confidence=confidence)
        del tr_data1, un_data1, tr_target1

        if do_backprop:
            self.iter_nb += 1

        return l.detach().cpu().numpy()
    

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
                elif '029' in self.dataset_directory or '030' in self.dataset_directory:
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
        
        
        if self.unlabeled or self.middle_unlabeled:
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
                if self.adversarial_loss:
                    self.discriminator_scheduler.step()
                    #if self.unlabeled:
                    #else:
                    #    self.seg_discriminator_scheduler.step()
                    #    self.rec_discriminator_scheduler.step()
            else:
                self.optimizer.param_groups[0]['lr'] = poly_lr(ep, self.max_num_epochs, self.initial_lr, 0.9)
                if self.adversarial_loss:
                    self.discriminator_optimizer.param_groups[0]['lr'] = poly_lr(ep, self.max_num_epochs, self.discriminator_lr, 0.9)
                    #if self.unlabeled:
                    #else:
                    #    self.seg_discriminator_optimizer.param_groups[0]['lr'] = poly_lr(ep, self.max_num_epochs, self.discriminator_lr, 0.9)
                    #    self.rec_discriminator_optimizer.param_groups[0]['lr'] = poly_lr(ep, self.max_num_epochs, self.discriminator_lr, 0.9)
            if self.adversarial_loss:
                lrs = {'Unet_lr': self.optimizer.param_groups[0]['lr'], 'discriminator_lr': self.discriminator_optimizer.param_groups[0]['lr']}
                #if self.unlabeled:
                #else:
                #    lrs = {'Unet_lr': self.optimizer.param_groups[0]['lr'], 'seg_discriminator_lr': self.seg_discriminator_optimizer.param_groups[0]['lr'], 'rec_discriminator_lr': self.rec_discriminator_optimizer.param_groups[0]['lr']}
                self.writer.add_scalars('Epoch/Learning rate', lrs, self.epoch)
            else:
                self.writer.add_scalar('Epoch/Learning rate', self.optimizer.param_groups[0]['lr'], self.epoch)
        else:
            ep = epoch
            if not self.config['scheduler'] == 'cosine':
                self.optimizer.param_groups[0]['lr'] = poly_lr(ep, self.max_num_epochs, self.initial_lr, 0.9)
                if self.adversarial_loss:
                    self.discriminator_optimizer.param_groups[0]['lr'] = poly_lr(ep, self.max_num_epochs, self.discriminator_lr, 0.9)
                    #if self.unlabeled:
                    #else:
                    #    self.seg_discriminator_optimizer.param_groups[0]['lr'] = poly_lr(ep, self.max_num_epochs, self.discriminator_lr, 0.9)
                    #    self.rec_discriminator_optimizer.param_groups[0]['lr'] = poly_lr(ep, self.max_num_epochs, self.discriminator_lr, 0.9)

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
    
    def run_training_mtl(self):
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

            if self.use_progress_bar:
                with trange(self.num_batches_per_epoch) as tbar:
                    for b in tbar:
                        tbar.set_description("Epoch {}/{}".format(self.epoch+1, self.max_num_epochs))

                        if self.binary:
                            l = self.run_iteration(self.dl_tr, do_backprop=True)
                        else:
                            l = self.run_iteration(self.tr_gen, do_backprop=True)

                        tbar.set_postfix(loss=l)
                        train_losses_epoch.append(l)
            else:
                for _ in range(self.num_batches_per_epoch):
                    if self.binary:
                        l = self.run_iteration(self.dl_tr, do_backprop=True)
                    else:
                        l = self.run_iteration(self.tr_gen, do_backprop=True)
                    train_losses_epoch.append(l)

            self.all_tr_losses.append(np.mean(train_losses_epoch))
            self.print_to_log_file("train loss : %.4f" % self.all_tr_losses[-1])

            if self.epoch % self.config['epoch_log'] == 0:
                with torch.no_grad():
                    # validation with train=False
                    self.network.eval()
                    val_losses = []
                    for b in range(self.num_val_batches_per_epoch):
                        if self.binary:
                            l = self.run_iteration(self.dl_val, do_backprop=False, run_online_evaluation=True)
                        else:
                            l = self.run_iteration(self.val_gen, do_backprop=False, run_online_evaluation=True)
                        val_losses.append(l)
                    self.all_val_losses.append(np.mean(val_losses))
                    self.print_to_log_file("validation loss: %.4f" % self.all_val_losses[-1])

                    if self.also_val_in_tr_mode:
                        self.network.train()
                        # validation with train=True
                        val_losses = []
                        for b in range(self.num_val_batches_per_epoch):
                            if self.binary:
                                l = self.run_iteration(self.dl_val, do_backprop=False)
                            else:
                                l = self.run_iteration(self.val_gen, do_backprop=False)
                            val_losses.append(l)
                        self.all_val_losses_tr_mode.append(np.mean(val_losses))
                        self.print_to_log_file("validation loss (train=True): %.4f" % self.all_val_losses_tr_mode[-1])

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
        # now we can delete latest as it will be identical with final
        if isfile(join(self.output_folder, "model_latest.model")):
            os.remove(join(self.output_folder, "model_latest.model"))
        if isfile(join(self.output_folder, "model_latest.model.pkl")):
            os.remove(join(self.output_folder, "model_latest.model.pkl"))
    
    def run_training_mtl_middle(self):
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

            if self.use_progress_bar:
                with trange(self.num_batches_per_epoch) as tbar:
                    for b in tbar:
                        tbar.set_description("Epoch {}/{}".format(self.epoch+1, self.max_num_epochs))

                        l = self.run_iteration_middle(self.tr_gen, do_backprop=True)

                        tbar.set_postfix(loss=l)
                        train_losses_epoch.append(l)
            else:
                for _ in range(self.num_batches_per_epoch):
                    l = self.run_iteration_middle(self.tr_gen, True)
                    train_losses_epoch.append(l)

            self.all_tr_losses.append(np.mean(train_losses_epoch))
            self.print_to_log_file("train loss : %.4f" % self.all_tr_losses[-1])

            if self.epoch % self.config['epoch_log'] == 0:
                #if self.middle and not self.middle_unlabeled:
                #    self.get_slice_percent()
                #    self.dl_val.percent = self.percent
                with torch.no_grad():
                    # validation with train=False
                    self.network.eval()
                    val_losses = []
                    for b in range(self.num_val_batches_per_epoch):
                        l = self.run_iteration_middle(self.val_gen, do_backprop=False, run_online_evaluation=True)
                        val_losses.append(l)
                    self.all_val_losses.append(np.mean(val_losses))
                    self.print_to_log_file("validation loss: %.4f" % self.all_val_losses[-1])

                    if self.also_val_in_tr_mode:
                        self.network.train()
                        # validation with train=True
                        val_losses = []
                        for b in range(self.num_val_batches_per_epoch):
                            l = self.run_iteration_middle(self.val_gen, do_backprop=False)
                            val_losses.append(l)
                        self.all_val_losses_tr_mode.append(np.mean(val_losses))
                        self.print_to_log_file("validation loss (train=True): %.4f" % self.all_val_losses_tr_mode[-1])

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
        
        #if self.middle and not self.middle_unlabeled:
        #    self.get_slice_percent()
        #    self.network.percent = self.percent

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
        ds = self.network.do_ds
        #self.network.do_ds = True
        self.save_debug_information()
        if self.middle:
            ret = self.run_training_mtl_middle()
        else:
            ret = self.run_training_mtl()
        self.network.do_ds = ds
        return ret

    
    def run_training_unlabeled(self):
        """
        if we run with -c then we need to set the correct lr for the first epoch, otherwise it will run the first
        continued epoch with self.initial_lr

        we also need to make sure deep supervision in the network is enabled for training, thus the wrapper
        :return:
        """
        self.maybe_update_lr(self.epoch)  # if we dont overwrite epoch then self.epoch+1 is used which is not what we
        # want at the start of the training
        ds = self.network.do_ds
        self.network.do_ds = True
        ret = self.training_unlabeled()
        self.network.do_ds = ds
        return ret


    def training_unlabeled(self):
        if not torch.cuda.is_available():
            self.print_to_log_file("WARNING!!! You are attempting to run training on a CPU (torch.cuda.is_available() is False). This can be VERY slow!")

        _ = self.tr_gen.next()
        _ = self.val_gen.next()
        _ = self.tr_un_gen.next()
        _ = self.val_un_gen.next()

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

            if self.use_progress_bar:
                with trange(self.num_batches_per_epoch) as tbar:
                    for b in tbar:
                        tbar.set_description("Epoch {}/{}".format(self.epoch+1, self.max_num_epochs))

                        l = self.run_iteration_unlabeled(self.tr_gen, un_generator=self.tr_un_gen, do_backprop=True)

                        tbar.set_postfix(loss=l)
                        train_losses_epoch.append(l)
            else:
                for _ in range(self.num_batches_per_epoch):
                    l = self.run_iteration_unlabeled(self.tr_gen, un_generator=self.tr_un_gen, do_backprop=True)
                    train_losses_epoch.append(l)

            self.all_tr_losses.append(np.mean(train_losses_epoch))
            self.print_to_log_file("train loss : %.4f" % self.all_tr_losses[-1])

            if self.epoch % self.config['epoch_log'] == 0:
                with torch.no_grad():
                    # validation with train=False
                    self.network.eval()
                    val_losses = []
                    for b in range(self.num_val_batches_per_epoch):
                        l = self.run_iteration_unlabeled(self.val_gen, un_generator=self.val_un_gen, do_backprop=False, run_online_evaluation=True)
                        val_losses.append(l)
                    self.all_val_losses.append(np.mean(val_losses))
                    self.print_to_log_file("validation loss: %.4f" % self.all_val_losses[-1])

                    if self.also_val_in_tr_mode:
                        self.network.train()
                        # validation with train=True
                        val_losses = []
                        for b in range(self.num_val_batches_per_epoch):
                            l = self.run_iteration_unlabeled(self.val_gen, un_generator=self.val_un_gen, do_backprop=False)
                            val_losses.append(l)
                        self.all_val_losses_tr_mode.append(np.mean(val_losses))
                        self.print_to_log_file("validation loss (train=True): %.4f" % self.all_val_losses_tr_mode[-1])
                    
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
        # now we can delete latest as it will be identical with final
        if isfile(join(self.output_folder, "model_latest.model")):
            os.remove(join(self.output_folder, "model_latest.model"))
        if isfile(join(self.output_folder, "model_latest.model.pkl")):
            os.remove(join(self.output_folder, "model_latest.model.pkl"))

    
    def load_dataset(self):
        self.dataset = load_dataset(self.folder_with_preprocessed_data)
        if self.unlabeled or self.middle_unlabeled:
            self.unlabeled_dataset = load_unlabeled_dataset(self.folder_with_preprocessed_data)
    
    def get_basic_generators(self):
        self.load_dataset()
        self.do_split()
        dl_un_tr = None
        dl_un_val = None
        if self.binary:
            dl_tr = DataLoader2DBinary(self.dataset_tr, self.basic_generator_patch_size, self.patch_size, self.batch_size, self.deep_supervision,
                                isval=False,
                                oversample_foreground_percent=self.oversample_foreground_percent,
                                pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
            dl_val = DataLoader2DBinary(self.dataset_val, self.patch_size, self.patch_size, self.batch_size, self.deep_supervision,
                                isval=True,
                                oversample_foreground_percent=self.oversample_foreground_percent,
                                pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
        else:
            dl_tr = DataLoader2D(self.dataset_tr, self.basic_generator_patch_size, self.patch_size, self.batch_size,
                                oversample_foreground_percent=self.oversample_foreground_percent,
                                pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
            dl_val = DataLoader2D(self.dataset_val, self.patch_size, self.patch_size, self.batch_size,
                                oversample_foreground_percent=self.oversample_foreground_percent,
                                pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
            if self.unlabeled:
                dl_un_tr = DataLoader2DUnlabeled(self.dataset_un_tr, self.basic_generator_patch_size, self.patch_size, self.batch_size,
                                            oversample_foreground_percent=self.oversample_foreground_percent,
                                            pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
                dl_un_val = DataLoader2DUnlabeled(self.dataset_un_val, self.basic_generator_patch_size, self.patch_size, self.batch_size,
                                            oversample_foreground_percent=self.oversample_foreground_percent,
                                            pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
        return dl_tr, dl_val, dl_un_tr, dl_un_val


    def learn_discriminator_unlabeled(self,  discriminator, discriminator_optimizer, real, fakes):
        B, C, H, W = real.shape
        label = torch.full((B, 1, H, W), 1, dtype=torch.float, device=real.device)
        discriminator.zero_grad()
        discriminator_optimizer.zero_grad()
        output_real = discriminator(real)[0]
        loss_real = self.adv_loss(output_real, label)
        self.writer.add_scalar('Discriminator/real', output_real.mean(), self.iter_nb)
        #loss_real.backward() #retain_graph if reuse output of discriminator for r1 penalty

        r1_penalty = 0
        #if iter_nb % self.r1_penalty_iteration == 0:
        #    r1_penalty = logisticGradientPenalty(real, discriminator, weight=5)
    
        loss_real_r1 = loss_real + r1_penalty
        loss_real_r1.backward()

        label.fill_(0)
        loss_fake = 0
        log_output_fake = 0
        for fake in fakes:
            assert fake.size() == real.size()

            output_fake = discriminator(fake.detach())[0]
            log_output_fake += output_fake.mean()
            current_loss_fake = self.adv_loss(output_fake, label).clone()
            loss_fake = loss_fake + current_loss_fake
        loss_fake = loss_fake / len(fakes)
        self.writer.add_scalar('Discriminator/fake', log_output_fake / len(fakes), self.iter_nb)

        loss_fake.backward()
        discriminator_loss = loss_real + loss_fake + r1_penalty
        discriminator_optimizer.step()
        
        return discriminator_loss
    
    def get_adversarial_loss_unlabeled(self, discriminator, fake):
        B, C, H, W = fake.shape
        label = torch.full((B, 1, H, W), 1, dtype=torch.float, device=fake.device)

        output = discriminator(fake)
        adversarial_loss = self.adv_loss(output[0], label)

        w_list = []
        for i in range(len(output)):
            w = output[i].numel() * (output[i] / output[i].sum())
            w_list.append(w)

        return adversarial_loss, w_list

    #def get_adversarial_loss(self, discriminator, discriminator_optimizer, real, fake, do_backprop):
    #    discriminator_loss = None
    #    adversarial_loss = None
    #    output_real = None
    #    output_fake = None
    #    label = torch.full((self.batch_size,), 1, dtype=torch.float, device=real.device)
    #    if do_backprop:
    #        discriminator.zero_grad()
    #        discriminator_optimizer.zero_grad()
    #        output_real = discriminator(real).reshape(-1)
    #        loss_real = self.adv_loss(output_real, label)
    #        #loss_real.backward() #retain_graph if reuse output of discriminator for r1 penalty
#
    #        r1_penalty = 0
    #        #if iter_nb % self.r1_penalty_iteration == 0:
    #        #    r1_penalty = logisticGradientPenalty(real, discriminator, weight=5)
    #    
    #        loss_real_r1 = loss_real + r1_penalty
    #        loss_real_r1.backward()
#
    #        assert fake.size() == real.size()
    #        label.fill_(0)
    #        output_fake = discriminator(fake.detach()).view(-1)
    #        loss_fake = self.adv_loss(output_fake, label)
    #        loss_fake.backward()
#
    #        discriminator_loss = loss_real + loss_fake + r1_penalty
    #        discriminator_optimizer.step()
    #        #discriminator_scheduler.step()

        ##self.model.reconstruction.zero_grad()
        #label.fill_(1)
        #output = discriminator(fake).view(-1)
        #adversarial_loss = self.adv_loss(output, label)
#
        #return adversarial_loss, discriminator_loss, output_real, output_fake

    def get_adversarial_loss(self, fake):
        label = torch.full((self.batch_size,), 1, dtype=torch.float, device=fake.device)
        output = self.discriminator(fake).view(-1)
        adversarial_loss = self.adv_loss(output, label)
        return adversarial_loss
    
    def get_discriminator_loss(self, real, fake):
        label = torch.full((self.batch_size,), 1, dtype=torch.float, device=real.device)
        output_real = self.discriminator(real).reshape(-1)
        loss_real = self.adv_loss(output_real, label)
        #loss_real.backward() #retain_graph if reuse output of discriminator for r1 penalty

        r1_penalty = 0
        #if iter_nb % self.r1_penalty_iteration == 0:
        #    r1_penalty = logisticGradientPenalty(real, discriminator, weight=5)
    
        loss_real_r1 = loss_real + r1_penalty
        loss_real_r1.backward()

        assert fake.size() == real.size()
        label.fill_(0)
        output_fake = self.discriminator(fake.detach()).view(-1)
        loss_fake = self.adv_loss(output_fake, label)
        loss_fake.backward()

        discriminator_loss = loss_real + loss_fake + r1_penalty
        return discriminator_loss, output_real, output_fake
    
    def train_discriminator(self, target, pseudo_label):
        self.discriminator.zero_grad()
        self.discriminator_optimizer.zero_grad()
        real = torch.nn.functional.one_hot(target.squeeze(1).long(), num_classes=4).permute(0, 3, 1, 2).float()
        fake = torch.nn.functional.one_hot(pseudo_label.squeeze(1).long(), num_classes=4).permute(0, 3, 1, 2).float()

        discriminator_loss, output_real, output_fake = self.get_discriminator_loss(real, fake)

        self.writer.add_scalar('Discriminator/Real', output_real.mean(), self.iter_nb)
        self.writer.add_scalar('Discriminator/Fake', output_fake.mean(), self.iter_nb)
        self.writer.add_scalar('Discriminator/Loss', discriminator_loss, self.iter_nb)

        self.discriminator_optimizer.step()