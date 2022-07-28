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
from typing import Tuple
import matplotlib
from datetime import datetime

from torch.nn.functional import interpolate
import psutil
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

import cv2 as cv
import sys
import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt
from copy import copy
from time import time, sleep, strftime
import yaml
import numpy as np
import torch
from torchinfo import summary
from nnunet.lib.ssim import ssim
from nnunet.utilities.tensor_utilities import sum_tensor
from nnunet.training.data_augmentation.data_augmentation_moreDA import get_moreDA_augmentation
from nnunet.training.loss_functions.deep_supervision import MultipleOutputLoss2
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
from nnunet.network_architecture.generic_UNet import Generic_UNet
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.network_architecture.neural_network import SegmentationNetwork
from nnunet.training.data_augmentation.default_data_augmentation import default_2D_augmentation_params, \
    get_patch_size, default_3D_augmentation_params
from nnunet.training.dataloading.dataset_loading import unpack_dataset
from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer
from nnunet.utilities.nd_softmax import softmax_helper
from sklearn.model_selection import KFold
from torch import nn
from torch.cuda.amp import autocast
from nnunet.training.learning_rate.poly_lr import poly_lr
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.lib.training_utils import build_2d_model, read_config
from nnunet.lib.loss import DirectionalFieldLoss
from nnunet.lib.dataset_utils import normalize_0_1
from pathlib import Path
from monai.losses import DiceFocalLoss, DiceLoss
from torch.utils.tensorboard import SummaryWriter
from nnunet.lib.boundary_utils import simplex
from nnunet.training.loss_functions.dice_loss import DC_and_CE_loss, DC_and_focal_loss, DC_and_topk_loss
from nnunet.training.loss_functions.crossentropy import RobustCrossEntropyLoss


class nnMTLTrainerV2(nnUNetTrainer):
    """
    Info for Fabian: same as internal nnUNetTrainerV2_2
    """

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.config = read_config(os.path.join(Path.cwd(), 'adversarial_acdc.yaml'))
        self.max_num_epochs = self.config['max_num_epochs']
        self.initial_lr = self.config['initial_lr']
        self.weight_decay = self.config['weight_decay']
        self.deep_supervision_scales = None
        self.ds_loss_weights = None
        self.use_progress_bar=True
        self.directional_field = self.config['directional_field']

        self.eval_images = self.initialize_image_data()
        self.iter_nb = 0

        self.val_loss = []
        self.train_loss = []
        
        self.min_max_normalization = self.config['min_max_normalization']
        self.image_size = self.config['image_size']

        if self.image_size == 224:
            loss_weights = torch.tensor(self.config['224_loss_weights'], device=self.config['device'])
        elif self.image_size == 128:
            loss_weights = torch.tensor(self.config['128_loss_weights_125'], device=self.config['device'])

        timestr = strftime("%Y-%m-%d_%HH%M")
        self.log_dir = os.path.join(self.output_folder, timestr)
        self.writer = SummaryWriter(log_dir=self.log_dir)

        self.loss_data = {'segmentation': [1.0, float('nan')]}

        if self.config['reconstruction']:
            self.loss_data['reconstruction'] = [self.config['reconstruction_loss_weight'], float('nan')]
            self.loss_data['similarity'] = [self.config['similarity_weight'], float('nan')]
            self.mse_loss = nn.MSELoss()
            self.similarity_loss = nn.L1Loss()

        if self.config['learn_transforms']:
            self.loss_data['rotation'] = [self.config['rotation_loss_weight'], float('nan')]
            self.loss_data['rotation_reconstruction'] = [self.config['reconstruction_rotation_loss_weight'], float('nan')]
            self.loss_data['scaling'] = [self.config['scaling_loss_weight'], float('nan')]
            self.loss_data['scaling_reconstruction'] = [self.config['reconstruction_scaling_loss_weight'], float('nan')]
        
        if self.directional_field:
            self.directional_field_loss = DirectionalFieldLoss(weights=loss_weights, writer=self.writer)
            self.loss_data['directional_field'] = [self.config['directional_field_weight'], float('nan')]

        #self.segmentation_loss = DiceFocalLoss(include_background=False, focal_weight=loss_weights[1:] if not self.config['binary'] else None, softmax=True)
        #self.segmentation_loss = torch.nn.CrossEntropyLoss(weight=loss_weights, ignore_index=0)
        #self.segmentation_loss = DiceLoss(include_background=False, softmax=True)
        if self.config['loss'] == 'focal_and_dice':
            self.segmentation_loss = DC_and_focal_loss({'batch_dice': self.batch_dice, 'smooth': 1e-5, 'do_bg': False}, {'apply_nonlin': nn.Softmax(dim=1), 'alpha':0.5, 'gamma':2, 'smooth':1e-5})
        elif self.config['loss'] == 'topk_and_dice':
            self.segmentation_loss = DC_and_topk_loss({'batch_dice': self.batch_dice, 'smooth': 1e-5, 'do_bg': False}, {'k': 10})
        elif self.config['loss'] == 'ce_and_dice':
            self.segmentation_loss = DC_and_CE_loss({'batch_dice': self.batch_dice, 'smooth': 1e-5, 'do_bg': False}, {'weight': loss_weights})
        elif self.config['loss'] == 'ce':
            self.segmentation_loss = RobustCrossEntropyLoss(weight=loss_weights, ignore_index=0)

        self.pin_memory = True
        self.similarity_downscale = self.config['similarity_down_scale']

        self.reconstruction_ssim_list = []
        self.df_ssim_list = []
        self.sim_l2_list = []

    def initialize_image_data(self):

        log_images_nb = 8
        eval_images = {'rec': None,
                        'sim': None,
                        'df': None,
                        'seg': None}

        for key in eval_images.keys():
            data = []
            scores = []
            if key == 'rec':
                payload = {'input': None,
                            'reconstruction': None}
                score = -1
                data.append(payload)
                scores.append(score)
            elif key == 'sim':
                for i in range(log_images_nb):
                    payload = {'input': None,
                                'decoder_sm': None,
                                'reconstruction_sm': None}
                    score = float('inf')
                    data.append(payload)
                    scores.append(score)
            elif key == 'df':
                for i in range(log_images_nb):
                    payload = {'input': None,
                                'gt_df': None,
                                'pred_df': None}
                    score = -1
                    data.append(payload)
                    scores.append(score)
            elif key == 'seg':
                for i in range(log_images_nb):
                    payload = {'input': None,
                                'pred': None,
                                'gt': None}
                    score = 1
                    data.append(payload)
                    scores.append(score)
            eval_images[key] = np.stack([np.array(data), np.array(scores)], axis=0)
    
        return eval_images
    
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

            self.segmentation_loss = MultipleOutputLoss2(self.segmentation_loss, seg_weights)
            self.reconstruction_loss = MultipleOutputLoss2(self.mse_loss, weights)
            ################# END ###################

            self.folder_with_preprocessed_data = join(self.dataset_directory, self.plans['data_identifier'] +
                                                      "_stage%d" % self.stage)
            if training:
                self.dl_tr, self.dl_val = self.get_basic_generators()
                if self.unpack_data:
                    print("unpacking dataset")
                    unpack_dataset(self.folder_with_preprocessed_data)
                    print("done")
                else:
                    print(
                        "INFO: Not unpacking data! Training may be slow due to that. Pray you are not using 2d or you "
                        "will wait all winter for your model to finish!")

                self.tr_gen, self.val_gen = get_moreDA_augmentation(
                    self.dl_tr, self.dl_val,
                    self.data_aug_params[
                        'patch_size_for_spatialtransform'],
                    directional_field=self.directional_field,
                    params=self.data_aug_params,
                    deep_supervision_scales=self.deep_supervision_scales,
                    min_max_normalization=self.min_max_normalization,
                    pin_memory=self.pin_memory,
                    use_nondetMultiThreadedAugmenter=False
                )
                self.print_to_log_file("TRAINING KEYS:\n %s" % (str(self.dataset_tr.keys())),
                                       also_print_to_console=False)
                self.print_to_log_file("VALIDATION KEYS:\n %s" % (str(self.dataset_val.keys())),
                                       also_print_to_console=False)
            else:
                pass

            self.initialize_network()
            self.initialize_optimizer_and_scheduler()

            assert isinstance(self.network, (SegmentationNetwork, nn.DataParallel))
        else:
            self.print_to_log_file('self.was_initialized is True, not running self.initialize again')
        self.was_initialized = True
    
    def print_to_log_file(self, *args, also_print_to_console=True, add_timestamp=True):

        timestamp = time()
        dt_object = datetime.fromtimestamp(timestamp)

        if add_timestamp:
            args = ("%s:" % dt_object, *args)

        if self.log_file is None:
            maybe_mkdir_p(self.log_dir)
            timestamp = datetime.now()
            self.log_file = join(self.log_dir, "training_log_%d_%d_%d_%02.0d_%02.0d_%02.0d.txt" %
                                 (timestamp.year, timestamp.month, timestamp.day, timestamp.hour, timestamp.minute,
                                  timestamp.second))
            with open(self.log_file, 'w') as f:
                f.write("Starting... \n")
        successful = False
        max_attempts = 5
        ctr = 0
        while not successful and ctr < max_attempts:
            try:
                with open(self.log_file, 'a+', encoding='utf-8') as f:
                    for a in args:
                        f.write(str(a))
                        f.write(" ")
                    f.write("\n")
                successful = True
            except IOError:
                print("%s: failed to log: " % datetime.fromtimestamp(timestamp), sys.exc_info())
                sleep(0.5)
                ctr += 1
        if also_print_to_console:
            print(*args)
    
    def count_parameters(self, config, models):
        params_sum = 0
        for k, v in models.items():
            nb_params =  sum(p.numel() for p in v[0].parameters() if p.requires_grad)
            params_sum += nb_params
            self.print_to_log_file(yaml.safe_dump(config, default_flow_style=None, sort_keys=False), also_print_to_console=False)

            model_stats = summary(v[0], input_size=v[1], 
                                col_names=["input_size", "output_size", "num_params", "mult_adds"], 
                                col_width=16,
                                verbose=0)
            model_stats.formatting.verbose = 1
            self.print_to_log_file(model_stats, also_print_to_console=False)
        
        self.print_to_log_file("The model has", "{:,}".format(params_sum), "parameters")

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

        self.network = build_2d_model(self.config, norm=getattr(torch.nn, self.config['norm']))

        models = {}
        model_input_size = (self.config['batch_size'], 1, self.image_size, self.image_size)
        models['model'] = (self.network, model_input_size)
        self.count_parameters(self.config, models)

        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper

    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, "self.initialize_network must be called first"
        if self.config['optimizer'] == 'sgd':
            self.optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                            momentum=0.99, nesterov=True)
        elif self.config['optimizer'] == 'adam':
            self.optimizer = torch.optim.AdamW(self.network.parameters(), lr=self.initial_lr, weight_decay=self.weight_decay)

        if self.config['scheduler'] == 'cosine':
            self.lr_scheduler = CosineAnnealingLR(self.optimizer, T_max=self.max_num_epochs)
            #self.warmup = LinearLR(optimizer=self.optimizer, start_factor=0.1, end_factor=1, total_iters=self.num_batches_per_epoch)
            #self.lr_scheduler = SequentialLR(optimizer=self.optimizer, schedulers=[warmup, cosine_scheduler], milestones=[1])
        else:
            self.lr_scheduler = None

    def get_images_ready_for_display(self, image, colormap):
        if colormap is not None:
            if np.count_nonzero(image) > 0:
                image = normalize_0_1(image)
            image = colormap(image)[:, :, :-1]
        image = cv.normalize(image, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX).astype(np.uint8)
        if len(image.shape) < 3:
            image = image[:, :, None]
        return image
    
    def get_seg_images_ready_for_display(self, image, colormap, norm):
        image = norm(image)
        image = colormap(image)[:, :, :-1]
        image = cv.normalize(image, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX).astype(np.uint8)
        if len(image.shape) < 3:
            image = image[:, :, None]
        return image
    
    def log_rec_images(self):
        input_image = self.eval_images['rec'][0, 0]['input']
        input_image = self.get_images_ready_for_display(input_image, colormap=None)

        reconstruction = self.eval_images['rec'][0, 0]['reconstruction']
        reconstruction = self.get_images_ready_for_display(reconstruction, colormap=None)

        self.writer.add_image(os.path.join('Reconstruction', 'input').replace('\\', '/'), input_image, self.epoch, dataformats='HWC')
        self.writer.add_image(os.path.join('Reconstruction', 'reconstruction').replace('\\', '/'), reconstruction, self.epoch, dataformats='HWC')
    
    def log_sim_images(self, colormap):
        input_list = [x['input'] for x in self.eval_images['sim'][0]]
        input_list = [self.get_images_ready_for_display(x, colormap=None) for x in input_list]
        input_list = np.stack(input_list, axis=0)

        decoder_sm_list = [x['decoder_sm'] for x in self.eval_images['sim'][0]]
        decoder_sm_list = [self.get_images_ready_for_display(x, colormap) for x in decoder_sm_list]
        decoder_sm_list = np.stack(decoder_sm_list, axis=0)

        reconstruction_sm_list = [x['reconstruction_sm'] for x in self.eval_images['sim'][0]]
        reconstruction_sm_list = [self.get_images_ready_for_display(x, colormap) for x in reconstruction_sm_list]
        reconstruction_sm_list = np.stack(reconstruction_sm_list, axis=0)

        self.writer.add_images(os.path.join('Similarity', 'input').replace('\\', '/'), input_list, self.epoch, dataformats='NHWC')
        self.writer.add_images(os.path.join('Similarity', 'decoder_similarity').replace('\\', '/'), decoder_sm_list, self.epoch, dataformats='NHWC')
        self.writer.add_images(os.path.join('Similarity', 'reconstruction_similarity').replace('\\', '/'), reconstruction_sm_list, self.epoch, dataformats='NHWC')

    def log_df_images(self, colormap):
        input_list = [x['input'] for x in self.eval_images['df'][0]]
        input_list = [self.get_images_ready_for_display(x, colormap=None) for x in input_list]
        input_list = np.stack(input_list, axis=0)

        gt_df_list = [x['gt_df'] for x in self.eval_images['df'][0]]
        gt_df_list = [self.get_images_ready_for_display(x, colormap) for x in gt_df_list]
        gt_df_list = np.stack(gt_df_list, axis=0)

        pred_df_list = [x['pred_df'] for x in self.eval_images['df'][0]]
        pred_df_list = [self.get_images_ready_for_display(x, colormap) for x in pred_df_list]
        pred_df_list = np.stack(pred_df_list, axis=0)

        self.writer.add_images(os.path.join('Directional_field', 'input').replace('\\', '/'), input_list, self.epoch, dataformats='NHWC')
        self.writer.add_images(os.path.join('Directional_field', 'ground_truth_directional_field').replace('\\', '/'), gt_df_list, self.epoch, dataformats='NHWC')
        self.writer.add_images(os.path.join('Directional_field', 'predicted_directional_field').replace('\\', '/'), pred_df_list, self.epoch, dataformats='NHWC')
    
    def log_seg_images(self, colormap, norm):
        input_list = [x['input'] for x in self.eval_images['seg'][0]]
        input_list = [self.get_images_ready_for_display(x, colormap=None) for x in input_list]
        input_list = np.stack(input_list, axis=0)

        gt_list = [x['gt'] for x in self.eval_images['seg'][0]]
        gt_list = [self.get_seg_images_ready_for_display(x, colormap, norm) for x in gt_list]
        gt_list = np.stack(gt_list, axis=0)

        pred_list = [x['pred'] for x in self.eval_images['seg'][0]]
        pred_list = [self.get_seg_images_ready_for_display(x, colormap, norm) for x in pred_list]
        pred_list = np.stack(pred_list, axis=0)

        self.writer.add_images(os.path.join('Segmentation', 'input').replace('\\', '/'), input_list, self.epoch, dataformats='NHWC')
        self.writer.add_images(os.path.join('Segmentation', 'ground_truth').replace('\\', '/'), gt_list, self.epoch, dataformats='NHWC')
        self.writer.add_images(os.path.join('Segmentation', 'prediction').replace('\\', '/'), pred_list, self.epoch, dataformats='NHWC')
    
    def set_up_image_seg(self, seg_dice, gt, pred, x):
        seg_dice = seg_dice.cpu().numpy()
        gt = gt.cpu().numpy()
        pred = pred.cpu().numpy()
        x = x.cpu().numpy()

        if self.eval_images['seg'][1, -1] > seg_dice:

            self.eval_images['seg'][0, -1]['gt'] = gt.astype(np.float32)
            self.eval_images['seg'][0, -1]['pred'] = pred.astype(np.float32)
            self.eval_images['seg'][0, -1]['input'] = x.astype(np.float32)
            self.eval_images['seg'][1, -1] = seg_dice

            sorted_indices = self.eval_images['seg'][1, :].argsort()
            self.eval_images['seg'] = self.eval_images['seg'][:, sorted_indices]

    def set_up_image_df(self, df_ssim, gt_df, pred_df, x):
        df_ssim = df_ssim.cpu().numpy()
        gt_df = gt_df.cpu().numpy()
        pred_df = pred_df.cpu().numpy()
        x = x.cpu().numpy()

        if self.eval_images['df'][1, 0] < df_ssim:

            self.eval_images['df'][0, 0]['gt_df'] = gt_df.astype(np.float32)
            self.eval_images['df'][0, 0]['pred_df'] = pred_df.astype(np.float32)
            self.eval_images['df'][0, 0]['input'] = x.astype(np.float32)
            self.eval_images['df'][1, 0] = df_ssim

            self.eval_images['df'] = self.eval_images['df'][:, self.eval_images['df'][1, :].argsort()]
    
    def set_up_image_sim(self, sim_l2, rec_sim, dec_sim, x):
        sim_l2 = sim_l2.cpu().numpy()
        rec_sim = rec_sim.cpu().numpy()
        dec_sim = dec_sim.cpu().numpy()
        x = x.cpu().numpy()

        if self.eval_images['sim'][1, -1] > sim_l2:

            self.eval_images['sim'][0, -1]['decoder_sm'] = dec_sim.astype(np.float32)
            self.eval_images['sim'][0, -1]['reconstruction_sm'] = rec_sim.astype(np.float32)
            self.eval_images['sim'][0, -1]['input'] = x.astype(np.float32)
            self.eval_images['sim'][1, -1] = sim_l2

            self.eval_images['sim'] = self.eval_images['sim'][:, self.eval_images['sim'][1, :].argsort()]
    
    def set_up_image_rec(self, rec_ssim, reconstructed, x):
        rec_ssim = rec_ssim.cpu().numpy()
        reconstructed = reconstructed.cpu().numpy()
        x = x.cpu().numpy()

        if self.eval_images['rec'][1] < rec_ssim:
            self.eval_images['rec'][0, 0]['reconstruction'] = reconstructed.astype(np.float32)
            self.eval_images['rec'][0, 0]['input'] = x.astype(np.float32)
            self.eval_images['rec'][1, 0] = rec_ssim

    def get_similarity_ready(self, sim):
        view_size = int(self.image_size / self.similarity_downscale)
        sim = sim[0].view(view_size, view_size)[None, None, :, :]
        min_sim = sim.min()
        max_sim = sim.max()
        sim = interpolate(input=sim, scale_factor=self.similarity_downscale, mode='bicubic', antialias=True).squeeze()
        sim = torch.clamp(sim, min_sim, max_sim)
        return sim

    def start_online_evaluation(self, pred, pred_df, reconstructed, x, target, gt_df, rec_sim, dec_sim):

        with torch.no_grad():
            num_classes = pred.shape[1]
            output_softmax = softmax_helper(pred)
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

            seg_dice = ((2 * tp_hard) / (2 * tp_hard + fp_hard + fn_hard + 1e-8))
            for t in range(len(target)):
                current_x = x[t, 0]
                current_reconstructed = reconstructed[t, 0]
                current_rec_sim = rec_sim[t]
                current_dec_sim = dec_sim[t]

                rec_ssim = ssim(current_x.type(torch.float32)[None, None, :, :], current_reconstructed.type(torch.float32)[None, None, :, :])
                self.reconstruction_ssim_list.append(rec_ssim)

                sim_l2 = torch.linalg.norm(current_rec_sim.type(torch.float32) - current_dec_sim.type(torch.float32), ord=2)
                self.sim_l2_list.append(sim_l2)

                if self.directional_field:
                    current_pred_df = pred_df[t]
                    current_gt_df = gt_df[t]
                    df_ssim = ssim(current_gt_df.type(torch.float32)[None, :, :, :], current_pred_df.type(torch.float32)[None, :, :, :])
                    self.df_ssim_list.append(df_ssim)
                    current_gt_df = current_gt_df[0]
                    current_pred_df = current_pred_df[0]
                    self.set_up_image_df(df_ssim=df_ssim, gt_df=current_gt_df, pred_df=current_pred_df, x=current_x)

                current_pred = torch.argmax(pred[t], dim=0)
                current_target = target[t]

                current_rec_sim = self.get_similarity_ready(current_rec_sim)
                current_dec_sim = self.get_similarity_ready(current_dec_sim)

                self.set_up_image_seg(seg_dice=seg_dice[t].mean(), gt=current_target, pred=current_pred, x=current_x)
                self.set_up_image_sim(sim_l2=sim_l2, rec_sim=current_rec_sim, dec_sim=current_dec_sim, x=current_x)
                self.set_up_image_rec(rec_ssim=rec_ssim, reconstructed=current_reconstructed, x=current_x)
                
                #with autocast():

            tp_hard = tp_hard.sum(0, keepdim=False).detach().cpu().numpy()
            fp_hard = fp_hard.sum(0, keepdim=False).detach().cpu().numpy()
            fn_hard = fn_hard.sum(0, keepdim=False).detach().cpu().numpy()

            self.online_eval_foreground_dc.append(list((2 * tp_hard) / (2 * tp_hard + fp_hard + fn_hard + 1e-8)))
            self.online_eval_tp.append(list(tp_hard))
            self.online_eval_fp.append(list(fp_hard))
            self.online_eval_fn.append(list(fn_hard))

    def get_custom_colormap(self, colormap):
        # extract all colors from the .jet map
        cmaplist = [colormap(i) for i in range(colormap.N)]

        cmaplist[0] = (0, 0, 0, 1.0)

        # create the new map
        cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, colormap.N)

        # define the bins and normalize
        bounds = np.linspace(0, 4, 5)
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

        return cmap, norm
        
    def finish_online_evaluation(self):
        self.online_eval_tp = np.sum(self.online_eval_tp, 0)
        self.online_eval_fp = np.sum(self.online_eval_fp, 0)
        self.online_eval_fn = np.sum(self.online_eval_fn, 0)

        global_dc_per_class = [i for i in [2 * i / (2 * i + j + k) for i, j, k in
                                           zip(self.online_eval_tp, self.online_eval_fp, self.online_eval_fn)]
                               if not np.isnan(i)]
        self.all_val_eval_metrics.append(np.mean(global_dc_per_class))

        self.print_to_log_file("(interpret this as an estimate for the Dice of the different classes. This is not exact.)")
        self.print_to_log_file("Average global foreground Dice:", [np.round(i, 4) for i in global_dc_per_class])
        self.print_to_log_file("Average reconstruction ssim:", torch.tensor(self.reconstruction_ssim_list).mean().item())
        self.print_to_log_file("Average similarity L2 distance:", torch.tensor(self.sim_l2_list).mean().item())

        class_dice = {'RV': global_dc_per_class[0], 'MYO': global_dc_per_class[1], 'LV': global_dc_per_class[2]}
        overfit_data = {'Train': torch.tensor(self.train_loss).mean().item(), 'Val': torch.tensor(self.val_loss).mean().item()}
        self.writer.add_scalars('Epoch/Train_vs_val_loss', overfit_data, self.epoch)
        self.writer.add_scalars('Epoch/Class dice', class_dice, self.epoch)
        self.writer.add_scalar('Epoch/Dice', torch.tensor(global_dc_per_class).mean().item(), self.epoch)
        self.writer.add_scalar('Epoch/Reconstruction ssim', torch.tensor(self.reconstruction_ssim_list).mean().item(), self.epoch)
        self.writer.add_scalar('Epoch/Similarity L2 distance', torch.tensor(self.sim_l2_list).mean().item(), self.epoch)

        self.log_rec_images()
        if self.directional_field:
            self.print_to_log_file("Average df ssim:", torch.tensor(self.df_ssim_list).mean().item())
            self.writer.add_scalar('Epoch/Directional field ssim', torch.tensor(self.df_ssim_list).mean().item(), self.epoch)
            self.log_df_images(colormap=cm.viridis)
        self.log_sim_images(colormap=cm.plasma)
        cmap, norm = self.get_custom_colormap(colormap=cm.jet)
        self.log_seg_images(colormap=cmap, norm=norm)

        max_memory_allocated = torch.cuda.max_memory_allocated(device=self.network.get_device())
        self.print_to_log_file("Max GPU Memory allocated:", max_memory_allocated / 10e8, "Gb")
        self.print_to_log_file("Max CPU Memory allocated:", psutil.Process(os.getpid()).memory_info().rss / 10e8, "Gb")

        self.online_eval_foreground_dc = []
        self.online_eval_tp = []
        self.online_eval_fp = []
        self.online_eval_fn = []
        self.reconstruction_ssim_list = []
        self.df_ssim_list = []
        self.sim_l2_list = []
        self.val_loss = []
        self.train_loss = []

        self.eval_images = self.initialize_image_data()

    def run_online_evaluation(self, data, target, output, gt_df):
        """
        due to deep supervision the return value and the reference are now lists of tensors. We only need the full
        resolution output because this is what we are interested in in the end. The others are ignored
        :param output:
        :param target:
        :return:
        """
        pred = output['pred'][0]
        pred_df = output['directional_field']
        reconstructed = output['reconstructed'][0]
        rec_sim = output['reconstruction_sm']
        dec_sim = output['decoder_sm']

        x = data[0]
        target = target[0]
        return self.start_online_evaluation(pred=pred, 
                                            pred_df=pred_df, 
                                            reconstructed=reconstructed, 
                                            x=x, 
                                            target=target, 
                                            gt_df=gt_df, 
                                            rec_sim=rec_sim, 
                                            dec_sim=dec_sim)
        #return super().run_online_evaluation(output, target)

    def validate(self, do_mirroring: bool = True, use_sliding_window: bool = True,
                 step_size: float = 0.5, save_softmax: bool = True, use_gaussian: bool = True, overwrite: bool = True,
                 validation_folder_name: str = 'validation_raw', debug: bool = False, all_in_gpu: bool = False,
                 segmentation_export_kwargs: dict = None, run_postprocessing_on_folds: bool = True):
        """
        We need to wrap this because we need to enforce self.network.do_ds = False for prediction
        """
        ds = self.network.do_ds
        self.network.do_ds = False
        ret = super().validate(do_mirroring=do_mirroring, use_sliding_window=use_sliding_window, step_size=step_size,
                               save_softmax=save_softmax, use_gaussian=use_gaussian,
                               overwrite=overwrite, validation_folder_name=validation_folder_name, debug=debug,
                               all_in_gpu=all_in_gpu, segmentation_export_kwargs=segmentation_export_kwargs,
                               run_postprocessing_on_folds=run_postprocessing_on_folds)

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

    def compute_losses(self, x, output, target, gt_df, do_backprop):
        for t in target:
            assert t.shape[1] == 1

        pred = output['pred']
        reconstructed = output['reconstructed']
        decoder_sm = output['decoder_sm']
        reconstruction_sm = output['reconstruction_sm']
        #predicted_parameters = output['parameters']
        assert pred[0].shape[-2:] == target[0].shape[-2:] == reconstructed[0].shape[-2:] == x[0].shape[-2:]
        assert len(pred) == len(target) == len(reconstructed) == len(x)

        #print(x[0].dtype)
        #print(reconstructed[0].dtype)
        #print(reconstruction_sm.dtype)
        #print(decoder_sm.dtype)
        #print(output['directional_field'].dtype)

        if self.directional_field:
            df_pred = output['directional_field']
            directional_field_loss = self.directional_field_loss(pred=df_pred, y_df=gt_df, y_seg=target[0].squeeze(1), iter_nb=self.iter_nb, do_backprop=do_backprop)
            self.loss_data['directional_field'][1] = directional_field_loss
        
        sim_loss = 0
        rec_loss = 0
        if reconstructed is not None:

            #a = (self.end_similarity - self.start_similarity) / self.total_nb_of_iterations
            #b = self.start_similarity
            #self.current_similarity_weight = a * iter_nb + b
            #self.loss_data['similarity'][0] = self.current_similarity_weight
            rec_loss = self.reconstruction_loss(reconstructed, x)
            self.loss_data['reconstruction'][1] = rec_loss

            assert decoder_sm.shape == reconstruction_sm.shape
            sim_loss = self.similarity_loss(decoder_sm, reconstruction_sm)
            self.loss_data['similarity'][1] = sim_loss
            #if self.dynamic_weight_averaging:
            #    self.epoch_average_losses[self.epoch]['reconstruction'] += reconstruction_loss.item() / nb_iters
            #    self.epoch_average_losses[self.epoch]['similarity'] += similarity_loss.item() / nb_iters

        #seg_target = []
        #for t in range(len(target)):
        #    temp = torch.clone(target[t]).squeeze().long()
        #    temp = torch.nn.functional.one_hot(temp, num_classes=4).permute(0, 3, 1, 2).float()
        #    seg_target.append(temp)
            
        seg_loss = self.segmentation_loss(pred, target)
        self.loss_data['segmentation'][1] = seg_loss

        loss = 0
        for key, value in self.loss_data.items():
            if do_backprop:
                self.writer.add_scalar('Iteration/' + key + ' loss', value[1], self.iter_nb)
            loss += value[0] * value[1]

        if do_backprop:
            self.writer.add_scalars('Iteration/loss weights', {key:value[0] for key, value in self.loss_data.items()}, self.iter_nb)
            self.writer.add_scalar('Iteration/Training loss', loss, self.iter_nb)

        return loss


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

        if self.fp16:
            with autocast():

                #matplotlib.use('QtAgg')
                #plt.imshow(data[0][0, 0].detach().cpu(), cmap='gray')
                #plt.show()

                output = self.network(data[0])

                l = self.compute_losses(x=data, output=output, target=target, gt_df=gt_df, do_backprop=do_backprop)

                if not do_backprop:
                    self.val_loss.append(l.detach().cpu())
                else:
                    self.train_loss.append(l.detach().cpu())
                #del data

                #l = self.loss(output, target)

            if do_backprop:
                self.amp_grad_scaler.scale(l).backward()
                self.amp_grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.amp_grad_scaler.step(self.optimizer)
                self.amp_grad_scaler.update()
        else:
            output = self.network(data[0])
            l = self.compute_losses(x=data, output=output, target=target, gt_df=gt_df, do_backprop=do_backprop)

            if not do_backprop:
                self.val_loss.append(l.detach().cpu())
            else:
                self.train_loss.append(l.detach().cpu())
            #l = self.loss(output, target)

            if do_backprop:
                l.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.optimizer.step()

        if run_online_evaluation:
            self.run_online_evaluation(data=data, target=target, output=output, gt_df=gt_df)
        del data

        del target

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
                self.print_to_log_file("Creating new 5-fold cross-validation split...")
                splits = []
                all_keys_sorted = np.sort(list(self.dataset.keys()))
                kfold = KFold(n_splits=5, shuffle=True, random_state=12345)
                for i, (train_idx, test_idx) in enumerate(kfold.split(all_keys_sorted)):
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

    def setup_DA_params(self):
        """
        - we increase roation angle from [-15, 15] to [-30, 30]
        - scale range is now (0.7, 1.4), was (0.85, 1.25)
        - we don't do elastic deformation anymore

        :return:
        """
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
            else:
                self.optimizer.param_groups[0]['lr'] = poly_lr(ep, self.max_num_epochs, self.initial_lr, 0.9)
            self.writer.add_scalar('Epoch/Learning rate', self.optimizer.param_groups[0]['lr'], self.epoch)
        else:
            ep = epoch
            if not self.config['scheduler'] == 'cosine':
                self.optimizer.param_groups[0]['lr'] = poly_lr(ep, self.max_num_epochs, self.initial_lr, 0.9)

        self.print_to_log_file("lr:", np.round(self.optimizer.param_groups[0]['lr'], decimals=6))

    def on_epoch_end(self):
        """
        overwrite patient-based early stopping. Always run to 1000 epochs
        :return:
        """
        super().on_epoch_end()
        continue_training = self.epoch < self.max_num_epochs

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
        self.network.do_ds = True
        ret = super().run_training()
        self.network.do_ds = ds
        return ret
