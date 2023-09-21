# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

from tqdm import tqdm
from scipy.ndimage import distance_transform_edt
from monai.transforms import NormalizeIntensity
import matplotlib
from copy import copy
from math import ceil
from torch.nn import init
import torch
import sys
import torch.nn as nn
import matplotlib.pyplot as plt
from ..lib.encoder import Encoder, Encoder1D, Encoder3D, Encoder2D
from ..lib.utils import MLP, MotionEstimation, DeformableTransformer, ConvBlocks2D, Filter, ConvBlock, GetSeparability, GetCrossSimilarityMatrix, ReplicateChannels, To_image, From_image, rescale, CCA
from ..lib import swin_transformer_2
from ..lib import decoder_alt
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from nnunet.network_architecture.neural_network import SegmentationNetwork
from nnunet.utilities.to_torch import to_cuda, maybe_to_torch
import numpy as np
from typing import Union
from ..lib import swin_cross_attention
from ..lib.vq_vae import VectorQuantizer, VectorQuantizerEMA, VanillaVAE, Quantize
from torch.cuda.amp import autocast
from nnunet.utilities.random_stuff import no_op
from typing import Union, Tuple
from ..lib.vit_transformer import TransformerFlowSegEncoderMutualSimple, TransformerFlowSegEncoderIterative, TransformerFlowSegEncoder2, TransformerLocalMotion, Interpolator, TransformerFlowEncoderFirstSeparate, TransformerFlowEncoderAllOnlyContext, TransformerFlowEncoderFirst, TransformerFlowEncoderAllSeparate
from batchgenerators.augmentations.utils import pad_nd_image
from ..training.dataloading.dataset_loading import get_idx, select_idx
from ..lib.position_embedding import PositionEmbeddingSine2d, PositionEmbeddingSine1d
from torchvision.transforms.functional import gaussian_blur
from torch.nn.functional import interpolate

class ModelWrap(SegmentationNetwork):
    def __init__(self, model1, model2, do_ds):
        super().__init__()
        self.model1 = model1
        self.model2 = model2
        self.conv_op=nn.Conv2d
        self.num_classes = 4

        self._do_ds = do_ds
    
    @property
    def do_ds(self):
        return self.model1.do_ds

    @do_ds.setter
    def do_ds(self, value):
        self.model1.do_ds = value

    def forward(self, x, model_nb=1):
        #if not self.training:
        #    return self.model1(x)

        if model_nb == 1:
            return self.model1(x)
        elif model_nb == 2:
            return self.model2(x)
    
    def _internal_maybe_mirror_and_pred_2D(self, x: Union[np.ndarray, torch.tensor], mirror_axes: tuple,
                                           do_mirroring: bool = True,
                                           mult: np.ndarray or torch.tensor = None) -> torch.tensor:
        return self.model1._internal_maybe_mirror_and_pred_2D(x, mirror_axes, do_mirroring, mult)


class OpticalFlowModelSimple(SegmentationNetwork):
    def __init__(self,
                deep_supervision,
                out_encoder_dims,
                in_dims,
                nb_layers,
                image_size,
                conv_depth,
                bottleneck_heads,
                drop_path_rate,
                log_function,
                dot_multiplier,
                inference_mode,
                nb_iters,
                padding,
                split,
                one_to_all,
                all_to_all,
                only_first):
        super(OpticalFlowModelSimple, self).__init__()
        self.num_stages = (len(conv_depth))
        self.d_model = out_encoder_dims[-1] * 2
        self.bottleneck_size = [int(image_size / (2**self.num_stages)), int(image_size / (2**self.num_stages))]
        self.image_size = image_size
        self.bottleneck_heads = bottleneck_heads
        self.do_ds = deep_supervision
        self.conv_op=nn.Conv2d
        self.nb_layers = nb_layers
        self.log_function = log_function
        #self.regularization_loss_weight = nn.Parameter(torch.tensor([-30.0], requires_grad=True)) # -6.8
        self.inference_mode = inference_mode
        self.one_to_all = one_to_all
        self.all_to_all = all_to_all
        self.split = split
        self.only_first = only_first
        
        self.num_classes = 4

        # stochastic depth
        num_blocks = conv_depth + [nb_layers]
        my_iter = torch.cat([torch.tensor([0]), torch.cumsum(torch.tensor(num_blocks), dim=0)])
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(num_blocks))]
        dpr = [dpr[my_iter[j]:my_iter[j+1]] for j in range(len(num_blocks))]
        dpr_encoder = dpr[:self.num_stages]
        dpr_decoder = [x[::-1] for x in dpr_encoder[::-1]]
        dpr_bottleneck = dpr[-1]

        self.motion_estimation = MotionEstimation()
        
        #encoder_in_dims = in_dims[:]
        #encoder_in_dims[0] = 2
        #self.context_encoder = Encoder2D(out_dims=out_encoder_dims, in_dims=encoder_in_dims, conv_depth=conv_depth)

        self.encoder = Encoder2D(out_dims=out_encoder_dims, in_dims=in_dims, conv_depth=conv_depth)

        decoder_in_dims = in_dims[:]
        decoder_in_dims[0] = self.num_classes
        conv_depth_decoder = conv_depth[::-1]

        self.flow_decoder = decoder_alt.Decoder2D(dot_multiplier=dot_multiplier, deep_supervision=deep_supervision, conv_depth=conv_depth_decoder, in_encoder_dims=decoder_in_dims[::-1], out_encoder_dims=out_encoder_dims[::-1], num_classes=2, img_size=image_size)
        #self.flow_decoder = decoder_alt.FlowDecoder3DInterp(dot_multiplier=dot_multiplier, deep_supervision=deep_supervision, conv_depth=conv_depth_decoder, dpr=dpr_decoder, in_encoder_dims=decoder_in_dims[::-1], out_encoder_dims=out_encoder_dims[::-1], num_classes=2, img_size=image_size, nb_interp_frame=nb_interp_frame)
        #self.seg_decoder = decoder_alt.FlowDecoder3D(dot_multiplier=dot_multiplier, deep_supervision=deep_supervision, conv_depth=conv_depth_decoder, dpr=dpr_decoder, in_encoder_dims=decoder_in_dims[::-1], out_encoder_dims=out_encoder_dims[::-1], num_classes=self.num_classes, img_size=image_size)

        H, W = (int(image_size / 2**(self.num_stages)), int(image_size / 2**(self.num_stages)))
        #d_ffn = min(2048, self.d_model * 4)
        #self.transformer_flow_encoder = TransformerFlowEncoderFirst(dim=self.d_model, nhead=self.bottleneck_heads, num_layers=self.nb_layers, padding='border')
        #self.flow_seg_encoder = TransformerFlowSegEncoder(dim=self.d_model, nhead=self.bottleneck_heads, num_layers=self.nb_layers)
        self.flow_seg_encoder = TransformerFlowSegEncoderMutualSimple(dim=self.d_model, nhead=self.bottleneck_heads, num_layers=self.nb_layers)
        #self.transformer_flow_encoder = TransformerFlowEncoderAllOnlyContext(dim=self.d_model, nhead=self.bottleneck_heads, num_layers=self.nb_layers, padding='border')
        
        self.skip_co_reduction_list = nn.ModuleList()
        for idx, dim in enumerate(out_encoder_dims):
            reduction = nn.Conv2d(in_channels=dim * 2, out_channels=dim, kernel_size=1)
            self.skip_co_reduction_list.append(reduction)

        #self.cat_reduction = nn.Conv2d(in_channels=self.d_model * 2, out_channels=self.d_model, kernel_size=1)

        #self.interpolator = Interpolator(dim=self.d_model, nb_interpolated_frame=nb_interp_frame)
    

    def dot(self, memory_bus, output_feature_map):
        B, M, C = memory_bus.shape
        B, L, C = output_feature_map.shape
        output_feature_map = output_feature_map.permute(0, 2, 1).contiguous() # B, C, L
        output_feature_map = memory_bus @ output_feature_map # B, M, L
        return output_feature_map

    def organize_deep_supervision(self, outputs):
        all_scale_list = []
        for i in range(self.num_stages):
            one_scale_list = []
            for t in range(len(outputs)):
                one_scale_list.append(outputs[t][i])
            one_scale_list = torch.stack(one_scale_list, dim=0)
            all_scale_list.append(one_scale_list)
        return all_scale_list
    


    #def forward(self, unlabeled):
    #    out = {'forward_flow': []}
    #    
    #    #if self.training and self.blackout and not unlabeled.requires_grad:
    #    #    nb_masked = np.rint(np.random.uniform(0, 0.2) * len(unlabeled)).astype(int)
    #    #    indices = np.random.randint(0, len(unlabeled), nb_masked)
    #    #    unlabeled[indices] = 0
    #        #for idx in indices:
    #        #    cropped_unlabeled[idx, j] = torch.zeros_like(cropped_unlabeled[idx, j])
#
    #    #matplotlib.use('QtAgg')
    #    #fig, ax = plt.subplots(1, len(unlabeled))
    #    #for t in range(len(unlabeled)):
    #    #    ax[t].imshow(unlabeled[t, 0, 0].cpu(), cmap='gray')
    #    #plt.show()
#
    #    unlabeled_feature_list = []
    #    unlabeled_skip_co_list = []
#
    #    #matplotlib.use('QtAgg')
    #    #fig, ax = plt.subplots(1, len(unlabeled))
    #    #for u in range(len(unlabeled)):
    #    #    ax[u].imshow(unlabeled[u, 0, 0].cpu(), cmap='gray')
    #    #plt.show()
    #    
    #    for t in range(len(unlabeled)):
    #        unlabeled_features, unlabeled_skip_connections = self.encoder(unlabeled[t])
    #        unlabeled_skip_co_list.append(unlabeled_skip_connections)
    #        unlabeled_feature_list.append(unlabeled_features)
    #    unlabeled_features = torch.stack(unlabeled_feature_list, dim=0) # T, B, C, H, W
#
    #    pos_1d = torch.flatten(unlabeled_features, start_dim=-2).mean(-1) # T, B, C
    #    pos_1d = pos_1d.permute(1, 2, 0).contiguous()
    #    #pos_1d = self.conv_1d(pos_1d)
    #    pos_1d = torch.zeros_like(pos_1d)
#
    #    T, B, C, H, W = unlabeled_features.shape
#
    #    unlabeled_features, _ = self.transformer_flow_encoder(spatial_features=unlabeled_features) # T, B, H*W, C
#
    #    for t in range(len(unlabeled_features)):
#
    #        flow_skip_co = []
    #        for s, reduction_layer in enumerate(self.skip_co_reduction_list):
    #            concatenated_forward = torch.cat([unlabeled_skip_co_list[0][s], unlabeled_skip_co_list[t][s]], dim=1)
    #            concatenated_forward = reduction_layer(concatenated_forward)
    #            flow_skip_co.append(concatenated_forward)
#
    #        to_decode_feature = unlabeled_features[t] # B, L, C
    #        to_decode_feature = to_decode_feature.permute(0, 2, 1).contiguous().view(B, C, H, W)
#
    #        forward_flow = self.flow_decoder(to_decode_feature, flow_skip_co)
    #        out['forward_flow'].append(forward_flow)
#
    #    for k in out.keys():
    #        out[k] = self.organize_deep_supervision(out[k])
    #        if not self.do_ds:
    #            out[k] = out[k][0]
#
    #    #middle = torch.stack([seg_1[1:], seg_2[:-1]], dim=0).mean(0)
    #    #seg = torch.cat([seg_1[0][None], middle, seg_2[-1][None]], dim=0)
#
    #    return out
    

    
    def forward(self, unlabeled):
        out = {'forward_flow': []}

        #matplotlib.use('QtAgg')
        #fig, ax = plt.subplots(1, unlabeled.shape[2])
        #for t in range(unlabeled.shape[2]):
        #    ax[t].imshow(unlabeled[0, 0, t].cpu(), cmap='gray')
        #plt.show()

        #matplotlib.use('QtAgg')
        #fig, ax = plt.subplots(1, len(unlabeled))
        #for u in range(len(unlabeled)):
        #    ax[u].imshow(unlabeled[u, 0, 0].cpu(), cmap='gray')
        #plt.show()

        #context_feature_list = []
        #context_skip_co_list = []
#
        #for t in range(len(unlabeled)):
        #    x1 = unlabeled[0]
        #    x2 = unlabeled[t]
        #    to_encode = torch.cat([x1, x2], dim=1)
        #    context_features, context_skip_connections =self.context_encoder(to_encode)
        #    context_skip_co_list.append(context_skip_connections)
        #    context_feature_list.append(context_features)
        #context_features = torch.stack(context_feature_list, dim=0) # T, B, C, H, W
        
        unlabeled_feature_list = []
        unlabeled_skip_co_list = []
        
        for t in range(len(unlabeled)):
            unlabeled_features, unlabeled_skip_connections = self.encoder(unlabeled[t])
            unlabeled_skip_co_list.append(unlabeled_skip_connections)
            unlabeled_feature_list.append(unlabeled_features)
        unlabeled_features = torch.stack(unlabeled_feature_list, dim=0) # T, B, C, H, W

        x2 = self.flow_seg_encoder(unlabeled_features) # T, B, C, H, W
        B, C, H, W = x2.shape

        flow_skip_co_forward = []
        for s, reduction_layer in enumerate(self.skip_co_reduction_list):
            concatenated_forward = torch.cat([unlabeled_skip_co_list[0][s], unlabeled_skip_co_list[1][s]], dim=1)
            concatenated_forward = reduction_layer(concatenated_forward)
            flow_skip_co_forward.append(concatenated_forward)

        forward_flow = self.flow_decoder(x2, flow_skip_co_forward)
        out['forward_flow'].append(forward_flow)

        out['forward_flow'] = self.organize_deep_supervision(out['forward_flow'])

        for k in out.keys():
            if not self.do_ds:
                out[k] = out[k][0]

        return out

    

    def predict_3D_flow(self, unlabeled, target, target_mask, processor, do_mirroring: bool, mirror_axes: Tuple[int, ...] = (0, 1, 2),
                   use_sliding_window: bool = False,
                   step_size: float = 0.5, patch_size: Tuple[int, ...] = None, regions_class_order: Tuple[int, ...] = None,
                   use_gaussian: bool = False, pad_border_mode: str = "constant",
                   pad_kwargs: dict = None, all_in_gpu: bool = False,
                   verbose: bool = True, mixed_precision: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Use this function to predict a 3D image. It does not matter whether the network is a 2D or 3D U-Net, it will
        detect that automatically and run the appropriate code.

        When running predictions, you need to specify whether you want to run fully convolutional of sliding window
        based inference. We very strongly recommend you use sliding window with the default settings.

        It is the responsibility of the user to make sure the network is in the proper mode (eval for inference!). If
        the network is not in eval mode it will print a warning.

        :param x: Your input data. Must be a nd.ndarray of shape (c, x, y, z).
        :param do_mirroring: If True, use test time data augmentation in the form of mirroring
        :param mirror_axes: Determines which axes to use for mirroing. Per default, mirroring is done along all three
        axes
        :param use_sliding_window: if True, run sliding window prediction. Heavily recommended! This is also the default
        :param step_size: When running sliding window prediction, the step size determines the distance between adjacent
        predictions. The smaller the step size, the denser the predictions (and the longer it takes!). Step size is given
        as a fraction of the patch_size. 0.5 is the default and means that wen advance by patch_size * 0.5 between
        predictions. step_size cannot be larger than 1!
        :param patch_size: The patch size that was used for training the network. Do not use different patch sizes here,
        this will either crash or give potentially less accurate segmentations
        :param regions_class_order: Fabian only
        :param use_gaussian: (Only applies to sliding window prediction) If True, uses a Gaussian importance weighting
         to weigh predictions closer to the center of the current patch higher than those at the borders. The reason
         behind this is that the segmentation accuracy decreases towards the borders. Default (and recommended): True
        :param pad_border_mode: leave this alone
        :param pad_kwargs: leave this alone
        :param all_in_gpu: experimental. You probably want to leave this as is it
        :param verbose: Do you want a wall of text? If yes then set this to True
        :param mixed_precision: if True, will run inference in mixed precision with autocast()
        :return:
        """

        # P, T, 1, D, H, W
        torch.cuda.empty_cache()

        assert step_size <= 1, 'step_size must be smaller than 1. Otherwise there will be a gap between consecutive ' \
                               'predictions'

        #if verbose: print("debug: mirroring", do_mirroring, "mirror_axes", mirror_axes)
        if verbose: self.log_function("debug: mirroring", do_mirroring, "mirror_axes", mirror_axes)

        if pad_kwargs is None:
            pad_kwargs = {'constant_values': 0}

        # A very long time ago the mirror axes were (2, 3, 4) for a 3d network. This is just to intercept any old
        # code that uses this convention
        if len(mirror_axes):
            if self.conv_op == nn.Conv2d:
                if max(mirror_axes) > 1:
                    raise ValueError("mirror axes. duh")
            if self.conv_op == nn.Conv3d:
                if max(mirror_axes) > 2:
                    raise ValueError("mirror axes. duh")

        if self.training:
            self.log_function("WARNING! Network is in train mode during inference. This may be intended, or not...")
            #print('WARNING! Network is in train mode during inference. This may be intended, or not...')

        assert len(unlabeled[0].shape) == 4, "data must have shape (c,x,y,z)"

        if mixed_precision:
            context = autocast
        else:
            context = no_op

        context = no_op

        with context():
            with torch.no_grad():
                if use_sliding_window:
                    res = self._internal_predict_3D_2Dconv_tiled_flow(unlabeled, target, target_mask, processor, patch_size, do_mirroring, mirror_axes, step_size,
                                                                                regions_class_order, use_gaussian, pad_border_mode,
                                                                                pad_kwargs, all_in_gpu, False)
                else:
                    res = self._internal_predict_3D_2Dconv(unlabeled, patch_size, do_mirroring, mirror_axes, regions_class_order,
                                                               pad_border_mode, pad_kwargs, all_in_gpu, False)

        return res
    



    def _internal_maybe_mirror_and_pred_2D(self, unlabeled, target, processor, mirror_axes: tuple,
                                           do_mirroring: bool = True) -> torch.tensor:
        # if cuda available:
        #   everything in here takes place on the GPU. If x and mult are not yet on GPU this will be taken care of here
        #   we now return a cuda tensor! Not numpy array!

        assert len(unlabeled[0].shape) == 4, 'x must be (b, c, x, y)'
        unlabeled = maybe_to_torch(unlabeled)
        target = maybe_to_torch(target)

        result_torch_seg = torch.zeros(list(unlabeled.shape[:2]) + [self.num_classes] + [processor.crop_size, processor.crop_size], dtype=torch.float)
        result_torch_flow = torch.zeros(list(unlabeled.shape[:2]) + [2] + [processor.crop_size, processor.crop_size], dtype=torch.float)
        if torch.cuda.is_available():
            unlabeled = to_cuda(unlabeled, gpu_id=self.get_device())
            #target = to_cuda(target, gpu_id=self.get_device())
            result_torch_seg = result_torch_seg.cuda(self.get_device(), non_blocking=True)
            result_torch_flow = result_torch_flow.cuda(self.get_device(), non_blocking=True)

        #if mult is not None:
        #    mult = maybe_to_torch(mult)
        #    if torch.cuda.is_available():
        #        mult = to_cuda(mult, gpu_id=self.get_device())

        if do_mirroring:
            mirror_idx = 4
            num_results = 2 ** len(mirror_axes)
        else:
            mirror_idx = 1
            num_results = 1

        with torch.no_grad():
            mean_centroid, _ = processor.preprocess_no_registration(data=unlabeled[:, 0]) # T, C(1), H, W

            cropped_unlabeled, padding_need = processor.crop_and_pad(data=unlabeled[:, 0], mean_centroid=mean_centroid)
            cropped_target, _ = processor.crop_and_pad(data=target[:, 0], mean_centroid=mean_centroid)
            padding_need = padding_need[None]
            cropped_unlabeled = cropped_unlabeled[:, None]
            cropped_target = cropped_target[:, None]
        
        cropped_unlabeled[:, 0] = NormalizeIntensity(nonzero=True)(cropped_unlabeled[:, 0])

        #cropped_unlabeled = cropped_unlabeled.permute(1, 2, 0, 3, 4).contiguous()

        #print(cropped_unlabeled.shape)
        #matplotlib.use('QtAgg')
        #fig, ax = plt.subplots(1, 1)
        #ax.imshow(cropped_unlabeled[0, 0, 0].cpu(), cmap='gray')
        #plt.show()
        #plt.waitforbuttonpress()
        #plt.close(fig)

        for m in range(mirror_idx):
            if m == 0:
                output = self(cropped_unlabeled)
                flow_pred = output['forward_flow'][-1]

                #step = int(len(flow_pred) / len(target))
                #flow_pred = flow_pred[::step]
                #assert len(flow_pred) == len(target)

                #seg_pred = self.inference_apply_nonlin(seg_pred)

                #matplotlib.use('QtAgg')
                #fig, ax = plt.subplots(1, 2)
                #ax[0].imshow(cropped_unlabeled[0, 0, 0].cpu(), cmap='gray')
                #ax[1].imshow(torch.argmax(seg_pred[0, 0], dim=0).cpu(), cmap='gray')
                #plt.show()
                #plt.waitforbuttonpress()
                #plt.close(fig)

                result_torch_flow = flow_pred
                #result_torch_flow += 1 / num_results * flow_pred
        
        #matplotlib.use('QtAgg')
        #fig, ax = plt.subplots(1, 1)
        #ax.imshow(torch.argmax(result_torch_seg[0, 0], dim=0).cpu(), cmap='gray')
        #plt.show()
        #plt.waitforbuttonpress()
        #plt.close(fig)
        
        return result_torch_flow, cropped_target, padding_need
    



    def _internal_maybe_mirror_and_pred_2D_2(self, unlabeled, target, processor, mirror_axes: tuple,
                                           do_mirroring: bool = True) -> torch.tensor:
        # if cuda available:
        #   everything in here takes place on the GPU. If x and mult are not yet on GPU this will be taken care of here
        #   we now return a cuda tensor! Not numpy array!

        assert len(unlabeled[0].shape) == 4, 'x must be (b, c, x, y)'
        unlabeled = maybe_to_torch(unlabeled)
        if target is not None:
            target = maybe_to_torch(target)

        if torch.cuda.is_available():
            unlabeled = to_cuda(unlabeled, gpu_id=self.get_device())

        if do_mirroring:
            mirror_idx = 4
            num_results = 2 ** len(mirror_axes)
        else:
            mirror_idx = 1
            num_results = 1

        with torch.no_grad():
            mean_centroid, _ = processor.preprocess_no_registration(data=unlabeled[:, 0]) # T, C(1), H, W

            cropped_unlabeled, padding_need = processor.crop_and_pad(data=unlabeled[:, 0], mean_centroid=mean_centroid)
            padding_need = padding_need[None]
            cropped_unlabeled = cropped_unlabeled[:, None]
            if target is not None:
                cropped_target, _ = processor.crop_and_pad(data=target[:, 0], mean_centroid=mean_centroid)
                cropped_target = cropped_target[:, None]
            else:
                cropped_target = None
        
        cropped_unlabeled[:, 0] = NormalizeIntensity(nonzero=True)(cropped_unlabeled[:, 0])

        #cropped_unlabeled = cropped_unlabeled.permute(1, 2, 0, 3, 4).contiguous()

        #print(cropped_unlabeled.shape)
        #matplotlib.use('QtAgg')
        #fig, ax = plt.subplots(1, 1)
        #ax.imshow(cropped_unlabeled[0, 0, 0].cpu(), cmap='gray')
        #plt.show()
        #plt.waitforbuttonpress()
        #plt.close(fig)

        global_motion_list = []
        for i in tqdm(range(1, len(cropped_unlabeled))):
            current_cropped_unlabeled = torch.cat([cropped_unlabeled[0][None], cropped_unlabeled[i][None]], dim=0)

            result_torch_global_motion = torch.zeros([unlabeled.shape[1], 2, processor.crop_size, processor.crop_size], dtype=torch.float)
            if torch.cuda.is_available():
                #target = to_cuda(target, gpu_id=self.get_device())
                result_torch_global_motion = result_torch_global_motion.cuda(self.get_device(), non_blocking=True)

            for m in range(mirror_idx):
                if m == 0:
                    output = self(current_cropped_unlabeled)
                    global_motion = output['forward_flow'][-1]
                    #seg_pred = self.inference_apply_nonlin(seg_pred)

                    #matplotlib.use('QtAgg')
                    #fig, ax = plt.subplots(1, 2)
                    #ax[0].imshow(cropped_unlabeled[0, 0, 0].cpu(), cmap='gray')
                    #ax[1].imshow(torch.argmax(seg_pred[0, 0], dim=0).cpu(), cmap='gray')
                    #plt.show()
                    #plt.waitforbuttonpress()
                    #plt.close(fig)

                    result_torch_global_motion = global_motion
                    #result_torch_flow += 1 / num_results * flow_pred

            global_motion_list.append(result_torch_global_motion)

        result_torch_flow = torch.stack(global_motion_list, dim=0)
        #matplotlib.use('QtAgg')
        #fig, ax = plt.subplots(1, 1)
        #ax.imshow(torch.argmax(result_torch_seg[0, 0], dim=0).cpu(), cmap='gray')
        #plt.show()
        #plt.waitforbuttonpress()
        #plt.close(fig)
        
        return result_torch_flow, cropped_target, padding_need
    

    def _internal_predict_3D_2Dconv_tiled_flow(self, unlabeled, target, target_mask, processor, patch_size: Tuple[int, int], do_mirroring: bool,
                                                mirror_axes: tuple = (0, 1), step_size: float = 0.5,
                                                regions_class_order: tuple = None, use_gaussian: bool = False,
                                                pad_border_mode: str = "edge", pad_kwargs: dict =None,
                                                all_in_gpu: bool = False,
                                                verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        
        # P, T, 1, D, H, W

        if all_in_gpu:
            raise NotImplementedError

        assert len(unlabeled[0].shape) == 4, "data must be c, x, y, z"

        flow_pred_list = []
        registered_pred_list = []

        for depth_idx in range(unlabeled.shape[2]):

            #current_video = [x[:, depth_idx] for x in frame_list]

            flow_pred, registered_pred = self._internal_predict_2D_2Dconv_tiled_flow(
                unlabeled[:, :, depth_idx], target[:, :, depth_idx], target_mask, processor, step_size, do_mirroring, mirror_axes, patch_size, regions_class_order, use_gaussian,
                pad_border_mode, pad_kwargs, all_in_gpu, verbose)

            flow_pred_list.append(flow_pred[None])
            registered_pred_list.append(registered_pred[None])

        flow_pred = np.vstack(flow_pred_list).transpose((1, 2, 0, 3, 4)) # T, C, depth, H, W
        registered_pred = np.vstack(registered_pred_list).transpose((1, 2, 0, 3, 4)) # T_other, C, depth, H, W

        return flow_pred, registered_pred


    def _internal_predict_2D_2Dconv_tiled_flow(self, unlabeled, target, target_mask, processor, step_size: float, do_mirroring: bool, mirror_axes: tuple,
                                          patch_size: tuple, regions_class_order: tuple, use_gaussian: bool,
                                          pad_border_mode: str, pad_kwargs: dict, all_in_gpu: bool,
                                          verbose: bool) -> Tuple[np.ndarray, np.ndarray]:
        # unlabeled = P, T1, 1, H, W
        # target = T2, 1, H, W
        # better safe than sorry
        assert len(unlabeled[0].shape) == 3, "x must be (c, x, y)"

        if verbose: print("step_size:", step_size)
        if verbose: print("do mirror:", do_mirroring)

        assert patch_size is not None, "patch_size cannot be None for tiled prediction"

        # for sliding window inference the image must at least be as large as the patch size. It does not matter
        # whether the shape is divisible by 2**num_pool as long as the patch size is
        unlabeled_data, unlabeled_slicer = pad_nd_image(unlabeled, patch_size, pad_border_mode, pad_kwargs, True, None)
        target_data, target_slicer = pad_nd_image(target, patch_size, pad_border_mode, pad_kwargs, True, None)
        unlabeled_shape = unlabeled_data.shape

        # compute the steps for sliding window
        steps = self._compute_steps_for_sliding_window(patch_size, unlabeled_shape[-2:], step_size)
        num_tiles = len(steps[0]) * len(steps[1])

        if verbose:
            print("data shape:", unlabeled_shape)
            print("patch size:", patch_size)
            print("steps (x, y, and z):", steps)
            print("number of tiles:", num_tiles)

        # we only need to compute that once. It can take a while to compute this due to the large sigma in
        # gaussian_filter
        #if use_gaussian and num_tiles > 1:
        #    if self._gaussian_2d is None or not all(
        #            [i == j for i, j in zip(patch_size, self._patch_size_for_gaussian_2d)]):
        #        if verbose: print('computing Gaussian')
        #        gaussian_importance_map = self._get_gaussian(patch_size, sigma_scale=1. / 8)
#
        #        self._gaussian_2d = gaussian_importance_map
        #        self._patch_size_for_gaussian_2d = patch_size
        #    else:
        #        if verbose: print("using precomputed Gaussian")
        #        gaussian_importance_map = self._gaussian_2d
#
        #    gaussian_importance_map = torch.from_numpy(gaussian_importance_map)
        #    if torch.cuda.is_available():
        #        gaussian_importance_map = gaussian_importance_map.cuda(self.get_device(), non_blocking=True)
#
        #else:
        #    gaussian_importance_map = None

        nb_frames = len(unlabeled)

        if all_in_gpu:
            # If we run the inference in GPU only (meaning all tensors are allocated on the GPU, this reduces
            # CPU-GPU communication but required more GPU memory) we need to preallocate a few things on GPU

            #if use_gaussian and num_tiles > 1:
            #    # half precision for the outputs should be good enough. If the outputs here are half, the
            #    # gaussian_importance_map should be as well
            #    gaussian_importance_map = gaussian_importance_map.half()
#
            #    # make sure we did not round anything to 0
            #    gaussian_importance_map[gaussian_importance_map == 0] = gaussian_importance_map[
            #        gaussian_importance_map != 0].min()
#
            #    add_for_nb_of_preds = gaussian_importance_map
            #else:
            #    add_for_nb_of_preds = torch.ones(patch_size, device=self.get_device())

            if verbose: print("initializing result array (on GPU)")
            aggregated_results_seg = torch.zeros([nb_frames, self.num_classes] + list(unlabeled_data.shape[2:]), dtype=torch.half,
                                             device=self.get_device())
            aggregated_results_flow = torch.zeros([nb_frames, 2] + list(unlabeled_data.shape[2:]), dtype=torch.half,
                                             device=self.get_device())
            aggregated_results_registered = torch.zeros([nb_frames, 1] + list(unlabeled_data.shape[2:]), dtype=torch.half,
                                             device=self.get_device())

            if verbose: print("moving data to GPU")
            unlabeled_data = torch.from_numpy(unlabeled_data).cuda(self.get_device(), non_blocking=True)
            target_data = torch.from_numpy(target_data).cuda(self.get_device(), non_blocking=True)

            #if verbose: print("initializing result_numsamples (on GPU)")
            #aggregated_nb_of_predictions_seg = torch.zeros([nb_frames, self.num_classes] + list(unlabeled_data.shape[2:]), dtype=torch.half,
            #                                           device=self.get_device())
        else:
            #if use_gaussian and num_tiles > 1:
            #    add_for_nb_of_preds = self._gaussian_2d
            #else:
            #    add_for_nb_of_preds = np.ones(patch_size, dtype=np.float32)
            aggregated_results_seg = np.zeros([nb_frames, self.num_classes] + list(unlabeled_data.shape[2:]), dtype=np.float32)
            aggregated_results_flow = np.zeros([nb_frames, 2] + list(unlabeled_data.shape[2:]), dtype=np.float32)
            aggregated_results_registered = np.zeros([nb_frames, 1] + list(unlabeled_data.shape[2:]), dtype=np.float32)
            #aggregated_nb_of_predictions_seg = np.zeros([nb_frames, self.num_classes] + list(unlabeled_data.shape[2:]), dtype=np.float32)

        H, W = unlabeled_data.shape[-2:]
        y1 = int((H / 2) - (patch_size[0] / 2))
        y2 = int((H / 2) + (patch_size[0] / 2))
        x1 = int((W / 2) - (patch_size[1] / 2))
        x2 = int((W / 2) + (patch_size[1] / 2))

        flow, target, padding_need = self._internal_maybe_mirror_and_pred_2D_2(unlabeled_data[:, None, :, y1:y2, x1:x2], target_data[:, None, :, y1:y2, x1:x2], processor, mirror_axes, do_mirroring)

        flow = torch.nn.functional.pad(flow, pad=(0, 0, 0, 0, 0, 0, 0, 0, 1, 0))

        assert nb_frames == len(target) == len(flow)
        #flow = B=1, C=T, T, W, H

        #matplotlib.use('QtAgg')
        #fig, ax = plt.subplots(1, len(seg))
        #for a in range(len(seg)):
        #    current_img = seg[a, 0].cpu()
        #    current_img = torch.softmax(current_img, dim=0)
        #    current_img = torch.argmax(current_img, dim=0)
        #    ax[a].imshow(current_img, cmap='gray')
        #plt.show()
        #plt.waitforbuttonpress()
        #plt.close(fig)

        #print(target.shape)
        #matplotlib.use('QtAgg')
        #fig, ax = plt.subplots(1, 2)
        #ax[0].imshow(target[indices[0].item(), 0, 0].cpu(), cmap='gray')
        #ax[1].imshow(target[indices[1].item(), 0, 0].cpu(), cmap='gray')
        #plt.show()

        if target_mask is None:
            all_motions = self.warp_linear_lib(flow, target)
        else:
            all_motions = self.warp_linear(flow, target, target_mask)

        assert len(all_motions) == nb_frames
        #for t in range(indices[1].item(), indices[0].item(), -1):
        #    current_motion = self.motion_estimation(flow=flow[t], original=current_motion, mode='nearest')

        flow = flow.permute(1, 0, 2, 3, 4).contiguous()
        all_motions = all_motions.permute(1, 0, 2, 3, 4).contiguous()

        flow = processor.uncrop_no_registration(flow, padding_need)[0]
        all_motions = processor.uncrop_no_registration(all_motions, padding_need)[0]

        if all_in_gpu:
            flow = flow.half()
            all_motions = all_motions.half()
        else:
            flow = flow.cpu().numpy()
            all_motions = all_motions.cpu().numpy()
        
        aggregated_results_flow[:, :, y1:y2, x1:x2] += flow
        aggregated_results_registered[:, :, y1:y2, x1:x2] += all_motions

        #for x in steps[0]:
        #    lb_x = x
        #    ub_x = x + patch_size[0]
        #    for y in steps[1]:
        #        lb_y = y
        #        ub_y = y + patch_size[1]
#
        #        seg = self._internal_maybe_mirror_and_pred_2D_seg(unlabeled_data[:, None, :, lb_x:ub_x, lb_y:ub_y], 
        #                                                          processor, mirror_axes, do_mirroring,
        #                                                          gaussian_importance_map)
        #        
        #        seg = seg[:, 0]
#
        #        if all_in_gpu:
        #            seg = seg.half()
        #        else:
        #            seg = seg.cpu().numpy()
#
        #        aggregated_results_seg[:, :, lb_x:ub_x, lb_y:ub_y] += seg
        #        aggregated_nb_of_predictions_seg[:, :, lb_x:ub_x, lb_y:ub_y] += add_for_nb_of_preds

        # we reverse the padding here (remeber that we padded the input to be at least as large as the patch size

        unlabeled_slicer_flow = tuple(
            [slice(0, aggregated_results_flow.shape[i]) for i in
             range(len(aggregated_results_flow.shape) - 2)] + unlabeled_slicer[-2:])
        
        unlabeled_slicer_registered = tuple(
            [slice(0, aggregated_results_registered.shape[i]) for i in
             range(len(aggregated_results_registered.shape) - 2)] + unlabeled_slicer[-2:])
        
        aggregated_results_flow = aggregated_results_flow[unlabeled_slicer_flow]
        aggregated_results_registered = aggregated_results_registered[unlabeled_slicer_registered]
        #aggregated_nb_of_predictions_seg = aggregated_nb_of_predictions_seg[unlabeled_slicer_seg]

        if verbose: print("prediction done")
        return aggregated_results_flow, aggregated_results_registered
    

    def warp_nearest_sequential(self, current_motion, flow, indices):
        current_motion = to_cuda(current_motion, gpu_id=self.get_device())
        for t in range(indices[0].item(), indices[1].item()):
            current_motion = self.motion_estimation(flow=flow[t], original=current_motion, mode='nearest')
        return current_motion
    
    def warp_distance_sequential(self, current_motion, flow, indices):
        """current_motion: B, 1, H, W"""
        current_motion = torch.nn.functional.one_hot(current_motion[:, 0].long(), num_classes=4).permute(0, 3, 1, 2).contiguous().float()

        motion_list_batch = []
        for i in range(current_motion.shape[0]):
            motion_list_channels = []
            for j in range(current_motion.shape[1]):
                current = distance_transform_edt(current_motion[i, j].numpy())
                motion_list_channels.append(current)
            motion_list_channels = np.stack(motion_list_channels, axis=0)
            motion_list_batch.append(motion_list_channels)
        current_motion = np.stack(motion_list_batch, axis=0)

        #matplotlib.use('QtAgg')
        #fig, ax = plt.subplots(1, 4)
        #for i in range(current_motion.shape[1]):
        #    ax[i].imshow(current_motion[0, i], cmap='gray')
        #plt.show()
        #plt.waitforbuttonpress()

        current_motion = torch.from_numpy(current_motion).float()
        current_motion = to_cuda(current_motion, gpu_id=self.get_device())
        for t in range(indices[0].item(), indices[1].item()):
            current_motion = self.motion_estimation(flow=flow[t], original=current_motion, mode='bilinear')
        current_motion = torch.argmax(current_motion, dim=1, keepdim=True)
        return current_motion
    

    def warp_linear(self, flow, target, target_mask):
        ed_index = np.where(target_mask)[0][0]
        current_motion = target[ed_index]
        current_motion = torch.nn.functional.one_hot(current_motion[:, 0].long(), num_classes=4).permute(0, 3, 1, 2).contiguous().float()
        current_motion = to_cuda(current_motion, gpu_id=self.get_device())
        registered_list = []
        for t in range(len(target)):
            current_flow = flow[t]
            registered = self.motion_estimation(flow=current_flow, original=current_motion, mode='bilinear')
            registered_list.append(torch.argmax(registered, dim=1, keepdim=True))
        registered_list = torch.stack(registered_list, dim=0)
        return registered_list
    
    def warp_linear_lib(self, flow, target):
        current_motion = target[0]
        current_motion = torch.nn.functional.one_hot(current_motion[:, 0].long(), num_classes=4).permute(0, 3, 1, 2).contiguous().float()
        current_motion = to_cuda(current_motion, gpu_id=self.get_device())
        registered_list = []
        for t in range(len(target)):
            current_flow = flow[t]
            registered = self.motion_estimation(flow=current_flow, original=current_motion, mode='bilinear')
            registered_list.append(torch.argmax(registered, dim=1, keepdim=True))
        registered_list = torch.stack(registered_list, dim=0)
        return registered_list