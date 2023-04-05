# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import matplotlib
from copy import copy
from math import ceil
from torch.nn import init
import torch
import sys
import torch.nn as nn
import matplotlib.pyplot as plt
from ..lib.encoder import Encoder
from ..lib.utils import MLP, MotionEstimation, DeformableTransformer, ConvBlocks, Filter, ConvBlock, GetSeparability, GetCrossSimilarityMatrix, ReplicateChannels, To_image, From_image, rescale, CCA
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
from ..lib.vit_transformer import TransformerFlowEncoder, ModulationTransformer, TransformerFlowDecoder, SpatioTemporalTransformer, SpatialTransformerLayer, SlotTransformer, TransformerEncoderLayer, SlotAttention, CrossTransformerEncoderLayer, CrossRelativeSpatialTransformerLayer
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


class OpticalFlowModel(SegmentationNetwork):
    def __init__(self,
                deep_supervision,
                out_encoder_dims,
                device,
                in_dims,
                nb_layers,
                video_length,
                image_size,
                num_bottleneck_layers,
                conv_layer,
                conv_depth,
                bottleneck_heads,
                drop_path_rate,
                norm_2d):
        super(OpticalFlowModel, self).__init__()
        
        self.num_stages = (len(conv_depth))
        self.num_bottleneck_layers = num_bottleneck_layers
        self.d_model = out_encoder_dims[-1] * 2
        self.bottleneck_size = [int(image_size / (2**self.num_stages)), int(image_size / (2**self.num_stages))]
        self.image_size = image_size
        self.bottleneck_heads = bottleneck_heads
        self.do_ds = deep_supervision
        self.conv_op=nn.Conv2d
        self.video_length = video_length
        self.nb_layers = nb_layers
        
        self.num_classes = 4

        # stochastic depth
        num_blocks = conv_depth + [num_bottleneck_layers]
        my_iter = torch.cat([torch.tensor([0]), torch.cumsum(torch.tensor(num_blocks), dim=0)])
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(num_blocks))]
        dpr = [dpr[my_iter[j]:my_iter[j+1]] for j in range(len(num_blocks))]
        dpr_encoder = dpr[:self.num_stages]
        dpr_decoder = [x[::-1] for x in dpr_encoder[::-1]]
        dpr_bottleneck = dpr[-1]

        self.encoder = Encoder(conv_layer=conv_layer, norm=norm_2d, out_dims=out_encoder_dims, device=device, in_dims=in_dims, conv_depth=conv_depth, dpr=dpr_encoder)
        in_dims[0] = self.num_classes
        conv_depth_decoder = conv_depth[::-1]
        self.seg_decoder = decoder_alt.SegmentationDecoder2(deep_supervision=deep_supervision, conv_depth=conv_depth_decoder, conv_layer=conv_layer, dpr=dpr_decoder, in_encoder_dims=in_dims[::-1], out_encoder_dims=out_encoder_dims[::-1], num_classes=self.num_classes, img_size=image_size, norm=norm_2d)
        self.flow_decoder = decoder_alt.SegmentationDecoder2(deep_supervision=deep_supervision, conv_depth=conv_depth_decoder, conv_layer=conv_layer, dpr=dpr_decoder, in_encoder_dims=in_dims[::-1], out_encoder_dims=out_encoder_dims[::-1], num_classes=2, img_size=image_size, norm=norm_2d)

        H, W = (int(image_size / 2**(self.num_stages)), int(image_size / 2**(self.num_stages)))
        d_ffn = min(2048, self.d_model * 4)
        
        self.transformer_flow_encoder = TransformerFlowEncoder(dim=self.d_model, nhead=self.bottleneck_heads, num_layers=self.nb_layers, video_length=video_length)
        #self.transformerDecoder = TransformerFlowDecoder(dim=self.d_model, num_layers=self.nb_layers, num_heads=self.bottleneck_heads, d_ffn=d_ffn)

        #self.final_proj_layer = nn.Conv2d(out_encoder_dims[0], self.d_model, kernel_size=1)
        #self.final_proj_layer_heatmap = nn.Conv2d(out_encoder_dims[0], self.d_model, kernel_size=1)
        #self.classification_linear = nn.Linear(self.d_model, self.video_length)

        self.motion_estimation = MotionEstimation()

        if deep_supervision:
            self.output_proj_list = nn.ModuleList()
            for dim in out_encoder_dims:
                hidden_dim = (self.d_model + dim) // 2
                mlp = MLP(input_dim=self.d_model, hidden_dim=hidden_dim, output_dim=dim, num_layers=2)
                self.output_proj_list.append(mlp)
        else:
            dim = out_encoder_dims[0]
            hidden_dim = (self.d_model + dim) // 2
            self.mlp = MLP(input_dim=self.d_model, hidden_dim=hidden_dim, output_dim=dim, num_layers=2)
    

    def dot(self, memory_bus, output_feature_map):
        B, M, C = memory_bus.shape
        B, C, H, W = output_feature_map.shape
        output_feature_map = output_feature_map.view(B, C, H * W)
        output_feature_map = memory_bus @ output_feature_map
        output_feature_map = output_feature_map.view(B, M, H, W)
        return output_feature_map

    def forward(self, labeled, unlabeled):
        out = {'labeled_predictions': [], 
                'unlabeled_predictions': [], 
                'motion_flow_u_to_l': [], 
                'motion_flow_l_to_u': [], 
                'registered_input_u_to_l': [], 
                'registered_input_l_to_u': [], 
                'registered_seg': []}

        #matplotlib.use('QtAgg')
        #print(i)
        #fig, ax = plt.subplots(1, 1)
        #ax.imshow(x[i, 0, 0].detach().cpu(), cmap='gray')
        #plt.show()
        labeled_features, labeled_skip_connections = self.encoder(labeled)
        assert torch.all(torch.isfinite(labeled))

        B, C, H, W = labeled_features.shape

        unlabeled_feature_list = []
        unlabeled_skip_co_list = []
        for t in range(len(unlabeled)):
            unlabeled_features, unlabeled_skip_connections = self.encoder(unlabeled[t])
            unlabeled_skip_co_list.append(unlabeled_skip_connections)
            unlabeled_feature_list.append(unlabeled_features)

        unlabeled_features = torch.stack(unlabeled_feature_list, dim=0) # T, B, C, H, W
        labeled_flow, unlabeled_flow, labeled_tokens, unlabeled_tokens = self.transformer_flow_encoder(labeled_spatial_features=labeled_features, unlabeled_spatial_features=unlabeled_features)

        for t in range(len(unlabeled)):
            flow_pred_unlabeled = self.flow_decoder(unlabeled_flow[t], unlabeled_skip_co_list[t])
            flow_pred_labeled = self.flow_decoder(labeled_flow[t], labeled_skip_connections)
            labeled_pred = self.seg_decoder(labeled_flow[t], labeled_skip_connections)
            unlabeled_pred = self.seg_decoder(unlabeled_flow[t], unlabeled_skip_co_list[t])
        
            #token_list = self.transformerDecoder(spatial_tokens=flow_features) # B, M, C 

            current_labeled_tokens = self.mlp(labeled_tokens[t]) # B, 6, C
            current_unlabeled_tokens = self.mlp(unlabeled_tokens[t]) # B, 6, C

            labeled_seg_tokens = current_labeled_tokens[:, :4]
            labeled_flow_tokens = current_labeled_tokens[:, 4:]

            unlabeled_seg_tokens = current_unlabeled_tokens[:, :4]
            unlabeled_flow_tokens = current_unlabeled_tokens[:, 4:]

            labeled_pred = self.dot(labeled_seg_tokens, labeled_pred)
            unlabeled_pred = self.dot(unlabeled_seg_tokens, unlabeled_pred)
            flow_pred_labeled = self.dot(labeled_flow_tokens, flow_pred_labeled)
            flow_pred_unlabeled = self.dot(unlabeled_flow_tokens, flow_pred_unlabeled)

            registered_input_unlabeled = self.motion_estimation(flow=flow_pred_unlabeled, original=unlabeled[t])
            registered_seg = self.motion_estimation(flow=flow_pred_unlabeled, original=unlabeled_pred)
            registered_input_labeled = self.motion_estimation(flow=flow_pred_labeled, original=labeled)

            out['labeled_predictions'].append(labeled_pred)
            out['unlabeled_predictions'].append(unlabeled_pred)
            out['motion_flow_u_to_l'].append(flow_pred_unlabeled)
            out['motion_flow_l_to_u'].append(flow_pred_labeled)
            out['registered_input_u_to_l'].append(registered_input_unlabeled)
            out['registered_input_l_to_u'].append(registered_input_labeled)
            out['registered_seg'].append(registered_seg)
        
        out['labeled_predictions'] = torch.stack(out['labeled_predictions'], dim=0)
        out['unlabeled_predictions'] = torch.stack(out['unlabeled_predictions'], dim=0)
        out['motion_flow_u_to_l'] = torch.stack(out['motion_flow_u_to_l'], dim=0)
        out['motion_flow_l_to_u'] = torch.stack(out['motion_flow_l_to_u'], dim=0)
        out['registered_input_u_to_l'] = torch.stack(out['registered_input_u_to_l'], dim=0)
        out['registered_input_l_to_u'] = torch.stack(out['registered_input_l_to_u'], dim=0)
        out['registered_seg'] = torch.stack(out['registered_seg'], dim=0)
            
        return out
    

    def predict_3D_flow(self, labeled, unlabeled, processor, do_mirroring: bool, mirror_axes: Tuple[int, ...] = (0, 1, 2),
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
        assert len(labeled.shape) == 4, "data must have shape (c,x,y,z)"

        if mixed_precision:
            context = autocast
        else:
            context = no_op

        context = no_op

        with context():
            with torch.no_grad():
                if use_sliding_window:
                    res = self._internal_predict_3D_2Dconv_tiled_flow(labeled, unlabeled, processor, patch_size, do_mirroring, mirror_axes, step_size,
                                                                                regions_class_order, use_gaussian, pad_border_mode,
                                                                                pad_kwargs, all_in_gpu, False)
                else:
                    res = self._internal_predict_3D_2Dconv(labeled, unlabeled, patch_size, do_mirroring, mirror_axes, regions_class_order,
                                                               pad_border_mode, pad_kwargs, all_in_gpu, False)

        return res
    

    def _internal_maybe_mirror_and_pred_2D_flow(self, labeled, unlabeled, processor, mirror_axes: tuple,
                                           do_mirroring: bool = True,
                                           mult: np.ndarray or torch.tensor = None) -> torch.tensor:
        # if cuda available:
        #   everything in here takes place on the GPU. If x and mult are not yet on GPU this will be taken care of here
        #   we now return a cuda tensor! Not numpy array!

        assert len(unlabeled[0].shape) == 4, 'x must be (b, c, x, y)'
        assert len(labeled.shape) == 4, 'x must be (b, c, x, y)'

        labeled = maybe_to_torch(labeled)
        unlabeled = maybe_to_torch(unlabeled)

        result_torch = torch.zeros([labeled.shape[0], self.num_classes] + list(labeled.shape[2:]), dtype=torch.float)
        result_torch = torch.zeros([video.shape[1], self.num_classes] + list(video.shape[3:]), dtype=torch.float)

        if torch.cuda.is_available():
            video = to_cuda(video, gpu_id=self.get_device())
            result_torch = result_torch.cuda(self.get_device(), non_blocking=True)

        if mult is not None:
            mult = maybe_to_torch(mult)
            if torch.cuda.is_available():
                mult = to_cuda(mult, gpu_id=self.get_device())

        if do_mirroring:
            mirror_idx = 4
            num_results = 2 ** len(mirror_axes)
        else:
            mirror_idx = 1
            num_results = 1

        for m in range(mirror_idx):
            if m == 0:
                with torch.no_grad():
                    network_input, padding_need, _, _ = processor.preprocess_no_registration(data_list=labeled_data)
                output = self(network_input)
                output = processor.uncrop_no_registration(output['predictions'], padding_need)
                pred = self.inference_apply_nonlin(output[idx])
                result_torch += 1 / num_results * pred

            if m == 1 and (1 in mirror_axes):
                video_list = []
                for t in range(len(video)):
                    video_list.append(torch.flip(video[t], (3, )))
                unlabeled = torch.stack(video_list, dim=0)
                with torch.no_grad():
                    network_input, padding_need, _, _ = processor.preprocess_no_registration(data_list=labeled_data)
                output = self(network_input)
                output = processor.uncrop_no_registration(output['predictions'], padding_need)
                pred = self.inference_apply_nonlin(output[idx])
                result_torch += 1 / num_results * torch.flip(pred, (3, ))

            if m == 2 and (0 in mirror_axes):
                labeled_data = []
                for t in range(len(video)):
                    labeled_data.append(torch.flip(video[t], (2, )))
                with torch.no_grad():
                    network_input, padding_need, _, _ = processor.preprocess_no_registration(data_list=labeled_data)
                output = self(network_input)
                output = processor.uncrop_no_registration(output['predictions'], padding_need)
                pred = self.inference_apply_nonlin(output[idx])
                result_torch += 1 / num_results * torch.flip(pred, (2, ))

            if m == 3 and (0 in mirror_axes) and (1 in mirror_axes):
                labeled_data = []
                for t in range(len(video)):
                    labeled_data.append(torch.flip(video[t], (3, 2)))
                with torch.no_grad():
                    network_input, padding_need, _, _ = processor.preprocess_no_registration(data_list=labeled_data)
                output = self(network_input)
                output = processor.uncrop_no_registration(output['predictions'], padding_need)
                pred = self.inference_apply_nonlin(output[idx])
                result_torch += 1 / num_results * torch.flip(pred, (3, 2))

        if mult is not None:
            result_torch[:, :] *= mult

        return result_torch

    def _internal_predict_3D_2Dconv_tiled_flow(self, labeled, unlabeled, processor, patch_size: Tuple[int, int], do_mirroring: bool,
                                                mirror_axes: tuple = (0, 1), step_size: float = 0.5,
                                                regions_class_order: tuple = None, use_gaussian: bool = False,
                                                pad_border_mode: str = "edge", pad_kwargs: dict =None,
                                                all_in_gpu: bool = False,
                                                verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        if all_in_gpu:
            raise NotImplementedError

        assert len(unlabeled[0].shape) == 4, "data must be c, x, y, z"
        assert len(labeled.shape) == 4, "data must be c, x, y, z"

        predicted_segmentation = []
        softmax_pred = []

        for depth_idx in range(labeled.shape[2]):

            #current_video = [x[:, depth_idx] for x in frame_list]

            pred_seg, softmax_pres = self._internal_predict_2D_2Dconv_tiled_flow(labeled[:, depth_idx],
                unlabeled[:, :, depth_idx], processor, step_size, do_mirroring, mirror_axes, patch_size, regions_class_order, use_gaussian,
                pad_border_mode, pad_kwargs, all_in_gpu, verbose)

            predicted_segmentation.append(pred_seg[None])
            softmax_pred.append(softmax_pres[None])

        predicted_segmentation = np.vstack(predicted_segmentation)
        softmax_pred = np.vstack(softmax_pred).transpose((1, 0, 2, 3))

        return predicted_segmentation, softmax_pred


    def _internal_predict_2D_2Dconv_tiled_flow(self, labeled, unlabeled, processor, step_size: float, do_mirroring: bool, mirror_axes: tuple,
                                          patch_size: tuple, regions_class_order: tuple, use_gaussian: bool,
                                          pad_border_mode: str, pad_kwargs: dict, all_in_gpu: bool,
                                          verbose: bool) -> Tuple[np.ndarray, np.ndarray]:
        # better safe than sorry
        assert len(unlabeled[0].shape) == 3, "x must be (c, x, y)"
        assert len(labeled.shape) == 3, "x must be (c, x, y)"

        if verbose: print("step_size:", step_size)
        if verbose: print("do mirror:", do_mirroring)

        assert patch_size is not None, "patch_size cannot be None for tiled prediction"

        # for sliding window inference the image must at least be as large as the patch size. It does not matter
        # whether the shape is divisible by 2**num_pool as long as the patch size is
        labeled_data, labeled_slicer = pad_nd_image(labeled, patch_size, pad_border_mode, pad_kwargs, True, None)
        unlabeled_data, _ = pad_nd_image(unlabeled, patch_size, pad_border_mode, pad_kwargs, True, None)

        labeled_shape = labeled_data.shape  # still c, x, y

        # compute the steps for sliding window
        steps = self._compute_steps_for_sliding_window(patch_size, labeled_shape[1:], step_size)
        num_tiles = len(steps[0]) * len(steps[1])

        if verbose:
            print("data shape:", labeled_shape)
            print("patch size:", patch_size)
            print("steps (x, y, and z):", steps)
            print("number of tiles:", num_tiles)

        # we only need to compute that once. It can take a while to compute this due to the large sigma in
        # gaussian_filter
        if use_gaussian and num_tiles > 1:
            if self._gaussian_2d is None or not all(
                    [i == j for i, j in zip(patch_size, self._patch_size_for_gaussian_2d)]):
                if verbose: print('computing Gaussian')
                gaussian_importance_map = self._get_gaussian(patch_size, sigma_scale=1. / 8)

                self._gaussian_2d = gaussian_importance_map
                self._patch_size_for_gaussian_2d = patch_size
            else:
                if verbose: print("using precomputed Gaussian")
                gaussian_importance_map = self._gaussian_2d

            gaussian_importance_map = torch.from_numpy(gaussian_importance_map)
            if torch.cuda.is_available():
                gaussian_importance_map = gaussian_importance_map.cuda(self.get_device(), non_blocking=True)

        else:
            gaussian_importance_map = None

        if all_in_gpu:
            # If we run the inference in GPU only (meaning all tensors are allocated on the GPU, this reduces
            # CPU-GPU communication but required more GPU memory) we need to preallocate a few things on GPU

            if use_gaussian and num_tiles > 1:
                # half precision for the outputs should be good enough. If the outputs here are half, the
                # gaussian_importance_map should be as well
                gaussian_importance_map = gaussian_importance_map.half()

                # make sure we did not round anything to 0
                gaussian_importance_map[gaussian_importance_map == 0] = gaussian_importance_map[
                    gaussian_importance_map != 0].min()

                add_for_nb_of_preds = gaussian_importance_map
            else:
                add_for_nb_of_preds = torch.ones(patch_size, device=self.get_device())

            if verbose: print("initializing result array (on GPU)")
            aggregated_results = torch.zeros([self.num_classes] + list(labeled_data.shape[1:]), dtype=torch.half,
                                             device=self.get_device())

            if verbose: print("moving data to GPU")
            labeled_data = torch.from_numpy(labeled_data).cuda(self.get_device(), non_blocking=True)
            unlabeled_data = torch.from_numpy(unlabeled_data).cuda(self.get_device(), non_blocking=True)

            if verbose: print("initializing result_numsamples (on GPU)")
            aggregated_nb_of_predictions = torch.zeros([self.num_classes] + list(labeled_data.shape[1:]), dtype=torch.half,
                                                       device=self.get_device())
        else:
            if use_gaussian and num_tiles > 1:
                add_for_nb_of_preds = self._gaussian_2d
            else:
                add_for_nb_of_preds = np.ones(patch_size, dtype=np.float32)
            aggregated_results = np.zeros([self.num_classes] + list(labeled_data.shape[1:]), dtype=np.float32)
            aggregated_nb_of_predictions = np.zeros([self.num_classes] + list(labeled_data.shape[1:]), dtype=np.float32)

        for x in steps[0]:
            lb_x = x
            ub_x = x + patch_size[0]
            for y in steps[1]:
                lb_y = y
                ub_y = y + patch_size[1]

                predicted_patch = self._internal_maybe_mirror_and_pred_2D_flow(labeled_data[None, :, lb_x:ub_x, lb_y:ub_y],
                    unlabeled_data[:, None, :, lb_x:ub_x, lb_y:ub_y], processor, mirror_axes, do_mirroring,
                    gaussian_importance_map)[0]

                if all_in_gpu:
                    predicted_patch = predicted_patch.half()
                else:
                    predicted_patch = predicted_patch.cpu().numpy()

                aggregated_results[:, lb_x:ub_x, lb_y:ub_y] += predicted_patch
                aggregated_nb_of_predictions[:, lb_x:ub_x, lb_y:ub_y] += add_for_nb_of_preds

        # we reverse the padding here (remeber that we padded the input to be at least as large as the patch size
        labeled_slicer = tuple(
            [slice(0, aggregated_results.shape[i]) for i in
             range(len(aggregated_results.shape) - (len(labeled_slicer) - 1))] + labeled_slicer[1:])
        aggregated_results = aggregated_results[labeled_slicer]
        aggregated_nb_of_predictions = aggregated_nb_of_predictions[labeled_slicer]

        # computing the class_probabilities by dividing the aggregated result with result_numsamples
        class_probabilities = aggregated_results / aggregated_nb_of_predictions

        if regions_class_order is None:
            predicted_segmentation = class_probabilities.argmax(0)
        else:
            if all_in_gpu:
                class_probabilities_here = class_probabilities.detach().cpu().numpy()
            else:
                class_probabilities_here = class_probabilities
            predicted_segmentation = np.zeros(class_probabilities_here.shape[1:], dtype=np.float32)
            for i, c in enumerate(regions_class_order):
                predicted_segmentation[class_probabilities_here[i] > 0.5] = c

        if all_in_gpu:
            if verbose: print("copying results to CPU")

            if regions_class_order is None:
                predicted_segmentation = predicted_segmentation.detach().cpu().numpy()

            class_probabilities = class_probabilities.detach().cpu().numpy()

        if verbose: print("prediction done")
        return predicted_segmentation, class_probabilities