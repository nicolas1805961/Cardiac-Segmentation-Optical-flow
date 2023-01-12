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
from ..lib.utils import ConvBlocks, Filter, ConvBlock, GetSeparability, GetCrossSimilarityMatrix, ReplicateChannels, To_image, From_image, rescale, CCA
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
from ..lib.vit_transformer import SpatioTemporalTransformer, SpatialTransformerLayer, TransformerEncoderLayer, SlotAttention, TemporalTransformerLayer, CrossTransformerEncoderLayer, CrossRelativeSpatialTransformerLayer
from batchgenerators.augmentations.utils import pad_nd_image
from ..training.dataloading.dataset_loading import get_idx, select_idx
from ..lib.position_embedding import PositionEmbeddingSine2d
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


class VideoModel(SegmentationNetwork):
    def __init__(self,
                window_size,
                deep_supervision,
                out_encoder_dims,
                use_conv_mlp,
                device,
                similarity_down_scale,
                nb_memory_bus,
                concat_spatial_cross_attention,
                spatial_cross_attention_num_heads,
                log_function,
                in_dims,
                conv_layer_1d,
                merge_temporal_tokens,
                use_patches,
                nb_layers,
                video_length,
                proj_qkv,
                nb_zones,
                area_size,
                learn_indices,
                softmax_indices,
                image_size,
                num_bottleneck_layers,
                conv_layer,
                deformable_points,
                conv_depth,
                num_heads,
                filter_skip_co_segmentation,
                bottleneck_heads,
                drop_path_rate,
                norm_1d,
                norm_2d):
        super(VideoModel, self).__init__()
        
        self.num_stages = (len(conv_depth))
        self.num_bottleneck_layers = num_bottleneck_layers
        self.d_model = out_encoder_dims[-1] * 2
        self.bottleneck_size = [int(image_size / (2**self.num_stages)), int(image_size / (2**self.num_stages))]
        self.image_size = image_size
        self.bottleneck_heads = bottleneck_heads
        self.do_ds = deep_supervision
        self.conv_op=nn.Conv2d
        self.percent = None
        self.log_function = log_function
        self.nb_memory_bus = nb_memory_bus
        self.video_length = video_length
        self.nb_layers = nb_layers
        self.learn_indices = learn_indices
        self.softmax_indices = softmax_indices
        self.merge_temporal_tokens = merge_temporal_tokens
        
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
        self.decoder = decoder_alt.VideoSegmentationDecoder(deformable_points=deformable_points, use_patches=use_patches, nb_zones=nb_zones, area_size=area_size, video_length=video_length, nb_memory_bus=nb_memory_bus, norm_1d=norm_1d, conv_layer_1d=conv_layer_1d, learn_indices=learn_indices, softmax_indices=softmax_indices, conv_layer=conv_layer, norm=norm_2d, similarity_down_scale=similarity_down_scale, filter_skip_co_segmentation=filter_skip_co_segmentation, concat_spatial_cross_attention=concat_spatial_cross_attention, spatial_cross_attention_num_heads=spatial_cross_attention_num_heads[::-1], shortcut=False, proj_qkv=proj_qkv, out_encoder_dims=out_encoder_dims[::-1], use_conv_mlp=use_conv_mlp, last_activation='identity', img_size=image_size, num_classes=self.num_classes, device=device, swin_abs_pos=False, in_encoder_dims=in_dims[::-1], merge=False, conv_depth=conv_depth_decoder, transformer_depth=0, dpr=dpr_decoder, rpe_mode=False, rpe_contextual_tensor=False, num_heads=num_heads, window_size=window_size, drop_path_rate=drop_path_rate, deep_supervision=self.do_ds)
        self.pos = PositionEmbeddingSine2d(num_pos_feats=self.d_model // 2, normalize=True)

        self.extra_bottleneck_block_1 = conv_layer(in_dim=self.d_model, out_dim=self.d_model, nb_blocks=1, dpr=dpr_bottleneck, norm=norm_2d, kernel_size=3)
        self.extra_bottleneck_block_2 = conv_layer(in_dim=self.d_model, out_dim=self.d_model, nb_blocks=1, dpr=dpr_bottleneck, norm=norm_2d, kernel_size=3)

        H, W = (int(image_size / 2**(self.num_stages)), int(image_size / 2**(self.num_stages)))

        temporal_transformer_layer = TemporalTransformerLayer(d_model=self.d_model, nhead=self.bottleneck_heads, dim_feedforward=self.d_model * 4)
        spatial_transformer_layer = TransformerEncoderLayer(d_model=self.d_model, nhead=self.bottleneck_heads, dim_feedforward=self.d_model * 4)
        slot_layer = SlotAttention(dim=self.d_model, iters=4, hidden_dim=self.d_model)
        self.spatio_temporal_encoder = SpatioTemporalTransformer(conv_layer=None, dim=self.d_model, slot_layer=slot_layer, temporal_layer=temporal_transformer_layer, spatial_layer=spatial_transformer_layer, num_layers=self.nb_layers, nb_memory_bus=self.nb_memory_bus, norm_1d=norm_1d, video_length=self.video_length, conv_layer_1d=conv_layer_1d, area_size=area_size, use_patches=False)
        
        self.slots_mu = nn.Parameter(torch.randn(1, 1, self.d_model))
        self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, self.d_model))
        init.xavier_uniform_(self.slots_logsigma)
        #self.query_embed = nn.Embedding(4, self.d_model)

        self.transformer_decoder = decoder_alt.TransformerVideoDecoder(dim=self.d_model, 
                                                                        num_heads=bottleneck_heads, 
                                                                        deformable_points=deformable_points, 
                                                                        video_length=video_length, 
                                                                        num_stages=self.num_stages)
        self.proj_layers = nn.ModuleList()
        for i in reversed(range(self.num_stages)):
            proj = nn.Linear(out_encoder_dims[i], self.d_model)
            self.proj_layers.append(proj)

        #ffn_layers = []
        #for i in range(self.num_stages):
        #    ffn_layers.append(nn.Linear(self.d_model // 2**i, self.d_model // 2**(i+1)))
        #    if i < self.num_stages - 1:
        #        ffn_layers.append(nn.GELU())
        #self.ffn = nn.Sequential(*ffn_layers)
        #self.ffn = nn.Linear(self.d_model, self.d_model)
        #self.out_proj = nn.Conv2d(in_channels=out_encoder_dims[0] * 2, out_channels=self.d_model, kernel_size=1)
    
    def get_mask(self, softmax_volume):
        "softmax_volume: B, T, C, H, W"
        B, T, C, H, W = softmax_volume.shape
        k = int((H / (2**self.num_stages)) ** 2)
        scale_factors = [1/2**i for i in range(self.num_stages)]
        scale_list = []
        for scale_factor in scale_factors:
            temp_blurred = 1 - torch.max(softmax_volume, dim=2)[0]
            temp_blurred = gaussian_blur(temp_blurred, kernel_size=[19, 19])
            temp_blurred = interpolate(temp_blurred, scale_factor=(scale_factor, scale_factor), mode='bilinear', antialias=True)
            B, T, H, W = temp_blurred.shape

            matplotlib.use('QtAgg')
            fig, ax = plt.subplots(1, 1)
            ax.imshow(temp_blurred[0, 0].cpu(), cmap='plasma')
            plt.show()

            temp_blurred_flattened = torch.flatten(temp_blurred, start_dim=-2)
            values, indices = torch.topk(temp_blurred_flattened, k=k, dim=-1, largest=True)
            #mask = torch.zeros_like(temp_blurred_flattened)
            #mask.scatter_(dim=-1, index=indices, src=torch.ones_like(mask))
            #mask = mask.view(B, T, H, W).unsqueeze(2)
            scale_list.append(indices)

            
        #matplotlib.use('QtAgg')
        #fig, ax = plt.subplots(1, 3)
        #ax[0].imshow(scale_list[0][0, 0, 0].cpu(), cmap='plasma')
        #ax[1].imshow(scale_list[1][0, 0, 0].cpu(), cmap='plasma')
        #ax[2].imshow(scale_list[2][0, 0, 0].cpu(), cmap='plasma')
        #plt.show()

        return scale_list

    def dot(self, memory_bus, output_feature_map):
        T, B, C, H, W = output_feature_map.shape
        output_feature_map = output_feature_map.view(T, B, C, H * W).view(T * B, C, H * W)
        if self.merge_temporal_tokens:
            memory_bus = torch.mean(memory_bus.view(T, B, self.nb_memory_bus, self.d_model), dim=0, keepdim=True)
            memory_bus = memory_bus.repeat(T, 1, 1, 1).view(T * B, self.nb_memory_bus, self.d_model)
        #memory_bus = self.ffn(memory_bus)
        output_feature_map = memory_bus @ output_feature_map
        output_feature_map = output_feature_map.view(T, B, self.nb_memory_bus, H * W).view(T, B, self.nb_memory_bus, H, W)
        return output_feature_map

    def forward(self, x):
        theta = None
        out = {}
        encoded_list = []
        skip_co_list = []
        out_list = []
        for i in range(len(x)):
            encoded, skip_connections = self.encoder(x[i])
            assert torch.all(torch.isfinite(encoded))

            encoded = self.extra_bottleneck_block_1(encoded)
            encoded_list.append(encoded)
            skip_co_list.append(skip_connections)
        
        spatial_tokens = torch.stack(encoded_list, dim=0)
        T, B, C, H, W = spatial_tokens.shape

        mu = self.slots_mu.expand(T * B, self.nb_memory_bus, -1)
        sigma = self.slots_logsigma.exp().expand(T * B, self.nb_memory_bus, -1)
        memory_bus = mu + sigma * torch.randn(mu.shape, device=spatial_tokens.device)

        spatial_tokens, memory_bus, advanced_pos = self.spatio_temporal_encoder(spatial_tokens, memory_bus) # memory_bus = T*B, M, C 
        memory_bus = memory_bus.view(T, B, self.nb_memory_bus, C)
        memory_bus = memory_bus.permute(1, 0, 2, 3).contiguous().view(B, T * self.nb_memory_bus, C)

        if self.softmax_indices or self.learn_indices:
            stage_list = []
            for stage in range(self.num_stages):
                video_list = [skip_co_list[i][stage] for i in range(self.video_length)]
                video_volume = torch.stack(video_list, dim=0)
                stage_list.append(video_volume)
            skip_co_list = stage_list

        sampling_point_list = []
        attention_weight_list = []
        out_level_list = []
        for i in range(self.video_length):
            decoded = self.extra_bottleneck_block_2(spatial_tokens[i])
            if self.softmax_indices:
                prediction_list, _ = self.decoder(decoded, skip_co_list, softmax_volume=data_dict['mask_data'], frame_index=i)
            elif self.learn_indices:
                prediction_list, sampling_points, attention_weights = self.decoder(decoded, skip_co_list, frame_index=i)
                out_level_list.append(prediction_list)
                #seg, theta = self.decoder(decoded, skip_co_list, frame_index=i)
            seg = prediction_list[-1]
            #seg = self.out_proj(seg)
            out_list.append(seg)
            sampling_point_list.append(sampling_points)
            attention_weight_list.append(attention_weights)
        
        s_list = []
        for i, proj_layer in enumerate(self.proj_layers):
            v_list = []
            for j in range(self.video_length):
                v_list.append(out_level_list[j][i])
            v_list = torch.stack(v_list, dim=0)
            v_list = v_list.permute(0, 1, 3, 4, 2)
            v_list = proj_layer(v_list)
            v_list = v_list.permute(0, 1, 4, 2, 3)
            s_list.append(v_list)

        attention_weights = torch.stack(attention_weight_list, dim=0)
        sampling_points = torch.stack(sampling_point_list, dim=0)
        output_feature_map = torch.stack(out_list, dim=0)

        memory_bus = self.transformer_decoder(memory_bus=memory_bus, skip_connection_list=s_list, advanced_pos=advanced_pos) # B, T*M, C
        memory_bus = memory_bus.view(B, T, self.nb_memory_bus, C)
        memory_bus = memory_bus.permute(1, 0, 2, 3).contiguous().view(T * B, self.nb_memory_bus, C)
        output_feature_map = self.dot(memory_bus, s_list[-1])
            
        out['attention_weights'] = attention_weights
        out['sampling_points'] = sampling_points
        out['predictions'] = output_feature_map
        #out['theta'] = theta
            
        return out
    

    def predict_3D_video(self, data, idx, video_padding, processor, do_mirroring: bool, mirror_axes: Tuple[int, ...] = (0, 1, 2),
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

        assert len(data[0].shape) == 4, "data must have shape (c,x,y,z)"

        if mixed_precision:
            context = autocast
        else:
            context = no_op

        with context():
            with torch.no_grad():
                if use_sliding_window:
                    res = self._internal_predict_3D_2Dconv_tiled_video(data, idx, video_padding, processor, patch_size, do_mirroring, mirror_axes, step_size,
                                                                                regions_class_order, use_gaussian, pad_border_mode,
                                                                                pad_kwargs, all_in_gpu, False)
                else:
                    res = self._internal_predict_3D_2Dconv(data, idx, video_padding, patch_size, do_mirroring, mirror_axes, regions_class_order,
                                                               pad_border_mode, pad_kwargs, all_in_gpu, False)

        return res


    def _internal_maybe_mirror_and_pred_2D_middle(self, x: Union[np.ndarray, torch.tensor], middle: Union[np.ndarray, torch.tensor],
                                                    mirror_axes: tuple,
                                                    do_mirroring: bool = True,
                                                    mult: np.ndarray or torch.tensor = None) -> torch.tensor:
        # if cuda available:
        #   everything in here takes place on the GPU. If x and mult are not yet on GPU this will be taken care of here
        #   we now return a cuda tensor! Not numpy array!

        assert len(x.shape) == 4, 'x must be (b, c, x, y)'

        x = maybe_to_torch(x)
        middle = maybe_to_torch(middle)
        result_torch = torch.zeros([x.shape[0], self.num_classes] + list(x.shape[2:]), dtype=torch.float)

        if torch.cuda.is_available():
            x = to_cuda(x, gpu_id=self.get_device())
            middle = to_cuda(middle, gpu_id=self.get_device())
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

            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(1, 2)
            #print(x.shape)
            #ax[0].imshow(x.cpu()[0, 0], cmap='gray')
            #ax[1].imshow(middle.cpu()[0, 0], cmap='gray')
            #plt.show()

            if m == 0:
                network_input = {'l1': x,
                                'l2': middle,
                                'u1': None,
                                'u2': None}
                pred = self.inference_apply_nonlin(self(network_input)['pred_l_1'])
                result_torch += 1 / num_results * pred

            if m == 1 and (1 in mirror_axes):
                network_input = {'l1': torch.flip(x, (3,)),
                                'l2': torch.flip(middle, (3,)),
                                'u1': None,
                                'u2': None}
                pred = self.inference_apply_nonlin(self(network_input)['pred_l_1'])
                result_torch += 1 / num_results * torch.flip(pred, (3, ))

            if m == 2 and (0 in mirror_axes):
                network_input = {'l1': torch.flip(x, (2,)),
                                'l2': torch.flip(middle, (2,)),
                                'u1': None,
                                'u2': None}
                pred = self.inference_apply_nonlin(self(network_input)['pred_l_1'])
                result_torch += 1 / num_results * torch.flip(pred, (2, ))

            if m == 3 and (0 in mirror_axes) and (1 in mirror_axes):
                network_input = {'l1': torch.flip(x, (3, 2)),
                                'l2': torch.flip(middle, (3, 2)),
                                'u1': None,
                                'u2': None}
                pred = self.inference_apply_nonlin(self(network_input)['pred_l_1'])
                result_torch += 1 / num_results * torch.flip(pred, (3, 2))

        if mult is not None:
            result_torch[:, :] *= mult

        return result_torch


    def _internal_maybe_mirror_and_pred_2D_middle_unlabeled(self, l1: Union[np.ndarray, torch.tensor], l2: Union[np.ndarray, torch.tensor], u1: Union[np.ndarray, torch.tensor],
                                                    mirror_axes: tuple,
                                                    do_mirroring: bool = True,
                                                    mult: np.ndarray or torch.tensor = None) -> torch.tensor:
        # if cuda available:
        #   everything in here takes place on the GPU. If x and mult are not yet on GPU this will be taken care of here
        #   we now return a cuda tensor! Not numpy array!

        assert len(l1.shape) == 4, 'x must be (b, c, x, y)'
        assert len(l2.shape) == 4, 'x must be (b, c, x, y)'
        assert len(u1.shape) == 4, 'x must be (b, c, x, y)'

        l1 = maybe_to_torch(l1)
        l2 = maybe_to_torch(l2)
        u1 = maybe_to_torch(u1)
        result_torch = torch.zeros([l1.shape[0], self.num_classes] + list(l1.shape[2:]), dtype=torch.float)

        if torch.cuda.is_available():
            l1 = to_cuda(l1, gpu_id=self.get_device())
            l2 = to_cuda(l2, gpu_id=self.get_device())
            u1 = to_cuda(u1, gpu_id=self.get_device())
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
            

            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(1, 2)
            #print(x.shape)
            #ax[0].imshow(x.cpu()[0, 0], cmap='gray')
            #ax[1].imshow(middle.cpu()[0, 0], cmap='gray')
            #plt.show()

            if m == 0:
                network_input = {'l1': l1,
                                'l2': l2,
                                'u1': u1}
                pred = self.inference_apply_nonlin(self(network_input)['pred_l_1'])
                result_torch += 1 / num_results * pred

            if m == 1 and (1 in mirror_axes):
                network_input = {'l1': torch.flip(l1, (3, )),
                                'l2': torch.flip(l2, (3, )),
                                'u1': torch.flip(u1, (3, ))}
                pred = self.inference_apply_nonlin(self(network_input)['pred_l_1'])
                result_torch += 1 / num_results * torch.flip(pred, (3, ))

            if m == 2 and (0 in mirror_axes):
                network_input = {'l1': torch.flip(l1, (2, )),
                                'l2': torch.flip(l2, (2, )),
                                'u1': torch.flip(u1, (2, ))}
                pred = self.inference_apply_nonlin(self(network_input)['pred_l_1'])
                result_torch += 1 / num_results * torch.flip(pred, (2, ))

            if m == 3 and (0 in mirror_axes) and (1 in mirror_axes):
                network_input = {'l1': torch.flip(l1, (3, 2)),
                                'l2': torch.flip(l2, (3, 2)),
                                'u1': torch.flip(u1, (3, 2))}
                pred = self.inference_apply_nonlin(self(network_input)['pred_l_1'])
                result_torch += 1 / num_results * torch.flip(pred, (3, 2))

        if mult is not None:
            result_torch[:, :] *= mult

        return result_torch
    

    def _internal_maybe_mirror_and_pred_2D_video(self, video, idx, video_padding, processor, mirror_axes: tuple,
                                           do_mirroring: bool = True,
                                           mult: np.ndarray or torch.tensor = None) -> torch.tensor:
        # if cuda available:
        #   everything in here takes place on the GPU. If x and mult are not yet on GPU this will be taken care of here
        #   we now return a cuda tensor! Not numpy array!

        assert len(video[0].shape) == 4, 'x must be (b, c, x, y)'

        video = maybe_to_torch(video)
        video_padding = maybe_to_torch(video_padding).bool()

        result_torch = torch.zeros([video.shape[1], self.num_classes] + list(video.shape[3:]), dtype=torch.float)

        if torch.cuda.is_available():
            video_padding = to_cuda(video_padding, gpu_id=self.get_device())
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
                labeled_data = [video[i] for i in range(len(video))]
                with torch.no_grad():
                    network_input, padding_need, _, _ = processor.preprocess_no_registration(data_list=labeled_data, video_padding=video_padding)
                output = self(network_input)
                output = processor.uncrop_no_registration(output, padding_need)
                pred = self.inference_apply_nonlin(output['predictions'][idx])
                result_torch += 1 / num_results * pred

            if m == 1 and (1 in mirror_axes):
                labeled_data = []
                for t in range(len(video)):
                    labeled_data.append(torch.flip(video[t], (3, )))
                with torch.no_grad():
                    network_input, padding_need, _, _ = processor.preprocess_no_registration(data_list=labeled_data, video_padding=video_padding)
                output = self(network_input)
                output = processor.uncrop_no_registration(output, padding_need)
                pred = self.inference_apply_nonlin(output['predictions'][idx])
                result_torch += 1 / num_results * torch.flip(pred, (3, ))

            if m == 2 and (0 in mirror_axes):
                labeled_data = []
                for t in range(len(video)):
                    labeled_data.append(torch.flip(video[t], (2, )))
                with torch.no_grad():
                    network_input, padding_need, _, _ = processor.preprocess_no_registration(data_list=labeled_data, video_padding=video_padding)
                output = self(network_input)
                output = processor.uncrop_no_registration(output, padding_need)
                pred = self.inference_apply_nonlin(output['predictions'][idx])
                result_torch += 1 / num_results * torch.flip(pred, (2, ))

            if m == 3 and (0 in mirror_axes) and (1 in mirror_axes):
                labeled_data = []
                for t in range(len(video)):
                    labeled_data.append(torch.flip(video[t], (3, 2)))
                with torch.no_grad():
                    network_input, padding_need, _, _ = processor.preprocess_no_registration(data_list=labeled_data, video_padding=video_padding)
                output = self(network_input)
                output = processor.uncrop_no_registration(output, padding_need)
                pred = self.inference_apply_nonlin(output['predictions'][idx])
                result_torch += 1 / num_results * torch.flip(pred, (3, 2))

        if mult is not None:
            result_torch[:, :] *= mult

        return result_torch

    def _internal_predict_3D_2Dconv_tiled_video(self, data, idx, video_padding, processor, patch_size: Tuple[int, int], do_mirroring: bool,
                                                mirror_axes: tuple = (0, 1), step_size: float = 0.5,
                                                regions_class_order: tuple = None, use_gaussian: bool = False,
                                                pad_border_mode: str = "edge", pad_kwargs: dict =None,
                                                all_in_gpu: bool = False,
                                                verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        if all_in_gpu:
            raise NotImplementedError

        assert len(data[0].shape) == 4, "data must be c, x, y, z"

        predicted_segmentation = []
        softmax_pred = []

        for depth_idx in range(data.shape[2]):

            #current_video = [x[:, depth_idx] for x in frame_list]

            pred_seg, softmax_pres = self._internal_predict_2D_2Dconv_tiled_video(
                data[:, :, depth_idx], idx, video_padding, processor, step_size, do_mirroring, mirror_axes, patch_size, regions_class_order, use_gaussian,
                pad_border_mode, pad_kwargs, all_in_gpu, verbose)

            predicted_segmentation.append(pred_seg[None])
            softmax_pred.append(softmax_pres[None])

        predicted_segmentation = np.vstack(predicted_segmentation)
        softmax_pred = np.vstack(softmax_pred).transpose((1, 0, 2, 3))

        return predicted_segmentation, softmax_pred


    def _internal_predict_2D_2Dconv_tiled_video(self, video, idx, video_padding, processor, step_size: float, do_mirroring: bool, mirror_axes: tuple,
                                          patch_size: tuple, regions_class_order: tuple, use_gaussian: bool,
                                          pad_border_mode: str, pad_kwargs: dict, all_in_gpu: bool,
                                          verbose: bool) -> Tuple[np.ndarray, np.ndarray]:
        # better safe than sorry
        assert len(video[0].shape) == 3, "x must be (c, x, y)"

        if verbose: print("step_size:", step_size)
        if verbose: print("do mirror:", do_mirroring)

        assert patch_size is not None, "patch_size cannot be None for tiled prediction"

        # for sliding window inference the image must at least be as large as the patch size. It does not matter
        # whether the shape is divisible by 2**num_pool as long as the patch size is
        data, slicer = pad_nd_image(video, patch_size, pad_border_mode, pad_kwargs, True, None)
        slicer = slicer[1:]
        
        #data_list = []
        #slicer_list = []
        #for x in video:
        #    data, slicer = pad_nd_image(x, patch_size, pad_border_mode, pad_kwargs, True, None)
        #    data_list.append(data)
        #    slicer_list.append(slicer)

        data_shape = data[0].shape  # still c, x, y

        # compute the steps for sliding window
        steps = self._compute_steps_for_sliding_window(patch_size, data_shape[1:], step_size)
        num_tiles = len(steps[0]) * len(steps[1])

        if verbose:
            print("data shape:", data_shape)
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
            aggregated_results = torch.zeros([self.num_classes] + list(data[0].shape[1:]), dtype=torch.half,
                                             device=self.get_device())

            if verbose: print("moving data to GPU")
            data = torch.from_numpy(data).cuda(self.get_device(), non_blocking=True)

            if verbose: print("initializing result_numsamples (on GPU)")
            aggregated_nb_of_predictions = torch.zeros([self.num_classes] + list(data[0].shape[1:]), dtype=torch.half,
                                                       device=self.get_device())
        else:
            if use_gaussian and num_tiles > 1:
                add_for_nb_of_preds = self._gaussian_2d
            else:
                add_for_nb_of_preds = np.ones(patch_size, dtype=np.float32)
            aggregated_results = np.zeros([self.num_classes] + list(data[0].shape[1:]), dtype=np.float32)
            aggregated_nb_of_predictions = np.zeros([self.num_classes] + list(data[0].shape[1:]), dtype=np.float32)

        for x in steps[0]:
            lb_x = x
            ub_x = x + patch_size[0]
            for y in steps[1]:
                lb_y = y
                ub_y = y + patch_size[1]

                predicted_patch = self._internal_maybe_mirror_and_pred_2D_video(
                    data[:, None, :, lb_x:ub_x, lb_y:ub_y], idx, video_padding, processor, mirror_axes, do_mirroring,
                    gaussian_importance_map)[0]

                if all_in_gpu:
                    predicted_patch = predicted_patch.half()
                else:
                    predicted_patch = predicted_patch.cpu().numpy()

                aggregated_results[:, lb_x:ub_x, lb_y:ub_y] += predicted_patch
                aggregated_nb_of_predictions[:, lb_x:ub_x, lb_y:ub_y] += add_for_nb_of_preds

        # we reverse the padding here (remeber that we padded the input to be at least as large as the patch size
        slicer = tuple(
            [slice(0, aggregated_results.shape[i]) for i in
             range(len(aggregated_results.shape) - (len(slicer) - 1))] + slicer[1:])
        aggregated_results = aggregated_results[slicer]
        aggregated_nb_of_predictions = aggregated_nb_of_predictions[slicer]

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