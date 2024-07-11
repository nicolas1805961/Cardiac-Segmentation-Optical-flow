# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

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
from nnunet.lib.position_embedding import PositionEmbeddingSine1d
from nnunet.lib.encoder import Encoder, Encoder1D, EncoderMotionAppearance, Encoder2D
from nnunet.lib.utils import KeyProjection, ConvBlocks2DBatch, ConvBlocks2DGroupLegacy, ConvBlock, ConvBlocks3DEmbedding, ConvBlocks2DGroup, ConvBlocks2DEmbedding, GetSeparability, GetCrossSimilarityMatrix, ReplicateChannels, To_image, From_image, rescale, CCA
from nnunet.lib import swin_transformer_2
from nnunet.lib import decoder_alt
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from nnunet.network_architecture.neural_network import SegmentationNetwork
from nnunet.utilities.to_torch import to_cuda, maybe_to_torch
import numpy as np
from typing import Union
from nnunet.lib import sfb
from torch.cuda.amp import autocast
from nnunet.utilities.random_stuff import no_op
from typing import Union, Tuple
from nnunet.lib.vit_transformer import CrossAttentionLayer, TransformerEncoderLayer, SpecialLayer, DeformableTransformer6, DeformableTransformer3, DynamicPositionalEmbedding, TransformerConv, SpatioTemporalTransformerPos, BatchedTokenSelfAttention, TokenSample, TokenSelfAttention, DeformableTransformer, AlignLinear, SpatioTemporalTransformerTwoMemory, SpatioTemporalTransformer, TransformerContext, TransformerFlowEncoderSuccessiveNoEmb, TransformerFlowEncoderFromStartNoEmb, TransformerFlowEncoderFromStart, TransformerFlowEncoderAllDistance, TransformerFlowSegEncoderAggregationDistance, TransformerFlowSegEncoderAggregationDistanceNoEmb, TransformerFlowEncoderAllOnlyContext, TransformerFlowEncoderFirst, TransformerFlowEncoderAllSeparate
from batchgenerators.augmentations.utils import pad_nd_image
from nnunet.network_architecture.integration import VecInt, SpatialTransformer
import math
from nnunet.lib.loss import NCC
from nnunet.network_architecture.convGRU import ConvGRUCell
import cv2 as cv
from nnunet.lib.spacetimeAttention import SpaceTimeAttention
from nnunet.lib.raft import CorrVolume
from nnunet.lib.raft_initial import CorrBlock, BasicUpdateBlock, coords_grid
from nnunet.lib.raft_extractor import BasicEncoder
from nnunet.lib.raft_extractor_seg import segDecoder, BasicEncoderSeg
import argparse
#from nnunet.lib.mamba import VimLayer
from nnunet.lib.position_embedding import PositionEmbeddingSine2d
from nnunet.lib.gma import Attention, GMAUpdateBlock
from nnunet.lib.utils import warp
from nnunet.lib.swin_cross_attention_return import SwinCrossAttention


class SinusoidalPositionEmbeddings(nn.Module):
    """Positional embeddings"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class SegFlowGaussian(SegmentationNetwork):
    def __init__(self,
                deep_supervision,
                out_encoder_dims,
                in_dims,
                nb_layers,
                image_size,
                conv_depth,
                skip_co_depth,
                bottleneck_heads,
                drop_path_rate,
                log_function,
                motion_from_ed,
                cost_volume,
                final_stride,
                inference_mode,
                nb_conv,
                legacy,
                shrink_select,
                bottleneck_type,
                P,
                nb_inputs,
                raft_iters,
                norm,
                one_to_all,
                supervise_iterations,
                skip_co_type,
                marginal,
                timesformer,
                backward_flow,
                no_residual,
                no_skip_co,
                small_memory,
                query_type,
                gaussian,
                d_model,
                logits_input,
                nb_inputs_memory,
                no_label,
                topk,
                pos_1d,
                memory_read,
                residual,
                deformable,
                downsample_conv,
                conv_bottleneck,
                mamba,
                transformer,
                remove_GRU,
                use_context_encoder,
                cat_correlation,
                memory_attn,
                label_pretrained,
                radius,
                warp,
                append_cat,
                stride,
                match_first,
                motion_appearance,
                raft,
                extra_block,
                dim_feedforward,
                memory_length,
                cross_attn_before_corr,
                correlation_value,
                nb_merging_block,
                all_to_all,
                only_first):
        super(SegFlowGaussian, self).__init__()
        self.num_stages = len(conv_depth)
        self.d_model = d_model
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
        self.only_first = only_first
        self.final_stride = final_stride
        self.motion_from_ed = motion_from_ed
        self.topk = topk
        if not self.topk:
            self.topk = None
        self.pos_1d = pos_1d
        self.bottleneck_type = bottleneck_type
        self.shrink_select = shrink_select
        self.P = P
        self.skip_co_type = skip_co_type
        self.deformable = deformable
        self.supervise_iterations = supervise_iterations
        self.marginal = marginal
        self.timesformer = timesformer
        self.gaussian = gaussian
        self.nb_inputs = nb_inputs
        self.nb_inputs_memory = nb_inputs_memory
        self.backward_flow = backward_flow
        self.logits_input = logits_input
        self.no_label = no_label
        self.no_skip_co = no_skip_co
        self.no_residual = no_residual
        self.small_memory = small_memory
        self.extra_block = extra_block
        self.query_type = query_type
        self.nb_merging_block = nb_merging_block
        self.conv_bottleneck = conv_bottleneck
        self.deformable = deformable
        self.memory_length = memory_length
        self.mamba = mamba
        self.raft = raft
        self.skip_co_depth = skip_co_depth
        self.cost_volume = cost_volume
        self.warp = warp
        self.raft_iters = raft_iters
        self.memory_read = memory_read
        self.downsample_conv = downsample_conv
        self.remove_GRU = remove_GRU
        self.use_context_encoder = use_context_encoder
        self.cat_correlation = cat_correlation
        self.correlation_value = correlation_value
        self.memory_attn = memory_attn
        self.cross_attn_before_corr = cross_attn_before_corr
        self.match_first = match_first
        self.append_cat = append_cat
        self.label_pretrained = label_pretrained
        self.dim_feedforward = dim_feedforward
        self.motion_appearance = motion_appearance
        
        self.num_classes = 4
        self.alpha = 0.5

        if skip_co_type == 'no_conv':
            dot_multiplier = 3
        else:
            dot_multiplier = 2

        # stochastic depth
        num_blocks = conv_depth + [nb_layers]
        my_iter = torch.cat([torch.tensor([0]), torch.cumsum(torch.tensor(num_blocks), dim=0)])
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(num_blocks))]
        dpr = [dpr[my_iter[j]:my_iter[j+1]] for j in range(len(num_blocks))]
        dpr_encoder = dpr[:self.num_stages]
        dpr_decoder = [x[::-1] for x in dpr_encoder[::-1]]
        dpr_bottleneck = dpr[-1]

        self.integration = VecInt((image_size, image_size), 7)
        self.motion_estimation = SpatialTransformer((image_size, image_size))

        in_dims_encoder = copy(in_dims)
        in_dims_past = copy(in_dims)
        self.radius = radius
        self.stride = stride

        in_dims_past[0] = 6
        self.memory_encoder = Encoder2D(d_model=self.d_model, out_dims=out_encoder_dims, in_dims=in_dims_past, conv_depth=conv_depth, norm=norm, legacy=legacy, nb_conv=nb_conv, extra_block=extra_block, residual=residual, expand=False, nhead=bottleneck_heads, downsample_conv=self.downsample_conv)
        
        if self.label_pretrained:
            in_dims_context = copy(in_dims_encoder)
            in_dims_context[0] = 4
            self.reduce_after_gru = nn.Conv2d(in_channels=384, out_channels=self.d_model, kernel_size=1)
            self.context_encoder = Encoder2D(d_model=self.d_model, out_dims=out_encoder_dims, in_dims=in_dims_context, conv_depth=conv_depth, norm=norm, legacy=legacy, nb_conv=nb_conv, extra_block=extra_block, residual=residual, expand=False, nhead=bottleneck_heads, downsample_conv=self.downsample_conv)
            self.bottleneck3 = CrossAttentionLayer(dim=self.d_model, nhead=self.bottleneck_heads, num_layers=self.nb_layers, dim_feedforward=dim_feedforward, expand_value=False)
        
        if not self.motion_appearance:
            in_dims_encoder[0] = 1
            self.query_encoder = Encoder2D(d_model=self.d_model, out_dims=out_encoder_dims, in_dims=in_dims_encoder, conv_depth=conv_depth, norm=norm, legacy=legacy, nb_conv=nb_conv, extra_block=extra_block, residual=residual, expand=False, nhead=bottleneck_heads, downsample_conv=self.downsample_conv)
        
            self.out_encoder_dims = out_encoder_dims
            self.skip_co_reduction_list = nn.ModuleList()
            self.cost_volume_encoder_list = nn.ModuleList()
            self.cost_volume_computation_list = nn.ModuleList()
                    
            dim_list = out_encoder_dims
            nb_blocks_list = self.skip_co_depth
            if self.correlation_value:
                dim_list = dim_list + [self.d_model]
                nb_blocks_list = nb_blocks_list + [1]

            for idx, (dim, nb_blocks) in enumerate(zip(dim_list, nb_blocks_list)):
                corr_dim = (2*self.radius[idx] + 1)**2
                if idx < len(out_encoder_dims):
                    if self.skip_co_type != 'past':
                        corrLayer = CorrVolume(radius=self.radius[idx], stride=self.stride[idx])
                        cv_layer = ConvBlocks2DGroupLegacy(in_dim=corr_dim, out_dim=dim, nb_blocks=1, residual=residual)
                        self.cost_volume_computation_list.append(corrLayer)
                    else:
                        cv_layer = nn.Identity()
                        self.cost_volume_computation_list.append(nn.Identity())
                else:
                    cor_planes = len(out_encoder_dims + [self.d_model]) * corr_dim
                    cv_layer = ConvBlocks2DGroupLegacy(in_dim=cor_planes, out_dim=dim, nb_blocks=1, residual=residual)
                    self.cost_volume_computation_list.append(nn.Identity())
                self.cost_volume_encoder_list.append(cv_layer)

                if not (self.memory_read and idx == len(out_encoder_dims)):
                    if self.skip_co_type == 'no_conv':
                        reduction_1 = nn.Identity()
                    elif self.skip_co_type == 'both':
                        input_dim = 3*dim if self.label_pretrained else 2*dim
                        reduction_1 = ConvBlocks2DGroupLegacy(in_dim=input_dim, out_dim=dim, nb_blocks=nb_blocks, residual=residual)
                    else:
                        reduction_1 = ConvBlocks2DGroupLegacy(in_dim=dim, out_dim=dim, nb_blocks=nb_blocks, residual=residual)
                    self.skip_co_reduction_list.append(reduction_1)
        
        else:
            in_dims_encoder[0] = 2
            self.query_encoder = EncoderMotionAppearance(d_model=self.d_model, out_dims=out_encoder_dims, in_dims=in_dims_encoder, conv_depth=conv_depth, norm=norm, legacy=legacy, nb_conv=nb_conv, residual=residual, expand=False, nhead=bottleneck_heads, downsample_conv=self.downsample_conv)
            
            self.out_encoder_dims = out_encoder_dims
            self.skip_co_reduction_list = nn.ModuleList()
            for idx, (dim, nb_blocks) in enumerate(zip(out_encoder_dims, self.skip_co_depth)):
                reduction_1 = ConvBlocks2DGroupLegacy(in_dim=2*dim, out_dim=dim, nb_blocks=nb_blocks, residual=residual)
                self.skip_co_reduction_list.append(reduction_1)

        decoder_in_dims = in_dims[:]
        decoder_in_dims[0] = self.num_classes
        conv_depth_decoder = conv_depth[::-1]

        self.flow_decoder = decoder_alt.Decoder2D(d_model=self.d_model, dot_multiplier=dot_multiplier, deep_supervision=deep_supervision, conv_depth=conv_depth_decoder, in_encoder_dims=decoder_in_dims[::-1], out_encoder_dims=out_encoder_dims[::-1], num_classes=2, img_size=image_size, norm=norm, last_activation='identity', legacy=legacy, nb_conv=nb_conv, residual=residual)

        self.H, self.W = (int(image_size / 2**(self.num_stages)), int(image_size / 2**(self.num_stages)))

        



        #self.block_1_list = nn.ModuleList()
        #self.block_2_list = nn.ModuleList()
        #for idx, (dim, nb_blocks) in enumerate(zip(dim_list, nb_blocks_list)):
        #    input_resolution=(image_size//(2**idx), image_size//(2**idx))
        #    blocks_1 = nn.ModuleList([
        #        SwinCrossAttention(dim=dim,
        #                        input_resolution=input_resolution,
        #                        num_heads=1,
        #                        device='cuda:0',
        #                        window_size=8,
        #                        shift_size=0 if (i % 2 == 0) else 8 // 2) 
        #                        for i in range(2)])
        #    self.block_1_list.append(blocks_1)
        #    
        #    blocks_2 = nn.ModuleList([
        #        SwinCrossAttention(dim=dim,
        #                        input_resolution=input_resolution,
        #                        num_heads=1,
        #                        device='cuda:0',
        #                        window_size=8,
        #                        shift_size=0 if (i % 2 == 0) else 8 // 2) 
        #                        for i in range(2)])
        #    self.block_2_list.append(blocks_2)
#
        #    if not (self.memory_read and idx == len(out_encoder_dims)):
        #        if self.skip_co_type == 'no_conv':
        #            reduction_1 = nn.Identity()
        #        elif self.skip_co_type == 'both':
        #            input_dim = 3*dim if self.label_pretrained else 2*dim
        #            reduction_1 = ConvBlocks2DGroupLegacy(in_dim=input_dim, out_dim=dim, nb_blocks=nb_blocks, residual=residual)
        #        else:
        #            reduction_1 = ConvBlocks2DGroupLegacy(in_dim=dim, out_dim=dim, nb_blocks=nb_blocks, residual=residual)
        #        self.skip_co_reduction_list.append(reduction_1)
                

            
        if not self.remove_GRU:
            self.gru_cell = ConvGRUCell(input_size=(self.H, self.W),
                                            input_dim=self.d_model if not self.label_pretrained else 384,
                                            hidden_dim=self.d_model if not self.label_pretrained else 384,
                                            kernel_size=(3, 3),
                                            bias=True,
                                            dtype=torch.cuda.FloatTensor)
        
        #self.reduce_transformer_1 = nn.Conv2d(in_channels=self.d_model, out_channels=self.d_model//2, kernel_size=1)
        #self.reduce_transformer_2 = nn.Conv2d(in_channels=self.d_model, out_channels=self.d_model//2, kernel_size=1)
        self.reduce_transformer = ConvBlocks2DGroupLegacy(in_dim=self.d_model*2 if not self.label_pretrained else self.d_model*3, out_dim=self.d_model if not self.label_pretrained else 384, nb_blocks=1, residual=residual)
        self.bottleneck1 = CrossAttentionLayer(dim=self.d_model, nhead=self.bottleneck_heads, num_layers=self.nb_layers, dim_feedforward=dim_feedforward, expand_value=False)
        self.bottleneck2 = CrossAttentionLayer(dim=self.d_model, nhead=self.bottleneck_heads, num_layers=self.nb_layers, dim_feedforward=dim_feedforward, expand_value=False)




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
    
    def forward(self, x, label=None, step=1):
        if self.label_pretrained:
            return self.forward_multi_task_flow_deformable_cost_volume_transformer_cat_label(x, label=label, step=step)
        else:
            if self.motion_appearance:
                return self.forward_motion_appearance(x, step=step)
            else:
                #return self.forward_multi_task_flow_deformable_cost_volume_transformer_swin(x, step=step)
                return self.forward_multi_task_flow_deformable_cost_volume_transformer_cat(x, step=step)

    #def forward(self, x, label=None, step=1):
    #    if self.raft:
    #        return self.forward_multi_task_flow_deformable_raft(x, label, step=step)
    #    #elif label is None:
    #    #    return self.forward_multi_task_flow_no_label(x, step=step)
    #    elif self.no_skip_co:
    #        return self.forward_multi_task_flow_no_skip_co(x, label, step=step)
    #    elif self.no_residual:
    #        return self.forward_multi_task_flow_no_residual(x, label, step=step)
    #    elif self.small_memory:
    #        return self.forward_multi_task_flow_small_memory(x, label, step=step)
    #    elif self.deformable:
    #        if self.cost_volume:
    #            if self.cat_correlation:
    #                if self.label_pretrained:
    #                    return self.forward_multi_task_flow_deformable_cost_volume_transformer_cat_label(x, label, step=step)
    #                else:
    #                    return self.forward_multi_task_flow_deformable_cost_volume_transformer_cat(x, label, step=step)
    #            else:
    #                return self.forward_multi_task_flow_deformable_cost_volume_transformer(x, label, step=step)
    #        elif self.use_context_encoder:
    #            return self.forward_multi_task_flow_context(x, label, step=step)
    #        else:
    #            return self.forward_multi_task_flow_cat(x, label, step=step)
    #    elif self.mamba:
    #        return self.forward_multi_task_flow_mamba(x, label, step=step)
    #    else:
    #        return self.forward_multi_task_flow(x, label, step=step)
    

    def get_max_index(self, x1, x2):
        B, C, H, W = x1.shape
        """x1: B, C, H, W
        x2: B, C, H, W"""

        pad_value = self.kernel_size // 2
        x2 = torch.nn.functional.pad(x2, pad=(pad_value, pad_value, pad_value, pad_value, 0, 0, 0, 0))
        x1 = torch.nn.functional.pad(x1, pad=(pad_value, pad_value, pad_value, pad_value, 0, 0, 0, 0))

        out_list = []
        for i in range(pad_value, x1.shape[2] - pad_value):
            for j in range(pad_value, x1.shape[3] - pad_value):
                current_center_pixel = x1[:, :, i, j][:, :, None] # B, C, 1
                current_window = x2[:, :, i - (self.kernel_size // 2): i + (self.kernel_size // 2) + 1, j - (self.kernel_size // 2): j + (self.kernel_size // 2) + 1].contiguous()
                current_window = current_window.view(B, C, self.kernel_size**2)
                affinity = torch.matmul(current_center_pixel.transpose(2, 1), current_window)
                argmax_idx = affinity.max(-1)[1].view(B, 1)

                # Convert argmax index to coordinates relative to the original image
                h_idx = (i - pad_value) + argmax_idx // self.kernel_size - pad_value
                w_idx = (j - pad_value) + argmax_idx % self.kernel_size - pad_value

                idx = torch.cat([h_idx, w_idx], dim=-1) # B, 2
                out_list.append(idx)
        out = torch.stack(out_list, dim=1) # B, HW, 2

        return out
    


    def get_input_query(self, x1, x2, x3, label1, label2, flow_forward, error_forward, registered_forward, flow_backward, error_backward, registered_backward):
        if self.query_type == 'simple':
            if self.nb_inputs == 'small':
                current_input = x3
            elif self.nb_inputs == 'big':
                current_input = torch.cat([x3, label2, flow_forward, error_forward, registered_forward], dim=1)
            else:
                current_input = torch.cat([x3, label2], dim=1)
        elif self.query_type == 'double':
            if self.nb_inputs == 'small':
                current_input = torch.cat([x2, x3], dim=1)
            elif self.nb_inputs == 'big':
                current_input = torch.cat([x2, x3, label2, flow_forward, error_forward, registered_forward], dim=1)
            else:
                current_input = torch.cat([x2, x3, label2], dim=1)
        elif self.query_type == 'triple':
            if self.nb_inputs == 'small':
                current_input = torch.cat([x1, x2, x3], dim=1)
            elif self.nb_inputs == 'big':
                current_input = torch.cat([x1, x2, x3, label1, label2, flow_forward, error_forward, registered_forward, flow_backward, error_backward, registered_backward], dim=1)
            else:
                current_input = torch.cat([x1, x2, x3, label1, label2], dim=1)
        return current_input
    

    def get_input_memory(self, 
                         x1, 
                         x2, 
                         label2, 
                         flow_forward, 
                         error_forward, 
                         registered_forward,
                         flow_backward, 
                         error_backward, 
                         registered_backward):
        if self.nb_inputs_memory == 'small':
            current_input = torch.cat([x1, 
                                       x2, 
                                       label2], dim=1)
        elif self.nb_inputs_memory == 'big':
            current_input = torch.cat([x1, 
                                       x2, 
                                       label2, 
                                       flow_forward, 
                                       error_forward, 
                                       registered_forward,
                                       flow_backward,
                                       error_backward,
                                       registered_backward], dim=1)
        else:
            current_input = torch.cat([x1, 
                                       x2, 
                                       label2, 
                                       flow_forward, 
                                       error_forward, 
                                       registered_forward], dim=1)
        return current_input
    
    

    def transform_label(self, flow, gt):
        registered_gt = self.motion_estimation(flow=flow, original=gt)

        if not self.logits_input:
            #print(flow.mean())
            registered_gt = torch.argmax(registered_gt, dim=1, keepdim=True)
            registered_gt = torch.nn.functional.one_hot(registered_gt[:, 0].long(), num_classes=4).permute(0, 3, 1, 2).contiguous().float()
            softmaxed_label = gt
        else:
            registered_gt = torch.softmax(registered_gt, dim=1)
            softmaxed_label = torch.softmax(gt, dim=1)

        return registered_gt, softmaxed_label
    


    def is_softmaxed(self, tensor, dim=1):
        """Check if a tensor is in softmaxed form."""
        softmax_sum = torch.sum(tensor, dim=dim)
        is_normalized = torch.allclose(softmax_sum, torch.ones_like(softmax_sum))
        return is_normalized
    



    def forward_single_task(self, x, label, step=1):
        out = {'backward_flow': [], 'weights': []}
        T, B, C, H, W = x.shape
        flow = torch.zeros(size=(B, 2, H, W), device=x.device)
        cumulated_backward = torch.zeros(size=(B, 2, H, W), device=x.device)

        past_feature_list = []
        max_indices_list = []
        past_query_list = []
        memory_skip_co_list = []

        registered_backward = self.motion_estimation(flow=cumulated_backward, original=x[0])
        error_backward = x[0] - registered_backward

        current_input = torch.cat([x[0], x[0], error_backward, registered_backward])

        current_input = self.get_input_query(x[0], 
                                       x[0], 
                                       None, 
                                       None, 
                                       None, 
                                       None, 
                                       error_backward, 
                                       registered_backward)
        
        current_feature, skip_co_current = self.query_encoder(current_input) # B, C, H, W
        past_query_list.append(current_feature)

        past_input = self.get_input_memory(x[0], 
                                       x[0], 
                                       None, 
                                       None, 
                                       None, 
                                       None, 
                                       error_backward, 
                                       registered_backward)

        #past_input = torch.cat([x[0], x[0], registered_seg, cumulated_forward, error, registered, error_backward, registered_backward], dim=1)
        past_new, past_skip_co = self.memory_encoder(past_input)
        past_feature_list.append(past_new)
        memory_skip_co_list.append(past_skip_co)

        for t in range(1, T):

            current_input = self.get_input_query(x[t], 
                                           x[t - 1], 
                                           None, 
                                           None, 
                                           None, 
                                           None, 
                                           error_backward, 
                                           registered_backward)
            
            current_feature, skip_co_current = self.query_encoder(current_input) # B, C, H, W

            if self.gaussian:
                max_idx = self.get_max_index(past_query_list[-1], current_feature)
                max_indices_list.append(max_idx)
                max_idx = torch.stack(max_indices_list, dim=0) # T, B, H*W
            else:
                max_idx = None

            past_features = torch.stack(past_feature_list, dim=0) # T, B, C, H, W
            past_query = torch.stack(past_query_list, dim=0) # T, B, C, H, W
            
            if not self.training:
                past_features = torch.cat([past_features[:2], past_features[2:-1:step], past_features[-1][None]], dim=0)
                past_query = torch.cat([past_query[:2], past_query[2:-1:step], past_query[-1][None]], dim=0)

            current_feature, weights = self.bottleneck(query=current_feature, 
                                                       key=past_query, 
                                                       value=past_features, 
                                                       max_idx=max_idx)

            acc_skip_co = []
            for s in range(self.num_stages):
                concatenated = torch.cat([skip_co_current[s], memory_skip_co_list[-1][s], memory_skip_co_list[0][s]], dim=1)
                concatenated = self.skip_co_reduction_list[s](concatenated)
                acc_skip_co.append(concatenated)

            flow, intermediary = self.flow_decoder_backward(current_feature, acc_skip_co)
            cumulated_backward = cumulated_backward + flow
            out['backward_flow'].append(cumulated_backward)

            registered_backward = self.motion_estimation(flow=cumulated_backward, original=x[t])
            error_backward = x[0] - registered_backward

            past_input = self.get_input_memory(x[t], 
                                       x[0], 
                                       None, 
                                       None, 
                                       None, 
                                       None, 
                                       error_backward, 
                                       registered_backward)

            #past_input = torch.cat([x[t], x[0], registered_seg, cumulated_forward, error, registered, error_backward, registered_backward], dim=1)
            past_new, skip_co_past_new = self.memory_encoder(past_input)
            memory_skip_co_list.append(skip_co_past_new)

            past_query_list.append(current_feature)
            past_feature_list.append(past_new)
        
        out['backward_flow'] = torch.stack(out['backward_flow'], dim=0)
        out['weights'] = weights

        return out
    



    def forward_multi_task_flow(self, x, label, step=1):
        out = {'forward_flow': [], 'backward_flow': [], 'weights': []}
        T, B, C, H, W = x.shape
        flow = torch.zeros(size=(B, 2, H, W), device=x.device)
        cumulated_forward = torch.zeros(size=(B, 2, H, W), device=x.device)
        cumulated_backward = torch.zeros(size=(B, 2, H, W), device=x.device)

        past_feature_list = []
        max_indices_list = []
        past_query_list = []
        memory_skip_co_list = []

        registered_backward = self.motion_estimation(flow=cumulated_backward, original=x[0])
        error_backward = x[0] - registered_backward

        registered_forward = self.motion_estimation(flow=cumulated_forward, original=x[0])
        error_forward = x[0] - registered_forward

        registered_seg, softmaxed_label = self.transform_label(cumulated_forward, label)

        current_input = self.get_input_query(x[0], 
                                       x[0], 
                                       x[0], 
                                       softmaxed_label,
                                       registered_seg, 
                                       cumulated_forward, 
                                       error_forward, 
                                       registered_forward, 
                                       cumulated_backward,
                                       error_backward, 
                                       registered_backward)
        
        current_feature, skip_co_current = self.query_encoder(current_input) # B, C, H, W
        past_query_list.append(current_feature)

        past_input = self.get_input_memory(x[0],
                                           x[0],
                                           registered_seg,
                                           cumulated_forward, 
                                           error_forward, 
                                           registered_forward,
                                           cumulated_backward,
                                           error_backward,
                                           registered_backward)

        past_new, past_skip_co = self.memory_encoder(past_input)

        memory_skip_co_list.append(past_skip_co)
        past_feature_list.append(past_new)

        for t in range(1, T):

            #print(self.is_softmaxed(softmaxed_label))
            #print(self.is_softmaxed(registered_seg))
#
            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(1, 4)
            #for o in range(4):
            #    ax[o].imshow(registered_seg[0, o].detach().cpu(), cmap='gray')
            #plt.show()

            current_input = self.get_input_query(x[0], 
                                           x[t - 1], 
                                           x[t], 
                                           softmaxed_label,
                                           registered_seg, 
                                           cumulated_forward, 
                                           error_forward, 
                                           registered_forward, 
                                           cumulated_backward,
                                           error_backward, 
                                           registered_backward)
            
            current_feature, skip_co_current = self.query_encoder(current_input) # B, C, H, W

            if self.gaussian:
                max_idx = self.get_max_index(past_query_list[-1], current_feature)
                max_indices_list.append(max_idx)
                max_idx = torch.stack(max_indices_list, dim=0) # T, B, H*W
            else:
                max_idx = None

            past_features = torch.stack(past_feature_list, dim=0) # T, B, C, H, W
            past_query = torch.stack(past_query_list, dim=0) # T, B, C, H, W
            
            if not self.training and t > 2:
                past_features = torch.cat([past_features[:2], past_features[2:-1:step], past_features[-1][None]], dim=0)
                past_query = torch.cat([past_query[:2], past_query[2:-1:step], past_query[-1][None]], dim=0)

            current_feature, weights = self.bottleneck(query=current_feature, 
                                                       key=past_query, 
                                                       value=past_features, 
                                                       max_idx=max_idx,
                                                       video_length=len(x)-1)

            acc_skip_co = []
            for s in range(self.num_stages):
                concatenated = torch.cat([skip_co_current[s], self.dropout_layer(memory_skip_co_list[-1][s]), self.dropout_layer(memory_skip_co_list[0][s])], dim=1)
                concatenated = self.skip_co_reduction_list[s](concatenated)
                acc_skip_co.append(concatenated)

            flow, intermediary = self.flow_decoder(current_feature, acc_skip_co)
            cumulated_forward = cumulated_forward + flow
            out['forward_flow'].append(cumulated_forward)

            flow, intermediary = self.flow_decoder_backward(current_feature, acc_skip_co)
            cumulated_backward = cumulated_backward + flow
            out['backward_flow'].append(cumulated_backward)

            registered_backward = self.motion_estimation(flow=cumulated_backward, original=x[t])
            error_backward = x[0] - registered_backward

            registered_forward = self.motion_estimation(flow=cumulated_forward, original=x[0])
            error_forward = x[t] - registered_forward

            registered_seg, softmaxed_label = self.transform_label(cumulated_forward, label)

            #if self.training:
            #    assert torch.all(registered_seg == softmaxed_label)

            past_input = self.get_input_memory(x[0],
                                           x[t],
                                           registered_seg,
                                           cumulated_forward, 
                                           error_forward, 
                                           registered_forward,
                                           cumulated_backward,
                                           error_backward,
                                           registered_backward)

            past_new, past_skip_co = self.memory_encoder(past_input)
            memory_skip_co_list.append(past_skip_co)

            past_query_list.append(current_feature)
            past_feature_list.append(past_new)
        
        out['forward_flow'] = torch.stack(out['forward_flow'], dim=0)
        out['backward_flow'] = torch.stack(out['backward_flow'], dim=0)
        out['weights'] = weights

        return out
    



    def forward_multi_task_flow_mamba(self, x, label, step=1):
        out = {'backward_flow': [], 
               'sampling_locations': [],
               'offsets': [],
               'attention_weights': []}
        T, B, C, H, W = x.shape
        flow = torch.zeros(size=(B, 2, H, W), device=x.device)
        cumulated_backward = torch.zeros(size=(B, 2, H, W), device=x.device)

        query_list_bottleneck = []
        query_list_skip_co = []

        for t in range(1, T):

            if self.warp:
                registered_error = self.motion_estimation(flow=cumulated_backward, original=x[t])
                error_error = x[0] - registered_error
                current_input = torch.cat([x[0], registered_error, error_error, label], dim=1)
            else:
                current_input = torch.cat([x[0], x[t], label], dim=1)
                
            current_feature, skip_co_current = self.query_encoder(current_input) # B, C, H, W
            query_list_bottleneck.append(current_feature)
            query_list_skip_co.append(skip_co_current)
        
        x = torch.stack(query_list_bottleneck, dim=0) # T-1, B, C, H, W

        bottleneck_features = self.bottleneck(x=x, video_length=len(x))

        for t in range(len(x)):

            flow = self.flow_decoder(bottleneck_features[t], 
                                    query_list_skip_co[t])
        
            cumulated_backward = cumulated_backward + flow
            out['backward_flow'].append(cumulated_backward)
        
        out['backward_flow'] = torch.stack(out['backward_flow'], dim=0)

        return out
    


    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//8, W//8, device=img.device)
        coords1 = coords_grid(N, H//8, W//8, device=img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1
    

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8*H, 8*W)
    

    def upsample_seg(self, seg, mask):
        """ Upsample seg field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = seg.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_seg = F.unfold(8 * seg, [3,3], padding=1)
        up_seg = up_seg.view(N, 4, 9, 1, 1, H, W)

        up_seg = torch.sum(mask * up_seg, dim=2)
        up_seg = up_seg.permute(0, 1, 4, 2, 5, 3)
        return up_seg.reshape(N, 4, 8*H, 8*W)



    def forward_multi_task_flow_deformable_raft(self, x, label, step=1):
        out = {'backward_flow': []}
        T, B, C, H, W = x.shape
        flow_up = torch.zeros(size=(B, 2, H, W), device=x.device)

        motion_list = []

        registered_backward = self.motion_estimation(flow=flow_up, original=x[0])
        error_backward = x[0] - registered_backward

        past_input = torch.cat([x[0], x[0], flow_up, error_backward, registered_backward], dim=1)

        cnet = self.memory_encoder(past_input)
        net, inp = torch.split(cnet, [self.d_model//2, self.d_model//2], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)

        motion_list.append(inp)

        #pos_1d = self.pos_obj_1d(shape_util=(B, T-1), device=x.device)

        #pos_1d = self.pos_1d.permute(1, 0).contiguous()
        #pos_1d = pos_1d[None].repeat(B, 1, 1)
#
        #pos_1d_list = []
        #for pos_reduction in self.pos_reduction_list:
        #    pos_1d = pos_reduction(pos_1d)
        #    pos_1d_list.append(pos_1d)
        #pos_1d_list = pos_1d_list[::-1]

        coords0, coords1 = self.initialize_flow(x[0])
        current_feature_1 = self.query_encoder(x[0]) # B, C, H, W
        for t in range(1, T):

            #print(self.is_softmaxed(softmaxed_label))
            #print(self.is_softmaxed(registered_seg))
#
            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(1, 4)
            #for o in range(4):
            #    ax[o].imshow(registered_seg[0, o].detach().cpu(), cmap='gray')
            #plt.show()

            #registered_error = self.motion_estimation(flow=flow_up.detach(), original=x[t])
            #error_registered = x[0] - registered_error

            current_feature_2 = self.query_encoder(x[t]) # B, C, H, W

            if self.cross_attn_before_corr:
                current_feature_2 = self.transformer(query=current_feature_2, key=current_feature_1, value=current_feature_1)

            if self.memory_attn:
                attention = self.att(current_feature_1, current_feature_2)

            corr_fn = CorrBlock(current_feature_1, current_feature_2, radius=4)

            iteration_flow_list = []
            for itr in range(self.raft_iters):
                coords1 = coords1.detach()
            
                corr = corr_fn(coords1) # index correlation volume

                flow = coords1 - coords0

                if self.memory_attn:
                    net, up_mask, delta_flow = self.update_block(net, motion_list[-1], corr, flow, attention)
                else:
                    net, up_mask, delta_flow = self.update_block(net, motion_list[-1], corr, flow)

                # F(t+1) = F(t) + \Delta(t)
                coords1 = coords1 + delta_flow

                flow_up = self.upsample_flow(coords1 - coords0, up_mask)
                iteration_flow_list.append(flow_up)

            iter_stack = torch.stack(iteration_flow_list, dim=0)
                
            out['backward_flow'].append(iter_stack)

            registered_backward = self.motion_estimation(flow=flow_up, original=x[t])
            error_backward = x[0] - registered_backward

            past_input = torch.cat([x[0], x[t], flow_up, error_backward, registered_backward], dim=1)

            cnet = self.memory_encoder(past_input)
            _, inp = torch.split(cnet, [self.d_model//2, self.d_model//2], dim=1)
            inp = torch.relu(inp)

            motion_list.append(inp)
        
        out['backward_flow'] = torch.stack(out['backward_flow'], dim=1) # iters, T, 2, H, W

        return out
    




    def forward_multi_task_flow_deformable_raft_adjacent(self, x, label, step=1):
        out = {'backward_flow_large': [], 'backward_flow_small': []}
        T, B, C, H, W = x.shape
        flow_up = torch.zeros(size=(B, 2, H, W), device=x.device)
        prev_flow = torch.zeros(size=(B, 2, self.H, self.W), device=x.device)

        motion_list = []

        registered_backward = self.motion_estimation(flow=flow_up.detach(), original=x[0])
        error_backward = x[0] - registered_backward

        past_input = torch.cat([x[0], x[0], flow_up.detach(), error_backward.detach(), registered_backward.detach()], dim=1)

        cnet = self.memory_encoder(past_input)
        net, inp = torch.split(cnet, [self.d_model//2, self.d_model//2], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)

        motion_list.append(inp)

        #pos_1d = self.pos_obj_1d(shape_util=(B, T-1), device=x.device)

        #pos_1d = self.pos_1d.permute(1, 0).contiguous()
        #pos_1d = pos_1d[None].repeat(B, 1, 1)
#
        #pos_1d_list = []
        #for pos_reduction in self.pos_reduction_list:
        #    pos_1d = pos_reduction(pos_1d)
        #    pos_1d_list.append(pos_1d)
        #pos_1d_list = pos_1d_list[::-1]

        coords0_large, coords1_large = self.initialize_flow(x[0])

        feature_init = self.query_encoder(x[0]) # B, C, H, W
        feature_prev = feature_init
        for t in range(1, T):

            #print(self.is_softmaxed(softmaxed_label))
            #print(self.is_softmaxed(registered_seg))
#
            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(1, 4)
            #for o in range(4):
            #    ax[o].imshow(registered_seg[0, o].detach().cpu(), cmap='gray')
            #plt.show()

            #registered_error = self.motion_estimation(flow=flow_up.detach(), original=x[t])
            #error_registered = x[0] - registered_error

            current_feature_2 = self.query_encoder(x[t]) # B, C, H, W

            if self.cross_attn_before_corr:
                current_feature_2 = self.transformer(query=current_feature_2, key=feature_init)

            if self.memory_attn:
                attention = self.att(feature_init, current_feature_2)

            corr_fn_large = CorrBlock(feature_init, current_feature_2, radius=4)
            corr_fn_small = CorrBlock(feature_prev, current_feature_2, radius=4)

            iteration_flow_list_large = []
            iteration_flow_list_small = []
            coords0_small, coords1_small = self.initialize_flow(x[0])
            for itr in range(self.raft_iters):
                coords1_small = coords1_small.detach()
                coords1_large = coords1_large.detach()
            
                corr_large = corr_fn_large(coords1_large) # index correlation volume
                corr_small = corr_fn_small(coords1_small) # index correlation volume

                if self.warp:
                    corr_small = warp(corr_small, prev_flow)

                flow_large = coords1_large - coords0_large
                flow_small = coords1_small - coords0_small

                if self.memory_attn:
                    net, up_mask, delta_flow = self.update_block(net, motion_list[-1], corr_large, flow_large, corr_small, flow_small, attention)
                else:
                    net, up_mask, delta_flow = self.update_block(net, motion_list[-1], corr_large, flow_large, corr_small, flow_small)

                large_up_mask, small_up_mask = torch.split(up_mask, [8**2*9, 8**2*9], dim=1)

                # F(t+1) = F(t) + \Delta(t)
                coords1_large = coords1_large + delta_flow[:, 0:2, :, :]
                coords1_small = coords1_small + delta_flow[:, 2:4, :, :]

                flow_up_large = self.upsample_flow(coords1_large - coords0_large, large_up_mask)
                flow_up_small = self.upsample_flow(coords1_small - coords0_small, small_up_mask)

                iteration_flow_list_large.append(flow_up_large)
                iteration_flow_list_small.append(flow_up_small)

            iter_stack_large = torch.stack(iteration_flow_list_large, dim=0)
            iter_stack_small = torch.stack(iteration_flow_list_small, dim=0)
                
            out['backward_flow_large'].append(iter_stack_large)
            out['backward_flow_small'].append(iter_stack_small)

            registered_backward = self.motion_estimation(flow=flow_up_large.detach(), original=x[t])
            error_backward = x[0] - registered_backward

            past_input = torch.cat([x[0], x[t], flow_up_large.detach(), error_backward.detach(), registered_backward.detach()], dim=1)

            cnet = self.memory_encoder(past_input)
            _, inp = torch.split(cnet, [self.d_model//2, self.d_model//2], dim=1)
            inp = torch.relu(inp)

            motion_list.append(inp)

            feature_prev = current_feature_2
            if self.warp:
                prev_flow = coords1_large - coords0_large
        
        out['backward_flow_large'] = torch.stack(out['backward_flow_large'], dim=1) # iters, T, 2, H, W
        out['backward_flow_small'] = torch.stack(out['backward_flow_small'], dim=1) # iters, T, 2, H, W

        return out
    




    def forward_multi_task_flow_deformable_raft_seg(self, x, label, step=1):
        out = {'backward_flow': [], 'seg': []}
        T, B, C, H, W = x.shape
        flow_up = torch.zeros(size=(B, 2, H, W), device=x.device)
        seg_up = torch.zeros(size=(B, 4, H, W), device=x.device)

        motion_list = []
        memory_seg_list = []

        registered_backward = self.motion_estimation(flow=flow_up.detach(), original=x[0])
        error_backward = x[0] - registered_backward

        past_input = torch.cat([x[0], x[0], flow_up.detach(), error_backward.detach(), registered_backward.detach()], dim=1)
        mem = self.memory_encoder(past_input)
        mem = torch.relu(mem)
        motion_list.append(mem)

        context_input = torch.cat([x[0], label], dim=1)
        cnet = self.context_encoder(context_input)
        net, inp = torch.split(cnet, [self.d_model//2, self.d_model//2], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)

        memory_seg_list.append(inp)

        #pos_1d = self.pos_obj_1d(shape_util=(B, T-1), device=x.device)

        #pos_1d = self.pos_1d.permute(1, 0).contiguous()
        #pos_1d = pos_1d[None].repeat(B, 1, 1)
#
        #pos_1d_list = []
        #for pos_reduction in self.pos_reduction_list:
        #    pos_1d = pos_reduction(pos_1d)
        #    pos_1d_list.append(pos_1d)
        #pos_1d_list = pos_1d_list[::-1]

        coords0_large, coords1_large = self.initialize_flow(x[0])

        feature_init, _ = self.query_encoder(x[0]) # B, C, H, W
        query_history_list = [feature_init]
        for t in range(1, T):

            #print(self.is_softmaxed(softmaxed_label))
            #print(self.is_softmaxed(registered_seg))
#
            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(1, 4)
            #for o in range(4):
            #    ax[o].imshow(registered_seg[0, o].detach().cpu(), cmap='gray')
            #plt.show()

            #registered_error = self.motion_estimation(flow=flow_up.detach(), original=x[t])
            #error_registered = x[0] - registered_error

            current_feature_2, skip_co_seg = self.query_encoder(x[t]) # B, C, H, W

            past_features_seg = torch.stack(memory_seg_list, dim=0) # T, B, C, H, W
            query_history = torch.stack(query_history_list, dim=0) # T, B, C, H, W
            if not self.training:
                query_history = torch.cat([query_history[:2], query_history[2:-1:step], query_history[-1][None]], dim=0)
                past_features_seg = torch.cat([past_features_seg[:2], past_features_seg[2:-1:step], past_features_seg[-1][None]], dim=0)
            current_feature_2 = self.transformer(query=current_feature_2, key=query_history, value=past_features_seg)

            seg_up = self.seg_decoder(current_feature_2, skip_co_seg)

            corr_fn_large = CorrBlock(feature_init, current_feature_2, radius=4)

            iteration_flow_list_large = []
            for itr in range(self.raft_iters):
                coords1_large = coords1_large.detach()
            
                corr_large = corr_fn_large(coords1_large) # index correlation volume

                flow_large = coords1_large - coords0_large

                net, up_mask, delta_flow = self.update_block(net, motion_list[-1], corr_large, flow_large, memory_seg_list[-1])

                #large_up_mask, seg_up_mask = torch.split(up_mask, [8**2*9, 8**2*9], dim=1)

                # F(t+1) = F(t) + \Delta(t)
                coords1_large = coords1_large + delta_flow
                #coords1_large = coords1_large + delta_flow[:, 0:2, :, :]

                flow_up_large = self.upsample_flow(coords1_large - coords0_large, up_mask)

                iteration_flow_list_large.append(flow_up_large)

            iter_stack_large = torch.stack(iteration_flow_list_large, dim=0)
                
            out['backward_flow'].append(iter_stack_large)
            out['seg'].append(seg_up)

            registered_backward = self.motion_estimation(flow=flow_up_large.detach(), original=x[t])
            error_backward = x[0] - registered_backward

            past_input = torch.cat([x[0], x[t], flow_up_large.detach(), error_backward.detach(), registered_backward.detach()], dim=1)

            mem = self.memory_encoder(past_input)
            mem = torch.relu(mem)

            motion_list.append(mem)
            query_history_list.append(current_feature_2)

            context_input = torch.cat([x[t], seg_up.detach()], dim=1)
            cnet = self.context_encoder(context_input)
            _, inp = torch.split(cnet, [self.d_model//2, self.d_model//2], dim=1)
            inp = torch.relu(inp)
            memory_seg_list.append(inp)
        
        out['backward_flow'] = torch.stack(out['backward_flow'], dim=1) # iters, T, 2, H, W
        out['seg'] = torch.stack(out['seg'], dim=0) # T, 4, H, W

        return out
    



    def forward_multi_task_flow_deformable_cost_volume_transformer_swin(self, x, step=1):
        out = {'backward_flow': [], 
               'weights': []}
        weights = None
        T, B, C, H, W = x.shape
        flow = torch.zeros(size=(B, 2, H, W), device=x.device)
        cumulated_backward = torch.zeros(size=(B, 2, H, W), device=x.device)
        init = torch.zeros(size=(B, self.d_model, self.H, self.W), device=x.device)
        hidden_state = init

        registered_backward = self.motion_estimation(flow=cumulated_backward.detach(), original=x[0])
        error_backward = x[0] - registered_backward

        past_input = torch.cat([x[0], x[0], cumulated_backward, error_backward, registered_backward], dim=1)

        past_motion, past_skip_co = self.memory_encoder(past_input)

        coords0, coords1 = self.initialize_flow(x[0])
        first_feature, first_skip_co = self.query_encoder(x[0]) # B, C, H, W

        previous_feature = first_feature
        previous_skip_co = first_skip_co

        for t in range(1, T):

            #print(self.is_softmaxed(softmaxed_label))
            #print(self.is_softmaxed(registered_seg))
#
            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(1, 4)
            #for o in range(4):
            #    ax[o].imshow(registered_seg[0, o].detach().cpu(), cmap='gray')
            #plt.show()

            if self.warp:
                registered_error = self.motion_estimation(flow=cumulated_backward.detach(), original=x[t])
                current_feature, skip_co_current = self.query_encoder(registered_error) # B, C, H, W
            else:
                current_feature, skip_co_current = self.query_encoder(x[t]) # B, C, H, W
            
            #error_registered = x[0] - registered_error


            new_skip_co = []
            for s in range(self.num_stages):
                x_current_a = skip_co_current[s]
                x_current_b = skip_co_current[s]
                for blk in self.block_1_list[s]:
                    x_current_a = blk(x_current_a, previous_skip_co[s], previous_skip_co[s])
                for blk in self.block_2_list[s]:
                    x_current_b = blk(x_current_b, first_skip_co[s], past_skip_co[s])
                concatenated = torch.cat([x_current_a, x_current_b], dim=1)
                concatenated = self.skip_co_reduction_list[s](concatenated)
                new_skip_co.append(concatenated)


            if self.correlation_value:
                corr_fn = CorrBlock(previous_feature, current_feature, radius=4)
                coords1 = coords1.detach()
                corr = corr_fn(coords1) # index correlation volume
                corr = self.cost_volume_encoder_list[-1](corr)

                current_feature_1 = self.bottleneck1(query=current_feature, 
                                            key=previous_feature,
                                            value=corr)
                current_feature_1 = self.reduce_transformer_1(current_feature_1)
            else:
                current_feature_1 = self.bottleneck1(query=current_feature, 
                                                key=previous_feature,
                                                value=previous_feature)
                #current_feature_1 = self.reduce_transformer_1(current_feature_1)
                
            current_feature_2 = self.bottleneck2(query=current_feature, 
                                                key=first_feature,
                                                value=past_motion)
            #current_feature_2 = self.reduce_transformer_2(current_feature_2)

            gru_input = torch.cat([current_feature_1, current_feature_2], dim=1)
            gru_input = self.reduce_transformer(gru_input)
            if self.remove_GRU:
                hidden_state = gru_input
            else:
                hidden_state = self.gru_cell(gru_input, hidden_state)


            flow, intermediary = self.flow_decoder(hidden_state, new_skip_co)
            cumulated_backward = cumulated_backward + flow
            out['backward_flow'].append(cumulated_backward)

            #fig, ax = plt.subplots(1, 1)
            #temp = torch.abs(torch.stack(out['backward_flow'], dim=0))
            #ax.imshow(temp.mean(0)[0, 0].cpu(), cmap='hot')
            #plt.show()

            registered_backward = self.motion_estimation(flow=cumulated_backward.detach(), original=x[t])
            error_backward = x[0] - registered_backward

            #if self.training:
            #    assert torch.all(registered_seg == softmaxed_label)

            past_input = torch.cat([x[0], x[t], cumulated_backward, error_backward, registered_backward], dim=1)

            past_motion, past_skip_co = self.memory_encoder(past_input)

            if not self.warp:
                previous_feature = current_feature
                previous_skip_co = skip_co_current
        
        out['backward_flow'] = torch.stack(out['backward_flow'], dim=0)
        out['weights'] = weights


        return out
    



    def forward_multi_task_flow_deformable_cost_volume_transformer_cat(self, x, step=1):
        out = {'backward_flow': [], 
               'weights': []}
        weights = None
        T, B, C, H, W = x.shape
        flow = torch.zeros(size=(B, 2, H, W), device=x.device)
        cumulated_backward = torch.zeros(size=(B, 2, H, W), device=x.device)
        init = torch.zeros(size=(B, self.d_model, self.H, self.W), device=x.device)
        hidden_state = init

        registered_backward = self.motion_estimation(flow=cumulated_backward.detach(), original=x[0])
        error_backward = x[0] - registered_backward

        past_input = torch.cat([x[0], x[0], cumulated_backward, error_backward, registered_backward], dim=1)

        past_motion, past_skip_co = self.memory_encoder(past_input)

        coords0, coords1 = self.initialize_flow(x[0])
        first_feature, first_skip_co = self.query_encoder(x[0]) # B, C, H, W

        previous_feature = first_feature
        previous_skip_co = first_skip_co

        for t in range(1, T):

            #print(self.is_softmaxed(softmaxed_label))
            #print(self.is_softmaxed(registered_seg))
#
            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(1, 4)
            #for o in range(4):
            #    ax[o].imshow(registered_seg[0, o].detach().cpu(), cmap='gray')
            #plt.show()

            if self.warp:
                registered_error = self.motion_estimation(flow=cumulated_backward.detach(), original=x[t])
                current_feature, skip_co_current = self.query_encoder(registered_error) # B, C, H, W
            else:
                current_feature, skip_co_current = self.query_encoder(x[t]) # B, C, H, W
            
            #error_registered = x[0] - registered_error


            new_skip_co = []
            for s in range(self.num_stages):
                if self.skip_co_type == 'both' or self.skip_co_type == 'no_conv':
                    corr = self.cost_volume_computation_list[s](skip_co_current[s], previous_skip_co[s])
                    corr = self.cost_volume_encoder_list[s](corr)
                    concatenated = torch.cat([corr, past_skip_co[s]], dim=1)
                elif self.skip_co_type == 'past':
                    concatenated = past_skip_co[s]
                elif self.skip_co_type == 'current':
                    corr = self.cost_volume_computation_list[s](skip_co_current[s], previous_skip_co[s])
                    corr = self.cost_volume_encoder_list[s](corr)
                    concatenated = corr
                concatenated = self.skip_co_reduction_list[s](concatenated)
                new_skip_co.append(concatenated)


            if self.correlation_value:
                corr_fn = CorrBlock(previous_feature, current_feature, radius=4)
                coords1 = coords1.detach()
                corr = corr_fn(coords1) # index correlation volume
                corr = self.cost_volume_encoder_list[-1](corr)

                current_feature_1 = self.bottleneck1(query=current_feature, 
                                            key=previous_feature,
                                            value=corr)
                current_feature_1 = self.reduce_transformer_1(current_feature_1)
            else:
                current_feature_1 = self.bottleneck1(query=current_feature, 
                                                key=previous_feature,
                                                value=previous_feature)
                #current_feature_1 = self.reduce_transformer_1(current_feature_1)
                
            current_feature_2 = self.bottleneck2(query=current_feature, 
                                                key=first_feature,
                                                value=past_motion)
            #current_feature_2 = self.reduce_transformer_2(current_feature_2)

            gru_input = torch.cat([current_feature_1, current_feature_2], dim=1)
            gru_input = self.reduce_transformer(gru_input)
            if self.remove_GRU:
                hidden_state = gru_input
            else:
                hidden_state = self.gru_cell(gru_input, hidden_state)


            flow, intermediary = self.flow_decoder(hidden_state, new_skip_co)
            cumulated_backward = cumulated_backward + flow
            out['backward_flow'].append(cumulated_backward)

            #fig, ax = plt.subplots(1, 1)
            #temp = torch.abs(torch.stack(out['backward_flow'], dim=0))
            #ax.imshow(temp.mean(0)[0, 0].cpu(), cmap='hot')
            #plt.show()

            registered_backward = self.motion_estimation(flow=cumulated_backward.detach(), original=x[t])
            error_backward = x[0] - registered_backward

            #if self.training:
            #    assert torch.all(registered_seg == softmaxed_label)

            past_input = torch.cat([x[0], x[t], cumulated_backward, error_backward, registered_backward], dim=1)

            past_motion, past_skip_co = self.memory_encoder(past_input)

            if not self.warp:
                previous_feature = current_feature
                previous_skip_co = skip_co_current
        
        out['backward_flow'] = torch.stack(out['backward_flow'], dim=0)
        out['weights'] = weights


        return out
    



    def forward_multi_task_flow_deformable_cost_volume_transformer_cat_label(self, x, label, step=1):
        #matplotlib.use('QtAgg')
        #fig, ax = plt.subplots(1, 4)
        #ax[0].imshow(x.cpu()[0, 0, 0], cmap='gray')
        #ax[1].imshow(label.cpu()[0, 0, 0], cmap='gray')
        #ax[2].imshow(label.cpu()[0, 0, 1], cmap='gray')
        #ax[3].imshow(label.cpu()[0, 0, 2], cmap='gray')
        #plt.show()

        out = {'backward_flow': [], 
               'weights': []}
        weights = None
        T, B, C, H, W = x.shape
        flow = torch.zeros(size=(B, 2, H, W), device=x.device)
        cumulated_backward = torch.zeros(size=(B, 2, H, W), device=x.device)
        init = torch.zeros(size=(B, self.d_model if not self.label_pretrained else 384, self.H, self.W), device=x.device)
        hidden_state = init

        context_input = torch.cat([x[0], label[0]], dim=1)
        context_feature, context_skip_co = self.context_encoder(context_input)

        registered_backward = self.motion_estimation(flow=cumulated_backward.detach(), original=x[0])
        error_backward = x[0] - registered_backward

        past_input = torch.cat([x[0], x[0], cumulated_backward, error_backward, registered_backward], dim=1)

        past_motion, past_skip_co = self.memory_encoder(past_input)

        coords0, coords1 = self.initialize_flow(x[0])
        first_feature, first_skip_co = self.query_encoder(x[0]) # B, C, H, W

        previous_feature = first_feature
        previous_skip_co = first_skip_co

        for t in range(1, T):

            #print(self.is_softmaxed(softmaxed_label))
            #print(self.is_softmaxed(registered_seg))
#
            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(1, 4)
            #for o in range(4):
            #    ax[o].imshow(registered_seg[0, o].detach().cpu(), cmap='gray')
            #plt.show()

            if self.warp:
                registered_error = self.motion_estimation(flow=cumulated_backward.detach(), original=x[t])
                current_feature, skip_co_current = self.query_encoder(registered_error) # B, C, H, W
            else:
                current_feature, skip_co_current = self.query_encoder(x[t]) # B, C, H, W
            
            #error_registered = x[0] - registered_error


            new_skip_co = []
            for s in range(self.num_stages):
                if self.skip_co_type == 'both' or self.skip_co_type == 'no_conv':
                    corr = self.cost_volume_computation_list[s](skip_co_current[s], previous_skip_co[s])
                    corr = self.cost_volume_encoder_list[s](corr)
                    concatenated = torch.cat([corr, past_skip_co[s], context_skip_co[s]], dim=1)
                elif self.skip_co_type == 'past':
                    concatenated = past_skip_co[s]
                elif self.skip_co_type == 'current':
                    corr = self.cost_volume_computation_list[s](skip_co_current[s], previous_skip_co[s])
                    corr = self.cost_volume_encoder_list[s](corr)
                    concatenated = corr
                concatenated = self.skip_co_reduction_list[s](concatenated)
                new_skip_co.append(concatenated)


            if self.correlation_value:
                corr_fn = CorrBlock(previous_feature, current_feature, radius=4)
                coords1 = coords1.detach()
                corr = corr_fn(coords1) # index correlation volume
                corr = self.cost_volume_encoder_list[-1](corr)

                current_feature_1 = self.bottleneck1(query=current_feature, 
                                            key=previous_feature,
                                            value=corr)
            else:
                current_feature_1 = self.bottleneck1(query=current_feature, 
                                                key=previous_feature,
                                                value=previous_feature)
                
            current_feature_2 = self.bottleneck2(query=current_feature, 
                                                key=first_feature,
                                                value=past_motion)

            current_feature_3 = self.bottleneck3(query=current_feature, 
                                                key=first_feature,
                                                value=context_feature)

            gru_input = torch.cat([current_feature_1, current_feature_2, current_feature_3], dim=1)
            gru_input = self.reduce_transformer(gru_input)
            if self.remove_GRU:
                hidden_state = gru_input
            else:
                hidden_state = self.gru_cell(gru_input, hidden_state)

            to_decode = self.reduce_after_gru(hidden_state)

            flow, intermediary = self.flow_decoder(to_decode, new_skip_co)
            cumulated_backward = cumulated_backward + flow
            out['backward_flow'].append(cumulated_backward)

            #fig, ax = plt.subplots(1, 1)
            #temp = torch.abs(torch.stack(out['backward_flow'], dim=0))
            #ax.imshow(temp.mean(0)[0, 0].cpu(), cmap='hot')
            #plt.show()

            registered_backward = self.motion_estimation(flow=cumulated_backward.detach(), original=x[t])
            error_backward = x[0] - registered_backward

            #if self.training:
            #    assert torch.all(registered_seg == softmaxed_label)

            past_input = torch.cat([x[0], x[t], cumulated_backward, error_backward, registered_backward], dim=1)

            past_motion, past_skip_co = self.memory_encoder(past_input)

            if not self.warp:
                previous_feature = current_feature
                previous_skip_co = skip_co_current
        
        out['backward_flow'] = torch.stack(out['backward_flow'], dim=0)
        out['weights'] = weights


        return out
    



    def forward_motion_appearance(self, x, step=1):
        out = {'backward_flow': [], 
               'weights': []}
        weights = None
        T, B, C, H, W = x.shape
        flow = torch.zeros(size=(B, 2, H, W), device=x.device)
        cumulated_backward = torch.zeros(size=(B, 2, H, W), device=x.device)
        init = torch.zeros(size=(B, self.d_model, self.H, self.W), device=x.device)
        hidden_state = init

        registered_backward = self.motion_estimation(flow=cumulated_backward.detach(), original=x[0])
        error_backward = x[0] - registered_backward

        past_input = torch.cat([x[0], x[0], cumulated_backward, error_backward, registered_backward], dim=1)

        past_motion, past_skip_co = self.memory_encoder(past_input)

        coords0, coords1 = self.initialize_flow(x[0])
        query_input = torch.cat([x[0], x[0]], dim=1)
        first_appearance_feature, _, _ = self.query_encoder(query_input) # B, C, H, W

        previous_appearance_feature = first_appearance_feature

        for t in range(1, T):

            #print(self.is_softmaxed(softmaxed_label))
            #print(self.is_softmaxed(registered_seg))
#
            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(1, 4)
            #for o in range(4):
            #    ax[o].imshow(registered_seg[0, o].detach().cpu(), cmap='gray')
            #plt.show()

            query_input = torch.cat([x[t], x[t-1]], dim=1)
            current_appearance_feature, current_motion_feature, skip_co_current = self.query_encoder(query_input) # B, C, H, W
            
            #error_registered = x[0] - registered_error


            new_skip_co = []
            for s in range(self.num_stages):
                concatenated = torch.cat([skip_co_current[s], past_skip_co[s]], dim=1)
                concatenated = self.skip_co_reduction_list[s](concatenated)
                new_skip_co.append(concatenated)


            if self.correlation_value:

                current_feature_1 = self.bottleneck1(query=current_appearance_feature, 
                                            key=previous_appearance_feature,
                                            value=current_motion_feature)
                #current_feature_1 = self.reduce_transformer_1(current_feature_1)
            else:
                current_feature_1 = self.bottleneck1(query=current_appearance_feature, 
                                                key=previous_appearance_feature,
                                                value=previous_appearance_feature)
                #current_feature_1 = self.reduce_transformer_1(current_feature_1)
                
            current_feature_2 = self.bottleneck2(query=current_appearance_feature, 
                                                key=first_appearance_feature,
                                                value=past_motion)
            #current_feature_2 = self.reduce_transformer_2(current_feature_2)

            gru_input = torch.cat([current_feature_1, current_feature_2], dim=1)
            gru_input = self.reduce_transformer(gru_input)
            if self.remove_GRU:
                hidden_state = gru_input
            else:
                hidden_state = self.gru_cell(gru_input, hidden_state)


            flow, intermediary = self.flow_decoder(hidden_state, new_skip_co)
            cumulated_backward = cumulated_backward + flow
            out['backward_flow'].append(cumulated_backward)

            #fig, ax = plt.subplots(1, 1)
            #temp = torch.abs(torch.stack(out['backward_flow'], dim=0))
            #ax.imshow(temp.mean(0)[0, 0].cpu(), cmap='hot')
            #plt.show()

            registered_backward = self.motion_estimation(flow=cumulated_backward.detach(), original=x[t])
            error_backward = x[0] - registered_backward

            #if self.training:
            #    assert torch.all(registered_seg == softmaxed_label)

            past_input = torch.cat([x[0], x[t], cumulated_backward, error_backward, registered_backward], dim=1)

            past_motion, past_skip_co = self.memory_encoder(past_input)

            if not self.warp:
                previous_appearance_feature = current_appearance_feature
        
        out['backward_flow'] = torch.stack(out['backward_flow'], dim=0)
        out['weights'] = weights


        return out
    



    def forward_multi_task_flow_context(self, x, label, step=1):
        out = {'backward_flow': [], 
               'weights': []}
        weights = None
        T, B, C, H, W = x.shape
        flow = torch.zeros(size=(B, 2, H, W), device=x.device)
        cumulated_backward = torch.zeros(size=(B, 2, H, W), device=x.device)
        init = torch.zeros(size=(B, self.d_model, self.H, self.W), device=x.device)

        memory_list = []
        query_list = []

        registered_backward = self.motion_estimation(flow=cumulated_backward.detach(), original=x[0])
        error_backward = x[0] - registered_backward

        past_input = torch.cat([x[0], x[0], cumulated_backward, error_backward, registered_backward], dim=1)

        past_new, past_skip_co = self.memory_encoder(past_input)
        memory_list.append(past_new)

        if self.warp:
            registered_error = self.motion_estimation(flow=cumulated_backward.detach(), original=x[0])
            current_input = torch.cat([x[0], registered_error], dim=1)
        else:
            current_input = torch.cat([x[0], x[0]], dim=1)
        current_feature, skip_co_current = self.query_encoder(current_input) # B, C, H, W
        query_list.append(current_feature)

        context_input = torch.cat([x[0], label], dim=1)
        context_feature, _ = self.context_encoder(context_input)
        hidden_state = context_feature
        for t in range(1, T):

            #print(self.is_softmaxed(softmaxed_label))
            #print(self.is_softmaxed(registered_seg))
#
            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(1, 4)
            #for o in range(4):
            #    ax[o].imshow(registered_seg[0, o].detach().cpu(), cmap='gray')
            #plt.show()
            if self.warp:
                registered_error = self.motion_estimation(flow=cumulated_backward.detach(), original=x[t])
                current_input = torch.cat([x[0], registered_error], dim=1)
            else:
                current_input = torch.cat([x[t], x[t-1]], dim=1)
            current_feature, skip_co_current = self.query_encoder(current_input) # B, C, H, W
            skip_co_current.append(current_feature)
            #error_registered = x[0] - registered_error


            #past_features = torch.stack(past_feature_list, dim=0) # T, B, C, H, W
#
            #if not self.training and t > 2:
            #    past_features = torch.cat([past_features[:2], past_features[2:-1:step], past_features[-1][None]], dim=0)

            new_skip_co = []
            for s in range(self.num_stages):
                if self.skip_co_type == 'current':
                    concatenated = skip_co_current[s]
                else:
                    concatenated = torch.cat([skip_co_current[s], past_skip_co[s]], dim=1)
                concatenated = self.skip_co_reduction_list[s](concatenated)
                new_skip_co.append(concatenated)

            current_feature = torch.cat([context_feature, past_new, current_feature], dim=1)
            current_feature = self.context_reduction(current_feature)

            # Transformer
            B, C, H, W = current_feature.shape
            pos_2d = self.pos_obj_2d(shape_util=(B, H, W), device=x.device)
            pos_2d = pos_2d.permute(0, 2, 3, 1).contiguous()
            pos_2d = pos_2d.view(B, H * W, C)

            current_feature = current_feature.permute(0, 2, 3, 1).contiguous() # B, self.H, W, C
            current_feature = current_feature.view(B, H * W, C)

            current_feature = self.transformer_layer(current_feature, pos=pos_2d)[0]

            current_feature = current_feature.permute(0, 2, 1).contiguous()
            current_feature = current_feature.view(B, C, H, W)


            if self.remove_GRU:
                hidden_state = current_feature
            else:
                hidden_state = self.gru_cell(current_feature, hidden_state)

            flow, intermediary = self.flow_decoder(hidden_state, new_skip_co)
            cumulated_backward = cumulated_backward + flow
            out['backward_flow'].append(cumulated_backward)

            #fig, ax = plt.subplots(1, 1)
            #temp = torch.abs(torch.stack(out['backward_flow'], dim=0))
            #ax.imshow(temp.mean(0)[0, 0].cpu(), cmap='hot')
            #plt.show()

            registered_backward = self.motion_estimation(flow=cumulated_backward.detach(), original=x[t])
            error_backward = x[0] - registered_backward

            #if self.training:
            #    assert torch.all(registered_seg == softmaxed_label)

            past_input = torch.cat([x[0], x[t], cumulated_backward, error_backward, registered_backward], dim=1)

            past_new, past_skip_co = self.memory_encoder(past_input)
        
        out['backward_flow'] = torch.stack(out['backward_flow'], dim=0)
        out['weights'] = weights

        return out
    


    def forward_multi_task_flow_deformable_cost_volume_no_memory_read(self, x, label, step=1):
        out = {'backward_flow': [], 
               'weights': []}
        weights = None
        T, B, C, H, W = x.shape
        flow = torch.zeros(size=(B, 2, H, W), device=x.device)
        cumulated_backward = torch.zeros(size=(B, 2, H, W), device=x.device)

        registered_backward = self.motion_estimation(flow=cumulated_backward.detach(), original=x[0])
        error_backward = x[0] - registered_backward

        past_input = torch.cat([x[0], x[0], cumulated_backward, error_backward, registered_backward], dim=1)

        past_new, past_skip_co = self.memory_encoder(past_input)
        past_skip_co.append(past_new)
        hidden_state = past_new
        #pos_1d = self.pos_obj_1d(shape_util=(B, T-1), device=x.device)

        #pos_1d = self.pos_1d.permute(1, 0).contiguous()
        #pos_1d = pos_1d[None].repeat(B, 1, 1)
#
        #pos_1d_list = []
        #for pos_reduction in self.pos_reduction_list:
        #    pos_1d = pos_reduction(pos_1d)
        #    pos_1d_list.append(pos_1d)
        #pos_1d_list = pos_1d_list[::-1]

        coords0, coords1 = self.initialize_flow(x[0])
        first_feature, skip_co_first = self.query_encoder(x[0]) # B, C, H, W

        previous_feature = first_feature
        previous_skip_co = skip_co_first
        for t in range(1, T):

            #print(self.is_softmaxed(softmaxed_label))
            #print(self.is_softmaxed(registered_seg))
#
            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(1, 4)
            #for o in range(4):
            #    ax[o].imshow(registered_seg[0, o].detach().cpu(), cmap='gray')
            #plt.show()

            if self.warp:
                registered_error = self.motion_estimation(flow=cumulated_backward.detach(), original=x[t])
                current_feature, skip_co_current = self.query_encoder(registered_error) # B, C, H, W
            else:
                current_feature, skip_co_current = self.query_encoder(x[t]) # B, C, H, W

            #error_registered = x[0] - registered_error


            #past_features = torch.stack(past_feature_list, dim=0) # T, B, C, H, W
#
            #if not self.training and t > 2:
            #    past_features = torch.cat([past_features[:2], past_features[2:-1:step], past_features[-1][None]], dim=0)

            new_skip_co = []
            for s in range(self.num_stages + 1):

                if s == self.num_stages:
                    corr_fn = CorrBlock(current_feature, previous_feature, radius=4)
                    coords1 = coords1.detach()
                    corr = corr_fn(coords1) # index correlation volume
                    corr = self.cost_volume_encoder_list[s](corr)
                else:
                    corr = self.cost_volume_computation_list[s](skip_co_current[s], previous_skip_co[s])
                    corr = self.cost_volume_encoder_list[s](corr)

                concatenated = torch.cat([corr, past_skip_co[s]], dim=1)
                concatenated = self.skip_co_reduction_list[s](concatenated)
                new_skip_co.append(concatenated)
            
            hidden_state = self.gru_cell(new_skip_co[-1], hidden_state)

            flow, intermediary = self.flow_decoder(hidden_state, new_skip_co[:-1])
            cumulated_backward = cumulated_backward + flow
            out['backward_flow'].append(cumulated_backward)

            #fig, ax = plt.subplots(1, 1)
            #temp = torch.abs(torch.stack(out['backward_flow'], dim=0))
            #ax.imshow(temp.mean(0)[0, 0].cpu(), cmap='hot')
            #plt.show()

            registered_backward = self.motion_estimation(flow=cumulated_backward.detach(), original=x[t])
            error_backward = x[0] - registered_backward

            #if self.training:
            #    assert torch.all(registered_seg == softmaxed_label)

            past_input = torch.cat([x[0], x[t], cumulated_backward, error_backward, registered_backward], dim=1)

            past_new, past_skip_co = self.memory_encoder(past_input)
            past_skip_co.append(past_new)

            if not self.warp:
                previous_feature = current_feature
                previous_skip_co = skip_co_current
        
        out['backward_flow'] = torch.stack(out['backward_flow'], dim=0)
        out['weights'] = weights


        return out
    




    def forward_multi_task_flow_small_memory(self, x, label, step=1):
        out = {'forward_flow': [], 'backward_flow': [], 'weights': []}
        T, B, C, H, W = x.shape
        flow = torch.zeros(size=(B, 2, H, W), device=x.device)
        cumulated_forward = torch.zeros(size=(B, 2, H, W), device=x.device)
        cumulated_backward = torch.zeros(size=(B, 2, H, W), device=x.device)

        past_feature_list = []
        max_indices_list = []
        past_query_list = []
        memory_skip_co_list = []

        registered_backward = self.motion_estimation(flow=cumulated_backward, original=x[0])
        error_backward = x[0] - registered_backward

        registered_forward = self.motion_estimation(flow=cumulated_forward, original=x[0])
        error_forward = x[0] - registered_forward

        registered_seg, softmaxed_label = self.transform_label(cumulated_forward, label)

        current_input = self.get_input_query(x[0], 
                                       x[0], 
                                       x[0], 
                                       softmaxed_label,
                                       registered_seg, 
                                       cumulated_forward, 
                                       error_forward, 
                                       registered_forward, 
                                       cumulated_backward,
                                       error_backward, 
                                       registered_backward)
        
        current_feature, skip_co_current = self.query_encoder(current_input) # B, C, H, W
        past_query_list.append(current_feature)

        past_input = self.get_input_memory(x[0],
                                           x[0],
                                           registered_seg,
                                           cumulated_forward, 
                                           error_forward, 
                                           registered_forward,
                                           cumulated_backward,
                                           error_backward,
                                           registered_backward)

        past_new, past_skip_co = self.memory_encoder(past_input)

        memory_skip_co_list.append(past_skip_co)
        past_feature_list.append(past_new)

        for t in range(1, T):

            #print(self.is_softmaxed(softmaxed_label))
            #print(self.is_softmaxed(registered_seg))
#
            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(1, 4)
            #for o in range(4):
            #    ax[o].imshow(registered_seg[0, o].cpu(), cmap='gray')
            #plt.show()

            current_input = self.get_input_query(x[0], 
                                           x[t - 1], 
                                           x[t], 
                                           softmaxed_label,
                                           registered_seg, 
                                           cumulated_forward, 
                                           error_forward, 
                                           registered_forward, 
                                           cumulated_backward,
                                           error_backward, 
                                           registered_backward)
            
            current_feature, skip_co_current = self.query_encoder(current_input) # B, C, H, W

            if self.gaussian:
                max_idx = self.get_max_index(past_query_list[-1], current_feature)
                max_indices_list.append(max_idx)
                max_idx = torch.stack(max_indices_list, dim=0) # T, B, H*W
            else:
                max_idx = None

            past_features = torch.stack(past_feature_list, dim=0) # T, B, C, H, W
            past_query = torch.stack(past_query_list, dim=0) # T, B, C, H, W
            
            past_features = past_features[-1][None]
            past_query = past_features[-1][None]

            current_feature, weights = self.bottleneck(query=current_feature, 
                                                       key=past_query, 
                                                       value=past_features, 
                                                       max_idx=max_idx)

            acc_skip_co = []
            for s in range(self.num_stages):
                concatenated = torch.cat([skip_co_current[s], self.dropout_layer(memory_skip_co_list[-1][s]), self.dropout_layer(memory_skip_co_list[0][s])], dim=1)
                concatenated = self.skip_co_reduction_list[s](concatenated)
                acc_skip_co.append(concatenated)

            flow, intermediary = self.flow_decoder(current_feature, acc_skip_co)
            cumulated_forward = cumulated_forward + flow
            out['forward_flow'].append(cumulated_forward)

            flow, intermediary = self.flow_decoder_backward(current_feature, acc_skip_co)
            cumulated_backward = cumulated_backward + flow
            out['backward_flow'].append(cumulated_backward)

            registered_backward = self.motion_estimation(flow=cumulated_backward, original=x[t])
            error_backward = x[0] - registered_backward

            registered_forward = self.motion_estimation(flow=cumulated_forward, original=x[0])
            error_forward = x[t] - registered_forward

            registered_seg, softmaxed_label = self.transform_label(cumulated_forward, label)

            #if self.training:
            #    assert torch.all(registered_seg == softmaxed_label)

            past_input = self.get_input_memory(x[0],
                                           x[t],
                                           registered_seg,
                                           cumulated_forward, 
                                           error_forward, 
                                           registered_forward,
                                           cumulated_backward,
                                           error_backward,
                                           registered_backward)

            past_new, past_skip_co = self.memory_encoder(past_input)
            memory_skip_co_list.append(past_skip_co)

            past_query_list.append(current_feature)
            past_feature_list.append(past_new)
        
        out['forward_flow'] = torch.stack(out['forward_flow'], dim=0)
        out['backward_flow'] = torch.stack(out['backward_flow'], dim=0)
        out['weights'] = weights

        return out
    




    def forward_multi_task_flow_no_skip_co(self, x, label, step=1):
        out = {'forward_flow': [], 'backward_flow': [], 'weights': []}
        T, B, C, H, W = x.shape
        flow = torch.zeros(size=(B, 2, H, W), device=x.device)
        cumulated_forward = torch.zeros(size=(B, 2, H, W), device=x.device)
        cumulated_backward = torch.zeros(size=(B, 2, H, W), device=x.device)

        past_feature_list = []
        max_indices_list = []
        past_query_list = []
        memory_skip_co_list = []

        registered_backward = self.motion_estimation(flow=cumulated_backward, original=x[0])
        error_backward = x[0] - registered_backward

        registered_forward = self.motion_estimation(flow=cumulated_forward, original=x[0])
        error_forward = x[0] - registered_forward

        registered_seg, softmaxed_label = self.transform_label(cumulated_forward, label)

        current_input = self.get_input_query(x[0], 
                                       x[0], 
                                       x[0], 
                                       softmaxed_label,
                                       registered_seg, 
                                       cumulated_forward, 
                                       error_forward, 
                                       registered_forward, 
                                       cumulated_backward,
                                       error_backward, 
                                       registered_backward)
        
        current_feature, skip_co_current = self.query_encoder(current_input) # B, C, H, W
        past_query_list.append(current_feature)

        past_input = self.get_input_memory(x[0],
                                           x[0],
                                           registered_seg,
                                           cumulated_forward, 
                                           error_forward, 
                                           registered_forward,
                                           cumulated_backward,
                                           error_backward,
                                           registered_backward)

        past_new, past_skip_co = self.memory_encoder(past_input)

        memory_skip_co_list.append(past_skip_co)
        past_feature_list.append(past_new)

        for t in range(1, T):

            #print(self.is_softmaxed(softmaxed_label))
            #print(self.is_softmaxed(registered_seg))
#
            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(1, 4)
            #for o in range(4):
            #    ax[o].imshow(registered_seg[0, o].cpu(), cmap='gray')
            #plt.show()

            current_input = self.get_input_query(x[0], 
                                           x[t - 1], 
                                           x[t], 
                                           softmaxed_label,
                                           registered_seg, 
                                           cumulated_forward, 
                                           error_forward, 
                                           registered_forward, 
                                           cumulated_backward,
                                           error_backward, 
                                           registered_backward)
            
            current_feature, skip_co_current = self.query_encoder(current_input) # B, C, H, W

            if self.gaussian:
                max_idx = self.get_max_index(past_query_list[-1], current_feature)
                max_indices_list.append(max_idx)
                max_idx = torch.stack(max_indices_list, dim=0) # T, B, H*W
            else:
                max_idx = None

            past_features = torch.stack(past_feature_list, dim=0) # T, B, C, H, W
            past_query = torch.stack(past_query_list, dim=0) # T, B, C, H, W
            
            if not self.training:
                past_features = torch.cat([past_features[:2], past_features[2:-1:step], past_features[-1][None]], dim=0)
                past_query = torch.cat([past_query[:2], past_query[2:-1:step], past_query[-1][None]], dim=0)

            current_feature, weights = self.bottleneck(query=current_feature, 
                                                       key=past_query, 
                                                       value=past_features, 
                                                       max_idx=max_idx)

            acc_skip_co = []
            for s in range(self.num_stages):
                concatenated = self.skip_co_reduction_list[s](skip_co_current[s])
                acc_skip_co.append(concatenated)

            flow, intermediary = self.flow_decoder(current_feature, acc_skip_co)
            cumulated_forward = cumulated_forward + flow
            out['forward_flow'].append(cumulated_forward)

            flow, intermediary = self.flow_decoder_backward(current_feature, acc_skip_co)
            cumulated_backward = cumulated_backward + flow
            out['backward_flow'].append(cumulated_backward)

            registered_backward = self.motion_estimation(flow=cumulated_backward, original=x[t])
            error_backward = x[0] - registered_backward

            registered_forward = self.motion_estimation(flow=cumulated_forward, original=x[0])
            error_forward = x[t] - registered_forward

            registered_seg, softmaxed_label = self.transform_label(cumulated_forward, label)

            #if self.training:
            #    assert torch.all(registered_seg == softmaxed_label)

            past_input = self.get_input_memory(x[0],
                                           x[t],
                                           registered_seg,
                                           cumulated_forward, 
                                           error_forward, 
                                           registered_forward,
                                           cumulated_backward,
                                           error_backward,
                                           registered_backward)

            past_new, past_skip_co = self.memory_encoder(past_input)
            memory_skip_co_list.append(past_skip_co)

            past_query_list.append(current_feature)
            past_feature_list.append(past_new)
        
        out['forward_flow'] = torch.stack(out['forward_flow'], dim=0)
        out['backward_flow'] = torch.stack(out['backward_flow'], dim=0)
        out['weights'] = weights

        return out
    



    def forward_multi_task_flow_no_residual(self, x, label, step=1):
        out = {'forward_flow': [], 'backward_flow': [], 'weights': []}
        T, B, C, H, W = x.shape
        flow = torch.zeros(size=(B, 2, H, W), device=x.device)
        cumulated_forward = torch.zeros(size=(B, 2, H, W), device=x.device)
        cumulated_backward = torch.zeros(size=(B, 2, H, W), device=x.device)

        past_feature_list = []
        max_indices_list = []
        past_query_list = []
        memory_skip_co_list = []

        registered_backward = self.motion_estimation(flow=cumulated_backward, original=x[0])
        error_backward = x[0] - registered_backward

        registered_forward = self.motion_estimation(flow=cumulated_forward, original=x[0])
        error_forward = x[0] - registered_forward

        registered_seg, softmaxed_label = self.transform_label(cumulated_forward, label)

        current_input = self.get_input_query(x[0], 
                                       x[0], 
                                       x[0], 
                                       softmaxed_label,
                                       registered_seg, 
                                       cumulated_forward, 
                                       error_forward, 
                                       registered_forward, 
                                       cumulated_backward,
                                       error_backward, 
                                       registered_backward)
        
        current_feature, skip_co_current = self.query_encoder(current_input) # B, C, H, W
        past_query_list.append(current_feature)

        past_input = self.get_input_memory(x[0],
                                           x[0],
                                           registered_seg,
                                           cumulated_forward, 
                                           error_forward, 
                                           registered_forward,
                                           cumulated_backward,
                                           error_backward,
                                           registered_backward)

        past_new, past_skip_co = self.memory_encoder(past_input)

        memory_skip_co_list.append(past_skip_co)
        past_feature_list.append(past_new)

        for t in range(1, T):

            #print(self.is_softmaxed(softmaxed_label))
            #print(self.is_softmaxed(registered_seg))
#
            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(1, 4)
            #for o in range(4):
            #    ax[o].imshow(registered_seg[0, o].cpu(), cmap='gray')
            #plt.show()

            current_input = self.get_input_query(x[0], 
                                           x[0], 
                                           x[t], 
                                           softmaxed_label,
                                           registered_seg, 
                                           cumulated_forward, 
                                           error_forward, 
                                           registered_forward, 
                                           cumulated_backward,
                                           error_backward, 
                                           registered_backward)
            
            current_feature, skip_co_current = self.query_encoder(current_input) # B, C, H, W

            if self.gaussian:
                max_idx = self.get_max_index(past_query_list[-1], current_feature)
                max_indices_list.append(max_idx)
                max_idx = torch.stack(max_indices_list, dim=0) # T, B, H*W
            else:
                max_idx = None

            past_features = torch.stack(past_feature_list, dim=0) # T, B, C, H, W
            past_query = torch.stack(past_query_list, dim=0) # T, B, C, H, W
            
            if not self.training:
                past_features = torch.cat([past_features[:2], past_features[2:-1:step], past_features[-1][None]], dim=0)
                past_query = torch.cat([past_query[:2], past_query[2:-1:step], past_query[-1][None]], dim=0)

            current_feature, weights = self.bottleneck(query=current_feature, 
                                                       key=past_query, 
                                                       value=past_features, 
                                                       max_idx=max_idx)

            acc_skip_co = []
            for s in range(self.num_stages):
                concatenated = torch.cat([skip_co_current[s], self.dropout_layer(memory_skip_co_list[-1][s]), self.dropout_layer(memory_skip_co_list[0][s])], dim=1)
                concatenated = self.skip_co_reduction_list[s](concatenated)
                acc_skip_co.append(concatenated)

            cumulated_forward, intermediary = self.flow_decoder(current_feature, acc_skip_co)
            out['forward_flow'].append(cumulated_forward)

            cumulated_backward, intermediary = self.flow_decoder_backward(current_feature, acc_skip_co)
            out['backward_flow'].append(cumulated_backward)

            registered_backward = self.motion_estimation(flow=cumulated_backward, original=x[t])
            error_backward = x[0] - registered_backward

            registered_forward = self.motion_estimation(flow=cumulated_forward, original=x[0])
            error_forward = x[t] - registered_forward

            registered_seg, softmaxed_label = self.transform_label(cumulated_forward, label)

            #if self.training:
            #    assert torch.all(registered_seg == softmaxed_label)

            past_input = self.get_input_memory(x[0],
                                           x[t],
                                           registered_seg,
                                           cumulated_forward, 
                                           error_forward, 
                                           registered_forward,
                                           cumulated_backward,
                                           error_backward,
                                           registered_backward)

            past_new, past_skip_co = self.memory_encoder(past_input)
            memory_skip_co_list.append(past_skip_co)

            past_query_list.append(current_feature)
            past_feature_list.append(past_new)
        
        out['forward_flow'] = torch.stack(out['forward_flow'], dim=0)
        out['backward_flow'] = torch.stack(out['backward_flow'], dim=0)
        out['weights'] = weights

        return out
    
    

    def forward_multi_task_flow_no_label(self, x, step=1):
        out = {'forward_flow': [], 'backward_flow': [], 'weights': []}
        T, B, C, H, W = x.shape
        flow = torch.zeros(size=(B, 2, H, W), device=x.device)
        cumulated_forward = torch.zeros(size=(B, 2, H, W), device=x.device)
        cumulated_backward = torch.zeros(size=(B, 2, H, W), device=x.device)

        past_feature_list = []
        max_indices_list = []
        past_query_list = []
        memory_skip_co_list = []

        registered_backward = self.motion_estimation(flow=cumulated_backward, original=x[0])
        error_backward = x[0] - registered_backward

        registered_forward = self.motion_estimation(flow=cumulated_forward, original=x[0])
        error_forward = x[0] - registered_forward

        #current_input = torch.cat([x[0], x[0]], dim=1)
        current_input = x[0]
        
        current_feature, skip_co_current = self.query_encoder(current_input) # B, C, H, W
        past_query_list.append(current_feature)

        past_input = torch.cat([x[0], 
                                x[0], 
                                cumulated_forward, 
                                error_forward, 
                                registered_forward, 
                                cumulated_backward, 
                                error_backward, 
                                registered_backward], dim=1)

        past_new, past_skip_co = self.memory_encoder(past_input)

        memory_skip_co_list.append(past_skip_co)
        past_feature_list.append(past_new)

        for t in range(1, T):

            #print(self.is_softmaxed(softmaxed_label))
            #print(self.is_softmaxed(registered_seg))
#
            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(1, 4)
            #for o in range(4):
            #    ax[o].imshow(registered_seg[0, o].cpu(), cmap='gray')
            #plt.show()

            #current_input = torch.cat([x[t-1], x[t]], dim=1)
            current_input = x[t]
            
            current_feature, skip_co_current = self.query_encoder(current_input) # B, C, H, W

            if self.gaussian:
                max_idx = self.get_max_index(past_query_list[-1], current_feature)
                max_indices_list.append(max_idx)
                max_idx = torch.stack(max_indices_list, dim=0) # T, B, H*W
            else:
                max_idx = None

            past_features = torch.stack(past_feature_list, dim=0) # T, B, C, H, W
            past_query = torch.stack(past_query_list, dim=0) # T, B, C, H, W
            
            if not self.training:
                past_features = torch.cat([past_features[:2], past_features[2:-1:step], past_features[-1][None]], dim=0)
                past_query = torch.cat([past_query[:2], past_query[2:-1:step], past_query[-1][None]], dim=0)

            current_feature, weights = self.bottleneck(query=current_feature, 
                                                       key=past_query, 
                                                       value=past_features, 
                                                       max_idx=max_idx)

            acc_skip_co = []
            for s in range(self.num_stages):
                concatenated = torch.cat([skip_co_current[s], self.dropout_layer(memory_skip_co_list[-1][s]), self.dropout_layer(memory_skip_co_list[0][s])], dim=1)
                concatenated = self.skip_co_reduction_list[s](concatenated)
                acc_skip_co.append(concatenated)

            flow, intermediary = self.flow_decoder(current_feature, acc_skip_co)
            cumulated_forward = cumulated_forward + flow
            out['forward_flow'].append(cumulated_forward)

            flow, intermediary = self.flow_decoder_backward(current_feature, acc_skip_co)
            cumulated_backward = cumulated_backward + flow
            out['backward_flow'].append(cumulated_backward)

            registered_backward = self.motion_estimation(flow=cumulated_backward, original=x[t])
            error_backward = x[0] - registered_backward

            registered_forward = self.motion_estimation(flow=cumulated_forward, original=x[0])
            error_forward = x[t] - registered_forward

            #if self.training:
            #    assert torch.all(registered_seg == softmaxed_label)

            past_input = torch.cat([x[0], 
                                x[t], 
                                cumulated_forward, 
                                error_forward, 
                                registered_forward, 
                                cumulated_backward, 
                                error_backward, 
                                registered_backward], dim=1)

            past_new, past_skip_co = self.memory_encoder(past_input)
            memory_skip_co_list.append(past_skip_co)

            past_query_list.append(current_feature)
            past_feature_list.append(past_new)
        
        out['forward_flow'] = torch.stack(out['forward_flow'], dim=0)
        out['backward_flow'] = torch.stack(out['backward_flow'], dim=0)
        out['weights'] = weights

        return out
    


    def forward_multi_task(self, x, label, step=1):
        out = {'flow': [], 'weights': [], 'seg': []}
        T, B, C, H, W = x.shape

        flow = torch.zeros(size=(B, 2, H, W), device=x.device)
        cumulated = torch.zeros(size=(B, 2, H, W), device=x.device)

        past_feature_list = []
        max_indices_list = []
        past_query_list = []
        memory_skip_co_list = []

        registered = self.motion_estimation(flow=cumulated, original=x[0])
        error = x[0] - registered

        current_input = self.get_input_query(x[0], x[0], label, cumulated, error, registered)
        current_feature, skip_co_current = self.query_encoder(current_input) # B, C, H, W
        past_query_list.append(current_feature)

        past_input = torch.cat([x[0], x[0], label, cumulated, error, registered], dim=1)
        past_new, past_skip_co = self.memory_encoder(past_input)
        past_feature_list.append(past_new)
        memory_skip_co_list.append(past_skip_co)

        for t in range(1, T):
            current_input = self.get_input_query(x[t], x[t - 1], label, cumulated, error, registered)
            current_feature, skip_co_current = self.query_encoder(current_input) # B, C, H, W

            if self.gaussian:
                max_idx = self.get_max_index(past_query_list[-1], current_feature)
                max_indices_list.append(max_idx)
                max_idx = torch.stack(max_indices_list, dim=0) # T, B, H*W
            else:
                max_idx = None

            past_features = torch.stack(past_feature_list, dim=0) # T, B, C, H, W
            past_query = torch.stack(past_query_list, dim=0) # T, B, C, H, W
            
            if not self.training:
                past_features = torch.cat([past_features[:2], past_features[2:-1:step], past_features[-1][None]], dim=0)
                past_query = torch.cat([past_query[:2], past_query[2:-1:step], past_query[-1][None]], dim=0)

            current_feature, weights = self.bottleneck(query=current_feature, 
                                                       key=past_query, 
                                                       value=past_features, 
                                                       max_idx=max_idx)

            acc_skip_co = []
            for s in range(self.num_stages):
                concatenated = torch.cat([skip_co_current[s], memory_skip_co_list[-1][s], memory_skip_co_list[0][s]], dim=1)
                concatenated = self.skip_co_reduction_list[s](concatenated)
                acc_skip_co.append(concatenated)

            flow, _ = self.flow_decoder(current_feature, acc_skip_co)
            cumulated = cumulated + flow
            out['flow'].append(cumulated)

            seg, _ = self.seg_decoder(current_feature, acc_skip_co)
            out['seg'].append(seg)

            label = torch.softmax(seg, dim=1)

            registered = self.motion_estimation(flow=cumulated, original=x[0])
            error = x[t] - registered

            past_input = torch.cat([x[t], x[0], label, cumulated, error, registered], dim=1)
            past_new, skip_co_past_new = self.memory_encoder(past_input)
            memory_skip_co_list.append(skip_co_past_new)

            past_query_list.append(current_feature)
            past_feature_list.append(past_new)
        
        out['seg'] = torch.stack(out['seg'], dim=0)
        out['flow'] = torch.stack(out['flow'], dim=0)
        out['weights'] = weights

        return out
    


    def forward_two_memory(self, x, step=1):
        out = {'flow': [], 'weights': []}
        T, B, C, H, W = x.shape
        flow = torch.zeros(size=(B, 2, H, W), device=x.device)
        cumulated = torch.zeros(size=(B, 2, H, W), device=x.device)

        current_feature = torch.zeros(size=(B, self.d_model, int(self.image_size / 2**(self.num_stages)), int(self.image_size / 2**(self.num_stages))), device=x.device)
        past_new = torch.zeros(size=(B, self.d_model, int(self.image_size / 2**(self.num_stages)), int(self.image_size / 2**(self.num_stages))), device=x.device)
        skip_co_past_new = [torch.zeros(size=(B, dim, int(H//(2**i)), int(W//(2**i))), device=x.device) for i, dim in enumerate(self.out_encoder_dims)]

        past_query_list = [current_feature]
        past_feature_list = [past_new]
        past_skip_co_list = [skip_co_past_new]
        max_indices_list = []
        for t in range(1, T):
            
            current_input = torch.cat([x[t], x[t - 1]], dim=1)
            current_feature, skip_co_current = self.encoder_current(current_input) # B, C, H, W

            if self.gaussian:
                max_idx = self.get_max_index(past_query_list[-1], current_feature)
                max_indices_list.append(max_idx)
                max_idx = torch.stack(max_indices_list, dim=0) # T, B, H*W
            else:
                max_idx = None

            past_features = torch.stack(past_feature_list, dim=0) # T, B, C, H, W
            past_query = torch.stack(past_query_list, dim=0) # T, B, C, H, W
            
            if not self.training:
                past_features = torch.cat([past_features[:2], past_features[2:-1:step], past_features[-1][None]], dim=0)
                past_query = torch.cat([past_query[:2], past_query[2:-1:step], past_query[-1][None]], dim=0)

            current_feature, weights = self.bottleneck(query=current_feature, 
                                                       key=past_query, 
                                                       value=past_features, 
                                                       max_idx=max_idx)

            acc_skip_co = []
            for s in range(self.num_stages):
                concatenated = torch.cat([skip_co_current[s], past_skip_co_list[-1][s]], dim=1)
                concatenated = self.skip_co_reduction_list[s](concatenated)
                acc_skip_co.append(concatenated)

            #flow, intermediary = self.flow_decoder(current_feature, skip_co_current)
            flow, intermediary = self.flow_decoder(current_feature, acc_skip_co)
            cumulated = cumulated + flow
            out['flow'].append(cumulated)

            registered = self.motion_estimation(flow=cumulated, original=x[t])
            error = x[0] - registered

            if self.nb_inputs == 'big':
                past_input = torch.cat([x[t], x[t - 1], x[0], cumulated, flow, registered, error], dim=1)
            elif self.nb_inputs == 'small':
                past_input = torch.cat([x[t], x[t - 1], flow], dim=1)
            else:
                past_input = torch.cat([x[t], x[0], cumulated, registered, error], dim=1)
            past_new, skip_co_past_new = self.encoder_past(past_input)
            past_skip_co_list.append(skip_co_past_new)

            past_query_list.append(current_feature)
            past_feature_list.append(past_new)
        
        out['flow'] = torch.stack(out['flow'], dim=0)
        out['weights'] = weights

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



    def _internal_maybe_mirror_and_pred_2D_old(self, unlabeled, target, processor, mirror_axes: tuple,
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
            mean_centroid, _ = processor.preprocess_no_registration(data=torch.clone(unlabeled[:, 0])) # T, C(1), H, W

            cropped_unlabeled, padding_need = processor.crop_and_pad(data=unlabeled[:, 0], mean_centroid=mean_centroid)
            cropped_target, _ = processor.crop_and_pad(data=target[:, 0], mean_centroid=mean_centroid)
            padding_need = padding_need[None]
            cropped_unlabeled = cropped_unlabeled[:, None]
            cropped_target = cropped_target[:, None]
        
        cropped_unlabeled[:, 0] = NormalizeIntensity()(cropped_unlabeled[:, 0])

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
                seg_pred = torch.softmax(output['seg'], dim=2)
                flow_pred = output['global_motion_forward']

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

                result_torch_seg += 1 / num_results * seg_pred
                result_torch_flow = flow_pred
                #result_torch_flow += 1 / num_results * flow_pred

            if m == 1 and (1 in mirror_axes):
                cropped_unlabeled_flipped = torch.flip(cropped_unlabeled, (4, ))
                output = self(cropped_unlabeled_flipped)
                seg_pred = torch.softmax(output['seg'], dim=2)
                flow_pred = output['global_motion_forward']

                #step = int(len(flow_pred) / len(target))
                #flow_pred = flow_pred[::step]

                #matplotlib.use('QtAgg')
                #fig, ax = plt.subplots(1, 3)
                #ax[0].imshow(cropped_unlabeled_flipped[0, 0, 0].cpu(), cmap='gray')
                #ax[1].imshow(torch.argmax(seg_pred[0, 0], dim=0).cpu(), cmap='gray')
                #ax[2].imshow(torch.argmax(torch.flip(seg_pred, (4, ))[0, 0], dim=0).cpu(), cmap='gray')
                #plt.show()
                #plt.waitforbuttonpress()
                #plt.close(fig)

                result_torch_seg += 1 / num_results * torch.flip(seg_pred, (4, ))
                #result_torch_flow += 1 / num_results * torch.flip(flow_pred, (4, ))

            if m == 2 and (0 in mirror_axes):
                cropped_unlabeled_flipped = torch.flip(cropped_unlabeled, (3, ))
                output = self(cropped_unlabeled_flipped)
                seg_pred = torch.softmax(output['seg'], dim=2)
                flow_pred = output['global_motion_forward']

                #step = int(len(flow_pred) / len(target))
                #flow_pred = flow_pred[::step]

                #matplotlib.use('QtAgg')
                #fig, ax = plt.subplots(1, 3)
                #ax[0].imshow(cropped_unlabeled_flipped[0, 0, 0].cpu(), cmap='gray')
                #ax[1].imshow(torch.argmax(seg_pred[0, 0], dim=0).cpu(), cmap='gray')
                #ax[2].imshow(torch.argmax(torch.flip(seg_pred, (3, ))[0, 0], dim=0).cpu(), cmap='gray')
                #plt.show()
                #plt.waitforbuttonpress()
                #plt.close(fig)

                result_torch_seg += 1 / num_results * torch.flip(seg_pred, (3, ))
                #result_torch_flow += 1 / num_results * torch.flip(flow_pred, (3, ))

            if m == 3 and (0 in mirror_axes) and (1 in mirror_axes):
                cropped_unlabeled_flipped = torch.flip(cropped_unlabeled, (4, 3))
                output = self(cropped_unlabeled_flipped)
                seg_pred = torch.softmax(output['seg'], dim=2)
                flow_pred = output['global_motion_forward']

                #step = int(len(flow_pred) / len(target))
                #flow_pred = flow_pred[::step]

                #matplotlib.use('QtAgg')
                #fig, ax = plt.subplots(1, 3)
                #ax[0].imshow(cropped_unlabeled_flipped[0, 0, 0].cpu(), cmap='gray')
                #ax[1].imshow(torch.argmax(seg_pred[0, 0], dim=0).cpu(), cmap='gray')
                #ax[2].imshow(torch.argmax(torch.flip(seg_pred, (4, 3))[0, 0], dim=0).cpu(), cmap='gray')
                #plt.show()
                #plt.waitforbuttonpress()
                #plt.close(fig)

                result_torch_seg += 1 / num_results * torch.flip(seg_pred, (4, 3))
                #result_torch_flow += 1 / num_results * torch.flip(flow_pred, (4, 3))
        
        #matplotlib.use('QtAgg')
        #fig, ax = plt.subplots(1, 1)
        #ax.imshow(torch.argmax(result_torch_seg[0, 0], dim=0).cpu(), cmap='gray')
        #plt.show()
        #plt.waitforbuttonpress()
        #plt.close(fig)
        
        return result_torch_seg, result_torch_flow, cropped_target, padding_need




    def _internal_maybe_mirror_and_pred_2D(self, unlabeled, target, processor, mirror_axes: tuple,
                                           do_mirroring: bool = True) -> torch.tensor:
        # if cuda available:
        #   everything in here takes place on the GPU. If x and mult are not yet on GPU this will be taken care of here
        #   we now return a cuda tensor! Not numpy array!

        assert len(unlabeled[0].shape) == 4, 'x must be (b, c, x, y)'
        unlabeled = maybe_to_torch(unlabeled)
        target = maybe_to_torch(target)

        unlabeled = to_cuda(unlabeled, gpu_id=self.get_device())
        
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
            mean_centroid, _ = processor.preprocess_no_registration(data=torch.clone(unlabeled[:, 0])) # T, C(1), H, W

            cropped_unlabeled, padding_need = processor.crop_and_pad(data=unlabeled[:, 0], mean_centroid=mean_centroid)
            cropped_target, _ = processor.crop_and_pad(data=target[:, 0], mean_centroid=mean_centroid)
            padding_need = padding_need[None]
            cropped_unlabeled = cropped_unlabeled[:, None]
            cropped_target = cropped_target[:, None]
        
        cropped_unlabeled[:, 0] = NormalizeIntensity()(cropped_unlabeled[:, 0])

        #cropped_unlabeled = cropped_unlabeled.permute(1, 2, 0, 3, 4).contiguous()

        #print(cropped_unlabeled.shape)
        #matplotlib.use('QtAgg')
        #fig, ax = plt.subplots(1, 1)
        #ax.imshow(cropped_unlabeled[0, 0, 0].cpu(), cmap='gray')
        #plt.show()
        #plt.waitforbuttonpress()
        #plt.close(fig)

        indices = torch.arange(1, len(cropped_unlabeled))
        chunk1, chunk2 = torch.chunk(indices, 2)

        chunk1 = torch.cat([torch.tensor([0]), chunk1])
        chunk2 = torch.flip(chunk2, dims=[0])
        chunk2 = torch.cat([torch.tensor([0]), chunk2])

        chunk_list_seg = []
        chunk_list_flow = []

        for chunk_idx, chunk in enumerate([chunk1, chunk2]):

            chunk_input = cropped_unlabeled[chunk]

            result_torch_seg = torch.zeros(list(chunk_input.shape[:2]) + [self.num_classes] + [processor.crop_size, processor.crop_size], dtype=torch.float)
            result_torch_flow = torch.zeros(list(chunk_input.shape[:2]) + [2] + [processor.crop_size, processor.crop_size], dtype=torch.float)
            if torch.cuda.is_available():
                #target = to_cuda(target, gpu_id=self.get_device())
                result_torch_seg = result_torch_seg.cuda(self.get_device(), non_blocking=True)
                result_torch_flow = result_torch_flow.cuda(self.get_device(), non_blocking=True)

            for m in range(mirror_idx):
                if m == 0:
                    output = self(chunk_input)
                    seg_pred = torch.softmax(output['seg'], dim=2)
                    flow_pred = output['global_motion_forward']

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

                    result_torch_seg += 1 / num_results * seg_pred
                    result_torch_flow = flow_pred
                    #result_torch_flow += 1 / num_results * flow_pred

                if m == 1 and (1 in mirror_axes):
                    chunk_input_flipped = torch.flip(chunk_input, (4, ))
                    output = self(chunk_input_flipped)
                    seg_pred = torch.softmax(output['seg'], dim=2)
                    flow_pred = output['global_motion_forward']

                    #step = int(len(flow_pred) / len(target))
                    #flow_pred = flow_pred[::step]

                    #matplotlib.use('QtAgg')
                    #fig, ax = plt.subplots(1, 3)
                    #ax[0].imshow(cropped_unlabeled_flipped[0, 0, 0].cpu(), cmap='gray')
                    #ax[1].imshow(torch.argmax(seg_pred[0, 0], dim=0).cpu(), cmap='gray')
                    #ax[2].imshow(torch.argmax(torch.flip(seg_pred, (4, ))[0, 0], dim=0).cpu(), cmap='gray')
                    #plt.show()
                    #plt.waitforbuttonpress()
                    #plt.close(fig)

                    result_torch_seg += 1 / num_results * torch.flip(seg_pred, (4, ))
                    #result_torch_flow += 1 / num_results * torch.flip(flow_pred, (4, ))

                if m == 2 and (0 in mirror_axes):
                    chunk_input_flipped = torch.flip(chunk_input, (3, ))
                    output = self(chunk_input_flipped)
                    seg_pred = torch.softmax(output['seg'], dim=2)
                    flow_pred = output['global_motion_forward']

                    #step = int(len(flow_pred) / len(target))
                    #flow_pred = flow_pred[::step]

                    #matplotlib.use('QtAgg')
                    #fig, ax = plt.subplots(1, 3)
                    #ax[0].imshow(cropped_unlabeled_flipped[0, 0, 0].cpu(), cmap='gray')
                    #ax[1].imshow(torch.argmax(seg_pred[0, 0], dim=0).cpu(), cmap='gray')
                    #ax[2].imshow(torch.argmax(torch.flip(seg_pred, (3, ))[0, 0], dim=0).cpu(), cmap='gray')
                    #plt.show()
                    #plt.waitforbuttonpress()
                    #plt.close(fig)

                    result_torch_seg += 1 / num_results * torch.flip(seg_pred, (3, ))
                    #result_torch_flow += 1 / num_results * torch.flip(flow_pred, (3, ))

                if m == 3 and (0 in mirror_axes) and (1 in mirror_axes):
                    chunk_input_flipped = torch.flip(chunk_input, (4, 3))
                    output = self(chunk_input_flipped)
                    seg_pred = torch.softmax(output['seg'], dim=2)
                    flow_pred = output['global_motion_forward']

                    #step = int(len(flow_pred) / len(target))
                    #flow_pred = flow_pred[::step]

                    #matplotlib.use('QtAgg')
                    #fig, ax = plt.subplots(1, 3)
                    #ax[0].imshow(cropped_unlabeled_flipped[0, 0, 0].cpu(), cmap='gray')
                    #ax[1].imshow(torch.argmax(seg_pred[0, 0], dim=0).cpu(), cmap='gray')
                    #ax[2].imshow(torch.argmax(torch.flip(seg_pred, (4, 3))[0, 0], dim=0).cpu(), cmap='gray')
                    #plt.show()
                    #plt.waitforbuttonpress()
                    #plt.close(fig)

                    result_torch_seg += 1 / num_results * torch.flip(seg_pred, (4, 3))
                    #result_torch_flow += 1 / num_results * torch.flip(flow_pred, (4, 3))
            
            chunk_list_seg.append(result_torch_seg)
            chunk_list_flow.append(result_torch_flow)
        
        assert len(chunk_list_flow) == 2

        result_torch_seg = torch.cat([chunk_list_seg[0], torch.flip(chunk_list_seg[1][1:], dims=[0])], dim=0)
        result_torch_flow = torch.cat([chunk_list_flow[0], torch.flip(chunk_list_flow[1][1:], dims=[0])], dim=0)

        assert len(result_torch_seg) == len(result_torch_flow) == len(unlabeled)
        
        #matplotlib.use('QtAgg')
        #fig, ax = plt.subplots(1, 1)
        #ax.imshow(torch.argmax(result_torch_seg[0, 0], dim=0).cpu(), cmap='gray')
        #plt.show()
        #plt.waitforbuttonpress()
        #plt.close(fig)
        
        return result_torch_seg, result_torch_flow, cropped_target, padding_need, cropped_unlabeled
    

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

        predicted_segmentation = []
        softmax_pred_list = []
        flow_pred_list = []
        registered_pred_list = []
        raw_flow_list = []

        for depth_idx in range(unlabeled.shape[2]):

            #current_video = [x[:, depth_idx] for x in frame_list]

            current_unlabeled = unlabeled[:, :, depth_idx]
            current_target = target[:, :, depth_idx] if target is not None else None

            pred_seg, softmax_pred, flow_pred, registered_pred, raw_flow = self._internal_predict_2D_2Dconv_tiled_flow(
                current_unlabeled, current_target, target_mask, processor, step_size, do_mirroring, mirror_axes, patch_size, regions_class_order, use_gaussian,
                pad_border_mode, pad_kwargs, all_in_gpu, verbose)

            predicted_segmentation.append(pred_seg[None])
            softmax_pred_list.append(softmax_pred[None])
            flow_pred_list.append(flow_pred[None])
            registered_pred_list.append(registered_pred[None])
            raw_flow_list.append(raw_flow[None])

        predicted_segmentation = np.vstack(predicted_segmentation).transpose((1, 0, 2, 3)) # T, depth, H, W
        flow_pred = np.vstack(flow_pred_list).transpose((1, 2, 0, 3, 4)) # T, C, depth, H, W
        softmax_pred = np.vstack(softmax_pred_list).transpose((1, 2, 0, 3, 4)) # T, C, depth, H, W
        registered_pred = np.vstack(registered_pred_list).transpose((1, 2, 0, 3, 4)) # T_other, C, depth, H, W
        raw_flow_pred = np.vstack(raw_flow_list).transpose((1, 2, 0, 3, 4)) # T_other, C, depth, H, W

        return predicted_segmentation, softmax_pred, flow_pred, registered_pred, raw_flow_pred


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
        unlabeled_shape = unlabeled_data.shape
        if target is not None:
            target_data, target_slicer = pad_nd_image(target, patch_size, pad_border_mode, pad_kwargs, True, None)

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

        unlabeled_data_input = unlabeled_data[:, None, :, y1:y2, x1:x2]
        target_input_query = target_data[:, None, :, y1:y2, x1:x2] if target is not None else None

        seg, flow, target, padding_need, cropped_input = self._internal_maybe_mirror_and_pred_2D(unlabeled_data_input, target_input_query, processor, mirror_axes, do_mirroring)
        #seg, flow, target, padding_need = self._internal_maybe_mirror_and_pred_2D(unlabeled_data_input, target_input_query, processor, mirror_axes, do_mirroring)

        #flow = torch.nn.functional.pad(flow, pad=(0, 0, 0, 0, 0, 0, 0, 0, 1, 0))

        raw_flow = torch.cat([cropped_input, flow], dim=2).squeeze(1)

        assert nb_frames == len(seg) == len(flow)
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

        all_motions = self.warp_linear(flow, target)

        #if target is not None:
        #    ed_index = np.where(target_mask)[0][0]
        #    assert ed_index == 0
        #    all_motions = self.warp_linear(flow, target)
        #else:
        #    all_motions = self.warp_linear(flow, torch.argmax(seg, dim=2, keepdim=True))

        #delta_list = []
        #for t in range(len(flow)):
        #    delta = self.motion_estimation.get_delta(flow[t])
        #    delta_list.append(delta)
        #flow = torch.stack(delta_list, dim=0)

        assert len(all_motions) == nb_frames
        #for t in range(indices[1].item(), indices[0].item(), -1):
        #    current_motion = self.motion_estimation(flow=flow[t], original=current_motion, mode='nearest')

        flow = flow.permute(1, 0, 2, 3, 4).contiguous()
        seg = seg.permute(1, 0, 2, 3, 4).contiguous()
        all_motions = all_motions.permute(1, 0, 2, 3, 4).contiguous()

        flow = processor.uncrop_no_registration(flow, padding_need)[0]
        seg = processor.uncrop_no_registration(seg, padding_need)[0]
        all_motions = processor.uncrop_no_registration(all_motions, padding_need)[0]

        if all_in_gpu:
            flow = flow.half()
            seg = seg.half()
            all_motions = all_motions.half()
            raw_flow = raw_flow.half()
        else:
            flow = flow.cpu().numpy()
            seg = seg.cpu().numpy()
            all_motions = all_motions.cpu().numpy()
            raw_flow = raw_flow.cpu().numpy()
        
        aggregated_results_flow[:, :, y1:y2, x1:x2] += flow
        aggregated_results_seg[:, :, y1:y2, x1:x2] += seg
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
        unlabeled_slicer_seg = tuple(
            [slice(0, aggregated_results_seg.shape[i]) for i in
             range(len(aggregated_results_seg.shape) - 2)] + unlabeled_slicer[-2:])
        
        unlabeled_slicer_flow = tuple(
            [slice(0, aggregated_results_flow.shape[i]) for i in
             range(len(aggregated_results_flow.shape) - 2)] + unlabeled_slicer[-2:])
        
        unlabeled_slicer_registered = tuple(
            [slice(0, aggregated_results_registered.shape[i]) for i in
             range(len(aggregated_results_registered.shape) - 2)] + unlabeled_slicer[-2:])
        
        aggregated_results_seg = aggregated_results_seg[unlabeled_slicer_seg]
        aggregated_results_flow = aggregated_results_flow[unlabeled_slicer_flow]
        aggregated_results_registered = aggregated_results_registered[unlabeled_slicer_registered]
        #aggregated_nb_of_predictions_seg = aggregated_nb_of_predictions_seg[unlabeled_slicer_seg]

        # computing the class_probabilities by dividing the aggregated result with result_numsamples
        class_probabilities = aggregated_results_seg
        #class_probabilities = aggregated_results_seg / aggregated_nb_of_predictions_seg
        T, C, H, W = class_probabilities.shape

        if regions_class_order is None:
            predicted_segmentation = class_probabilities.argmax(1) # H, W --> T, H, W
        else:
            if all_in_gpu:
                class_probabilities_here = class_probabilities.detach().cpu().numpy()
            else:
                class_probabilities_here = class_probabilities
            predicted_segmentation = np.zeros((T, H, W), dtype=np.float32)
            for i, c in enumerate(regions_class_order):
                predicted_segmentation[class_probabilities_here[:, i] > 0.5] = c

        if all_in_gpu:
            if verbose: print("copying results to CPU")

            if regions_class_order is None:
                predicted_segmentation = predicted_segmentation.detach().cpu().numpy()

            class_probabilities = class_probabilities.detach().cpu().numpy()

        if verbose: print("prediction done")
        return predicted_segmentation, class_probabilities, aggregated_results_flow, aggregated_results_registered, raw_flow
    

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
    
    
    def warp_linear(self, flow, target):
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