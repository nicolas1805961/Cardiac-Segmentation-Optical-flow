import torch
import torch.nn as nn
import torch.nn.functional as F
from .position_embedding import PositionEmbeddingSine2d, PositionEmbeddingSine1d, PositionEmbeddingSine3d, PositionEmbeddingSine2dDeformable
from torch.nn.utils.parametrizations import spectral_norm
from einops import rearrange
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from math import log, sqrt
from typing import Optional
from torch import Tensor
import copy
import matplotlib.patches as patches
from torch.nn.parameter import Parameter

from timm.models.layers import to_2tuple

from .seresnet import ResnetBasicBlock1D, RescaleBasicBlock, RescaleBasicBlock1D, RescaleBasicBlock3D, SEBasicBlock_3d, ECABasicBlock, rescale_layer, DiscriminatorECABasicBlock, AdaINECABasicBlock, ResnetBasicBlock, rescale_layer_3d

import logging
import matplotlib

from mmcv.utils import get_logger
import sys
from torch.nn.functional import affine_grid
from torch.nn.functional import grid_sample

class MotionEstimation(nn.Module):
    def __init__(self):
        super().__init__()
    
    def generate_grid(self, x, offset):
        # x only provides shape parameters to the model
        x_shape = x.size()
        grid_w, grid_h = torch.meshgrid([torch.linspace(-1, 1, x_shape[2]), torch.linspace(-1, 1, x_shape[3])])  # (h, w)
        grid_w = grid_w.cuda().float()
        grid_h = grid_h.cuda().float()

        grid_w = nn.Parameter(grid_w, requires_grad=False)
        grid_h = nn.Parameter(grid_h, requires_grad=False)

        offset_h, offset_w = torch.split(offset, 1, 1)
        offset_w = offset_w.contiguous().view(-1, int(x_shape[2]), int(x_shape[3]))  # (b*c, h, w)
        offset_h = offset_h.contiguous().view(-1, int(x_shape[2]), int(x_shape[3]))  # (b*c, h, w)

        offset_w = grid_w + offset_w
        offset_h = grid_h + offset_h

        offsets = torch.stack((offset_h, offset_w), 3) # 1 x img_size x img_size x2
        # print(offsets.shape) 
        return offsets

    def forward(self, flow, original, mode='bilinear'):
        grid = self.generate_grid(original, flow)
        return grid_sample(original, grid, mode=mode)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class CrossTransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.0,
                 activation="gelu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     query, key, value,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None,
                     key_pos: Optional[Tensor] = None):
        src = query
        query = self.with_pos_embed(query, query_pos)
        key = self.with_pos_embed(key, key_pos)
        src2, weights = self.self_attn(query, key, value=value, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, weights

    def forward(self, query, key, value,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                key_pos: Optional[Tensor] = None):
        return self.forward_post(query, key, value, src_mask, src_key_padding_mask, query_pos=query_pos, key_pos=key_pos)


class ZoomAttention(nn.Module):
    def __init__(self, dim, n_heads, nb_zones, video_length, area_size, d_model, n_points, res):
        super(ZoomAttention, self).__init__()
        self.n_points = n_points
        self.area_size = area_size
        self.n_heads = n_heads
        self.nb_zones = nb_zones
        self.head_dim = dim // n_heads
        #self.centers = nn.Linear(dim, 2*nb_zones)
        self.centers = nn.Linear(d_model, 2*nb_zones)
        #self.centers = SpatialTransformerNetwork(in_dim=dim, norm=norm, area_size=area_size, input_resolution=input_resolution, nb_zones=nb_zones)
        #self.skip_co_proj = nn.Linear(dim, dim)
        #self.output_proj1 = nn.Linear(dim, dim)
        #self.output_proj2 = nn.Linear(dim, dim)
        self.norm1 = nn.InstanceNorm2d(dim)
        self.norm2 = nn.InstanceNorm2d(dim)
        self.pos_2d = nn.Parameter(torch.randn(dim, res, res))
        #self.pos_2d = PositionEmbeddingSine2dDeformable(num_pos_feats=dim // 2, normalize=True)
        #self.temperature = Parameter(torch.ones(size=(1,)))
        #self.cross_transformer = CrossTransformerEncoderLayer(d_model=dim, nhead=n_heads, dim_feedforward=4*dim)
        self.conv_weight_getter = nn.Conv2d(in_channels=self.nb_zones+1, out_channels=1, kernel_size=7, padding='same')

        #self.sampling_offsets = nn.Linear(dim, n_heads * video_length * n_points * 2)
        #self.attention_weights = nn.Linear(dim, n_heads * video_length * n_points)
        self.sampling_offsets = nn.Conv2d(dim, n_heads * video_length * n_points * 2, kernel_size=3, padding='same')
        self.attention_weights = nn.Conv2d(dim, n_heads * video_length * n_points, kernel_size=3, padding='same')
        self.skip_co_proj = nn.Linear(dim, dim)
        self.output_proj = nn.Linear(dim, dim)
        self.output_proj_zones = nn.Linear(dim, dim)

    
    #def get_zones(self, query, key):
    #    # Get zones
    #    T, B, C, H, W = key.shape
#
    #    query = query.permute(0, 2, 1).view(B, C, H, W)
    #    theta, theta_coords = self.centers(query) # B, n_heads, 2, 3
#
    #    query = query.unsqueeze(1).repeat(1, self.n_heads, 1, 1, 1).view(B * self.n_heads, C, H, W)
    #    theta = theta.view(B * self.n_heads, 2, 3)
    #    grid = affine_grid(theta, size=(B * self.n_heads, C, self.area_size, self.area_size))
    #    query_zones = F.grid_sample(query, grid, mode='nearest') # B*n_heads, C, area_size, area_size
    #    query_zones = query_zones.permute(0, 2, 3, 1).view(B * self.n_heads, -1, C) # B*n_heads, area_size*area_size, C
#
    #    if self.wide_attention:
    #        key_zones  = key.unsqueeze(2).repeat(1, 1, self.n_heads, 1, 1, 1).view(T, B * self.n_heads, C, H, W)
    #    else:
    #        grid = grid.unsqueeze(0).repeat(T, 1, 1, 1, 1).view(T * B * self.n_heads, self.area_size, self.area_size, 2)
    #        key = key.unsqueeze(2).repeat(1, 1, self.n_heads, 1, 1, 1).view(T * B * self.n_heads, C, H, W)
    #        key_zones = F.grid_sample(key, grid, mode='nearest') # T*B*n_heads, C, area_size, area_size
    #        key_zones = key_zones.view(T, B * self.n_heads, C, self.area_size, self.area_size)
    #    
    #    return query_zones, key_zones, theta, theta_coords.detach()

    def sigmoid(self, x, weight):
        return 1 / (1 + torch.exp(-weight * x))
    
    def min_max_normalization(self, x, prev_min, prev_max, new_min, new_max):
        temp = (x - prev_min) / (prev_max - prev_min)
        return temp * (new_max - new_min) + new_min
    
    def get_centers(self, x, H, W):
        B, C = x.shape
        x = self.centers(x)
        x = torch.tanh(x).view(B, self.nb_zones, 2)
        x = self.min_max_normalization(x, -1, 1, -1 + (1/H), 1 - (1/H))
        x = x.repeat(1, 1, 2)
        x[:, :, 0] = x[:, :, 0] - self.area_size / H
        x[:, :, 1] = x[:, :, 1] - self.area_size / H
        x[:, :, 2] = x[:, :, 2] + self.area_size / H
        x[:, :, 3] = x[:, :, 3] + self.area_size / H

        #theta = torch.zeros(size=(B, self.nb_zones, 2, 3), device=x.device)
        #theta[:, :, 0, 0] = (x[:, :, 2] - x[:, :, 0]) / W
        #theta[:, :, 0, 1] = 0.0
        #theta[:, :, 0, 2] = -1 + (x[:, :, 2] + x[:, :, 0]) / W
        #theta[:, :, 1, 0] = 0.0
        #theta[:, :, 1, 1] = (x[:, :, 3] - x[:, :, 1]) / H
        #theta[:, :, 1, 2] = -1 + (x[:, :, 3] + x[:, :, 1]) / H

        theta = torch.zeros(size=(B, self.nb_zones, 2, 3), device=x.device)
        theta[:, :, 0, 0] = (x[:, :, 2] - x[:, :, 0]) / 2
        theta[:, :, 0, 1] = 0.0
        theta[:, :, 0, 2] = (x[:, :, 2] + x[:, :, 0]) / 2
        theta[:, :, 1, 0] = 0.0
        theta[:, :, 1, 1] = (x[:, :, 3] - x[:, :, 1]) / 2
        theta[:, :, 1, 2] = (x[:, :, 3] + x[:, :, 1]) / 2

        theta_coords = (x + 1) * (H / 2)
        theta_coords = torch.round(theta_coords)

        return theta, theta_coords


    def get_uncropped_images(self, zones, theta, H, W, downsample=False):
        inv_theta = torch.clone(theta)
        inv_theta[:, 0, 0] = 1 / theta[:, 0, 0]
        inv_theta[:, 1, 1] = 1 / theta[:, 1, 1]
        inv_theta[:, 0, 2] = -theta[:, 0, 2] / theta[:, 0, 0]
        inv_theta[:, 1, 2] = -theta[:, 1, 2] / theta[:, 1, 1]

        B, C, _, _ = zones.shape
        inv_grid = torch.nn.functional.affine_grid(inv_theta, size=(B, C, H, W))
        out = torch.nn.functional.grid_sample(zones, inv_grid, mode='nearest') # B, C, H, W


    def get_zones(self, query, memory_bus):
        # Get zones
        B, C, H, W = query.shape

        theta, theta_coords = self.get_centers(memory_bus, H, W) # B, n_zones, 2, 3

        #rects = []
        #theta_coords = theta_coords.cpu()
        #for i in range(self.nb_zones):
        #    rect = patches.Rectangle((theta_coords[0, i, 0], theta_coords[0, i, 1]), theta_coords[0, i, 2] - theta_coords[0, i, 0], theta_coords[0, i, 3] - theta_coords[0, i, 1], linewidth=1, edgecolor='r', facecolor='none')
        #    rects.append(rect)

        theta = theta.view(B * self.nb_zones, 2, 3)
        #grid = affine_grid(theta, size=(B * self.n_heads, C, self.area_size, self.area_size))
        grid = affine_grid(theta, size=(B * self.nb_zones, C, self.area_size, self.area_size))

        #grid = grid.unsqueeze(0).repeat(T, 1, 1, 1, 1).view(T * B * self.n_heads, self.area_size, self.area_size, 2)
        query = query.unsqueeze(1).repeat(1, self.nb_zones, 1, 1, 1).view(B * self.nb_zones, C, H, W)
        out_query_zones = F.grid_sample(query, grid, mode='nearest') # B*n_zones, C, area_size, area_size
        #out_query_zones = query_zones.permute(0, 2, 3, 1).contiguous() # B*n_zones, area_size, area_size, C
        #out_query_zones = out_query_zones.view(B * self.nb_zones, self.area_size * self.area_size, C)

        whole_reference_points = self._get_ref_points(out_query_zones, H, W) # B*nb_zones, H, W, 2
        reference_points = whole_reference_points.permute(0, 3, 1, 2).contiguous() # B*nb_zones, 2, H, W
        reference_points = F.grid_sample(reference_points, grid, mode='nearest') # B*n_zones, 2, area_size, area_size
        out_reference_points = reference_points.permute(0, 2, 3, 1).contiguous() # B*nb_zones, area_size, area_size, 2
        out_reference_points = out_reference_points.view(B * self.nb_zones, self.area_size*self.area_size, 2)

        #pos_2d = self.pos_2d(shape_util=(B * self.nb_zones, H, W), device=query.device)
        pos_2d = self.pos_2d.view(1, C, H, W).repeat(B * self.nb_zones, 1, 1, 1)
        pos_2d = F.grid_sample(pos_2d, grid, mode='nearest') # B*n_zones, C, area_size, area_size
        pos_2d = pos_2d.permute(0, 2, 3, 1).contiguous()
        pos_2d = pos_2d.view(B * self.nb_zones, self.area_size * self.area_size, C)

        #matplotlib.use('QtAgg')
        #fig, ax = plt.subplots(2, 2)
        #ax[0, 0].imshow(key[0, 0].detach().cpu())
        #ax[0, 1].imshow(key_zones[0, 0].detach().cpu())
        #ax[1, 0].imshow(whole_reference_points[0, :, :, 0].detach().cpu())
        #ax[1, 1].imshow(reference_points[0, 0].detach().cpu())
        #for rect in rects:
        #    ax[0, 0].add_patch(copy.copy(rect))
        #    ax[1, 0].add_patch(copy.copy(rect))
        #plt.show()
        
        return out_query_zones, theta, theta_coords.detach(), out_reference_points, pos_2d


    def get_weights(self, x, saved_key):
        """saved_key: B, C, H, W
            x: B, H, C, H, W"""

        _, C, _, _ = saved_key.shape

        saved_key = saved_key.unsqueeze(1)
        x = torch.cat([saved_key, x], dim=1) # B, H+1, C, H, W
        x = x.permute(0, 1, 3, 4, 2) # B, H+1, H, W, C
        x = self.weights_getter(x).squeeze(-1) # B, H+1, H, W

        w = self.conv_weight_getter(x).repeat(1, C, 1, 1)
        w = torch.sigmoid(w)

        return w


    def uncrop_zones(self, theta, zones, saved_key, theta_coords=None):
        #rects = []
        #theta_coords = theta_coords.cpu()
        #for i in range(self.nb_zones):
        #    rect = patches.Rectangle((theta_coords[0, i, 0], theta_coords[0, i, 1]), theta_coords[0, i, 2] - theta_coords[0, i, 0], theta_coords[0, i, 3] - theta_coords[0, i, 1], linewidth=1, edgecolor='r', facecolor='none')
        #    rects.append(rect)

        inv_theta = torch.clone(theta)
        inv_theta[:, 0, 0] = 1 / theta[:, 0, 0]
        inv_theta[:, 1, 1] = 1 / theta[:, 1, 1]
        inv_theta[:, 0, 2] = -theta[:, 0, 2] / theta[:, 0, 0]
        inv_theta[:, 1, 2] = -theta[:, 1, 2] / theta[:, 1, 1]

        B, L, C = zones.shape
        H, W = saved_key.shape[-2:]
        zones = zones.permute(0, 2, 1).view(B, C, self.area_size, self.area_size)
        inv_grid = torch.nn.functional.affine_grid(inv_theta, size=(B, C, H, W))

        #zones = self.norm1(zones)
        #saved_key = self.norm2(saved_key)

        out = torch.nn.functional.grid_sample(zones, inv_grid, mode='nearest') # B, C, H, W
        B = B // self.nb_zones
        out = out.view(B, self.nb_zones, C, H, W)

        out = out.permute(0, 2, 1, 3, 4).contiguous()
        out = out.view(B * C, self.nb_zones, H, W)
        saved_key = saved_key.view(B*C, H, W).unsqueeze(1)
        out = torch.cat([saved_key, out], dim=1)
        out = self.conv_weight_getter(out) # B*C, 1, H, W
        out = out.view(B, C, H, W)

        #nonzero = torch.count_nonzero(out, dim=1)
        #my_sum = out.sum(dim=1)
        #out = torch.where(nonzero == 0, saved_key, my_sum / (nonzero + 1e-7)) # B, C, H, W
        #out = out * w


        #matplotlib.use('QtAgg')
        #fig, ax = plt.subplots(1, 2)
        #ax[0].imshow(out[0, 0].detach().cpu(), cmap='plasma', extent=[0, W, H, 0])
        #ax[1].imshow(saved_key[0, 0].detach().cpu(), cmap='plasma', extent=[0, W, H, 0])
        ##ax[2].imshow(nonzero[0, 0].detach().cpu(), cmap='gray', extent=[0, W, H, 0])
        ## Add the patch to the Axes
        #for rect in rects:
        #    ax[0].add_patch(copy.copy(rect))
        #    ax[1].add_patch(copy.copy(rect))
        #plt.show()

        out = out.permute(0, 2, 3, 1).view(B, H*W, C)
        out = self.output_proj_zones(out)
        return out


    def get_pos_2d(self, zones):
        B, _, C = zones.shape
        pos_2d = self.pos_2d(shape_util=(B, self.area_size, self.area_size), device=zones.device)
        pos_2d = pos_2d.permute(0, 2, 3, 1).view(B, self.area_size*self.area_size, C)
        return pos_2d


    def _get_ref_points(self, x, H, W):
        B = x.shape[0]
        
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H - 0.5, H, dtype=x.dtype, device=x.device), 
            torch.linspace(0.5, W - 0.5, W, dtype=x.dtype, device=x.device)
        )
        ref = torch.stack((ref_x, ref_y), -1)
        ref[..., 1].div_(W).mul_(2).sub_(1)
        ref[..., 0].div_(H).mul_(2).sub_(1)
        ref = ref[None, ...].expand(B, -1, -1, -1) # B H W 2
        
        return ref


    #def cropped_attention(self, query, key, frame_idx):
    #    T, B, C, H, W = key.shape
    #    shape = key.shape
    #    B, L, C = query.shape
    #    saved_key = key
#
    #    query_zones, key_zones, theta, theta_coords = self.get_zones(query, key)
    #    # key_zones = T, B*n_heads, C, area_size, area_size
    #    # query_zones = B*n_heads, area_size*area_size, C
#
    #    pos = self.get_pos_2d(query_zones) # B, L, C
    #    query_zones = query_zones + pos
    #    reference_points = self._get_ref_points(query_zones, self.area_size, self.area_size).view(B * self.n_heads, self.area_size*self.area_size, 2)
#
    #    key_zones, sampling_locations, attention_weights = self.attention(query=query_zones, key=key_zones, reference_points=reference_points, frame_idx=frame_idx) # B*n_heads, area_size*area_size, C
#
    #    out = self.uncrop_zones(theta, key_zones, saved_key[frame_idx], theta_coords=theta_coords) # B, L, C
#
#
    #    # for display
    #    # attention_weights = B * n_heads, T, n_heads, area_size*area_size, n_points
    #    # sampling_locations = B * n_heads, T, n_heads, area_size*area_size, n_points, 2
    #    attention_weights = attention_weights.view(B, self.n_heads, T, self.n_heads, self.area_size*self.area_size, self.n_points)
    #    sampling_locations = sampling_locations.view(B, self.n_heads, T, self.n_heads, self.area_size*self.area_size, self.n_points, 2)
    #    theta_coords = theta_coords.unsqueeze(2).repeat(1, 1, T, 1)
#
    #    return out, sampling_locations, attention_weights, theta_coords


    def attention(self, query, key, reference_points):
        T, B, C, H_k, W_k = key.shape
        B, C, H, W = query.shape
        L = H * W

        key = key.permute(1, 0, 2, 3, 4) # B, T, C, H, W

        key = key.permute(0, 1, 3, 4, 2)
        key = self.skip_co_proj(key).view(B, T, H_k, W_k, self.n_heads, self.head_dim)

        sampling_offsets = self.sampling_offsets(query).view(B, L, self.n_heads, T, self.n_points, 2)
        sampling_offsets = F.tanh(sampling_offsets) * 2
        attention_weights = self.attention_weights(query).view(B, L, self.n_heads, T * self.n_points) # <-- softmax dim normalization
        attention_weights = F.softmax(attention_weights, -1).view(B, L, self.n_heads, T, self.n_points)

        sampling_locations  = reference_points[:, :, None, None, None, :] + sampling_offsets # B, L, self.n_heads, T, self.n_points, 2

        sampling_locations = sampling_locations.permute(0, 3, 2, 1, 4, 5).contiguous().view(B * T * self.n_heads, L, self.n_points, 2)
        key = key.permute(0, 1, 4, 5, 2, 3).contiguous().view(B * T * self.n_heads, self.head_dim, H_k, W_k)

        sampled_values = F.grid_sample(key, grid=sampling_locations, mode='bilinear', padding_mode='zeros', align_corners=False).view(B, T, self.n_heads, self.head_dim, L, self.n_points)
        sampled_values = sampled_values.permute(0, 4, 2, 3, 1, 5) # B, L, n_heads, head_dim, T, n_points
        out = sampled_values * attention_weights[:, :, :, None, :, :] # B, L, n_heads, head_dim, T, n_points

        out = out.permute(0, 1, 4, 5, 2, 3).contiguous().view(B, L, T, self.n_points, C)
        out = self.output_proj(out)
        out = out.permute(0, 1, 4, 2, 3).contiguous() # B, L, C, T, n_points
        out = out.view(B, L, C, T * self.n_points).mean(-1)

        # for display
        sampling_locations = sampling_locations.detach()
        attention_weights = attention_weights.detach()
        sampling_locations = sampling_locations.view(B, T, self.n_heads, L, self.n_points, 2)
        attention_weights = attention_weights.permute(0, 3, 2, 1, 4).contiguous() # B, T, n_heads, L, n_points

        return out, sampling_locations, attention_weights

    
    def cropped_attention(self, query, key, frame_idx, temporal_pos, memory_bus):
        T, B, C, H, W = key.shape
        # query = B, C, H, W
        saved_key = key

        query_zones, theta, theta_coords, reference_points, pos_2d = self.get_zones(query, memory_bus) # maybe replace query with decoder feature map
        # query_zones = B*nb_zones, area_size*area_size, C
        key = key.unsqueeze(2).repeat(1, 1, self.nb_zones, 1, 1, 1).view(T, B*self.nb_zones, C, H, W)
        #pos_2d = self.pos_2d.unsqueeze(0).repeat(B, 1, 1)
        #pos = temporal_pos
        #query_zones = query_zones + pos

        query_zones, sampling_locations, attention_weights = self.attention(query=query_zones, key=key, reference_points=reference_points) # B*nb_zones, area_size*area_size, C

        #query_zone = key_zones[frame_idx]
        #query_zone = query_zone.permute(0, 2, 3, 1).view(B * self.n_heads, self.area_size*self.area_size, C)
        #query_zone = query_zone.permute(0, 2, 3, 1).view(B * self.nb_zones, 1, C)

        #pos_2d = self.get_pos_2d(query_zone) # B, L, C
        #pos_2d = pos_2d.unsqueeze(0).repeat(T, 1, 1, 1)# T, B, L, C

        #key_zones = key_zones.permute(1, 0, 3, 4, 2).contiguous() # B*n_heads, T, area_size, area_size, C
        #key_zones = key_zones.view(B * self.n_heads, T * self.area_size * self.area_size, C)# B*n_heads, T*area_size*area_size, C
        #key_zones = key_zones.view(B * self.n_heads, T, C)# B*n_heads, T*area_size*area_size, C
        #key_pos = key_pos.permute(1, 0, 2, 3).contiguous().view(B*self.n_heads, T*self.area_size*self.area_size, C)

        #query_zone, attention_weights = self.cross_transformer(query=query_zone, key=key_zones, value=key_zones, query_pos=query_pos, key_pos=key_pos) # B*n_heads, area_size*area_size, C
        out = self.uncrop_zones(theta, query_zones, saved_key[frame_idx], theta_coords=theta_coords) # B, H*W, C

        # for display
        # attention_weights = # B*nb_zones, T, n_heads, area_size*area_size, n_points
        # sampling_locations = # B*nb_zones, T, n_heads, area_size*area_size, n_points, 2
        attention_weights = attention_weights.view(B, self.nb_zones, T, self.n_heads, self.area_size, self.area_size, self.n_points)
        sampling_locations = sampling_locations.view(B, self.nb_zones, T, self.n_heads, self.area_size, self.area_size, self.n_points, 2)

        return out, sampling_locations, attention_weights, theta_coords


    def forward(self, key, query, frame_idx, temporal_pos, memory_bus):
        T, B, C, H, W = key.shape
        B, C, H, W = query.shape

        out, sampling_locations, attention_weights, theta_coords = self.cropped_attention(query, key, frame_idx, temporal_pos=temporal_pos, memory_bus=memory_bus)

        return out, sampling_locations, attention_weights, theta_coords


class DeformableAttention(nn.Module):
    def __init__(self, dim, n_heads, n_points, video_length):
        super(DeformableAttention, self).__init__()
        self.n_heads = n_heads
        self.n_points = n_points
        self.head_dim = dim // n_heads
        self.sampling_offsets = nn.Linear(dim, n_heads * video_length * n_points * 2)
        self.attention_weights = nn.Linear(dim, n_heads * video_length * n_points)
        self.skip_co_proj = nn.Linear(dim, dim)
        self.output_proj = nn.Linear(dim, dim)
    
    
    def attention(self, query, key, reference_points):
        T, B, C, H, W = key.shape
        B, L, C = query.shape

        key = key.permute(1, 0, 2, 3, 4) # B, T, C, H, W

        key = key.permute(0, 1, 3, 4, 2)
        key = self.skip_co_proj(key).view(B, T, H, W, self.n_heads, self.head_dim)

        sampling_offsets = self.sampling_offsets(query).view(B, L, self.n_heads, T, self.n_points, 2)
        sampling_offsets = F.tanh(sampling_offsets) * 2
        attention_weights = self.attention_weights(query).view(B, L, self.n_heads, T * self.n_points) # <-- softmax dim normalization
        attention_weights = F.softmax(attention_weights, -1).view(B, L, self.n_heads, T, self.n_points)

        sampling_locations  = reference_points[:, :, None, None, None, :] + sampling_offsets # B, L, self.n_heads, T, self.n_points, 2

        sampling_locations = sampling_locations.permute(0, 3, 2, 1, 4, 5).contiguous().view(B * T * self.n_heads, L, self.n_points, 2)
        key = key.permute(0, 1, 4, 5, 2, 3).contiguous().view(B * T * self.n_heads, self.head_dim, H, W)

        sampled_values = F.grid_sample(key, grid=sampling_locations, mode='bilinear', padding_mode='zeros', align_corners=False).view(B, T, self.n_heads, self.head_dim, L, self.n_points)
        sampled_values = sampled_values.permute(0, 4, 2, 3, 1, 5) # B, L, n_heads, head_dim, T, n_points
        out = sampled_values * attention_weights[:, :, :, None, :, :] # B, L, n_heads, head_dim, T, n_points

        out = out.permute(0, 1, 4, 5, 2, 3).contiguous().view(B, L, T, self.n_points, C)
        out = self.output_proj(out)
        out = out.permute(0, 1, 4, 2, 3).contiguous() # B, L, C, T, n_points
        out = out.view(B, L, C, T * self.n_points).mean(-1)

        # for display
        sampling_locations = sampling_locations.detach()
        attention_weights = attention_weights.detach()
        sampling_locations = sampling_locations.view(B, T, self.n_heads, L, self.n_points, 2)
        attention_weights = attention_weights.permute(0, 3, 2, 1, 4).contiguous() # B, T, n_heads, L, n_points

        return out, sampling_locations, attention_weights

    def forward(self, key, query, reference_points):
        T, B, C, H, W = key.shape
        B, L, C = query.shape

        out, sampling_locations, attention_weights = self.attention(query, key, reference_points) # out = B, L, C

        return out, sampling_locations, attention_weights


class SkipCoDeformableAttention(nn.Module):
    def __init__(self, dim, n_heads, nb_zones, video_length, norm, area_size, n_points, d_model, res):
        super(SkipCoDeformableAttention, self).__init__()
        self.area_size = area_size
        self.n_heads = n_heads
        self.nb_zones = nb_zones
        #self.level_embed = nn.Parameter(torch.randn(size=(video_length, dim)))
        self.attn = ZoomAttention(d_model=d_model, 
                                    n_points=n_points, 
                                    dim=dim, 
                                    n_heads=n_heads, 
                                    nb_zones=nb_zones, 
                                    video_length=video_length, 
                                    area_size=area_size,
                                    res=res)

        self.reduce_temporal_pos = nn.Linear(d_model, dim)

        self.norm = norm(dim)
        self.gelu = nn.GELU()

    #def forward(self, skip_connection, x, frame_idx):
    #    T, B, C, H, W = skip_connection.shape
    #    B, C, H, W = x.shape
#
    #    x = x.permute(0, 2, 3, 1).view(B, H * W, C)
    #    skip_connection, sampling_locations, attention_weights, theta_coords = self.attn(key=skip_connection, query=x, frame_idx=frame_idx) # skip_connection = B, L, C
    #    skip_connection = skip_connection.permute(0, 2, 1).view(B, C, H, W)
#
    #    skip_connection = self.norm(skip_connection)
    #    skip_connection = self.gelu(skip_connection)
#
    #    # for display
    #    # attention_weights = B, n_heads, T, n_heads, area_size*area_size, n_points
    #    # sampling_locations = B, n_heads, T, n_heads, area_size*area_size, n_points, 2
    #    sampling_locations = sampling_locations.view(B, self.n_heads, T, self.n_heads, self.area_size, self.area_size, self.n_points, 2)
    #    attention_weights = attention_weights.view(B, self.n_heads, T, self.n_heads, self.area_size, self.area_size, self.n_points)
#
    #    return skip_connection, sampling_locations, attention_weights, theta_coords

    def forward(self, key, query, frame_idx, temporal_pos, memory_bus):
        T, B, C, H, W = key.shape
        B, C, H, W = query.shape

        temporal_pos = self.reduce_temporal_pos(temporal_pos)
        temporal_pos = temporal_pos.view(1, 1, C).repeat(B, self.area_size * self.area_size, 1)

        skip_connection, sampling_locations, attention_weights, theta_coords = self.attn(key=key, query=query, frame_idx=frame_idx, temporal_pos=temporal_pos, memory_bus=memory_bus) # skip_connection = B, L, C
        skip_connection = skip_connection.permute(0, 2, 1).view(B, C, H, W)

        skip_connection = self.norm(skip_connection)
        skip_connection = self.gelu(skip_connection)

        return skip_connection, sampling_locations, attention_weights, theta_coords


class DeformableTransformer(nn.Module):
    def __init__(self, dim, n_heads, video_length, n_points, d_ffn, res, dropout=0.0):
        super(DeformableTransformer, self).__init__()
        self.n_heads = n_heads
        self.attn = DeformableAttention(dim=dim, n_heads=n_heads, n_points=n_points, video_length=video_length)

        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(dim)

        self.pos_2d = nn.Parameter(torch.randn(res**2, dim))

        # ffn
        self.linear1 = nn.Linear(dim, d_ffn)
        self.activation = nn.GELU()
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, dim)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim)
    
    def _get_ref_points(self, x, H, W):
        B = x.shape[0]
        
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H - 0.5, H, dtype=x.dtype, device=x.device), 
            torch.linspace(0.5, W - 0.5, W, dtype=x.dtype, device=x.device)
        )
        ref = torch.stack((ref_x, ref_y), -1)
        ref[..., 1].div_(W).mul_(2).sub_(1)
        ref[..., 0].div_(H).mul_(2).sub_(1)
        ref = ref[None, ...].expand(B, -1, -1, -1) # B H W 2
        
        return ref
    
    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src
    
    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, key, query, temporal_pos, spatial_pos):
        T, B, C, H, W = key.shape
        B, C, H, W = query.shape

        reference_points = self._get_ref_points(query, H, W)
        reference_points = reference_points.view(B, H * W, 2)

        temporal_pos = temporal_pos.view(1, 1, C).repeat(B, H * W, 1)
        src = query.permute(0, 2, 3, 1).contiguous().view(B, -1, C)

        pos = temporal_pos + spatial_pos

        #src2, sampling_locations, attention_weights = self.attn(key=key, query=src, reference_points=reference_points) # query = B, L, C
        src2, sampling_locations, attention_weights = self.attn(key=key, query=self.with_pos_embed(src, pos), reference_points=reference_points) # query = B, L, C
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        src = src.permute(0, 2, 1).view(B, C, H, W)
        return src, sampling_locations, attention_weights



class ConvDecoderDeformableAttentionIdentity(nn.Module):
    def __init__(self):
        super(ConvDecoderDeformableAttentionIdentity, self).__init__()

    def forward(self, skip_connection, x, frame_idx):
        return skip_connection[frame_idx], None, None


class TransformerDecoderBlock(nn.Module):
    def __init__(self, dim, num_heads, n_points, video_length, d_ffn, dropout=0.):
        super(TransformerDecoderBlock, self).__init__()


        # ffn
        self.linear1 = nn.Linear(dim, d_ffn)
        self.activation = nn.GELU()
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, dim)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(dim)

        #self.get_reference_points = nn.Linear(dim, 2)
        #self.cross_attn_layer = DeformableAttention(dim=dim, n_heads=num_heads, n_points=n_points, video_length=video_length)
        #self.norm1 = nn.LayerNorm(dim)
        #self.dropout1 = nn.Dropout(dropout)

        self.self_attn_layer = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt
    
    def with_pos_embed(self, x, pos):
        return x + pos

    def forward(self, skip_connection, x, advanced_pos):
        #reference_points = self.get_reference_points(advanced_pos)
        #reference_points = F.tanh(reference_points)
        #cross_attn_out, _, _ = self.cross_attn_layer(key=skip_connection, query=self.with_pos_embed(x, advanced_pos), reference_points=reference_points)
        #x = x + self.dropout1(cross_attn_out)
        #x = self.norm1(x)

        self_attn_out, _ = self.self_attn_layer(query=self.with_pos_embed(x, advanced_pos), key=self.with_pos_embed(x, advanced_pos), value=self.with_pos_embed(x, advanced_pos))
        x = x + self.dropout2(self_attn_out)
        x = self.norm2(x)

        x = self.forward_ffn(x)
        return x




class SpatialTransformerNetwork(nn.Module):
    def __init__(self, in_dim, norm, area_size, input_resolution, nb_zones):
        super(SpatialTransformerNetwork, self).__init__()
        self.depth = int(log(input_resolution/2) / log(2))
        self.area_size = area_size
        self.nb_zones = nb_zones

        #dims = torch.linspace(in_dim, 1, self.depth+1).int()
        self.layers = nn.ModuleList()
        for i in range(self.depth):
            #conv = conv_layer(in_dim, in_dim, 1, norm, kernel_size=3)
            downsample = nn.Sequential(nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=3, stride=2, padding=1),
                                        norm(in_dim),
                                        nn.GELU())
            #self.layers.append(conv)
            self.layers.append(downsample)
        self.regressor = nn.Sequential(nn.Linear(in_dim * 4, in_dim * 2), nn.GELU(), nn.Linear(in_dim * 2, 2 * nb_zones))
    
    def sigmoid(self, x, weight):
        return 1 / (1 + torch.exp(-weight * x))
    
    def min_max_normalization(self, x, new_min, new_max):
        #new_max = size - self.area_size // 2
        #new_min = self.area_size // 2
        return x * (new_max - new_min) + new_min

    def forward(self, x):
        B, C, H, W = x.shape
        for layer in self.layers:
            x = layer(x)
        x = torch.flatten(x, start_dim=1)
        x = self.regressor(x)
        x = torch.sigmoid(x)
        #x = self.sigmoid(x, weight=10)
        #x = self.min_max_normalization(x, new_min=self.area_size // 2, new_max=H - self.area_size // 2)
        x = self.min_max_normalization(x, new_min=0, new_max=H - 1)
        zones = torch.split(x, 2, dim=-1)
        #x = torch.clamp(x, self.area_size // 2, H - self.area_size // 2)

        #width, coords = torch.split(x, [1, 2], dim=-1)
        #coords = self.min_max_normalization(coords, new_min=0, new_max=H)
        #width = self.min_max_normalization(width, new_min=max(1, self.area_size ** 2 / H), new_max=min(H, self.area_size ** 2))

        #height = (self.area_size ** 2) / width[:, 0]

        thetas = []
        theta_coords = []
        for zone in zones:
            #x1 = zone[:, 0] - self.area_size // 2
            #x2 = zone[:, 0] + self.area_size // 2
            #y1 = zone[:, 1] - self.area_size // 2
            #y2 = zone[:, 1] + self.area_size // 2

            x1 = zone[:, 0] - 0.5
            x2 = zone[:, 0] + 0.5
            y1 = zone[:, 1] - 0.5
            y2 = zone[:, 1] + 0.5

            theta = torch.zeros(size=(B, 2, 3), device=x.device)
            theta[:, 0, 0] = (x2 - x1) / W
            theta[:, 0, 1] = 0.0
            theta[:, 0, 2] = -1 + (x2 + x1) / W
            theta[:, 1, 0] = 0.0
            theta[:, 1, 1] = (y2 - y1) / H
            theta[:, 1, 2] = -1 + (y2 + y1) / H
            theta_coord = torch.stack([x1, y1, x2, y2], dim=-1)
            thetas.append(theta)
            theta_coords.append(theta_coord)
        thetas = torch.stack(thetas, dim=1)
        theta_coords = torch.stack(theta_coords, dim=1)
        theta_coords = torch.round(theta_coords)

        #x = x.unsqueeze(-1).repeat(1, 1, 3)
        #x[:, 0, 0] = self.area_size / W
        #x[:, 0, 1] = 0.0
        #x[:, 0, 2] = -1 + (2 * x[:, 0, 2]) / W
        #x[:, 1, 0] = 0.0
        #x[:, 1, 1] = self.area_size / H
        #x[:, 1, 2] = -1 + (2 * x[:, 1, 2]) / H
        return thetas, theta_coords

class LayerNorm(nn.LayerNorm):
    """
    this is just a compatibility layer because my target tensor is float and has an extra dimension
    """
    def __init__(self, dim):
        super(LayerNorm, self).__init__(dim)

    def forward(self, x):
        assert x.dim() == 4, print(x.shape)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)
        x = super().forward(x)
        x = x.permute(0, 3, 1, 2)
        return x

class Filter(nn.Module):
    def __init__(self, dim, num_heads, norm):
        super(Filter, self).__init__()

        self.attention = torch.nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads)
        self.rescaler_conv = ConvBlock(in_dim=dim, out_dim=dim, norm=norm, kernel_size=1)
        self.rescaled_conv = ConvBlock(in_dim=dim, out_dim=dim, norm=norm, kernel_size=1)
        self.output_conv = nn.Sequential(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1),
                                        norm(dim),
                                        nn.Sigmoid())
        self.pos = PositionEmbeddingSine2d(num_pos_feats=dim // 2, normalize=True)

    def forward(self, rescaled, rescaler):
        B, C, H, W = rescaled.shape
        pos = self.pos(shape_util=(B, H, W), device=rescaled.device)

        ready_rescaler = self.rescaler_conv(rescaler)
        ready_rescaled = self.rescaled_conv(rescaled)

        temp = torch.stack([ready_rescaled, ready_rescaler], dim=-1)
        weights = torch.mean(temp, dim=-1)

        #ready_rescaler = ready_rescaler + pos
        #ready_rescaled = ready_rescaled + pos
#
        #ready_rescaler = ready_rescaler.permute(2, 3, 0, 1).view(H * W, B, C)
        #ready_rescaled = ready_rescaled.permute(2, 3, 0, 1).view(H * W, B, C)
#
        #weights = self.attention(query=ready_rescaler, key=ready_rescaler, value=ready_rescaled)[0]
#
        #weights = weights.permute(1, 2, 0).view(B, C, H, W)

        weights = self.output_conv(weights)
        out = rescaled * weights

        return out, weights.detach()



class ConvBlocksLegacy(nn.Module):
    def __init__(self, in_dim, out_dim, nb_blocks, norm, kernel_size=3, dpr=None):
        super(ConvBlocksLegacy, self).__init__()
        dims = torch.linspace(in_dim, out_dim, nb_blocks+1).int()

        self.blocks = nn.ModuleList()

        for i in range(nb_blocks):
            block = nn.Sequential(nn.Conv2d(in_channels=dims[i], out_channels=dims[i+1], kernel_size=kernel_size, padding='same'), 
                                        norm(dims[i+1]), 
                                        nn.GELU(),
                                        nn.Conv2d(in_channels=dims[i+1], out_channels=dims[i+1], kernel_size=kernel_size, padding='same'), 
                                        norm(dims[i+1]), 
                                        nn.GELU())
            self.blocks.append(block)
    
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x
    

#class ConvBlocks2D(nn.Module):
#    def __init__(self, in_dim, out_dim, nb_blocks, kernel_size=3):
#        super(ConvBlocks2D, self).__init__()
#        dims = torch.linspace(in_dim, out_dim, nb_blocks+1).int()
#        dims[1:-1] = torch.IntTensor([8 * round(x.item() / 8) for x in dims[1:-1]])
#
#        self.blocks = nn.ModuleList()
#
#        for i in range(nb_blocks):
#            block = nn.Sequential(nn.Conv2d(in_channels=dims[i], out_channels=dims[i+1], kernel_size=kernel_size, padding='same'), 
#                                        nn.GroupNorm(8, dims[i+1]),
#                                        nn.GELU(),)
#            self.blocks.append(block)
#    
#    def forward(self, x):
#        for block in self.blocks:
#            x = block(x)
#        return x

class ResnetBlock2DGroup(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3):
        super(ResnetBlock2DGroup, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=kernel_size, padding='same')
        self.norm1 = nn.GroupNorm(8, out_dim)
        self.conv2 = nn.Conv2d(in_channels=out_dim, out_channels=out_dim, kernel_size=kernel_size, padding='same')
        self.norm2 = nn.GroupNorm(8, out_dim)
        
        self.res_conv = nn.Conv2d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()
        self.gelu = nn.GELU()
    
    def forward(self, x):
        identity = x
        
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.gelu(x)

        x = self.conv2(x)
        x = self.norm2(x)

        x = x + self.res_conv(identity)
        x = self.gelu(x)

        return x


class ResnetBlock3DGroup(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3):
        super(ResnetBlock3DGroup, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=in_dim, out_channels=out_dim, kernel_size=kernel_size, padding='same')
        self.norm1 = nn.GroupNorm(8, out_dim)
        self.conv2 = nn.Conv3d(in_channels=out_dim, out_channels=out_dim, kernel_size=kernel_size, padding='same')
        self.norm2 = nn.GroupNorm(8, out_dim)
        
        self.res_conv = nn.Conv3d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()
        self.gelu = nn.GELU()
    
    def forward(self, x):
        """"dist_emb: T-1, B, C"""
        identity = x
        B, C, T, H, W = x.shape
        
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.gelu(x)

        x = self.conv2(x)
        x = self.norm2(x)

        x = x + self.res_conv(identity)
        x = self.gelu(x)

        return x



class ResnetBlock3DEmbedding(nn.Module):
    def __init__(self, in_dim, out_dim, d_model, kernel_size=3):
        super(ResnetBlock3DEmbedding, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=in_dim, out_channels=out_dim, kernel_size=kernel_size, padding='same')
        self.norm1 = nn.GroupNorm(8, out_dim)
        self.conv2 = nn.Conv3d(in_channels=out_dim, out_channels=out_dim, kernel_size=kernel_size, padding='same')
        self.norm2 = nn.GroupNorm(8, out_dim)
        
        self.res_conv = nn.Conv3d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()
        self.gelu = nn.GELU()

        self.modulate = nn.Linear(d_model, out_dim)
    
    def forward(self, x, dist_emb):
        """"dist_emb: T, B, C"""
        identity = x
        B, C, T, H, W = x.shape
        
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.gelu(x)

        dist_emb = self.modulate(dist_emb)
        dist_emb = dist_emb.permute(1, 2, 0).contiguous() # B, C, T
        dist_emb = dist_emb[:, :, :, None, None].repeat(1, 1, 1, H, W)
        x = x + dist_emb

        x = self.conv2(x)
        x = self.norm2(x)

        x = x + self.res_conv(identity)
        x = self.gelu(x)

        return x



class ResnetBlock2DEmbedding(nn.Module):
    def __init__(self, in_dim, out_dim, d_model=None, kernel_size=3):
        super(ResnetBlock2DEmbedding, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=kernel_size, padding='same')
        self.norm1 = nn.GroupNorm(8, out_dim)
        self.conv2 = nn.Conv2d(in_channels=out_dim, out_channels=out_dim, kernel_size=kernel_size, padding='same')
        self.norm2 = nn.GroupNorm(8, out_dim)
        
        self.res_conv = nn.Conv2d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()
        self.gelu = nn.GELU()
        self.embedding = False

        if d_model is not None:
            self.embedding = True
            self.modulate = nn.Linear(d_model, out_dim)
    
    def forward(self, x, dist_emb):
        """"dist_emb: B, C"""
        identity = x
        B, C, H, W = x.shape
        
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.gelu(x)

        if self.embedding:
            dist_emb = self.modulate(dist_emb)
            dist_emb = dist_emb[:, :, None, None].repeat(1, 1, H, W)
            x = x + dist_emb

        x = self.conv2(x)
        x = self.norm2(x)

        x = x + self.res_conv(identity)
        x = self.gelu(x)

        return x
    

class ResnetBlock2DBatch(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3):
        super(ResnetBlock2DBatch, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=kernel_size, padding='same')
        self.norm1 = nn.BatchNorm2d(out_dim)
        self.conv2 = nn.Conv2d(in_channels=out_dim, out_channels=out_dim, kernel_size=kernel_size, padding='same')
        self.norm2 = nn.BatchNorm2d(out_dim)
        
        self.res_conv = nn.Conv2d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()
        self.gelu = nn.GELU()
    
    def forward(self, x):
        identity = x
        
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.gelu(x)

        x = self.conv2(x)
        x = self.norm2(x)

        x = x + self.res_conv(identity)
        x = self.gelu(x)

        return x
    

class ConvBlocks2DGroup(nn.Module):
    def __init__(self, in_dim, out_dim, nb_blocks, kernel_size=3):
        super(ConvBlocks2DGroup, self).__init__()
        dims = torch.linspace(in_dim, out_dim, nb_blocks+1).int()
        dims[1:] = (torch.round(dims[1:] / 8) * 8).int()

        self.blocks = nn.ModuleList()

        for i in range(nb_blocks):
            block = ResnetBlock2DGroup(in_dim=dims[i], out_dim=dims[i+1], kernel_size=kernel_size)
            self.blocks.append(block)
    
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x



class DoubleConv(nn.Module):
    def __init__(self, in_dim, out_dim, d_model=None, kernel_size=3):
        super(DoubleConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=kernel_size, padding='same')
        self.norm1 = nn.GroupNorm(8, out_dim)
        self.conv2 = nn.Conv2d(in_channels=out_dim, out_channels=out_dim, kernel_size=kernel_size, padding='same')
        self.norm2 = nn.GroupNorm(8, out_dim)
        self.gelu = nn.GELU()
        self.embedding = False

        if d_model is not None:
            self.embedding = True
            self.modulate = nn.Linear(d_model, out_dim)
    
    def forward(self, x, dist_emb=None):
        """"dist_emb: B, C"""
        identity = x
        B, C, H, W = x.shape
        
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.gelu(x)

        if self.embedding:
            dist_emb = self.modulate(dist_emb)
            dist_emb = dist_emb[:, :, None, None].repeat(1, 1, H, W)
            x = x + dist_emb

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.gelu(x)

        return x
    


class SingleConv(nn.Module):
    def __init__(self, in_dim, out_dim, d_model=None, kernel_size=3):
        super(SingleConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=kernel_size, padding='same')
        self.norm1 = nn.GroupNorm(8, out_dim)
        self.gelu = nn.GELU()
        self.embedding = False

        if d_model is not None:
            self.embedding = True
            self.modulate = nn.Linear(d_model, out_dim)
    
    def forward(self, x, dist_emb=None):
        """"dist_emb: B, C"""
        identity = x
        B, C, H, W = x.shape
        
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.gelu(x)

        return x
    

class OneConv3D(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, dilation):
        super(OneConv3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=in_dim, out_channels=out_dim, kernel_size=kernel_size, dilation=dilation, padding='same')
        self.norm1 = nn.GroupNorm(8, out_dim)
        self.gelu = nn.GELU()
    
    def forward(self, x):
        """"dist_emb: B, C"""
        B, C, T, H, W = x.shape
        
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.gelu(x)

        return x


class Pooling3D(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Pooling3D, self).__init__()
        self.pooling = nn.AdaptiveAvgPool3d(1)
        self.conv1 = nn.Conv3d(in_channels=in_dim, out_channels=out_dim, kernel_size=1, dilation=1, padding='same')
        self.norm1 = nn.GroupNorm(8, out_dim)
        self.gelu = nn.GELU()
    
    def forward(self, x):
        """"dist_emb: B, C"""
        B, C, T, H, W = x.shape

        size = x.shape[-3:]
        
        x = self.pooling(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.gelu(x)

        return nn.functional.interpolate(x, size=size, mode='trilinear', align_corners=False)

    


class DoubleConv3D(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size):
        super(DoubleConv3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=in_dim, out_channels=out_dim, kernel_size=kernel_size, padding='same')
        self.norm1 = nn.GroupNorm(8, out_dim)
        self.conv2 = nn.Conv3d(in_channels=out_dim, out_channels=out_dim, kernel_size=kernel_size, padding='same')
        self.norm2 = nn.GroupNorm(8, out_dim)
        self.gelu = nn.GELU()
        self.embedding = False

        #if d_model is not None:
        #    self.embedding = True
        #    self.modulate = nn.Linear(d_model, out_dim)
    
    def forward(self, x, dist_emb=None):
        """"dist_emb: B, C"""
        identity = x
        B, C, T, H, W = x.shape
        
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.gelu(x)

        #if self.embedding:
        #    dist_emb = self.modulate(dist_emb)
        #    dist_emb = dist_emb[:, :, None, None].repeat(1, 1, H, W)
        #    x = x + dist_emb

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.gelu(x)

        return x
    


class ConvBlocks2DGroupLegacy(nn.Module):
    def __init__(self, in_dim, out_dim, nb_blocks, kernel_size=3, d_model=None, nb_conv=2):
        super(ConvBlocks2DGroupLegacy, self).__init__()
        dims = torch.linspace(in_dim, out_dim, nb_blocks+1).int()
        dims[1:] = (torch.round(dims[1:] / 8) * 8).int()

        self.blocks = nn.ModuleList()

        if nb_conv == 2:
            blockfunction = DoubleConv
        elif nb_conv == 1:
            blockfunction = SingleConv

        for i in range(nb_blocks):
            block = blockfunction(in_dim=dims[i], out_dim=dims[i+1], d_model=d_model)
            self.blocks.append(block)
    
    def forward(self, x, dist_emb=None):
        for block in self.blocks:
            x = block(x, dist_emb)
        return x
    



class ConvBlocks3DGroupLegacy(nn.Module):
    def __init__(self, in_dim, out_dim, nb_blocks, kernel_size):
        super(ConvBlocks3DGroupLegacy, self).__init__()
        dims = torch.linspace(in_dim, out_dim, nb_blocks+1).int()
        dims[1:] = (torch.round(dims[1:] / 8) * 8).int()

        self.blocks = nn.ModuleList()

        for i in range(nb_blocks):
            block = DoubleConv3D(in_dim=dims[i], out_dim=dims[i+1], kernel_size=kernel_size)
            self.blocks.append(block)
    
    def forward(self, x, dist_emb=None):
        for block in self.blocks:
            x = block(x, dist_emb)
        return x



class ConvBlocks2DEmbedding(nn.Module):
    def __init__(self, in_dim, out_dim, nb_blocks, kernel_size=3, d_model=None):
        super(ConvBlocks2DEmbedding, self).__init__()
        dims = torch.linspace(in_dim, out_dim, nb_blocks+1).int()
        dims[1:] = (torch.round(dims[1:] / 8) * 8).int()

        self.blocks = nn.ModuleList()

        for i in range(nb_blocks):
            block = ResnetBlock2DEmbedding(in_dim=dims[i], out_dim=dims[i+1], d_model=d_model, kernel_size=kernel_size)
            self.blocks.append(block)
    
    def forward(self, x, dist_emb=None):
        for block in self.blocks:
            x = block(x, dist_emb)
        return x


class ConvBlocks3DEmbedding(nn.Module):
    def __init__(self, in_dim, out_dim, nb_blocks, d_model, kernel_size=3):
        super(ConvBlocks3DEmbedding, self).__init__()
        dims = torch.linspace(in_dim, out_dim, nb_blocks+1).int()

        self.blocks = nn.ModuleList()

        for i in range(nb_blocks):
            block = ResnetBlock3DEmbedding(in_dim=dims[i], out_dim=dims[i+1], d_model=d_model, kernel_size=kernel_size)
            self.blocks.append(block)
    
    def forward(self, x, dist_emb):
        for block in self.blocks:
            x = block(x, dist_emb)
        return x


class ConvBlocks3DGroup(nn.Module):
    def __init__(self, in_dim, out_dim, nb_blocks, kernel_size=3):
        super(ConvBlocks3DGroup, self).__init__()
        dims = torch.linspace(in_dim, out_dim, nb_blocks+1).int()

        self.blocks = nn.ModuleList()

        for i in range(nb_blocks):
            block = ResnetBlock3DGroup(in_dim=dims[i], out_dim=dims[i+1], kernel_size=kernel_size)
            self.blocks.append(block)
    
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x



class ConvBlocks2DBatch(nn.Module):
    def __init__(self, in_dim, out_dim, nb_blocks, kernel_size=3):
        super(ConvBlocks2DBatch, self).__init__()
        dims = torch.linspace(in_dim, out_dim, nb_blocks+1).int()

        self.blocks = nn.ModuleList()

        for i in range(nb_blocks):
            block = ResnetBlock2DBatch(in_dim=dims[i], out_dim=dims[i+1], kernel_size=kernel_size)
            self.blocks.append(block)
    
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class ConvBlocks3D(nn.Module):
    def __init__(self, in_dim, out_dim, nb_blocks, kernel_size=3):
        super(ConvBlocks3D, self).__init__()
        dims = torch.linspace(in_dim, out_dim, nb_blocks+1).int()

        self.blocks = nn.ModuleList()
        for i in range(nb_blocks):
            block = nn.Sequential(nn.Conv3d(in_channels=dims[i], out_channels=dims[i+1], kernel_size=kernel_size, padding='same'), 
                                        nn.GroupNorm(8, dims[i+1]),
                                        nn.GELU())
            self.blocks.append(block)
    
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class ConvBlocks3DPos(nn.Module):
    def __init__(self, in_dim, out_dim, nb_blocks, embedding_dim, kernel_size=3):
        super(ConvBlocks3DPos, self).__init__()
        dims = torch.linspace(in_dim, out_dim, nb_blocks+1).int()

        self.blocks_3d = nn.ModuleList()
        self.linear_layers = nn.ModuleList()
        for i in range(nb_blocks):
            block = nn.Sequential(nn.Conv3d(in_channels=dims[i], out_channels=dims[i+1], kernel_size=kernel_size, padding='same'), 
                                        nn.GroupNorm(8, dims[i+1]),
                                        nn.GELU())
            linear_layer = nn.Linear(embedding_dim, dims[i+1])
            self.blocks_3d.append(block)
            self.linear_layers.append(linear_layer)
    
    def forward(self, x, embedding):
        B, C, T, H, W = x.shape
        for block, linear_layer in zip(self.blocks_3d, self.linear_layers):
            x = block(x)
            h = linear_layer(embedding)
            x = x + h[:, :, None, None, None].repeat(1, 1, T, H, W)
        return x
    

class PatchMerging3D2D(nn.Module):
    """ Patch Merging Layer
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.reduction_3d = nn.Sequential(
            nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=(1, 2, 2), padding=1),
            nn.GroupNorm(8, out_dim),
            nn.GELU())
        self.reduction_2d = nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(8, out_dim),
                nn.GELU())

    def forward(self, x_3d, x_2d):
        x_3d = self.reduction(x_3d)
        x_2d = self.reduction(x_2d)
        x_3d[:, :, 0] = x_3d[:, :, 0] + x_2d
        return x_3d, x_2d



class ConvBlocks2D2D(nn.Module):
    def __init__(self, in_dim, out_dim, nb_blocks, kernel_size=3, dpr=None):
        super(ConvBlocks2D2D, self).__init__()
        dims = torch.linspace(in_dim, out_dim, nb_blocks+1).int()

        self.blocks_2d = nn.ModuleList()
        self.reduction_layers = nn.ModuleList()
        for i in range(nb_blocks):
            block_2d = nn.Sequential(nn.Conv2d(in_channels=dims[i], out_channels=dims[i+1], kernel_size=kernel_size, padding='same'), 
                                        nn.GroupNorm(8, dims[i+1]), 
                                        nn.GELU())
            reduction_layer = nn.Conv2d(in_channels=2 * dims[i+1], out_channels=dims[i+1], kernel_size=1)

            self.blocks_2d.append(block_2d)
            self.reduction_layers.append(reduction_layer)
        
    
    def forward(self, x, label):
        for block_2d, reduction_layer in zip(self.blocks_2d, self.reduction_layers):
            x = block_2d(x)
            x = torch.cat([x, label], dim=1)
            x = reduction_layer(x)

            x_3d[:, :, 0] = x_3d[:, :, 0] + x_2d
        return x_3d, x_2d


class ConvBlocks3D2D(nn.Module):
    def __init__(self, in_dim, out_dim, nb_blocks, kernel_size=3, dpr=None):
        super(ConvBlocks3D, self).__init__()
        dims = torch.linspace(in_dim, out_dim, nb_blocks+1).int()

        self.blocks_3d = nn.ModuleList()
        self.blocks_2d = nn.ModuleList()
        for i in range(nb_blocks):
            block_3d = nn.Sequential(nn.Conv3d(in_channels=dims[i], out_channels=dims[i+1], kernel_size=kernel_size, padding='same'), 
                                        nn.GroupNorm(8, dims[i+1]), 
                                        nn.GELU())
            block_2d = nn.Sequential(nn.Conv2d(in_channels=dims[i], out_channels=dims[i+1], kernel_size=kernel_size, padding='same'), 
                                        nn.GroupNorm(8, dims[i+1]), 
                                        nn.GELU())
            self.blocks_3d.append(block_3d)
            self.blocks_2d.append(block_2d)
    
    def forward(self, x_3d, x_2d):
        for block_3d, block_2d in zip(self.blocks_3d, self.blocks_2d):
            x_3d = block_3d(x_3d)
            x_2d = block_2d(x_2d)
            x_3d[:, :, 0] = x_3d[:, :, 0] + x_2d
        return x_3d, x_2d
    


class ConvBlocks1D(nn.Module):
    def __init__(self, in_dim, out_dim, nb_blocks, norm, kernel_size=3, dpr=None):
        super(ConvBlocks1D, self).__init__()
        dims = torch.linspace(in_dim, out_dim, nb_blocks+1).int().tolist()

        self.blocks = nn.ModuleList()

        for i in range(nb_blocks):
            block = nn.Sequential(nn.Conv1d(in_channels=dims[i], out_channels=dims[i+1], kernel_size=kernel_size, padding='same'), 
                                        norm(dims[i+1]), 
                                        nn.GELU(),
                                        nn.Conv1d(in_channels=dims[i+1], out_channels=dims[i+1], kernel_size=kernel_size, padding='same'), 
                                        norm(dims[i+1]), 
                                        nn.GELU())
            self.blocks.append(block)
    
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, norm, kernel_size):
        super(ConvBlock, self).__init__()

        self.block = nn.Sequential(nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=kernel_size, padding='same'), 
                                        norm(out_dim), 
                                        nn.GELU())
    
    def forward(self, x):
        x = self.block(x)
        return x

def get_indices_2d(input_resolution, num_memory_bus=0, rpe_table_size=None):
    # get pair-wise relative position index for each token inside the window
    coords_h = torch.arange(input_resolution[0])
    coords_w = torch.arange(input_resolution[1])
    coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
    coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww     i-i', j-j' [-(wh-1), wh-1]
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
    relative_coords[:, :, 0] += input_resolution[0] - 1  # shift to start from 0    i-i'+ h-1, j-j'+ w-1 [0, 2wh-2]
    relative_coords[:, :, 1] += input_resolution[1] - 1  # shift to start from 0    i-i'+ h-1, j-j'+ w-1 [0, 2wh-2]
    relative_coords[:, :, 0] *= 2 * input_resolution[1] - 1 # 1d indexing because parameters are 1d [0, 10wh-10, step=5]
    relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww  parameters weights are 1d [0, (2wh-1) * (2ww-1)]
    #if num_memory_bus > 0:
    #    relative_position_index = add_rpe_mem_bus(relative_position_index, num_memory_bus, rpe_table_size)
    return relative_position_index

class ConvFilterBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int, norm, input_resolution):
        super(ConvFilterBlock,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            norm(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            norm(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            norm(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        filtered = x * psi
        out = torch.cat((filtered, g), dim=1)

        return out


def get_root_logger(log_file=None, log_level=logging.INFO):
    """Use ``get_logger`` method in mmcv to get the root logger.
    The logger will be initialized if it has not been initialized. By default a
    StreamHandler will be added. If ``log_file`` is specified, a FileHandler
    will also be added. The name of the root logger is the top-level package
    name, e.g., "mmaction".
    Args:
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the root logger.
        log_level (int): The root logger level. Note that only the process of
            rank 0 is affected, while other processes will set the level to
            "Error" and be silent most of the time.
    Returns:
        :obj:`logging.Logger`: The root logger.
    """
    return get_logger(__name__.split('.')[0], log_file, log_level)

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class depthwise_separable_conv_3d(nn.Module):
    def __init__(self, nin, nout):
        super(depthwise_separable_conv_3d, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv3d(nin, nin, kernel_size=3, padding=1, groups=nin),
            nn.BatchNorm3d(nin),
            nn.GELU(),
            nn.Conv3d(nin, nout, kernel_size=1),
        )

    def forward(self, x):
        out = self.layer(x)
        return out

class depthwise_conv(nn.Module):
    def __init__(self, nin, input_resolution):
        super(depthwise_conv, self).__init__()
        self.layer = nn.Sequential(
            To_image(input_resolution),
            nn.Conv2d(nin, nin, kernel_size=3, padding=1, groups=nin),
            From_image()
        )

    def forward(self, x):
        out = self.layer(x)
        return out

class Mlp(nn.Module):
    def __init__(self, input_resolution, in_features, use_conv=False, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.use_conv = use_conv
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        if use_conv:
            self.conv = depthwise_conv(hidden_features, input_resolution)
            self.layer_norm = nn.LayerNorm(hidden_features)
        else:
            self.conv = nn.Identity()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        if self.use_conv:
            conv_out = self.conv(x)
            x = x + conv_out
            x = self.layer_norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout, norm):
        super(depthwise_separable_conv, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(nin, nin, kernel_size=3, padding=1, groups=nin),
            norm(nin),
            nn.GELU(),
            nn.Conv2d(nin, nout, kernel_size=1),
        )

    def forward(self, x):
        out = self.layer(x)
        return out


class depthwise_conv_3d(nn.Module):
    def __init__(self, nin):
        super(depthwise_conv_3d, self).__init__()
        self.layer = nn.Conv3d(nin, nin, kernel_size=3, padding=1, groups=nin)

    def forward(self, x):
        x = x.permute(0, 4, 1, 2, 3)
        out = self.layer(x)
        out = out.permute(0, 2, 3, 4, 1)
        return out

class Mlp3D(nn.Module):
    def __init__(self, in_features, use_conv=False, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.use_conv = use_conv
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        if use_conv:
            self.conv = depthwise_conv_3d(hidden_features)
            self.layer_norm = nn.LayerNorm(hidden_features)
        else:
            self.conv = nn.Identity()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        if self.use_conv:
            conv_out = self.conv(x)
            x = x + conv_out
            x = self.layer_norm(x)
            x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class AutoencoderMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        return x

class DeepSupervision(nn.Module):
    def __init__(self, dim, num_classes, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor
        self.conv = nn.Conv2d(dim, num_classes, 1, padding=0)

    def forward(self, x):
        x = self.conv(x)
        #my_max = torch.max(torch.flatten(x, start_dim=2), dim=-1)[0].unsqueeze(-1).unsqueeze(-1)
        #my_min = torch.min(torch.flatten(x, start_dim=2), dim=-1)[0].unsqueeze(-1).unsqueeze(-1)
        #x = torch.nn.functional.interpolate(x, scale_factor=self.scale_factor, mode='bicubic', antialias=True)
        #x = torch.clamp(x, my_min, my_max)

        return x

class DeepSupervisionLearn(nn.Module):
    def __init__(self, dim, num_classes, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor
        self.conv = nn.ConvTranspose2d(in_channels=dim, out_channels=num_classes, kernel_size=scale_factor)

    def forward(self, x):
        x = self.conv(x)
        return x

class ReshapeLayer(nn.Module):
    def __init__(self, input_resolution):
        super().__init__()
        self.input_resolution = input_resolution

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.permute(0, 2, 1).view(B, C, H, W)

        return x

class DeepSupervision_3d(nn.Module):
    def __init__(self, dim, num_classes, scale_factor):
        super().__init__()
        self.conv = nn.Conv3d(dim, num_classes, 1, padding=0)
        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear')

    def forward(self, x):
        x = self.conv(x)
        x = self.up(x)
        return x

class PatchExpandSwin(nn.Module):
    def __init__(self, blur, blur_kernel, input_resolution, in_dim, out_dim, swin_abs_pos, device):
        super().__init__()
        self.input_resolution = input_resolution
        self.out_dim = out_dim
        self.blur = blur
        self.up = nn.ConvTranspose2d(in_dim, out_dim, 2, 2)
        if blur:
            self.blur_layer = BlurLayer(blur_kernel, stride=1)
        self.swin_abs_pos = swin_abs_pos
        self.activation = nn.GELU()
        self.layer_norm = nn.LayerNorm(out_dim)
        self.device = device
        self.pos_object = PositionEmbeddingSine2d(num_pos_feats=out_dim//2, normalize=True)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        x = x.permute(0, 2, 1).view(B, C, H, W)
        x = self.up(x)
        x = x.permute(0, 2, 3, 1).view(B, -1, self.out_dim)
        x = self.layer_norm(x)
        x = self.activation(x)
        x = x.permute(0, 2, 1).view(B, self.out_dim, H * 2, W * 2)
        if self.blur:
            x = self.blur_layer(x)
        if self.swin_abs_pos:
            abs_pos = self.pos_object(shape_util=(B, H * 2, W * 2), device=self.device)
            x = x + abs_pos
        x = x.permute((0, 2, 3, 1)).view(B, -1, self.out_dim)

        return x


class PatchExpandSwin3D(nn.Module):
    def __init__(self, blur, blur_kernel, in_dim, out_dim, swin_abs_pos, device):
        super().__init__()
        self.out_dim = out_dim
        self.blur = blur
        self.up = nn.ConvTranspose3d(in_dim, out_dim, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        if blur:
            self.blur_layer = BlurLayer3D(blur_kernel, stride=1)
        self.swin_abs_pos = swin_abs_pos
        self.activation = nn.GELU()
        self.layer_norm = nn.LayerNorm(out_dim)
        self.device = device
        self.pos_object = PositionEmbeddingSine3d(num_pos_feats=out_dim//3, normalize=True)

    def forward(self, x):
        B, C, D, H, W = x.shape
        #print(x.shape)
        x = self.up(x)
        x = x.permute(0, 2, 3, 4, 1)
        x = self.layer_norm(x)
        x = self.activation(x)
        x = x.permute(0, 4, 1, 2, 3)
        if self.blur:
            x = self.blur_layer(x)
        if self.swin_abs_pos:
            abs_pos = self.pos_object(shape_util=(B, D, H * 2, W * 2), device=self.device)
            x = x + abs_pos

        return x

class CatLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        return torch.cat([x1, x2], dim=1)


class PatchExpandLegacy(nn.Module):
    def __init__(self, in_dim, out_dim, swin_abs_pos, norm):
        super().__init__()
        self.out_dim = out_dim
        #if blur:
        #    self.up =nn.Sequential(
        #        nn.ConvTranspose2d(in_dim, out_dim, 2, 2),
        #        BlurLayer(blur_kernel, stride=1))
        #else:
        #    self.up = nn.ConvTranspose2d(in_dim, out_dim, 2, 2)
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_dim, out_dim, 2, 2),
            norm(out_dim),
            nn.GELU())
        self.swin_abs_pos = swin_abs_pos
        if swin_abs_pos:
            self.pos_object = PositionEmbeddingSine2d(num_pos_feats=out_dim//2, normalize=True)

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.up(x)
        if self.swin_abs_pos:
            abs_pos = self.pos_object(shape_util=(x.shape[0], H * 2, W * 2), device=x.device)
            x = x + abs_pos
        return x



class PatchExpand2DBatch(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.out_dim = out_dim
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_dim, out_dim, 2, 2),
            nn.BatchNorm2d(out_dim),
            nn.GELU())

    def forward(self, x):
        x = self.up(x)
        return x
    


class PatchExpand2DGroup(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.out_dim = out_dim
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_dim, out_dim, 2, 2),
            nn.GroupNorm(8, out_dim),
            nn.GELU())

    def forward(self, x):
        x = self.up(x)
        return x



class PatchExpand3DGroup(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.out_dim = out_dim
        self.up = nn.Sequential(
            nn.ConvTranspose3d(in_dim, out_dim, (1, 2, 2), (1, 2, 2)),
            nn.GroupNorm(8, out_dim),
            nn.GELU())

    def forward(self, x):
        x = self.up(x)
        return x



class concat_merge_conv_rescale(nn.Module):
    def __init__(self, in_dim, out_dim, input_resolution, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.rescale = rescale_layer(in_dim)
        self.merge = nn.Conv2d(in_dim, out_dim, 3, padding=1)
        self.norm = norm_layer(out_dim)
        self.out_dim = out_dim
    
    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        x = x.permute(0, 2, 1).view(B, C, H, W)

        x = self.rescale(x)
        x = self.merge(x)
        x = x.permute((0, 2, 3, 1)).view(B, -1, self.out_dim)
        x= self.norm(x)
        x = F.gelu(x)

        return x

class concat_merge_linear_rescale(nn.Module):
    def __init__(self, in_dim, out_dim, input_resolution, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.rescale = rescale_layer(in_dim)
        self.merge = nn.Linear(in_dim, out_dim)
        self.norm = norm_layer(out_dim)
    
    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        x = x.permute(0, 2, 1).view(B, C, H, W)
        x = self.rescale(x)
        x = x.permute(0, 2, 3, 1).view(B, L, C)
        x = self.merge(x)
        x= self.norm(x)
        x = F.gelu(x)

        return x

class concat_merge_linear(nn.Module):
    def __init__(self, in_dim, out_dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.merge = nn.Linear(in_dim, out_dim)
        self.norm = norm_layer(out_dim)
    
    def forward(self, x):
        x = self.merge(x)
        x= self.norm(x)
        x = F.gelu(x)

        return x



class PatchExpandConv_3d(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.up = nn.ConvTranspose3d(dim, dim//2, 2, 2)
        self.norm = norm_layer(dim//2)

    def forward(self, x):
        B, C, D, H, W = x.shape
        assert D % 2 == 0 and H % 2 == 0 and W % 2 == 0, f"x size ({D}*{H}*{W}) are not even."

        x = self.up(x)
        x = x.permute(0, 2, 3, 4, 1)
        x= self.norm(x)

        return x

class ConvUpBlock_3d(nn.Module):
    def __init__(self, in_dim, out_dim, stride, kernel_size, padding, num_classes, patch_size, deep_supervision):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.patch_size=patch_size
        up1 = nn.Conv3d(in_channels=2*in_dim, out_channels=in_dim, kernel_size=1, stride=1, padding=0)
        up2 = nn.Conv3d(in_channels=in_dim, out_channels=out_dim, kernel_size=1, stride=1, padding=0)
        self.convtransposed1 = nn.ConvTranspose3d(in_dim, in_dim, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv1_1 = SEBasicBlock_3d(inplanes=2*in_dim, planes=in_dim, stride=1, downsample=up1)
        self.conv1_2 = SEBasicBlock_3d(inplanes=in_dim, planes=out_dim, stride=1, downsample=up2)
        # deep supervision
        if self.deep_supervision:
            self.deep_supervision_layers = DeepSupervision_3d(dim=out_dim, num_classes=num_classes, scale_factor=2)
        else:
            self.deep_supervision_layers = None

    def forward(self, x, saved_out):
        out = None
        B, C, D, H, W = x.shape
        x = self.convtransposed1(x)
        x = torch.cat([x, saved_out], dim=1)
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        if self.deep_supervision:
            out = self.deep_supervision_layers(x)
        return x, out

class ConvBlock_3d(nn.Module):
    def __init__(self, stride, in_chans, embed_dim):
        super().__init__()
        self.conv = nn.Conv3d(in_chans, embed_dim, kernel_size=3, stride=stride, padding=1)
        self.activate = nn.GELU()
        nb_groups = embed_dim//16 if embed_dim > 16 else 1
        self.norm = nn.GroupNorm(nb_groups, embed_dim)
    
    def forward(self, x):
        x = self.conv(x)
        B, C, D, H, W = x.shape
        x = self.activate(x)
        #x = x.permute((0, 2, 3, 1)).view(B, -1, C) # B, L, C
        x = self.norm(x)
        return x
        #return x.permute((0, 2, 1)).view(B, C, H, W)

class PatchMergingSwin(nn.Module):
    """ Patch Merging Layer
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, blur, blur_kernel, input_resolution, in_dim, out_dim, swin_abs_pos, device):
        super().__init__()
        self.blur = blur
        self.out_dim = out_dim
        self.input_resolution = input_resolution
        if blur:
            self.reduction = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1)
            self.blur_layer = BlurLayer(blur_kernel, stride=2)
        else:
            self.reduction = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1)
        self.swin_abs_pos = swin_abs_pos
        self.activation = nn.GELU()
        self.layer_norm = nn.LayerNorm(out_dim)
        self.device = device
        self.pos_object = PositionEmbeddingSine2d(num_pos_feats=out_dim//2, normalize=True)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        x = x.permute(0, 2, 1).view(B, C, H, W)
        x = self.reduction(x)
        x = x.permute(0, 2, 3, 1).view(B, -1, self.out_dim)
        x = self.layer_norm(x)
        x = self.activation(x)
        if self.blur:
            x = x.permute(0, 2, 1).view(B, self.out_dim, H, W)
            x = self.blur_layer(x)
        else:
            x = x.permute(0, 2, 1).view(B, self.out_dim, H // 2, W // 2)
        if self.swin_abs_pos:
            abs_pos = self.pos_object(shape_util=(B, H // 2, W // 2), device=self.device)
            x = x + abs_pos
        x = x.permute(0, 2, 3, 1).contiguous().view(B, -1, self.out_dim)
        
        return x

class PatchMergingLegacy(nn.Module):
    """ Patch Merging Layer
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, in_dim, out_dim, device, norm, swin_abs_pos=False, blur=False, blur_kernel=[1, 2, 1]):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        if blur:
            self.reduction = nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
                norm(out_dim),
                nn.GELU(),
                BlurLayer(blur_kernel, stride=2))
        else:
            self.reduction = nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1),
                norm(out_dim),
                nn.GELU())
        self.swin_abs_pos = swin_abs_pos
        self.device = device
        self.pos_object = PositionEmbeddingSine2d(num_pos_feats=out_dim//2, normalize=True)

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.reduction(x)
        if self.swin_abs_pos:
            abs_pos = self.pos_object(shape_util=(x.shape[0], H // 2, W // 2), device=self.device)
            x = x + abs_pos
        
        return x


class PatchMerging2DGroup(nn.Module):
    """ Patch Merging Layer
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.reduction = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, out_dim),
            nn.GELU())

    def forward(self, x):
        x = self.reduction(x)
        return x
    


class PatchMerging3DGroup(nn.Module):
    """ Patch Merging Layer
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.reduction = nn.Sequential(
            nn.Conv3d(in_dim, out_dim, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.GroupNorm(8, out_dim),
            nn.GELU())

    def forward(self, x):
        x = self.reduction(x)
        return x


class PatchMerging2DBatch(nn.Module):
    """ Patch Merging Layer
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.reduction = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.GELU())

    def forward(self, x):
        x = self.reduction(x)
        return x


class PatchMerging3D(nn.Module):
    """ Patch Merging Layer
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.reduction = nn.Sequential(
            nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=(1, 2, 2), padding=1),
            nn.GroupNorm(8, out_dim),
            nn.GELU())

    def forward(self, x):
        x = self.reduction(x)
        return x

class PatchMergingSwin3D(nn.Module):
    """ Patch Merging Layer
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, blur, blur_kernel, in_dim, out_dim, swin_abs_pos, device, stride=(1, 2, 2)):
        super().__init__()
        self.blur = blur
        self.out_dim = out_dim
        padding = (0, 1, 1) if stride == (2, 2, 2) else 1
        if blur:
            self.reduction = nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=stride, padding=padding)
            self.blur_layer = BlurLayer3D(blur_kernel, stride=stride)
        else:
            self.reduction = nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=stride, padding=padding)
        self.swin_abs_pos = swin_abs_pos
        self.activation = nn.GELU()
        self.layer_norm = nn.LayerNorm(out_dim)
        self.device = device
        self.pos_object = PositionEmbeddingSine3d(num_pos_feats=out_dim//3, normalize=True)

    def forward(self, x):
        x = self.reduction(x)
        B, C, D, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1)
        x = self.layer_norm(x)
        x = self.activation(x)
        if self.blur:
            x = x.permute(0, 4, 1, 2, 3)
            x = self.blur_layer(x)
        else:
            x = x.permute(0, 4, 1, 2, 3).view(B, self.out_dim, D, H, W)
        if self.swin_abs_pos:
            abs_pos = self.pos_object(shape_util=(B, D, H, W), device=self.device)
            x = x + abs_pos
        
        return x


class ConvLayerDiscriminator(nn.Module):
    def __init__(self, input_resolution, in_dim, out_dim, nb_se_blocks, dpr, shortcut):
        super().__init__()
        self.input_resolution = input_resolution
        self.out_dim = out_dim
        dims = torch.linspace(in_dim, out_dim, nb_se_blocks+1).int()
        self.shortcut = shortcut
        if shortcut:
            self.norm = nn.Sequential(nn.BatchNorm2d(out_dim), nn.GELU())

        self.blocks = nn.ModuleList()
        for i in range(nb_se_blocks):
            down = spectral_norm(nn.Conv2d(dims[i], dims[i+1], 1, padding=0))
            layer = DiscriminatorECABasicBlock(inplanes=dims[i], planes=dims[i+1], drop_path=dpr[i], stride=1, downsample=down)
            self.blocks.append(layer)
    
    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        x = x.permute(0, 2, 1).view(B, C, H, W)

        if self.shortcut:
            shortcut = x
        
        for layer in self.blocks:
            x = layer(x)

        if self.shortcut:
            x = self.norm(x + shortcut)

        x = x.permute(0, 2, 3, 1).view(B, L, self.out_dim)

        return x

class RFR(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, nb_blocks, norm, dpr=None):
        super().__init__()
        self.out_dim = out_dim
        dims = torch.linspace(in_dim, out_dim, nb_blocks+1).int().tolist()

        self.blocks = nn.ModuleList()
        for i in range(nb_blocks):
            down = nn.Conv2d(dims[i], dims[i+1], 1, padding=0)
            #layer = ECABasicBlock(inplanes=dims[i], planes=dims[i+1], drop_path=dpr[i], stride=1, downsample=down, norm=norm)
            layer = RescaleBasicBlock(inplanes=dims[i], planes=dims[i+1], drop_path=dpr[i], stride=1, downsample=down, norm=norm)
            self.blocks.append(layer)
    
    def forward(self, x):
        assert x.dim() == 4
        
        for layer in self.blocks:
            x = layer(x)
    
        return x

class RFR_1d(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, nb_blocks, norm, dpr=None):
        super().__init__()
        self.out_dim = out_dim
        dims = torch.linspace(in_dim, out_dim, nb_blocks+1).int().tolist()

        self.blocks = nn.ModuleList()
        for i in range(nb_blocks):
            down = nn.Conv1d(dims[i], dims[i+1], 1, padding=0)
            #layer = ECABasicBlock(inplanes=dims[i], planes=dims[i+1], drop_path=dpr[i], stride=1, downsample=down, norm=norm)
            layer = RescaleBasicBlock1D(inplanes=dims[i], planes=dims[i+1], drop_path=dpr[i], kernel_size=kernel_size, downsample=down, norm=norm)
            self.blocks.append(layer)
    
    def forward(self, x):
        assert x.dim() == 3
        
        for layer in self.blocks:
            x = layer(x)
    
        return x


class Resblock1D(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, nb_blocks, norm, dpr=None):
        super().__init__()
        self.out_dim = out_dim
        dims = torch.linspace(in_dim, out_dim, nb_blocks+1).int().tolist()

        self.blocks = nn.ModuleList()
        for i in range(nb_blocks):
            down = nn.Conv1d(dims[i], dims[i+1], 1, padding=0)
            #layer = ECABasicBlock(inplanes=dims[i], planes=dims[i+1], drop_path=dpr[i], stride=1, downsample=down, norm=norm)
            layer = ResnetBasicBlock1D(inplanes=dims[i], planes=dims[i+1], drop_path=dpr[i], kernel_size=kernel_size, downsample=down, norm=norm)
            self.blocks.append(layer)
    
    def forward(self, x):
        assert x.dim() == 3
        
        for layer in self.blocks:
            x = layer(x)
    
        return x


class Resblock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, nb_blocks, norm, dpr=None):
        super().__init__()
        self.out_dim = out_dim
        dims = torch.linspace(in_dim, out_dim, nb_blocks+1).int()

        self.blocks = nn.ModuleList()
        for i in range(nb_blocks):
            down = nn.Conv2d(dims[i], dims[i+1], 1, padding=0)
            #layer = ECABasicBlock(inplanes=dims[i], planes=dims[i+1], drop_path=dpr[i], stride=1, downsample=down, norm=norm)
            layer = ResnetBasicBlock(inplanes=dims[i], planes=dims[i+1], drop_path=dpr[i], stride=1, downsample=down, norm=norm)
            self.blocks.append(layer)
    
    def forward(self, x):
        assert x.dim() == 4
        
        for layer in self.blocks:
            x = layer(x)
    
        return x


class ConvLayer3D(nn.Module):
    def __init__(self, in_dim, out_dim, nb_blocks, dpr, stride=1):
        super().__init__()
        dims = torch.linspace(in_dim, out_dim, nb_blocks+1).int()

        self.blocks = nn.ModuleList()
        for i in range(nb_blocks):
            down = nn.Conv3d(dims[i], dims[i+1], 1, padding=0, stride=stride)
            layer = RescaleBasicBlock3D(inplanes=dims[i], planes=dims[i+1], drop_path=dpr[i], stride=1, downsample=down)
            self.blocks.append(layer)
    
    def forward(self, x):

        for layer in self.blocks:
            x = layer(x)
    
        return x


#class ConvLayer3D(nn.Module):
#    def __init__(self, in_dim, out_dim, nb_blocks):
#        super().__init__()
#        self.out_dim = out_dim
#        dims = torch.linspace(in_dim, out_dim, nb_blocks+1).int()
#
#        self.blocks = nn.ModuleList()
#        for i in range(nb_blocks):
#            layer = nn.Sequential(nn.Conv3d(in_channels=dims[i], out_channels=dims[i+1], kernel_size=3, padding=1),
#                                rescale_layer_3d(dims[i+1]),
#                                nn.BatchNorm3d(dims[i+1]),
#                                nn.GELU())
#            self.blocks.append(layer)
#    
#    def forward(self, x):
#        
#        for layer in self.blocks:
#            x = layer(x)
#    
#        return x


class ResnetConvLayer(nn.Module):
    def __init__(self, input_resolution, in_dim, out_dim, nb_se_blocks, dpr):
        super().__init__()
        self.input_resolution = input_resolution
        self.out_dim = out_dim
        dims = torch.linspace(in_dim, out_dim, nb_se_blocks+1).int()

        self.blocks = nn.ModuleList()
        for i in range(nb_se_blocks):
            down = nn.Conv2d(dims[i], dims[i+1], 1, padding=0)
            layer = ResnetBasicBlock(inplanes=dims[i], planes=dims[i+1], drop_path=dpr[i], stride=1, downsample=down)
            self.blocks.append(layer)
    
    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        x = x.permute(0, 2, 1).view(B, C, H, W)
        
        for layer in self.blocks:
            x = layer(x)

        x = x.permute(0, 2, 3, 1).view(B, L, self.out_dim)
    
        return x

class ConvLayerAdaIN(nn.Module):
    def __init__(self, input_resolution, latent_size, in_dim, out_dim, nb_se_blocks, dpr):
        super().__init__()
        self.input_resolution = input_resolution
        self.out_dim = out_dim
        dims = torch.linspace(in_dim, out_dim, nb_se_blocks+1).int()

        self.blocks = nn.ModuleList()
        for i in range(nb_se_blocks):
            down = nn.Conv2d(dims[i], dims[i+1], 1, padding=0)
            layer = AdaINECABasicBlock(input_resolution=input_resolution, latent_size=latent_size, inplanes=dims[i], planes=dims[i+1], drop_path=dpr[i], stride=1, downsample=down)
            self.blocks.append(layer)
    
    def forward(self, x, y):
        H, W = self.input_resolution
        B, L, C = x.shape
        x = x.permute(0, 2, 1).view(B, C, H, W)
        
        for layer in self.blocks:
            x = layer(x, y)

        x = x.permute(0, 2, 3, 1).view(B, L, self.out_dim)
    
        return x

class ConvDownBlock_3d(nn.Module):
    def __init__(self, in_dim, out_dim, stride):
        super().__init__()

        down1 = nn.Conv3d(in_channels=in_dim, out_channels=out_dim, kernel_size=1, stride=1, padding=0)
        self.conv_block_1 = SEBasicBlock_3d(inplanes=in_dim, planes=out_dim, stride=1, downsample=down1)
        self.conv_block_2 = SEBasicBlock_3d(inplanes=out_dim, planes=out_dim, stride=1)
        self.conv_block_3 = ConvBlock_3d(stride=stride, in_chans=out_dim, embed_dim=out_dim)
    
    def forward(self, x):
        B, C, D, H, W = x.shape
        
        x1 = self.conv_block_1(x)
        x2 = self.conv_block_2(x1)
        x3 = self.conv_block_3(x2)

        return x3, x2

class DownsampleLayer(nn.Module):
    def __init__(self, input_resolution, blur, blur_kernel):
        super().__init__()
        self.input_resolution = input_resolution
        if blur:
            self.down_layer = BlurLayer(blur_kernel=blur_kernel, stride=2)
        else:
            self.down_layer = nn.AvgPool2d(2)
    
    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        x = x.permute(0, 2, 1).view(B, C, H, W)

        x = self.down_layer(x)

        x = x.permute(0, 2, 3, 1).view(B, -1, C)
        return x

class ToGrayscale(nn.Module):
    def __init__(self, input_resolution, in_dim, out_dim):
        super().__init__()
        self.input_resolution = input_resolution
        self.to_grayscale = nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=1)
    
    def forward(self, x):
        B, L, C = x.shape
        H, W = self.input_resolution
        x = x.permute(0, 2, 1).view(B, C, H, W)
        x = self.to_grayscale(x)
        return x

class FromGrayscale(nn.Module):
    def __init__(self, out_dim, in_dim):
        super().__init__()
        self.from_grayscale = nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=1)
        self.out_dim = out_dim
        self.in_dim = in_dim
    
    def forward(self, x):
        x = self.from_grayscale(x)
        return x


class BlurLayer(nn.Module):
    def __init__(self, blur_kernel, stride):
        super(BlurLayer, self).__init__()
        kernel = torch.tensor(blur_kernel, dtype=torch.float32)
        kernel = kernel[:, None] * kernel[None, :]
        kernel = kernel[None, None]
        kernel = kernel / kernel.sum()
        self.register_buffer('kernel', kernel)
        self.stride = stride

    def forward(self, x):
        # expand kernel channels
        kernel = self.kernel.expand(x.size(1), -1, -1, -1)
        x = F.conv2d(
            x,
            kernel,
            stride=self.stride,
            padding=int((self.kernel.size(2) - 1) / 2),
            groups=x.size(1)
        )
        return x


class BlurLayer3D(nn.Module):
    def __init__(self, blur_kernel, stride):
        super(BlurLayer3D, self).__init__()
        kernel = torch.tensor(blur_kernel, dtype=torch.float32)
        kernel = kernel[:, None, None] * kernel[None, None, :] * kernel[None, :, None]
        kernel = kernel[None, None]
        kernel = kernel / kernel.sum()
        self.register_buffer('kernel', kernel)
        self.stride = stride

    def forward(self, x):
        # expand kernel channels
        kernel = self.kernel.expand(x.size(1), -1, -1, -1, -1)
        x = F.conv3d(
            x,
            kernel,
            stride=self.stride,
            padding=int((self.kernel.size(2) - 1) / 2),
            groups=x.size(1)
        )
        return x


# function to calculate the Exponential moving averages for the Generator weights
# This function updates the exponential average weights based on the current training
# Copy from: https://github.com/akanimax/pro_gan_pytorch
def update_average(model_tgt, model_src, beta):
    """
    update the model_target using exponential moving averages
    :param model_tgt: target model
    :param model_src: source model
    :param beta: value of decay beta
    :return: None (updates the target model)
    """

    # utility function for toggling the gradient requirements of the models
    def toggle_grad(model, requires_grad):
        for p in model.parameters():
            p.requires_grad_(requires_grad)

    # turn off gradient calculation
    toggle_grad(model_tgt, False)
    toggle_grad(model_src, False)

    param_dict_src = dict(model_src.named_parameters())

    for p_name, p_tgt in model_tgt.named_parameters():
        p_src = param_dict_src[p_name]
        assert (p_src is not p_tgt)
        p_tgt.copy_(beta * p_tgt + (1. - beta) * p_src)

    # turn back on the gradient calculation
    toggle_grad(model_tgt, True)
    toggle_grad(model_src, True)


class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16*dim, bias=False)
        self.output_dim = dim 
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C//(self.dim_scale**2))
        x = x.view(B,-1,self.output_dim)
        x= self.norm(x)

        return x

#class CCA(nn.Module):
#    def __init__(self, dim):
#        super().__init__()
#        self.sigmoid = nn.Sigmoid()
#        self.avg_pool = nn.AdaptiveAvgPool2d(1)
#        self.linear1 = nn.Linear(dim, dim)
#        self.linear2 = nn.Linear(dim, dim)
#    
#    def forward(self, rescaled, rescale):
#        rescale_out = self.avg_pool(rescale)
#        rescaled_out = self.avg_pool(rescaled)
#
#        rescale_out = self.linear1(rescale_out)
#        rescaled_out = self.linear2(rescaled_out)
#
#        weights = (rescaled_out + rescale_out) / 2
#        weights = self.sigmoid(weights).expand_as(rescaled)
#
#        rescaled = rescaled * weights
#
#        return rescaled

class CCA(nn.Module):
    def __init__(self, dim, input_resolution):
        super().__init__()
        self.input_resolution = input_resolution
        #self.reduction = reduction
        #self.sigmoid = nn.Sigmoid()
        #self.avg_pool = nn.AdaptiveAvgPool2d(1)
        #self.linear = nn.Linear(dim, dim)
        #self.norm = nn.LayerNorm(dim)
        down = nn.Conv2d(dim, dim//2, 1, padding=0)
        self.reduction = RescaleBasicBlock(inplanes=dim, planes=dim//2, drop_path=0, stride=1, downsample=down)
        #self.linear2 = nn.Linear(dim, dim//2)
    
    def forward(self, x):
        B, L, C = x.shape
        H, W = self.input_resolution

        x = x.permute(0, 2, 1).view(B, C, H, W)
        self.reduction(x)
        x = x.permute(0, 2, 3, 1).view(B, L, C)

        #x = self.norm(x)

        #weights = self.avg_pool(x).squeeze(-1).squeeze(-1)
        #weights = self.linear(weights)[:, :, None, None]
        #weights = self.sigmoid(weights).expand_as(x)

        #x = x * weights

        #x = self.linear2(x)

        return x

class CCA3D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.linear = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.linear2 = nn.Linear(dim, dim//2)
    
    def forward(self, x):
        B, C, D, H, W = x.shape

        x = x.permute(0, 2, 3, 4, 1).view(B, -1, C)
        x = self.norm(x)
        x = x.permute(0, 2, 1).view(B, C, D, H, W)

        weights = self.avg_pool(x).squeeze(-1).squeeze(-1).squeeze(-1)
        weights = self.linear(weights)[:, :, None, None, None]
        weights = self.sigmoid(weights).expand_as(x)

        x = x * weights

        x = x.permute(0, 2, 3, 4, 1).view(B, -1, C)
        x = self.linear2(x)
        x = x.permute(0, 2, 1).view(B, C//2, D, H, W)

        return x

def rescale(rescaled, rescaler):
    rescaled_mean = torch.mean(rescaled, dim=(2, 3), keepdim=True)
    rescaled_std = torch.std(rescaled, dim=(2, 3), keepdim=True)
    rescaler_mean = torch.mean(rescaler, dim=(2, 3), keepdim=True)
    rescaler_std = torch.std(rescaler, dim=(2, 3), keepdim=True)
    out = ((rescaled - rescaled_mean) / rescaled_std) * rescaler_std + rescaler_mean
    return out

class MixLayer(nn.Module):
    def __init__(self, in_dim, out_dim, output_res):
        super().__init__()
        self.output_res = output_res[0]

        decoder_cca_layer = CCA(dim=in_dim)
        self.decoder_fusion_layer = nn.Sequential(decoder_cca_layer, nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=1))

    def forward(self, second_input, first_input):
        B, L, C = second_input.shape
        second_input = second_input.permute(0, 2, 1).contiguous().view(B, C, self.output_res // 2, self.output_res // 2)
        first_input = first_input.permute(0, 2, 1).contiguous().view(B, C, self.output_res, self.output_res)

        #second_input = rescale(rescaled=second_input, rescaler=first_input)

        pad_size = second_input.shape[-1] // 2
        pad = (pad_size, pad_size, pad_size, pad_size)
        second_input = torch.nn.functional.pad(second_input, pad, mode='constant', value=0.0)

        #new[:, :, :(pad_size)] = first_input[:, :, :(pad_size)]
        #new[:, :, (pad_size * 3):] = first_input[:, :, (pad_size * 3):]
        #new[:, :, :, :(pad_size)] = first_input[:, :, :, :(pad_size)]
        #new[:, :, :, (pad_size * 3):] = first_input[:, :, :, (pad_size * 3):]

        out = torch.cat([second_input, first_input], dim=1)
        out = self.decoder_fusion_layer(out)
        B, C, H, W = out.shape
        #new = self.norm_layer(new)

        out = out.permute(0, 2, 3, 1).view(B, H * W, C)
        return out

class To_image(nn.Module):
    def __init__(self, input_resolution):
        super().__init__()
        self.input_resolution = input_resolution

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        x = x.permute(0, 2, 1)
        x = x.view(B, C, H, W)
        return x

class To_image3D(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        B, D, H, W, C = x.shape
        x = x.permute(0, 4, 1, 2, 3)
        return x

class From_image(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)
        x = x.view(B, -1, C)
        return x

class From_image3D(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        B, C, D, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1)
        return x

class GetCrossSimilarityMatrix(nn.Module):
    def __init__(self, similarity_down_scale=None):
        super().__init__()
        self.scale = similarity_down_scale

    def rescale(self, x):
        assert x.shape[1] == 4
        my_max = torch.max(torch.flatten(x, start_dim=2), dim=-1)[0].unsqueeze(-1).unsqueeze(-1)
        my_min = torch.min(torch.flatten(x, start_dim=2), dim=-1)[0].unsqueeze(-1).unsqueeze(-1)
        x = torch.nn.functional.interpolate(x, scale_factor=(1/self.scale), mode='bicubic', antialias=True)
        x = torch.clamp(x, my_min, my_max)
        return x
    
    def normalize(self, x):
        x = torch.flatten(x, start_dim=2)
        norm = torch.linalg.norm(x, dim=1, keepdim=True)
        x = x / torch.max(norm, torch.tensor([1e-8], device=x.device))
        return x
    
    def forward(self, x1, x2=None):
        if x2 == None:
            x2 = x1
        if self.scale is not None:
            x1 = self.rescale(x1)
            x2 = self.rescale(x2)
        
        x1 = self.normalize(x1)
        x2 = self.normalize(x2)
        
        x = torch.matmul(torch.transpose(x1, dim0=1, dim1=2), x2)
        return x

class GetSeparability(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        B, C, H, W = x.shape
        x = torch.flatten(x, start_dim=2)
        x = x - x.mean(dim=2).unsqueeze(-1)
        norm = torch.linalg.norm(x, dim=2, keepdim=True)
        x = x / torch.max(norm, torch.tensor([1e-8], device=x.device))
        x = torch.matmul(x, torch.transpose(x, dim0=1, dim1=2))
        #x = torch.matmul(x, torch.transpose(x, dim0=1, dim1=2)) / (H * W - 1)
        x = torch.clamp(x, -1, 1)

        #matplotlib.use('QtAgg')
        #mask = torch.eye(384, dtype=bool).unsqueeze(0).repeat(2, 1, 1)
        #print(torch.unique(x[mask]))
        #fig, ax = plt.subplots(1, 1)
        #ax.imshow(x[0].detach().cpu(), cmap='plasma')
        #plt.show()

        x = torch.abs(x)
        #x = (x + 1) / 2
        return x


class ReplicateChannels(nn.Module):
    def __init__(self, nb_classes):
        super().__init__()
        self.nb_classes = nb_classes
    
    def forward(self, x):
        assert x.dim() == 4
        out = torch.tile(x, dims=(1, self.nb_classes, 1, 1))
        return out


class SelFuseFeature(nn.Module):
    def __init__(self, in_channels, shift_n, norm):
        super(SelFuseFeature, self).__init__()
        
        self.shift_n = shift_n
        self.fuse_conv = nn.Sequential(nn.Conv2d(in_channels*2, in_channels, kernel_size=1, padding=0),
                                    norm(in_channels),
                                    nn.GELU(),
                                    )
        

    def forward(self, x, df):
        N, _, H, W = df.shape
        mag = torch.sqrt(torch.sum(df ** 2, dim=1))
        greater_mask = mag > 0.5
        greater_mask = torch.stack([greater_mask, greater_mask], dim=1)
        df[~greater_mask] = 0

        scale = 1.
        
        grid = torch.stack(torch.meshgrid(torch.arange(H), torch.arange(W)), dim=0)
        grid = grid.expand(N, -1, -1, -1).to(x.device, dtype=torch.float).requires_grad_()
        grid = grid + scale * df

        grid = grid.permute(0, 2, 3, 1).transpose(1, 2)
        grid_ = grid + 0.
        grid[...,0] = 2*grid_[..., 0] / (H-1) - 1
        grid[...,1] = 2*grid_[..., 1] / (W-1) - 1

        select_x = x.clone()
        for _ in range(self.shift_n):
            select_x = F.grid_sample(select_x, grid, mode='bilinear', padding_mode='border')

        select_x = self.fuse_conv(torch.cat([x, select_x], dim=1))
        return select_x

class GetProbabilitiesMatrix(nn.Module):
    def __init__(self, similarity_down_scale):
        super().__init__()
        self.scale = similarity_down_scale
    
    def forward(self, x):
        my_max = torch.max(torch.flatten(x, start_dim=2), dim=-1)[0].unsqueeze(-1).unsqueeze(-1)
        my_min = torch.min(torch.flatten(x, start_dim=2), dim=-1)[0].unsqueeze(-1).unsqueeze(-1)
        x = torch.nn.functional.interpolate(x, scale_factor=(1/self.scale), mode='bicubic', antialias=True)
        x = torch.clamp(x, my_min, my_max)
        x = torch.flatten(x, start_dim=2)
        x = torch.matmul(torch.transpose(x, dim0=1, dim1=2), x)
        return x

def plot_grad_flow(named_parameters, keywords):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    #max_grads= []
    layers = []
    for n, p in named_parameters:
        for key in keywords:
            if(p.requires_grad and key in n) and ("bias" not in n):
                layers.append(n)
                ave_grads.append(p.grad.abs().mean().cpu())
                #max_grads.append(p.grad.abs().max().cpu())

    layers = layers[::1]
    ave_grads = ave_grads[::1]
    print(f'Average gradient is {np.array(ave_grads).mean()}')
    #max_grads = max_grads[::1]
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    #ax.bar(np.arange(len(ave_grads)), max_grads, lw=1, color="c")
    ax.bar(np.arange(len(ave_grads)), ave_grads, lw=1, color="b")
    ax.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    ax.set_xticks(range(0,len(ave_grads), 1), minor=False)
    ax.set_xticklabels(layers, rotation=90, fontdict=None, minor=False)
    #ax.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    #ax.axis(xmin=0, xmax=len(ave_grads), ymin= -0.001, ymax=0.02)
    #ax.xlim(left=0, right=len(ave_grads))
    #ax.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    ax.set_xlabel("Layers")
    ax.set_ylabel("average gradient")
    ax.set_title("Gradient flow")
    #ax.xlabel("Layers")
    #ax.ylabel("average gradient")
    #ax.title("Gradient flow")
    ax.grid(True)
    ax.legend([Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['mean-gradient', 'zero-gradient'])
    fig.subplots_adjust(bottom=0.8)
    
    plt.show()


def compute_losses(labeled_loss_object, reconstruction_loss, similarity_loss, x, y, out, dist_map):
    reconstruction_loss = reconstruction_loss(out['reconstructed'][-1], x)

    similarity_loss = similarity_loss(out['decoder_sm'], out['reconstruction_sm'])

    computed_loss = labeled_loss_object(out['pred'][-1], y, 1, dist_maps=dist_map)

    return torch.tensor([computed_loss, reconstruction_loss, similarity_loss])