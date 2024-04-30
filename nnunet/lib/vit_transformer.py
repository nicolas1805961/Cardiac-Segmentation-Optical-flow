import copy
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch import nn
from .vit_rpe import MultiheadAttention, get_indices_2d, get_indices_3d
from timm.models.layers import DropPath
from .position_embedding import PositionEmbeddingSine2d, PositionEmbeddingSine3d, PositionEmbeddingSine1d
from .seresnet import rescale_layer
from .utils import ConvBlocks2DGroupLegacy, ConvBlocks2DBatch, Mlp, To_image, From_image, RFR_1d, MLP
import matplotlib
import matplotlib.pyplot as plt
from math import ceil
from torch.nn.functional import grid_sample
from torch.nn.functional import pad
import matplotlib.patches as patches
from torchvision.ops import roi_align
import numpy as np
import math
import numbers
from torch.nn.init import xavier_uniform_, constant_

from einops import rearrange
from einops.layers.torch import Rearrange
from torch.nn import init
from torch.nn.functional import affine_grid

from nnunet.lib.spacetimeAttention import AttentionLearnedSin


class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel.to('cuda:0'))
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups)


class AlignConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
                nn.Conv2d(in_channels=2 * dim, out_channels=dim, kernel_size=3, padding='same'),
                nn.GELU(),
                nn.Conv2d(in_channels=dim, out_channels=2, kernel_size=3, padding='same'),
            )
    
    
    def forward(self, x1, x2):
        B, C, H, W = x1.shape

        x = torch.cat([x1, x2], dim=1)

        offsets = self.mlp(x)
        offsets = offsets.permute(0, 2, 3, 1).contiguous()

        ref_y, ref_x = torch.meshgrid(torch.linspace(-1, 1, H, dtype=torch.float32, device=x1.device), torch.linspace(-1, 1, W, dtype=torch.float32, device=x1.device))  # (h, w)
        ref = torch.stack((ref_x, ref_y), -1)[None].repeat(B, 1, 1, 1)

        return grid_sample(x1, ref + offsets, mode='bilinear', align_corners=True)


class AlignLinear(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        self.linear1 = nn.Linear(in_features=dim, out_features=dim)
        self.linear2 = nn.Linear(in_features=dim, out_features=dim)

        self.mlp = nn.Sequential(
                nn.Linear(2 * dim, 2048),
                nn.GELU(),
                nn.Linear(2048, 2),
            )

        self.pos_obj_2d = PositionEmbeddingSine2d(num_pos_feats=dim//2, normalize=True)
    
    
    def forward(self, x1, x2):
        B, C, H, W = x1.shape

        pos_2d = self.pos_obj_2d(shape_util=(B, H, W), device=x1.device)
        pos_2d = pos_2d.permute(0, 2, 3, 1).contiguous()
        pos_2d = pos_2d.view(B, H * W, C)

        x1 = x1.permute(0, 2, 3, 1).contiguous() # B, H, W, C
        x1 = x1.view(B, H * W, C)

        x2 = x2.permute(0, 2, 3, 1).contiguous() # B, H, W, C
        x2 = x2.view(B, H * W, C)

        x1 = self.linear1(x1 + pos_2d)
        x2 = self.linear2(x2 + pos_2d)

        x = torch.cat([x1, x2], dim=2)

        offsets = self.mlp(x)
        offsets = offsets.view(B, H, W, 2)

        ref_y, ref_x = torch.meshgrid(torch.linspace(-1, 1, H, dtype=torch.float32, device=x1.device), torch.linspace(-1, 1, W, dtype=torch.float32, device=x1.device))  # (h, w)
        ref = torch.stack((ref_x, ref_y), -1)[None].repeat(B, 1, 1, 1)

        x1 = x1.view(B, H, W, C)
        x1 = x1.permute(0, 3, 1, 2).contiguous()

        return grid_sample(x1, ref + offsets, mode='bilinear', align_corners=True)
       

class Sampler(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Linear(dim, 2 * 50)   
        self.pos_obj_2d = PositionEmbeddingSine2d(num_pos_feats=dim // 2, normalize=True) 
    
    def forward(self, query, key, value):
        T, B, C, H, W = key.shape

        query = query.permute(0, 2, 3, 1).contiguous() # B, H, W, C
        query = query.view(B, H * W, C)

        pos_2d = self.pos_obj_2d(shape_util=(B, H, W), device=query.device)
        pos_2d = pos_2d.permute(0, 2, 3, 1).contiguous()
        pos_2d = pos_2d.view(B, H * W, C)

        offsets = self.mlp(query + pos_2d)
        offsets = offsets.view(B, H*W, 50, 2)
        offsets = offsets[None].repeat(T, 1, 1, 1, 1)
        offsets = offsets.view(T*B, H*W, 50, 2)

        ref_y, ref_x = torch.meshgrid(torch.linspace(-1, 1, H, dtype=torch.float32, device=query.device), torch.linspace(-1, 1, W, dtype=torch.float32, device=query.device))  # (h, w)
        ref = torch.stack((ref_x, ref_y), -1)
        ref = ref.view(-1, 2)[None, :, None, :].repeat(T*B, 1, 50, 1)

        key = key.view(T*B, C, H, W)
        value = value.view(T*B, C, H, W)

        print(key.shape)
        print((ref + offsets).shape)

        sampled_key = grid_sample(key, ref + offsets, mode='bilinear', align_corners=True)
        sampled_value = grid_sample(value, ref + offsets, mode='bilinear', align_corners=True)

        print(sampled_key.shape)
        
        return sampled_key, sampled_value


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



def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class RelativeAttention1D(nn.Module):
    def __init__(self, inp, oup, seq_length, heads, dim_head, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == inp)
        self.seq_length = seq_length

        self.heads = heads
        self.scale = dim_head ** -0.5

        # parameter table of relative position bias
        self.relative_bias_table = nn.Parameter(torch.zeros(2 * seq_length - 1, heads))

        coords = torch.arange(seq_length)
        coords = torch.stack(torch.meshgrid([coords]))
        coords = torch.flatten(coords, 1)
        relative_coords = coords[:, :, None] - coords[:, None, :]

        relative_coords[0] += seq_length - 1
        relative_coords = rearrange(relative_coords, 'c h w -> h w c')
        relative_index = relative_coords.sum(-1).flatten().unsqueeze(1)
        self.register_buffer("relative_index", relative_index)

        self.attend = nn.Softmax(dim=-1)
        self.to_q = nn.Linear(inp, inner_dim, bias=False)
        self.to_k = nn.Linear(inp, inner_dim, bias=False)
        self.to_v = nn.Linear(inp, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, oup),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        # B, L, E
        q = self.to_q(query)
        k = self.to_k(key)
        v = self.to_v(value)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), [q, k, v])

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # Use "gather" for more efficiency on GPUs
        relative_bias = self.relative_bias_table.gather(0, self.relative_index.repeat(1, self.heads))
        relative_bias = rearrange(relative_bias, '(h w) c -> 1 c h w', h=self.seq_length, w=self.seq_length)
        dots = dots + relative_bias

        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class ContextualRelativeAttention1D(nn.Module):
    def __init__(self, inp, oup, seq_length, heads, dim_head, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == inp)
        self.seq_length = seq_length

        self.heads = heads
        self.scale = dim_head ** -0.5

        # parameter table of relative position bias
        self.relative_bias_table = nn.Parameter(torch.zeros(2 * seq_length - 1, heads))

        coords = torch.arange(seq_length)
        coords = torch.stack(torch.meshgrid([coords]))
        coords = torch.flatten(coords, 1)
        relative_coords = coords[:, :, None] - coords[:, None, :]

        relative_coords[0] += seq_length - 1
        relative_coords = rearrange(relative_coords, 'c h w -> h w c')
        relative_index = relative_coords.sum(-1).flatten().unsqueeze(1)
        self.register_buffer("relative_index", relative_index)

        self.attend = nn.Softmax(dim=-1)
        self.to_q = nn.Linear(inp, inner_dim, bias=False)
        self.to_k = nn.Linear(inp, inner_dim, bias=False)
        self.to_v = nn.Linear(inp, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, oup),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        # B, L, E
        q = self.to_q(query)
        k = self.to_k(key)
        v = self.to_v(value)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), [q, k, v])

        # Use "gather" for more efficiency on GPUs
        relative_bias = self.relative_bias_table.gather(0, self.relative_index.repeat(1, self.heads))
        relative_bias = rearrange(relative_bias, '(h w) c -> 1 c h w', h=self.seq_length, w=self.seq_length)

        relative_bias_q = torch.matmul(q.transpose(-1, -2), relative_bias).transpose(-1, -2)
        relative_bias_k = torch.matmul(k.transpose(-1, -2), relative_bias).transpose(-1, -2)
        relative_bias_v = torch.matmul(v.transpose(-1, -2), relative_bias).transpose(-1, -2)
        q = q + relative_bias_q
        k = k + relative_bias_k
        v = v + relative_bias_v

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class ContextualRelativeAttention2D(nn.Module):
    def __init__(self, inp, oup, image_size, heads, dim_head, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == inp)

        self.ih, self.iw = image_size

        self.heads = heads
        self.scale = dim_head ** -0.5

        # parameter table of relative position bias
        self.relative_bias_table = nn.Parameter(
            torch.zeros((2 * self.ih - 1) * (2 * self.iw - 1), heads))

        coords = torch.meshgrid((torch.arange(self.ih), torch.arange(self.iw)))
        coords = torch.flatten(torch.stack(coords), 1)
        relative_coords = coords[:, :, None] - coords[:, None, :]

        relative_coords[0] += self.ih - 1
        relative_coords[1] += self.iw - 1
        relative_coords[0] *= 2 * self.iw - 1
        relative_coords = rearrange(relative_coords, 'c h w -> h w c')
        relative_index = relative_coords.sum(-1).flatten().unsqueeze(1)
        self.register_buffer("relative_index", relative_index)

        self.attend = nn.Softmax(dim=-1)
        self.to_q = nn.Linear(inp, inner_dim, bias=False)
        self.to_k = nn.Linear(inp, inner_dim, bias=False)
        self.to_v = nn.Linear(inp, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, oup),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        # B, L, E
        q = self.to_q(query)
        k = self.to_k(key)
        v = self.to_v(value)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), [q, k, v])

        # Use "gather" for more efficiency on GPUs
        relative_bias = self.relative_bias_table.gather(0, self.relative_index.repeat(1, self.heads))
        relative_bias = rearrange(relative_bias, '(h w) c -> 1 c h w', h=self.ih*self.iw, w=self.ih*self.iw)

        relative_bias_q = torch.matmul(q.transpose(-1, -2), relative_bias).transpose(-1, -2)
        relative_bias_k = torch.matmul(k.transpose(-1, -2), relative_bias).transpose(-1, -2)
        relative_bias_v = torch.matmul(v.transpose(-1, -2), relative_bias).transpose(-1, -2)
        q = q + relative_bias_q
        k = k + relative_bias_k
        v = v + relative_bias_v

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class RelativeAttention2D(nn.Module):
    def __init__(self, inp, oup, image_size, heads, dim_head, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == inp)

        self.ih, self.iw = image_size

        self.heads = heads
        self.scale = dim_head ** -0.5

        # parameter table of relative position bias
        self.relative_bias_table = nn.Parameter(
            torch.zeros((2 * self.ih - 1) * (2 * self.iw - 1), heads))

        coords = torch.meshgrid((torch.arange(self.ih), torch.arange(self.iw)))
        coords = torch.flatten(torch.stack(coords), 1)
        relative_coords = coords[:, :, None] - coords[:, None, :]

        relative_coords[0] += self.ih - 1
        relative_coords[1] += self.iw - 1
        relative_coords[0] *= 2 * self.iw - 1
        relative_coords = rearrange(relative_coords, 'c h w -> h w c')
        relative_index = relative_coords.sum(-1).flatten().unsqueeze(1)
        self.register_buffer("relative_index", relative_index)

        self.attend = nn.Softmax(dim=-1)
        self.to_q = nn.Linear(inp, inner_dim, bias=False)
        self.to_k = nn.Linear(inp, inner_dim, bias=False)
        self.to_v = nn.Linear(inp, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, oup),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        # B, L, E
        q = self.to_q(query)
        k = self.to_k(key)
        v = self.to_v(value)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), [q, k, v])

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # Use "gather" for more efficiency on GPUs
        relative_bias = self.relative_bias_table.gather(0, self.relative_index.repeat(1, self.heads))
        relative_bias = rearrange(relative_bias, '(h w) c -> 1 c h w', h=self.ih*self.iw, w=self.ih*self.iw)
        dots = dots + relative_bias

        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class CrossRelativeAttention(nn.Module):
    def __init__(self, inp, oup, size_2d, rescaled, heads, dim_head, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == inp)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.rescaled = rescaled

        self.ih = self.iw = size_2d
        # parameter table of relative position bias
        self.relative_bias_table_2d = nn.Parameter(torch.zeros((2 * self.ih - 1) * (2 * self.iw - 1), heads))

        coords_2d = torch.meshgrid((torch.arange(self.ih), torch.arange(self.iw)))
        coords_2d = torch.flatten(torch.stack(coords_2d), 1)
        relative_coords_2d = coords_2d[:, :, None] - coords_2d[:, None, :]

        relative_coords_2d[0] += self.ih - 1
        relative_coords_2d[1] += self.iw - 1
        relative_coords_2d[0] *= 2 * self.iw - 1
        relative_coords_2d = rearrange(relative_coords_2d, 'c h w -> h w c')
        relative_index_2d = relative_coords_2d.sum(-1).flatten().unsqueeze(1)
        self.register_buffer("relative_index_2d", relative_index_2d)

        self.attend = nn.Softmax(dim=-1)
        self.to_q = nn.Linear(inp, inner_dim, bias=False)
        self.to_k = nn.Linear(inp, inner_dim, bias=False)
        self.to_v = nn.Linear(inp, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, oup),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        # B, L, E
        q = self.to_q(query)
        k = self.to_k(key)
        v = self.to_v(value)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), [q, k, v])

        relative_bias_2d = self.relative_bias_table_2d.gather(0, self.relative_index_2d.repeat(1, self.heads))
        relative_bias_2d = rearrange(relative_bias_2d, '(h w) c -> 1 c h w', h=self.ih*self.iw, w=self.ih*self.iw)

        if self.rescaled == '1d':
            relative_bias_k = torch.matmul(k.transpose(-1, -2), relative_bias_2d).transpose(-1, -2)
            relative_bias_v = torch.matmul(v.transpose(-1, -2), relative_bias_2d).transpose(-1, -2)
            k = k + relative_bias_k
            v = v + relative_bias_v
        elif self.rescaled == '2d':
            relative_bias_q = torch.matmul(q.transpose(-1, -2), relative_bias_2d).transpose(-1, -2)
            q = q + relative_bias_q

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


#class TransformerDecoder(nn.Module):
#
#    def __init__(self, decoder_layer, num_layers, num_frames, norm=None, return_intermediate=False):
#        super().__init__()
#        self.layers = _get_clones(decoder_layer, num_layers)
#        self.num_layers = num_layers
#        self.num_frames = num_frames
#        self.norm = norm
#        self.return_intermediate = return_intermediate
#
#    def pad_zero(self, x, pad, dim=0):
#        if x is None:
#            return None
#        pad_shape = list(x.shape)
#        pad_shape[dim] = pad
#        return torch.cat((x, x.new_zeros(pad_shape)), dim=dim)
#
#    def forward(self, tgt, memory, memory_bus, memory_pos,
#                tgt_mask: Optional[Tensor] = None,
#                memory_mask: Optional[Tensor] = None,
#                tgt_key_padding_mask: Optional[Tensor] = None,
#                memory_key_padding_mask: Optional[Tensor] = None,
#                pos: Optional[Tensor] = None,
#                query_pos: Optional[Tensor] = None,
#                is_train: bool = True):
#        output = tgt
#
#        return_intermediate = (self.return_intermediate and is_train)
#        intermediate = []
#
#        M, bt, c = memory_bus.shape
#        bs = bt // self.num_frames if is_train else 1
#        t = bt // bs
#
#        memory_bus = memory_bus.view(M, bs, t, c).permute(2,0,1,3).flatten(0,1) # TMxBxC
#        memory = torch.cat((memory, memory_bus))
#
#        memory_pos = memory_pos[None, :, None, :].repeat(t,1,bs,1).flatten(0,1) # TMxBxC
#        pos = torch.cat((pos, memory_pos))
#
#        memory_key_padding_mask = self.pad_zero(memory_key_padding_mask, t*M, dim=1) # B, THW
#
#        for layer in self.layers:
#            output = layer(output, memory, tgt_mask=tgt_mask,
#                           memory_mask=memory_mask,
#                           tgt_key_padding_mask=tgt_key_padding_mask,
#                           memory_key_padding_mask=memory_key_padding_mask,
#                           pos=pos, query_pos=query_pos)
#            if return_intermediate:
#                intermediate.append(self.norm(output))
#
#        if self.norm is not None:
#            output = self.norm(output)
#            if return_intermediate:
#                intermediate.pop()
#                intermediate.append(output)
#
#        if return_intermediate:
#            return torch.stack(intermediate)
#
#        return output.unsqueeze(0)


#class TransformerEncoderLayer(nn.Module):
#
#    def __init__(self, 
#                d_model, 
#                nhead, 
#                input_resolution,
#                proj,
#                device,
#                relative_position_index,
#                drop_path,
#                num_memory_token=0,
#                rpe_mode='contextual',
#                rpe_contextual_tensor='qkv',
#                dim_feedforward=3072, 
#                dropout=0.1,
#                activation="gelu", 
#                normalize_before=False):
#        super().__init__()
#
#        self.norm1 = nn.LayerNorm(d_model)
#
#        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#
#        #self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout)
#        self.self_attn = MultiheadAttention(embed_dim=d_model, 
#                                            num_heads=nhead, 
#                                            input_resolution=input_resolution,
#                                            device=device,
#                                            relative_position_index=relative_position_index,
#                                            proj=proj, 
#                                            num_memory_token=num_memory_token, 
#                                            rpe_mode=rpe_mode, 
#                                            rpe_contextual_tensor=rpe_contextual_tensor, 
#                                            dropout=dropout)
#
#        self.norm2 = nn.LayerNorm(d_model)
#        # Implementation of Feedforward model
#        self.linear1 = nn.Linear(d_model, dim_feedforward)
#        self.dropout = nn.Dropout(dropout)
#        self.linear2 = nn.Linear(dim_feedforward, d_model)
#
#        self.dropout1 = nn.Dropout(dropout)
#        self.dropout2 = nn.Dropout(dropout)
#
#        self.activation = _get_activation_fn(activation)
#        self.normalize_before = normalize_before
#
#    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
#        return tensor if pos is None else tensor + pos
#
#    def forward_post(self,
#                     src,
#                     src_mask: Optional[Tensor] = None,
#                     src_key_padding_mask: Optional[Tensor] = None,
#                     pos: Optional[Tensor] = None):
#        q = k = self.with_pos_embed(src, pos)
#        src2 = self.self_attn(q, k, 
#                                value=src, 
#                                attn_mask=src_mask, 
#                                key_padding_mask=src_key_padding_mask)[0]
#        src = src + self.dropout1(src2)
#        src = self.norm1(src)
#        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
#        src = src + self.dropout2(src2)
#        src = self.norm2(src)
#        src = self.drop_path(src)
#        return src
#
#    def forward_pre(self, src,
#                    src_mask: Optional[Tensor] = None,
#                    src_key_padding_mask: Optional[Tensor] = None,
#                    pos: Optional[Tensor] = None):
#        src2 = self.norm1(src)
#        q = k = self.with_pos_embed(src2, pos)
#        src2 = self.self_attn(q, k, 
#                                value=src2, 
#                                attn_mask=src_mask,
#                                key_padding_mask=src_key_padding_mask)[0]
#        src = src + self.dropout1(src2)
#        src2 = self.norm2(src)
#        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
#        src = src + self.dropout2(src2)
#        src = self.drop_path(src)
#        return src
#
#    def forward(self, src, pos,
#                src_mask: Optional[Tensor] = None,
#                src_key_padding_mask: Optional[Tensor] = None):
#        if self.normalize_before:
#            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
#        return self.forward_post(src, src_mask, src_key_padding_mask, pos)

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class CrossTransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, query, key, value,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):

        B, C, H, W = query.shape

        query = torch.flatten(query, start_dim=2).permute(0, 2, 1)
        key = torch.flatten(key, start_dim=2).permute(0, 2, 1)
        value = torch.flatten(value, start_dim=2).permute(0, 2, 1)
        pos = torch.flatten(pos, start_dim=2).permute(0, 2, 1)

        for layer in self.layers:
            output = layer(query=query, key=key, value=value, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        output = output.permute(0, 2, 1).view(B, C, H, W)

        return 

class SlotTransformer(nn.Module):
    def __init__(self, dim, slot_layer, temporal_layer, num_layers, nb_memory_bus, video_length, conv_layer_1d, area_size, use_patches):
        super().__init__()
        self.use_patches = use_patches
        self.area_size = area_size
        self.video_length = video_length
        self.conv_1d = _get_clones(conv_layer_1d, num_layers)
        self.to_slot = _get_clones(slot_layer, num_layers)
        self.temporal_layers = _get_clones(temporal_layer, num_layers)
        self.num_layers = num_layers
        self.nb_memory_bus = nb_memory_bus
        
        self.memory_pos = nn.Parameter(torch.randn(self.nb_memory_bus, dim))
        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, dim))
        init.xavier_uniform_(self.slots_logsigma)
    
    def get_advanced_pos(self, memory_pos, shape, temporal_pos):
        T, B, C, H, W = shape
        temporal_pos = temporal_pos.view(1, T, C).repeat(B, self.nb_memory_bus, 1)
        memory_pos = memory_pos.view(T, B, self.nb_memory_bus, C).permute(1, 0, 2, 3).contiguous().view(B, T * self.nb_memory_bus, C)
        advanced_pos = temporal_pos + memory_pos
        return advanced_pos

    def process_temporal(self, memory_bus, shape, layer_idx, advanced_pos):
        T, B, C, H, W = shape
        memory_bus = memory_bus.view(T, B, self.nb_memory_bus, C).permute(1, 2, 3, 0).flatten(0, 1) # B*M, C, T
        memory_bus = self.conv_1d[layer_idx](memory_bus)
        memory_bus = memory_bus.permute(0, 2, 1).contiguous() # B*M, T, C
        memory_bus = memory_bus.view(B, self.nb_memory_bus, T, C).permute(0, 2, 1, 3).contiguous().view(B, T * self.nb_memory_bus, C)

        memory_bus = self.temporal_layers[layer_idx](memory_bus, pos=advanced_pos) # B, T*M, C

        memory_bus = memory_bus.view(B, T, self.nb_memory_bus, C).permute(1, 0, 2, 3).flatten(0, 1) # T*B, M, C 
        #memory_bus = memory_bus.view(B, self.nb_memory_bus, T, C).permute(2, 0, 1, 3).flatten(0, 1) # T*B, M, C 
        return memory_bus


    def forward(self, spatial_tokens, temporal_pos, spatial_pos):
        shape = spatial_tokens.shape
        T, B, C, H, W = shape

        mu = self.slots_mu.expand(T * B, self.nb_memory_bus, -1)
        sigma = self.slots_logsigma.exp().expand(T * B, self.nb_memory_bus, -1)
        memory_bus = mu + sigma * torch.randn(mu.shape, device=spatial_tokens.device)

        memory_pos = self.memory_pos
        memory_pos = memory_pos[None, :, :].repeat(T * B, 1, 1)

        advanced_pos = self.get_advanced_pos(memory_pos, shape, temporal_pos)

        #spatial_pos = self.pos_2d(shape_util=(T * B, H, W), device=spatial_tokens.device)
        #spatial_pos = spatial_pos.permute(0, 2, 3, 1).view(T * B, H * W, C)
        
        spatial_pos = spatial_pos.repeat(T, 1, 1)
        spatial_tokens = spatial_tokens.permute(0, 1, 3, 4, 2).view(T * B, H, W, C).view(T * B, H * W, C)

        for i in range(self.num_layers):
            #spatial_tokens = self.spatial_layers[i](spatial_tokens, pos=spatial_pos)

            memory_bus = self.to_slot[i](query=memory_bus, key=spatial_tokens, value=spatial_tokens, query_pos=memory_pos, key_pos=spatial_pos)

            memory_bus = self.process_temporal(memory_bus=memory_bus, shape=shape, layer_idx=i, advanced_pos=advanced_pos)

            #spatial_tokens = self.from_slot[i](query=spatial_tokens, key=memory_bus, value=memory_bus, query_pos=spatial_pos, key_pos=memory_pos)

        #spatial_tokens = spatial_tokens.permute(0, 2, 1).view(T, B, C, H * W).view(T, B, C, H, W)
        return memory_bus


#class SpatioTemporalTransformer(nn.Module):
#    def __init__(self, dim, num_heads, num_layers, conv_layer_1d, d_ffn, dropout=0.0):
#        super().__init__()
#        self.num_layers = num_layers
#        
#        spatial_attn_layer = TransformerEncoderLayer(d_model=dim, nhead=num_heads, dim_feedforward=d_ffn)
#        self.spatial_attention = _get_clones(spatial_attn_layer, num_layers)
#
#        temporal_attn_layer = TransformerEncoderLayer(d_model=dim, nhead=num_heads, dim_feedforward=d_ffn)
#        self.temporal_attention = _get_clones(temporal_attn_layer, num_layers)
#
#        self.conv_1d = _get_clones(conv_layer_1d, num_layers)
#        
#        self.pos_weight = nn.Parameter(torch.randn(1, dim))
#
#
#    def forward(self, spatial_tokens, memory_bus, pos_2d, pos_1d):
#        shape = spatial_tokens.shape
#        T, B, C, H, W = shape
#
#        pos_2d = pos_2d.view(1, H * W, C).repeat(T * B, 1, 1)
#        pos_1d = pos_1d.view(1, T, C).repeat(B, 1, 1)
#        pos_weight = self.pos_weight.view(1, 1, C).repeat(T * B, 1, 1)
#
#        memory_bus = memory_bus.view(T, 1, 1, C).repeat(1, B, 1, 1).view(T * B, 1, C)
#        spatial_tokens = spatial_tokens.permute(0, 1, 3, 4, 2).contiguous()
#        spatial_tokens = spatial_tokens.view(T * B, H, W, C).view(T * B, H * W, C)
#
#        for i in range(self.num_layers):
#            src = torch.cat([memory_bus, spatial_tokens], dim=1)
#            pos = torch.cat([pos_weight, pos_2d], dim=1)
#            src = self.spatial_attention[i](src=src, pos=pos)
#
#            spatial_tokens = src[:, 1:] # T*B, H*W, C
#            memory_bus = src[:, 0] # T*B, C
#            memory_bus = memory_bus.view(T, B, C)
#            memory_bus = memory_bus.permute(1, 2, 0).contiguous() # B, C, T
#
#            memory_bus = self.conv_1d[i](memory_bus) # B, C, T
#            memory_bus = memory_bus.permute(0, 2, 1).contiguous() # B, T, C
#
#            memory_bus = self.temporal_attention[i](src=memory_bus, pos=pos_1d) # B, T, C
#
#            memory_bus = memory_bus.permute(1, 0, 2).contiguous() # T, B, C
#            memory_bus = memory_bus.view(T * B, C).unsqueeze(1) # T * B, 1, C
#
#        memory_bus = memory_bus.view(T, B, 1, C).squeeze(2) # T, B, C
#        memory_bus = memory_bus.permute(1, 0, 2).contiguous() # B, T, C
#        spatial_tokens = spatial_tokens.permute(0, 2, 1).contiguous() # T*B, C, H*W
#        spatial_tokens = spatial_tokens.view(T, B, C, H*W).view(T, B, C, H, W)
#        return memory_bus, spatial_tokens


#class ModulationTransformer(nn.Module):
#    def __init__(self, dim, num_heads, num_layers, d_ffn, video_length, dropout=0.0):
#        super().__init__()
#        self.num_layers = num_layers
#
#        self.pos_1d = nn.Parameter(torch.randn(video_length, dim))
#
#        temporal_attn_layer = TransformerEncoderLayer(d_model=dim, nhead=num_heads, dim_feedforward=d_ffn)
#        self.temporal_attention = _get_clones(temporal_attn_layer, num_layers)
#
#    def forward(self, memory_bus, modulation_embedding):
#        B, T, C = memory_bus.shape
#        
#        modulation_embedding = modulation_embedding.view(1, 1, C).repeat(B, T, 1)
#        pos_1d = self.pos_1d.view(1, T, C).repeat(B, 1, 1)
#        pos = pos_1d + modulation_embedding
#
#        for i in range(self.num_layers):
#            #src = torch.cat([memory_bus, modulation_token], dim=1)
#            memory_bus = self.temporal_attention[i](src=memory_bus, pos=pos)
#
#            #memory_bus = src[:, :-1] # B, T, C
#            #modulation_token = src[:, -1] # B, C
#            #modulation_token = modulation_token.unsqueeze(1)
#
#        return memory_bus
    


class ModulationTransformer(nn.Module):
    def __init__(self, dim, num_heads, num_layers, d_ffn, dropout=0.0):
        super().__init__()
        self.num_layers = num_layers

        temporal_attn_layer = TransformerEncoderLayer(d_model=dim, nhead=num_heads, dim_feedforward=d_ffn)
        self.temporal_attention = _get_clones(temporal_attn_layer, num_layers)

        conv_1d = nn.Conv1d(dim, dim, kernel_size=3, padding='same')
        self.conv_1d = _get_clones(conv_1d, num_layers)

        self.pos_obj_2d = PositionEmbeddingSine2d(num_pos_feats=dim // 2, normalize=True)

    def forward(self, memory_bus, spatial_tokens):
        B, T, C = memory_bus.shape
        B, C, H, W = spatial_tokens.shape

        pos_2d = self.pos_obj_2d(shape_util=(B, H, W), device=spatial_tokens.device)
        pos_2d = pos_2d.permute(0, 2, 3, 1).contiguous()
        pos_2d = pos_2d.view(B, H * W, C)

        spatial_tokens = spatial_tokens.permute(0, 2, 3, 1).contiguous()
        spatial_tokens = spatial_tokens.view(B, H * W, C)
        

        for i in range(self.num_layers):
            memory_bus = memory_bus.permute(0, 2, 1).contiguous() # B, C, T

            pos_1d = self.conv_1d[i](memory_bus) # B, C, T
            pos_1d = pos_1d.permute(0, 2, 1).contiguous() # B, T, C
            memory_bus = memory_bus.permute(0, 2, 1).contiguous() # B, T, C

            pos = torch.cat([pos_1d, pos_2d], dim=1)

            src = torch.cat([memory_bus, spatial_tokens], dim=1)
            src = self.temporal_attention[i](src=src, pos=pos)[0]

            memory_bus = src[:, :T] # B, T, C
            spatial_tokens = src[:, T:]

        return memory_bus, spatial_tokens



#class TransformerDecoderLayer(nn.Module):
#    def __init__(self, dim, num_heads, d_ffn, dropout=0.0, nb_memory_bus=4):
#        super().__init__()
#        self.nb_memory_bus = nb_memory_bus
#
#        #self.cross_attn_layer = SlotAttention(dim=dim)
#        self.cross_attn_layer = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True, dropout=dropout)
#        self.norm1 = nn.LayerNorm(dim)
#        self.dropout1 = nn.Dropout(dropout)
#
#        self.self_attn_layer = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True, dropout=dropout)
#        self.norm2 = nn.LayerNorm(dim)
#        self.dropout2 = nn.Dropout(dropout)
#
#        # ffn
#        self.linear1 = nn.Linear(dim, d_ffn)
#        self.activation = nn.GELU()
#        self.dropout3 = nn.Dropout(dropout)
#        self.linear2 = nn.Linear(d_ffn, dim)
#        self.dropout4 = nn.Dropout(dropout)
#        self.norm3 = nn.LayerNorm(dim)
#
#    def forward_ffn(self, tgt):
#        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
#        tgt = tgt + self.dropout4(tgt2)
#        tgt = self.norm3(tgt)
#        return tgt
#    
#    def with_pos_embed(self, x, pos):
#        return x + pos
#
#    def forward(self, query, key, query_pos, key_pos):
#        tgt2 = self.cross_attn_layer(query=self.with_pos_embed(query, query_pos), key=self.with_pos_embed(key, key_pos), value=key)[0]
#        #tgt2 = self.cross_attention[i](query=self.with_pos_embed(object_tokens, memory_pos), key=self.with_pos_embed(spatial_tokens, pos_2d), value=spatial_tokens)[0]
#        query = query + self.dropout1(tgt2)
#        query = self.norm1(query)
#
#        tgt2 = self.self_attn_layer(query=self.with_pos_embed(query, query_pos), key=self.with_pos_embed(query, query_pos), value=query)[0]
#        query = query + self.dropout2(tgt2)
#        query = self.norm2(query)
#
#        query = self.forward_ffn(query)
#
#        #spatial_tokens = self.from_slot[i](query=spatial_tokens, key=object_tokens, value=object_tokens, query_pos=spatial_pos, key_pos=memory_pos)
#
#        #spatial_tokens = spatial_tokens.permute(0, 2, 1).view(T, B, C, H * W).view(T, B, C, H, W)
#        #object_tokens = object_tokens.permute(1, 0, 2).contiguous() # T, B, C
#        return query
    

class TransformerDecoder(nn.Module):
    def __init__(self, dim, num_heads, num_layers, d_ffn, nb_object_bus):
        super().__init__()
        self.nb_object_bus = nb_object_bus
        self.num_layers = num_layers
        self.object_pos = nn.Parameter(torch.randn(self.nb_object_bus, dim))
        self.object_tokens = nn.Parameter(torch.randn(self.nb_object_bus, dim))

        layer = TransformerDecoderLayer(dim=dim, num_heads=num_heads, d_ffn=d_ffn)
        #layer = TransformerEncoderLayer(d_model=dim, nhead=num_heads, dim_feedforward=d_ffn)
        self.layers = _get_clones(layer, num_layers)
    

    def forward(self, spatial_tokens, pos_2d):
        B, L, C = spatial_tokens.shape

        query = self.object_tokens[None, :, :].repeat(B, 1, 1)
        query_pos = self.object_pos[None, :, :].repeat(B, 1, 1)
        pos_2d = pos_2d.view(1, L, C).repeat(B, 1, 1)

        for layer in self.layers:
            query = layer(query=query, key=spatial_tokens, query_pos=query_pos, key_pos=pos_2d)

        #object_tokens = query[:, :4]
        #heatmap_token = query[:, -1].unsqueeze(1)
        return query
    

class TransformerFlowDecoder(nn.Module):
    def __init__(self, dim, num_heads, num_layers, d_ffn):
        super().__init__()
        self.num_layers = num_layers
        self.object_pos = nn.Parameter(torch.randn(6, dim))
        self.object_tokens = nn.Parameter(torch.randn(6, dim))

        layer = TransformerDecoderLayer(dim=dim, num_heads=num_heads, d_ffn=d_ffn)
        #layer = TransformerEncoderLayer(d_model=dim, nhead=num_heads, dim_feedforward=d_ffn)
        self.layers = _get_clones(layer, num_layers)

        self.pos_obj = PositionEmbeddingSine2d(num_pos_feats=dim // 2, normalize=True)
    

    def forward(self, feature_map):
        B, C, H, W = feature_map.shape

        pos = self.pos_obj(shape_util=(B, H, W), device=feature_map.device)
        pos = pos.permute(0, 2, 3, 1).contiguous()
        pos = pos.view(B, H * W, C)

        feature_map = feature_map.permute(0, 2, 3, 1).contiguous()
        feature_map = feature_map.view(B, H * W, C)

        query = self.object_tokens[None, :, :].repeat(B, 1, 1)
        query_pos = self.object_pos[None, :, :].repeat(B, 1, 1)

        out_list = []
        for layer in self.layers:
            query = layer(query=query, key=feature_map, query_pos=query_pos, key_pos=pos)
            out_list.append(query)

        return out_list[::-1]


class TransformerFlowLayerContext(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.context_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)

        self.activation = nn.GELU()
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, myself, other, context, myself_pos, other_pos, context_pos):
        q = k = self.with_pos_embed(myself, myself_pos)
        tgt2 = self.self_attn(q, k, value=myself)[0]
        myself = myself + self.dropout1(tgt2)
        myself = self.norm1(myself)

        tgt2 = self.cross_attn(query=self.with_pos_embed(myself, myself_pos), key=self.with_pos_embed(other, other_pos), value=other)[0]
        myself = myself + self.dropout2(tgt2)
        myself = self.norm2(myself)

        tgt2, weights = self.context_attn(query=self.with_pos_embed(myself, myself_pos), key=self.with_pos_embed(context, context_pos), value=context)
        myself = myself + self.dropout3(tgt2)
        myself = self.norm3(myself)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(myself))))
        myself = myself + self.dropout4(tgt2)
        myself = self.norm4(myself)
        return myself, weights

    def forward(self, myself, other, context, myself_pos, other_pos, context_pos):
        return self.forward_post(myself, other, context, myself_pos, other_pos, context_pos)



class TransformerFlowLayerOneWay(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn_1 = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.self_attn_2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)

        self.activation = nn.GELU()
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, query, key, query_pos, key_pos):
        q = k = self.with_pos_embed(query, query_pos)
        tgt2 = self.self_attn_1(q, k, value=query)[0]
        query = query + self.dropout1(tgt2)
        query = self.norm1(query)

        q = k = self.with_pos_embed(key, key_pos)
        tgt2 = self.self_attn_2(q, k, value=key)[0]
        key = key + self.dropout2(tgt2)
        key = self.norm2(key)

        tgt2 = self.cross_attn(query=self.with_pos_embed(query, query_pos), key=self.with_pos_embed(key, key_pos), value=key)[0]
        query = query + self.dropout3(tgt2)
        query = self.norm3(query)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(query))))
        query = query + self.dropout4(tgt2)
        query = self.norm4(query)
        return query

    def forward(self, query, key, query_pos, key_pos):
        return self.forward_post(query, key, query_pos, key_pos)
    

class TransformerFlowLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.GELU()
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, myself, context, myself_pos, context_pos):
        q = k = self.with_pos_embed(myself, myself_pos)
        tgt2 = self.self_attn(q, k, value=myself)[0]
        myself = myself + self.dropout1(tgt2)
        myself = self.norm1(myself)

        tgt2, weights = self.cross_attn(query=self.with_pos_embed(myself, myself_pos), key=self.with_pos_embed(context, context_pos), value=context)
        myself = myself + self.dropout2(tgt2)
        myself = self.norm2(myself)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(myself))))
        myself = myself + self.dropout3(tgt2)
        myself = self.norm3(myself)
        return myself

    def forward(self, query, key, query_pos, key_pos):
        return self.forward_post(query, key, query_pos, key_pos)
    



class TransformerFlowLayerSeparated(nn.Module):

    def __init__(self, d_model, nhead, distance, pos_1d, dim_feedforward, topk=None, dropout=0.0,
                 activation="relu", normalize_before=False, gaussian_type='query'):
        super().__init__()
        self.distance = distance
        self.pos_1d = pos_1d
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        if distance == 'cos':
            if pos_1d == 'learnable_sin':
                self.cross_attn = AttentionLearnedSin(d_model, num_heads=nhead)
            else:
                self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        elif distance == 'l2':
            self.cross_attn = L2Attention(d_model, nb_heads=nhead, topk=topk, pos_1d=pos_1d, gaussian_type=gaussian_type)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.GELU()
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, query, key, value, query_pos, key_pos, shape, max_idx=None, pos=None):
        q = k = self.with_pos_embed(query, query_pos)
        tgt2 = self.self_attn(q, k, value=query)[0]
        query = query + self.dropout1(tgt2)
        query = self.norm1(query)

        if self.distance == 'l2':
            tgt2, weights = self.cross_attn(query=self.with_pos_embed(query, query_pos), 
                                        key=self.with_pos_embed(key, key_pos), 
                                        value=value,
                                        shape=shape,
                                        pos=pos,
                                        max_idx=max_idx)
        else:
            if self.pos_1d == 'learnable_sin':
                tgt2, weights = self.cross_attn(query=self.with_pos_embed(query, query_pos), 
                                                key=self.with_pos_embed(key, key_pos), 
                                                value=value,
                                                shape=shape,
                                                pos=pos)
            else:
                tgt2, weights = self.cross_attn(query=self.with_pos_embed(query, query_pos), 
                                                key=self.with_pos_embed(key, key_pos), 
                                                value=value)
            
        query = query + self.dropout2(tgt2)
        query = self.norm2(query)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(query))))
        query = query + self.dropout3(tgt2)
        query = self.norm3(query)
        return query, weights

    def forward(self, query, key, value, query_pos, key_pos, shape, max_idx=None, pos=None):
        return self.forward_post(query, key, value, query_pos, key_pos, shape, max_idx, pos)
    




class MemoryToQueryCross(nn.Module):

    def __init__(self, d_model, nhead, distance, pos_1d, topk=None, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.distance = distance
        if distance == 'cos':
            if topk is not None:
                self.cross_attn = TopKAttention(d_model, d_model, heads=nhead, topk=topk)
            else:
                self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        elif distance == 'l2':
            self.cross_attn = L2Attention(d_model, topk=topk, pos_1d=pos_1d)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.GELU()
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, query, key, value, query_pos, key_pos, shape, max_idx=None, pos=None):

        if self.distance == 'l2':
            tgt2, weights = self.cross_attn(query=self.with_pos_embed(query, query_pos), 
                                        key=self.with_pos_embed(key, key_pos), 
                                        value=value,
                                        shape=shape,
                                        pos=pos,
                                        max_idx=max_idx)
        else:
            tgt2, weights = self.cross_attn(query=self.with_pos_embed(query, query_pos), 
                                            key=self.with_pos_embed(key, key_pos), 
                                            value=value)
            
        query = query + self.dropout2(tgt2)
        query = self.norm2(query)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(query))))
        query = query + self.dropout3(tgt2)
        query = self.norm3(query)
        return query, weights

    def forward(self, query, key, value, query_pos, key_pos, shape, max_idx=None, pos=None):
        return self.forward_post(query, key, value, query_pos, key_pos, shape, max_idx, pos)
    




class TransformerFlowLayerSeparatedDumb(nn.Module):

    def __init__(self, d_model, nhead, distance, pos_1d, kernel_size=(5, 5), topk=None, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.distance = distance
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        if distance == 'cos':
            if topk is not None:
                self.cross_attn = TopKAttention(d_model, d_model, heads=nhead, topk=topk)
            else:
                self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        elif distance == 'l2':
            self.cross_attn = L2Attention(d_model, topk=topk, pos_1d=pos_1d)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.GELU()
        self.normalize_before = normalize_before

        #padding = kernel_size[0]//2
        #self.unfolder = torch.nn.Unfold(kernel_size=kernel_size, dilation=1, padding=padding, stride=1)
        #self.nb_tokens = kernel_size[0] * kernel_size[1]

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos
    
    def reshape_tokens(self, x, shape):
        T, B, C, H, W = shape
        x = x.view(B, T, H, W, C)
        
        x = x.permute(1, 0, 4, 2, 3).contiguous() # T, B, C, H, W
        window_size_list = (torch.floor((-4 / T) * torch.arange(T) + 7) // 2 * 2 + 1).int().tolist()

        #window_size_list = [val for val in range(3, 9, 2) for _ in range(5)]
        window_size_list = window_size_list[:len(x)]
        data_list = []
        split_list = []
        for t in range(len(x)):
            window_size = window_size_list[t]
            pad_value = window_size // 2
            unfolder = torch.nn.Unfold(kernel_size=window_size, dilation=1, padding=pad_value, stride=1)
            unfolded = unfolder(x[t])
            unfolded = unfolded.view(B, C, window_size**2, H * W)
            unfolded = unfolded.permute(0, 3, 2, 1).contiguous()
            unfolded = unfolded.view(B * H * W, window_size**2, C)
            data_list.append(unfolded)
            split_list.append(unfolded.shape[1])

        x = torch.cat(data_list, dim=1)
        return x, split_list
    
    #def reshape_tokens(self, x, shape):
    #    T, B, C, H, W = shape
    #    x = x.view(B, T, H, W, C)
    #    x = x.permute(0, 1, 4, 2, 3).contiguous() # B, T, C, H, W
    #    x = x.view(B * T, C, H, W)
    #    x = self.unfolder(x)
    #    x = x.view(B, T, C, self.nb_tokens, H * W)
    #    x = x.permute(0, 4, 1, 3, 2).contiguous()
    #    x = x.view(B * H * W, T * self.nb_tokens, C)
    #    return x

    def forward_post(self, query, key, value, query_pos, key_pos, shape, max_idx=None, pos=None):
        T, B, C, H, W = shape
        
        q = k = self.with_pos_embed(query, query_pos)
        tgt2 = self.self_attn(q, k, value=query)[0]
        query = query + self.dropout1(tgt2)
        query = self.norm1(query)

        query = self.with_pos_embed(query, query_pos)
        key = self.with_pos_embed(key, key_pos)

        key, split_list = self.reshape_tokens(key, shape=shape)
        value, _ = self.reshape_tokens(value, shape=shape)
        if pos is not None:
            pos, _ = self.reshape_tokens(pos, shape=shape)

        query = query.view(B * H * W, C)
        query = query[:, None, :]

        tgt2, weights = self.cross_attn(query=query, 
                                        key=key, 
                                        value=value,
                                        shape=shape,
                                        pos=pos,
                                        max_idx=max_idx)
        
        weights = torch.split(weights, split_list, dim=1)[1:]
            
        query = query + self.dropout2(tgt2)

        query = query.view(B, H, W, 1, C).squeeze(3)
        query = query.view(B, H * W, C)

        query = self.norm2(query)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(query))))
        query = query + self.dropout3(tgt2)
        query = self.norm3(query)
        return query, weights

    def forward(self, query, key, value, query_pos, key_pos, shape, max_idx=None, pos=None):
        return self.forward_post(query, key, value, query_pos, key_pos, shape, max_idx, pos)




class PosAttention(nn.Module):

    def __init__(self, d_model, nhead, distance, topk=None, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.distance = distance
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = L2AttentionPos(d_model, topk)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.GELU()
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, query, key, value, query_pos, key_pos, pos):
        q = k = self.with_pos_embed(query, query_pos)
        tgt2 = self.self_attn(q, k, value=query)[0]
        query = query + self.dropout1(tgt2)
        query = self.norm1(query)

        tgt2, weights = self.cross_attn(query=self.with_pos_embed(query, query_pos), 
                                        key=self.with_pos_embed(key, key_pos), 
                                        value=value,
                                        pos=pos)
            
        query = query + self.dropout2(tgt2)
        query = self.norm2(query)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(query))))
        query = query + self.dropout3(tgt2)
        query = self.norm3(query)
        return query, weights

    def forward(self, query, key, value, query_pos, key_pos, pos):
        return self.forward_post(query, key, value, query_pos, key_pos, pos)
    


class TransformerFlowLayerSelect(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = AttentionSelect(dim=d_model, heads=nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.GELU()
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, myself, context, myself_pos, context_pos):
        q = k = self.with_pos_embed(myself, myself_pos)
        tgt2 = self.self_attn(q, k, value=myself)[0]
        myself = myself + self.dropout1(tgt2)
        myself = self.norm1(myself)

        tgt2 = self.cross_attn(query=myself, key=context, value=context)
        myself = myself + self.dropout2(tgt2)
        myself = self.norm2(myself)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(myself))))
        myself = myself + self.dropout3(tgt2)
        myself = self.norm3(myself)
        return myself

    def forward(self, query, key, query_pos, key_pos):
        return self.forward_post(query, key, query_pos, key_pos)
    



class TransformerFlowLayerTopK(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = TopKAttention(inp=d_model, oup=d_model, heads=nhead)
        self.cross_attn = TopKAttention(inp=d_model, oup=d_model, heads=nhead)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.GELU()
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, myself, context, myself_pos, context_pos):
        q = k = self.with_pos_embed(myself, myself_pos)
        tgt2 = self.self_attn(q, k, value=myself)[0]
        myself = myself + self.dropout1(tgt2)
        myself = self.norm1(myself)

        tgt2 = self.cross_attn(query=self.with_pos_embed(myself, myself_pos), key=self.with_pos_embed(context, context_pos), value=context)
        myself = myself + self.dropout2(tgt2)
        myself = self.norm2(myself)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(myself))))
        myself = myself + self.dropout3(tgt2)
        myself = self.norm3(myself)
        return myself

    def forward(self, query, key, query_pos, key_pos):
        return self.forward_post(query, key, query_pos, key_pos)
    



class LSSTransformer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn_short = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn_long = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.GELU()
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, myself, context_short, context_long, myself_pos, context_pos):
        q = k = self.with_pos_embed(myself, myself_pos)
        tgt = self.self_attn(q, k, value=myself)[0]
        tgt = myself + self.dropout1(tgt)
        myself = self.norm1(tgt)

        tgt2, weights = self.cross_attn_short(query=self.with_pos_embed(myself, myself_pos), key=self.with_pos_embed(context_short, context_pos), value=context_short)
        tgt3, weights = self.cross_attn_long(query=self.with_pos_embed(myself, myself_pos), key=self.with_pos_embed(context_long, context_pos), value=context_long)
        myself = tgt + tgt2 + tgt3
        myself = self.norm2(myself)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(myself))))
        myself = myself + self.dropout3(tgt2)
        myself = self.norm3(myself)
        return myself, weights

    def forward(self, query, key, query_pos, key_pos):
        return self.forward_post(query, key, query_pos, key_pos)
    


class TransformerCell(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.cross_attn_1 = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn_2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn_3 = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.activation = nn.GELU()
        self.normalize_before = normalize_before

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.dropout3 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, myself, context, myself_pos, context_pos):
        #q = k = self.with_pos_embed(myself, myself_pos)
        #tgt2 = self.self_attn(q, k, value=myself)[0]
        #myself = myself + self.dropout1(tgt2)
        #myself = self.norm1(myself)

        zt, weights = self.cross_attn_1(query=self.with_pos_embed(myself, myself_pos), key=self.with_pos_embed(context, context_pos), value=context)
        zt = self.sigmoid(zt)

        rt, weights = self.cross_attn_2(query=self.with_pos_embed(context, context_pos), key=self.with_pos_embed(myself, myself_pos), value=myself)
        rt = self.sigmoid(rt)

        ht, weights = self.cross_attn_3(query=self.with_pos_embed(myself, myself_pos), key=self.with_pos_embed(context * rt, context_pos), value=context)
        ht = self.tanh(ht)

        h_next = (1 - zt) * context + zt * ht

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(h_next))))
        h_next = h_next + self.dropout3(tgt2)
        h_next = self.norm3(h_next)
        return h_next

    def forward(self, query, key, query_pos, key_pos):
        return self.forward_post(query, key, query_pos, key_pos)
    


class TransformerFlowEncoder(nn.Module):
    def __init__(self, dim, nhead, num_layers, nb_tokens):
        super().__init__()

        self.nb_tokens = nb_tokens

        spatial_layer = TransformerFlowLayer(d_model=dim, nhead=nhead)
        self.spatial_layers = _get_clones(spatial_layer, num_layers)

        temporal_layer = TransformerEncoderLayer(d_model=dim, nhead=nhead)
        self.temporal_layers = _get_clones(temporal_layer, num_layers)

        #temporal_layer = TransformerEncoderLayer(d_model=dim, nhead=nhead)
        #self.temporal_layers = _get_clones(temporal_layer, num_layers)

        self.pos_obj_2d = PositionEmbeddingSine2d(num_pos_feats=dim // 2, normalize=True)
        #self.pos_obj_1d = PositionEmbeddingSine1d(num_pos_feats=dim, normalize=True)

        self.token_pos = nn.Parameter(torch.randn(self.nb_tokens, dim))
        self.temporal_tokens = nn.Parameter(torch.randn(self.nb_tokens, dim))
    
    
    def forward(self, spatial_features_1, spatial_features_2, pos_1d_1, pos_1d_2):
        '''spatial_features_1: T-1, B, C, H, W
            spatial_features_2: T-1, B, C, H, W
            pos_1d_1: B, C, T-1
            pos_1d_2: B, C, T-1'''
        
        shape = spatial_features_1.shape
        T, B, C, H, W = shape

        pos_1d_1 = pos_1d_1.permute(0, 2, 1)[:, :, None, :].repeat(1, 1, self.nb_tokens, 1).view(B, T * self.nb_tokens, C)
        pos_1d_2 = pos_1d_2.permute(0, 2, 1)[:, :, None, :].repeat(1, 1, self.nb_tokens, 1).view(B, T * self.nb_tokens, C)

        spatial_features_1 = spatial_features_1.permute(0, 1, 3, 4, 2).contiguous() # T, B, H, W, C
        spatial_features_1 = spatial_features_1.view(T, B, H * W, C).view(T * B, H * W, C)

        spatial_features_2 = spatial_features_2.permute(0, 1, 3, 4, 2).contiguous() # T, B, H, W, C
        spatial_features_2 = spatial_features_2.view(T, B, H * W, C).view(T * B, H * W, C)

        pos_2d = self.pos_obj_2d(shape_util=(B, H, W), device=spatial_features_1.device)
        pos_2d = pos_2d.permute(0, 2, 3, 1).contiguous()
        pos_2d = pos_2d.view(B, H * W, C).repeat(T, 1, 1)

        temporal_features_1 = self.temporal_tokens[None, None].repeat(B, T, 1, 1).view(B, T * self.nb_tokens, C)
        temporal_features_2 = self.temporal_tokens[None, None].repeat(B, T, 1, 1).view(B, T * self.nb_tokens, C)

        token_pos = self.token_pos[None].repeat(T * B, 1, 1) # T*B, M, C
        spatial_pos = torch.cat([pos_2d, token_pos], dim=1)

        token_pos = self.token_pos[None, None].repeat(B, T, 1, 1).view(B, T * self.nb_tokens, C)

        temporal_pos_1 = pos_1d_1 + token_pos
        temporal_pos_2 = pos_1d_2 + token_pos
        temporal_pos = torch.cat([temporal_pos_1, temporal_pos_2], dim=1)

        #pos_1d = self.pos_obj_1d(shape_util=(B, T), device=labeled_spatial_features.device) # B, C, T
        #pos_1d = pos_1d.permute(0, 2, 1).contiguous() # B, T, C
        #pos_1d = pos_1d.repeat(1, 2*self.nb_tokens, 1) # B, 2*T*nb_tokens, C
        #pos_1d = self.pos_1d[None, :, :].repeat(B, 1, 1)
        #token_pos = self.token_pos[None, :, :].repeat(T * B, 1, 1)

        #temporal_tokens = self.temporal_tokens[:, None, :, :, :].repeat(1, B, 1, 1, 1) # (T-1), B, 2, nb_tokens, C
        #temporal_tokens = temporal_tokens.view(T * B, 2, self.nb_tokens, C)
        #temporal_features_1 = temporal_tokens[:, 0]
        #temporal_features_2 = temporal_tokens[:, 1]

        #pos = torch.cat([pos_2d, token_pos], dim=1)


        for spatial_layer, temporal_layer in zip(self.spatial_layers, self.temporal_layers):
            temporal_features = torch.cat([temporal_features_1, temporal_features_2], dim=1)
            temporal_features = temporal_layer(temporal_features, pos=temporal_pos)[0]

            temporal_features_1 = temporal_features[:, :T*self.nb_tokens]
            temporal_features_2 = temporal_features[:, T*self.nb_tokens:]

            temporal_features_1 = temporal_features_1.permute(1, 0, 2).contiguous() # T*M, B, C
            temporal_features_2 = temporal_features_2.permute(1, 0, 2).contiguous() # T*M, B, C

            temporal_features_1 = temporal_features_1.view(T, self.nb_tokens, B, C)
            temporal_features_2 = temporal_features_2.view(T, self.nb_tokens, B, C)

            temporal_features_1 = temporal_features_1.permute(0, 2, 1, 3).contiguous() # T, B, M, C
            temporal_features_2 = temporal_features_2.permute(0, 2, 1, 3).contiguous() # T, B, M, C

            temporal_features_1 = temporal_features_1.view(T*B, self.nb_tokens, C)
            temporal_features_2 = temporal_features_2.view(T*B, self.nb_tokens, C)

            feature_1 = torch.cat([spatial_features_1, temporal_features_1], dim=1)
            feature_2 = torch.cat([spatial_features_2, temporal_features_2], dim=1)

            feature_1 = spatial_layer(feature_1, feature_2, spatial_pos)
            feature_2 = spatial_layer(feature_2, feature_1, spatial_pos)

            spatial_features_1 = feature_1[:, :H*W] # T*B, H*W, C
            temporal_features_1 = feature_1[:, H*W:] # T*B, M, C
            spatial_features_2 = feature_2[:, :H*W] # T*B, H*W, C
            temporal_features_2 = feature_2[:, H*W:] # T*B, M, C

            temporal_features_1 = temporal_features_1.view(T, B, self.nb_tokens, C)
            temporal_features_2 = temporal_features_2.view(T, B, self.nb_tokens, C)

            temporal_features_1 = temporal_features_1.permute(1, 0, 2, 3).contiguous() # B, T, M, C
            temporal_features_2 = temporal_features_2.permute(1, 0, 2, 3).contiguous() # B, T, M, C

            temporal_features_1 = temporal_features_1.view(B, T * self.nb_tokens, C)
            temporal_features_2 = temporal_features_2.view(B, T * self.nb_tokens, C)
        

        spatial_features_1 = spatial_features_1.view(T, B, H * W, C)
        spatial_features_2 = spatial_features_2.view(T, B, H * W, C)

        #spatial_features_seg = spatial_features_seg.view((T + 1) * B, H * W, C)
        #pos_2d = torch.cat([pos_2d, pos_2d[-1][None].repeat(B, 1, 1)], dim=0)
        #for seg_layer in self.seg_layers:
        #    spatial_features_seg = seg_layer(spatial_features_seg, pos=pos_2d)[0]
        #spatial_features_seg = spatial_features_seg.view((T + 1), B, H * W, C)

        return spatial_features_1, spatial_features_2
    


class TransformerFlowEncoderConv(nn.Module):
    def __init__(self, dim, nhead, num_layers, nb_tokens):
        super().__init__()

        self.nb_tokens = nb_tokens

        spatial_layer = TransformerFlowLayer(d_model=dim, nhead=nhead)
        self.spatial_layers = _get_clones(spatial_layer, num_layers)

        temporal_layer = TransformerEncoderLayer(d_model=dim, nhead=nhead)
        self.temporal_layers = _get_clones(temporal_layer, num_layers)

        #seg_layer = TransformerEncoderLayer(d_model=dim, nhead=nhead)
        #self.seg_layers = _get_clones(seg_layer, num_layers)

        #temporal_layer = TransformerEncoderLayer(d_model=dim, nhead=nhead)
        #self.temporal_layers = _get_clones(temporal_layer, num_layers)

        self.pos_obj_2d = PositionEmbeddingSine2d(num_pos_feats=dim // 2, normalize=True)
        #self.pos_obj_1d = PositionEmbeddingSine1d(num_pos_feats=dim, normalize=True)

        self.token_pos = nn.Parameter(torch.randn(self.nb_tokens, dim))
        self.temporal_tokens = nn.Parameter(torch.randn(self.nb_tokens, dim))
    
    
    def forward(self, spatial_features_1, spatial_features_2, pos_1d_1, pos_1d_2):
        '''spatial_features_1: T-1, B, C, H, W
            spatial_features_2: T-1, B, C, H, W
            pos_1d_1: B, C, T-1
            pos_1d_2: B, C, T-1'''
        
        shape = spatial_features_1.shape
        T, B, C, H, W = shape

        pos_1d_1 = pos_1d_1.permute(0, 2, 1)[:, :, None, :].repeat(1, 1, self.nb_tokens, 1).view(B, T * self.nb_tokens, C)
        pos_1d_2 = pos_1d_2.permute(0, 2, 1)[:, :, None, :].repeat(1, 1, self.nb_tokens, 1).view(B, T * self.nb_tokens, C)

        spatial_features_1 = spatial_features_1.permute(0, 1, 3, 4, 2).contiguous() # T, B, H, W, C
        spatial_features_1 = spatial_features_1.view(T, B, H * W, C).view(T * B, H * W, C)

        spatial_features_2 = spatial_features_2.permute(0, 1, 3, 4, 2).contiguous() # T, B, H, W, C
        spatial_features_2 = spatial_features_2.view(T, B, H * W, C).view(T * B, H * W, C)

        pos_2d = self.pos_obj_2d(shape_util=(B, H, W), device=spatial_features_1.device)
        pos_2d = pos_2d.permute(0, 2, 3, 1).contiguous()
        pos_2d = pos_2d.view(B, H * W, C).repeat(T, 1, 1)

        temporal_features_1 = self.temporal_tokens[None, None].repeat(B, T, 1, 1).view(B, T * self.nb_tokens, C)
        temporal_features_2 = self.temporal_tokens[None, None].repeat(B, T, 1, 1).view(B, T * self.nb_tokens, C)

        token_pos = self.token_pos[None].repeat(T * B, 1, 1) # T*B, M, C
        spatial_pos = torch.cat([pos_2d, token_pos], dim=1)

        token_pos = self.token_pos[None, None].repeat(B, T, 1, 1).view(B, T * self.nb_tokens, C)

        temporal_pos_1 = pos_1d_1 + token_pos
        temporal_pos_2 = pos_1d_2 + token_pos
        temporal_pos = torch.cat([temporal_pos_1, temporal_pos_2], dim=1)

        #pos_1d = self.pos_obj_1d(shape_util=(B, T), device=labeled_spatial_features.device) # B, C, T
        #pos_1d = pos_1d.permute(0, 2, 1).contiguous() # B, T, C
        #pos_1d = pos_1d.repeat(1, 2*self.nb_tokens, 1) # B, 2*T*nb_tokens, C
        #pos_1d = self.pos_1d[None, :, :].repeat(B, 1, 1)
        #token_pos = self.token_pos[None, :, :].repeat(T * B, 1, 1)

        #temporal_tokens = self.temporal_tokens[:, None, :, :, :].repeat(1, B, 1, 1, 1) # (T-1), B, 2, nb_tokens, C
        #temporal_tokens = temporal_tokens.view(T * B, 2, self.nb_tokens, C)
        #temporal_features_1 = temporal_tokens[:, 0]
        #temporal_features_2 = temporal_tokens[:, 1]

        #pos = torch.cat([pos_2d, token_pos], dim=1)


        for spatial_layer, temporal_layer in zip(self.spatial_layers, self.temporal_layers):
            temporal_features = torch.cat([temporal_features_1, temporal_features_2], dim=1)
            temporal_features = temporal_layer(temporal_features, pos=temporal_pos)[0]

            temporal_features_1 = temporal_features[:, :T*self.nb_tokens]
            temporal_features_2 = temporal_features[:, T*self.nb_tokens:]

            temporal_features_1 = temporal_features_1.permute(1, 0, 2).contiguous() # T*M, B, C
            temporal_features_2 = temporal_features_2.permute(1, 0, 2).contiguous() # T*M, B, C

            temporal_features_1 = temporal_features_1.view(T, self.nb_tokens, B, C)
            temporal_features_2 = temporal_features_2.view(T, self.nb_tokens, B, C)

            temporal_features_1 = temporal_features_1.permute(0, 2, 1, 3).contiguous() # T, B, M, C
            temporal_features_2 = temporal_features_2.permute(0, 2, 1, 3).contiguous() # T, B, M, C

            temporal_features_1 = temporal_features_1.view(T*B, self.nb_tokens, C)
            temporal_features_2 = temporal_features_2.view(T*B, self.nb_tokens, C)

            feature_1 = torch.cat([spatial_features_1, temporal_features_1], dim=1)
            feature_2 = torch.cat([spatial_features_2, temporal_features_2], dim=1)

            feature_1 = spatial_layer(feature_1, feature_2, sa_query_pos=spatial_pos, ca_query_pos=spatial_pos, key_pos=spatial_pos)
            feature_2 = spatial_layer(feature_2, feature_1, sa_query_pos=spatial_pos, ca_query_pos=spatial_pos, key_pos=spatial_pos)

            spatial_features_1 = feature_1[:, :H*W] # T*B, H*W, C
            temporal_features_1 = feature_1[:, H*W:] # T*B, M, C
            spatial_features_2 = feature_2[:, :H*W] # T*B, H*W, C
            temporal_features_2 = feature_2[:, H*W:] # T*B, M, C

            #spatial_features_1 = spatial_features_1.view(T, B, H * W, C)
            #spatial_features_2 = spatial_features_2.view(T, B, H * W, C)

            #padding = torch.zeros_like(spatial_features_1[0]).unsqueeze(0)
            #spatial_features_1_seg = torch.cat([spatial_features_1, padding], dim=0)
            #spatial_features_2_seg = torch.cat([padding, spatial_features_2], dim=0)
            #spatial_features_seg = spatial_features_1_seg + spatial_features_2_seg

            #spatial_features_1 = spatial_features_seg[:-1]
            #spatial_features_2 = spatial_features_seg[1:]
#
            #spatial_features_1 = spatial_features_1.view(T * B, H * W, C)
            #spatial_features_2 = spatial_features_2.view(T * B, H * W, C)

            temporal_features_1 = temporal_features_1.view(T, B, self.nb_tokens, C)
            temporal_features_2 = temporal_features_2.view(T, B, self.nb_tokens, C)

            temporal_features_1 = temporal_features_1.permute(1, 0, 2, 3).contiguous() # B, T, M, C
            temporal_features_2 = temporal_features_2.permute(1, 0, 2, 3).contiguous() # B, T, M, C

            temporal_features_1 = temporal_features_1.view(B, T * self.nb_tokens, C)
            temporal_features_2 = temporal_features_2.view(B, T * self.nb_tokens, C)
        

        spatial_features_1 = spatial_features_1.view(T, B, H * W, C)
        spatial_features_2 = spatial_features_2.view(T, B, H * W, C)

        padding = torch.zeros_like(spatial_features_1[0]).unsqueeze(0)
        spatial_features_1_seg = torch.cat([spatial_features_1, padding], dim=0)
        spatial_features_2_seg = torch.cat([padding, spatial_features_2], dim=0)
        spatial_features_seg = spatial_features_1_seg + spatial_features_2_seg

        #spatial_features_seg = spatial_features_seg.view((T + 1) * B, H * W, C)
        #pos_2d = torch.cat([pos_2d, pos_2d[-1][None].repeat(B, 1, 1)], dim=0)
        #for seg_layer in self.seg_layers:
        #    spatial_features_seg = seg_layer(spatial_features_seg, pos=pos_2d)[0]
        #spatial_features_seg = spatial_features_seg.view((T + 1), B, H * W, C)

        return spatial_features_1, spatial_features_2, spatial_features_seg


class TransformerFlowEncoderIterativeStep(nn.Module):
    def __init__(self, dim, nhead, num_layers, nb_tokens, temporal_kernel_size):
        super().__init__()

        self.nb_tokens = nb_tokens
        self.temporal_kernel_size = temporal_kernel_size

        spatial_layer = TransformerFlowLayer(d_model=dim, nhead=nhead)
        self.spatial_layers = _get_clones(spatial_layer, num_layers)

        #temporal_layer = TransformerEncoderLayer(d_model=dim, nhead=nhead)
        #self.temporal_layers = _get_clones(temporal_layer, num_layers)

        #seg_layer = TransformerEncoderLayer(d_model=dim, nhead=nhead)
        #self.seg_layers = _get_clones(seg_layer, num_layers)

        #temporal_layer = TransformerEncoderLayer(d_model=dim, nhead=nhead)
        #self.temporal_layers = _get_clones(temporal_layer, num_layers)

        self.pos_obj_2d = PositionEmbeddingSine2d(num_pos_feats=dim // 2, normalize=True)
        #self.pos_obj_1d = PositionEmbeddingSine1d(num_pos_feats=dim, normalize=True)

        #self.token_pos = nn.Parameter(torch.randn(self.nb_tokens, dim))
        #self.temporal_tokens = nn.Parameter(torch.randn(self.nb_tokens, dim))
    
    
    def forward(self, spatial_features, pos_1d):
        '''spatial_features_1: T, B, C, H, W
            pos_1d: B, C, T'''
        
        shape = spatial_features.shape
        T, B, C, H, W = shape

        pos_1d = pos_1d.permute(2, 0, 1)[:, :, None, :].repeat(1, 1, H * W, 1)

        spatial_features = spatial_features.permute(0, 1, 3, 4, 2).contiguous() # T, B, H, W, C
        spatial_features = spatial_features.view(T, B, H * W, C)

        pos_2d = self.pos_obj_2d(shape_util=(B, H, W), device=spatial_features.device)
        pos_2d = pos_2d.permute(0, 2, 3, 1).contiguous()
        pos_2d = pos_2d.view(B, H * W, C)

        #pos_1d = self.pos_obj_1d(shape_util=(B, T), device=labeled_spatial_features.device) # B, C, T
        #pos_1d = pos_1d.permute(0, 2, 1).contiguous() # B, T, C
        #pos_1d = pos_1d.repeat(1, 2*self.nb_tokens, 1) # B, 2*T*nb_tokens, C
        #pos_1d = self.pos_1d[None, :, :].repeat(B, 1, 1)
        #token_pos = self.token_pos[None, :, :].repeat(T * B, 1, 1)

        #temporal_tokens = self.temporal_tokens[:, None, :, :, :].repeat(1, B, 1, 1, 1) # (T-1), B, 2, nb_tokens, C
        #temporal_tokens = temporal_tokens.view(T * B, 2, self.nb_tokens, C)
        #temporal_features_1 = temporal_tokens[:, 0]
        #temporal_features_2 = temporal_tokens[:, 1]

        #pos = torch.cat([pos_2d, token_pos], dim=1)

        for spatial_layer in self.spatial_layers:
            past_features = torch.zeros_like(spatial_features) # T, B, H*W, C
            for i in range(len(spatial_features)):

                #matplotlib.use('QtAgg')
                #fig, ax = plt.subplots(2, len(spatial_features))
                #past_features = past_features.permute(0, 1, 3, 2).contiguous().view(T - 1, B, C, H, W)
                #spatial_features = spatial_features.permute(0, 1, 3, 2).contiguous().view(T, B, C, H, W)
                #for j in range(len(spatial_features)):
                #    ax[0, j].imshow(spatial_features[j, 0, 0].detach().cpu())
                #    if j < len(spatial_features) - 1:
                #        ax[1, j].imshow(past_features[j, 0, 0].detach().cpu())
                #plt.show()
                #past_features = past_features.permute(0, 1, 3, 4, 2).contiguous().view(T-1, B, H * W, C)
                #spatial_features = spatial_features.permute(0, 1, 3, 4, 2).contiguous().view(T, B, H * W, C)

                #fig, ax = plt.subplots(2, 1)
                #ax[0].imshow(past_features[i, 0, 0].detach().cpu())
                #ax[1].imshow(spatial_features[i, 0, 0].detach().cpu())
                #plt.show()

                current_spatial_feature = spatial_features[i]
                current_pos_1d = pos_1d[i]
                if i == 0:
                    previous_spatial_feature = torch.zeros_like(current_spatial_feature).unsqueeze(0).repeat(self.temporal_kernel_size, 1, 1, 1)
                    previous_pos_1d = torch.zeros_like(current_pos_1d).unsqueeze(0).repeat(self.temporal_kernel_size, 1, 1, 1)
                else:
                    if i > 2:
                        step = int(ceil((i - 2) / (self.temporal_kernel_size - 2)))
                        middle_indices = torch.arange(step, i-1, step)
                        previous_spatial_feature = torch.cat([past_features[0][None], past_features[middle_indices], past_features[i - 1][None]])
                        previous_pos_1d = torch.cat([pos_1d[0][None], pos_1d[middle_indices], pos_1d[i - 1][None]])
                    else:
                        previous_spatial_feature = torch.cat([past_features[0][None], past_features[i - 1][None]])
                        previous_pos_1d = torch.cat([pos_1d[0][None], pos_1d[i - 1][None]])
                    previous_spatial_feature = torch.nn.functional.pad(previous_spatial_feature, pad=(0, 0, 0, 0, 0, 0, max(0, self.temporal_kernel_size - len(previous_spatial_feature)), 0))
                    previous_pos_1d = torch.nn.functional.pad(previous_pos_1d, pad=(0, 0, 0, 0, 0, 0, max(0, self.temporal_kernel_size - len(previous_pos_1d)), 0))

                previous_pos_1d = previous_pos_1d.permute(0, 2, 1, 3).contiguous()
                previous_pos_1d = previous_pos_1d.view(self.temporal_kernel_size * H * W, B, C)
                previous_pos_1d = previous_pos_1d.permute(1, 0, 2).contiguous()

                ca_query_pos = pos_2d + current_pos_1d

                key_pos_2d = pos_2d[:, None, :, :].repeat(1, self.temporal_kernel_size, 1, 1)
                key_pos_2d = key_pos_2d.permute(1, 2, 0, 3).contiguous()
                key_pos_2d = key_pos_2d.view(self.temporal_kernel_size * H * W, B, C)
                key_pos_2d = key_pos_2d.permute(1, 0, 2).contiguous()
                key_pos = key_pos_2d + previous_pos_1d

                previous_spatial_feature = previous_spatial_feature.permute(0, 2, 1, 3).contiguous()
                previous_spatial_feature = previous_spatial_feature.view(self.temporal_kernel_size * H * W, B, C)
                previous_spatial_feature = previous_spatial_feature.permute(1, 0, 2).contiguous()

                current_spatial_feature = spatial_layer(current_spatial_feature, previous_spatial_feature, sa_query_pos=pos_2d, ca_query_pos=ca_query_pos, key_pos=key_pos)
                past_features[i] = current_spatial_feature
            spatial_features = past_features
        return spatial_features
    

class TransformerFlowEncoderStep(nn.Module):
    def __init__(self, dim, nhead, num_layers, nb_tokens, temporal_kernel_size):
        super().__init__()

        self.nb_tokens = nb_tokens
        self.temporal_kernel_size = temporal_kernel_size

        spatial_layer = TransformerFlowLayer(d_model=dim, nhead=nhead)
        self.spatial_layers = _get_clones(spatial_layer, num_layers)

        #temporal_layer = TransformerEncoderLayer(d_model=dim, nhead=nhead)
        #self.temporal_layers = _get_clones(temporal_layer, num_layers)

        #seg_layer = TransformerEncoderLayer(d_model=dim, nhead=nhead)
        #self.seg_layers = _get_clones(seg_layer, num_layers)

        #temporal_layer = TransformerEncoderLayer(d_model=dim, nhead=nhead)
        #self.temporal_layers = _get_clones(temporal_layer, num_layers)

        self.pos_obj_2d = PositionEmbeddingSine2d(num_pos_feats=dim // 2, normalize=True)
        #self.pos_obj_1d = PositionEmbeddingSine1d(num_pos_feats=dim, normalize=True)

        #self.token_pos = nn.Parameter(torch.randn(self.nb_tokens, dim))
        #self.temporal_tokens = nn.Parameter(torch.randn(self.nb_tokens, dim))
    
    
    def forward(self, spatial_features, pos_1d):
        '''spatial_features_1: T, B, C, H, W
            pos_1d: B, C, T'''
        
        shape = spatial_features.shape
        T, B, C, H, W = shape

        pos_1d = pos_1d.permute(2, 0, 1)[:, :, None, :].repeat(1, 1, H * W, 1)

        spatial_features = spatial_features.permute(0, 1, 3, 4, 2).contiguous() # T, B, H, W, C
        spatial_features = spatial_features.view(T, B, H * W, C)

        pos_2d = self.pos_obj_2d(shape_util=(B, H, W), device=spatial_features.device)
        pos_2d = pos_2d.permute(0, 2, 3, 1).contiguous()
        pos_2d = pos_2d.view(B, H * W, C)

        #pos_1d = self.pos_obj_1d(shape_util=(B, T), device=labeled_spatial_features.device) # B, C, T
        #pos_1d = pos_1d.permute(0, 2, 1).contiguous() # B, T, C
        #pos_1d = pos_1d.repeat(1, 2*self.nb_tokens, 1) # B, 2*T*nb_tokens, C
        #pos_1d = self.pos_1d[None, :, :].repeat(B, 1, 1)
        #token_pos = self.token_pos[None, :, :].repeat(T * B, 1, 1)

        #temporal_tokens = self.temporal_tokens[:, None, :, :, :].repeat(1, B, 1, 1, 1) # (T-1), B, 2, nb_tokens, C
        #temporal_tokens = temporal_tokens.view(T * B, 2, self.nb_tokens, C)
        #temporal_features_1 = temporal_tokens[:, 0]
        #temporal_features_2 = temporal_tokens[:, 1]

        #pos = torch.cat([pos_2d, token_pos], dim=1)

        for spatial_layer in self.spatial_layers:
            out_list = []
            for i in range(len(spatial_features)):

                #matplotlib.use('QtAgg')
                #fig, ax = plt.subplots(2, len(spatial_features))
                #past_features = past_features.permute(0, 1, 3, 2).contiguous().view(T - 1, B, C, H, W)
                #spatial_features = spatial_features.permute(0, 1, 3, 2).contiguous().view(T, B, C, H, W)
                #for j in range(len(spatial_features)):
                #    ax[0, j].imshow(spatial_features[j, 0, 0].detach().cpu())
                #    if j < len(spatial_features) - 1:
                #        ax[1, j].imshow(past_features[j, 0, 0].detach().cpu())
                #plt.show()
                #past_features = past_features.permute(0, 1, 3, 4, 2).contiguous().view(T-1, B, H * W, C)
                #spatial_features = spatial_features.permute(0, 1, 3, 4, 2).contiguous().view(T, B, H * W, C)

                #fig, ax = plt.subplots(2, 1)
                #ax[0].imshow(past_features[i, 0, 0].detach().cpu())
                #ax[1].imshow(spatial_features[i, 0, 0].detach().cpu())
                #plt.show()

                current_spatial_feature = spatial_features[i]
                current_pos_1d = pos_1d[i]
                if i == 0:
                    previous_spatial_feature = torch.zeros_like(current_spatial_feature).unsqueeze(0).repeat(self.temporal_kernel_size, 1, 1, 1)
                    previous_pos_1d = torch.zeros_like(current_pos_1d).unsqueeze(0).repeat(self.temporal_kernel_size, 1, 1, 1)
                else:
                    if i > 2:
                        step = int(ceil((i - 2) / (self.temporal_kernel_size - 2)))
                        middle_indices = torch.arange(step, i-1, step)
                        previous_spatial_feature = torch.cat([spatial_features[0][None], spatial_features[middle_indices], spatial_features[i - 1][None]])
                        previous_pos_1d = torch.cat([pos_1d[0][None], pos_1d[middle_indices], pos_1d[i - 1][None]])
                    else:
                        previous_spatial_feature = torch.cat([spatial_features[0][None], spatial_features[i - 1][None]])
                        previous_pos_1d = torch.cat([pos_1d[0][None], pos_1d[i - 1][None]])
                    previous_spatial_feature = torch.nn.functional.pad(previous_spatial_feature, pad=(0, 0, 0, 0, 0, 0, max(0, self.temporal_kernel_size - len(previous_spatial_feature)), 0))
                    previous_pos_1d = torch.nn.functional.pad(previous_pos_1d, pad=(0, 0, 0, 0, 0, 0, max(0, self.temporal_kernel_size - len(previous_pos_1d)), 0))

                previous_pos_1d = previous_pos_1d.permute(0, 2, 1, 3).contiguous()
                previous_pos_1d = previous_pos_1d.view(self.temporal_kernel_size * H * W, B, C)
                previous_pos_1d = previous_pos_1d.permute(1, 0, 2).contiguous()

                ca_query_pos = pos_2d + current_pos_1d

                key_pos_2d = pos_2d[:, None, :, :].repeat(1, self.temporal_kernel_size, 1, 1)
                key_pos_2d = key_pos_2d.permute(1, 2, 0, 3).contiguous()
                key_pos_2d = key_pos_2d.view(self.temporal_kernel_size * H * W, B, C)
                key_pos_2d = key_pos_2d.permute(1, 0, 2).contiguous()
                key_pos = key_pos_2d + previous_pos_1d

                previous_spatial_feature = previous_spatial_feature.permute(0, 2, 1, 3).contiguous()
                previous_spatial_feature = previous_spatial_feature.view(self.temporal_kernel_size * H * W, B, C)
                previous_spatial_feature = previous_spatial_feature.permute(1, 0, 2).contiguous()

                current_spatial_feature = spatial_layer(current_spatial_feature, previous_spatial_feature, sa_query_pos=pos_2d, ca_query_pos=ca_query_pos, key_pos=key_pos)
                out_list.append(current_spatial_feature)
            spatial_features = torch.stack(out_list, dim=0)
        return spatial_features


class TransformerFlowEncoderStepRegular(nn.Module):
    def __init__(self, dim, nhead, num_layers, nb_tokens, temporal_kernel_size):
        super().__init__()

        self.nb_tokens = nb_tokens
        self.temporal_kernel_size = temporal_kernel_size

        spatial_layer = TransformerFlowLayer(d_model=dim, nhead=nhead)
        self.spatial_layers = _get_clones(spatial_layer, num_layers)

        #temporal_layer = TransformerEncoderLayer(d_model=dim, nhead=nhead)
        #self.temporal_layers = _get_clones(temporal_layer, num_layers)

        #seg_layer = TransformerEncoderLayer(d_model=dim, nhead=nhead)
        #self.seg_layers = _get_clones(seg_layer, num_layers)

        #temporal_layer = TransformerEncoderLayer(d_model=dim, nhead=nhead)
        #self.temporal_layers = _get_clones(temporal_layer, num_layers)

        self.pos_obj_2d = PositionEmbeddingSine2d(num_pos_feats=dim // 2, normalize=True)
        #self.pos_obj_1d = PositionEmbeddingSine1d(num_pos_feats=dim, normalize=True)

        #self.token_pos = nn.Parameter(torch.randn(self.nb_tokens, dim))
        #self.temporal_tokens = nn.Parameter(torch.randn(self.nb_tokens, dim))
    
    
    def forward(self, spatial_features, pos_1d):
        '''spatial_features_1: T, B, C, H, W
            pos_1d: B, C, T'''
        
        shape = spatial_features.shape
        T, B, C, H, W = shape

        pos_1d = pos_1d.permute(2, 0, 1)[:, :, None, :].repeat(1, 1, H * W, 1)

        spatial_features = spatial_features.permute(0, 1, 3, 4, 2).contiguous() # T, B, H, W, C
        spatial_features = spatial_features.view(T, B, H * W, C)

        pos_2d = self.pos_obj_2d(shape_util=(B, H, W), device=spatial_features.device)
        pos_2d = pos_2d.permute(0, 2, 3, 1).contiguous()
        pos_2d = pos_2d.view(B, H * W, C)

        #pos_1d = self.pos_obj_1d(shape_util=(B, T), device=labeled_spatial_features.device) # B, C, T
        #pos_1d = pos_1d.permute(0, 2, 1).contiguous() # B, T, C
        #pos_1d = pos_1d.repeat(1, 2*self.nb_tokens, 1) # B, 2*T*nb_tokens, C
        #pos_1d = self.pos_1d[None, :, :].repeat(B, 1, 1)
        #token_pos = self.token_pos[None, :, :].repeat(T * B, 1, 1)

        #temporal_tokens = self.temporal_tokens[:, None, :, :, :].repeat(1, B, 1, 1, 1) # (T-1), B, 2, nb_tokens, C
        #temporal_tokens = temporal_tokens.view(T * B, 2, self.nb_tokens, C)
        #temporal_features_1 = temporal_tokens[:, 0]
        #temporal_features_2 = temporal_tokens[:, 1]

        #pos = torch.cat([pos_2d, token_pos], dim=1)

        for spatial_layer in self.spatial_layers:
            out_list = []
            for i in range(len(spatial_features)):

                #matplotlib.use('QtAgg')
                #fig, ax = plt.subplots(2, len(spatial_features))
                #past_features = past_features.permute(0, 1, 3, 2).contiguous().view(T - 1, B, C, H, W)
                #spatial_features = spatial_features.permute(0, 1, 3, 2).contiguous().view(T, B, C, H, W)
                #for j in range(len(spatial_features)):
                #    ax[0, j].imshow(spatial_features[j, 0, 0].detach().cpu())
                #    if j < len(spatial_features) - 1:
                #        ax[1, j].imshow(past_features[j, 0, 0].detach().cpu())
                #plt.show()
                #past_features = past_features.permute(0, 1, 3, 4, 2).contiguous().view(T-1, B, H * W, C)
                #spatial_features = spatial_features.permute(0, 1, 3, 4, 2).contiguous().view(T, B, H * W, C)

                #fig, ax = plt.subplots(2, 1)
                #ax[0].imshow(past_features[i, 0, 0].detach().cpu())
                #ax[1].imshow(spatial_features[i, 0, 0].detach().cpu())
                #plt.show()

                current_spatial_feature = spatial_features[i]
                current_pos_1d = pos_1d[i]
                if i == 0:
                    previous_spatial_feature = torch.zeros_like(current_spatial_feature).unsqueeze(0).repeat(self.temporal_kernel_size, 1, 1, 1)
                    previous_pos_1d = torch.zeros_like(current_pos_1d).unsqueeze(0).repeat(self.temporal_kernel_size, 1, 1, 1)
                else:
                    step = int(ceil(i / self.temporal_kernel_size))
                    middle_indices = torch.arange(0, i, step)
                    previous_spatial_feature = spatial_features[middle_indices]
                    previous_pos_1d = pos_1d[middle_indices]
                    previous_spatial_feature = torch.nn.functional.pad(previous_spatial_feature, pad=(0, 0, 0, 0, 0, 0, max(0, self.temporal_kernel_size - len(previous_spatial_feature)), 0))
                    previous_pos_1d = torch.nn.functional.pad(previous_pos_1d, pad=(0, 0, 0, 0, 0, 0, max(0, self.temporal_kernel_size - len(previous_pos_1d)), 0))

                previous_pos_1d = previous_pos_1d.permute(0, 2, 1, 3).contiguous()
                previous_pos_1d = previous_pos_1d.view(self.temporal_kernel_size * H * W, B, C)
                previous_pos_1d = previous_pos_1d.permute(1, 0, 2).contiguous()

                ca_query_pos = pos_2d + current_pos_1d

                key_pos_2d = pos_2d[:, None, :, :].repeat(1, self.temporal_kernel_size, 1, 1)
                key_pos_2d = key_pos_2d.permute(1, 2, 0, 3).contiguous()
                key_pos_2d = key_pos_2d.view(self.temporal_kernel_size * H * W, B, C)
                key_pos_2d = key_pos_2d.permute(1, 0, 2).contiguous()
                key_pos = key_pos_2d + previous_pos_1d

                previous_spatial_feature = previous_spatial_feature.permute(0, 2, 1, 3).contiguous()
                previous_spatial_feature = previous_spatial_feature.view(self.temporal_kernel_size * H * W, B, C)
                previous_spatial_feature = previous_spatial_feature.permute(1, 0, 2).contiguous()

                current_spatial_feature = spatial_layer(current_spatial_feature, previous_spatial_feature, sa_query_pos=pos_2d, ca_query_pos=ca_query_pos, key_pos=key_pos)
                out_list.append(current_spatial_feature)
            spatial_features = torch.stack(out_list, dim=0)
        return spatial_features
    

class TransformerFlowEncoderEmbedding(nn.Module):
    def __init__(self, dim, nhead, num_layers, padding, embedding_dim):
        super().__init__()

        self.padding = padding

        spatial_layer = TransformerFlowLayer(d_model=dim, nhead=nhead)
        self.spatial_layers = _get_clones(spatial_layer, num_layers)

        self.embedding_proj = nn.Linear(embedding_dim, dim)

        #temporal_layer = TransformerEncoderLayer(d_model=dim, nhead=nhead)
        #self.temporal_layers = _get_clones(temporal_layer, num_layers)

        #seg_layer = TransformerEncoderLayer(d_model=dim, nhead=nhead)
        #self.seg_layers = _get_clones(seg_layer, num_layers)

        #temporal_layer = TransformerEncoderLayer(d_model=dim, nhead=nhead)
        #self.temporal_layers = _get_clones(temporal_layer, num_layers)

        self.pos_obj_2d = PositionEmbeddingSine2d(num_pos_feats=dim // 2, normalize=True)
        #self.pos_obj_1d = PositionEmbeddingSine1d(num_pos_feats=dim, normalize=True)

        #self.token_pos = nn.Parameter(torch.randn(self.nb_tokens, dim))
        #self.pos_1d = nn.Parameter(torch.randn(self.video_length, dim))
    
    
    def forward(self, spatial_features, embedding):
        '''spatial_features_1: T, B, C, H, W
            pos_1d: B, C, T'''
        
        shape = spatial_features.shape
        T, B, C, H, W = shape

        embedding = self.embedding_proj(embedding)
        embedding = embedding[None, :, None, :].repeat(T, 1, H * W, 1)
        
        pos_1d = torch.zeros(size=(T, C), device=spatial_features.device)
        pos_1d = pos_1d[:, None, None, :].repeat(1, B, H * W, 1)
        #pos_1d = self.pos_1d[:, None, None, :].repeat(1, B, H * W, 1)

        spatial_features = spatial_features.permute(0, 1, 3, 4, 2).contiguous() # T, B, H, W, C
        spatial_features = spatial_features.view(T, B, H * W, C)
        spatial_features = spatial_features + embedding

        pos_2d = self.pos_obj_2d(shape_util=(B, H, W), device=spatial_features.device)
        pos_2d = pos_2d.permute(0, 2, 3, 1).contiguous()
        pos_2d = pos_2d.view(B, H * W, C)

        #pos_1d = self.pos_obj_1d(shape_util=(B, T), device=labeled_spatial_features.device) # B, C, T
        #pos_1d = pos_1d.permute(0, 2, 1).contiguous() # B, T, C
        #pos_1d = pos_1d.repeat(1, 2*self.nb_tokens, 1) # B, 2*T*nb_tokens, C
        #pos_1d = self.pos_1d[None, :, :].repeat(B, 1, 1)
        #token_pos = self.token_pos[None, :, :].repeat(T * B, 1, 1)

        #temporal_tokens = self.temporal_tokens[:, None, :, :, :].repeat(1, B, 1, 1, 1) # (T-1), B, 2, nb_tokens, C
        #temporal_tokens = temporal_tokens.view(T * B, 2, self.nb_tokens, C)
        #temporal_features_1 = temporal_tokens[:, 0]
        #temporal_features_2 = temporal_tokens[:, 1]

        #pos = torch.cat([pos_2d, token_pos], dim=1)

        for spatial_layer in self.spatial_layers:
            out_list = []
            for i in range(len(spatial_features)):

                #matplotlib.use('QtAgg')
                #fig, ax = plt.subplots(2, len(spatial_features))
                #past_features = past_features.permute(0, 1, 3, 2).contiguous().view(T - 1, B, C, H, W)
                #spatial_features = spatial_features.permute(0, 1, 3, 2).contiguous().view(T, B, C, H, W)
                #for j in range(len(spatial_features)):
                #    ax[0, j].imshow(spatial_features[j, 0, 0].detach().cpu())
                #    if j < len(spatial_features) - 1:
                #        ax[1, j].imshow(past_features[j, 0, 0].detach().cpu())
                #plt.show()
                #past_features = past_features.permute(0, 1, 3, 4, 2).contiguous().view(T-1, B, H * W, C)
                #spatial_features = spatial_features.permute(0, 1, 3, 4, 2).contiguous().view(T, B, H * W, C)

                #fig, ax = plt.subplots(2, 1)
                #ax[0].imshow(past_features[i, 0, 0].detach().cpu())
                #ax[1].imshow(spatial_features[i, 0, 0].detach().cpu())
                #plt.show()

                current_spatial_feature = spatial_features[i]
                current_pos_1d = pos_1d[i]
                all_spatial_features = spatial_features
                all_pos_1d = pos_1d

                if self.padding == 'border':
                    all_spatial_features = torch.nn.functional.pad(all_spatial_features, pad=(0, 0, 0, 0, 0, 0, 1, 1))
                    all_pos_1d = torch.nn.functional.pad(all_pos_1d, pad=(0, 0, 0, 0, 0, 0, 1, 1))
                
                myself_pos = pos_2d + current_pos_1d

                key_pos_2d = pos_2d[:, None, :, :].repeat(1, len(all_spatial_features), 1, 1)
                key_pos_2d = key_pos_2d.permute(1, 2, 0, 3).contiguous()
                key_pos_2d = key_pos_2d.view(len(all_spatial_features) * H * W, B, C)
                key_pos_2d = key_pos_2d.permute(1, 0, 2).contiguous()

                all_pos_1d = all_pos_1d.permute(0, 2, 1, 3).contiguous() # T, H * W, B, C
                all_pos_1d = all_pos_1d.view(len(all_spatial_features) * H * W, B, C)
                all_pos_1d = all_pos_1d.permute(1, 0, 2).contiguous()
                context_pos = key_pos_2d + all_pos_1d

                T_pad = len(all_spatial_features)

                all_spatial_features = all_spatial_features.permute(0, 2, 1, 3).contiguous()
                all_spatial_features = all_spatial_features.view(len(all_spatial_features) * H * W, B, C)
                all_spatial_features = all_spatial_features.permute(1, 0, 2).contiguous()

                current_spatial_feature, weights = spatial_layer(myself=current_spatial_feature,
                                                                 context=all_spatial_features, 
                                                                 myself_pos=myself_pos,
                                                                 context_pos=context_pos)
                out_list.append(current_spatial_feature)
            spatial_features = torch.stack(out_list, dim=0)

        weights = weights.permute(2, 1, 0).contiguous().mean(1) # T_pad * H * W, B
        weights = weights.view(T_pad, H * W, B).mean(1) # T_pad, B
        if self.padding == 'border':
            weights = weights[1:-1]
        return spatial_features, weights.detach()
    

class BeforeFlow(nn.Module):
    def __init__(self, dim, nhead, nb_layers):
        super().__init__()
        
        self.pos_obj_2d = PositionEmbeddingSine2d(num_pos_feats=dim // 2, normalize=True)

        self_attention_layer = TransformerEncoderLayer(d_model=dim, nhead=nhead)
        self.self_attention_layers = _get_clones(self_attention_layer, nb_layers)
        
    
    def forward(self, x1, x2):
        # B, C, H, W
        B, C, H, W = x1.shape

        pos_2d = self.pos_obj_2d(shape_util=(B, H, W), device=x1.device) # B, C, H, W
        pos_2d = pos_2d.permute(0, 2, 3, 1).contiguous()
        pos_2d = pos_2d.view(B, H * W, C)
        pos_2d = torch.cat([pos_2d, pos_2d], dim=0)

        x1 = x1.permute(0, 2, 3, 1).contiguous()
        x1 = x1.view(B, H * W, C)

        x2 = x2.permute(0, 2, 3, 1).contiguous()
        x2 = x2.view(B, H * W, C)

        for layer in self.self_attention_layers:
            concat0 = torch.cat([x1, x2], dim=0)
            concat0 = layer(concat0, pos=pos_2d)[0]
            x1, x2 = concat0.chunk(2, dim=0)

        x1 = x1.permute(0, 2, 1).contiguous()
        x1 = x1.view(B, C, H, W)

        x2 = x2.permute(0, 2, 1).contiguous()
        x2 = x2.view(B, C, H, W)

        return x1, x2



class Interpolator(nn.Module):
    def __init__(self, dim, nhead, nb_layers, nb_interpolated_frame):
        super().__init__()
        self.nb_interpolated_frame = nb_interpolated_frame
        self.nb_layers = nb_layers
        
        self.pos_obj_2d = PositionEmbeddingSine2d(num_pos_feats=dim // 2, normalize=True)

        self.pos_1d_1 = nn.Parameter(torch.randn(nb_interpolated_frame + 2, dim))
        self.pos_1d_2 = nn.Parameter(torch.randn(nb_interpolated_frame, dim))

        cross_attention_layer_1 = TransformerFlowLayer(d_model=dim, nhead=nhead)
        self.cross_attention_layers_1 = _get_clones(cross_attention_layer_1, nb_layers)

        cross_attention_layer_2 = TransformerFlowLayer(d_model=dim, nhead=nhead)
        self.cross_attention_layers_2 = _get_clones(cross_attention_layer_2, nb_layers)

        self_attention_layer = TransformerEncoderLayer(d_model=dim, nhead=nhead)
        self.self_attention_layers = _get_clones(self_attention_layer, nb_layers)

        decoder_layer = TransformerFlowLayer(d_model=dim, nhead=nhead)
        self.decoder_layers = _get_clones(decoder_layer, nb_layers)

        self.reduce1 = nn.Linear(dim * 2, dim)
        #self.reduce2 = nn.Linear(dim * 2, dim)
        
    
    def forward(self, x1, x2):
        # B, C, H, W
        B, C, H, W = x1.shape

        pos_2d = self.pos_obj_2d(shape_util=(B, H, W), device=x1.device) # B, C, H, W
        pos_2d = pos_2d.permute(0, 2, 3, 1).contiguous()
        pos_2d = pos_2d.view(B, H * W, C) # B, H * W, C
        
        pos_1d_1 = self.pos_1d_1[:, None, None].repeat(1, B, H * W, 1)
        pos_1d_2 = self.pos_1d_2[:, None, None].repeat(1, B, H * W, 1)

        x1 = x1.permute(0, 2, 3, 1).contiguous()
        x1 = x1.view(B, H * W, C)

        x2 = x2.permute(0, 2, 3, 1).contiguous()
        x2 = x2.view(B, H * W, C)


        for l in range(self.nb_layers):
            query = torch.cat([x2, x1], dim=0)
            key = torch.cat([x1, x2], dim=0)
            pos = torch.cat([pos_2d, pos_2d], dim=0)
            query = self.cross_attention_layers_1[l](query=query, key=key, query_pos=pos, key_pos=pos)[0]
            x2, x1 = query.chunk(2, dim=0)

        inter = torch.cat([x2, x1], dim=-1)
        inter = self.reduce1(inter)

        inter = inter[None].repeat(self.nb_interpolated_frame + 2, 1, 1, 1)
        inter = inter.view((self.nb_interpolated_frame + 2) * B, H * W, C)

        pos = pos_2d[None] + pos_1d_1
        pos = pos.view((self.nb_interpolated_frame + 2) * B, H * W, C)

        for l in range(self.nb_layers):
            inter = self.self_attention_layers[l](inter, pos=pos)[0]
        
        inter = inter.view(self.nb_interpolated_frame + 2, B, H * W, C)
        inter1 = inter[:-1]
        inter2 = inter[1:]
        inter1 = inter1.view((self.nb_interpolated_frame + 1) * B, H * W, C)
        inter2 = inter2.view((self.nb_interpolated_frame + 1) * B, H * W, C)

        pos = pos_2d[None].repeat(self.nb_interpolated_frame + 1, 1, 1, 1)
        pos = pos.view((self.nb_interpolated_frame + 1) * B, H * W, C)

        for l in range(self.nb_layers):
            inter2 = self.cross_attention_layers_2[l](query=inter2, key=inter1, query_pos=pos, key_pos=pos)[0]

        inter2 = inter2.view(self.nb_interpolated_frame + 1, B, H * W, C)

        key = inter2[0]
        for i in range(1, len(inter2)):
            pos = pos_2d + pos_1d_2[i - 1]
            for l in range(self.nb_layers):
                attn_out = self.decoder_layers[l](query=inter2[i], key=key, query_pos=pos, key_pos=pos)[0]
            key = attn_out
        global_motion = key
        
        global_motion = global_motion.permute(0, 2, 1).contiguous()
        global_motion = global_motion.view(B, C, H, W)

        inter2 = inter2.permute(0, 1, 3, 2).contiguous()
        inter2 = inter2.view((self.nb_interpolated_frame + 1), B, C, H, W)

        inter = inter.permute(0, 1, 3, 2).contiguous()
        inter = inter.view(self.nb_interpolated_frame + 2, B, C, H, W)

        return global_motion, inter, inter2
            


class TransformerFlowEncoderInter(nn.Module):
    def __init__(self, dim, nhead):
        super().__init__()
        self.spatial_layer = TransformerFlowLayerContext(d_model=dim, nhead=nhead)
    
    
    def forward(self, query, key, query_pos, key_pos):
        '''spatial_features_1: T, B, C, H, W
            pos_1d: B, C, T'''
        
        T_q, B, C, H, W = query.shape

        query = query.permute(0, 1, 3, 4, 2).contiguous() # T, B, H, W, C
        query = query.view(T_q, B, H * W, C)

        query_pos = query_pos.permute(0, 1, 3, 4, 2).contiguous() # T, B, H, W, C
        query_pos = query_pos.view(T_q, B, H * W, C)

        key = key.permute(0, 2, 3, 1).contiguous() # B, H, W, C
        key = key.view(B, H * W, C)
        key_pos = key_pos.permute(0, 2, 3, 1).contiguous() # B, H, W, C
        key_pos = key_pos.view(B, H * W, C)

        out_list = []
        for i in range(len(query)):

            myself = query[i] # B, H * W, C
            myself_pos = query_pos[i]
            context = query
            context_pos = query_pos
            other = key # T_k, B, H * W, C
            other_pos = key_pos

            context = context.permute(0, 2, 1, 3).contiguous()
            context = context.view(len(context) * H * W, B, C)
            context = context.permute(1, 0, 2).contiguous()

            context_pos = context_pos.permute(0, 2, 1, 3).contiguous()
            context_pos = context_pos.view(len(context_pos) * H * W, B, C)
            context_pos = context_pos.permute(1, 0, 2).contiguous()

            myself, _ = self.spatial_layer(myself=myself,
                                            other=other,
                                            context=context,
                                            myself_pos=myself_pos,
                                            other_pos=other_pos,
                                            context_pos=context_pos)
            out_list.append(myself)

        query = torch.stack(out_list, dim=0) # T_q, B, L, C
        query = query.permute(0, 1, 3, 2).contiguous() # T_q, B, C, L
        query = query.view(T_q, B, C, H, W)

        return query
    

class TransformerFlowEncoderAllOnlyContext(nn.Module):
    def __init__(self, dim, nhead, num_layers, padding):
        super().__init__()

        self.padding = padding

        spatial_layer = TransformerFlowLayer(d_model=dim, nhead=nhead)
        self.spatial_layers = _get_clones(spatial_layer, num_layers)

        #temporal_layer = TransformerEncoderLayer(d_model=dim, nhead=nhead)
        #self.temporal_layers = _get_clones(temporal_layer, num_layers)

        #seg_layer = TransformerEncoderLayer(d_model=dim, nhead=nhead)
        #self.seg_layers = _get_clones(seg_layer, num_layers)

        #temporal_layer = TransformerEncoderLayer(d_model=dim, nhead=nhead)
        #self.temporal_layers = _get_clones(temporal_layer, num_layers)

        self.pos_obj_2d = PositionEmbeddingSine2d(num_pos_feats=dim // 2, normalize=True)
        #self.pos_obj_1d = PositionEmbeddingSine1d(num_pos_feats=dim, normalize=True)

        #self.token_pos = nn.Parameter(torch.randn(self.nb_tokens, dim))
        #self.pos_1d = nn.Parameter(torch.randn(self.video_length, dim))
    
    
    def forward(self, spatial_features):
        '''spatial_features_1: T, B, C, H, W
            pos_1d: B, C, T'''
        
        shape = spatial_features.shape
        T, B, C, H, W = shape

        pos_1d = torch.zeros(size=(T, C), device=spatial_features.device)
        pos_1d = pos_1d[:, None, None, :].repeat(1, B, H * W, 1)
        #pos_1d = self.pos_1d[:, None, None, :].repeat(1, B, H * W, 1)

        spatial_features = spatial_features.permute(0, 1, 3, 4, 2).contiguous() # T, B, H, W, C
        spatial_features = spatial_features.view(T, B, H * W, C)

        pos_2d = self.pos_obj_2d(shape_util=(B, H, W), device=spatial_features.device)
        pos_2d = pos_2d.permute(0, 2, 3, 1).contiguous()
        pos_2d = pos_2d.view(B, H * W, C)

        #pos_1d = self.pos_obj_1d(shape_util=(B, T), device=labeled_spatial_features.device) # B, C, T
        #pos_1d = pos_1d.permute(0, 2, 1).contiguous() # B, T, C
        #pos_1d = pos_1d.repeat(1, 2*self.nb_tokens, 1) # B, 2*T*nb_tokens, C
        #pos_1d = self.pos_1d[None, :, :].repeat(B, 1, 1)
        #token_pos = self.token_pos[None, :, :].repeat(T * B, 1, 1)

        #temporal_tokens = self.temporal_tokens[:, None, :, :, :].repeat(1, B, 1, 1, 1) # (T-1), B, 2, nb_tokens, C
        #temporal_tokens = temporal_tokens.view(T * B, 2, self.nb_tokens, C)
        #temporal_features_1 = temporal_tokens[:, 0]
        #temporal_features_2 = temporal_tokens[:, 1]

        #pos = torch.cat([pos_2d, token_pos], dim=1)

        for spatial_layer in self.spatial_layers:
            out_list = []
            for i in range(len(spatial_features)):

                #matplotlib.use('QtAgg')
                #fig, ax = plt.subplots(2, len(spatial_features))
                #past_features = past_features.permute(0, 1, 3, 2).contiguous().view(T - 1, B, C, H, W)
                #spatial_features = spatial_features.permute(0, 1, 3, 2).contiguous().view(T, B, C, H, W)
                #for j in range(len(spatial_features)):
                #    ax[0, j].imshow(spatial_features[j, 0, 0].detach().cpu())
                #    if j < len(spatial_features) - 1:
                #        ax[1, j].imshow(past_features[j, 0, 0].detach().cpu())
                #plt.show()
                #past_features = past_features.permute(0, 1, 3, 4, 2).contiguous().view(T-1, B, H * W, C)
                #spatial_features = spatial_features.permute(0, 1, 3, 4, 2).contiguous().view(T, B, H * W, C)

                #fig, ax = plt.subplots(2, 1)
                #ax[0].imshow(past_features[i, 0, 0].detach().cpu())
                #ax[1].imshow(spatial_features[i, 0, 0].detach().cpu())
                #plt.show()

                current_spatial_feature = spatial_features[i]
                current_pos_1d = pos_1d[i]
                all_spatial_features = spatial_features
                all_pos_1d = pos_1d

                if self.padding == 'border':
                    all_spatial_features = torch.nn.functional.pad(all_spatial_features, pad=(0, 0, 0, 0, 0, 0, 1, 1))
                    all_pos_1d = torch.nn.functional.pad(all_pos_1d, pad=(0, 0, 0, 0, 0, 0, 1, 1))
                
                myself_pos = pos_2d + current_pos_1d

                key_pos_2d = pos_2d[:, None, :, :].repeat(1, len(all_spatial_features), 1, 1)
                key_pos_2d = key_pos_2d.permute(1, 2, 0, 3).contiguous()
                key_pos_2d = key_pos_2d.view(len(all_spatial_features) * H * W, B, C)
                key_pos_2d = key_pos_2d.permute(1, 0, 2).contiguous()

                all_pos_1d = all_pos_1d.permute(0, 2, 1, 3).contiguous() # T, H * W, B, C
                all_pos_1d = all_pos_1d.view(len(all_spatial_features) * H * W, B, C)
                all_pos_1d = all_pos_1d.permute(1, 0, 2).contiguous()
                context_pos = key_pos_2d + all_pos_1d

                T_pad = len(all_spatial_features)

                all_spatial_features = all_spatial_features.permute(0, 2, 1, 3).contiguous()
                all_spatial_features = all_spatial_features.view(len(all_spatial_features) * H * W, B, C)
                all_spatial_features = all_spatial_features.permute(1, 0, 2).contiguous()

                current_spatial_feature, weights = spatial_layer(query=current_spatial_feature,
                                                                 key=all_spatial_features, 
                                                                 query_pos=myself_pos,
                                                                 key_pos=context_pos)
                out_list.append(current_spatial_feature)
            spatial_features = torch.stack(out_list, dim=0)

        weights = weights.permute(2, 1, 0).contiguous().mean(1) # T_pad * H * W, B
        weights = weights.view(T_pad, H * W, B).mean(1) # T_pad, B
        if self.padding == 'border':
            weights = weights[1:-1]
        return spatial_features, weights.detach()



class Gating(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.state_out_to_gate = nn.Linear(dim, dim)
        self.learned_ema_beta = nn.Parameter(torch.randn(dim))

    def forward(self, after, before):
        z = self.state_out_to_gate(after)
        learned_ema_decay = self.learned_ema_beta.sigmoid()
        return learned_ema_decay * z + (1 - learned_ema_decay) * before


class TransformerFlowSegEncoderIterative(nn.Module):
    def __init__(self, dim, nhead, nb_iters):
        super().__init__()
        self.nb_iters = nb_iters

        self.self_attention_layer = TransformerEncoderLayer(d_model=dim, nhead=nhead)
        self.bilateral_attention_layer = TransformerFlowLayer(d_model=dim, nhead=nhead)
        self.multilateral_attention_layer = TransformerFlowLayer(d_model=dim, nhead=nhead)
        self.pos_obj_2d = PositionEmbeddingSine2d(num_pos_feats=dim // 2, normalize=True)
        self.time_emb = nn.Parameter(torch.randn(nb_iters, dim))
        self.cross_attn_emb1 = nn.Parameter(torch.randn(dim))
        self.cross_attn_emb2 = nn.Parameter(torch.randn(dim))

        self.flow_gating = Gating(dim)
        self.seg_gating = Gating(dim)
        self.mean_gating = Gating(dim)
    
    
    def forward(self, unlabeled):
        '''unlabeled: T, B, C, H, W'''
        
        shape = unlabeled.shape
        T, B, C, H, W = shape

        time_emb = self.time_emb[:, None, None, :].repeat(1, T * B, H * W, 1)

        cross_attn_emb1 = self.cross_attn_emb1[None, None].repeat(T * B, H * W, 1)
        cross_attn_emb2 = self.cross_attn_emb2[None, None].repeat(T * B, H * W, 1)

        unlabeled = unlabeled.permute(0, 1, 3, 4, 2).contiguous() # T, B, H, W, C
        unlabeled = unlabeled.view(T, B, H * W, C)

        pos_2d = self.pos_obj_2d(shape_util=(B, H, W), device=unlabeled.device)
        pos_2d = pos_2d.permute(0, 2, 3, 1).contiguous()
        pos_2d = pos_2d.view(B, H * W, C)
        pos_2d = pos_2d[None].repeat(T, 1, 1, 1)
        pos_2d = pos_2d.view(T * B, H * W, C)


        anchor = unlabeled[0][None].repeat(T, 1, 1, 1) # T, B, H * W, C
        frames = unlabeled # T, B, H * W, C

        anchor = anchor.view(T * B, H * W, C)
        frames = frames.view(T * B, H * W, C)
        flow = frames

        for i in range(self.nb_iters):
            current_time_emb = time_emb[i]
            pos = current_time_emb + pos_2d

            before = torch.cat([frames, anchor], dim=0)
            pos = torch.cat([pos, pos], dim=0)
            after = self.self_attention_layer(before, pos=pos)[0]
            after = self.seg_gating(after, before)
            
            frames, anchor = after.chunk(2, dim=0)
            pos = pos.chunk(2, dim=0)[0]

            key = torch.cat([frames, anchor], dim=1)
            key_pos = torch.cat([pos, pos], dim=1) + torch.cat([cross_attn_emb1, cross_attn_emb2], dim=1)

            before_flow = flow
            flow = self.bilateral_attention_layer(query=flow, key=key, query_pos=pos, key_pos=key_pos)[0]
            flow = self.flow_gating(flow, before_flow)

            flow_mean = flow.view(T, B, H * W, C).mean(0)
            flow_mean = flow_mean[None].repeat(T, 1, 1, 1)
            flow_mean = flow_mean.view(T * B, H * W, C)

            frames_mean = frames.view(T, B, H * W, C).mean(0)
            frames_mean = frames_mean[None].repeat(T, 1, 1, 1)
            frames_mean = frames_mean.view(T * B, H * W, C)

            anchor_mean = anchor.view(T, B, H * W, C).mean(0)
            anchor_mean = anchor_mean[None].repeat(T, 1, 1, 1)
            anchor_mean = anchor_mean.view(T * B, H * W, C)

            concat0 = torch.cat([frames, flow, anchor], dim=0)
            concat1 = torch.cat([frames_mean, flow_mean, anchor_mean], dim=0)

            pos_2d = pos_2d.repeat(3, 1, 1)
            current_time_emb = current_time_emb.repeat(3, 1, 1)

            pos = pos_2d + current_time_emb

            before = concat0
            concat0 = self.multilateral_attention_layer(query=concat0, key=concat1, query_pos=pos, key_pos=pos)[0]
            concat0 = self.mean_gating(concat0, before)

            frames, flow, anchor = concat0.chunk(3, dim=0)
            pos_2d = pos_2d.chunk(3, dim=0)[0]

        frames = frames.view(T, B, H * W, C)
        frames = frames.permute(0, 1, 3, 2).contiguous()
        frames = frames.view(T, B, C, H, W)

        flow = flow.view(T, B, H * W, C)
        flow = flow.permute(0, 1, 3, 2).contiguous()
        flow = flow.view(T, B, C, H, W)

        anchor = anchor.view(T, B, H * W, C)
        anchor = anchor.permute(0, 1, 3, 2).contiguous()
        anchor = anchor.view(T, B, C, H, W)

        return frames, flow, anchor



class TransformerFlowSegEncoder2(nn.Module):
    def __init__(self, dim, nhead, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self_attention_layer = TransformerEncoderLayer(d_model=dim, nhead=nhead)
        bilateral_attention_layer = TransformerFlowLayer(d_model=dim, nhead=nhead)
        multilateral_attention_layer = TransformerFlowLayer(d_model=dim, nhead=nhead)

        self.self_attention_layers = _get_clones(self_attention_layer, num_layers)
        self.bilateral_attention_layers = _get_clones(bilateral_attention_layer, num_layers)
        self.multilateral_attention_layers = _get_clones(multilateral_attention_layer, num_layers)

        self.pos_obj_2d = PositionEmbeddingSine2d(num_pos_feats=dim // 2, normalize=True)
        self.cross_attn_emb1 = nn.Parameter(torch.randn(dim))
        self.cross_attn_emb2 = nn.Parameter(torch.randn(dim))
    
    
    def forward(self, unlabeled, label=None):
        '''unlabeled: T, B, C, H, W'''
        
        shape = unlabeled.shape
        T, B, C, H, W = shape

        cross_attn_emb1 = self.cross_attn_emb1[None, None].repeat(T * B, H * W, 1)
        cross_attn_emb2 = self.cross_attn_emb2[None, None].repeat(T * B, H * W, 1)

        unlabeled = unlabeled.permute(0, 1, 3, 4, 2).contiguous() # T, B, H, W, C
        unlabeled = unlabeled.view(T, B, H * W, C)

        pos_2d = self.pos_obj_2d(shape_util=(B, H, W), device=unlabeled.device)
        pos_2d = pos_2d.permute(0, 2, 3, 1).contiguous()
        pos_2d = pos_2d.view(B, H * W, C)
        pos_2d = pos_2d[None].repeat(T, 1, 1, 1)
        pos_2d = pos_2d.view(T * B, H * W, C)

        if label is not None:
            label = label.permute(0, 2, 3, 1).contiguous()
            label = label.view(B, H * W, C)
            anchor = label[None].repeat(T, 1, 1, 1) # T, B, H * W, C
        else:
            anchor = unlabeled[0][None].repeat(T, 1, 1, 1) # T, B, H * W, C

        frames = unlabeled # T, B, H * W, C

        anchor = anchor.view(T * B, H * W, C)
        frames = frames.view(T * B, H * W, C)
        flow = frames

        for l in range(self.num_layers):

            concat0 = torch.cat([frames, anchor], dim=0)
            pos = torch.cat([pos_2d, pos_2d], dim=0)
            concat0 = self.self_attention_layers[l](concat0, pos=pos)[0]
            
            frames, anchor = concat0.chunk(2, dim=0)

            key = torch.cat([frames, anchor], dim=1)
            key_pos = torch.cat([pos_2d, pos_2d], dim=1) + torch.cat([cross_attn_emb1, cross_attn_emb2], dim=1)

            flow = self.bilateral_attention_layers[l](query=flow, key=key, query_pos=pos_2d, key_pos=key_pos)[0]

            flow_mean = flow.view(T, B, H * W, C).mean(0)
            flow_mean = flow_mean[None].repeat(T, 1, 1, 1)
            flow_mean = flow_mean.view(T * B, H * W, C)

            frames_mean = frames.view(T, B, H * W, C).mean(0)
            frames_mean = frames_mean[None].repeat(T, 1, 1, 1)
            frames_mean = frames_mean.view(T * B, H * W, C)

            anchor_mean = anchor.view(T, B, H * W, C).mean(0)
            anchor_mean = anchor_mean[None].repeat(T, 1, 1, 1)
            anchor_mean = anchor_mean.view(T * B, H * W, C)

            concat0 = torch.cat([frames, flow, anchor], dim=0)
            concat1 = torch.cat([frames_mean, flow_mean, anchor_mean], dim=0)

            pos = pos_2d.repeat(3, 1, 1)

            concat0 = self.multilateral_attention_layers[l](query=concat0, key=concat1, query_pos=pos, key_pos=pos)[0]

            frames, flow, anchor = concat0.chunk(3, dim=0)

        frames = frames.view(T, B, H * W, C)
        frames = frames.permute(0, 1, 3, 2).contiguous()
        frames = frames.view(T, B, C, H, W)

        flow = flow.view(T, B, H * W, C)
        flow = flow.permute(0, 1, 3, 2).contiguous()
        flow = flow.view(T, B, C, H, W)

        anchor = anchor.view(T, B, H * W, C)
        anchor = anchor.permute(0, 1, 3, 2).contiguous()
        anchor = anchor.view(T, B, C, H, W)

        return frames, flow, anchor




class TransformerFlowSegEncoderMutual(nn.Module):
    def __init__(self, dim, nhead, num_layers):
        super().__init__()
        self.num_layers = num_layers
        bilateral_attention_layer = TransformerFlowLayer(d_model=dim, nhead=nhead)
        #multilateral_attention_layer = TransformerFlowLayer(d_model=dim, nhead=nhead)

        self.bilateral_attention_layers = _get_clones(bilateral_attention_layer, num_layers)
        #self.multilateral_attention_layers = _get_clones(multilateral_attention_layer, num_layers)

        self.pos_obj_2d = PositionEmbeddingSine2d(num_pos_feats=dim // 2, normalize=True)
    
    
    def forward(self, unlabeled):
        '''unlabeled: T, B, C, H, W'''
        
        shape = unlabeled.shape
        T, B, C, H, W = shape

        unlabeled = unlabeled.permute(0, 1, 3, 4, 2).contiguous() # T, B, H, W, C
        unlabeled = unlabeled.view(T, B, H * W, C)

        pos_2d = self.pos_obj_2d(shape_util=(B, H, W), device=unlabeled.device)
        pos_2d = pos_2d.permute(0, 2, 3, 1).contiguous()
        pos_2d = pos_2d.view(B, H * W, C)
        pos_2d = pos_2d[None].repeat(T, 1, 1, 1)
        pos_2d = pos_2d.view(T * B, H * W, C)

        anchor = unlabeled[0][None].repeat(T, 1, 1, 1) # T, B, H * W, C
        frames = unlabeled # T, B, H * W, C

        anchor = anchor.view(T * B, H * W, C)
        frames = frames.view(T * B, H * W, C)

        for l in range(self.num_layers):

            concat0 = torch.cat([frames, anchor], dim=0)
            concat1 = torch.cat([anchor, frames], dim=0)
            pos = torch.cat([pos_2d, pos_2d], dim=0)
            concat0 = self.bilateral_attention_layers[l](query=concat0, key=concat1, query_pos=pos, key_pos=pos)[0]
            
            frames, anchor = concat0.chunk(2, dim=0)

            #frames_mean = frames.view(T, B, H * W, C).mean(0)
            #frames_mean = frames_mean[None].repeat(T, 1, 1, 1)
            #frames_mean = frames_mean.view(T * B, H * W, C)
#
            #anchor_mean = anchor.view(T, B, H * W, C).mean(0)
            #anchor_mean = anchor_mean[None].repeat(T, 1, 1, 1)
            #anchor_mean = anchor_mean.view(T * B, H * W, C)
#
            #concat0 = torch.cat([frames, anchor], dim=0)
            #concat1 = torch.cat([frames_mean, anchor_mean], dim=0)
#
            #pos = pos_2d.repeat(2, 1, 1)
#
            #concat0 = self.multilateral_attention_layers[l](query=concat0, key=concat1, query_pos=pos, key_pos=pos)[0]
#
            #frames, anchor = concat0.chunk(2, dim=0)

        frames = frames.view(T, B, H * W, C)
        frames = frames.permute(0, 1, 3, 2).contiguous()
        frames = frames.view(T, B, C, H, W)

        anchor = anchor.view(T, B, H * W, C)
        anchor = anchor.permute(0, 1, 3, 2).contiguous()
        anchor = anchor.view(T, B, C, H, W)

        return frames, anchor




class TransformerFlowSegEncoderMutualSimple(nn.Module):
    def __init__(self, dim, nhead, num_layers):
        super().__init__()
        self.num_layers = num_layers

        bilateral_attention_layer = TransformerFlowLayer(d_model=dim, nhead=nhead)
        self.bilateral_attention_layers = _get_clones(bilateral_attention_layer, num_layers)

        self.pos_obj_2d = PositionEmbeddingSine2d(num_pos_feats=dim // 2, normalize=True)
    
    
    def forward(self, unlabeled):
        '''unlabeled: T, B, C, H, W'''
        
        shape = unlabeled.shape
        T, B, C, H, W = shape

        unlabeled = unlabeled.permute(0, 1, 3, 4, 2).contiguous() # T, B, H, W, C
        unlabeled = unlabeled.view(T, B, H * W, C)

        pos_2d = self.pos_obj_2d(shape_util=(B, H, W), device=unlabeled.device)
        pos_2d = pos_2d.permute(0, 2, 3, 1).contiguous()
        pos_2d = pos_2d.view(B, H * W, C)

        x1 = unlabeled[0] # B, H * W, C
        x2 = unlabeled[-1] # B, H * W, C

        for l in range(self.num_layers):
            concat0 = torch.cat([x2, x1], dim=0)
            concat1 = torch.cat([x1, x2], dim=0)
            pos = torch.cat([pos_2d, pos_2d], dim=0)
            concat0 = self.bilateral_attention_layers[l](query=concat0, key=concat1, query_pos=pos, key_pos=pos)[0]
            x2, x1 = torch.chunk(concat0, chunks=2, dim=0)

        x2 = x2.permute(0, 2, 1).contiguous()
        x2 = x2.view(B, C, H, W)

        return x2
    

    


class TransformerFlowSegEncoderAggregation(nn.Module):
    def __init__(self, dim, nhead, num_layers):
        super().__init__()
        self.num_layers = num_layers

        bilateral_attention_layer = TransformerFlowLayer(d_model=dim, nhead=nhead)
        self.bilateral_attention_layers = _get_clones(bilateral_attention_layer, num_layers)

        self.decoder_layer = TransformerFlowLayer(d_model=dim, nhead=nhead)

        self.pos_obj_2d = PositionEmbeddingSine2d(num_pos_feats=dim // 2, normalize=True)
    
    
    def forward(self, unlabeled):
        '''unlabeled: T, B, C, H, W'''
        
        shape = unlabeled.shape
        T, B, C, H, W = shape

        unlabeled = unlabeled.permute(0, 1, 3, 4, 2).contiguous() # T, B, H, W, C
        unlabeled = unlabeled.view(T, B, H * W, C)

        pos_2d = self.pos_obj_2d(shape_util=(B, H, W), device=unlabeled.device)
        pos_2d = pos_2d.permute(0, 2, 3, 1).contiguous()
        pos_2d = pos_2d.view(B, H * W, C)
        pos_2d = pos_2d[None].repeat(T, 1, 1, 1)
        pos_2d = pos_2d.view(T * B, H * W, C)

        backward = unlabeled[:-1]
        forward = unlabeled
        backward = torch.cat([unlabeled[0][None], backward], dim=0)

        backward = backward.view(T * B, H * W, C)
        forward = forward.view(T * B, H * W, C)

        for l in range(self.num_layers):
            concat0 = torch.cat([forward, backward], dim=0)
            concat1 = torch.cat([backward, forward], dim=0)
            pos = torch.cat([pos_2d, pos_2d], dim=0)
            concat0 = self.bilateral_attention_layers[l](query=concat0, key=concat1, query_pos=pos, key_pos=pos)[0]
            forward, backward = torch.chunk(concat0, chunks=2, dim=0)

        forward = forward.view(T, B, H * W, C)
        pos_2d = pos_2d.view(T, B, H * W, C)

        global_motion_forward_list = []
        key = forward[0]
        for i in range(len(forward)):
            attn_out = self.decoder_layer(query=forward[i], key=key, query_pos=pos_2d[i], key_pos=pos_2d[i])[0]
            key = attn_out
            global_motion_forward_list.append(key)
        global_motion_forward = torch.stack(global_motion_forward_list, dim=0)

        global_motion_forward = global_motion_forward.permute(0, 1, 3, 2).contiguous()
        global_motion_forward = global_motion_forward.view(T, B, C, H, W)

        forward = forward.permute(0, 1, 3, 2).contiguous()
        forward = forward.view(T, B, C, H, W)

        return forward, global_motion_forward



class TransformerFlowEncoderFromStart(nn.Module):
    def __init__(self, dim, nhead, num_layers):
        super().__init__()
        self.num_layers = num_layers

        bilateral_attention_layer = TransformerFlowLayer(d_model=dim, nhead=nhead)
        self.bilateral_attention_layers = _get_clones(bilateral_attention_layer, num_layers)

        decoder_layer = TransformerFlowLayer(d_model=dim, nhead=nhead)
        self.decoder_layers = _get_clones(decoder_layer, num_layers)

        self.pos_obj_2d = PositionEmbeddingSine2d(num_pos_feats=dim // 2, normalize=True)

        proj1 = nn.Linear(dim, dim)
        self.proj1 = _get_clones(proj1, num_layers)
        proj2 = nn.Linear(dim, dim)
        self.proj2 = _get_clones(proj2, num_layers)
    
    
    def forward(self, unlabeled, dist_emb):
        '''unlabeled: T, B, C, H, W
        dist_emb: B, T-1, C'''
        
        shape = unlabeled.shape
        T, B, C, H, W = shape

        dist_emb = dist_emb.permute(1, 0, 2).contiguous()
        dist_emb = dist_emb.view((T-1) * B, C)
        dist_emb = dist_emb[:, None, :].repeat(1, H * W, 1) # (T-1) * B, H * W, C

        unlabeled = unlabeled.permute(0, 1, 3, 4, 2).contiguous() # T, B, H, W, C
        unlabeled = unlabeled.view(T, B, H * W, C)

        pos_2d = self.pos_obj_2d(shape_util=(B, H, W), device=unlabeled.device)
        pos_2d = pos_2d.permute(0, 2, 3, 1).contiguous()
        pos_2d = pos_2d.view(B, H * W, C)
        pos_2d = pos_2d[None].repeat(T - 1, 1, 1, 1)
        pos_2d = pos_2d.view((T-1) * B, H * W, C)

        backward = unlabeled[0][None].repeat(len(unlabeled) - 1, 1, 1, 1)
        forward = unlabeled[1:]

        backward = backward.view((T-1) * B, H * W, C)
        forward = forward.view((T-1) * B, H * W, C)

        for l in range(self.num_layers):
            dist_emb = self.proj1[l](dist_emb)
            concat0 = torch.cat([forward, backward], dim=0)
            concat1 = torch.cat([backward, forward], dim=0)
            pos = torch.cat([pos_2d + dist_emb, pos_2d + dist_emb], dim=0)
            concat0 = self.bilateral_attention_layers[l](query=concat0, key=concat1, query_pos=pos, key_pos=pos)[0]
            forward, backward = torch.chunk(concat0, chunks=2, dim=0)

            forward = forward.view(T - 1, B, H * W, C)
            pos_2d = pos_2d.view(T - 1, B, H * W, C)
            dist_emb = dist_emb.view(T - 1, B, H * W, C)

            dist_emb = self.proj2[l](dist_emb)

            key_list = [forward[0]]
            for i in range(1, len(forward)):
                query_pos = pos_2d[i] + dist_emb[i]
                key_pos = pos_2d[i-1] + dist_emb[i-1]
                attn_out = self.decoder_layers[l](query=forward[i], key=key_list[i - 1], query_pos=query_pos, key_pos=key_pos)[0]
                key_list.append(attn_out)
            forward = torch.stack(key_list, dim=0)

            forward = forward.permute(0, 1, 3, 2).contiguous()
            forward = forward.view(T - 1, B, C, H, W)

        return forward




class TransformerFlowEncoderLocalGlobal(nn.Module):
    def __init__(self, dim, nhead, num_layers):
        super().__init__()
        self.num_layers = num_layers

        bilateral_attention_layer = TransformerFlowLayer(d_model=dim, nhead=nhead)
        self.bilateral_attention_layers = _get_clones(bilateral_attention_layer, num_layers)

        decoder_layer = TransformerFlowLayer(d_model=dim, nhead=nhead)
        self.decoder_layers = _get_clones(decoder_layer, num_layers)

        self.pos_obj_2d = PositionEmbeddingSine2d(num_pos_feats=dim // 2, normalize=True)

        proj2 = nn.Linear(dim, dim)
        self.proj2 = _get_clones(proj2, num_layers)
    
    
    def forward(self, unlabeled, dist_emb):
        '''unlabeled: T, B, C, H, W
        dist_emb: B, T-1, C'''
        
        shape = unlabeled.shape
        T, B, C, H, W = shape

        dist_emb = dist_emb.permute(1, 0, 2).contiguous()
        dist_emb = dist_emb.view((T-1) * B, C)
        dist_emb = dist_emb[:, None, :].repeat(1, H * W, 1) # (T-1) * B, H * W, C

        unlabeled = unlabeled.permute(0, 1, 3, 4, 2).contiguous() # T, B, H, W, C
        unlabeled = unlabeled.view(T, B, H * W, C)

        pos_2d = self.pos_obj_2d(shape_util=(B, H, W), device=unlabeled.device)
        pos_2d = pos_2d.permute(0, 2, 3, 1).contiguous()
        pos_2d = pos_2d.view(B, H * W, C)
        pos_2d = pos_2d[None].repeat(T - 1, 1, 1, 1)
        pos_2d = pos_2d.view((T-1) * B, H * W, C)

        local_backward = unlabeled[:-1]
        local_forward = unlabeled[1:]

        for l in range(self.num_layers):

            local_backward = local_backward.view((T-1) * B, H * W, C)
            local_forward = local_forward.view((T-1) * B, H * W, C) 

            concat0 = torch.cat([local_forward, local_backward], dim=0)
            concat1 = torch.cat([local_backward, local_forward], dim=0)
            pos = torch.cat([pos_2d, pos_2d], dim=0)
            concat0 = self.bilateral_attention_layers[l](query=concat0, key=concat1, query_pos=pos, key_pos=pos)[0]
            local_forward, local_backward = torch.chunk(concat0, chunks=2, dim=0)

            local_forward = local_forward.view(T - 1, B, H * W, C)
            local_backward = local_backward.view(T - 1, B, H * W, C)
        
        pos_2d = pos_2d.view(T - 1, B, H * W, C)
        dist_emb = dist_emb.view(T - 1, B, H * W, C)
        global_forward = local_forward
        
        for l in range(self.num_layers):

            dist_emb = self.proj2[l](dist_emb)

            key_list = [global_forward[0]]
            for i in range(1, len(global_forward)):
                query_pos = pos_2d[i] + dist_emb[i]
                key_pos = pos_2d[i] + dist_emb[i]
                attn_out = self.decoder_layers[l](query=global_forward[i], key=key_list[i - 1], query_pos=query_pos, key_pos=key_pos)[0]
                key_list.append(attn_out)
            global_forward = torch.stack(key_list, dim=0)

        local_forward = local_forward.permute(0, 1, 3, 2).contiguous()
        local_forward = local_forward.view(T - 1, B, C, H, W)

        global_forward = global_forward.permute(0, 1, 3, 2).contiguous()
        global_forward = global_forward.view(T - 1, B, C, H, W)

        return local_forward, global_forward
    



class TransformerFlowEncoderSuccessive2All(nn.Module):
    def __init__(self, dim, nhead, num_layers):
        super().__init__()
        self.num_layers = num_layers

        decoder_layer = TransformerFlowLayer(d_model=dim, nhead=nhead)
        self.decoder_layers = _get_clones(decoder_layer, num_layers)

        self.pos_obj_2d = PositionEmbeddingSine2d(num_pos_feats=dim // 2, normalize=True)

        proj2 = nn.Linear(dim, dim)
        self.proj2 = _get_clones(proj2, num_layers)
    
    
    def forward(self, unlabeled, dist_emb):
        '''unlabeled: T, B, C, H, W
        dist_emb: B, T, C'''
        
        shape = unlabeled.shape
        T, B, C, H, W = shape

        dist_emb = dist_emb.permute(1, 0, 2).contiguous()
        dist_emb = dist_emb[:, :, None, :].repeat(1, 1, H * W, 1) # T, B, H * W, C

        unlabeled = unlabeled.permute(0, 1, 3, 4, 2).contiguous() # T, B, H, W, C
        unlabeled = unlabeled.view(T, B, H * W, C)

        pos_2d = self.pos_obj_2d(shape_util=(B, H, W), device=unlabeled.device)
        pos_2d = pos_2d.permute(0, 2, 3, 1).contiguous()
        pos_2d = pos_2d.view(B, H * W, C)
        pos_2d = pos_2d[None].repeat(T, 1, 1, 1)

        for l in range(self.num_layers):

            dist_emb = self.proj2[l](dist_emb)

            key = unlabeled.permute(1, 0, 2, 3).contiguous() # B, T, H*W, C
            key = key.view(B, T * H * W, C)
            key = key[None, :, :, :].repeat(T, 1, 1, 1)
            key = key.view(T * B, T * H * W, C)

            key_pos_2d = pos_2d.permute(1, 0, 2, 3).contiguous()
            key_pos_2d = key_pos_2d.view(B, T * H * W, C)
            key_pos_2d = key_pos_2d[None, :, :, :].repeat(T, 1, 1, 1)
            key_pos_2d = key_pos_2d.view(T * B, T * H * W, C)

            key_pos_1d = dist_emb.permute(1, 0, 2, 3).contiguous()
            key_pos_1d = key_pos_1d.view(B, T * H * W, C)
            key_pos_1d = key_pos_1d[None, :, :, :].repeat(T, 1, 1, 1)
            key_pos_1d = key_pos_1d.view(T * B, T * H * W, C)

            query_pos = pos_2d + dist_emb
            key_pos = key_pos_2d + key_pos_1d
            query_pos = query_pos.view(T * B, H * W, C)

            unlabeled = unlabeled.view(T * B, H * W, C)

            unlabeled, weights = self.decoder_layers[l](query=unlabeled,
                                                    key=key, 
                                                    query_pos=query_pos,
                                                    key_pos=key_pos)
            
            unlabeled = unlabeled.view(T, B, H * W, C)
        
        weights = weights.view(T, B, H * W, T * H * W).mean(2) # T, B, T * H * W
        weights = weights.permute(2, 0, 1).contiguous()
        weights = weights.view(T, H * W, T, B).mean(1)
        weights = weights.mean(2).permute(1, 0).contiguous()

        unlabeled = unlabeled.permute(0, 1, 3, 2).contiguous()
        unlabeled = unlabeled.view(T, B, C, H, W)

        return unlabeled, weights
    




class TransformerFlowEncoderSuccessive2Iterative(nn.Module):
    def __init__(self, dim, nhead, num_layers):
        super().__init__()
        self.num_layers = num_layers

        decoder_layer = TransformerFlowLayer(d_model=dim, nhead=nhead)
        self.decoder_layers = _get_clones(decoder_layer, num_layers)

        self.pos_obj_2d = PositionEmbeddingSine2d(num_pos_feats=dim // 2, normalize=True)

        proj2 = nn.Linear(dim, dim)
        self.proj2 = _get_clones(proj2, num_layers)
    
    
    def forward(self, unlabeled, dist_emb):
        '''unlabeled: T, B, C, H, W
        dist_emb: B, T, C'''
        
        shape = unlabeled.shape
        T, B, C, H, W = shape

        dist_emb = dist_emb.permute(1, 0, 2).contiguous()
        dist_emb = dist_emb[:, :, None, :].repeat(1, 1, H * W, 1) # T, B, H * W, C

        unlabeled = unlabeled.permute(0, 1, 3, 4, 2).contiguous() # T, B, H, W, C
        unlabeled = unlabeled.view(T, B, H * W, C)

        pos_2d = self.pos_obj_2d(shape_util=(B, H, W), device=unlabeled.device)
        pos_2d = pos_2d.permute(0, 2, 3, 1).contiguous()
        pos_2d = pos_2d.view(B, H * W, C)
        pos_2d = pos_2d[None].repeat(T, 1, 1, 1)
        
        for l in range(self.num_layers):

            dist_emb = self.proj2[l](dist_emb)

            key_list = [unlabeled[0]]
            for i in range(1, len(unlabeled)):
                query_pos = pos_2d[i] + dist_emb[i]
                key_pos = pos_2d[i] + dist_emb[i]
                attn_out = self.decoder_layers[l](query=unlabeled[i], key=key_list[i - 1], query_pos=query_pos, key_pos=key_pos)[0]
                key_list.append(attn_out)
            unlabeled = torch.stack(key_list, dim=0)

        unlabeled = unlabeled.permute(0, 1, 3, 2).contiguous()
        unlabeled = unlabeled.view(T, B, C, H, W)

        return unlabeled
    



class TransformerFlowEncoderSuccessive(nn.Module):
    def __init__(self, dim, nhead, num_layers):
        super().__init__()
        self.num_layers = num_layers

        bilateral_attention_layer = TransformerFlowLayer(d_model=dim, nhead=nhead)
        self.bilateral_attention_layers = _get_clones(bilateral_attention_layer, num_layers)

        self.pos_obj_2d = PositionEmbeddingSine2d(num_pos_feats=dim // 2, normalize=True)

        self.mlp = MLP(input_dim=dim, hidden_dim=2048, output_dim=dim, num_layers=2)
    
    
    def forward(self, unlabeled, embedding):
        '''unlabeled: T, B, C, H, W
        embedding: 1 * B, H * W, C'''
        
        shape = unlabeled.shape
        T, B, C, H, W = shape

        unlabeled = unlabeled.permute(0, 1, 3, 4, 2).contiguous() # T, B, H, W, C
        unlabeled = unlabeled.view(T, B, H * W, C)

        pos_2d = self.pos_obj_2d(shape_util=(B, H, W), device=unlabeled.device)
        pos_2d = pos_2d.permute(0, 2, 3, 1).contiguous()
        pos_2d = pos_2d.view(B, H * W, C)
        pos_2d = pos_2d[None].repeat(T - 1, 1, 1, 1)
        pos_2d = pos_2d.view((T-1) * B, H * W, C)

        local_backward = unlabeled[:-1]
        local_forward = unlabeled[1:]

        local_backward = local_backward.view((T-1) * B, H * W, C)
        local_forward = local_forward.view((T-1) * B, H * W, C) 

        for l in range(self.num_layers):

            concat0 = torch.cat([local_forward, local_backward], dim=0)
            concat1 = torch.cat([local_backward, local_forward], dim=0)
            pos0 = torch.cat([pos_2d + embedding, pos_2d + embedding], dim=0)
            pos1 = torch.cat([pos_2d + embedding, pos_2d + embedding], dim=0)
            concat0 = self.bilateral_attention_layers[l](query=concat0, key=concat1, query_pos=pos0, key_pos=pos1)[0]
            local_forward, local_backward = torch.chunk(concat0, chunks=2, dim=0)
        
        embedding = self.mlp(local_forward)

        local_forward = local_forward.view(T - 1, B, H * W, C)
        local_forward = local_forward.permute(0, 1, 3, 2).contiguous()
        local_forward = local_forward.view(T - 1, B, C, H, W)

        return local_forward, embedding
    



class TransformerFlowEncoderSuccessiveNoEmb(nn.Module):
    def __init__(self, dim, nhead, num_layers):
        super().__init__()
        self.num_layers = num_layers

        bilateral_attention_layer = TransformerFlowLayer(d_model=dim, nhead=nhead)
        self.bilateral_attention_layers = _get_clones(bilateral_attention_layer, num_layers)

        self.pos_obj_2d = PositionEmbeddingSine2d(num_pos_feats=dim // 2, normalize=True)
    
    
    def forward(self, unlabeled):
        '''unlabeled: T, B, C, H, W'''
        
        shape = unlabeled.shape
        T, B, C, H, W = shape

        unlabeled = unlabeled.permute(0, 1, 3, 4, 2).contiguous() # T, B, H, W, C
        unlabeled = unlabeled.view(T, B, H * W, C)

        pos_2d = self.pos_obj_2d(shape_util=(B, H, W), device=unlabeled.device)
        pos_2d = pos_2d.permute(0, 2, 3, 1).contiguous()
        pos_2d = pos_2d.view(B, H * W, C)
        pos_2d = pos_2d[None].repeat(T - 1, 1, 1, 1)
        pos_2d = pos_2d.view((T-1) * B, H * W, C)

        local_backward = unlabeled[:-1]
        local_forward = unlabeled[1:]

        local_backward = local_backward.view((T-1) * B, H * W, C)
        local_forward = local_forward.view((T-1) * B, H * W, C) 
        
        for l in range(self.num_layers):

            concat0 = torch.cat([local_forward, local_backward], dim=0)
            concat1 = torch.cat([local_backward, local_forward], dim=0)
            pos = torch.cat([pos_2d, pos_2d], dim=0)
            concat0 = self.bilateral_attention_layers[l](query=concat0, key=concat1, query_pos=pos, key_pos=pos)
            local_forward, local_backward = torch.chunk(concat0, chunks=2, dim=0)

        local_forward = local_forward.view(T - 1, B, H * W, C)
        local_forward = local_forward.permute(0, 1, 3, 2).contiguous()
        local_forward = local_forward.view(T - 1, B, C, H, W)

        return local_forward



class TransformerContext(nn.Module):
    def __init__(self, dim, nhead, num_layers):
        super().__init__()
        self.num_layers = num_layers

        bilateral_attention_layer = TransformerFlowLayer(d_model=dim, nhead=nhead)
        self.bilateral_attention_layers = _get_clones(bilateral_attention_layer, num_layers)

        self.pos_obj_2d = PositionEmbeddingSine2d(num_pos_feats=dim // 2, normalize=True)
    
    
    def forward(self, x1, x2):
        '''x1: B, C, H, W'''
        
        shape = x1.shape
        B, C, H, W = shape

        x1 = x1.permute(0, 2, 3, 1).contiguous() # B, H, W, C
        x1 = x1.view(B, H * W, C)

        x2 = x2.permute(0, 2, 3, 1).contiguous() # B, H, W, C
        x2 = x2.view(B, H * W, C)

        pos_2d = self.pos_obj_2d(shape_util=(B, H, W), device=x1.device)
        pos_2d = pos_2d.permute(0, 2, 3, 1).contiguous()
        pos_2d = pos_2d.view(B, H * W, C)
        
        for l in range(self.num_layers):

            concat0 = torch.cat([x1, x2], dim=0)
            concat1 = torch.cat([x2, x1], dim=0)
            pos = torch.cat([pos_2d, pos_2d], dim=0)
            concat0 = self.bilateral_attention_layers[l](query=concat0, key=concat1, query_pos=pos, key_pos=pos)
            x1, x2 = torch.chunk(concat0, chunks=2, dim=0)

        x1 = x1.permute(0, 1, 2).contiguous()
        x1 = x1.view(B, C, H, W)

        return x1
    


class TopKAttention(nn.Module):
    def __init__(self, inp, oup, topk, heads, dim_head=64, dropout=0.):
        super().__init__()
        self.topk = topk
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == inp)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_q = nn.Linear(inp, inner_dim, bias=False)
        self.to_k = nn.Linear(inp, inner_dim, bias=False)
        self.to_v = nn.Linear(inp, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, oup),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, query, key, value):
        q = self.to_q(query)
        k = self.to_k(key)
        v = self.to_v(value)
        qkv = [q, k, v]
        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        values, indices = torch.topk(dots, k=self.topk, dim=-1)

        values = self.attend(values)
        dots.zero_().scatter_(-1, indices, values) # B * h * HW * HWT

        #matplotlib.use('QtAgg')
        #fig, ax = plt.subplots(1, 1)
        #ax.imshow(dots[0].mean(0).detach().cpu(), cmap='plasma')
        #plt.show()

        out = torch.matmul(dots, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out, dots.mean(1)
    



class MemoryReaderPos(nn.Module):
    def __init__(self, topk=None):
        super().__init__()
        self.topk = topk
    
    def get_similarity(self, mk, ms, qk, qe):
        # used for training/inference and memory reading/memory potentiation
        # mk: B x CK x [N]    - Memory keys
        # ms: B x  1 x [N]    - Memory shrinkage
        # qk: B x CK x [HW/P] - Query keys
        # qe: B x CK x [HW/P] - Query selection
        # Dimensions in [] are flattened
        CK = mk.shape[1]
        mk = mk.flatten(start_dim=2)
        ms = ms.flatten(start_dim=1).unsqueeze(2) if ms is not None else None
        qk = qk.flatten(start_dim=2)
        qe = qe.flatten(start_dim=2) if qe is not None else None

        if qe is not None:
            # See appendix for derivation
            # or you can just trust me ヽ(ー_ー )ノ
            mk = mk.transpose(1, 2)
            a_sq = (mk.pow(2) @ qe)
            two_ab = 2 * (mk @ (qk * qe))
            b_sq = (qe * qk.pow(2)).sum(1, keepdim=True)
            similarity = (-a_sq+two_ab-b_sq)
        else:
            # similar to STCN if we don't have the selection term
            a_sq = mk.pow(2).sum(1).unsqueeze(2)
            two_ab = 2 * (mk.transpose(1, 2) @ qk)
            similarity = (-a_sq+two_ab)

        if ms is not None:
            similarity = similarity * ms / math.sqrt(CK)   # B*N*HW
        else:
            similarity = similarity / math.sqrt(CK)   # B*N*HW

        return similarity

 
    def get_affinity(self, mk, qk, pos):
        B, head, HWT, hidden_dim = mk.shape
        B, head, HW, hidden_dim = qk.shape
        #print(shrink.shape)
        #print(select.shape)

        pos = pos.view(B * head, HWT, hidden_dim)

        mk = mk.view(B * head, HWT, hidden_dim)
        mk = mk.permute(0, 2, 1).contiguous()

        qk = qk.view(B * head, HW, hidden_dim)
        pos_match = torch.matmul(qk, pos.transpose(2, 1)).permute(0, 2, 1).contiguous()
        qk = qk.permute(0, 2, 1).contiguous()

        affinity = self.get_similarity(mk, None, qk, None) # B, N, HW
        affinity = affinity + pos_match
        affinity = affinity.view(B, head, HWT, HW)

        # See supplementary material
        #a_sq = mk.pow(2).sum(-1).unsqueeze(-1)
        #ab = mk @ qk.transpose(2, 3)
#
        #affinity = (2*ab-a_sq) / math.sqrt(hidden_dim)   # B, h, THW, HW

        #matplotlib.use('QtAgg')
        #temp = affinity[0, 0].view(-1, 24, 24, 576).view(-1, 576, 576)
        #fig, ax = plt.subplots(1, temp.shape[0])
        #for i in range(temp.shape[0]):
        #    ax[i].imshow(temp[i].detach().cpu(), cmap='plasma', vmin=affinity[0, 0].min(), vmax=affinity[0, 0].max())
        #plt.show()

        #matplotlib.use('QtAgg')
        #temp = affinity[0, 0]
        #temp = temp.view(-1, 24, 24)[:, 12, 12]
        #temp = temp.view(-1, 24, 24).detach().cpu()
        #fig, ax = plt.subplots(1, temp.shape[0])
        #for i in range(temp.shape[0]):
        #    ax[i].imshow(temp[i], cmap='plasma', vmin=temp.min(), vmax=temp.max())
        #plt.show()

        if self.topk is not None:
            values, indices = torch.topk(affinity, k=self.topk, dim=2)

            # softmax operation; aligned the evaluation style
            maxes = torch.max(values, dim=2, keepdim=True)[0]
            x_exp = torch.exp(values - maxes)
            x_exp_sum = torch.sum(x_exp, dim=2, keepdim=True)
            values = x_exp / x_exp_sum 

            affinity.zero_().scatter_(2, indices, values) # B * h * HW * HWT
        else:
            # softmax operation; aligned the evaluation style
            maxes = torch.max(affinity, dim=2, keepdim=True)[0]
            x_exp = torch.exp(affinity - maxes)
            x_exp_sum = torch.sum(x_exp, dim=2, keepdim=True)
            affinity = x_exp / x_exp_sum 

        #matplotlib.use('QtAgg')
        #temp = affinity[0].mean(0)
        #temp = temp.view(-1, 24, 24)[:, 12, 12]
        #temp = temp.view(-1, 24, 24).detach().cpu()
        #fig, ax = plt.subplots(1, temp.shape[0])
        #for i in range(temp.shape[0]):
        #    ax[i].imshow(temp[i], cmap='hot', vmin=temp.min(), vmax=temp.max())
        #plt.show()
            

        return affinity

    def readout(self, affinity, mv, qv):
        B, CV, T, H, W = mv.shape

        mo = mv.view(B, CV, T*H*W) 
        mem = torch.bmm(mo, affinity) # Weighted-sum B, CV, HW
        mem = mem.view(B, CV, H, W)

        mem_out = torch.cat([mem, qv], dim=1)

        return mem_out
    



class MemoryReader(nn.Module):
    def __init__(self, topk=None):
        super().__init__()
        self.topk = topk
    
    def get_similarity(self, mk, ms, qk, qe):
        # used for training/inference and memory reading/memory potentiation
        # mk: B x CK x [N]    - Memory keys
        # ms: B x  1 x [N]    - Memory shrinkage
        # qk: B x CK x [HW/P] - Query keys
        # qe: B x CK x [HW/P] - Query selection
        # Dimensions in [] are flattened
        CK = mk.shape[1]
        mk = mk.flatten(start_dim=2)
        ms = ms.flatten(start_dim=1).unsqueeze(2) if ms is not None else None
        qk = qk.flatten(start_dim=2)
        qe = qe.flatten(start_dim=2) if qe is not None else None

        if qe is not None:
            # See appendix for derivation
            # or you can just trust me ヽ(ー_ー )ノ
            mk = mk.transpose(1, 2)
            a_sq = (mk.pow(2) @ qe)
            two_ab = 2 * (mk @ (qk * qe))
            b_sq = (qe * qk.pow(2)).sum(1, keepdim=True)
            similarity = (-a_sq+two_ab-b_sq)
        else:
            # similar to STCN if we don't have the selection term
            a_sq = mk.pow(2).sum(1).unsqueeze(2)
            two_ab = 2 * (mk.transpose(1, 2) @ qk)
            similarity = (-a_sq+two_ab)

        if ms is not None:
            similarity = similarity * ms / math.sqrt(CK)   # B*N*HW
        else:
            similarity = similarity / math.sqrt(CK)   # B*N*HW

        return similarity

 
    def get_affinity(self, mk, qk, pos=None):
        B, head, HWT, hidden_dim = mk.shape
        B, head, HW, hidden_dim = qk.shape
        #print(shrink.shape)
        #print(select.shape)

        mk = mk.view(B * head, HWT, hidden_dim)
        mk = mk.permute(0, 2, 1).contiguous()

        qk = qk.view(B * head, HW, hidden_dim)

        if pos is not None:
            pos = pos.view(B * head, HWT, hidden_dim)
            pos_match = torch.matmul(qk, pos.transpose(2, 1)).permute(0, 2, 1).contiguous()
        else:
            pos_match = torch.zeros(size=(B * head, HWT, HW), device=mk.device)

        qk = qk.permute(0, 2, 1).contiguous()

        affinity = self.get_similarity(mk, None, qk, None)
        affinity = affinity + pos_match
        affinity = affinity.view(B, head, HWT, HW)

        # See supplementary material
        #a_sq = mk.pow(2).sum(-1).unsqueeze(-1)
        #ab = mk @ qk.transpose(2, 3)
#
        #affinity = (2*ab-a_sq) / math.sqrt(hidden_dim)   # B, h, THW, HW

        #matplotlib.use('QtAgg')
        #temp = affinity[0, 0].view(-1, 24, 24, 576).view(-1, 576, 576)
        #fig, ax = plt.subplots(1, temp.shape[0])
        #for i in range(temp.shape[0]):
        #    ax[i].imshow(temp[i].detach().cpu(), cmap='plasma', vmin=affinity[0, 0].min(), vmax=affinity[0, 0].max())
        #plt.show()

        #matplotlib.use('QtAgg')
        #temp = affinity[0, 0]
        #temp = temp.view(-1, 24, 24)[:, 12, 12]
        #temp = temp.view(-1, 24, 24).detach().cpu()
        #fig, ax = plt.subplots(1, temp.shape[0])
        #for i in range(temp.shape[0]):
        #    ax[i].imshow(temp[i], cmap='plasma', vmin=temp.min(), vmax=temp.max())
        #plt.show()

        if self.topk is not None:
            values, indices = torch.topk(affinity, k=self.topk, dim=2)

            # softmax operation; aligned the evaluation style
            maxes = torch.max(values, dim=2, keepdim=True)[0]
            x_exp = torch.exp(values - maxes)
            x_exp_sum = torch.sum(x_exp, dim=2, keepdim=True)
            values = x_exp / x_exp_sum 

            affinity.zero_().scatter_(2, indices, values) # B * h * HW * HWT
        else:
            # softmax operation; aligned the evaluation style
            maxes = torch.max(affinity, dim=2, keepdim=True)[0]
            x_exp = torch.exp(affinity - maxes)
            x_exp_sum = torch.sum(x_exp, dim=2, keepdim=True)
            affinity = x_exp / x_exp_sum 

        #matplotlib.use('QtAgg')
        #temp = affinity[0].mean(0)
        #temp = temp.view(-1, 24, 24)[:, 12, 12]
        #temp = temp.view(-1, 24, 24).detach().cpu()
        #fig, ax = plt.subplots(1, temp.shape[0])
        #for i in range(temp.shape[0]):
        #    ax[i].imshow(temp[i], cmap='hot', vmin=temp.min(), vmax=temp.max())
        #plt.show()
            

        return affinity

    def readout(self, affinity, mv, qv):
        B, CV, T, H, W = mv.shape

        mo = mv.view(B, CV, T*H*W) 
        mem = torch.bmm(mo, affinity) # Weighted-sum B, CV, HW
        mem = mem.view(B, CV, H, W)

        mem_out = torch.cat([mem, qv], dim=1)

        return mem_out

    


class L2Attention(nn.Module):
    def __init__(self, dim, pos_1d, nb_heads, topk=None, dropout=0., gaussian_type='query'):
        super().__init__()

        self.memory = MemoryReader(topk)
        self.pos_1d = pos_1d
        self.gaussian_type = gaussian_type

        if self.pos_1d == 'learnable_sin':
            self.pos_proj = nn.Linear(dim, dim, bias=False)

        self.heads = nb_heads

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )


    
    def make_gaussian_query(self, y_idx, x_idx, shape, sigma=7):
        T, B, C, H, W = shape

        y_idx = y_idx.view(B, T, H * W)
        x_idx = x_idx.view(B, T, H * W)

        y_idx = y_idx.permute(1, 0, 2).contiguous() # T, B, HW
        x_idx = x_idx.permute(1, 0, 2).contiguous() # T, B, HW

        kernel_sizes = torch.arange(T) * (-4/T) + 7

        gauss_kernel_list = []
        for t, k in enumerate(kernel_sizes):
            current_y_idx = y_idx[t] #B, HW
            current_x_idx = x_idx[t] #B, HW

            current_y_idx = current_y_idx.view(B, H*W, 1, 1).float()
            current_x_idx = current_x_idx.view(B, H*W, 1, 1).float()

            current_y = np.linspace(0,H-1,H)
            current_y = torch.FloatTensor(current_y).to('cuda:0')
            current_y = current_y.view(1,1,H,1).expand(B, H*W, H, 1)

            current_x = np.linspace(0,W-1,W)
            current_x = torch.FloatTensor(current_x).to('cuda:0')
            current_x = current_x.view(1,1,1,W).expand(B, H*W, 1, W)

            gauss_kernel = torch.exp(-((current_x-current_x_idx)**2 + (current_y-current_y_idx)**2) / (2 * (k)**2))
            gauss_kernel = gauss_kernel.view(B, H*W, H*W)
            gauss_kernel_list.append(gauss_kernel)

        gauss_kernel = torch.stack(gauss_kernel_list, dim=1)
        gauss_kernel = gauss_kernel.view(B, T*H*W, H*W)

        
        #y_idx = y_idx.view(B, T*H*W, 1, 1).float()
        #x_idx = x_idx.view(B, T*H*W, 1, 1).float()
#
        #y = np.linspace(0,H-1,H)
        #y = torch.FloatTensor(y).to('cuda:0')
        #y = y.view(1,1,H,1).expand(B, T*H*W, H, 1)
#
        #x = np.linspace(0,W-1,W)
        #x = torch.FloatTensor(x).to('cuda:0')
        #x = x.view(1,1,1,W).expand(B, T*H*W, 1, W)
        #        
        #gauss_kernel = torch.exp(-((x-x_idx)**2 + (y-y_idx)**2) / (2 * (sigma)**2))
        #gauss_kernel = gauss_kernel.view(B, T*H*W, H*W)

        return gauss_kernel
    

    
    def make_gaussian_memory(self, y_idx, x_idx, shape, sigma=7):
        T, B, C, H, W = shape

        y_idx = y_idx.view(B, T, H * W)
        x_idx = x_idx.view(B, T, H * W)

        y_idx = y_idx.permute(1, 0, 2).contiguous() # T, B, HW
        x_idx = x_idx.permute(1, 0, 2).contiguous() # T, B, HW

        kernel_sizes = torch.arange(T) * (-4/T) + 7

        gauss_kernel_list = []
        for t, k in enumerate(kernel_sizes):
            current_y_idx = y_idx[t] #B, HW
            current_x_idx = x_idx[t] #B, HW

            current_y_idx = current_y_idx.view(B, 1, 1, H*W).float()
            current_x_idx = current_x_idx.view(B, 1, 1, H*W).float()

            current_y = np.linspace(0,H-1,H)
            current_y = torch.FloatTensor(current_y).to('cuda:0')
            current_y = current_y.view(1,H,1,1).expand(B, H, 1, H*W)

            current_x = np.linspace(0,W-1,W)
            current_x = torch.FloatTensor(current_x).to('cuda:0')
            current_x = current_x.view(1,1,W,1).expand(B, 1, W, H*W)

            gauss_kernel = torch.exp(-((current_x-current_x_idx)**2 + (current_y-current_y_idx)**2) / (2 * (k)**2))
            gauss_kernel = gauss_kernel.view(B, H*W, H*W)
            gauss_kernel_list.append(gauss_kernel)

        gauss_kernel = torch.stack(gauss_kernel_list, dim=1)
        gauss_kernel = gauss_kernel.view(B, T*H*W, H*W)

        
        #y_idx = y_idx.view(B, T*H*W, 1, 1).float()
        #x_idx = x_idx.view(B, T*H*W, 1, 1).float()
#
        #y = np.linspace(0,H-1,H)
        #y = torch.FloatTensor(y).to('cuda:0')
        #y = y.view(1,1,H,1).expand(B, T*H*W, H, 1)
#
        #x = np.linspace(0,W-1,W)
        #x = torch.FloatTensor(x).to('cuda:0')
        #x = x.view(1,1,1,W).expand(B, T*H*W, 1, W)
        #        
        #gauss_kernel = torch.exp(-((x-x_idx)**2 + (y-y_idx)**2) / (2 * (sigma)**2))
        #gauss_kernel = gauss_kernel.view(B, T*H*W, H*W)

        return gauss_kernel
    


    def get_gaussians(self, argmax_idx, shape):
        T, B, C, H, W = shape
        y_idx, x_idx = argmax_idx[:, :, 0], argmax_idx[:, :, 1]
        #y_idx, x_idx = argmax_idx//W, argmax_idx%W
        if self.gaussian_type == 'memory':
            g = self.make_gaussian_memory(y_idx, x_idx, shape)
        elif self.gaussian_type == 'query':
            g = self.make_gaussian_query(y_idx, x_idx, shape)
        
        #matplotlib.use('QtAgg')
        #fig, ax = plt.subplots(1, 2)
        #temp = g.view(B, T, H * W, H * W)
        #temp = temp.view(B, T, H, W, H, W)
        #ax[0].imshow(temp[0, 0, :, :, 0, 0].view(H, W).detach().cpu(), cmap='gray')
        #ax[1].imshow(temp[0, -1, :, :, 0, 0].view(H, W).detach().cpu(), cmap='gray')
        #plt.show()
    
        return g
    


    def forward(self, query, key, value, shape, pos=None, max_idx=None):
        T, B, C, H, W = shape

        if pos is not None:
            pos = self.pos_proj(pos)
            pos = rearrange(pos, 'b n (h d) -> b h n d', h=self.heads).contiguous()

        q = self.to_q(query)
        k = self.to_k(key)
        v = self.to_v(value)
        qkv = [q, k, v]
        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> b h n d', h=self.heads).contiguous(), qkv)

        affinity = self.memory.get_affinity(k, q, pos=pos) # B, head, THW, HW
        
        if max_idx is not None:
            gaussian = self.get_gaussians(max_idx, shape)
            affinity = affinity * gaussian[:, None, :, :]

        #matplotlib.use('QtAgg')
        #temp = affinity[0].mean(0).view(T, H, W, H * W).view(T, H, W, H, W)
        #temp = temp[:, :, :, H//2, W//2]
        #fig, ax = plt.subplots(1, temp.shape[0])
        #for i in range(temp.shape[0]):
        #    if temp.shape[0] == 1:
        #        ax.imshow(temp[i].detach().cpu(), cmap='hot', vmin=temp.min(), vmax=temp.max())
        #    else:
        #        ax[i].imshow(temp[i].detach().cpu(), cmap='hot', vmin=temp.min(), vmax=temp.max())
        #plt.show()

        out = torch.matmul(affinity.transpose(3, 2), v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out, affinity.mean(1)
    


class L2AttentionPos(nn.Module):
    def __init__(self, dim, topk=None, dropout=0.):
        super().__init__()

        self.memory = MemoryReaderPos(topk)

        self.pos_proj = nn.Linear(dim, dim)

        self.heads = 8

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, query, key, value, pos):

        pos = self.pos_proj(pos)
        q = self.to_q(query)
        k = self.to_k(key)
        v = self.to_v(value)
        qkvp = [q, k, v, pos]
        q, k, v, pos = map(lambda t: rearrange(
            t, 'b n (h d) -> b h n d', h=self.heads).contiguous(), qkvp)

        affinity = self.memory.get_affinity(k, q, pos=pos) # B, head, THW, HW

        #matplotlib.use('QtAgg')
        #temp = affinity[0, 0].view(-1, 24, 24, 576).view(-1, 576, 576)
        #fig, ax = plt.subplots(1, temp.shape[0])
        #for i in range(temp.shape[0]):
        #    ax[i].imshow(temp[i].detach().cpu(), cmap='plasma')
        #plt.show()

        out = torch.matmul(affinity.transpose(3, 2), v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out, affinity.mean(1)



class deformableAttention(nn.Module):
    def __init__(self, dim, nhead, memory_length, points=4):
        super().__init__()

        self.heads = nhead
        self.points = points
        self.M = memory_length - 1

        self.tanh = nn.Tanh()

        self.to_q = nn.Linear(dim, dim)
        self.offsets = nn.Linear(dim, nhead * points * 2)

        self.pos_1d = nn.Parameter(torch.randn(self.M, dim))
        
        self.attention_weights = nn.Linear(dim, nhead * points)

        self.to_v = nn.Conv3d(in_channels=dim, out_channels=dim, kernel_size=1)
        self.output_proj = nn.Linear(dim, dim)

        self._reset_parameters()

    
    def _reset_parameters(self):
        constant_(self.offsets.weight.data, 0.)

        thetas = torch.arange(self.heads, dtype=torch.float32) * (2.0 * math.pi / self.heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.heads, 1, 1, 2).repeat(1, 1, self.points, 1)
        for i in range(self.points):
            grid_init[:, :, i, :] *= 0.01*i

        with torch.no_grad():
            self.offsets.bias = nn.Parameter(grid_init.view(-1))

        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.to_v.weight.data)
        xavier_uniform_(self.to_q.weight.data)
        constant_(self.to_q.bias.data, 0.)
        constant_(self.to_v.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    
    def get_grid(self, value):

        B, C, T, H, W = value.shape

        # Generate coordinates along each axis
        y = torch.linspace(-1, 1, H, device=value.device)
        x = torch.linspace(-1, 1, W, device=value.device)

        # Create the meshgrid
        grid_y, grid_x = torch.meshgrid(y, x)
        ref = torch.stack((grid_y, grid_x), -1)[None].repeat(B, 1, 1, 1, 1)
        return ref
    

    def forward(self, query, value, video_length):
        """query: B, HW, C,
        reference_points: B, HW, 2,
        value: B, C, T, H, W"""

        B, C, T, H, W = value.shape
        B, L, C = query.shape

        reference_points = self.get_grid(value)
        reference_points = reference_points.view(B, H*W, 2) # B, L, 2

        #pos_seq = torch.arange(T-1, -1, -1.0, device=query.device)
        #pos_1d = self.pos_obj_1d(pos_seq=pos_seq) # M, C

        value = self.to_v(value)
        value = value.view(B, self.heads, C // self.heads, T, H, W)
        value = value.view(B * self.heads, C // self.heads, T, H, W)

        pos_1d = self.pos_1d[None].repeat(B, 1, 1) # B, M, C
        pos_1d = pos_1d.permute(0, 2, 1).contiguous() # B, C, M

        if not self.training:
            pos_1d = torch.nn.functional.interpolate(pos_1d, size=video_length, mode='linear')

        sampling_location_list = []
        attention_weights_list = []
        sample_list = []
        offset_list = []
        for t in range(T):

            current_query = query + pos_1d[:, :, t][:, None, :]
            current_query = self.to_q(current_query)

            offsets = self.tanh(self.offsets(current_query)) # B, H*W, nh * P * 2
            offsets = offsets.view(B, H*W, self.heads, self.points, 2)
            offset_list.append(offsets)

            attention_weights = self.attention_weights(current_query) # B, H*W, nh * P
            attention_weights = attention_weights.view(B, H*W, self.heads, self.points)
            attention_weights = attention_weights.permute(0, 2, 1, 3).contiguous() # B, nh, H*W, P
            attention_weights = attention_weights.view(B * self.heads, H*W, self.points)[:, None]
            attention_weights_list.append(attention_weights)

            sampling_locations = offsets + reference_points[:, :, None, None, :] # B, H*W, nh, P, 2
            sampling_locations = sampling_locations.permute(0, 2, 1, 3, 4).contiguous() # B, nh, H*W, P, 2
            sampling_location_list.append(sampling_locations)
            sampling_locations = sampling_locations.view(B * self.heads, H*W, self.points, 2)

            samples = F.grid_sample(value[:, :, t], sampling_locations, mode='bilinear', padding_mode='zeros', align_corners=True).squeeze(2) # B*nh, hidden_dim, H*W, P
            sample_list.append(samples)

        samples = torch.stack(sample_list, dim=3) # B*nh, hidden_dim, H*W, M, P
        samples = samples.view(B * self.heads, C // self.heads, H*W, T * self.points)

        attention_weights = torch.stack(attention_weights_list, dim=3) # B*nh, 1, H*W, M, P
        attention_weights = attention_weights.view(B * self.heads, 1, H*W, T * self.points)
        attention_weights = F.softmax(attention_weights, dim=-1)

        out = (samples * attention_weights).sum(-1) # B*nh, hidden_dim, H*W
        out = out.view(B, self.heads, C // self.heads, H*W)
        out = out.view(B, C, H*W)
        out = out.permute(0, 2, 1).contiguous() # B, L, C

        out = self.output_proj(out)

        offsets = torch.stack(offset_list, dim=0)

        attention_weights = attention_weights.view(B, self.heads, 1, H*W, T*self.points).squeeze(2)
        attention_weights = attention_weights.view(B, self.heads, H*W, T, self.points)
        attention_weights = attention_weights.permute(3, 0, 2, 1, 4).contiguous() # M, B, HW, nh, P
        attention_weights = attention_weights.view(T, B, H, W, self.heads, self.points)

        #if not self.training:

        sampling_locations = torch.stack(sampling_location_list, dim=0) # M, B, nh, H*W, P, 2
        sampling_locations = sampling_locations.permute(0, 1, 3, 2, 4, 5).contiguous() # M, B, H*W, nh, P, 2
        sampling_locations = sampling_locations.view(T, B, H, W, self.heads, self.points, 2)

        return out, sampling_locations, attention_weights, offsets
    



class deformableAttention6(nn.Module):
    def __init__(self, dim, nhead, memory_length, add_motion_cues, points=4):
        super().__init__()

        self.heads = nhead
        self.points = points
        self.M = memory_length - 1
        self.add_motion_cues = add_motion_cues

        self.tanh = nn.Tanh()

        self.to_q = nn.Linear(dim, dim)
        self.offsets = nn.Linear(dim, nhead * points * 2)

        self.pos_1d = nn.Parameter(torch.randn(self.M, dim))
        
        self.attention_weights = nn.Linear(dim, nhead * points)

        self.to_v = nn.Conv3d(in_channels=dim, out_channels=dim, kernel_size=1)
        if add_motion_cues:
            self.to_k = nn.Conv3d(in_channels=dim, out_channels=dim, kernel_size=1)
        self.output_proj = nn.Linear(dim, dim)

        self._reset_parameters()

    
    def _reset_parameters(self):
        constant_(self.offsets.weight.data, 0.)

        thetas = torch.arange(self.heads, dtype=torch.float32) * (2.0 * math.pi / self.heads)
        
        #thetas *= 0.001  # Adjust the scaling factor as needed
        
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.heads, 1, 1, 2).repeat(1, 1, self.points, 1)
        for i in range(self.points):
            grid_init[:, :, i, :] *= 0.01*i
            #grid_init[:, :, i, :] *= i + 1

        with torch.no_grad():
            self.offsets.bias = nn.Parameter(grid_init.view(-1))

        #matplotlib.use('QtAgg')
        #plt.scatter(grid_init[0, 0, :, 0], grid_init[0, 0, :, 1])
        #plt.show()

        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.to_v.weight.data)
        xavier_uniform_(self.to_q.weight.data)
        constant_(self.to_q.bias.data, 0.)
        constant_(self.to_v.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

        if self.add_motion_cues:
            xavier_uniform_(self.to_k.weight.data)
            constant_(self.to_k.bias.data, 0.)

    
    def get_grid(self, value):

        B, C, T, H, W = value.shape

        # Generate coordinates along each axis
        y = torch.linspace(-1, 1, H, device=value.device)
        x = torch.linspace(-1, 1, W, device=value.device)

        # Create the meshgrid
        grid_y, grid_x = torch.meshgrid(y, x)
        ref = torch.stack((grid_y, grid_x), -1)[None].repeat(B, 1, 1, 1, 1)
        return ref
    

    def forward(self, query, value, key, video_length):
        """query: B, HW, C,
        reference_points: B, HW, 2,
        key: B, C, T, H, W,
        value: B, C, T, H, W"""

        B, C, T, H, W = value.shape
        B, L, C = query.shape

        reference_points = self.get_grid(value)
        reference_points = reference_points.view(B, H*W, 2) # B, L, 2

        #pos_seq = torch.arange(T-1, -1, -1.0, device=query.device)
        #pos_1d = self.pos_obj_1d(pos_seq=pos_seq) # M, C

        value = self.to_v(value)
        value = value.view(B, self.heads, C // self.heads, T, H, W)
        value = value.view(B * self.heads, C // self.heads, T, H, W)

        if self.add_motion_cues:
            key = self.to_k(key)
            key = key.permute(2, 0, 1, 3, 4).contiguous()
            key = key.view(T, B, C, H*W).contiguous()
            key = key.permute(0, 1, 3, 2).contiguous() # T, B, H*W, C

        pos_1d = self.pos_1d[None].repeat(B, 1, 1) # B, M, C
        pos_1d = pos_1d.permute(0, 2, 1).contiguous() # B, C, M

        if not self.training:
            pos_1d = torch.nn.functional.interpolate(pos_1d, size=video_length, mode='linear')

        sampling_location_list = []
        attention_weights_list = []
        sample_list = []
        offset_list = []
        for t in range(T):

            current_query = query + pos_1d[:, :, t][:, None, :]
            current_query = self.to_q(current_query)

            if self.add_motion_cues:
                offsets = self.tanh(self.offsets(current_query - key[t])) # B, H*W, nh * P * 2
            else:
                offsets = self.tanh(self.offsets(current_query)) # B, H*W, nh * P * 2
            offsets = offsets.view(B, H*W, self.heads, self.points, 2)
            offset_list.append(offsets)

            attention_weights = self.attention_weights(current_query) # B, H*W, nh * P
            attention_weights = attention_weights.view(B, H*W, self.heads, self.points)
            attention_weights = attention_weights.permute(0, 2, 1, 3).contiguous() # B, nh, H*W, P
            attention_weights = attention_weights.view(B * self.heads, H*W, self.points)[:, None]
            attention_weights_list.append(attention_weights)

            sampling_locations = offsets + reference_points[:, :, None, None, :] # B, H*W, nh, P, 2
            sampling_locations = sampling_locations.permute(0, 2, 1, 3, 4).contiguous() # B, nh, H*W, P, 2
            sampling_location_list.append(sampling_locations)
            sampling_locations = sampling_locations.view(B * self.heads, H*W, self.points, 2)

            samples = F.grid_sample(value[:, :, t], sampling_locations, mode='bilinear', padding_mode='zeros', align_corners=True).squeeze(2) # B*nh, hidden_dim, H*W, P
            sample_list.append(samples)

        samples = torch.stack(sample_list, dim=3) # B*nh, hidden_dim, H*W, M, P
        samples = samples.view(B * self.heads, C // self.heads, H*W, T * self.points)

        attention_weights = torch.stack(attention_weights_list, dim=3) # B*nh, 1, H*W, M, P
        attention_weights = attention_weights.view(B * self.heads, 1, H*W, T * self.points)
        attention_weights = F.softmax(attention_weights, dim=-1)

        out = (samples * attention_weights).sum(-1) # B*nh, hidden_dim, H*W
        out = out.view(B, self.heads, C // self.heads, H*W)
        out = out.view(B, C, H*W)
        out = out.permute(0, 2, 1).contiguous() # B, L, C

        out = self.output_proj(out)

        offsets = torch.stack(offset_list, dim=0)

        attention_weights = attention_weights.view(B, self.heads, 1, H*W, T*self.points).squeeze(2)
        attention_weights = attention_weights.view(B, self.heads, H*W, T, self.points)
        attention_weights = attention_weights.permute(3, 0, 2, 1, 4).contiguous() # M, B, HW, nh, P
        attention_weights = attention_weights.view(T, B, H, W, self.heads, self.points)

        #if not self.training:

        sampling_locations = torch.stack(sampling_location_list, dim=0) # M, B, nh, H*W, P, 2
        sampling_locations = sampling_locations.permute(0, 1, 3, 2, 4, 5).contiguous() # M, B, H*W, nh, P, 2
        sampling_locations = sampling_locations.view(T, B, H, W, self.heads, self.points, 2)

        return out, sampling_locations, attention_weights, offsets
    

    


class deformableAttention4Cross(nn.Module):
    def __init__(self, dim, nhead, memory_length, points=4):
        super().__init__()

        self.heads = nhead
        self.points = points
        self.M = memory_length - 1
        self.level = 3

        self.tanh = nn.Tanh()

        self.to_q = nn.Linear(dim, dim)
        self.offsets = nn.Linear(dim, self.level * nhead * points * 2)

        self.pos_1d = nn.Parameter(torch.randn(self.M, dim))
        
        self.attention_weights = nn.Linear(dim, self.level * nhead * points)

        self.to_v = nn.Linear(in_features=dim, out_features=dim)
        self.output_proj = nn.Linear(dim, dim)

        self._reset_parameters()

    
    def _reset_parameters(self):
        constant_(self.offsets.weight.data, 0.)
        constant_(self.offsets.bias.data, 0.)

        #thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        #grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        #grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        #for i in range(self.n_points):
        #    grid_init[:, :, i, :] *= i + 1
        #with torch.no_grad():
        #    self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))

        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.to_v.weight.data)
        xavier_uniform_(self.to_q.weight.data)
        constant_(self.to_q.bias.data, 0.)
        constant_(self.to_v.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)
    

    def forward(self, query, reference_points, value, split_value):
        """query: B, 4*H*W, C,
        reference_points: B, 4*HW, 2
        value: B, 4*T*H*W, C"""

        B, L, C = query.shape
        value = self.to_v(value)

        value_list = []

        value_list = list(torch.split(value, [torch.prod(x) for x in split_value], dim=1))
        for idx, current_value in enumerate(value_list):
            current_value = current_value.permute(0, 2, 1).contiguous() # B, C, G
            current_value = current_value.view((B, C,) + tuple(split_value[idx].tolist()))
            _, _, T, H, W = current_value.shape
            current_value = current_value.view(B, self.heads, C // self.heads, T, H, W)
            current_value = current_value.view(B * self.heads, C // self.heads, T, H, W)
            value_list[idx] = current_value

        pos_1d = self.pos_1d[None].repeat(B, 1, 1) # B, M, C
        pos_1d = pos_1d.permute(0, 2, 1).contiguous() # B, C, M

        if T > self.M:
            pos_1d = torch.nn.functional.interpolate(pos_1d, size=T, mode='linear')

        sampling_location_list = []
        attention_weights_list = []
        sample_list = []
        offset_list = []
        for t in range(T):
            current_query = query + pos_1d[:, :, t][:, None, :]
            current_query = self.to_q(current_query)

            offsets = self.tanh(self.offsets(current_query)) # B, L, l * nh * P * 2
            offsets = offsets.view(B, L, self.level, self.heads, self.points, 2)
            offset_list.append(offsets)

            attention_weights = self.attention_weights(current_query) # B, L, l * nh * P
            attention_weights = attention_weights.view(B, L, self.level, self.heads, self.points)
            attention_weights = attention_weights.permute(0, 3, 1, 2, 4).contiguous() # B, nh, L, l, P
            attention_weights = attention_weights.view(B * self.heads, L, self.level, self.points)
            attention_weights = attention_weights.view(B * self.heads, L, self.level * self.points)
            attention_weights_list.append(attention_weights)

            sampling_locations = offsets + reference_points[:, :, None, None, None, :] # B, L, l, nh, P, 2
            sampling_locations = sampling_locations.permute(0, 3, 1, 2, 4, 5).contiguous() # B, nh, L, l, P, 2
            sampling_locations = sampling_locations.view(B * self.heads, L, self.level, self.points, 2)
            sampling_location_list.append(sampling_locations)

            sample_level_list = []
            for l in range(self.level):
                samples = F.grid_sample(value_list[l][:, :, t], sampling_locations[:, :, l, :, :], mode='bilinear', padding_mode='zeros', align_corners=True) # B*nh, hidden_dim, L, P
                sample_level_list.append(samples)
            
            samples = torch.stack(sample_level_list, dim=3) # B*nh, hidden_dim, L, l, P
            sample_list.append(samples)
        
        samples = torch.stack(sample_list, dim=3) # B*nh, hidden_dim, L, T, l, P
        samples = samples.view(B * self.heads, C // self.heads, L, T * self.level * self.points)

        attention_weights = torch.stack(attention_weights_list, dim=2) # B*nh, L, T, l*P
        attention_weights = attention_weights.view(B * self.heads, L, T * self.level * self.points)
        attention_weights = F.softmax(attention_weights, dim=-1)

        out = (samples * attention_weights[:, None]).sum(-1) # B*nh, hidden_dim, L
        out = out.view(B, self.heads, C // self.heads, L)
        out = out.view(B, C, L)
        out = out.permute(0, 2, 1).contiguous() # B, L, C

        out = self.output_proj(out)

        offsets = torch.stack(offset_list, dim=0)

        attention_weights = attention_weights.view(B, self.heads, L, T, self.level, self.points)
        attention_weights = attention_weights.permute(3, 0, 2, 1, 4, 5).contiguous() # T, B, L, nh, l, P
        attention_weights = attention_weights.view(T, B, L, self.heads * self.level * self.points)

        sampling_locations = torch.stack(sampling_location_list, dim=0) # T, b*nh, L, l * P, 2
        sampling_locations = sampling_locations.view(T, B, self.heads, L, self.level, self.points, 2)
        sampling_locations = sampling_locations.permute(0, 1, 3, 2, 4, 5, 6).contiguous() # T, B, L, nh, l, P, 2
        sampling_locations = sampling_locations.view(T, B, L, self.heads * self.level * self.points, 2)

        return out, sampling_locations, attention_weights, offsets


class AttentionSelect(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.):
        super().__init__()

        self.pos_obj_2d = PositionEmbeddingSine2d(num_pos_feats=dim // 2, normalize=True)
        self.pos_obj_1d = PositionEmbeddingSine1d(num_pos_feats=dim, normalize=True)

        self.heads = heads

        self.to_q = nn.Conv2d(dim, dim, kernel_size=1)
        self.to_k = nn.Conv2d(dim, dim, kernel_size=1)
        self.to_v = nn.Conv2d(dim, dim, kernel_size=1)

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

    def select_windows(self, x):
        T, B, C, H, W = x.shape

        window_size_list = [val for val in range(5, 100, 2) for _ in range(2 * val)]
        window_size_list = window_size_list[:len(x)]
        data_list = []
        for t in range(len(x)):
            window_size = window_size_list[t]
            pad_value = window_size // 2
            current_x = torch.nn.functional.pad(x[t], pad=(pad_value, pad_value, pad_value, pad_value))
            current_windows = current_x.unfold(2, window_size, 1).unfold(3, window_size, 1)
            current_windows = current_windows.permute(0, 1, 4, 5, 2, 3).contiguous()
            current_windows = current_windows.view(B, C, window_size * window_size, H * W)
            data_list.append(current_windows)

        out = torch.cat(data_list, dim=2)
        return out

    def forward(self, query, key, value):
        T, B, C, H, W = key.shape

        dim_head = C // self.heads

        key = key.view(T * B, C, H, W)
        value = value.view(T * B, C, H, W)

        q = self.to_q(query)
        k = self.to_k(key)
        v = self.to_v(value)

        k = k.view(T, B, C, H, W)
        v = v.view(T, B, C, H, W)

        # Scale
        q = q / (dim_head**0.5)

        pos_1d = self.pos_obj_1d(shape_util=(B, T), device=q.device)
        pos_1d = pos_1d[:, :, :, None, None].repeat(1, 1, 1, H, W)
        pos_1d = pos_1d.permute(2, 0, 1, 3, 4).contiguous()

        pos_2d = self.pos_obj_2d(shape_util=(B, H, W), device=q.device)
        pos_2d_key = pos_2d[None].repeat(T, 1, 1, 1, 1)

        #pos_2d_key = pos_2d_key
        pos_2d_key = pos_2d_key + pos_1d

        q = q + pos_2d
        k = k + pos_2d_key

        k = self.select_windows(k) # B, C, w*w, H * W
        v = self.select_windows(v) # B, C, w*w, H * W
        ww = k.shape[2]

        q = q.view(B, self.heads, dim_head, H * W)
        q = q.view(B * self.heads, dim_head, H * W)

        k = k.view(B, self.heads, dim_head, ww, H * W)
        k = k.view(B * self.heads, dim_head, ww, H * W)

        v = v.view(B, self.heads, dim_head, ww, H * W)

        qk = (q.unsqueeze(2) * k).sum(dim=1) #B*h, w*w, H*W
        qk = qk.view(B, self.heads, ww, H * W)

        attn = torch.softmax(qk, dim=2)

        out = (attn.unsqueeze(2) * v).sum(dim=3) # B, h, dim_head, H*W
        out = out.view(B, C, H * W)
        out = out.permute(0, 2, 1).contiguous()

        out = self.to_out(out)
        return out
    


class SpatioTemporalTransformer(nn.Module):
    def __init__(self, dim, nhead, num_layers, topk, pos_1d):
        super().__init__()
        self.num_layers = num_layers
        self.pos_1d = pos_1d
        self.topk = topk
        self.dim = dim

        if topk:
            bilateral_attention_layer = TransformerFlowLayerTopK(d_model=dim, nhead=nhead)
        else:
            bilateral_attention_layer = TransformerFlowLayer(d_model=dim, nhead=nhead)

        self.bilateral_attention_layers = _get_clones(bilateral_attention_layer, num_layers)

        self.pos_obj_2d = PositionEmbeddingSine2d(num_pos_feats=dim // 2, normalize=True)

        if pos_1d:
            self.pos_obj_1d = PositionEmbeddingSine1d(num_pos_feats=dim, normalize=True)
    
    
    def forward(self, x1, x2):
        '''x1: B, C, H, W,
        x2: T, B, C, H, W'''
        
        T, B, C, H, W = x2.shape

        if self.pos_1d:
            pos_1d = self.pos_obj_1d(shape_util=(B, T), device=x1.device)

            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(1, 1)
            #cos = torch.nn.CosineSimilarity(dim=1)
            #out_cos = cos(pos_1d.permute(2, 1, 0), pos_1d)
            #ax.imshow(out_cos.cpu())
            #plt.show()

            pos_1d = pos_1d[:, :, :, None, None].repeat(1, 1, 1, H, W)
            pos_1d = pos_1d.view(B, C, T * H * W)
            pos_1d = pos_1d.permute(0, 2, 1).contiguous()
        else:
            pos_1d = torch.zeros(size=(B, T * H * W, C), device=x1.device)

        x1 = x1.permute(0, 2, 3, 1).contiguous() # B, H, W, C
        x1 = x1.view(B, H * W, C)

        x2 = x2.permute(0, 3, 4, 1, 2).contiguous()
        x2 = x2.view(T * H * W, B, C)
        x2 = x2.permute(1, 0, 2).contiguous() # B, T * H * W, C

        pos_2d = self.pos_obj_2d(shape_util=(B, H, W), device=x1.device)
        pos_2d = pos_2d.permute(0, 2, 3, 1).contiguous()
        pos_2d = pos_2d.view(B, H * W, C)

        pos_2d_key = pos_2d[:, None, :, :].repeat(1, T, 1, 1)
        pos_2d_key = pos_2d_key.view(B, T * H * W, C)

        #pos_2d_key = pos_2d_key
        pos_2d_key = pos_2d_key + pos_1d
        
        for l in range(self.num_layers):
            x1 = self.bilateral_attention_layers[l](query=x1, key=x2, query_pos=pos_2d, key_pos=pos_2d_key)

        x1 = x1.permute(0, 1, 2).contiguous()
        x1 = x1.view(B, C, H, W)

        return x1
    



class DeformableTransformer(nn.Module):
    def __init__(self, dim, nhead, num_layers, memory_length, self_attention):
        super().__init__()
        self.num_layers = num_layers
        self.dim = dim
        self.memory_length = memory_length

        bilateral_attention_layer = DeformableTransformerEncoderLayer(d_model=dim, nhead=nhead, memory_length=memory_length, self_attention=self_attention)

        self.bilateral_attention_layers = _get_clones(bilateral_attention_layer, num_layers)

        self.pos_obj_2d = PositionEmbeddingSine2d(num_pos_feats=dim // 2, normalize=True)

        self.pos_obj_1d = DynamicPositionalEmbedding(demb=dim)
    
    
    def forward(self, query, value, video_length):
        '''query: B, C, H, W,
        value: T, B, C, H, W'''
        
        shape = value.shape
        T, B, C, H, W = shape

        #memory_length_info = torch.count_nonzero(value.view(T, -1).sum(-1)) / T

        query = query.permute(0, 2, 3, 1).contiguous() # B, H, W, C
        query = query.view(B, H * W, C)

        value = value.permute(1, 2, 0, 3, 4).contiguous()

        pos_2d = self.pos_obj_2d(shape_util=(B, H, W), device=query.device)
        pos_2d = pos_2d.permute(0, 2, 3, 1).contiguous()
        pos_2d = pos_2d.view(B, H * W, C)

        #reference_points = self.get_grid(value)
        #reference_points = reference_points.view(B, H*W, 2) # B, L, 2

        #pos_1d = self.pos_obj_1d(pos_seq=memory_length_info.view(1,)) # 1, C
        #pos_1d = pos_1d[None].repeat(B, H*W, 1)

        pos_self = pos_2d
        pos_cross = pos_2d

        #pos_cross = torch.zeros_like(pos_2d)
        
        for l in range(self.num_layers):
            query, sampling_locations, attention_weights, offsets = self.bilateral_attention_layers[l](src=query, 
                                                                                              pos_self=pos_self, 
                                                                                              pos_cross=pos_cross,
                                                                                              value=value,
                                                                                              video_length=video_length)
        query = query.permute(0, 1, 2).contiguous()
        query = query.view(B, C, H, W)

        return query, sampling_locations, attention_weights, offsets
    




class DeformableTransformer6(nn.Module):
    def __init__(self, dim, nhead, num_layers, memory_length, add_motion_cues, self_attention):
        super().__init__()
        self.num_layers = num_layers
        self.dim = dim
        self.memory_length = memory_length
        self.add_motion_cues = add_motion_cues

        bilateral_attention_layer = DeformableTransformerEncoderLayer6(d_model=dim, nhead=nhead, memory_length=memory_length, self_attention=self_attention, add_motion_cues=add_motion_cues)

        self.bilateral_attention_layers = _get_clones(bilateral_attention_layer, num_layers)

        self.pos_obj_2d = PositionEmbeddingSine2d(num_pos_feats=dim // 2, normalize=True)

        self.pos_obj_1d = DynamicPositionalEmbedding(demb=dim)
    
    
    def forward(self, query, value, key, video_length):
        '''query: B, C, H, W,
        value: T, B, C, H, W'''
        
        shape = value.shape
        T, B, C, H, W = shape

        #memory_length_info = torch.count_nonzero(value.view(T, -1).sum(-1)) / T

        query = query.permute(0, 2, 3, 1).contiguous() # B, H, W, C
        query = query.view(B, H * W, C)

        if self.add_motion_cues:
            key = key.permute(1, 2, 0, 3, 4).contiguous()
        value = value.permute(1, 2, 0, 3, 4).contiguous()

        pos_2d = self.pos_obj_2d(shape_util=(B, H, W), device=query.device)
        pos_2d = pos_2d.permute(0, 2, 3, 1).contiguous()
        pos_2d = pos_2d.view(B, H * W, C)

        #reference_points = self.get_grid(value)
        #reference_points = reference_points.view(B, H*W, 2) # B, L, 2

        #pos_1d = self.pos_obj_1d(pos_seq=memory_length_info.view(1,)) # 1, C
        #pos_1d = pos_1d[None].repeat(B, H*W, 1)

        pos_self = pos_2d
        pos_cross = pos_2d

        #pos_cross = torch.zeros_like(pos_2d)
        
        for l in range(self.num_layers):
            query, sampling_locations, attention_weights, offsets = self.bilateral_attention_layers[l](src=query, 
                                                                                              pos_self=pos_self, 
                                                                                              pos_cross=pos_cross,
                                                                                              value=value,
                                                                                              key=key,
                                                                                              video_length=video_length)
        query = query.permute(0, 1, 2).contiguous()
        query = query.view(B, C, H, W)

        return query, sampling_locations, attention_weights, offsets
    



class DeformableTransformer3(nn.Module):
    def __init__(self, dim, nhead, num_layers, memory_length):
        super().__init__()
        self.num_layers = num_layers
        self.dim = dim
        self.memory_length = memory_length

        bilateral_attention_layer = DeformableTransformerEncoderLayer3(d_model=dim, nhead=nhead, memory_length=memory_length)

        self.bilateral_attention_layers = _get_clones(bilateral_attention_layer, num_layers)

        self.pos_obj_2d = PositionEmbeddingSine2d(num_pos_feats=dim // 2, normalize=True)

        self.pos_obj_1d = DynamicPositionalEmbedding(demb=dim)
    
    def get_grid(self, value):

        B, C, T, H, W = value.shape

        # Generate coordinates along each axis
        y = torch.linspace(-1, 1, H, device=value.device)
        x = torch.linspace(-1, 1, W, device=value.device)

        # Create the meshgrid
        grid_y, grid_x = torch.meshgrid(y, x)
        ref = torch.stack((grid_y, grid_x), -1)[None].repeat(B, 1, 1, 1, 1)
        return ref
    
    
    def forward(self, query, value):
        '''query: B, C, H, W,
        value: T, B, C, H, W'''

        
        shape = value.shape
        T, B, C, H, W = shape

        memory_length_info = torch.count_nonzero(value.view(T, -1).sum(-1)) / (T-1)
        
        query = query.permute(0, 2, 3, 1).contiguous() # B, H, W, C
        query = query.view(B, H * W, C)

        value = value.permute(1, 2, 0, 3, 4).contiguous()

        pos_2d = self.pos_obj_2d(shape_util=(B, H, W), device=query.device)
        pos_2d = pos_2d.permute(0, 2, 3, 1).contiguous()
        pos_2d = pos_2d.view(B, H * W, C)

        reference_points = self.get_grid(value)
        reference_points = reference_points.view(B, H*W, 2) # B, L, 2

        pos_1d = self.pos_obj_1d(pos_seq=memory_length_info.view(1,)) # 1, C
        pos_1d = pos_1d[None].repeat(B, H*W, 1)

        pos_cross = pos_2d + pos_1d

        #pos_cross = torch.zeros_like(pos_2d)
        
        for l in range(self.num_layers):
            query, sampling_locations, attention_weights, offsets = self.bilateral_attention_layers[l](src=query,
                                                                                              pos_cross=pos_cross, 
                                                                                              value=value, 
                                                                                              reference_points=reference_points)
        query = query.permute(0, 1, 2).contiguous()
        query = query.view(B, C, H, W)

        return query, sampling_locations, attention_weights, offsets
    




class DeformableTransformer4(nn.Module):
    def __init__(self, dim, nhead, num_layers, memory_length):
        super().__init__()
        self.num_layers = num_layers
        self.dim = dim
        self.memory_length = memory_length

        bilateral_attention_layer = DeformableTransformerEncoderLayer4(d_model=dim, nhead=nhead, memory_length=memory_length)

        self.bilateral_attention_layers = _get_clones(bilateral_attention_layer, num_layers)

        self.pos_obj_2d = PositionEmbeddingSine2d(num_pos_feats=dim // 2, normalize=True)

        self.pos_level = nn.Parameter(torch.randn(3, dim))

        self.query_reduction_layers = nn.ModuleList()
        self.query_expansion_layers = nn.ModuleList()
        self.value_expansion_layers = nn.ModuleList()
        for i in range(1, 4):
            query_reduction = nn.Conv2d(in_channels=dim, out_channels=int(dim/(2**(4-i-1))), kernel_size=1)
            query_expansion = nn.Conv2d(in_channels=int(dim/(2**(4-i-1))), out_channels=dim, kernel_size=1)
            value_expansion = nn.Conv3d(in_channels=int(dim/(2**(4-i-1))), out_channels=dim, kernel_size=1)
            self.query_reduction_layers.append(query_reduction)
            self.query_expansion_layers.append(query_expansion)
            self.value_expansion_layers.append(value_expansion)

    

    def get_grid_2d(self, query):

        B, C, H, W = query.shape

        # Generate coordinates along each axis
        y = torch.linspace(-1, 1, H, device=query.device)
        x = torch.linspace(-1, 1, W, device=query.device)

        # Create the meshgrid
        grid_y, grid_x = torch.meshgrid(y, x)
        ref = torch.stack((grid_y, grid_x), -1)[None].repeat(B, 1, 1, 1)
        return ref
    
    
    def forward(self, query, value):
        '''query: [B, C, H, W] * 3,
        value: [T, B, C, H, W] * 3'''

        query_cat = []
        value_cat = []
        pos_self_cat = []
        reference_points_2d_cat = []
        split_query = []
        split_value = []
        for idx, (query_expansion, value_expansion) in enumerate(zip(self.query_expansion_layers, self.value_expansion_layers)):
            current_value = value[idx].permute(1, 2, 0, 3, 4).contiguous() # B, C, T, H, W

            reduced_query = query_expansion(query[idx])
            reduced_value = value_expansion(current_value)

            B, C, T, H, W = reduced_value.shape
            
            pos_2d = self.pos_obj_2d(shape_util=(B, H, W), device=query[0].device)
            pos_2d = pos_2d.permute(0, 2, 3, 1).contiguous()
            pos_2d = pos_2d.view(B, H * W, C)
            split_query.append(torch.tensor([H, W]))
            split_value.append(torch.tensor([T, H, W]))

            reference_points_2d = self.get_grid_2d(reduced_query)
            reference_points_2d = reference_points_2d.view(B, H*W, 2) # B, L, 2
            reference_points_2d_cat.append(reference_points_2d)

            reduced_query = reduced_query.permute(0, 2, 3, 1).contiguous() # B, H, W, C
            reduced_query = reduced_query.view(B, H * W, C)

            reduced_value = reduced_value.permute(0, 2, 3, 4, 1).contiguous() # B, T, H, W, C
            reduced_value = reduced_value.view(B, T * H * W, C)

            pos_level = self.pos_level[idx][None, None, :].repeat(B, H * W, 1)

            query_cat.append(reduced_query)
            value_cat.append(reduced_value)
            pos_self_cat.append(pos_2d + pos_level)
        
        query = torch.cat(query_cat, dim=1)
        value = torch.cat(value_cat, dim=1)
        pos_self = torch.cat(pos_self_cat, dim=1)
        reference_points_2d = torch.cat(reference_points_2d_cat, dim=1)
        
        for l in range(self.num_layers):
            query, sampling_locations, attention_weights, offsets = self.bilateral_attention_layers[l](src=query, 
                                                                                              pos_self=pos_self,
                                                                                              value=value, 
                                                                                              split_value=split_value,
                                                                                              reference_points_2d=reference_points_2d)

        query = torch.split(query, [torch.prod(x) for x in split_query], dim=1)
        attention_weights = torch.split(attention_weights, [torch.prod(x) for x in split_query], dim=2)
        sampling_locations = torch.split(sampling_locations, [torch.prod(x) for x in split_query], dim=2)

        out = []
        for i in range(len(query)):
            current_query = query[i].permute(0, 1, 2).contiguous()
            current_query = current_query.view((B, C,) + tuple(split_query[i].tolist()))
            current_query = self.query_reduction_layers[i](current_query)
            out.append(current_query)

        return out, sampling_locations, attention_weights, offsets
    




class MemoryAttention(nn.Module):
    def __init__(self, dim, nhead):
        super().__init__()
        self.heads = nhead

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        self.o_proj = nn.Linear(dim, dim)

        # Each head has its own gate
        # init with -100 to make it close to 0 effect at the beginning
        self.gate = nn.Parameter(torch.full((1, self.num_heads, 1, 1), 0.0))
    
    def _retrieve_from_memory(self, query_states, memory, norm_term):
         # Apply ELU activation
        query_states = F.elu(query_states) + 1  # ELU activation + 1 for stability
        memory_output = torch.matmul(
            # GQA
            query_states,
            memory.repeat(1, self.num_key_value_groups, 1, 1),
        )

        # Broadcast norm_term to the shape of query_states, then sum across head_dim for normalization
        norm_term_broadcastable = torch.matmul(
            query_states,
            # GQA
            norm_term.transpose(-2, -1).repeat(1, self.num_key_value_groups, 1, 1),
        )

        # Perform division
        memory_output = memory_output / norm_term_broadcastable

        return memory_output


    def _update_memory(self, key_states, value_states, memory, norm_term):
        # key_states: [batch_size, num_heads, seq_len, head_dim]
        # value_states: [batch_size, num_heads, seq_len, value_dim]

        key_states = F.elu(key_states) + 1  # Apply ELU activation

        memory = memory + torch.matmul(key_states.transpose(-2, -1), value_states)

        norm_term = norm_term + key_states.sum(dim=2, keepdim=True)  # Update normalization termtialize normalization term

        return memory, norm_term
    


    def forward(self, x, memory, norm_term):



        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        qkv = [q, k, v]
        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> b h n d', h=self.heads).contiguous(), qkv)
        
        memory_output = self._retrieve_from_memory(
                q,
                memory,
                norm_term,
            )
        
        updated_memory, updated_norm_term = self._update_memory(
                k,
                v,
                memory,
                norm_term,
            )
        
        attn_output = torch.nn.functional.scaled_dot_product_attention(q,k,v)

        combined_output = (
                F.sigmoid(self.gate) * memory_output
                + (1 - F.sigmoid(self.gate)) * attn_output
            )
        
        # Prepare output for this segment
        combined_output = combined_output.transpose(1, 2).contiguous()
        combined_output = combined_output.view(bsz, q_len, self.hidden_size)

        final_output = self.o_proj(combined_output)
    

    


class CrossAttentionLayer(nn.Module):
    def __init__(self, dim, nhead, topk, pos_1d, distance):
        super().__init__()
        self.pos_1d = pos_1d
        self.topk = topk
        self.dim = dim

        self.cross_attention_layer = CrossTransformerEncoderLayer(d_model=dim, nhead=nhead, distance=distance, topk=topk)

        self.pos_obj_2d = PositionEmbeddingSine2d(num_pos_feats=dim // 2, normalize=True)

        if pos_1d:
            self.pos_obj_1d = PositionEmbeddingSine1d(num_pos_feats=dim, normalize=True)
    
    
    def forward(self, query, key, value):
        '''query: B, C, H, W,
        key: T, B, C, H, W,
        value: T, B, C, H, W'''
        
        T, B, C, H, W = key.shape

        if self.pos_1d:
            pos_1d = self.pos_obj_1d(shape_util=(B, T), device=query.device)

            pos_1d = pos_1d[:, :, :, None, None].repeat(1, 1, 1, H, W)
            pos_1d = pos_1d.view(B, C, T * H * W)
            pos_1d = pos_1d.permute(0, 2, 1).contiguous()
        else:
            pos_1d = torch.zeros(size=(B, T * H * W, C), device=query.device)

        query = query.permute(0, 2, 3, 1).contiguous() # B, H, W, C
        query = query.view(B, H * W, C)

        key = key.permute(0, 3, 4, 1, 2).contiguous()
        key = key.view(T * H * W, B, C)
        key = key.permute(1, 0, 2).contiguous() # B, T * H * W, C

        value = value.permute(0, 3, 4, 1, 2).contiguous()
        value = value.view(T * H * W, B, C)
        value = value.permute(1, 0, 2).contiguous() # B, T * H * W, C

        pos_2d = self.pos_obj_2d(shape_util=(B, H, W), device=query.device)
        pos_2d = pos_2d.permute(0, 2, 3, 1).contiguous()
        pos_2d = pos_2d.view(B, H * W, C)

        pos_2d_key = pos_2d[:, None, :, :].repeat(1, T, 1, 1)
        pos_2d_key = pos_2d_key.view(B, T * H * W, C)

        #pos_2d_key = pos_2d_key
        pos_2d_key = pos_2d_key + pos_1d
        
        query = self.cross_attention_layer(query=query, key=key, value=value, query_pos=pos_2d, key_pos=pos_2d_key)

        query = query.permute(0, 1, 2).contiguous()
        query = query.view(B, C, H, W)

        return query
    



class SelfAttentionLayer(nn.Module):
    def __init__(self, dim, nhead):
        super().__init__()
        self.dim = dim

        self.cross_attention_layer = TransformerEncoderLayer(d_model=dim, nhead=nhead)

        self.pos_obj_2d = PositionEmbeddingSine2d(num_pos_feats=dim // 2, normalize=True)
    
    
    def forward(self, src):
        '''src: B, C, H, W'''
        
        B, C, H, W = src.shape

        src = src.permute(0, 2, 3, 1).contiguous() # B, H, W, C
        src = src.view(B, H * W, C)

        pos_2d = self.pos_obj_2d(shape_util=(B, H, W), device=src.device)
        pos_2d = pos_2d.permute(0, 2, 3, 1).contiguous()
        pos_2d = pos_2d.view(B, H * W, C)
        
        query = self.cross_attention_layer(src=src, pos=pos_2d)

        query = query.permute(0, 1, 2).contiguous()
        query = query.view(B, C, H, W)

        return query
    



class SpatioTemporalTransformerTwoMemory(nn.Module):
    def __init__(self, dim, nhead, num_layers, topk, pos_1d, distance, dumb, kernel_size, gaussian_type, memory_length):
        super().__init__()
        self.num_layers = num_layers
        self.pos_1d = pos_1d
        self.topk = topk
        self.dim = dim
        self.dumb = dumb
        self.memory_length = memory_length - 1

        if dumb:
            bilateral_attention_layer = TransformerFlowLayerSeparatedDumb(d_model=dim, nhead=nhead, distance=distance, topk=topk, kernel_size=kernel_size, pos_1d=pos_1d)
        else:
            bilateral_attention_layer = TransformerFlowLayerSeparated(d_model=dim, nhead=nhead, distance=distance, topk=topk, pos_1d=pos_1d, gaussian_type=gaussian_type, dim_feedforward=4*dim)

        self.bilateral_attention_layers = _get_clones(bilateral_attention_layer, num_layers)

        self.pos_obj_2d = PositionEmbeddingSine2d(num_pos_feats=dim // 2, normalize=True)

        if pos_1d == 'learnable_sin':
            self.pos_obj_1d = DynamicPositionalEmbedding(demb=dim)
            #self.pos_obj_1d = PositionEmbeddingSine1d(num_pos_feats=dim, normalize=True)
        elif pos_1d == 'sin':
            self.pos_obj_1d = PositionEmbeddingSine1d(num_pos_feats=dim, normalize=False)
        elif pos_1d == 'learn':
            self.pos_obj_1d = nn.Parameter(torch.randn(self.memory_length, dim))
    
    
    def forward(self, query, key, value, video_length, max_idx=None):
        '''query: B, C, H, W,
        key: T, B, C, H, W,
        value: T, B, C, H, W,
        max_idx: T, B, HW, 2'''
        
        shape = key.shape
        T, B, C, H, W = shape

        if max_idx is not None:
            max_idx = max_idx.permute(1, 0, 2, 3).contiguous()
            max_idx = max_idx.view(B, T*H*W, 2)

        if self.pos_1d == 'learnable_sin':
            #pos_1d = self.pos_obj_1d(shape_util=(B, T), device=query.device)
            #pos_1d = pos_1d[:, :, :, None, None].repeat(1, 1, 1, H, W)

            pos_seq = torch.arange(T-1, -1, -1.0, device=query.device)
            pos_1d = self.pos_obj_1d(pos_seq=pos_seq) # T, C
            pos_1d = pos_1d.permute(1, 0).contiguous()
            pos_1d = pos_1d[None, :, :, None, None].repeat(B, 1, 1, H, W)
        elif self.pos_1d == "sin":
            pos_1d = self.pos_obj_1d(shape_util=(B, T), device=query.device)
            pos_1d = pos_1d[:, :, :, None, None].repeat(1, 1, 1, H, W)
        elif self.pos_1d == "learn":
            pos_1d = self.pos_obj_1d[None].repeat(B, 1, 1) # B, M, C
            pos_1d = pos_1d.permute(0, 2, 1).contiguous() # B, C, M
            if not self.training:
                pos_1d = torch.nn.functional.interpolate(pos_1d, size=video_length, mode='linear')
            pos_1d = pos_1d[:, :, :T, None, None].repeat(1, 1, 1, H, W)
        else:
            pos_1d = torch.zeros(size=(B, C, T, H, W), device=query.device)

        pos_1d = pos_1d.view(B, C, T * H * W)
        pos_1d = pos_1d.permute(0, 2, 1).contiguous()

        query = query.permute(0, 2, 3, 1).contiguous() # B, H, W, C
        query = query.view(B, H * W, C)

        key = key.permute(0, 3, 4, 1, 2).contiguous()
        key = key.view(T * H * W, B, C)
        key = key.permute(1, 0, 2).contiguous() # B, T * H * W, C

        value = value.permute(0, 3, 4, 1, 2).contiguous()
        value = value.view(T * H * W, B, C)
        value = value.permute(1, 0, 2).contiguous() # B, T * H * W, C

        pos_2d = self.pos_obj_2d(shape_util=(B, H, W), device=query.device)
        pos_2d = pos_2d.permute(0, 2, 3, 1).contiguous()
        pos_2d = pos_2d.view(B, H * W, C)

        pos_2d_key = pos_2d[:, None, :, :].repeat(1, T, 1, 1)
        pos_2d_key = pos_2d_key.view(B, T * H * W, C)

        if self.pos_1d == 'learnable_sin':
            pos_2d_key = pos_2d_key
            learnable_pos = pos_1d
        else:
            pos_2d_key = pos_2d_key + pos_1d
            learnable_pos = None
        
        for l in range(self.num_layers):
            query, weights = self.bilateral_attention_layers[l](query=query, 
                                                                key=key, 
                                                                value=value, 
                                                                query_pos=pos_2d, 
                                                                key_pos=pos_2d_key,
                                                                shape=shape,
                                                                max_idx=max_idx,
                                                                pos=learnable_pos)
        if not self.dumb:
            weights = weights.view(B, T, H*W, H*W)
            weights = weights.mean((0, 2, 3))[1:]
        #weights = weights.view(B, T, H, W, H*W)
        #weights = weights.permute(0, 4, 1, 2, 3).contiguous()
        #weights = weights.view(B * H * W, T, H, W)

        #matplotlib.use('QtAgg')
        #weights = torch.nn.functional.pad(weights, (7, 7, 7, 7), mode='reflect')
        #weights = GaussianSmoothing(channels=T, kernel_size=15, sigma=2.0)(weights) # B*H*W, T, H, W
        #weights = weights.view(B, H, W, T, H, W).detach().cpu()
        #temp = weights[0, 12, 12]
        #fig, ax = plt.subplots(1, temp.shape[0])
        #for i in range(temp.shape[0]):
        #    if temp.shape[0] == 1:
        #        ax.imshow(temp[i].detach().cpu(), cmap='hot', vmin=temp.min(), vmax=temp.max())
        #    else:
        #        ax[i].imshow(temp[i].detach().cpu(), cmap='hot', vmin=temp.min(), vmax=temp.max())
        #plt.show()

        query = query.permute(0, 1, 2).contiguous()
        query = query.view(B, C, H, W)

        return query, weights





class TransformerConv(nn.Module):
    def __init__(self, dim, norm, nhead, num_layers, topk, pos_1d, pos_2d, distance, dumb, kernel_size, gaussian_type):
        super().__init__()
        self.num_layers = num_layers
        self.pos_1d = pos_1d
        self.topk = topk
        self.dim = dim
        self.pos_2d = pos_2d

        self.memory = MemoryReader(topk)

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)

        if norm == 'group':
            self.out_conv = ConvBlocks2DGroupLegacy(in_dim=2 * dim, out_dim=dim, nb_blocks=1, residual=False)
        elif norm == 'batch':
            self.out_conv = ConvBlocks2DBatch(in_dim=2 * dim, out_dim=dim, nb_blocks=1, residual=False)
        
        self.pos_obj_2d = PositionEmbeddingSine2d(num_pos_feats=dim // 2, normalize=True)

        if pos_1d == 'learnable_sin':
            self.pos_obj_1d = DynamicPositionalEmbedding(demb=dim)
            self.pos_proj = nn.Linear(dim, dim, bias=False)
            #self.pos_obj_1d = PositionEmbeddingSine1d(num_pos_feats=dim, normalize=True)
        elif pos_1d == 'sin':
            self.pos_obj_1d = PositionEmbeddingSine1d(num_pos_feats=dim, normalize=False)
        elif pos_1d == 'conv':
            self.temporal_embedding = nn.Conv3d(in_channels=dim, out_channels=dim, kernel_size=(9, 1, 1), padding='same')
    
    
    def forward(self, query, key, value, max_idx=None):
        '''query: B, C, H, W,
        key: T, B, C, H, W,
        value: T, B, C, H, W,
        max_idx: T, B, HW, 2'''

        shape = key.shape
        T, B, C, H, W = shape

        residual = query

        if max_idx is not None:
            max_idx = max_idx.permute(1, 0, 2, 3).contiguous()
            max_idx = max_idx.view(B, T*H*W, 2)

        if self.pos_1d == 'learnable_sin':
            #pos_1d = self.pos_obj_1d(shape_util=(B, T), device=query.device)
            #pos_1d = pos_1d[:, :, :, None, None].repeat(1, 1, 1, H, W)

            pos_seq = torch.arange(T-1, -1, -1.0, device=query.device)
            pos_1d = self.pos_obj_1d(pos_seq=pos_seq) # T, C
            pos_1d = self.pos_proj(pos_1d)
            pos_1d = pos_1d.permute(1, 0).contiguous()
            pos_1d = pos_1d[None, :, :, None, None].repeat(B, 1, 1, H, W)
        elif self.pos_1d == "sin":
            pos_1d = self.pos_obj_1d(shape_util=(B, T), device=query.device)
            pos_1d = pos_1d[:, :, :, None, None].repeat(1, 1, 1, H, W)
        elif self.pos_1d == "conv":
            pos_1d = key.permute(1, 2, 0, 3, 4).contiguous()
            pos_1d = self.temporal_embedding(pos_1d) # B, C, T, H, W
        else:
            pos_1d = torch.zeros(size=(B, C, T, H, W), device=query.device)

        pos_1d = pos_1d.view(B, C, T * H * W)
        pos_1d = pos_1d.permute(0, 2, 1).contiguous()

        query = query.permute(0, 2, 3, 1).contiguous() # B, H, W, C
        query = query.view(B, H * W, C)

        key = key.permute(0, 3, 4, 1, 2).contiguous()
        key = key.view(T * H * W, B, C)
        key = key.permute(1, 0, 2).contiguous() # B, T * H * W, C

        value = value.permute(0, 3, 4, 1, 2).contiguous()
        value = value.view(T * H * W, B, C)
        value = value.permute(1, 0, 2).contiguous() # B, T * H * W, C

        if self.pos_2d:
            pos_2d = self.pos_obj_2d(shape_util=(B, H, W), device=query.device)
            pos_2d = pos_2d.permute(0, 2, 3, 1).contiguous()
            pos_2d = pos_2d.view(B, H * W, C)
            pos_2d_key = pos_2d[:, None, :, :].repeat(1, T, 1, 1)
            pos_2d_key = pos_2d_key.view(B, T * H * W, C)

            query = query + pos_2d
            key = key + pos_2d_key

        if self.pos_1d != 'learnable_sin':
            key = key + pos_1d

        query = self.to_q(query)
        key = self.to_k(key)
        value = self.to_v(value)

        query = query[:, None, :, :]
        key = key[:, None, :, :]
        value = value[:, None, :, :]

        affinity, weights = self.memory.get_affinity(key, query, pos=pos_1d) # B, head, THW, HW

        if max_idx is not None:
            gaussian = self.get_gaussians(max_idx, shape)
            affinity = affinity * gaussian[:, None, :, :]

        out = torch.matmul(affinity.transpose(3, 2), value)
        out = rearrange(out, 'b h n d -> b n (h d)')

        out = out.permute(0, 2, 1).contiguous()
        out = out.view(B, C, H, W)

        out = torch.cat([residual, out], dim=1)
        out = self.out_conv(out)

        weights = weights.mean(1)
        weights = weights.view(B, T, H*W, H*W)
        weights = weights.mean((0, 2, 3))[1:]

        return out, weights
    
    


    def make_gaussian_query(self, y_idx, x_idx, shape, sigma=7):
        T, B, C, H, W = shape

        y_idx = y_idx.view(B, T, H * W)
        x_idx = x_idx.view(B, T, H * W)

        y_idx = y_idx.permute(1, 0, 2).contiguous() # T, B, HW
        x_idx = x_idx.permute(1, 0, 2).contiguous() # T, B, HW

        kernel_sizes = torch.arange(T) * (-4/T) + 7

        gauss_kernel_list = []
        for t, k in enumerate(kernel_sizes):
            current_y_idx = y_idx[t] #B, HW
            current_x_idx = x_idx[t] #B, HW

            current_y_idx = current_y_idx.view(B, H*W, 1, 1).float()
            current_x_idx = current_x_idx.view(B, H*W, 1, 1).float()

            current_y = np.linspace(0,H-1,H)
            current_y = torch.FloatTensor(current_y).to('cuda:0')
            current_y = current_y.view(1,1,H,1).expand(B, H*W, H, 1)

            current_x = np.linspace(0,W-1,W)
            current_x = torch.FloatTensor(current_x).to('cuda:0')
            current_x = current_x.view(1,1,1,W).expand(B, H*W, 1, W)

            gauss_kernel = torch.exp(-((current_x-current_x_idx)**2 + (current_y-current_y_idx)**2) / (2 * (k)**2))
            gauss_kernel = gauss_kernel.view(B, H*W, H*W)
            gauss_kernel_list.append(gauss_kernel)

        gauss_kernel = torch.stack(gauss_kernel_list, dim=1)
        gauss_kernel = gauss_kernel.view(B, T*H*W, H*W)

        
        #y_idx = y_idx.view(B, T*H*W, 1, 1).float()
        #x_idx = x_idx.view(B, T*H*W, 1, 1).float()
#
        #y = np.linspace(0,H-1,H)
        #y = torch.FloatTensor(y).to('cuda:0')
        #y = y.view(1,1,H,1).expand(B, T*H*W, H, 1)
#
        #x = np.linspace(0,W-1,W)
        #x = torch.FloatTensor(x).to('cuda:0')
        #x = x.view(1,1,1,W).expand(B, T*H*W, 1, W)
        #        
        #gauss_kernel = torch.exp(-((x-x_idx)**2 + (y-y_idx)**2) / (2 * (sigma)**2))
        #gauss_kernel = gauss_kernel.view(B, T*H*W, H*W)

        return gauss_kernel
    

    
    def make_gaussian_memory(self, y_idx, x_idx, shape, sigma=7):
        T, B, C, H, W = shape

        y_idx = y_idx.view(B, T, H * W)
        x_idx = x_idx.view(B, T, H * W)

        y_idx = y_idx.permute(1, 0, 2).contiguous() # T, B, HW
        x_idx = x_idx.permute(1, 0, 2).contiguous() # T, B, HW

        kernel_sizes = torch.arange(T) * (-4/T) + 7

        gauss_kernel_list = []
        for t, k in enumerate(kernel_sizes):
            current_y_idx = y_idx[t] #B, HW
            current_x_idx = x_idx[t] #B, HW

            current_y_idx = current_y_idx.view(B, 1, 1, H*W).float()
            current_x_idx = current_x_idx.view(B, 1, 1, H*W).float()

            current_y = np.linspace(0,H-1,H)
            current_y = torch.FloatTensor(current_y).to('cuda:0')
            current_y = current_y.view(1,H,1,1).expand(B, H, 1, H*W)

            current_x = np.linspace(0,W-1,W)
            current_x = torch.FloatTensor(current_x).to('cuda:0')
            current_x = current_x.view(1,1,W,1).expand(B, 1, W, H*W)

            gauss_kernel = torch.exp(-((current_x-current_x_idx)**2 + (current_y-current_y_idx)**2) / (2 * (k)**2))
            gauss_kernel = gauss_kernel.view(B, H*W, H*W)
            gauss_kernel_list.append(gauss_kernel)

        gauss_kernel = torch.stack(gauss_kernel_list, dim=1)
        gauss_kernel = gauss_kernel.view(B, T*H*W, H*W)

        
        #y_idx = y_idx.view(B, T*H*W, 1, 1).float()
        #x_idx = x_idx.view(B, T*H*W, 1, 1).float()
#
        #y = np.linspace(0,H-1,H)
        #y = torch.FloatTensor(y).to('cuda:0')
        #y = y.view(1,1,H,1).expand(B, T*H*W, H, 1)
#
        #x = np.linspace(0,W-1,W)
        #x = torch.FloatTensor(x).to('cuda:0')
        #x = x.view(1,1,1,W).expand(B, T*H*W, 1, W)
        #        
        #gauss_kernel = torch.exp(-((x-x_idx)**2 + (y-y_idx)**2) / (2 * (sigma)**2))
        #gauss_kernel = gauss_kernel.view(B, T*H*W, H*W)

        return gauss_kernel
    


    def get_gaussians(self, argmax_idx, shape):
        T, B, C, H, W = shape
        y_idx, x_idx = argmax_idx[:, :, 0], argmax_idx[:, :, 1]
        #y_idx, x_idx = argmax_idx//W, argmax_idx%W
        if self.gaussian_type == 'memory':
            g = self.make_gaussian_memory(y_idx, x_idx, shape)
        elif self.gaussian_type == 'query':
            g = self.make_gaussian_query(y_idx, x_idx, shape)
        
        #matplotlib.use('QtAgg')
        #fig, ax = plt.subplots(1, 2)
        #temp = g.view(B, T, H * W, H * W)
        #temp = temp.view(B, T, H, W, H, W)
        #ax[0].imshow(temp[0, 0, :, :, 0, 0].view(H, W).detach().cpu(), cmap='gray')
        #ax[1].imshow(temp[0, -1, :, :, 0, 0].view(H, W).detach().cpu(), cmap='gray')
        #plt.show()
    
        return g





class MatchMemoryQuery(nn.Module):
    def __init__(self, dim, nhead, num_layers, topk, distance):
        super().__init__()
        self.num_layers = num_layers
        self.topk = topk
        self.dim = dim

        bilateral_attention_layer = TransformerFlowLayerSeparated(d_model=dim, nhead=nhead, distance=distance, topk=topk, pos_1d=None)

        self.bilateral_attention_layers = _get_clones(bilateral_attention_layer, num_layers)

        self.pos_obj_2d = PositionEmbeddingSine2d(num_pos_feats=dim // 2, normalize=True)
    
    
    def forward(self, query, key, value):
        '''query: B, C, H, W,
        key: B, C, H, W,
        value: B, C, H, W'''
        
        shape = key.shape
        B, C, H, W = shape

        query = query.permute(0, 2, 3, 1).contiguous() # B, H, W, C
        query = query.view(B, H * W, C)

        key = key.permute(0, 2, 3, 1).contiguous() # B, H, W, C
        key = key.view(B, H * W, C)

        value = value.permute(0, 2, 3, 1).contiguous() # B, H, W, C
        value = value.view(B, H * W, C)

        pos_2d = self.pos_obj_2d(shape_util=(B, H, W), device=query.device)
        pos_2d = pos_2d.permute(0, 2, 3, 1).contiguous()
        pos_2d = pos_2d.view(B, H * W, C)
        
        for l in range(self.num_layers):
            query, weights = self.bilateral_attention_layers[l](query=query, 
                                                                key=key, 
                                                                value=value, 
                                                                query_pos=pos_2d, 
                                                                key_pos=pos_2d,
                                                                shape=shape,
                                                                max_idx=None,
                                                                pos=None)

        query = query.permute(0, 1, 2).contiguous()
        query = query.view(B, C, H, W)

        return query





class DynamicPositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super(DynamicPositionalEmbedding, self).__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
        return pos_emb
    


class SpatioTemporalTransformerPos(nn.Module):
    def __init__(self, dim, nhead, num_layers, topk, pos_1d, distance):
        super().__init__()
        self.num_layers = num_layers
        self.pos_1d = pos_1d
        self.topk = topk
        self.dim = dim

        bilateral_attention_layer = PosAttention(d_model=dim, nhead=nhead, distance=distance, topk=topk)

        self.bilateral_attention_layers = _get_clones(bilateral_attention_layer, num_layers)

        self.pos_obj_2d = PositionEmbeddingSine2d(num_pos_feats=dim // 2, normalize=True)

        if pos_1d == 'sin':
            self.pos_obj_1d = DynamicPositionalEmbedding(demb=dim)
        else:
            self.temporal_embedding = nn.Conv3d(in_channels=dim, out_channels=dim, kernel_size=(9, 1, 1), padding='same')
    
    
    def forward(self, query, key, value, max_idx=None):
        '''query: B, C, H, W,
        key: T, B, C, H, W,
        value: T, B, C, H, W'''
        
        T, B, C, H, W = key.shape

        if self.pos_1d == "sin":
            pos_seq = torch.arange(T-1, -1, -1.0, device=query.device)
            pos_1d = self.pos_obj_1d(pos_seq=pos_seq) # T, C
            pos_1d = pos_1d.permute(1, 0).contiguous()
            pos_1d = pos_1d[None, :, :, None, None].repeat(B, 1, 1, H, W)
        else:
            pos_1d = key.permute(1, 2, 0, 3, 4).contiguous()
            pos_1d = self.temporal_embedding(pos_1d) # B, C, T, H, W

        pos_1d = pos_1d.view(B, C, T * H * W)
        pos_1d = pos_1d.permute(0, 2, 1).contiguous()

        query = query.permute(0, 2, 3, 1).contiguous() # B, H, W, C
        query = query.view(B, H * W, C)

        key = key.permute(0, 3, 4, 1, 2).contiguous()
        key = key.view(T * H * W, B, C)
        key = key.permute(1, 0, 2).contiguous() # B, T * H * W, C

        value = value.permute(0, 3, 4, 1, 2).contiguous()
        value = value.view(T * H * W, B, C)
        value = value.permute(1, 0, 2).contiguous() # B, T * H * W, C

        pos_2d = self.pos_obj_2d(shape_util=(B, H, W), device=query.device)
        pos_2d = pos_2d.permute(0, 2, 3, 1).contiguous()
        pos_2d = pos_2d.view(B, H * W, C)

        pos_2d_key = pos_2d[:, None, :, :].repeat(1, T, 1, 1)
        pos_2d_key = pos_2d_key.view(B, T * H * W, C)
        
        for l in range(self.num_layers):
            query, weights = self.bilateral_attention_layers[l](query=query, 
                                                                key=key, 
                                                                value=value, 
                                                                query_pos=pos_2d, 
                                                                key_pos=pos_2d_key,
                                                                pos=pos_1d)
        
        weights = weights.view(B, T, H*W, H*W)
        weights = weights.view(B, T, H, W, H*W)
        weights = weights.permute(0, 4, 1, 2, 3).contiguous()
        weights = weights.view(B * H * W, T, H, W)

        #matplotlib.use('QtAgg')
        #weights = torch.nn.functional.pad(weights, (7, 7, 7, 7), mode='reflect')
        #weights = GaussianSmoothing(channels=T, kernel_size=15, sigma=2.0)(weights) # B*H*W, T, H, W
        #weights = weights.view(B, H, W, T, H, W).detach().cpu()
        #temp = weights[0, 12, 12]
        #fig, ax = plt.subplots(1, temp.shape[0])
        #for i in range(temp.shape[0]):
        #    ax[i].imshow(temp[i], cmap='hot', vmin=temp.min(), vmax=temp.max())
        #plt.show()

        query = query.permute(0, 1, 2).contiguous()
        query = query.view(B, C, H, W)

        return query, weights



class TokenSelfAttention(nn.Module):
    def __init__(self, dim, nhead, pos_1d, num_layers, P):
        super().__init__()
        self.num_layers = num_layers
        self.dim = dim
        self.pos_1d = pos_1d

        bilateral_attention_layer = TransformerEncoderLayer(d_model=dim, nhead=nhead)

        self.bilateral_attention_layers = _get_clones(bilateral_attention_layer, num_layers)

        if pos_1d == 'sin':
            self.pos_obj_1d = PositionEmbeddingSine1d(num_pos_feats=dim, normalize=True)
        else:
            self.temporal_embedding = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=9, padding='same')
    
        if P > 0:
            self.token_pos = nn.Parameter(torch.randn(P, dim))


    def forward(self, tokens, pos=None):
        '''tokens: B, T, P, C'''
        
        B, T, P, C = tokens.shape

        if self.pos_1d == "sin":
            pos_1d = self.pos_obj_1d(shape_util=(B, T), device=tokens.device) # B, C, T
            pos_1d = pos_1d.permute(0, 2, 1).contiguous()
            pos_1d = pos_1d[:, :, None, :].repeat(1, 1, P, 1)
        else:
            pos_1d = tokens.permute(0, 2, 3, 1).contiguous() # B, P, C, T
            pos_1d = pos_1d.view(B * P, C, T)
            pos_1d = self.temporal_embedding(pos_1d)
            pos_1d = pos_1d.view(B, P, C, T)
            pos_1d = pos_1d.permute(0, 3, 1, 2).contiguous()

        tokens = tokens.view(B, T * P, C)
        pos_1d = pos_1d.view(B, T * P, C)
        if pos is not None:
            pos = pos.view(B, T * P, C)
        else:
            pos = self.token_pos[None, None].repeat(B, T, 1, 1)
            pos = pos.view(B, T * P, C)
        pos_1d = pos_1d + pos
        
        for l in range(self.num_layers):
            tokens, weights = self.bilateral_attention_layers[l](src=tokens, pos=pos_1d)

        tokens = tokens.view(B, T, P, C)

        return tokens


class TokenSample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        #self.reduce = nn.Linear(dim, 2)
        self.reduce = nn.Sequential(
                nn.Linear(dim, dim),
                nn.GELU(),
                nn.Linear(dim, dim),
                nn.GELU(),
                nn.Linear(dim, 2),
            )
        self.tanh = nn.Tanh()
        self.pos_obj_2d = PositionEmbeddingSine2d(num_pos_feats=dim // 2, normalize=True)
    
    def forward(self, tokens, memory):
        '''tokens: B, T, P, C,
        memory: T, B, C, H, W'''
        
        B, T, P, C = tokens.shape
        T, B, C, H, W = memory.shape

        memory = memory.view(T * B, C, H, W)

        tokens = tokens.permute(1, 0, 2, 3).contiguous() # T, B, P, C
        tokens = tokens.view(T * B, P, C)

        tokens = self.tanh(self.reduce(tokens)) # T * B, P, 2
        tokens = tokens[:, None, :, :] # T * B, 1, P, 2

        pos_2d = self.pos_obj_2d(shape_util=(T * B, H, W), device=memory.device)

        samples = torch.nn.functional.grid_sample(memory, tokens, mode='bilinear', align_corners=True) # T*B, C, 1, P
        samples = samples.view(T, B, C, 1, P).squeeze(3)
        samples = samples.permute(1, 0, 3, 2).contiguous() # B, T, P, C

        pos = torch.nn.functional.grid_sample(pos_2d, tokens, mode='bilinear', align_corners=True) # T*B, C, 1, P
        pos = pos.view(T, B, C, 1, P).squeeze(3)
        pos = pos.permute(1, 0, 3, 2).contiguous() # B, T, P, C

        return samples, pos, tokens.view(T, B, 1, P, 2)
    


class SpatioTemporalTransformerDynamic(nn.Module):
    def __init__(self, dim, nhead, num_layers, topk, pos_1d, distance):
        super().__init__()
        self.num_layers = num_layers
        self.pos_1d = pos_1d
        self.topk = topk
        self.dim = dim

        bilateral_attention_layer = TransformerFlowLayerSeparated(d_model=dim, nhead=nhead, distance=distance, topk=topk)

        self.bilateral_attention_layers = _get_clones(bilateral_attention_layer, num_layers)

        self.pos_obj_2d = PositionEmbeddingSine2d(num_pos_feats=dim // 2, normalize=True)

        if pos_1d:
            self.pos_obj_1d = PositionEmbeddingSine1d(num_pos_feats=dim, normalize=True)
    
    
    def forward(self, query, key, value, tokens):
        '''query: B, C, H, W,
        key: T, B, C, H, W,
        value: T, B, C, H, W,
        tokens: B, N, C'''
        
        T, B, C, H, W = key.shape

        if self.pos_1d:
            pos_1d = self.pos_obj_1d(shape_util=(B, T), device=query.device)

            pos_1d_tokens = pos_1d # B, C, T
            pos_1d_tokens = pos_1d_tokens.permute(0, 2, 1).contiguous()

            pos_1d = pos_1d[:, :, :, None, None].repeat(1, 1, 1, H, W)
            pos_1d = pos_1d.view(B, C, T * H * W)
            pos_1d = pos_1d.permute(0, 2, 1).contiguous()
        else:
            pos_1d = torch.zeros(size=(B, T * H * W, C), device=query.device)

        query = query.permute(0, 2, 3, 1).contiguous() # B, H, W, C
        query = query.view(B, H * W, C)

        key = key.permute(0, 3, 4, 1, 2).contiguous()
        key = key.view(T * H * W, B, C)
        key = key.permute(1, 0, 2).contiguous() # B, T * H * W, C

        value = value.permute(0, 3, 4, 1, 2).contiguous()
        value = value.view(T * H * W, B, C)
        value = value.permute(1, 0, 2).contiguous() # B, T * H * W, C

        pos_2d = self.pos_obj_2d(shape_util=(B, H, W), device=query.device)
        pos_2d = pos_2d.permute(0, 2, 3, 1).contiguous()
        pos_2d = pos_2d.view(B, H * W, C)

        pos_2d_key = pos_2d[:, None, :, :].repeat(1, T, 1, 1)
        pos_2d_key = pos_2d_key.view(B, T * H * W, C)

        #pos_2d_key = pos_2d_key
        pos_2d_key = pos_2d_key + pos_1d
        
        for l in range(self.num_layers):
            query = torch.cat([query, tokens], dim=1)
            query_pos = torch.cat([pos_2d, pos_1d_tokens], dim=1)

            query, weights = self.bilateral_attention_layers[l](query=query, key=key, value=value, query_pos=query_pos, key_pos=pos_2d_key)

            query, tokens = torch.split(query, [H * W, T], dim=1)

        #matplotlib.use('QtAgg')
        #weights = torch.nn.functional.pad(weights, (7, 7, 7, 7), mode='reflect')
        #weights = GaussianSmoothing(channels=T, kernel_size=15, sigma=2.0)(weights) # B*H*W, T, H, W
        #weights = weights.view(B, H, W, T, H, W).detach().cpu()
        #temp = weights[0, 12, 12]
        #fig, ax = plt.subplots(1, temp.shape[0])
        #for i in range(temp.shape[0]):
        #    ax[i].imshow(temp[i], cmap='hot', vmin=temp.min(), vmax=temp.max())
        #plt.show()

        query = query.permute(0, 1, 2).contiguous()
        query = query.view(B, C, H, W)

        return query, tokens
    





class CrossAttentionDynamic(nn.Module):
    def __init__(self, dim, nhead, num_layers, topk, pos_1d, distance):
        super().__init__()
        self.num_layers = num_layers
        self.pos_1d = pos_1d
        self.topk = topk
        self.dim = dim

        bilateral_attention_layer = CrossTransformerEncoderLayer(d_model=dim, nhead=nhead, distance=distance, topk=topk)

        self.bilateral_attention_layers = _get_clones(bilateral_attention_layer, num_layers)

        self.pos_obj_2d = PositionEmbeddingSine2d(num_pos_feats=dim // 2, normalize=True)

        if pos_1d:
            self.pos_obj_1d = PositionEmbeddingSine1d(num_pos_feats=dim, normalize=True)
    
    
    def forward(self, query, key, value, tokens):
        '''query: B, C, H, W,
        key: T, B, C, H, W,
        value: T, B, C, H, W,
        tokens: B, N, C'''
        
        T, B, C, H, W = key.shape

        if self.pos_1d:
            pos_1d = self.pos_obj_1d(shape_util=(B, T), device=query.device)

            pos_1d_tokens = pos_1d # B, C, T
            pos_1d_tokens = pos_1d_tokens.permute(0, 2, 1).contiguous()

            pos_1d = pos_1d[:, :, :, None, None].repeat(1, 1, 1, H, W)
            pos_1d = pos_1d.view(B, C, T * H * W)
            pos_1d = pos_1d.permute(0, 2, 1).contiguous()
        else:
            pos_1d = torch.zeros(size=(B, T * H * W, C), device=query.device)

        query = query.permute(0, 2, 3, 1).contiguous() # B, H, W, C
        query = query.view(B, H * W, C)

        key = key.permute(0, 3, 4, 1, 2).contiguous()
        key = key.view(T * H * W, B, C)
        key = key.permute(1, 0, 2).contiguous() # B, T * H * W, C

        value = value.permute(0, 3, 4, 1, 2).contiguous()
        value = value.view(T * H * W, B, C)
        value = value.permute(1, 0, 2).contiguous() # B, T * H * W, C

        pos_2d = self.pos_obj_2d(shape_util=(B, H, W), device=query.device)
        pos_2d = pos_2d.permute(0, 2, 3, 1).contiguous()
        pos_2d = pos_2d.view(B, H * W, C)

        pos_2d_key = pos_2d[:, None, :, :].repeat(1, T, 1, 1)
        pos_2d_key = pos_2d_key.view(B, T * H * W, C)

        #pos_2d_key = pos_2d_key
        pos_2d_key = pos_2d_key + pos_1d
        
        query_pos = torch.cat([pos_2d, pos_1d_tokens], dim=1)
        for l in range(self.num_layers):
            query = torch.cat([query, tokens], dim=1)

            query = self.bilateral_attention_layers[l](query=query, key=key, value=value, query_pos=query_pos, key_pos=pos_2d_key)

            query, tokens = torch.split(query, [H * W, T], dim=1)

        #matplotlib.use('QtAgg')
        #weights = torch.nn.functional.pad(weights, (7, 7, 7, 7), mode='reflect')
        #weights = GaussianSmoothing(channels=T, kernel_size=15, sigma=2.0)(weights) # B*H*W, T, H, W
        #weights = weights.view(B, H, W, T, H, W).detach().cpu()
        #temp = weights[0, 12, 12]
        #fig, ax = plt.subplots(1, temp.shape[0])
        #for i in range(temp.shape[0]):
        #    ax[i].imshow(temp[i], cmap='hot', vmin=temp.min(), vmax=temp.max())
        #plt.show()

        query = query.permute(0, 1, 2).contiguous()
        query = query.view(B, C, H, W)

        return query, tokens
    



class UnbatchedTokenSelfAttention(nn.Module):
    def __init__(self, dim, nhead, num_layers, pos_1d):
        super().__init__()
        self.num_layers = num_layers
        self.pos_1d = pos_1d
        self.dim = dim

        bilateral_attention_layer = TransformerEncoderLayer(d_model=dim, nhead=nhead)

        self.bilateral_attention_layers = _get_clones(bilateral_attention_layer, num_layers)

        self.pos_obj_2d = PositionEmbeddingSine2d(num_pos_feats=dim // 2, normalize=True)

        if pos_1d:
            self.pos_obj_1d = PositionEmbeddingSine1d(num_pos_feats=dim, normalize=True)
    
    
    def forward(self, query, tokens):
        '''query: B, C, H, W,
        tokens: B, T, C'''
        
        B, C, H, W = query.shape
        B, T, C = tokens.shape

        if self.pos_1d:
            pos_1d = self.pos_obj_1d(shape_util=(B, T), device=query.device) # B, C, T
            pos_1d = pos_1d.permute(0, 2, 1).contiguous() # B, T, C
        else:
            pos_1d = torch.zeros(size=(B, T, C), device=query.device)

        query = query.permute(0, 2, 3, 1).contiguous() # B, H, W, C
        query = query.view(B, H * W, C)

        pos_2d = self.pos_obj_2d(shape_util=(B, H, W), device=query.device)
        pos_2d = pos_2d.permute(0, 2, 3, 1).contiguous()
        pos_2d = pos_2d.view(B, H * W, C)
        
        query_pos = torch.cat([pos_2d, pos_1d], dim=1)

        for l in range(self.num_layers):
            query = torch.cat([query, tokens], dim=1)

            query = self.bilateral_attention_layers[l](src=query, pos=query_pos)

            query, tokens = torch.split(query, [H * W, T], dim=1)

        #matplotlib.use('QtAgg')
        #weights = torch.nn.functional.pad(weights, (7, 7, 7, 7), mode='reflect')
        #weights = GaussianSmoothing(channels=T, kernel_size=15, sigma=2.0)(weights) # B*H*W, T, H, W
        #weights = weights.view(B, H, W, T, H, W).detach().cpu()
        #temp = weights[0, 12, 12]
        #fig, ax = plt.subplots(1, temp.shape[0])
        #for i in range(temp.shape[0]):
        #    ax[i].imshow(temp[i], cmap='hot', vmin=temp.min(), vmax=temp.max())
        #plt.show()

        query = query.permute(0, 1, 2).contiguous()
        query = query.view(B, C, H, W)

        return query, tokens
    




class TransformerLayer(nn.Module):
    def __init__(self, dim, nhead, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.dim = dim

        bilateral_attention_layer = TransformerEncoderLayer(d_model=dim, nhead=nhead)

        self.bilateral_attention_layers = _get_clones(bilateral_attention_layer, num_layers)

        self.pos_obj_2d = PositionEmbeddingSine2d(num_pos_feats=dim // 2, normalize=True)
    
    
    def forward(self, query):
        '''query: T, B, C, H, W'''
        
        T, B, C, H, W = query.shape

        query = query.permute(0, 1, 3, 4, 2).contiguous() # T, B, H, W, C
        query = query.view(T * B, H * W, C)

        pos_2d = self.pos_obj_2d(shape_util=(T * B, H, W), device=query.device)
        pos_2d = pos_2d.permute(0, 2, 3, 1).contiguous()
        pos_2d = pos_2d.view(T * B, H * W, C)
        
        for l in range(self.num_layers):
            query, weights = self.bilateral_attention_layers[l](src=query, pos=pos_2d)

        query = query.view(T, B, H, W, C)
        query = query.permute(0, 1, 4, 2, 3).contiguous()

        return query
    



class BatchedTokenSelfAttention(nn.Module):
    def __init__(self, dim, nhead, num_layers, P):
        super().__init__()
        self.num_layers = num_layers
        self.dim = dim

        bilateral_attention_layer = TransformerEncoderLayer(d_model=dim, nhead=nhead)

        self.bilateral_attention_layers = _get_clones(bilateral_attention_layer, num_layers)

        self.pos_obj_2d = PositionEmbeddingSine2d(num_pos_feats=dim // 2, normalize=True)

        if P > 0:
            self.token_pos = nn.Parameter(torch.randn(P, dim))
    
    
    def forward(self, query, tokens, pos=None):
        '''query: T, B, C, H, W,
        tokens: B, T, P, C,
        pos: B, T, P, C'''
        
        T, B, C, H, W = query.shape
        B, T, P, C = tokens.shape

        if pos is not None:
            pos = pos.permute(1, 0, 2, 3).contiguous() # T, B, P, C
            pos = pos.view(T * B, P, C)
        else:
            pos = self.token_pos[None].repeat(T * B, 1, 1)

        tokens = tokens.permute(1, 0, 2, 3).contiguous() # T, B, P, C
        tokens = tokens.view(T * B, P, C)

        query = query.permute(0, 1, 3, 4, 2).contiguous() # T, B, H, W, C
        query = query.view(T * B, H, W, C)
        query = query.view(T * B, H * W, C)

        pos_2d = self.pos_obj_2d(shape_util=(T * B, H, W), device=query.device)
        pos_2d = pos_2d.permute(0, 2, 3, 1).contiguous()
        pos_2d = pos_2d.view(T * B, H * W, C)

        query_pos = torch.cat([pos_2d, pos], dim=1)
        
        for l in range(self.num_layers):
            src = torch.cat([query, tokens], dim=1)

            src, weights = self.bilateral_attention_layers[l](src=src, pos=query_pos)

            query = src[:, :H*W]
            tokens = src[:, H*W:]

        #matplotlib.use('QtAgg')
        #weights = torch.nn.functional.pad(weights, (7, 7, 7, 7), mode='reflect')
        #weights = GaussianSmoothing(channels=T, kernel_size=15, sigma=2.0)(weights) # B*H*W, T, H, W
        #weights = weights.view(B, H, W, T, H, W).detach().cpu()
        #temp = weights[0, 12, 12]
        #fig, ax = plt.subplots(1, temp.shape[0])
        #for i in range(temp.shape[0]):
        #    ax[i].imshow(temp[i], cmap='hot', vmin=temp.min(), vmax=temp.max())
        #plt.show()
        
        tokens = tokens.view(T, B, P, C)
        tokens = tokens.permute(1, 0, 2, 3).contiguous()

        query = query.view(T, B, H, W, C)
        query = query.permute(0, 1, 4, 2, 3).contiguous()

        return query, tokens
    


class TransformerFlowEncoderLocalGlobalAll(nn.Module):
    def __init__(self, dim, nhead, num_layers):
        super().__init__()
        self.num_layers = num_layers

        bilateral_attention_layer = TransformerFlowLayer(d_model=dim, nhead=nhead)
        self.bilateral_attention_layers = _get_clones(bilateral_attention_layer, num_layers)

        decoder_layer = TransformerFlowLayer(d_model=dim, nhead=nhead)
        self.decoder_layers = _get_clones(decoder_layer, num_layers)

        self.pos_obj_2d = PositionEmbeddingSine2d(num_pos_feats=dim // 2, normalize=True)

        proj2 = nn.Linear(dim, dim)
        self.proj2 = _get_clones(proj2, num_layers)
    
    
    def forward(self, unlabeled, dist_emb):
        '''unlabeled: T, B, C, H, W
        dist_emb: B, T-1, C'''
        
        shape = unlabeled.shape
        T, B, C, H, W = shape

        dist_emb = dist_emb.permute(1, 0, 2).contiguous()
        dist_emb = dist_emb.view((T-1) * B, C)
        dist_emb = dist_emb[:, None, :].repeat(1, H * W, 1) # (T-1) * B, H * W, C

        unlabeled = unlabeled.permute(0, 1, 3, 4, 2).contiguous() # T, B, H, W, C
        unlabeled = unlabeled.view(T, B, H * W, C)

        pos_2d = self.pos_obj_2d(shape_util=(B, H, W), device=unlabeled.device)
        pos_2d = pos_2d.permute(0, 2, 3, 1).contiguous()
        pos_2d = pos_2d.view(B, H * W, C)
        pos_2d = pos_2d[None].repeat(T - 1, 1, 1, 1)
        pos_2d = pos_2d.view((T-1) * B, H * W, C)

        local_backward = unlabeled[:-1]
        local_forward = unlabeled[1:]

        for l in range(self.num_layers):

            local_backward = local_backward.view((T-1) * B, H * W, C)
            local_forward = local_forward.view((T-1) * B, H * W, C) 

            concat0 = torch.cat([local_forward, local_backward], dim=0)
            concat1 = torch.cat([local_backward, local_forward], dim=0)
            pos = torch.cat([pos_2d, pos_2d], dim=0)
            concat0 = self.bilateral_attention_layers[l](query=concat0, key=concat1, query_pos=pos, key_pos=pos)[0]
            local_forward, local_backward = torch.chunk(concat0, chunks=2, dim=0)

            local_forward = local_forward.view(T - 1, B, H * W, C)
            local_backward = local_backward.view(T - 1, B, H * W, C)
        
        pos_2d = pos_2d.view(T - 1, B, H * W, C)
        dist_emb = dist_emb.view(T - 1, B, H * W, C)
        global_forward = local_forward

        T_dim = T - 1

        for l in range(self.num_layers):

            dist_emb = self.proj2[l](dist_emb)

            key = global_forward.permute(1, 0, 2, 3).contiguous()
            key = key.view(B, T_dim * H * W, C)
            key = key[None, :, :, :].repeat((T-1), 1, 1, 1)
            key = key.view((T-1) * B, T_dim * H * W, C)

            key_pos_2d = pos_2d.permute(1, 0, 2, 3).contiguous()
            key_pos_2d = key_pos_2d.view(B, T_dim * H * W, C)
            key_pos_2d = key_pos_2d[None, :, :, :].repeat((T-1), 1, 1, 1)
            key_pos_2d = key_pos_2d.view((T-1) * B, T_dim * H * W, C)

            key_pos_1d = dist_emb.permute(1, 0, 2, 3).contiguous()
            key_pos_1d = key_pos_1d.view(B, T_dim * H * W, C)
            key_pos_1d = key_pos_1d[None, :, :, :].repeat((T-1), 1, 1, 1)
            key_pos_1d = key_pos_1d.view((T-1) * B, T_dim * H * W, C)

            query_pos = pos_2d + dist_emb
            key_pos = key_pos_2d + key_pos_1d
            query_pos = query_pos.view((T-1) * B, H * W, C)

            global_forward = global_forward.view((T-1) * B, H * W, C)

            global_forward, weights = self.decoder_layers[l](query=global_forward,
                                                    key=key, 
                                                    query_pos=query_pos,
                                                    key_pos=key_pos)
            
            global_forward = global_forward.view((T-1), B, H * W, C)
        
        weights = weights.view((T-1), B, H * W, T_dim * H * W).mean(2) # (T-1), B, T_dim * H * W
        weights = weights.permute(2, 0, 1).contiguous()
        weights = weights.view(T_dim, H * W, (T-1), B).mean(1)
        weights = weights.mean(2).permute(1, 0).contiguous()

        local_forward = local_forward.permute(0, 1, 3, 2).contiguous()
        local_forward = local_forward.view(T - 1, B, C, H, W)

        global_forward = global_forward.permute(0, 1, 3, 2).contiguous()
        global_forward = global_forward.view(T - 1, B, C, H, W)

        return local_forward, global_forward, weights



class TransformerFlowEncoderFromStartNoEmb(nn.Module):
    def __init__(self, dim, nhead, num_layers):
        super().__init__()
        self.num_layers = num_layers

        bilateral_attention_layer = TransformerFlowLayer(d_model=dim, nhead=nhead)
        self.bilateral_attention_layers = _get_clones(bilateral_attention_layer, num_layers)

        decoder_layer = TransformerFlowLayer(d_model=dim, nhead=nhead)
        self.decoder_layers = _get_clones(decoder_layer, num_layers)

        self.pos_obj_2d = PositionEmbeddingSine2d(num_pos_feats=dim // 2, normalize=True)
    
    
    def forward(self, unlabeled):
        '''unlabeled: T, B, C, H, W
        dist_emb: B, T-1, C'''
        
        shape = unlabeled.shape
        T, B, C, H, W = shape

        unlabeled = unlabeled.permute(0, 1, 3, 4, 2).contiguous() # T, B, H, W, C
        unlabeled = unlabeled.view(T, B, H * W, C)

        pos_2d = self.pos_obj_2d(shape_util=(B, H, W), device=unlabeled.device)
        pos_2d = pos_2d.permute(0, 2, 3, 1).contiguous()
        pos_2d = pos_2d.view(B, H * W, C)
        pos_2d = pos_2d[None].repeat(T - 1, 1, 1, 1)
        pos_2d = pos_2d.view((T-1) * B, H * W, C)

        backward = unlabeled[0][None].repeat(len(unlabeled) - 1, 1, 1, 1)
        forward = unlabeled[1:]

        backward = backward.view((T-1) * B, H * W, C)
        forward = forward.view((T-1) * B, H * W, C)

        for l in range(self.num_layers):
            concat0 = torch.cat([forward, backward], dim=0)
            concat1 = torch.cat([backward, forward], dim=0)
            pos = torch.cat([pos_2d, pos_2d], dim=0)
            concat0 = self.bilateral_attention_layers[l](query=concat0, key=concat1, query_pos=pos, key_pos=pos)[0]
            forward, backward = torch.chunk(concat0, chunks=2, dim=0)

            forward = forward.view(T - 1, B, H * W, C)
            pos_2d = pos_2d.view(T - 1, B, H * W, C)

            key_list = [forward[0]]
            for i in range(1, len(forward)):
                query_pos = pos_2d[i]
                key_pos = pos_2d[i-1]
                attn_out = self.decoder_layers[l](query=forward[i], key=key_list[i - 1], query_pos=query_pos, key_pos=key_pos)[0]
                key_list.append(attn_out)
            forward = torch.stack(key_list, dim=0)

            forward = forward.permute(0, 1, 3, 2).contiguous()
            forward = forward.view(T - 1, B, C, H, W)

        return forward



class TransformerFlowSegEncoderAggregationDistance(nn.Module):
    def __init__(self, dim, nhead, num_layers):
        super().__init__()
        self.num_layers = num_layers

        bilateral_attention_layer = TransformerFlowLayer(d_model=dim, nhead=nhead)
        self.bilateral_attention_layers = _get_clones(bilateral_attention_layer, num_layers)

        self.decoder_layer = TransformerFlowLayer(d_model=dim, nhead=nhead)

        self.pos_obj_2d = PositionEmbeddingSine2d(num_pos_feats=dim // 2, normalize=True)
    
    
    def forward(self, unlabeled, dist_emb):
        '''unlabeled: T, B, C, H, W
        dist_emb: B, T-1, C'''
        
        shape = unlabeled.shape
        T, B, C, H, W = shape

        dist_emb = dist_emb.permute(1, 0, 2).contiguous()
        dist_emb = dist_emb.view((T-1) * B, C)
        dist_emb = dist_emb[:, None, :].repeat(1, H * W, 1) # (T-1) * B, H * W, C

        unlabeled = unlabeled.permute(0, 1, 3, 4, 2).contiguous() # T, B, H, W, C
        unlabeled = unlabeled.view(T, B, H * W, C)

        pos_2d = self.pos_obj_2d(shape_util=(B, H, W), device=unlabeled.device)
        pos_2d = pos_2d.permute(0, 2, 3, 1).contiguous()
        pos_2d = pos_2d.view(B, H * W, C)
        pos_2d = pos_2d[None].repeat(T - 1, 1, 1, 1)
        pos_2d = pos_2d.view((T-1) * B, H * W, C)

        backward = unlabeled[:-1]
        forward = unlabeled[1:]

        backward = backward.view((T-1) * B, H * W, C)
        forward = forward.view((T-1) * B, H * W, C)

        for l in range(self.num_layers):
            concat0 = torch.cat([forward, backward], dim=0)
            concat1 = torch.cat([backward, forward], dim=0)
            pos = torch.cat([pos_2d + dist_emb, pos_2d + dist_emb], dim=0)
            concat0 = self.bilateral_attention_layers[l](query=concat0, key=concat1, query_pos=pos, key_pos=pos)[0]
            forward, backward = torch.chunk(concat0, chunks=2, dim=0)

        forward = forward.view(T - 1, B, H * W, C)
        pos_2d = pos_2d.view(T - 1, B, H * W, C)
        dist_emb = dist_emb.view(T - 1, B, H * W, C)

        key_list = [forward[0]]
        for i in range(1, len(forward)):
            current_pos = pos_2d[i]
            attn_out = self.decoder_layer(query=forward[i], key=key_list[i - 1], query_pos=current_pos, key_pos=current_pos)[0]
            key_list.append(attn_out)
        global_motion_forward = torch.stack(key_list, dim=0)

        global_motion_forward = global_motion_forward.permute(0, 1, 3, 2).contiguous()
        global_motion_forward = global_motion_forward.view(T - 1, B, C, H, W)

        forward = forward.permute(0, 1, 3, 2).contiguous()
        forward = forward.view(T - 1, B, C, H, W)

        return forward, global_motion_forward



class TransformerFlowEncoderAllDistanceNoEmb(nn.Module):
    def __init__(self, dim, nhead, num_layers):
        super().__init__()
        self.num_layers = num_layers

        bilateral_attention_layer = TransformerFlowLayer(d_model=dim, nhead=nhead)
        self.bilateral_attention_layers = _get_clones(bilateral_attention_layer, num_layers)

        decoder_layer = TransformerFlowLayer(d_model=dim, nhead=nhead)
        self.decoder_layers = _get_clones(decoder_layer, num_layers)

        self.pos_obj_2d = PositionEmbeddingSine2d(num_pos_feats=dim // 2, normalize=True)
    
    
    def forward(self, unlabeled):
        '''unlabeled: T, B, C, H, W
        dist_emb: B, T-1, C'''

        
        shape = unlabeled.shape
        T, B, C, H, W = shape

        unlabeled = unlabeled.permute(0, 1, 3, 4, 2).contiguous() # T, B, H, W, C
        unlabeled = unlabeled.view(T, B, H * W, C)

        pos_2d = self.pos_obj_2d(shape_util=(B, H, W), device=unlabeled.device)
        pos_2d = pos_2d.permute(0, 2, 3, 1).contiguous()
        pos_2d = pos_2d.view(B, H * W, C)
        pos_2d = pos_2d[None].repeat(T - 1, 1, 1, 1)
        pos_2d = pos_2d.view((T-1) * B, H * W, C)

        backward = unlabeled[:-1]
        forward = unlabeled[1:]

        backward = unlabeled[0][None].repeat(len(unlabeled) - 1, 1, 1, 1)
        forward = unlabeled[1:]

        backward = backward.view((T-1) * B, H * W, C)
        forward = forward.view((T-1) * B, H * W, C)

        for l in range(self.num_layers):
            concat0 = torch.cat([forward, backward], dim=0)
            concat1 = torch.cat([backward, forward], dim=0)
            pos = torch.cat([pos_2d, pos_2d], dim=0)
            concat0 = self.bilateral_attention_layers[l](query=concat0, key=concat1, query_pos=pos, key_pos=pos)[0]
            forward, backward = torch.chunk(concat0, chunks=2, dim=0)

            forward = forward.view(T - 1, B, H * W, C)
            pos_2d = pos_2d.view(T - 1, B, H * W, C)

            T_dim = T - 1

            key = forward.permute(1, 0, 2, 3).contiguous()
            key = key.view(B, T_dim * H * W, C)
            key = key[None, :, :, :].repeat((T-1), 1, 1, 1)
            key = key.view((T-1) * B, T_dim * H * W, C)

            key_pos_2d = pos_2d.permute(1, 0, 2, 3).contiguous()
            key_pos_2d = key_pos_2d.view(B, T_dim * H * W, C)
            key_pos_2d = key_pos_2d[None, :, :, :].repeat((T-1), 1, 1, 1)
            key_pos_2d = key_pos_2d.view((T-1) * B, T_dim * H * W, C)
        
            query_pos = pos_2d
            key_pos = key_pos_2d

            query_pos = query_pos.view((T-1) * B, H * W, C)
            forward = forward.view((T-1) * B, H * W, C)

            forward, weights = self.decoder_layers[l](query=forward,
                                                    key=key, 
                                                    query_pos=query_pos,
                                                    key_pos=key_pos)
            
        
        weights = weights.view((T-1), B, H * W, T_dim * H * W).mean(2) # (T-1), B, T_dim * H * W
        weights = weights.permute(2, 0, 1).contiguous()
        weights = weights.view(T_dim, H * W, (T-1), B).mean(1)
        weights = weights.mean(2).permute(1, 0).contiguous()

        forward = forward.view(T - 1, B, H * W, C)
        forward = forward.permute(0, 1, 3, 2).contiguous()
        forward = forward.view(T - 1, B, C, H, W)

        return forward, weights.detach()



class TransformerFlowEncoderAllDistance(nn.Module):
    def __init__(self, dim, nhead, num_layers):
        super().__init__()
        self.num_layers = num_layers

        bilateral_attention_layer = TransformerFlowLayer(d_model=dim, nhead=nhead)
        self.bilateral_attention_layers = _get_clones(bilateral_attention_layer, num_layers)

        decoder_layer = TransformerFlowLayer(d_model=dim, nhead=nhead)
        self.decoder_layers = _get_clones(decoder_layer, num_layers)

        self.pos_obj_2d = PositionEmbeddingSine2d(num_pos_feats=dim // 2, normalize=True)

        proj1 = nn.Linear(dim, dim)
        self.proj1 = _get_clones(proj1, num_layers)
        proj2 = nn.Linear(dim, dim)
        self.proj2 = _get_clones(proj2, num_layers)
    
    
    def forward(self, unlabeled, dist_emb):
        '''unlabeled: T, B, C, H, W
        dist_emb: B, T-1, C'''

        
        shape = unlabeled.shape
        T, B, C, H, W = shape

        dist_emb = dist_emb.permute(1, 0, 2).contiguous()
        dist_emb = dist_emb.view((T-1) * B, C)
        dist_emb = dist_emb[:, None, :].repeat(1, H * W, 1) # (T-1) * B, H * W, C

        unlabeled = unlabeled.permute(0, 1, 3, 4, 2).contiguous() # T, B, H, W, C
        unlabeled = unlabeled.view(T, B, H * W, C)

        pos_2d = self.pos_obj_2d(shape_util=(B, H, W), device=unlabeled.device)
        pos_2d = pos_2d.permute(0, 2, 3, 1).contiguous()
        pos_2d = pos_2d.view(B, H * W, C)
        pos_2d = pos_2d[None].repeat(T - 1, 1, 1, 1)
        pos_2d = pos_2d.view((T-1) * B, H * W, C)

        backward = unlabeled[:-1]
        forward = unlabeled[1:]

        backward = unlabeled[0][None].repeat(len(unlabeled) - 1, 1, 1, 1)
        forward = unlabeled[1:]

        backward = backward.view((T-1) * B, H * W, C)
        forward = forward.view((T-1) * B, H * W, C)

        for l in range(self.num_layers):
            dist_emb = self.proj1[l](dist_emb)
            concat0 = torch.cat([forward, backward], dim=0)
            concat1 = torch.cat([backward, forward], dim=0)
            pos = torch.cat([pos_2d + dist_emb, pos_2d + dist_emb], dim=0)
            concat0 = self.bilateral_attention_layers[l](query=concat0, key=concat1, query_pos=pos, key_pos=pos)[0]
            forward, backward = torch.chunk(concat0, chunks=2, dim=0)

            forward = forward.view(T - 1, B, H * W, C)
            pos_2d = pos_2d.view(T - 1, B, H * W, C)
            dist_emb = dist_emb.view(T - 1, B, H * W, C)

            dist_emb = self.proj2[l](dist_emb)

            T_dim = T - 1

            key = forward.permute(1, 0, 2, 3).contiguous()
            key = key.view(B, T_dim * H * W, C)
            key = key[None, :, :, :].repeat((T-1), 1, 1, 1)
            key = key.view((T-1) * B, T_dim * H * W, C)

            key_pos_2d = pos_2d.permute(1, 0, 2, 3).contiguous()
            key_pos_2d = key_pos_2d.view(B, T_dim * H * W, C)
            key_pos_2d = key_pos_2d[None, :, :, :].repeat((T-1), 1, 1, 1)
            key_pos_2d = key_pos_2d.view((T-1) * B, T_dim * H * W, C)

            key_pos_1d = dist_emb.permute(1, 0, 2, 3).contiguous()
            key_pos_1d = key_pos_1d.view(B, T_dim * H * W, C)
            key_pos_1d = key_pos_1d[None, :, :, :].repeat((T-1), 1, 1, 1)
            key_pos_1d = key_pos_1d.view((T-1) * B, T_dim * H * W, C)
        
            query_pos = pos_2d + dist_emb
            key_pos = key_pos_2d + key_pos_1d

            query_pos = query_pos.view((T-1) * B, H * W, C)
            forward = forward.view((T-1) * B, H * W, C)

            forward, weights = self.decoder_layers[l](query=forward,
                                                    key=key, 
                                                    query_pos=query_pos,
                                                    key_pos=key_pos)
            
        
        weights = weights.view((T-1), B, H * W, T_dim * H * W).mean(2) # (T-1), B, T_dim * H * W
        weights = weights.permute(2, 0, 1).contiguous()
        weights = weights.view(T_dim, H * W, (T-1), B).mean(1)
        weights = weights.mean(2).permute(1, 0).contiguous()

        forward = forward.view(T - 1, B, H * W, C)
        forward = forward.permute(0, 1, 3, 2).contiguous()
        forward = forward.view(T - 1, B, C, H, W)

        return forward, weights.detach()
    


class TransformerFlowSegEncoderAggregationDistanceNoEmb(nn.Module):
    def __init__(self, dim, nhead, num_layers):
        super().__init__()
        self.num_layers = num_layers

        bilateral_attention_layer = TransformerFlowLayer(d_model=dim, nhead=nhead)
        self.bilateral_attention_layers = _get_clones(bilateral_attention_layer, num_layers)

        self.decoder_layer = TransformerFlowLayer(d_model=dim, nhead=nhead)

        self.pos_obj_2d = PositionEmbeddingSine2d(num_pos_feats=dim // 2, normalize=True)
    
    
    def forward(self, unlabeled):
        '''unlabeled: T, B, C, H, W
        dist_emb: B, T-1, C'''
        
        shape = unlabeled.shape
        T, B, C, H, W = shape

        unlabeled = unlabeled.permute(0, 1, 3, 4, 2).contiguous() # T, B, H, W, C
        unlabeled = unlabeled.view(T, B, H * W, C)

        pos_2d = self.pos_obj_2d(shape_util=(B, H, W), device=unlabeled.device)
        pos_2d = pos_2d.permute(0, 2, 3, 1).contiguous()
        pos_2d = pos_2d.view(B, H * W, C)
        pos_2d = pos_2d[None].repeat(T - 1, 1, 1, 1)
        pos_2d = pos_2d.view((T-1) * B, H * W, C)

        backward = unlabeled[:-1]
        forward = unlabeled[1:]

        backward = backward.view((T-1) * B, H * W, C)
        forward = forward.view((T-1) * B, H * W, C)

        for l in range(self.num_layers):
            concat0 = torch.cat([forward, backward], dim=0)
            concat1 = torch.cat([backward, forward], dim=0)
            pos = torch.cat([pos_2d, pos_2d], dim=0)
            concat0 = self.bilateral_attention_layers[l](query=concat0, key=concat1, query_pos=pos, key_pos=pos)[0]
            forward, backward = torch.chunk(concat0, chunks=2, dim=0)

        forward = forward.view(T - 1, B, H * W, C)
        pos_2d = pos_2d.view(T - 1, B, H * W, C)

        key_list = [forward[0]]
        for i in range(1, len(forward)):
            current_pos = pos_2d[i]
            attn_out = self.decoder_layer(query=forward[i], key=key_list[i - 1], query_pos=current_pos, key_pos=current_pos)[0]
            key_list.append(attn_out)
        global_motion_forward = torch.stack(key_list, dim=0)

        global_motion_forward = global_motion_forward.permute(0, 1, 3, 2).contiguous()
        global_motion_forward = global_motion_forward.view(T - 1, B, C, H, W)

        forward = forward.permute(0, 1, 3, 2).contiguous()
        forward = forward.view(T - 1, B, C, H, W)

        return forward, global_motion_forward
    




class TransformerFlowSegEncoderAggregation3D(nn.Module):
    def __init__(self, dim, nhead, num_layers):
        super().__init__()
        self.num_layers = num_layers

        bilateral_attention_layer = TransformerFlowLayer(d_model=dim, nhead=nhead)
        self.bilateral_attention_layers = _get_clones(bilateral_attention_layer, num_layers)

        self.decoder_layer = TransformerFlowLayer(d_model=dim, nhead=nhead)

        self.pos_obj_2d = PositionEmbeddingSine2d(num_pos_feats=dim // 2, normalize=True)
        self.pos_z = nn.Parameter(torch.randn(2, dim))
    
    
    def forward(self, unlabeled):
        '''unlabeled: T, B, C, D, H, W'''
        
        shape = unlabeled.shape
        T, B, C, D, H, W = shape

        unlabeled = unlabeled.permute(0, 1, 3, 4, 5, 2).contiguous() # T, B, D, H, W, C
        unlabeled = unlabeled.view(T, B, D * H * W, C)

        pos_2d = self.pos_obj_2d(shape_util=(B, H, W), device=unlabeled.device)
        pos_2d = pos_2d.permute(0, 2, 3, 1).contiguous()
        pos_2d = pos_2d[:, None, :, :, :].repeat(1, D, 1, 1, 1)
        pos_2d = pos_2d.view(B, D * H * W, C)
        pos_2d = pos_2d[None].repeat(T, 1, 1, 1)
        pos_2d = pos_2d.view(T * B, D * H * W, C)

        pos_z = self.pos_z[None, :, None, None, :].repeat(B, 1, H, W, 1)
        pos_z = pos_z.view(B, D * H * W, C)
        pos_z = pos_z[None].repeat(T, 1, 1, 1)
        pos_z = pos_z.view(T * B, D * H * W, C)

        backward = unlabeled[:-1]
        forward = unlabeled
        backward = torch.cat([unlabeled[0][None], backward], dim=0)

        backward = backward.view(T * B, D * H * W, C)
        forward = forward.view(T * B, D * H * W, C)

        for l in range(self.num_layers):
            concat0 = torch.cat([forward, backward], dim=0)
            concat1 = torch.cat([backward, forward], dim=0)
            pos = torch.cat([pos_2d + pos_z, pos_2d + pos_z], dim=0)
            concat0 = self.bilateral_attention_layers[l](query=concat0, key=concat1, query_pos=pos, key_pos=pos)[0]
            forward, backward = torch.chunk(concat0, chunks=2, dim=0)

        forward = forward.view(T, B, D * H * W, C)
        pos_2d = pos_2d.view(T, B, D * H * W, C)
        pos_z = pos_z.view(T, B, D * H * W, C)

        global_motion_forward_list = []
        key = forward[0]
        for i in range(len(forward)):
            pos = pos_2d[i] + pos_z[i]
            attn_out = self.decoder_layer(query=forward[i], key=key, query_pos=pos, key_pos=pos)[0]
            key = attn_out
            global_motion_forward_list.append(key)
        global_motion_forward = torch.stack(global_motion_forward_list, dim=0) # T, B, D * H * W, C

        global_motion_forward = global_motion_forward.permute(0, 1, 3, 2).contiguous()
        global_motion_forward = global_motion_forward.view(T, B, C, D, H, W)

        forward = forward.permute(0, 1, 3, 2).contiguous()
        forward = forward.view(T, B, C, D, H, W)

        return forward, global_motion_forward



class TransformerLib(nn.Module):
    def __init__(self, dim, nhead, num_layers, video_length):
        super().__init__()
        self.num_layers = num_layers

        bilateral_attention_layer = TransformerFlowLayer(d_model=dim, nhead=nhead)
        self.bilateral_attention_layers = _get_clones(bilateral_attention_layer, num_layers)

        strain_layer = TransformerEncoderLayer(d_model=dim, nhead=nhead)
        self.strain_layers = _get_clones(strain_layer, num_layers)

        conv_1d = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=video_length, padding='same')
        self.conv_1d_layers = _get_clones(conv_1d, num_layers)
        
        self.pos_obj_2d = PositionEmbeddingSine2d(num_pos_feats=dim // 2, normalize=True)

        self.strain = nn.Parameter(torch.randn(dim))

        self.lv_strain_pos = nn.Parameter(torch.randn(dim))
        self.rv_strain_pos = nn.Parameter(torch.randn(dim))

        self.mlp = MLP(input_dim=512, hidden_dim=256, output_dim=1, num_layers=2)
    
    
    def forward(self, unlabeled):
        '''unlabeled: T, B, C, H, W'''
        
        shape = unlabeled.shape
        T, B, C, H, W = shape

        unlabeled = unlabeled.permute(0, 1, 3, 4, 2).contiguous() # T, B, H, W, C
        unlabeled = unlabeled.view(T, B, H * W, C)

        pos_2d = self.pos_obj_2d(shape_util=(B, H, W), device=unlabeled.device)
        pos_2d = pos_2d.permute(0, 2, 3, 1).contiguous()
        pos_2d = pos_2d.view(B, H * W, C)
        pos_2d = pos_2d[None].repeat(T, 1, 1, 1)
        pos_2d = pos_2d.view(T * B, H * W, C)

        strain = self.strain[None, None].repeat(T * B, 1, 1) # T * B, 1, C
        lv_strain = strain
        rv_strain = strain

        lv_strain_pos = self.lv_strain_pos[None, None].repeat(T * B, 1, 1) # T * B, 1, C
        rv_strain_pos = self.rv_strain_pos[None, None].repeat(T * B, 1, 1) # T * B, 1, C

        forward = unlabeled[1:]
        backward = unlabeled[:-1]

        forward = torch.cat([unlabeled[0][None], forward], dim=0)
        backward = torch.cat([unlabeled[0][None], backward], dim=0)

        backward = backward.view(T * B, H * W, C)
        forward = forward.view(T * B, H * W, C)

        for l in range(self.num_layers):
            forward = torch.cat([forward, lv_strain, rv_strain], dim=1) # T * B, H * W + 2, C
            pos = torch.cat([pos_2d, lv_strain_pos, rv_strain_pos], dim=1) # T * B, H * W + 2, C
            forward, weights = self.bilateral_attention_layers[l](query=forward, key=backward, query_pos=pos, key_pos=pos_2d)
            forward, lv_strain, rv_strain = torch.split(forward, [H*W, 1, 1], dim=1)

            lv_strain = lv_strain.view(T, B, 1, C).squeeze(2) # T, B, C
            rv_strain = rv_strain.view(T, B, 1, C).squeeze(2) # T, B, C

            lv_strain = lv_strain.permute(1, 2, 0).contiguous() # B, C, T
            rv_strain = rv_strain.permute(1, 2, 0).contiguous() # B, C, T

            pos_lv = self.conv_1d_layers[l](lv_strain) # B, C, T
            pos_rv = self.conv_1d_layers[l](rv_strain) # B, C, T

            pos_lv = pos_lv.permute(0, 2, 1).contiguous() # B, T, C
            pos_rv = pos_rv.permute(0, 2, 1).contiguous() # B, T, C
            lv_strain = lv_strain.permute(0, 2, 1).contiguous() # B, T, C
            rv_strain = rv_strain.permute(0, 2, 1).contiguous() # B, T, C

            concat0 = torch.cat([lv_strain, rv_strain], dim=0)
            pos = torch.cat([pos_lv, pos_rv], dim=0)

            concat0 = self.strain_layers[l](concat0, pos=pos)[0]
            lv_strain, rv_strain = torch.chunk(concat0, 2, 0)

            lv_strain = lv_strain.permute(1, 0, 2).contiguous() # T, B, C
            rv_strain = rv_strain.permute(1, 0, 2).contiguous() # T, B, C

            lv_strain = lv_strain.view(T * B, C)[:, None] # T * B, 1, C
            rv_strain = rv_strain.view(T * B, C)[:, None] # T * B, 1, C
        
        forward = forward.view(T, B, H * W, C)

        forward = forward.permute(0, 1, 3, 2).contiguous() # T, B, C, H * W
        forward = forward.view(T, B, C, H, W)

        lv_strain = lv_strain.permute(0, 2, 1).contiguous() # T * B, C, 1
        lv_strain = lv_strain.view(T, B, C)

        rv_strain = rv_strain.permute(0, 2, 1).contiguous() # T * B, C, 1
        rv_strain = rv_strain.view(T, B, C)

        lv_strain = self.mlp(lv_strain).squeeze(2) # T, B
        rv_strain = self.mlp(rv_strain).squeeze(2) # T, B

        weights = weights[:, -2, :].view(T, B, H * W).view(T, B, H, W)

        return forward, lv_strain, rv_strain, weights
    


class TransformerFlowSegEncoderLabel(nn.Module):
    def __init__(self, dim, nhead, num_layers):
        super().__init__()
        self.num_layers = num_layers
        #self.nb_iters = nb_iters

        self_attention_layer = TransformerEncoderLayer(d_model=dim, nhead=nhead)
        self.self_attention_layers = _get_clones(self_attention_layer, num_layers)

        bilateral_attention_layer = TransformerFlowLayer(d_model=dim, nhead=nhead)
        self.bilateral_attention_layers = _get_clones(bilateral_attention_layer, num_layers)

        multilateral_attention_layer = TransformerFlowLayer(d_model=dim, nhead=nhead)
        self.multilateral_attention_layers = _get_clones(multilateral_attention_layer, num_layers)

        self.pos_obj_2d = PositionEmbeddingSine2d(num_pos_feats=dim // 2, normalize=True)

        cat_reduction = nn.Linear(2 * dim, dim)
        self.cat_reductions = _get_clones(cat_reduction, num_layers)
    
    
    def forward(self, unlabeled, label):
        '''unlabeled: T, B, C, H, W'''
        
        shape = unlabeled.shape
        T, B, C, H, W = shape

        unlabeled = unlabeled.permute(0, 1, 3, 4, 2).contiguous() # T, B, H, W, C
        unlabeled = unlabeled.view(T, B, H * W, C)

        label = label.permute(0, 2, 3, 1).contiguous() # B, H, W, C
        label = label.view(B, H * W, C)
        label = label[None].repeat(T, 1, 1, 1)

        pos_2d = self.pos_obj_2d(shape_util=(B, H, W), device=unlabeled.device)
        pos_2d = pos_2d.permute(0, 2, 3, 1).contiguous()
        pos_2d = pos_2d.view(B, H * W, C)
        pos_2d = pos_2d[None].repeat(T, 1, 1, 1)
        pos_2d = pos_2d.view(T * B, H * W, C)


        anchor = label # T, B, H * W, C
        frames = unlabeled # T, B, H * W, C

        anchor = anchor.view(T * B, H * W, C)
        frames = frames.view(T * B, H * W, C)
        flow = frames

        for l in range(self.num_layers):
            concat0 = torch.cat([frames, anchor], dim=0)
            pos_2d = torch.cat([pos_2d, pos_2d], dim=0)
            concat0 = self.self_attention_layers[l](concat0, pos=pos_2d)[0]

            frames, anchor = concat0.chunk(2, dim=0)
            pos_2d = pos_2d.chunk(2, dim=0)[0]
            
            flow = torch.cat([frames, flow], dim=-1)
            flow = self.cat_reductions[l](flow)

            flow = self.bilateral_attention_layers[l](query=flow, key=anchor, query_pos=pos_2d, key_pos=pos_2d)[0]

            flow_mean = flow.view(T, B, H * W, C).mean(0)
            flow_mean = flow_mean[None].repeat(T, 1, 1, 1)
            flow_mean = flow_mean.view(T * B, H * W, C)

            frames_mean = frames.view(T, B, H * W, C).mean(0)
            frames_mean = frames_mean[None].repeat(T, 1, 1, 1)
            frames_mean = frames_mean.view(T * B, H * W, C)

            anchor_mean = anchor.view(T, B, H * W, C).mean(0)
            anchor_mean = anchor_mean[None].repeat(T, 1, 1, 1)
            anchor_mean = anchor_mean.view(T * B, H * W, C)

            concat0 = torch.cat([frames, flow, anchor], dim=0)
            concat1 = torch.cat([frames_mean, flow_mean, anchor_mean], dim=0)
            pos_2d = pos_2d.repeat(3, 1, 1)
            concat0 = self.multilateral_attention_layers[l](query=concat0, key=concat1, query_pos=pos_2d, key_pos=pos_2d)[0]
            frames, flow, anchor = concat0.chunk(3, dim=0)
            pos_2d = pos_2d.chunk(3, dim=0)[0]

        frames = frames.view(T, B, H * W, C)
        frames = frames.permute(0, 1, 3, 2).contiguous()
        frames = frames.view(T, B, C, H, W)

        flow = flow.view(T, B, H * W, C)
        flow = flow.permute(0, 1, 3, 2).contiguous()
        flow = flow.view(T, B, C, H, W)

        anchor = anchor.view(T, B, H * W, C)
        anchor = anchor.permute(0, 1, 3, 2).contiguous()
        anchor = anchor.view(T, B, C, H, W)

        return frames, flow, anchor

    

class TransformerLocalMotion(nn.Module):
    def __init__(self, dim, nhead, num_layers):
        super().__init__()

        spatial_layer = TransformerFlowLayer(d_model=dim, nhead=nhead)
        self.spatial_layers = _get_clones(spatial_layer, num_layers)

        self.pos_obj_2d = PositionEmbeddingSine2d(num_pos_feats=dim // 2, normalize=True)
    
    
    def forward(self, spatial_features):
        '''spatial_features_1: T, B, C, H, W
            pos_1d: B, C, T'''
        
        shape = spatial_features.shape
        T, B, C, H, W = shape

        spatial_features = spatial_features.permute(0, 1, 3, 4, 2).contiguous() # T, B, H, W, C
        spatial_features = spatial_features.view(T, B, H * W, C)

        pos_2d = self.pos_obj_2d(shape_util=(B, H, W), device=spatial_features.device)
        pos_2d = pos_2d.permute(0, 2, 3, 1).contiguous()
        pos_2d = pos_2d.view(B, H * W, C)

        for spatial_layer in self.spatial_layers:
            out_list = []
            for i in range(len(spatial_features)):

                query = spatial_features[0]
                key = spatial_features[i]

                query, _ = spatial_layer(query=query,
                                        key=key, 
                                        query_pos=pos_2d,
                                        key_pos=pos_2d)
                out_list.append(query)
            spatial_features = torch.stack(out_list, dim=0)
        spatial_features = spatial_features.permute(0, 1, 3, 2).contiguous()
        spatial_features = spatial_features.view(T, B, C, H, W)
        return spatial_features


class FlowTransformerIterative(nn.Module):
    def __init__(self, dim, nhead, num_layers, nb_iters):
        super().__init__()
        self.nb_iters = nb_iters

        spatial_layer = TransformerFlowLayer(d_model=dim, nhead=nhead)
        self.spatial_layers = _get_clones(spatial_layer, num_layers)

        #self_attention_layer = TransformerEncoderLayer(d_model=dim, nhead=nhead)
        #self.self_attention_layers = _get_clones(self_attention_layer, num_layers)

        self.pos_obj_2d = PositionEmbeddingSine2d(num_pos_feats=dim // 2, normalize=True)

        self.pos_1d = nn.Parameter(torch.randn(nb_iters, dim))

        self.reduction = nn.Linear(2 * dim, dim)
    
    
    def forward(self, x1, x2, init):
        '''spatial_features_1: B, C, H, W'''
        
        shape = x1.shape
        B, C, H, W = shape

        init = init.permute(0, 2, 3, 1).contiguous() # B, H, W, C
        init = init.view(B, H * W, C)

        x1 = x1.permute(0, 2, 3, 1).contiguous() # B, H, W, C
        x1 = x1.view(B, H * W, C)

        x2 = x2.permute(0, 2, 3, 1).contiguous() # B, H, W, C
        x2 = x2.view(B, H * W, C)

        pos_2d = self.pos_obj_2d(shape_util=(B, H, W), device=x1.device)
        pos_2d = pos_2d.permute(0, 2, 3, 1).contiguous()
        pos_2d = pos_2d.view(B, H * W, C)

        pos_1d = self.pos_1d[:, None, None].repeat(1, B, H * W, 1) # T, B, H*W, C
        pos = pos_1d + pos_2d[None]
        #pos = pos_2d[None].repeat(self.nb_iters, 1, 1, 1)

        key = torch.cat([x1, x2], dim=-1)
        key = self.reduction(key)

        for i in range(self.nb_iters):
            #for self_attention_layer in self.self_attention_layers:
            #    key = self_attention_layer(key, pos=pos[i])[0]
            
            for spatial_layer in self.spatial_layers:
                init = spatial_layer(query=init,
                                        key=key, 
                                        query_pos=pos[i],
                                        key_pos=pos[i])[0]
                
        init = init.permute(0, 2, 1).contiguous()
        init = init.view(B, C, H, W)
        return init
    

class TransformerFlowEncoderFirst(nn.Module):
    def __init__(self, dim, nhead, num_layers, padding):
        super().__init__()

        self.padding = padding

        spatial_layer = TransformerFlowLayer(d_model=dim, nhead=nhead)
        self.spatial_layers = _get_clones(spatial_layer, num_layers)

        #temporal_layer = TransformerEncoderLayer(d_model=dim, nhead=nhead)
        #self.temporal_layers = _get_clones(temporal_layer, num_layers)

        #seg_layer = TransformerEncoderLayer(d_model=dim, nhead=nhead)
        #self.seg_layers = _get_clones(seg_layer, num_layers)

        #temporal_layer = TransformerEncoderLayer(d_model=dim, nhead=nhead)
        #self.temporal_layers = _get_clones(temporal_layer, num_layers)

        self.pos_obj_2d = PositionEmbeddingSine2d(num_pos_feats=dim // 2, normalize=True)
        #self.pos_obj_1d = PositionEmbeddingSine1d(num_pos_feats=dim, normalize=True)

        #self.token_pos = nn.Parameter(torch.randn(self.nb_tokens, dim))
        #self.temporal_tokens = nn.Parameter(torch.randn(self.nb_tokens, dim))
    
    
    def forward(self, spatial_features):
        '''spatial_features_1: T, B, C, H, W
            pos_1d: B, C, T'''
        
        shape = spatial_features.shape
        T, B, C, H, W = shape

        pos_1d = torch.zeros(size=(T, C), device=spatial_features.device)
        pos_1d = pos_1d[:, None, None, :].repeat(1, B, H * W, 1)

        spatial_features = spatial_features.permute(0, 1, 3, 4, 2).contiguous() # T, B, H, W, C
        spatial_features = spatial_features.view(T, B, H * W, C)

        pos_2d = self.pos_obj_2d(shape_util=(B, H, W), device=spatial_features.device)
        pos_2d = pos_2d.permute(0, 2, 3, 1).contiguous()
        pos_2d = pos_2d.view(B, H * W, C)

        #pos_1d = self.pos_obj_1d(shape_util=(B, T), device=labeled_spatial_features.device) # B, C, T
        #pos_1d = pos_1d.permute(0, 2, 1).contiguous() # B, T, C
        #pos_1d = pos_1d.repeat(1, 2*self.nb_tokens, 1) # B, 2*T*nb_tokens, C
        #pos_1d = self.pos_1d[None, :, :].repeat(B, 1, 1)
        #token_pos = self.token_pos[None, :, :].repeat(T * B, 1, 1)

        #temporal_tokens = self.temporal_tokens[:, None, :, :, :].repeat(1, B, 1, 1, 1) # (T-1), B, 2, nb_tokens, C
        #temporal_tokens = temporal_tokens.view(T * B, 2, self.nb_tokens, C)
        #temporal_features_1 = temporal_tokens[:, 0]
        #temporal_features_2 = temporal_tokens[:, 1]

        #pos = torch.cat([pos_2d, token_pos], dim=1)

        for spatial_layer in self.spatial_layers:
            out_list = []
            for i in range(len(spatial_features)):

                #matplotlib.use('QtAgg')
                #fig, ax = plt.subplots(2, len(spatial_features))
                #past_features = past_features.permute(0, 1, 3, 2).contiguous().view(T - 1, B, C, H, W)
                #spatial_features = spatial_features.permute(0, 1, 3, 2).contiguous().view(T, B, C, H, W)
                #for j in range(len(spatial_features)):
                #    ax[0, j].imshow(spatial_features[j, 0, 0].detach().cpu())
                #    if j < len(spatial_features) - 1:
                #        ax[1, j].imshow(past_features[j, 0, 0].detach().cpu())
                #plt.show()
                #past_features = past_features.permute(0, 1, 3, 4, 2).contiguous().view(T-1, B, H * W, C)
                #spatial_features = spatial_features.permute(0, 1, 3, 4, 2).contiguous().view(T, B, H * W, C)

                #fig, ax = plt.subplots(2, 1)
                #ax[0].imshow(past_features[i, 0, 0].detach().cpu())
                #ax[1].imshow(spatial_features[i, 0, 0].detach().cpu())
                #plt.show()

                current_spatial_feature = spatial_features[i]
                current_pos_1d = pos_1d[i]
                all_spatial_features = spatial_features[0][None]

                if self.padding == 'border':
                    all_spatial_features = torch.nn.functional.pad(all_spatial_features, pad=(0, 0, 0, 0, 0, 0, 1, 1))
                
                myself_pos = pos_2d + current_pos_1d

                key_pos_2d = pos_2d[:, None, :, :].repeat(1, len(all_spatial_features), 1, 1)
                key_pos_2d = key_pos_2d.permute(1, 2, 0, 3).contiguous()
                key_pos_2d = key_pos_2d.view(len(all_spatial_features) * H * W, B, C)
                key_pos_2d = key_pos_2d.permute(1, 0, 2).contiguous()
                context_pos = key_pos_2d

                T_pad = len(all_spatial_features)

                all_spatial_features = all_spatial_features.permute(0, 2, 1, 3).contiguous()
                all_spatial_features = all_spatial_features.view(len(all_spatial_features) * H * W, B, C)
                all_spatial_features = all_spatial_features.permute(1, 0, 2).contiguous()

                current_spatial_feature, weights = spatial_layer(query=current_spatial_feature,
                                                                 key=all_spatial_features, 
                                                                 query_pos=myself_pos,
                                                                 key_pos=context_pos)
                out_list.append(current_spatial_feature)
            spatial_features = torch.stack(out_list, dim=0)

        weights = weights.permute(2, 1, 0).contiguous().mean(1) # T_pad * H * W, B
        weights = weights.view(T_pad, H * W, B).mean(1) # T_pad, B
        if self.padding == 'border':
            weights = weights[1:-1]
        return spatial_features, weights.detach()
    

class TransformerFlowEncoderAllSeparate(nn.Module):
    def __init__(self, dim, nhead, num_layers, padding):
        super().__init__()
        self.padding = padding
        self.num_layers = num_layers

        self_attention_layer = TransformerEncoderLayer(d_model=dim, nhead=nhead)
        self.self_attention_layers = _get_clones(self_attention_layer, num_layers)

        cross_attention_layer = CrossTransformerEncoderLayer(d_model=dim, nhead=nhead)
        self.cross_attention_layers = _get_clones(cross_attention_layer, num_layers)

        cross_attention_layer_all = CrossTransformerEncoderLayer(d_model=dim, nhead=nhead)
        self.cross_attention_layers_all = _get_clones(cross_attention_layer_all, num_layers)

        self.pos_obj_2d = PositionEmbeddingSine2d(num_pos_feats=dim // 2, normalize=True)
    
    
    def forward(self, spatial_features):
        '''spatial_features_1: T, B, C, H, W
            pos_1d: B, C, T'''
        
        shape = spatial_features.shape
        T, B, C, H, W = shape

        pos_1d = torch.zeros(size=(T, C), device=spatial_features.device)
        pos_1d = pos_1d[:, None, None, :].repeat(1, B, H * W, 1)
        #pos_1d = self.pos_1d[:, None, None, :].repeat(1, B, H * W, 1)

        spatial_features = spatial_features.permute(0, 1, 3, 4, 2).contiguous() # T, B, H, W, C
        spatial_features = spatial_features.view(T, B, H * W, C)

        pos_2d = self.pos_obj_2d(shape_util=(B, H, W), device=spatial_features.device)
        pos_2d = pos_2d.permute(0, 2, 3, 1).contiguous()
        pos_2d = pos_2d.view(B, H * W, C)

        #pos_1d = self.pos_obj_1d(shape_util=(B, T), device=labeled_spatial_features.device) # B, C, T
        #pos_1d = pos_1d.permute(0, 2, 1).contiguous() # B, T, C
        #pos_1d = pos_1d.repeat(1, 2*self.nb_tokens, 1) # B, 2*T*nb_tokens, C
        #pos_1d = self.pos_1d[None, :, :].repeat(B, 1, 1)
        #token_pos = self.token_pos[None, :, :].repeat(T * B, 1, 1)

        #temporal_tokens = self.temporal_tokens[:, None, :, :, :].repeat(1, B, 1, 1, 1) # (T-1), B, 2, nb_tokens, C
        #temporal_tokens = temporal_tokens.view(T * B, 2, self.nb_tokens, C)
        #temporal_features_1 = temporal_tokens[:, 0]
        #temporal_features_2 = temporal_tokens[:, 1]

        #pos = torch.cat([pos_2d, token_pos], dim=1)

        for l in range(self.num_layers):
            out_list = []
            for i in range(len(spatial_features)):

                #matplotlib.use('QtAgg')
                #fig, ax = plt.subplots(2, len(spatial_features))
                #past_features = past_features.permute(0, 1, 3, 2).contiguous().view(T - 1, B, C, H, W)
                #spatial_features = spatial_features.permute(0, 1, 3, 2).contiguous().view(T, B, C, H, W)
                #for j in range(len(spatial_features)):
                #    ax[0, j].imshow(spatial_features[j, 0, 0].detach().cpu())
                #    if j < len(spatial_features) - 1:
                #        ax[1, j].imshow(past_features[j, 0, 0].detach().cpu())
                #plt.show()
                #past_features = past_features.permute(0, 1, 3, 4, 2).contiguous().view(T-1, B, H * W, C)
                #spatial_features = spatial_features.permute(0, 1, 3, 4, 2).contiguous().view(T, B, H * W, C)

                #fig, ax = plt.subplots(2, 1)
                #ax[0].imshow(past_features[i, 0, 0].detach().cpu())
                #ax[1].imshow(spatial_features[i, 0, 0].detach().cpu())
                #plt.show()

                current_spatial_feature = spatial_features[i]
                current_pos_1d = pos_1d[i]
                first_spatial_feature = spatial_features[0]
                all_spatial_features = spatial_features
                all_pos_1d = pos_1d

                if self.padding == 'border':
                    all_spatial_features = torch.nn.functional.pad(all_spatial_features, pad=(0, 0, 0, 0, 0, 0, 1, 1))
                    all_pos_1d = torch.nn.functional.pad(all_pos_1d, pad=(0, 0, 0, 0, 0, 0, 1, 1))

                myself_pos = pos_2d + current_pos_1d

                key_pos_2d = pos_2d[:, None, :, :].repeat(1, len(all_spatial_features), 1, 1)
                key_pos_2d = key_pos_2d.permute(1, 2, 0, 3).contiguous()
                key_pos_2d = key_pos_2d.view(len(all_spatial_features) * H * W, B, C)
                key_pos_2d = key_pos_2d.permute(1, 0, 2).contiguous()

                all_pos_1d = all_pos_1d.permute(0, 2, 1, 3).contiguous()
                all_pos_1d = all_pos_1d.view(len(all_spatial_features) * H * W, B, C)
                all_pos_1d = all_pos_1d.permute(1, 0, 2).contiguous()
                context_pos = key_pos_2d + all_pos_1d

                T_pad = len(all_spatial_features)

                all_spatial_features = all_spatial_features.permute(0, 2, 1, 3).contiguous()
                all_spatial_features = all_spatial_features.view(len(all_spatial_features) * H * W, B, C)
                all_spatial_features = all_spatial_features.permute(1, 0, 2).contiguous()

                current_spatial_feature = self.self_attention_layers[l](current_spatial_feature, pos=pos_2d)[0]
                current_spatial_feature = self.cross_attention_layers[l](query=current_spatial_feature, 
                                                                         key=first_spatial_feature, 
                                                                         value=first_spatial_feature, 
                                                                         query_pos=pos_2d, 
                                                                         key_pos=pos_2d)[0]
                current_spatial_feature, weights = self.cross_attention_layers_all[l](query=current_spatial_feature, 
                                                                         key=all_spatial_features, 
                                                                         value=all_spatial_features, 
                                                                         query_pos=myself_pos, 
                                                                         key_pos=context_pos)

                out_list.append(current_spatial_feature)
            spatial_features = torch.stack(out_list, dim=0)
        
        weights = weights.permute(2, 1, 0).contiguous().mean(1) # T_pad * H * W, B
        weights = weights.view(T_pad, H * W, B).mean(1) # T_pad, B
        if self.padding == 'border':
            weights = weights[1:-1]
        return spatial_features, weights.detach()
    

class TransformerFlowEncoderFirstSeparate(nn.Module):
    def __init__(self, dim, nhead, num_layers):
        super().__init__()
        self.num_layers = num_layers

        self_attention_layer = TransformerEncoderLayer(d_model=dim, nhead=nhead)
        self.self_attention_layers = _get_clones(self_attention_layer, num_layers)

        cross_attention_layer = CrossTransformerEncoderLayer(d_model=dim, nhead=nhead)
        self.cross_attention_layers = _get_clones(cross_attention_layer, num_layers)

        cross_attention_layer_all = CrossTransformerEncoderLayer(d_model=dim, nhead=nhead)
        self.cross_attention_layers_all = _get_clones(cross_attention_layer_all, num_layers)

        self.pos_obj_2d = PositionEmbeddingSine2d(num_pos_feats=dim // 2, normalize=True)
    
    
    def forward(self, spatial_features, pos_1d):
        '''spatial_features_1: T, B, C, H, W
            pos_1d: B, C, T'''
        
        shape = spatial_features.shape
        T, B, C, H, W = shape

        pos_1d = pos_1d.permute(2, 0, 1)[:, :, None, :].repeat(1, 1, H * W, 1)

        spatial_features = spatial_features.permute(0, 1, 3, 4, 2).contiguous() # T, B, H, W, C
        spatial_features = spatial_features.view(T, B, H * W, C)

        pos_2d = self.pos_obj_2d(shape_util=(B, H, W), device=spatial_features.device)
        pos_2d = pos_2d.permute(0, 2, 3, 1).contiguous()
        pos_2d = pos_2d.view(B, H * W, C)

        #pos_1d = self.pos_obj_1d(shape_util=(B, T), device=labeled_spatial_features.device) # B, C, T
        #pos_1d = pos_1d.permute(0, 2, 1).contiguous() # B, T, C
        #pos_1d = pos_1d.repeat(1, 2*self.nb_tokens, 1) # B, 2*T*nb_tokens, C
        #pos_1d = self.pos_1d[None, :, :].repeat(B, 1, 1)
        #token_pos = self.token_pos[None, :, :].repeat(T * B, 1, 1)

        #temporal_tokens = self.temporal_tokens[:, None, :, :, :].repeat(1, B, 1, 1, 1) # (T-1), B, 2, nb_tokens, C
        #temporal_tokens = temporal_tokens.view(T * B, 2, self.nb_tokens, C)
        #temporal_features_1 = temporal_tokens[:, 0]
        #temporal_features_2 = temporal_tokens[:, 1]

        #pos = torch.cat([pos_2d, token_pos], dim=1)

        for l in range(self.num_layers):
            out_list = []
            for i in range(len(spatial_features)):

                #matplotlib.use('QtAgg')
                #fig, ax = plt.subplots(2, len(spatial_features))
                #past_features = past_features.permute(0, 1, 3, 2).contiguous().view(T - 1, B, C, H, W)
                #spatial_features = spatial_features.permute(0, 1, 3, 2).contiguous().view(T, B, C, H, W)
                #for j in range(len(spatial_features)):
                #    ax[0, j].imshow(spatial_features[j, 0, 0].detach().cpu())
                #    if j < len(spatial_features) - 1:
                #        ax[1, j].imshow(past_features[j, 0, 0].detach().cpu())
                #plt.show()
                #past_features = past_features.permute(0, 1, 3, 4, 2).contiguous().view(T-1, B, H * W, C)
                #spatial_features = spatial_features.permute(0, 1, 3, 4, 2).contiguous().view(T, B, H * W, C)

                #fig, ax = plt.subplots(2, 1)
                #ax[0].imshow(past_features[i, 0, 0].detach().cpu())
                #ax[1].imshow(spatial_features[i, 0, 0].detach().cpu())
                #plt.show()

                current_spatial_feature = spatial_features[i]
                first_spatial_feature = spatial_features[0]

                current_spatial_feature = self.self_attention_layers[l](current_spatial_feature, pos=pos_2d)[0]
                current_spatial_feature = self.cross_attention_layers[l](query=current_spatial_feature, 
                                                                         key=first_spatial_feature, 
                                                                         value=first_spatial_feature, 
                                                                         query_pos=pos_2d, 
                                                                         key_pos=pos_2d)[0]
                current_spatial_feature, weights = self.cross_attention_layers_all[l](query=current_spatial_feature, 
                                                                         key=first_spatial_feature, 
                                                                         value=first_spatial_feature, 
                                                                         query_pos=pos_2d, 
                                                                         key_pos=pos_2d)

                out_list.append(current_spatial_feature)
            spatial_features = torch.stack(out_list, dim=0)
        
        return spatial_features, weights.detach()
    

class TransformerFlowEncoderWindowFirst(nn.Module):
    def __init__(self, dim, nhead, num_layers, nb_tokens, temporal_kernel_size):
        super().__init__()

        self.nb_tokens = nb_tokens
        self.temporal_kernel_size = temporal_kernel_size
        assert temporal_kernel_size % 2 == 0

        self.half_window_size = (temporal_kernel_size - 1) // 2

        spatial_layer = TransformerFlowLayer(d_model=dim, nhead=nhead)
        self.spatial_layers = _get_clones(spatial_layer, num_layers)

        #temporal_layer = TransformerEncoderLayer(d_model=dim, nhead=nhead)
        #self.temporal_layers = _get_clones(temporal_layer, num_layers)

        #seg_layer = TransformerEncoderLayer(d_model=dim, nhead=nhead)
        #self.seg_layers = _get_clones(seg_layer, num_layers)

        #temporal_layer = TransformerEncoderLayer(d_model=dim, nhead=nhead)
        #self.temporal_layers = _get_clones(temporal_layer, num_layers)

        self.pos_obj_2d = PositionEmbeddingSine2d(num_pos_feats=dim // 2, normalize=True)
        #self.pos_obj_1d = PositionEmbeddingSine1d(num_pos_feats=dim, normalize=True)

        #self.token_pos = nn.Parameter(torch.randn(self.nb_tokens, dim))
        #self.temporal_tokens = nn.Parameter(torch.randn(self.nb_tokens, dim))
    
    
    def forward(self, spatial_features, pos_1d):
        '''spatial_features_1: T, B, C, H, W
            pos_1d: B, C, T'''
        
        shape = spatial_features.shape
        T, B, C, H, W = shape

        pos_1d = pos_1d.permute(2, 0, 1)[:, :, None, :].repeat(1, 1, H * W, 1)

        spatial_features = spatial_features.permute(0, 1, 3, 4, 2).contiguous() # T, B, H, W, C
        spatial_features = spatial_features.view(T, B, H * W, C)

        pos_2d = self.pos_obj_2d(shape_util=(B, H, W), device=spatial_features.device)
        pos_2d = pos_2d.permute(0, 2, 3, 1).contiguous()
        pos_2d = pos_2d.view(B, H * W, C)

        #pos_1d = self.pos_obj_1d(shape_util=(B, T), device=labeled_spatial_features.device) # B, C, T
        #pos_1d = pos_1d.permute(0, 2, 1).contiguous() # B, T, C
        #pos_1d = pos_1d.repeat(1, 2*self.nb_tokens, 1) # B, 2*T*nb_tokens, C
        #pos_1d = self.pos_1d[None, :, :].repeat(B, 1, 1)
        #token_pos = self.token_pos[None, :, :].repeat(T * B, 1, 1)

        #temporal_tokens = self.temporal_tokens[:, None, :, :, :].repeat(1, B, 1, 1, 1) # (T-1), B, 2, nb_tokens, C
        #temporal_tokens = temporal_tokens.view(T * B, 2, self.nb_tokens, C)
        #temporal_features_1 = temporal_tokens[:, 0]
        #temporal_features_2 = temporal_tokens[:, 1]

        #pos = torch.cat([pos_2d, token_pos], dim=1)
        for spatial_layer in self.spatial_layers:
            out_list = []
            for i in range(len(spatial_features)):

                #matplotlib.use('QtAgg')
                #fig, ax = plt.subplots(2, len(spatial_features))
                #past_features = past_features.permute(0, 1, 3, 2).contiguous().view(T - 1, B, C, H, W)
                #spatial_features = spatial_features.permute(0, 1, 3, 2).contiguous().view(T, B, C, H, W)
                #for j in range(len(spatial_features)):
                #    ax[0, j].imshow(spatial_features[j, 0, 0].detach().cpu())
                #    if j < len(spatial_features) - 1:
                #        ax[1, j].imshow(past_features[j, 0, 0].detach().cpu())
                #plt.show()
                #past_features = past_features.permute(0, 1, 3, 4, 2).contiguous().view(T-1, B, H * W, C)
                #spatial_features = spatial_features.permute(0, 1, 3, 4, 2).contiguous().view(T, B, H * W, C)

                #fig, ax = plt.subplots(2, 1)
                #ax[0].imshow(past_features[i, 0, 0].detach().cpu())
                #ax[1].imshow(spatial_features[i, 0, 0].detach().cpu())
                #plt.show()

                start = max(0, i - self.half_window_size)
                stop = min(len(spatial_features), i + self.half_window_size + 1)

                current_spatial_feature = spatial_features[i]
                current_pos_1d = pos_1d[i]
                key_spatial_feature = spatial_features[start:stop]
                key_pos_1d = pos_1d[start:stop]
                key_spatial_feature = torch.nn.functional.pad(key_spatial_feature, pad=(0, 0, 0, 0, 0, 0, max(0, self.half_window_size - i), max(0, self.half_window_size - (len(spatial_features) - i - 1))))
                key_pos_1d = torch.nn.functional.pad(key_pos_1d, pad=(0, 0, 0, 0, 0, 0, max(0, self.half_window_size - i), max(0, self.half_window_size - (len(spatial_features) - i - 1))))
                key_spatial_feature = torch.cat([spatial_features[0][None], key_spatial_feature], dim=0)
                key_pos_1d = torch.cat([pos_1d[0][None], key_pos_1d], dim=0)

                assert len(key_spatial_feature) == self.temporal_kernel_size

                key_pos_1d = key_pos_1d.permute(0, 2, 1, 3).contiguous()
                key_pos_1d = key_pos_1d.view(self.temporal_kernel_size * H * W, B, C)
                key_pos_1d = key_pos_1d.permute(1, 0, 2).contiguous()

                ca_query_pos = pos_2d + current_pos_1d

                key_pos_2d = pos_2d[:, None, :, :].repeat(1, self.temporal_kernel_size, 1, 1)
                key_pos_2d = key_pos_2d.permute(1, 2, 0, 3).contiguous()
                key_pos_2d = key_pos_2d.view(self.temporal_kernel_size * H * W, B, C)
                key_pos_2d = key_pos_2d.permute(1, 0, 2).contiguous()

                key_pos = key_pos_2d + key_pos_1d

                key_spatial_feature = key_spatial_feature.permute(0, 2, 1, 3).contiguous()
                key_spatial_feature = key_spatial_feature.view(self.temporal_kernel_size * H * W, B, C)
                key_spatial_feature = key_spatial_feature.permute(1, 0, 2).contiguous()

                current_spatial_feature = spatial_layer(current_spatial_feature, key_spatial_feature, sa_query_pos=pos_2d, ca_query_pos=ca_query_pos, key_pos=key_pos)
                out_list.append(current_spatial_feature)
            spatial_features = torch.stack(out_list, dim=0)
        return spatial_features
    

class TransformerFlowEncoderDoubleWindow(nn.Module):
    def __init__(self, dim, nhead, num_layers, nb_tokens, temporal_kernel_size):
        super().__init__()

        self.nb_tokens = nb_tokens
        self.temporal_kernel_size = temporal_kernel_size
        self.window_2_size = int(((2/3) * temporal_kernel_size) // 2 * 2 + 1)
        self.window_1_size = temporal_kernel_size - self.window_2_size

        self.half_window_size_2 = (self.window_2_size - 1) // 2

        spatial_layer = TransformerFlowLayer(d_model=dim, nhead=nhead)
        self.spatial_layers = _get_clones(spatial_layer, num_layers)

        #temporal_layer = TransformerEncoderLayer(d_model=dim, nhead=nhead)
        #self.temporal_layers = _get_clones(temporal_layer, num_layers)

        #seg_layer = TransformerEncoderLayer(d_model=dim, nhead=nhead)
        #self.seg_layers = _get_clones(seg_layer, num_layers)

        #temporal_layer = TransformerEncoderLayer(d_model=dim, nhead=nhead)
        #self.temporal_layers = _get_clones(temporal_layer, num_layers)

        self.pos_obj_2d = PositionEmbeddingSine2d(num_pos_feats=dim // 2, normalize=True)
        #self.pos_obj_1d = PositionEmbeddingSine1d(num_pos_feats=dim, normalize=True)

        #self.token_pos = nn.Parameter(torch.randn(self.nb_tokens, dim))
        #self.temporal_tokens = nn.Parameter(torch.randn(self.nb_tokens, dim))
    
    
    def forward(self, spatial_features, pos_1d):
        '''spatial_features_1: T, B, C, H, W
            pos_1d: B, C, T'''
        
        shape = spatial_features.shape
        T, B, C, H, W = shape

        pos_1d = pos_1d.permute(2, 0, 1)[:, :, None, :].repeat(1, 1, H * W, 1)

        spatial_features = spatial_features.permute(0, 1, 3, 4, 2).contiguous() # T, B, H, W, C
        spatial_features = spatial_features.view(T, B, H * W, C)

        pos_2d = self.pos_obj_2d(shape_util=(B, H, W), device=spatial_features.device)
        pos_2d = pos_2d.permute(0, 2, 3, 1).contiguous()
        pos_2d = pos_2d.view(B, H * W, C)

        #pos_1d = self.pos_obj_1d(shape_util=(B, T), device=labeled_spatial_features.device) # B, C, T
        #pos_1d = pos_1d.permute(0, 2, 1).contiguous() # B, T, C
        #pos_1d = pos_1d.repeat(1, 2*self.nb_tokens, 1) # B, 2*T*nb_tokens, C
        #pos_1d = self.pos_1d[None, :, :].repeat(B, 1, 1)
        #token_pos = self.token_pos[None, :, :].repeat(T * B, 1, 1)

        #temporal_tokens = self.temporal_tokens[:, None, :, :, :].repeat(1, B, 1, 1, 1) # (T-1), B, 2, nb_tokens, C
        #temporal_tokens = temporal_tokens.view(T * B, 2, self.nb_tokens, C)
        #temporal_features_1 = temporal_tokens[:, 0]
        #temporal_features_2 = temporal_tokens[:, 1]

        #pos = torch.cat([pos_2d, token_pos], dim=1)
        for spatial_layer in self.spatial_layers:
            out_list = []
            for i in range(len(spatial_features)):

                #matplotlib.use('QtAgg')
                #fig, ax = plt.subplots(2, len(spatial_features))
                #past_features = past_features.permute(0, 1, 3, 2).contiguous().view(T - 1, B, C, H, W)
                #spatial_features = spatial_features.permute(0, 1, 3, 2).contiguous().view(T, B, C, H, W)
                #for j in range(len(spatial_features)):
                #    ax[0, j].imshow(spatial_features[j, 0, 0].detach().cpu())
                #    if j < len(spatial_features) - 1:
                #        ax[1, j].imshow(past_features[j, 0, 0].detach().cpu())
                #plt.show()
                #past_features = past_features.permute(0, 1, 3, 4, 2).contiguous().view(T-1, B, H * W, C)
                #spatial_features = spatial_features.permute(0, 1, 3, 4, 2).contiguous().view(T, B, H * W, C)

                #fig, ax = plt.subplots(2, 1)
                #ax[0].imshow(past_features[i, 0, 0].detach().cpu())
                #ax[1].imshow(spatial_features[i, 0, 0].detach().cpu())
                #plt.show()

                start = max(0, i - self.half_window_size_2)
                stop = min(len(spatial_features), i + self.half_window_size_2 + 1)

                current_spatial_feature = spatial_features[i]
                current_pos_1d = pos_1d[i]
                key_spatial_feature = spatial_features[start:stop]
                key_pos_1d = pos_1d[start:stop]
                key_spatial_feature = torch.nn.functional.pad(key_spatial_feature, pad=(0, 0, 0, 0, 0, 0, max(0, self.half_window_size_2 - i), max(0, self.half_window_size_2 - (len(spatial_features) - i - 1))))
                key_pos_1d = torch.nn.functional.pad(key_pos_1d, pad=(0, 0, 0, 0, 0, 0, max(0, self.half_window_size_2 - i), max(0, self.half_window_size_2 - (len(spatial_features) - i - 1))))
                key_spatial_feature = torch.cat([spatial_features[:self.window_1_size], key_spatial_feature], dim=0)
                key_pos_1d = torch.cat([pos_1d[:self.window_1_size], key_pos_1d], dim=0)

                assert len(key_spatial_feature) == self.temporal_kernel_size

                key_pos_1d = key_pos_1d.permute(0, 2, 1, 3).contiguous()
                key_pos_1d = key_pos_1d.view(self.temporal_kernel_size * H * W, B, C)
                key_pos_1d = key_pos_1d.permute(1, 0, 2).contiguous()

                ca_query_pos = pos_2d + current_pos_1d

                key_pos_2d = pos_2d[:, None, :, :].repeat(1, self.temporal_kernel_size, 1, 1)
                key_pos_2d = key_pos_2d.permute(1, 2, 0, 3).contiguous()
                key_pos_2d = key_pos_2d.view(self.temporal_kernel_size * H * W, B, C)
                key_pos_2d = key_pos_2d.permute(1, 0, 2).contiguous()

                key_pos = key_pos_2d + key_pos_1d

                key_spatial_feature = key_spatial_feature.permute(0, 2, 1, 3).contiguous()
                key_spatial_feature = key_spatial_feature.view(self.temporal_kernel_size * H * W, B, C)
                key_spatial_feature = key_spatial_feature.permute(1, 0, 2).contiguous()

                current_spatial_feature = spatial_layer(current_spatial_feature, key_spatial_feature, sa_query_pos=pos_2d, ca_query_pos=ca_query_pos, key_pos=key_pos)
                out_list.append(current_spatial_feature)
            spatial_features = torch.stack(out_list, dim=0)
        return spatial_features


class TransformerFlowEncoderIterative(nn.Module):
    def __init__(self, dim, nhead, num_layers, nb_tokens, temporal_kernel_size):
        super().__init__()

        self.nb_tokens = nb_tokens
        self.temporal_kernel_size = temporal_kernel_size
        assert temporal_kernel_size % 2 == 1

        self.half_window_size = temporal_kernel_size // 2

        spatial_layer = TransformerFlowLayer(d_model=dim, nhead=nhead)
        self.spatial_layers = _get_clones(spatial_layer, num_layers)

        #temporal_layer = TransformerEncoderLayer(d_model=dim, nhead=nhead)
        #self.temporal_layers = _get_clones(temporal_layer, num_layers)

        #seg_layer = TransformerEncoderLayer(d_model=dim, nhead=nhead)
        #self.seg_layers = _get_clones(seg_layer, num_layers)

        #temporal_layer = TransformerEncoderLayer(d_model=dim, nhead=nhead)
        #self.temporal_layers = _get_clones(temporal_layer, num_layers)

        self.pos_obj_2d = PositionEmbeddingSine2d(num_pos_feats=dim // 2, normalize=True)
        #self.pos_obj_1d = PositionEmbeddingSine1d(num_pos_feats=dim, normalize=True)

        #self.token_pos = nn.Parameter(torch.randn(self.nb_tokens, dim))
        #self.temporal_tokens = nn.Parameter(torch.randn(self.nb_tokens, dim))
    
    
    def forward(self, spatial_features, pos_1d):
        '''spatial_features_1: T, B, C, H, W
            pos_1d: B, C, T'''
        
        shape = spatial_features.shape
        T, B, C, H, W = shape

        pos_1d = pos_1d.permute(2, 0, 1)[:, :, None, :].repeat(1, 1, H * W, 1)

        spatial_features = spatial_features.permute(0, 1, 3, 4, 2).contiguous() # T, B, H, W, C
        spatial_features = spatial_features.view(T, B, H * W, C)

        pos_2d = self.pos_obj_2d(shape_util=(B, H, W), device=spatial_features.device)
        pos_2d = pos_2d.permute(0, 2, 3, 1).contiguous()
        pos_2d = pos_2d.view(B, H * W, C)

        #pos_1d = self.pos_obj_1d(shape_util=(B, T), device=labeled_spatial_features.device) # B, C, T
        #pos_1d = pos_1d.permute(0, 2, 1).contiguous() # B, T, C
        #pos_1d = pos_1d.repeat(1, 2*self.nb_tokens, 1) # B, 2*T*nb_tokens, C
        #pos_1d = self.pos_1d[None, :, :].repeat(B, 1, 1)
        #token_pos = self.token_pos[None, :, :].repeat(T * B, 1, 1)

        #temporal_tokens = self.temporal_tokens[:, None, :, :, :].repeat(1, B, 1, 1, 1) # (T-1), B, 2, nb_tokens, C
        #temporal_tokens = temporal_tokens.view(T * B, 2, self.nb_tokens, C)
        #temporal_features_1 = temporal_tokens[:, 0]
        #temporal_features_2 = temporal_tokens[:, 1]

        #pos = torch.cat([pos_2d, token_pos], dim=1)

        for spatial_layer in self.spatial_layers:
            past_features = torch.zeros_like(spatial_features) # T, B, H*W, C
            for i in range(len(spatial_features)):

                #matplotlib.use('QtAgg')
                #fig, ax = plt.subplots(2, len(spatial_features))
                #past_features = past_features.permute(0, 1, 3, 2).contiguous().view(T - 1, B, C, H, W)
                #spatial_features = spatial_features.permute(0, 1, 3, 2).contiguous().view(T, B, C, H, W)
                #for j in range(len(spatial_features)):
                #    ax[0, j].imshow(spatial_features[j, 0, 0].detach().cpu())
                #    if j < len(spatial_features) - 1:
                #        ax[1, j].imshow(past_features[j, 0, 0].detach().cpu())
                #plt.show()
                #past_features = past_features.permute(0, 1, 3, 4, 2).contiguous().view(T-1, B, H * W, C)
                #spatial_features = spatial_features.permute(0, 1, 3, 4, 2).contiguous().view(T, B, H * W, C)

                #fig, ax = plt.subplots(2, 1)
                #ax[0].imshow(past_features[i, 0, 0].detach().cpu())
                #ax[1].imshow(spatial_features[i, 0, 0].detach().cpu())
                #plt.show()

                start = max(0, i - self.temporal_kernel_size)
                stop = i

                current_spatial_feature = spatial_features[i]
                current_pos_1d = pos_1d[i]
                if i == 0:
                    previous_spatial_feature = torch.zeros_like(current_spatial_feature).unsqueeze(0).repeat(self.temporal_kernel_size, 1, 1, 1)
                    previous_pos_1d = torch.zeros_like(current_pos_1d).unsqueeze(0).repeat(self.temporal_kernel_size, 1, 1, 1)
                else:
                    previous_spatial_feature = past_features[start:stop]
                    previous_spatial_feature = torch.nn.functional.pad(previous_spatial_feature, pad=(0, 0, 0, 0, 0, 0, max(0, self.temporal_kernel_size - len(previous_spatial_feature)), 0))
                    previous_pos_1d = pos_1d[start:stop]
                    previous_pos_1d = torch.nn.functional.pad(previous_pos_1d, pad=(0, 0, 0, 0, 0, 0, max(0, self.temporal_kernel_size - len(previous_pos_1d)), 0))

                previous_pos_1d = previous_pos_1d.permute(0, 2, 1, 3).contiguous()
                previous_pos_1d = previous_pos_1d.view(self.temporal_kernel_size * H * W, B, C)
                previous_pos_1d = previous_pos_1d.permute(1, 0, 2).contiguous()

                ca_query_pos = pos_2d + current_pos_1d

                key_pos_2d = pos_2d[:, None, :, :].repeat(1, self.temporal_kernel_size, 1, 1)
                key_pos_2d = key_pos_2d.permute(1, 2, 0, 3).contiguous()
                key_pos_2d = key_pos_2d.view(self.temporal_kernel_size * H * W, B, C)
                key_pos_2d = key_pos_2d.permute(1, 0, 2).contiguous()
                key_pos = key_pos_2d + previous_pos_1d

                previous_spatial_feature = previous_spatial_feature.permute(0, 2, 1, 3).contiguous()
                previous_spatial_feature = previous_spatial_feature.view(self.temporal_kernel_size * H * W, B, C)
                previous_spatial_feature = previous_spatial_feature.permute(1, 0, 2).contiguous()

                current_spatial_feature = spatial_layer(current_spatial_feature, previous_spatial_feature, sa_query_pos=pos_2d, ca_query_pos=ca_query_pos, key_pos=key_pos)
                past_features[i] = current_spatial_feature
            spatial_features = past_features
        return spatial_features
            

class TransformerFlowEncoderIterativeMiddle(nn.Module):
    def __init__(self, dim, nhead, num_layers, nb_tokens, temporal_kernel_size):
        super().__init__()

        self.nb_tokens = nb_tokens
        self.temporal_kernel_size = temporal_kernel_size
        assert temporal_kernel_size % 2 == 1

        self.half_window_size = temporal_kernel_size // 2

        spatial_layer = TransformerFlowLayer(d_model=dim, nhead=nhead)
        self.spatial_layers = _get_clones(spatial_layer, num_layers)

        #seg_layer = TransformerEncoderLayer(d_model=dim, nhead=nhead)
        #self.seg_layers = _get_clones(seg_layer, num_layers)

        #temporal_layer = TransformerEncoderLayer(d_model=dim, nhead=nhead)
        #self.temporal_layers = _get_clones(temporal_layer, num_layers)

        self.pos_obj_2d = PositionEmbeddingSine2d(num_pos_feats=dim // 2, normalize=True)
        #self.pos_obj_1d = PositionEmbeddingSine1d(num_pos_feats=dim, normalize=True)

        #self.token_pos = nn.Parameter(torch.randn(self.nb_tokens, dim))
        #self.temporal_tokens = nn.Parameter(torch.randn(self.nb_tokens, dim))
    
    
    def forward(self, spatial_features, pos_1d):
        '''spatial_features_1: T, B, C, H, W
            pos_1d: B, C, T'''
        
        shape = spatial_features.shape
        T, B, C, H, W = shape

        pos_1d = pos_1d.permute(2, 0, 1)[:, :, None, :].repeat(1, 1, H * W, 1)

        spatial_features = spatial_features.permute(0, 1, 3, 4, 2).contiguous() # T, B, H, W, C
        spatial_features = spatial_features.view(T, B, H * W, C)

        pos_2d = self.pos_obj_2d(shape_util=(B, H, W), device=spatial_features.device)
        pos_2d = pos_2d.permute(0, 2, 3, 1).contiguous()
        pos_2d = pos_2d.view(B, H * W, C)

        #pos_1d = self.pos_obj_1d(shape_util=(B, T), device=labeled_spatial_features.device) # B, C, T
        #pos_1d = pos_1d.permute(0, 2, 1).contiguous() # B, T, C
        #pos_1d = pos_1d.repeat(1, 2*self.nb_tokens, 1) # B, 2*T*nb_tokens, C
        #pos_1d = self.pos_1d[None, :, :].repeat(B, 1, 1)
        #token_pos = self.token_pos[None, :, :].repeat(T * B, 1, 1)

        #temporal_tokens = self.temporal_tokens[:, None, :, :, :].repeat(1, B, 1, 1, 1) # (T-1), B, 2, nb_tokens, C
        #temporal_tokens = temporal_tokens.view(T * B, 2, self.nb_tokens, C)
        #temporal_features_1 = temporal_tokens[:, 0]
        #temporal_features_2 = temporal_tokens[:, 1]

        #pos = torch.cat([pos_2d, token_pos], dim=1)
        for spatial_layer in self.spatial_layers:
            past_features = torch.zeros_like(spatial_features) # T, B, H*W, C
            for i in range(len(spatial_features)):

                #matplotlib.use('QtAgg')
                #fig, ax = plt.subplots(2, len(spatial_features))
                #past_features = past_features.permute(0, 1, 3, 2).contiguous().view(T - 1, B, C, H, W)
                #spatial_features = spatial_features.permute(0, 1, 3, 2).contiguous().view(T, B, C, H, W)
                #for j in range(len(spatial_features)):
                #    ax[0, j].imshow(spatial_features[j, 0, 0].detach().cpu())
                #    if j < len(spatial_features) - 1:
                #        ax[1, j].imshow(past_features[j, 0, 0].detach().cpu())
                #plt.show()
                #past_features = past_features.permute(0, 1, 3, 4, 2).contiguous().view(T-1, B, H * W, C)
                #spatial_features = spatial_features.permute(0, 1, 3, 4, 2).contiguous().view(T, B, H * W, C)

                #fig, ax = plt.subplots(2, 1)
                #ax[0].imshow(past_features[i, 0, 0].detach().cpu())
                #ax[1].imshow(spatial_features[i, 0, 0].detach().cpu())
                #plt.show()

                start = max(0, i - self.half_window_size)
                stop = min(len(spatial_features), i + self.half_window_size + 1)

                current_spatial_feature = spatial_features[i]
                current_pos_1d = pos_1d[i]
                if i == 0:
                    previous_spatial_feature = torch.zeros_like(current_spatial_feature).unsqueeze(0).repeat(self.half_window_size, 1, 1, 1)
                    previous_pos_1d = torch.zeros_like(current_pos_1d).unsqueeze(0).repeat(self.half_window_size, 1, 1, 1)
                else:
                    previous_spatial_feature = past_features[start:i]
                    previous_spatial_feature = torch.nn.functional.pad(previous_spatial_feature, pad=(0, 0, 0, 0, 0, 0, max(0, self.half_window_size - len(previous_spatial_feature)), 0))
                    previous_pos_1d = pos_1d[start:i]
                    previous_pos_1d = torch.nn.functional.pad(previous_pos_1d, pad=(0, 0, 0, 0, 0, 0, max(0, self.half_window_size - len(previous_pos_1d)), 0))
                
                if i == len(spatial_features) - 1:
                    next_spatial_feature = torch.zeros_like(current_spatial_feature).unsqueeze(0).repeat(self.half_window_size, 1, 1, 1)
                    next_pos_1d = torch.zeros_like(current_pos_1d).unsqueeze(0).repeat(self.half_window_size, 1, 1, 1)
                else:
                    next_spatial_feature = spatial_features[i:stop]
                    next_spatial_feature = torch.nn.functional.pad(next_spatial_feature, pad=(0, 0, 0, 0, 0, 0, 0, max(0, self.half_window_size - len(next_spatial_feature))))
                    next_pos_1d = pos_1d[i:stop]
                    next_pos_1d = torch.nn.functional.pad(next_pos_1d, pad=(0, 0, 0, 0, 0, 0, 0, max(0, self.half_window_size - len(next_pos_1d))))

                key_pos_1d = torch.cat([previous_pos_1d, next_pos_1d], dim=0)
                key_spatial_feature = torch.cat([previous_spatial_feature, next_spatial_feature], dim=0)

                key_pos_1d = key_pos_1d.permute(0, 2, 1, 3).contiguous()
                key_pos_1d = key_pos_1d.view(self.temporal_kernel_size * H * W, B, C)
                key_pos_1d = key_pos_1d.permute(1, 0, 2).contiguous()

                ca_query_pos = pos_2d + current_pos_1d

                key_pos_2d = pos_2d[:, None, :, :].repeat(1, self.temporal_kernel_size, 1, 1)
                key_pos_2d = key_pos_2d.permute(1, 2, 0, 3).contiguous()
                key_pos_2d = key_pos_2d.view(self.temporal_kernel_size * H * W, B, C)
                key_pos_2d = key_pos_2d.permute(1, 0, 2).contiguous()

                key_pos = key_pos_2d + key_pos_1d

                key_spatial_feature = key_spatial_feature.permute(0, 2, 1, 3).contiguous()
                key_spatial_feature = key_spatial_feature.view(self.temporal_kernel_size * H * W, B, C)
                key_spatial_feature = key_spatial_feature.permute(1, 0, 2).contiguous()

                current_spatial_feature = spatial_layer(current_spatial_feature, key_spatial_feature, sa_query_pos=pos_2d, ca_query_pos=ca_query_pos, key_pos=key_pos)
                past_features[i] = current_spatial_feature
            spatial_features = past_features
        return spatial_features



class TransformerFlowEncoderWindow(nn.Module):
    def __init__(self, dim, nhead, num_layers, nb_tokens, temporal_kernel_size):
        super().__init__()

        self.nb_tokens = nb_tokens
        self.temporal_kernel_size = temporal_kernel_size
        assert temporal_kernel_size % 2 == 1

        self.half_window_size = temporal_kernel_size // 2

        spatial_layer = TransformerFlowLayer(d_model=dim, nhead=nhead)
        self.spatial_layers = _get_clones(spatial_layer, num_layers)

        #seg_layer = TransformerEncoderLayer(d_model=dim, nhead=nhead)
        #self.seg_layers = _get_clones(seg_layer, num_layers)

        #temporal_layer = TransformerEncoderLayer(d_model=dim, nhead=nhead)
        #self.temporal_layers = _get_clones(temporal_layer, num_layers)

        self.pos_obj_2d = PositionEmbeddingSine2d(num_pos_feats=dim // 2, normalize=True)
        #self.pos_obj_1d = PositionEmbeddingSine1d(num_pos_feats=dim, normalize=True)

        #self.token_pos = nn.Parameter(torch.randn(self.nb_tokens, dim))
        #self.temporal_tokens = nn.Parameter(torch.randn(self.nb_tokens, dim))
    
    
    def forward(self, spatial_features, pos_1d):
        '''spatial_features_1: T, B, C, H, W
            pos_1d: B, C, T'''
        
        shape = spatial_features.shape
        T, B, C, H, W = shape

        pos_1d = pos_1d.permute(2, 0, 1)[:, :, None, :].repeat(1, 1, H * W, 1)

        spatial_features = spatial_features.permute(0, 1, 3, 4, 2).contiguous() # T, B, H, W, C
        spatial_features = spatial_features.view(T, B, H * W, C)

        pos_2d = self.pos_obj_2d(shape_util=(B, H, W), device=spatial_features.device)
        pos_2d = pos_2d.permute(0, 2, 3, 1).contiguous()
        pos_2d = pos_2d.view(B, H * W, C)

        #pos_1d = self.pos_obj_1d(shape_util=(B, T), device=labeled_spatial_features.device) # B, C, T
        #pos_1d = pos_1d.permute(0, 2, 1).contiguous() # B, T, C
        #pos_1d = pos_1d.repeat(1, 2*self.nb_tokens, 1) # B, 2*T*nb_tokens, C
        #pos_1d = self.pos_1d[None, :, :].repeat(B, 1, 1)
        #token_pos = self.token_pos[None, :, :].repeat(T * B, 1, 1)

        #temporal_tokens = self.temporal_tokens[:, None, :, :, :].repeat(1, B, 1, 1, 1) # (T-1), B, 2, nb_tokens, C
        #temporal_tokens = temporal_tokens.view(T * B, 2, self.nb_tokens, C)
        #temporal_features_1 = temporal_tokens[:, 0]
        #temporal_features_2 = temporal_tokens[:, 1]

        #pos = torch.cat([pos_2d, token_pos], dim=1)
        for spatial_layer in self.spatial_layers:
            out_list = []
            for i in range(len(spatial_features)):

                #matplotlib.use('QtAgg')
                #fig, ax = plt.subplots(2, len(spatial_features))
                #past_features = past_features.permute(0, 1, 3, 2).contiguous().view(T - 1, B, C, H, W)
                #spatial_features = spatial_features.permute(0, 1, 3, 2).contiguous().view(T, B, C, H, W)
                #for j in range(len(spatial_features)):
                #    ax[0, j].imshow(spatial_features[j, 0, 0].detach().cpu())
                #    if j < len(spatial_features) - 1:
                #        ax[1, j].imshow(past_features[j, 0, 0].detach().cpu())
                #plt.show()
                #past_features = past_features.permute(0, 1, 3, 4, 2).contiguous().view(T-1, B, H * W, C)
                #spatial_features = spatial_features.permute(0, 1, 3, 4, 2).contiguous().view(T, B, H * W, C)

                #fig, ax = plt.subplots(2, 1)
                #ax[0].imshow(past_features[i, 0, 0].detach().cpu())
                #ax[1].imshow(spatial_features[i, 0, 0].detach().cpu())
                #plt.show()

                start = max(0, i - self.half_window_size)
                stop = min(len(spatial_features), i + self.half_window_size + 1)

                current_spatial_feature = spatial_features[i]
                current_pos_1d = pos_1d[i]
                key_spatial_feature = spatial_features[start:stop]
                key_pos_1d = pos_1d[start:stop]
                key_spatial_feature = torch.nn.functional.pad(key_spatial_feature, pad=(0, 0, 0, 0, 0, 0, max(0, self.half_window_size - i), max(0, self.half_window_size - (len(spatial_features) - i - 1))))
                key_pos_1d = torch.nn.functional.pad(key_pos_1d, pad=(0, 0, 0, 0, 0, 0, max(0, self.half_window_size - i), max(0, self.half_window_size - (len(spatial_features) - i - 1))))

                key_pos_1d = key_pos_1d.permute(0, 2, 1, 3).contiguous()
                key_pos_1d = key_pos_1d.view(self.temporal_kernel_size * H * W, B, C)
                key_pos_1d = key_pos_1d.permute(1, 0, 2).contiguous()

                ca_query_pos = pos_2d + current_pos_1d

                key_pos_2d = pos_2d[:, None, :, :].repeat(1, self.temporal_kernel_size, 1, 1)
                key_pos_2d = key_pos_2d.permute(1, 2, 0, 3).contiguous()
                key_pos_2d = key_pos_2d.view(self.temporal_kernel_size * H * W, B, C)
                key_pos_2d = key_pos_2d.permute(1, 0, 2).contiguous()

                key_pos = key_pos_2d + key_pos_1d

                key_spatial_feature = key_spatial_feature.permute(0, 2, 1, 3).contiguous()
                key_spatial_feature = key_spatial_feature.view(self.temporal_kernel_size * H * W, B, C)
                key_spatial_feature = key_spatial_feature.permute(1, 0, 2).contiguous()

                current_spatial_feature = spatial_layer(current_spatial_feature, key_spatial_feature, sa_query_pos=pos_2d, ca_query_pos=ca_query_pos, key_pos=key_pos)
                out_list.append(current_spatial_feature)
            spatial_features = torch.stack(out_list, dim=0)
        return spatial_features
    


#class TransformerFlowEncoderIterative(nn.Module):
#    def __init__(self, dim, nhead, num_layers, nb_tokens, lookback):
#        super().__init__()
#
#        self.nb_tokens = nb_tokens
#        self.lookback = lookback
#
#        spatial_layer = TransformerFlowLayer(d_model=dim, nhead=nhead)
#        self.spatial_layers = _get_clones(spatial_layer, num_layers)
#
#        self.conv_1d = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=self.lookback + 1, padding='same')
#
#        #temporal_layer = TransformerEncoderLayer(d_model=dim, nhead=nhead)
#        #self.temporal_layers = _get_clones(temporal_layer, num_layers)
#
#        #seg_layer = TransformerEncoderLayer(d_model=dim, nhead=nhead)
#        #self.seg_layers = _get_clones(seg_layer, num_layers)
#
#        #temporal_layer = TransformerEncoderLayer(d_model=dim, nhead=nhead)
#        #self.temporal_layers = _get_clones(temporal_layer, num_layers)
#
#        self.pos_obj_2d = PositionEmbeddingSine2d(num_pos_feats=dim // 2, normalize=True)
#        #self.pos_obj_1d = PositionEmbeddingSine1d(num_pos_feats=dim, normalize=True)
#
#        #self.token_pos = nn.Parameter(torch.randn(self.nb_tokens, dim))
#        #self.temporal_tokens = nn.Parameter(torch.randn(self.nb_tokens, dim))
#    
#    
#    def forward(self, spatial_features):
#        '''spatial_features_1: T, B, C, H, W
#            pos_1d: B, C, T'''
#        
#        shape = spatial_features.shape
#        T, B, C, H, W = shape
#
#        #pos_1d = pos_1d.permute(2, 0, 1)[:, :, None, :].repeat(1, 1, H * W, 1)
#
#        spatial_features = spatial_features.permute(0, 1, 3, 4, 2).contiguous() # T, B, H, W, C
#        spatial_features = spatial_features.view(T, B, H * W, C)
#
#        pos_2d = self.pos_obj_2d(shape_util=(B, H, W), device=spatial_features.device)
#        pos_2d = pos_2d.permute(0, 2, 3, 1).contiguous()
#        pos_2d = pos_2d.view(B, H * W, C)
#
#        #pos_1d = self.pos_obj_1d(shape_util=(B, T), device=labeled_spatial_features.device) # B, C, T
#        #pos_1d = pos_1d.permute(0, 2, 1).contiguous() # B, T, C
#        #pos_1d = pos_1d.repeat(1, 2*self.nb_tokens, 1) # B, 2*T*nb_tokens, C
#        #pos_1d = self.pos_1d[None, :, :].repeat(B, 1, 1)
#        #token_pos = self.token_pos[None, :, :].repeat(T * B, 1, 1)
#
#        #temporal_tokens = self.temporal_tokens[:, None, :, :, :].repeat(1, B, 1, 1, 1) # (T-1), B, 2, nb_tokens, C
#        #temporal_tokens = temporal_tokens.view(T * B, 2, self.nb_tokens, C)
#        #temporal_features_1 = temporal_tokens[:, 0]
#        #temporal_features_2 = temporal_tokens[:, 1]
#
#        #pos = torch.cat([pos_2d, token_pos], dim=1)
#
#        #past_pos_1d = pos_1d[:-1]
#        for spatial_layer in self.spatial_layers:
#            past_features = torch.zeros_like(spatial_features)[:-1] # T - 1, B, H*W, C
#            for i in range(len(spatial_features)):
#
#                #matplotlib.use('QtAgg')
#                #fig, ax = plt.subplots(2, len(spatial_features))
#                #past_features = past_features.permute(0, 1, 3, 2).contiguous().view(T - 1, B, C, H, W)
#                #spatial_features = spatial_features.permute(0, 1, 3, 2).contiguous().view(T, B, C, H, W)
#                #for j in range(len(spatial_features)):
#                #    ax[0, j].imshow(spatial_features[j, 0, 0].detach().cpu())
#                #    if j < len(spatial_features) - 1:
#                #        ax[1, j].imshow(past_features[j, 0, 0].detach().cpu())
#                #plt.show()
#                #past_features = past_features.permute(0, 1, 3, 4, 2).contiguous().view(T-1, B, H * W, C)
#                #spatial_features = spatial_features.permute(0, 1, 3, 4, 2).contiguous().view(T, B, H * W, C)
#
#                #fig, ax = plt.subplots(2, 1)
#                #ax[0].imshow(past_features[i, 0, 0].detach().cpu())
#                #ax[1].imshow(spatial_features[i, 0, 0].detach().cpu())
#                #plt.show()
#
#                start = max(0, i - self.lookback)
#                stop = i
#
#                current_spatial_feature = spatial_features[i]
#                #current_pos_1d = pos_1d[i]
#                if i == 0:
#                    previous_spatial_feature = torch.zeros_like(current_spatial_feature).unsqueeze(0).repeat(self.lookback, 1, 1, 1)
#                    #previous_pos_1d = torch.zeros_like(current_pos_1d).unsqueeze(0).repeat(self.lookback, 1, 1, 1)
#                else:
#                    previous_spatial_feature = past_features[start:stop]
#                    previous_spatial_feature = torch.nn.functional.pad(previous_spatial_feature, pad=(0, 0, 0, 0, 0, 0, max(0, self.lookback - len(previous_spatial_feature)), 0))
#                    #previous_pos_1d = past_pos_1d[start:stop]
#                    #previous_pos_1d = torch.nn.functional.pad(previous_pos_1d, pad=(0, 0, 0, 0, 0, 0, max(0, self.lookback - len(previous_pos_1d)), 0))
#
#                all_features = torch.cat([previous_spatial_feature, current_spatial_feature.unsqueeze(0)], dim=0) # lookback + 1, B, H*W, C
#                all_features = all_features.permute(1, 3, 0, 2).mean(-1) # B, C, lookback + 1
#                #print(torch.any(previous_spatial_feature.flatten(1, -1), dim=-1))
#                pos_1d = self.conv_1d(all_features)
#                current_pos_1d = pos_1d[:, :, -1]
#                previous_pos_1d = pos_1d[:, :, :-1]
#
#                current_pos_1d = current_pos_1d[:, None, :].repeat(1, H * W, 1)
#                
#                previous_pos_1d = previous_pos_1d.permute(2, 0, 1).contiguous()
#                previous_pos_1d = previous_pos_1d[:, None, :, :].repeat(1, H * W, 1, 1)
#                previous_pos_1d = previous_pos_1d.view(len(previous_pos_1d) * H * W, B, C)
#                previous_pos_1d = previous_pos_1d.permute(1, 0, 2).contiguous()
#
#                query_pos = pos_2d + current_pos_1d
#
#                key_pos_2d = pos_2d[:, None, :, :].repeat(1, len(previous_spatial_feature), 1, 1)
#                key_pos_2d = key_pos_2d.permute(1, 2, 0, 3).contiguous()
#                key_pos_2d = key_pos_2d.view(len(previous_spatial_feature) * H * W, B, C)
#                key_pos_2d = key_pos_2d.permute(1, 0, 2).contiguous()
#                key_pos = key_pos_2d + previous_pos_1d
#
#                previous_spatial_feature = previous_spatial_feature.permute(0, 2, 1, 3).contiguous()
#                previous_spatial_feature = previous_spatial_feature.view(len(previous_spatial_feature) * H * W, B, C)
#                previous_spatial_feature = previous_spatial_feature.permute(1, 0, 2).contiguous()
#
#                current_spatial_feature = spatial_layer(current_spatial_feature, previous_spatial_feature, query_pos=query_pos, key_pos=key_pos)
#                if i == len(spatial_features) - 1:
#                    spatial_features = torch.cat([past_features, current_spatial_feature.unsqueeze(0)], dim=0)
#                else:
#                    past_features[i] = current_spatial_feature
#        return spatial_features
    


class TransformerFlowEncoderContext(nn.Module):
    def __init__(self, dim, nhead, num_layers, video_length, nb_tokens):
        super().__init__()

        self.nb_tokens = nb_tokens

        spatial_layer = TransformerFlowLayerContext(d_model=dim, nhead=nhead)
        self.spatial_layers = _get_clones(spatial_layer, num_layers)

        context_layer = TransformerEncoderLayer(d_model=dim, nhead=nhead)
        self.context_layers = _get_clones(context_layer, num_layers)

        #temporal_layer = TransformerEncoderLayer(d_model=dim, nhead=nhead)
        #self.temporal_layers = _get_clones(temporal_layer, num_layers)

        self.pos_obj_2d = PositionEmbeddingSine2d(num_pos_feats=dim // 2, normalize=True)
        #self.pos_obj_1d = PositionEmbeddingSine1d(num_pos_feats=dim, normalize=True)

        self.pos_1d = nn.Parameter(torch.randn(video_length, dim))

        #self.token_pos = nn.Parameter(torch.randn(self.nb_tokens, dim))
        #self.temporal_tokens = nn.Parameter(torch.randn(video_length - 1, 2, self.nb_tokens, dim))
    
    
    def forward(self, spatial_features_1, spatial_features_2, context_1):
        '''spatial_features_1: T-1, B, C, H, W
            spatial_features_2: T-1, B, C, H, W'''
        
        shape = spatial_features_1.shape
        T, B, C, H, W = shape

        context_1 = context_1[None].repeat(T, 1, 1, 1, 1)
        context_1 = context_1.permute(0, 1, 3, 4, 2).contiguous() # T, B, H, W, C
        context_1 = context_1.view(T, B, H * W, C).view(T * B, H * W, C)

        #context_2 = context_2[None].repeat(T, 1, 1, 1, 1)
        #context_2 = context_2.permute(0, 1, 3, 4, 2).contiguous() # T, B, H, W, C
        #context_2 = context_2.view(T, B, H * W, C).view(T * B, H * W, C)

        spatial_features_1 = spatial_features_1.permute(0, 1, 3, 4, 2).contiguous() # T, B, H, W, C
        spatial_features_1 = spatial_features_1.view(T, B, H * W, C).view(T * B, H * W, C)

        spatial_features_2 = spatial_features_2.permute(0, 1, 3, 4, 2).contiguous() # T, B, H, W, C
        spatial_features_2 = spatial_features_2.view(T, B, H * W, C).view(T * B, H * W, C)

        pos_2d = self.pos_obj_2d(shape_util=(B, H, W), device=spatial_features_1.device)
        pos_2d = pos_2d.permute(0, 2, 3, 1).contiguous()
        pos_2d = pos_2d.view(B, H * W, C).repeat(T, 1, 1)

        pos_1d = self.pos_1d[:, None, None, :].repeat(1, B, H*W, 1) # T, B, H*W, C
        pos_1d_1 = pos_1d[:-1]
        pos_1d_2 = pos_1d[1:]
        pos_1d_1 = pos_1d_1.view(T * B, H * W, C)
        pos_1d_2 = pos_1d_2.view(T * B, H * W, C)

        #pos_1d = self.pos_obj_1d(shape_util=(B, T), device=labeled_spatial_features.device) # B, C, T
        #pos_1d = pos_1d.permute(0, 2, 1).contiguous() # B, T, C
        #pos_1d = pos_1d.repeat(1, 2*self.nb_tokens, 1) # B, 2*T*nb_tokens, C
        #pos_1d = self.pos_1d[None, :, :].repeat(B, 1, 1)
        #token_pos = self.token_pos[None, :, :].repeat(T * B, 1, 1)

        #temporal_tokens = self.temporal_tokens[:, None, :, :, :].repeat(1, B, 1, 1, 1) # (T-1), B, 2, nb_tokens, C
        #temporal_tokens = temporal_tokens.view(T * B, 2, self.nb_tokens, C)
        #feature_1_tokens = temporal_tokens[:, 0]
        #feature_2_tokens = temporal_tokens[:, 1]

        #pos = torch.cat([pos_2d, token_pos], dim=1)

        for spatial_layer, context_layer in zip(self.spatial_layers, self.context_layers):
            context_1 = context_layer(context_1, pos=pos_2d)[0]
            #context_2 = context_layer(context_2, pos=pos_2d)[0]

            spatial_features_1 = spatial_features_1 + pos_1d_1
            spatial_features_2 = spatial_features_2 + pos_1d_2

            spatial_features_1 = spatial_layer(spatial_features_1, spatial_features_2, context_1, pos_2d)
            spatial_features_2 = spatial_layer(spatial_features_2, spatial_features_1, context_1, pos_2d)

        spatial_features_1 = spatial_features_1.view(T, B, H * W, C)
        spatial_features_2 = spatial_features_2.view(T, B, H * W, C)

        padding = torch.zeros_like(spatial_features_1[0]).unsqueeze(0)
        spatial_features_1_seg = torch.cat([spatial_features_1, padding], dim=0)
        spatial_features_2_seg = torch.cat([padding, spatial_features_2], dim=0)
        spatial_features_seg = spatial_features_1_seg + spatial_features_2_seg

        return spatial_features_1, spatial_features_2, spatial_features_seg

class TransformerDecoderLayer(nn.Module):
    def __init__(self, dim, num_heads, d_ffn, dropout=0.0):
        super().__init__()

        #self.cross_attn_layer = SlotAttention(dim=dim)
        self.cross_attn_layer = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.dropout1 = nn.Dropout(dropout)

        self.self_attn_layer = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout2 = nn.Dropout(dropout)

        # ffn
        self.linear1 = nn.Linear(dim, d_ffn)
        self.activation = nn.GELU()
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, dim)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(dim)

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt
    
    def with_pos_embed(self, x, pos):
        return x + pos

    def forward(self, query, key, query_pos, key_pos):
        tgt2 = self.self_attn_layer(query=self.with_pos_embed(query, query_pos), key=self.with_pos_embed(query, query_pos), value=query)[0]
        query = query + self.dropout1(tgt2)
        query = self.norm1(query)

        tgt2 = self.cross_attn_layer(query=self.with_pos_embed(query, query_pos), key=self.with_pos_embed(key, key_pos), value=key)[0]
        query = query + self.dropout2(tgt2)
        query = self.norm2(query)

        query = self.forward_ffn(query)

        #spatial_tokens = self.from_slot[i](query=spatial_tokens, key=object_tokens, value=object_tokens, query_pos=spatial_pos, key_pos=memory_pos)

        #spatial_tokens = spatial_tokens.permute(0, 2, 1).view(T, B, C, H * W).view(T, B, C, H, W)
        #object_tokens = object_tokens.permute(1, 0, 2).contiguous() # T, B, C
        return query


class ChannelAttention(nn.Module):
    """
    Input -> x: [B, N, C]
    Output -> [B, N, C]
    """

    def __init__(self, dim, use_conv_mlp, num_heads, input_resolution, qkv_bias=False, drop_path=0., mlp_drop=0., attn_drop=0, proj_drop=0, mlp_ratio=4.0):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, input_resolution=input_resolution, use_conv=use_conv_mlp, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=mlp_drop)

    def forward(self, x):
        """x: [B, N, C]"""
        B, N, C = x.shape

        shortcut = x
        x = self.norm1(x)

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        # -------------------
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
        # ------------------
        x = self.proj(x)
        x = self.proj_drop(x)

        # FFN
        x = shortcut + self.drop_path(x)
        mlp_out = self.drop_path(self.mlp(self.norm2(x)))
        x = x + mlp_out

        return x


class SpatialTransformerLayer(nn.Module):
    """
    Input -> x: [B, N, C]
    Output -> [B, N, C]
    """

    def __init__(self, dim, use_conv_mlp, num_heads, input_resolution, qkv_bias=False, qk_scale=None, drop_path=0., mlp_drop=0., attn_drop=0, proj_drop=0, mlp_ratio=4.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, input_resolution=input_resolution, use_conv=use_conv_mlp, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=mlp_drop)

    def forward(self, x):
        """x: [B, N, C]"""
        B, N, C = x.shape

        shortcut = x
        x = self.norm1(x)

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale

        attn = (q @ k.transpose(-2, -1))
        # -------------------
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # ------------------
        x = self.proj(x)
        x = self.proj_drop(x)

        # FFN
        x = shortcut + self.drop_path(x)
        mlp_out = self.drop_path(self.mlp(self.norm2(x)))
        x = x + mlp_out

        return x


class SlotTransformer(nn.Module):
    def __init__(self, dim, iters, hidden_dim, eps = 1e-8):
        super().__init__()
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        self.gru = nn.GRUCell(dim, dim)

        hidden_dim = max(dim, hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )

        self.norm_input  = nn.LayerNorm(dim)
        self.norm_slots  = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

    def forward(self, query, key, value, query_pos, key_pos):
        B, L, C = query.shape

        query = query + query_pos
        key = key + key_pos

        key = self.norm_input(key)
        value = self.norm_input(value)
        k, v = self.to_k(key), self.to_v(value)

        slots = query

        for _ in range(self.iters):
            slots_prev = slots

            slots = self.norm_slots(slots)
            q = self.to_q(slots)

            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            attn = dots.softmax(dim=1) + self.eps
            attn = attn / attn.sum(dim=-1, keepdim=True)

            updates = torch.einsum('bjd,bij->bid', v, attn)

            slots = self.gru(
                updates.reshape(-1, C),
                slots_prev.reshape(-1, C)
            )

            slots = slots.reshape(B, -1, C)
            slots = slots + self.mlp(self.norm_pre_ff(slots))

        return slots


class SlotAttention(nn.Module):
    def __init__(self, dim, dropout=0.0, eps = 1e-8):
        super().__init__()
        self.eps = eps
        self.scale = dim ** -0.5

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        self.norm_input  = nn.LayerNorm(dim)
        self.norm_slots  = nn.LayerNorm(dim)

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, query, key, value, query_pos, key_pos):
        B, L, C = query.shape

        query = query + query_pos
        key = key + key_pos

        key = self.norm_input(key)
        value = self.norm_input(value)
        k, v = self.to_k(key), self.to_v(value)

        query = self.norm_slots(query)
        q = self.to_q(query)

        dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
        dots = dots.softmax(dim=1)
        attn = dots + self.eps
        attn = attn / attn.sum(dim=-1, keepdim=True)

        updates = torch.einsum('bjd,bij->bid', v, attn)

        slots = self.to_out(updates)

        return slots, dots

class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        B, C, H, W = output.shape

        output = torch.flatten(output, start_dim=2).permute(0, 2, 1)

        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos)[0]

        if self.norm is not None:
            output = self.norm(output)

        output = output.permute(0, 2, 1).view(B, C, H, W)

        return output

class CrossRelativeSpatialTransformerLayer(nn.Module):

    def __init__(self, d_model, nhead, size_2d, rescaled, dim_feedforward, dropout=0.0,
                 activation="gelu", normalize_before=False):
        super().__init__()
        dim_head = int(d_model / nhead)
        self.cross_attn = CrossRelativeAttention(inp=d_model, oup=d_model, size_2d=size_2d, rescaled=rescaled, heads=nhead, dim_head=dim_head)
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
        src2 = self.cross_attn(query=query, key=key, value=value, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward(self, query, key, value,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                key_pos: Optional[Tensor] = None):
        return self.forward_post(query, key, value, src_mask, src_key_padding_mask, query_pos=query_pos, key_pos=key_pos)



class CrossTransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, distance, topk=None, dim_feedforward=2048, dropout=0.0,
                 activation="gelu", normalize_before=False):
        super().__init__()
        if distance == 'cos':
            if topk is not None:
                self.self_attn = TopKAttention(d_model, d_model, heads=nhead, topk=topk)
            else:
                self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        elif distance == 'l2':
            self.self_attn = L2Attention(d_model, topk)
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
        src2, weights = self.self_attn(query=self.with_pos_embed(query, query_pos), key=self.with_pos_embed(key, key_pos), value=value)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward(self, query, key, value,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                key_pos: Optional[Tensor] = None):
        return self.forward_post(query, key, value, src_mask, src_key_padding_mask, query_pos=query_pos, key_pos=key_pos)


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.0,
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
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2, weights = self.self_attn(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, weights

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2, weights = self.self_attn(q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src,  weights

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)



class DeformableTransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, memory_length, self_attention, d_ffn=2048, dropout=0.0,
                 activation="gelu", normalize_before=False):
        super().__init__()
        self.self_attention = self_attention
        self.cross_attn = deformableAttention(d_model, nhead, memory_length)
        # Implementation of Feedforward model
        # self attention
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        d_ffn = 4 * d_model

        if self_attention:
            # self attention
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
            self.dropout1 = nn.Dropout(dropout)
            self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, 4 * d_model)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(4 * d_model, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos
    
    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos_self, pos_cross, value, video_length):
        if self.self_attention:
            q = k = self.with_pos_embed(src, pos_self)
            tgt2 = self.self_attn(q, k, value=src)[0]
            src = src + self.dropout1(tgt2)
            src = self.norm1(src)

        # self attention
        src2, sampling_locations, attention_weights, offsets = self.cross_attn(self.with_pos_embed(src, pos_cross), value, video_length=video_length)
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        # ffn
        src = self.forward_ffn(src)

        return src, sampling_locations, attention_weights, offsets



class DeformableTransformerEncoderLayer6(nn.Module):

    def __init__(self, d_model, nhead, memory_length, self_attention, add_motion_cues, d_ffn=2048, dropout=0.0,
                 activation="gelu", normalize_before=False):
        super().__init__()
        self.self_attention = self_attention
        self.cross_attn = deformableAttention6(d_model, nhead, memory_length, add_motion_cues=add_motion_cues)
        # Implementation of Feedforward model
        # self attention
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        d_ffn = 4 * d_model

        if self_attention:
            # self attention
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
            self.dropout1 = nn.Dropout(dropout)
            self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, 4 * d_model)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(4 * d_model, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos
    
    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos_self, pos_cross, value, key, video_length):
        if self.self_attention:
            q = k = self.with_pos_embed(src, pos_self)
            tgt2 = self.self_attn(q, k, value=src)[0]
            src = src + self.dropout1(tgt2)
            src = self.norm1(src)

        # self attention
        src2, sampling_locations, attention_weights, offsets = self.cross_attn(self.with_pos_embed(src, pos_cross), 
                                                                               value=value, 
                                                                               key=key, 
                                                                               video_length=video_length)
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        # ffn
        src = self.forward_ffn(src)

        return src, sampling_locations, attention_weights, offsets
    


class DeformableTransformerEncoderLayer2(nn.Module):

    def __init__(self, d_model, nhead, memory_length, d_ffn=2048, dropout=0.0,
                 activation="gelu", normalize_before=False):
        super().__init__()
        self.cross_attn = deformableAttention2(d_model, nhead, memory_length)
        # Implementation of Feedforward model
        # self attention
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos
    
    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos_self, pos_cross, value, reference_points):
        q = k = self.with_pos_embed(src, pos_self)
        tgt2 = self.self_attn(q, k, value=src)[0]
        src = src + self.dropout1(tgt2)
        src = self.norm1(src)

        # self attention
        src2, sampling_locations, attention_weights, offsets = self.cross_attn(self.with_pos_embed(src, pos_cross), reference_points, value)
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        # ffn
        src = self.forward_ffn(src)

        return src, sampling_locations, attention_weights, offsets
    


class DeformableTransformerEncoderLayer3(nn.Module):

    def __init__(self, d_model, nhead, memory_length, d_ffn=2048, dropout=0.0,
                 activation="gelu", normalize_before=False):
        super().__init__()
        self.cross_attn = deformableAttention2(d_model, nhead, memory_length)
        # Implementation of Feedforward model
        # self attention
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, 4*d_model)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(4*d_model, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos
    
    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos_cross, value, reference_points):

        # cross attention
        src2, sampling_locations, attention_weights, offsets = self.cross_attn(self.with_pos_embed(src, pos_cross), reference_points, value)
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        # ffn
        src = self.forward_ffn(src)

        return src, sampling_locations, attention_weights, offsets




class DeformableTransformerEncoderLayer4(nn.Module):

    def __init__(self, d_model, nhead, memory_length, d_ffn=2048, dropout=0.0,
                 activation="gelu", normalize_before=False):
        super().__init__()
        self.cross_attn = deformableAttention4Cross(d_model, nhead, memory_length)
        # Implementation of Feedforward model
        # self attention
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # self attention
        #self.self_attn = deformableAttention4Self(d_model, nhead)
        #self.dropout1 = nn.Dropout(dropout)
        #self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos
    
    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos_self, value, split_value, reference_points_2d):
        #tgt2, _, _, _ = self.self_attn(self.with_pos_embed(src, pos_self), reference_points_2d, src)
        #src = src + self.dropout1(tgt2)
        #src = self.norm1(src)

        # self attention
        src2, sampling_locations, attention_weights, offsets = self.cross_attn(self.with_pos_embed(src, pos_self), reference_points_2d, value, split_value)
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        # ffn
        src = self.forward_ffn(src)

        return src, sampling_locations, attention_weights, offsets
    


class CCATransformerLayer(nn.Module):

    def __init__(self, 
                dim, 
                nhead,
                input_resolution,
                drop_path,
                use_conv,
                dropout=0.0,
                activation="gelu", 
                normalize_before=False):
        super().__init__()

        dim_feedforward = int(dim * 4)
        self.use_conv = use_conv

        self.norm1 = nn.LayerNorm(dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        #self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout)
        self.self_attn = nn.MultiheadAttention(embed_dim=dim, 
                                                num_heads=nhead, 
                                                dropout=dropout)
        
        if use_conv:
            self.conv = nn.Conv2d()
            self.layer_norm = nn.LayerNorm(dim_feedforward)
        else:
            self.conv = nn.Identity()

        self.norm2 = nn.LayerNorm(dim)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, dim)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     skip_connection,
                     x,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None):
        src = skip_connection
        k = v = x
        src2 = self.self_attn(src, k, 
                                value=v, 
                                attn_mask=src_mask, 
                                key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.activation(self.linear1(src))
        if self.use_conv:
            conv_out = self.conv(src2)
            src2 = src2 + conv_out
            src2 = self.layer_norm(src2)
            src2 = self.act(src2)
        src2 = self.linear2(self.dropout(src2))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        src = self.drop_path(src)
        return src

    def forward_pre(self, skip_connection, x,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None):
        src2 = self.norm1(x)
        q = skip_connection
        k = v = x
        src2 = self.self_attn(q, k, 
                                value=v, 
                                attn_mask=src_mask,
                                key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.activation(self.linear1(src2))
        if self.use_conv:
            conv_out = self.conv(src2)
            src2 = src2 + conv_out
            src2 = self.layer_norm(src2)
            src2 = self.act(src2)
        src2 = self.linear2(self.dropout(src2))
        src = src + self.dropout2(src2)
        src = self.drop_path(src)
        return src

    def forward(self, skip_connection, x,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(skip_connection, x, src_mask, src_key_padding_mask)
        return self.forward_post(skip_connection, x, src_mask, src_key_padding_mask)

class ChannelTransformerEncoderLayer(nn.Module):

    def __init__(self, 
                d_model, 
                nhead,
                device,
                drop_path,
                dim_feedforward=3072, 
                dropout=0.1,
                activation="gelu", 
                normalize_before=False):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        #self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout)
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, device=device)

        self.norm2 = nn.LayerNorm(d_model)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     concat,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = self.with_pos_embed(src, pos)
        k = v = concat
        src2 = self.self_attn(q, k, 
                                value=v, 
                                attn_mask=src_mask, 
                                key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        src = self.drop_path(src)
        return src

    def forward_pre(self, src, concat,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = self.with_pos_embed(src2, pos)
        k = v = concat
        src2 = self.self_attn(q, k, 
                                value=v, 
                                attn_mask=src_mask,
                                key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        src = self.drop_path(src)
        return src

    def forward(self, src, concat, pos,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, concat, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, concat, src_mask, src_key_padding_mask, pos)

class VitCCABasicLayer(nn.Module):
    def __init__(self, in_dim, nhead, out_encoder_dims, scale_factor, use_conv_mlp, input_resolution, device, nb_blocks, dpr, dropout):
        super().__init__()
        self.in_dim=in_dim
        self.device = device
        self.nb_stages = len(out_encoder_dims)
        output_res = int(input_resolution[0] / scale_factor)
        
        self.pos_embedding = PositionEmbeddingSine1d(num_pos_feats=output_res**2, normalize=True)
#
        #self.skip_projs = nn.ModuleList()
        #for patch_size, nb_channels in zip(patch_sizes, out_encoder_dims):
        #    proj = nn.Conv2d(nb_channels, nb_channels, kernel_size=patch_size, stride=patch_size)
        #    res = int(patch_size * output_res)
        #    proj = nn.Sequential(To_image([res, res]), proj, nn.BatchNorm2d(nb_channels), nn.GELU(), From_image())
        #    self.skip_projs.append(proj)
        #
        out_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=3, padding=1)
        up = nn.Upsample(scale_factor=scale_factor, mode='bilinear')
        self.out = nn.Sequential(To_image([output_res, output_res]), up, out_conv, nn.BatchNorm2d(in_dim), nn.GELU(), From_image())
        
        decoder_shrink_proj = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=scale_factor, stride=scale_factor)
        self.decoder_shrink_proj = nn.Sequential(To_image(input_resolution), decoder_shrink_proj, nn.BatchNorm2d(in_dim), nn.GELU(), From_image())

        self.blocks = nn.ModuleList()
        for i in range(nb_blocks):
            layer = CCATransformerLayer(dim=output_res**2,
                                        nhead=nhead,
                                        use_conv=use_conv_mlp,
                                        drop_path=dpr[i],
                                        input_resolution=input_resolution,
                                        dropout=dropout)
            self.blocks.append(layer)
    
    def forward(self, skip_connection, skip_connection_range):
        skip_connection = self.decoder_shrink_proj(skip_connection)
        skip_connection = skip_connection.permute(2, 0, 1)
        skip_connection_range = skip_connection_range.permute(2, 0, 1)
        skip_connection_pos = self.pos_embedding(shape_util=(skip_connection.shape[1], skip_connection.shape[0]), device=self.device)
        skip_connection_range_pos = self.pos_embedding(shape_util=(skip_connection_range.shape[1], skip_connection_range.shape[0]), device=self.device)
        for layer in self.blocks:
            skip_connection_range = skip_connection_range + skip_connection_range_pos
            skip_connection = skip_connection + skip_connection_pos
            skip_connection = layer(skip_connection, skip_connection_range)
        skip_connection = skip_connection.permute(1, 2, 0)
        skip_connection = self.out(skip_connection)
        return skip_connection
    
    def concat_skip_connections(self, skip_connections):
        sp_list = []
        for skip_connection, proj in zip(skip_connections, self.skip_projs):
            skip_connection = proj(skip_connection)
            sp_list.append(skip_connection)
        return torch.cat(sp_list, dim=-1)

class VitBasicLayer(nn.Module):
    def __init__(self, in_dim, nhead, rpe_mode, rpe_contextual_tensor, input_resolution, proj, device, nb_blocks, dpr, dropout):
        super().__init__()
        self.in_dim=in_dim
        self.device = device
        self.input_resolution = input_resolution
        if len(input_resolution) == 2:
            self.pos_embedding = PositionEmbeddingSine2d(num_pos_feats=in_dim//2, normalize=True)
            relative_position_index = get_indices_2d(self.input_resolution)
        elif len(input_resolution) == 3:
            self.pos_embedding = PositionEmbeddingSine3d(num_pos_feats=in_dim//3, normalize=True)
            relative_position_index = get_indices_3d(self.input_resolution)
        self.blocks = nn.ModuleList()
        for i in range(nb_blocks):
            layer = TransformerEncoderLayer(d_model=in_dim,
                                            nhead=nhead,
                                            input_resolution=input_resolution,
                                            proj=proj,
                                            device=device,
                                            relative_position_index=relative_position_index,
                                            drop_path=dpr[i],
                                            rpe_mode=rpe_mode,
                                            rpe_contextual_tensor=rpe_contextual_tensor,
                                            dropout=dropout)
            self.blocks.append(layer)
    
    def forward(self, x):
        x = x.permute(1, 0, 2)
        for layer in self.blocks:
            pos = self.pos_embedding(shape_util=(x.shape[1],) + tuple(self.input_resolution), device=self.device)
            if len(self.input_resolution) == 2:
                pos = pos.permute(2, 3, 0, 1).view(-1, x.shape[1], self.in_dim)
            else:
                pos = pos.permute(2, 3, 4, 0, 1).view(-1, x.shape[1], self.in_dim)
            x = layer(x, pos=pos)
        x = x.permute(1, 0, 2)
        return x

class VitChannelLayer(nn.Module):
    def __init__(self, nhead, img_size, device, nb_blocks, dropout, input_encoder_dims, batch_size):
        super().__init__()
        channels = [input_encoder_dims[1] * (2**idx) for idx in range(len(input_encoder_dims))]
        patch_sizes = [2**(4-idx) for idx in range(len(input_encoder_dims))]
        self.resolutions = [img_size//(2**idx) for idx in range(len(input_encoder_dims))]
        self.embedding_resolution = img_size//patch_sizes[0]
        self.device = device
        self.blocks = nn.ModuleList()
        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.pos_embeddings = []

        for i in range(nb_blocks):
            layer = ChannelTransformerEncoderLayer(d_model=self.embedding_resolution**2, nhead=nhead, device=device, drop_path=0, dropout=dropout)
            self.blocks.append(layer)

        for idx, (nb_channels, patch_size) in enumerate(zip(channels, patch_sizes)):
            conv = nn.Conv2d(nb_channels, nb_channels, kernel_size=patch_size, stride=patch_size)
            self.convs.append(conv)

            upsample = nn.Upsample(scale_factor=2**(4-idx), mode='bilinear')
            self.upsamples.append(upsample)

            p = PositionEmbeddingSine2d(nb_channels//2, normalize=True)
            pos = p(shape_util=(batch_size, self.embedding_resolution, self.embedding_resolution), device=device)
            pos = pos.permute(1, 0, 2, 3).view(nb_channels, batch_size, self.embedding_resolution**2)
            self.pos_embeddings.append(pos)
    
    def forward(self, skip_connections):
        # B L C
        embeddings = []
        for conv, skip_connection, res in zip(self.convs, skip_connections, self.resolutions):
            B, L, C = skip_connection.shape
            skip_connection = skip_connection.permute(0, 2, 1).view(B, C, res, res)
            embed = conv(skip_connection)
            embed = embed.permute(1, 0, 2, 3).view(C, B, -1)
            embeddings.append(embed)

        out = []
        cat_embeddings = torch.cat(embeddings, dim=0)
        for embedding, pos, upsample in zip(embeddings, self.pos_embeddings, self.upsamples):
            C, B, L = embedding.shape
            for block in self.blocks:
                embedding = block(src=embedding, concat=cat_embeddings, pos=pos)
            embedding = embedding.permute(1, 0, 2).view(B, C, self.embedding_resolution, self.embedding_resolution)
            embedding = upsample(embedding)
            embedding = embedding.permute(0, 2, 3, 1).view(B, -1, C)
            out.append(embedding)

        return out