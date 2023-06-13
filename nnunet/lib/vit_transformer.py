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
from .utils import depthwise_conv, Mlp, To_image, From_image, RFR_1d
import matplotlib
import matplotlib.pyplot as plt
from math import ceil
from torch.nn.functional import grid_sample
from torch.nn.functional import pad
import matplotlib.patches as patches
from torchvision.ops import roi_align

from einops import rearrange
from einops.layers.torch import Rearrange
from torch.nn import init
from torch.nn.functional import affine_grid

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


class SpatioTemporalTransformer(nn.Module):
    def __init__(self, dim, num_heads, num_layers, d_ffn, dropout=0.0):
        super().__init__()
        self.num_layers = num_layers
        
        spatial_attn_layer = TransformerEncoderLayer(d_model=dim, nhead=num_heads, dim_feedforward=d_ffn)
        self.spatial_attention = _get_clones(spatial_attn_layer, num_layers)
        
        self.pos_weight = nn.Parameter(torch.randn(1, dim))


    def forward(self, spatial_tokens, memory_bus, pos_2d):
        shape = spatial_tokens.shape
        T, B, C, H, W = shape

        pos_2d = pos_2d.view(1, H * W, C).repeat(T * B, 1, 1)
        pos_weight = self.pos_weight.view(1, 1, C).repeat(T * B, 1, 1)

        memory_bus = memory_bus.view(T, 1, 1, C).repeat(1, B, 1, 1).view(T * B, 1, C)
        spatial_tokens = spatial_tokens.permute(0, 1, 3, 4, 2).contiguous()
        spatial_tokens = spatial_tokens.view(T * B, H, W, C).view(T * B, H * W, C)

        src = torch.cat([memory_bus, spatial_tokens], dim=1)
        pos = torch.cat([pos_weight, pos_2d], dim=1)

        for i in range(self.num_layers):
            src = self.spatial_attention[i](src=src, pos=pos)

        memory_bus = src[:, 0] # T*B, C
        spatial_tokens = src[:, 1:] # T*B, H*W, C
        memory_bus = memory_bus.view(T, B, C)

        memory_bus = memory_bus.permute(1, 0, 2).contiguous() # B, T, C
        spatial_tokens = spatial_tokens.permute(0, 2, 1).contiguous() # T*B, C, H*W
        spatial_tokens = spatial_tokens.view(T, B, C, H*W).view(T, B, C, H, W)
        return memory_bus, spatial_tokens


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
    def __init__(self, dim, num_heads, num_layers, d_ffn, conv_layer_1d, dropout=0.0):
        super().__init__()
        self.num_layers = num_layers

        temporal_attn_layer = TransformerEncoderLayer(d_model=dim, nhead=num_heads, dim_feedforward=d_ffn)
        self.temporal_attention = _get_clones(temporal_attn_layer, num_layers)

        self.conv_1d = _get_clones(conv_layer_1d, num_layers)

    def forward(self, memory_bus, spatial_tokens, pos_1d, pos_2d):
        B, T, C = memory_bus.shape
        B, C, H, W = spatial_tokens.shape

        spatial_tokens = spatial_tokens.permute(0, 2, 3, 1).contiguous()
        spatial_tokens = spatial_tokens.view(B, H * W, C)
        
        pos_1d = pos_1d.view(1, T, C).repeat(B, 1, 1)
        pos_2d = pos_2d.view(1, H*W, C).repeat(B, 1, 1)
        pos = torch.cat([pos_1d, pos_2d], dim=1)

        for i in range(self.num_layers):
            memory_bus = memory_bus.permute(0, 2, 1).contiguous() # B, C, T

            memory_bus = self.conv_1d[i](memory_bus) # B, C, T
            memory_bus = memory_bus.permute(0, 2, 1).contiguous() # B, T, C

            src = torch.cat([memory_bus, spatial_tokens], dim=1)
            src = self.temporal_attention[i](src=src, pos=pos)

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

    def forward_post(self, myself, other, context, pos):
        q = k = self.with_pos_embed(myself, pos)
        tgt2 = self.self_attn(q, k, value=myself)[0]
        myself = myself + self.dropout1(tgt2)
        myself = self.norm1(myself)

        tgt2 = self.cross_attn(query=self.with_pos_embed(myself, pos), key=self.with_pos_embed(other, pos), value=other)[0]
        myself = myself + self.dropout2(tgt2)
        myself = self.norm2(myself)

        tgt2 = self.context_attn(query=self.with_pos_embed(myself, pos), key=self.with_pos_embed(context, pos), value=context)[0]
        myself = myself + self.dropout3(tgt2)
        myself = self.norm3(myself)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(myself))))
        myself = myself + self.dropout4(tgt2)
        myself = self.norm4(myself)
        return myself

    def forward(self, myself, other, context, pos):
        return self.forward_post(myself, other, context, pos)
    

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

    def forward_post(self, myself, other, sa_query_pos, ca_query_pos, key_pos):
        q = k = self.with_pos_embed(myself, sa_query_pos)
        tgt2 = self.self_attn(q, k, value=myself)[0]
        myself = myself + self.dropout1(tgt2)
        myself = self.norm1(myself)

        tgt2 = self.cross_attn(query=self.with_pos_embed(myself, ca_query_pos), key=self.with_pos_embed(other, key_pos), value=other)[0]
        myself = myself + self.dropout2(tgt2)
        myself = self.norm2(myself)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(myself))))
        myself = myself + self.dropout3(tgt2)
        myself = self.norm3(myself)
        return myself

    def forward(self, myself, other, sa_query_pos, ca_query_pos, key_pos):
        return self.forward_post(myself, other, sa_query_pos, ca_query_pos, key_pos)
    

class TransformerFlowEncoder3(nn.Module):
    def __init__(self, dim, nhead, num_layers, video_length, nb_tokens):
        super().__init__()

        self.nb_tokens = nb_tokens

        spatial_layer = TransformerFlowLayer(d_model=dim, nhead=nhead)
        self.spatial_layers = _get_clones(spatial_layer, num_layers)

        temporal_layer = TransformerEncoderLayer(d_model=dim, nhead=nhead)
        self.temporal_layers = _get_clones(temporal_layer, num_layers)

        self.pos_obj_2d = PositionEmbeddingSine2d(num_pos_feats=dim // 2, normalize=True)
        #self.pos_obj_1d = PositionEmbeddingSine1d(num_pos_feats=dim, normalize=True)

        self.unlabeled_pos_1d = nn.Parameter(torch.randn(video_length - 1, dim))
        self.labeled_pos_1d = nn.Parameter(torch.randn(1, dim))

        self.token_pos = nn.Parameter(torch.randn(self.nb_tokens, dim))
        self.unlabeled_temporal_tokens = nn.Parameter(torch.randn(video_length - 1, self.nb_tokens, dim))
        self.labeled_temporal_tokens = nn.Parameter(torch.randn(1, self.nb_tokens, dim))
    
    
    def forward(self, labeled_spatial_features, unlabeled_spatial_features):
        '''labeled_spatial_features: B, C, H, W
            unlabeled_spatial_features: T-1, B, C, H, W'''
        
        shape = unlabeled_spatial_features.shape
        T, B, C, H, W = shape

        labeled_spatial_features = labeled_spatial_features[None].repeat(len(unlabeled_spatial_features), 1, 1, 1, 1)
        labeled_spatial_features = labeled_spatial_features.permute(0, 1, 3, 4, 2).contiguous() # T, B, H, W, C
        labeled_spatial_features = labeled_spatial_features.view(T, B, H * W, C).view(T * B, H * W, C)

        unlabeled_spatial_features = unlabeled_spatial_features.permute(0, 1, 3, 4, 2).contiguous() # T, B, H, W, C
        unlabeled_spatial_features = unlabeled_spatial_features.view(T, B, H * W, C).view(T * B, H * W, C)

        pos_2d = self.pos_obj_2d(shape_util=(B, H, W), device=labeled_spatial_features.device)
        pos_2d = pos_2d.permute(0, 2, 3, 1).contiguous()
        pos_2d = pos_2d.view(B, H * W, C).repeat(T, 1, 1)

        unlabeled_pos_1d = self.unlabeled_pos_1d[None, :, None, :].repeat(B, 1, self.nb_tokens, 1).view(B, T * self.nb_tokens, C)
        labeled_pos_1d = self.labeled_pos_1d[None, :, None, :].repeat(B, T, self.nb_tokens, 1).view(B, T * self.nb_tokens, C)

        unlabeled_temporal_features = self.unlabeled_temporal_tokens[None].repeat(B, 1, 1, 1).view(B, T * self.nb_tokens, C)
        labeled_temporal_features = self.labeled_temporal_tokens[None].repeat(B, T, 1, 1).view(B, T * self.nb_tokens, C)

        token_pos = self.token_pos[None].repeat(T * B, 1, 1) # T*B, M, C
        spatial_pos = torch.cat([pos_2d, token_pos], dim=1)

        token_pos = self.token_pos[None, None].repeat(B, T, 1, 1).view(B, T * self.nb_tokens, C)

        labeled_temporal_pos = labeled_pos_1d + token_pos
        unlabeled_temporal_pos = unlabeled_pos_1d + token_pos
        temporal_pos = torch.cat([labeled_temporal_pos, unlabeled_temporal_pos], dim=1)

        for spatial_layer, temporal_layer in zip(self.spatial_layers, self.temporal_layers):
            #temporal_features = torch.cat([labeled_temporal_features, unlabeled_temporal_features], dim=1)
            #temporal_features = temporal_layer(temporal_features, pos=temporal_pos)[0]
#
            #labeled_temporal_features = temporal_features[:, :T*self.nb_tokens]
            #unlabeled_temporal_features = temporal_features[:, T*self.nb_tokens:]
#
            #labeled_temporal_features = labeled_temporal_features.permute(1, 0, 2).contiguous() # T*M, B, C
            #unlabeled_temporal_features = unlabeled_temporal_features.permute(1, 0, 2).contiguous() # T*M, B, C
#
            #labeled_temporal_features = labeled_temporal_features.view(T, self.nb_tokens, B, C)
            #unlabeled_temporal_features = unlabeled_temporal_features.view(T, self.nb_tokens, B, C)
#
            #labeled_temporal_features = labeled_temporal_features.permute(0, 2, 1, 3).contiguous() # T, B, M, C
            #unlabeled_temporal_features = unlabeled_temporal_features.permute(0, 2, 1, 3).contiguous() # T, B, M, C
#
            #labeled_temporal_features = labeled_temporal_features.view(T*B, self.nb_tokens, C)
            #unlabeled_temporal_features = unlabeled_temporal_features.view(T*B, self.nb_tokens, C)
#
            #labeled_feature = torch.cat([labeled_spatial_features, labeled_temporal_features], dim=1)
            #unlabeled_feature = torch.cat([unlabeled_spatial_features, unlabeled_temporal_features], dim=1)

            #labeled_feature = spatial_layer(labeled_feature, unlabeled_feature, spatial_pos)
            #unlabeled_feature = spatial_layer(unlabeled_feature, labeled_feature, spatial_pos)

            labeled_spatial_features = spatial_layer(labeled_spatial_features, unlabeled_spatial_features, pos_2d)
            unlabeled_spatial_features = spatial_layer(unlabeled_spatial_features, labeled_spatial_features, pos_2d)

            #labeled_spatial_features = labeled_feature[:, :H*W] # T*B, H*W, C
            #labeled_temporal_features = labeled_feature[:, H*W:] # T*B, M, C
            #unlabeled_spatial_features = unlabeled_feature[:, :H*W] # T*B, H*W, C
            #unlabeled_temporal_features = unlabeled_feature[:, H*W:] # T*B, M, C
#
            #labeled_temporal_features = labeled_temporal_features.view(T, B, self.nb_tokens, C)
            #unlabeled_temporal_features = unlabeled_temporal_features.view(T, B, self.nb_tokens, C)
#
            #labeled_temporal_features = labeled_temporal_features.permute(1, 0, 2, 3).contiguous() # B, T, M, C
            #unlabeled_temporal_features = unlabeled_temporal_features.permute(1, 0, 2, 3).contiguous() # B, T, M, C
#
            #labeled_temporal_features = labeled_temporal_features.view(B, T * self.nb_tokens, C)
            #unlabeled_temporal_features = unlabeled_temporal_features.view(B, T * self.nb_tokens, C)

        labeled_spatial_features = labeled_spatial_features.view(T, B, H * W, C)
        unlabeled_spatial_features = unlabeled_spatial_features.view(T, B, H * W, C)

        return labeled_spatial_features, unlabeled_spatial_features


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



class TransformerFlowEncoderIterative(nn.Module):
    def __init__(self, dim, nhead, num_layers, nb_tokens, lookback):
        super().__init__()

        self.nb_tokens = nb_tokens
        self.lookback = lookback

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

        past_pos_1d = pos_1d[:-1]
        for spatial_layer in self.spatial_layers:
            past_features = torch.zeros_like(spatial_features)[:-1] # T - 1, B, H*W, C
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

                start = max(0, i - self.lookback)
                stop = i

                current_spatial_feature = spatial_features[i]
                current_pos_1d = pos_1d[i]
                if i == 0:
                    previous_spatial_feature = torch.zeros_like(current_spatial_feature).unsqueeze(0).repeat(self.lookback, 1, 1, 1)
                    previous_pos_1d = torch.zeros_like(current_pos_1d).unsqueeze(0).repeat(self.lookback, 1, 1, 1)
                else:
                    previous_spatial_feature = past_features[start:stop]
                    previous_spatial_feature = torch.nn.functional.pad(previous_spatial_feature, pad=(0, 0, 0, 0, 0, 0, max(0, self.lookback - len(previous_spatial_feature)), 0))
                    previous_pos_1d = past_pos_1d[start:stop]
                    previous_pos_1d = torch.nn.functional.pad(previous_pos_1d, pad=(0, 0, 0, 0, 0, 0, max(0, self.lookback - len(previous_pos_1d)), 0))

                previous_pos_1d = previous_pos_1d.permute(0, 2, 1, 3).contiguous()
                previous_pos_1d = previous_pos_1d.view(self.lookback * H * W, B, C)
                previous_pos_1d = previous_pos_1d.permute(1, 0, 2).contiguous()

                ca_query_pos = pos_2d + current_pos_1d

                key_pos_2d = pos_2d[:, None, :, :].repeat(1, self.lookback, 1, 1)
                key_pos_2d = key_pos_2d.permute(1, 2, 0, 3).contiguous()
                key_pos_2d = key_pos_2d.view(self.lookback * H * W, B, C)
                key_pos_2d = key_pos_2d.permute(1, 0, 2).contiguous()
                key_pos = key_pos_2d + previous_pos_1d

                previous_spatial_feature = previous_spatial_feature.permute(0, 2, 1, 3).contiguous()
                previous_spatial_feature = previous_spatial_feature.view(self.lookback * H * W, B, C)
                previous_spatial_feature = previous_spatial_feature.permute(1, 0, 2).contiguous()

                current_spatial_feature = spatial_layer(current_spatial_feature, previous_spatial_feature, sa_query_pos=pos_2d, ca_query_pos=ca_query_pos, key_pos=key_pos)
                if i == len(spatial_features) - 1:
                    spatial_features = torch.cat([past_features, current_spatial_feature.unsqueeze(0)], dim=0)
                else:
                    past_features[i] = current_spatial_feature
        return spatial_features
            

class TransformerFlowEncoderIterativeMiddle(nn.Module):
    def __init__(self, dim, nhead, num_layers, nb_tokens, lookback):
        super().__init__()

        self.nb_tokens = nb_tokens
        self.lookback = lookback
        assert lookback % 2 == 0

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
        past_pos_1d = pos_1d[:-1]
        for spatial_layer in self.spatial_layers:
            past_features = torch.zeros_like(spatial_features)[:-1] # T - 1, B, H*W, C
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

                start = max(0, i - (self.lookback // 2))
                stop = min(len(spatial_features), i + (self.lookback // 2) + 1)

                current_spatial_feature = spatial_features[i]
                current_pos_1d = pos_1d[i]
                if i == 0:
                    previous_spatial_feature = torch.zeros_like(current_spatial_feature).unsqueeze(0).repeat(self.lookback // 2, 1, 1, 1)
                    previous_pos_1d = torch.zeros_like(current_pos_1d).unsqueeze(0).repeat(self.lookback // 2, 1, 1, 1)
                else:
                    previous_spatial_feature = past_features[start:i]
                    previous_spatial_feature = torch.nn.functional.pad(previous_spatial_feature, pad=(0, 0, 0, 0, 0, 0, max(0, (self.lookback // 2) - len(previous_spatial_feature)), 0))
                    previous_pos_1d = past_pos_1d[start:i]
                    previous_pos_1d = torch.nn.functional.pad(previous_pos_1d, pad=(0, 0, 0, 0, 0, 0, max(0, (self.lookback // 2) - len(previous_pos_1d)), 0))
                
                if i == len(spatial_features) - 1:
                    next_spatial_feature = torch.zeros_like(current_spatial_feature).unsqueeze(0).repeat(self.lookback // 2, 1, 1, 1)
                    next_pos_1d = torch.zeros_like(current_pos_1d).unsqueeze(0).repeat(self.lookback // 2, 1, 1, 1)
                else:
                    next_spatial_feature = spatial_features[i+1:stop]
                    next_spatial_feature = torch.nn.functional.pad(next_spatial_feature, pad=(0, 0, 0, 0, 0, 0, 0, max(0, (self.lookback // 2) - len(next_spatial_feature))))
                    next_pos_1d = pos_1d[i+1:stop]
                    next_pos_1d = torch.nn.functional.pad(next_pos_1d, pad=(0, 0, 0, 0, 0, 0, 0, max(0, (self.lookback // 2) - len(next_pos_1d))))

                key_pos_1d = torch.cat([previous_pos_1d, next_pos_1d], dim=0)
                key_spatial_feature = torch.cat([previous_spatial_feature, next_spatial_feature], dim=0)

                key_pos_1d = key_pos_1d.permute(0, 2, 1, 3).contiguous()
                key_pos_1d = key_pos_1d.view(self.lookback * H * W, B, C)
                key_pos_1d = key_pos_1d.permute(1, 0, 2).contiguous()

                ca_query_pos = pos_2d + current_pos_1d

                key_pos_2d = pos_2d[:, None, :, :].repeat(1, self.lookback, 1, 1)
                key_pos_2d = key_pos_2d.permute(1, 2, 0, 3).contiguous()
                key_pos_2d = key_pos_2d.view(self.lookback * H * W, B, C)
                key_pos_2d = key_pos_2d.permute(1, 0, 2).contiguous()

                key_pos = key_pos_2d + key_pos_1d

                key_spatial_feature = key_spatial_feature.permute(0, 2, 1, 3).contiguous()
                key_spatial_feature = key_spatial_feature.view(self.lookback * H * W, B, C)
                key_spatial_feature = key_spatial_feature.permute(1, 0, 2).contiguous()

                current_spatial_feature = spatial_layer(current_spatial_feature, key_spatial_feature, sa_query_pos=pos_2d, ca_query_pos=ca_query_pos, key_pos=key_pos)
                if i == len(spatial_features) - 1:
                    spatial_features = torch.cat([past_features, current_spatial_feature.unsqueeze(0)], dim=0)
                else:
                    past_features[i] = current_spatial_feature
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
                     query, key, value,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None,
                     key_pos: Optional[Tensor] = None):
        src = query
        src2, weights = self.self_attn(self.with_pos_embed(query, query_pos), self.with_pos_embed(key, key_pos), value=value, attn_mask=src_mask,
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
        return self.forward_post(query, key, value, src_mask, src_key_padding_mask, query_pos=query_pos, key_pos=key_pos)[0]


class RelativeTransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, image_size, dim_feedforward, dropout=0.0,
                 activation="gelu", normalize_before=False):
        super().__init__()
        dim_head = int(d_model / nhead)
        self.self_attn = ContextualRelativeAttention2D(inp=d_model, oup=d_model, image_size=image_size, heads=nhead, dim_head=dim_head)
        #self.self_attn = RelativeAttention2D(inp=d_model, oup=d_model, image_size=image_size, heads=nhead, dim_head=dim_head)
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
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)

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
        return src, weights

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)

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