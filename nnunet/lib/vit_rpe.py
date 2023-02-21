from torch.nn.modules import Module
import warnings
from typing import Optional, Tuple
import math
from timm.models.layers import trunc_normal_
from .utils import depthwise_separable_conv

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
from torch.nn.init import xavier_normal_
from torch.nn.parameter import Parameter
from torch import functional as F
from torch.nn.functional import linear, _in_projection_packed, _in_projection, pad, softmax, dropout
from torch.overrides import has_torch_function, handle_torch_function
from torch.nn.init import zeros_

def add_rpe_mem_bus(relative_position_index, num_memory_bus, rpe_table_size):
    relative_position_index_shape = relative_position_index.shape
    new_relative_position_index = relative_position_index.new_empty(size=(num_memory_bus + relative_position_index_shape[0], num_memory_bus + relative_position_index_shape[1]))
    new_relative_position_index[:num_memory_bus] = rpe_table_size
    new_relative_position_index[:, :num_memory_bus] = rpe_table_size
    new_relative_position_index[num_memory_bus:, num_memory_bus:] = relative_position_index
    return new_relative_position_index

def get_indices_1d(input_resolution):
    # get pair-wise relative position index for each token inside the window
    coords = torch.arange(input_resolution[0])
    coords_flatten = torch.stack(torch.meshgrid([coords]))
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww     i-i', j-j' [-(wh-1), wh-1]
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
    relative_coords[:, :, 0] += input_resolution[0] - 1  # shift to start from 0    i-i'+ h-1, j-j'+ w-1 [0, 2wh-2]
    relative_position_index = relative_coords.sum(-1)  # Wh, Ww  parameters weights are 1d [0, (2wh-1) * (2ww-1)]
    return relative_position_index

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
    if num_memory_bus > 0:
        relative_position_index = add_rpe_mem_bus(relative_position_index, num_memory_bus, rpe_table_size)
    return relative_position_index

def get_indices_3d(input_resolution):
    # get pair-wise relative position index for each token inside the window
    coords_d = torch.arange(input_resolution[0])
    coords_h = torch.arange(input_resolution[1])
    coords_w = torch.arange(input_resolution[2])
    coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))  # 3, Wd, Wh, Ww
    coords_flatten = torch.flatten(coords, 1)  # 3, Wd*Wh*Ww
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Wd*Wh*Ww, Wd*Wh*Ww
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wd*Wh*Ww, Wd*Wh*Ww, 3
    relative_coords[:, :, 0] += input_resolution[0] - 1  # shift to start from 0
    relative_coords[:, :, 1] += input_resolution[1] - 1
    relative_coords[:, :, 2] += input_resolution[2] - 1

    relative_coords[:, :, 0] *= (2 * input_resolution[1] - 1) * (2 * input_resolution[2] - 1)
    relative_coords[:, :, 1] *= (2 * input_resolution[2] - 1)
    relative_position_index = relative_coords.sum(-1)  # Wd*Wh*Ww, Wd*Wh*Ww
    return relative_position_index

def _mha_shape_check(query: Tensor, key: Tensor, value: Tensor,
                     key_padding_mask: Optional[Tensor], attn_mask: Optional[Tensor], num_heads: int):
    # Verifies the expected shape for `query, `key`, `value`, `key_padding_mask` and `attn_mask`
    # and returns if the input is batched or not.
    # Raises an error if `query` is not 2-D (unbatched) or 3-D (batched) tensor.

    # Shape check.
    if query.dim() == 3:
        # Batched Inputs
        is_batched = True
        assert key.dim() == 3 and value.dim() == 3, \
            ("For batched (3-D) `query`, expected `key` and `value` to be 3-D"
             f" but found {key.dim()}-D and {value.dim()}-D tensors respectively")
        if key_padding_mask is not None:
            assert key_padding_mask.dim() == 2, \
                ("For batched (3-D) `query`, expected `key_padding_mask` to be `None` or 2-D"
                 f" but found {key_padding_mask.dim()}-D tensor instead")
        if attn_mask is not None:
            assert attn_mask.dim() in (2, 3), \
                ("For batched (3-D) `query`, expected `attn_mask` to be `None`, 2-D or 3-D"
                 f" but found {attn_mask.dim()}-D tensor instead")
    elif query.dim() == 2:
        # Unbatched Inputs
        is_batched = False
        assert key.dim() == 2 and value.dim() == 2, \
            ("For unbatched (2-D) `query`, expected `key` and `value` to be 2-D"
             f" but found {key.dim()}-D and {value.dim()}-D tensors respectively")

        if key_padding_mask is not None:
            assert key_padding_mask.dim() == 1, \
                ("For unbatched (2-D) `query`, expected `key_padding_mask` to be `None` or 1-D"
                 f" but found {key_padding_mask.dim()}-D tensor instead")

        if attn_mask is not None:
            assert attn_mask.dim() in (2, 3), \
                ("For unbatched (2-D) `query`, expected `attn_mask` to be `None`, 2-D or 3-D"
                 f" but found {attn_mask.dim()}-D tensor instead")
            if attn_mask.dim() == 3:
                expected_shape = (num_heads, query.shape[0], key.shape[0])
                assert attn_mask.shape == expected_shape, \
                    (f"Expected `attn_mask` shape to be {expected_shape} but got {attn_mask.shape}")
    else:
        raise AssertionError(
            f"query should be unbatched 2D or batched 3D tensor but received {query.dim()}-D query tensor")

    return is_batched

class MultiheadAttention(Module):
    r"""Allows the model to jointly attend to information
    from different representation subspaces.
    See `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_

    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O

    where :math:`head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)`.

    Args:
        embed_dim: total dimension of the model.
        num_heads: parallel attention heads.
        dropout: a Dropout layer on attn_output_weights. Default: 0.0.
        bias: add bias as module parameter. Default: True.
        add_bias_kv: add bias to the key and value sequences at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        kdim: total number of features in key. Default: None.
        vdim: total number of features in value. Default: None.
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).

    Note that if :attr:`kdim` and :attr:`vdim` are None, they will be set
    to :attr:`embed_dim` such that query, key, and value have the same
    number of features.

    Examples::

        >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)
    """
    __constants__ = ['batch_first']
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    def __init__(self, 
                embed_dim, 
                num_heads, 
                input_resolution,
                proj,
                device,
                relative_position_index,
                num_memory_token=0,
                rpe_mode='contextual',
                rpe_contextual_tensor='qkv',
                dropout=0., 
                bias=True, 
                add_bias_kv=False, 
                add_zero_attn=False,
                kdim=None, 
                vdim=None, 
                batch_first=False, 
                dtype=None) -> None:

        factory_kwargs = {'device': device, 'dtype': dtype}
        super(MultiheadAttention, self).__init__()

        self.rpe_mode = rpe_mode
        self.rpe_contextual_tensor = rpe_contextual_tensor
        self.input_resolution = input_resolution
        self.proj = proj
        self.device=device
        self.relative_position_index = relative_position_index

        self.rpe_table = None
        self.q_rpe_table = None
        self.k_rpe_table = None
        self.v_rpe_table = None
        if len(input_resolution) == 2:
            rpe_table_size = (2 * input_resolution[0] - 1) * (2 * input_resolution[1] - 1)
            if rpe_mode == 'contextual':
                if 'q' in rpe_contextual_tensor:
                    self.q_rpe_table = nn.Parameter(torch.zeros(num_heads, embed_dim//num_heads, rpe_table_size))  # nh, head_dim, 2*Wh-1 * 2*Ww-1
                if 'k' in rpe_contextual_tensor:
                    self.k_rpe_table = nn.Parameter(torch.zeros(num_heads, embed_dim//num_heads, rpe_table_size))  # nh, head_dim, 2*Wh-1 * 2*Ww-1
                if 'v' in rpe_contextual_tensor:
                    self.v_rpe_table = nn.Parameter(torch.zeros(num_heads, rpe_table_size, embed_dim//num_heads))  # nh, head_dim, 2*Wh-1 * 2*Ww-1
            elif rpe_mode == 'bias':
                self.rpe_table = nn.Parameter(torch.zeros(num_heads, rpe_table_size))  # nh, 2*Wh-1 * 2*Ww-1
        elif len(input_resolution) == 3:
            rpe_table_size = (2 * input_resolution[0] - 1) * (2 * input_resolution[1] - 1) * (2 * input_resolution[2] - 1)
            if rpe_mode == 'contextual':
                if 'q' in rpe_contextual_tensor:
                    self.q_rpe_table = nn.Parameter(torch.zeros(num_heads, embed_dim//num_heads, rpe_table_size))  # nh, head_dim, 2*Wh-1 * 2*Ww-1
                if 'k' in rpe_contextual_tensor:
                    self.k_rpe_table = nn.Parameter(torch.zeros(num_heads, embed_dim//num_heads, rpe_table_size))  # nh, head_dim, 2*Wh-1 * 2*Ww-1
                if 'v' in rpe_contextual_tensor:
                    self.v_rpe_table = nn.Parameter(torch.zeros(num_heads, rpe_table_size, embed_dim//num_heads))  # nh, head_dim, 2*Wh-1 * 2*Ww-1
            elif rpe_mode == 'bias':
                self.rpe_table = nn.Parameter(torch.zeros(num_heads, rpe_table_size))  # nh, 2*Wh-1 * 2*Ww-1
        
        if self.q_rpe_table is not None:
            trunc_normal_(self.q_rpe_table, std=.02)
        if self.k_rpe_table is not None:
            trunc_normal_(self.k_rpe_table, std=.02)
        if self.v_rpe_table is not None:
            trunc_normal_(self.v_rpe_table, std=.02)
        if self.rpe_table is not None:
            trunc_normal_(self.rpe_table, std=.02)

        if num_memory_token > 0:
            if self.q_rpe_table is not None:
                memory_rpe_table_q = nn.Parameter(torch.zeros(num_heads, embed_dim//num_heads, 1))
                zeros_(memory_rpe_table_q)
                self.q_rpe_table = nn.Parameter(torch.cat([self.q_rpe_table, memory_rpe_table_q], dim=2))
            if self.k_rpe_table is not None:
                memory_rpe_table_k = nn.Parameter(torch.zeros(num_heads, embed_dim//num_heads, 1))
                zeros_(memory_rpe_table_k)
                self.k_rpe_table = nn.Parameter(torch.cat([self.k_rpe_table, memory_rpe_table_k], dim=2))
            if self.v_rpe_table is not None:
                memory_rpe_table_v = nn.Parameter(torch.zeros(num_heads, 1, embed_dim//num_heads))
                zeros_(memory_rpe_table_v)
                self.v_rpe_table = nn.Parameter(torch.cat([self.v_rpe_table, memory_rpe_table_v], dim=1))
            if self.rpe_table is not None:
                memory_rpe_table = nn.Parameter(torch.zeros(num_heads, 1))
                zeros_(memory_rpe_table)
                self.rpe_table = nn.Parameter(torch.cat([self.rpe_table, memory_rpe_table], dim=1))

        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj_weight = None
        self.k_proj_weight = None
        self.v_proj_weight = None
        self.in_proj_weight = None
        self.in_proj_bias = None
        self.conv_q = None
        self.conv_k = None
        self.conv_v = None

        if self.proj == 'linear' or len(input_resolution) == 1:
            if self._qkv_same_embed_dim is False:
                self.q_proj_weight = Parameter(torch.empty((embed_dim, embed_dim), **factory_kwargs))
                self.k_proj_weight = Parameter(torch.empty((embed_dim, self.kdim), **factory_kwargs))
                self.v_proj_weight = Parameter(torch.empty((embed_dim, self.vdim), **factory_kwargs))
                #self.register_parameter('in_proj_weight', None)
            else:
                self.in_proj_weight = Parameter(torch.empty((3 * embed_dim, embed_dim), **factory_kwargs))
                #self.register_parameter('q_proj_weight', None)
                #self.register_parameter('k_proj_weight', None)
                #self.register_parameter('v_proj_weight', None)

            if bias:
                self.in_proj_bias = Parameter(torch.empty(3 * embed_dim, **factory_kwargs))
            else:
                self.register_parameter('in_proj_bias', None)
        elif self.proj == 'conv':
            self.conv_q = depthwise_separable_conv(embed_dim, embed_dim)
            self.conv_k = depthwise_separable_conv(embed_dim, embed_dim)
            self.conv_v = depthwise_separable_conv(embed_dim, embed_dim)

        self.out_proj = NonDynamicallyQuantizableLinear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

        if add_bias_kv:
            self.bias_k = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
            self.bias_v = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        if self.proj == 'linear' or len(self.input_resolution) == 1:
            if self._qkv_same_embed_dim:
                xavier_uniform_(self.in_proj_weight)
            else:
                xavier_uniform_(self.q_proj_weight)
                xavier_uniform_(self.k_proj_weight)
                xavier_uniform_(self.v_proj_weight)

            if self.in_proj_bias is not None:
                constant_(self.in_proj_bias, 0.)
                constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def __setstate__(self, state):
        # Support loading old MultiheadAttention checkpoints generated by v1.1.0
        if '_qkv_same_embed_dim' not in state:
            state['_qkv_same_embed_dim'] = True

        super(MultiheadAttention, self).__setstate__(state)

    def forward(self, 
                query: Tensor, 
                key: Tensor, 
                value: Tensor, 
                key_padding_mask: Optional[Tensor] = None,
                need_weights: bool = True, 
                attn_mask: Optional[Tensor] = None
                ) -> Tuple[Tensor, Optional[Tensor]]:
        r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. When given a binary mask and a value is True,
            the corresponding value on the attention layer will be ignored. When given
            a byte mask and a value is non-zero, the corresponding value on the attention
            layer will be ignored
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.

    Shapes for inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension. :math:`(N, L, E)` if ``batch_first`` is ``True``.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension. :math:`(N, S, E)` if ``batch_first`` is ``True``.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension. :math:`(N, S, E)` if ``batch_first`` is ``True``.
        - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a ByteTensor is provided, the non-zero positions will be ignored while the position
          with the zero positions will be unchanged. If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: if a 2D mask: :math:`(L, S)` where L is the target sequence length, S is the
          source sequence length.

          If a 3D mask: :math:`(N\cdot\text{num\_heads}, L, S)` where N is the batch size, L is the target sequence
          length, S is the source sequence length. ``attn_mask`` ensure that position i is allowed to attend
          the unmasked positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
          while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
          is not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.

    Shapes for outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension. :math:`(N, L, E)` if ``batch_first`` is ``True``.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
        """
        if self.batch_first:
            query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        if not self._qkv_same_embed_dim:
            attn_output, attn_output_weights = multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.device, self.out_proj.weight, self.out_proj.bias,
                in_proj_weight=self.in_proj_weight,
                in_proj_bias=self.in_proj_bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight,
                rpe_weight_table=self.rpe_table,
                rpe_weight_table_q=self.q_rpe_table,
                rpe_weight_table_k=self.k_rpe_table,
                rpe_weight_table_v=self.v_rpe_table,
                rpe_indices=self.relative_position_index,
                conv_proj_q=self.conv_q,
                conv_proj_k=self.conv_k,
                conv_proj_v=self.conv_v,
                input_resolution=self.input_resolution,
                proj=self.proj)
        else:
            attn_output, attn_output_weights = multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.device, self.out_proj.weight, self.out_proj.bias,
                in_proj_weight=self.in_proj_weight,
                in_proj_bias=self.in_proj_bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask,
                rpe_weight_table=self.rpe_table,
                rpe_weight_table_q=self.q_rpe_table,
                rpe_weight_table_k=self.k_rpe_table,
                rpe_weight_table_v=self.v_rpe_table,
                rpe_indices=self.relative_position_index,
                conv_proj_q=self.conv_q,
                conv_proj_k=self.conv_k,
                conv_proj_v=self.conv_v,
                input_resolution=self.input_resolution,
                proj=self.proj)
        if self.batch_first:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights    


def multi_head_attention_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    embed_dim_to_check: int,
    num_heads: int,
    bias_k: Optional[Tensor],
    bias_v: Optional[Tensor],
    add_zero_attn: bool,
    dropout_p: float,
    device,
    out_proj_weight: Tensor,
    out_proj_bias: Optional[Tensor],
    in_proj_weight: Tensor = None,
    in_proj_bias: Optional[Tensor] = None,
    training: bool = True,
    key_padding_mask: Optional[Tensor] = None,
    need_weights: bool = True,
    attn_mask: Optional[Tensor] = None,
    use_separate_proj_weight: bool = False,
    q_proj_weight: Optional[Tensor] = None,
    k_proj_weight: Optional[Tensor] = None,
    v_proj_weight: Optional[Tensor] = None,
    static_k: Optional[Tensor] = None,
    static_v: Optional[Tensor] = None,
    average_attn_weights: bool = True,
    rpe_weight_table = None,
    rpe_weight_table_q = None,
    rpe_weight_table_k = None,
    rpe_weight_table_v = None,
    rpe_indices = None,
    conv_proj_q=None,
    conv_proj_k=None,
    conv_proj_v=None,
    input_resolution = None,
    proj = 'linear'
) -> Tuple[Tensor, Optional[Tensor]]:
    r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        embed_dim_to_check: total dimension of the model.
        num_heads: parallel attention heads.
        in_proj_weight, in_proj_bias: input projection weight and bias.
        bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        dropout_p: probability of an element to be zeroed.
        out_proj_weight, out_proj_bias: the output projection weight and bias.
        training: apply dropout if is ``True``.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.
        use_separate_proj_weight: the function accept the proj. weights for query, key,
            and value in different forms. If false, in_proj_weight will be used, which is
            a combination of q_proj_weight, k_proj_weight, v_proj_weight.
        q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
        static_k, static_v: static key and value used for attention operators.
        average_attn_weights: If true, indicates that the returned ``attn_weights`` should be averaged across heads.
            Otherwise, ``attn_weights`` are provided separately per head. Note that this flag only has an effect
            when ``need_weights=True.``. Default: True
    Shape:
        Inputs:
        - query: :math:`(L, E)` or :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, E)` or :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, E)` or :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(S)` or :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a ByteTensor is provided, the non-zero positions will be ignored while the zero positions
          will be unchanged. If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
          3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length. attn_mask ensures that position i is allowed to attend the unmasked
          positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
          while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
          are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.
        - static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        - static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        Outputs:
        - attn_output: :math:`(L, E)` or :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: Only returned when ``need_weights=True``. If ``average_attn_weights=True``, returns
          attention weights averaged across heads of shape :math:`(L, S)` when input is unbatched or
          :math:`(N, L, S)`, where :math:`N` is the batch size, :math:`L` is the target sequence length, and
          :math:`S` is the source sequence length. If ``average_weights=False``, returns attention weights per
          head of shape :math:`(num_heads, L, S)` when input is unbatched or :math:`(N, num_heads, L, S)`.
    """
    tens_ops = (query, key, value, in_proj_weight, in_proj_bias, bias_k, bias_v, out_proj_weight, out_proj_bias)
    if has_torch_function(tens_ops):
        return handle_torch_function(
            multi_head_attention_forward,
            tens_ops,
            query,
            key,
            value,
            embed_dim_to_check,
            num_heads,
            in_proj_weight,
            in_proj_bias,
            bias_k,
            bias_v,
            add_zero_attn,
            dropout_p,
            out_proj_weight,
            out_proj_bias,
            training=training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            use_separate_proj_weight=use_separate_proj_weight,
            q_proj_weight=q_proj_weight,
            k_proj_weight=k_proj_weight,
            v_proj_weight=v_proj_weight,
            static_k=static_k,
            static_v=static_v,
        )
    is_batched = _mha_shape_check(query, key, value, key_padding_mask, attn_mask, num_heads)

    # For unbatched input, we unsqueeze at the expected batch-dim to pretend that the input
    # is batched, run the computation and before returning squeeze the
    # batch dimension so that the output doesn't carry this temporary batch dimension.
    if not is_batched:
        # unsqueeze if the input is unbatched
        query = query.unsqueeze(1)
        key = key.unsqueeze(1)
        value = value.unsqueeze(1)
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(0)

    # set up shape vars
    tgt_len, bsz, embed_dim = query.shape
    src_len, _, _ = key.shape
    assert embed_dim == embed_dim_to_check, \
        f"was expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}"
    if isinstance(embed_dim, torch.Tensor):
        # embed_dim can be a tensor when JIT tracing
        head_dim = embed_dim.div(num_heads, rounding_mode='trunc')
    else:
        head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
    if use_separate_proj_weight:
        # allow MHA to have different embedding dimensions when separate projection weights are used
        assert key.shape[:2] == value.shape[:2], \
            f"key's sequence and batch dims {key.shape[:2]} do not match value's {value.shape[:2]}"
    else:
        assert key.shape == value.shape, f"key shape {key.shape} does not match value shape {value.shape}"

    #
    # compute in-projection
    #
    if proj == 'linear':
        if not use_separate_proj_weight:
            q, k, v = _in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)
        else:
            assert q_proj_weight is not None, "use_separate_proj_weight is True but q_proj_weight is None"
            assert k_proj_weight is not None, "use_separate_proj_weight is True but k_proj_weight is None"
            assert v_proj_weight is not None, "use_separate_proj_weight is True but v_proj_weight is None"
            if in_proj_bias is None:
                b_q = b_k = b_v = None
            else:
                b_q, b_k, b_v = in_proj_bias.chunk(3)
            q, k, v = _in_projection(query, key, value, q_proj_weight, k_proj_weight, v_proj_weight, b_q, b_k, b_v)
    elif proj == 'conv':
        query = query.permute(1, 0, 2)
        key = key.permute(1, 0, 2)
        value = value.permute(1, 0, 2)
        q = conv_proj_q(query).permute(1, 0, 2)
        k = conv_proj_k(key).permute(1, 0, 2)
        v = conv_proj_v(value).permute(1, 0, 2)

    # prep attention mask
    if attn_mask is not None:
        if attn_mask.dtype == torch.uint8:
            warnings.warn("Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
            attn_mask = attn_mask.to(torch.bool)
        else:
            assert attn_mask.is_floating_point() or attn_mask.dtype == torch.bool, \
                f"Only float, byte, and bool types are supported for attn_mask, not {attn_mask.dtype}"
        # ensure attn_mask's dim is 3
        if attn_mask.dim() == 2:
            correct_2d_size = (tgt_len, src_len)
            if attn_mask.shape != correct_2d_size:
                raise RuntimeError(f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}.")
            attn_mask = attn_mask.unsqueeze(0)
        elif attn_mask.dim() == 3:
            correct_3d_size = (bsz * num_heads, tgt_len, src_len)
            if attn_mask.shape != correct_3d_size:
                raise RuntimeError(f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}.")
        else:
            raise RuntimeError(f"attn_mask's dimension {attn_mask.dim()} is not supported")

    # prep key padding mask
    if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
        warnings.warn("Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
        key_padding_mask = key_padding_mask.to(torch.bool)

    # add bias along batch dimension (currently second)
    if bias_k is not None and bias_v is not None:
        assert static_k is None, "bias cannot be added to static key."
        assert static_v is None, "bias cannot be added to static value."
        k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
        v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))
    else:
        assert bias_k is None
        assert bias_v is None

    #
    # reshape q, k, v for multihead attention and make em batch first
    #
    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if static_k is None:
        k = k.contiguous().view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    else:
        # TODO finish disentangling control flow so we don't do in-projections when statics are passed
        assert static_k.size(0) == bsz * num_heads, \
            f"expecting static_k.size(0) of {bsz * num_heads}, but got {static_k.size(0)}"
        assert static_k.size(2) == head_dim, \
            f"expecting static_k.size(2) of {head_dim}, but got {static_k.size(2)}"
        k = static_k
    if static_v is None:
        v = v.contiguous().view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    else:
        # TODO finish disentangling control flow so we don't do in-projections when statics are passed
        assert static_v.size(0) == bsz * num_heads, \
            f"expecting static_v.size(0) of {bsz * num_heads}, but got {static_v.size(0)}"
        assert static_v.size(2) == head_dim, \
            f"expecting static_v.size(2) of {head_dim}, but got {static_v.size(2)}"
        v = static_v

    # add zero attention along batch dimension (now first)
    if add_zero_attn:
        zero_attn_shape = (bsz * num_heads, 1, head_dim)
        k = torch.cat([k, torch.zeros(zero_attn_shape, dtype=k.dtype, device=k.device)], dim=1)
        v = torch.cat([v, torch.zeros(zero_attn_shape, dtype=v.dtype, device=v.device)], dim=1)
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))

    # update source sequence length after adjustments
    src_len = k.size(1)

    # merge key padding and attention masks
    if key_padding_mask is not None:
        assert key_padding_mask.shape == (bsz, src_len), \
            f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"
        key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len).   \
            expand(-1, num_heads, -1, -1).reshape(bsz * num_heads, 1, src_len)
        if attn_mask is None:
            attn_mask = key_padding_mask
        elif attn_mask.dtype == torch.bool:
            attn_mask = attn_mask.logical_or(key_padding_mask)
        else:
            attn_mask = attn_mask.masked_fill(key_padding_mask, float("-inf"))

    # convert mask to float
    if attn_mask is not None and attn_mask.dtype == torch.bool:
        new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
        new_attn_mask.masked_fill_(attn_mask, float("-inf"))
        attn_mask = new_attn_mask

    # adjust dropout probability
    if not training:
        dropout_p = 0.0

    #
    # (deep breath) calculate attention and out projection
    #
    attn_output, attn_output_weights = _scaled_dot_product_attention(q, k, v, bsz, num_heads, device, rpe_weight_table, rpe_weight_table_q, rpe_weight_table_k, rpe_weight_table_v, rpe_indices, attn_mask, dropout_p)
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = linear(attn_output, out_proj_weight, out_proj_bias)

    if need_weights:
        # optionally average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        if average_attn_weights:
            attn_output_weights = attn_output_weights.sum(dim=1) / num_heads

        if not is_batched:
            # squeeze the output if input was unbatched
            attn_output = attn_output.squeeze(1)
            attn_output_weights = attn_output_weights.squeeze(0)
        return attn_output, attn_output_weights
    else:
        if not is_batched:
            # squeeze the output if input was unbatched
            attn_output = attn_output.squeeze(1)
        return attn_output, None


def _scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    batch_size,
    n_heads,
    device,
    rpe_weight_table = None,
    rpe_weight_table_q = None,
    rpe_weight_table_k = None,
    rpe_weight_table_v = None,
    rpe_indices = None,
    attn_mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
) -> Tuple[Tensor, Tensor]:
    r"""
    Computes scaled dot product attention on query, key and value tensors, using
    an optional attention mask if passed, and applying dropout if a probability
    greater than 0.0 is specified.
    Returns a tensor pair containing attended values and attention weights.
    Args:
        q, k, v: query, key and value tensors. See Shape section for shape details.
        attn_mask: optional tensor containing mask values to be added to calculated
            attention. May be 2D or 3D; see Shape section for details.
        dropout_p: dropout probability. If greater than 0.0, dropout is applied.
    Shape:
        - q: :math:`(B, Nt, E)` where B is batch size, Nt is the target sequence length,
            and E is embedding dimension.
        - key: :math:`(B, Ns, E)` where B is batch size, Ns is the source sequence length,
            and E is embedding dimension.
        - value: :math:`(B, Ns, E)` where B is batch size, Ns is the source sequence length,
            and E is embedding dimension.
        - attn_mask: either a 3D tensor of shape :math:`(B, Nt, Ns)` or a 2D tensor of
            shape :math:`(Nt, Ns)`.
        - Output: attention values have shape :math:`(B, Nt, E)`; attention weights
            have shape :math:`(B, Nt, Ns)`
    """
    B, Nt, E = q.shape
    B, Ns, E = k.shape
    q = q / math.sqrt(E)
    if rpe_weight_table is not None:
        b = rpe_weight_table[:, rpe_indices]
        attn = torch.bmm(q, k.transpose(-2, -1)).view((batch_size, n_heads) + rpe_indices.shape)
        attn = (attn + b).view((B,) + rpe_indices.shape)
    elif rpe_weight_table_q is not None or rpe_weight_table_k is not None:
        b = torch.zeros((B,) + rpe_indices.shape, device=device)
        if rpe_weight_table_q is not None:
            b_q = torch.matmul(q.view(batch_size, n_heads, Nt, E), rpe_weight_table_q)
            b_q = b_q.flatten(2)[:, :, rpe_indices].view((B,) + rpe_indices.shape)
            b += b_q
        if rpe_weight_table_k is not None:
            b_k = torch.matmul(k.view(batch_size, n_heads, Ns, E), rpe_weight_table_k)
            b_k = b_k.flatten(2)[:, :, rpe_indices].view((B,) + rpe_indices.shape)
            b += b_k
        # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
        attn = torch.bmm(q, k.transpose(-2, -1))
        attn = attn + b
    else:
        attn = torch.bmm(q, k.transpose(-2, -1))

    if attn_mask is not None:
        attn += attn_mask
    attn = softmax(attn, dim=-1)
    if dropout_p > 0.0:
        attn = dropout(attn, p=dropout_p)

    if rpe_weight_table_v is not None and rpe_weight_table is None:
        w = rpe_weight_table_v[:, rpe_indices.flatten()] # 24 49*49 32
        w = w.view(n_heads, Nt, Ns, E)
        r_v = torch.matmul(attn.view(batch_size, n_heads, Nt, Ns).permute(1, 2, 0, 3), w) # 24 49 4 32
        r_v = r_v.permute(2, 0, 1, 3).contiguous().view(B, Nt, E)
        # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
        output = torch.bmm(attn, v)
        output = output + r_v
    else:
        # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
        output = torch.bmm(attn, v)

    return output, attn