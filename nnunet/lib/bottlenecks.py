import torch
import torch.nn as nn
from torch import Tensor
import copy
from typing import Optional
from vit_transformer import SpatialTransformerEncoderLayer, TransformerEncoderLayer
from position_embedding import PositionEmbeddingSine2d, PositionEmbeddingSine1d
import matplotlib.pyplot as plt
from torch.nn.init import zeros_
from vit_rpe import get_indices_1d, get_indices_2d

class temporalTransformer(nn.Module):
    def __init__(self,
                device,
                dpr,
                batch_size,
                num_frames,
                proj,
                d_model=768, 
                nhead=8,
                num_encoder_layers=3, 
                bottleneck_size=[7, 7], 
                num_memory_bus=8,
                dim_feedforward=2048, 
                dropout=0.1,
                activation="gelu", 
                normalize_before=False,
                return_intermediate_dec=False,
                rpe_mode=None,
                rpe_contextual_tensor=None):
        super().__init__()

        #rpe_config = get_rpe_config(ratio=1.9,
        #                            method="product",
        #                            mode='ctx',
        #                            shared_head=False,
        #                            skip=num_memory_bus,
        #                            rpe_on='qkv')

        self.num_frames = num_frames
        self.num_memory_bus = num_memory_bus
        dpr_index = len(dpr) // 2

        rpe_table_size = (2 * bottleneck_size[0] - 1) * (2 * bottleneck_size[1] - 1)
        relative_position_index_2d = get_indices_2d(input_resolution=bottleneck_size, num_memory_bus=num_memory_bus, rpe_table_size=rpe_table_size)
        relative_position_index_1d = get_indices_1d(input_resolution=[batch_size*num_frames])

        spatial_layer = TransformerEncoderLayer(d_model=d_model,
                                            nhead=nhead, 
                                            input_resolution=bottleneck_size,
                                            proj=proj,
                                            device=device,
                                            relative_position_index=relative_position_index_2d,
                                            drop_path=dpr[dpr_index],
                                            num_memory_token=num_memory_bus, 
                                            rpe_mode=rpe_mode, 
                                            rpe_contextual_tensor=rpe_contextual_tensor, 
                                            dim_feedforward=dim_feedforward,
                                            dropout=dropout, 
                                            activation=activation, 
                                            normalize_before=normalize_before)
        
        #spatial_layer = SpatialTransformerEncoderLayer(d_model, nhead, bottleneck_size, rpe_config, dim_feedforward,
        #                                                dropout, activation, normalize_before)

        temporal_layer = TransformerEncoderLayer(d_model=d_model,
                                            nhead=nhead, 
                                            input_resolution=[batch_size*num_frames],
                                            proj=proj,
                                            device=device,
                                            relative_position_index=relative_position_index_1d,
                                            drop_path=dpr[dpr_index],
                                            num_memory_token=0, 
                                            rpe_mode=rpe_mode, 
                                            rpe_contextual_tensor=rpe_contextual_tensor, 
                                            dim_feedforward=dim_feedforward,
                                            dropout=dropout, 
                                            activation=activation, 
                                            normalize_before=normalize_before)

        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = IFCEncoder(num_frames, spatial_layer, temporal_layer, num_encoder_layers, encoder_norm)

        self._reset_parameters()

        N_steps = d_model // 2
        self.position_encoding2d = PositionEmbeddingSine2d(N_steps, normalize=True)
        self.position_encoding1d = PositionEmbeddingSine1d(d_model, normalize=True)
        self.pos = self.position_encoding2d(shape_util=(num_frames, bottleneck_size[0], bottleneck_size[1]), device=device)
        self.temporal_pos = self.position_encoding1d(shape_util=(num_memory_bus, num_frames), device=device)

        self.memory_bus = torch.nn.Parameter(torch.randn(num_memory_bus, d_model))
        self.memory_pos = torch.nn.Parameter(torch.randn(num_memory_bus, d_model))
        if num_memory_bus:
            nn.init.kaiming_normal_(self.memory_bus, mode="fan_out", nonlinearity="relu")
            zeros_(self.memory_pos)

        self.return_intermediate_dec = return_intermediate_dec

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def pad_zero(self, x, pad, dim=0):
        if x is None:
            return None
        pad_shape = list(x.shape)
        pad_shape[dim] = pad
        return torch.cat((x, x.new_zeros(pad_shape)), dim=dim)

    def forward(self, src, is_train):
        # prepare for enc-dec
        bs = src.shape[0] // self.num_frames if is_train else 1
        t = src.shape[0] // bs
        _, c, h, w = src.shape

        memory_bus = self.memory_bus
        memory_pos = self.memory_pos

        # encoder
        src = src.view(bs*t, c, h*w).permute(2, 0, 1)               # HW, BT, C
        frame_pos = self.pos.view(bs*t, c, h*w).permute(2, 0, 1)   # HW, BT, C
        #frame_mask = mask.view(bs*t, h*w)                          # BT, HW
        frame_mask=None

        src, memory_bus = self.encoder(src, memory_bus, memory_pos, self.temporal_pos, src_key_padding_mask=frame_mask, pos=frame_pos, is_train=is_train)

        return src, memory_bus

class IFCEncoder(nn.Module):
    def __init__(self, num_frames, spatial_layer, temporal_layer, num_layers, norm=None):
        super().__init__()
        self.num_frames = num_frames
        self.num_layers = num_layers
        self.enc_layers = nn.ModuleList([copy.deepcopy(spatial_layer) for i in range(num_layers)])
        self.bus_layers = nn.ModuleList([copy.deepcopy(temporal_layer) for i in range(num_layers)])
        norm = [copy.deepcopy(norm) for i in range(2)]
        self.out_norm, self.bus_norm = norm

    def pad_zero(self, x, pad, dim=0):
        if x is None:
            return None
        pad_shape = list(x.shape)
        pad_shape[dim] = pad
        return torch.cat((x, x.new_zeros(pad_shape)), dim=dim)

    def forward(self, src, memory_bus, memory_pos, temporal_pos,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                is_train: bool = True):
        bs = src.shape[1] // self.num_frames if is_train else 1
        t = src.shape[1] // bs
        hw, _, c = src.shape
        M = len(memory_bus)

        memory_bus = memory_bus[:, None, :].repeat(1, bs*t, 1)
        memory_pos = memory_pos[:, None, :].repeat(1, bs*t, 1)

        pos = torch.cat((memory_pos, pos))
        mask = self.pad_zero(mask, M, dim=1)
        src_key_padding_mask = self.pad_zero(src_key_padding_mask, M, dim=1)

        output = src

        for layer_idx in range(self.num_layers):
            output = torch.cat((memory_bus, output))

            output = self.enc_layers[layer_idx](output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)
            output, memory_bus = output[M:, :, :], output[:M, :, :]

            memory_bus = memory_bus.view(M, bs, t, c).permute(2,1,0,3).flatten(1,2) # TxBMxC
            memory_bus = self.bus_layers[layer_idx](memory_bus, pos=temporal_pos)
            memory_bus = memory_bus.view(t, bs, M, c).permute(2,1,0,3).flatten(1,2) # MxBTxC

        if self.out_norm is not None:
            output = self.out_norm(output)
        if self.bus_norm is not None:
            memory_bus = self.bus_norm(memory_bus)

        return output, memory_bus