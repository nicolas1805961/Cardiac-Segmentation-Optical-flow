# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import psutil
import os
from nnunet.analysis import flop_count_operators
import matplotlib
from copy import copy
from math import ceil
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
from ..lib.vit_transformer import SpatialTransformerLayer, ChannelAttention, TransformerEncoderLayer, TransformerEncoder, CrossTransformerEncoderLayer, CrossTransformerEncoder, RelativeTransformerEncoderLayer
from batchgenerators.augmentations.utils import pad_nd_image
from ..training.dataloading.dataset_loading import get_idx, select_idx
from ..lib.position_embedding import PositionEmbeddingSine2d, PositionEmbeddingLearned
from torch.nn import init

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


class MTLmodel(SegmentationNetwork):
    def __init__(self,
                binary,
                attention_map,
                shortcut,
                patch_size, 
                window_size,
                swin_abs_pos,
                deep_supervision,
                proj,
                num_classes,
                out_encoder_dims,
                use_conv_mlp,
                uncertainty_weighting,
                device,
                similarity_down_scale,
                concat_spatial_cross_attention,
                encoder_attention_type,
                spatial_cross_attention_num_heads,
                merge,
                reconstruction,
                reconstruction_skip,
                middle,
                middle_unlabeled,
                classification,
                log_function,
                batch_size,
                in_dims,
                image_size,
                num_bottleneck_layers, 
                directional_field,
                conv_layer,
                conv_depth,
                middle_classification,
                num_heads,
                one_vs_all,
                separability,
                transformer_depth,
                filter_skip_co_segmentation,
                bottleneck_heads, 
                adversarial_loss,
                nb_repeat,
                v1,
                mix_residual,
                transformer_bottleneck,
                registered_seg,
                affinity,
                asymmetric_unet,
                norm,
                add_extra_bottleneck_blocks,
                bottleneck='swin',
                qkv_bias=True, 
                qk_scale=None,
                drop_rate=0., 
                attn_drop_rate=0., 
                drop_path_rate=0.1,
                use_checkpoint=False, 
                rpe_mode=None, 
                rpe_contextual_tensor=None):
        super(MTLmodel, self).__init__()
        
        self.num_stages = (len(transformer_depth) + len(conv_depth))
        self.num_bottleneck_layers = num_bottleneck_layers
        self.bottleneck_name = bottleneck
        self.d_model = out_encoder_dims[-1] * 2
        self.bottleneck_size = [int(image_size / (2**self.num_stages)), int(image_size / (2**self.num_stages))]
        self.reconstruction = reconstruction
        self.registered_seg = registered_seg
        self.image_size = image_size
        self.batch_size = batch_size
        self.v1 = v1
        self.middle_unlabeled = middle_unlabeled
        self.classification = classification
        self.binary = binary
        self.do_ds = deep_supervision
        self.conv_op=nn.Conv2d
        self.middle = middle
        self.one_vs_all = one_vs_all
        self.percent = None
        self.nb_repeat = nb_repeat
        self.middle_classification = middle_classification
        self.asymmetric_unet = asymmetric_unet
        self.log_function = log_function
        self.transformer_bottleneck = transformer_bottleneck
        self.affinity = affinity
        self.mix_residual = mix_residual
        self.separability = separability
        self.add_extra_bottleneck_blocks = add_extra_bottleneck_blocks
        if uncertainty_weighting:
            self.logsigma = nn.Parameter(torch.FloatTensor([1.61, -0.7, -0.7]))
        else:
            self.logsigma = [None] * 3
        
        self.num_classes = num_classes

        self.adversarial_loss = adversarial_loss

        # stochastic depth
        num_blocks = conv_depth + transformer_depth + [num_bottleneck_layers]
        my_iter = torch.cat([torch.tensor([0]), torch.cumsum(torch.tensor(num_blocks), dim=0)])
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(num_blocks))]
        dpr = [dpr[my_iter[j]:my_iter[j+1]] for j in range(len(num_blocks))]
        dpr_encoder = dpr[:self.num_stages]
        dpr_decoder = [x[::-1] for x in dpr_encoder[::-1]]
        dpr_bottleneck = dpr[-1]

        H, W = (int(image_size / 2**(self.num_stages)), int(image_size / 2**(self.num_stages)))

        self.patch_size = patch_size
        self.encoder = Encoder(conv_layer=conv_layer, norm=norm, out_dims=out_encoder_dims, device=device, in_dims=in_dims, conv_depth=conv_depth, dpr=dpr_encoder)

        if asymmetric_unet:
            conv_depth_decoder = [x//2 for x in conv_depth[::-1]]
        else:
            conv_depth_decoder = conv_depth[::-1]

        if not self.middle:
            decoder_output_dims = in_dims[::-1]
            decoder_output_dims[-1] = self.num_classes
            H, W = (int(image_size / 2**(self.num_stages)), int(image_size / 2**(self.num_stages)))
            self.decoder = decoder_alt.SegmentationDecoder(conv_layer=conv_layer, norm=norm, similarity_down_scale=similarity_down_scale, filter_skip_co_segmentation=filter_skip_co_segmentation, directional_field=directional_field, attention_map=attention_map, reconstruction=reconstruction, reconstruction_skip=reconstruction_skip, concat_spatial_cross_attention=concat_spatial_cross_attention, attention_type=encoder_attention_type, spatial_cross_attention_num_heads=spatial_cross_attention_num_heads[::-1], shortcut=shortcut, proj_qkv=proj, out_encoder_dims=out_encoder_dims[::-1], use_conv_mlp=use_conv_mlp, last_activation='identity', img_size=image_size, num_classes=self.num_classes, device=device, swin_abs_pos=swin_abs_pos, in_encoder_dims=decoder_output_dims, merge=merge, conv_depth=conv_depth_decoder, transformer_depth=transformer_depth[::-1], dpr=dpr_decoder, rpe_mode=rpe_mode, rpe_contextual_tensor=rpe_contextual_tensor, num_heads=num_heads, window_size=window_size, drop_path_rate=drop_path_rate, deep_supervision=self.do_ds)
            
            self.pos = PositionEmbeddingSine2d(num_pos_feats=self.d_model // 2, normalize=True)
            #self.spatial_pos = nn.Parameter(torch.randn(size=(self.bottleneck_size[0]**2, self.d_model)))
            
            self.extra_bottleneck_block_1 = conv_layer(in_dim=self.d_model, out_dim=self.d_model, nb_blocks=1, dpr=dpr_bottleneck, norm=norm, kernel_size=3)
            if transformer_bottleneck:
                encoder_layer = TransformerEncoderLayer(d_model=int(self.d_model), nhead=bottleneck_heads, dim_feedforward=4 * int(self.d_model))
                self.bottleneck = TransformerEncoder(encoder_layer=encoder_layer, num_layers=self.num_bottleneck_layers)
            else:
                self.bottleneck = nn.Sequential(nn.Conv2d(in_channels=self.d_model, out_channels=self.d_model, kernel_size=3, padding='same'),
                                                norm(self.d_model),
                                                nn.GELU())
                #self.bottleneck = conv_layer(in_dim=self.d_model, out_dim=self.d_model, nb_blocks=1, dpr=dpr_bottleneck, norm=norm, kernel_size=3)
            self.extra_bottleneck_block_2 = conv_layer(in_dim=self.d_model, out_dim=self.d_model, nb_blocks=1, dpr=dpr_bottleneck, norm=norm, kernel_size=3)

            if classification:
                #self.classification_conv = ConvBlock(in_dim=self.d_model, out_dim=1, kernel_size=1, norm=norm, stride=1)

                self.classification_conv = nn.Sequential(nn.Conv2d(in_channels=self.d_model, out_channels=1, kernel_size=1),
                                                        nn.GELU(),
                                                        nn.Dropout(0.5))
                #in_dim_linear = int(H * W * self.d_model)
                self.classification_net = nn.Sequential(nn.Linear(784, 392), nn.GELU(), nn.Dropout(0.5),
                                                        nn.Linear(392, 5))
                #self.classification_net[-1].weight.data.zero_()
                #self.classification_net[-1].bias.data.copy_(torch.tensor([0, 1], dtype=torch.float))
        else:
            self.decoder = decoder_alt.SegmentationDecoder(conv_layer=conv_layer, norm=norm, filter_skip_co_segmentation=filter_skip_co_segmentation, concat_spatial_cross_attention=concat_spatial_cross_attention, spatial_cross_attention_num_heads=spatial_cross_attention_num_heads[::-1], shortcut=shortcut, proj_qkv=proj, out_encoder_dims=out_encoder_dims[::-1], use_conv_mlp=use_conv_mlp, last_activation='identity', img_size=image_size, num_classes=self.num_classes, device=device, swin_abs_pos=swin_abs_pos, in_encoder_dims=in_dims[::-1], conv_depth=conv_depth_decoder, transformer_depth=transformer_depth[::-1], dpr=dpr_decoder, rpe_mode=rpe_mode, rpe_contextual_tensor=rpe_contextual_tensor, num_heads=num_heads, window_size=window_size, drop_path_rate=drop_path_rate, deep_supervision=self.do_ds)
            if registered_seg:
                self.motion_decoder = decoder_alt.SimpleDecoderStages(in_dim=self.d_model, out_dim=num_classes, nb_stages=len(conv_depth), norm=norm, conv_layer=conv_layer, conv_depth=conv_depth_decoder, dpr=dpr_decoder, deep_supervision=False)
            if middle_classification:
                self.reduce = nn.Sequential(torch.nn.MaxPool2d(2),
                    torch.nn.Conv2d(in_channels=self.d_model, out_channels=self.d_model // 4, kernel_size=H // 2))
                self.ff = nn.Sequential(torch.nn.Linear(self.d_model // 4, self.d_model // 8),
                                            torch.nn.GELU(),
                                            torch.nn.Linear(self.d_model // 8, 1))

            dpr_bottleneck = dpr_bottleneck * 2
            self.pos1 = PositionEmbeddingSine2d(num_pos_feats=self.d_model // 2, normalize=True)
            self.pos2 = PositionEmbeddingSine2d(num_pos_feats=self.d_model, normalize=True)
            self.motion_estimation = decoder_alt.MotionEstimation()

            #if self.siamese:
            #    self.compute_cross_sim = GetCrossSimilarityMatrix()
            #    self.bottleneck_layers = nn.ModuleList()
            #    for i in range(self.nb_repeat):
            #        cross_attention_layer = CrossTransformerEncoderLayer(d_model=self.d_model, nhead=bottleneck_heads, dim_feedforward=2048)
            #        cross_attention_bottleneck = CrossTransformerEncoder(encoder_layer=cross_attention_layer, num_layers=1)
            #        self_attention_layer = TransformerEncoderLayer(d_model=self.d_model * 2, nhead=bottleneck_heads, dim_feedforward=2048)
            #        self_attention_bottleneck = TransformerEncoder(encoder_layer=self_attention_layer, num_layers=1)
            #        conv_block = conv_layer(in_dim=self.d_model * 2, out_dim=self.d_model, nb_blocks=1, dpr=dpr_bottleneck, norm=norm, kernel_size=3)
            #        self.bottleneck_layers.append(cross_attention_bottleneck)
            #        self.bottleneck_layers.append(self_attention_bottleneck)
            #        self.bottleneck_layers.append(conv_block)
#
            #else:
            self.compute_cross_sim_md = GetCrossSimilarityMatrix()
            
            if self.middle_unlabeled and not self.v1:
                #cross_attention_layer = CrossTransformerEncoderLayer(d_model=self.d_model, nhead=bottleneck_heads, dim_feedforward=2048)
                #self.cross_attention_bottleneck = CrossTransformerEncoder(encoder_layer=cross_attention_layer, num_layers=1)

                self.filter = Filter(dim=self.d_model, num_heads=bottleneck_heads, norm=norm)

                self_attention_layer = TransformerEncoderLayer(d_model=self.d_model * 2, nhead=bottleneck_heads, dim_feedforward=self.d_model * 2 * 4)
                self.self_attention_bottleneck = TransformerEncoder(encoder_layer=self_attention_layer, num_layers=1)
                self.conv_blocks = conv_layer(in_dim=self.d_model * 2, out_dim=self.d_model, nb_blocks=2, dpr=dpr_bottleneck * 2, norm=norm, kernel_size=3)

                #self_attention_layer_pre = TransformerEncoderLayer(d_model=self.d_model, nhead=bottleneck_heads, dim_feedforward=2048)
                #self.self_attention_bottleneck_pre = TransformerEncoder(encoder_layer=self_attention_layer_pre, num_layers=1)
                self.conv_blocks_pre = conv_layer(in_dim=self.d_model, out_dim=self.d_model, nb_blocks=2, dpr=dpr_bottleneck * 2, norm=norm, kernel_size=3)
                self.filter_pre = Filter(dim=self.d_model, num_heads=bottleneck_heads, norm=norm)
                #cross_attention_layer = CrossTransformerEncoderLayer(d_model=self.d_model, nhead=bottleneck_heads, dim_feedforward=2048)
                #self_attention_layer = TransformerEncoderLayer(d_model=self.d_model * 2, nhead=bottleneck_heads, dim_feedforward=2048)
                #self.cross_attention_bottleneck_1 = CrossTransformerEncoder(encoder_layer=cross_attention_layer, num_layers=1)
                #self.cross_attention_bottleneck_2 = CrossTransformerEncoder(encoder_layer=cross_attention_layer, num_layers=1)
                #self.self_attention_bottleneck_1 = TransformerEncoder(encoder_layer=self_attention_layer, num_layers=1)
                #self.self_attention_bottleneck_2 = TransformerEncoder(encoder_layer=self_attention_layer, num_layers=1)
                #self.conv_blocks_1 = conv_layer(in_dim=self.d_model * 2, out_dim=self.d_model, nb_blocks=2, dpr=dpr_bottleneck * 2, norm=norm, kernel_size=3)
                #self.conv_blocks_2 = conv_layer(in_dim=self.d_model * 2, out_dim=self.d_model, nb_blocks=2, dpr=dpr_bottleneck * 2, norm=norm, kernel_size=3)
            else:
                cross_attention_layer = CrossTransformerEncoderLayer(d_model=self.d_model, nhead=bottleneck_heads, dim_feedforward=self.d_model * 4)
                self_attention_layer_pre = TransformerEncoderLayer(d_model=self.d_model, nhead=bottleneck_heads, dim_feedforward=self.d_model * 4)
                self_attention_layer_post = TransformerEncoderLayer(d_model=self.d_model * 2, nhead=bottleneck_heads, dim_feedforward=self.d_model * 2 * 4)

                self.self_attention_bottleneck_pre = TransformerEncoder(encoder_layer=self_attention_layer_pre, num_layers=1)
                self.self_attention_bottleneck_post = TransformerEncoder(encoder_layer=self_attention_layer_post, num_layers=1)

                self.cross_attention_bottleneck = CrossTransformerEncoder(encoder_layer=cross_attention_layer, num_layers=1)
                self.conv_blocks_pre_1 = conv_layer(in_dim=self.d_model, out_dim=self.d_model, nb_blocks=1, dpr=dpr_bottleneck * 1, norm=norm, kernel_size=3)
                self.conv_blocks_pre_2 = conv_layer(in_dim=self.d_model, out_dim=self.d_model, nb_blocks=1, dpr=dpr_bottleneck * 1, norm=norm, kernel_size=3)
                self.conv_blocks_post_1 = conv_layer(in_dim=self.d_model * 2, out_dim=self.d_model * 2, nb_blocks=1, dpr=dpr_bottleneck * 1, norm=norm, kernel_size=3)
                self.conv_blocks_post_2 = conv_layer(in_dim=self.d_model * 2, out_dim=self.d_model, nb_blocks=1, dpr=dpr_bottleneck * 1, norm=norm, kernel_size=3)

                #self.cross_layers = nn.ModuleList()
                #for i in range(self.nb_repeat):
                #    cross_attention_bottleneck = CrossTransformerEncoder(encoder_layer=cross_attention_layer, num_layers=1)
                #    conv_blocks = conv_layer(in_dim=self.d_model, out_dim=self.d_model, nb_blocks=1, dpr=dpr_bottleneck * 1, norm=norm, kernel_size=3)
                #    self.cross_layers.append(cross_attention_bottleneck)
                #    self.cross_layers.append(conv_blocks)


                #self.cross_attention_bottleneck = CrossTransformerEncoder(encoder_layer=cross_attention_layer, num_layers=self.nb_repeat)
                #self.conv_blocks_post = conv_layer(in_dim=self.d_model * 2, out_dim=self.d_model, nb_blocks=self.nb_repeat, dpr=dpr_bottleneck * self.nb_repeat, norm=norm, kernel_size=3)
                #self.conv_blocks_post = conv_layer(in_dim=self.d_model * 2, out_dim=self.d_model, nb_blocks=self.nb_repeat, dpr=dpr_bottleneck * self.nb_repeat, norm=norm, kernel_size=3)
                

                #self_attention_layer = TransformerEncoderLayer(d_model=self.d_model * 2, nhead=bottleneck_heads, dim_feedforward=2048)
                #self.self_attention_bottleneck = TransformerEncoder(encoder_layer=self_attention_layer, num_layers=1)


            if self.mix_residual:
                self.motion_head = nn.Conv2d(in_channels=4, out_channels=2, kernel_size=1)
                self.reduce_layers = nn.ModuleList()
                for i in range(len(conv_depth)):
                    reduce_layer = conv_layer(in_dim=out_encoder_dims[i] * 2, 
                                            out_dim=out_encoder_dims[i], 
                                            nb_blocks=conv_depth[i], 
                                            kernel_size=3,
                                            dpr=dpr[i],
                                            norm=norm)
                    self.reduce_layers.append(reduce_layer)
    
    
    def mix_skip_co(self, encoder_skip_connections_1, encoder_skip_connections_2):
        out_1 = []
        out_2 = []
        out_3 = []
        for i, layer in enumerate(self.reduce_layers):
            encoder_skip_connection = torch.cat([encoder_skip_connections_1[i], encoder_skip_connections_2[i]], dim=1)
            encoder_skip_connection = layer(encoder_skip_connection)
            encoder_skip_connection_1 = encoder_skip_connection + encoder_skip_connections_1[i]
            encoder_skip_connection_2 = encoder_skip_connection + encoder_skip_connections_2[i]
            encoder_skip_connection_3 = encoder_skip_connection + encoder_skip_connection
            out_1.append(encoder_skip_connection_1)
            out_2.append(encoder_skip_connection_2)
            out_3.append(encoder_skip_connection_3)
        return out_1, out_2, out_3
    
    def process_bottleneck(self, x_encoded_1, x_encoded_2, pos1, pos2):
        #for i in range(0, 2 * self.nb_repeat, 2):
        #    x_encoded_1 = self.cross_layers[i](query=x_encoded_2, key=x_encoded_1, value=x_encoded_1, pos=pos1)
        #    x_encoded_2 = self.cross_layers[i](query=x_encoded_1, key=x_encoded_2, value=x_encoded_2, pos=pos1)
#
        #    x_encoded_1 = self.cross_layers[i + 1](x_encoded_1)
        #    x_encoded_2 = self.cross_layers[i + 1](x_encoded_2)

        x_encoded_1 = self.conv_blocks_pre_1(x_encoded_1)
        x_encoded_2 = self.conv_blocks_pre_1(x_encoded_2)

        x_encoded_1 = self.self_attention_bottleneck_pre(x_encoded_1, pos=pos1)
        x_encoded_2 = self.self_attention_bottleneck_pre(x_encoded_2, pos=pos1)

        x_encoded_1 = self.conv_blocks_pre_2(x_encoded_1)
        x_encoded_2 = self.conv_blocks_pre_2(x_encoded_2)

        value_1 = self.cross_attention_bottleneck(query=x_encoded_2, key=x_encoded_1, value=x_encoded_1, pos=pos1)
        value_2 = self.cross_attention_bottleneck(query=x_encoded_1, key=x_encoded_2, value=x_encoded_2, pos=pos1)

        x_decoded = torch.cat([value_1, value_2], dim=1)

        x_decoded = self.conv_blocks_post_1(x_decoded)
        x_decoded = self.self_attention_bottleneck_post(x_decoded, pos=pos2)
        x_decoded = self.conv_blocks_post_2(x_decoded)

        return x_decoded

    def bottleneck_pre(self, x, pos):
        #x = self.self_attention_bottleneck_pre(x, pos=pos)
        x = self.conv_blocks_pre(x)
        return x

    def forward(self, x):
        reconstruction_sm = None
        reconstructed = None
        decoder_sm = None
        df = None
        seg_aff = None
        parameters = None
        vq_loss = None
        perplexity = None
        classification_out = None
        aff = None
        separability = None
        classification_pred1 = None
        classification_pred2 = None
        decode_out = None
        cross_sim_md = None
        cross_sim_md1 = None
        cross_sim_md2 = None
        cross_sim_l_1 = cross_sim_l_2 = None
        registered_seg_2 = registered_seg_1 = None
        motion_2 = motion_1 = None

        if isinstance(x, dict):
            x_encoded_1, encoder_skip_connections_1 = self.encoder(x['l1'])
            x_encoded_2, encoder_skip_connections_2 = self.encoder(x['l2'])

            B, C, H, W = x_encoded_1.shape
            pos1 = self.pos1(shape_util=(B, H, W), device=x_encoded_1.device)
            pos2 = self.pos2(shape_util=(B, H, W), device=x_encoded_1.device)

            x_decoded = self.process_bottleneck(x_encoded_1, x_encoded_2, pos1=pos1, pos2=pos2)

            with torch.no_grad():
                x_decoded_1 = self.process_bottleneck(x_encoded_1, x_encoded_1, pos1=pos1, pos2=pos2)
                x_decoded_2 = self.process_bottleneck(x_encoded_2, x_encoded_2, pos1=pos1, pos2=pos2)
            cross_sim_l_1 = self.compute_cross_sim_md(x_decoded, x_decoded_1)
            cross_sim_l_2 = self.compute_cross_sim_md(x_decoded, x_decoded_2)

            pred_l_1 = self.decoder(x_decoded, encoder_skip_connections_1)
            pred_l_2 = self.decoder(x_decoded, encoder_skip_connections_2)
            if self.registered_seg:
                motion = self.motion_decoder(x_decoded)[0]
                forward_motion, backward_motion = torch.split(motion, 2, dim=1)
                registered_seg_1 = self.motion_estimation(x=forward_motion, original=pred_l_1[0].detach())
                registered_seg_2 = self.motion_estimation(x=backward_motion, original=pred_l_2[0].detach())
                forward_motion = forward_motion.detach()
                backward_motion = backward_motion.detach()

            if not self.do_ds:
                pred_l_1 = pred_l_1[0]
                pred_l_2 = pred_l_2[0]
                
            out = {'pred_l_1': pred_l_1,
                    'pred_l_2': pred_l_2,
                    'registered_seg_1': registered_seg_1,
                    'registered_seg_2': registered_seg_2,
                    'forward_motion': forward_motion,
                    'backward_motion': backward_motion,
                    'cross_sim_l_1': cross_sim_l_1,
                    'cross_sim_l_2': cross_sim_l_2,
                    }
            return out
            
        else:
            x_encoded, encoder_skip_connections = self.encoder(x)
            x_encoded = self.extra_bottleneck_block_1(x_encoded)
            if self.transformer_bottleneck:
                B, C, H, W = x_encoded.shape
                pos = self.pos(shape_util=(B, H, W), device=x.device)
                pos = torch.flatten(pos, start_dim=2).permute(0, 2, 1)
                #pos = self.spatial_pos.unsqueeze(0).repeat(B, 1, 1)
                x_encoded = self.bottleneck(x_encoded, pos=pos)
            else:
                x_encoded = self.bottleneck(x_encoded)
            x_encoded = self.extra_bottleneck_block_2(x_encoded)

            seg = self.decoder(x_encoded, encoder_skip_connections)
            
            if not self.do_ds:
                seg = seg[0]

            out = {'pred': seg}

            return out
    
    def predict_3D_middle_unlabeled(self, x: np.ndarray, x2: np.ndarray, do_mirroring: bool, mirror_axes: Tuple[int, ...] = (0, 1, 2),
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

        assert len(x.shape) == 4, "data must have shape (c,x,y,z)"
        assert len(x2.shape) == 4, "data must have shape (c,x,y,z)"

        if mixed_precision:
            context = autocast
        else:
            context = no_op

        with context():
            with torch.no_grad():
                if use_sliding_window:
                    res = self._internal_predict_3D_2Dconv_tiled_middle_unlabeled(x, x2, patch_size, do_mirroring, mirror_axes, step_size,
                                                                                regions_class_order, use_gaussian, pad_border_mode,
                                                                                pad_kwargs, all_in_gpu, False)
                        
                else:
                    res = self._internal_predict_3D_2Dconv(x, x2, patch_size, do_mirroring, mirror_axes, regions_class_order,
                                                            pad_border_mode, pad_kwargs, all_in_gpu, False)

        return res

    def predict_3D(self, x: np.ndarray, do_mirroring: bool, mirror_axes: Tuple[int, ...] = (0, 1, 2),
                   use_sliding_window: bool = False,
                   step_size: float = 0.5, patch_size: Tuple[int, ...] = None, regions_class_order: Tuple[int, ...] = None,
                   use_gaussian: bool = False, pad_border_mode: str = "constant",
                   pad_kwargs: dict = None, all_in_gpu: bool = False,
                   verbose: bool = True, mixed_precision: bool = True, get_flops=False) -> Tuple[np.ndarray, np.ndarray]:
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

        assert len(x.shape) == 4, "data must have shape (c,x,y,z)"

        if mixed_precision:
            context = autocast
        else:
            context = no_op

        with context():
            with torch.no_grad():
                if self.conv_op == nn.Conv3d:
                    if use_sliding_window:
                        res = self._internal_predict_3D_3Dconv_tiled(x, step_size, do_mirroring, mirror_axes, patch_size,
                                                                     regions_class_order, use_gaussian, pad_border_mode,
                                                                     pad_kwargs=pad_kwargs, all_in_gpu=all_in_gpu,
                                                                     verbose=verbose)
                    else:
                        res = self._internal_predict_3D_3Dconv(x, patch_size, do_mirroring, mirror_axes, regions_class_order,
                                                               pad_border_mode, pad_kwargs=pad_kwargs, verbose=verbose)
                elif self.conv_op == nn.Conv2d:
                    if use_sliding_window:
                        if self.middle:
                            res = self._internal_predict_3D_2Dconv_tiled_middle(x, patch_size, do_mirroring, mirror_axes, step_size,
                                                                                regions_class_order, use_gaussian, pad_border_mode,
                                                                                pad_kwargs, all_in_gpu, False)
                        else:
                            res = self._internal_predict_3D_2Dconv_tiled(x, patch_size, do_mirroring, mirror_axes, step_size,
                                                                        regions_class_order, use_gaussian, pad_border_mode,
                                                                        pad_kwargs, all_in_gpu, False, get_flops=get_flops)
                    else:
                        res = self._internal_predict_3D_2Dconv(x, patch_size, do_mirroring, mirror_axes, regions_class_order,
                                                               pad_border_mode, pad_kwargs, all_in_gpu, False)
                else:
                    raise RuntimeError("Invalid conv op, cannot determine what dimensionality (2d/3d) the network is")

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
    

    def _internal_maybe_mirror_and_pred_2D(self, x: Union[np.ndarray, torch.tensor], mirror_axes: tuple,
                                            get_time_stats=False,
                                           do_mirroring: bool = True,
                                           mult: np.ndarray or torch.tensor = None):
        # if cuda available:
        #   everything in here takes place on the GPU. If x and mult are not yet on GPU this will be taken care of here
        #   we now return a cuda tensor! Not numpy array!

        assert len(x.shape) == 4, 'x must be (b, c, x, y)'

        x = maybe_to_torch(x)
        result_torch = torch.zeros([x.shape[0], self.num_classes] + list(x.shape[2:]), dtype=torch.float)

        if torch.cuda.is_available():
            x = to_cuda(x, gpu_id=self.get_device())
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

        if get_time_stats:
            out_flop = flop_count_operators(self, x)

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _ = self(x)
            end.record()
            torch.cuda.synchronize()
            inference_time = start.elapsed_time(end)
        else:
            out_flop = inference_time = None

        for m in range(mirror_idx):
            if m == 0:
                pred = self.inference_apply_nonlin(self(x)['pred'])
                result_torch += 1 / num_results * pred

            if m == 1 and (1 in mirror_axes):
                pred = self.inference_apply_nonlin(self(torch.flip(x, (3, )))['pred'])
                result_torch += 1 / num_results * torch.flip(pred, (3, ))

            if m == 2 and (0 in mirror_axes):
                pred = self.inference_apply_nonlin(self(torch.flip(x, (2, )))['pred'])
                result_torch += 1 / num_results * torch.flip(pred, (2, ))

            if m == 3 and (0 in mirror_axes) and (1 in mirror_axes):
                pred = self.inference_apply_nonlin(self(torch.flip(x, (3, 2)))['pred'])
                result_torch += 1 / num_results * torch.flip(pred, (3, 2))
            
            del pred

        if mult is not None:
            result_torch[:, :] *= mult

        return result_torch, out_flop, inference_time

    def _internal_predict_3D_2Dconv_tiled_middle(self, x: np.ndarray, patch_size: Tuple[int, int], do_mirroring: bool,
                                                mirror_axes: tuple = (0, 1), step_size: float = 0.5,
                                                regions_class_order: tuple = None, use_gaussian: bool = False,
                                                pad_border_mode: str = "edge", pad_kwargs: dict =None,
                                                all_in_gpu: bool = False,
                                                verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        if all_in_gpu:
            raise NotImplementedError

        assert len(x.shape) == 4, "data must be c, x, y, z"

        predicted_segmentation = []
        softmax_pred = []

        for s in range(x.shape[1]):

            if self.one_vs_all:
                #self.percent = dict(sorted(self.percent.items()))
                #idx = select_idx(nb_slices=x.shape[1], random_slice=s, keys=list(self.percent.keys()))
                ##idx = min(list(self.percent.keys()), key=lambda t:abs(t - (s / x.shape[1])))
                #percent = self.percent[idx]
                #middle_idx = get_idx(x.shape[1] * percent)
                if s >= 1:
                    middle_idx = s - 1
                else:
                    middle_idx = 1
            else:
                middle_idx = get_idx(x.shape[1] * self.percent)

            pred_seg, softmax_pres = self._internal_predict_2D_2Dconv_tiled_middle(
                x[:, s], x[:, middle_idx], step_size, do_mirroring, mirror_axes, patch_size, regions_class_order, use_gaussian,
                pad_border_mode, pad_kwargs, all_in_gpu, verbose)

            predicted_segmentation.append(pred_seg[None])
            softmax_pred.append(softmax_pres[None])

        predicted_segmentation = np.vstack(predicted_segmentation)
        softmax_pred = np.vstack(softmax_pred).transpose((1, 0, 2, 3))

        return predicted_segmentation, softmax_pred
    
    def _internal_predict_3D_2Dconv_tiled_middle_unlabeled(self, x: np.ndarray, x2: np.ndarray, patch_size: Tuple[int, int], do_mirroring: bool,
                                                mirror_axes: tuple = (0, 1), step_size: float = 0.5,
                                                regions_class_order: tuple = None, use_gaussian: bool = False,
                                                pad_border_mode: str = "edge", pad_kwargs: dict =None,
                                                all_in_gpu: bool = False,
                                                verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        if all_in_gpu:
            raise NotImplementedError

        assert len(x.shape) == 4, "data must be c, x, y, z"
        assert len(x2.shape) == 4, "data must be c, x, y, z"

        predicted_segmentation = []
        softmax_pred = []

        for s in range(x.shape[1]):

            if self.one_vs_all:
                #self.percent = dict(sorted(self.percent.items()))
                #idx = select_idx(nb_slices=x.shape[1], random_slice=s, keys=list(self.percent.keys()))
                ##idx = min(list(self.percent.keys()), key=lambda t:abs(t - (s / x.shape[1])))
                #percent = self.percent[idx]
                #middle_idx = get_idx(x.shape[1] * percent)
                if s >= 1:
                    middle_idx = s - 1
                else:
                    middle_idx = 1
            else:
                middle_idx = get_idx(x.shape[1] * self.percent)

            pred_seg, softmax_pres = self._internal_predict_2D_2Dconv_tiled_middle_unlabeled(
                x[:, s], x2[:, s], x[:, middle_idx], step_size, do_mirroring, mirror_axes, patch_size, regions_class_order, use_gaussian,
                pad_border_mode, pad_kwargs, all_in_gpu, verbose)

            predicted_segmentation.append(pred_seg[None])
            softmax_pred.append(softmax_pres[None])

        predicted_segmentation = np.vstack(predicted_segmentation)
        softmax_pred = np.vstack(softmax_pred).transpose((1, 0, 2, 3))

        return predicted_segmentation, softmax_pred

    def _internal_predict_2D_2Dconv_tiled_middle(self, x: np.ndarray, middle, step_size: float, do_mirroring: bool, mirror_axes: tuple,
                                          patch_size: tuple, regions_class_order: tuple, use_gaussian: bool,
                                          pad_border_mode: str, pad_kwargs: dict, all_in_gpu: bool,
                                          verbose: bool) -> Tuple[np.ndarray, np.ndarray]:
        # better safe than sorry
        assert len(x.shape) == 3, "x must be (c, x, y)"

        if verbose: print("step_size:", step_size)
        if verbose: print("do mirror:", do_mirroring)

        assert patch_size is not None, "patch_size cannot be None for tiled prediction"

        # for sliding window inference the image must at least be as large as the patch size. It does not matter
        # whether the shape is divisible by 2**num_pool as long as the patch size is
        data, slicer = pad_nd_image(x, patch_size, pad_border_mode, pad_kwargs, True, None)
        middle_data, _ = pad_nd_image(middle, patch_size, pad_border_mode, pad_kwargs, True, None)
        data_shape = data.shape  # still c, x, y


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
            aggregated_results = torch.zeros([self.num_classes] + list(data.shape[1:]), dtype=torch.half,
                                             device=self.get_device())

            if verbose: print("moving data to GPU")
            data = torch.from_numpy(data).cuda(self.get_device(), non_blocking=True)
            middle_data = torch.from_numpy(middle_data).cuda(self.get_device(), non_blocking=True)

            if verbose: print("initializing result_numsamples (on GPU)")
            aggregated_nb_of_predictions = torch.zeros([self.num_classes] + list(data.shape[1:]), dtype=torch.half,
                                                       device=self.get_device())
        else:
            if use_gaussian and num_tiles > 1:
                add_for_nb_of_preds = self._gaussian_2d
            else:
                add_for_nb_of_preds = np.ones(patch_size, dtype=np.float32)
            aggregated_results = np.zeros([self.num_classes] + list(data.shape[1:]), dtype=np.float32)
            aggregated_nb_of_predictions = np.zeros([self.num_classes] + list(data.shape[1:]), dtype=np.float32)

        for x in steps[0]:
            lb_x = x
            ub_x = x + patch_size[0]
            for y in steps[1]:
                lb_y = y
                ub_y = y + patch_size[1]
                predicted_patch = self._internal_maybe_mirror_and_pred_2D_middle(
                    data[None, :, lb_x:ub_x, lb_y:ub_y], middle_data[None, :, lb_x:ub_x, lb_y:ub_y], mirror_axes, do_mirroring,
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


    def _internal_predict_2D_2Dconv_tiled_middle_unlabeled(self, l1: np.ndarray, l2: np.ndarray, u1: np.ndarray, step_size: float, do_mirroring: bool, mirror_axes: tuple,
                                          patch_size: tuple, regions_class_order: tuple, use_gaussian: bool,
                                          pad_border_mode: str, pad_kwargs: dict, all_in_gpu: bool,
                                          verbose: bool) -> Tuple[np.ndarray, np.ndarray]:
        # better safe than sorry
        assert len(l1.shape) == 3, "x must be (c, x, y)"
        assert len(l2.shape) == 3, "x must be (c, x, y)"
        assert len(u1.shape) == 3, "x must be (c, x, y)"

        if verbose: print("step_size:", step_size)
        if verbose: print("do mirror:", do_mirroring)

        assert patch_size is not None, "patch_size cannot be None for tiled prediction"

        # for sliding window inference the image must at least be as large as the patch size. It does not matter
        # whether the shape is divisible by 2**num_pool as long as the patch size is
        data_l1, slicer = pad_nd_image(l1, patch_size, pad_border_mode, pad_kwargs, True, None)
        data_l2, _ = pad_nd_image(l2, patch_size, pad_border_mode, pad_kwargs, True, None)
        data_u1, _ = pad_nd_image(u1, patch_size, pad_border_mode, pad_kwargs, True, None)
        data_shape = data_l1.shape  # still c, x, y

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
            aggregated_results = torch.zeros([self.num_classes] + list(data_l1.shape[1:]), dtype=torch.half,
                                             device=self.get_device())

            if verbose: print("moving data to GPU")
            data_l1 = torch.from_numpy(data_l1).cuda(self.get_device(), non_blocking=True)
            data_l2 = torch.from_numpy(data_l2).cuda(self.get_device(), non_blocking=True)
            data_u1 = torch.from_numpy(data_u1).cuda(self.get_device(), non_blocking=True)

            if verbose: print("initializing result_numsamples (on GPU)")
            aggregated_nb_of_predictions = torch.zeros([self.num_classes] + list(data_l1.shape[1:]), dtype=torch.half,
                                                       device=self.get_device())
        else:
            if use_gaussian and num_tiles > 1:
                add_for_nb_of_preds = self._gaussian_2d
            else:
                add_for_nb_of_preds = np.ones(patch_size, dtype=np.float32)
            aggregated_results = np.zeros([self.num_classes] + list(data_l1.shape[1:]), dtype=np.float32)
            aggregated_nb_of_predictions = np.zeros([self.num_classes] + list(data_l1.shape[1:]), dtype=np.float32)

        for x in steps[0]:
            lb_x = x
            ub_x = x + patch_size[0]
            for y in steps[1]:
                lb_y = y
                ub_y = y + patch_size[1]
                predicted_patch = self._internal_maybe_mirror_and_pred_2D_middle_unlabeled(
                    data_l1[None, :, lb_x:ub_x, lb_y:ub_y], data_l2[None, :, lb_x:ub_x, lb_y:ub_y], data_u1[None, :, lb_x:ub_x, lb_y:ub_y], mirror_axes, do_mirroring,
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


class PolicyNet(nn.Module):
    def __init__(self, 
                blur,
                blur_kernel,
                shortcut,
                patch_size, 
                window_size, 
                swin_abs_pos,
                proj,
                out_encoder_dims,
                use_conv_mlp,
                device,
                mlp_intermediary_dim,
                in_dims,
                image_size,
                conv_depth,
                transformer_depth, 
                num_heads,
                bottleneck_heads, 
                transformer_type='swin',
                bottleneck='swin',
                num_bottleneck_layers=2, 
                mlp_ratio=4., 
                qkv_bias=True, 
                qk_scale=None,
                drop_rate=0., 
                attn_drop_rate=0., 
                drop_path_rate=0.1, 
                norm_layer=nn.LayerNorm, 
                use_checkpoint=False, 
                rpe_mode=None, 
                rpe_contextual_tensor=None):
        super().__init__()
        
        self.num_stages = (len(transformer_depth) + len(conv_depth))
        self.num_bottleneck_layers = num_bottleneck_layers
        self.bottleneck_name = bottleneck
        self.d_model = out_encoder_dims[-1] * 2
        self.bottleneck_size = [int(image_size / (x * 2**(self.num_stages - 2))) for x in patch_size]

        # stochastic depth
        num_blocks = conv_depth + transformer_depth + [num_bottleneck_layers]
        my_iter = torch.cat([torch.tensor([0]), torch.cumsum(torch.tensor(num_blocks), dim=0)])
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(num_blocks))]
        dpr = [dpr[my_iter[j]:my_iter[j+1]] for j in range(len(num_blocks))]
        dpr_encoder = dpr[:self.num_stages]
        dpr_bottleneck = dpr[-1]

        self.patch_size = patch_size
        self.encoder = Encoder(shortcut=shortcut, proj=proj, use_conv_mlp=use_conv_mlp, blur=blur, img_size=image_size, blur_kernel=blur_kernel, device=device, swin_abs_pos=swin_abs_pos, in_dims=in_dims, conv_depth=conv_depth, transformer_depth=transformer_depth, transformer_type=transformer_type, bottleneck_type=self.bottleneck_name, dpr=dpr_encoder, rpe_mode=rpe_mode, rpe_contextual_tensor=rpe_contextual_tensor, patch_size=patch_size, num_heads=num_heads, window_size=window_size, drop_path_rate=drop_path_rate)

        last_res = int(image_size / 2**(self.num_stages))                           
        in_dim_linear = int(last_res * last_res * self.d_model)
        self.linear_layers = nn.Sequential(nn.Linear(in_dim_linear, mlp_intermediary_dim), nn.GELU(), nn.Linear(mlp_intermediary_dim, 6))
        self.linear_layers[-1].weight.data.zero_()
        self.linear_layers[-1].bias.data.copy_(torch.tensor([0, 0, 0, 0, 0, 0], dtype=torch.float))

        self.bottleneck = swin_transformer_2.BasicLayer(dim=int(self.d_model),
                                        input_resolution=self.bottleneck_size,
                                        shortcut=shortcut,
                                        depth=self.num_bottleneck_layers,
                                        num_heads=bottleneck_heads,
                                        proj=proj,
                                        use_conv_mlp=use_conv_mlp,
                                        device=device,
                                        rpe_mode=rpe_mode,
                                        rpe_contextual_tensor=rpe_contextual_tensor,
                                        window_size=window_size,
                                        mlp_ratio=mlp_ratio,
                                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                                        drop=drop_rate, attn_drop=attn_drop_rate,
                                        drop_path=dpr_bottleneck,
                                        norm_layer=norm_layer,
                                        use_checkpoint=use_checkpoint)

    def forward(self, x):
        x_bottleneck, skip_connections = self.encoder(x)
        x_bottleneck = self.bottleneck(x_bottleneck)
        linear_layers_input = torch.flatten(x_bottleneck, start_dim=1)
        q_values = self.linear_layers(linear_layers_input)
        return q_values