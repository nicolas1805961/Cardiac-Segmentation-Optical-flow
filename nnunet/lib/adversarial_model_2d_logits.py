# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import torch
import sys
import torch.nn as nn
import matplotlib.pyplot as plt
from encoder import FusionEncoder
from utils import GetSimilarityMatrix, To_image, From_image, ConvLayer, rescale, CCA
import swin_transformer_2
import decoder_alt
import torchvision.transforms.functional as TF
import torch.nn.functional as F

class WholeModel(nn.Module):
    def __init__(self, model1, model2=None):
        super().__init__()
        self.model1 = model1
        self.model2 = model2

    def forward(self, x, model_nb=1):
        if not self.training:
            return self.model1(x)

        if model_nb == 1:
            return self.model1(x)
        elif model_nb == 2:
            return self.model2(x)


class my_model(nn.Module):
    def __init__(self, 
                blur,
                blur_kernel,
                shortcut,
                patch_size, 
                window_size, 
                swin_abs_pos,
                deep_supervision,
                proj,
                binary,
                out_encoder_dims,
                use_conv_mlp,
                device,
                concat_spatial_cross_attention,
                encoder_attention_type,
                spatial_cross_attention_num_heads,
                merge,
                reconstruction,
                batch_size,
                in_dims,
                image_size,
                num_bottleneck_layers, 
                conv_depth,
                num_heads,
                transformer_depth, 
                bottleneck_heads, 
                transformer_type='swin',
                bottleneck='swin',
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
        self.bottleneck_size = [int(image_size / (2**self.num_stages)), int(image_size / (2**self.num_stages))]
        self.reconstruction = reconstruction
        self.binary = binary
        self.image_size = image_size
        self.batch_size = batch_size

        # stochastic depth
        num_blocks = conv_depth + transformer_depth + [num_bottleneck_layers]
        my_iter = torch.cat([torch.tensor([0]), torch.cumsum(torch.tensor(num_blocks), dim=0)])
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(num_blocks))]
        dpr = [dpr[my_iter[j]:my_iter[j+1]] for j in range(len(num_blocks))]
        dpr_encoder = dpr[:self.num_stages]
        dpr_decoder = [x[::-1] for x in dpr_encoder[::-1]]
        dpr_bottleneck = dpr[-1]

        self.patch_size = patch_size
        self.encoder = FusionEncoder(out_dims=out_encoder_dims, shortcut=shortcut, proj=proj, use_conv_mlp=use_conv_mlp, blur=blur, img_size=image_size, blur_kernel=blur_kernel, device=device, swin_abs_pos=swin_abs_pos, in_dims=in_dims, conv_depth=conv_depth, transformer_depth=transformer_depth, transformer_type=transformer_type, bottleneck_type=self.bottleneck_name, dpr=dpr_encoder, rpe_mode=rpe_mode, rpe_contextual_tensor=rpe_contextual_tensor, patch_size=patch_size, num_heads=num_heads, window_size=window_size, drop_path_rate=drop_path_rate, deep_supervision=deep_supervision)
        self.decoder = decoder_alt.FusionDecoder(reconstruction=reconstruction, concat_spatial_cross_attention=concat_spatial_cross_attention, attention_type=encoder_attention_type, spatial_cross_attention_num_heads=spatial_cross_attention_num_heads, shortcut=shortcut, proj_qkv=proj, out_encoder_dims=out_encoder_dims[::-1], use_conv_mlp=use_conv_mlp, last_activation='identity', blur=blur, img_size=image_size, num_classes=4 if not binary else 2, blur_kernel=blur_kernel, device=device, swin_abs_pos=swin_abs_pos, in_encoder_dims=in_dims[::-1], merge=merge, conv_depth=conv_depth, transformer_depth=transformer_depth, dpr=dpr_decoder, rpe_mode=rpe_mode, rpe_contextual_tensor=rpe_contextual_tensor, num_heads=num_heads, window_size=window_size, drop_path_rate=drop_path_rate, deep_supervision=deep_supervision)

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
        
        self.cca_layer = nn.Sequential(CCA(dim=int(self.d_model) * 2, input_resolution=self.bottleneck_size))


    def forward(self, x1, x2):
        x1, x2, skip_connection_list = self.encoder(x1, x2)
        x = torch.cat([x1, x2], dim=-1)
        x = self.cca_layer(x)
        x = self.bottleneck(x)
        pred = self.decoder(x, skip_connection_list)
        out = {'pred': pred}
        return out