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
from ..lib.encoder import Encoder
from ..lib.utils import ConvBlock, ConvBlocks, GetSimilarityMatrix, ReplicateChannels, To_image, From_image, ConvLayer, rescale, CCA
from ..lib import swin_transformer_2
from ..lib import decoder_alt
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from nnunet.network_architecture.neural_network import SegmentationNetwork

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


class MTLmodel(SegmentationNetwork):
    def __init__(self, 
                shift_nb,
                blur,
                binary,
                blur_kernel,
                attention_map,
                shortcut,
                patch_size, 
                window_size,
                swin_abs_pos,
                deep_supervision,
                proj,
                out_encoder_dims,
                use_conv_mlp,
                uncertainty_weighting,
                device,
                similarity_down_scale,
                concat_spatial_cross_attention,
                reconstruction_attention_type,
                encoder_attention_type,
                spatial_cross_attention_num_heads,
                merge,
                reconstruction,
                reconstruction_skip,
                mlp_intermediary_dim,
                learn_transforms,
                batch_size,
                in_dims,
                image_size,
                num_bottleneck_layers, 
                directional_field,
                conv_depth,
                swin_bottleneck,
                num_heads,
                transformer_depth, 
                filter_skip_co,
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
        super(MTLmodel, self).__init__()
        
        self.num_stages = (len(transformer_depth) + len(conv_depth))
        self.num_bottleneck_layers = num_bottleneck_layers
        self.bottleneck_name = bottleneck
        self.d_model = out_encoder_dims[-1] * 2
        self.bottleneck_size = [int(image_size / (2**self.num_stages)), int(image_size / (2**self.num_stages))]
        self.reconstruction = reconstruction
        self.image_size = image_size
        self.batch_size = batch_size
        self.learn_transforms = learn_transforms
        self.binary = binary
        self.do_ds = deep_supervision
        if uncertainty_weighting:
            self.logsigma = nn.Parameter(torch.FloatTensor([1.61, -0.7, -0.7]))
        else:
            self.logsigma = [None] * 3
        
        seg_out_dim = 2 if binary else 4

        # stochastic depth
        num_blocks = conv_depth + transformer_depth + [num_bottleneck_layers]
        my_iter = torch.cat([torch.tensor([0]), torch.cumsum(torch.tensor(num_blocks), dim=0)])
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(num_blocks))]
        dpr = [dpr[my_iter[j]:my_iter[j+1]] for j in range(len(num_blocks))]
        dpr_encoder = dpr[:self.num_stages]
        dpr_decoder = [x[::-1] for x in dpr_encoder[::-1]]
        dpr_bottleneck = dpr[-1]

        self.patch_size = patch_size
        self.encoder = Encoder(attention_map=attention_map, out_dims=out_encoder_dims, shortcut=shortcut, proj=proj, use_conv_mlp=use_conv_mlp, blur=blur, img_size=image_size, blur_kernel=blur_kernel, device=device, swin_abs_pos=swin_abs_pos, in_dims=in_dims, conv_depth=conv_depth, transformer_depth=transformer_depth, transformer_type=transformer_type, bottleneck_type=self.bottleneck_name, dpr=dpr_encoder, rpe_mode=rpe_mode, rpe_contextual_tensor=rpe_contextual_tensor, patch_size=patch_size, num_heads=num_heads, window_size=window_size, drop_path_rate=drop_path_rate, deep_supervision=deep_supervision)
        if self.reconstruction:
            #sm_computation = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=seg_out_dim, kernel_size=1), 
            #                                    nn.BatchNorm2d(seg_out_dim), 
            #                                    nn.GELU(), 
            #                                    GetSimilarityMatrix(similarity_down_scale))
            
            sm_computation = nn.Sequential(ReplicateChannels(4),
                                                GetSimilarityMatrix(similarity_down_scale))

            self.reconstruction = decoder_alt.ReconstructionDecoder(filter_skip_co=filter_skip_co, reconstruction=reconstruction, reconstruction_skip=reconstruction_skip, sm_computation=sm_computation, concat_spatial_cross_attention=concat_spatial_cross_attention, attention_type=reconstruction_attention_type, spatial_cross_attention_num_heads=spatial_cross_attention_num_heads[::-1], shortcut=shortcut, proj_qkv=proj, out_encoder_dims=out_encoder_dims[::-1], use_conv_mlp=use_conv_mlp, last_activation='identity', blur=blur, img_size=image_size, num_classes=1, blur_kernel=blur_kernel, device=device, swin_abs_pos=swin_abs_pos, in_encoder_dims=in_dims[::-1], merge=merge, conv_depth=conv_depth[::-1], transformer_depth=transformer_depth[::-1], dpr=dpr_decoder, rpe_mode=rpe_mode, rpe_contextual_tensor=rpe_contextual_tensor, num_heads=num_heads, window_size=window_size, drop_path_rate=drop_path_rate, deep_supervision=deep_supervision)

        #final_intermediary_dim = 64 if directional_field else 16 if reconstruction else 4
        final_intermediary_dim = 64 if directional_field or reconstruction else 4
        sm_computation = nn.Sequential(ConvBlock(in_dim=final_intermediary_dim, out_dim=4, kernel_size=1),
                                        GetSimilarityMatrix(similarity_down_scale))

        self.decoder = decoder_alt.SegmentationDecoder(filter_skip_co=filter_skip_co, shift_nb=shift_nb, final_intermediary_dim=final_intermediary_dim, directional_field=directional_field, attention_map=attention_map, reconstruction=reconstruction, reconstruction_skip=reconstruction_skip, sm_computation=sm_computation, concat_spatial_cross_attention=concat_spatial_cross_attention, attention_type=encoder_attention_type, spatial_cross_attention_num_heads=spatial_cross_attention_num_heads[::-1], shortcut=shortcut, proj_qkv=proj, out_encoder_dims=out_encoder_dims[::-1], use_conv_mlp=use_conv_mlp, last_activation='identity', blur=blur, img_size=image_size, num_classes=seg_out_dim, blur_kernel=blur_kernel, device=device, swin_abs_pos=swin_abs_pos, in_encoder_dims=in_dims[::-1], merge=merge, conv_depth=conv_depth[::-1], transformer_depth=transformer_depth[::-1], dpr=dpr_decoder, rpe_mode=rpe_mode, rpe_contextual_tensor=rpe_contextual_tensor, num_heads=num_heads, window_size=window_size, drop_path_rate=drop_path_rate, deep_supervision=deep_supervision)
        
        H, W = (int(image_size / 2**(self.num_stages)), int(image_size / 2**(self.num_stages)))
        if learn_transforms:  
            in_dim_linear = int(H * W * self.d_model)
            self.rotation_net = nn.Sequential(nn.Linear(in_dim_linear, mlp_intermediary_dim), nn.GELU(), nn.Linear(mlp_intermediary_dim, 2))
            self.rotation_net[-1].weight.data.zero_()
            self.rotation_net[-1].bias.data.copy_(torch.tensor([0, 1], dtype=torch.float))

        if swin_bottleneck:

            self.bottleneck = swin_transformer_2.BasicLayer(dim=int(self.d_model),
                                            attention_map=attention_map,
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
        else:
            #self.bottleneck = ConvBlocks(in_dim=int(self.d_model), out_dim=int(self.d_model), nb_block=2)

            self.bottleneck = ConvLayer(in_dim=int(self.d_model),
                                out_dim=int(self.d_model),
                                nb_se_blocks=self.num_bottleneck_layers, 
                                dpr=dpr_bottleneck)


    def forward(self, x, attention_map=None):
        reconstruction_sm = None
        reconstructed = None
        decoder_sm = None
        df = None
        parameters = None
        reconstruction_skip_connections = None
        x_bottleneck, encoder_skip_connections = self.encoder(x, attention_map=attention_map)
        x_bottleneck = self.bottleneck(x_bottleneck, attention_map=attention_map)
        if self.reconstruction:
            reconstructed, reconstruction_sm, reconstruction_skip_connections = self.reconstruction(x_bottleneck, encoder_skip_connections)
        pred, decoder_sm, df = self.decoder(x_bottleneck, encoder_skip_connections, reconstruction_skip_connections, attention_map=attention_map)
        if self.learn_transforms:
            rotation_net_input = torch.flatten(x_bottleneck, start_dim=1)
            parameters = self.rotation_net(rotation_net_input)
        out = {'pred': pred, 'reconstructed': reconstructed, 'decoder_sm': decoder_sm, 'reconstruction_sm': reconstruction_sm, 'parameters': parameters, 'logsigma': self.logsigma, 'directional_field': df}
        return out


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