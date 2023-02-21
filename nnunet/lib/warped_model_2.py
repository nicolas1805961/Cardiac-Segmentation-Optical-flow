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
from encoder import Encoder
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


class my_adversarial_model(nn.Module):
    def __init__(self, 
                blur,
                blur_kernel,
                shortcut,
                embed_dim, 
                patch_size, 
                window_size, 
                swin_abs_pos,
                deep_supervision,
                proj,
                binary,
                out_encoder_dims,
                use_conv_mlp,
                device,
                mlp_intermediary_dim,
                similarity_down_scale,
                concat_spatial_cross_attention,
                reconstruction_attention_type,
                encoder_attention_type,
                spatial_cross_attention_num_heads,
                merge,
                reconstruction,
                warped,
                second_conv_depth,
                second_transformer_depth,
                second_in_dims,
                second_num_heads,
                second_bottleneck_heads,
                batch_size,
                test,
                in_dims,
                image_size,
                swin_layer_type,
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
        self.learn_angle = True if binary else False
        self.warped = warped
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
        self.encoder = Encoder(out_dims=out_encoder_dims, shortcut=shortcut, proj=proj, use_conv_mlp=use_conv_mlp, blur=blur, img_size=image_size, blur_kernel=blur_kernel, device=device, swin_layer_type=swin_layer_type, swin_abs_pos=swin_abs_pos, in_dims=in_dims, conv_depth=conv_depth, transformer_depth=transformer_depth, transformer_type=transformer_type, bottleneck_type=self.bottleneck_name, dpr=dpr_encoder, rpe_mode=rpe_mode, rpe_contextual_tensor=rpe_contextual_tensor, patch_size=patch_size, num_heads=num_heads, window_size=window_size, drop_path_rate=drop_path_rate, deep_supervision=deep_supervision)
        if self.reconstruction and not test:
            sm_computation = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=4, kernel_size=1), 
                                                nn.BatchNorm2d(4), 
                                                nn.GELU(), 
                                                GetSimilarityMatrix(similarity_down_scale))

            self.reconstruction = decoder_alt.Decoder(reconstruction=True, sm_computation=sm_computation, concat_spatial_cross_attention=concat_spatial_cross_attention, attention_type=reconstruction_attention_type, spatial_cross_attention_num_heads=spatial_cross_attention_num_heads, shortcut=shortcut, proj_qkv=proj, out_encoder_dims=out_encoder_dims, use_conv_mlp=use_conv_mlp, last_activation='identity', blur=blur, img_size=image_size, num_classes=1, blur_kernel=blur_kernel, device=device, swin_abs_pos=swin_abs_pos, in_encoder_dims=in_dims[::-1], merge=merge, conv_depth=conv_depth, transformer_depth=transformer_depth, transformer_type=transformer_type, dpr=dpr_decoder, rpe_mode=rpe_mode, rpe_contextual_tensor=rpe_contextual_tensor, embed_dim=embed_dim, num_heads=num_heads, window_size=window_size, drop_path_rate=drop_path_rate, deep_supervision=deep_supervision)

        sm_computation = GetSimilarityMatrix(similarity_down_scale)
        if warped:
            self.fisrt_resolutions = [int(image_size / (2**i)) for i in range(self.num_stages)]
            self.first_encoder_fusion_layers = nn.ModuleList()
            self.decoder = decoder_alt.FirstDecoder(deep_supervision=deep_supervision, spatial_cross_attention_num_heads=spatial_cross_attention_num_heads, proj_qkv=proj, shortcut=shortcut, blur=blur, use_conv_mlp=use_conv_mlp, last_activation='softmax', blur_kernel=blur_kernel, device=device, dpr=dpr_decoder, in_encoder_dims=in_dims[::-1], out_encoder_dims=out_encoder_dims[::-1], num_classes=4, window_size=window_size, img_size=image_size, swin_abs_pos=swin_abs_pos, rpe_mode=rpe_mode, rpe_contextual_tensor=rpe_contextual_tensor, conv_depth=conv_depth, transformer_depth=transformer_depth, num_heads=num_heads)
            self.second_encoder = Encoder(out_dims=out_encoder_dims, shortcut=shortcut, proj=proj, use_conv_mlp=use_conv_mlp, blur=blur, img_size=image_size // 2, blur_kernel=blur_kernel, device=device, swin_layer_type=swin_layer_type, swin_abs_pos=swin_abs_pos, in_dims=second_in_dims, conv_depth=second_conv_depth, transformer_depth=second_transformer_depth, transformer_type=transformer_type, bottleneck_type=self.bottleneck_name, dpr=dpr_encoder, rpe_mode=rpe_mode, rpe_contextual_tensor=rpe_contextual_tensor, patch_size=patch_size, num_heads=second_num_heads, window_size=window_size, drop_path_rate=drop_path_rate, deep_supervision=deep_supervision)
            self.second_decoder = decoder_alt.SecondDecoder(mix=False, deep_supervision=deep_supervision, proj_qkv=proj, shortcut=shortcut, blur=blur, use_conv_mlp=use_conv_mlp, last_activation='softmax', blur_kernel=blur_kernel, device=device, dpr=dpr_decoder, in_encoder_dims=second_in_dims[::-1], out_encoder_dims=out_encoder_dims[::-1], num_classes=4, window_size=window_size, img_size=image_size // 2, swin_abs_pos=swin_abs_pos, rpe_mode=rpe_mode, rpe_contextual_tensor=rpe_contextual_tensor, conv_depth=second_conv_depth, transformer_depth=second_transformer_depth, num_heads=second_num_heads)
            
            for i_layer in range(self.num_stages):
                first_encoder_cca_layer = CCA(dim=out_encoder_dims[i_layer] * 2)
                first_encoder_fusion_layer = nn.Sequential(first_encoder_cca_layer, nn.Conv2d(in_channels=out_encoder_dims[i_layer] * 2, out_channels=out_encoder_dims[i_layer], kernel_size=1))
                self.first_encoder_fusion_layers.append(first_encoder_fusion_layer)
        else:
            self.decoder = decoder_alt.Decoder(reconstruction=reconstruction, sm_computation=sm_computation, concat_spatial_cross_attention=concat_spatial_cross_attention, attention_type=encoder_attention_type, spatial_cross_attention_num_heads=spatial_cross_attention_num_heads, shortcut=shortcut, proj_qkv=proj, out_encoder_dims=out_encoder_dims[::-1], use_conv_mlp=use_conv_mlp, last_activation='softmax', blur=blur, img_size=image_size, num_classes=4 if not binary else 2, blur_kernel=blur_kernel, device=device, swin_abs_pos=swin_abs_pos, in_encoder_dims=in_dims[::-1], merge=merge, conv_depth=conv_depth, transformer_depth=transformer_depth, transformer_type=transformer_type, dpr=dpr_decoder, rpe_mode=rpe_mode, rpe_contextual_tensor=rpe_contextual_tensor, embed_dim=embed_dim, num_heads=num_heads, window_size=window_size, drop_path_rate=drop_path_rate, deep_supervision=deep_supervision)
        last_res = int(image_size / 2**(self.num_stages))
        if self.learn_angle:                             
            in_dim_linear = int(last_res * last_res * self.d_model)
            self.rotation_net = nn.Sequential(nn.Linear(in_dim_linear, mlp_intermediary_dim), nn.GELU(), nn.Linear(mlp_intermediary_dim, 1))
            self.rotation_net[-1].weight.data.zero_()
            self.rotation_net[-1].bias.data.copy_(torch.tensor([0], dtype=torch.float))

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
        if warped:
            self.second_bottleneck = swin_transformer_2.BasicLayer(dim=int(self.d_model),
                                        input_resolution=[x // 2 for x in self.bottleneck_size],
                                        shortcut=shortcut,
                                        depth=self.num_bottleneck_layers,
                                        num_heads=second_bottleneck_heads,
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
            
            #self.fusion_layers = nn.ModuleList()
            #for i in range(self.num_stages):
            #    if i < self.num_stages - 1 and deep_supervision:
            #        fusion = ConvLayer(input_resolution=[image_size, image_size], 
            #                    in_dim=8,
            #                    out_dim=4,
            #                    nb_se_blocks=fusion_depth, 
            #                    dpr=[0.0, 0.0],
            #                    shortcut=shortcut)
            #        self.fusion_layers.append(fusion)
            #    elif i == self.num_stages - 1:
            #        fusion = ConvLayer(input_resolution=[image_size, image_size], 
            #                    in_dim=8,
            #                    out_dim=4,
            #                    nb_se_blocks=fusion_depth, 
            #                    dpr=[0.0, 0.0],
            #                    shortcut=shortcut)
            #        self.fusion_layers.append(fusion)
    
    def fuse(self, pred_list):
        out_list = []
        for pred, fusion_layer in zip(pred_list, self.fusion_layers):
            out = fusion_layer(pred)
            out = out.permute(0, 2, 1).view(self.batch_size, 4, self.image_size, self.image_size)
            out = F.softmax(out, dim=1)
            out_list.append(out)
        return out_list


    def forward(self, x, x2=None):
        reconstruction_sm = None
        reconstructed = None
        angle = None
        decoder_sm = None
        second_pred = None
        x_bottleneck, skip_connections = self.encoder(x)
        x_bottleneck = self.bottleneck(x_bottleneck)

        if x2 is not None:
            x2 = TF.center_crop(x2, self.image_size // 2).contiguous()

            second_x_bottleneck, second_skip_connections = self.second_encoder(x2)
            second_x_bottleneck = self.second_bottleneck(second_x_bottleneck)
            input_mix = [None] * self.num_stages
            second_pred, second_input_list = self.second_decoder(second_x_bottleneck, input_mix)

            encoder_skip_connections = self.mix(second_skip_connections, skip_connections, self.first_encoder_fusion_layers, self.fisrt_resolutions)
            pred = self.decoder(x_bottleneck, encoder_skip_connections, second_input_list)
            #pred = self.mix_ds(second_pred, pred)
            #pred = self.fuse(pred)
        else:
            pred, decoder_sm = self.decoder(x_bottleneck, skip_connections)
            if self.learn_angle:
                rotation_net_input = torch.flatten(x_bottleneck, start_dim=1)
                angle = self.rotation_net(rotation_net_input)
            if self.reconstruction and self.training:
                reconstructed, reconstruction_sm = self.reconstruction(x_bottleneck, skip_connections)
        out = {'pred': pred, 'reconstructed': reconstructed, 'decoder_sm': decoder_sm, 'reconstruction_sm': reconstruction_sm, 'angle': angle, 'second_pred': second_pred}
        return out

    def mix(self, small_skip_connections, big_skip_connections, encoder_fusion_layers, resolutions):
        mixed_skip_connections = []
        for small_skip_connection, big_skip_connection, encoder_fusion_layer, resolution in zip(small_skip_connections, big_skip_connections, encoder_fusion_layers, resolutions):
            B, L, C = small_skip_connection.shape
            small_skip_connection = small_skip_connection.permute(0, 2, 1).contiguous().view(B, C, resolution // 2, resolution // 2)
            big_skip_connection = big_skip_connection.permute(0, 2, 1).contiguous().view(B, C, resolution, resolution)
            pad_size = small_skip_connection.shape[-1] // 2
            pad = (pad_size, pad_size, pad_size, pad_size)

            #print(small_skip_connection[0, 0].mean())
            #print(big_skip_connection[0, 0].mean())
            #print('////////////////////')

            #fig, ax = plt.subplots(1, 2)
            #ax[0].imshow(small_skip_connection.detach().cpu()[0, 0], cmap='plasma', vmin=big_skip_connection.min(), vmax=big_skip_connection.max())
            #ax[1].imshow(big_skip_connection.detach().cpu()[0, 0, pad_size:3*pad_size, pad_size:3*pad_size], cmap='plasma')
            #plt.show()

            #small_skip_connection = rescale(rescaled=small_skip_connection, rescaler=big_skip_connection)
            small_skip_connection = torch.nn.functional.pad(small_skip_connection, pad, mode='constant', value=0.0)

            #new[:, :, :(pad_size)] = big_skip_connection[:, :, :(pad_size)]
            #new[:, :, (pad_size * 3):] = big_skip_connection[:, :, (pad_size * 3):]
            #new[:, :, :, :(pad_size)] = big_skip_connection[:, :, :, :(pad_size)]
            #new[:, :, :, (pad_size * 3):] = big_skip_connection[:, :, :, (pad_size * 3):]

            #print(new[0, 0].mean())
            #print(big_skip_connection[0, 0].mean())

            #fig, ax = plt.subplots(1, 2)
            #ax[0].imshow(new.detach().cpu()[0, 0, pad_size:3*pad_size, pad_size:3*pad_size], cmap='plasma', vmin=big_skip_connection.min(), vmax=big_skip_connection.max())
            #ax[1].imshow(big_skip_connection.detach().cpu()[0, 0, pad_size:3*pad_size, pad_size:3*pad_size], cmap='plasma')
            #plt.show()

            #print('******************************')

            encoder_skip_connection = torch.cat([big_skip_connection, small_skip_connection], dim=1)
            encoder_skip_connection = encoder_fusion_layer(encoder_skip_connection)
            #encoder_skip_connection = mix_norm_layer(encoder_skip_connection)
            encoder_skip_connection = encoder_skip_connection.permute(0, 2, 3, 1).view(B, resolution * resolution, C)
            mixed_skip_connections.append(encoder_skip_connection)

        return mixed_skip_connections


    def mix_ds(self, small_skip_connections, big_skip_connections):
        mixed_skip_connections = []
        for small_skip_connection, big_skip_connection in zip(small_skip_connections, big_skip_connections):
            pad_size = small_skip_connection.shape[-1] // 2
            pad = (pad_size, pad_size, pad_size, pad_size)
            new = torch.nn.functional.pad(small_skip_connection, pad, mode='constant', value=0.0)

            new[:(pad_size)] = big_skip_connection[:(pad_size)]
            new[(pad_size * 3):] = big_skip_connection[(pad_size * 3):]
            new[:, :(pad_size)] = big_skip_connection[:, :(pad_size)]
            new[:, (pad_size * 3):] = big_skip_connection[:, (pad_size * 3):]

            big_skip_connection = big_skip_connection.permute(0, 2, 3, 1).view(self.batch_size, -1, 4)
            new = new.permute(0, 2, 3, 1).view(self.batch_size, -1, 4)

            encoder_skip_connection = torch.cat([big_skip_connection, new], dim=-1)

            mixed_skip_connections.append(encoder_skip_connection)

        return mixed_skip_connections



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