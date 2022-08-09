# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import matplotlib
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
from nnunet.utilities.to_torch import to_cuda, maybe_to_torch
import numpy as np
from typing import Union
from ..lib import swin_cross_attention
from ..lib.vq_vae import VectorQuantizer, VectorQuantizerEMA, VanillaVAE

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
                start_reconstruction_dim,
                merge,
                reconstruction,
                reconstruction_skip,
                mlp_intermediary_dim,
                middle,
                learn_transforms,
                batch_size,
                in_dims,
                image_size,
                num_bottleneck_layers, 
                directional_field,
                conv_depth,
                swin_bottleneck,
                num_heads,
                vae_noise,
                transformer_depth, 
                filter_skip_co_reconstruction,
                filter_skip_co_segmentation,
                bottleneck_heads, 
                vae,
                vq_vae,
                similarity,
                norm,
                add_extra_bottleneck_blocks,
                transformer_type='swin',
                bottleneck='swin',
                mlp_ratio=4., 
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
        self.image_size = image_size
        self.batch_size = batch_size
        self.learn_transforms = learn_transforms
        self.binary = binary
        self.do_ds = deep_supervision
        self.conv_op=nn.Conv2d
        self.middle = middle
        self.vae = vae
        self.vq_vae = vq_vae
        self.similarity = similarity
        self.add_extra_bottleneck_blocks = add_extra_bottleneck_blocks
        if uncertainty_weighting:
            self.logsigma = nn.Parameter(torch.FloatTensor([1.61, -0.7, -0.7]))
        else:
            self.logsigma = [None] * 3
        
        self.num_classes = 2 if binary else 4

        # stochastic depth
        num_blocks = conv_depth + transformer_depth + [num_bottleneck_layers]
        my_iter = torch.cat([torch.tensor([0]), torch.cumsum(torch.tensor(num_blocks), dim=0)])
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(num_blocks))]
        dpr = [dpr[my_iter[j]:my_iter[j+1]] for j in range(len(num_blocks))]
        dpr_encoder = dpr[:self.num_stages]
        dpr_decoder = [x[::-1] for x in dpr_encoder[::-1]]
        dpr_bottleneck = dpr[-1]

        self.patch_size = patch_size
        self.encoder = Encoder(norm=norm, attention_map=attention_map, out_dims=out_encoder_dims, shortcut=shortcut, proj=proj, use_conv_mlp=use_conv_mlp, blur=blur, img_size=image_size, blur_kernel=blur_kernel, device=device, swin_abs_pos=swin_abs_pos, in_dims=in_dims, conv_depth=conv_depth, transformer_depth=transformer_depth, transformer_type=transformer_type, bottleneck_type=self.bottleneck_name, dpr=dpr_encoder, rpe_mode=rpe_mode, rpe_contextual_tensor=rpe_contextual_tensor, patch_size=patch_size, num_heads=num_heads, window_size=window_size, drop_path_rate=drop_path_rate, deep_supervision=self.do_ds)
        if self.reconstruction:
            #sm_computation = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=self.num_classes, kernel_size=1), 
            #                                    nn.BatchNorm2d(self.num_classes), 
            #                                    nn.GELU(), 
            #                                    GetSimilarityMatrix(similarity_down_scale))
            
            sm_computation = nn.Sequential(ReplicateChannels(4),
                                                GetSimilarityMatrix(similarity_down_scale))

            self.reconstruction = decoder_alt.ReconstructionDecoder(similarity=similarity, norm=norm, filter_skip_co_reconstruction=filter_skip_co_reconstruction, reconstruction=reconstruction, reconstruction_skip=reconstruction_skip, sm_computation=sm_computation, concat_spatial_cross_attention=concat_spatial_cross_attention, attention_type=reconstruction_attention_type, spatial_cross_attention_num_heads=spatial_cross_attention_num_heads[::-1], shortcut=shortcut, proj_qkv=proj, out_encoder_dims=out_encoder_dims[::-1], use_conv_mlp=use_conv_mlp, last_activation='identity', blur=blur, img_size=image_size, num_classes=1, blur_kernel=blur_kernel, device=device, swin_abs_pos=swin_abs_pos, in_encoder_dims=in_dims[::-1], merge=merge, conv_depth=conv_depth[::-1], transformer_depth=transformer_depth[::-1], dpr=dpr_decoder, rpe_mode=rpe_mode, rpe_contextual_tensor=rpe_contextual_tensor, num_heads=num_heads, window_size=window_size, drop_path_rate=drop_path_rate, deep_supervision=self.do_ds)

        self.decoder = decoder_alt.SegmentationDecoder(similarity=similarity, norm=norm, similarity_down_scale=similarity_down_scale, filter_skip_co_segmentation=filter_skip_co_segmentation, shift_nb=shift_nb, start_reconstruction_dim=start_reconstruction_dim, directional_field=directional_field, attention_map=attention_map, reconstruction=reconstruction, reconstruction_skip=reconstruction_skip, concat_spatial_cross_attention=concat_spatial_cross_attention, attention_type=encoder_attention_type, spatial_cross_attention_num_heads=spatial_cross_attention_num_heads[::-1], shortcut=shortcut, proj_qkv=proj, out_encoder_dims=out_encoder_dims[::-1], use_conv_mlp=use_conv_mlp, last_activation='identity', blur=blur, img_size=image_size, num_classes=self.num_classes, blur_kernel=blur_kernel, device=device, swin_abs_pos=swin_abs_pos, in_encoder_dims=in_dims[::-1], merge=merge, conv_depth=conv_depth[::-1], transformer_depth=transformer_depth[::-1], dpr=dpr_decoder, rpe_mode=rpe_mode, rpe_contextual_tensor=rpe_contextual_tensor, num_heads=num_heads, window_size=window_size, drop_path_rate=drop_path_rate, deep_supervision=self.do_ds)
        if self.add_extra_bottleneck_blocks:
            self.extra_bottleneck_block_1 = ConvLayer(in_dim=self.d_model, out_dim=self.d_model, nb_se_blocks=1, dpr=dpr_bottleneck, norm=norm)
            self.extra_bottleneck_block_2 = ConvLayer(in_dim=self.d_model, out_dim=self.d_model, nb_se_blocks=1, dpr=dpr_bottleneck, norm=norm)
        if self.vae:
            H, W = (int(image_size / 2**(self.num_stages)), int(image_size / 2**(self.num_stages)))
            vae_dim = self.d_model
            vae_dim = 16
            in_dim_linear = int(H * W * vae_dim)
            self.pre_vae = ConvBlock(in_dim=self.d_model, out_dim=vae_dim, kernel_size=1, norm=norm)
            self.post_vae = nn.Conv2d(in_channels=vae_dim, out_channels=self.d_model, kernel_size=1)
            self.vae_block = VanillaVAE(vae_noise=vae_noise, flatten_dim=in_dim_linear, latent_dim=128)
        elif self.vq_vae:
            vae_dim = 64
            self.pre_vae = nn.Conv2d(in_channels=self.d_model, out_channels=64, kernel_size=1)
            self.post_vae = nn.Conv2d(in_channels=64, out_channels=self.d_model, kernel_size=1)
            self.vq_vae_block = VectorQuantizer(num_embeddings=512, embedding_dim=vae_dim, beta=0.25)
            #self.vq_vae = VectorQuantizerEMA(num_embeddings=512, embedding_dim=vae_dim, commitment_cost=0.25, decay=0.99)

        H, W = (int(image_size / 2**(self.num_stages)), int(image_size / 2**(self.num_stages)))
        if learn_transforms:  
            in_dim_linear = int(H * W * self.d_model)
            self.rotation_net = nn.Sequential(nn.Linear(in_dim_linear, mlp_intermediary_dim), nn.GELU(), nn.Linear(mlp_intermediary_dim, 2))
            self.rotation_net[-1].weight.data.zero_()
            self.rotation_net[-1].bias.data.copy_(torch.tensor([0, 1], dtype=torch.float))

        if self.middle:
            self.middle_encoder = Encoder(norm=norm, attention_map=attention_map, out_dims=out_encoder_dims, shortcut=shortcut, proj=proj, use_conv_mlp=use_conv_mlp, blur=blur, img_size=image_size, blur_kernel=blur_kernel, device=device, swin_abs_pos=swin_abs_pos, in_dims=in_dims, conv_depth=conv_depth, transformer_depth=transformer_depth, transformer_type=transformer_type, bottleneck_type=self.bottleneck_name, dpr=dpr_encoder, rpe_mode=rpe_mode, rpe_contextual_tensor=rpe_contextual_tensor, patch_size=patch_size, num_heads=num_heads, window_size=window_size, drop_path_rate=drop_path_rate, deep_supervision=self.do_ds)
            self.reduce_layer = nn.Conv2d(in_channels=int(self.d_model) * 2, out_channels=int(self.d_model), kernel_size=1)
            self.big_attention = swin_transformer_2.BasicLayer(dim=int(self.d_model) * 2,
                                                                norm=norm,
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
                                                                norm_layer=nn.LayerNorm,
                                                                use_checkpoint=use_checkpoint)
            #self.ca_layer = swin_cross_attention.BasicLayer(swin_abs_pos=False, 
            #                                                norm=norm,
            #                                                same_key_query=False,
            #                                                dim=int(self.d_model),
            #                                                proj=proj,
            #                                                input_resolution=self.bottleneck_size,
            #                                                use_conv_mlp=use_conv_mlp,
            #                                                depth=self.num_bottleneck_layers,
            #                                                num_heads=bottleneck_heads,
            #                                                device=device,
            #                                                rpe_mode=rpe_mode,
            #                                                rpe_contextual_tensor=rpe_contextual_tensor,
            #                                                window_size=window_size,
            #                                                mlp_ratio=mlp_ratio,
            #                                                qkv_bias=qkv_bias, qk_scale=qk_scale,
            #                                                drop=drop_rate, attn_drop=attn_drop_rate,
            #                                                drop_path=dpr_bottleneck,
            #                                                norm_layer=nn.LayerNorm,
            #                                                use_checkpoint=use_checkpoint)

            #self.filter_layer = swin_cross_attention.SwinFilterBlock(in_dim=int(self.d_model),
            #                                                        out_dim=int(self.d_model),
            #                                                        input_resolution=self.bottleneck_size,
            #                                                        num_heads=bottleneck_heads,
            #                                                        norm=norm,
            #                                                        proj=proj,
            #                                                        device=device,
            #                                                        rpe_mode=rpe_mode,
            #                                                        rpe_contextual_tensor=rpe_contextual_tensor,
            #                                                        window_size=window_size,
            #                                                        depth=self.num_bottleneck_layers)

            #if swin_bottleneck:
            #    self.middle_bottleneck = swin_transformer_2.BasicLayer(dim=int(self.d_model),
            #                                                            norm=norm,
            #                                                            attention_map=attention_map,
            #                                                            input_resolution=self.bottleneck_size,
            #                                                            shortcut=shortcut,
            #                                                            depth=self.num_bottleneck_layers,
            #                                                            num_heads=bottleneck_heads,
            #                                                            proj=proj,
            #                                                            use_conv_mlp=use_conv_mlp,
            #                                                            device=device,
            #                                                            rpe_mode=rpe_mode,
            #                                                            rpe_contextual_tensor=rpe_contextual_tensor,
            #                                                            window_size=window_size,
            #                                                            mlp_ratio=mlp_ratio,
            #                                                            qkv_bias=qkv_bias, qk_scale=qk_scale,
            #                                                            drop=drop_rate, attn_drop=attn_drop_rate,
            #                                                            drop_path=dpr_bottleneck,
            #                                                            norm_layer=nn.LayerNorm,
            #                                                            use_checkpoint=use_checkpoint)
            #else:
            #    self.middle_bottleneck = ConvLayer(in_dim=int(self.d_model),
            #                                    out_dim=int(self.d_model),
            #                                    nb_se_blocks=self.num_bottleneck_layers, 
            #                                    dpr=dpr_bottleneck)

        if swin_bottleneck:
            self.bottleneck = swin_transformer_2.BasicLayer(dim=int(self.d_model),
                                            norm=norm,
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
                                            norm_layer=nn.LayerNorm,
                                            use_checkpoint=use_checkpoint)
        else:
            #self.bottleneck = ConvBlocks(in_dim=int(self.d_model), out_dim=int(self.d_model), nb_block=2)

            self.bottleneck = ConvLayer(in_dim=int(self.d_model),
                                out_dim=int(self.d_model),
                                nb_se_blocks=self.num_bottleneck_layers, 
                                dpr=dpr_bottleneck)


    def forward(self, x, middle=None, attention_map=None):
        reconstruction_sm = None
        reconstructed = None
        decoder_sm = None
        df = None
        parameters = None
        vq_loss = None
        reconstruction_skip_connections = None
        x_encoded, encoder_skip_connections = self.encoder(x, attention_map=attention_map)
        if self.add_extra_bottleneck_blocks:
            x_encoded = self.extra_bottleneck_block_1(x_encoded)
        x_encoded = self.bottleneck(x_encoded, attention_map=attention_map)
        if self.add_extra_bottleneck_blocks:
            x_encoded = self.extra_bottleneck_block_2(x_encoded)
        if self.vae:
            x_bottleneck = self.pre_vae(x_encoded)
            x_bottleneck, vq_loss = self.vae_block(x_bottleneck)
            x_bottleneck = self.post_vae(x_bottleneck)
        elif self.vq_vae:
            x_bottleneck = self.pre_vae(x_encoded)
            x_bottleneck, vq_loss = self.vq_vae_block(x_bottleneck)
            #vq_loss, x_bottleneck, perplexity, _ = self.vq_vae(x_bottleneck)
            x_bottleneck = self.post_vae(x_bottleneck)
        else:
            x_bottleneck = x_encoded
        if self.reconstruction:
            reconstructed, reconstruction_sm, reconstruction_skip_connections = self.reconstruction(x_bottleneck, encoder_skip_connections)
        if self.middle:
            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(1, 4)
            #ax[0].imshow((x - middle).cpu()[0, 0])
            #ax[1].imshow(torch.abs((x - middle).cpu()[0, 0]))
            #ax[2].imshow(x.cpu()[0, 0], cmap='gray')
            #ax[3].imshow(middle.cpu()[0, 0], cmap='gray')
            #plt.show()

            middle, _ = self.middle_encoder(middle, attention_map=attention_map)
            #middle = self.middle_bottleneck(middle, attention_map=attention_map)
            x_bottleneck = torch.cat([x_encoded, middle], dim=1)
            x_bottleneck = self.big_attention(x_bottleneck, attention_map=attention_map)
            #filtered = self.filter_layer(x_bottleneck + middle, x_bottleneck)
            #attention = self.ca_layer(x_bottleneck, middle)
            #x_bottleneck = torch.cat([filtered, x_bottleneck], dim=1)
            #x_bottleneck = torch.cat([attention, x_bottleneck], dim=1)
            x_bottleneck = self.reduce_layer(x_bottleneck)

        pred, decoder_sm, df = self.decoder(x_encoded, encoder_skip_connections, reconstruction_skip_connections, attention_map=attention_map)
        
        if not self.do_ds:
            pred = pred[0]
            reconstructed = reconstructed[0]
        out = {'pred': pred, 'reconstructed': reconstructed, 'decoder_sm': decoder_sm, 'reconstruction_sm': reconstruction_sm, 'parameters': parameters, 'logsigma': self.logsigma, 'directional_field': df, 'vq_loss': vq_loss}
        return out
    

    def _internal_maybe_mirror_and_pred_2D(self, x: Union[np.ndarray, torch.tensor], mirror_axes: tuple,
                                           do_mirroring: bool = True,
                                           mult: np.ndarray or torch.tensor = None) -> torch.tensor:
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

        if mult is not None:
            result_torch[:, :] *= mult

        return result_torch


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