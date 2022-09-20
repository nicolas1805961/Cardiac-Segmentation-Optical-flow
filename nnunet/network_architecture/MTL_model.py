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
from ..lib.vq_vae import VectorQuantizer, VectorQuantizerEMA, VanillaVAE, Quantize
from torch.cuda.amp import autocast
from nnunet.utilities.random_stuff import no_op
from typing import Union, Tuple
from ..lib.vit_transformer import TransformerEncoderLayer, TransformerEncoder

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
                middle,
                classification,
                log_function,
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
                adversarial_loss,
                simple_decoder,
                vae,
                vq_vae,
                affinity,
                asymmetric_unet,
                expansion_ratio,
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
        self.classification = classification
        self.binary = binary
        self.do_ds = deep_supervision
        self.conv_op=nn.Conv2d
        self.middle = middle
        self.vae = vae
        self.vq_vae = vq_vae
        self.asymmetric_unet = asymmetric_unet
        self.log_function = log_function
        self.similarity = similarity
        self.affinity = affinity
        self.add_extra_bottleneck_blocks = add_extra_bottleneck_blocks
        if uncertainty_weighting:
            self.logsigma = nn.Parameter(torch.FloatTensor([1.61, -0.7, -0.7]))
        else:
            self.logsigma = [None] * 3
        
        self.num_classes = 2 if binary else 4

        self.adversarial_loss = adversarial_loss

        self.expanded_dim = expansion_ratio * self.d_model

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
        
        if asymmetric_unet:
            conv_depth_decoder = [x//2 for x in conv_depth[::-1]]
        else:
            conv_depth_decoder = conv_depth[::-1]
        
        if self.affinity:
            self.SegAffinityComputer = GetSimilarityMatrix(similarity_down_scale)
            if simple_decoder:
                self.affinity_decoder = nn.Sequential(decoder_alt.AffinityDecoder2(dim=self.d_model),
                                                   GetSimilarityMatrix(similarity_down_scale))    
            else: 
                self.affinity_decoder = nn.Sequential(decoder_alt.AffinityDecoder(norm=norm, shortcut=shortcut, proj_qkv=proj, out_encoder_dims=out_encoder_dims[::-1], use_conv_mlp=use_conv_mlp, last_activation='identity', blur=blur, img_size=image_size, num_classes=4, blur_kernel=blur_kernel, device=device, swin_abs_pos=swin_abs_pos, in_encoder_dims=in_dims[::-1], merge=merge, conv_depth=conv_depth_decoder, transformer_depth=transformer_depth[::-1], dpr=dpr_decoder, rpe_mode=rpe_mode, rpe_contextual_tensor=rpe_contextual_tensor, num_heads=num_heads, window_size=window_size, drop_path_rate=drop_path_rate, deep_supervision=False),
                                                        GetSimilarityMatrix(similarity_down_scale))
        
        if self.reconstruction:
            sm_computation = nn.Sequential(ConvBlock(in_dim=1, out_dim=4, kernel_size=1, norm=norm),
                                                GetSimilarityMatrix(similarity_down_scale))

            if simple_decoder:
                self.reconstruction = decoder_alt.ReconstructionDecoder2(dim=self.d_model)
            else:
                self.reconstruction = decoder_alt.ReconstructionDecoder(similarity=similarity, norm=norm, filter_skip_co_reconstruction=filter_skip_co_reconstruction, reconstruction=reconstruction, reconstruction_skip=reconstruction_skip, sm_computation=sm_computation, concat_spatial_cross_attention=concat_spatial_cross_attention, attention_type=reconstruction_attention_type, spatial_cross_attention_num_heads=spatial_cross_attention_num_heads[::-1], shortcut=shortcut, proj_qkv=proj, out_encoder_dims=out_encoder_dims[::-1], use_conv_mlp=use_conv_mlp, last_activation='identity', blur=blur, img_size=image_size, num_classes=1, blur_kernel=blur_kernel, device=device, swin_abs_pos=swin_abs_pos, in_encoder_dims=in_dims[::-1], merge=merge, conv_depth=conv_depth_decoder, transformer_depth=transformer_depth[::-1], dpr=dpr_decoder, rpe_mode=rpe_mode, rpe_contextual_tensor=rpe_contextual_tensor, num_heads=num_heads, window_size=window_size, drop_path_rate=drop_path_rate, deep_supervision=self.do_ds)

        self.decoder = decoder_alt.SegmentationDecoder(similarity=similarity, norm=norm, similarity_down_scale=similarity_down_scale, filter_skip_co_segmentation=filter_skip_co_segmentation, shift_nb=shift_nb, start_reconstruction_dim=start_reconstruction_dim, directional_field=directional_field, attention_map=attention_map, reconstruction=reconstruction, reconstruction_skip=reconstruction_skip, concat_spatial_cross_attention=concat_spatial_cross_attention, attention_type=encoder_attention_type, spatial_cross_attention_num_heads=spatial_cross_attention_num_heads[::-1], shortcut=shortcut, proj_qkv=proj, out_encoder_dims=out_encoder_dims[::-1], use_conv_mlp=use_conv_mlp, last_activation='identity', blur=blur, img_size=image_size, num_classes=self.num_classes, blur_kernel=blur_kernel, device=device, swin_abs_pos=swin_abs_pos, in_encoder_dims=in_dims[::-1], merge=merge, conv_depth=conv_depth_decoder, transformer_depth=transformer_depth[::-1], dpr=dpr_decoder, rpe_mode=rpe_mode, rpe_contextual_tensor=rpe_contextual_tensor, num_heads=num_heads, window_size=window_size, drop_path_rate=drop_path_rate, deep_supervision=self.do_ds)
        if self.add_extra_bottleneck_blocks:
            self.extra_bottleneck_block_1 = ConvLayer(in_dim=self.d_model, out_dim=self.expanded_dim, nb_se_blocks=1, dpr=dpr_bottleneck, norm=norm)
            self.extra_bottleneck_block_2 = ConvLayer(in_dim=self.expanded_dim, out_dim=self.d_model, nb_se_blocks=1, dpr=dpr_bottleneck, norm=norm)
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
            #self.vq_vae_block = VectorQuantizer(num_embeddings=512, embedding_dim=vae_dim, beta=0.25)
            self.vq_vae_block = VectorQuantizerEMA(num_embeddings=512, embedding_dim=vae_dim, commitment_cost=0.25, decay=0.99)
            #self.vq_vae_block = Quantize(dim=64, n_embed=512, decay=0.99)

        H, W = (int(image_size / 2**(self.num_stages)), int(image_size / 2**(self.num_stages)))
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
        
        encoder_layer = TransformerEncoderLayer(d_model=int(self.expanded_dim), nhead=bottleneck_heads, dim_feedforward=2048)

        self.bottleneck = TransformerEncoder(encoder_layer=encoder_layer, num_layers=self.num_bottleneck_layers)

        #if swin_bottleneck:
        #    self.bottleneck = swin_transformer_2.BasicLayer(dim=int(self.d_model),
        #                                    norm=norm,
        #                                    attention_map=attention_map,
        #                                    input_resolution=self.bottleneck_size,
        #                                    shortcut=shortcut,
        #                                    depth=self.num_bottleneck_layers,
        #                                    num_heads=bottleneck_heads,
        #                                    proj=proj,
        #                                    use_conv_mlp=use_conv_mlp,
        #                                    device=device,
        #                                    rpe_mode=rpe_mode,
        #                                    rpe_contextual_tensor=rpe_contextual_tensor,
        #                                    window_size=H,
        #                                    mlp_ratio=mlp_ratio,
        #                                    qkv_bias=qkv_bias, qk_scale=qk_scale,
        #                                    drop=drop_rate, attn_drop=attn_drop_rate,
        #                                    drop_path=dpr_bottleneck,
        #                                    norm_layer=nn.LayerNorm,
        #                                    use_checkpoint=use_checkpoint)
        #else:
        #    #self.bottleneck = ConvBlocks(in_dim=int(self.d_model), out_dim=int(self.d_model), nb_block=2)
#
        #    self.bottleneck = ConvLayer(in_dim=int(self.d_model),
        #                        out_dim=int(self.d_model),
        #                        nb_se_blocks=self.num_bottleneck_layers, 
        #                        dpr=dpr_bottleneck)


    def forward(self, x, middle=None, attention_map=None):
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
        reconstruction_skip_connections = None
        x_encoded, encoder_skip_connections = self.encoder(x, attention_map=attention_map)
        if self.add_extra_bottleneck_blocks:
            x_encoded = self.extra_bottleneck_block_1(x_encoded)
        x_encoded = self.bottleneck(x_encoded)
        if self.add_extra_bottleneck_blocks:
            x_encoded = self.extra_bottleneck_block_2(x_encoded)
        if self.classification:
            #classification_out = self.reduce(x_encoded)
            classification_out = self.classification_conv(x_encoded)
            classification_out = torch.flatten(classification_out, start_dim=1)
            classification_out = self.classification_net(classification_out)
        if self.vae:
            x_bottleneck = self.pre_vae(x_encoded)
            x_bottleneck, vq_loss = self.vae_block(x_bottleneck)
            x_bottleneck = self.post_vae(x_bottleneck)
        elif self.vq_vae:
            x_bottleneck = self.pre_vae(x_encoded)
            #x_bottleneck, vq_loss, _ = self.vq_vae_block(x_bottleneck)
            vq_loss, x_bottleneck, perplexity, _ = self.vq_vae_block(x_bottleneck)
            x_bottleneck = self.post_vae(x_bottleneck)
        else:
            x_bottleneck = x_encoded
        if self.reconstruction:
            reconstructed, reconstruction_sm = self.reconstruction(x_bottleneck)
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

        pred, decoder_sm, df = self.decoder(x_encoded, encoder_skip_connections, attention_map=attention_map)

        if self.affinity:
            aff = self.affinity_decoder(x_encoded)
            seg_aff = self.SegAffinityComputer(pred[0])
        
        if not self.do_ds:
            pred = pred[0]
            reconstructed = reconstructed[0]
        out = {'pred': pred, 'reconstructed': reconstructed, 'decoder_sm': decoder_sm, 'reconstruction_sm': reconstruction_sm, 'parameters': parameters, 'logsigma': self.logsigma, 'directional_field': df, 'vq_loss': vq_loss, 'perplexity': perplexity, 'classification': classification_out, 'affinity': aff, 'seg_affinity': seg_aff}
        return out
    

    def predict_3D(self, x: np.ndarray, do_mirroring: bool, mirror_axes: Tuple[int, ...] = (0, 1, 2),
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
                        res = self._internal_predict_3D_2Dconv_tiled(x, patch_size, do_mirroring, mirror_axes, step_size,
                                                                     regions_class_order, use_gaussian, pad_border_mode,
                                                                     pad_kwargs, all_in_gpu, False)
                    else:
                        res = self._internal_predict_3D_2Dconv(x, patch_size, do_mirroring, mirror_axes, regions_class_order,
                                                               pad_border_mode, pad_kwargs, all_in_gpu, False)
                else:
                    raise RuntimeError("Invalid conv op, cannot determine what dimensionality (2d/3d) the network is")

        return res
    

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