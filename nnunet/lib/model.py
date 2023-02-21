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
from bottlenecks import temporalTransformer
from vit_transformer import VitBasicLayer, VitChannelLayer
from encoder import Encoder, CCSSS_3d, CCVVV, CCCVV, EncoderNoConv, ConvEncoder
from utils import MLP, ConvLayer, ResnetConvLayer
import swin_transformer_2
import swin_transformer_3d
import decoder_alt
import generator

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
                num_frames, 
                embed_dim, 
                similarity_down_scale,
                reconstruction,
                batch_size, 
                patch_size, 
                window_size, 
                swin_abs_pos,
                encoder_attention_type,
                reconstruction_attention_type,
                concat_spatial_cross_attention,
                deep_supervision,
                convolutional_patch_embedding,
                proj,
                spatial_cross_attention_num_heads,
                out_encoder_dims,
                binary,
                merge,
                use_conv_mlp,
                device,
                in_dims,
                image_size,
                swin_layer_type,
                channel_attention,
                transformer_type='swin',
                bottleneck='swin',
                conv_depth=[2, 2],
                transformer_depth=[2, 2, 2], 
                num_heads=[3, 6, 12],
                bottleneck_heads=8, 
                num_bottleneck_layers=3, 
                num_memory_bus=8, 
                mlp_ratio=4., 
                qkv_bias=True, 
                qk_scale=None,
                drop_rate=0., 
                attn_drop_rate=0., 
                drop_path_rate=0.1, 
                norm_layer=nn.LayerNorm, 
                use_checkpoint=False, 
                dim_feedforward=3072, 
                dropout=0.0, 
                activation="gelu", 
                normalize_before=False, 
                return_intermediate_dec=False, 
                rpe_mode=None, 
                rpe_contextual_tensor=None):
        super().__init__()
        
        self.num_stages = (len(transformer_depth) + len(conv_depth))
        self.num_bottleneck_layers = num_bottleneck_layers
        self.bottleneck_name = bottleneck
        self.d_model = out_encoder_dims[-1] * 2
        self.bottleneck_size = [int(image_size / (x * 2**(self.num_stages - 2))) for x in patch_size]
        self.reconstruction_attention_type = reconstruction_attention_type
        self.reconstruction = reconstruction

        if channel_attention:
            self.channel_attention = VitChannelLayer(nhead=4, device=device, img_size=image_size, input_encoder_dims=in_dims, nb_blocks=4, batch_size=batch_size, dropout=0)
        else:
            self.channel_attention = None

        # stochastic depth
        num_blocks = conv_depth + transformer_depth + [num_bottleneck_layers]
        my_iter = torch.cat([torch.tensor([0]), torch.cumsum(torch.tensor(num_blocks), dim=0)])
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(num_blocks))]
        dpr = [dpr[my_iter[j]:my_iter[j+1]] for j in range(len(num_blocks))]
        dpr_encoder = dpr[:self.num_stages]
        dpr_decoder = [x[::-1] for x in dpr_encoder[::-1]]
        dpr_bottleneck = dpr[-1]
        
        if reconstruction:
            self.reconstruction_branch = decoder_alt.Decoder(reconstruction=reconstruction, similarity_down_scale=similarity_down_scale, concat_spatial_cross_attention=concat_spatial_cross_attention, attention_type=reconstruction_attention_type, spatial_cross_attention_num_heads=spatial_cross_attention_num_heads, shortcut=shortcut, proj_qkv=proj, out_encoder_dims=out_encoder_dims, use_conv_mlp=use_conv_mlp, last_activation='identity', blur=blur, img_size=image_size, num_classes=1, blur_kernel=blur_kernel, device=device, swin_abs_pos=swin_abs_pos, in_encoder_dims=in_dims[::-1], merge=merge, conv_depth=conv_depth, transformer_depth=transformer_depth, transformer_type=transformer_type, dpr=dpr_decoder, rpe_mode=rpe_mode, rpe_contextual_tensor=rpe_contextual_tensor, embed_dim=embed_dim, num_heads=num_heads, window_size=window_size, drop_path_rate=drop_path_rate, deep_supervision=deep_supervision)
            #self.vq_layer = VectorQuantizerEMA(bottleneck_resolution=self.bottleneck_size, num_embeddings=512, embedding_dim=self.d_model, commitment_cost=0.25, decay=0.99)
            #self.reconstruction_branch = ReconstructionBranch(nb_classes=1, shortcut=shortcut, proj=proj, use_conv_mlp=use_conv_mlp, blur=blur, img_size=image_size, blur_kernel=blur_kernel, device=device, swin_layer_type=swin_layer_type, swin_abs_pos=swin_abs_pos, in_encoder_dims=in_dims[::-1], conv_depth=conv_depth, transformer_depth=transformer_depth, transformer_type=transformer_type, bottleneck_type=self.bottleneck_name, dpr=dpr_decoder, rpe_mode=rpe_mode, rpe_contextual_tensor=rpe_contextual_tensor, patch_size=patch_size, embed_dim=embed_dim, num_heads=num_heads, window_size=window_size, drop_path_rate=drop_path_rate, deep_supervision=deep_supervision)

        self.patch_size = patch_size
        if len(patch_size) == 2 and '3d' in self.bottleneck_name:
            self.bottleneck_size = [num_frames] + self.bottleneck_size
        if len(patch_size) == 2:
            bottleneck_num_patches = self.bottleneck_size[0] * self.bottleneck_size[1]  
            if convolutional_patch_embedding:
                self.encoder = Encoder(shortcut=shortcut, proj=proj, use_conv_mlp=use_conv_mlp, blur=blur, img_size=image_size, blur_kernel=blur_kernel, device=device, swin_layer_type=swin_layer_type, swin_abs_pos=swin_abs_pos, in_dims=in_dims, conv_depth=conv_depth, transformer_depth=transformer_depth, transformer_type=transformer_type, bottleneck_type=self.bottleneck_name, dpr=dpr_encoder, rpe_mode=rpe_mode, rpe_contextual_tensor=rpe_contextual_tensor, patch_size=patch_size, embed_dim=embed_dim, num_heads=num_heads, window_size=window_size, drop_path_rate=drop_path_rate, deep_supervision=deep_supervision)
                self.decoder = decoder_alt.Decoder(reconstruction=reconstruction, similarity_down_scale=similarity_down_scale, concat_spatial_cross_attention=concat_spatial_cross_attention, attention_type=encoder_attention_type, spatial_cross_attention_num_heads=spatial_cross_attention_num_heads, shortcut=shortcut, proj_qkv=proj, out_encoder_dims=out_encoder_dims, use_conv_mlp=use_conv_mlp, last_activation='softmax', blur=blur, img_size=image_size, num_classes=2 if binary else 4, blur_kernel=blur_kernel, device=device, swin_abs_pos=swin_abs_pos, in_encoder_dims=in_dims[::-1], merge=merge, conv_depth=conv_depth, transformer_depth=transformer_depth, transformer_type=transformer_type, dpr=dpr_decoder, rpe_mode=rpe_mode, rpe_contextual_tensor=rpe_contextual_tensor, embed_dim=embed_dim, num_heads=num_heads, window_size=window_size, drop_path_rate=drop_path_rate, deep_supervision=deep_supervision)
                #if channel_cross_attention:
                #    self.decoder = decoder_alt.DecoderChannelAttention(attention_type=channel_attention_type, shortcut=shortcut, proj_qkv=proj, cross_attention_num_heads=channel_cross_attention_num_heads, out_encoder_dims=out_encoder_dims, use_conv_mlp=use_conv_mlp, last_activation='softmax', blur=blur, img_size=image_size, num_classes=2 if binary else 4, blur_kernel=blur_kernel, device=device, swin_layer_type=swin_layer_type, swin_abs_pos=swin_abs_pos, in_encoder_dims=in_dims[::-1], merge=merge, conv_depth=conv_depth, transformer_depth=transformer_depth, transformer_type=transformer_type, bottleneck_type=self.bottleneck_name, dpr=dpr_decoder, rpe_mode=rpe_mode, rpe_contextual_tensor=rpe_contextual_tensor, patch_size=patch_size, embed_dim=embed_dim, num_heads=num_heads, window_size=window_size, drop_path_rate=drop_path_rate, deep_supervision=deep_supervision)
                #elif spatial_cross_attention:
                #    self.decoder = decoder_alt.DecoderSpatialAttention(attention_type=spatial_attention_type, shortcut=shortcut, proj_qkv=proj, cross_attention_num_heads=spatial_cross_attention_num_heads, out_encoder_dims=out_encoder_dims, use_conv_mlp=use_conv_mlp, last_activation='softmax', blur=blur, img_size=image_size, num_classes=2 if binary else 4, blur_kernel=blur_kernel, device=device, swin_layer_type=swin_layer_type, swin_abs_pos=swin_abs_pos, in_encoder_dims=in_dims[::-1], merge=merge, conv_depth=conv_depth, transformer_depth=transformer_depth, transformer_type=transformer_type, bottleneck_type=self.bottleneck_name, dpr=dpr_decoder, rpe_mode=rpe_mode, rpe_contextual_tensor=rpe_contextual_tensor, patch_size=patch_size, embed_dim=embed_dim, num_heads=num_heads, window_size=window_size, drop_path_rate=drop_path_rate, deep_supervision=deep_supervision)
                #else:
                #    self.decoder = Decoder(shortcut=shortcut, proj=proj, use_conv_mlp=use_conv_mlp, last_activation='softmax', blur=blur, img_size=image_size, num_classes=2 if binary else 4, blur_kernel=blur_kernel, device=device, swin_layer_type=swin_layer_type, swin_abs_pos=swin_abs_pos, in_encoder_dims=in_dims[::-1], merge=merge, conv_depth=conv_depth, transformer_depth=transformer_depth, transformer_type=transformer_type, bottleneck_type=self.bottleneck_name, dpr=dpr_decoder, rpe_mode=rpe_mode, rpe_contextual_tensor=rpe_contextual_tensor, patch_size=patch_size, embed_dim=embed_dim, num_heads=num_heads, window_size=window_size, drop_path_rate=drop_path_rate, deep_supervision=deep_supervision)
            else:
                self.encoder = EncoderNoConv(proj=proj, blur=blur, img_size=image_size, blur_kernel=blur_kernel, device=device, swin_layer_type=swin_layer_type, swin_abs_pos=swin_abs_pos, in_dims=in_dims, conv_depth=conv_depth, transformer_depth=transformer_depth, transformer_type=transformer_type, bottleneck_type=self.bottleneck_name, dpr=dpr_encoder, rpe_mode=rpe_mode, rpe_contextual_tensor=rpe_contextual_tensor, patch_size=patch_size, embed_dim=embed_dim, num_heads=num_heads, window_size=window_size, drop_path_rate=drop_path_rate, deep_supervision=deep_supervision)
                self.decoder = DecoderNoConv(proj=proj, last_activation='softmax', blur=blur, img_size=image_size, num_classes=2 if binary else 4, blur_kernel=blur_kernel, device=device, swin_layer_type=swin_layer_type, swin_abs_pos=swin_abs_pos, in_encoder_dims=in_dims[::-1], merge=merge, conv_depth=conv_depth, transformer_depth=transformer_depth, transformer_type=transformer_type, bottleneck_type=self.bottleneck_name, dpr=dpr_decoder, rpe_mode=rpe_mode, rpe_contextual_tensor=rpe_contextual_tensor, patch_size=patch_size, embed_dim=embed_dim, num_heads=num_heads, window_size=window_size, drop_path_rate=drop_path_rate, deep_supervision=deep_supervision)

            if self.bottleneck_name == 'factorized':
                self.bottleneck = temporalTransformer(batch_size=batch_size,
                                                        device=device,
                                                        dpr=dpr_bottleneck,
                                                        num_frames=num_frames,
                                                        proj='linear',
                                                        d_model=self.d_model, 
                                                        nhead=bottleneck_heads, 
                                                        num_encoder_layers=num_bottleneck_layers, 
                                                        bottleneck_size=self.bottleneck_size, 
                                                        num_memory_bus=num_memory_bus, 
                                                        dim_feedforward=dim_feedforward, 
                                                        dropout=dropout,
                                                        activation=activation, 
                                                        normalize_before=normalize_before, 
                                                        return_intermediate_dec=return_intermediate_dec, 
                                                        rpe_mode=rpe_mode, 
                                                        rpe_contextual_tensor=rpe_contextual_tensor)
                self.mlp = MLP(num_memory_bus + bottleneck_num_patches, bottleneck_num_patches, 2)
            elif self.bottleneck_name == 'swin':
                #self.bottleneck = swin_transformer.BasicLayer(dim=int(self.d_model),
                #               input_resolution=self.bottleneck_size,
                #               depth=self.num_bottleneck_layers,
                #               num_heads=bottleneck_heads,
                #               window_size=window_size,
                #               mlp_ratio=mlp_ratio,
                #               qkv_bias=qkv_bias, qk_scale=qk_scale,
                #               drop=drop_rate, attn_drop=attn_drop_rate,
                #               drop_path=dpr_bottleneck,
                #               norm_layer=norm_layer,
                #               downsample=None,
                #               use_checkpoint=use_checkpoint)
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
            elif self.bottleneck_name == 'vit':
                #if irpe:
                #    rpe_config = get_rpe_config(ratio=1.9,
                #                        method="product",
                #                        mode='ctx',
                #                        shared_head=False,
                #                        skip=0,
                #                        rpe_on='qkv')
                #    bottleneck_layer_type = SpatialTransformerEncoderLayer(self.d_model, bottleneck_heads, 
                #                                                    self.bottleneck_size, rpe_config, 
                #                                                    dim_feedforward, dropout, activation, normalize_before)
                #else:
                #bottleneck_layer_type = TransformerEncoderLayer(d_model=self.d_model, 
                #                                                nhead=bottleneck_heads, 
                #                                                input_resolution=self.bottleneck_size,
                #                                                proj=proj,
                #                                                device=device,
                #                                                num_memory_token=0,
                #                                                rpe_mode=rpe_mode, 
                #                                                rpe_contextual_tensor=rpe_contextual_tensor, 
                #                                                dim_feedforward=dim_feedforward, 
                #                                                dropout=dropout, 
                #                                                activation=activation, 
                #                                                normalize_before=normalize_before)
                #self.bottleneck_layers = nn.ModuleList([copy.deepcopy(bottleneck_layer_type) for i in range(self.num_bottleneck_layers)])
                self.bottleneck = VitBasicLayer(in_dim=self.d_model, 
                                        nhead=bottleneck_heads, 
                                        rpe_mode=rpe_mode, 
                                        rpe_contextual_tensor=rpe_contextual_tensor, 
                                        input_resolution=self.bottleneck_size, 
                                        dropout=drop_rate,
                                        device=device, 
                                        nb_blocks=self.num_bottleneck_layers,
                                        dpr=dpr_bottleneck)
        else:
            self.encoder = CCSSS_3d(patch_size=patch_size, window_size=window_size)
            self.decoder = SSSCC_3d(patch_size=patch_size, window_size=window_size, deep_supervision=deep_supervision)
            
        if self.bottleneck_name == 'vit_3d':
            #self.position_encoding3d = PositionEmbeddingSine3d(N_steps, normalize=True)
            #pos = self.position_encoding3d(shape_util=(batch_size, self.bottleneck_size[0], self.bottleneck_size[1], self.bottleneck_size[2])).contiguous()
            #self.pos = pos.view(batch_size, self.d_model, -1).permute(2, 0, 1)
            #if irpe:
            #    rpe_config = get_rpe_config(ratio=1.9,
            #                        method="product",
            #                        mode='ctx',
            #                        shared_head=False,
            #                        skip=0,
            #                        rpe_on='qkv')
            #    bottleneck_layer_type = SpatialTransformerEncoderLayer(self.d_model, bottleneck_heads, 
            #                                                    (self.bottleneck_size, self.bottleneck_size), rpe_config, 
            #                                                    dim_feedforward, dropout, activation, normalize_before)
            #else:
            #bottleneck_layer_type = TransformerEncoderLayer(d_model=self.d_model, 
            #                                                nhead=bottleneck_heads, 
            #                                                input_resolution=self.bottleneck_size,
            #                                                proj=proj,
            #                                                num_memory_token=0,
            #                                                rpe_mode=rpe_mode, 
            #                                                rpe_contextual_tensor=rpe_contextual_tensor, 
            #                                                dim_feedforward=dim_feedforward, 
            #                                                dropout=dropout, 
            #                                                activation=activation, 
            #                                                normalize_before=normalize_before)
            #self.bottleneck_layers = nn.ModuleList([copy.deepcopy(bottleneck_layer_type) for i in range(self.num_bottleneck_layers)])
            self.bottleneck = VitBasicLayer(in_dim=self.d_model,
                                                    nhead=bottleneck_heads, 
                                                    rpe_mode=rpe_mode, 
                                                    rpe_contextual_tensor=rpe_contextual_tensor, 
                                                    input_resolution=self.bottleneck_size, 
                                                    device=device, 
                                                    nb_blocks=self.num_bottleneck_layers,
                                                    dpr=dpr_bottleneck)
        elif self.bottleneck_name == 'swin_3d':
            window_size = [num_frames, window_size, window_size] if isinstance(window_size, int) else window_size
            self.bottleneck = swin_transformer_3d.BasicLayer(dim=int(embed_dim * 8),
                                                            depth=self.num_bottleneck_layers,
                                                            num_heads=bottleneck_heads,
                                                            rpe_mode=rpe_mode,
                                                            device=device,
                                                            rpe_contextual_tensor=rpe_contextual_tensor,
                                                            batch_size=batch_size,
                                                            input_resolution=self.bottleneck_size,
                                                            window_size=window_size,
                                                            mlp_ratio=mlp_ratio,
                                                            qkv_bias=qkv_bias,
                                                            qk_scale=qk_scale,
                                                            drop=drop_rate,
                                                            attn_drop=attn_drop_rate,
                                                            drop_path=dpr_bottleneck,
                                                            norm_layer=norm_layer,
                                                            downsample=None,
                                                            use_checkpoint=use_checkpoint)
    
    def forward_single_frame(self, x):
        reconstructed = None
        vq_loss = None
        reconstruction_sm = None
        decoder_sm = None
        x, skip_connections = self.encoder(x)
        if self.channel_attention is not None:
            skip_connections = self.channel_attention(skip_connections)
        x = self.bottleneck(x)
        if self.reconstruction:
            #quantized, vq_loss = self.vq_layer(x)
            reconstructed, reconstruction_sm = self.reconstruction_branch(x, skip_connections)
        pred, decoder_sm = self.decoder(x, skip_connections)
        out = {'pred': pred, 'vq_loss': vq_loss, 'reconstructed': reconstructed, 'decoder_sm': decoder_sm, 'reconstruction_sm': reconstruction_sm}
        return out
    
    def forward_factorized(self, x):
        T, B, C, H, W = x.shape
        preds = []
        token_list = []
        frames_skip_connection = []

        if self.bottleneck_name == 'factorized':
            for frame in x:
                token, skip_connections = self.encoder(frame)
                token = token.permute((0, 2, 1)).view(B, self.d_model, self.bottleneck_size[0], self.bottleneck_size[1])
                token_list.append(token)
                frames_skip_connection.append(skip_connections)

            frames = torch.stack(token_list).view(-1, self.d_model, self.bottleneck_size[0], self.bottleneck_size[1]) # will be reshaped properly inside the bottleneck
            tokens, memory_busses = self.bottleneck(src=frames, is_train=True)
            tokens = tokens.permute((1, 0, 2)).view(T, B, -1, self.d_model)
            memory_busses = memory_busses.permute((1, 0, 2)).view(T, B, -1, self.d_model)

            for token, memory_bus, skip_connections in zip(tokens, memory_busses, frames_skip_connection):
                input_token = torch.cat([token, memory_bus], dim=1)
                input_token = self.mlp(input_token.permute((0, 2, 1))).permute((0, 2, 1))
                pred = self.decoder(input_token, skip_connections)
                preds.append(pred)
        return preds

    def forward_vit_3d(self, x):
        T, B, C, H, W = x.shape
        preds = []
        token_list = []
        frames_skip_connection = []
        if len(self.patch_size) == 2:
            for frame in x:
                token, skip_connections = self.encoder(frame)
                #token = token.permute((0, 2, 1)).view(B, self.d_model, self.bottleneck_size[1], self.bottleneck_size[2])
                token_list.append(token)
                frames_skip_connection.append(skip_connections)
            
            B, L, C = token_list[0].shape
            frames = torch.cat(token_list, dim=1) # bottleneck needs tensor in B L C format
            frames = self.bottleneck(frames)
            frames = torch.split(frames, L, dim=1)

            for token, skip_connections in zip(frames, frames_skip_connection):
                pred = self.decoder(token, skip_connections)
                preds.append(pred)
        return preds
    
    def forward_swin_3d(self, x):
        T, B, C, H, W = x.shape
        preds = []
        token_list = []
        frames_skip_connection = []
        if len(self.patch_size) == 2:
            for frame in x:
                token, skip_connections = self.encoder(frame)
                token = token.permute((0, 2, 1)).view(B, self.d_model, self.bottleneck_size[1], self.bottleneck_size[2])
                token_list.append(token)
                frames_skip_connection.append(skip_connections)

            frames = torch.stack(token_list)
            frames = frames.permute(1, 2, 0, 3, 4).contiguous() # B C D H W
            frames, layer_out = self.bottleneck(frames)

            frames = frames.permute(2, 0, 3, 4, 1).view(T, B, -1, self.d_model)

            for token, skip_connections in zip(frames, frames_skip_connection):
                pred = self.decoder(token, skip_connections)
                preds.append(pred)
        elif len(self.patch_size) == 3:
            x = x.permute(1, 2, 0, 3, 4) # B C D H W
            token, skip_connections = self.encoder(x)
        return preds

    def forward(self, x):
        if x.shape[0] == 1 and x.dim() == 5:
            x = torch.squeeze(x, dim=0)
            return [self.forward_single_frame(x)]
        elif x.dim() == 4:
            return self.forward_single_frame(x)
        else:
            if self.bottleneck_name == 'factorized':
                return self.forward_factorized(x)
            elif self.bottleneck_name == 'vit_3d':
                return self.forward_vit_3d(x)
            elif self.bottleneck_name == 'swin_3d':
                return self.forward_swin_3d(x)


class ConvUnet(nn.Module):
    def __init__(self, 
                blur,
                blur_kernel,
                deep_supervision,
                merge,
                conv_unet_depth,
                device,
                in_dims,
                image_size,
                num_bottleneck_layers=2,
                drop_path_rate=0.1):

        super(ConvUnet, self).__init__()
        num_stages = len(conv_unet_depth)
        d_model = in_dims[1] * (2 ** (num_stages - 1))

        # stochastic depth
        num_blocks = conv_unet_depth + [num_bottleneck_layers]
        my_iter = torch.cat([torch.tensor([0]), torch.cumsum(torch.tensor(num_blocks), dim=0)])
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(num_blocks))]
        dpr = [dpr[my_iter[j]:my_iter[j+1]] for j in range(len(num_blocks))]
        dpr_encoder = dpr[:num_stages]
        dpr_decoder = [x[::-1] for x in dpr_encoder[::-1]]
        dpr_bottleneck = dpr[-1]

        decoder_in_dims = [x * 2 for x in in_dims]
        decoder_in_dims[0] = in_dims[1]
        decoder_in_dims = decoder_in_dims[::-1]

        self.encoder = ConvEncoder(blur=blur,
                                    device=device,
                                    blur_kernel=blur_kernel,
                                    dpr=dpr_encoder,
                                    in_localizer_dims=in_dims,
                                    img_size=image_size,
                                    localizer_conv_depth=conv_unet_depth)
        self.decoder = ConvDecoder(last_activation='softmax',
                                    blur=blur,
                                    device=device,
                                    merge=merge,
                                    blur_kernel=blur_kernel,
                                    deep_supervision=deep_supervision,
                                    dpr=dpr_decoder,
                                    in_localizer_dims=decoder_in_dims,
                                    img_size=image_size,
                                    localizer_conv_depth=conv_unet_depth)
        last_res = int(image_size / (2 ** num_stages))
        self.bottleneck = ResnetConvLayer(input_resolution=[last_res, last_res], in_dim=d_model, out_dim=d_model * 2, nb_se_blocks=num_bottleneck_layers, dpr=dpr_bottleneck)

    def forward(self, x):
        B, C, H, W = x.shape
        x, skip_connections = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x, skip_connections)

        return x