from locale import normalize
import torch
import torch.nn as nn

from timm.models.layers import trunc_normal_
from .utils import get_root_logger, ConvDownBlock_3d, FromGrayscale, DownsampleLayer, MLP, ConvLayerDiscriminator
from torch.nn.utils.parametrizations import spectral_norm
from .swin_transformer_2 import BasicLayer
from .encoder import Encoder
from.decoder_alt import SegmentationDecoder
from .vit_transformer import TransformerEncoderLayer, TransformerEncoder
from .position_embedding import PositionEmbeddingSine2d

class Discriminator(nn.Module):
    def __init__(self,
                conv_depth,
                image_size,
                num_bottleneck_layers,
                drop_path_rate,
                out_encoder_dims,
                norm,
                device,
                in_encoder_dims,
                conv_layer,
                bottleneck_heads
                ):
        super().__init__()

        num_stages = len(conv_depth)
        self.d_model = out_encoder_dims[-1] * 2

        # stochastic depth
        num_blocks = conv_depth + [num_bottleneck_layers]
        my_iter = torch.cat([torch.tensor([0]), torch.cumsum(torch.tensor(num_blocks), dim=0)])
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(num_blocks))]
        dpr = [dpr[my_iter[j]:my_iter[j+1]] for j in range(len(num_blocks))]
        dpr_encoder = dpr[:num_stages]

        self.final_conv = nn.Conv2d(in_channels=self.d_model, out_channels=1, kernel_size=int(image_size/2**3))

        print(out_encoder_dims)
        print(in_encoder_dims)
        print(conv_depth)
        print(bottleneck_heads)

        self.discriminator_enc = Encoder(conv_layer=conv_layer, norm=getattr(torch.nn, norm), out_dims=out_encoder_dims, device=device, in_dims=in_encoder_dims, conv_depth=conv_depth, dpr=dpr_encoder)

        self.pos = PositionEmbeddingSine2d(num_pos_feats=self.d_model // 2, normalize=True)
        self_attention_layer = TransformerEncoderLayer(d_model=self.d_model, nhead=bottleneck_heads, dim_feedforward=2048)
        self.self_attention_bottleneck = TransformerEncoder(encoder_layer=self_attention_layer, num_layers=1)
    
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x, _ = self.discriminator_enc(x)
        B, C, H, W = x.shape
        pos = self.pos(shape_util=(B, H, W), device=x.device)
        x = self.self_attention_bottleneck(x, pos=pos)
        x = self.final_conv(x)
        return self.sigmoid(x)


class ConfidenceNetwork(nn.Module):
    def __init__(self,
                transformer_depth,
                deep_supervision,
                conv_depth,
                image_size,
                num_bottleneck_layers,
                drop_path_rate,
                out_encoder_dims,
                norm,
                attention_map,
                shortcut,
                proj,
                use_conv_mlp,
                blur,
                blur_kernel,
                device,
                swin_abs_pos,
                in_encoder_dims,
                filter_skip_co_segmentation,
                window_size,
                num_heads,
                rpe_contextual_tensors,
                rpe_mode,
                bottleneck,
                bottleneck_heads,
                spatial_cross_attention_num_heads
                ):
        super().__init__()

        num_stages = (len(transformer_depth) + len(conv_depth))
        bottleneck_size = [int(image_size / (2**num_stages)), int(image_size / (2**num_stages))]

        # stochastic depth
        num_blocks = conv_depth + transformer_depth + [num_bottleneck_layers]
        my_iter = torch.cat([torch.tensor([0]), torch.cumsum(torch.tensor(num_blocks), dim=0)])
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(num_blocks))]
        dpr = [dpr[my_iter[j]:my_iter[j+1]] for j in range(len(num_blocks))]
        dpr_encoder = dpr[:num_stages]
        dpr_decoder = [x[::-1] for x in dpr_encoder[::-1]]
        dpr_bottleneck = dpr[-1]

        print(out_encoder_dims)
        print(in_encoder_dims)
        print(conv_depth)

        self.discriminator_enc = Encoder(norm=getattr(torch.nn, norm), attention_map=attention_map, out_dims=out_encoder_dims, shortcut=shortcut, proj=proj, use_conv_mlp=use_conv_mlp, blur=blur, img_size=image_size, blur_kernel=blur_kernel, device=device, swin_abs_pos=swin_abs_pos, in_dims=in_encoder_dims, conv_depth=conv_depth[::-1], transformer_depth=transformer_depth, bottleneck_type=bottleneck, dpr=dpr_encoder, rpe_mode=rpe_mode, rpe_contextual_tensor=rpe_contextual_tensors, num_heads=num_heads, window_size=window_size, deep_supervision=False)
        conv_depth_decoder = conv_depth[::-1]

        self.decoder = SegmentationDecoder(last_activation='sigmoid', similarity=False, norm=getattr(torch.nn, norm), similarity_down_scale=8, filter_skip_co_segmentation=filter_skip_co_segmentation, shift_nb=5, start_reconstruction_dim=4, directional_field=False, attention_map=False, reconstruction=False, reconstruction_skip=False, concat_spatial_cross_attention=True, attention_type=[], spatial_cross_attention_num_heads=spatial_cross_attention_num_heads[::-1], shortcut=shortcut, proj_qkv=proj, out_encoder_dims=out_encoder_dims[::-1], use_conv_mlp=use_conv_mlp, blur=blur, img_size=image_size, num_classes=1, blur_kernel=blur_kernel, device=device, swin_abs_pos=swin_abs_pos, in_encoder_dims=in_encoder_dims[::-1], conv_depth=conv_depth_decoder, transformer_depth=transformer_depth[::-1], dpr=dpr_decoder, rpe_mode=rpe_mode, rpe_contextual_tensor=rpe_contextual_tensors, num_heads=num_heads, window_size=window_size, drop_path_rate=drop_path_rate, deep_supervision=deep_supervision)

        self.bottleneck = BasicLayer(dim=int(out_encoder_dims[-1]*2),
                                    norm=getattr(torch.nn, norm),
                                    attention_map=attention_map,
                                    input_resolution=bottleneck_size,
                                    shortcut=shortcut,
                                    depth=num_bottleneck_layers,
                                    num_heads=bottleneck_heads,
                                    proj=proj,
                                    use_conv_mlp=use_conv_mlp,
                                    device=device,
                                    rpe_mode=rpe_mode,
                                    rpe_contextual_tensor=rpe_contextual_tensors,
                                    window_size=window_size,
                                    mlp_ratio=4.,
                                    qkv_bias=True, qk_scale=None,
                                    drop=0., attn_drop=0.,
                                    drop_path=dpr_bottleneck,
                                    norm_layer=nn.LayerNorm,
                                    use_checkpoint=False)
    
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x, encoder_skip_connections = self.discriminator_enc(x)
        x = self.bottleneck(x)
        x, _, _ = self.decoder(x=x, encoder_skip_connections=encoder_skip_connections)
        return x


#class Discriminator(nn.Module):
#    r""" Swin Transformer
#        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
#          https://arxiv.org/pdf/2103.14030
#
#    Args:
#        img_size (int | tuple(int)): Input image size. Default 224
#        patch_size (int | tuple(int)): Patch size. Default: 4
#        in_chans (int): Number of input image channels. Default: 3
#        num_classes (int): Number of classes for classification head. Default: 1000
#        embed_dim (int): Patch embedding dimension. Default: 96
#        depths (tuple(int)): Depth of each Swin Transformer layer.
#        num_heads (tuple(int)): Number of attention heads in different layers.
#        window_size (int): Window size. Default: 7
#        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
#        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
#        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
#        drop_rate (float): Dropout rate. Default: 0
#        attn_drop_rate (float): Attention dropout rate. Default: 0
#        drop_path_rate (float): Stochastic depth rate. Default: 0.1
#        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
#        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
#        patch_norm (bool): If True, add normalization after patch embedding. Default: True
#        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
#    """
#
#    def __init__(self, device, dpr, in_discriminator_dims, swin_abs_pos, swin_layer_type, transformer_type='swin', proj='linear', rpe_mode=None, rpe_contextual_tensor='qkv', img_size=224, patch_size=[4, 4],
#                 embed_dim=96, discriminator_conv_depth=[2, 2], discriminator_transformer_depth=[2, 2, 2, 2], discriminator_num_heads=[3, 6, 12, 24],
#                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
#                 drop_rate=0., attn_drop_rate=0.,
#                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
#                 use_checkpoint=False, deep_supervision=True, **kwargs):
#        super().__init__()
#
#        self.num_stages = len(discriminator_conv_depth) + len(discriminator_transformer_depth)
#        self.embed_dim = embed_dim
#        self.ape = ape
#        self.patch_norm = patch_norm
#        #self.num_features = int(embed_dim * 2 ** (self.num_stages - 2))
#        self.mlp_ratio = mlp_ratio
#        self.deep_supervision = deep_supervision
#        self.d_model = embed_dim * 2**(len(discriminator_transformer_depth) - 1)
#
#        patches_resolution = [img_size // patch_size[0], img_size // patch_size[1]]
#        num_patches = patches_resolution[0] * patches_resolution[1]
#
#        # absolute position embedding
#        if self.ape:
#            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
#            trunc_normal_(self.absolute_pos_embed, std=.02)
#
#        self.pos_drop = nn.Dropout(p=drop_rate)
#
#        # build encoder layers
#        self.layers = nn.ModuleList()
#        self.downsample_layers = nn.ModuleList()
#        for i_layer in range(self.num_stages):
#            input_resolution=(img_size//(2**i_layer), img_size//(2**i_layer))
#            if i_layer < len(discriminator_conv_depth):
#                if i_layer == 0:
#                    layer = ConvLayer(input_resolution=input_resolution, 
#                                    in_dim=in_discriminator_dims[i_layer],
#                                    out_dim=in_discriminator_dims[i_layer + 1],
#                                    nb_se_blocks=discriminator_conv_depth[i_layer], 
#                                    dpr=dpr[i_layer])
#                    downsample_layer = PatchMergingConv(input_resolution=input_resolution, in_dim=in_discriminator_dims[i_layer + 1], out_dim=in_discriminator_dims[i_layer + 1], swin_abs_pos=False, device=device)
#                elif i_layer == 1:
#                    layer = ConvLayer(input_resolution=input_resolution, 
#                                    in_dim=in_discriminator_dims[i_layer],
#                                    out_dim=2 * in_discriminator_dims[i_layer],
#                                    nb_se_blocks=discriminator_conv_depth[i_layer], 
#                                    dpr=dpr[i_layer])
#                    downsample_layer = PatchMergingConv(input_resolution=input_resolution, in_dim=2 * in_discriminator_dims[i_layer], out_dim=in_discriminator_dims[i_layer + 1], swin_abs_pos=swin_abs_pos, device=device)
#            else:
#                if i_layer < self.num_stages - 1:
#                    downsample_layer = PatchMergingConv(input_resolution=input_resolution, in_dim=in_discriminator_dims[i_layer], out_dim=2 * in_discriminator_dims[i_layer], swin_abs_pos=swin_abs_pos, device=device)
#                else:
#                    self.last_input_resolution = input_resolution
#                    self.final_conv = nn.Conv2d(in_channels=in_discriminator_dims[-1], out_channels=1, kernel_size=input_resolution[0])
#                    downsample_layer = nn.Identity()
#                transformer_index = i_layer-len(discriminator_conv_depth)
#                if transformer_type == 'swin':
#                    layer = swin_layer_type.BasicLayer(dim=in_discriminator_dims[i_layer],
#                                    input_resolution=input_resolution,
#                                    depth=discriminator_transformer_depth[transformer_index],
#                                    num_heads=discriminator_num_heads[transformer_index],
#                                    proj=proj,
#                                    device=device,
#                                    rpe_mode=rpe_mode,
#                                    rpe_contextual_tensor=rpe_contextual_tensor,
#                                    window_size=window_size,
#                                    mlp_ratio=self.mlp_ratio,
#                                    qkv_bias=qkv_bias, 
#                                    qk_scale=qk_scale,
#                                    drop=drop_rate, 
#                                    attn_drop=attn_drop_rate,
#                                    drop_path=dpr[i_layer],
#                                    norm_layer=norm_layer,
#                                    use_checkpoint=use_checkpoint)
#                elif transformer_type == 'vit':
#                    layer = VitBasicLayer(in_dim=in_discriminator_dims[i_layer], 
#                                        nhead=discriminator_num_heads[transformer_index], 
#                                        rpe_mode=rpe_mode, 
#                                        rpe_contextual_tensor=rpe_contextual_tensor, 
#                                        input_resolution=input_resolution, 
#                                        proj=proj, 
#                                        device=device, 
#                                        nb_blocks=discriminator_transformer_depth[transformer_index],
#                                        dpr=dpr[i_layer])
#            self.layers.append(layer)
#            self.downsample_layers.append(downsample_layer)
#
#        self.sigmoid = torch.nn.Sigmoid()
#
#        #self.norm = norm_layer(self.num_features)
#        #self.norm_after_conv = norm_layer(embed_dim)
#
#        self.apply(self._init_weights)
#
#    def _init_weights(self, m):
#        if isinstance(m, nn.Linear):
#            trunc_normal_(m.weight, std=.02)
#            if isinstance(m, nn.Linear) and m.bias is not None:
#                nn.init.constant_(m.bias, 0)
#        elif isinstance(m, nn.LayerNorm):
#            nn.init.constant_(m.bias, 0)
#            nn.init.constant_(m.weight, 1.0)
#
#    @torch.jit.ignore
#    def no_weight_decay(self):
#        return {'absolute_pos_embed'}
#
#    @torch.jit.ignore
#    def no_weight_decay_keywords(self):
#        return {'relative_position_bias_table'}
#
#    def forward(self, x):
#        B, C, H, W = x.shape
#        x = x.permute(0, 2, 3, 1).view(B, -1, C)
#
#        for layer, downsample_layer in zip(self.layers, self.downsample_layers):
#            x = layer(x)
#            x = downsample_layer(x)
#
#        x = x.permute(0, 2, 1).view(B, self.d_model, self.last_input_resolution[0], self.last_input_resolution[1])
#        
#        x = self.final_conv(x)
#
#        x = self.sigmoid(x)
#        
#        return x

class SpectralConvDiscriminator(nn.Module):

    def __init__(self, blur, shortcut, blur_kernel, dpr, in_discriminator_dims, out_discriminator_dims, img_size=224, discriminator_conv_depth=[2, 2, 2]):
        super().__init__()

        self.num_stages = len(discriminator_conv_depth)
        self.d_model = in_discriminator_dims[-1] * 2
        self.last_res_size = int(img_size / (2**(self.num_stages - 1)))

        # build encoder layers
        self.layers = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        for i_layer in range(self.num_stages):
            input_resolution = (int(img_size / (2**(i_layer))), int(img_size / (2**(i_layer))))
            layer = ConvLayerDiscriminator(input_resolution=input_resolution,
                                            in_dim=in_discriminator_dims[i_layer], 
                                            out_dim=out_discriminator_dims[i_layer], 
                                            nb_se_blocks=discriminator_conv_depth[i_layer], 
                                            dpr=dpr[i_layer],
                                            shortcut=shortcut)
            if i_layer == self.num_stages - 1:
                downsample_layer = nn.Identity()
            else:
                #down_dim = in_discriminator_dims[i_layer + 1] if i_layer == 0 else in_discriminator_dims[i_layer] * 2
                downsample_layer = DownsampleLayer(input_resolution=input_resolution, blur=blur, blur_kernel=blur_kernel)
            self.layers.append(layer)
            self.downsample_layers.append(downsample_layer)
        
        self.final_conv = nn.Conv2d(in_channels=self.d_model, out_channels=1, kernel_size=self.last_res_size)

        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).view(B, -1, C)

        for layer, downsample_layer in zip(self.layers, self.downsample_layers):
            x  = layer(x)
            x = downsample_layer(x)
        
        x = x.permute(0, 2, 1).view(B, self.d_model, self.last_res_size, self.last_res_size)
        
        x = self.final_conv(x)
        x = self.sigmoid(x)
        
        return x

#class SpectralConvDiscriminator(nn.Module):
#
#    def __init__(self, progressive_growing, blur, blur_kernel, dpr, in_discriminator_dims, out_discriminator_dims, img_size=224, discriminator_conv_depth=[2, 2, 2, 2]):
#        super().__init__()
#
#        self.num_stages = len(discriminator_conv_depth)
#        self.d_model = in_discriminator_dims[-1] * 2
#        self.last_res_size = int(img_size / (2**(self.num_stages - 1)))
#        self.progressive_growing = progressive_growing
#
#        # build encoder layers
#        self.layers = nn.ModuleList()
#        self.downsample_layers = nn.ModuleList()
#        self.from_grayscales = nn.ModuleList()
#        for i_layer in range(self.num_stages):
#            input_resolution = (int(img_size / (2**(i_layer))), int(img_size / (2**(i_layer))))
#            from_grayscale = FromGrayscale(out_dim=in_discriminator_dims[i_layer], in_dim=1)
#            layer = ConvLayerDiscriminator(input_resolution=input_resolution,
#                                            in_dim=in_discriminator_dims[i_layer], 
#                                            out_dim=out_discriminator_dims[i_layer], 
#                                            nb_se_blocks=discriminator_conv_depth[i_layer], 
#                                            dpr=dpr[i_layer])
#            if i_layer == self.num_stages - 1:
#                downsample_layer = nn.Identity()
#            else:
#                #down_dim = in_discriminator_dims[i_layer + 1] if i_layer == 0 else in_discriminator_dims[i_layer] * 2
#                downsample_layer = DownsampleLayer(input_resolution=input_resolution, blur=blur, blur_kernel=blur_kernel)
#            self.layers.append(layer)
#            self.downsample_layers.append(downsample_layer)
#            if self.progressive_growing:
#                self.from_grayscales.append(from_grayscale)
#        
#        self.final_conv = nn.Conv2d(in_channels=self.d_model, out_channels=1, kernel_size=self.last_res_size)
#
#        self.sigmoid = torch.nn.Sigmoid()
#    
#    def forward_plain(self, x):
#        B, C, H, W = x.shape
#        x = x.permute(0, 2, 3, 1).view(B, -1, C)
#
#        for layer, downsample_layer in zip(self.layers, self.downsample_layers):
#            x, fm = layer(x)
#            x = downsample_layer(x)
#        
#        x = x.permute(0, 2, 1).view(B, self.d_model, self.last_res_size, self.last_res_size)
#        
#        x = self.final_conv(x)
#        x = self.sigmoid(x)
#        
#        return x
#    
#    def forward_progressive_growing(self, x, stage_nb, alpha, fade_in):
#        layers = self.layers[-stage_nb:]
#        downsample_layers = self.downsample_layers[-stage_nb:]
#        from_grayscale = self.from_grayscales[self.num_stages - stage_nb]
#
#        for i, (layer, downsample_layer) in enumerate(zip(layers, downsample_layers)):
#            if i == 0:
#                if fade_in:
#                    x1 = downsample_layer(x)
#                    x1 = self.from_grayscales[self.num_stages - stage_nb + 1](x1)
#
#                    x2 = from_grayscale(x)
#                    x2, fm = layer(x2)
#                    x2 = downsample_layer(x2)
#                    x = alpha * x2 + (1 - alpha) * x1
#                else:
#                    x = from_grayscale(x)
#                    x, fm = layer(x)
#                    x = downsample_layer(x)
#            else:
#                x, fm = layer(x)
#                x = downsample_layer(x)
#        
#        x = self.final_conv(x)
#
#        x = self.sigmoid(x)
#        
#        return x
#    
#    def forward(self, x, stage_nb=3, alpha=0, fade_in=False):
#        if self.progressive_growing:
#            return self.forward_progressive_growing(x, stage_nb, alpha, fade_in)
#        else:
#            return self.forward_plain(x)