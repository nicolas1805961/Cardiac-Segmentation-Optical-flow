import torch
import torch.nn as nn

from timm.models.layers import trunc_normal_
import swin_transformer_3d
from utils import ConvUpBlock_3d, DeepSupervision, get_root_logger, GetSimilarityMatrix, PatchExpandSwin, ToGrayscale, ConvLayerAdaIN, ConvLayer, PatchExpandConv, PatchExpandConv_3d, concat_merge_linear_rescale, concat_merge_conv_rescale, concat_merge_linear
from einops import rearrange
from vit_transformer import TransformerEncoderLayer, VitBasicLayer
import copy
from position_embedding import PositionEmbeddingSine2d
import swin_transformer
from adain import adaptive_instance_normalization
import random

class Generator(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, mapping_function_nb_layers, batch_size, style_mixing_p, device, dpr, in_generator_dims, swin_layer_type, swin_abs_pos, transformer_type='swin', proj='linear', rpe_mode=None, rpe_contextual_tensor='qkv', img_size=224, num_classes=4,
                 embed_dim=96, generator_conv_depth=[2, 2], generator_transformer_depth=[2, 2, 2, 2], generator_num_heads=[24, 12, 6, 3],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0.,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, deep_supervision=True, **kwargs):
        super().__init__()

        self.num_stages = len(generator_conv_depth) + len(generator_transformer_depth)
        d_model = embed_dim * (2 ** (len(generator_transformer_depth) - 1))
        self.mapping_function = MappingFunction(mapping_function_nb_layers, d_model)
        last_res = int(img_size / (2 ** (self.num_stages - 1)))
        self.fixed_input = torch.full(size=(batch_size, last_res*last_res, d_model), fill_value=1, device=device)
        self.embed_dim = embed_dim
        self.style_mixing_p = style_mixing_p
        self.ape = ape
        self.patch_norm = patch_norm
        #self.num_features = int(embed_dim * 2 ** self.num_layers)
        self.mlp_ratio = mlp_ratio
        self.deep_supervision = deep_supervision
        self.img_size = img_size
        
        # build decoder layers
        self.layers = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()
        for i_layer in range(self.num_stages):
            conv_index = len(generator_transformer_depth) - i_layer
            input_resolution=(img_size//(2**(self.num_stages-i_layer-1)), img_size//(2**(self.num_stages-i_layer-1)))
            #in_dim = int(embed_dim*(2**(self.num_stages-i_layer-3)))

            if i_layer < len(generator_transformer_depth):
                upsample_layer = PatchExpandConv(input_resolution=input_resolution, in_dim=in_generator_dims[i_layer], out_dim=in_generator_dims[i_layer + 1], swin_abs_pos=swin_abs_pos, device=device)
                if transformer_type == 'swin':
                    layer_up = swin_layer_type.BasicLayerUp(dim=in_generator_dims[i_layer],
                                                                input_resolution=input_resolution,
                                                                depth=generator_transformer_depth[i_layer],
                                                                num_heads=generator_num_heads[i_layer],
                                                                proj=proj,
                                                                device=device,
                                                                rpe_mode=rpe_mode,
                                                                rpe_contextual_tensor=rpe_contextual_tensor,
                                                                window_size=window_size,
                                                                mlp_ratio=self.mlp_ratio,
                                                                qkv_bias=qkv_bias, 
                                                                qk_scale=qk_scale,
                                                                drop=drop_rate, 
                                                                attn_drop=attn_drop_rate,
                                                                drop_path=dpr[i_layer],
                                                                norm_layer=norm_layer,
                                                                use_checkpoint=use_checkpoint, 
                                                                deep_supervision=self.deep_supervision)
                elif transformer_type == 'vit':
                    layer_up = VitBasicLayer(in_dim=in_generator_dims[i_layer], 
                                            nhead=generator_num_heads[i_layer], 
                                            rpe_mode=rpe_mode, 
                                            rpe_contextual_tensor=rpe_contextual_tensor, 
                                            input_resolution=input_resolution, 
                                            proj=proj, 
                                            device=device, 
                                            nb_blocks=generator_transformer_depth[i_layer],
                                            dpr=dpr[i_layer])
            else:
                if i_layer == 4:
                    upsample_layer = PatchExpandConv(input_resolution=input_resolution, in_dim=in_generator_dims[i_layer + 1], out_dim=in_generator_dims[i_layer + 1], swin_abs_pos=False, device=device)
                    layer_up = ConvLayer(input_resolution=input_resolution, 
                                    in_dim=in_generator_dims[i_layer], 
                                    out_dim=in_generator_dims[i_layer + 1],
                                    nb_se_blocks=generator_conv_depth[conv_index], 
                                    dpr=dpr[i_layer])
                elif i_layer == 5:
                    upsample_layer = nn.Identity()
                    layer_up = ConvLayer(input_resolution=input_resolution, 
                                        in_dim=in_generator_dims[i_layer], 
                                        out_dim=1,
                                        nb_se_blocks=generator_conv_depth[conv_index], 
                                        dpr=dpr[i_layer])
            self.layers.append(layer_up)
            self.upsample_layers.append(upsample_layer)

        self.norm = nn.LayerNorm(embed_dim*2**3)
        #self.norm_up = norm_layer(self.embed_dim)
        self.sigmoid = torch.nn.Sigmoid()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}
    
    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).view(B, -1, C)
        #x = self.norm(x)
        for i, (layer_up, upsample_layer) in enumerate(zip(self.layers, self.upsample_layers)):
            x = layer_up(x)
            x = upsample_layer(x)

        x = x.permute(0, 2, 1).view(B, 1, self.img_size, self.img_size)
        x = self.sigmoid(x)

        #x = self.norm_up(x)  # B L C

        return x
    
    def forward_alt(self, x):
        B, C = x.shape
        w1 = self.mapping_function(x)

        if random.random() < self.style_mixing_p:
            noise = torch.rand(size=(B, C), dtype=x.dtype, device=x.device)
            w2 = self.mapping_function(noise)
            w1 = torch.repeat_interleave(w1[:, None, :], self.num_stages//2, dim=1)
            w2 = torch.repeat_interleave(w2[:, None, :], self.num_stages//2, dim=1)
            w = torch.cat([w1, w2], dim=1)
        else:
            w = torch.repeat_interleave(w1[:, None, :], self.num_stages, dim=1)

        x_temp = self.fixed_input
        for i, (layer_up, upsample_layer) in enumerate(zip(self.layers, self.upsample_layers)):
            x_temp = layer_up(x_temp, w[:, i, :])
            x_temp = upsample_layer(x_temp)

        x = x.permute(0, 2, 1).view(B, 1, self.img_size, self.img_size)
        x = self.sigmoid(x)

        #x = self.norm_up(x)  # B L C

        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops


class Reconstruction(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, proj, similarity_down_scale, blur, shortcut, reconstruction_attention_type, nb_classes, use_conv_mlp, blur_kernel, device, dpr, out_encoder_dims, in_encoder_dims, window_size, img_size, swin_layer_type, swin_abs_pos, rpe_mode=None, rpe_contextual_tensor='qkv',
                 conv_depth=[2, 2], transformer_depth=[2, 2, 2], num_heads=[3, 6, 12],
                 mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0.,
                 norm_layer=nn.LayerNorm,
                 use_checkpoint=False, deep_supervision=True, **kwargs):
        super().__init__()

        self.num_stages = len(conv_depth) + len(transformer_depth)
        self.mlp_ratio = mlp_ratio
        self.deep_supervision = deep_supervision
        self.img_size = img_size
        self.reconstruction_attention_type = reconstruction_attention_type
        
        # build decoder layers
        self.layers = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()
        self.to_grayscales = nn.ModuleList()
        self.to_grayscales_residual = nn.ModuleList()
        self.sm_computation_layers = nn.ModuleList()
        for i_layer in range(self.num_stages):
            conv_index = self.num_stages - i_layer - 1
            input_resolution=(img_size//(2**(self.num_stages-i_layer-1)), img_size//(2**(self.num_stages-i_layer-1)))
            to_grayscale_residual = ToGrayscale(input_resolution=input_resolution, in_dim=out_encoder_dims[self.num_stages - i_layer - 1], out_dim=1)

            scale = similarity_down_scale / (2**(self.num_stages - i_layer - 1))
            sm_computation = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=4, kernel_size=1), 
                                                nn.BatchNorm2d(4), 
                                                nn.GELU(), 
                                                GetSimilarityMatrix(scale))
            self.sm_computation_layers.append(sm_computation)

            if i_layer < len(transformer_depth):
                transformer_index = len(transformer_depth) - i_layer - 1
                to_grayscale = ToGrayscale(input_resolution=input_resolution, in_dim=in_encoder_dims[i_layer], out_dim=1)
                upsample_layer = PatchExpandSwin(blur=blur, blur_kernel=blur_kernel, input_resolution=[x//2 for x in input_resolution], in_dim=in_encoder_dims[i_layer]*2, out_dim=in_encoder_dims[i_layer], swin_abs_pos=swin_abs_pos, device=device)
                layer_up = swin_layer_type.BasicLayerUp(dim=in_encoder_dims[i_layer],
                                                        input_resolution=input_resolution,
                                                        proj=proj,
                                                        shortcut=shortcut,
                                                        depth=transformer_depth[transformer_index],
                                                        num_heads=num_heads[transformer_index],
                                                        device=device,
                                                        use_conv_mlp=use_conv_mlp,
                                                        rpe_mode=rpe_mode,
                                                        rpe_contextual_tensor=rpe_contextual_tensor,
                                                        window_size=window_size,
                                                        mlp_ratio=self.mlp_ratio,
                                                        qkv_bias=qkv_bias, 
                                                        qk_scale=qk_scale,
                                                        drop=drop_rate, 
                                                        attn_drop=attn_drop_rate,
                                                        drop_path=dpr[i_layer],
                                                        norm_layer=norm_layer,
                                                        use_checkpoint=use_checkpoint, 
                                                        deep_supervision=self.deep_supervision)
            else:
                if i_layer == self.num_stages - 2:
                    upsample_layer = PatchExpandConv(blur=blur, blur_kernel=blur_kernel, input_resolution=[x//2 for x in input_resolution], in_dim=in_encoder_dims[i_layer]*4, out_dim=2*in_encoder_dims[i_layer], swin_abs_pos=False, device=device)
                    to_grayscale = ToGrayscale(input_resolution=input_resolution, in_dim=in_encoder_dims[i_layer], out_dim=1)
                    layer_up = ConvLayer(input_resolution=input_resolution, 
                                    in_dim=2*in_encoder_dims[i_layer], 
                                    out_dim=in_encoder_dims[i_layer],
                                    nb_se_blocks=conv_depth[conv_index], 
                                    dpr=dpr[i_layer])
                elif i_layer == self.num_stages - 1:
                    upsample_layer = PatchExpandConv(blur=blur, blur_kernel=blur_kernel, input_resolution=[x//2 for x in input_resolution], in_dim=in_encoder_dims[i_layer-1], out_dim=in_encoder_dims[i_layer-1], swin_abs_pos=False, device=device)
                    to_grayscale = ToGrayscale(input_resolution=input_resolution, in_dim=1, out_dim=1)
                    layer_up = ConvLayer(input_resolution=input_resolution, 
                                        in_dim=in_encoder_dims[i_layer-1], 
                                        out_dim=nb_classes,
                                        nb_se_blocks=conv_depth[conv_index], 
                                        dpr=dpr[i_layer])
            self.layers.append(layer_up)
            self.upsample_layers.append(upsample_layer)
            self.to_grayscales.append(to_grayscale)
            self.to_grayscales_residual.append(to_grayscale_residual)

        self.norm = nn.LayerNorm(in_encoder_dims[0] * 2)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x, stage_nb, alpha, fade_in=False):
        skip_connections = [None] * self.num_stages

        layers = self.layers[:stage_nb]
        upsample_layers = self.upsample_layers[:stage_nb]
        to_grayscale = self.to_grayscales[stage_nb - 1]
        to_grayscale_residual = self.to_grayscales_residual[stage_nb - 1]

        for i, (layer_up, upsample_layer, sm_computation_layer) in enumerate(zip(layers, upsample_layers, self.sm_computation_layers)):
            if i == stage_nb - 1:
                if fade_in:
                    x_up = upsample_layer(x)
                    if self.reconstruction_attention_type:
                        skip_connections[i] = x_up
                    x = to_grayscale_residual(x_up)
                    x_up = layer_up(x_up)
                    x_up = to_grayscale(x_up)
                    x = alpha * x_up + (1 - alpha) * x
                else:
                    x = upsample_layer(x)
                    if self.reconstruction_attention_type:
                        skip_connections[i] = x
                    x = layer_up(x)
                    x = to_grayscale(x)
                similarity_matrix = sm_computation_layer(x)
            else:
                x = upsample_layer(x)
                if self.reconstruction_attention_type:
                    skip_connections[i] = x
                x = layer_up(x)

        #x = self.norm_up(x)  # B L C

        return [x], skip_connections, similarity_matrix


class Generator2(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, out_dim_proj, blur, blur_kernel, latent_size, mapping_function_nb_layers, batch_size, style_mixing_p, device, dpr, in_generator_dims, swin_layer_type, swin_abs_pos, transformer_type='swin', proj='linear', rpe_mode=None, rpe_contextual_tensor='qkv', img_size=224, num_classes=4,
                 embed_dim=96, generator_conv_depth=[2, 2], generator_transformer_depth=[2, 2, 2, 2], generator_num_heads=[24, 12, 6, 3],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0.,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, deep_supervision=True, **kwargs):
        super().__init__()

        self.num_stages = len(generator_conv_depth) + len(generator_transformer_depth)
        self.d_model = embed_dim * (2 ** (len(generator_transformer_depth) - 1))
        self.mapping_function = MappingFunction(mapping_function_nb_layers, self.d_model)
        self.last_res = int(img_size / (2 ** (self.num_stages - 1)))
        self.embed_dim = embed_dim
        self.style_mixing_p = style_mixing_p
        self.ape = ape
        self.patch_norm = patch_norm
        #self.num_features = int(embed_dim * 2 ** self.num_layers)
        self.mlp_ratio = mlp_ratio
        self.deep_supervision = deep_supervision
        self.img_size = img_size
        
        # build decoder layers
        self.layers = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()
        self.to_grayscales = nn.ModuleList()
        self.to_grayscales_residual = nn.ModuleList()
        for i_layer in range(self.num_stages):
            conv_index = len(generator_transformer_depth) - i_layer
            input_resolution=(img_size//(2**(self.num_stages-i_layer-1)), img_size//(2**(self.num_stages-i_layer-1)))
            to_grayscale_residual = ToGrayscale(input_resolution=input_resolution, in_dim=in_generator_dims[i_layer], out_dim=out_dim_proj)

            if i_layer < len(generator_transformer_depth):
                to_grayscale = ToGrayscale(input_resolution=input_resolution, in_dim=in_generator_dims[i_layer], out_dim=out_dim_proj)
                if i_layer == 0:
                    upsample_layer = nn.Identity()
                else:
                    upsample_layer = PatchExpandSwin(blur=blur, blur_kernel=blur_kernel, input_resolution=[x//2 for x in input_resolution], in_dim=in_generator_dims[i_layer - 1], out_dim=in_generator_dims[i_layer], swin_abs_pos=swin_abs_pos, device=device)
                if transformer_type == 'swin':
                    layer_up = swin_layer_type.BasicLayerUp(dim=in_generator_dims[i_layer],
                                                            latent_size=latent_size,
                                                            input_resolution=input_resolution,
                                                            depth=generator_transformer_depth[i_layer],
                                                            num_heads=generator_num_heads[i_layer],
                                                            proj=proj,
                                                            device=device,
                                                            rpe_mode=rpe_mode,
                                                            rpe_contextual_tensor=rpe_contextual_tensor,
                                                            window_size=window_size,
                                                            mlp_ratio=self.mlp_ratio,
                                                            qkv_bias=qkv_bias, 
                                                            qk_scale=qk_scale,
                                                            drop=drop_rate, 
                                                            attn_drop=attn_drop_rate,
                                                            drop_path=dpr[i_layer],
                                                            norm_layer=norm_layer,
                                                            use_checkpoint=use_checkpoint, 
                                                            deep_supervision=self.deep_supervision)
                elif transformer_type == 'vit':
                    layer_up = VitBasicLayer(in_dim=in_generator_dims[i_layer], 
                                            nhead=generator_num_heads[i_layer], 
                                            rpe_mode=rpe_mode, 
                                            rpe_contextual_tensor=rpe_contextual_tensor, 
                                            input_resolution=input_resolution, 
                                            proj=proj, 
                                            device=device, 
                                            nb_blocks=generator_transformer_depth[i_layer],
                                            dpr=dpr[i_layer])
            else:
                if i_layer == 4:
                    upsample_layer = PatchExpandConv(blur=blur, blur_kernel=blur_kernel, input_resolution=[x//2 for x in input_resolution], in_dim=in_generator_dims[i_layer - 1], out_dim=in_generator_dims[i_layer], swin_abs_pos=False, device=device)
                    to_grayscale = ToGrayscale(input_resolution=input_resolution, in_dim=in_generator_dims[i_layer + 1], out_dim=out_dim_proj)
                    layer_up = ConvLayerAdaIN(latent_size=latent_size,
                                    input_resolution=input_resolution, 
                                    in_dim=in_generator_dims[i_layer], 
                                    out_dim=in_generator_dims[i_layer + 1],
                                    nb_se_blocks=generator_conv_depth[conv_index], 
                                    dpr=dpr[i_layer])
                elif i_layer == 5:
                    upsample_layer = PatchExpandConv(blur=blur, blur_kernel=blur_kernel, input_resolution=[x//2 for x in input_resolution], in_dim=in_generator_dims[i_layer], out_dim=in_generator_dims[i_layer], swin_abs_pos=False, device=device)
                    to_grayscale = ToGrayscale(input_resolution=input_resolution, in_dim=out_dim_proj, out_dim=out_dim_proj)
                    layer_up = ConvLayerAdaIN(latent_size=latent_size,
                                        input_resolution=input_resolution, 
                                        in_dim=in_generator_dims[i_layer], 
                                        out_dim=out_dim_proj,
                                        nb_se_blocks=generator_conv_depth[conv_index], 
                                        dpr=dpr[i_layer])
            self.layers.append(layer_up)
            self.upsample_layers.append(upsample_layer)
            self.to_grayscales.append(to_grayscale)
            self.to_grayscales_residual.append(to_grayscale_residual)

        self.norm = nn.LayerNorm(embed_dim*2**3)
        self.last_activation = torch.nn.Sigmoid() if out_dim_proj == 1 else torch.nn.Softmax(dim=1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x, stage_nb, alpha, fade_in=False):
        B, C = x.shape
        w1 = self.mapping_function(x)

        if random.random() < self.style_mixing_p:
            noise = torch.rand(size=(B, C), dtype=x.dtype, device=x.device)
            w2 = self.mapping_function(noise)
            w1 = torch.repeat_interleave(w1[:, None, :], self.num_stages//2, dim=1)
            w2 = torch.repeat_interleave(w2[:, None, :], self.num_stages//2, dim=1)
            w = torch.cat([w1, w2], dim=1)
        else:
            w = torch.repeat_interleave(w1[:, None, :], self.num_stages, dim=1)
        
        layers = self.layers[:stage_nb]
        upsample_layers = self.upsample_layers[:stage_nb]
        to_grayscale = self.to_grayscales[stage_nb - 1]
        to_grayscale_residual = self.to_grayscales_residual[stage_nb - 1]

        x_temp = torch.full(size=(B, self.last_res * self.last_res, self.d_model), fill_value=1.0, device=x.device)
        for i, (layer_up, upsample_layer) in enumerate(zip(layers, upsample_layers)):
            if i == stage_nb - 1:
                if fade_in:
                    x_up = upsample_layer(x_temp)
                    x_temp = to_grayscale_residual(x_up)
                    x_up = layer_up(x_up, w[:, i, :])
                    x_up = to_grayscale(x_up)
                    x_temp = alpha * x_up + (1 - alpha) * x_temp
                else:
                    x_temp = upsample_layer(x_temp)
                    x_temp = layer_up(x_temp, w[:, i, :])
                    x_temp = to_grayscale(x_temp)
            else:
                x_temp = upsample_layer(x_temp)
                x_temp = layer_up(x_temp, w[:, i, :])
        

        x_temp = self.last_activation(x_temp)

        #x = self.norm_up(x)  # B L C

        return x_temp

class PixelNormLayer(nn.Module):
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x):
        return x * torch.rsqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.epsilon)
class MappingFunction(nn.Module):
    def __init__(self, nb_layers=8, dim=768):
        super().__init__()
        self.normalize_latent = PixelNormLayer()
        self.layers = nn.ModuleList()
        self.relu = nn.ReLU()
        for i in range(nb_layers):
            self.layers.append(nn.Linear(dim, dim))
    
    def forward(self, x):
        B, C = x.shape
        x = self.normalize_latent(x)
        for layer in self.layers:
            x = layer(x)
            x = self.relu(x)
        return x

