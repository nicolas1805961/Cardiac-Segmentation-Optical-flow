from numpy import identity
import torch
import torch.nn as nn

import psutil
import os
from timm.models.layers import trunc_normal_
from . import swin_transformer_3d
from .vit_transformer import SlotAttention, SpatioTemporalTransformer, TransformerEncoderLayer
from .utils import TransformerDecoderBlock, ConvDecoderDeformableAttentionIdentity, DeformableAttention, SkipCoDeformableAttention, SpatialTransformerNetwork, CCA, ReplicateChannels, ConvBlock, GetCrossSimilarityMatrix, SelFuseFeature, DeepSupervision3D, CCA3D, ConvLayer3D, From_image3D, From_image, To_image, To_image3D, DeepSupervision, PatchExpand, concat_merge_linear_rescale, concat_merge_conv_rescale, concat_merge_linear, PatchExpandSwin3D, PatchExpandConv3D
from . import swin_transformer_2
from . import swin_cross_attention
import matplotlib.pyplot as plt
#from .swin_cross_attention_2 import SwinFilterBlock, SwinFilterBlockIdentity
from .swin_cross_attention import SwinFilterBlock, SwinFilterBlockIdentity
from torch.nn.functional import grid_sample
from torchvision.transforms.functional import gaussian_blur
from torch.nn.functional import interpolate
import matplotlib

def show_attention_weights(attention_weights, res):
    print(attention_weights.std())
    B, L, C = attention_weights.shape
    temp = attention_weights.permute(0, 2, 1).view(B, C, res, res)
    plt.imshow(temp.detach().cpu()[0, 0], cmap='plasma')
    plt.show()

def show_feature_map(layer_output, x, res):
    B, L, C = layer_output.shape
    fig, ax = plt.subplots(2, 3)
    layer_output = layer_output.permute(0, 2, 1).view(B, C, res, res)
    x = x.permute(0, 2, 1).view(B, 24, res, res)
    im = ax[0, 0].imshow(layer_output[0, 12].detach().cpu(), cmap='jet')
    ax[0, 1].imshow(layer_output[0, 36].detach().cpu(), cmap='jet')
    ax[0, 2].imshow(layer_output[0, 60].detach().cpu(), cmap='jet')
    ax[1, 0].imshow(layer_output[0, 84].detach().cpu(), cmap='jet')
    ax[1, 1].imshow(layer_output[0, 108].detach().cpu(), cmap='jet')
    ax[1, 2].imshow(x[0, 12].detach().cpu(), cmap='jet')
    plt.colorbar(im, orientation='vertical')
    plt.show()

def show_feature_map(layer_output, x, res):
    B, L, C = layer_output.shape
    fig, ax = plt.subplots(2, 3)
    layer_output = layer_output.permute(0, 2, 1).view(B, C, res, res)
    x = x.permute(0, 2, 1).view(B, C, res, res)
    im = ax[0, 0].imshow(layer_output[0, 12].detach().cpu(), cmap='jet')
    ax[0, 1].imshow(layer_output[0, 36].detach().cpu(), cmap='jet')
    ax[0, 2].imshow(layer_output[0, 60].detach().cpu(), cmap='jet')
    ax[1, 0].imshow(layer_output[0, 84].detach().cpu(), cmap='jet')
    ax[1, 1].imshow(layer_output[0, 108].detach().cpu(), cmap='jet')
    ax[1, 2].imshow(x[0, 12].detach().cpu(), cmap='jet')
    plt.colorbar(im, orientation='vertical')
    plt.show()


class Decoder(nn.Module):
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

    def __init__(self, 
                proj_qkv,
                reconstruction,
                sm_computation, 
                concat_spatial_cross_attention,
                conv_depth, 
                transformer_depth, 
                num_heads, 
                shortcut, 
                blur, 
                spatial_cross_attention_num_heads, 
                use_conv_mlp, 
                last_activation, 
                blur_kernel, 
                device, 
                dpr, 
                in_encoder_dims, 
                out_encoder_dims, 
                num_classes, 
                window_size, 
                img_size, 
                merge, 
                swin_abs_pos, 
                rpe_mode=None, 
                rpe_contextual_tensor='qkv',
                mlp_ratio=4., 
                qkv_bias=True, 
                qk_scale=None,
                drop_rate=0., 
                attn_drop_rate=0.,
                norm_layer=nn.LayerNorm,
                use_checkpoint=False, 
                deep_supervision=True, 
                **kwargs):
        super().__init__()

        if merge == 'linear':
            merge_layer = concat_merge_linear
        elif merge == 'rescale_linear':
            merge_layer = concat_merge_linear_rescale
        elif merge == 'rescale_conv':
            merge_layer = concat_merge_conv_rescale

        self.num_stages = len(conv_depth) + len(transformer_depth)
        self.mlp_ratio = mlp_ratio
        self.deep_supervision = deep_supervision
        self.img_size = img_size
        self.concat_spatial_cross_attention = concat_spatial_cross_attention
        self.reconstruction = reconstruction
        
        # build decoder layers
        self.spatial_cross_attention_layers = nn.ModuleList()
        self.layers = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()
        self.deep_supervision_layers = nn.ModuleList()
        self.spatial_skip_projs = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.cca_layers = nn.ModuleList()
        self.cat_cca_layers = nn.ModuleList()

        if self.reconstruction:
            self.sm_computation = sm_computation

        for i_layer in range(self.num_stages):
            input_resolution=(img_size//(2**(self.num_stages-i_layer-1)), img_size//(2**(self.num_stages-i_layer-1)))

            self.norm_layers.append(nn.LayerNorm(out_encoder_dims[i_layer]))
            spatial_cross_attention_layer = swin_cross_attention.BasicLayer(dim=out_encoder_dims[i_layer], #out_encoder_dims[i_layer],
                                                                            input_resolution=input_resolution,
                                                                            proj=proj_qkv,
                                                                            depth=2,
                                                                            use_conv_mlp=use_conv_mlp,
                                                                            num_heads=spatial_cross_attention_num_heads[i_layer],
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
                                                                            use_checkpoint=use_checkpoint)

            self.spatial_cross_attention_layers.append(spatial_cross_attention_layer)

            if concat_spatial_cross_attention:
                skip_proj_one_layer = nn.ModuleList()
                strides = torch.flip(2 ** torch.abs(torch.arange(-i_layer, self.num_stages - i_layer)), dims=[0])
                div = self.num_stages
                num = out_encoder_dims[i_layer] #out_encoder_dims[i_layer] # 120
                nb_channels = [num // div + (1 if x < num % div else 0)  for x in range (div)]
                for j, stride in enumerate(strides):
                    h_in = img_size // 2**j
                    if j < self.num_stages - i_layer:
                        proj =  nn.Conv2d(in_channels=out_encoder_dims[self.num_stages - j - 1], out_channels=nb_channels[j], kernel_size=stride.item(), stride=stride.item())
                        proj = nn.Sequential(To_image([h_in, h_in]), proj, nn.BatchNorm2d(nb_channels[j]), nn.GELU(), From_image())
                        #proj = nn.Sequential(To_image([h_in, h_in]), BlurLayer(blur_kernel=blur_kernel, stride=1), proj, nn.BatchNorm2d(nb_channels[j]), nn.GELU(), From_image())
                    else:
                        up = nn.Upsample(scale_factor=stride.item(), mode='bilinear')
                        proj =  nn.Conv2d(in_channels=out_encoder_dims[self.num_stages - j - 1], out_channels=nb_channels[j], kernel_size=3, stride=1, padding=1)
                        proj = nn.Sequential(To_image([h_in, h_in]), up, proj, nn.BatchNorm2d(nb_channels[j]), nn.GELU(), From_image())
                        #proj = nn.Sequential(To_image([h_in, h_in]), up, BlurLayer(blur_kernel=blur_kernel, stride=1), proj, nn.BatchNorm2d(nb_channels[j]), nn.GELU(), From_image())
                    skip_proj_one_layer.append(proj)
                self.spatial_skip_projs.append(skip_proj_one_layer)

                cat_cca_layer = CCA(dim=out_encoder_dims[i_layer])
                self.cat_cca_layers.append(nn.Sequential(To_image(input_resolution), cat_cca_layer, From_image()))
            else:
                self.spatial_skip_projs.append(None)
                self.cat_cca_layers.append(None)

            cca_layer = CCA(dim=out_encoder_dims[i_layer] * 2)
            self.cca_layers.append(nn.Sequential(To_image(input_resolution), cca_layer, From_image()))

            concat_merge = merge_layer(out_encoder_dims[i_layer] * 2, out_encoder_dims[i_layer])
            if i_layer < len(transformer_depth):
                ds_layer = DeepSupervision(input_resolution=input_resolution, dim=in_encoder_dims[i_layer], num_classes=num_classes, scale_factor=2**(self.num_stages - i_layer - 1)) if deep_supervision else nn.Identity()
                transformer_index = len(transformer_depth) - i_layer - 1
                upsample_layer = PatchExpandSwin(blur=blur, blur_kernel=blur_kernel, input_resolution=[x//2 for x in input_resolution], in_dim=out_encoder_dims[i_layer] * 2 if i_layer == 0 else in_encoder_dims[i_layer - 1], out_dim=out_encoder_dims[i_layer], swin_abs_pos=swin_abs_pos, device=device)
                layer_up = swin_transformer_2.BasicLayerUp(dim=in_encoder_dims[i_layer],
                                                        shortcut=shortcut,
                                                        input_resolution=input_resolution,
                                                        proj=proj_qkv,
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
                upsample_layer = PatchExpandConv(blur=blur, blur_kernel=blur_kernel, input_resolution=[x//2 for x in input_resolution], in_dim=in_encoder_dims[i_layer - 1], out_dim=out_encoder_dims[i_layer], swin_abs_pos=False, device=device)
                ds_layer = nn.Identity() if i_layer == self.num_stages - 1 else DeepSupervision(input_resolution=input_resolution, dim=in_encoder_dims[i_layer], num_classes=num_classes, scale_factor=2) if deep_supervision else nn.Identity()
                layer_up = ConvLayer(input_resolution=input_resolution, 
                                in_dim=out_encoder_dims[i_layer], 
                                out_dim=num_classes if i_layer == self.num_stages - 1 else in_encoder_dims[i_layer],
                                nb_se_blocks=conv_depth[i_layer - len(transformer_depth)], 
                                dpr=dpr[i_layer])
            self.layers.append(layer_up)
            self.concat_back_dim.append(concat_merge)
            self.upsample_layers.append(upsample_layer)
            self.deep_supervision_layers.append(ds_layer)

        self.norm = nn.LayerNorm(in_encoder_dims[0] * 2)
        #self.norm_up = norm_layer(self.embed_dim)
        if last_activation == 'sigmoid':
            self.last_activation = torch.nn.Sigmoid()
        elif last_activation == 'softmax':
            self.last_activation = torch.nn.Softmax(dim=1)
        elif last_activation == 'identity':
            self.last_activation = nn.Identity()

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
    
    #Dencoder and Skip connection
    #def forward_up_features(self, x, skip_connection):
    #    output_list = []
    #    x = self.first_up(x)
    #    for (layer_up, concat_back_dim, layer_output) in zip(self.layers_up, self.concat_back_dim, reversed(skip_connection)):
    #        x = torch.cat([x, layer_output], -1)
    #        x = concat_back_dim(x)
    #        x, out = layer_up(x)
    #        output_list.append(out)
#
    #    x = self.norm_up(x)  # B L C
#
    #    return x, 
    
    def forward(self, x, encoder_skip_connections):
        output_list = []
        similarity_matrix = None

        x = self.norm(x)
        for i, (layer_up, 
                concat_back_dim, 
                upsample_layer, 
                ds_layer, 
                encoder_skip_connection,
                spatial_cross_attention_layer,
                spatial_encoder_skip_proj,
                encoder_norm,
                cca_layer,
                cat_cca_layer
                ) in enumerate(zip(
                    self.layers, 
                    self.concat_back_dim, 
                    self.upsample_layers, 
                    self.deep_supervision_layers, 
                    reversed(encoder_skip_connections),
                    self.spatial_cross_attention_layers,
                    self.spatial_skip_projs,
                    self.norm_layers,
                    self.cca_layers,
                    self.cat_cca_layers)):
            x = upsample_layer(x)
            shortcut = encoder_skip_connection
            if self.concat_spatial_cross_attention:
                spatial_encoder_skip_connection_range = self.concat_skip_connections(encoder_skip_connections, spatial_encoder_skip_proj)
                spatial_encoder_skip_connection_range = cat_cca_layer(spatial_encoder_skip_connection_range)
                encoder_skip_connection = spatial_cross_attention_layer(encoder_skip_connection, spatial_encoder_skip_connection_range)
            else:
                encoder_skip_connection = spatial_cross_attention_layer(encoder_skip_connection, x)
            encoder_skip_connection = encoder_skip_connection + shortcut
            encoder_skip_connection = encoder_norm(encoder_skip_connection)
            x = torch.cat([encoder_skip_connection, x], dim=-1)
            x = cca_layer(x)
            x = concat_back_dim(x)
            x = layer_up(x)
            if i == self.num_stages - 1:
                B, L, C = x.shape
                x = x.permute(0, 2, 1).view(B, C, self.img_size, self.img_size)
                if self.reconstruction:
                    similarity_matrix = self.sm_computation(x)
                output_list.append(x)
            elif self.deep_supervision:
                output_list.append(ds_layer(x))
        
        output_list = [self.last_activation(x) for x in output_list if x is not None]

        #x = self.norm_up(x)  # B L C

        return output_list, similarity_matrix

    def concat_skip_connections(self, skip_connections, skip_projs):
        sp_list = []
        for skip_connection, proj in zip(skip_connections, skip_projs):
            skip_connection = proj(skip_connection)
            sp_list.append(skip_connection)
        return torch.cat(sp_list, dim=-1)


class SkipConnectionHandler(nn.Module):
    def __init__(self, 
                swin_abs_pos,
                num_stages,
                concat_spatial_cross_attention, 
                i_layer, 
                out_encoder_dims, 
                input_resolution,
                img_size, 
                proj_qkv, 
                use_conv_mlp, 
                spatial_cross_attention_num_heads,
                device,
                rpe_mode,
                rpe_contextual_tensor,
                window_size,
                reduction,
                dpr):
        super().__init__()

        self.reduction = reduction
        
        self.concat_spatial_cross_attention = concat_spatial_cross_attention
        self.input_resolution = input_resolution
        self.same_key_query = True if not concat_spatial_cross_attention else False

        self.spatial_cross_attention_layer = swin_cross_attention.BasicLayer(dim=out_encoder_dims[i_layer], #out_encoder_dims[i_layer],
                                                                        swin_abs_pos=swin_abs_pos,
                                                                        input_resolution=input_resolution,
                                                                        proj=proj_qkv,
                                                                        same_key_query=self.same_key_query,
                                                                        depth=2,
                                                                        use_conv_mlp=use_conv_mlp,
                                                                        num_heads=spatial_cross_attention_num_heads[i_layer],
                                                                        device=device,
                                                                        rpe_mode=rpe_mode,
                                                                        rpe_contextual_tensor=rpe_contextual_tensor,
                                                                        window_size=window_size,
                                                                        mlp_ratio=4.,
                                                                        qkv_bias=True, 
                                                                        qk_scale=None,
                                                                        drop=0., 
                                                                        attn_drop=0.,
                                                                        drop_path=dpr[i_layer],
                                                                        norm_layer=nn.LayerNorm,
                                                                        use_checkpoint=False)
        
        if concat_spatial_cross_attention:
            skip_proj_one_layer = nn.ModuleList()
            strides = torch.flip(2 ** torch.abs(torch.arange(-i_layer, num_stages - i_layer)), dims=[0])
            div = num_stages
            num = out_encoder_dims[i_layer] #out_encoder_dims[i_layer] # 120
            nb_channels = [num // div + (1 if x < num % div else 0)  for x in range (div)]
            for j, stride in enumerate(strides):
                h_in = img_size // 2**j
                if j < num_stages - i_layer:
                    proj =  nn.Conv2d(in_channels=out_encoder_dims[num_stages - j - 1], out_channels=nb_channels[j], kernel_size=stride.item(), stride=stride.item())
                    proj = nn.Sequential(To_image([h_in, h_in]), proj, nn.BatchNorm2d(nb_channels[j]), nn.GELU(), From_image())
                    #proj = nn.Sequential(To_image([h_in, h_in]), BlurLayer(blur_kernel=blur_kernel, stride=1), proj, nn.BatchNorm2d(nb_channels[j]), nn.GELU(), From_image())
                else:
                    up = nn.Upsample(scale_factor=stride.item(), mode='bilinear')
                    proj =  nn.Conv2d(in_channels=out_encoder_dims[num_stages - j - 1], out_channels=nb_channels[j], kernel_size=3, stride=1, padding=1)
                    proj = nn.Sequential(To_image([h_in, h_in]), up, proj, nn.BatchNorm2d(nb_channels[j]), nn.GELU(), From_image())
                    #proj = nn.Sequential(To_image([h_in, h_in]), up, BlurLayer(blur_kernel=blur_kernel, stride=1), proj, nn.BatchNorm2d(nb_channels[j]), nn.GELU(), From_image())
                skip_proj_one_layer.append(proj)
            self.spatial_skip_projs = skip_proj_one_layer

        self.cat_cca_layer = CCA(dim=out_encoder_dims[i_layer] * 2, input_resolution=input_resolution)
        if self.reduction:
            self.cca_layer = CCA(dim=out_encoder_dims[i_layer] * 2, input_resolution=input_resolution)
        #self.sigmoid = nn.Sigmoid()
        #self.layer_norm = nn.LayerNorm(out_encoder_dims[i_layer])
        #self.conv = nn.Conv2d(in_channels=out_encoder_dims[i_layer], out_channels=out_encoder_dims[i_layer], kernel_size=1)

    
    def forward(self, x, skip_connections, skip_connection):
        if self.concat_spatial_cross_attention:
            #spatial_skip_connection_range = self.concat_skip_connections(skip_connections, self.spatial_skip_projs)
            #ca_out = self.spatial_cross_attention_layer(skip_connection, spatial_skip_connection_range)
            #skip_connection = torch.cat([ca_out, skip_connection], dim=-1)
            #skip_connection = self.cat_cca_layer(skip_connection)

            spatial_skip_connection_range = self.concat_skip_connections(skip_connections, self.spatial_skip_projs)
            ca_out = self.spatial_cross_attention_layer(x, spatial_skip_connection_range)
            x = torch.cat([ca_out, x], dim=-1)
            x = self.cat_cca_layer(x)
        else:
            ca_out = self.spatial_cross_attention_layer(x, skip_connection)
            x = torch.cat([ca_out, x], dim=-1)
            x = self.cat_cca_layer(x)
        x = torch.cat([skip_connection, x], dim=-1)
        if self.reduction:
            x = self.cca_layer(x)
        return x
    
    def concat_skip_connections(self, skip_connections, skip_projs):
        sp_list = []
        for skip_connection, proj in zip(skip_connections, skip_projs):
            skip_connection = proj(skip_connection)
            sp_list.append(skip_connection)
        return torch.cat(sp_list, dim=-1)


class FullDecoder(nn.Module):
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

    def __init__(self,
                conv_depth,
                conv_layer,
                dpr,
                in_encoder_dims,
                out_encoder_dims,
                num_classes,
                img_size,
                norm,
                last_activation='identity',
                mlp_ratio=4.,
                deep_supervision=True, 
                **kwargs):
        super().__init__()

        self.num_stages = len(conv_depth)
        self.mlp_ratio = mlp_ratio
        self.deep_supervision = deep_supervision
        self.img_size = img_size
        self.num_classes = num_classes
        
        # build decoder layers
        self.layers = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()
        self.deep_supervision_layers = nn.ModuleList()

        for i_layer in range(self.num_stages):
            in_dim = out_encoder_dims[i_layer] * 2 if i_layer == 0 else in_encoder_dims[i_layer - 1]

            

            upsample_layer = PatchExpand(norm=norm, in_dim=in_dim, out_dim=out_encoder_dims[i_layer], swin_abs_pos=False)
            ds_layer = nn.Identity() if i_layer == self.num_stages - 1 else DeepSupervision(dim=in_encoder_dims[i_layer], num_classes=num_classes, scale_factor=2**(self.num_stages - i_layer - 1)) if deep_supervision else nn.Identity()

            layer_up = conv_layer(in_dim=out_encoder_dims[i_layer] * 2, 
                            out_dim=in_encoder_dims[i_layer],
                            nb_blocks=conv_depth[i_layer], 
                            kernel_size=3,
                            dpr=dpr[i_layer],
                            norm=norm)
            self.layers.append(layer_up)
            self.upsample_layers.append(upsample_layer)
            self.deep_supervision_layers.append(ds_layer)

        #self.norm = nn.LayerNorm(out_encoder_dims[0] * 2)
        #self.norm_up = norm_layer(self.embed_dim)
        if last_activation == 'sigmoid':
            self.last_activation = torch.nn.Sigmoid()
        elif last_activation == 'softmax':
            self.last_activation = torch.nn.Softmax(dim=1)
        elif last_activation == 'identity':
            self.last_activation = nn.Identity()

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
    

    def forward(self, x, encoder_skip_connections):
        output_list = []

        #x = self.norm(x)
        for i, (layer_up,
                upsample_layer, 
                ds_layer, 
                encoder_skip_connection
                ) in enumerate(zip(
                    self.layers,
                    self.upsample_layers, 
                    self.deep_supervision_layers, 
                    reversed(encoder_skip_connections)
                    )):
            x = upsample_layer(x)
            x = torch.cat((encoder_skip_connection, x), dim=1)
            x = layer_up(x)
            if i == self.num_stages - 1:
                output_list.append(self.last_activation(x))
            elif self.deep_supervision:
                output_list.append(self.last_activation(ds_layer(x)))

        output_list = output_list[::-1]

        return output_list[0]


class SegmentationDecoder(nn.Module):
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

    def __init__(self,
                proj_qkv,
                concat_spatial_cross_attention,
                conv_depth,
                spatial_cross_attention_num_heads,
                conv_layer,
                device, 
                dpr,
                in_encoder_dims, 
                out_encoder_dims,
                num_classes, 
                window_size, 
                filter_skip_co_segmentation,
                img_size,
                norm,
                last_activation='identity', 
                rpe_mode=None, 
                rpe_contextual_tensor='qkv',
                mlp_ratio=4., 
                qkv_bias=True, 
                qk_scale=None,
                drop_rate=0., 
                attn_drop_rate=0.,
                use_checkpoint=False, 
                deep_supervision=True, 
                **kwargs):
        super().__init__()

        self.num_stages = len(conv_depth)
        self.deep_supervision = deep_supervision
        self.img_size = img_size
        self.concat_spatial_cross_attention = concat_spatial_cross_attention
        self.num_classes = num_classes
        
        # build decoder layers
        self.layers = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()
        self.deep_supervision_layers = nn.ModuleList()
        self.encoder_skip_layers = nn.ModuleList()
        self.resizer_layers = nn.ModuleList()

        self.filter_skip_co_segmentation = filter_skip_co_segmentation

        for i_layer in range(self.num_stages):
            in_dim = out_encoder_dims[i_layer] * 2 if i_layer == 0 else in_encoder_dims[i_layer - 1]
            input_resolution=(img_size//(2**(self.num_stages-i_layer-1)), img_size//(2**(self.num_stages-i_layer-1)))

            if filter_skip_co_segmentation:
                encoder_skip_connection_handler = SwinFilterBlock(in_dim=out_encoder_dims[i_layer], 
                                                                out_dim=out_encoder_dims[i_layer],
                                                                input_resolution=input_resolution,
                                                                num_heads=spatial_cross_attention_num_heads[i_layer],
                                                                device=device,
                                                                rpe_mode=rpe_mode,
                                                                rpe_contextual_tensor=rpe_contextual_tensor,
                                                                window_size=window_size,
                                                                norm=norm,
                                                                depth=2)

                #encoder_skip_connection_handler = SwinFilterBlock(in_dim=out_encoder_dims[i_layer], 
                #                                                out_dim=out_encoder_dims[i_layer],
                #                                                input_resolution=input_resolution,
                #                                                num_heads=spatial_cross_attention_num_heads[i_layer],
                #                                                proj=proj_qkv,
                #                                                device=device,
                #                                                rpe_mode=rpe_mode,
                #                                                rpe_contextual_tensor=rpe_contextual_tensor,
                #                                                window_size=window_size,
                #                                                norm=norm,
                #                                                depth=2)
                

                self.encoder_skip_layers.append(encoder_skip_connection_handler)
            else:
                self.encoder_skip_layers.append(nn.Identity())

            self.resizer_layers.append(nn.Identity())
            upsample_layer = PatchExpand(norm=norm, in_dim=in_dim, out_dim=out_encoder_dims[i_layer], swin_abs_pos=False)
            ds_layer = nn.Identity() if i_layer == self.num_stages - 1 else DeepSupervision(dim=in_encoder_dims[i_layer], num_classes=num_classes, scale_factor=2**(self.num_stages - i_layer - 1)) if deep_supervision else nn.Identity()

            layer_up = conv_layer(in_dim=out_encoder_dims[i_layer] * 2, 
                            out_dim=in_encoder_dims[i_layer],
                            nb_blocks=conv_depth[i_layer], 
                            kernel_size=3,
                            dpr=dpr[i_layer],
                            norm=norm)
            self.layers.append(layer_up)
            self.upsample_layers.append(upsample_layer)
            self.deep_supervision_layers.append(ds_layer)

        #self.norm = nn.LayerNorm(out_encoder_dims[0] * 2)
        #self.norm_up = norm_layer(self.embed_dim)
        if last_activation == 'sigmoid':
            self.last_activation = torch.nn.Sigmoid()
        elif last_activation == 'softmax':
            self.last_activation = torch.nn.Softmax(dim=1)
        elif last_activation == 'identity':
            self.last_activation = nn.Identity()

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
    

    def forward(self, x, encoder_skip_connections):
        output_list = []

        #x = self.norm(x)
        for i, (layer_up,
                upsample_layer, 
                ds_layer, 
                encoder_skip_connection,
                encoder_skip_layer,
                resizer_layer
                ) in enumerate(zip(
                    self.layers,
                    self.upsample_layers, 
                    self.deep_supervision_layers, 
                    reversed(encoder_skip_connections),
                    self.encoder_skip_layers,
                    self.resizer_layers)):
            x = upsample_layer(x)
            #x = encoder_skip_layer(x, encoder_skip_connections, encoder_skip_connection)
            if encoder_skip_connection is not None:
                if self.filter_skip_co_segmentation:
                    encoder_skip_connection = encoder_skip_layer(x, encoder_skip_connection)
                x = torch.cat((encoder_skip_connection, x), dim=1)
            x = resizer_layer(x)
            x = layer_up(x)
            if i == self.num_stages - 1:
                #if self.directional_field:
                #    df = self.df_conv(x)
                #    df = torch.tanh(df)
                #    refined = self.refinement_module(x, torch.clone(df))
                #    final_seg = self.final_refined_conv(refined)
                #    initial_seg = self.last_conv(x)
                #    x = self.last_conv_layer(x)
                #    if self.reconstruction:
                #        similarity_matrix = self.sm_computation(x)
                #    output_list.append(initial_seg)
                #    output_list.append(final_seg)
                #else:
                #if self.similarity:
                #    similarity_matrix = self.sm_computation(x)
                    #if self.start_reconstruction_dim > self.num_classes:
                    #    x = self.last_conv(x)
                output_list.append(self.last_activation(x))
            elif self.deep_supervision:
                output_list.append(self.last_activation(ds_layer(x)))

        #x = self.norm_up(x)  # B L C

        output_list = output_list[::-1]

        return output_list


class VideoSegmentationDecoder(nn.Module):
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

    def __init__(self,
                proj_qkv,
                concat_spatial_cross_attention,
                conv_depth,
                spatial_cross_attention_num_heads,
                conv_layer,
                device, 
                dpr,
                in_encoder_dims, 
                out_encoder_dims,
                num_classes, 
                window_size, 
                filter_skip_co_segmentation,
                img_size,
                area_size,
                n_points,
                video_length,
                norm,
                nb_zones,
                last_activation='identity', 
                rpe_mode=None, 
                rpe_contextual_tensor='qkv',
                mlp_ratio=4., 
                qkv_bias=True, 
                qk_scale=None,
                drop_rate=0., 
                attn_drop_rate=0.,
                use_checkpoint=False, 
                deep_supervision=True, 
                **kwargs):
        super().__init__()

        self.num_stages = len(conv_depth)
        self.deep_supervision = deep_supervision
        self.img_size = img_size
        self.concat_spatial_cross_attention = concat_spatial_cross_attention
        self.num_classes = num_classes
        self.area_size = area_size
        
        # build decoder layers
        self.layers = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()
        self.deep_supervision_layers = nn.ModuleList()
        self.encoder_skip_layers = nn.ModuleList()
        self.resizer_layers = nn.ModuleList()

        self.filter_skip_co_segmentation = filter_skip_co_segmentation

        for i_layer in range(self.num_stages):
            in_dim = out_encoder_dims[i_layer] * 2 if i_layer == 0 else in_encoder_dims[i_layer - 1]
            input_resolution=(img_size//(2**(self.num_stages-i_layer-1)), img_size//(2**(self.num_stages-i_layer-1)))

            if filter_skip_co_segmentation:
                encoder_skip_connection_handler = SwinFilterBlock(in_dim=out_encoder_dims[i_layer], 
                                                                out_dim=out_encoder_dims[i_layer],
                                                                input_resolution=input_resolution,
                                                                num_heads=spatial_cross_attention_num_heads[i_layer],
                                                                proj=proj_qkv,
                                                                device=device,
                                                                rpe_mode=rpe_mode,
                                                                rpe_contextual_tensor=rpe_contextual_tensor,
                                                                window_size=window_size,
                                                                norm=norm,
                                                                depth=2)
                

                self.encoder_skip_layers.append(encoder_skip_connection_handler)
            else:
                self.encoder_skip_layers.append(nn.Identity())

            self.resizer_layers.append(nn.Identity())
            upsample_layer = PatchExpand(norm=norm, in_dim=in_dim, out_dim=out_encoder_dims[i_layer], swin_abs_pos=False)
            ds_layer = nn.Identity() if i_layer == self.num_stages - 1 else DeepSupervision(dim=in_encoder_dims[i_layer], num_classes=num_classes, scale_factor=2**(self.num_stages - i_layer - 1)) if deep_supervision else nn.Identity()

            #out_dim = out_encoder_dims[i_layer] * 2 if i_layer == self.num_stages - 1 else in_encoder_dims[i_layer]
            out_dim = out_encoder_dims[i_layer] if i_layer == self.num_stages - 1 else in_encoder_dims[i_layer]
            layer_up = conv_layer(in_dim=out_encoder_dims[i_layer] * 2, 
                            out_dim=out_dim,
                            nb_blocks=conv_depth[i_layer], 
                            kernel_size=3,
                            dpr=dpr[i_layer],
                            norm=norm)
            self.layers.append(layer_up)
            self.upsample_layers.append(upsample_layer)
            self.deep_supervision_layers.append(ds_layer)

        #self.norm = nn.LayerNorm(out_encoder_dims[0] * 2)
        #self.norm_up = norm_layer(self.embed_dim)
        if last_activation == 'sigmoid':
            self.last_activation = torch.nn.Sigmoid()
        elif last_activation == 'softmax':
            self.last_activation = torch.nn.Softmax(dim=1)
        elif last_activation == 'identity':
            self.last_activation = nn.Identity()

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
    
    def get_indices_softmax(self, softmax_volume, stage):
        "softmax_volume: B, T, C, H, W"
        B, T, C, H, W = softmax_volume.shape
        k = int((H / (2**self.num_stages)) ** 2)
        scale_factor = 1/2**(self.num_stages - 1 - stage)
        temp_blurred = 1 - torch.max(softmax_volume, dim=2)[0]

        #kernel_size = self.gaussian_kernel_size
        #kernel_size = torch.sigmoid(kernel_size)
        #kernel_size = 2 * floor((H * kernel_size[stage].item()) / 2) + 1

        temp_blurred = gaussian_blur(temp_blurred, kernel_size=[19, 19])
        temp_blurred = interpolate(temp_blurred, scale_factor=(scale_factor, scale_factor), mode='bilinear', antialias=True)
        B, T, H, W = temp_blurred.shape

        #matplotlib.use('QtAgg')
        #fig, ax = plt.subplots(1, 1)
        #ax.imshow(temp_blurred[0, 0].cpu(), cmap='plasma')
        #plt.show()

        temp_blurred_flattened = torch.flatten(temp_blurred, start_dim=-2)
        values, indices = torch.topk(temp_blurred_flattened, k=k, dim=-1, largest=True)
        indices = indices.permute(0, 2, 1)
            
        #matplotlib.use('QtAgg')
        #fig, ax = plt.subplots(1, 3)
        #ax[0].imshow(scale_list[0][0, 0, 0].cpu(), cmap='plasma')
        #ax[1].imshow(scale_list[1][0, 0, 0].cpu(), cmap='plasma')
        #ax[2].imshow(scale_list[2][0, 0, 0].cpu(), cmap='plasma')
        #plt.show()

        return indices
    
    def get_indices_learn(self, x, decoder_map_getter):
        "x: B, C, H, W"
        k = int((self.img_size / 2**self.num_stages)**2)
        decoder_map = decoder_map_getter(x)
        decoder_map = torch.sigmoid(decoder_map)

        #matplotlib.use('QtAgg')
        #fig, ax = plt.subplots(1, 1)
        #ax.imshow(decoder_map[0, 0].detach().cpu(), cmap='plasma')
        #plt.show()

        decoder_map_flattened = torch.flatten(decoder_map, start_dim=-2)
        values, indices = torch.topk(decoder_map_flattened, k=k, dim=-1, largest=True)
        indices = indices.permute(0, 2, 1)
            
        #matplotlib.use('QtAgg')
        #fig, ax = plt.subplots(1, 3)
        #ax[0].imshow(scale_list[0][0, 0, 0].cpu(), cmap='plasma')
        #ax[1].imshow(scale_list[1][0, 0, 0].cpu(), cmap='plasma')
        #ax[2].imshow(scale_list[2][0, 0, 0].cpu(), cmap='plasma')
        #plt.show()

        return indices
    

    def forward(self, x, encoder_skip_connections, memory_bus=None, temporal_pos=None, frame_index=None):
        output_list = []
        sampling_points = attention_weights = theta_coords = None

        #x = self.norm(x)
        for i, (layer_up,
                upsample_layer, 
                ds_layer, 
                encoder_skip_connection,
                encoder_skip_layer,
                resizer_layer
                ) in enumerate(zip(
                    self.layers,
                    self.upsample_layers, 
                    self.deep_supervision_layers, 
                    reversed(encoder_skip_connections),
                    self.encoder_skip_layers,
                    self.resizer_layers)):
            x = upsample_layer(x)
            #x = encoder_skip_layer(x, encoder_skip_connections, encoder_skip_connection)
            if encoder_skip_connection is not None:
                #if self.filter_skip_co_segmentation:
                #    encoder_skip_connection = encoder_skip_layer(x, encoder_skip_connection)
                #elif self.learn_indices:
                #    encoder_skip_connection, sampling_points, attention_weights, theta_coords = deformable_attention_layer(key=encoder_skip_connection, query=x, frame_idx=frame_index, temporal_pos=temporal_pos, memory_bus=memory_bus)
                x = torch.cat((encoder_skip_connection, x), dim=1)
            x = resizer_layer(x)
            x = layer_up(x)
            output_list.append(x)

        return output_list, sampling_points, attention_weights, theta_coords
        

class SimpleDecoder(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.reduce = nn.ConvTranspose2d(in_channels=in_dim, out_channels=out_dim, kernel_size=8, stride=8)
        #self.reduce1 = nn.ConvTranspose2d(in_channels=dim, out_channels=dim//2, kernel_size=2, stride=2)
        #self.reduce2 = nn.ConvTranspose2d(in_channels=dim//2, out_channels=dim//4, kernel_size=2, stride=2)
        #self.reduce3 = nn.ConvTranspose2d(in_channels=dim//4, out_channels=1, kernel_size=2, stride=2)
        #self.up = torch.nn.Upsample(size=image_size, mode='bilinear')

    def forward(self, x, skip_connections=None):
        #x = self.reduce1(x)
        #x = self.reduce2(x)
        #x = self.reduce3(x)
        x = self.reduce(x)
        #x = self.up(x)
        return [x]

class MotionEstimation(nn.Module):
    def __init__(self):
        super().__init__()
        self.tanh = torch.nn.Tanh()
    
    def generate_grid(self, x, offset):
        x_shape = x.size()
        grid_w, grid_h = torch.meshgrid([torch.linspace(-1, 1, x_shape[2]), torch.linspace(-1, 1, x_shape[3])])  # (h, w)
        grid_w = grid_w.cuda().float()
        grid_h = grid_h.cuda().float()

        grid_w = nn.Parameter(grid_w, requires_grad=False)
        grid_h = nn.Parameter(grid_h, requires_grad=False)

        offset_h, offset_w = torch.split(offset, 1, 1)
        offset_w = offset_w.contiguous().view(-1, int(x_shape[2]), int(x_shape[3]))  # (b*c, h, w)
        offset_h = offset_h.contiguous().view(-1, int(x_shape[2]), int(x_shape[3]))  # (b*c, h, w)

        offset_w = grid_w + offset_w
        offset_h = grid_h + offset_h

        offsets = torch.stack((offset_h, offset_w), 3)
        return offsets

    def forward(self, x, original):
        x = self.tanh(x)
        grid = self.generate_grid(x=original, offset=x)
        return grid_sample(original, grid)

class SimpleDecoderStages(nn.Module):
    def __init__(self, 
                in_dim, 
                out_dim, 
                nb_stages,
                norm,
                conv_layer,
                conv_depth,
                dpr,
                deep_supervision):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.nb_stages = nb_stages
        self.layers = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()
        self.deep_supervision_layers = nn.ModuleList()
        self.encoder_skip_layers = nn.ModuleList()

        for i in range(nb_stages):
            current_in_dim = in_dim // 2**i
            current_out_dim = current_in_dim // 2
            ds_layer = nn.Identity() if i == nb_stages - 1 else DeepSupervision(dim=current_out_dim, num_classes=out_dim, scale_factor=2**(nb_stages - i - 1)) if deep_supervision else nn.Identity()
            upsample_layer = PatchExpand(norm=norm, in_dim=current_in_dim, out_dim=current_out_dim, swin_abs_pos=False)
            conv_out_dim = out_dim if i == nb_stages - 1 else current_out_dim
            layer_up = conv_layer(in_dim=current_out_dim, 
                                out_dim=conv_out_dim,
                                nb_blocks=conv_depth[i], 
                                kernel_size=3,
                                dpr=dpr[i],
                                norm=norm)
            self.layers.append(layer_up)
            self.upsample_layers.append(upsample_layer)
            self.deep_supervision_layers.append(ds_layer)

    def forward(self, x):
        output_list = []
        for i, (layer_up,
                upsample_layer, 
                ds_layer,
                ) in enumerate(zip(
                    self.layers,
                    self.upsample_layers, 
                    self.deep_supervision_layers)):
            x = upsample_layer(x)
            x = layer_up(x)
            if i == self.nb_stages - 1:
                output_list.append(x)
            elif self.deep_supervision:
                output_list.append(ds_layer(x))

        output_list = output_list[::-1]
        return output_list

class TransformerVideoDecoder(nn.Module):
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

    def __init__(self,
                dim,
                num_heads,
                deformable_points,
                video_length,
                num_stages):
        super().__init__()
        
        # build decoder layers
        self.transformer_blocks = nn.ModuleList()

        for i_layer in range(num_stages):
            d_ffn = min(2048, 4 * dim)
            transformer_block = TransformerDecoderBlock(dim=dim, num_heads=num_heads, n_points=deformable_points, video_length=video_length, d_ffn=d_ffn)
            self.transformer_blocks.append(transformer_block)

    def forward(self, memory_bus, skip_connection_list, advanced_pos):

        for i, (transformer_block, skip_connection) in enumerate(zip(self.transformer_blocks, skip_connection_list)):
            memory_bus = transformer_block(skip_connection=skip_connection, x=memory_bus, advanced_pos=advanced_pos)

        return memory_bus


def show_feature_map(layer_output, x, res):
    B, L, C = layer_output.shape
    fig, ax = plt.subplots(2, 3)
    layer_output = layer_output.permute(0, 2, 1).view(B, C, res, res)
    x = x.permute(0, 2, 1).view(B, 24, res, res)
    im = ax[0, 0].imshow(layer_output[0, 12].detach().cpu(), cmap='jet')
    ax[0, 1].imshow(layer_output[0, 36].detach().cpu(), cmap='jet')
    ax[0, 2].imshow(layer_output[0, 60].detach().cpu(), cmap='jet')
    ax[1, 0].imshow(layer_output[0, 84].detach().cpu(), cmap='jet')
    ax[1, 1].imshow(layer_output[0, 108].detach().cpu(), cmap='jet')
    ax[1, 2].imshow(x[0, 12].detach().cpu(), cmap='jet')
    plt.colorbar(im, orientation='vertical')
    plt.show()

def show_feature_map(layer_output, x, res):
    B, L, C = layer_output.shape
    fig, ax = plt.subplots(2, 3)
    layer_output = layer_output.permute(0, 2, 1).view(B, C, res, res)
    x = x.permute(0, 2, 1).view(B, C, res, res)
    im = ax[0, 0].imshow(layer_output[0, 12].detach().cpu(), cmap='jet')
    ax[0, 1].imshow(layer_output[0, 36].detach().cpu(), cmap='jet')
    ax[0, 2].imshow(layer_output[0, 60].detach().cpu(), cmap='jet')
    ax[1, 0].imshow(layer_output[0, 84].detach().cpu(), cmap='jet')
    ax[1, 1].imshow(layer_output[0, 108].detach().cpu(), cmap='jet')
    ax[1, 2].imshow(x[0, 12].detach().cpu(), cmap='jet')
    plt.colorbar(im, orientation='vertical')
    plt.show()