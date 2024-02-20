from locale import normalize
import torch
import torch.nn as nn

from timm.models.layers import trunc_normal_
from mmcv.runner import load_checkpoint
#import swin_transformer_3d
from .utils import ConvBlocks2DBatch, PatchMerging3DGroup, ConvBlocks3DGroupLegacy, ConvBlocks2DGroupLegacy, ConvBlocks2DGroup, ConvBlocks3DPos, ConvBlocksLegacy, PatchMerging2DBatch, PatchMerging2DGroup, ConvBlocks3D, PatchMerging3D2D, ConvBlocks3D2D, PatchMerging3D, get_root_logger, CCA, ConvLayer3D, PatchMergingLegacy, ResnetConvLayer
from einops import rearrange
from .swin_transformer_2 import BasicLayer
#from vit_transformer import TransformerEncoderLayer, VitBasicLayer
import copy
from .position_embedding import PositionEmbeddingSine2d
from einops.layers.torch import Rearrange
import sys
import matplotlib.pyplot as plt
from . import swin_cross_attention

class FusionModuleConv(nn.Module):
    def __init__(self, 
                input_resolution, 
                in_dim, 
                out_dim, 
                nb_blocks, 
                dpr, 
                proj_qkv, 
                use_conv_mlp, 
                ca_head_nb, 
                device, 
                rpe_mode, 
                rpe_contextual_tensor, 
                window_size):
        super().__init__()
        self.layer1 = ConvLayer(input_resolution=input_resolution, 
                            in_dim=in_dim,
                            out_dim=out_dim,
                            nb_se_blocks=nb_blocks, 
                            dpr=dpr)
        
        self.layer2 = ConvLayer(input_resolution=input_resolution, 
                            in_dim=in_dim,
                            out_dim=out_dim,
                            nb_se_blocks=nb_blocks, 
                            dpr=dpr)
        
        self.ca1 = swin_cross_attention.BasicLayer(dim=out_dim,
                                                input_resolution=input_resolution,
                                                proj=proj_qkv,
                                                depth=2,
                                                use_conv_mlp=use_conv_mlp,
                                                num_heads=ca_head_nb,
                                                device=device,
                                                rpe_mode=rpe_mode,
                                                rpe_contextual_tensor=rpe_contextual_tensor,
                                                window_size=window_size,
                                                mlp_ratio=4.,
                                                qkv_bias=True, 
                                                qk_scale=None,
                                                drop=0., 
                                                attn_drop=0.,
                                                drop_path=dpr,
                                                norm_layer=nn.LayerNorm,
                                                use_checkpoint=False)
        
        self.ca2 = swin_cross_attention.BasicLayer(dim=out_dim,
                                                input_resolution=input_resolution,
                                                proj=proj_qkv,
                                                depth=2,
                                                use_conv_mlp=use_conv_mlp,
                                                num_heads=ca_head_nb,
                                                device=device,
                                                rpe_mode=rpe_mode,
                                                rpe_contextual_tensor=rpe_contextual_tensor,
                                                window_size=window_size,
                                                mlp_ratio=4.,
                                                qkv_bias=True, 
                                                qk_scale=None,
                                                drop=0., 
                                                attn_drop=0.,
                                                drop_path=dpr,
                                                norm_layer=nn.LayerNorm,
                                                use_checkpoint=False)
        
        self.cca_layer1 = nn.Sequential(CCA(dim=out_dim * 2, input_resolution=input_resolution))
        self.cca_layer2 = nn.Sequential(CCA(dim=out_dim * 2, input_resolution=input_resolution))
    
        self.last_cca_layer = nn.Sequential(CCA(dim=out_dim * 2, input_resolution=input_resolution))

    def forward(self, x1, x2):
        x1 = self.layer1(x1)
        x2 = self.layer2(x2)

        x1_ca = self.ca1(x1, x2)
        x2_ca = self.ca2(x2, x1)

        x1 = torch.cat([x1, x1_ca], dim=-1)
        x2 = torch.cat([x2, x2_ca], dim=-1)

        x1 = self.cca_layer1(x1)
        x2 = self.cca_layer2(x2)

        skip_connection = torch.cat([x1, x2], dim=-1)
        skip_connection = self.last_cca_layer(skip_connection)

        return x1, x2, skip_connection


class FusionModuleSwin(nn.Module):
    def __init__(self, 
                input_resolution, 
                in_dim, 
                shortcut, 
                depth,
                dpr, 
                proj_qkv, 
                use_conv_mlp, 
                head_nb, 
                ca_head_nb, 
                device, 
                rpe_mode, 
                rpe_contextual_tensor, 
                window_size):
        super().__init__()
        self.layer1 = BasicLayer(dim=in_dim,
                                proj=proj_qkv,
                                shortcut=shortcut,
                                input_resolution=input_resolution,
                                depth=depth,
                                num_heads=head_nb,
                                device=device,
                                use_conv_mlp=use_conv_mlp,
                                rpe_mode=rpe_mode,
                                rpe_contextual_tensor=rpe_contextual_tensor,
                                window_size=window_size,
                                mlp_ratio=4.,
                                qkv_bias=True, 
                                qk_scale=None,
                                drop=0., 
                                attn_drop=0.,
                                drop_path=dpr,
                                norm_layer=nn.LayerNorm,
                                use_checkpoint=False)
        
        self.layer2 = BasicLayer(dim=in_dim,
                                proj=proj_qkv,
                                shortcut=shortcut,
                                input_resolution=input_resolution,
                                depth=depth,
                                num_heads=head_nb,
                                device=device,
                                use_conv_mlp=use_conv_mlp,
                                rpe_mode=rpe_mode,
                                rpe_contextual_tensor=rpe_contextual_tensor,
                                window_size=window_size,
                                mlp_ratio=4.,
                                qkv_bias=True, 
                                qk_scale=None,
                                drop=0., 
                                attn_drop=0.,
                                drop_path=dpr,
                                norm_layer=nn.LayerNorm,
                                use_checkpoint=False)
        
        self.ca1 = swin_cross_attention.BasicLayer(dim=in_dim,
                                                input_resolution=input_resolution,
                                                proj=proj_qkv,
                                                depth=2,
                                                use_conv_mlp=use_conv_mlp,
                                                num_heads=ca_head_nb,
                                                device=device,
                                                rpe_mode=rpe_mode,
                                                rpe_contextual_tensor=rpe_contextual_tensor,
                                                window_size=window_size,
                                                mlp_ratio=4.,
                                                qkv_bias=True, 
                                                qk_scale=None,
                                                drop=0., 
                                                attn_drop=0.,
                                                drop_path=dpr,
                                                norm_layer=nn.LayerNorm,
                                                use_checkpoint=False)
        
        self.ca2 = swin_cross_attention.BasicLayer(dim=in_dim,
                                                input_resolution=input_resolution,
                                                proj=proj_qkv,
                                                depth=2,
                                                use_conv_mlp=use_conv_mlp,
                                                num_heads=ca_head_nb,
                                                device=device,
                                                rpe_mode=rpe_mode,
                                                rpe_contextual_tensor=rpe_contextual_tensor,
                                                window_size=window_size,
                                                mlp_ratio=4.,
                                                qkv_bias=True, 
                                                qk_scale=None,
                                                drop=0., 
                                                attn_drop=0.,
                                                drop_path=dpr,
                                                norm_layer=nn.LayerNorm,
                                                use_checkpoint=False)
        
        self.cca_layer1 = nn.Sequential(CCA(dim=in_dim * 2, input_resolution=input_resolution))
        self.cca_layer2 = nn.Sequential(CCA(dim=in_dim * 2, input_resolution=input_resolution))
    
        self.last_cca_layer = nn.Sequential(CCA(dim=in_dim * 2, input_resolution=input_resolution))

    def forward(self, x1, x2):
        x1 = self.layer1(x1)
        x2 = self.layer2(x2)

        x1_ca = self.ca1(x1, x2)
        x2_ca = self.ca2(x2, x1)

        x1 = torch.cat([x1, x1_ca], dim=-1)
        x2 = torch.cat([x2, x2_ca], dim=-1)

        x1 = self.cca_layer1(x1)
        x2 = self.cca_layer2(x2)

        skip_connection = torch.cat([x1, x2], dim=-1)
        skip_connection = self.last_cca_layer(skip_connection)

        return x1, x2, skip_connection


class FusionEncoder(nn.Module):
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
                blur, 
                proj, 
                shortcut, 
                use_conv_mlp, 
                ca_head_numbers, 
                conv_depth, 
                transformer_depth, 
                num_heads, 
                blur_kernel, 
                device, 
                dpr, 
                in_dims, 
                out_dims, 
                swin_abs_pos, 
                window_size, 
                img_size, 
                rpe_mode=None, 
                rpe_contextual_tensor='qkv'):
        super().__init__()

        self.num_stages = len(conv_depth) + len(transformer_depth)

        self.pos_drop = nn.Dropout(p=0.)

        # build encoder layers
        self.layers = nn.ModuleList()
        self.downsample_layers1 = nn.ModuleList()
        self.downsample_layers2 = nn.ModuleList()
        for i_layer in range(self.num_stages):
            input_resolution=(img_size//(2**i_layer), img_size//(2**i_layer))
            if i_layer < len(conv_depth):
                layer = FusionModuleConv(input_resolution=input_resolution, 
                                        in_dim=in_dims[i_layer],
                                        out_dim=out_dims[i_layer],
                                        nb_blocks=conv_depth[i_layer],
                                        dpr=dpr[i_layer],
                                        proj_qkv=proj,
                                        use_conv_mlp=use_conv_mlp,
                                        ca_head_nb=ca_head_numbers[i_layer],
                                        device=device,
                                        rpe_mode=rpe_mode,
                                        rpe_contextual_tensor=rpe_contextual_tensor,
                                        window_size=window_size)
                downsample_layer1 = PatchMergingConv(blur=blur, blur_kernel=blur_kernel, input_resolution=input_resolution, in_dim=out_dims[i_layer], out_dim=in_dims[i_layer+1], swin_abs_pos=swin_abs_pos if i_layer == len(conv_depth) - 1 else False, device=device)
                downsample_layer2 = PatchMergingConv(blur=blur, blur_kernel=blur_kernel, input_resolution=input_resolution, in_dim=out_dims[i_layer], out_dim=in_dims[i_layer+1], swin_abs_pos=swin_abs_pos if i_layer == len(conv_depth) - 1 else False, device=device)
            else:
                downsample_layer1 = PatchMergingSwin(blur=blur, blur_kernel=blur_kernel, input_resolution=input_resolution, in_dim=out_dims[i_layer], out_dim=2*out_dims[i_layer] if i_layer == self.num_stages - 1 else in_dims[i_layer+1], swin_abs_pos=swin_abs_pos, device=device)
                downsample_layer2 = PatchMergingSwin(blur=blur, blur_kernel=blur_kernel, input_resolution=input_resolution, in_dim=out_dims[i_layer], out_dim=2*out_dims[i_layer] if i_layer == self.num_stages - 1 else in_dims[i_layer+1], swin_abs_pos=swin_abs_pos, device=device)
                transformer_index = i_layer-len(conv_depth)
                if i_layer == self.num_stages - 1:
                    layer = FusionModuleSwin(input_resolution=input_resolution,
                                            in_dim=in_dims[i_layer],
                                            shortcut=shortcut,
                                            depth=transformer_depth[transformer_index],
                                            dpr=dpr[i_layer],
                                            proj_qkv=proj,
                                            use_conv_mlp=use_conv_mlp,
                                            head_nb=num_heads[transformer_index],
                                            ca_head_nb=ca_head_numbers[i_layer],
                                            device=device,
                                            rpe_mode=rpe_mode,
                                            rpe_contextual_tensor=rpe_contextual_tensor,
                                            window_size=window_size)
            self.layers.append(layer)
            self.downsample_layers1.append(downsample_layer1)
            self.downsample_layers2.append(downsample_layer2)

        #self.norm = norm_layer(self.num_features)
        #self.norm_after_conv = norm_layer(embed_dim)

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

    def forward(self, x1, x2):
        skip_connection_list = []
        B, C, H, W = x1.shape
        x1 = x1.permute(0, 2, 3, 1).view(B, -1, C)
        x2 = x2.permute(0, 2, 3, 1).view(B, -1, C)

        for layer, downsample_layer1, downsample_layer2 in zip(self.layers, self.downsample_layers1, self.downsample_layers2):
            x1, x2, skip_connection = layer(x1, x2)
            skip_connection_list.append(skip_connection)
            x1 = downsample_layer1(x1)
            x2 = downsample_layer2(x2)
        
        return x1, x2, skip_connection_list

class Encoder(nn.Module):
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

    def __init__(self, conv_layer, norm, conv_depth, device, dpr, in_dims, out_dims):
        super().__init__()

        self.num_stages = len(conv_depth)

        # build encoder layers
        self.layers = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        for i_layer in range(self.num_stages):
            out_dim = 2*out_dims[i_layer] if i_layer == self.num_stages - 1 else in_dims[i_layer+1]
            layer = ConvBlocksLegacy(in_dim=in_dims[i_layer],
                                kernel_size=3,
                                out_dim=out_dims[i_layer],
                                nb_blocks=conv_depth[i_layer], 
                                dpr=dpr[i_layer],
                                norm=norm)
                #layer = ConvBlocks(in_dim=in_dims[i_layer], out_dim=out_dims[i_layer], nb_block=conv_depth[i_layer])
                                
            downsample_layer = PatchMergingLegacy(norm=norm, in_dim=out_dims[i_layer], out_dim=out_dim, device=device)
            self.layers.append(layer)
            self.downsample_layers.append(downsample_layer)

        #self.norm = norm_layer(self.num_features)
        #self.norm_after_conv = norm_layer(embed_dim)

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
        skip_connections = []

        for layer, downsample_layer in zip(self.layers, self.downsample_layers):
            x = layer(x)
            skip_connections.append(x)
            x = downsample_layer(x)
        
        return x, skip_connections
    



class Encoder3D(nn.Module):
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

    def __init__(self, conv_depth, in_dims, out_dims, kernel_size):
        super().__init__()

        self.num_stages = len(conv_depth)

        # build encoder layers
        self.layers = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        for i_layer in range(self.num_stages):
            out_dim = 2*out_dims[i_layer] if i_layer == self.num_stages - 1 else in_dims[i_layer+1]
            layer = ConvBlocks3DGroupLegacy(in_dim=in_dims[i_layer],
                                kernel_size=kernel_size,
                                out_dim=out_dims[i_layer],
                                nb_blocks=conv_depth[i_layer])
                                    
            downsample_layer = PatchMerging3DGroup(in_dim=out_dims[i_layer], out_dim=out_dim)
            self.layers.append(layer)
            self.downsample_layers.append(downsample_layer)

        #self.norm = norm_layer(self.num_features)
        #self.norm_after_conv = norm_layer(embed_dim)

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
        skip_connections = []

        for layer, downsample_layer in zip(self.layers, self.downsample_layers):
            x = layer(x)
            skip_connections.append(x)
            x = downsample_layer(x)
        
        return x, skip_connections
    



class Encoder2D(nn.Module):
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

    def __init__(self, conv_depth, in_dims, out_dims, norm, legacy, nb_conv):
        super().__init__()

        self.num_stages = len(conv_depth)

        # build encoder layers
        self.layers = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        for i_layer in range(self.num_stages):
            out_dim = 2*out_dims[i_layer] if i_layer == self.num_stages - 1 else in_dims[i_layer+1]
            if norm == 'group':
                if legacy:
                    layer = ConvBlocks2DGroupLegacy(in_dim=in_dims[i_layer],
                                        kernel_size=3,
                                        d_model=None,
                                        out_dim=out_dims[i_layer],
                                        nb_blocks=conv_depth[i_layer],
                                        nb_conv=nb_conv)
                else:
                    layer = ConvBlocks2DGroup(in_dim=in_dims[i_layer],
                                        kernel_size=3,
                                        out_dim=out_dims[i_layer],
                                        nb_blocks=conv_depth[i_layer])
                                    
                downsample_layer = PatchMerging2DGroup(in_dim=out_dims[i_layer], out_dim=out_dim)
            elif norm == 'batch':
                layer = ConvBlocks2DBatch(in_dim=in_dims[i_layer],
                                        kernel_size=3,
                                        out_dim=out_dims[i_layer],
                                        nb_blocks=conv_depth[i_layer])
                                    
                downsample_layer = PatchMerging2DBatch(in_dim=out_dims[i_layer], out_dim=out_dim)
            self.layers.append(layer)
            self.downsample_layers.append(downsample_layer)

        #self.norm = norm_layer(self.num_features)
        #self.norm_after_conv = norm_layer(embed_dim)

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
        skip_connections = []

        for layer, downsample_layer in zip(self.layers, self.downsample_layers):
            x = layer(x)
            skip_connections.append(x)
            x = downsample_layer(x)
        
        return x, skip_connections
    

class Encoder3DPos(nn.Module):
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

    def __init__(self, conv_depth, in_dims, out_dims, embedding_dim):
        super().__init__()

        self.num_stages = len(conv_depth)

        # build encoder layers
        self.layers = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        for i_layer in range(self.num_stages):
            out_dim = 2*out_dims[i_layer] if i_layer == self.num_stages - 1 else in_dims[i_layer+1]
            layer = ConvBlocks3DPos(in_dim=in_dims[i_layer],
                                kernel_size=3,
                                out_dim=out_dims[i_layer],
                                nb_blocks=conv_depth[i_layer],
                                embedding_dim=embedding_dim)
                #layer = ConvBlocks(in_dim=in_dims[i_layer], out_dim=out_dims[i_layer], nb_block=conv_depth[i_layer])
                                
            downsample_layer = PatchMerging3D(in_dim=out_dims[i_layer], out_dim=out_dim)
            self.layers.append(layer)
            self.downsample_layers.append(downsample_layer)

        #self.norm = norm_layer(self.num_features)
        #self.norm_after_conv = norm_layer(embed_dim)

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

    def forward(self, x, embedding):
        skip_connections = []

        for layer, downsample_layer in zip(self.layers, self.downsample_layers):
            x = layer(x, embedding)
            skip_connections.append(x)
            x = downsample_layer(x)
        
        return x, skip_connections
    


class Encoder3D2D(nn.Module):
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

    def __init__(self, conv_depth, dpr, in_dims, out_dims):
        super().__init__()

        self.num_stages = len(conv_depth)

        # build encoder layers
        self.layers = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        for i_layer in range(self.num_stages):
            out_dim = 2*out_dims[i_layer] if i_layer == self.num_stages - 1 else in_dims[i_layer+1]
            layer = ConvBlocks3D2D(in_dim=in_dims[i_layer],
                                        kernel_size=3,
                                        out_dim=out_dims[i_layer],
                                        nb_blocks=conv_depth[i_layer], 
                                        dpr=dpr[i_layer])
                #layer = ConvBlocks(in_dim=in_dims[i_layer], out_dim=out_dims[i_layer], nb_block=conv_depth[i_layer])
                                
            downsample_layer = PatchMerging3D2D(in_dim=out_dims[i_layer], out_dim=out_dim)
            self.layers.append(layer)
            self.downsample_layers.append(downsample_layer)

        #self.norm = norm_layer(self.num_features)
        #self.norm_after_conv = norm_layer(embed_dim)

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

    def forward(self, x_3d, x_2d):
        skip_connections = []

        for layer, downsample_layer in zip(self.layers, self.downsample_layers):
            x_3d, x_2d = layer(x_3d, x_2d)
            skip_connections.append(x_3d)
            x_3d, x_2d = downsample_layer(x_3d, x_2d)
        
        return x_3d, skip_connections


class Encoder1D(nn.Module):
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

    def __init__(self, conv_layer, norm, conv_depth, dpr, out_dims):
        super().__init__()

        self.num_stages = len(conv_depth)
        in_dims = out_dims
        in_dims[0] = 1
        out_dims = [2*x for x in out_dims]
        out_dims[0] = in_dims[1]

        self.conv_1d = nn.Sequential(nn.Conv1d(in_channels=out_dims[-1], out_channels=out_dims[-1], kernel_size=3, padding=1),
                                    nn.BatchNorm1d(out_dims[-1]))

        # build encoder layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_stages):
            layer = conv_layer(in_dim=in_dims[i_layer],
                                kernel_size=3,
                                out_dim=out_dims[i_layer],
                                nb_blocks=conv_depth[i_layer], 
                                dpr=dpr[i_layer],
                                norm=norm)
            self.layers.append(layer)

        #self.norm = norm_layer(self.num_features)
        #self.norm_after_conv = norm_layer(embed_dim)

    def forward(self, x):
        x = torch.flatten(x, start_dim=-2).mean(-1) # T, B, 1
        x = x.permute(1, 2, 0).contiguous()

        for layer in self.layers:
            x = layer(x)
        
        x = self.conv_1d(x)
        
        return x
    
    

class ReconstructionEncoder(nn.Module):
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

    def __init__(self, blur, proj, shortcut, use_conv_mlp, blur_kernel, device, dpr, in_dims, swin_abs_pos, window_size, swin_layer_type, img_size, transformer_type='swin', rpe_mode=None, rpe_contextual_tensor='qkv', 
                 conv_depth=[2, 2], transformer_depth=[2, 2, 2], num_heads=[3, 6, 12],
                 mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0.,
                 norm_layer=nn.LayerNorm,
                 use_checkpoint=False, **kwargs):
        super().__init__()

        self.num_stages = len(conv_depth) + len(transformer_depth)
        self.mlp_ratio = mlp_ratio

        self.pos_drop = nn.Dropout(p=drop_rate)

        # build encoder layers
        self.layers = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        for i_layer in range(self.num_stages):
            input_resolution=(img_size//(2**i_layer), img_size//(2**i_layer))
            if i_layer < len(conv_depth):
                if i_layer == 0:
                    layer = ConvLayer(input_resolution=input_resolution, 
                                    in_dim=in_dims[i_layer],
                                    out_dim=in_dims[i_layer+1],
                                    nb_se_blocks=conv_depth[i_layer], 
                                    dpr=dpr[i_layer])
                    downsample_layer = PatchMergingConv(blur=blur, blur_kernel=blur_kernel, input_resolution=input_resolution, in_dim=in_dims[i_layer+1], out_dim=in_dims[i_layer+1], swin_abs_pos=False, device=device)
                elif i_layer == 1:
                    layer = ConvLayer(input_resolution=input_resolution, 
                                    in_dim=in_dims[i_layer],
                                    out_dim=2*in_dims[i_layer],
                                    nb_se_blocks=conv_depth[i_layer], 
                                    dpr=dpr[i_layer])
                    downsample_layer = PatchMergingConv(blur=blur, blur_kernel=blur_kernel, input_resolution=input_resolution, in_dim=2*in_dims[i_layer], out_dim=4*in_dims[i_layer], swin_abs_pos=swin_abs_pos, device=device)
            else:
                downsample_layer = PatchMergingSwin(blur=blur, blur_kernel=blur_kernel, input_resolution=input_resolution, in_dim=in_dims[i_layer], out_dim=2*in_dims[i_layer], swin_abs_pos=swin_abs_pos, device=device)
                #downsample_layer = PatchMergingConv(input_resolution=input_resolution, in_dim=in_dim, out_dim=2*in_dim)
                transformer_index = i_layer-len(conv_depth)
                if i_layer == self.num_stages - 1:
                    if transformer_type == 'swin':
                        layer = swin_layer_type.BasicLayer(dim=in_dims[i_layer],
                                        proj=proj,
                                        shortcut=shortcut,
                                        input_resolution=input_resolution,
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
                                        use_checkpoint=use_checkpoint)
                    elif transformer_type == 'vit':
                        layer = VitBasicLayer(in_dim=in_dims[i_layer], 
                                            nhead=num_heads[transformer_index], 
                                            rpe_mode=rpe_mode, 
                                            proj=proj,
                                            shortcut=shortcut,
                                            use_conv_mlp=use_conv_mlp,
                                            rpe_contextual_tensor=rpe_contextual_tensor, 
                                            input_resolution=input_resolution, 
                                            dropout=drop_rate,
                                            device=device, 
                                            nb_blocks=transformer_depth[transformer_index],
                                            dpr=dpr[i_layer])
                else:
                    layer = swin_layer_type.BasicLayer(dim=in_dims[i_layer],
                                    proj=proj,
                                    shortcut=shortcut,
                                    input_resolution=input_resolution,
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
                                    use_checkpoint=use_checkpoint)
            self.layers.append(layer)
            self.downsample_layers.append(downsample_layer)

        #self.norm = norm_layer(self.num_features)
        #self.norm_after_conv = norm_layer(embed_dim)

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

        for layer, downsample_layer in zip(self.layers, self.downsample_layers):
            x = layer(x)
            x = downsample_layer(x)
        
        return x


class EncoderNoConv(nn.Module):
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

    def __init__(self, blur, blur_kernel, embed_dim, device, dpr, in_dims, swin_abs_pos, window_size, swin_layer_type, img_size, transformer_type='swin', proj='linear', rpe_mode=None, rpe_contextual_tensor='qkv', 
                 transformer_depth=[2, 2, 2], num_heads=[3, 6, 12],
                 mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0.,
                 norm_layer=nn.LayerNorm,
                 use_checkpoint=False, **kwargs):
        super().__init__()

        self.num_stages = len(transformer_depth)
        self.mlp_ratio = mlp_ratio

        self.pos_drop = nn.Dropout(p=drop_rate)

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = 4, p2 = 4),
            nn.Linear((1 * 4 * 4), embed_dim),
        )

        # build encoder layers
        self.layers = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        for i_layer in range(2, 2 + self.num_stages):
            transformer_index = i_layer-2
            input_resolution=(img_size//(2**i_layer), img_size//(2**i_layer))
            downsample_layer = PatchMergingSwin(blur=blur, blur_kernel=blur_kernel, input_resolution=input_resolution, in_dim=in_dims[i_layer], out_dim=2*in_dims[i_layer], swin_abs_pos=swin_abs_pos, device=device)
            #downsample_layer = PatchMergingConv(input_resolution=input_resolution, in_dim=in_dim, out_dim=2*in_dim)
            if i_layer == self.num_stages - 1:
                if transformer_type == 'swin':
                    layer = swin_layer_type.BasicLayer(dim=in_dims[i_layer],
                                    input_resolution=input_resolution,
                                    depth=transformer_depth[transformer_index],
                                    num_heads=num_heads[transformer_index],
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
                                    use_checkpoint=use_checkpoint)
                elif transformer_type == 'vit':
                    layer = VitBasicLayer(in_dim=in_dims[i_layer], 
                                        nhead=num_heads[transformer_index], 
                                        rpe_mode=rpe_mode, 
                                        rpe_contextual_tensor=rpe_contextual_tensor, 
                                        input_resolution=input_resolution, 
                                        proj=proj, 
                                        dropout=drop_rate,
                                        device=device, 
                                        nb_blocks=transformer_depth[transformer_index],
                                        dpr=dpr[i_layer])
            else:
                layer = swin_layer_type.BasicLayer(dim=in_dims[i_layer],
                                input_resolution=input_resolution,
                                depth=transformer_depth[transformer_index],
                                num_heads=num_heads[transformer_index],
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
                                use_checkpoint=use_checkpoint)
            self.layers.append(layer)
            self.downsample_layers.append(downsample_layer)

        #self.norm = norm_layer(self.num_features)
        #self.norm_after_conv = norm_layer(embed_dim)

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
        skip_connections = []
        B, C, H, W = x.shape
        x = self.to_patch_embedding(x)

        for layer, downsample_layer in zip(self.layers, self.downsample_layers):
            x = layer(x)
            skip_connections.append(x)
            x = downsample_layer(x)
        
        return x, skip_connections

        

class ConvEncoder(nn.Module):

    def __init__(self, blur, device, blur_kernel, dpr, in_localizer_dims, img_size, localizer_conv_depth=[2, 2, 2, 2, 2]):
        super().__init__()

        self.num_stages = len(localizer_conv_depth)

        # build encoder layers
        self.layers = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        for i_layer in range(self.num_stages):
            input_resolution = (int(img_size / (2**(i_layer))), int(img_size / (2**(i_layer))))
            out_dim = in_localizer_dims[i_layer + 1] if i_layer == 0 else in_localizer_dims[i_layer] * 2
            layer = ResnetConvLayer(input_resolution=input_resolution, 
                              in_dim=in_localizer_dims[i_layer], 
                              out_dim=out_dim, 
                              nb_se_blocks=localizer_conv_depth[i_layer], 
                              dpr=dpr[i_layer])
            downsample_layer = PatchMergingConv(blur=blur, blur_kernel=blur_kernel, input_resolution=input_resolution, in_dim=out_dim, out_dim=out_dim, swin_abs_pos=False, device=device)
            self.layers.append(layer)
            self.downsample_layers.append(downsample_layer)
    
    def forward(self, x):
        skip_connections = []
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).view(B, -1, C)

        for layer, downsample_layer in zip(self.layers, self.downsample_layers):
            x = layer(x)
            skip_connections.append(x)
            x = downsample_layer(x)
        
        return x, skip_connections

class CCVVV(nn.Module):
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

    def __init__(self, device, proj='linear', rpe_mode=None, rpe_contextual_tensor='qkv', img_size=224, patch_size=[4, 4], in_chans=1,
                 embed_dim=96, conv_depth=[2, 2], transformer_depth=[2, 2, 2], num_heads=[3, 6, 12],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, deep_supervision=True, **kwargs):
        super().__init__()

        self.num_stages = len(conv_depth) + len(transformer_depth)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_stages - 2))
        self.mlp_ratio = mlp_ratio
        self.deep_supervision = deep_supervision
        self.device=device

        self.conv_block_1 = ConvDownBlock(img_size=img_size, in_dim=in_chans, out_dim=embed_dim//2)
        self.conv_block_2 = ConvDownBlock(img_size=img_size//2, in_dim=embed_dim//2, out_dim=embed_dim)

        patches_resolution = [img_size // patch_size[0], img_size // patch_size[1]]
        num_patches = patches_resolution[0] * patches_resolution[1]

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(transformer_depth))]  # stochastic depth decay rule

        # build encoder layers
        self.layers = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        self.sine_position_encodings = nn.ModuleList()
        self.input_resolutions = []
        self.dims = []
        for i_layer in range(len(transformer_depth)):
            dim = int(embed_dim * 2 ** i_layer)
            input_resolution = (patches_resolution[0] // (2 ** i_layer), patches_resolution[1] // (2 ** i_layer))
            layer_type = TransformerEncoderLayer(d_model=dim,
                                                nhead=num_heads[i_layer],
                                                input_resolution=input_resolution,
                                                proj=proj,
                                                device=device,
                                                rpe_mode=rpe_mode,
                                                rpe_contextual_tensor=rpe_contextual_tensor)
            layer = nn.ModuleList([copy.deepcopy(layer_type) for i in range(transformer_depth[i_layer])])
            self.layers.append(layer)
            self.sine_position_encodings.append(PositionEmbeddingSine2d(num_pos_feats=dim//2, normalize=True))
            self.input_resolutions.append(input_resolution)
            self.downsample_layers.append(PatchMergingConv(input_resolution=input_resolution, dim=dim))
            self.dims.append(dim)

        self.norm = norm_layer(self.num_features)
        self.norm_after_embedding = norm_layer(embed_dim)

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
        skip_connections = []

        x, saved_out_1 = self.conv_block_1(x)
        skip_connections.append(saved_out_1)

        x, saved_out_2 = self.conv_block_2(x)
        skip_connections.append(saved_out_2)

        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C

        if self.norm is not None:
            x = self.norm_after_embedding(x)

        x = self.pos_drop(x)


        for layer, pos_embed_obj, input_resolution, downsample_layer, dim in zip(self.layers, self.sine_position_encodings, self.input_resolutions, self.downsample_layers, self.dims):
            x = x.permute(1, 0, 2)
            for bloc in layer:
                shape_util = (x.shape[1],) + input_resolution
                pos_embed = pos_embed_obj(shape_util=shape_util, device=self.device).permute(2, 3, 0, 1).view(-1, x.shape[1], dim)
                x = bloc(x, pos=pos_embed)
            x = x.permute(1, 0, 2)
            skip_connections.append(x)
            x = downsample_layer(x)

        x = self.norm(x)  # B L C
  
        return x, skip_connections


class CCCVV(nn.Module):
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

    def __init__(self, device, proj='linear', rpe_mode=None, rpe_contextual_tensor='qkv', img_size=224, patch_size=[4, 4], in_chans=1,
                 embed_dim=96, conv_depth=[2, 2], transformer_depth=[2, 2, 2], num_heads=[6, 12],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, deep_supervision=True, **kwargs):
        super().__init__()

        self.num_stages = len(conv_depth) + len(transformer_depth)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_stages - 2))
        self.mlp_ratio = mlp_ratio
        self.deep_supervision = deep_supervision
        self.device=device

        self.conv_block_1 = ConvDownBlock(img_size=img_size, in_dim=in_chans, out_dim=embed_dim//2)
        self.conv_block_2 = ConvDownBlock(img_size=img_size//2, in_dim=embed_dim//2, out_dim=embed_dim)
        self.conv_block_3 = ConvDownBlock(img_size=img_size//4, in_dim=embed_dim, out_dim=2*embed_dim)

        patches_resolution = [img_size // patch_size[0], img_size // patch_size[1]]
        num_patches = patches_resolution[0] * patches_resolution[1]

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(transformer_depth))]  # stochastic depth decay rule

        # build encoder layers
        self.layers = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        self.sine_position_encodings = nn.ModuleList()
        self.input_resolutions = []
        self.dims = []
        for i_layer in range(len(transformer_depth)):
            dim = int(embed_dim * 2 ** (i_layer + 1))
            input_resolution = (patches_resolution[0] // (2 ** (i_layer + 1)), patches_resolution[1] // (2 ** (i_layer + 1)))
            layer_type = TransformerEncoderLayer(d_model=dim,
                                                nhead=num_heads[i_layer],
                                                input_resolution=input_resolution,
                                                proj=proj,
                                                device=device,
                                                rpe_mode=rpe_mode,
                                                rpe_contextual_tensor=rpe_contextual_tensor)
            layer = nn.ModuleList([copy.deepcopy(layer_type) for i in range(transformer_depth[i_layer])])
            self.layers.append(layer)
            self.sine_position_encodings.append(PositionEmbeddingSine2d(num_pos_feats=dim//2, normalize=True))
            self.input_resolutions.append(input_resolution)
            self.downsample_layers.append(PatchMergingConv(input_resolution=input_resolution, dim=dim))
            self.dims.append(dim)

        self.norm = norm_layer(self.num_features)
        self.norm_after_conv = norm_layer(embed_dim*2)

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
        skip_connections = []

        x, saved_out_1 = self.conv_block_1(x)
        skip_connections.append(saved_out_1)

        x, saved_out_2 = self.conv_block_2(x)
        skip_connections.append(saved_out_2)

        x, saved_out_3 = self.conv_block_3(x)
        skip_connections.append(saved_out_3)

        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C

        if self.norm is not None:
            x = self.norm_after_conv(x)

        x = self.pos_drop(x)


        for layer, pos_embed_obj, input_resolution, downsample_layer, dim in zip(self.layers, self.sine_position_encodings, self.input_resolutions, self.downsample_layers, self.dims):
            x = x.permute(1, 0, 2)
            for bloc in layer:
                shape_util = (x.shape[1],) + input_resolution
                pos_embed = pos_embed_obj(shape_util=shape_util, device=self.device).permute(2, 3, 0, 1).view(-1, x.shape[1], dim)
                x = bloc(x, pos=pos_embed)
            x = x.permute(1, 0, 2)
            skip_connections.append(x)
            x = downsample_layer(x)

        x = self.norm(x)  # B L C
  
        return x, skip_connections

class CCSSS_3d(nn.Module):
    """ Swin Transformer backbone.
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        patch_size (int | tuple(int)): Patch size. Default: (4,4,4).
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        depths (tuple[int]): Depths of each Swin Transformer stage.
        num_heads (tuple[int]): Number of attention head of each stage.
        window_size (int): Window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: Truee
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer: Normalization layer. Default: nn.LayerNorm.
        patch_norm (bool): If True, add normalization after patch embedding. Default: False.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
    """

    def __init__(self,
                 img_size=224,
                 pretrained=None,
                 pretrained2d=True,
                 patch_size=(4,4,4),
                 in_chans=1,
                 embed_dim=96,
                 conv_depth=[2, 2], 
                 transformer_depth=[2, 2, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=(2,7,7),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 patch_norm=False,
                 frozen_stages=-1,
                 use_checkpoint=False):
        super().__init__()

        self.pretrained = pretrained
        self.pretrained2d = pretrained2d
        self.num_stages = len(conv_depth) + len(transformer_depth)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.frozen_stages = frozen_stages
        self.window_size = window_size
        self.patch_size = patch_size

        self.conv_block_1 = ConvDownBlock_3d(in_dim=in_chans, out_dim=embed_dim//2, stride=(patch_size[0]//2, patch_size[1]//2, patch_size[2]//2))
        self.conv_block_2 = ConvDownBlock_3d(in_dim=embed_dim//2, out_dim=embed_dim, stride=(2, patch_size[1]//2, patch_size[2]//2))

        # split image into non-overlapping patches
        #self.patch_embed = PatchEmbed3D(
        #    patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
        #    norm_layer=norm_layer if self.patch_norm else None)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(transformer_depth))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(len(transformer_depth)):
            layer = swin_transformer_3d.BasicLayer(
                dim=int(embed_dim * 2**i_layer),
                depth=transformer_depth[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(transformer_depth[:i_layer]):sum(transformer_depth[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMergingConv_3d,
                use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        self.num_features = int(embed_dim * 2**(self.num_stages - 2))

        # add a norm layer for each output
        self.norm = norm_layer(self.num_features)

        self.norm_after_embedding = norm_layer(embed_dim)

        self._freeze_stages()

        self.apply(self._init_weights)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.conv_block_1.eval()
            self.conv_block_2.eval()
            for param in self.conv_block_1.parameters():
                param.requires_grad = False
            for param in self.conv_block_2.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def inflate_weights(self, logger):
        """Inflate the swin2d parameters to swin3d.

        The differences between swin3d and swin2d mainly lie in an extra
        axis. To utilize the pretrained parameters in 2d model,
        the weight of swin2d models should be inflated to fit in the shapes of
        the 3d counterpart.

        Args:
            logger (logging.Logger): The logger used to print
                debugging infomation.
        """
        checkpoint = torch.load(self.pretrained, map_location='cpu')
        state_dict = checkpoint['model']

        # delete relative_position_index since we always re-init it
        relative_position_index_keys = [k for k in state_dict.keys() if "relative_position_index" in k]
        for k in relative_position_index_keys:
            del state_dict[k]

        # delete attn_mask since we always re-init it
        attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
        for k in attn_mask_keys:
            del state_dict[k]

        state_dict['patch_embed.proj.weight'] = state_dict['patch_embed.proj.weight'].unsqueeze(2).repeat(1,1,self.patch_size[0],1,1) / self.patch_size[0]

        # bicubic interpolate relative_position_bias_table if not match
        relative_position_bias_table_keys = [k for k in state_dict.keys() if "relative_position_bias_table" in k]
        for k in relative_position_bias_table_keys:
            relative_position_bias_table_pretrained = state_dict[k]
            relative_position_bias_table_current = self.state_dict()[k]
            L1, nH1 = relative_position_bias_table_pretrained.size()
            L2, nH2 = relative_position_bias_table_current.size()
            L2 = (2*self.window_size[1]-1) * (2*self.window_size[2]-1)
            wd = self.window_size[0]
            if nH1 != nH2:
                logger.warning(f"Error in loading {k}, passing")
            else:
                if L1 != L2:
                    S1 = int(L1 ** 0.5)
                    relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
                        relative_position_bias_table_pretrained.permute(1, 0).view(1, nH1, S1, S1), size=(2*self.window_size[1]-1, 2*self.window_size[2]-1),
                        mode='bicubic')
                    relative_position_bias_table_pretrained = relative_position_bias_table_pretrained_resized.view(nH2, L2).permute(1, 0)
            state_dict[k] = relative_position_bias_table_pretrained.repeat(2*wd-1,1)

        msg = self.load_state_dict(state_dict, strict=False)
        logger.info(msg)
        logger.info(f"=> loaded successfully '{self.pretrained}'")
        del checkpoint
        torch.cuda.empty_cache()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        if pretrained:
            self.pretrained = pretrained
        if isinstance(self.pretrained, str):
            #self.apply(_init_weights)
            logger = get_root_logger()
            logger.info(f'load model from: {self.pretrained}')

            if self.pretrained2d:
                # Inflate 2D model into 3D model.
                self.inflate_weights(logger)
            else:
                # Directly load 3D model.
                load_checkpoint(self, self.pretrained, strict=False, logger=logger)
        #elif self.pretrained is None:
        #    self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        """Forward function."""
        B, C, D, H, W = x.shape

        x_downsample = []

        x, saved_out_1 = self.conv_block_1(x)
        x_downsample.append(saved_out_1)

        print(x.shape)
        x, saved_out_2 = self.conv_block_2(x)
        print(x.shape)
        x_downsample.append(saved_out_2)

        x = x.permute(0, 2, 3, 4, 1)  # B D H W C

        if self.norm is not None:
            x = self.norm_after_embedding(x)
            x = x.permute(0, 4, 1, 2, 3)


        x = self.pos_drop(x)

        for layer in self.layers:
            x, layer_out = layer(x.contiguous())
            x_downsample.append(layer_out)

        x = rearrange(x, 'b c d h w -> b d h w c')
        x = self.norm(x)
        x = rearrange(x, 'b d h w c -> b c d h w')
  
        return x, x_downsample

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(CCSSS_3d, self).train(mode)
        self._freeze_stages()