import torch.nn.functional as F
import torch.nn as nn
import torch
from utils import ConvLayer

from timm.models.layers import trunc_normal_
from utils import ConvLayer, PatchMergingConv, PatchMergingSwin
from vit_transformer import VitBasicLayer
import math
from encoder import ConvEncoder, Encoder
import swin_transformer_2

class LocalisationNet(nn.Module):
    def __init__(self, blur, blur_kernel, device, dpr, in_dims, swin_abs_pos, swin_layer_type, transformer_type='swin', proj='linear', rpe_mode=None, rpe_contextual_tensor='qkv', img_size=224, patch_size=[4, 4], in_chans=1,
                 embed_dim=96, conv_depth=[2, 2], transformer_depth=[2, 2, 2, 2], num_heads=[3, 6, 12],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0.,
                 norm_layer=nn.LayerNorm,
                 use_checkpoint=False, **kwargs):

        super(LocalisationNet, self).__init__()

        self.num_stages = len(conv_depth) + len(transformer_depth)
        self.embed_dim = embed_dim
        self.d_model = embed_dim * (2 ** (len(transformer_depth) - 1))
        self.last_res_size = int(img_size / (2**(self.num_stages - 1)))
        #self.num_features = int(embed_dim * 2 ** (self.num_stages - 2))
        self.mlp_ratio = mlp_ratio

        self.pos_drop = nn.Dropout(p=drop_rate)

        # build encoder layers
        self.layers = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        for i_layer in range(self.num_stages):
            input_resolution=(img_size//(2**i_layer), img_size//(2**i_layer))
            #in_dim = int(embed_dim*(2**(i_layer-2)))
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
                if i_layer == self.num_stages - 1:
                    downsample_layer = nn.Identity()
                else:
                    downsample_layer = PatchMergingSwin(blur=blur, blur_kernel=blur_kernel, input_resolution=input_resolution, in_dim=in_dims[i_layer], out_dim=2*in_dims[i_layer], swin_abs_pos=swin_abs_pos, device=device)
                #downsample_layer = PatchMergingConv(input_resolution=input_resolution, in_dim=in_dim, out_dim=2*in_dim)
                transformer_index = i_layer-len(conv_depth)
                if i_layer > 3:
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
            self.final_conv = nn.Conv2d(in_channels=self.d_model, out_channels=self.d_model, kernel_size=self.last_res_size)
            self.final = nn.Sequential(nn.Linear(768, 384), nn.ReLU(True), nn.Linear(384, 192), nn.ReLU(True), nn.Linear(192, 96), nn.ReLU(True), nn.Linear(96, 48), nn.ReLU(True), nn.Linear(48, 24), nn.ReLU(True), nn.Linear(24, 12), nn.ReLU(True), nn.Linear(12, 4))
            self.final[-1].weight.data.zero_()
            self.final[-1].bias.data.copy_(torch.tensor([0, 0, 1, 0], dtype=torch.float))

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
        x1 = x.permute(0, 2, 3, 1).view(B, -1, C)

        for layer, downsample_layer in zip(self.layers, self.downsample_layers):
            x1 = layer(x1)
            x1 = downsample_layer(x1)
        
        x1 = x1.permute(0, 2, 1).view(B, self.d_model, self.last_res_size, self.last_res_size)
        
        x1 = self.final_conv(x1).view(B, -1)

        theta = self.final(x1)

        tx = theta[:, 0]
        ty = theta[:, 1]
        scale = theta[:, 2]
        angle = theta[:, 3]

        metadata = {'tx': tx, 'ty': ty, 'scale': scale, 'angle': angle}

        r = get_rotation_batched_matrices(angle)
        t = get_translation_batched_matrices(tx, ty)
        s = get_scaling_batched_matrices(scale)
        theta = torch.bmm(torch.bmm(t, r), s)[:, :-1]

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x, metadata


class LocalisationNet2(nn.Module):
    def __init__(self, 
                blur,
                blur_kernel,
                mlp_intermediary_dim,
                deep_supervision,
                merge,
                localizer_conv_depth,
                device,
                in_localizer_dims,
                image_size,
                num_bottleneck_layers=2,
                drop_path_rate=0.1):

        super(LocalisationNet2, self).__init__()
        num_stages = len(localizer_conv_depth)
        d_model = in_localizer_dims[1] * (2 ** (num_stages - 1))

        # stochastic depth
        num_blocks = localizer_conv_depth + [num_bottleneck_layers]
        my_iter = torch.cat([torch.tensor([0]), torch.cumsum(torch.tensor(num_blocks), dim=0)])
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(num_blocks))]
        dpr = [dpr[my_iter[j]:my_iter[j+1]] for j in range(len(num_blocks))]
        dpr_encoder = dpr[:num_stages]
        dpr_decoder = [x[::-1] for x in dpr_encoder[::-1]]
        dpr_bottleneck = dpr[-1]

        decoder_in_dims = [x * 2 for x in in_localizer_dims]
        decoder_in_dims[0] = in_localizer_dims[1]
        decoder_in_dims = decoder_in_dims[::-1]

        self.localization_encoder = ConvEncoder(blur=blur,
                                                device=device,
                                                blur_kernel=blur_kernel,
                                                dpr=dpr_encoder,
                                                in_localizer_dims=in_localizer_dims,
                                                img_size=image_size,
                                                localizer_conv_depth=localizer_conv_depth)
        self.localization_decoder = ConvDecoder(blur=blur,
                                                device=device,
                                                merge=merge,
                                                blur_kernel=blur_kernel,
                                                deep_supervision=deep_supervision,
                                                dpr=dpr_decoder,
                                                in_localizer_dims=decoder_in_dims,
                                                img_size=image_size,
                                                localizer_conv_depth=localizer_conv_depth)
        last_res = image_size / (2 ** num_stages)
        self.localization_bottleneck = ConvLayer(input_resolution=last_res, in_dim=d_model, out_dim=d_model * 2, nb_se_blocks=num_bottleneck_layers, dpr=dpr_bottleneck)
        self.localization_net = nn.Sequential(nn.Linear(last_res * last_res * d_model * 2, mlp_intermediary_dim), nn.ReLU(), nn.Linear(mlp_intermediary_dim, 4))
        self.localization_net[-1].weight.data.zero_()
        self.localization_net[-1].bias.data.copy_(torch.tensor([0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        B, C, H, W = x.shape
        x, skip_connections = self.localization_encoder(x)
        x = self.localization_bottleneck(x)
        theta = self.localization_net(x.view(B, -1))
        x = self.localization_decoder(x, skip_connections)

        tx = theta[:, 0]
        ty = theta[:, 1]
        scale = theta[:, 2]
        angle = theta[:, 3]

        metadata = {'tx': tx, 'ty': ty, 'scale': scale, 'angle': angle}

        return x, metadata


class LocalisationNet3(nn.Module):
    def __init__(self, 
                blur,
                blur_kernel,
                mlp_intermediary_dim,
                deep_supervision,
                window_size,
                merge,
                conv_depth,
                transformer_depth,
                num_heads,
                transformer_type,
                device,
                in_dims,
                swin_abs_pos,
                swin_layer_type,
                proj,
                rpe_mode,
                img_size,
                rpe_contextual_tensor,
                num_bottleneck_layers=2,
                bottleneck_heads=24,
                drop_path_rate=0.1):

        super(LocalisationNet3, self).__init__()
        num_stages = (len(transformer_depth) + len(conv_depth))
        d_model = in_dims[-1] * 2

        # stochastic depth
        num_blocks = conv_depth + transformer_depth + [num_bottleneck_layers]
        my_iter = torch.cat([torch.tensor([0]), torch.cumsum(torch.tensor(num_blocks), dim=0)])
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(num_blocks))]
        dpr = [dpr[my_iter[j]:my_iter[j+1]] for j in range(len(num_blocks))]
        dpr_encoder = dpr[:num_stages]
        dpr_decoder = [x[::-1] for x in dpr_encoder[::-1]]
        dpr_bottleneck = dpr[-1]

        self.localization_encoder = Encoder(blur=blur,
                                            blur_kernel=blur_kernel,
                                            device=device,
                                            dpr=dpr_encoder,
                                            in_dims=in_dims,
                                            swin_abs_pos=swin_abs_pos,
                                            swin_layer_type=swin_layer_type,
                                            transformer_type=transformer_type,
                                            window_size=window_size,
                                            proj=proj,
                                            rpe_mode=rpe_mode,
                                            rpe_contextual_tensor=rpe_contextual_tensor,
                                            img_size=img_size,
                                            conv_depth=conv_depth,
                                            transformer_depth=transformer_depth,
                                            num_heads=num_heads)
        self.localization_decoder = Decoder(last_activation='softmax',
                                            blur=blur,
                                            blur_kernel=blur_kernel,
                                            device=device,
                                            dpr=dpr_decoder,
                                            num_classes=2,
                                            in_encoder_dims=in_dims[::-1],
                                            swin_layer_type=swin_layer_type,
                                            merge=merge,
                                            swin_abs_pos=swin_abs_pos,
                                            transformer_type=transformer_type,
                                            window_size=window_size,
                                            img_size=img_size,
                                            proj=proj,
                                            rpe_mode=rpe_mode,
                                            rpe_contextual_tensor=rpe_contextual_tensor,
                                            conv_depth=conv_depth,
                                            transformer_depth=transformer_depth,
                                            num_heads=num_heads,
                                            deep_supervision=deep_supervision)
        last_res = int(img_size / (2 ** num_stages))
        self.localization_bottleneck = swin_transformer_2.BasicLayer(dim=d_model,
                                                                     input_resolution=[last_res,last_res],
                                                                     depth=num_bottleneck_layers,
                                                                     num_heads=bottleneck_heads,
                                                                     proj=proj,
                                                                     device=device,
                                                                     rpe_mode=rpe_mode,
                                                                     rpe_contextual_tensor=rpe_contextual_tensor,
                                                                     window_size=window_size,
                                                                     drop_path=dpr_bottleneck)
        in_dim_linear = int(last_res * last_res * d_model)
        self.localization_net = nn.Sequential(nn.Linear(in_dim_linear, mlp_intermediary_dim), nn.ReLU(), nn.Linear(mlp_intermediary_dim, 4))
        self.localization_net[-1].weight.data.zero_()
        self.localization_net[-1].bias.data.copy_(torch.tensor([0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        B, C, H, W = x.shape
        x, skip_connections = self.localization_encoder(x)
        x = self.localization_bottleneck(x)
        theta = self.localization_net(x.view(B, -1))
        x = self.localization_decoder(x, skip_connections)

        tx = theta[:, 0]
        ty = theta[:, 1]
        scale = theta[:, 2]
        angle = theta[:, 3]

        metadata = {'tx': tx, 'ty': ty, 'scale': scale, 'angle': angle}

        return x, metadata


def get_rotation_batched_matrices(angle):
    matrices = []
    for i in range(angle.size(0)):
        m = torch.tensor([[math.cos(angle[i]), -1.0*math.sin(angle[i]), 0.], 
                          [math.sin(angle[i]), math.cos(angle[i]), 0.], 
                          [0, 0, 1]], device=angle.device).float()
        matrices.append(m)
    return torch.stack(matrices, dim=0)

def get_translation_batched_matrices(tx, ty):
    matrices = []
    for i in range(tx.size(0)):
        m = torch.tensor([[1, 0, tx[i]], 
                          [0, 1, ty[i]], 
                          [0, 0, 1]], device=tx.device).float()
        matrices.append(m)
    return torch.stack(matrices, dim=0)

def get_scaling_batched_matrices(scale):
    matrices = []
    for i in range(scale.size(0)):
        m = torch.tensor([[scale[i], 0, 0], 
                          [0, scale[i], 0], 
                          [0, 0, 1]], device=scale.device).float()
        matrices.append(m)
    return torch.stack(matrices, dim=0)


def unnormalize_translation(x):
    return 2 * x - 1

def unnormalize_rotation(x):
    return x * 3.1378278068987537

def unnormalize_scale(x):
    return x * (1.7744 - 0.0447) + 0.0447


class LocalisationNet2(nn.Module):
    def __init__(self, blur, blur_kernel, device, dpr, in_dims, swin_abs_pos, swin_layer_type, transformer_type='swin', proj='linear', rpe_mode=None, rpe_contextual_tensor='qkv', img_size=224, patch_size=[4, 4], in_chans=1,
                 embed_dim=96, conv_depth=[2, 2], transformer_depth=[2, 2, 2, 2], num_heads=[3, 6, 12],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0.,
                 norm_layer=nn.LayerNorm,
                 use_checkpoint=False, **kwargs):

        super(LocalisationNet2, self).__init__()

        self.net = nn.Sequential(nn.Conv2d(1, 24, kernel_size=3, padding=1), nn.BatchNorm2d(24), nn.ReLU(True), nn.MaxPool2d(2, stride=2),
                            nn.Conv2d(24, 48, kernel_size=3, padding=1), nn.BatchNorm2d(48), nn.ReLU(True), nn.MaxPool2d(2, stride=2))
        
        self.final_conv = nn.Conv2d(in_channels=48, out_channels=48, kernel_size=56)
        self.final = nn.Sequential(nn.Linear(48, 24), nn.ReLU(True), nn.Linear(24, 4))
        self.final[-1].weight.data.zero_()
        self.final[-1].bias.data.copy_(torch.tensor([0, 0, 1, 0], dtype=torch.float))

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
        x1 = self.net(x)
        
        x1 = self.final_conv(x1).view(B, -1)

        theta = self.final(x1)

        tx = theta[:, 0]
        ty = theta[:, 1]
        scale = theta[:, 2]
        angle = theta[:, 3]

        metadata = {'tx': tx, 'ty': ty, 'scale': scale, 'angle': angle}

        r = get_rotation_batched_matrices(angle)
        t = get_translation_batched_matrices(tx, ty)
        s = get_scaling_batched_matrices(scale)
        theta = torch.bmm(torch.bmm(t, r), s)[:, :-1]

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x, metadata