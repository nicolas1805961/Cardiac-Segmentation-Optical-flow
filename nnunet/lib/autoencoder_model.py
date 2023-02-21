import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from bottlenecks import temporalTransformer
from vit_transformer import VitBasicLayer, VitChannelLayer
from position_embedding import PositionEmbeddingSine2d, PositionEmbeddingSine3d
import copy
from encoder import Encoder, CCSSS_3d, CCVVV, CCCVV, EncoderNoConv, ConvEncoder, ReconstructionEncoder
from utils import MLP, ConvLayer, ResnetConvLayer, AutoencoderMlp
import swin_transformer_2
import swin_transformer_3d
import swin_transformer


class Autoencoder(nn.Module):
    def __init__(self, 
                blur,
                autoencoder_dim,
                blur_kernel,
                batch_size, 
                patch_size, 
                window_size, 
                swin_abs_pos,
                deep_supervision,
                proj,
                out_encoder_dims,
                use_conv_mlp,
                device,
                in_dims,
                image_size,
                swin_layer_type,
                transformer_type='swin',
                conv_depth=[2, 2],
                transformer_depth=[2, 2, 2], 
                num_heads=[3, 6, 12],
                drop_path_rate=0.1, 
                rpe_mode=None, 
                rpe_contextual_tensor=None):
        super().__init__()
        
        num_stages = (len(transformer_depth) + len(conv_depth))
        self.batch_size = batch_size
        self.d_model = out_encoder_dims[-1] * 2
        self.bottleneck_size = [int(image_size / (x * 2**(num_stages - 2))) for x in patch_size]

        # stochastic depth
        num_blocks = conv_depth + transformer_depth
        my_iter = torch.cat([torch.tensor([0]), torch.cumsum(torch.tensor(num_blocks), dim=0)])
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(num_blocks))]
        dpr = [dpr[my_iter[j]:my_iter[j+1]] for j in range(len(num_blocks))]
        dpr_encoder = dpr[:num_stages]
        dpr_decoder = [x[::-1] for x in dpr_encoder[::-1]]

        self.encoder = ReconstructionEncoder(proj=proj, use_conv_mlp=use_conv_mlp, blur=blur, img_size=image_size, blur_kernel=blur_kernel, device=device, swin_layer_type=swin_layer_type, swin_abs_pos=swin_abs_pos, in_dims=in_dims, conv_depth=conv_depth, transformer_depth=transformer_depth, transformer_type=transformer_type, dpr=dpr_encoder, rpe_mode=rpe_mode, rpe_contextual_tensor=rpe_contextual_tensor, patch_size=patch_size, num_heads=num_heads, window_size=window_size, drop_path_rate=drop_path_rate, deep_supervision=deep_supervision)
        self.decoder = ReconstructionBranch(nb_classes=1, proj=proj, blur=blur, use_conv_mlp=use_conv_mlp, blur_kernel=blur_kernel, device=device, dpr=dpr_decoder, in_encoder_dims=in_dims[::-1], window_size=window_size, img_size=image_size, swin_layer_type=swin_layer_type, swin_abs_pos=swin_abs_pos, transformer_type=transformer_type, rpe_mode=rpe_mode, rpe_contextual_tensor=rpe_contextual_tensor, conv_depth=conv_depth, transformer_depth=transformer_depth, num_heads=num_heads, deep_supervision=deep_supervision)
        mlp_in_dim = self.bottleneck_size[0] * self.bottleneck_size[1] * self.d_model
        self.mlp = AutoencoderMlp(in_features=mlp_in_dim, hidden_features=autoencoder_dim, out_features=mlp_in_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        x = self.mlp(x)
        x = x.view(B, -1, self.d_model)
        x = self.decoder(x)
        return x