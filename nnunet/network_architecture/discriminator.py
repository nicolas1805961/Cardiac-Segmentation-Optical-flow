import torch
import torch.nn as nn
from ..lib.encoder import Encoder
from ..lib.vit_transformer import TransformerEncoder, TransformerEncoderLayer
from ..lib.position_embedding import PositionEmbeddingSine2d

class Discriminator(nn.Module):
    def __init__(self,
                out_encoder_dims,
                device,
                in_dims,
                image_size,
                conv_layer,
                conv_depth,
                drop_path_rate,
                bottleneck_heads,
                norm_2d
                ):
        super().__init__()

        num_stages = len(conv_depth)
        self.d_model = out_encoder_dims[-1] * 2

        # stochastic depth
        num_blocks = conv_depth
        my_iter = torch.cat([torch.tensor([0]), torch.cumsum(torch.tensor(num_blocks), dim=0)])
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(num_blocks))]
        dpr = [dpr[my_iter[j]:my_iter[j+1]] for j in range(len(num_blocks))]
        dpr_encoder = dpr[:num_stages]

        self.final_conv = nn.Conv2d(in_channels=self.d_model, out_channels=1, kernel_size=int(image_size/2**3))

        self.encoder = Encoder(conv_layer=conv_layer, norm=norm_2d, out_dims=out_encoder_dims, device=device, in_dims=in_dims, conv_depth=conv_depth, dpr=dpr_encoder)

        #self.pos = PositionEmbeddingSine2d(num_pos_feats=self.d_model // 2, normalize=True)
        #self_attention_layer = TransformerEncoderLayer(d_model=self.d_model, nhead=bottleneck_heads, dim_feedforward=2048)
        #self.self_attention_bottleneck = TransformerEncoder(encoder_layer=self_attention_layer, num_layers=1)
    
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x, _ = self.encoder(x)
        B, C, H, W = x.shape
        #pos = self.pos(shape_util=(B, H, W), device=x.device)
        #pos = torch.flatten(pos, start_dim=2).permute(0, 2, 1)
        #x = self.self_attention_bottleneck(x, pos=pos)
        x = self.final_conv(x)
        return self.sigmoid(x)