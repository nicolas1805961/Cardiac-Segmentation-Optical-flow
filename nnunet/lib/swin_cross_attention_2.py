import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from .utils import depthwise_separable_conv, Mlp, get_indices_2d
import torch.utils.checkpoint as checkpoint
import matplotlib.pyplot as plt
from .position_embedding import PositionEmbeddingSine2d


class SwinCrossAttention(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, proj, same_key_query, input_resolution, num_heads, device, relative_position_index, rpe_mode, rpe_contextual_tensor, window_size, shift_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"
        
        self.before_cross_attention_img1 = BeforeCrossAttention(dim=dim, input_resolution=input_resolution, window_size=self.window_size, shift_size=self.shift_size)
        self.before_cross_attention_img2 = BeforeCrossAttention(dim=dim, input_resolution=input_resolution, window_size=self.window_size, shift_size=self.shift_size)
        self.cross_attn = CrossAttention(dim=dim, 
                                            same_key_query=same_key_query,
                                            input_resolution=input_resolution,
                                            proj=proj,
                                            window_size=to_2tuple(self.window_size), 
                                            num_heads=num_heads,
                                            device=device,
                                            relative_position_index=relative_position_index,
                                            rpe_mode=rpe_mode,
                                            rpe_contextual_tensor=rpe_contextual_tensor,
                                            qkv_bias=qkv_bias, 
                                            qk_scale=qk_scale, 
                                            attn_drop=attn_drop, 
                                            proj_drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, rescaled, rescaler):
        #H, W = self.input_resolution
        B, C, H, W = rescaled.shape
        rescaled = rescaled.permute(0, 2, 3, 1).view(B, H * W, C)
        rescaler = rescaler.permute(0, 2, 3, 1).view(B, H * W, C)
        #assert L == H * W, "input feature has wrong size"

        x_windows_rescaled = self.before_cross_attention_img1(rescaled)
        x_windows_rescaler = self.before_cross_attention_img2(rescaler)

        # W-MSA/SW-MSA
        attn_windows = self.cross_attn(x_windows_rescaled, x_windows_rescaler, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x1 = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x1 = shifted_x
        x1 = x1.view(B, H * W, C)

        x1 = x1.permute(0, 2, 1).view(B, C, H, W)

        return x1



class SwinFilterBlock(nn.Module):
    def __init__(self, in_dim, out_dim, input_resolution, num_heads, proj, device, rpe_mode, rpe_contextual_tensor, window_size, depth, norm):
        super(SwinFilterBlock,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=1,stride=1,padding=0,bias=True),
            norm(out_dim),
            nn.GELU()
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=1,stride=1,padding=0,bias=True),
            norm(out_dim),
            nn.GELU()
        )

        if rpe_mode is not None:
            relative_position_index = get_indices_2d(to_2tuple(min(window_size, input_resolution[0])))
        else:
            relative_position_index = None

        self.blocks = nn.ModuleList([
            SwinCrossAttention(dim=out_dim,
                                proj=proj,
                                same_key_query=True,
                                input_resolution=input_resolution,
                                num_heads=num_heads,
                                device=device,
                                relative_position_index=relative_position_index,
                                rpe_mode=rpe_mode,
                                rpe_contextual_tensor=rpe_contextual_tensor,
                                window_size=window_size,
                                shift_size=0 if (i % 2 == 0) else window_size // 2) 
                                for i in range(depth)])

        self.psi = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(out_dim),
            nn.Sigmoid()
        )
        self.abs_pos_encoding = PositionEmbeddingSine2d(in_dim//2, normalize=True)
        
    def forward(self, x, skip_co):
        B, C, H, W = x.shape
        pos = self.abs_pos_encoding(shape_util=(B, H, W), device=x.device)
        x = x + pos
        skip_co = skip_co + pos
        g1 = self.W_g(skip_co)
        x1 = self.W_x(x)
        for blk in self.blocks:
            g1 = blk(g1, x1)
        psi = self.psi(g1)

        filtered = skip_co * psi

        return filtered


class SwinFilterBlockIdentity(nn.Module):
    def __init__(self):
        super(SwinFilterBlockIdentity, self).__init__()
 
    def forward(self, x, skip_co):
        return skip_co



def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class CrossAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, proj, same_key_query, input_resolution, window_size, num_heads, rpe_mode, rpe_contextual_tensor, device, relative_position_index=None, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        self.qkv = None
        self.device=device
        self.same_key_query = same_key_query

        self.get_qkv_object_rescaled = get_qkv(proj=proj, num_heads=num_heads, dim=dim, qkv_bias=qkv_bias, window_size=window_size, input_resolution=input_resolution)
        self.get_qkv_object_rescaler = get_qkv(proj=proj, num_heads=num_heads, dim=dim, qkv_bias=qkv_bias, window_size=window_size, input_resolution=input_resolution)

        self.rpe_table = None
        self.q_rpe_table = None
        self.k_rpe_table = None
        self.v_rpe_table = None
        
        self.rpe_table_size = int((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1))
        if relative_position_index is not None:
            self.rpe_indices = relative_position_index
            if rpe_mode == 'contextual':
                if 'q' in rpe_contextual_tensor:
                    self.q_rpe_table = nn.Parameter(torch.zeros(num_heads, self.head_dim, self.rpe_table_size))  # nh, head_dim, 2*Wh-1 * 2*Ww-1
                    trunc_normal_(self.q_rpe_table, std=.02)
                if 'k' in rpe_contextual_tensor:
                    self.k_rpe_table = nn.Parameter(torch.zeros(num_heads, self.head_dim, self.rpe_table_size))  # nh, head_dim, 2*Wh-1 * 2*Ww-1
                    trunc_normal_(self.k_rpe_table, std=.02)
                if 'v' in rpe_contextual_tensor:
                    self.v_rpe_table = nn.Parameter(torch.zeros(num_heads, self.rpe_table_size, self.head_dim))  # nh, head_dim, 2*Wh-1 * 2*Ww-1
                    trunc_normal_(self.v_rpe_table, std=.02)
            elif rpe_mode == 'bias':
                self.rpe_table = nn.Parameter(torch.zeros(num_heads, self.rpe_table_size))  # nh, 2*Wh-1 * 2*Ww-1
                trunc_normal_(self.rpe_table, std=.02)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, rescaled, rescaler, mask=None):
        """
        Args:
            shape = shape of x before call to 'get_qkv'
        """
        B_, N, C = rescaled.shape

        q_rescaled, k_rescaled, v_rescaled = self.get_qkv_object_rescaled(rescaled)
        q_rescaler, k_rescaler, v_rescaler = self.get_qkv_object_rescaler(rescaler)

        if self.same_key_query:
            q, k, v = q_rescaler, k_rescaler, v_rescaled
        else:
            q, k, v = q_rescaler, k_rescaled, v_rescaled

        q = q * self.scale

        if self.rpe_table is not None:
            rpe = self.rpe_table[:, self.rpe_indices.view(-1)].view(-1, self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1])
            attn = (q @ k.transpose(-2, -1))
            attn = (attn + rpe)
        elif self.q_rpe_table is not None or self.k_rpe_table is not None:
            b = torch.zeros((B_, self.num_heads) + self.rpe_indices.shape, device=self.device)
            if self.q_rpe_table is not None:
                b_q = torch.matmul(q.view(B_, self.num_heads, self.window_size[0]**2, self.head_dim), self.q_rpe_table)
                b_q = b_q.flatten(2)[:, :, self.rpe_indices].view((B_, self.num_heads) + self.rpe_indices.shape)
                b += b_q
            if self.k_rpe_table is not None:
                b_k = torch.matmul(k.view(B_, self.num_heads, self.window_size[1]**2, self.head_dim), self.k_rpe_table)
                b_k = b_k.flatten(2)[:, :, self.rpe_indices].view((B_, self.num_heads) + self.rpe_indices.shape)
                b += b_k
            attn = (q @ k.transpose(-2, -1))
            attn = attn + b
        else:
            attn = (q @ k.transpose(-2, -1))

        #relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
        #    self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        #relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        #attn = attn + relative_position_bias.unsqueeze(0)


        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        if self.v_rpe_table is not None and self.rpe_table is None:
            w = self.v_rpe_table[:, self.rpe_indices.flatten()] # 24 49*49 32
            w = w.view(self.num_heads, self.window_size[0]**2, self.window_size[1]**2, self.head_dim)
            r_v = torch.matmul(attn.permute(1, 2, 0, 3), w) # 24 49 4 32
            r_v = r_v.permute(2, 1, 0, 3).contiguous().view(B_, self.window_size[0]**2, self.head_dim * self.num_heads)
            x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
            x = x + r_v
        else:
            x = (attn @ v).transpose(1, 2).reshape(B_, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops

class get_qkv(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, proj, dim, num_heads, window_size, input_resolution, qkv_bias=True):

        super().__init__()
        self.window_size = window_size
        self.input_resolution = input_resolution
        self.dim = dim
        self.proj_type = proj
        self.num_heads = num_heads
        if proj == 'conv':
            self.qkv = depthwise_separable_conv(dim, 3*dim)
        elif proj == 'linear':
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

    def forward(self, x):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        if self.proj_type == 'conv':
            x = x.view(B_, self.window_size[0], self.window_size[1], C)
            x = window_reverse(x, self.window_size[0], self.input_resolution[0], self.input_resolution[1]).permute(0, 3, 1, 2)
            qkv = self.qkv(x).permute(0, 2, 3, 1)
            qkv = window_partition(qkv, self.window_size[0])
            qkv = qkv.reshape(B_, N, -1)
        elif self.proj_type == 'linear':
            qkv = self.qkv(x)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        #qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        return q, k, v

    def extra_repr(self) -> str:
        return f'dim={self.dim}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, proj, same_key_query, use_conv_mlp, input_resolution, num_heads, device, relative_position_index, rpe_mode, rpe_contextual_tensor, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"
        
        self.before_cross_attention_img1 = BeforeCrossAttention(dim=dim, input_resolution=input_resolution, window_size=self.window_size, shift_size=self.shift_size)
        self.before_cross_attention_img2 = BeforeCrossAttention(dim=dim, input_resolution=input_resolution, window_size=self.window_size, shift_size=self.shift_size)
        self.cross_attn = CrossAttention(dim=dim, 
                                            same_key_query=same_key_query,
                                            input_resolution=input_resolution,
                                            proj=proj,
                                            window_size=to_2tuple(self.window_size), 
                                            num_heads=num_heads,
                                            device=device,
                                            relative_position_index=relative_position_index,
                                            rpe_mode=rpe_mode,
                                            rpe_contextual_tensor=rpe_contextual_tensor,
                                            qkv_bias=qkv_bias, 
                                            qk_scale=qk_scale, 
                                            attn_drop=attn_drop, 
                                            proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, input_resolution=input_resolution, use_conv=use_conv_mlp, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x1, x2):
        H, W = self.input_resolution
        B, L, C = x1.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x1

        x_windows_img1 = self.before_cross_attention_img1(x1)
        x_windows_img2 = self.before_cross_attention_img2(x2)

        # W-MSA/SW-MSA
        attn_windows = self.cross_attn(x_windows_img1, x_windows_img2, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x1 = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x1 = shifted_x
        x1 = x1.view(B, H * W, C)

        # FFN
        x1 = shortcut + self.drop_path(x1)
        x1 = x1 + self.drop_path(self.mlp(self.norm2(x1)))

        return x1

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops

class BeforeCrossAttention(nn.Module):
    def __init__(self, dim, input_resolution, window_size=7, shift_size=0, norm_layer=nn.LayerNorm):
        super().__init__()
        self.window_size = window_size
        self.input_resolution = input_resolution
        self.shift_size = shift_size
        self.norm1 = norm_layer(dim)
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"
    
    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        return x_windows

class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, swin_abs_pos, same_key_query, dim, proj, input_resolution, use_conv_mlp, depth, num_heads, device, rpe_mode, rpe_contextual_tensor, window_size=7,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.swin_abs_pos = swin_abs_pos

        if rpe_mode is not None:
            relative_position_index = get_indices_2d(to_2tuple(min(window_size, input_resolution[0])))
        else:
            relative_position_index = None

        self.pos_object = PositionEmbeddingSine2d(num_pos_feats=dim//2, normalize=True)

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, 
                                proj=proj,
                                same_key_query=same_key_query,
                                input_resolution=input_resolution,
                                num_heads=num_heads, 
                                use_conv_mlp=use_conv_mlp,
                                device=device,
                                relative_position_index=relative_position_index,
                                rpe_mode=rpe_mode,
                                rpe_contextual_tensor=rpe_contextual_tensor,
                                window_size=window_size,
                                shift_size=0 if (i % 2 == 0) else window_size // 2,
                                mlp_ratio=mlp_ratio,
                                qkv_bias=qkv_bias, 
                                qk_scale=qk_scale,
                                drop=drop, 
                                attn_drop=attn_drop,
                                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                norm_layer=norm_layer)
            for i in range(depth)])

        ## patch merging layer
        #if downsample is not None:
        #    self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        #else:
        #    self.downsample = None

    def forward(self, x1, x2):
        H, W = self.input_resolution
        B, L, C = x1.shape

        if self.swin_abs_pos:
            x1 = x1.permute(0, 2, 1).view(B, C, H, W)
            x2 = x2.permute(0, 2, 1).view(B, C, H, W)
            abs_pos = self.pos_object(shape_util=(B, H, W), device=x1.device)
            x1 = x1 + abs_pos
            x2 = x2 + abs_pos
            x1 = x1.permute(0, 2, 3, 1).view(B, L, C)
            x2 = x2.permute(0, 2, 3, 1).view(B, L, C)

        for blk in self.blocks:
            if self.use_checkpoint:
                x1 = checkpoint.checkpoint(blk, x1)
            else:
                x1 = blk(x1, x2)
        #layer_out = x
        #if self.downsample is not None:
        #    x = self.downsample(x)
        #return x, layer_out
        return x1

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops