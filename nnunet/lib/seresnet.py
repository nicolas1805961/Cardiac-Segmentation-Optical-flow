import torch.nn as nn
from timm.models.layers import DropPath
from math import log
from torch.nn.utils.parametrizations import spectral_norm
import torch
import matplotlib.pyplot as plt
import sys

class AdaIN(nn.Module):
    def __init__(self, input_resolution, latent_size, dim):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.linear = nn.Linear(latent_size, 2*dim)
        self.instance_norm = nn.InstanceNorm2d(dim)
    
    def forward(self, x, y):
        B, C = y.shape
        y = self.linear(y)
        y = y.view(B, 2, self.dim)
        H, W = self.input_resolution
        if x.dim() == 3:
            B, L, C = x.shape
            x = x.permute(0, 2, 1).view(B, C, H, W)
            gamma = y[:, 0, :][:, :, None, None]
            beta = y[:, 1, :][:, :, None, None]
            x = self.instance_norm(x)
            x = x * gamma + beta
            x = x.permute(0, 2, 3, 1).view(B, -1, C)
        else:
            gamma = y[:, 0, :][:, :, None, None]
            beta = y[:, 1, :][:, :, None, None]
            x = self.instance_norm(x)
            x = x * gamma + beta
        return x

class rescale_layer(nn.Module):
    def __init__(self, channel):
        super(rescale_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(channel, channel)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        y = self.linear(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class rescale_layer_1d(nn.Module):
    def __init__(self, channel):
        super(rescale_layer_1d, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.linear = nn.Linear(channel, channel)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        y = self.linear(y.transpose(-1, -2)).transpose(-1, -2)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class rescale_layer_big(nn.Module):
    def __init__(self, channel):
        super(rescale_layer_big, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(channel, channel**2)
        self.sigmoid = nn.Sigmoid()
        self.channel = channel

    def forward(self, x):
        B, C, H, W = x.shape
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        y = self.linear(y.squeeze(-1).transpose(-1, -2))
        y = y.view(1, self.channel, self.channel)

        # Multi-scale information fusion
        y = self.sigmoid(y)
        x = x.view(B, C, -1)
        out = torch.matmul(y, x)


        return out.view(B, C, H, W)


class rescale_layer_3d(nn.Module):
    def __init__(self, channel):
        super(rescale_layer_3d, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.linear = nn.Linear(channel, channel)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        y = self.linear(y.squeeze(-1).squeeze(-1).squeeze(-1))[:, :, None, None, None]

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        red = channel // reduction
        if red == 0:
            red = 1
        self.fc = nn.Sequential(
            nn.Linear(channel, red, bias=False),
            nn.GELU(),
            nn.Linear(red, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SELayer_3d(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer_3d, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.GELU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1)

def conv3x3_3d(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1)

class SEBasicBlock_3d(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16):
        super(SEBasicBlock_3d, self).__init__()
        self.conv1 = conv3x3_3d(inplanes, planes, stride)
        nb_groups = planes//16 if planes > 16 else 1
        self.bn1 = nn.GroupNorm(nb_groups, planes)
        self.gelu = nn.GELU()
        self.conv2 = conv3x3_3d(planes, planes, 1)
        self.bn2 = nn.GroupNorm(nb_groups, planes)
        self.se = SELayer_3d(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.gelu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.gelu(out)

        return out

class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, drop_path=0.03, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16):
        super(SEBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        nb_groups = planes//16 if planes > 16 else 1
        self.bn1 = nn.GroupNorm(nb_groups, planes)
        self.gelu = nn.GELU()
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.GroupNorm(nb_groups, planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.gelu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.gelu(out)

        #out = self.drop_path(out)

        return out

class AdaINECABasicBlock(nn.Module):
    expansion = 1

    def __init__(self, input_resolution, inplanes, planes, latent_size, drop_path=0., stride=1, downsample=None, b=1, gamma=2):
        super(AdaINECABasicBlock, self).__init__()

        t = int(abs((log(inplanes, 2) + b) / gamma))
        k_size = t if t%2 else t + 1

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.noise_weights1 = nn.Parameter(torch.zeros(planes))
        self.adain1 = AdaIN(input_resolution=input_resolution, latent_size=latent_size, dim=planes)
        self.gelu = nn.GELU()
        self.conv2 = conv3x3(planes, planes, 1)
        self.noise_weights2 = nn.Parameter(torch.zeros(planes))
        self.adain2 = AdaIN(input_resolution=input_resolution, latent_size=latent_size, dim=planes)
        self.eca = eca_layer(planes, k_size=k_size)
        self.downsample = downsample
        self.stride = stride
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, y):
        residual = x
        out = self.conv1(x)

        noise = torch.randn(out.size(0), 1, out.size(2), out.size(3), device=out.device, dtype=out.dtype)
        out = out + self.noise_weights1.view(1, -1, 1, 1) * noise

        out = self.adain1(out, y)
        out = self.gelu(out)

        out = self.conv2(out)

        noise = torch.randn(out.size(0), 1, out.size(2), out.size(3), device=out.device, dtype=out.dtype)
        out = out + self.noise_weights2.view(1, -1, 1, 1) * noise

        out = self.adain2(out, y)
        out = self.eca(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.gelu(out)

        out = self.drop_path(out)

        return out

class ECABasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, norm, drop_path=0., stride=1, downsample=None, b=1, gamma=2):
        super(ECABasicBlock, self).__init__()

        t = int(abs((log(inplanes, 2) + b) / gamma))
        k_size = t if t%2 else t + 1

        self.conv1 = conv3x3(inplanes, planes, stride)
        #nb_groups = planes//16 if planes > 16 else 1
        self.bn1 = norm(planes)
        self.gelu = nn.GELU()
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = norm(planes)
        self.eca = eca_layer(planes, k_size=k_size)
        self.downsample = downsample
        self.stride = stride
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.gelu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.eca(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.gelu(out)

        out = self.drop_path(out)

        return out


class RescaleBasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, drop_path=0., stride=1, downsample=None, b=1, gamma=2):
        super(RescaleBasicBlock3D, self).__init__()

        self.conv1 = conv3x3_3d(inplanes, planes, stride)
        #nb_groups = planes//16 if planes > 16 else 1
        self.bn1 = nn.BatchNorm3d(planes)
        self.gelu = nn.GELU()
        self.conv2 = conv3x3_3d(planes, planes, 1)
        self.bn2 = nn.BatchNorm3d(planes)
        self.rescale = rescale_layer_3d(planes)
        self.downsample = downsample
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.gelu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.rescale(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.gelu(out)

        out = self.drop_path(out)

        return out


class RescaleBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, norm, drop_path=0., stride=1, downsample=None, b=1, gamma=2):
        super(RescaleBasicBlock, self).__init__()

        self.conv1 = conv3x3(inplanes, planes, stride)
        #nb_groups = planes//16 if planes > 16 else 1
        self.bn1 = norm(planes)
        self.gelu = nn.GELU()
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = norm(planes)
        self.rescale = rescale_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.gelu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.rescale(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.gelu(out)

        out = self.drop_path(out)

        return out

class RescaleBasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, kernel_size, norm, drop_path=0., downsample=None):
        super(RescaleBasicBlock1D, self).__init__()

        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=kernel_size, padding='same')
        #nb_groups = planes//16 if planes > 16 else 1
        self.bn1 = norm(planes)
        self.gelu = nn.GELU()
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=kernel_size, padding='same')
        self.bn2 = norm(planes)
        self.rescale = rescale_layer_1d(planes)
        self.downsample = downsample
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.gelu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.rescale(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.gelu(out)

        out = self.drop_path(out)

        return out

class ResnetBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, norm, drop_path=0., stride=1, downsample=None):
        super(ResnetBasicBlock, self).__init__()

        self.conv1 = conv3x3(inplanes, planes, stride)
        #nb_groups = planes//16 if planes > 16 else 1
        self.bn1 = norm(planes)
        self.gelu = nn.GELU()
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = norm(planes)
        self.downsample = downsample
        self.stride = stride
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.gelu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.gelu(out)

        out = self.drop_path(out)

        return out


class ResnetBasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, kernel_size, norm, drop_path=0., downsample=None):
        super(ResnetBasicBlock1D, self).__init__()

        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=kernel_size, padding='same')
        #nb_groups = planes//16 if planes > 16 else 1
        self.bn1 = norm(planes)
        self.gelu = nn.GELU()
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=kernel_size, padding='same')
        self.bn2 = norm(planes)
        self.downsample = downsample
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.gelu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.gelu(out)

        out = self.drop_path(out)

        return out


class discriminator_eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(discriminator_eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = spectral_norm(nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

def discriminatorconv3x3(in_planes, out_planes, stride=1):
    return spectral_norm(nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False))


class DiscriminatorECABasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, drop_path=0., stride=1, downsample=None, b=1, gamma=2):
        super(DiscriminatorECABasicBlock, self).__init__()

        t = int(abs((log(inplanes, 2) + b) / gamma))
        k_size = t if t%2 else t + 1

        self.conv1 = discriminatorconv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        #nb_groups = planes//16 if planes > 16 else 1
        self.gelu = nn.GELU()
        self.conv2 = discriminatorconv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.eca = eca_layer(planes, k_size=k_size)
        self.downsample = downsample
        self.stride = stride
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.gelu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.eca(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.gelu(out)

        out = self.drop_path(out)

        return out