from math import nan
import torch
import torch.nn as nn
from nnunet.network_architecture.MTL_model import MTLmodel, PolicyNet
from nnunet.network_architecture.temporal_model import VideoModel
#from nnunet.network_architecture.Optical_flow_model import OpticalFlowModel
#from nnunet.network_architecture.Optical_flow_model_2 import OpticalFlowModel
from nnunet.network_architecture.Optical_flow_model_3 import OpticalFlowModel
from nnunet.network_architecture.Optical_flow_model_4 import OpticalFlowModel4
from nnunet.network_architecture.Optical_flow_model_label import OpticalFlowModelLabeled
from nnunet.network_architecture.Optical_flow_model_prediction import OpticalFlowModelPrediction
from nnunet.network_architecture.Optical_flow_model_simple import OpticalFlowModelSimple
from nnunet.network_architecture.Optical_flow_model_recursive import OpticalFlowModelRecursive
from nnunet.network_architecture.Optical_flow_model_variable_length import OpticalFlowModelVariableLength
from nnunet.network_architecture.Optical_flow_model_recursive_video import OpticalFlowModelRecursiveVideo
from nnunet.network_architecture.Optical_flow_model_lib import OpticalFlowModelLib
from nnunet.network_architecture.discriminator import Discriminator
from tqdm import tqdm
import logging
import os
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
#import yaml
from nnunet.torchinfo.torchinfo.torchinfo import summary
from .boundary_utils import one_hot, simplex
from scipy.spatial.distance import directed_hausdorff
from scipy.ndimage import distance_transform_edt as eucl_distance
from medpy import metric
from .loss import PerimeterLoss, ScaleLoss, TopkLoss3D, SurfaceLoss, GeneralizedDice, MyCrossEntropy, TopkLoss, FocalLoss
from . import swin_transformer_2
import segmentation_models_pytorch as smp
from . import adversarial_model_3d
from monai.losses import DiceLoss, DiceFocalLoss, GeneralizedDiceLoss, FocalLoss, DiceCELoss
from monai.transforms import KeepLargestConnectedComponent
from kornia.morphology import opening, closing
import torch.nn.functional as F
import math
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from skimage.transform import resize
from .encoder import Encoder
from copy import copy
from ruamel.yaml import YAML

def resample_logits_scipy(image, original_label):
    current_device = image.device
    image = image.squeeze()
    assert image.dim() == 3

    image = image.cpu().numpy()
    min_max_values = []
    for t in range(len(image)):
        max_value = image[t].max()
        min_value = image[t].min()
        min_max_values.append((min_value, max_value))
    
    image = resize(image, output_shape=(image.shape[0],) + original_label.shape, order=3)

    zoomed_image3 = []
    for idx, min_max in enumerate(min_max_values):
        temp = cv.normalize(image[idx], None, alpha=min_max[0], beta=min_max[1], norm_type=cv.NORM_MINMAX)
        zoomed_image3.append(temp)
    image = np.round(np.stack(zoomed_image3, axis=0))
    image = torch.from_numpy(image).unsqueeze(0).to(current_device)

    assert image.dim() == 4
    return image

def resample_logits(image, original_label):
    image = image.squeeze()
    assert image.dim() == 3

    my_max = torch.max(torch.flatten(image, start_dim=1), dim=-1)[0].unsqueeze(-1).unsqueeze(-1)
    my_min = torch.min(torch.flatten(image, start_dim=1), dim=-1)[0].unsqueeze(-1).unsqueeze(-1)
    
    image = F.interpolate(image.unsqueeze(0), size=original_label.shape, mode='bicubic', antialias=True, align_corners=False).squeeze()
    image = torch.clamp(image, my_min, my_max).unsqueeze(0)

    assert image.dim() == 4
    return image

def resample_softmax(image, original_label):
    assert simplex(image,  axis=1)
    image = image.squeeze()
    assert image.dim() == 3

    #my_max = torch.max(torch.flatten(image, start_dim=1), dim=-1)[0].unsqueeze(-1).unsqueeze(-1)
    #my_min = torch.min(torch.flatten(image, start_dim=1), dim=-1)[0].unsqueeze(-1).unsqueeze(-1)

    image = F.interpolate(image.unsqueeze(0), size=original_label.shape, mode='bicubic', antialias=True, align_corners=False).squeeze()
    image = image.unsqueeze(0)
    #image = torch.clamp(image, my_min, my_max).unsqueeze(0)

    #image = image.cpu().numpy()
    #class_images = []
    #dtype_data = image.dtype
    #for i in range(len(image)):
    #    class_images.append(resize(image[i].astype(float), original_label.shape, 2, "edge", anti_aliasing=False).astype(dtype_data))
    #image = np.stack(class_images, axis=0).astype(dtype_data)
    #image = torch.from_numpy(image).unsqueeze(0)

    #print(torch.unique(image.sum(dim=1)))

    assert simplex(image,  axis=1)

    assert image.dim() == 4
    return image

def min_max_normalization(image, new_min, new_max):
    image_min = image.min()
    image_max = image.max()
    new_image = (image - image_min) / (image_max - image_min) * (new_max - new_min) + new_min
    return new_image

def get_rotation_batched_matrices(angle):
    matrices = []
    for i in range(angle.size(0)):
        m = torch.tensor([[torch.cos(angle), -1.0*torch.sin(angle), 0.], 
                        [torch.sin(angle), torch.cos(angle), 0.], 
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

def transform_image(image, angle, tx, ty, scale, mode):
    for i in range(4 - image.dim()):
        image = image.unsqueeze(0)

    if torch.all(angle == 0) and torch.all(scale == 1):
        return image

    r = get_rotation_batched_matrices(angle)
    t = get_translation_batched_matrices(tx, ty)
    s = get_scaling_batched_matrices(1/scale)
    print(r)
    theta = torch.bmm(torch.bmm(t, r), s)[:, :-1]

    print(theta)
    print(theta.shape)
    print(image.size())

    grid = F.affine_grid(theta, image.size())
    image_min = torch.min(torch.flatten(image, start_dim=2), dim=-1)[0].unsqueeze(-1).unsqueeze(-1)
    image_max = torch.max(torch.flatten(image, start_dim=2), dim=-1)[0].unsqueeze(-1).unsqueeze(-1)
    transformed = F.grid_sample(image, grid, mode=mode)
    transformed = torch.clamp(transformed, image_min, image_max)
    
    return transformed

def remove_rv_last_slice(img):
    if np.count_nonzero(img == 1) > np.count_nonzero(img == 2) + np.count_nonzero(img == 3):
        img[img == 1] = 0
    return img


def choose_value(img, mask):
    nb_components, output, stats, centroids = cv.connectedComponentsWithStats(mask, cv.CV_8U)

    while nb_components > 1:
        sizes = stats[:, -1]
        sorted_indices = np.flip(np.argsort(sizes))
        components = np.arange(nb_components)[sorted_indices]
        sizes = sizes[sorted_indices]
        j = components[1]

        if sizes[1] < 10:
            np.putmask(img, (output == j).astype(bool), 0)
            np.putmask(mask, (output == j).astype(bool), 0)
            nb_components, output, stats, centroids = cv.connectedComponentsWithStats(mask, cv.CV_8U)
            continue

        blob_image = (output == j).astype(np.uint8)
        frontier_image = cv.morphologyEx(blob_image, cv.MORPH_DILATE, cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))).astype(np.uint8) - blob_image
        frontier_values = img[frontier_image.astype(bool)]

        value_list = []
        for u in range(4):
            value_list.append((frontier_values == u).sum())
        value_array = np.array(value_list)
        max_value = np.amax(value_array)
        value = np.max(np.argwhere(value_array == max_value))

        if max_value < value_array.sum() * (2 / 3):
            value = 2
        
        np.putmask(img, (output == j).astype(bool), value)
        np.putmask(mask, (output == j).astype(bool), 0)
        nb_components, output, stats, centroids = cv.connectedComponentsWithStats(mask, cv.CV_8U)
    return img


def get_largest_connected_component(img):
    for i in range(1, 4):
        current_img = (img == i).astype(np.uint8)
        nb_components, output, stats, centroids = cv.connectedComponentsWithStats(current_img, cv.CV_8U)

        while nb_components > 2:
            
            sizes = stats[:, -1]
            sorted_indices = np.flip(np.argsort(sizes))
            components = np.arange(nb_components)[sorted_indices]

            j = components[2]

            blob_image = (output == j).astype(np.uint8)
            frontier_image = cv.morphologyEx(blob_image, cv.MORPH_DILATE, cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))).astype(np.uint8) - blob_image
            frontier_values = img[frontier_image.astype(bool)]

            value_list = []
            for u in range(4):
                value_list.append((frontier_values == u).sum())
            value_array = np.array(value_list)
            max_value = np.amax(value_array)
            value = np.max(np.argwhere(value_array == max_value))

            np.putmask(img, (output == j).astype(bool), value)
            current_img = (img == i).astype(np.uint8)
            nb_components, output, stats, centroids = cv.connectedComponentsWithStats(current_img, cv.CV_8U)
            
    return img

def kmeans_model(input_data, zoomed_gt):
    model = KMeans(n_clusters=4)
    model.fit(input_data)

    cc = model.cluster_centers_
    temp = np.stack([np.argsort(cc[:, 0]), np.arange(4)], axis=1)
    temp = temp[temp[:, 0].argsort()]

    relabel = np.choose(model.labels_, temp[:, 1]).astype(np.uint16)
    label2 = relabel.reshape(zoomed_gt.shape)
    return label2

def choose_value(img, mask):
    nb_components, output, stats, centroids = cv.connectedComponentsWithStats(mask, cv.CV_8U)

    while nb_components > 1:
        sizes = stats[:, -1]
        sorted_indices = np.flip(np.argsort(sizes))
        components = np.arange(nb_components)[sorted_indices]
        sizes = sizes[sorted_indices]
        j = components[1]

        blob_image = (output == j).astype(np.uint8)
        frontier_image = cv.morphologyEx(blob_image, cv.MORPH_DILATE, cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))).astype(np.uint8) - blob_image
        frontier_values = img[frontier_image.astype(bool)]

        value_list = []
        for u in range(4):
            value_list.append((frontier_values == u).sum())
        value_array = np.array(value_list)
        max_value = np.amax(value_array)
        value = np.max(np.argwhere(value_array == max_value))
        
        if max_value < value_array.sum() * (2 / 3):
            value = 2

        np.putmask(img, (output == j).astype(bool), value)
        np.putmask(mask, (output == j).astype(bool), 0)
        nb_components, output, stats, centroids = cv.connectedComponentsWithStats(mask, cv.CV_8U)
    return img


def postprocess(label):
    slice_list = []
    for i in range(len(label)):

        if i == len(label) - 1:
            label[i, :, :] = remove_rv_last_slice(label[i, :, :])

        mask = (label[i, :, :] == 1).astype(np.uint8) - cv.morphologyEx((label[i, :, :] == 1).astype(np.uint8), cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))).astype(np.uint8)
        
        out = choose_value(label[i, :, :], mask)

        out = get_largest_connected_component(out)
        slice_list.append(out)
    
    label = np.stack(slice_list, axis=0)
    return label

def improve_label(zoomed_gt):
    zoomed_gt[zoomed_gt < 5e-4] = 0

    input_data = zoomed_gt.reshape(-1, 1)
    scaler = StandardScaler()
    input_data = scaler.fit_transform(input_data)

    label = kmeans_model(input_data, zoomed_gt)
    label = postprocess(label)
    return label

def resample(image, zoom):
    image = zoom(image, zoom, order=0)
    if image.shape[0] > image.shape[1]:
        image = np.rot90(image, axes=(1, 0))
    elif image.shape[0] == image.shape[1]:
        temp = cv.normalize(image, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_16S).astype(np.uint8)
        _, th2 = cv.threshold(temp, 0, 1, cv.THRESH_BINARY+cv.THRESH_OTSU)

        if np.count_nonzero(th2[:, :30]) == 0:
            image = np.rot90(image, axes=(1, 0))
    return image


def crop_image(centroid, small_image_size, image):
    big_shape = image.shape

    y1 = int(math.ceil(centroid[1] - (small_image_size / 2)))
    y2 = int(math.ceil(centroid[1] + (small_image_size / 2)))
    x1 = int(math.ceil(centroid[0] - (small_image_size / 2)))
    x2 = int(math.ceil(centroid[0] + (small_image_size / 2)))

    pad_top = y1
    pad_bottom = big_shape[-2] - y2
    pad_left = x1
    pad_right = big_shape[-1] - x2

    if y1 < 0:
        pad = (0, 0, abs(y1), 0)
        image = torch.nn.functional.pad(image, pad, mode='constant', value=0.0)
        #centroid[1] += int(math.ceil(abs(y1)))
        y1 = 0
        y2 = small_image_size
    if y2 - big_shape[-2] > 0:
        #print('pad_bottom')
        pad = (0, 0, 0, y2 - big_shape[-2])
        image = torch.nn.functional.pad(image, pad, mode='constant', value=0.0)
        y2 = image.shape[-2]
    if x1 < 0:
        #print('pad_left')
        pad = (abs(x1), 0, 0, 0)
        image = torch.nn.functional.pad(image, pad, mode='constant', value=0.0)
        #centroid[1] += int(math.ceil(abs(x1)))
        x1 = 0
        x2 = small_image_size
    if x2 - big_shape[-1] > 0:
        pad = (0, x2 - big_shape[-1], 0, 0)
        image = torch.nn.functional.pad(image, pad, mode='constant', value=0.0)
        x2 = image.shape[-1]

    image = image[y1:y2, x1:x2]

    return image, torch.tensor([pad_left, pad_right, pad_top, pad_bottom])

def translate(image, tx, ty, interpolation_mode):
    t = torch.tensor([[1, 0, tx], [0, 1, ty], [0, 0, 1]], device=image.device).float()

    theta = t[:-1]
    theta = theta.unsqueeze(0)

    for i in range(4 - image.dim()):
        image = image.unsqueeze(0)

    grid = F.affine_grid(theta, image.size())
    image_min = image.min()
    image_max = image.max()
    translated = F.grid_sample(image, grid, mode=interpolation_mode)
    translated = torch.clamp(translated, image_min, image_max)
    
    return translated

def rotate_image(x, angle, interpolation_mode):
    for i in range(4 - x.dim()):
        x = x.unsqueeze(0)
    
    if angle == 0:
        return x
    
    r = torch.tensor([[torch.cos(angle), -1.0*torch.sin(angle), 0.], [torch.sin(angle), torch.cos(angle), 0.], [0, 0, 1]], device=x.device).float()
    theta = r[:-1]
    theta = theta.unsqueeze(0)

    grid = F.affine_grid(theta, x.size())
    x_min = x.min()
    x_max = x.max()
    rotated = F.grid_sample(x, grid, mode=interpolation_mode)
    rotated = torch.clamp(rotated, x_min, x_max)

    return rotated

def revert(image, parameters, interpolation_mode, padding):
    z = torch.zeros(size=(1, 1), device=image.device)
    image = transform_image(image=image, angle=-parameters[:, 0], tx=z, ty=z, scale=1/parameters[:, 1], mode=interpolation_mode)
    assert image.dim() == 4

    pad_left = padding[0]
    pad_right = padding[1]
    pad_top = padding[2]
    pad_bottom = padding[3]
    if pad_left < 0:
        image = image[:, :, :, abs(pad_left):]
        pad_left = 0
    if pad_right < 0:
        image = image[:, :, :, :abs(pad_right)]
        pad_right = 0
    if pad_top < 0:
        image = image[:, :, abs(pad_top):, :]
        pad_top = 0
    if pad_bottom < 0:
        image = image[:, :, :abs(pad_bottom), :]
        pad_bottom = 0
    pad_sequence = (pad_left, pad_right, pad_top, pad_bottom)
    image = F.pad(image, pad_sequence, mode='constant')

    return image

def batched_distance_transform(seg, resolution=None, dtype=None):
    assert one_hot(torch.tensor(seg), axis=1)
    K: int = seg.shape[1]

    res = np.zeros_like(seg, dtype=dtype)
    for k in range(K):
        posmask = seg[:, k, :, :].astype(bool)

        if posmask.any():
            negmask = ~posmask
            res[:, k, :, :] = eucl_distance(negmask, sampling=resolution) * negmask \
                - (eucl_distance(posmask, sampling=resolution) - 1) * posmask
        # The idea is to leave blank the negative classes
        # since this is one-hot encoded, another class will supervise that pixel

    return torch.tensor(res, dtype=torch.float32)

def read_config(filename, middle, video):
    yaml = YAML()
    with open(filename) as file:
        config = yaml.load(file)
    #with open(filename) as file:
    #    config = yaml.load(file, Loader=yaml.FullLoader)

    if config['bottleneck'] == 'swin_3d' or config['bottleneck'] == 'vit_3d' or config['bottleneck'] == 'factorized':
        assert config['nb_frames'] > 1, "bottleneck mode 'swin_3d', 'vit_3d' and 'factorized' require nb_frames to be more than 1"
    if config['bottleneck'] == 'factorized':
        assert len(config['patch_size']) == 2, "bottleneck mode 'factorized' require len(patch_size) to be 2"
    if filename == 'lib_config.yaml':
        assert config['semi_supervised'] == False, "can not run in a semi supervised manner with the lib dataset"
    if config['semi_supervised'] == True:
        assert config['use_spatial_transformer'] == False, "Semi supervised model can not be used with spatial transformer"
    assert len(config['transformer_depth']) == len(config['num_heads']), "transformer_depth and num_heads must have the same size"
    return config

def read_config_video(filename):
    yaml = YAML()
    with open(filename) as file:
        config = yaml.load(file)

    if config['only_first']:
        assert not config['split']

    return config

def write_model_parameters(model):
    with open('model.txt', 'a') as out_file:
        model_dict = model.state_dict()
        out_file.write('*'*200)
        out_file.write('\n')
        for param_tensor in model_dict:
            out_file.write(f'{param_tensor} \t {model_dict[param_tensor].size()}\n')

    with open('parameters.txt', 'a') as out_file:
        params = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                if 'encoder.layers.0' in name or 'encoder.layers.1' in name or 'decoder.layers.3' in name or 'decoder.layers.4' in name or '.upsample_layers.3' in name or '.upsample_layers.4' in name or '.downsample_layers.0' in name or '.downsample_layers.1' in name:
                    params += param.numel()
                out_file.write(f'{name}\n')
                out_file.write(f'{param.size()}\n')
                out_file.write('*'*10)
                out_file.write('\n')
        out_file.write(f'{params}\n')
        out_file.write('/'*1000)
        out_file.write('\n')

def build_gan(config):

    # stochastic depth
    num_blocks = config['gan_conv_depth'] + config['gan_transformer_depth']
    my_iter = torch.cat([torch.tensor([0]), torch.cumsum(torch.tensor(num_blocks), dim=0)])
    dpr = [x.item() for x in torch.linspace(0, config['drop_path_rate'], sum(num_blocks))]
    dpr_discriminator = [dpr[my_iter[j]:my_iter[j+1]] for j in range(len(num_blocks))]
    dpr_generator = [x[::-1] for x in dpr_discriminator[::-1]]

    out_dim = 1 if config['image_or_label'] == 'image' else 4

    if config['conv_discriminator']:
        discriminator = SpectralConvDiscriminator(in_dim_proj=out_dim, blur=config['blur'], blur_kernel=config['blur_kernel'], dpr=dpr_discriminator, in_discriminator_dims=config['in_conv_discriminator_dims'], discriminator_conv_depth=config['conv_discriminator_depth']).to(config['device'])
    else:
        discriminator = Discriminator(blur=config['blur'], blur_kernel=config['blur_kernel'], device=config['device'], swin_layer_type=swin_gan, swin_abs_pos=config['swin_abs_pos'], in_discriminator_dims=config['in_discriminator_dims'], proj=config['proj'], discriminator_conv_depth=config['gan_conv_depth'], discriminator_transformer_depth=config['gan_transformer_depth'], transformer_type=config['transformer_type'], dpr=dpr_discriminator, rpe_mode=config['rpe_mode'], rpe_contextual_tensor=config['rpe_contextual_tensors'], patch_size=config['patch_size'], embed_dim=config['embed_dim'], discriminator_num_heads=config['gan_num_heads'], window_size=config['window_size'], drop_path_rate=config['drop_path_rate'], deep_supervision=config['deep_supervision']).to(config['device'])
    generator = Generator2(out_dim_proj=out_dim, blur=config['blur'], blur_kernel=config['blur_kernel'], latent_size=config['latent_size'], style_mixing_p=config['style_mixing_p'], mapping_function_nb_layers=config['mapping_function_nb_layers'], batch_size=config['batch_size'], device=config['device'], swin_layer_type=swin_gan, swin_abs_pos=config['swin_abs_pos'], in_generator_dims=config['in_generator_dims'], merge=config['merge'], proj=config['proj'], generator_conv_depth=config['gan_conv_depth'][::-1], generator_transformer_depth=config['gan_transformer_depth'][::-1], transformer_type=config['transformer_type'], dpr=dpr_generator, rpe_mode=config['rpe_mode'], rpe_contextual_tensor=config['rpe_contextual_tensors'], patch_size=config['patch_size'], embed_dim=config['embed_dim'], generator_num_heads=config['gan_num_heads'][::-1], window_size=config['window_size'], drop_path_rate=config['drop_path_rate'], deep_supervision=config['deep_supervision']).to(config['device'])
    return generator, discriminator

def build_spatial_transformer(config):
    if config['swin_layer_type'] =='double':
        swin_layer_type = swin_double_attn
    elif config['swin_layer_type'] == 'single':
        swin_layer_type = swin_transformer_2
    if config['big_image_size'] == 224:
        spatial_transformer = LocalisationNet3(blur=config['blur'], 
                                                blur_kernel=config['blur_kernel'], 
                                                mlp_intermediary_dim=config['mlp_intermediary_dim'], 
                                                deep_supervision=config['deep_supervision'],
                                                window_size=config['big_window_size'],
                                                merge=config['merge'],
                                                conv_depth=config['conv_depth'],
                                                transformer_depth=config['transformer_depth'],
                                                num_heads=config['num_heads'],
                                                transformer_type=config['transformer_type'],
                                                device=config['device'],
                                                in_dims=config['in_encoder_dims'],
                                                swin_abs_pos=config['swin_abs_pos'],
                                                swin_layer_type=swin_layer_type,
                                                proj=config['proj'],
                                                rpe_mode=config['rpe_mode'],
                                                img_size=config['big_image_size'],
                                                rpe_contextual_tensor=config['rpe_contextual_tensors'],
                                                num_bottleneck_layers=config['num_bottleneck_layers'],
                                                bottleneck_heads=config['bottleneck_heads'],
                                                drop_path_rate=config['drop_path_rate']).to(config['device'])
    else:
        spatial_transformer = LocalisationNet2(blur=config['blur'], 
                                                blur_kernel=config['blur_kernel'], 
                                                mlp_intermediary_dim=config['mlp_intermediary_dim'], 
                                                deep_supervision=config['deep_supervision'], 
                                                merge=config['merge'], 
                                                device=config['device'], 
                                                image_size=config['big_image_size'], 
                                                num_bottleneck_layers=config['num_bottleneck_layers'], 
                                                in_localizer_dims=config['in_localizer_dims'], 
                                                localizer_conv_depth=config['localizer_conv_depth'], 
                                                drop_path_rate=config['drop_path_rate']).to(config['device'])

    return spatial_transformer

def build_autoencoder(config):
    if config['swin_layer_type'] =='double':
        swin_layer_type = swin_double_attn
    elif config['swin_layer_type'] == 'single':
        swin_layer_type = swin_transformer_2
    
    image_size = 224 #config['big_image_size'] if config['binary'] else config['image_size']
    window_size = 7 #config['big_window_size'] if config['binary'] else config['window_size']
    model = Autoencoder(device=config['device'],
                        autoencoder_dim=config['autoencoder_dim'],
                        proj=config['proj'],
                        blur=config['blur'],
                        use_conv_mlp=config['use_conv_mlp'],
                        out_encoder_dims=config['autoencoder_out_encoder_dims'],
                        blur_kernel=config['blur_kernel'],
                        swin_abs_pos=config['swin_abs_pos'],
                        swin_layer_type=swin_layer_type,
                        batch_size=config['batch_size'], 
                        patch_size=config['patch_size'], 
                        window_size=window_size,
                        in_dims=config['autoencoder_in_encoder_dims'],
                        deep_supervision=config['deep_supervision'],
                        drop_path_rate=config['drop_path_rate'],
                        transformer_type=config['transformer_type'],
                        image_size=image_size,
                        conv_depth=config['autoencoder_conv_depth'],
                        transformer_depth=config['autoencoder_transformer_depth'],
                        num_heads=config['autoencoder_num_heads'], 
                        rpe_mode=config['rpe_mode'],
                        rpe_contextual_tensor=config['rpe_contextual_tensors']).to(config['device'])

    return model

#def build_discriminator(config, discriminator_type):
#    num_stages = len(config['conv_depth']) + len(config['transformer_depth'])
#    # stochastic depth
#    num_blocks = config['conv_depth'] + config['transformer_depth'] + [config['num_bottleneck_layers']]
#    my_iter = torch.cat([torch.tensor([0]), torch.cumsum(torch.tensor(num_blocks), dim=0)])
#    dpr = [x.item() for x in torch.linspace(0, 0.0, sum(num_blocks))]
#    dpr = [dpr[my_iter[j]:my_iter[j+1]] for j in range(len(num_blocks))]
#    dpr_encoder = dpr[:num_stages]
#
#    if discriminator_type == 'seg':
#        in_discriminator_dims = config['seg_in_discriminator_dims']
#    elif discriminator_type == 'rec':
#        in_discriminator_dims = config['rec_in_discriminator_dims']
#
#    discriminator = SpectralConvDiscriminator(blur=config['blur'], shortcut=config['shortcut'], blur_kernel=config['blur_kernel'], dpr=dpr_encoder, in_discriminator_dims=in_discriminator_dims, out_discriminator_dims=config['out_discriminator_dims'], discriminator_conv_depth=config['discriminator_depth']).to(config['device'])
#    return discriminator

def build_policy_net(config):
    policy_net = PolicyNet(blur=config['blur'], 
                            blur_kernel=config['blur_kernel'], 
                            shortcut=config['shortcut'],
                            patch_size=[4, 4],
                            window_size=7,
                            swin_abs_pos=config['swin_abs_pos'],
                            proj=config['proj'],
                            num_heads=config['num_heads'],
                            bottleneck_heads=config['bottleneck_heads'],
                            out_encoder_dims=config['out_encoder_dims'],
                            use_conv_mlp=config['use_conv_mlp'],
                            device=config['device'],
                            mlp_intermediary_dim=config['mlp_intermediary_dim'],
                            in_dims=config['in_encoder_dims'],
                            image_size=224,
                            conv_depth=config['conv_depth'],
                            transformer_depth=config['transformer_depth']).to(config['device'])
    return policy_net

def build_3d_model_crop(config):
    image_size, window_size = 160, [12, 10, 10]
    if config['model'] == 'conv':
        model = smp.Unet(encoder_name='resnet50', encoder_depth=5, encoder_weights=None, in_channels=1, classes=4, activation='softmax2d').to(config['device'])
    elif config['model'] == 'swin':
        model = adversarial_model_3d.my_3d_model(device=config['device'],
                                                        batch_size=config['batch_size'],
                                                        reconstruction=config['reconstruction'],
                                                        nb_nets=config['nb_nets'],
                                                        logits=False,
                                                        proj=config['proj'],
                                                        cropping_network=True,
                                                        mlp_intermediary_dim=config['mlp_intermediary_dim'],
                                                        shortcut=config['shortcut'],
                                                        blur=config['blur'],
                                                        use_conv_mlp=config['use_conv_mlp'],
                                                        concat_spatial_cross_attention=config['concat_spatial_cross_attention'],
                                                        encoder_attention_type=config['encoder_attention_type'],
                                                        spatial_cross_attention_num_heads=config['spatial_cross_attention_num_heads'],
                                                        merge=config['merge'],
                                                        out_encoder_dims=config['out_encoder_dims'],
                                                        blur_kernel=config['blur_kernel'],
                                                        swin_abs_pos=config['swin_abs_pos'],
                                                        embed_dim=config['embed_dim'], 
                                                        patch_size=config['patch_size'], 
                                                        window_size=window_size,
                                                        in_dims=config['in_encoder_dims'],
                                                        deep_supervision=config['deep_supervision'],
                                                        bottleneck=config['bottleneck'],
                                                        drop_path_rate=config['drop_path_rate'],
                                                        image_size=image_size,
                                                        conv_depth=config['conv_depth'],
                                                        transformer_depth=config['transformer_depth'],
                                                        num_heads=config['num_heads'], 
                                                        bottleneck_heads=config['bottleneck_heads'], 
                                                        num_bottleneck_layers=config['num_bottleneck_layers'], 
                                                        rpe_mode=config['rpe_mode'],
                                                        rpe_contextual_tensor=config['rpe_contextual_tensors']).to(config['device'])
        
        if config['load_weights']:
            load_weights(model, config, starting_layer_number=2)

    return model

def build_3d_model(config):
    image_size, window_size = 128, [12, 8, 8]
    if config['model'] == 'conv':
        model = smp.Unet(encoder_name='resnet50', encoder_depth=5, encoder_weights=None, in_channels=1, classes=4, activation='softmax2d').to(config['device'])
    elif config['model'] == 'swin':
        model = adversarial_model_3d.my_3d_model(device=config['device'],
                                                        batch_size=config['batch_size'],
                                                        reconstruction=config['reconstruction'],
                                                        nb_nets=config['nb_nets'],
                                                        logits=False,
                                                        proj=config['proj'],
                                                        cropping_network=False,
                                                        mlp_intermediary_dim=config['mlp_intermediary_dim'],
                                                        shortcut=config['shortcut'],
                                                        blur=config['blur'],
                                                        use_conv_mlp=config['use_conv_mlp'],
                                                        concat_spatial_cross_attention=config['concat_spatial_cross_attention'],
                                                        encoder_attention_type=config['encoder_attention_type'],
                                                        spatial_cross_attention_num_heads=config['spatial_cross_attention_num_heads'],
                                                        merge=config['merge'],
                                                        out_encoder_dims=config['out_encoder_dims'],
                                                        blur_kernel=config['blur_kernel'],
                                                        swin_abs_pos=config['swin_abs_pos'],
                                                        embed_dim=config['embed_dim'], 
                                                        patch_size=config['patch_size'], 
                                                        window_size=window_size,
                                                        in_dims=config['in_encoder_dims'],
                                                        deep_supervision=config['deep_supervision'],
                                                        bottleneck=config['bottleneck'],
                                                        drop_path_rate=config['drop_path_rate'],
                                                        image_size=image_size,
                                                        conv_depth=config['conv_depth'],
                                                        transformer_depth=config['transformer_depth'],
                                                        num_heads=config['num_heads'], 
                                                        bottleneck_heads=config['bottleneck_heads'], 
                                                        num_bottleneck_layers=config['num_bottleneck_layers'], 
                                                        rpe_mode=config['rpe_mode'],
                                                        rpe_contextual_tensor=config['rpe_contextual_tensors']).to(config['device'])
        
        if config['load_weights']:
            load_weights(model, config, starting_layer_number=2)

    return model

def build_3d_model_logits(config):
    image_size, window_size = 128, [9, 8, 8]
    if config['model'] == 'conv':
        model = smp.Unet(encoder_name='resnet50', encoder_depth=5, encoder_weights=None, in_channels=1, classes=4, activation='softmax2d').to(config['device'])
    elif config['model'] == 'swin':
        model = adversarial_model_3d.my_3d_model(device=config['device'],
                                                        batch_size=config['batch_size'],
                                                        reconstruction=config['reconstruction'],
                                                        nb_nets=config['nb_nets'],
                                                        logits=True,
                                                        proj=config['proj'],
                                                        cropping_network=False,
                                                        mlp_intermediary_dim=config['mlp_intermediary_dim'],
                                                        shortcut=config['shortcut'],
                                                        blur=config['blur'],
                                                        use_conv_mlp=config['use_conv_mlp'],
                                                        concat_spatial_cross_attention=config['concat_spatial_cross_attention'],
                                                        encoder_attention_type=config['encoder_attention_type'],
                                                        spatial_cross_attention_num_heads=config['spatial_cross_attention_num_heads'],
                                                        merge=config['merge'],
                                                        out_encoder_dims=config['out_encoder_dims'],
                                                        blur_kernel=config['blur_kernel'],
                                                        swin_abs_pos=config['swin_abs_pos'],
                                                        embed_dim=config['embed_dim'], 
                                                        patch_size=config['patch_size'], 
                                                        window_size=window_size,
                                                        in_dims=config['in_encoder_dims'],
                                                        deep_supervision=config['deep_supervision'],
                                                        bottleneck=config['bottleneck'],
                                                        drop_path_rate=config['drop_path_rate'],
                                                        image_size=image_size,
                                                        conv_depth=config['conv_depth'],
                                                        transformer_depth=config['transformer_depth'],
                                                        num_heads=config['num_heads'], 
                                                        bottleneck_heads=config['bottleneck_heads'], 
                                                        num_bottleneck_layers=config['num_bottleneck_layers'], 
                                                        rpe_mode=config['rpe_mode'],
                                                        rpe_contextual_tensor=config['rpe_contextual_tensors']).to(config['device'])
        
        if config['load_weights']:
            load_weights(model, config, starting_layer_number=2)

    return model

def build_confidence_network(config, alpha, image_size, window_size):
    discriminator_in_dim = copy(config['in_encoder_dims'])
    discriminator_in_dim = [int(alpha * x) for x in discriminator_in_dim]
    discriminator_in_dim[0] = 4

    confidence_network = ConfidenceNetwork(transformer_depth=config['transformer_depth'],
                                    spatial_cross_attention_num_heads=config['spatial_cross_attention_num_heads'],
                                    deep_supervision=config['deep_supervision'],
                                    blur=config['blur'],
                                    conv_depth=[1, 1, 1],
                                    filter_skip_co_segmentation=config['filter_skip_co_segmentation'],
                                    image_size=image_size,
                                    num_bottleneck_layers=config['num_bottleneck_layers'],
                                    drop_path_rate=config['drop_path_rate'],
                                    out_encoder_dims=[int(alpha * x) for x in config['out_encoder_dims']],
                                    norm=config['norm'],
                                    attention_map=config['attention_map'],
                                    shortcut=config['shortcut'],
                                    proj=config['proj'],
                                    use_conv_mlp=config['use_conv_mlp'],
                                    blur_kernel=config['blur_kernel'],
                                    device=config['device'],
                                    swin_abs_pos=config['swin_abs_pos'],
                                    in_encoder_dims=discriminator_in_dim,
                                    window_size=window_size,
                                    num_heads=config['num_heads'],
                                    rpe_contextual_tensors=config['rpe_contextual_tensors'],
                                    rpe_mode=config['rpe_mode'],
                                    bottleneck=config['bottleneck'],
                                    bottleneck_heads=int(alpha * config['bottleneck_heads']))

    return confidence_network

#def build_discriminator(config, conv_layer, alpha, in_channel, image_size):
#    discriminator_in_dim = copy(config['in_encoder_dims'])
#    discriminator_in_dim = [int(round((alpha * x) / 2) * 2) for x in discriminator_in_dim]
#    discriminator_in_dim[0] = in_channel
#
#    discriminator = Discriminator(conv_depth=config['conv_depth'],
#                                    image_size=image_size,
#                                    num_bottleneck_layers=config['num_bottleneck_layers'],
#                                    drop_path_rate=config['drop_path_rate'],
#                                    out_encoder_dims=[int(round((alpha * x) / 2) * 2) for x in config['out_encoder_dims']],
#                                    norm=config['norm'],
#                                    device=config['device'],
#                                    in_encoder_dims=discriminator_in_dim,
#                                    conv_layer=conv_layer,
#                                    bottleneck_heads=int(alpha * config['bottleneck_heads']))
#
#    return discriminator

def load_weights(model, path):
    model.load_state_dict(path)
    for name, param in model.named_parameters():
        if 'spatio_temporal' not in name:
            param.requires_grad = False

def build_video_model(config, conv_layer, conv_layer_1d, norm_2d, norm_1d, log_function, image_size, window_size):

    model = VideoModel(device=config['device'],
                    conv_layer_1d=conv_layer_1d,
                    merge_temporal_tokens=config['merge_temporal_tokens'],
                    conv_layer=conv_layer,
                    nb_zones=config['nb_zones'],
                    deformable_points=config['deformable_points'],
                    area_size=config['area_size'],
                    log_function=log_function,
                    video_length=config['video_length'],
                    nb_layers=config['nb_layers'],
                    norm_2d=norm_2d,
                    norm_1d=norm_1d,
                    nb_memory_bus=config['nb_memory_bus'],
                    proj_qkv=config['proj'],
                    filter_skip_co_segmentation=config['filter_skip_co_segmentation'],
                    use_conv_mlp=config['use_conv_mlp'],
                    similarity_down_scale=config['similarity_down_scale'],
                    concat_spatial_cross_attention=config['concat_spatial_cross_attention'],
                    spatial_cross_attention_num_heads=config['spatial_cross_attention_num_heads'],
                    out_encoder_dims=config['out_encoder_dims'],
                    window_size=window_size,
                    in_dims=config['in_encoder_dims'],
                    deep_supervision=config['deep_supervision'],
                    drop_path_rate=config['drop_path_rate'],
                    image_size=image_size,
                    conv_depth=config['conv_depth'],
                    num_heads=config['num_heads'], 
                    bottleneck_heads=config['bottleneck_heads'], 
                    num_bottleneck_layers=config['num_bottleneck_layers'])
        
    model = model.to(config['device'])

    #if config['feature_extractor']:
    #    model = load_weights(model)

    return model


def build_discriminator(config, conv_layer, norm, image_size):
    model = Discriminator(out_encoder_dims=config['discriminator_out_dims'],
                          device=config['device'],
                          in_dims=config['discriminator_in_dims'],
                          image_size=image_size,
                          conv_layer=conv_layer,
                          conv_depth=config['discriminator_depth'],
                          drop_path_rate=config['drop_path_rate'],
                          bottleneck_heads=config['bottleneck_heads'],
                          norm_2d=norm)
        
    model = model.to(config['device'])

    return model


def build_flow_model(config, conv_layer_2d, conv_layer_1d, norm_2d, norm_1d, image_size, log_function):

    model = OpticalFlowModel(deep_supervision=config['deep_supervision'],
             out_encoder_dims=config['out_encoder_dims'],
             device=config['device'],
             in_dims=config['in_encoder_dims'],
             nb_layers=config['nb_layers'],
             image_size=image_size,
             blackout=config['blackout'],
             num_bottleneck_layers=config['num_bottleneck_layers'],
             conv_layer_2d=conv_layer_2d,
             conv_layer_1d=conv_layer_1d,
             conv_depth=config['conv_depth'],
             bottleneck_heads=config['bottleneck_heads'],
             drop_path_rate=config['drop_path_rate'],
             log_function=log_function,
             nb_tokens=config['nb_tokens'],
             dot_multiplier=config['dot_multiplier'],
             norm_1d=norm_1d,
             norm_2d=norm_2d)
        
    model = model.to(config['device'])

    return model


def build_flow_model_4(config, conv_layer_2d, norm_2d, image_size, log_function):

    model = OpticalFlowModel4(deep_supervision=config['deep_supervision'],
             video_length=config['video_length'],
             split=config['split'],
             one_to_all=config['one_to_all'],
             all_to_all=config['all_to_all'],
             out_encoder_dims=config['out_encoder_dims'],
             inference_mode=config['inference_mode'],
             padding=config['padding'],
             device=config['device'],
             in_dims=config['in_encoder_dims'],
             nb_layers=config['nb_layers'],
             image_size=image_size,
             conv_layer_2d=conv_layer_2d,
             conv_depth=config['conv_depth'],
             bottleneck_heads=config['bottleneck_heads'],
             drop_path_rate=config['drop_path_rate'],
             log_function=log_function,
             only_first=config['only_first'],
             dot_multiplier=config['dot_multiplier'],
             temporal_kernel_size=config['temporal_kernel_size'],
             norm_2d=norm_2d)
        
    model = model.to(config['device'])

    return model


def build_flow_model_variable_length(config, image_size, log_function):

    model = OpticalFlowModelVariableLength(deep_supervision=config['deep_supervision'],
             split=config['split'],
             one_to_all=config['one_to_all'],
             all_to_all=config['all_to_all'],
             out_encoder_dims=config['out_encoder_dims'],
             inference_mode=config['inference_mode'],
             padding=config['padding'],
             in_dims=config['in_encoder_dims'],
             embedding_dim=config['embedding_dim'],
             nb_layers=config['nb_layers'],
             image_size=image_size,
             conv_depth=config['conv_depth'],
             bottleneck_heads=config['bottleneck_heads'],
             drop_path_rate=config['drop_path_rate'],
             log_function=log_function,
             only_first=config['only_first'],
             dot_multiplier=config['dot_multiplier'])
        
    model = model.to(config['device'])

    return model



def build_flow_model_prediction(config, image_size, log_function):

    model = OpticalFlowModelPrediction(deep_supervision=config['deep_supervision'],
             split=config['split'],
             one_to_all=config['one_to_all'],
             all_to_all=config['all_to_all'],
             out_encoder_dims=config['out_encoder_dims'],
             inference_mode=config['inference_mode'],
             nb_iters=config['nb_iters'],
             padding=config['padding'],
             in_dims=config['in_encoder_dims'],
             nb_layers=config['nb_layers'],
             image_size=image_size,
             conv_depth=config['conv_depth'],
             bottleneck_heads=config['bottleneck_heads'],
             drop_path_rate=config['drop_path_rate'],
             log_function=log_function,
             only_first=config['only_first'],
             dot_multiplier=config['dot_multiplier'])
        
    model = model.to(config['device'])

    return model



def build_flow_model_simple(config, image_size, log_function):

    model = OpticalFlowModelSimple(deep_supervision=config['deep_supervision'],
             split=config['split'],
             one_to_all=config['one_to_all'],
             all_to_all=config['all_to_all'],
             out_encoder_dims=config['out_encoder_dims'],
             inference_mode=config['inference_mode'],
             nb_iters=config['nb_iters'],
             padding=config['padding'],
             in_dims=config['in_encoder_dims'],
             nb_layers=config['nb_layers'],
             image_size=image_size,
             conv_depth=config['conv_depth'],
             bottleneck_heads=config['bottleneck_heads'],
             drop_path_rate=config['drop_path_rate'],
             log_function=log_function,
             only_first=config['only_first'],
             dot_multiplier=config['dot_multiplier'])
        
    model = model.to(config['device'])

    return model




def build_flow_model_recursive(config, image_size, log_function):

    model = OpticalFlowModelRecursive(deep_supervision=config['deep_supervision'],
             split=config['split'],
             one_to_all=config['one_to_all'],
             all_to_all=config['all_to_all'],
             out_encoder_dims=config['out_encoder_dims'],
             inference_mode=config['inference_mode'],
             padding=config['padding'],
             in_dims=config['in_encoder_dims'],
             nb_interp_frame=config['nb_interp_frame'],
             nb_layers=config['nb_layers'],
             image_size=image_size,
             conv_depth=config['conv_depth'],
             bottleneck_heads=config['bottleneck_heads'],
             drop_path_rate=config['drop_path_rate'],
             log_function=log_function,
             only_first=config['only_first'],
             dot_multiplier=config['dot_multiplier'])
        
    model = model.to(config['device'])

    return model



def build_flow_model_recursive_video(config, image_size, log_function):

    model = OpticalFlowModelRecursiveVideo(deep_supervision=config['deep_supervision'],
             split=config['split'],
             one_to_all=config['one_to_all'],
             all_to_all=config['all_to_all'],
             out_encoder_dims=config['out_encoder_dims'],
             inference_mode=config['inference_mode'],
             padding=config['padding'],
             in_dims=config['in_encoder_dims'],
             nb_interp_frame=config['nb_interp_frame'],
             nb_layers=config['nb_layers'],
             image_size=image_size,
             conv_depth=config['conv_depth'],
             bottleneck_heads=config['bottleneck_heads'],
             drop_path_rate=config['drop_path_rate'],
             log_function=log_function,
             only_first=config['only_first'],
             dot_multiplier=config['dot_multiplier'])
        
    model = model.to(config['device'])

    return model



def build_flow_model_lib(config, image_size, log_function):

    model = OpticalFlowModelLib(deep_supervision=config['deep_supervision'],
             split=config['split'],
             one_to_all=config['one_to_all'],
             all_to_all=config['all_to_all'],
             out_encoder_dims=config['out_encoder_dims'],
             inference_mode=config['inference_mode'],
             in_dims=config['in_encoder_dims'],
             nb_interp_frame=config['nb_interp_frame'],
             nb_layers=config['nb_layers'],
             image_size=image_size,
             video_length=config['video_length'],
             conv_depth=config['conv_depth'],
             bottleneck_heads=config['bottleneck_heads'],
             drop_path_rate=config['drop_path_rate'],
             log_function=log_function,
             only_first=config['only_first'],
             dot_multiplier=config['dot_multiplier'])
        
    model = model.to(config['device'])

    return model



def build_flow_model_labeled(config, conv_layer_2d, norm_2d, image_size, log_function):

    model = OpticalFlowModelLabeled(deep_supervision=config['deep_supervision'],
             only_first=config['only_first'],
             split=config['split'],
             one_to_all=config['one_to_all'],
             all_to_all=config['all_to_all'],
             out_encoder_dims=config['out_encoder_dims'],
             inference_mode=config['inference_mode'],
             padding=config['padding'],
             in_dims=config['in_encoder_dims'],
             nb_layers=config['nb_layers'],
             image_size=image_size,
             conv_layer_2d=conv_layer_2d,
             conv_depth=config['conv_depth'],
             bottleneck_heads=config['bottleneck_heads'],
             drop_path_rate=config['drop_path_rate'],
             log_function=log_function,
             dot_multiplier=config['dot_multiplier'],
             norm_2d=norm_2d)
        
    model = model.to(config['device'])

    return model


def build_flow_model_2(config, conv_layer, norm, image_size, log_function):

    model = OpticalFlowModel2(deep_supervision=config['deep_supervision'],
             out_encoder_dims=config['out_encoder_dims'],
             device=config['device'],
             in_dims=config['in_encoder_dims'],
             nb_layers=config['nb_layers'],
             video_length=config['video_length'],
             image_size=image_size,
             blackout=config['blackout'],
             num_bottleneck_layers=config['num_bottleneck_layers'],
             conv_layer=conv_layer,
             conv_depth=config['conv_depth'],
             bottleneck_heads=config['bottleneck_heads'],
             drop_path_rate=config['drop_path_rate'],
             log_function=log_function,
             nb_tokens=config['nb_tokens'],
             dot_multiplier=config['dot_multiplier'],
             norm_2d=norm)
        
    model = model.to(config['device'])

    return model


def build_2d_model(config, conv_layer, norm, log_function, image_size, window_size, middle, num_classes):

    #image_size = 128 if config['use_cropped_images'] else 224
    #window_size = 16 if config['use_cropped_images'] else 14

    if config['model'] == 'conv':
        model = smp.Unet(encoder_name='resnet50', encoder_depth=5, encoder_weights=None, in_channels=1, classes=4, activation=None).to(config['device'])
    elif config['model'] == 'swin':
        model = MTLmodel(device=config['device'],
                        mix_residual=config['mix_residual'],
                        nb_repeat=config['nb_repeat'],
                        v1=config['v1'],
                        middle_unlabeled=config['middle_unlabeled'],
                        registered_seg=config['registered_seg'],
                        conv_layer=conv_layer,
                        num_classes=num_classes,
                        transformer_bottleneck=config['transformer_bottleneck'],
                        one_vs_all=config['one_vs_all'],
                        middle_classification=config['middle_classification'],
                        separability=config['separability'],
                        adversarial_loss=config['adversarial_loss'],
                        log_function=log_function,
                        asymmetric_unet=config['asymmetric_unet'],
                        norm=norm,
                        affinity=config['affinity'],
                        add_extra_bottleneck_blocks=config['add_extra_bottleneck_blocks'],
                        middle=middle,
                        filter_skip_co_segmentation=config['filter_skip_co_segmentation'],
                        directional_field=config['directional_field'],
                        classification=config['classification'],
                        attention_map=config['attention_map'],
                        batch_size=config['batch_size'],
                        uncertainty_weighting=config['uncertainty_weighting'],
                        reconstruction=config['reconstruction'],
                        reconstruction_skip=config['reconstruction_skip'],
                        proj=config['proj'],
                        shortcut=config['shortcut'],
                        use_conv_mlp=config['use_conv_mlp'],
                        similarity_down_scale=config['similarity_down_scale'],
                        concat_spatial_cross_attention=config['concat_spatial_cross_attention'],
                        encoder_attention_type=config['encoder_attention_type'],
                        spatial_cross_attention_num_heads=config['spatial_cross_attention_num_heads'],
                        merge=config['merge'],
                        out_encoder_dims=config['out_encoder_dims'],
                        swin_abs_pos=config['swin_abs_pos'],
                        patch_size=config['patch_size'], 
                        window_size=window_size,
                        in_dims=config['in_encoder_dims'],
                        deep_supervision=config['deep_supervision'],
                        bottleneck=config['bottleneck'],
                        drop_path_rate=config['drop_path_rate'],
                        image_size=image_size,
                        conv_depth=config['conv_depth'],
                        transformer_depth=config['transformer_depth'],
                        num_heads=config['num_heads'], 
                        bottleneck_heads=config['bottleneck_heads'], 
                        num_bottleneck_layers=config['num_bottleneck_layers'], 
                        rpe_mode=config['rpe_mode'],
                        rpe_contextual_tensor=config['rpe_contextual_tensors'])
        
        model = model.to(config['device'])
        
        if config['load_weights']:
            load_weights(model, config, starting_layer_number=2)

    return model


def build_2d_model_crop(config):

    image_size = 160

    if config['model'] == 'conv':
        model = smp.Unet(encoder_name='resnet50', encoder_depth=5, encoder_weights=None, in_channels=1, classes=4, activation='softmax2d').to(config['device'])
    elif config['model'] == 'swin':
        model = MTLmodel(device=config['device'],
                        mlp_intermediary_dim=config['mlp_intermediary_dim'],
                        cropping_network=True,
                        binary=config['binary'],
                        attention_map=False,
                        uncertainty_weighting=config['uncertainty_weighting'],
                        batch_size=config['batch_size'],
                        reconstruction=config['reconstruction'],
                        reconstruction_skip=False,
                        proj=config['proj'],
                        shortcut=config['shortcut'],
                        blur=config['blur'],
                        use_conv_mlp=config['use_conv_mlp'],
                        similarity_down_scale=config['similarity_down_scale'],
                        concat_spatial_cross_attention=config['concat_spatial_cross_attention'],
                        reconstruction_attention_type=config['reconstruction_attention_type'],
                        encoder_attention_type=config['encoder_attention_type'],
                        spatial_cross_attention_num_heads=config['spatial_cross_attention_num_heads'],
                        merge=config['merge'],
                        out_encoder_dims=config['out_encoder_dims'],
                        blur_kernel=config['blur_kernel'],
                        swin_abs_pos=config['swin_abs_pos'],
                        patch_size=config['patch_size'], 
                        window_size=config['window_size'],
                        in_dims=config['in_encoder_dims'],
                        deep_supervision=config['deep_supervision'],
                        bottleneck=config['bottleneck'],
                        drop_path_rate=config['drop_path_rate'],
                        image_size=image_size,
                        conv_depth=config['conv_depth'],
                        transformer_depth=config['transformer_depth'],
                        num_heads=config['num_heads'], 
                        bottleneck_heads=config['bottleneck_heads'], 
                        num_bottleneck_layers=config['num_bottleneck_layers'], 
                        rpe_mode=config['rpe_mode'],
                        rpe_contextual_tensor=config['rpe_contextual_tensors']).to(config['device'])
        
        if config['load_weights']:
            load_weights(model, config, starting_layer_number=2)

    return model


def build_model_logits(config):

    image_size, window_size = 128, 8

    if config['model'] == 'conv':
        model = smp.Unet(encoder_name='resnet50', encoder_depth=5, encoder_weights=None, in_channels=1, classes=4, activation='softmax2d').to(config['device'])
    elif config['model'] == 'swin':
        model = adversarial_model_2d_logits.my_model(device=config['device'],
                                                    use_attention_map=config['use_attention_map'],
                                                    batch_size=config['batch_size'],
                                                    reconstruction=config['reconstruction'],
                                                    binary=config['binary'],
                                                    proj=config['proj'],
                                                    shortcut=config['shortcut'],
                                                    blur=config['blur'],
                                                    use_conv_mlp=config['use_conv_mlp'],
                                                    similarity_down_scale=config['similarity_down_scale'],
                                                    concat_spatial_cross_attention=config['concat_spatial_cross_attention'],
                                                    reconstruction_attention_type=config['reconstruction_attention_type'],
                                                    encoder_attention_type=config['encoder_attention_type'],
                                                    spatial_cross_attention_num_heads=config['spatial_cross_attention_num_heads'],
                                                    merge=config['merge'],
                                                    out_encoder_dims=config['out_encoder_dims'],
                                                    blur_kernel=config['blur_kernel'],
                                                    swin_abs_pos=config['swin_abs_pos'],
                                                    patch_size=config['patch_size'], 
                                                    window_size=window_size,
                                                    in_dims=config['in_encoder_dims'],
                                                    deep_supervision=config['deep_supervision'],
                                                    bottleneck=config['bottleneck'],
                                                    drop_path_rate=config['drop_path_rate'],
                                                    image_size=image_size,
                                                    conv_depth=config['conv_depth'],
                                                    transformer_depth=config['transformer_depth'],
                                                    num_heads=config['num_heads'], 
                                                    bottleneck_heads=config['bottleneck_heads'], 
                                                    num_bottleneck_layers=config['num_bottleneck_layers'], 
                                                    rpe_mode=config['rpe_mode'],
                                                    rpe_contextual_tensor=config['rpe_contextual_tensors']).to(config['device'])

    return model

def create_loggers(path):
    console_logger = logging.getLogger('console_logger')
    file_logger = logging.getLogger('file_logger')
    console_logger.setLevel(logging.INFO)
    file_logger.setLevel(logging.INFO)
    my_handler = TqdmLoggingHandler()
    fileHandler = logging.FileHandler(os.path.join(path, 'results.txt'), mode='a+', encoding="utf-8")
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fileHandler.setFormatter(formatter)
    console_logger.addHandler(my_handler)
    file_logger.addHandler(fileHandler)
    return console_logger, file_logger


def set_augmentations(config, module, img_size, autoencoder):
    data_augmentation_utils = []
    if config['noise_p'] > 0.0 and not autoencoder:
        data_augmentation_utils.append((module.GaussianNoise(shape=(img_size, img_size), std=config['std_noise']), config['noise_p'], 1))
    if config['brightness_p'] > 0.0:
        data_augmentation_utils.append((module.RandomBrightnessAdjust(config['brightness_range']), config['brightness_p'], 1))
    if config['rotation_p'] > 0.0:
        data_augmentation_utils.append((module.RandomRotation(config['rotation_degree']), config['rotation_p'], 2))
    if config['zoom_p'] > 0.0:
        data_augmentation_utils.append((module.RandomCenterZoom(config['zoom_scale']), config['zoom_p'], 2))
    if config['translate_p'] > 0.0:
        data_augmentation_utils.append((module.RandomTranslate(config['translate_scale']), config['translate_p'], 2))
    if config['flipv_p'] > 0.0:
        data_augmentation_utils.append((module.RandomVerticalFlip(), config['flipv_p'], 2))
    if config['fliph_p'] > 0.0:
        data_augmentation_utils.append((module.RandomHorizontalFlip(), config['fliph_p'], 2))
    if config['sharp_p'] > 0.0 and not autoencoder:
        data_augmentation_utils.append((module.RandomAdjustSharpness(config['sharp_range']), config['sharp_p'], 1))
    if config['gamma_p'] > 0.0 and not autoencoder:
        data_augmentation_utils.append((module.AdjustGamma(config['gamma_range']), config['gamma_p'], 1))
    if config['elastic_p'] > 0.0:
        data_augmentation_utils.append((module.RandomElasticDeformation(config['elastic_std']), config['elastic_p'], 2))
    if config['mixup_p'] > 0.0 and not autoencoder:
        data_augmentation_utils.append((module.Mixup(use_spatial_transformer=config['use_spatial_transformer'], device=config['device']), config['mixup_p'], 2))
    if config['cutmix_p'] > 0.0 and not autoencoder:
        data_augmentation_utils.append((module.Cutmix(use_spatial_transformer=config['use_spatial_transformer'], device=config['device']), config['cutmix_p'], 2))
    if config['cowmix_p'] > 0.0 and not autoencoder:
        data_augmentation_utils.append((module.Cowmix(config['cowmix_sigma_range'], config['cowmix_proportion_range'], config['cowmix_kernel_size'], use_spatial_transformer=config['use_spatial_transformer'], device=config['device']), config['cowmix_p'], 2))
    if config['my_augmentation_p'] > 0.0 and not autoencoder:
        data_augmentation_utils.append((module.MyAugment(use_spatial_transformer=config['use_spatial_transformer'], device=config['device']), config['my_augmentation_p'], 2))
    return data_augmentation_utils

def set_losses(config, add, loss_weights):
    losses = []
    loss_name = config['labeled_loss']
    if config['uncertainty_weighting']:
        loss = MyCrossEntropy(class_weights=loss_weights, ignore_index=0)
        losses.append({'loss': loss, 'weight': 1, 'add': 0})
    else:
        if loss_name == 'cross_entropy':
            loss = MyCrossEntropy(class_weights=loss_weights, ignore_index=-100)
            losses.append({'loss': loss, 'weight': 1, 'add': 0})
        elif loss_name == 'dice_and_perimeter':
            perimeter_loss = PerimeterLoss(device=config['device'])
            dice_loss = DiceLoss(include_background=False)
            losses.append({'loss': dice_loss, 'weight': 1 - config['lambda_start'], 'add': -add})
            losses.append({'loss': perimeter_loss, 'weight': config['lambda_start'], 'add': add})
        elif loss_name == 'dice':
            loss = DiceLoss(include_background=False)
            losses.append({'loss': loss, 'weight': 1, 'add': 0})
        elif loss_name == 'dice_and_boundary':
            dice_loss = DiceLoss(include_background=False)
            boundary_loss = SurfaceLoss()
            losses.append({'loss': dice_loss, 'weight': 1 - config['lambda_start'], 'add': -add})
            losses.append({'loss': boundary_loss, 'weight': config['lambda_start'], 'add': add})
        elif loss_name == 'generalized_dice_and_boundary':
            dice_loss = GeneralizedDice()
            boundary_loss = SurfaceLoss()
            losses.append({'loss': dice_loss, 'weight': 1 - config['lambda_start'], 'add': -add})
            losses.append({'loss': boundary_loss, 'weight': config['lambda_start'], 'add': add})
        elif loss_name == 'dice_cross_entropy':
            loss = DiceCELoss(include_background=False)
            losses.append({'loss': loss, 'weight': 1, 'add': 0})
        elif loss_name == 'generalized_dice_cross_entropy':
            loss1 = GeneralizedDiceLoss(include_background=False)
            loss2 = MyCrossEntropy(class_weights=loss_weights)
            losses.append({'loss': loss1, 'weight': 1, 'add': 0})
            losses.append({'loss': loss2, 'weight': 1, 'add': 0})
        elif loss_name == 'generalized_dice':
            loss = GeneralizedDiceLoss(include_background=False)
            losses.append({'loss': loss, 'weight': 1, 'add': 0})
        elif loss_name == 'topk_and_dice':
            topk = TopkLoss(class_weights=loss_weights, percent=config['topk_percent'])
            dice = DiceLoss(include_background=False, softmax=True)
            losses.append({'loss': topk, 'weight': 1, 'add': 0})
            losses.append({'loss': dice, 'weight': 1, 'add': 0})
        elif loss_name == 'topk_and_generalized_dice':
            topk = TopkLoss(class_weights=loss_weights, percent=config['topk_percent'])
            dice = GeneralizedDice()
            losses.append({'loss': topk, 'weight': 1, 'add': 0})
            losses.append({'loss': dice, 'weight': 1, 'add': 0})
        elif loss_name == 'focal_and_dice':
            loss = DiceFocalLoss(include_background=False, focal_weight=loss_weights[1:] if not config['binary'] else None, softmax=True)
            losses.append({'loss': loss, 'weight': 1, 'add': 0})
        elif loss_name == 'focal_and_generalized_dice':
            focal = FocalLoss(include_background=False)
            dice = GeneralizedDiceLoss(include_background=False)
            losses.append({'loss': focal, 'weight': 1, 'add': 0})
            losses.append({'loss': dice, 'weight': 1, 'add': 0})
        elif loss_name == 'scale_and_dice':
            scale_loss = ScaleLoss(class_weights=loss_weights, slope=config['slope'])
            dice_loss = DiceLoss(include_background=False, softmax=True)
            losses.append({'loss': scale_loss, 'weight': 1, 'add': 0})
            losses.append({'loss': dice_loss, 'weight': 1, 'add': 0})


    return losses

class Postprocessing2D():
    def __init__(self):
        self.keep_largest_component = KeepLargestConnectedComponent(applied_labels=[1, 2, 3])

    def __call__(self, pred):

        if pred.dim() == 3:
            temp = pred.unsqueeze(1)
        elif pred.dim() == 2:
            temp = pred[None, None, :, :]
        
        kernel = torch.ones(3, 3, device=pred.device)
        temp = closing(temp, kernel)
        pred = opening(temp, kernel)
        pred = self.keep_largest_component(pred.squeeze(0)).squeeze()

        return pred

def remove_padding(x, pred):
    temp = torch.flatten(x, start_dim=1).squeeze()
    mask = torch.all(temp == 0, dim=-1)
    pred = pred[:, :, ~mask]
    return x, pred

def rotate90(to_rotate, to_check_volume, axes):
    if to_check_volume.shape[-2] > to_check_volume.shape[-1]:
        plt.imshow(to_rotate, cmap='gray')
        plt.show()
        to_rotate = np.rot90(to_rotate, axes=axes)
        plt.imshow(to_rotate, cmap='gray')
        plt.show()
    elif to_check_volume.shape[-2] == to_check_volume.shape[-1]:
        temp = to_check_volume[-1]
        temp = cv.normalize(temp, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_16S).astype(np.uint8)
        _, th2 = cv.threshold(temp, 0, 1, cv.THRESH_BINARY+cv.THRESH_OTSU)
        if np.count_nonzero(th2[:, :30]) == 0:
            to_rotate = np.rot90(to_rotate, axes=axes)
    return to_rotate


class Postprocessing3D():
    def __init__(self):
        self.keep_largest_component = KeepLargestConnectedComponent(applied_labels=[1, 2, 3])

    def __call__(self, pred):

        kernel = torch.ones(3, 3, device=pred.device)
        for i in range(len(pred)):
            temp = pred[i][None, None, :, :]
            temp = closing(temp, kernel)
            pred[i] = opening(temp, kernel).squeeze()
        pred = self.keep_largest_component(pred.squeeze(0)).squeeze()
        return pred


class TqdmLoggingHandler(logging.StreamHandler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)

def get_validation_images_lib(model, val_random_dataloader, device):
    data = next(iter(val_random_dataloader))
    x = data['x'].permute((1, 0, 2, 3, 4))
    y = data['y'].permute((1, 0, 2, 3, 4))
    embedding = torch.nn.Embedding(4, 3)
    embedding.weight.data = torch.tensor([[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255]], device=device)
    pred = model(x)
    pred_in = pred[len(pred)//2][-1]
    input_image = x[len(pred)//2].squeeze()
    label = y[len(pred)//2]
    label = torch.argmax(label.squeeze(0), dim=0)
    pred_out = torch.argmax(pred_in.squeeze(0), dim=0)
    with torch.no_grad():
        label = embedding(label)
        pred_out = embedding(pred_out)
    input_image = cv.normalize(input_image.cpu().numpy(), None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F).astype(np.uint8)[..., np.newaxis]
    out_dict = {'pred': pred_out.cpu().numpy().astype(np.uint8), 'y': label.cpu().numpy().astype(np.uint8), 'x': input_image}
    return out_dict

def get_attention_image(model, val_random_dataloader, patch_size, device):
    x, y = next(iter(val_random_dataloader))
    attn = model.get_last_attention(x)
    mean_heads_attention = attn.mean(dim=1)
    attentions = nn.functional.interpolate(mean_heads_attention.unsqueeze(1), scale_factor=patch_size, mode="nearest")[0, 0]
    attentions = (attentions - torch.min(attentions)) / (torch.max(attentions) - torch.min(attentions))
    return attentions * x

def log_metrics(console_logger, file_logger, writer, metrics, iter_nb, location):
    for k, v in metrics.items():
        if v is not None:
            if type(v) is dict:
                writer.add_scalars(os.path.join(location, k).replace('\\', '/'), v, iter_nb)
            else:
                writer.add_scalar(os.path.join(location, k).replace('\\', '/'), v, iter_nb)
                console_logger.info(location + f" number {iter_nb}, " + k + f": {v}")
                file_logger.info(location + f" number {iter_nb}, " + k + f": {v}")
    file_logger.info("**************************************************")

def log_reinforcement_metrics(logger, writer, correct, iter_nb, location):
    logger.info(location + f" number {iter_nb}, Correct q_values predicted %: {correct}")
    writer.add_scalar(os.path.join(location, 'Validation correct q_values predicted %'), correct, iter_nb)

def log_ssim(logger, writer, ssim_metric, iter_nb, location):
    logger.info(location + f" number {iter_nb}, Average ssim: {ssim_metric}")
    writer.add_scalar(os.path.join(location, 'Validation ssim'), ssim_metric, iter_nb)

def log_gan_metrics(logger, writer, fid, iter_nb, location):
    logger.info(location + f" number {iter_nb}, Average fid: {fid}")
    writer.add_scalar(os.path.join(location, 'Validation average fid'), fid, iter_nb)

def get_class_hd(y_true, preds):
    class_hd = []
    nb_classes = y_true.size(0)
    assert nb_classes == 4
    for i in range(1, nb_classes):
        pred = preds[i].cpu().numpy()
        y = y_true[i].cpu().numpy()
        d1 = directed_hausdorff(y, pred)[0]
        d2 = directed_hausdorff(pred, y)[0]
        class_hd.append(max(d1, d2))
    return np.array(class_hd)
    #return {'average_hd': np.array(class_hd).mean(), 'bg_hd': class_hd[0], 'lv_hd': class_hd[1], 'rv_hd': class_hd[2], 'myo_hd': class_hd[3]}

#def calculate_metric_percase(pred, gt):
#    pred[pred > 0] = 1
#    gt[gt > 0] = 1
#    if pred.sum() > 0 and gt.sum()>0:
#        dice = metric.binary.dc(pred, gt)
#        hd95 = metric.binary.hd95(pred, gt)
#        return dice, hd95
#    elif pred.sum() > 0 and gt.sum()==0:
#        return 1, 0
#    else:
#        return 0, 0

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() == 0 and gt.sum() == 0:
        return 1, 0
    else:
        dice = metric.binary.dc(pred, gt)
        if pred.sum() == 0 or gt.sum() == 0:
            hd = np.nan
        else:
            hd = metric.binary.hd(pred, gt)
        return dice, hd


def get_metrics(y_true, preds, nb_classes):
    assert preds.size() == y_true.size(), 'predict {} & target {} shape do not match'.format(preds.size(), y_true.size())
    with torch.no_grad():
        preds = preds.cpu().detach().numpy()
        y_true = y_true.cpu().detach().numpy()
    #class_hd = get_class_hd(y_true, preds)
    class_dice = []
    class_hds = []
    for i in range(1, nb_classes):
        #y_true_temp = y_true[i]
        #pred_temp = preds[i]
        #dice = f1_score(y_true_temp.flatten().cpu().numpy(), pred_temp.flatten().cpu().numpy())
        dice, hd = calculate_metric_percase(preds == i, y_true == i)
        class_dice.append(dice)
        class_hds.append(hd)
    #y_true = torch.argmax(y_true, dim=0)
    #class_dice = f1_score(y_true.flatten().cpu().numpy(), preds.flatten().cpu().numpy(), average=None)[1:]
    return torch.tensor(class_dice).reshape(1, nb_classes - 1), torch.tensor(class_hds).reshape(1, nb_classes - 1)
    #metric_dict = {'average_dice': class_dice.mean(), 'bg_dice': class_dice[0], 'lv_dice': class_dice[1], 'rv_dice': class_dice[2], 'myo_dice': class_dice[3]}
    #return metric_dict.update(hd_dict)