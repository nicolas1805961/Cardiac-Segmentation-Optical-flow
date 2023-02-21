import torch
import cv2 as cv
import numpy as np
import os
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import random
import torchvision.transforms.functional as TF
from scipy.ndimage import distance_transform_edt as eucl_distance
from .boundary_utils import one_hot
from torch.utils.data import Sampler
import torch.nn.functional as F
import math
from skimage.measure import regionprops
import sys
from monai.transforms import NormalizeIntensity
from .training_utils import crop_image, resample, transform_image

def get_distance_image(label, norm):

    h, w = label.shape

    accumulation = np.zeros((2, h, w), dtype=np.float32)
    for t in range(1, 4):
        current_class = (label == t).astype(np.uint8)
        dst, labels = cv.distanceTransformWithLabels(current_class, cv.DIST_L2, cv.DIST_MASK_PRECISE, labelType=cv.DIST_LABEL_PIXEL)
        # labels is a LABEL map indicating LABEL (not index) of nearest zero pixel. Zero pixels have different labels.
        #  As a result som labels in backgound and in heart structure can have the same label.
        index = np.copy(labels)
        index[current_class > 0] = 0
        place = np.argwhere(index > 0) # get coords of background pixels
        nearCord = place[labels-1,:] # get coords of nearest zero pixel of EVERY pixels of the image. For background this is current coords.
        nearPixel = np.transpose(nearCord, axes=(2, 0, 1))
        grid = np.indices(current_class.shape).astype(float)
        diff = grid - nearPixel

        if norm:
            dr = np.sqrt(np.sum(diff**2, axis = 0))
        else:
            dr = np.ones_like(current_class)

        direction = np.zeros((2, h, w), dtype=np.float32)
        direction[0, current_class>0] = np.divide(diff[0, current_class>0], dr[current_class>0])
        direction[1, current_class>0] = np.divide(diff[1, current_class>0], dr[current_class>0])

        accumulation[:, current_class>0] = 0
        accumulation = accumulation + direction

        #fig, ax = plt.subplots(1, 5)
        #ax[0].imshow(labels, cmap='gray')
        #ax[1].imshow(index, cmap='gray')
        #ax[2].imshow(nearPixel[0], cmap='gray')
        #ax[3].imshow(diff[0], cmap='gray')
        #ax[4].imshow(accumulation[0], cmap='gray')
        #plt.show()
        #plt.waitforbuttonpress()
        #plt.close(fig)

    assert accumulation.max() <= 1.0 and accumulation.min() >= -1.0
    return torch.from_numpy(accumulation)

def get_parameters_2d(max_label, target_ratio):
    label_myo_lv = (max_label > 1).astype(np.uint8)
    label_rv = (max_label == 1).astype(np.uint8)
    if np.count_nonzero(label_myo_lv) > 0 and np.count_nonzero(label_rv) > 0:
        myo_lv_center = list(regionprops(label_myo_lv.astype(int))[0].centroid)
        rv_center = list(regionprops(label_rv.astype(int))[0].centroid)
        tx = -2 * ((max_label.shape[-1] / 2) - myo_lv_center[1]) / (max_label.shape[-1])
        ty = -2 * ((max_label.shape[-2] / 2) - myo_lv_center[0]) / (max_label.shape[-2])

        m = ((max_label.shape[-2] - rv_center[0]) - (max_label.shape[-2] - myo_lv_center[0])) / (rv_center[1] - myo_lv_center[1])
        angle = math.degrees(math.atan(m))
        if angle > 0:
            angle = math.degrees(np.pi - math.atan(m))
        else:
            angle = math.degrees(abs(math.atan(m)))
        angle = math.radians(angle)
    else:
        angle = 0
        tx = 0
        ty = 0
    positive_label = (max_label > 0).astype(bool)
    if np.count_nonzero(positive_label) > 0:
        ratio = np.sum(positive_label) / 128**2
        scale = math.sqrt(target_ratio / ratio)
    else:
        scale = 0
    #label_1[label_1 > 0] = 1
#
    #if len(np.unique(max_label)) > 3:
    #    center = list(regionprops(label_1.astype(int))[0].centroid)
    #    tx = -2 * ((max_label.shape[-1] / 2) - center[1]) / (max_label.shape[-1])
    #    ty = -2 * ((max_label.shape[-2] / 2) - center[0]) / (max_label.shape[-2])
#
    #    rv_centroid, lv_centroid = get_rv_centroid(max_label)
    #    m = ((max_label.shape[-2] - rv_centroid[0]) - (max_label.shape[-2] - center[0])) / (rv_centroid[1] - center[1])
    #    angle = math.degrees(math.atan(m))
    #    if angle > 0:
    #        angle = math.degrees(np.pi - math.atan(m))
    #    else:
    #        angle = math.degrees(abs(math.atan(m)))
    #    angle = math.radians(angle)
    #else:
    #    angle = 0
    #    tx = 0
    #    ty = 0
    #    #angle = 1.8939664
    parameters = [angle, tx, ty, scale]
    return parameters 

def get_parameters_3d(label_volume):
    index = get_max_index(torch.clone(label_volume))
    max_label = label_volume[index].numpy()
    label_1 = np.copy(max_label)
    label_1[label_1 > 0] = 1

    center = list(regionprops(label_1.astype(int))[0].centroid)
    tx = -2 * ((max_label.shape[-1] / 2) - center[1]) / (max_label.shape[-1])
    ty = -2 * ((max_label.shape[-2] / 2) - center[0]) / (max_label.shape[-2])

    if len(np.unique(max_label)) > 3:
        rv_centroid, lv_centroid = get_rv_centroid(max_label)
        m = ((max_label.shape[-2] - rv_centroid[0]) - (max_label.shape[-2] - center[0])) / (rv_centroid[1] - center[1])
        angle = math.degrees(math.atan(m))
        if angle > 0:
            angle = math.degrees(np.pi - math.atan(m))
        else:
            angle = math.degrees(abs(math.atan(m)))
        angle = math.radians(angle)
    else:
        angle = 1.8939664
    parameters = [angle, tx, ty]
    return parameters 

def get_max_index(volume_pred):
    volume_pred[volume_pred > 0] = 1
    volume_pred = torch.flatten(volume_pred, start_dim=1)
    volume_pred = torch.sum(volume_pred, dim=-1)
    return torch.argmax(volume_pred, dim=0)

def get_angle(label_volume):
    index = get_max_index(torch.clone(label_volume))
    max_label = label_volume[index].numpy()
    if len(torch.unique(label_volume)) > 3:
        angle = get_max_angle(max_label)
    else:
        angle = 1.8939664
    return angle

def get_rv_centroid(label):
    regions = regionprops(label.astype(int))
    data = [(regions[i].eccentricity, regions[i].centroid) for i in range(len(regions))]
    data = sorted(data, key=lambda x: x[0])
    rv_centroid = data[-1][1]
    lv_centroid = data[0][1]

    return rv_centroid, lv_centroid

def get_center(label):
    label_1 = np.copy(label)
    label_1[label_1 > 0] = 1

    center = list(regionprops(label_1.astype(int))[0].centroid)
    tx = -2 * ((label.shape[-1] / 2) - center[1]) / (label.shape[-1])
    ty = -2 * ((label.shape[-2] / 2) - center[0]) / (label.shape[-2])

    return center
    

def get_max_angle(label):

    rv_centroid, lv_centroid = get_rv_centroid(label)

    m = ((label.shape[-1] - rv_centroid[0]) - (label.shape[-1] - lv_centroid[0])) / (rv_centroid[1] - lv_centroid[1])

    angle = math.degrees(math.atan(m))
    if angle > 0:
        angle = math.degrees(np.pi - math.atan(m))
    else:
        angle = math.degrees(abs(math.atan(m)))
    angle = math.radians(angle)
    return angle 

def rotate_image(x, angle, interpolation_mode):
    for i in range(4 - x.dim()):
        x = x.unsqueeze(0)
    
    if angle == 0:
        return x    

    r = torch.tensor([[torch.cos(angle), -1.0*torch.sin(angle), 0.], [torch.sin(angle), torch.cos(angle), 0.], [0, 0, 1]], device=x.device).float()
    theta = r[:-1]
    theta = theta.unsqueeze(0)

    grid = F.affine_grid(theta, x.size())
    rotated = F.grid_sample(x, grid, mode=interpolation_mode)
    return rotated

def rand_bbox(size, lam):
    W = size[-1]
    H = size[-2]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def normalize_0_1(img):
    img_max = img.max()
    img_min = img.min()
    img = (img - img_min) / (img_max - img_min + 1e-8)
    return img

def normalize_range(img, new_min, new_max):
    img_max = torch.max(img)
    img_min = torch.min(img)
    img = (img - img_min) * ((new_max - new_min) / (img_max - img_min)) + new_min
    return img

def normalize_0_1_autoencoder(img):
    img_max = 3
    img_min = 0
    return (img - img_min) / (img_max - img_min)

def normalize_translation(x):
    return x + 1 / 2

def normalize_rotation(x):
    return x / 3.1378278068987537

def normalize_scale(x):
    return (x - 0.0447) / (1.7744 - 0.0447)

def standardize(img, mean, std):
    return (img - mean) / std

def distance_transform(seg, resolution=None, dtype=None):
    assert one_hot(torch.tensor(seg), axis=0)
    K: int = len(seg)

    res = np.zeros_like(seg, dtype=dtype)
    for k in range(K):
        posmask = seg[k].astype(bool)

        if posmask.any():
            negmask = ~posmask
            res[k] = eucl_distance(negmask, sampling=resolution) * negmask \
                - (eucl_distance(posmask, sampling=resolution) - 1) * posmask
        # The idea is to leave blank the negative classes
        # since this is one-hot encoded, another class will supervise that pixel

    return torch.tensor(res, dtype=torch.float32)

def rearrange_channels(x):
    x[x == 3] = 4
    x[x == 1] = 3
    x[x == 4] = 1
    return x

def set_angle(label_3, angle, tx, ty):
    r = torch.tensor([[np.cos(angle), -1.0*np.sin(angle), 0.], [np.sin(angle), np.cos(angle), 0.], [0, 0, 1]]).float()
    t = torch.tensor([[1, 0, tx], [0, 1, ty], [0, 0, 1]]).float()
    s = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).float()
    theta = torch.mm(torch.mm(t, r), s)[:-1]
    theta = theta.unsqueeze(0)

    grid = F.affine_grid(theta, (1, 1,) + label_3.size())
    x2 = F.grid_sample(label_3[None, None], grid, mode='nearest')

    regions = regionprops(x2[0, 0].numpy().astype(int))
    if regions[0].centroid[1] > regions[1].centroid[1]:
        angle -= np.pi
    return angle

def set_metadata(label, img_size, device):
    label_3 = torch.argmax(label, dim=0).float()
    tx = 0
    ty = 0
    scale = 0
    angle = 0
    if len(torch.unique(label_3)) > 1:
        label_1 = torch.clone(label_3)
        label_1[label_1 > 0] = 1

        region = regionprops(label_1.numpy().astype(int))[0]

        a = (np.array(list(region.centroid)) - (label_1.size(-1) // 2))[::-1] / (label_1.size(-1) // 2)

        tx = a[0]
        ty = a[1]

        if 1 in torch.unique(label_3) and len(torch.unique(label_3)) > 2:
            rv_centroid = regionprops(label_3.numpy().astype(int))[0].centroid
            center = regionprops(label_1.numpy().astype(int))[0].centroid
            m = ((label_3.size(-1) - rv_centroid[0]) - (label_3.size(-1) - center[0])) / (rv_centroid[1] - center[1])
            angle = np.pi - math.atan(m)
            angle = set_angle(label_3, angle, tx, ty)

        out_area = (region.area / img_size**2)

        group = torch.linspace(0.01, 4, 1000000)

        while(True):
            index = len(group) // 2
            scale = group[index]

            r = torch.tensor([[np.cos(angle), -1.0*np.sin(angle), 0.], [np.sin(angle), np.cos(angle), 0.], [0, 0, 1]]).float()
            t = torch.tensor([[1, 0, tx], [0, 1, ty], [0, 0, 1]]).float()
            s = torch.tensor([[scale, 0, 0], [0, scale, 0], [0, 0, 1]]).float()
            theta = torch.mm(torch.mm(t, r), s)[:-1]
            theta = theta.unsqueeze(0)

            grid = F.affine_grid(theta, (1, 1, img_size, img_size))
            x2 = F.grid_sample(label_1[None, None], grid, mode='nearest')

            region = regionprops(x2[0, 0].numpy().astype(int))[0]
            out_area = (region.area / img_size**2)

            if abs(out_area - 0.1) < 0.001:
                break
            if out_area - 0.1 < 0:
                group = group[:index]
            if out_area - 0.1 > 0:
                group = group[index:]
        
    metadata = {'tx': torch.tensor(tx).float().to(device), 'ty': torch.tensor(ty).float().to(device), 'scale': torch.tensor(scale).float().to(device), 'angle': torch.tensor(angle).float().to(device)}
    return metadata

def standardize_parameters(parameters):
    parameters[0] = (parameters[0] - (-77.891)) / 33.006
    parameters[1] = (parameters[1] - (7.983)) / 24.212
    parameters[2] = (parameters[2] - (13.900)) / 16.783
    return parameters

def unstandardize_parameters(parameters):
    angle = int((parameters[0] * 33.006) + (-77.891))
    tx = int((parameters[1] * 24.212) + (7.983))
    ty = int((parameters[2] * 16.783) + (13.900))
    return torch.tensor([angle, tx, ty])

def update_center(center, crop_size, img_shape):
    row = center[0]
    col = center[1]
    row_margin = (img_shape[0] - crop_size) // 2
    col_margin = (img_shape[1] - crop_size) // 2
    new_row = row - row_margin
    new_col = col - col_margin
    return [new_row, new_col]

def process_labeled_image_3d_crop(path_list, img_size, augmentations, target_ratio, binary, learn_transforms, directional_field, apply_clahe):
    #print(augmentations)
    if augmentations is not None:
        [x.reset() for x in augmentations]
    out = {}
    x_list = []
    image_list = []
    label_list = []
    y_list = []
    original_label_list = []
    original_image_list = []
    parameter_list = []
    df_list = []
    for idx, path in enumerate(path_list):
        out = process_2d_image_crop_tool(path, augmentations, img_size, target_ratio, binary, learn_transforms, directional_field, apply_clahe)

        x_list.append(out['x'])
        y_list.append(out['y'])
        image_list.append(out['image'])
        original_label_list.append(out['original_label'])
        original_image_list.append(out['original_image'])
        label_list.append(out['label'])
        if learn_transforms:
            parameter_list.append(out['parameters'])
        if directional_field:
            df_list.append(out['directional_field'])
    
    x = torch.stack(x_list, dim=1)
    y = torch.stack(y_list, dim=1)
    image = torch.stack(image_list, dim=0)
    original_image = torch.stack(original_image_list, dim=0)
    original_label = torch.stack(original_label_list, dim=0)
    label = torch.stack(label_list, dim=0)
    if learn_transforms:
        parameters = torch.stack(parameter_list, dim=0)
        out['parameters'] = parameters
    if directional_field:
        df = torch.stack(df_list, dim=1)
        out['directional_field'] = df
    


    #parameters = get_parameters_3d(torch.argmax(label_volume, dim=0))

    #for i in range(image_out_volume.shape[1]):
    #    fig, ax = plt.subplots(1, 2)
    #    ax[0].imshow(image_out_volume[0, i].numpy(), cmap='gray')
    #    ax[1].imshow(torch.argmax(label_volume, dim=0, keepdim=True).numpy()[0, i], cmap='gray')
    #    #figManager = plt.get_current_fig_manager()
    #    #figManager.window.showMaximized()
    #    plt.show()

    out['x'] = x
    out['y'] = y
    out['original_label'] = original_label
    out['original_image'] = original_image
    out['image'] = image
    out['label'] = label

    return out


def process_labeled_image_3d(path_list, img_size, augmentations, directional_field, apply_clahe):
    #print(augmentations)
    if augmentations is not None:
        [x.reset() for x in augmentations]
    x_list = []
    image_list = []
    label_list = []
    y_list = []
    original_label_list = []
    original_image_list = []
    parameter_list = []
    padding_list = []
    df_list = []
    for idx, path in enumerate(path_list):
        out = process_2d_image_tool(path, augmentations, img_size, directional_field, apply_clahe)

        x_list.append(out['x'])
        y_list.append(out['y'])
        image_list.append(out['image'])
        original_label_list.append(out['original_label'])
        original_image_list.append(out['original_image'])
        label_list.append(out['label'])
        parameter_list.append(out['parameters'])
        padding_list.append(out['padding'])
        if directional_field:
            df_list.append(out['directional_field'])
    
    x = torch.stack(x_list, dim=1)
    y = torch.stack(y_list, dim=1)
    image = torch.stack(image_list, dim=0)
    original_image = torch.stack(original_image_list, dim=0)
    original_label = torch.stack(original_label_list, dim=0)
    parameters = torch.stack(parameter_list, dim=0)
    label = torch.stack(label_list, dim=0)
    padding = torch.stack(padding_list, dim=0)


    #parameters = get_parameters_3d(torch.argmax(label_volume, dim=0))

    #for i in range(image_out_volume.shape[1]):
    #    fig, ax = plt.subplots(1, 2)
    #    ax[0].imshow(image_out_volume[0, i].numpy(), cmap='gray')
    #    ax[1].imshow(torch.argmax(label_volume, dim=0, keepdim=True).numpy()[0, i], cmap='gray')
    #    #figManager = plt.get_current_fig_manager()
    #    #figManager.window.showMaximized()
    #    plt.show()

    out = {'x': x,
            'y': y,
            'original_label': original_label,
            'original_image': original_image,
            'image': image,
            'label': label,
            'parameters': parameters,
            'padding': padding}
    
    if directional_field:
        df = torch.stack(df_list, dim=1)
        out['directional_field'] = df

    return out


def process_labeled_image_3d_logits(path, augmentations):
    #print(augmentations)
    if augmentations is not None:
        [x.reset() for x in augmentations]

    data = np.load(path)
    clahed_image = data['arr_0']
    label = data['arr_1']
    logits_2d = data['logits_2d']
    logits_3d = data['logits_3d']
    
    clahed_image = torch.from_numpy(clahed_image).unsqueeze(0)
    label = torch.from_numpy(label).long()

    logits_2d = torch.from_numpy(logits_2d).float()
    logits_2d = normalize_0_1(logits_2d)
    logits_3d = torch.from_numpy(logits_3d).float()
    logits_3d = normalize_0_1(logits_3d)

    if augmentations is not None:
        for augmentation in augmentations:
            clahed_image, label, logits_2d, logits_3d = augmentation.augment_images(clahed_image, label, logits_2d, logits_3d)
        assert clahed_image.max() <= 1.0 and clahed_image.min() >= 0.0, augmentations
        assert logits_2d.max() <= 1.0 and logits_2d.min() >= 0.0, augmentations
        assert logits_3d.max() <= 1.0 and logits_3d.min() >= 0.0, augmentations

    clahed_image = NormalizeIntensity(nonzero=True)(clahed_image)
    logits_2d = NormalizeIntensity(nonzero=True)(logits_2d)
    logits_3d = NormalizeIntensity(nonzero=True)(logits_3d)

    input_volume = torch.cat([clahed_image, logits_2d, logits_3d], dim=0).unsqueeze(0)

    return input_volume, label, torch.tensor(data['angle']), torch.from_numpy(data['translation_parameters']), torch.from_numpy(data['padding']), torch.from_numpy(data['original_label']).long(), torch.from_numpy(data['original_image'])


def process_labeled_image_2d_logits(path, img_size, augmentations):
    if augmentations is not None:
        [x.reset() for x in augmentations]

    data = np.load(path)
    image = data['arr_0']
    unrotated_label = data['arr_1']
    angle = torch.tensor(data['angle'])
    logits_2d = data['logits_2d']
    logits_3d = data['logits_3d']

    assert np.all(image >= 0)
    
    unrotated_label = torch.from_numpy(unrotated_label.copy()).long()
    unrotated_label = TF.center_crop(unrotated_label, img_size)

    logits_2d = torch.from_numpy(logits_2d.copy()).float()
    logits_2d = TF.center_crop(logits_2d, img_size)
    logits_2d = normalize_0_1(logits_2d).unsqueeze(0)
    logits_3d = torch.from_numpy(logits_3d.copy()).float()
    logits_3d = TF.center_crop(logits_3d, img_size)
    logits_3d = normalize_0_1(logits_3d).unsqueeze(0)

    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(6, 6))
    clahed_image = clahe.apply(image.astype(np.uint16)).astype(np.int32)
    clahed_image = torch.from_numpy(clahed_image.copy()).float()
    clahed_image = TF.center_crop(clahed_image, img_size)
    clahed_image = normalize_0_1(clahed_image).unsqueeze(0)

    clahed_image = rotate_image(clahed_image, angle, interpolation_mode='bilinear').squeeze(0)
    label = rotate_image(unrotated_label.float(), angle, interpolation_mode='nearest').long().squeeze()

    #fig, ax = plt.subplots(1, 2)
    #ax[0].imshow(clahed_image[0], cmap='gray')
    #ax[1].imshow(label, cmap='gray')
    #plt.show()

    label = torch.nn.functional.one_hot(label, num_classes=4).permute(2, 0, 1).float()
    unrotated_label = torch.nn.functional.one_hot(unrotated_label, num_classes=4).permute(2, 0, 1).float()

    if augmentations is not None:
        for augmentation in augmentations:
            clahed_image, label, logits_2d, logits_3d = augmentation.augment_images(clahed_image, label, logits_2d, logits_3d)
        assert clahed_image.max() <= 1.0 and clahed_image.min() >= 0.0, augmentations
        assert logits_2d.max() <= 1.0 and logits_2d.min() >= 0.0, augmentations
        assert logits_3d.max() <= 1.0 and logits_3d.min() >= 0.0, augmentations

    logits_2d = logits_2d.squeeze()
    logits_3d = logits_3d.squeeze()

    clahed_image = NormalizeIntensity(nonzero=True)(clahed_image)
    logits_2d = NormalizeIntensity(nonzero=True)(logits_2d)
    logits_3d = NormalizeIntensity(nonzero=True)(logits_3d)

    input_2d = torch.cat([clahed_image, logits_2d], dim=0)
    input_3d = torch.cat([clahed_image, logits_3d], dim=0)
    

    #plt.imshow(attention_map[0], cmap='plasma')
    #plt.show()

    return input_2d, input_3d, label, angle, unrotated_label, image


def process_2d_image(path, img_size, augmentations, directional_field, apply_clahe):
    #print(augmentations)
    if augmentations is not None:
        [x.reset() for x in augmentations]

    out = process_2d_image_tool(path, augmentations, img_size, directional_field, apply_clahe)

    return out


def process_2d_image_tool(path, augmentations, img_size, directional_field, apply_clahe):
    #print(augmentations)
    out = {}
    data = np.load(path)
    image = data['image']
    out['image'] = torch.from_numpy(image)
    label = data['label']
    centroid = data['centroid']
    parameters = torch.from_numpy(data['parameters'])

    assert np.all(image >= 0)
    
    label = torch.from_numpy(label.astype(np.uint8)).float()
    out['label'] = label

    label, padding_y = crop_image(centroid=centroid, small_image_size=img_size, image=label)
    z = torch.zeros(size=(1, 1))
    label = transform_image(image=label, angle=parameters[0].reshape((1,)), tx=z, ty=z, scale=parameters[1].reshape((1,)), mode='nearest').squeeze()
    label = torch.nn.functional.one_hot(label.long(), num_classes=4).permute(2, 0, 1).float()

    if apply_clahe:
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image = clahe.apply(image.astype(np.uint16)).astype(np.int32)
    out_image = torch.from_numpy(image.copy()).float()
    out_image, padding_x = crop_image(centroid=centroid, small_image_size=img_size, image=out_image)
    out_image = transform_image(image=out_image, angle=parameters[0].reshape((1,)), tx=z, ty=z, scale=parameters[1].reshape((1,)), mode='bicubic').squeeze(0)

    assert torch.all(padding_x == padding_y)

    if augmentations is not None:
        for augmentation in augmentations:
            out_image, label = augmentation.augment_labeled(out_image, label)

    if directional_field:
        df = get_distance_image(torch.argmax(label, dim=0).numpy(), norm=True)
        out['directional_field'] = df
    
    #out_image = NormalizeIntensity(nonzero=True)(out_image)
    img_min = out_image.min().item()
    img_max = out_image.max().item()
    out_image = NormalizeIntensity(subtrahend=img_min, divisor=(img_max - img_min), nonzero=True)(out_image)

    #print(augmentations)
    #fig, ax = plt.subplots(1, 2)
    #ax[0].imshow(out_image[0], cmap='gray')
    #ax[1].imshow(torch.argmax(label, dim=0), cmap='gray')
    #plt.show()

    #fig, ax = plt.subplots(1, 4)
    #print(augmentations)
    #print(parameters[1])
    #print(math.degrees(parameters[0]))
    #print(path)
    #ax[0].imshow(data['image'], cmap='gray')
    #ax[1].imshow(clahed_image[0], cmap='gray')
    #ax[2].imshow(torch.argmax(label, dim=0), cmap='gray')
    #ax[3].imshow(data['label'], cmap='gray')
    #ax[3].scatter(x=centroid[0], y=centroid[1], c='r', s=10)
    #ax[0].axis('off')
    #ax[1].axis('off')
    #ax[2].axis('off')
    #ax[3].axis('off')
    #plt.show()

    out['x'] = out_image
    out['y'] = label
    out['padding'] = padding_x
    out['parameters'] = parameters
    out['original_label'] = torch.from_numpy(data['original_label'])
    out['original_image'] = torch.from_numpy(data['original_image'])

    return out


def process_2d_image_crop_tool(path, augmentations, img_size, target_ratio, binary, learn_transforms, directional_field, apply_clahe):
    out = {}
    data = np.load(path)
    image = data['image']
    out['image'] = torch.from_numpy(image)
    label = data['label']

    assert np.all(image >= 0)
    
    label = torch.from_numpy(label.astype(np.uint8)).float()
    out['label'] = label
    label = TF.center_crop(label, img_size)
    label = torch.nn.functional.one_hot(label.long(), num_classes=4).permute(2, 0, 1).float()

    if apply_clahe:
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image = clahe.apply(image.astype(np.uint16)).astype(np.int32)
    out_image = torch.from_numpy(image.copy()).float()
    out_image = TF.center_crop(out_image, img_size).unsqueeze(0)
    #out_image = normalize_0_1(out_image).unsqueeze(0)

    if augmentations is not None:
        for augmentation in augmentations:
            out_image, label = augmentation.augment_labeled(out_image, label)
    
    out_image = NormalizeIntensity(nonzero=True)(out_image)

    if directional_field:
        df = get_distance_image(torch.argmax(label, dim=0).numpy(), norm=True)
        out['directional_field'] = df

    original_label = torch.from_numpy(data['original_label'])
    if learn_transforms:
        parameters = torch.tensor(get_parameters_2d(torch.argmax(label, dim=0).numpy(), target_ratio)).float()
        out['parameters'] = parameters
    if binary:
        label = torch.argmax(label, dim=0)
        label[label > 0] = 1
        label = torch.nn.functional.one_hot(label.long(), num_classes=2).permute(2, 0, 1).float()
        original_label[original_label > 0] = 1

    #fig, ax = plt.subplots(1, 3)
    #print(augmentations)
    #print(math.degrees(parameters[0]))
    #print(parameters[-1])
    #print(path)
    #ax[0].imshow(clahed_image[0], cmap='gray')
    #ax[1].imshow(torch.argmax(label, dim=0), cmap='gray')
    #ax[2].imshow(data['original_label'], cmap='gray')
    #plt.show()

    out['x'] = out_image
    out['y'] = label
    out['original_label'] = original_label
    out['original_image'] = torch.from_numpy(data['original_image'])

    return out

def process_2d_image_crop(path, img_size, augmentations, target_ratio, binary, learn_transforms, directional_field, apply_clahe):
    #print(augmentations)
    if augmentations is not None:
        [x.reset() for x in augmentations]
    
    out = process_2d_image_crop_tool(path, augmentations, img_size, target_ratio, binary, learn_transforms, directional_field, apply_clahe)

    #fig, ax = plt.subplots(1, 4)
    #print(math.degrees(parameters[0]))
    #ax[0].imshow(clahed_image[0], cmap='gray')
    #ax[1].imshow(torch.argmax(label, dim=0), cmap='gray')
    #ax[2].imshow(data['original_label'], cmap='gray')
    #ax[3].imshow(data['original_image'], cmap='gray')
    #plt.show()

    #plt.imshow(attention_map[0], cmap='plasma')
    #plt.show()
    
    return out


#def process_labeled_image(path, img_size, tile_sizes, augmentations, use_mask):
#    print(augmentations)
#    if augmentations is not None:
#        [x.reset() for x in augmentations]
#
#    data = np.load(path)
#    image = data['arr_0']
#    label = data['arr_1']
#    zero_mask = torch.tensor(image == 0) if use_mask else torch.zeros(size=image.shape).bool()
#    
#    label = torch.from_numpy(label.copy()).long()
#    label = TF.center_crop(label, img_size)
#    label = torch.nn.functional.one_hot(label, num_classes=4).permute(2, 0, 1).float()
#
#    image_list = []
#    for idx, tile_size in enumerate(tile_sizes):
#
#        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=tile_size)
#        clahed_image = clahe.apply(image.astype(np.uint16)).astype(np.int32)
#        clahed_image = torch.from_numpy(clahed_image.copy()).unsqueeze(dim=0).float()
#        clahed_image = normalize_0_1(clahed_image, zero_mask)
#        clahed_image = TF.center_crop(clahed_image, img_size)
#
#        if augmentations is not None:
#            for augmentation in augmentations:
#                if idx == 0:
#                    clahed_image, label = augmentation.augment_labeled(clahed_image, label)
#                else:
#                    clahed_image = augmentation.augment_unlabeled(clahed_image)
#            assert clahed_image.max() <= 1.0 and clahed_image.min() >= 0.0, augmentations
#        clahed_image = NormalizeIntensity(nonzero=True)(clahed_image)
#        image_list.append(clahed_image)
#
#    parameters = get_parameters(torch.argmax(label, dim=0).numpy())
#
#    return image_list, label, torch.tensor(parameters)

def process_unlabeled_image(path, img_size, rotate, mean, std, augmentation=None):
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    data = np.load(path)
    image = data['arr_0']
    image = clahe.apply(image.astype(np.uint16)).astype(np.int32)

    if rotate and image.shape[0] > image.shape[1]:
        image = np.rot90(image, axes=(1, 0))

    image = torch.from_numpy(image.copy()).unsqueeze(dim=0).float()
    image = TF.center_crop(image, img_size)
    image = standardize(image, mean=mean, std=std)
    image = normalize_0_1(image)
    if augmentation is not None:
        image = augmentation.augment_unlabeled(image)
        image = normalize_0_1(image)
    return image

class MyIterator:
    def __init__(self, indices, cumsum):
        self.current_index = 0
        self.indices = indices
        self.cumsum = cumsum
        self.buckets = [list(range(cumsum[i] - cumsum[i - 1])) for i in range(1, len(cumsum))]

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_index > len(self.indices) - 1:
            raise StopIteration
        for i in range(len(self.cumsum) - 1):
            if self.indices[self.current_index] > self.cumsum[i] and self.indices[self.current_index] < self.cumsum[i + 1]:
                new_idx = random.choice(self.buckets[i])
                self.buckets[i].remove(new_idx)
                self.current_index += 1
                return new_idx

        raise StopIteration

class UnlabeledSampler1(Sampler):
    def __init__(self, indices):
        self.indices = indices
        
    def __iter__(self):
        return iter(self.indices)
    
    def __len__(self):
        return len(self.indices)

class UnlabeledSampler2(Sampler):
    def __init__(self, indices, cumsum):
        self.indices = indices
        self.cumsum = cumsum
        
    def __iter__(self):
        return MyIterator(self.indices, self.cumsum)
    
    def __len__(self):
        return len(self.indices)