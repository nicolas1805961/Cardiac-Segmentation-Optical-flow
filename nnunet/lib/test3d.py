import torch
from tqdm import tqdm
import numpy as np
from training_utils import rotate_image, translate, build_2d_model_crop, build_3d_model_crop, build_3d_model, build_2d_model, read_config, Postprocessing3D, Postprocessing2D, remove_padding
from acdc_dataset import create_ACDC_test_3d, create_ACDC_test_3d_crop
from torchvision.ops import masks_to_boxes
import warnings
import shutil
from pathlib import Path
import matplotlib.pyplot as plt
import os
import torchvision.transforms.functional as TF
from test_short_axis_process import process_data
from scipy import ndimage as ndi
import torch.nn.functional as F
import math
from torchvision.transforms import InterpolationMode
from skimage.measure import regionprops
from compute_temperature import ModelWithTemperature
from medpy.metric import dc
from dataset_utils import normalize_0_1
from boundary_utils import simplex
from torchvision.transforms.functional import center_crop
import sys
import matplotlib.animation as animation

import skimage
import cv2
from skimage.feature import peak_local_max

warnings.filterwarnings("ignore", category=UserWarning)

def revert(image, angle, interpolation_mode, padding, translation_params):
    if angle != 0:
        image = rotate_image(image, -angle, interpolation_mode=interpolation_mode).squeeze()
    
    if translation_params[0] != 0 and translation_params[1] != 0:
        image = translate(image, -translation_params[0], -translation_params[1], interpolation_mode).squeeze()

    pad_left = padding[0]
    pad_right = padding[1]
    pad_top = padding[2]
    pad_bottom = padding[3]
    if pad_left < 0:
        image = image[:, abs(pad_left):]
        pad_left = 0
    if pad_right < 0:
        image = image[:, :abs(pad_right)]
        pad_right = 0
    if pad_top < 0:
        image = image[abs(pad_top):, :]
        pad_top = 0
    if pad_bottom < 0:
        image = image[:abs(pad_bottom), :]
        pad_bottom = 0
    pad_sequence = (pad_left, pad_right, pad_top, pad_bottom)
    image = F.pad(image, pad_sequence, mode='constant')

    return image

def get_lv_centroid(image):
    mask = torch.all(torch.flatten(image, start_dim=1) == 0, dim=-1)
    image = image[~mask]
    boxes = masks_to_boxes(image)
    centroids = torch.stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2], dim=1) # (x, y)
    lv_centroid = centroids[-1]
    return lv_centroid

def get_angle_and_center(unrotated_label_volume):
    index = get_max_index(torch.clone(unrotated_label_volume))
    max_label = unrotated_label_volume[index].numpy()
    center = get_center(max_label)
    if len(torch.unique(unrotated_label_volume)) > 3:
        angle = get_angle(max_label, center)
    else:
        angle = 1.8939664
    return angle, center

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
    

    return center
    

def get_angle(label, center):

    rv_centroid, lv_centroid = get_rv_centroid(label)

    m = ((label.shape[-1] - rv_centroid[0]) - (label.shape[-1] - center[0])) / (rv_centroid[1] - center[1])

    angle = math.degrees(math.atan(m))
    if angle > 0:
        angle = math.degrees(np.pi - math.atan(m))
    else:
        angle = math.degrees(abs(math.atan(m)))
    angle = math.radians(angle)
    return angle 

#def rotate_image(x, parameters):
#    x = x[None, None].float()
#    r = torch.tensor([[math.cos(parameters[0]), -1.0*math.sin(parameters[0]), 0.], [math.sin(parameters[0]), math.cos(parameters[0]), 0.], [0, 0, 1]]).float()
#    t = torch.tensor([[1, 0, parameters[1]], [0, 1, parameters[2]], [0, 0, 1]]).float()
#    t_2 = torch.tensor([[1, 0, -parameters[1]], [0, 1, -parameters[2]], [0, 0, 1]]).float()
#    theta = torch.mm(t, torch.mm(r, t_2))[:-1].unsqueeze(0)
#
#    grid = F.affine_grid(theta, x.size())
#    rotated = F.grid_sample(x, grid, mode='nearest').squeeze()
#
#    return rotated

def crop(image, boxes, big_img_size, img_size, device):
    centroids = ((boxes[:, 0] + boxes[:, 2]) // 2, (boxes[:, 1] + boxes[:, 3]) // 2)
    boxes[:, 0] = torch.maximum(torch.tensor([0], device=device), centroids[0] - (img_size // 2))
    boxes[:, 1] = torch.maximum(torch.tensor([0], device=device), centroids[1] - (img_size // 2))
    boxes[:, 2] = torch.minimum(torch.tensor([big_img_size], device=device), centroids[0] + (img_size // 2))
    boxes[:, 3] = torch.minimum(torch.tensor([big_img_size], device=device), centroids[1] + (img_size // 2))

    indices = torch.nonzero(boxes[:, 0] == 0)
    boxes[:, 2][indices] = img_size

    indices = torch.nonzero(boxes[:, 1] == 0)
    boxes[:, 3][indices] = img_size

    indices = torch.nonzero(boxes[:, 2] == big_img_size)
    boxes[:, 0][indices] = big_img_size - img_size

    indices = torch.nonzero(boxes[:, 3] == big_img_size)
    boxes[:, 1][indices] = big_img_size - img_size

    return torch.stack([image[i, :, boxes[i, 1]: boxes[i, 3], boxes[i, 0]:boxes[i, 2]] for i in range(image.size(0))], dim=0)

def get_max_index(volume_pred):
    volume_pred[volume_pred > 0] = 1
    volume_pred = torch.flatten(volume_pred, start_dim=1)
    volume_pred = torch.sum(volume_pred, dim=-1)
    return torch.argmax(volume_pred, dim=0)


def get_images_registered(new_folder_name, config_path, test_path, weights_path, big_image_size, small_image_size):
    with torch.no_grad():
        postprocessing = Postprocessing3D()
        config = read_config(config_path)
        model = build_3d_model_crop(config)
        #model = ModelWithTemperature(orig_model)
        model.load_state_dict(torch.load(weights_path))
        model.eval()
        labeled_test_dataloader = create_ACDC_test_3d_crop(test_path, big_image_size, config['device'])
        dcs = []
        global_lv_list_before = []
        global_lv_list_after = []
        for idx, data in enumerate(tqdm(labeled_test_dataloader, desc='Labeled testing iteration: ')):
            image_volume = data['x']
            path_list = [x[0] for x in data['path']]

            out = model(image_volume)

            pred = out['pred'][-1]
            angle = out['angle']
            image_volume = image_volume.squeeze()
            image_volume, pred = remove_padding(image_volume, pred)

            pred = F.softmax(pred, dim=1)
            pred = torch.argmax(pred.squeeze(0), dim=0)
            pred = postprocessing(pred=pred)

            big_images_data = np.load(path_list[0])
            zoomed_shape = big_images_data['zoomed_image'].shape

            cc_list = []
            for j in range(len(pred)):
                cc_list.append(center_crop(pred[j], output_size=zoomed_shape))
            pred = torch.stack(cc_list, dim=0)

            assert pred.shape[1:] == zoomed_shape

            flat_pred = torch.any(pred > 0, dim=0).int()
            boxes = masks_to_boxes(flat_pred.unsqueeze(0)).squeeze()
            global_centroid = torch.tensor([(boxes[0].item() + boxes[2].item()) / 2, (boxes[1].item() + boxes[3].item()) / 2]) # (x, y)

            pred_one_hotted = F.one_hot(pred.long(), num_classes=4)
            myo_pred = pred_one_hotted[:, :, :, -2]
            lv_pred = pred_one_hotted[:, :, :, -1]

            flat_pred_lv = torch.any(lv_pred > 0, dim=0).int()
            boxes = masks_to_boxes(flat_pred_lv.unsqueeze(0)).squeeze()
            global_lv_centroid = torch.tensor([(boxes[0].item() + boxes[2].item()) / 2, (boxes[1].item() + boxes[3].item()) / 2]) # (x, y)

            #fig, ax = plt.subplots(1, 2)
            #ax[0].imshow(flat_pred_lv.cpu(), cmap='gray')
            #ax[0].scatter(x=global_lv_centroid[0], y=global_lv_centroid[1], c='r', s=10)
            #ax[1].imshow(flat_pred.cpu(), cmap='gray')
            #ax[1].scatter(x=global_centroid[0], y=global_centroid[1], c='r', s=10)
            #plt.show()

            #fig, ax = plt.subplots(1, 3)
            #ax[2].imshow(pred[3].cpu(), cmap='gray')
            #ax[0].imshow(flat_pred.cpu(), cmap='gray')
            #ax[1].imshow(image_volume[3].cpu(), cmap='gray')
            #print(path_list[0])
            #ax[0].scatter(x=centroid[0], y=centroid[1], c='r', s=10)
            #plt.show()


            #flat_max_pred = torch.clone(max_pred)
            #flat_max_pred[flat_max_pred > 0] = 1

            #boxes = masks_to_boxes(flat_max_pred.unsqueeze(0)).squeeze()
            #centroid = [(boxes[0].item() + boxes[2].item()) / 2, (boxes[1].item() + boxes[3].item()) / 2] # (x, y)

            pad_index_list = []
            distance_list = []
            assert len(pred) == len(path_list) == len(lv_pred) == len(myo_pred)
            #display_test_list = []
            #print(path_list[0])
            #if '014' in path_list[0]:
            #    fig, ax = plt.subplots(1, 2)
            for i, path in enumerate(path_list):
                
                current_pred = pred[i]
                current_lv_pred = lv_pred[i]
                current_myo_pred = myo_pred[i]

                if torch.count_nonzero(current_lv_pred) > 0:
                    boxes = masks_to_boxes(current_lv_pred.unsqueeze(0)).squeeze()
                    lv_centroid = torch.tensor([(boxes[0].item() + boxes[2].item()) / 2, (boxes[1].item() + boxes[3].item()) / 2]) # (x, y)
                    distance = lv_centroid - global_lv_centroid
                    distance_list.append(distance)
                    #centroid_list.append(global_centroid + distance)
                    #if '014' in path:
                    #    current_pred_cropped, padding = crop_image(centroid=global_centroid + distance, small_image_size=small_image_size, big_shape=current_pred.shape, image=current_pred)
                    #    current_pred_cropped = rotate_image(current_pred_cropped.float(), angle, interpolation_mode='nearest').squeeze()
                    #    im1 = ax[0].imshow(current_pred_cropped.cpu(), cmap='gray', animated=True)
                    #    im2 = ax[1].imshow(current_pred.cpu(), cmap='gray', animated=True)
                    #    display_test_list.append([im1, im2])
                elif torch.count_nonzero(current_myo_pred) > 0:
                    boxes = masks_to_boxes(current_myo_pred.unsqueeze(0)).squeeze()
                    myo_centroid = torch.tensor([(boxes[0].item() + boxes[2].item()) / 2, (boxes[1].item() + boxes[3].item()) / 2]) # (x, y)
                    distance = myo_centroid - global_lv_centroid
                    distance_list.append(distance)
                    #centroid_list.append(global_centroid + distance)
                else:
                    pad_index_list.append(i)

            #if '014' in path_list[0]:
            #    ani = animation.ArtistAnimation(fig, display_test_list, interval=1500, blit=True, repeat_delay=1000)
            #    plt.show()


            distances = torch.stack(distance_list, dim=0)

            pad_indices = torch.tensor(pad_index_list)
            pad_indices = (pad_indices / (len(pred) / 2)).int()
            pad_top = (pad_indices == 0).sum().item()
            pad_bottom = (pad_indices == 1).sum().item()
            pad_sequence = (0, 0, pad_top, pad_bottom)
            distances = F.pad(distances.unsqueeze(0), pad_sequence, mode='replicate').squeeze()

            assert len(distances) == len(pred) == len(path_list)

            for i, path in enumerate(path_list):
                metadata = '\\'.join(path.split('\\')[1:])
                folder_paths = '\\'.join(metadata.split('\\')[:-1])

                big_images_data = np.load(path)
                original_image_volume = big_images_data['original_image_volume']
                original_label_volume = big_images_data['original_label_volume']
                zoomed_image = big_images_data['zoomed_image']
                zoomed_label = big_images_data['zoomed_label']

                Path(os.path.join(new_folder_name, folder_paths)).mkdir(parents=True, exist_ok=True)
                np.savez_compressed(os.path.join(new_folder_name, metadata), zoomed_image=zoomed_image, zoomed_label=zoomed_label, angle=angle.cpu(), centroid=global_centroid, shift=distances[i], original_label_volume=original_label_volume, original_image_volume=original_image_volume)

            #lv_list_before = []
            #lv_list_after = []
            ##print(path_list[0])
            #for i, path in enumerate(path_list):
#
            #    metadata = '\\'.join(path.split('\\')[1:])
            #    folder_paths = '\\'.join(metadata.split('\\')[:-1])
#
            #    current_small_image = small_image_list[i]
            #    current_small_label = small_label_list[i]
            #    current_padding = padding_list[i]
            #    current_translation_parameters = translation_parameters[i]
            #    tx, ty = current_translation_parameters[0], current_translation_parameters[1]
#
            #    big_images_data = np.load(path)
            #    big_label = big_images_data['arr_1']
            #    big_image = big_images_data['arr_0']
#
            #    big_label = torch.from_numpy(big_label.copy()).float()
#
            #    temp_image = torch.clone(current_small_image)
            #    temp_label = torch.clone(current_small_label)
#
            #    if len(torch.unique(temp_label)) == 1:
            #        new_centroid = torch.tensor([temp_label.shape[-1] - 1, temp_label.shape[-2] - 1])
            #    else:
            #        new_centroid = get_lv_centroid(F.one_hot(temp_label.long(), num_classes=4).permute(2, 0, 1)).float()
            #        lv_list_before.append(new_centroid)
#
            #    temp_image = translate(torch.clone(temp_image), tx, ty, 'bilinear').squeeze()
            #    temp_label = translate(torch.clone(temp_label), tx, ty, 'nearest').squeeze()
#
            #    if len(torch.unique(temp_label)) == 1:
            #        new_centroid = torch.tensor([temp_label.shape[-1] - 1, temp_label.shape[-2] - 1])
            #    else:
            #        new_centroid = get_lv_centroid(F.one_hot(temp_label.long(), num_classes=4).permute(2, 0, 1)).float()
            #        lv_list_after.append(new_centroid)
#
            #    temp_image = rotate_image(temp_image, angle, interpolation_mode='bilinear').squeeze()
            #    temp_label = rotate_image(temp_label, angle, interpolation_mode='nearest').squeeze()
#
            #    assert temp_image.shape == temp_label.shape == (128, 128), temp_label.shape
#
            #    reverted = revert(image=temp_label, angle=torch.tensor(angle), interpolation_mode='nearest', padding=current_padding, translation_params=[tx, ty])
#
            #    #fig, ax = plt.subplots(1, 2)
            #    #print(path)
            #    #ax[0].imshow(current_small_image, cmap='gray')
            #    #ax[1].imshow(current_small_label, cmap='gray')
            #    #plt.show()
#
            #    assert reverted.shape == big_label.shape
            #    if torch.count_nonzero(big_label) > 0:
            #        dcs.append(dc(reverted.cpu().numpy(), big_label.cpu().numpy()))
            #
            #    #Path(os.path.join(new_folder_name, folder_paths)).mkdir(parents=True, exist_ok=True)
            #    #np.savez_compressed(os.path.join(new_folder_name, metadata), arr_0=current_small_image.cpu().numpy(), arr_1=current_small_label.cpu().numpy(), angle=angle.cpu(), translation_parameters=[tx, ty], padding=current_padding, original_label=big_label.cpu().numpy(), original_image=big_image)
#
            #lv_list_before = torch.stack(lv_list_before, dim=0)
            #lv_list_after = torch.stack(lv_list_after, dim=0)
            #global_lv_list_before.append(torch.max(lv_list_before, dim=0)[0] - torch.min(lv_list_before, dim=0)[0])
            #global_lv_list_after.append(torch.max(lv_list_after, dim=0)[0] - torch.min(lv_list_after, dim=0)[0])
        #dcs = torch.tensor(dcs)
        #print(dcs.mean())
        #print(torch.stack(global_lv_list_before, dim=0).mean(dim=0))
        #print(torch.stack(global_lv_list_after, dim=0).mean(dim=0))

def fill_list(x):
    for i in range(len(x)):
        t = 0
        idx = i
        v = 1
        top = bottom = False
        while torch.any(torch.isnan(x[idx])):
            if idx == 0:
                bottom = True
            elif idx == len(x) - 1:
                top = True
            if not top and not bottom:
                t = t + v if t <= 0 else t - v
                v += 2
            elif bottom:
                t = 1
            elif top:
                t = -1
            idx = idx + t
        x[i] = x[idx]
    return x

def get_images(new_folder_name, config_path, test_path, weights_path, big_image_size):
    with torch.no_grad():
        postprocessing_2d = Postprocessing2D()
        config = read_config(config_path)
        model = build_2d_model(config)
        #model = ModelWithTemperature(orig_model)
        model.load_state_dict(torch.load(weights_path))
        model.eval()
        labeled_test_dataloader = create_ACDC_test_3d_crop(test_path, big_image_size, config['device'], config['target_ratio'], config['binary'], config['learn_transforms'], config['directional_field'])
        for idx, data in enumerate(tqdm(labeled_test_dataloader, desc='Labeled testing iteration: ')):
            x = data['x'].to('cuda:0')
            path_list = [x[0] for x in data['path_list']]

            assert x.shape[2] == len(path_list)

            preds = []
            angles = []
            scales = []
            centroids = []
            for j in range(x.shape[2]):
                out = model(x[:, :, j])
                pred = out['pred'][-1]

                pred = torch.nn.functional.softmax(pred, dim=1)
                pred = torch.argmax(pred.squeeze(0), dim=0)
                pred = postprocessing_2d(pred)

                #if '79' in path_list[j]:
                #    fig, ax = plt.subplots(1, 2)
                #    ax[0].imshow(x[:, :, j].cpu()[0, 0], cmap='gray')
                #    ax[1].imshow(pred.cpu(), cmap='gray')
                #    plt.show()

                big_shape = data['image'].shape[-2:]
                pred = center_crop(pred, output_size=big_shape)
                assert pred.shape == big_shape
                preds.append(pred)

                parameters = out['parameters'].reshape(-1,) if config['learn_transforms'] else torch.tensor([0, 1])
                if not config['binary'] and (torch.count_nonzero(pred == 1) == 0 or torch.count_nonzero(pred == 3) == 0):
                    angles.append(torch.tensor(float('nan')))
                else:
                    angles.append(parameters[0])

                binary_pred = pred
                binary_pred[binary_pred > 0] = 1
                if torch.count_nonzero(binary_pred) == 0:
                    centroids.append(torch.tensor(float('nan')))
                    scales.append(torch.tensor(float('nan')))
                else:
                    boxes = masks_to_boxes(binary_pred.unsqueeze(0)).squeeze()
                    centroid = torch.tensor([(boxes[0].item() + boxes[2].item()) / 2, (boxes[1].item() + boxes[3].item()) / 2]) # (x, y)
                    centroids.append(centroid)
                    scales.append(parameters[1])
            
            centroids = fill_list(centroids)
            preds = torch.stack(preds, dim=0)
            centroids = torch.stack(centroids, dim=0)
            angles = fill_list(angles)
            scales = fill_list(scales)
            angles = torch.stack(angles, dim=0)
            scales = torch.stack(scales, dim=0)
            parameters = torch.stack([angles, scales], dim=1)

            assert len(centroids) == len(scales)== len(angles) == len(preds) == len(path_list)

            assert (~torch.isnan(centroids)).all()
            assert (~torch.isnan(parameters)).all()

            if not config['learn_transforms']:
                assert torch.all(parameters[:, 0] == 0)
                assert torch.all(parameters[:, 1] == 1)

            for i, path in enumerate(path_list):

                metadata = '\\'.join(path.split('\\')[1:])
                folder_paths = '\\'.join(metadata.split('\\')[:-1])

                original_data = np.load(path)
                image = original_data['image']
                label = original_data['label']
                original_label = original_data['original_label']
                original_image = original_data['original_image']
            
                Path(os.path.join(new_folder_name, folder_paths)).mkdir(parents=True, exist_ok=True)
                np.savez_compressed(os.path.join(new_folder_name, metadata), centroid=centroids[i].cpu(), parameters=parameters[i].cpu(), image=image, label=label, original_image=original_image, original_label=original_label)



def get_logits(new_folder_name, config_path_3d, config_path_2d, test_path, weights_path_3d, weights_path_2d, big_image_size):
    with torch.no_grad():
        config_3d = read_config(config_path_3d)
        config_2d = read_config(config_path_2d)
        model_3d = build_3d_model(config_3d)
        model_2d = build_2d_model(config_2d)
        #model = ModelWithTemperature(orig_model)
        model_3d.load_state_dict(torch.load(weights_path_3d))
        model_2d.load_state_dict(torch.load(weights_path_2d))
        model_3d.eval()
        model_2d.eval()
        labeled_test_dataloader = create_ACDC_test_3d(test_path, big_image_size, 'cuda:0')
        for idx, data in enumerate(tqdm(labeled_test_dataloader, desc='Labeled testing iteration: ')):
            x = data['x']
            label = data['y'].squeeze()
            angle = data['angle'].squeeze()
            translation_parameters = data['translation_parameters'].squeeze()
            padding = data['padding'].squeeze()
            original_label = data['original_label'].squeeze()
            original_image = data['original_image'].squeeze()
            x_before_norm = data['image_before_norm'].squeeze()
            path_list = [x[0] for x in data['path']]

            out = model_3d(x)
            logits_3d = out['pred'][-1]

            logits_3d = logits_3d.squeeze()
            x = x.squeeze()

            temp = torch.flatten(x, start_dim=1).squeeze()
            mask = torch.all(temp == 0, dim=-1)
            x = x[~mask]
            logits_3d = logits_3d[:, ~mask]

            #fig, ax = plt.subplots(2, len(pred))
            #for i in range(len(pred)):
            #    ax[0, i].imshow(pred[i].cpu(), cmap='gray')
            #    ax[1, i].imshow(image_volume[0, 0, i].cpu(), cmap='gray')
            #plt.show()

            assert len(path_list) == logits_3d.shape[1] == len(x) == len(x_before_norm) == label.shape[1] == len(original_label) == len(original_image) == len(padding) == len(translation_parameters)
            for i, path in enumerate(path_list):
                #print(path)

                metadata = '\\'.join(path.split('\\')[1:])
                folder_paths = '\\'.join(metadata.split('\\')[:-1])

                current_label = label[:, i]
                current_x = x[i]
                current_original_image = original_image[i]
                current_original_label = original_label[i]
                current_3d_logits = logits_3d[:, i]
                current_padding = padding[i]
                current_translation_parameters = translation_parameters[i]
                current_x_before_norm = x_before_norm[i]

                out_2d = model_2d(current_x.unsqueeze(0).unsqueeze(0))
                logits_2d = out_2d['pred'][-1].squeeze()

                #pred_3d = torch.argmax(F.softmax(current_3d_logits, dim=0), dim=0)
                #pred_2d = torch.argmax(F.softmax(logits_2d, dim=0), dim=0)

                #fig, ax = plt.subplots(1, 7)
                #ax[0].imshow(logits_2d[1].cpu(), cmap='plasma')
                #ax[1].imshow(current_3d_logits[1].cpu(), cmap='plasma')
                #ax[2].imshow(current_x.cpu(), cmap='gray')
                #ax[3].imshow(current_label.cpu(), cmap='gray')
                #ax[4].imshow(pred_2d.cpu(), cmap='gray')
                #ax[5].imshow(pred_3d.cpu(), cmap='gray')
                #ax[6].imshow(current_image_in.cpu(), cmap='gray')
                #plt.show()

                #print(dc(small_label.cpu().numpy(), small_pred.cpu().numpy()))

                #Path(os.path.join(new_folder_name, folder_paths)).mkdir(parents=True, exist_ok=True)
                #np.savez_compressed(os.path.join(new_folder_name, metadata), arr_0=current_x_before_norm.cpu().numpy(), arr_1=current_label.cpu().numpy(), angle=angle, logits_2d=logits_2d.cpu().numpy(), logits_3d=current_3d_logits.cpu().numpy(), translation_parameters=current_translation_parameters, padding=current_padding, original_label=current_original_label.cpu().numpy(), original_image=current_original_image.cpu().numpy())

#new_folder_name = 'caca'

#dirpath = Path(new_folder_name)
#if dirpath.exists() and dirpath.is_dir():
#    shutil.rmtree(dirpath)

#get_logits(new_folder_name, '3d_model_128_3\config.yaml', '2d_model_128_2\config.yaml', 'ACDC_resampled_cropped_2/*', '3d_model_128_3\weights.pth', '2d_model_128_2\weights.pth', 128)

#new_folder_name = 'ACDC_cropped_binary_125'

#dirpath = Path(new_folder_name)
#if dirpath.exists() and dirpath.is_dir():
#    shutil.rmtree(dirpath)

#get_images(new_folder_name, 'binary_crop_model\config.yaml', 'ACDC_resampled/*', 'binary_crop_model\weights.pth', 160, binary=True)
#get_images(new_folder_name, 'binary_crop_model_125\config.yaml', 'ACDC_resampled_125/*', 'binary_crop_model_125\weights.pth', 224)

#new_folder_name = 'ACDC_resampled_cropped'
#
#dirpath = Path(new_folder_name)
#if dirpath.exists() and dirpath.is_dir():
#    shutil.rmtree(dirpath)
#
#get_images_registered(new_folder_name, 'temp_160\config.yaml', 'ACDC_resampled/*', 'temp_160\weights.pth', 160, 128)