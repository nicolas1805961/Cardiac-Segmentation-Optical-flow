import torch
from tqdm import tqdm
import numpy as np
from training_utils import read_config
from acdc_dataset import create_ACDC_test
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

import skimage
import cv2
from skimage.feature import peak_local_max

warnings.filterwarnings("ignore", category=UserWarning)

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

def rotate_image(x, angle, interpolation_mode):
    r = torch.tensor([[torch.cos(angle), -1.0*torch.sin(angle), 0.], [torch.sin(angle), torch.cos(angle), 0.], [0, 0, 1]]).float()
    theta = r[:-1]
    theta = theta.unsqueeze(0)

    grid = F.affine_grid(theta, (1, 1,) + x.size())
    rotated = F.grid_sample(x[None, None], grid, mode=interpolation_mode)
    return rotated

def get_images(new_folder_name, config_path, test_path, weights_path, big_image_size, small_image_size):
    config = read_config(config_path)
    model = build_adversarial_model(config, test=True)
    model.load_state_dict(torch.load(weights_path))
    model.eval()
    labeled_test_dataloader, unlabeled_test_dataloader = create_ACDC_test(test_path, big_image_size, config['device'])
    for idx, data in enumerate(tqdm(labeled_test_dataloader, desc='Labeled testing iteration: ')):
        image = data['x']
        path = data['path'][0]

        metadata = '\\'.join(path.split('\\')[1:])
        folder_paths = '\\'.join(metadata.split('\\')[:-1])

        out = model(image)
        angle = out['angle']
        pred = out['pred'][-1]
        pred = torch.argmax(pred, dim=1).squeeze()

        big_images_data = np.load(path)
        big_image = big_images_data['arr_0']
        big_label = big_images_data['arr_1']

        big_image = torch.from_numpy(big_image.copy()).float()
        big_label = torch.from_numpy(big_label.copy()).float()
        big_shape = big_image.shape

        if torch.count_nonzero(torch.unique(pred)) == 0:
            small_image = TF.center_crop(big_image, small_image_size)
            small_label = TF.center_crop(big_label, small_image_size)
            small_image = rotate_image(small_image, angle, interpolation_mode='bilinear').squeeze()
            small_label = rotate_image(small_label, angle, interpolation_mode='nearest').squeeze()
            Path(os.path.join(new_folder_name, folder_paths)).mkdir(parents=True, exist_ok=True)
            np.savez_compressed(os.path.join(new_folder_name, metadata), arr_0=small_image.numpy(), arr_1=small_label.numpy())
            continue

        boxes = masks_to_boxes(pred.unsqueeze(0)).int()
        centroid = [(boxes[:, 0] + boxes[:, 2]) // 2, (boxes[:, 1] + boxes[:, 3]) // 2]

        centroid = [centroid[0] + ((big_shape[1] - big_image_size) // 2), centroid[1] + ((big_shape[0] - big_image_size) // 2)]
        y1 = centroid[1] - (small_image_size // 2)
        y2 = centroid[1] + (small_image_size // 2)
        x1 = centroid[0] - (small_image_size // 2)
        x2 = centroid[0] + (small_image_size // 2)

        if y1 < 0:
            pad = (0, 0, int(math.ceil(abs(y1))), 0)
            big_image = torch.nn.functional.pad(big_image, pad, mode='constant', value=0.0)
            big_label = torch.nn.functional.pad(big_label, pad, mode='constant', value=0.0)
            centroid[1] += int(math.ceil(abs(y1)))
            y1 = 0
        if y2 - big_shape[0] > 0:
            pad = (0, 0, 0, int(math.ceil(abs(y2 - big_shape[0]))))
            big_image = torch.nn.functional.pad(big_image, pad, mode='constant', value=0.0)
            big_label = torch.nn.functional.pad(big_label, pad, mode='constant', value=0.0)
            y2 = big_image.shape[0]
        if x1 < 0:
            pad = (int(math.ceil(abs(x1))), 0, 0, 0)
            big_image = torch.nn.functional.pad(big_image, pad, mode='constant', value=0.0)
            big_label = torch.nn.functional.pad(big_label, pad, mode='constant', value=0.0)
            centroid[1] += int(math.ceil(abs(x1)))
            x1 = 0
        if x2 - big_shape[1] > 0:
            pad = (0, int(math.ceil(abs(x2 - big_shape[1]))), 0, 0)
            big_image = torch.nn.functional.pad(big_image, pad, mode='constant', value=0.0)
            big_label = torch.nn.functional.pad(big_label, pad, mode='constant', value=0.0)
            x2 = big_image.shape[1]

        small_image = big_image[y1:y2, x1:x2]
        small_label = big_label[y1:y2, x1:x2]

        small_image = rotate_image(small_image, angle, interpolation_mode='bilinear').squeeze()
        small_label = rotate_image(small_label, angle, interpolation_mode='nearest').squeeze()

        #if big_shape[0] == big_shape[1]:
        #    print(path)
        #    fig, ax = plt.subplots(1, 5)
        #    ax[0].imshow(small_image, cmap='gray')
        #    ax[1].imshow(small_label, cmap='gray')
        #    ax[2].imshow(pred.cpu(), cmap='gray')
        #    ax[3].imshow(big_image.cpu(), cmap='gray')
        #    ax[4].imshow(big_label.cpu(), cmap='gray')
        #    ax[4].scatter(x=centroid[0].cpu(), y=centroid[1].cpu(), c='r', s=10)
        #    plt.show()

        Path(os.path.join(new_folder_name, folder_paths)).mkdir(parents=True, exist_ok=True)
        np.savez_compressed(os.path.join(new_folder_name, metadata), arr_0=small_image.numpy(), arr_1=small_label.numpy())

#new_folder_name = 'ACDC_cropped'

#dirpath = Path(new_folder_name)
#if dirpath.exists() and dirpath.is_dir():
#    shutil.rmtree(dirpath)

#get_images(new_folder_name, 'binary_rotation\config.yaml', 'ACDC_data', 'binary_rotation\weights.pth', 224, 160)

#def get_images(new_folder_name, config_path, test_path, weights_path, big_image_size, small_image_size):
#    config = read_config(config_path)
#    model = build_model(config)
#    model.load_state_dict(torch.load(weights_path))
#    model.eval()
#    labeled_test_dataloader, unlabeled_test_dataloader = create_ACDC_test(test_path, big_image_size, config['device'])
#    for data in tqdm(labeled_test_dataloader, desc='Labeled testing iteration: '):
#        images = data['x']
#        labels = data['y']
#        paths = data['path']
#        paths = [x[0] for x in paths]
#        pred_list = []
#        image_list = []
#        label_list = []
#        for i in range(len(images)):
#            path = paths[i]
#            image = images[i]
#            pred = model(image)['pred'][-1]
#            pred = torch.argmax(pred, dim=1).squeeze()
#
#            big_images_data = np.load(path)
#            big_image = big_images_data['arr_0']
#            big_label = big_images_data['arr_1']
#            image_list.append(big_image)
#            label_list.append(big_label)
#            big_shape = big_image.shape
#
#            #print(path)
#            #fig, ax = plt.subplots(1, 3)
#            #ax[0].imshow(big_image, cmap='gray')
            #ax[1].imshow(pred.cpu(), cmap='gray')
            #ax[2].imshow(image[0, 0].cpu(), cmap='gray')
            #plt.show()

#            pred = TF.center_crop(pred, big_shape)
#
#            pred_list.append(pred)
#
#        global_binary_mask = torch.stack(pred_list)
#        global_binary_mask = torch.sum(global_binary_mask, dim=0).cpu().numpy()
#        global_binary_mask[global_binary_mask > 0] = 1
#
#        big_images = np.stack(image_list)
#        big_labels = np.stack(label_list)
#
#        images, labels = process_data(global_binary_mask, big_labels, big_images, small_image_size//2)
#
#        assert len(images) == len(labels) == len(paths)
#        for i, (image, label, path) in enumerate(zip(images, labels, paths)):
#            metadata = '\\'.join(path.split('\\')[1:])
#            folder_paths = '\\'.join(metadata.split('\\')[:-1])
#
#            print(metadata)
#
#            fig, ax = plt.subplots(2, 2)
#            ax[0, 0].imshow(big_images[i], cmap='gray')
#            ax[0, 1].imshow(big_labels[i], cmap='gray')
#            ax[1, 0].imshow(image, cmap='gray')
#            ax[1, 1].imshow(label, cmap='gray')
#            plt.show()

            #Path(os.path.join(new_folder_name, folder_paths)).mkdir(parents=True, exist_ok=True)
            #np.savez_compressed(os.path.join(new_folder_name, metadata), arr_0=image, arr_1=label)
        

        

    #for data in tqdm(unlabeled_test_dataloader, desc='Unlabeled testing iteration: '):
    #    path = data['path'][0]
    #    metadata = '\\'.join(path.split('\\')[1:])
    #    folder_paths = '\\'.join(metadata.split('\\')[:-1])
    #    image = data['x']
    #    pred = model(image)[-1]
    #    pred = torch.argmax(pred, dim=1)
#
    #    big_images_data = np.load(path)
    #    big_image = big_images_data['arr_0']
    #    big_shape = big_image.shape
#
    #    if torch.count_nonzero(torch.unique(pred)) == 0:
    #        big_image = torch.from_numpy(big_image)
    #        small_image = TF.center_crop(big_image, small_image_size)
    #        Path(os.path.join(new_folder_name, folder_paths)).mkdir(parents=True, exist_ok=True)
    #        np.savez_compressed(os.path.join(new_folder_name, metadata), arr_0=small_image.numpy())
    #        continue
#
    #    boxes = masks_to_boxes(pred).int()
    #    centroid = ((boxes[:, 0] + boxes[:, 2]) // 2, (boxes[:, 1] + boxes[:, 3]) // 2)
#
    #    centroid = (centroid[0] + ((big_shape[1] - big_image_size) // 2), centroid[1] + ((big_shape[0] - big_image_size) // 2))
    #    y1 = torch.maximum(torch.tensor([0], device=config['device']), centroid[1] - (small_image_size // 2))
    #    y2 = torch.minimum(torch.tensor([big_shape[0]], device=config['device']), centroid[1] + (small_image_size // 2))
    #    x1 = torch.maximum(torch.tensor([0], device=config['device']), centroid[0] - (small_image_size // 2))
    #    x2 = torch.minimum(torch.tensor([big_shape[1]], device=config['device']), centroid[0] + (small_image_size // 2))
#
    #    if x1 == 0:
    #        x2 = small_image_size
    #    if y1 == 0:
    #        y2 = small_image_size
    #    if x2 == big_shape[1]:
    #        x1 = big_shape[1] - small_image_size
    #    if y2 == big_shape[0]:
    #        y1 = big_shape[0] - small_image_size
#
    #    small_image = big_image[y1:y2, x1:x2]
#
    #    Path(os.path.join(new_folder_name, folder_paths)).mkdir(parents=True, exist_ok=True)
    #    np.savez_compressed(os.path.join(new_folder_name, metadata), arr_0=small_image)

#config = read_config('acdc_config.yaml')
#get_images(config, 'ACDC_data', '')

#def test_loop(model, test_dataloader, device):
#    model.eval()
#    with torch.no_grad():
#        class_dice_sum = np.array([0, 0, 0], dtype=np.float64)
#        class_hd_sum = np.array([0, 0, 0], dtype=np.float64)
#        for x_vals, y_vals in tqdm(test_dataloader, desc='Testing iteration: '):
#            x_vals, y_vals = x_vals.to(device), y_vals.to(device)
#            x_vals = x_vals.permute((1, 0, 2, 3, 4))
#            y_vals = y_vals.permute((1, 0, 2, 3, 4))
#            pred_val_list = model(x_vals)
#            class_dice_sum_frames = np.array([0, 0, 0], dtype=np.float64)
#            class_hd_sum_frames = np.array([0, 0, 0], dtype=np.float64)
#            for ds_x, frame_y in zip(pred_val_list, y_vals):
#                class_dice, class_hd = get_metrics(frame_y, ds_x[-1])
#                class_dice_sum_frames += class_dice
#                class_hd_sum_frames += class_hd
#            class_dice_sum += class_dice_sum_frames / len(pred_val_list)
#            class_hd_sum += class_hd_sum_frames / len(pred_val_list)
#        class_dices = class_dice_sum / len(test_dataloader)
#        class_hds = class_hd_sum / len(test_dataloader)
#        return class_dices, class_hds


#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#model.load_state_dict(torch.load('out/weights.pth'))
#model.eval()
#class_dice, class_hd = test_loop(model, dataloaders['test_dataloader'], device)
#with open('results.txt', 'a') as result_file:
#    result_file.write(f'Class Dices: {class_dice}')
#    result_file.write(f'Class Hausdorff distance: {class_hd}')
#    result_file.write('****************************************************************')

#warnings.filterwarnings("ignore", category=UserWarning)
#
def set_angle(label, angle):
    (h, w) = label.shape[-2:]
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    plt.imshow(label.astype(np.uint8), cmap='gray')
    plt.show()
    print(angle)
    rotated = cv2.warpAffine(label.astype(np.uint8), M, (w, h), flags=cv2.INTER_LINEAR)
    plt.imshow(rotated, cmap='gray')
    plt.show()

    rv_centroid, lv_centroid = get_rv_centroid(rotated)
    if rv_centroid[1] > lv_centroid[1]:
        print('flipped !')
        angle -= 180
    return angle

def get_rv_centroid(label):
    label1 = label.astype(np.int32)
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(label1, cmap='gray')
    ax[1].imshow(label, cmap='gray')
    plt.show()
    distance = ndi.distance_transform_edt(label)
    coords = peak_local_max(distance, min_distance=2, labels=label)
    if len(coords) < 2:
        coords = peak_local_max(distance, min_distance=1, labels=label)

    #temp = [np.where(coords == x) for x in np.unique(coords)]
    #print(temp)
    #for t in temp:
    #    if len(t[0]) > 1:
    #        indices = t[0][t[1] == 1]
    #        for j in range(len(indices) - 1):
    #            coords = np.delete(coords, np.max(indices), axis=0)
    #coords = coords[:2]
    #if np.any(coords[0] == coords[1]):
    #    coords = np.delete(coords, 1, axis=0)
    #else:
    #    coords = coords[:3]
    print(coords)
    #sort peaks get only two
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    labels = skimage.segmentation.watershed(-distance, markers, mask=label)
    plt.imshow(labels, cmap='gray')
    plt.show()
    regions = skimage.measure.regionprops(labels)
    data = [(region.axis_minor_length, i) for i, region in enumerate(regions)]
    print(data)
    data = sorted(data, key=lambda x: x[0])
    print(data)
    indices = [x[1] for x in data[-2:]]
    print(indices)
    data = [(4 * np.pi * regions[i].area / (regions[i].perimeter ** 2), regions[i].centroid) for i in indices]
    print(data)
    data = sorted(data, key=lambda x: x[0])
    print(data)
    lv_centroid = data[1][1]
    rv_centroid = data[0][1]


    #region0_eccentricity = regions[0].eccentricity
    #region1_eccentricity = regions[1].eccentricity
    #if region0_eccentricity > region1_eccentricity:   
    #    rv_centroid = regions[0].centroid
    #    lv_centroid = regions[1].centroid
    #else:
    #    rv_centroid = regions[1].centroid
    #    lv_centroid = regions[0].centroid
    
    print(rv_centroid)
    print(lv_centroid)

    label_after = np.tile(label[:, :, None], (1, 1, 3)) * 255
    label_after[int(rv_centroid[0]), int(rv_centroid[1])] = [255, 0, 0]
    label_after[int(lv_centroid[0]), int(lv_centroid[1])] = [0, 255, 0]

    fig, axes = plt.subplots(ncols=4, figsize=(20, 8), sharex=True, sharey=True)
    ax = axes.ravel()

    ax[0].imshow(label, cmap=plt.cm.gray)
    ax[0].set_title('Overlapping objects')
    ax[1].imshow(-distance, cmap=plt.cm.gray)
    ax[1].set_title('Distances')
    ax[2].imshow(labels, cmap=plt.cm.gray)
    ax[2].set_title('Separated objects')
    ax[3].imshow(label_after, cmap=plt.cm.gray)
    plt.show()

    return rv_centroid, lv_centroid

def get_angle(label):
    rv_centroid, _ = get_rv_centroid(label)

    center = skimage.measure.regionprops(label)[0].centroid
    m = ((label.shape[-1] - rv_centroid[0]) - (label.shape[-1] - center[0])) / (rv_centroid[1] - center[1])
    angle = math.degrees(np.pi - math.atan(m))
    return set_angle(label, angle)

def watershed(label):
    label = label.astype(np.int32)
    distance = ndi.distance_transform_edt(label)
    coords = peak_local_max(distance, min_distance=2, footprint=np.ones((5, 5)), num_peaks=2, labels=label)
    print(coords)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    labels = skimage.segmentation.watershed(-distance, markers, mask=label)
    regions = skimage.measure.regionprops(labels)
    p = regions[0].centroid
    label_after = np.tile(label[:, :, None], (1, 1, 3)) * 255
    label_after[int(p[0]), int(p[1])] = [255, 0, 0]

    fig, axes = plt.subplots(ncols=4, figsize=(20, 8), sharex=True, sharey=True)
    ax = axes.ravel()

    ax[0].imshow(label, cmap=plt.cm.gray)
    ax[0].set_title('Overlapping objects')
    ax[1].imshow(-distance, cmap=plt.cm.gray)
    ax[1].set_title('Distances')
    ax[2].imshow(labels, cmap=plt.cm.gist_heat)
    ax[2].set_title('Separated objects')
    ax[3].imshow(label_after, cmap=plt.cm.gray)
    plt.show()





#new_folder_name = 'ACDC_cropped'
#
#dirpath = Path(new_folder_name)
#if dirpath.exists() and dirpath.is_dir():
#    shutil.rmtree(dirpath)
#
#get_images(new_folder_name, 'binary_rotated\config.yaml', 'ACDC_data_rotated', 'binary_rotated\weights.pth', 224, 160)