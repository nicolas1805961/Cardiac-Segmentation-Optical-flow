import numpy as np
import os
from glob import glob
from multiprocessing import Pool
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
import nibabel as nib
import pickle
import argparse
import csv
import torch
from network_architecture.integration import SpatialTransformerContour
from dateutil.parser import parse
from nnunet.lib.training_utils import read_config_video
from numpy.linalg import norm
import cv2 as cv
from skimage.measure import regionprops
import scipy

#fourcc = cv.VideoWriter_fourcc(*'MJPG')
fourcc = cv.VideoWriter_fourcc(*'mp4v')


def curvature(x, y):
    dx = np.gradient(x)
    dy = np.gradient(y)
    d2x = np.gradient(dx)
    d2y = np.gradient(dy)
    
    curvature = np.abs((dx * d2y - dy * d2x) / ((dx**2 + dy**2)**(3/2)))
    
    return curvature

def smoothness_measure(x, y):
    curv = curvature(x, y)
    smoothness = np.var(curv)
    
    return smoothness


def save_json(obj, file: str, indent: int = 4, sort_keys: bool = True) -> None:
    with open(file, 'w') as f:
        json.dump(obj, f, sort_keys=sort_keys, indent=indent)


def is_date(string, fuzzy=False):
    """
    Return whether the string can be interpreted as a date.

    :param string: str, string to check for date
    :param fuzzy: bool, ignore unknown tokens in string if True
    """
    try: 
        parse(string, fuzzy=fuzzy)
        return True

    except ValueError:
        return False


def from_ed_accumulation(patient_name, phase_list_pkl, phase_list_img, phase_list):

    patient_results = {'all': []}
    with open(phase_list_pkl[0], 'rb') as f:
        data = pickle.load(f)
        es_number = np.rint(data['es_number']).astype(int) % len(phase_list)
        ed_number = np.rint(data['ed_number']).astype(int) % len(phase_list)

    video = []
    for phase in phase_list:
        data = np.load(phase)
        arr = data['flow'] # H, W, D, C
        arr = arr.transpose((2, 3, 0, 1)) # D, C, H, W
        video.append(arr)
    flow = np.stack(video, axis=1) # D, T, C, H, W
    flow = flow.transpose(0, 1, 2, 4, 3) # D, T, C, W, H
    flow = np.insert(flow, ed_number, values=np.nan, axis=1)

    spatial_transformer = SpatialTransformerContour(size=flow.shape[-2:])

    video_img = []
    for phase_img in phase_list_img:
        data = nib.load(phase_img)
        arr = data.get_fdata()
        arr = arr.transpose((2, 0, 1)) # D, H, W
        video_img.append(arr)
    img = np.stack(video_img, axis=1) # D, T, H, W

    frame_indices = np.arange(len(phase_list_img))

    before_where = np.argwhere(frame_indices < ed_number).reshape(-1,)
    after_where = np.argwhere(frame_indices >= ed_number).reshape(-1,)

    all_where = np.concatenate([after_where, before_where])

    frame_indices = frame_indices[all_where]
    img = img[:, frame_indices]
    flow = flow[:, frame_indices]

    assert frame_indices[0] == ed_number

    #fig, ax = plt.subplots(1, 1)
    #X, Y = np.meshgrid(np.arange(0, flow.shape[-1], step=step), np.arange(flow.shape[-2], step=step))
    #ax.imshow(img[0, es_number, :, :], cmap='gray')
    #ax.quiver(X, Y, flow[0, es_number, 1, ::step, ::step], flow[0, es_number, 0, ::step, ::step], color='r', angles='xy', scale_units='xy', scale=1)
    #plt.show()

    slice_error_list = []
    for d in range(len(flow)):
        current_slice_flow = flow[d] # T, 2, H, W
        slice_nb = str(d + 1).zfill(2)
        filename = patient_name + '_slice' + slice_nb + '.npy'
        gt_path_lv = os.path.join(gt_folder_name, 'contour', 'LV', filename)
        gt_path_rv = os.path.join(gt_folder_name, 'contour', 'RV', filename)
        gt_lv_contour = np.load(gt_path_lv).transpose((2, 1, 0)) # T, P1, 4
        gt_rv_contour = np.load(gt_path_rv).transpose((2, 1, 0)) # T, P2, 2
        gt_endo_contour = gt_lv_contour[:, :, :2]
        gt_epi_contour = gt_lv_contour[:, :, 2:]
        split_index = np.cumsum([gt_endo_contour.shape[1], gt_epi_contour.shape[1]])
        contours = np.concatenate([gt_endo_contour, gt_epi_contour, gt_rv_contour], axis=1) # T, P, 2

        contours = contours[frame_indices] - 1

        contours = torch.from_numpy(contours.copy()).float()
        current_slice_flow = torch.from_numpy(current_slice_flow.copy()).float()

        temporal_error_list = []
        for t in range(1, len(current_slice_flow)):
            current_contours = contours[0] # P, 2
            next_contours = contours[t] # P, 2
            gt_delta = next_contours - current_contours

            current_contours = current_contours.transpose(1, 0) # 2, P
            current_contours = current_contours[None, :, None, :] # 1, 2, 1, P

            initial_contours = current_contours

            for t2 in range(1, t + 1):
                current_frame_flow = current_slice_flow[t2] # 2, H, W: index 0 is nan
                delta_pred = spatial_transformer(torch.clone(current_contours), current_frame_flow[None])
                current_contours = current_contours + delta_pred
            
            delta_pred = (current_contours - initial_contours).squeeze()
            delta_pred = delta_pred.permute(1, 0).numpy()

            error = norm(gt_delta - delta_pred, axis=1) # P, careful here x and y (voxelmorph)
            #error = np.abs(gt_delta - delta_pred).mean(-1) # P, careful here x and y (voxelmorph)
            #error = np.abs(gt_delta - np.flip(delta_pred, axis=-1)).mean(-1) # P, careful here x and y (not voxelmorph)
            temporal_error_list.append(error)
            #print(f'ERROR: {error.mean()}, ERROR2: {error_2.mean()}')

        temporal_error_list = np.stack(temporal_error_list, axis=0) # T, P

        error = np.split(temporal_error_list, indices_or_sections=split_index, axis=1) # [(T, P1) , (T, P2), (T, P3)]
        error = np.stack([x.mean(-1) for x in error], axis=-1) # T, 3

        slice_error_list.append(error)
    slice_error_list = np.stack(slice_error_list, axis=0) # D, T, 3

    #sorted_where = np.argsort(all_where)
    #slice_error_list_initial = slice_error_list[:, sorted_where]

    for i in range(slice_error_list.shape[0]):
        current_res = slice_error_list[i] # T, 3
        current_res_info = {'Name': patient_name,
                            'slice_number': i,
                            'ENDO': current_res[:, 0].tolist(),
                            'EPI': current_res[:, 1].tolist(),
                            'RV': current_res[:, 2].tolist()}
        patient_results['all'].append(current_res_info)

    return patient_results


def to_ed_accumulation(patient_name, phase_list_pkl, phase_list_img, phase_list):

    patient_results = {'all': []}
    with open(phase_list_pkl[0], 'rb') as f:
        data = pickle.load(f)
        es_number = np.rint(data['es_number']).astype(int) % len(phase_list)
        ed_number = np.rint(data['ed_number']).astype(int) % len(phase_list)

    video = []
    for phase in phase_list:
        data = np.load(phase)
        arr = data['flow'] # H, W, D, C
        arr = arr.transpose((2, 3, 0, 1)) # D, C, H, W
        video.append(arr)
    flow = np.stack(video, axis=1) # D, T, C, H, W
    flow = flow.transpose(0, 1, 2, 4, 3) # D, T, C, W, H
    flow = np.insert(flow, ed_number, values=np.nan, axis=1)

    spatial_transformer = SpatialTransformerContour(size=flow.shape[-2:])

    video_img = []
    for phase_img in phase_list_img:
        data = nib.load(phase_img)
        arr = data.get_fdata()
        arr = arr.transpose((2, 0, 1)) # D, H, W
        video_img.append(arr)
    img = np.stack(video_img, axis=1) # D, T, H, W

    frame_indices = np.arange(len(phase_list_img))

    before_where = np.argwhere(frame_indices < ed_number).reshape(-1,)
    after_where = np.argwhere(frame_indices >= ed_number).reshape(-1,)

    all_where = np.concatenate([after_where, before_where])

    frame_indices = frame_indices[all_where]
    img = img[:, frame_indices]
    flow = flow[:, frame_indices]

    assert frame_indices[0] == ed_number

    #fig, ax = plt.subplots(1, 1)
    #X, Y = np.meshgrid(np.arange(0, flow.shape[-1], step=step), np.arange(flow.shape[-2], step=step))
    #ax.imshow(img[0, es_number, :, :], cmap='gray')
    #ax.quiver(X, Y, flow[0, es_number, 1, ::step, ::step], flow[0, es_number, 0, ::step, ::step], color='r', angles='xy', scale_units='xy', scale=1)
    #plt.show()

    slice_error_list = []
    for d in range(len(flow)):
        current_slice_flow = flow[d] # T, 2, H, W
        slice_nb = str(d + 1).zfill(2)
        filename = patient_name + '_slice' + slice_nb + '.npy'
        gt_path_lv = os.path.join(gt_folder_name, 'contour', 'LV', filename)
        gt_path_rv = os.path.join(gt_folder_name, 'contour', 'RV', filename)
        gt_lv_contour = np.load(gt_path_lv).transpose((2, 1, 0)) # T, P1, 4
        gt_rv_contour = np.load(gt_path_rv).transpose((2, 1, 0)) # T, P2, 2
        gt_endo_contour = gt_lv_contour[:, :, :2]
        gt_epi_contour = gt_lv_contour[:, :, 2:]
        split_index = np.cumsum([gt_endo_contour.shape[1], gt_epi_contour.shape[1]])
        contours = np.concatenate([gt_endo_contour, gt_epi_contour, gt_rv_contour], axis=1) # T, P, 2

        contours = contours[frame_indices]

        current_slice_flow = np.flip(current_slice_flow, axis=0)
        # -1 here for matlab indexing
        contours = np.flip(contours, axis=0) - 1

        contours = torch.from_numpy(contours.copy()).float()
        current_slice_flow = torch.from_numpy(current_slice_flow.copy()).float()

        temporal_error_list = []
        for t in range(len(current_slice_flow) - 1):
            current_contours = contours[t] # P, 2
            next_contours = contours[-1] # P, 2
            gt_delta = next_contours - current_contours

            current_contours = current_contours.transpose(1, 0) # 2, P
            current_contours = current_contours[None, :, None, :] # 1, 2, 1, P
            initial_contours = current_contours
            
            for t2 in range(t, len(current_slice_flow) - 1):
                current_frame_flow = current_slice_flow[t2] # 2, H, W
                delta_pred = spatial_transformer(torch.clone(current_contours), current_frame_flow[None])
                current_contours = current_contours + delta_pred
            
            delta_pred = (current_contours - initial_contours).squeeze()
            delta_pred = delta_pred.permute(1, 0).numpy()

            error = norm(gt_delta - delta_pred, axis=1) # P, careful here x and y (voxelmorph)
            #error = np.abs(gt_delta - delta_pred).mean(-1) # P, careful here x and y (voxelmorph)
            #error = np.abs(gt_delta - np.flip(delta_pred, axis=-1)).mean(-1) # P, careful here x and y (not voxelmorph)
            temporal_error_list.append(error)
            #print(f'ERROR: {error.mean()}, ERROR2: {error_2.mean()}')

        temporal_error_list = np.stack(temporal_error_list, axis=0) # T, P
        temporal_error_list = np.flip(temporal_error_list, axis=0)

        error = np.split(temporal_error_list, indices_or_sections=split_index, axis=1) # [(T, P1) , (T, P2), (T, P3)]
        error = np.stack([x.mean(-1) for x in error], axis=-1) # T, 3

        slice_error_list.append(error)
    slice_error_list = np.stack(slice_error_list, axis=0) # D, T, 3

    #sorted_where = np.argsort(all_where)
    #slice_error_list_initial = slice_error_list[:, sorted_where]

    for i in range(slice_error_list.shape[0]):
        current_res = slice_error_list[i] # T, 3
        current_res_info = {'Name': patient_name, 
                            #'reference': gt_path_lv[:-12], 
                            #'test': phase_list[i + 1],
                            'slice_number': i,
                            'ENDO': current_res[:, 0].tolist(),
                            'EPI': current_res[:, 1].tolist(),
                            'RV': current_res[:, 2].tolist()}
        patient_results['all'].append(current_res_info)

    return patient_results



def to_ed(patient_name, phase_list_pkl, phase_list_img, phase_list):

    patient_results = {'all': []}
    with open(phase_list_pkl[0], 'rb') as f:
        data = pickle.load(f)
        es_number = np.rint(data['es_number']).astype(int) % len(phase_list)
        ed_number = np.rint(data['ed_number']).astype(int) % len(phase_list)

    video = []
    for phase in phase_list:
        data = np.load(phase)
        arr = data['flow'] # H, W, D, C
        arr = arr.transpose((2, 3, 0, 1)) # D, C, H, W
        video.append(arr)
    flow = np.stack(video, axis=1) # D, T, C, H, W
    flow = flow.transpose(0, 1, 2, 4, 3) # D, T, C, W, H
    flow = np.insert(flow, ed_number, values=np.nan, axis=1)

    spatial_transformer = SpatialTransformerContour(size=flow.shape[-2:])

    video_img = []
    for phase_img in phase_list_img:
        data = nib.load(phase_img)
        arr = data.get_fdata()
        arr = arr.transpose((2, 0, 1)) # D, H, W
        video_img.append(arr)
    img = np.stack(video_img, axis=1) # D, T, H, W

    frame_indices = np.arange(len(phase_list_img))

    before_where = np.argwhere(frame_indices < ed_number).reshape(-1,)
    after_where = np.argwhere(frame_indices >= ed_number).reshape(-1,)

    all_where = np.concatenate([after_where, before_where])

    frame_indices = frame_indices[all_where]
    img = img[:, frame_indices]
    flow = flow[:, frame_indices]

    assert frame_indices[0] == ed_number

    #fig, ax = plt.subplots(1, 1)
    #X, Y = np.meshgrid(np.arange(0, flow.shape[-1], step=step), np.arange(flow.shape[-2], step=step))
    #ax.imshow(img[0, es_number, :, :], cmap='gray')
    #ax.quiver(X, Y, flow[0, es_number, 1, ::step, ::step], flow[0, es_number, 0, ::step, ::step], color='r', angles='xy', scale_units='xy', scale=1)
    #plt.show()

    slice_error_list = []
    for d in range(len(flow)):
        current_slice_flow = flow[d] # T, 2, H, W
        slice_nb = str(d + 1).zfill(2)
        filename = patient_name + '_slice' + slice_nb + '.npy'
        gt_path_lv = os.path.join(gt_folder_name, 'contour', 'LV', filename)
        gt_path_rv = os.path.join(gt_folder_name, 'contour', 'RV', filename)
        gt_lv_contour = np.load(gt_path_lv).transpose((2, 1, 0)) # T, P1, 4
        gt_rv_contour = np.load(gt_path_rv).transpose((2, 1, 0)) # T, P2, 2
        gt_endo_contour = gt_lv_contour[:, :, :2]
        gt_epi_contour = gt_lv_contour[:, :, 2:]
        split_index = np.cumsum([gt_endo_contour.shape[1], gt_epi_contour.shape[1]])
        contours = np.concatenate([gt_endo_contour, gt_epi_contour, gt_rv_contour], axis=1) # T, P, 2

        contours = contours[frame_indices]

        current_slice_flow = np.flip(current_slice_flow, axis=0)
        # -1 here for matlab indexing
        contours = np.flip(contours, axis=0) - 1

        contours = torch.from_numpy(contours.copy()).float()
        current_slice_flow = torch.from_numpy(current_slice_flow.copy()).float()

        temporal_error_list = []
        for t in range(len(current_slice_flow) - 1):
            current_contours = contours[t] # P, 2
            next_contours = contours[-1] # P, 2
            gt_delta = next_contours - current_contours

            current_contours = current_contours.transpose(1, 0) # 2, P
            current_contours = current_contours[None, :, None, :] # 1, 2, 1, P

            current_frame_flow = current_slice_flow[t] # 2, H, W

            #current_frame_flow = current_frame_flow.transpose((1, 2, 0)) # H, W, 2
            #x = np.rint(current_contours[:, 0]).astype(int)
            #y = np.rint(current_contours[:, 1]).astype(int)
            
            #x_range = np.arange(0, current_frame_flow.shape[1])
            #y_range = np.arange(0, current_frame_flow.shape[0])
            #xv, yv = np.meshgrid(x_range, y_range)
            #grid_x = current_frame_flow[:, :, 0] + xv
            #grid_y = current_frame_flow[:, :, 1] + yv
            #grid_x = grid_x[y, x]
            #grid_y = grid_y[y, x]
            #grid_x_alt = current_frame_flow[:, :, 1] + xv
            #grid_y_alt = current_frame_flow[:, :, 0] + yv
            #grid_x_alt = grid_x_alt[y, x]
            #grid_y_alt = grid_y_alt[y, x]

            #fig, ax = plt.subplots(1, 2)
            #ax[0].imshow(img[d, -1, :, :], cmap='gray')
            #ax[1].imshow(img[d, -1, :, :], cmap='gray')
            #ax[0].scatter(grid_x, grid_y, c='r')
            #ax[0].scatter(next_contours[:, 0], next_contours[:, 1], c='g')
            #ax[1].scatter(grid_x_alt, grid_y_alt, c='b')
            #ax[1].scatter(next_contours[:, 0], next_contours[:, 1], c='g')
            #plt.show()

            delta_pred = spatial_transformer(torch.clone(current_contours), current_frame_flow[None]).squeeze()
            delta_pred = delta_pred.permute(1, 0).numpy()

            #delta_pred = current_frame_flow[y, x, :] # P, 2
            error = norm(gt_delta - delta_pred, axis=1) # P, careful here x and y (voxelmorph)
            #error = np.abs(gt_delta - delta_pred).mean(-1) # P, careful here x and y (voxelmorph)
            #error = np.abs(gt_delta - np.flip(delta_pred, axis=-1)).mean(-1) # P, careful here x and y (not voxelmorph)
            temporal_error_list.append(error)
            #print(f'ERROR: {error.mean()}, ERROR2: {error_2.mean()}')

        temporal_error_list = np.stack(temporal_error_list, axis=0) # T, P
        temporal_error_list = np.flip(temporal_error_list, axis=0)

        error = np.split(temporal_error_list, indices_or_sections=split_index, axis=1) # [(T, P1) , (T, P2), (T, P3)]
        error = np.stack([x.mean(-1) for x in error], axis=-1) # T, 3

        slice_error_list.append(error)
    slice_error_list = np.stack(slice_error_list, axis=0) # D, T, 3

    #sorted_where = np.argsort(all_where)
    #slice_error_list_initial = slice_error_list[:, sorted_where]

    for i in range(slice_error_list.shape[0]):
        current_res = slice_error_list[i] # T, 3
        current_res_info = {'Name': patient_name,
                            'slice_number': i,
                            'ENDO': current_res[:, 0].tolist(),
                            'EPI': current_res[:, 1].tolist(),
                            'RV': current_res[:, 2].tolist()}
        patient_results['all'].append(current_res_info)

    return patient_results


def from_ed(patient_name, phase_list_pkl, phase_list_img, phase_list):

    patient_results = {'all': []}
    with open(phase_list_pkl[0], 'rb') as f:
        data = pickle.load(f)
        mat_filename = data['FileName']
        mat_path = glob(os.path.join('data_lib_mat', '**', mat_filename), recursive=True)[0]
        mat = scipy.io.loadmat(mat_path, simplify_cells=True)
        my_list = mat['data']['orientations']['short_axis']['slices']
        ed_idx_list = np.rint(np.array([x['maxDilation'] for x in my_list if len(x['structures']) > 0])).astype(int) - 1
        es_idx_list = np.rint(np.array([x['systole'] for x in my_list if len(x['structures']) > 0])).astype(int) - 1
        es_number = np.rint(data['es_number']).astype(int) % len(phase_list)
        ed_number = np.rint(data['ed_number']).astype(int) % len(phase_list)
        #print(ed_number)
        #print(ed_idx_list)

    video = []
    for phase in phase_list:
        data = np.load(phase)
        arr = data['flow'] # H, W, D, C
        arr = arr.transpose((2, 3, 0, 1)) # D, C, H, W
        video.append(arr)
    flow = np.stack(video, axis=1) # D, T, C, H, W
    flow = flow.transpose(0, 1, 2, 4, 3) # D, T, C, W, H
    flow = np.insert(flow, ed_number, values=np.nan, axis=1)

    #assert len(flow) == len(ed_idx_list)

    spatial_transformer = SpatialTransformerContour(size=flow.shape[-2:])

    video_img = []
    for phase_img in phase_list_img:
        data = nib.load(phase_img)
        zoom = data.header.get_zooms()[:2]
        arr = data.get_fdata()
        arr = arr.transpose((2, 0, 1)) # D, H, W
        video_img.append(arr)
    img = np.stack(video_img, axis=1) # D, T, H, W

    zoom = np.array(list(zoom))
    assert zoom[0] == zoom[1]

    frame_indices = np.arange(len(phase_list_img))

    before_where = np.argwhere(frame_indices < ed_number).reshape(-1,)
    after_where = np.argwhere(frame_indices >= ed_number).reshape(-1,)

    to_roll = len(after_where)
    #print(ed_number)
    #print(to_roll)

    all_where = np.concatenate([after_where, before_where])

    frame_indices = frame_indices[all_where]
    img = img[:, frame_indices]
    flow = flow[:, frame_indices]

    assert frame_indices[0] == ed_number

    #fig, ax = plt.subplots(1, 1)
    #X, Y = np.meshgrid(np.arange(0, flow.shape[-1], step=step), np.arange(flow.shape[-2], step=step))
    #ax.imshow(img[0, es_number, :, :], cmap='gray')
    #ax.quiver(X, Y, flow[0, es_number, 1, ::step, ::step], flow[0, es_number, 0, ::step, ::step], color='r', angles='xy', scale_units='xy', scale=1)
    #plt.show()

    smooth_list = []

    #print(len(flow))

    for d in range(len(flow)):
        #print(d)
        #real_ed_idx = ed_idx_list[d]

        slice_nb = str(d + 1).zfill(2)
        filename = patient_name + '_slice' + slice_nb + '.npy'

        current_slice_img = img[d] # T, H, W
        current_slice_flow = flow[d] # T, 2, H, W
        gt_path_lv = os.path.join(gt_folder_name, 'contour', 'LV', filename)
        gt_path_rv = os.path.join(gt_folder_name, 'contour', 'RV', filename)
        gt_lv_contour = np.load(gt_path_lv).transpose((2, 1, 0)) # T, P1, 4
        gt_rv_contour = np.load(gt_path_rv).transpose((2, 1, 0)) # T, P2, 2
        gt_endo_contour = gt_lv_contour[:, :, :2]
        gt_epi_contour = gt_lv_contour[:, :, 2:]
        contours = np.stack([gt_endo_contour, gt_epi_contour], axis=0) # 2, T, P, 2

        contours = contours[:, frame_indices] - 1
        #contours = contours - 1

        contours = torch.from_numpy(contours.copy()).float()
        current_slice_flow = torch.from_numpy(current_slice_flow.copy()).float()
        current_slice_img = torch.from_numpy(current_slice_img.copy()).float()

        first_contours = contours[:, 0] # 2, P, 2
        first_contours = first_contours.permute(0, 2, 1) # 2, 2, P
        out_list = [torch.clone(first_contours)]
        first_contours = first_contours[:, :, None, :] # 2, 2, 1, P

        for t in range(1, len(current_slice_flow)):
            current_contours = contours[:, t].numpy() # 2, P, 2

            current_frame_flow = current_slice_flow[t] # 2, H, W; index 0 is nan so start at index 1
            current_frame_img = current_slice_img[t].numpy() # H, W

            #current_frame_flow = current_frame_flow.transpose((1, 2, 0)) # H, W, 2
            #x = np.rint(current_contours[:, 0]).astype(int)
            #y = np.rint(current_contours[:, 1]).astype(int)

            #x_range = np.arange(0, current_frame_flow.shape[1])
            #y_range = np.arange(0, current_frame_flow.shape[0])
            #xv, yv = np.meshgrid(x_range, y_range)
            #grid_x = current_frame_flow[:, :, 0] + xv
            #grid_y = current_frame_flow[:, :, 1] + yv
            #grid_x = grid_x[y, x]
            #grid_y = grid_y[y, x]
            #grid_x_alt = current_frame_flow[:, :, 1] + xv
            #grid_y_alt = current_frame_flow[:, :, 0] + yv
            #grid_x_alt = grid_x_alt[y, x]
            #grid_y_alt = grid_y_alt[y, x]

            #fig, ax = plt.subplots(1, 2)
            #ax[0].imshow(img[d, -1, :, :], cmap='gray')
            #ax[1].imshow(img[d, -1, :, :], cmap='gray')
            #ax[0].scatter(grid_x, grid_y, c='r')
            #ax[0].scatter(next_contours[:, 0], next_contours[:, 1], c='g')
            #ax[1].scatter(grid_x_alt, grid_y_alt, c='b')
            #ax[1].scatter(next_contours[:, 0], next_contours[:, 1], c='g')
            #plt.show()

            delta_pred = spatial_transformer(torch.clone(first_contours), current_frame_flow[None].repeat(2, 1, 1, 1))
            new_predicted_points = (first_contours + delta_pred).squeeze() # structure, C, P
            out_list.append(new_predicted_points)
        
        zoom_reshaped = zoom.reshape(1, 1, 2, 1)
        contour_points = torch.stack(out_list, dim=0) # T, structure, C, P
        contour_points = contour_points * zoom_reshaped # Convert to mm

        radial_distances = torch.linalg.norm(torch.diff(contour_points, dim=1), dim=2).squeeze() # T, P
        radial_strain = 0.5 * ((radial_distances**2 - radial_distances[0][None]**2) / radial_distances[0][None]**2)

        #to_roll = torch.argmin(radial_strain.mean(-1)).item()
        #to_roll = real_ed_idx - ed_number
        #print(to_roll)

        unfolded = torch.cat([contour_points, contour_points[:, :, :, 0][:, :, :, None]], dim=-1)
        unfolded = unfolded.unfold(-1, 2, 1)
        circ_distances = torch.linalg.norm(torch.diff(unfolded, dim=-1), dim=2).squeeze() # T, structure, P
        circ_distances = circ_distances.mean(1) # T, P
        circ_strain = 0.5 * ((circ_distances**2 - circ_distances[0][None]**2) / circ_distances[0][None]**2)

        radial_strain = torch.roll(radial_strain, shifts=-to_roll, dims=[0]).mean(-1)
        circ_strain = torch.roll(circ_strain, shifts=-to_roll, dims=[0]).mean(-1)

        fig, ax = plt.subplots(1, 2)
        ax[0].plot(radial_strain)
        ax[1].plot(circ_strain)
        plt.show()

        smooth_radial = smoothness_measure(np.arange(len(radial_strain)), radial_strain)
        smooth_circ = smoothness_measure(np.arange(len(circ_strain)), circ_strain)

        smooth = (smooth_radial + smooth_circ) / 2

        smooth_list.append(smooth)
    
    smooth = np.array(smooth_list)
    return smooth.mean()

        



if __name__ == "__main__":

    # parse commandline args
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_or_val', required=True, help='Whether this is testing or validation set')
    args = parser.parse_args()

    results = {"all": [], "mean": None}
    results_per_patient = {"all": {}, "mean": None}
    step = 2

    default_num_threads = 1

    if args.test_or_val == 'val':
        gt_folder_name = r"C:\Users\Portal\Documents\Isensee\nnUNet\nnunet\out\nnUNet_preprocessed\Task032_Lib"
        gt_folder_lib = 'Lib_training_2'
    elif args.test_or_val == 'test':
        gt_folder_name = r"C:\Users\Portal\Documents\Isensee\nnUNet\nnunet\out\nnUNet_preprocessed\Task036_Lib"
        gt_folder_lib = 'Lib_testing_2'

    #pred_file_folder = os.path.join(pred_folder, r"C:\Users\Portal\Documents\voxelmorph\results\Lib_regu_not_all_small\Validation\Flow")
    #pred_file_folder = os.path.join(pred_folder, r"C:\Users\Portal\Documents\voxelmorph\testings_no_regu\inference\Validation\Flow")
    #pred_file_folder = os.path.join(pred_folder, r"C:\Users\Portal\Documents\voxelmorph\testings2_small\inference\Validation\Flow")

    # Postprocessed Flow here
    #pred_file_folder = r"C:\Users\Portal\Documents\voxelmorph\2023-12-14_17H47\Task032_Lib\fold_0\Lib\val\Postprocessed\Flow"
    pred_file_folder = r"C:\Users\Portal\Documents\voxelmorph\2024-01-18_18H32\Task032_Lib\fold_0\Lib\val\Postprocessed\Flow"

    splitted = pred_file_folder.split(os.sep)
    is_date_list = [is_date(x, True) for x in splitted]
    try:
        stop_idx = np.where(is_date_list)[0][0] + 1
        splitted = splitted[:stop_idx]
        splitted[0] = splitted[0] + '/'
        config = read_config_video(os.path.join(*splitted, 'config.yaml'))
    except:
        config = {'motion_from_ed': True, 'dataloader_modality': 'other'}

    if config['motion_from_ed']:
        if config['dataloader_modality'] == 'all_adjacent':
            applied_function = from_ed_accumulation
        else:
            applied_function = from_ed
    else:
        if config['dataloader_modality'] == 'all_adjacent':
            applied_function = to_ed_accumulation
        else:
            applied_function = to_ed

    registered_patients = [name for name in os.listdir(pred_file_folder) if os.path.isdir(os.path.join(pred_file_folder, name))]

    smooth_measure_list = []

    for patient_name in tqdm(registered_patients[1:2]):
    
        all_files = glob(os.path.join(pred_file_folder, patient_name, '*.npz'))
        all_files = sorted([x for x in all_files if '_raw.npz' not in x])
        #patient_names = sorted(list(set([os.path.basename(x).split('_')[0] for x in all_files])))

        all_files_img = glob(os.path.join(gt_folder_lib, patient_name, '*.gz'))
        all_files_img = sorted([x for x in all_files_img if '_gt' not in x])

        all_files_pkl = glob(os.path.join(gt_folder_lib, patient_name, '*.pkl'))

        phase_list = np.array(sorted(all_files, key=lambda x: int(os.path.basename(x).split('.')[0][-2:])))
        phase_list_img = np.array(sorted(all_files_img, key=lambda x: int(os.path.basename(x).split('.')[0][-2:])))
        phase_list_pkl = np.array(sorted(all_files_pkl, key=lambda x: int(os.path.basename(x).split('.')[0][-2:])))

        current_smooth = applied_function(patient_name, phase_list_pkl, phase_list_img, phase_list)
        smooth_measure_list.append(current_smooth)

    print(np.array(smooth_measure_list).mean())

     