import os
from glob import glob
import numpy as np
from tqdm import tqdm
import pystrum.pynd.ndutils as nd
from batchgenerators.utilities.file_and_folder_operations import save_json
import argparse
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from kornia.filters import spatial_gradient3d
import torch
import csv
import pickle
import nibabel as nib

def jacobian_determinant(disp):
    """
    jacobian determinant of a displacement field.
    NB: to compute the spatial gradients, we use np.gradient.

    Parameters:
        disp: 2D or 3D displacement field of size [*vol_shape, nb_dims], 
              where vol_shape is of len nb_dims

    Returns:
        jacobian determinant (scalar)
    """

    # check inputs
    volshape = disp.shape[:-1]
    nb_dims = len(volshape)
    assert len(volshape) in (2, 3), 'flow has to be 2D or 3D'

    # compute grid
    grid_lst = nd.volsize2ndgrid(volshape)
    grid = np.stack(grid_lst, len(volshape))

    # compute gradients
    J = np.gradient(disp + grid)

    # 3D glow
    if nb_dims == 3:
        dx = J[0]
        dy = J[1]
        dz = J[2]

        # compute jacobian components
        Jdet0 = dx[..., 0] * (dy[..., 1] * dz[..., 2] - dy[..., 2] * dz[..., 1])
        Jdet1 = dx[..., 1] * (dy[..., 0] * dz[..., 2] - dy[..., 2] * dz[..., 0])
        Jdet2 = dx[..., 2] * (dy[..., 0] * dz[..., 1] - dy[..., 1] * dz[..., 0])

        return Jdet0 - Jdet1 + Jdet2

    else:  # must be 2

        dfdx = J[0]
        dfdy = J[1]

        return dfdx[..., 0] * dfdy[..., 1] - dfdy[..., 0] * dfdx[..., 1]


if __name__ == "__main__":

    # parse commandline args
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', required=True, type=int, help='image size after crop')
    args = parser.parse_args()

    # Raw here
    #dir_path = r"C:\Users\Portal\Documents\voxelmorph\results\VM-NCC\Lib\test\Raw\Flow"
    dir_path = r"C:\Users\Portal\Documents\voxelmorph\exposant\new\exposant_infini\Task032_Lib\fold_0\Lib\test\Raw\Backward_flow"

    if dir_path.split(os.sep)[-3] == 'val':
        gt_dirname = 'Lib_resampling_training_mask'
    else:
        gt_dirname = 'Lib_resampling_testing_mask'
    pkl_dirname = 'custom_lib_t_4'

    path_list = glob(os.path.join(dir_path, "**", "*.npz"), recursive=True)

    all_results = {'all': [], 'mean': {'Temporal gradient': None,
                                       'Spatial gradient': None,
                                       'abs(Mean jacobian - 1)': None,
                                       'total_sum': None,
                                       'negative_sum': None,
                                       'negative_%': None}}
    step = 1

    mean_jacobian_list = []
    mean_gradient_xy_list = []
    mean_gradient_z_list = []
    stat_list = []

    total = {'RV': 0.0, 'MYO': 0.0, 'LV': 0.0}

    patient_list = sorted(list(set([os.path.basename(x).split('_')[0] for x in path_list])))

    all_patient_paths = []
    for patient in patient_list:
        patient_files = []
        for path in path_list:
            if patient in path:
                patient_files.append(path)
        all_patient_paths.append(sorted(patient_files))
    
    for patient_path in tqdm(all_patient_paths):

        patient_nb = os.path.basename(os.path.dirname(patient_path[0]))
        pkl_path = os.path.join(pkl_dirname, patient_nb, "info_01.pkl")

        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
            es_idx = np.round(data['es_number']).astype(int)
            ed_idx = np.round(data['ed_number']).astype(int)

        patient_path.insert(ed_idx, None)
        es_path = patient_path[es_idx]
        del patient_path[ed_idx]

        video_flow = []
        video_flow_gt = []
        for i, path in enumerate(patient_path):

            filename = os.path.basename(path)[:-4] + '.npy'

            corresponding_gt_path = os.path.join(gt_dirname, filename)
            data = np.load(corresponding_gt_path)
            arr = data[1]
            video_flow_gt.append(arr)

            if path == es_path:
                es_idx = i

            data = np.load(path)
            flow = data['flow']
            video_flow.append(flow)
        video_flow = np.stack(video_flow, axis=0)
        video_flow_gt = np.stack(video_flow_gt, axis=0)
        T, H, W, D, C = video_flow.shape

        for d in range(video_flow.shape[3]):
            if 'patient029' in patient_path[0] and d == 0:
                continue
            slice_gt = video_flow_gt[:, :, :, d]
            slice_flow = video_flow[:, :, :, d]
            gradient = spatial_gradient3d(torch.from_numpy(slice_flow.transpose(3, 0, 1, 2)[None])).double().numpy()
            gradient_xy = np.abs(gradient[:, :, :2])
            gradient_z = np.abs(gradient[:, :, 2])
            
            for t in range(T):
                frame_gt = slice_gt[t]
                frame_flow = slice_flow[t]
                jacobian = jacobian_determinant(frame_flow)

                current_res = {'Name': os.path.basename(patient_path[t]).split('.')[0],
                               'Slice nb': float(d),
                               'Frame nb': float(t),
                               'Temporal gradient': gradient_z[:, :, t].mean(),
                               'Spatial gradient': gradient_xy[:, :, :, t].mean()
                            }

                for i, k in enumerate(total, 1):
                    current_jacobian = jacobian[frame_gt == i]

                    current_total = float(current_jacobian.size)
                    current_negative = float((current_jacobian < 0).sum())

                    current_mean_jacobian = current_jacobian.mean()

                    current_res['abs(Mean jacobian - 1)_' + k] = abs(current_mean_jacobian - 1)
                    current_res['total_' + k] = current_total
                    current_res['negative_' + k] = current_negative
                    current_res['negative_%_' + k] = (current_negative/current_total) * 100
                
                current_res['abs(Mean jacobian - 1)_average'] = (current_res['abs(Mean jacobian - 1)_LV'] + current_res['abs(Mean jacobian - 1)_RV'] + current_res['abs(Mean jacobian - 1)_MYO']) / 3
                current_res['negative_%_average'] = (current_res['negative_%_LV'] + current_res['negative_%_RV'] + current_res['negative_%_MYO']) / 3
                
                whole_negative = float((jacobian < 0).sum())
                whole_total = float(jacobian.size)
                current_res['abs(Mean jacobian - 1)'] = abs(jacobian.mean() - 1)
                current_res['total'] = whole_total
                current_res['negative'] = whole_negative
                current_res['negative_%'] = (whole_negative/whole_total) * 100
                
                if t == es_idx:
                    current_res['ES_negative_%'] = current_res['negative_%']
                    current_res['ES_negative_%_average'] = current_res['negative_%_average']
                    current_res['ES_negative_%_average_RV'] = current_res['negative_%_RV']
                    current_res['ES_negative_%_average_MYO'] = current_res['negative_%_MYO']
                    current_res['ES_negative_%_average_LV'] = current_res['negative_%_LV']
                else:
                    current_res['ES_negative_%'] = np.nan
                    current_res['ES_negative_%_average'] = np.nan
                    current_res['ES_negative_%_average_RV'] = np.nan
                    current_res['ES_negative_%_average_MYO'] = np.nan
                    current_res['ES_negative_%_average_LV'] = np.nan

                all_results['all'].append(current_res)


    all_results['mean']['ES_negative_%_mean'] = np.nanmean(np.array([x['ES_negative_%'] for x in all_results['all']]))
    all_results['mean']['Temporal gradient'] = np.array([x['Temporal gradient'] for x in all_results['all']]).mean()
    all_results['mean']['Spatial gradient'] = np.array([x['Spatial gradient'] for x in all_results['all']]).mean()
    all_results['mean']['negative_sum'] = np.array([x['negative'] for x in all_results['all']]).sum()
    all_results['mean']['total_sum'] = np.array([x['total'] for x in all_results['all']]).sum()
    all_results['mean']['negative_%'] = (all_results['mean']['negative_sum'] / all_results['mean']['total_sum']) * 100
    all_results['mean']['negative_%_mean'] = np.array([x['negative_%'] for x in all_results['all']]).mean()
    all_results['mean']['abs(Mean jacobian - 1)'] = np.array([x['abs(Mean jacobian - 1)'] for x in all_results['all']]).mean()

    all_results['mean']['negative_%_mean_average'] = np.array([x['negative_%_average'] for x in all_results['all']]).mean()
    all_results['mean']['abs(Mean jacobian - 1)_average'] = np.array([x['abs(Mean jacobian - 1)_average'] for x in all_results['all']]).mean()
    all_results['mean']['ES_negative_%_mean_average'] = np.nanmean(np.array([x['ES_negative_%_average'] for x in all_results['all']]))

    for i, k in enumerate(total):
        all_results['mean']['negative_sum_' + k] = np.array([x['negative_' + k] for x in all_results['all']]).sum()
        all_results['mean']['total_sum_' + k] = np.array([x['total_' + k] for x in all_results['all']]).sum()
        all_results['mean']['negative_%_' + k] = (all_results['mean']['negative_sum_' + k] / all_results['mean']['total_sum_' + k]) * 100
        all_results['mean']['negative_%_mean_' + k] = np.array([x['negative_%_' + k] for x in all_results['all']]).mean()
        all_results['mean']['abs(Mean jacobian - 1)_' + k] = np.array([x['abs(Mean jacobian - 1)_' + k] for x in all_results['all']]).mean()
        all_results['mean']['ES_negative_%_average_' + k] = np.array([x['ES_negative_%_average_' + k] for x in all_results['all']]).mean()

    save_json(all_results, os.path.join(dir_path, 'jacobian.json'))

    with open(os.path.join(dir_path, 'jacobian_metrics.csv'), 'w') as fd_csv:
        writer = csv.DictWriter(fd_csv, fieldnames=list(all_results['all'][0].keys()))
        writer.writeheader() 
        writer.writerows(all_results['all']) 