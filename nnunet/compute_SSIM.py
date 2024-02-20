import numpy as np
from glob import glob
import os
import torch
from nnunet.network_architecture.integration import SpatialTransformer, SpatialTransformerContour
import nibabel as nib
import matplotlib.pyplot as plt
import pickle
import cv2 as cv
from skimage.metrics import structural_similarity
import csv
import json
from tqdm import tqdm

def save_json(obj, file: str, indent: int = 4, sort_keys: bool = True) -> None:
    with open(file, 'w') as f:
        json.dump(obj, f, sort_keys=sort_keys, indent=indent)

results = {'all': []}
gt_directory = r'out\nnUNet_preprocessed\Task036_Lib'

path_method = r"C:\Users\Portal\Documents\voxelmorph\results\VM-DIF\Lib\test"

patient_list = [name for name in os.listdir(os.path.join(path_method, 'Postprocessed', 'Flow')) if os.path.isdir(os.path.join(path_method, 'Postprocessed', 'Flow', name))]

for patient_name in tqdm(patient_list):

    corresponding_pkl_file = os.path.join(gt_directory, 'custom_experiment_planner_stage0', patient_name + '_frame01.pkl')
    with open(corresponding_pkl_file, 'rb') as f:
        data = pickle.load(f)
        ed_number = np.rint(data['ed_number']).astype(int)
        es_number = np.rint(data['es_number']).astype(int)

    all_files = glob(os.path.join('Lib_testing_2', patient_name, '*.gz'))
    all_files_img = sorted([x for x in all_files if '_gt' not in x])
    phase_list_img = np.array(sorted(all_files_img, key=lambda x: int(os.path.basename(x).split('.')[0][-2:])))

    all_files_flow = glob(os.path.join(path_method, 'Postprocessed', 'Flow', patient_name, '*.npz'))
    flow_path_list = np.array(sorted(all_files_flow, key=lambda x: int(os.path.basename(x).split('.')[0][-2:])))

    video_img = []
    for phase_img in phase_list_img:
        data_img = nib.load(phase_img)
        arr_img = data_img.get_fdata()
        arr_img = arr_img.transpose((2, 0, 1)) # D, H, W
        video_img.append(arr_img)
    img = np.stack(video_img, axis=1) # D, T, H, W

    video_flow = []
    for path in flow_path_list:
        data = np.load(path)
        arr = data['flow'] # H, W, D, C
        arr = arr.transpose((2, 3, 0, 1)) # D, C, H, W
        video_flow.append(arr)
    big_flow = np.stack(video_flow, axis=1) # D, T, C, H, W
    big_flow = big_flow.transpose(0, 1, 2, 4, 3) # D, T, C, W, H
    big_flow = np.insert(big_flow, ed_number, values=np.nan, axis=1)

    assert big_flow.shape[1] == img.shape[1]

    frame_indices = np.arange(len(phase_list_img))

    before_where = np.argwhere(frame_indices < ed_number).reshape(-1,)
    after_where = np.argwhere(frame_indices >= ed_number).reshape(-1,)

    all_where = np.concatenate([after_where, before_where])

    frame_indices = frame_indices[all_where]
    img = img[:, frame_indices]
    big_flow = big_flow[:, frame_indices]

    assert frame_indices[0] == ed_number

    motion_estimation = SpatialTransformer(size=(img.shape[-2], img.shape[-1]))
    patient_ssim = []
    for d in range(len(big_flow)):
        current_fixed = img[d, 0] # H, W
        video_ssim = []
        for t in range(big_flow.shape[1]):
            current_flow = big_flow[d, t] # C, H, W
            current_moving = img[d, t] # H, W

            if np.all(~np.isfinite(current_flow)):
                continue

            current_moving = torch.from_numpy(current_moving).float() # H, W
            current_flow = torch.from_numpy(current_flow)

            registered = motion_estimation(flow=torch.flip(current_flow, dims=[0])[None], original=current_moving[None, None], mode='bilinear').squeeze().numpy()

            ssim = structural_similarity(current_fixed, registered, data_range=registered.max() - registered.min())

            video_ssim.append(ssim)

            #fig, ax = plt.subplots(1, 3)
            #ax[0].imshow(current_fixed, cmap='gray')
            #ax[1].imshow(registered, cmap='gray')
            #ax[2].imshow(current_moving, cmap='gray')
            #plt.show()
        
        video_ssim = np.stack(video_ssim, axis=0)
        patient_ssim.append(video_ssim)

    patient_ssim = np.stack(patient_ssim, axis=0) # D, T-1

    for d in range(patient_ssim.shape[0]):
        current_res = patient_ssim[d] # T-1
        current_res_info = {'Name': patient_name,
                            'slice_number': d,
                            'SSIM': current_res.tolist()}
        results['all'].append(current_res_info)

results_csv = []
for i in range(len(results['all'])):
    current_dict = results['all'][i].copy()
    current_dict['SSIM'] = np.array(current_dict['SSIM']).mean()
    results_csv.append(current_dict)

with open(os.path.join(os.path.join(path_method, 'Postprocessed', 'Flow', 'ssim_metrics.csv')), 'w') as fd_csv:
    writer = csv.DictWriter(fd_csv, fieldnames=list(results_csv[0].keys()))
    writer.writeheader() 
    writer.writerows(results_csv)

results['mean'] = np.array([np.array(x['SSIM']).mean() for x in results['all']]).mean()

save_json(results, os.path.join(path_method, 'Postprocessed', 'Flow', 'ssim_metrics.json'))