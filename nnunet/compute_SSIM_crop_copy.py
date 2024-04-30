from glob import glob
import os
import pickle
import numpy as np
import nibabel as nib
from skimage.metrics import structural_similarity
from nnunet.network_architecture.integration import SpatialTransformer
import torch
import csv
import json
from tqdm import tqdm

def save_json(obj, file: str, indent: int = 4, sort_keys: bool = True) -> None:
    with open(file, 'w') as f:
        json.dump(obj, f, sort_keys=sort_keys, indent=indent)

motion_estimation = SpatialTransformer(size=(192, 192))

# flow raw directory
path = r"C:\Users\Portal\Documents\voxelmorph\2023-12-15_19H01\Task032_Lib\fold_0\Lib\test\Raw\Flow"

patient_path_list = glob(os.path.join(path, 'patient*'))

results = {'all': [], 'ES': None}

for patient_path in tqdm(patient_path_list):
    npz_path_list = glob(os.path.join(patient_path, '*.npz'))
    frame_nb_list = [int(os.path.basename(x).split('.')[0][-2:]) for x in npz_path_list]
    idx = np.where(np.diff(frame_nb_list) == 2)[0][0]
    ed_frame_nb = frame_nb_list[idx] + 1
    assert ed_frame_nb not in frame_nb_list

    current_patient_nb = os.path.basename(patient_path)

    with open(os.path.join('voxelmorph_Lib_2D_testing', current_patient_nb + '_frame01.pkl'), 'rb') as f:
        data = pickle.load(f)
        es_idx = np.rint(data['es_number']).astype(int)

    ed_data = nib.load(os.path.join('voxelmorph_Lib_2D_testing', current_patient_nb + '_frame' + str(ed_frame_nb).zfill(2) + '.nii.gz'))
    arr_ed = ed_data.get_fdata()
    arr_ed = arr_ed.astype(float)

    img_path_list = sorted(glob(os.path.join('voxelmorph_Lib_2D_testing', current_patient_nb + '*.gz')))
    flow_path_list = sorted(glob(os.path.join(path, current_patient_nb, '*.npz')))

    img_numbers = [int(os.path.basename(x).split('.')[0][-2:]) for x in img_path_list]
    flow_numbers = [int(os.path.basename(x).split('.')[0][-2:]) for x in flow_path_list]
    ed_number = [x for x in img_numbers if x not in flow_numbers][0]
    assert ed_number not in flow_numbers
    img_numbers = np.array(img_numbers)
    img_numbers = np.concatenate([img_numbers[img_numbers >= ed_number], img_numbers[img_numbers < ed_number]])

    es_idx = np.where(img_numbers - 1 == es_idx)[0]

    img_path_list = np.array(img_path_list)
    flow_path_list = np.array(flow_path_list)
    img_path_list = img_path_list[img_numbers - 1]
    flow_path_list = np.insert(flow_path_list, ed_number - 1, np.nan)
    flow_path_list = flow_path_list[img_numbers - 1]
    assert flow_path_list[0] == 'nan'

    flow_list = []
    img_list = []
    for flow_path, moving_path in zip(flow_path_list, img_path_list):

        img_data = nib.load(moving_path)
        arr_img = img_data.get_fdata()
        arr_img = torch.from_numpy(arr_img).float()
        img_list.append(arr_img)

        if flow_path == 'nan':
            continue

        data = np.load(flow_path)
        flow = data['flow']
        flow = torch.from_numpy(flow).permute(3, 0, 1, 2).contiguous()
        flow_list.append(flow)

    flow = torch.stack(flow_list)
    img = torch.stack(img_list)

    depth_ssim_list = []
    depth_es_ssim_list = []
    for d in range(flow.shape[-1]):
        if 'patient029' in patient_path and d == 0:
            continue
        current_flow = flow[:, :, :, :, d]
        current_img = img[:, :, :, d]

        video_ssim = []
        es_ssim = []
        for t1 in reversed(range(1, len(current_img))):
            current_moving_img = current_img[t1][None, None]
            for t2 in reversed(range(t1)):
                current_moving_img = motion_estimation(flow=current_flow[t2][None], original=current_moving_img, mode='bilinear')

            current_moving_img = current_moving_img[0, 0].numpy()
            assert current_moving_img.min() >= 0
            assert current_moving_img.max() <= 1.0
            ssim = structural_similarity(current_img[0].numpy(), current_moving_img, data_range=current_moving_img.max() - current_moving_img.min())
            video_ssim.append(ssim)

            if t1 == es_idx:
                es_ssim = ssim
            
        video_ssim = np.stack(video_ssim, axis=0)
        assert len(video_ssim) == len(img) - 1
        depth_ssim_list.append(video_ssim)

        depth_es_ssim_list.append(es_ssim)

    patient_ssim = np.stack(depth_ssim_list, axis=-1) # T-1, D
    es_ssim = np.stack(depth_es_ssim_list, axis=-1).reshape(1, -1) # 1, D

    for d in range(patient_ssim.shape[-1]):
        current_res = patient_ssim[:, d] # T-1
        current_res_es = es_ssim[0, d] # T-1
        current_res_info = {'Name': current_patient_nb,
                            'slice_number': d,
                            'SSIM': current_res.tolist(),
                            'ES_SSIM': current_res_es}
        results['all'].append(current_res_info)


results_csv = []
for i in range(len(results['all'])):
    current_dict = results['all'][i].copy()
    current_dict['SSIM'] = np.array(current_dict['SSIM']).mean()
    current_dict['ES_SSIM'] = current_dict['ES_SSIM']
    results_csv.append(current_dict)

with open(os.path.join(os.path.join(path, 'ssim_metrics.csv')), 'w') as fd_csv:
    writer = csv.DictWriter(fd_csv, fieldnames=list(results_csv[0].keys()))
    writer.writeheader() 
    writer.writerows(results_csv)

results['mean'] = np.array([np.array(x['SSIM']).mean() for x in results['all']]).mean()

save_json(results, os.path.join(path, 'ssim_metrics.json'))