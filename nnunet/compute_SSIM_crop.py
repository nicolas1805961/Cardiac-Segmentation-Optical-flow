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
path = r"C:\Users\Portal\Documents\voxelmorph\icpr_models\2024-07-09_17H24_52s_461126\Task032_Lib\fold_0\Lib\test\Raw\Backward_flow"

patient_path_list = glob(os.path.join(path, 'patient*'))

results = {'all': [], 'ES': None}

for patient_path in tqdm(patient_path_list):
    npz_path_list = glob(os.path.join(patient_path, '*.npz'))
    frame_nb_list = [int(os.path.basename(x).split('.')[0][-2:]) for x in npz_path_list]
    try:
        idx = np.where(np.diff(frame_nb_list) == 2)[0][0]
    except:
        idx = frame_nb_list[-2]

    ed_frame_nb = frame_nb_list[idx] + 1
    assert ed_frame_nb not in frame_nb_list

    current_patient_nb = os.path.basename(patient_path)

    with open(os.path.join('custom_lib_t_4', current_patient_nb, 'info_01.pkl'), 'rb') as f:
        data = pickle.load(f)
        es_idx = np.rint(data['es_number']).astype(int)

    ed_data = np.load(os.path.join('Lib_resampling_testing_mask', current_patient_nb + '_frame' + str(ed_frame_nb).zfill(2) + '.npy'))
    arr_ed = ed_data[0]
    arr_ed = arr_ed.astype(float)

    img_path_list = glob(os.path.join('Lib_resampling_testing_mask', current_patient_nb + '*.npy'))
    img_path_list = sorted([x for x in img_path_list if str(ed_frame_nb).zfill(2) not in os.path.basename(x).split('_')[-1]])
    flow_path_list = sorted(glob(os.path.join(path, current_patient_nb, '*.npz')))

    indices = np.arange(len(img_path_list))
    indices = np.concatenate([indices[indices >= ed_frame_nb - 1], indices[indices < ed_frame_nb - 1]])
    
    img_path_list = np.array(img_path_list)[indices]
    flow_path_list = np.array(flow_path_list)[indices]

    video_ssim = []
    es_ssim = []
    for flow_path, moving_path in zip(flow_path_list, img_path_list):
        data = np.load(flow_path)

        if 'registered' in data.files:
            depth_ssim_list = []
            for d in range(arr_ed.shape[-1]):
                if 'patient029' in flow_path and d == 0:
                    continue

                registered = data['registered'][:, :, d]
                assert registered.min() >= 0
                assert registered.max() <= 1.0
                ssim = structural_similarity(arr_ed[:, :, d], registered, data_range=registered.max() - registered.min())
                depth_ssim_list.append(ssim)
        else:
            flow = data['flow']
            flow = torch.from_numpy(flow).permute(3, 0, 1, 2).contiguous()

            img_data = np.load(moving_path)
            arr_img = img_data[0]
            arr_img = torch.from_numpy(arr_img).float()

            depth_ssim_list = []
            for d in range(arr_ed.shape[-1]):
                if 'patient029' in flow_path and d == 0:
                    continue

                registered = motion_estimation(flow=flow[:, :, :, d][None], original=arr_img[:, :, d][None, None], mode='bilinear')[0, 0].numpy()
                assert registered.min() >= 0
                assert registered.max() <= 1.0
                ssim = structural_similarity(arr_ed[:, :, d], registered, data_range=registered.max() - registered.min())
                depth_ssim_list.append(ssim)

        depth_ssim = np.stack(depth_ssim_list, axis=0)
        video_ssim.append(depth_ssim)

        if int(os.path.basename(flow_path).split('.')[0][-2:]) == es_idx + 1:
            es_ssim.append(depth_ssim)

    patient_ssim = np.stack(video_ssim, axis=0) # T-1, D
    es_ssim = np.stack(es_ssim, axis=0) # 1, D

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