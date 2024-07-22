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
import matplotlib.pyplot as plt

def save_json(obj, file: str, indent: int = 4, sort_keys: bool = True) -> None:
    with open(file, 'w') as f:
        json.dump(obj, f, sort_keys=sort_keys, indent=indent)

motion_estimation = SpatialTransformer(size=(192, 192))

# flow raw directory
path = r"C:\Users\Portal\Documents\voxelmorph\iterative_warp\2024-07-02_11H29_59s_847044\iterative_sum\Lib\test\Raw\Backward_flow"

patient_path_list = glob(os.path.join(path, 'patient*'))

results = {'all': []}

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

    seg_ed = ed_data[1]

    img_path_list = glob(os.path.join('Lib_resampling_testing_mask', current_patient_nb + '*.npy'))
    img_path_list = sorted([x for x in img_path_list if str(ed_frame_nb).zfill(2) not in os.path.basename(x).split('_')[-1]])
    flow_path_list = sorted(glob(os.path.join(path, current_patient_nb, '*.npz')))

    indices = np.arange(len(img_path_list))
    indices = np.concatenate([indices[indices >= ed_frame_nb - 1], indices[indices < ed_frame_nb - 1]])
    
    img_path_list = np.array(img_path_list)[indices]
    flow_path_list = np.array(flow_path_list)[indices]

    video_ssim_mean = []
    video_ssim_rv = []
    video_ssim_myo = []
    video_ssim_lv = []
    video_ssim_whole = []
    es_ssim_mean = []
    es_ssim_rv = []
    es_ssim_myo = []
    es_ssim_lv = []
    es_ssim_whole = []
    for flow_path, moving_path in zip(flow_path_list, img_path_list):
        data = np.load(flow_path)

        depth_ssim_list_mean = []
        depth_ssim_list_rv = []
        depth_ssim_list_myo = []
        depth_ssim_list_lv = []
        depth_ssim_list_whole = []

        if 'registered' in data.files:
            depth_ssim_list = []
            for d in range(arr_ed.shape[-1]):
                if 'patient029' in flow_path and d == 0:
                    continue

                registered = data['registered'][:, :, d]
                assert registered.min() >= 0
                assert registered.max() <= 1.0
                ssim, ssim_img = structural_similarity(arr_ed[:, :, d], registered, data_range=registered.max() - registered.min(), full=True)
                
                rv_ssim = ssim_img[seg_ed[:, :, d] == 1].mean()
                myo_ssim = ssim_img[seg_ed[:, :, d] == 2].mean()
                lv_ssim = ssim_img[seg_ed[:, :, d] == 3].mean()
                depth_ssim_list_mean.append((rv_ssim + myo_ssim + lv_ssim) / 3)
                depth_ssim_list_rv.append(rv_ssim)
                depth_ssim_list_myo.append(myo_ssim)
                depth_ssim_list_lv.append(lv_ssim)
                depth_ssim_list_whole.append(ssim)
        else:
            flow = data['flow']
            flow = torch.from_numpy(flow).permute(3, 0, 1, 2).contiguous()

            img_data = np.load(moving_path)
            arr_img = img_data[0]
            arr_img = torch.from_numpy(arr_img).float()

            for d in range(arr_ed.shape[-1]):
                if 'patient029' in flow_path and d == 0:
                    continue

                registered = motion_estimation(flow=flow[:, :, :, d][None], original=arr_img[:, :, d][None, None], mode='bilinear')[0, 0].numpy()
                assert registered.min() >= 0
                assert registered.max() <= 1.0
                ssim, ssim_img = structural_similarity(arr_ed[:, :, d], registered, data_range=registered.max() - registered.min(), full=True)

                #plt.imshow(seg_ed[:, :, d], cmap='gray')
                #plt.show()
                #print(ssim.shape)

                rv_ssim = ssim_img[seg_ed[:, :, d] == 1].mean()
                myo_ssim = ssim_img[seg_ed[:, :, d] == 2].mean()
                lv_ssim = ssim_img[seg_ed[:, :, d] == 3].mean()
                
                depth_ssim_list_mean.append((rv_ssim + myo_ssim + lv_ssim) / 3)
                depth_ssim_list_rv.append(rv_ssim)
                depth_ssim_list_myo.append(myo_ssim)
                depth_ssim_list_lv.append(lv_ssim)
                depth_ssim_list_whole.append(ssim)

        depth_ssim_mean = np.stack(depth_ssim_list_mean, axis=0)
        depth_ssim_rv = np.stack(depth_ssim_list_rv, axis=0)
        depth_ssim_myo = np.stack(depth_ssim_list_myo, axis=0)
        depth_ssim_lv = np.stack(depth_ssim_list_lv, axis=0)
        depth_ssim_whole = np.stack(depth_ssim_list_whole, axis=0)
        video_ssim_mean.append(depth_ssim_mean)
        video_ssim_rv.append(depth_ssim_rv)
        video_ssim_myo.append(depth_ssim_myo)
        video_ssim_lv.append(depth_ssim_lv)
        video_ssim_whole.append(depth_ssim_whole)

        if int(os.path.basename(flow_path).split('.')[0][-2:]) == es_idx + 1:
            es_ssim_mean.append(depth_ssim_mean)
            es_ssim_rv.append(depth_ssim_rv)
            es_ssim_myo.append(depth_ssim_myo)
            es_ssim_lv.append(depth_ssim_lv)
            es_ssim_whole.append(depth_ssim_whole)

    patient_ssim_mean = np.stack(video_ssim_mean, axis=0) # T-1, D
    patient_ssim_rv = np.stack(video_ssim_rv, axis=0) # T-1, D
    patient_ssim_myo = np.stack(video_ssim_myo, axis=0) # T-1, D
    patient_ssim_lv = np.stack(video_ssim_lv, axis=0) # T-1, D
    patient_ssim_whole = np.stack(video_ssim_whole, axis=0) # T-1, D
    es_ssim_mean = np.stack(es_ssim_mean, axis=0) # 1, D
    es_ssim_rv = np.stack(es_ssim_rv, axis=0) # 1, D
    es_ssim_myo = np.stack(es_ssim_myo, axis=0) # 1, D
    es_ssim_lv = np.stack(es_ssim_lv, axis=0) # 1, D
    es_ssim_whole = np.stack(es_ssim_whole, axis=0) # 1, D

    for d in range(patient_ssim_mean.shape[-1]):
        current_res_mean = patient_ssim_mean[:, d] # T-1
        current_res_rv = patient_ssim_rv[:, d] # T-1
        current_res_myo = patient_ssim_myo[:, d] # T-1
        current_res_lv = patient_ssim_lv[:, d] # T-1
        current_res_whole = patient_ssim_whole[:, d] # T-1
        current_res_es_mean = es_ssim_mean[0, d] # T-1
        current_res_es_rv = es_ssim_rv[0, d] # T-1
        current_res_es_myo = es_ssim_myo[0, d] # T-1
        current_res_es_lv = es_ssim_lv[0, d] # T-1
        current_res_es_whole = es_ssim_whole[0, d] # T-1
        current_res_info = {'Name': current_patient_nb,
                            'slice_number': d,
                            'SSIM_mean': current_res_mean.tolist(),
                            'SSIM_rv': current_res_rv.tolist(),
                            'SSIM_myo': current_res_myo.tolist(),
                            'SSIM_lv': current_res_lv.tolist(),
                            'SSIM_whole': current_res_whole.tolist(),
                            'ES_SSIM_mean': current_res_es_mean,
                            'ES_SSIM_rv': current_res_es_rv,
                            'ES_SSIM_myo': current_res_es_myo,
                            'ES_SSIM_lv': current_res_es_lv,
                            'ES_SSIM_whole': current_res_es_whole}
        results['all'].append(current_res_info)


results_csv = []
for i in range(len(results['all'])):
    current_dict = results['all'][i].copy()
    current_dict['SSIM_mean'] = np.array(current_dict['SSIM_mean']).mean()
    current_dict['SSIM_rv'] = np.array(current_dict['SSIM_rv']).mean()
    current_dict['SSIM_myo'] = np.array(current_dict['SSIM_myo']).mean()
    current_dict['SSIM_lv'] = np.array(current_dict['SSIM_lv']).mean()
    current_dict['SSIM_whole'] = np.array(current_dict['SSIM_whole']).mean()
    current_dict['ES_SSIM_mean'] = current_dict['ES_SSIM_mean']
    current_dict['ES_SSIM_rv'] = current_dict['ES_SSIM_rv']
    current_dict['ES_SSIM_myo'] = current_dict['ES_SSIM_myo']
    current_dict['ES_SSIM_lv'] = current_dict['ES_SSIM_lv']
    current_dict['ES_SSIM_whole'] = current_dict['ES_SSIM_whole']
    results_csv.append(current_dict)

with open(os.path.join(os.path.join(path, 'ssim_metrics_all.csv')), 'w') as fd_csv:
    writer = csv.DictWriter(fd_csv, fieldnames=list(results_csv[0].keys()))
    writer.writeheader() 
    writer.writerows(results_csv)

results['mean'] = np.array([np.array(x['SSIM_mean']).mean() for x in results['all']]).mean()

save_json(results, os.path.join(path, 'ssim_metrics_all.json'))