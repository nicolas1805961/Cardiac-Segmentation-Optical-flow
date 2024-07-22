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
path = r"C:\Users\Portal\Documents\voxelmorph\iterative_warp\2024-07-02_11H29_59s_847044\Task032_Lib\fold_0\Lib\test\Raw\Backward_flow"

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

    img_path_list = glob(os.path.join('Lib_resampling_testing_mask', current_patient_nb + '*.npy'))
    flow_path_list = sorted(glob(os.path.join(path, current_patient_nb, '*.npz')))

    img_numbers = [int(os.path.basename(x).split('.')[0][-2:]) for x in img_path_list]
    flow_numbers = [int(os.path.basename(x).split('.')[0][-2:]) for x in flow_path_list]
    ed_number = [x for x in img_numbers if x not in flow_numbers][0]
    assert ed_number not in flow_numbers
    img_numbers = np.array(img_numbers)
    img_numbers = np.concatenate([img_numbers[img_numbers >= ed_number], img_numbers[img_numbers < ed_number]])

    es_idx = np.where(img_numbers - 1 == es_idx)[0][0]

    img_path_list = np.array(img_path_list)
    flow_path_list = np.array(flow_path_list)
    img_path_list = img_path_list[img_numbers - 1]
    flow_path_list = np.insert(flow_path_list, ed_number - 1, np.nan)
    flow_path_list = flow_path_list[img_numbers - 1]
    assert flow_path_list[0] == 'nan'

    flow_list = []
    img_list = []
    seg_list = []
    for flow_path, moving_path in zip(flow_path_list, img_path_list):

        img_data = np.load(moving_path)

        arr_img = img_data[0]
        img_list.append(arr_img)

        arr_seg = img_data[1]
        seg_list.append(arr_seg)

        if flow_path == 'nan':
            continue

        data = np.load(flow_path)
        flow = data['flow']
        flow = torch.from_numpy(flow).permute(3, 0, 1, 2).contiguous()
        flow_list.append(flow)

    flow = torch.stack(flow_list)
    img = np.stack(img_list)
    seg = np.stack(seg_list)

    depth_ssim_list = {'mean':[], 'lv':[], 'myo':[], 'rv':[], 'whole':[]}
    depth_es_ssim_list = {'mean':[], 'lv':[], 'myo':[], 'rv':[], 'whole':[]}
    for d in range(flow.shape[-1]):
        if 'patient029' in patient_path and d == 0:
            continue
        current_flow = flow[:, :, :, :, d]
        current_img = img[:, :, :, d]
        current_seg = seg[:, :, :, d]

        indices = torch.arange(1, len(current_img))
        chunk1 = indices[:es_idx-1]
        chunk2 = indices[es_idx-1:]
        chunk2 = torch.flip(chunk2, dims=[0])

        chunk1_0 = torch.cat([torch.tensor([0]), chunk1])
        chunk2_0 = torch.cat([torch.tensor([0]), chunk2])

        chunk_list_ssim = []

        for cn, chunk in enumerate([chunk1_0, chunk2_0]):
            current_img_chunk = current_img[chunk]
            current_seg_chunk = current_seg[chunk]
            current_flow_chunk = current_flow[chunk-1]

            video_ssim_list = {'mean':[], 'lv':[], 'myo':[], 'rv':[], 'whole':[]}
            video_es_ssim_list = {'mean':None, 'lv':None, 'myo':None, 'rv':None, 'whole':None}
            for t1 in reversed(range(1, len(current_img_chunk))):
                current_moving_img = torch.from_numpy(current_img_chunk[t1][None, None]).float()
                for t2 in reversed(range(t1)):
                    current_moving_img = motion_estimation(flow=current_flow_chunk[t2][None], original=current_moving_img, mode='bilinear')

                current_moving_img = current_moving_img[0, 0].numpy()
                assert current_moving_img.min() >= 0
                assert current_moving_img.max() <= 1.0

                ssim, ssim_img = structural_similarity(current_img[0], current_moving_img, data_range=current_moving_img.max() - current_moving_img.min(), full=True)
                #ssim = structural_similarity(current_img[0].numpy(), current_moving_img, data_range=current_moving_img.max() - current_moving_img.min())

                rv_ssim = ssim_img[current_seg[0] == 1].mean()
                myo_ssim = ssim_img[current_seg[0] == 2].mean()
                lv_ssim = ssim_img[current_seg[0] == 3].mean()
                
                video_ssim_list['rv'].append(rv_ssim)
                video_ssim_list['myo'].append(myo_ssim)
                video_ssim_list['lv'].append(lv_ssim)
                video_ssim_list['mean'].append((rv_ssim + myo_ssim + lv_ssim) / 3)
                video_ssim_list['whole'].append(ssim)

                if cn == 1 and t1 == len(current_img_chunk) - 1:

                    #fig, ax = plt.subplots(1, 2)
                    #ax[0].imshow(current_img[0], cmap='gray')
                    #ax[1].imshow(current_moving_img, cmap='gray')
                    #plt.show()

                    video_es_ssim_list['rv'] = rv_ssim
                    video_es_ssim_list['myo'] = myo_ssim
                    video_es_ssim_list['lv'] = lv_ssim
                    video_es_ssim_list['mean'] = (rv_ssim + myo_ssim + lv_ssim) / 3
                    video_es_ssim_list['whole'] = ssim
            
            for key in video_ssim_list.keys():
                structure_ssim = np.stack(video_ssim_list[key], axis=0)
                video_ssim_list[key] = np.flip(structure_ssim, axis=0)

            chunk_list_ssim.append(video_ssim_list)

        for key in video_ssim_list.keys():
            nan_tensor_1 = np.full(shape=(len(chunk1),) + chunk_list_ssim[0][key].shape[1:], fill_value=np.nan)
            nan_tensor_2 = np.full(shape=(len(chunk2),) + chunk_list_ssim[1][key].shape[1:], fill_value=np.nan)
            nan_tensor_1 = chunk_list_ssim[0][key]
            nan_tensor_2 = chunk_list_ssim[1][key]

            assert np.all(np.isfinite(nan_tensor_1))
            assert np.all(np.isfinite(nan_tensor_2))
            
            video_ssim = np.concatenate([nan_tensor_1, np.flip(nan_tensor_2, axis=0)], axis=0)

            assert len(video_ssim) == len(current_img) - 1
        
            depth_ssim_list[key].append(video_ssim)
            depth_es_ssim_list[key].append(video_es_ssim_list[key])

    for key in depth_ssim_list.keys():
        depth_ssim_list[key] = np.stack(depth_ssim_list[key], axis=-1) # T-1, D
        depth_es_ssim_list[key] = np.stack(depth_es_ssim_list[key], axis=-1).reshape(1, -1) # 1, D
        assert len(depth_ssim_list[key]) == len(img) - 1

    for d in range(depth_ssim_list['mean'].shape[-1]):
        current_res_info = {'Name': current_patient_nb,
                            'slice_number': d}
        for key in depth_ssim_list.keys():
            current_res = depth_ssim_list[key][:, d] # T-1
            current_res_es = depth_es_ssim_list[key][0, d] # T-1
            current_res_info['SSIM_' + key] = current_res.tolist()
            current_res_info['ES_SSIM_' + key] = current_res_es
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