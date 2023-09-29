import numpy as np
import os
from glob import glob
from multiprocessing import Pool
import json
from tqdm import tqdm

def save_json(obj, file: str, indent: int = 4, sort_keys: bool = True) -> None:
    with open(file, 'w') as f:
        json.dump(obj, f, sort_keys=sort_keys, indent=indent)


def compute_contour_metric(patient_name, all_files, gt_folder_name):
    results = {'all': []}
    phase_list = [x for x in all_files if patient_name in x]
    phase_list = np.array(sorted(phase_list, key=lambda x: int(os.path.basename(x).split('frame')[-1][:2])))
    video = []
    for phase in phase_list:
        data = np.load(phase)
        arr = data['flow'] # H, W, D, C
        arr = arr.transpose((2, 3, 0, 1)) # D, C, H, W
        video.append(arr)
    flow = np.stack(video, axis=1) # D, T, C, H, W
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

        current_slice_flow = np.flip(current_slice_flow, axis=0)
        contours = np.flip(contours, axis=0)

        temporal_error_list = []
        for t in range(len(current_slice_flow) - 1):
            current_contours = contours[t] # P, 2
            next_contours = contours[t + 1] # P, 2
            gt_delta = next_contours - current_contours
            current_frame_flow = current_slice_flow[t] # 2, H, W
            current_frame_flow = current_frame_flow.transpose((1, 2, 0)) # H, W, 2
            y = np.rint(current_contours[:, 0]).astype(int)
            x = np.rint(current_contours[:, 1]).astype(int)

            delta_pred = current_frame_flow[y, x, :] # P, 2
            error = np.abs(gt_delta - delta_pred).mean(-1) # P,
            temporal_error_list.append(error) 

        temporal_error_list = np.stack(temporal_error_list, axis=0) # T, P
        temporal_error_list = np.flip(temporal_error_list, axis=0)

        error = np.split(temporal_error_list, indices_or_sections=split_index, axis=1) # [(T, P1) , (T, P2), (T, P3)]
        error = np.stack([x.mean(-1) for x in error], axis=-1) # T, 3

        slice_error_list.append(error)
    slice_error_list = np.stack(slice_error_list, axis=0) # D, T, 3

    for i in range(slice_error_list.shape[1]):
        current_res = slice_error_list[:, i] # D, 3
        current_res_info = {'patient_name': patient_name, 
                            'reference': gt_path_lv[:-12], 
                            'test': phase_list[i + 1],
                            'ENDO_mae': current_res[:, 0].tolist(),
                            'EPI_mae': current_res[:, 1].tolist(),
                            'RV_mae': current_res[:, 2].tolist()}
        results['all'].append(current_res_info)
    return results



if __name__ == "__main__":

    default_num_threads = 1

    pred_folder= "2023-09-25_22H16"
    gt_folder_name = r"C:\Users\Portal\Documents\Isensee\nnUNet\nnunet\out\nnUNet_preprocessed\Task032_Lib"

    pred_file_folder = os.path.join(pred_folder, r"Validation\Task032_Lib\fold_0\Flow\validation_raw")
    
    all_files = glob(os.path.join(pred_file_folder, '*.npz'))
    patient_names = list(set([os.path.basename(x).split('_')[0] for x in all_files]))

    results = {"all": [], "mean": None}

    for patient_name in tqdm(patient_names[:1]):
        compute_contour_metric(patient_name, all_files, gt_folder_name)

    #p = Pool(default_num_threads)
    #all_res = p.starmap(compute_contour_metric, zip(patient_names, [all_files]*len(patient_names), [gt_folder_name]*len(patient_names)))
    #p.close()
    #p.join()

    for i in range(len(all_res)):
        results['all'].extend(all_res[i]['all'])

    current_res_info = {'ENDO_mae': np.concatenate([np.array(x['ENDO_mae']) for x in results['all']]).mean(),
                        'EPI_mae': np.concatenate([np.array(x['EPI_mae']) for x in results['all']]).mean(),
                        'RV_mae': np.concatenate([np.array(x['RV_mae']) for x in results['all']]).mean()}
    results['mean'] = current_res_info
    save_json(results, os.path.join(pred_file_folder, 'summary.json'))