import numpy as np
import os
from glob import glob
from multiprocessing import Pool
import json

def save_json(obj, file: str, indent: int = 4, sort_keys: bool = True) -> None:
    with open(file, 'w') as f:
        json.dump(obj, f, sort_keys=sort_keys, indent=indent)


def strain_compute_metric(patient_name, all_files, gt_folder_name):
    results = {'all': []}
    phase_list = [x for x in all_files if patient_name in x]
    phase_list = np.array(sorted(phase_list))
    video = []
    for phase in phase_list:
        arr = np.load(phase) # T
        video.append(arr)
    video = np.stack(video, axis=0) # D, T
    for d in range(len(video)):
        slice_nb = str(d + 1).zfill(2)
        filename = patient_name + '_slice' + slice_nb + '.npy'
        gt_path = os.path.join(gt_folder_name, 'strain', 'LV', 'tangential', filename)
        lv_tangential_strain = video[d]
        lv_tangential_strain_gt = np.load(gt_path)[0]
        current_res = {'reference': gt_path, 'test': phase_list[0][:-12]}
        current_res['lv_tangential'] = np.abs(lv_tangential_strain - lv_tangential_strain_gt).tolist()
        results['all'].append(current_res)
    return results



if __name__ == "__main__":

    default_num_threads = 8

    pred_folder= "2023-09-25_22H16"
    gt_folder_name = r"C:\Users\Portal\Documents\Isensee\nnUNet\nnunet\out\nnUNet_preprocessed\Task032_Lib"

    pred_file_folder = os.path.join(pred_folder, r"Validation\Task032_Lib\fold_0\Registered\temp_allClasses\Strain\LV\Tangential")
    
    all_files = glob(os.path.join(pred_file_folder, '*.npy'))
    patient_names = list(set([os.path.basename(x).split('_')[0] for x in all_files]))

    p = Pool(default_num_threads)
    results = {"all": [], "mean_lv_tangential": None}

    all_res = p.starmap(strain_compute_metric, zip(patient_names, [all_files]*len(patient_names), [gt_folder_name]*len(patient_names)))
    p.close()
    p.join()

    for i in range(len(all_res)):
        results['all'].extend(all_res[i]['all'])

    results['mean_lv_tangential'] = np.concatenate([np.array(x['lv_tangential']) for x in results['all']]).mean()
    save_json(results, os.path.join(pred_file_folder, 'strain_summary.json'))