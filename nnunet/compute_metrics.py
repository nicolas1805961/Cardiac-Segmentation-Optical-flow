import nibabel as nib
from evaluation.metrics import hausdorff_distance, dice, avg_surface_distance_symmetric
from glob import glob
import os
from tqdm import tqdm
import numpy as np
import pickle
from batchgenerators.utilities.file_and_folder_operations import save_json

if __name__ == "__main__":

    #pred_directory = r'2023-09-20_16H30\Validation\Task031_ACDC\fold_0\Registered\temp_allClasses'
    #gt_directory = r'out\nnUNet_preprocessed\Task031_ACDC'

    pred_directory = r'2023-09-25_22H16\fold_0\Registered\temp_allClasses'
    gt_directory = r'out\nnUNet_preprocessed\Task032_Lib'

    with open(os.path.join(gt_directory, 'splits_final.pkl'), 'rb') as f:
        data = pickle.load(f)
        validation_patients = data[0]['val']


    path_list = glob(os.path.join(pred_directory, '*.gz'))
    path_list = [x for x in path_list if os.path.basename(x)[:-7] in validation_patients]
    path_list = sorted([x for x in path_list if 'frame01' not in x])


    all_results = {'all': [], 'mean': {'RV': {'Dice': None, 'HD': None, 'ASSD': None},
                                    'MYO': {'Dice': None, 'HD': None, 'ASSD': None},
                                    'LV': {'Dice': None, 'HD': None, 'ASSD': None}}}

    for path in tqdm(path_list):
        filename = os.path.basename(path)
        corresponding_gt_file = os.path.join(gt_directory, 'gt_segmentations', filename)

        data = nib.load(path)
        arr = data.get_fdata()

        data_gt = nib.load(corresponding_gt_file)
        arr_gt = data_gt.get_fdata()

        zoom = data_gt.header.get_zooms()

        class_results = {'RV': {'Dice': None, 'HD': None, 'ASSD': None},
                        'MYO': {'Dice': None, 'HD': None, 'ASSD': None},
                        'LV': {'Dice': None, 'HD': None, 'ASSD': None},
                        'test': path,
                        'reference': corresponding_gt_file}

        dice_class_results = []
        hd_class_results = []
        assd_class_results = []
        for c, class_name in enumerate(['RV', 'MYO', 'LV'], 1):
            class_pred = arr == c
            class_gt = arr_gt == c

            dice_results = dice(class_pred, class_gt)
            hd_results = hausdorff_distance(class_pred, class_gt, voxel_spacing=zoom)
            assd_results = avg_surface_distance_symmetric(class_pred, class_gt, voxel_spacing=zoom)

            class_results[class_name]['Dice'] = dice_results
            class_results[class_name]['HD'] = hd_results
            class_results[class_name]['ASSD'] = assd_results
        
        all_results['all'].append(class_results)

    for k1 in all_results['mean'].keys():
        for k2 in all_results['mean'][k1].keys():
            all_results['mean'][k1][k2] = np.array([x[k1][k2] for x in all_results['all']]).mean()

    #save_json(all_results, os.path.join(os.path.dirname(path_list[0]), 'metrics.json'))
    save_json(all_results, os.path.join('metrics.json'))
