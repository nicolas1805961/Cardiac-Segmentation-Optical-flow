import nibabel as nib
from evaluation.metrics import hausdorff_distance, dice, avg_surface_distance_symmetric
from glob import glob
import os
from tqdm import tqdm
import numpy as np
import pickle
import csv
from batchgenerators.utilities.file_and_folder_operations import save_json
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='Dataset name')
    parser.add_argument('--test_or_val', required=True, help='Whether this is testing or validation set')
    args = parser.parse_args()

    #registered folder path here
    pred_directory = r"C:\Users\Portal\Documents\voxelmorph\2023-12-15_19H01\temp_sum\Lib\test\Postprocessed\Registered"

    registered_patients = [name for name in os.listdir(pred_directory) if os.path.isdir(os.path.join(pred_directory, name))]

    if args.dataset == 'Lib':
        if args.test_or_val == 'val':
            gt_directory = r'out\nnUNet_preprocessed\Task032_Lib'
        elif args.test_or_val == 'test':
            gt_directory = r'out\nnUNet_preprocessed\Task036_Lib'
    elif args.dataset == 'ACDC':
        if args.test_or_val == 'val':
            gt_directory = r'out\nnUNet_preprocessed\Task031_ACDC'
        elif args.test_or_val == 'test':
            gt_directory = r'out\nnUNet_preprocessed\Task035_ACDC'

    with open(os.path.join(gt_directory, 'splits_final.pkl'), 'rb') as f:
        data = pickle.load(f)
        validation_patients = data[0]['val']

    all_results = {'all': {}, 'mean': {'RV': {'Dice': None, 'HD': None, 'ASSD': None},
                                    'MYO': {'Dice': None, 'HD': None, 'ASSD': None},
                                    'LV': {'Dice': None, 'HD': None, 'ASSD': None}},
                            'std': {'RV': {'Dice': None, 'HD': None, 'ASSD': None},
                                    'MYO': {'Dice': None, 'HD': None, 'ASSD': None},
                                    'LV': {'Dice': None, 'HD': None, 'ASSD': None}}}
    
    results_list = []
    for patient_name in tqdm(registered_patients):

        path_list = glob(os.path.join(pred_directory, patient_name, 'temp_allClasses', '*.gz'))
        path_list = [x for x in path_list if os.path.basename(x)[:-7] in validation_patients]

        corresponding_pkl_file = os.path.join(gt_directory, 'custom_experiment_planner_stage0', patient_name + '_frame01.pkl')
        with open(corresponding_pkl_file, 'rb') as f:
            data = pickle.load(f)
            ed_number = np.rint(data['ed_number']).astype(int)

        patient_results = []

        for path in path_list:
            filename = os.path.basename(path)
            filename = filename.split('.')[0][:-2] + str(ed_number + 1).zfill(2) + '.nii.gz'
            corresponding_gt_file = os.path.join(gt_directory, 'gt_segmentations', filename)

            if not os.path.isfile(corresponding_gt_file):
                continue

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
            
            patient_results.append(class_results)

            results_list.append({'Name': os.path.basename(path).split('.')[0],
                                'Average_Dice': (class_results['RV']['Dice'] + class_results['MYO']['Dice'] + class_results['LV']['Dice']) / 3,
                                'Average_HD': (class_results['RV']['HD'] + class_results['MYO']['HD'] + class_results['LV']['HD']) / 3,
                                'Average_ASSD': (class_results['RV']['ASSD'] + class_results['MYO']['ASSD'] + class_results['LV']['ASSD']) / 3,
                                'RV_Dice': class_results['RV']['Dice'], 
                                'RV_HD': class_results['RV']['HD'], 
                                'RV_ASSD': class_results['RV']['ASSD'], 
                                'MYO_Dice': class_results['MYO']['Dice'], 
                                'MYO_HD': class_results['MYO']['HD'],
                                'MYO_ASSD': class_results['MYO']['ASSD'],
                                'LV_Dice': class_results['LV']['Dice'],
                                'LV_HD': class_results['LV']['HD'],
                                'LV_ASSD': class_results['LV']['ASSD'],
                                })
        
        all_results['all'][patient_name] = patient_results
    
    individual_results = []
    for p in all_results['all'].keys():
        for t in range(len(all_results['all'][p])):
            individual_results.append(all_results['all'][p][t])

    print(len(individual_results))

    for k1 in all_results['mean'].keys():
        for k2 in all_results['mean'][k1].keys():
            all_results['mean'][k1][k2] = np.nanmean(np.array([x[k1][k2] for x in individual_results]))
            all_results['std'][k1][k2] = np.std(np.array([x[k1][k2] for x in individual_results]))

    save_json(all_results, os.path.join(pred_directory, 'segmentation_metrics.json'))
    
    with open(os.path.join(pred_directory, 'segmentation_metrics.csv'), 'w') as fd_csv:
        writer = csv.DictWriter(fd_csv, fieldnames=list(results_list[0].keys()))
        writer.writeheader() 
        writer.writerows(results_list) 
