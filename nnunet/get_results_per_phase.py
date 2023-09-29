import nibabel as nib
from evaluation.metrics import hausdorff_distance, dice, avg_surface_distance_symmetric
from glob import glob
import os
from tqdm import tqdm
import numpy as np
import pickle
from batchgenerators.utilities.file_and_folder_operations import save_json
import json
import matplotlib.pyplot as plt

if __name__ == "__main__":

    pred_directory = r'2023-09-25_22H16\fold_0\Registered\temp_allClasses'
    pkl_directory = r'out\nnUNet_preprocessed\Task032_Lib\custom_experiment_planner_stage0'

    pkl_paths = glob(os.path.join(pkl_directory, '*.pkl'))
    patient_names = list(set([os.path.basename(x).split('_')[0] for x in pkl_paths]))

    with open(os.path.join(pred_directory, 'metrics.json'), 'r') as f:
        data = json.load(f)['all'] # list of dict
        out_data_list = []
        for patient_name in tqdm(patient_names):
            sequence_data = {'LV': [], 'MYO': [], 'RV': [], 'frame_number': []}

            with open(os.path.join(pkl_directory, patient_name + '_frame01.pkl'), 'rb') as fd:
                info = pickle.load(fd)
                sequence_data['es_number'] = info['es_number']
                sequence_data['ed_number'] = info['ed_number']

            patient_dice_list_lv = []
            frame_list = []
            for volume_metrics in data:
                current_splitted_basename = os.path.basename(volume_metrics['reference']).split('_')
                if current_splitted_basename[0] == patient_name:
                    sequence_data['frame_number'].append(int(current_splitted_basename[1][5:7]))
                    sequence_data['LV'].append(volume_metrics['LV']['Dice'])
            
            for k in sequence_data.keys():
                sequence_data[k] = np.array(sequence_data[k])
            
            arr_0 = np.abs(sequence_data['frame_number'] - sequence_data['ed_number']).reshape(1, -1)
            arr_1 = np.abs((sequence_data['frame_number'] + len(sequence_data['frame_number'])) - sequence_data['ed_number']).reshape(1, -1)
            arr_2 = np.abs((sequence_data['frame_number'] - len(sequence_data['frame_number'])) - sequence_data['ed_number']).reshape(1, -1)
            concatenated = np.concatenate([arr_0, arr_1, arr_2], axis=0)
            sequence_data['distance'] = concatenated.min(0)
            out_data_list.append(sequence_data)

        dice_list = []
        distance_list = []
        for data in out_data_list:
            dice_list.append(data['LV'].reshape(1, -1))
            distance_list.append(data['distance'].reshape(1, -1))
        Y = np.concatenate(dice_list, axis=-1)
        X = np.concatenate(distance_list, axis=-1)

        unique_distances = np.unique(X)
        new_X = []
        new_Y = []
        for unique_distance in unique_distances:
            new_X.append(unique_distance)
            mask = X == unique_distance
            masked_average_y = Y[mask].mean()
            new_Y.append(masked_average_y)
        
        new_X = np.array(new_X)
        new_Y = np.array(new_Y)
        
        m, b = np.polyfit(new_X, new_Y, 1)

        fig, ax = plt.subplots(1, 1)
        ax.scatter(new_X, new_Y)
        ax.plot(new_X, m * new_X + b, c='r')
        plt.show()
