import pickle
from glob import glob
import os
from collections import OrderedDict
from batchgenerators.utilities.file_and_folder_operations import save_pickle
import numpy as np

if __name__ == "__main__":
    splits = []

    path_list = glob(os.path.join(r'C:\Users\Portal\Documents\Isensee\nnUNet\nnunet\out\nnUNet_preprocessed\Task035_ACDC\custom_experiment_planner_stage0', '*.npz'))
    filename_list = sorted(list(set([os.path.basename(x)[:-4] for x in path_list])))
    filename_list = [x for x in filename_list if '_u' not in x]

    patient_names = np.array([x.split('_')[0] for x in filename_list])
    to_split = np.arange(len(filename_list))
    assert len(to_split) == len(patient_names)
    print(to_split)

    train_index = np.sort(to_split)
    test_index = np.sort(to_split)
    train_keys = np.array(filename_list)[train_index]
    test_keys = np.array(filename_list)[test_index]
    splits.append(OrderedDict())
    splits[-1]['train'] = train_keys
    splits[-1]['val'] = test_keys
    save_pickle(splits, os.path.join(r'C:\Users\Portal\Documents\Isensee\nnUNet\nnunet\out\nnUNet_preprocessed\Task035_ACDC', 'splits_final.pkl'))