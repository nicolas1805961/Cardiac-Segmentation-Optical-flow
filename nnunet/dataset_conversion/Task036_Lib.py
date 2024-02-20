#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from collections import OrderedDict
from batchgenerators.utilities.file_and_folder_operations import *
import shutil
import numpy as np
from sklearn.model_selection import KFold
import nibabel as nib
import sys
from tqdm import tqdm
import pickle
from glob import glob

def get_labeled_frame_nb(info_file):
    indices = []
    with open(info_file) as fd:
        lines = fd.readlines()
        indices.append(int(lines[0].split(' ')[-1]) - 1)
        indices.append(int(lines[1].split(' ')[-1]) - 1)
    return indices


def convert_to_submission(source_dir, target_dir):
    niftis = subfiles(source_dir, join=False, suffix=".nii.gz")
    patientids = np.unique([i[:10] for i in niftis])
    maybe_mkdir_p(target_dir)
    for p in patientids:
        files_of_that_patient = subfiles(source_dir, prefix=p, suffix=".nii.gz", join=False)
        assert len(files_of_that_patient)
        files_of_that_patient.sort()
        # first is ED, second is ES
        shutil.copy(join(source_dir, files_of_that_patient[0]), join(target_dir, p + "_ED.nii.gz"))
        shutil.copy(join(source_dir, files_of_that_patient[1]), join(target_dir, p + "_ES.nii.gz"))


if __name__ == "__main__":
    folder = "Lib_testing"
    folder_test = "/media/fabian/My Book/datasets/ACDC/testing/testing"
    out_folder = os.path.join('out', 'nnUNet_raw_data_base', 'nnUNet_raw_data', 'Task036_Lib')
    #out_folder = "out\\nnUNet_raw_data_base\\nnUNet_raw_data\Task027_ACDC"

    maybe_mkdir_p(join(out_folder, "imagesTr"))
    maybe_mkdir_p(join(out_folder, "imagesTs"))
    maybe_mkdir_p(join(out_folder, "labelsTr"))
    maybe_mkdir_p(join(out_folder, "strain", "LV", "radial"))
    maybe_mkdir_p(join(out_folder, "strain", "LV", "tangential"))
    maybe_mkdir_p(join(out_folder, "strain", "RV", "tangential"))
    maybe_mkdir_p(join(out_folder, "contour", "RV"))
    maybe_mkdir_p(join(out_folder, "contour", "LV"))

    # train
    all_train_files = []
    all_dict = []
    patient_dirs_train = subfolders(folder, prefix="patient")
    patient_dirs_train = sorted(patient_dirs_train)
    for p in tqdm(patient_dirs_train):
        current_dir = p
        data_files_train = [i for i in subfiles(current_dir, suffix=".nii.gz") if i.find("_gt") == -1 and i.find("_4d") == -1]
        corresponding_seg_files = [i[:-7] + "_gt.nii.gz" for i in data_files_train]
        data_files_train = sorted(data_files_train)
        corresponding_seg_files = sorted(corresponding_seg_files)
        for d, s in zip(data_files_train, corresponding_seg_files):
            patient_identifier = d.split(os.sep)[-1][:-7]
            all_train_files.append(patient_identifier + "_0000.nii.gz")
            shutil.copy(d, join(out_folder, "imagesTr", patient_identifier + "_0000.nii.gz"))
            shutil.copy(s, join(out_folder, "labelsTr", patient_identifier + ".nii.gz"))

            pickle_file = 'info_' + patient_identifier[-2:] + '.pkl'
            with open(os.path.join(current_dir, pickle_file), 'rb') as f:
                loaded_dict = pickle.load(f)
            current_dict = {'image': "./imagesTr/%s.nii.gz" % patient_identifier, "label": "./labelsTr/%s.nii.gz" % patient_identifier}
            #[x.update(current_dict) for x in loaded_dict]
            loaded_dict.update(current_dict)
            all_dict.append(loaded_dict)

        npy_paths = glob(os.path.join(current_dir, '**', '*.npy'), recursive=True)

        for npy_path in npy_paths:
            split_index = 4 if 'strain' in npy_path else 3
            start_path = [out_folder]
            start_path.extend(npy_path.split(os.sep)[-split_index:])
            shutil.copy(npy_path, os.path.join(*start_path))


    # test
    all_test_files = []
    #patient_dirs_test = subfolders(folder_test, prefix="patient")
    #for p in patient_dirs_test:
    #    current_dir = p
    #    data_files_test = [i for i in subfiles(current_dir, suffix=".nii.gz") if i.find("_gt") == -1 and i.find("_4d") == -1]
    #    for d in data_files_test:
    #        patient_identifier = d.split("/")[-1][:-7]
    #        all_test_files.append(patient_identifier + "_0000.nii.gz")
    #        shutil.copy(d, join(out_folder, "imagesTs", patient_identifier + "_0000.nii.gz"))


    json_dict = OrderedDict()
    json_dict['name'] = "Libvolumes"
    json_dict['description'] = "cardiac cine MRI segmentation"
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = ""
    json_dict['licence'] = ""
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": "noNorm",
    }
    json_dict['labels'] = {
        "0": "background",
        "1": "RV",
        "2": "MLV",
        "3": "LVC"
    }
    json_dict['numTraining'] = len(all_dict)
    json_dict['numTest'] = len(all_test_files)
    json_dict['training'] = all_dict
    json_dict['test'] = ["./imagesTs/%s.nii.gz" % i.split("/")[-1][:-12] for i in all_test_files]

    save_json(json_dict, os.path.join(out_folder, "dataset.json"))