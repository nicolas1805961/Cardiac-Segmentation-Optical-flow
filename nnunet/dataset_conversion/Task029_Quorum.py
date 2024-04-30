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
import pandas as pd
from tqdm import tqdm

def split_factor(all_train_files, sorter_factor_list):
    unique_subfactors = set(sorter_factor_list)
    out = {}
    for subfactor in unique_subfactors:
        out[subfactor] = []
    for name, factor_value in zip(all_train_files, sorter_factor_list):
        out[factor_value].append(name)
    return out


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
    folder = "Quorum_training"
    folder_test = "/media/fabian/My Book/datasets/ACDC/testing/testing"
    out_folder = os.path.join('out', 'nnUNet_raw_data_base', 'nnUNet_raw_data', 'Task040_Quorum_training')
    #out_folder = "out\\nnUNet_raw_data_base\\nnUNet_raw_data\Task027_ACDC"

    maybe_mkdir_p(join(out_folder, "imagesTr"))
    maybe_mkdir_p(join(out_folder, "imagesTs"))
    maybe_mkdir_p(join(out_folder, "labelsTr"))

    # train
    all_train_files = []
    all_phases = []
    all_strengths = []
    all_manufacturers = []
    all_centers = []
    all_spacing_between_slices = []
    all_slice_thickness = []
    patient_dirs_train = subfolders(folder, prefix="patient")
    for p in tqdm(patient_dirs_train):
        current_dir = p
        ed_info_path = os.path.join(current_dir, 'ed_info.csv')
        es_info_path = os.path.join(current_dir, 'es_info.csv')
        ed_df = pd.read_csv(ed_info_path)
        es_df = pd.read_csv(es_info_path)
        df_list = [ed_df, es_df]
        data_files_train = [i for i in subfiles(current_dir, suffix=".nii.gz") if i.find("_gt") == -1 and i.find("_4d") == -1]
        corresponding_seg_files = [i[:-7] + "_gt.nii.gz" for i in data_files_train]
        assert len(data_files_train) == len(corresponding_seg_files) == len(df_list)
        for d, s, df in zip(data_files_train, corresponding_seg_files, df_list):
            all_phases.append(df['Phase'].iloc[0])
            all_strengths.append(df['Field Strength'].iloc[0])
            all_manufacturers.append(df['Manufacturer'].iloc[0])
            all_centers.append(df['Name'].iloc[0])
            all_spacing_between_slices.append(df['Space Between Slices'].iloc[0])
            all_slice_thickness.append(df['Slice Thickness'].iloc[0])
            patient_identifier = d.split(os.sep)[-1][:-7]
            all_train_files.append(patient_identifier + "_0000.nii.gz")
            shutil.copy(d, join(out_folder, "imagesTr", patient_identifier + "_0000.nii.gz"))
            shutil.copy(s, join(out_folder, "labelsTr", patient_identifier + ".nii.gz"))
    
    assert len(all_train_files) == len(all_strengths) == len(all_manufacturers) == len(all_phases) == len(all_centers) == len(all_slice_thickness) == len(all_spacing_between_slices)

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
    json_dict['name'] = "Quorum"
    json_dict['description'] = "cardiac cine MRI segmentation"
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = "see Quorum"
    json_dict['licence'] = "see Quorum"
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": "MRI",
    }
    json_dict['labels'] = {
        "0": "background",
        "1": "RV",
        "2": "MLV",
        "3": "LVC"
    }
    json_dict['numTraining'] = len(all_train_files)
    json_dict['numTest'] = len(all_test_files)
    json_dict['training'] = [{'image': "./imagesTr/%s.nii.gz" % train_file.split("/")[-1][:-12],
                             "label": "./labelsTr/%s.nii.gz" % train_file.split("/")[-1][:-12],
                             "phase": "%s" % phase, 
                             "strength": "%s" % strength,
                             "manufacturer": "%s" % manufacturer,
                             "center": "%s" % center,
                             "spacing between slices": "%s" % sbs,
                             "slice thickness": "%s" % thickness} for (train_file, phase, strength, manufacturer, center, sbs, thickness) in
                             zip(all_train_files, all_phases, all_strengths, all_manufacturers, all_centers, all_spacing_between_slices, all_slice_thickness)]
    json_dict['test'] = ["./imagesTs/%s.nii.gz" % i.split("/")[-1][:-12] for i in all_test_files]

    save_json(json_dict, os.path.join(out_folder, "dataset.json"))

    #df = pd.DataFrame(data={'strength': all_strengths, 'manufacturer': all_manufacturers, 'center': all_centers})
    #df.to_excel('criterias.xlsx')

    #all_manufacturer_strength = [str(x) + '_' + str(y) for (x, y) in zip(all_manufacturers, all_strengths)]
    factor = all_manufacturers
    print("Creating new 5-fold cross-validation split...")
    unique_subfactors = set(factor)
    factor_dict = split_factor(all_train_files, factor)
    payload_dict = {'train': {x:[] for x in unique_subfactors}, 'val': {x:[] for x in unique_subfactors}}
    for k in factor_dict.keys():
        kf = KFold(n_splits=5, shuffle=True, random_state=12345)
        patients = np.unique([i[:10] for i in factor_dict[k]])
        for tr, val in kf.split(patients):
            tr_patients = patients[tr]
            payload_dict['train'][k].append(np.array([i for i in factor_dict[k] if i[:10] in tr_patients]))
            val_patients = patients[val]
            payload_dict['val'][k].append(np.array([i for i in factor_dict[k] if i[:10] in val_patients]))
    splits = [OrderedDict() for _ in range(5)]
    for j in range(5):
        current_folder_data_train = payload_dict['train']
        current_folder_data_val = payload_dict['val']
        train_list = []
        val_list = []
        for k in current_folder_data_train.keys():
            train_list.append(current_folder_data_train[k][j])
            val_list.append(current_folder_data_val[k][j])
        splits[j]['train'] = np.concatenate(train_list)
        splits[j]['val'] = np.concatenate(val_list)

    save_pickle(splits, os.path.join('out', 'nnUNet_raw_data_base', 'nnUNet_raw_data', 'Task040_Quorum_training', 'splits_final.pkl'))