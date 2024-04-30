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
from copy import copy

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
    folder = "ACDC_training"
    folder_test = "/media/fabian/My Book/datasets/ACDC/testing/testing"
    out_folder = os.path.join('out', 'nnUNet_raw_data_base', 'nnUNet_raw_data', 'Task039_ACDC')
    #out_folder = "out\\nnUNet_raw_data_base\\nnUNet_raw_data\Task027_ACDC"

    maybe_mkdir_p(join(out_folder, "imagesTr"))
    maybe_mkdir_p(join(out_folder, "imagesTs"))
    maybe_mkdir_p(join(out_folder, "labelsTr"))

    # train
    all_dict = []
    all_dict_unlabeled = []
    patient_dirs_train = subfolders(folder, prefix="patient")
    for p in tqdm(patient_dirs_train):
        current_dir = p

        with open(os.path.join(current_dir, 'Info.cfg')) as f:
            lines = f.read().splitlines() 
            patient_loaded_dict = {lines[2][:5]: lines[2][7:], lines[0][:2]: int(lines[0][4:]), lines[1][:2]: int(lines[1][4:])}

        data_files_train = [i for i in subfiles(current_dir, suffix=".nii.gz") if i.find("_gt") == -1 and i.find("_4d") == -1]
        corresponding_seg_files = [i[:-7] + "_gt.nii.gz" for i in data_files_train]
        assert len(data_files_train) == 2
        for d, s in zip(data_files_train, corresponding_seg_files):
            loaded_dict = copy(patient_loaded_dict)
            patient_identifier = d.split(os.sep)[-1][:-7]
            shutil.copy(d, join(out_folder, "imagesTr", patient_identifier + "_0000.nii.gz"))
            shutil.copy(s, join(out_folder, "labelsTr", patient_identifier + ".nii.gz"))

            current_dict = {'image': "./imagesTr/%s.nii.gz" % patient_identifier, "label": "./labelsTr/%s.nii.gz" % patient_identifier}
            loaded_dict.update(current_dict)
            all_dict.append(loaded_dict)

        info_file = [i for i in subfiles(current_dir, suffix=".cfg")][0]
        labeled_frame_indices = get_labeled_frame_nb(info_file)
        data_file_train_unlabeled = [i for i in subfiles(current_dir, suffix=".nii.gz") if i.find("_4d") >= 0][0]
        in_nib_img = nib.load(data_file_train_unlabeled)
        original_header = in_nib_img.header
        original_affine = in_nib_img.affine
        img = in_nib_img.get_fdata()
        for i in range(img.shape[-1]):
            if i in labeled_frame_indices:
                continue
            else:
                loaded_dict = copy(patient_loaded_dict)
                current_dict = {'image': "./imagesTr/%s.nii.gz" % patient_identifier}
                loaded_dict.update(current_dict)
                all_dict_unlabeled.append(loaded_dict)

                patient_identifier = data_file_train_unlabeled.split(os.sep)[-1].split('4d')[0] + 'frame' + str(i + 1).zfill(2) + '_u'
                out_path = join(out_folder, "imagesTr", patient_identifier + "_0000.nii.gz")
                out_nib_img = nib.Nifti1Image(img[:, :, :, i], original_affine, original_header)
                assert ~np.all(np.array(out_nib_img.header['pixdim']) == 1.0), print(p)
                nib.save(out_nib_img, out_path)


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
    json_dict['name'] = "ACDCNoNorm"
    json_dict['description'] = "cardiac cine MRI segmentation"
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = "see ACDC challenge"
    json_dict['licence'] = "see ACDC challenge"
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
    json_dict['numUnlabeled'] = len(all_dict_unlabeled)
    json_dict['numTest'] = len(all_test_files)
    json_dict['unlabeled'] = all_dict_unlabeled
    json_dict['training'] = all_dict
    json_dict['test'] = ["./imagesTs/%s.nii.gz" % i.split("/")[-1][:-12] for i in all_test_files]

    save_json(json_dict, os.path.join(out_folder, "dataset.json"))