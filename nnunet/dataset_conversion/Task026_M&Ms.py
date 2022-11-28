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
import nibabel as nib
import sys
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

def get_labeled_frame_nb(df, external_code):
    ed = int(df['ED'][df['External code'] == external_code])
    es = int(df['ES'][df['External code'] == external_code])
    if ed == es:
        return []
    else:
        return [ed, es]

def switch_label(label):
    assert len(label.shape) == 3
    out = []
    for i in range(label.shape[-1]):
        out_img = np.zeros_like(label[:, :, i])
        out_img[label[:, :, i] == 3] = 1
        out_img[label[:, :, i] == 2] = 2
        out_img[label[:, :, i] == 1] = 3

        #fig, ax = plt.subplots(1, 2)
        #ax[0].imshow(label[:, :, i], cmap='gray')
        #ax[1].imshow(out_img, cmap='gray')
        #plt.show()

        out.append(out_img)
    out = np.stack(out, axis=-1)
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
    labeled_train_folder = "OpenDataset/Training/Labeled"
    unlabeled_train_folder = "OpenDataset/Training/Unlabeled"
    val_folder = "OpenDataset/Validation"
    test_folder = "OpenDataset/Testing"
    out_folder = os.path.join('out', 'nnUNet_raw_data_base', 'nnUNet_raw_data', 'Task026_MMs')
    #out_folder = "out\\nnUNet_raw_data_base\\nnUNet_raw_data\Task027_ACDC"

    maybe_mkdir_p(join(out_folder, "imagesTr"))
    maybe_mkdir_p(join(out_folder, "imagesTs"))
    maybe_mkdir_p(join(out_folder, "labelsTr"))

    # train
    all_train_files = []
    all_unlabeled_train_files = []

    patient_dirs_labeled_train = subfolders(labeled_train_folder)
    patient_dirs_unlabeled_train = subfolders(unlabeled_train_folder)
    patient_dirs_val = subfolders(val_folder)
    patient_dirs_test = subfolders(test_folder)
    patient_dirs_train = patient_dirs_labeled_train + patient_dirs_unlabeled_train + patient_dirs_test + patient_dirs_val
    df = pd.read_csv(os.path.join('OpenDataset', '211230_M&Ms_Dataset_information_diagnosis_opendataset.csv'))

    for p in tqdm(patient_dirs_train):
        current_dir = p
        data_files_train = [i for i in subfiles(current_dir, suffix=".nii.gz") if i.find("_gt") == -1]
        corresponding_seg_files = [i[:-7] + "_gt.nii.gz" for i in data_files_train]
        for d, s in zip(data_files_train, corresponding_seg_files):
            patient_identifier = d.split(os.sep)[-1][:-7].split('_')[0]
            labeled_frame_indices = get_labeled_frame_nb(df, patient_identifier)
            in_nib_img = nib.load(d)
            in_nib_img_gt = nib.load(s)
            original_header = in_nib_img.header
            original_affine = in_nib_img.affine
            img = in_nib_img.get_fdata()
            label = in_nib_img_gt.get_fdata().astype(int)
            for i in range(img.shape[-1]):
                if i in labeled_frame_indices:
                    label[:, :, :, i] = switch_label(label[:, :, :, i])
                    file_name = patient_identifier + '_frame' + str(i + 1).zfill(2)
                    all_train_files.append(file_name + "_0000.nii.gz")
                    out_nib_img = nib.Nifti1Image(img[:, :, :, i], original_affine, original_header)
                    out_nib_img_gt = nib.Nifti1Image(label[:, :, :, i], original_affine, original_header)
                    out_path = join(out_folder, "imagesTr", file_name + "_0000.nii.gz")
                    out_path_gt = join(out_folder, "labelsTr", file_name + ".nii.gz")
                    assert ~np.all(np.array(out_nib_img.header['pixdim']) == 1.0)
                    nib.save(out_nib_img, out_path)
                    nib.save(out_nib_img_gt, out_path_gt)
                else:
                    file_name = patient_identifier + '_frame' + str(i + 1).zfill(2) + "_u_0000.nii.gz"
                    all_unlabeled_train_files.append(file_name)
                    out_path = join(out_folder, "imagesTr", file_name)
                    out_nib_img = nib.Nifti1Image(img[:, :, :, i], original_affine, original_header)
                    assert ~np.all(np.array(out_nib_img.header['pixdim']) == 1.0)
                    nib.save(out_nib_img, out_path)

    # test
    all_test_files = []
    #for p in patient_dirs_test:
    #    current_dir = p
    #    data_files_test = [i for i in subfiles(current_dir, suffix=".nii.gz") if i.find("_gt") == -1 and i.find("_4d") == -1]
    #    for d in data_files_test:
    #        patient_identifier = d.split("/")[-1][:-7]
    #        all_test_files.append(patient_identifier + "_0000.nii.gz")
    #        shutil.copy(d, join(out_folder, "imagesTs", patient_identifier + "_0000.nii.gz"))


    json_dict = OrderedDict()
    json_dict['name'] = "M&Ms"
    json_dict['description'] = "cardias cine MRI segmentation"
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = "see M&Ms challenge"
    json_dict['licence'] = "see M&Ms challenge"
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
    json_dict['numUnlabeled'] = len(all_unlabeled_train_files)
    json_dict['numTest'] = len(all_test_files)
    json_dict['unlabeled'] = [{'image': "./imagesTr/%s.nii.gz" % i.split("/")[-1][:-12]} for i in
                             all_unlabeled_train_files]
    json_dict['training'] = [{'image': "./imagesTr/%s.nii.gz" % i.split("/")[-1][:-12], "label": "./labelsTr/%s.nii.gz" % i.split("/")[-1][:-12]} for i in
                             all_train_files]
    json_dict['test'] = ["./imagesTs/%s.nii.gz" % i.split("/")[-1][:-12] for i in all_test_files]

    save_json(json_dict, os.path.join(out_folder, "dataset.json"))

    # create a dummy split (patients need to be separated)
    splits = []
    splits.append(OrderedDict())
    splits[-1]['train'] = []
    splits[-1]['val'] = []
    #patients = np.unique([i[:10] for i in all_train_files])
    patientids = [i[:-12] for i in all_train_files]
    print(patientids)
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

    test_id = [x.split(os.sep)[-1] for x in patient_dirs_test]
    print(test_id)
    
    for p in patientids:
        if any(p.split('_')[0] in x for x in test_id):
            splits[-1]['val'].append(p)
        else:
            splits[-1]['train'].append(p)


    #kf = KFold(n_splits=5, shuffle=True, random_state=12345)
    #for tr, val in kf.split(patients):
    #    splits.append(OrderedDict())
    #    tr_patients = patients[tr]
    #    splits[-1]['train'] = [i[:-12] for i in all_train_files if i[:10] in tr_patients]
    #    val_patients = patients[val]
    #    splits[-1]['val'] = [i[:-12] for i in all_train_files if i[:10] in val_patients]

    
    save_pickle(splits, os.path.join('out', 'nnUNet_raw_data_base', 'nnUNet_raw_data', 'Task026_MMs', 'splits_final.pkl'))