import numpy as np
import os
from skimage.measure import perimeter
from pathlib import Path
from glob import glob
from tqdm import tqdm
import nibabel as nib

def get_strain(target):
    rv_perim_list = []
    endo_perim_list = []
    epi_perim_list = []
    for t in range(len(target)):
        current_arr = target[t]
        binarized_rv = current_arr == 1
        binarized_endo = current_arr == 3
        binarized_epi = np.logical_or(current_arr == 2, binarized_endo)
        perim_rv = perimeter(binarized_rv)
        perim_endo = perimeter(binarized_endo)
        perim_epi = perimeter(binarized_epi)
        rv_perim_list.append(perim_rv)
        endo_perim_list.append(perim_endo)
        epi_perim_list.append(perim_epi)
    
    rv_strain = [(rv_perim_list[i] - rv_perim_list[0]) / (rv_perim_list[0] + 1e-8) for i in range(len(rv_perim_list))]
    endo_strain = [(endo_perim_list[i] - endo_perim_list[0]) / (endo_perim_list[0] + 1e-8) for i in range(len(endo_perim_list))]
    epi_strain = [(epi_perim_list[i] - epi_perim_list[0]) / (epi_perim_list[0] + 1e-8) for i in range(len(epi_perim_list))]

    rv_strain = np.array(rv_strain)
    endo_strain = np.array(endo_strain)
    epi_strain = np.array(epi_strain)

    lv_strain = (endo_strain + epi_strain) / 2

    return rv_strain * 100, lv_strain * 100



def save_strain(data, path, patient_name):
    #data: D, T, H, W
    for d in range(len(data)):
        slice_nb = str(d + 1).zfill(2)
        filename = patient_name + '_slice' + slice_nb + '.npy'
        rv_tangential_strain, lv_tangential_strain = get_strain(data[d])
        np.save(os.path.join(path, filename), lv_tangential_strain)


def delete_if_exist(folder_name):
    dirpath = Path(folder_name)
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(dirpath)




if __name__ == "__main__":

    pred_folder= "2023-09-25_22H16"
    pred_file_folder = os.path.join(pred_folder, r"Validation\Task032_Lib\fold_0\Registered\temp_allClasses")
    output_path = os.path.join(pred_file_folder, 'Strain', 'LV', 'Tangential')

    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    all_files = sorted(glob(os.path.join(pred_file_folder, '*.gz')))
    patient_names = list(set([os.path.basename(x).split('_')[0] for x in all_files]))

    for patient_name in tqdm(patient_names):
        array_list = []
        for filepath in all_files:
            if patient_name in filepath:
                data = nib.load(filepath)
                arr = data.get_fdata()
                array_list.append(arr)
        arr = np.stack(array_list, axis=0)
        arr = arr.transpose((3, 0, 1, 2)) # D, T, H, W
        save_strain(arr, output_path, patient_name)