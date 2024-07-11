from glob import glob
import os
import pickle
from tqdm import tqdm

pkl_list_in = glob(os.path.join('custom_lib_t_4', '**', '*.pkl'), recursive=True)
pkl_list_out = glob(os.path.join('Lib_resampling_training_mask', '*.pkl'))

patient_list = sorted(list(set([os.path.basename(x).split('_')[0] for x in pkl_list_out])))

all_patient_paths_pkl = []
for patient in patient_list:
    patient_files_pkl = []
    for pkl_path in pkl_list_out:
        if patient in pkl_path:
            patient_files_pkl.append(pkl_path)
    all_patient_paths_pkl.append(sorted(patient_files_pkl))

for patient_path_list in tqdm(all_patient_paths_pkl):
    patient_nb = os.path.basename(patient_path_list[0]).split('_')[0]
    new_property_path = os.path.join('custom_lib_t_4', patient_nb, 'info_01.pkl')
    with open(new_property_path, 'rb') as f:
        data_in = pickle.load(f)
    for path in patient_path_list:
        with open(path, 'rb') as f2:
            data_out = pickle.load(f2)
        data_out['ed_number'] = data_in['ed_number']
        data_out['es_number'] = data_in['es_number']

        with open(path, 'wb') as handle:
            pickle.dump(data_out, handle, protocol=pickle.HIGHEST_PROTOCOL)

