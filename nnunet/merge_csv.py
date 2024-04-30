import nibabel as nib
from evaluation.metrics import hausdorff_distance, dice, avg_surface_distance_symmetric
from glob import glob
import os
from tqdm import tqdm
import numpy as np
import pickle
from batchgenerators.utilities.file_and_folder_operations import save_json
import pandas as pd


if __name__ == "__main__":

    csv_file_1 = r"C:\Users\Portal\Documents\voxelmorph\2023-12-14_17H47\Task032_Lib\fold_0\Lib\test\Raw\Flow\ssim_metrics.csv"
    csv_file_2 = r"C:\Users\Portal\Documents\voxelmorph\results\VM-NCC\Lib\test\Raw\Flow\ssim_metrics.csv"

    #merge_on = ['File ID',
    #            'Patient ID',
    #            'Study description',
    #            'Study Date (Date Anonymisation??',
    #            'HR',
    #            'LVM_ED',
    #            'LVM_ES',
    #            'LVEDV',
    #            'LVESV',
    #            'LVEF',
    #            'LV_SV',
    #            'LV_CO',
    #            'RVM_ES',
    #            'RVM_ED',
    #            'RVEDV',
    #            'RVESV',
    #            'RVEF',
    #            'RV_SV',
    #            'RV_CO',
    #            'TH1_ED',
    #            'TH2_ED',
    #            'TH3_ED',
    #            'TH4_ED',
    #            'TH5_ED',
    #            'TH6_ED',
    #            'TH7_ED',
    #            'TH8_ED',
    #            'TH9_ED',
    #            'TH10_ED',
    #            'TH11_ED',
    #            'TH12_ED',
    #            'TH13_ED',
    #            'TH_14_ED',
    #            'TH15_ED',
    #            'TH16_ED',
    #            'TH1_ES',
    #            'TH2_ES',
    #            'TH3_ES',
    #            'TH4_ES',
    #            'TH5_ES',
    #            'TH6_ES',
    #            'TH7_ES',
    #            'TH8_ES',
    #            'TH9_ES',
    #            'TH10_ES',
    #            'TH11_ES',
    #            'TH12_ES',
    #            'TH13_ES',
    #            'TH_14_ES',
    #            'TH15_ES',
    #            'TH16_ES']

    #merge_on = ['Name',
    #            'Phase']
    
    merge_on = ['Name',
                'slice_number']

    df1 = pd.read_csv(csv_file_1)
    df2 = pd.read_csv(csv_file_2)
    df3 = pd.merge(df1, df2, on=merge_on)
    #df3 = pd.merge(df1, df2, on='Name')
    #df3 = pd.merge(df1, df2, on=['Name', 'Slice nb', 'Frame nb'])
    #df3 = pd.merge(df1, df2, on=['Name', 'Slice nb'])

    df3.to_csv('metric_comparison.csv')
    
