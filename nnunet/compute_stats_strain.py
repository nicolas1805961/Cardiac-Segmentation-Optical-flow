from glob import glob
import os
import scipy
import csv
from tqdm import tqdm

#strain folder here:
folder = r'C:\Users\Portal\Documents\voxelmorph\2024-02-18_20H44\Task045_Lib\fold_0\Lib\val\Postprocessed\Strain'

path_list = glob(os.path.join(folder, 'AI', '*.mat'))
out_list = []
for path in tqdm(path_list):
    corresponding_gt_basename = os.path.basename(path)[:-6] + 'gt.mat'
    corresponding_gt_path = os.path.join(folder, 'GT', corresponding_gt_basename)
    mat_ai = scipy.io.loadmat(path, simplify_cells=True)
    mat_gt = scipy.io.loadmat(corresponding_gt_path, simplify_cells=True)
    peak_ai_rv = mat_ai['Structure_ai']['Scirc_RV_peak']
    peak_gt_rv = mat_gt['Structure_gt']['Scirc_RV_peak']
    peak_ai_radial = mat_ai['Structure_ai']['Sradial_LV_peak']
    peak_gt_radial = mat_gt['Structure_gt']['Sradial_LV_peak']
    peak_ai_circ = mat_ai['Structure_ai']['Scirc_LV_peak']
    peak_gt_circ = mat_gt['Structure_gt']['Scirc_LV_peak']
    data = {'patient': os.path.basename(path).split('_')[0],
               'slice_nb': os.path.basename(path).split('_')[2],
               'ES_peak_index_ai_radial': peak_ai_radial[0, 0],
               'ED_peak_index_ai_radial': peak_ai_radial[0, 1],
               'ES_peak_value_ai_radial': peak_ai_radial[1, 0],
               'ED_peak_value_ai_radial': peak_ai_radial[1, 1],
               'ES_peak_index_gt_radial': peak_gt_radial[0, 0],
               'ED_peak_index_gt_radial': peak_gt_radial[0, 1],
               'ES_peak_value_gt_radial': peak_gt_radial[1, 0],
               'ED_peak_value_gt_radial': peak_gt_radial[1, 1],
               'ES_peak_index_ai_circ': peak_ai_circ[0, 0],
               'ED_peak_index_ai_circ': peak_ai_circ[0, 1],
               'ES_peak_value_ai_circ': peak_ai_circ[1, 0],
               'ED_peak_value_ai_circ': peak_ai_circ[1, 1],
               'ES_peak_index_gt_circ': peak_gt_circ[0, 0],
               'ED_peak_index_gt_circ': peak_gt_circ[0, 1],
               'ES_peak_value_gt_circ': peak_gt_circ[1, 0],
               'ED_peak_value_gt_circ': peak_gt_circ[1, 1],
               'ES_peak_index_ai_rv': peak_ai_rv[0, 0] if type(peak_ai_rv) != int else None,
               'ED_peak_index_ai_rv': peak_ai_rv[0, 1] if type(peak_ai_rv) != int else None,
               'ES_peak_value_ai_rv': peak_ai_rv[1, 0] if type(peak_ai_rv) != int else None,
               'ED_peak_value_ai_rv': peak_ai_rv[1, 1] if type(peak_ai_rv) != int else None,
               'ES_peak_index_gt_rv': peak_gt_rv[0, 0] if type(peak_gt_rv) != int else None,
               'ED_peak_index_gt_rv': peak_gt_rv[0, 1] if type(peak_gt_rv) != int else None,
               'ES_peak_value_gt_rv': peak_gt_rv[1, 0] if type(peak_gt_rv) != int else None,
               'ED_peak_value_gt_rv': peak_gt_rv[1, 1] if type(peak_gt_rv) != int else None}
    
    out_list.append(data)

with open(os.path.join(folder, 'strain_metrics.csv'), 'w') as fd_csv:
    writer = csv.DictWriter(fd_csv, fieldnames=list(out_list[0].keys()))
    writer.writeheader() 
    writer.writerows(out_list)