import os
import scipy
from glob import glob
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

path_list_method = {"V1": r'C:\Users\Portal\Documents\voxelmorph\multi_task\2024-04-15_02H11_10s_514482\Task045_Lib\fold_0\Lib\val\Postprocessed\Strain',
                    "V2": r'C:\Users\Portal\Documents\voxelmorph\new_models_256\2024-03-01_23H51_16s_864320\Task045_Lib\fold_0\Lib\val\Postprocessed\Strain',
                    "V3": r'C:\Users\Portal\Documents\voxelmorph\multi_task\2024-04-17_09H19_55s_569829\Task045_Lib\fold_0\Lib\val\Postprocessed\Strain',
                    }

fig, ax = plt.subplots(1, 2)
for key in path_list_method.keys():

    path_list_ai = sorted(glob(os.path.join(path_list_method[key], 'AI', '*.mat')))
    path_list_gt = sorted(glob(os.path.join(path_list_method[key], 'GT', '*.mat')))

    #out = {'AI': {"radial_lv": [], "circ_lv": [], "circ_rv": []},
    #        'GT': {"radial_lv": [], "circ_lv": [], "circ_rv": []}}

    out = {'AI': {"radial_lv": [], "circ_lv": []},
            'GT': {"radial_lv": [], "circ_lv": []}}

    for ai_path, gt_path in zip(path_list_ai, path_list_gt):

        ai_mat = scipy.io.loadmat(ai_path, simplify_cells=True)
        gt_mat = scipy.io.loadmat(gt_path, simplify_cells=True)

        Sradial_LV_curve_ai = ai_mat['Structure_ai']['Sradial_LV_curve']
        Sradial_LV_curve_gt = gt_mat['Structure_gt']['Sradial_LV_curve']

        Scirc_LV_curve_ai = ai_mat['Structure_ai']['Scirc_LV_curve']
        Scirc_LV_curve_gt = gt_mat['Structure_gt']['Scirc_LV_curve']

        #Scirc_RV_curve_ai = ai_mat['Structure_ai']['Scirc_RV_curve']
        #Scirc_RV_curve_gt = gt_mat['Structure_gt']['Scirc_RV_curve']
        
        out['AI']['radial_lv'].append(Sradial_LV_curve_ai)
        out['AI']['circ_lv'].append(Scirc_LV_curve_ai)
        #out['AI']['circ_rv'].append(Scirc_RV_curve_ai)
        out['GT']['radial_lv'].append(Sradial_LV_curve_gt)
        out['GT']['circ_lv'].append(Scirc_LV_curve_gt)
        #out['GT']['circ_rv'].append(Sradial_LV_curve_gt)

    m = max([len(x) for x in out['GT']['radial_lv']])

    #out_plot = {'AI': {"radial_lv": [], "circ_lv": [], "circ_rv": []},
    #        'GT': {"radial_lv": [], "circ_lv": [], "circ_rv": []}}

    out_plot = {'AI': {"radial_lv": [], "circ_lv": []},
            'GT': {"radial_lv": [], "circ_lv": []}}

    for k1 in out.keys():
        for k2 in out[k1].keys():
            for data in out[k1][k2]:

                x = np.arange(0, len(data))
                f1 = interpolate.interp1d(x, data)
                x_new = np.linspace(0, len(data) - 1, m)
                y_new = np.array(f1(x_new))

                out_plot[k1][k2].append(y_new)
            
            out_plot[k1][k2] = np.stack(out_plot[k1][k2], axis=0).mean(0)


    for idx, (data_ai, data_gt) in enumerate(zip(out_plot['AI'].values(), out_plot['GT'].values())):
        ax[idx].plot(data_ai, label=key)
        ax[idx].legend()

for idx, (data_ai, data_gt) in enumerate(zip(out_plot['AI'].values(), out_plot['GT'].values())):
        ax[idx].plot(data_gt, label='GT')
        ax[idx].legend()

plt.show()