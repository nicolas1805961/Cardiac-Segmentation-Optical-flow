from nnunet.inference.segmentation_export import save_segmentation_nifti_from_softmax
import shutil
import os
import numpy as np
from glob import glob
from tqdm import tqdm
from pathlib import Path
import nibabel as nib
import pickle
from monai.transforms import ResizeWithPadOrCrop

def delete_if_exist(folder_name):
    dirpath = Path(folder_name)
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(dirpath)

if __name__ == "__main__":

    dirname = "2023-09-23_17H55"

    pred_path = os.path.join(r"C:\Users\Portal\Documents\voxelmorph\results", dirname)
    pkl_path = r"C:\Users\Portal\Documents\voxelmorph\voxelmorph_ACDC"
    output_dir = os.path.join(pred_path, 'Validation')

    delete_if_exist(output_dir)
    os.makedirs(output_dir)

    newpath_flow = os.path.join(output_dir, 'Flow')
    delete_if_exist(newpath_flow)
    os.makedirs(newpath_flow)

    newpath_registered = os.path.join(output_dir, 'Registered')
    delete_if_exist(newpath_registered)
    os.makedirs(newpath_registered)

    newpath_seg = os.path.join(output_dir, 'Segmentation')
    delete_if_exist(newpath_seg)
    os.makedirs(newpath_seg)

    plan_path = r"C:\Users\Portal\Documents\Isensee\nnUNet\nnunet\out\nnUNet_preprocessed\Task033_ACDC\nnUNetPlansv2.1_plans_3D.pkl"

    with open(plan_path, 'rb') as f:
        plans = pickle.load(f)
    
    if plans.get('transpose_forward') is None or plans.get('transpose_backward') is None:
        print("WARNING! You seem to have data that was preprocessed with a previous version of nnU-Net. "
                "You should rerun preprocessing. We will proceed and assume that both transpose_foward "
                "and transpose_backward are [0, 1, 2]. If that is not correct then weird things will happen!")
        plans['transpose_forward'] = [0, 1, 2]
        plans['transpose_backward'] = [0, 1, 2]
    transpose_forward = plans['transpose_forward']
    transpose_backward = plans['transpose_backward']

    if 'segmentation_export_params' in plans.keys():
        force_separate_z = plans['segmentation_export_params']['force_separate_z']
        interpolation_order = plans['segmentation_export_params']['interpolation_order']
        interpolation_order_z = plans['segmentation_export_params']['interpolation_order_z']
    else:
        force_separate_z = None
        interpolation_order = 1
        interpolation_order_z = 0
    
    pred_path_list = glob(os.path.join(pred_path, "*.gz"))
    for path in tqdm(pred_path_list):
        fname = os.path.basename(path)[:-7]
        corresponding_pkl_file = os.path.join(pkl_path, fname + '.pkl')

        with open(corresponding_pkl_file, 'rb') as f:
            properties = pickle.load(f)
            old_shape = properties['shape_before_pre_processing']
            cropper = ResizeWithPadOrCrop(spatial_size=old_shape)
        
        data = nib.load(path)
        arr = data.get_fdata()[None]

        arr = cropper(arr)
        arr = arr.transpose((0, 3, 1, 2)) # C, depth, H, W
        C, D, H, W = arr.shape

        current_softmax_pred = np.zeros(shape=(4, D, H, W))
        current_flow = np.zeros(shape=(2, D, H, W))

        current_registered = arr.transpose([0] + [i + 1 for i in transpose_backward])
        current_softmax_pred = current_softmax_pred.transpose([0] + [i + 1 for i in transpose_backward])
        current_flow = current_flow.transpose([0] + [i + 1 for i in transpose_backward])

        registered_path = os.path.join(newpath_registered, fname + ".nii.gz")
        flow_path = os.path.join(newpath_flow, fname + ".npz")

        save_segmentation_nifti_from_softmax(current_softmax_pred, os.path.join(newpath_seg, fname + ".nii.gz"),
                                                            properties, interpolation_order, None,
                                                            None, None,
                                                            None, None, force_separate_z,
                                                            interpolation_order_z, False, current_flow, flow_path,
                                                            current_registered, registered_path)