from nnunet.inference.segmentation_export import save_segmentation_nifti_from_softmax
import shutil
import os
import numpy as np
from glob import glob
from tqdm import tqdm
from pathlib import Path
import nibabel as nib
import pickle
import torch
from nnunet.lib.training_utils import read_config
from nnunet.lib.utils import ConvBlocks2D
from nnunet.lib.training_utils import build_2d_model
from nnunet.training.network_training.processor import Processor
from monai.transforms import ResizeWithPadOrCrop
from nnunet.postprocessing.connected_components import determine_postprocessing_no_metric

def delete_if_exist(folder_name):
    dirpath = Path(folder_name)
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(dirpath)

if __name__ == "__main__":

    dirname = "2023-10-07_23H25"

    pred_path = os.path.join(r"C:\Users\Portal\Documents\voxelmorph\results", dirname)
    pkl_path = r"voxelmorph_ACDC_2D"
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

    cropper_weights_folder_path = r"C:\Users\Portal\Documents\Isensee\nnUNet\nnunet\binary"
    cropper_config = read_config(os.path.join(Path.cwd(), 'adversarial_acdc.yaml'), False, False)

    cropping_conv_layer = ConvBlocks2DGroup
    cropping_network = build_2d_model(cropper_config, conv_layer=cropping_conv_layer, norm=getattr(torch.nn, cropper_config['norm']), log_function=None, image_size=224, window_size=7, middle=False, num_classes=2)
    cropping_network.load_state_dict(torch.load(os.path.join(cropper_weights_folder_path, 'model_final_checkpoint.model'))['state_dict'], strict=True)
    cropping_network.eval()
    cropping_network.do_ds = False

    processor = Processor(crop_size=128, image_size=224, cropping_network=cropping_network)

    plan_path = r"C:\Users\Portal\Documents\Isensee\nnUNet\nnunet\out\nnUNet_preprocessed\Task031_ACDC\custom_experiment_planner_plans_2D.pkl"

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
    
    pred_path_list = sorted(glob(os.path.join(pred_path, "*.gz")))
    pred_path_list_flow = sorted(glob(os.path.join(pred_path, "*raw.npz")))
    for path, flow_path in tqdm(zip(pred_path_list, pred_path_list_flow), total=len(pred_path_list)):
        fname = os.path.basename(path)[:-7]
        corresponding_pkl_file = os.path.join(pkl_path, fname + '.pkl')

        with open(corresponding_pkl_file, 'rb') as f:
            properties = pickle.load(f)
            padding_need = properties['padding_need']
            size_before = properties['voxelmorph_size_before']
        
        current_flow = np.load(flow_path)['flow']
        current_flow = current_flow[None, None]
        current_flow = current_flow.transpose(4, 0, 1, 5, 2, 3) # D, 1, 1, 2, H, W

        #current_raw_flow = np.load(flow_path)
        
        data = nib.load(path)
        arr = data.get_fdata()[None, None, None] # 1, 1, 1, H, W, D
        arr = arr.transpose(5, 0, 1, 2, 3, 4) # D, 1, 1, 1, H, W

        slice_list = []
        slice_list_flow = []
        for d in range(len(arr)):
            to_uncrop = arr[d] # 1, 1, 1, H, W
            to_uncrop_flow = current_flow[d] # 1, 1, 2, H, W
            to_uncrop = torch.from_numpy(to_uncrop).to('cuda:0')
            to_uncrop_flow = torch.from_numpy(to_uncrop_flow).to('cuda:0')
            uncroped = processor.uncrop_no_registration(to_uncrop, padding_need=padding_need[:, d][None]) # B, T, 1, H, W
            uncroped_flow = processor.uncrop_no_registration(to_uncrop_flow, padding_need=padding_need[:, d][None]) # B, T, 1, H, W
            slice_list.append(uncroped[0, 0].cpu())
            slice_list_flow.append(uncroped_flow[0, 0].cpu())
        arr = np.stack(slice_list, axis=-1)
        current_flow = np.stack(slice_list_flow, axis=-1)

        assert list(arr.shape[1:-1]) == [224, 224]
        assert list(current_flow.shape[1:-1]) == [224, 224]

        cropper = ResizeWithPadOrCrop(spatial_size=size_before)

        #arr = np.pad(arr, padding)
        arr = cropper(arr)
        assert size_before == list(arr.shape[1:])
        arr = arr.transpose((0, 3, 1, 2)) # C, depth, H, W
        C, D, H, W = arr.shape

        current_flow = cropper(current_flow)
        assert size_before == list(current_flow.shape[1:])
        current_flow = current_flow.transpose((0, 3, 1, 2)) # C, depth, H, W

        current_softmax_pred = np.zeros(shape=(4, D, H, W))

        current_registered = arr.transpose([0] + [i + 1 for i in transpose_backward])
        current_softmax_pred = current_softmax_pred.transpose([0] + [i + 1 for i in transpose_backward])
        current_flow = current_flow.transpose([0] + [i + 1 for i in transpose_backward])
        #current_raw_flow = current_raw_flow.transpose([0] + [i + 1 for i in transpose_backward])

        registered_path = os.path.join(newpath_registered, fname + ".nii.gz")
        flow_path = os.path.join(newpath_flow, fname + ".npz")

        save_segmentation_nifti_from_softmax(current_softmax_pred, os.path.join(newpath_seg, fname + ".nii.gz"),
                                                            properties, interpolation_order, None,
                                                            None, None,
                                                            None, None, force_separate_z,
                                                            interpolation_order_z, False, current_flow, flow_path,
                                                            current_registered, registered_path, None)
        
        base_registered = os.path.join(output_dir, 'Registered')
        determine_postprocessing_no_metric(base_registered, '',
                                     final_subf_name='' + "_postprocessed", debug=True)