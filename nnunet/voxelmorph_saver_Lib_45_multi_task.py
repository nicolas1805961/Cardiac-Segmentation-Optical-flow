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
from nnunet.lib.utils import ConvBlocks2DGroup
from nnunet.lib.training_utils import build_2d_model
from nnunet.training.network_training.processor import Processor
from monai.transforms import ResizeWithPadOrCrop
from nnunet.postprocessing.connected_components import determine_postprocessing_custom
import argparse
import matplotlib.pyplot as plt


def postprocess(pred_path_list_registered_forward, 
                pred_path_list_registered_backward, 
                pred_path_list_flow_forward, 
                pred_path_list_flow_backward, 
                pred_path_list_registered_forward_ed, 
                pkl_path,
                full_image_size,
                newpath_flow_forward,
                newpath_flow_backward,
                newpath_registered_forward,
                newpath_registered_backward,
                patient_name):
    
    newpath_flow_forward = os.path.join(newpath_flow_forward, patient_name)
    newpath_flow_backward = os.path.join(newpath_flow_backward, patient_name)
    newpath_registered_forward = os.path.join(newpath_registered_forward, patient_name)
    newpath_registered_backward = os.path.join(newpath_registered_backward, patient_name)

    delete_if_exist(newpath_flow_forward)
    os.makedirs(newpath_flow_forward)

    delete_if_exist(newpath_flow_backward)
    os.makedirs(newpath_flow_backward)

    delete_if_exist(newpath_registered_forward)
    os.makedirs(newpath_registered_forward)

    delete_if_exist(newpath_registered_backward)
    os.makedirs(newpath_registered_backward)

    assert len(pred_path_list_registered_forward) == len(pred_path_list_registered_backward) == len(pred_path_list_flow_forward) == len(pred_path_list_flow_backward)

    for seg_path in pred_path_list_registered_forward_ed:
        fname = os.path.basename(seg_path)[:-4]
        corresponding_pkl_file = os.path.join(pkl_path, fname + '.pkl')

        with open(corresponding_pkl_file, 'rb') as f:
            properties = pickle.load(f)
            padding_need = properties['padding_need']
            size_before = properties['voxelmorph_size_before']

        
        arr_seg = np.load(seg_path)['seg']
        arr_seg = arr_seg[None, None] # 1, 1, C, H, W, D
        arr_seg = arr_seg.transpose(5, 0, 1, 2, 3, 4) # D, 1, 1, C, H, W

        slice_list_seg = []
        for d in range(len(arr_seg)):
            to_uncrop_seg = arr_seg[d] # 1, 1, C, H, W
            to_uncrop_seg = torch.from_numpy(to_uncrop_seg).to('cuda:0')
            uncroped_seg = processor.uncrop_no_registration(to_uncrop_seg, padding_need=padding_need[:, d][None]) # B, T, 1, H, W
            slice_list_seg.append(uncroped_seg[0, 0].cpu())
        arr_seg = np.stack(slice_list_seg, axis=-1)

        assert list(arr_seg.shape[1:-1]) == [full_image_size, full_image_size]

        cropper = ResizeWithPadOrCrop(spatial_size=size_before)

        arr_seg = cropper(arr_seg)
        assert size_before == list(arr_seg.shape[1:])
        arr_seg = arr_seg.transpose((0, 3, 1, 2)) # C, depth, H, W

        current_softmax_pred = arr_seg.transpose([0] + [i + 1 for i in transpose_backward])
        #current_raw_flow = current_raw_flow.transpose([0] + [i + 1 for i in transpose_backward])

        seg_path = os.path.join(newpath_registered_forward, fname + ".nii.gz")

        save_segmentation_nifti_from_softmax(current_softmax_pred, seg_path,
                                                            properties, interpolation_order, None,
                                                            None, None,
                                                            None, None, force_separate_z,
                                                            interpolation_order_z, False, None, None,
                                                            None, None)
        


    for registered_path_forward, registered_path_backward, flow_path_forward, flow_path_backward in zip(pred_path_list_registered_forward, pred_path_list_registered_backward, pred_path_list_flow_forward, pred_path_list_flow_backward):
        fname = os.path.basename(registered_path_forward)[:-4]
        corresponding_pkl_file = os.path.join(pkl_path, fname + '.pkl')

        with open(corresponding_pkl_file, 'rb') as f:
            properties = pickle.load(f)
            padding_need = properties['padding_need']
            size_before = properties['voxelmorph_size_before']

        current_flow_forward = np.load(flow_path_forward)['flow']
        current_flow_backward = np.load(flow_path_backward)['flow']
        #if rescale:
        #    current_flow = current_flow * (image_size / 2)

        current_flow_forward = current_flow_forward[None, None]
        current_flow_forward = current_flow_forward.transpose(4, 0, 1, 5, 2, 3) # D, 1, 1, 2, H, W

        current_flow_backward = current_flow_backward[None, None]
        current_flow_backward = current_flow_backward.transpose(4, 0, 1, 5, 2, 3) # D, 1, 1, 2, H, W
        
        arr_registered_forward = np.load(registered_path_forward)['seg']
        arr_registered_forward = arr_registered_forward[None, None] # 1, 1, C, H, W, D
        arr_registered_forward = arr_registered_forward.transpose(5, 0, 1, 2, 3, 4) # D, 1, 1, C, H, W

        arr_registered_backward = np.load(registered_path_backward)['seg']
        arr_registered_backward = arr_registered_backward[None, None] # 1, 1, C, H, W, D
        arr_registered_backward = arr_registered_backward.transpose(5, 0, 1, 2, 3, 4) # D, 1, 1, C, H, W

        slice_list_seg_forward = []
        slice_list_seg_backward = []
        slice_list_flow_forward = []
        slice_list_flow_backward = []
        for d in range(len(arr_registered_forward)):
            to_uncrop_seg_forward = arr_registered_forward[d] # 1, 1, 1, H, W
            to_uncrop_seg_backward = arr_registered_backward[d] # 1, 1, 1, H, W
            to_uncrop_flow_forward = current_flow_forward[d] # 1, 1, 2, H, W
            to_uncrop_flow_backward = current_flow_backward[d] # 1, 1, 2, H, W

            to_uncrop_seg_forward = torch.from_numpy(to_uncrop_seg_forward).to('cuda:0')
            to_uncrop_seg_backward = torch.from_numpy(to_uncrop_seg_backward).to('cuda:0')
            to_uncrop_flow_forward = torch.from_numpy(to_uncrop_flow_forward).to('cuda:0')
            to_uncrop_flow_backward = torch.from_numpy(to_uncrop_flow_backward).to('cuda:0')

            uncroped_seg_forward = processor.uncrop_no_registration(to_uncrop_seg_forward, padding_need=padding_need[:, d][None]) # B, T, 1, H, W
            uncroped_seg_backward = processor.uncrop_no_registration(to_uncrop_seg_backward, padding_need=padding_need[:, d][None]) # B, T, 1, H, W
            uncroped_flow_forward = processor.uncrop_no_registration(to_uncrop_flow_forward, padding_need=padding_need[:, d][None]) # B, T, 1, H, W
            uncroped_flow_backward = processor.uncrop_no_registration(to_uncrop_flow_backward, padding_need=padding_need[:, d][None]) # B, T, 1, H, W

            slice_list_seg_forward.append(uncroped_seg_forward[0, 0].cpu())
            slice_list_seg_backward.append(uncroped_seg_backward[0, 0].cpu())
            slice_list_flow_forward.append(uncroped_flow_forward[0, 0].cpu())
            slice_list_flow_backward.append(uncroped_flow_backward[0, 0].cpu())

        arr_seg_forward = np.stack(slice_list_seg_forward, axis=-1)
        arr_seg_backward = np.stack(slice_list_seg_backward, axis=-1)
        current_flow_forward = np.stack(slice_list_flow_forward, axis=-1)
        current_flow_backward = np.stack(slice_list_flow_backward, axis=-1)

        assert list(arr_seg_forward.shape[1:-1]) == [full_image_size, full_image_size]
        assert list(arr_seg_backward.shape[1:-1]) == [full_image_size, full_image_size]
        assert list(current_flow_forward.shape[1:-1]) == [full_image_size, full_image_size]
        assert list(current_flow_backward.shape[1:-1]) == [full_image_size, full_image_size]

        cropper = ResizeWithPadOrCrop(spatial_size=size_before)

        arr_seg_forward = cropper(arr_seg_forward)
        assert size_before == list(arr_seg_forward.shape[1:])
        arr_seg_forward = arr_seg_forward.transpose((0, 3, 1, 2)) # C, depth, H, W

        arr_seg_backward = cropper(arr_seg_backward)
        assert size_before == list(arr_seg_backward.shape[1:])
        arr_seg_backward = arr_seg_backward.transpose((0, 3, 1, 2)) # C, depth, H, W

        current_flow_forward = cropper(current_flow_forward)
        assert size_before == list(current_flow_forward.shape[1:])
        current_flow_forward = current_flow_forward.transpose((0, 3, 1, 2)) # C, depth, H, W

        current_flow_backward = cropper(current_flow_backward)
        assert size_before == list(current_flow_backward.shape[1:])
        current_flow_backward = current_flow_backward.transpose((0, 3, 1, 2)) # C, depth, H, W

        current_softmax_pred_forward = arr_seg_forward.transpose([0] + [i + 1 for i in transpose_backward])
        current_softmax_pred_backward = arr_seg_backward.transpose([0] + [i + 1 for i in transpose_backward])
        current_flow_forward = current_flow_forward.transpose([0] + [i + 1 for i in transpose_backward])
        current_flow_backward = current_flow_backward.transpose([0] + [i + 1 for i in transpose_backward])

        seg_path_forward = os.path.join(newpath_registered_forward, fname + ".nii.gz")
        seg_path_backward = os.path.join(newpath_registered_backward, fname + ".nii.gz")
        flow_path_forward = os.path.join(newpath_flow_forward, fname + ".npz")
        flow_path_backward = os.path.join(newpath_flow_backward, fname + ".npz")

        save_segmentation_nifti_from_softmax(current_softmax_pred_forward, seg_path_forward,
                                                            properties, interpolation_order, None,
                                                            None, None,
                                                            None, None, force_separate_z,
                                                            interpolation_order_z, False, current_flow_forward, flow_path_forward,
                                                            None, None)
        
        save_segmentation_nifti_from_softmax(current_softmax_pred_backward, seg_path_backward,
                                                            properties, interpolation_order, None,
                                                            None, None,
                                                            None, None, force_separate_z,
                                                            interpolation_order_z, False, current_flow_backward, flow_path_backward,
                                                            None, None)
        
    #base_registered = os.path.join(output_dir, 'Registered')
    determine_postprocessing_custom(newpath_registered_forward, '',
                                    final_subf_name='' + "_postprocessed", debug=True)
    
    #base_registered = os.path.join(output_dir, 'Segmentation')
    determine_postprocessing_custom(newpath_registered_backward, '',
                                    final_subf_name='' + "_postprocessed", debug=True)
    




def delete_if_exist(folder_name):
    dirpath = Path(folder_name)
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(dirpath)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='Dataset name')
    #parser.add_argument('--test_or_val', required=True, help='Whether this is testing or validation set')
    args = parser.parse_args()

    #dirname = "Lib_regu_not_all"

    #pred_path = os.path.join(r"../../voxelmorph/results", dirname)
    #pred_path = os.path.join(r"C:\Users\Portal\Documents\voxelmorph\results", dirname)
    #pred_path = r"C:\Users\Portal\Documents\voxelmorph\Flow_results\Lib\No_regularisation\Simple\inference_alt"
    #pred_path = r"../../voxelmorph/testings2/inference"

    #folder before Raw here
    pred_path = r"C:\Users\Portal\Documents\voxelmorph\multi_task\2024-04-25_21H46_45s_686392\Task045_Lib\fold_0\Lib\val"

    if args.dataset == 'Lib':
        cropper_weights_folder_path = r"binary_lib"
        image_size = 384
        crop_size = 192
        window_size = 8
        if '\\val' in pred_path:
            plan_path = r"out/nnUNet_preprocessed/Task045_Lib/custom_experiment_planner_plans_2D.pkl"
            pkl_path = r"Lib_no_resampling"
        elif '\\test' in pred_path:
            plan_path = r"out/nnUNet_preprocessed/Task046_Lib/custom_experiment_planner_plans_2D.pkl"
            pkl_path = r"Lib_no_resampling_testing"
    elif args.dataset == 'ACDC':
        plan_path = r"out/nnUNet_preprocessed/Task031_ACDC/custom_experiment_planner_plans_2D.pkl"
        image_size = 224
        crop_size = 128
        window_size = 7
        cropper_weights_folder_path = r"binary"
        if args.test_or_val == 'val':
            pkl_path = r"C:\Users\Portal\Documents\voxelmorph\voxelmorph_ACDC_2D"
        elif args.test_or_val == 'test':
            pkl_path = r"C:\Users\Portal\Documents\voxelmorph\voxelmorph_ACDC_2D_testing"
    #pkl_path = r"../../voxelmorph/voxelmorph_Lib_2D"

    output_dir = os.path.join(pred_path, 'Postprocessed')

    delete_if_exist(output_dir)
    os.makedirs(output_dir)

    newpath_flow_forward = os.path.join(output_dir, 'Forward_flow')
    delete_if_exist(newpath_flow_forward)
    os.makedirs(newpath_flow_forward)

    newpath_flow_backward = os.path.join(output_dir, 'Backward_flow')
    delete_if_exist(newpath_flow_backward)
    os.makedirs(newpath_flow_backward)

    newpath_registered_forward = os.path.join(output_dir, 'Registered_forward')
    delete_if_exist(newpath_registered_forward)
    os.makedirs(newpath_registered_forward)

    newpath_registered_backward = os.path.join(output_dir, 'Registered_backward')
    delete_if_exist(newpath_registered_backward)
    os.makedirs(newpath_registered_backward)

    cropper_config = read_config(os.path.join(Path.cwd(), 'adversarial_acdc.yaml'), False, False)

    cropping_conv_layer = ConvBlocks2DGroup
    cropping_network = build_2d_model(cropper_config, conv_layer=cropping_conv_layer, norm=getattr(torch.nn, cropper_config['norm']), log_function=None, image_size=image_size, window_size=window_size, middle=False, num_classes=2, processor=None)
    cropping_network.load_state_dict(torch.load(os.path.join(cropper_weights_folder_path, 'model_final_checkpoint.model'))['state_dict'], strict=True)
    cropping_network.eval()
    cropping_network.do_ds = False

    processor = Processor(crop_size=crop_size, image_size=image_size, cropping_network=cropping_network)

    #plan_path = r"C:\Users\Portal\Documents\Isensee\nnUNet\nnunet\out\nnUNet_preprocessed\Task032_Lib\custom_experiment_planner_plans_2D.pkl"

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

    registered_dir = os.path.join(pred_path, 'Raw', 'Registered_forward')
    registered_patients = [name for name in os.listdir(registered_dir) if os.path.isdir(os.path.join(registered_dir, name))]

    for patient_name in tqdm(registered_patients):
        
        pred_path_list_registered_forward = sorted(glob(os.path.join(pred_path, 'Raw', 'Registered_forward', patient_name, "*.npz")))
        pred_path_list_registered_backward = sorted(glob(os.path.join(pred_path, 'Raw', 'Registered_backward', patient_name, "*.npz")))
        pred_path_list_flow_forward = sorted(glob(os.path.join(pred_path, 'Raw', 'Forward_flow', patient_name, "*.npz")))
        pred_path_list_flow_backward = sorted(glob(os.path.join(pred_path, 'Raw', 'Backward_flow', patient_name, "*.npz")))

        pred_path_list_flow_basename = [os.path.basename(x)[:-4] for x in pred_path_list_flow_forward]

        pred_path_list_registered_backward = [x for x in pred_path_list_registered_backward if os.path.basename(x)[:-4] in pred_path_list_flow_basename]

        pred_path_list_registered_forward_ed = [x for x in pred_path_list_registered_forward if os.path.basename(x)[:-4] not in pred_path_list_flow_basename]
        pred_path_list_registered_forward = [x for x in pred_path_list_registered_forward if os.path.basename(x)[:-4] in pred_path_list_flow_basename]

        postprocess(pred_path_list_registered_forward, pred_path_list_registered_backward, pred_path_list_flow_forward, pred_path_list_flow_backward, pred_path_list_registered_forward_ed, pkl_path, image_size, newpath_flow_forward, newpath_flow_backward, newpath_registered_forward, newpath_registered_backward, patient_name)