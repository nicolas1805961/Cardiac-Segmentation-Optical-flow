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

def postprocess_no_seg(pred_path_list_registered, 
                        pred_path_list_flow, 
                        pkl_path,
                        full_image_size,
                        newpath_flow,
                        newpath_registered,
                        patient_name):


    newpath_flow = os.path.join(newpath_flow, patient_name)
    newpath_registered = os.path.join(newpath_registered, patient_name)

    delete_if_exist(newpath_flow)
    os.makedirs(newpath_flow)

    delete_if_exist(newpath_registered)
    os.makedirs(newpath_registered)

    assert len(pred_path_list_registered) == len(pred_path_list_flow)

    for registered_path, flow_path in tqdm(zip(pred_path_list_registered, pred_path_list_flow), total=len(pred_path_list_registered)):
        fname = os.path.basename(registered_path)[:-7]
        corresponding_pkl_file = os.path.join(pkl_path, fname + '.pkl')

        with open(corresponding_pkl_file, 'rb') as f:
            properties = pickle.load(f)
            padding_need = properties['padding_need']
            size_before = properties['voxelmorph_size_before']
        
        #print(properties['Database'])
        #print(properties['spacing_after_resampling'])
        #print(properties['size_after_resampling'])
        #print(properties['size_after_cropping'])
        #print(properties['original_spacing'])
        #print(properties['original_size_of_raw_data'])

        current_flow = np.load(flow_path)['flow']
        #if rescale:
        #    current_flow = current_flow * (image_size / 2)
        current_flow = current_flow[None, None]
        current_flow = current_flow.transpose(4, 0, 1, 5, 2, 3) # D, 1, 1, 2, H, W
        
        data_registered = nib.load(registered_path)
        arr_registered = data_registered.get_fdata()[None, None, None] # 1, 1, 1, H, W, D
        arr_registered = arr_registered.transpose(5, 0, 1, 2, 3, 4) # D, 1, 1, 1, H, W

        slice_list_registered = []
        slice_list_flow = []
        for d in range(len(arr_registered)):
            to_uncrop_registered = arr_registered[d] # 1, 1, 1, H, W
            to_uncrop_flow = current_flow[d] # 1, 1, 2, H, W
            to_uncrop_registered = torch.from_numpy(to_uncrop_registered).to('cuda:0')
            to_uncrop_flow = torch.from_numpy(to_uncrop_flow).to('cuda:0')
            uncroped_registered = processor.uncrop_no_registration(to_uncrop_registered, padding_need=padding_need[:, d][None]) # B, T, 1, H, W
            uncroped_flow = processor.uncrop_no_registration(to_uncrop_flow, padding_need=padding_need[:, d][None]) # B, T, 1, H, W
            slice_list_registered.append(uncroped_registered[0, 0].cpu())
            slice_list_flow.append(uncroped_flow[0, 0].cpu())
        arr_registered = np.stack(slice_list_registered, axis=-1)
        current_flow = np.stack(slice_list_flow, axis=-1)

        assert list(arr_registered.shape[1:-1]) == [full_image_size, full_image_size]
        assert list(current_flow.shape[1:-1]) == [full_image_size, full_image_size]

        cropper = ResizeWithPadOrCrop(spatial_size=size_before)

        #arr = np.pad(arr, padding)
        arr_registered = cropper(arr_registered)
        assert size_before == list(arr_registered.shape[1:])
        arr_registered = arr_registered.transpose((0, 3, 1, 2)) # C, depth, H, W
        C, D, H, W = arr_registered.shape

        current_flow = cropper(current_flow)
        assert size_before == list(current_flow.shape[1:])
        current_flow = current_flow.transpose((0, 3, 1, 2)) # C, depth, H, W

        arr_seg = np.zeros_like(arr_registered)

        current_registered = arr_registered.transpose([0] + [i + 1 for i in transpose_backward])
        current_softmax_pred = arr_seg.transpose([0] + [i + 1 for i in transpose_backward])
        current_flow = current_flow.transpose([0] + [i + 1 for i in transpose_backward])
        #current_raw_flow = current_raw_flow.transpose([0] + [i + 1 for i in transpose_backward])

        seg_path = os.path.join(newpath_seg, fname + ".nii.gz")
        registered_path = os.path.join(newpath_registered, fname + ".nii.gz")
        flow_path = os.path.join(newpath_flow, fname + ".npz")

        save_segmentation_nifti_from_softmax(current_softmax_pred, seg_path,
                                                            properties, interpolation_order, None,
                                                            None, None,
                                                            None, None, force_separate_z,
                                                            interpolation_order_z, False, current_flow, flow_path,
                                                            current_registered, registered_path)
        
    #base_registered = os.path.join(output_dir, 'Registered')
    determine_postprocessing_custom(newpath_registered, '',
                                    final_subf_name='' + "_postprocessed", debug=True)



def postprocess(pred_path_list_registered, 
                pred_path_list_seg, 
                pred_path_list_flow, 
                pred_path_list_seg_ed, 
                pkl_path,
                full_image_size,
                newpath_flow,
                newpath_registered,
                newpath_seg,
                patient_name):
    
    newpath_flow = os.path.join(newpath_flow, patient_name)
    newpath_registered = os.path.join(newpath_registered, patient_name)
    newpath_seg = os.path.join(newpath_seg, patient_name)

    delete_if_exist(newpath_flow)
    os.makedirs(newpath_flow)

    delete_if_exist(newpath_registered)
    os.makedirs(newpath_registered)

    delete_if_exist(newpath_seg)
    os.makedirs(newpath_seg)

    assert len(pred_path_list_registered) == len(pred_path_list_seg) == len(pred_path_list_flow)

    for seg_path in tqdm(pred_path_list_seg_ed):
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

        seg_path = os.path.join(newpath_seg, fname + ".nii.gz")

        save_segmentation_nifti_from_softmax(current_softmax_pred, seg_path,
                                                            properties, interpolation_order, None,
                                                            None, None,
                                                            None, None, force_separate_z,
                                                            interpolation_order_z, False, None, None,
                                                            None, None)
        


    for registered_path, flow_path, seg_path in tqdm(zip(pred_path_list_registered, pred_path_list_flow, pred_path_list_seg), total=len(pred_path_list_registered)):
        fname = os.path.basename(seg_path)[:-4]
        corresponding_pkl_file = os.path.join(pkl_path, fname + '.pkl')

        with open(corresponding_pkl_file, 'rb') as f:
            properties = pickle.load(f)
            padding_need = properties['padding_need']
            size_before = properties['voxelmorph_size_before']

        current_flow = np.load(flow_path)['flow']
        #if rescale:
        #    current_flow = current_flow * (image_size / 2)

        current_flow = current_flow[None, None]
        current_flow = current_flow.transpose(4, 0, 1, 5, 2, 3) # D, 1, 1, 2, H, W
        
        data_registered = nib.load(registered_path)
        arr_registered = data_registered.get_fdata()[None, None, None] # 1, 1, 1, H, W, D
        arr_registered = arr_registered.transpose(5, 0, 1, 2, 3, 4) # D, 1, 1, 1, H, W

        arr_seg = np.load(seg_path)['seg']
        arr_seg = arr_seg[None, None] # 1, 1, C, H, W, D
        arr_seg = arr_seg.transpose(5, 0, 1, 2, 3, 4) # D, 1, 1, C, H, W

        slice_list_seg = []
        slice_list_registered = []
        slice_list_flow = []
        for d in range(len(arr_registered)):
            to_uncrop_seg = arr_seg[d] # 1, 1, 1, H, W
            to_uncrop_registered = arr_registered[d] # 1, 1, 1, H, W
            to_uncrop_flow = current_flow[d] # 1, 1, 2, H, W
            to_uncrop_seg = torch.from_numpy(to_uncrop_seg).to('cuda:0')
            to_uncrop_registered = torch.from_numpy(to_uncrop_registered).to('cuda:0')
            to_uncrop_flow = torch.from_numpy(to_uncrop_flow).to('cuda:0')
            uncroped_seg = processor.uncrop_no_registration(to_uncrop_seg, padding_need=padding_need[:, d][None]) # B, T, 1, H, W
            uncroped_registered = processor.uncrop_no_registration(to_uncrop_registered, padding_need=padding_need[:, d][None]) # B, T, 1, H, W
            uncroped_flow = processor.uncrop_no_registration(to_uncrop_flow, padding_need=padding_need[:, d][None]) # B, T, 1, H, W
            slice_list_seg.append(uncroped_seg[0, 0].cpu())
            slice_list_registered.append(uncroped_registered[0, 0].cpu())
            slice_list_flow.append(uncroped_flow[0, 0].cpu())
        arr_seg = np.stack(slice_list_seg, axis=-1)
        arr_registered = np.stack(slice_list_registered, axis=-1)
        current_flow = np.stack(slice_list_flow, axis=-1)

        assert list(arr_seg.shape[1:-1]) == [full_image_size, full_image_size]
        assert list(arr_registered.shape[1:-1]) == [full_image_size, full_image_size]
        assert list(current_flow.shape[1:-1]) == [full_image_size, full_image_size]

        cropper = ResizeWithPadOrCrop(spatial_size=size_before)

        arr_seg = cropper(arr_seg)
        assert size_before == list(arr_seg.shape[1:])
        arr_seg = arr_seg.transpose((0, 3, 1, 2)) # C, depth, H, W

        #arr = np.pad(arr, padding)
        arr_registered = cropper(arr_registered)
        assert size_before == list(arr_registered.shape[1:])
        arr_registered = arr_registered.transpose((0, 3, 1, 2)) # C, depth, H, W
        C, D, H, W = arr_registered.shape

        current_flow = cropper(current_flow)
        assert size_before == list(current_flow.shape[1:])
        current_flow = current_flow.transpose((0, 3, 1, 2)) # C, depth, H, W

        current_registered = arr_registered.transpose([0] + [i + 1 for i in transpose_backward])
        current_softmax_pred = arr_seg.transpose([0] + [i + 1 for i in transpose_backward])
        current_flow = current_flow.transpose([0] + [i + 1 for i in transpose_backward])
        #current_raw_flow = current_raw_flow.transpose([0] + [i + 1 for i in transpose_backward])

        seg_path = os.path.join(newpath_seg, fname + ".nii.gz")
        registered_path = os.path.join(newpath_registered, fname + ".nii.gz")
        flow_path = os.path.join(newpath_flow, fname + ".npz")

        save_segmentation_nifti_from_softmax(current_softmax_pred, seg_path,
                                                            properties, interpolation_order, None,
                                                            None, None,
                                                            None, None, force_separate_z,
                                                            interpolation_order_z, False, current_flow, flow_path,
                                                            current_registered, registered_path)
        
    #base_registered = os.path.join(output_dir, 'Registered')
    determine_postprocessing_custom(newpath_registered, '',
                                    final_subf_name='' + "_postprocessed", debug=True)
    
    #base_registered = os.path.join(output_dir, 'Segmentation')
    determine_postprocessing_custom(newpath_seg, '',
                                    final_subf_name='' + "_postprocessed", debug=True)

def delete_if_exist(folder_name):
    dirpath = Path(folder_name)
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(dirpath)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--no_seg', required=False, action='store_true', help='Whether to save direct segmentation')
    parser.add_argument('--dataset', required=True, help='Dataset name')
    #parser.add_argument('--test_or_val', required=True, help='Whether this is testing or validation set')
    args = parser.parse_args()

    #dirname = "Lib_regu_not_all"

    #pred_path = os.path.join(r"../../voxelmorph/results", dirname)
    #pred_path = os.path.join(r"C:\Users\Portal\Documents\voxelmorph\results", dirname)
    #pred_path = r"C:\Users\Portal\Documents\voxelmorph\Flow_results\Lib\No_regularisation\Simple\inference_alt"
    #pred_path = r"../../voxelmorph/testings2/inference"

    #folder before Raw here
    pred_path = r"C:\Users\Portal\Documents\voxelmorph\2023-12-15_19H01\temp_sum\Lib\test"

    if args.dataset == 'Lib':
        cropper_weights_folder_path = r"binary_lib"
        image_size = 384
        crop_size = 192
        window_size = 8
        if '\\val' in pred_path:
            plan_path = r"out/nnUNet_preprocessed/Task032_Lib/custom_experiment_planner_plans_2D.pkl"
            pkl_path = r"voxelmorph_Lib_2D"
        elif '\\test' in pred_path:
            plan_path = r"out/nnUNet_preprocessed/Task036_Lib/custom_experiment_planner_plans_2D.pkl"
            pkl_path = r"voxelmorph_Lib_2D_testing"
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

    newpath_flow = os.path.join(output_dir, 'Flow')
    delete_if_exist(newpath_flow)
    os.makedirs(newpath_flow)

    newpath_registered = os.path.join(output_dir, 'Registered')
    delete_if_exist(newpath_registered)
    os.makedirs(newpath_registered)

    newpath_seg = os.path.join(output_dir, 'Segmentation')
    delete_if_exist(newpath_seg)
    os.makedirs(newpath_seg)

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

    registered_dir = os.path.join(pred_path, 'Raw', 'Registered')
    registered_patients = [name for name in os.listdir(registered_dir) if os.path.isdir(os.path.join(registered_dir, name))]

    for patient_name in tqdm(registered_patients):
        
        pred_path_list_registered = sorted(glob(os.path.join(pred_path, 'Raw', 'Registered', patient_name, "*.gz")))
        pred_path_list_seg = sorted(glob(os.path.join(pred_path, 'Raw', 'Segmentation', patient_name, "*.npz")))
        pred_path_list_flow = sorted(glob(os.path.join(pred_path, 'Raw', 'Flow', patient_name, "*.npz")))

        pred_path_list_registered_basename = [os.path.basename(x)[:-7] for x in pred_path_list_registered]
        pred_path_list_seg_ed = [x for x in pred_path_list_seg if os.path.basename(x)[:-4] not in pred_path_list_registered_basename]
        pred_path_list_seg = [x for x in pred_path_list_seg if os.path.basename(x)[:-4] in pred_path_list_registered_basename]

        print(patient_name)
        print(len(pred_path_list_seg))
        print(len(pred_path_list_registered))
        print(len(pred_path_list_flow))
        print(len(pred_path_list_seg_ed))

        if args.no_seg:
            postprocess_no_seg(pred_path_list_registered, pred_path_list_flow, pkl_path, image_size, newpath_flow, newpath_registered, patient_name)
        else:
            postprocess(pred_path_list_registered, pred_path_list_seg, pred_path_list_flow, pred_path_list_seg_ed, pkl_path, image_size, newpath_flow, newpath_registered, newpath_seg, patient_name)