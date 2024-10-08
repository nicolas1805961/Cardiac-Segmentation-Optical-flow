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

import SimpleITK as sitk
import numpy as np
import shutil
from batchgenerators.utilities.file_and_folder_operations import *
from multiprocessing import Pool
from collections import OrderedDict
import sys
import matplotlib
import matplotlib.pyplot as plt


def create_nonzero_mask(data):
    from scipy.ndimage import binary_fill_holes
    assert len(data.shape) == 4 or len(data.shape) == 3, "data must have shape (C, X, Y, Z) or shape (C, X, Y)"
    nonzero_mask = np.zeros(data.shape[1:], dtype=bool)
    for c in range(data.shape[0]):
        this_mask = data[c] != 0
        nonzero_mask = nonzero_mask | this_mask
    nonzero_mask = binary_fill_holes(nonzero_mask)

    #matplotlib.use('QtAgg')
    #fig, ax = plt.subplots(2, 3)
    #ax[0, 0].imshow(data[0, 0], cmap='gray')
    #ax[0, 1].imshow(data[0, 1], cmap='gray')
    #ax[0, 2].imshow(data[0, 2], cmap='gray')
    #ax[1, 0].imshow(nonzero_mask[0], cmap='gray')
    #ax[1, 1].imshow(nonzero_mask[1], cmap='gray')
    #ax[1, 2].imshow(nonzero_mask[2], cmap='gray')
    #plt.show()
    
    return nonzero_mask


def get_bbox_from_mask(mask, outside_value=0):
    mask_voxel_coords = np.where(mask != outside_value)
    minzidx = int(np.min(mask_voxel_coords[0]))
    maxzidx = int(np.max(mask_voxel_coords[0])) + 1
    minxidx = int(np.min(mask_voxel_coords[1]))
    maxxidx = int(np.max(mask_voxel_coords[1])) + 1
    minyidx = int(np.min(mask_voxel_coords[2]))
    maxyidx = int(np.max(mask_voxel_coords[2])) + 1
    return [[minzidx, maxzidx], [minxidx, maxxidx], [minyidx, maxyidx]]


def crop_to_bbox(image, bbox):
    assert len(image.shape) == 3, "only supports 3d images"
    resizer = (slice(bbox[0][0], bbox[0][1]), slice(bbox[1][0], bbox[1][1]), slice(bbox[2][0], bbox[2][1]))
    return image[resizer]


def get_case_identifier(case):
    case_identifier = case[0].split(os.sep)[-1].split(".nii.gz")[0][:-5]
    #case_identifier = case[0].split("/")[-1].split(".nii.gz")[0][:-5]
    return case_identifier


def get_case_identifier_from_npz(case):
    case_identifier = case.split(os.sep)[-1][:-4]
    return case_identifier


def load_case_from_list_of_files(data_files, seg_file=None, info_dict=None):
    assert isinstance(data_files, list) or isinstance(data_files, tuple), "case must be either a list or a tuple"
    properties = OrderedDict()
    data_itk = [sitk.ReadImage(f) for f in data_files]

    properties["original_size_of_raw_data"] = np.array(data_itk[0].GetSize())[[2, 1, 0]]
    properties["original_spacing"] = np.array(data_itk[0].GetSpacing())[[2, 1, 0]]
    properties["list_of_data_files"] = data_files
    properties["seg_file"] = seg_file

    properties['volume_per_voxel'] = float(np.prod(data_itk[0].GetSpacing(), dtype=np.float64))

    properties["itk_origin"] = data_itk[0].GetOrigin()
    properties["itk_spacing"] = data_itk[0].GetSpacing()
    properties["itk_direction"] = data_itk[0].GetDirection()

    if info_dict is not None:
        properties.update(info_dict)

    data_npy = np.vstack([sitk.GetArrayFromImage(d)[None] for d in data_itk])
    if seg_file is not None:
        seg_itk = sitk.ReadImage(seg_file)
        seg_npy = sitk.GetArrayFromImage(seg_itk)[None].astype(np.float32)
    else:
        seg_npy = None

    return data_npy.astype(np.float32), seg_npy, properties


def crop_to_nonzero(data, seg=None, nonzero_label=-1):
    """

    :param data:
    :param seg:
    :param nonzero_label: this will be written into the segmentation map
    :return:
    """
    nonzero_mask = create_nonzero_mask(data)
    bbox = get_bbox_from_mask(nonzero_mask, 0)

    cropped_data = []
    for c in range(data.shape[0]):
        cropped = crop_to_bbox(data[c], bbox)
        cropped_data.append(cropped[None])
    data = np.vstack(cropped_data)

    if seg is not None:
        cropped_seg = []
        for c in range(seg.shape[0]):
            cropped = crop_to_bbox(seg[c], bbox)
            cropped_seg.append(cropped[None])
        seg = np.vstack(cropped_seg)

    nonzero_mask = crop_to_bbox(nonzero_mask, bbox)[None]
    if seg is not None:
        seg[(seg == 0) & (nonzero_mask == 0)] = nonzero_label
    else:
        nonzero_mask = nonzero_mask.astype(int)
        nonzero_mask[nonzero_mask == 0] = nonzero_label
        nonzero_mask[nonzero_mask > 0] = 0
        seg = nonzero_mask
    return data, seg, bbox


def get_patient_identifiers_from_cropped_files(folder):
    out = [i.split(os.sep)[-1][:-4] for i in subfiles(folder, join=True, suffix=".npz")]
    return out


class ImageCropper(object):
    def __init__(self, num_threads, output_folder=None):
        """
        This one finds a mask of nonzero elements (must be nonzero in all modalities) and crops the image to that mask.
        In the case of BRaTS and ISLES data this results in a significant reduction in image size
        :param num_threads:
        :param output_folder: whete to store the cropped data
        :param list_of_files:
        """
        self.output_folder = output_folder
        self.num_threads = num_threads

        if self.output_folder is not None:
            maybe_mkdir_p(self.output_folder)

    @staticmethod
    def crop(data, properties, seg=None):
        shape_before = data.shape
        data, seg, bbox = crop_to_nonzero(data, seg, nonzero_label=-1)
        shape_after = data.shape
        print("before crop:", shape_before, "after crop:", shape_after, "spacing:",
              np.array(properties["original_spacing"]), "\n")

        properties["crop_bbox"] = bbox
        properties['classes'] = np.unique(seg)
        seg[seg < -1] = 0
        properties["size_after_cropping"] = data[0].shape
        return data, seg, properties

    @staticmethod
    def crop_from_list_of_files(data_files, seg_file=None, info_dict=None):
        data, seg, properties = load_case_from_list_of_files(data_files, seg_file, info_dict)
        return ImageCropper.crop(data, properties, seg)

    def load_crop_save(self, case, case_identifier, overwrite_existing=False, info_dict=None):
        # case = [[img_path, label_path], ...]
        # case_identifier = [case_identifier, ...]
        # info_dict = [dict, ...]

        data_list = []
        property_list = []
        seg_list = []

        for i in range(len(case)):
            current_case_identifier = case_identifier[i]
            current_case = case[i]
            current_info_dict = info_dict[i]
            print(current_case_identifier)
                
            data, seg, properties = self.crop_from_list_of_files(current_case[:-1], current_case[-1], current_info_dict)
            data_list.append(data)
            property_list.append(properties)
            seg_list.append(seg)

        array_bbox_list = []
        for i in range(len(data_list)):
            crop_bbox = property_list[i]['crop_bbox']
            array_bbox = np.stack([np.array(crop_bbox[i]) for i in range(len(crop_bbox))])
            array_bbox_list.append(array_bbox)
        array_bbox_list = np.stack(array_bbox_list)
        max_after = np.max(array_bbox_list[:, :, 1], axis=0)[None]
        min_before = np.min(array_bbox_list[:, :, 0], axis=0)[None]
        pad_after = max_after - array_bbox_list[:, :, 1]
        pad_before = array_bbox_list[:, :, 0] - min_before

        new_crop_bbox = np.stack([min_before[0], max_after[0]], axis=-1)
        new_crop_bbox = [list(x) for x in new_crop_bbox]

        for i in range(len(data_list)):
            current_case_identifier = case_identifier[i]

            current_padding = np.stack([pad_before[i], pad_after[i]], axis=-1)
            current_padding = np.concatenate([np.zeros(shape=(1, 2)).astype(int), current_padding])

            current_data = data_list[i]
            current_seg = seg_list[i]
            if 'Quorum' not in case[i][0]:
                current_data = np.pad(current_data, current_padding)
                current_seg = np.pad(current_seg, current_padding)

            current_property = property_list[i]
            current_property['size_after_cropping'] = current_data[0].shape
            current_property['crop_bbox'] = new_crop_bbox

            if overwrite_existing \
                    or (not os.path.isfile(os.path.join(self.output_folder, "%s.npz" % current_case_identifier))
                        or not os.path.isfile(os.path.join(self.output_folder, "%s.pkl" % current_case_identifier))):

                all_data = np.vstack((current_data, current_seg))
                np.savez_compressed(os.path.join(self.output_folder, "%s.npz" % current_case_identifier), data=all_data)
                with open(os.path.join(self.output_folder, "%s.pkl" % current_case_identifier), 'wb') as f:
                    pickle.dump(current_property, f)


    
    def load_crop_save_unlabeled(self, case, case_identifier, overwrite_existing=False, info_dict=None):
        # case = [[img_path, label_path], ...]
        # case_identifier = [case_identifier, ...]
        # info_dict = [dict, ...]

        data_list = []
        property_list = []

        for i in range(len(case)):
            current_case_identifier = case_identifier[i]
            current_case = case[i]
            current_info_dict = info_dict[i]
            print(current_case_identifier)
                
            data, seg, properties = self.crop_from_list_of_files(current_case, None, current_info_dict)
            data_list.append(data)
            property_list.append(properties)

        array_bbox_list = []
        for i in range(len(data_list)):
            crop_bbox = property_list[i]['crop_bbox']
            array_bbox = np.stack([np.array(crop_bbox[i]) for i in range(len(crop_bbox))])
            array_bbox_list.append(array_bbox)
        array_bbox_list = np.stack(array_bbox_list)
        max_after = np.max(array_bbox_list[:, :, 1], axis=0)[None]
        min_before = np.min(array_bbox_list[:, :, 0], axis=0)[None]
        pad_after = max_after - array_bbox_list[:, :, 1]
        pad_before = array_bbox_list[:, :, 0] - min_before

        new_crop_bbox = np.stack([min_before[0], max_after[0]], axis=-1)
        new_crop_bbox = [list(x) for x in new_crop_bbox]

        for i in range(len(data_list)):
            current_case_identifier = case_identifier[i]

            current_padding = np.stack([pad_before[i], pad_after[i]], axis=-1)
            current_padding = np.concatenate([np.zeros(shape=(1, 2)).astype(int), current_padding])
            current_data = np.pad(data_list[i], current_padding)

            current_property = property_list[i]
            current_property['size_after_cropping'] = current_data[0].shape
            current_property['crop_bbox'] = new_crop_bbox

            if overwrite_existing \
                    or (not os.path.isfile(os.path.join(self.output_folder, "%s.npz" % current_case_identifier))
                        or not os.path.isfile(os.path.join(self.output_folder, "%s.pkl" % current_case_identifier))):

                all_data = current_data
                np.savez_compressed(os.path.join(self.output_folder, "%s.npz" % current_case_identifier), data=all_data)
                with open(os.path.join(self.output_folder, "%s.pkl" % current_case_identifier), 'wb') as f:
                    pickle.dump(current_property, f)

    def get_list_of_cropped_files(self):
        return subfiles(self.output_folder, join=True, suffix=".npz")

    def get_patient_identifiers_from_cropped_files(self):
        return [i.split(os.sep)[-1][:-4] for i in self.get_list_of_cropped_files()]

    def run_cropping(self, list_of_files, overwrite_existing=False, output_folder=None, info_list=None):
        # list_of_files = patient_list -> patient_volume_list -> [img, label]
        # info_list = patient_list -> dict
        """
        also copied ground truth nifti segmentation into the preprocessed folder so that we can use them for evaluation
        on the cluster
        :param list_of_files: list of list of files [[PATIENTID_TIMESTEP_0000.nii.gz], [PATIENTID_TIMESTEP_0000.nii.gz]]
        :param overwrite_existing:
        :param output_folder:
        :return:
        """
        if output_folder is not None:
            self.output_folder = output_folder

        output_folder_gt = os.path.join(self.output_folder, "gt_segmentations")
        maybe_mkdir_p(output_folder_gt)
        list_of_args = []
        for p, patient_info in zip(list_of_files, info_list):
            case_identifier_list = []
            for case in p:
                case_identifier = get_case_identifier(case)
                case_identifier_list.append(case_identifier)
                if case[-1] is not None:
                    shutil.copy(case[-1], output_folder_gt)
            list_of_args.append((p, case_identifier_list, overwrite_existing, patient_info))

        #for j, case in enumerate(list_of_files):
        #    case_identifier = get_case_identifier(case)
        #    if info_list is not None:
        #        list_of_args.append((case, case_identifier, overwrite_existing, info_list[j]))
        #    else:
        #        list_of_args.append((case, case_identifier, overwrite_existing))

        p = Pool(self.num_threads)
        p.starmap(self.load_crop_save, list_of_args)
        p.close()
        p.join()
    
    def run_cropping_unlabeled(self, list_of_files, overwrite_existing=False, output_folder=None, info_list=None):
        """
        also copied ground truth nifti segmentation into the preprocessed folder so that we can use them for evaluation
        on the cluster
        :param list_of_files: list of list of files [[PATIENTID_TIMESTEP_0000.nii.gz], [PATIENTID_TIMESTEP_0000.nii.gz]]
        :param overwrite_existing:
        :param output_folder:
        :return:
        """
        if output_folder is not None:
            self.output_folder = output_folder

        list_of_args = []
        for p, patient_info in zip(list_of_files, info_list):
            case_identifier_list = []
            for case in p:
                case_identifier = get_case_identifier(case)
                case_identifier_list.append(case_identifier)
            list_of_args.append((p, case_identifier_list, overwrite_existing, patient_info))

        #for j, case in enumerate(list_of_files):
        #    case_identifier = get_case_identifier(case)
        #    if info_list is not None:
        #        list_of_args.append((case, case_identifier, overwrite_existing, info_list[j]))
        #    else:
        #        list_of_args.append((case, case_identifier, overwrite_existing))

        p = Pool(self.num_threads)
        p.starmap(self.load_crop_save_unlabeled, list_of_args)
        #p.starmap(self.load_crop_save_unlabeled, list_of_args)
        p.close()
        p.join()

    def load_properties(self, case_identifier):
        with open(os.path.join(self.output_folder, "%s.pkl" % case_identifier), 'rb') as f:
            properties = pickle.load(f)
        return properties

    def save_properties(self, case_identifier, properties):
        with open(os.path.join(self.output_folder, "%s.pkl" % case_identifier), 'wb') as f:
            pickle.dump(properties, f)
