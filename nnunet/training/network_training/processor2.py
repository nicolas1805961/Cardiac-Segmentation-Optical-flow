import torch
from torchvision.ops import masks_to_boxes
from torchvision.transforms.functional import affine
import matplotlib.pyplot as plt
import matplotlib
from torch.nn.functional import pad
from monai.transforms import NormalizeIntensity
from skimage import morphology
from skimage import measure
from skimage import filters
import numpy as np
from scipy.ndimage import distance_transform_edt
import logging
import cv2 as cv

logging.basicConfig(level=logging.DEBUG, filename="logfile.txt", filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")

class Processor2(object):
    def __init__(self, crop_size, image_size, cropping_network) -> None:
        self.crop_size = crop_size
        self.image_size = image_size
        self.cropping_network = cropping_network
    
    def get_coords(self, data):
        coords = masks_to_boxes(data.unsqueeze(0))
        x = coords[:, 0] + ((coords[:, 2] - coords[:, 0]) / 2)
        y = coords[:, 1] + ((coords[:, 3] - coords[:, 1]) / 2)
        coords = torch.cat([x, y])
        return coords
    
    def get_mask_values(self, data):
        for values in [[3], [2, 3], [1, 2, 3]]:
            values = torch.tensor(values, device=data.device)
            current_data = torch.clone(data)
            mask = torch.isin(current_data, values)
            if torch.any(mask):
                return values

    def get_masked_data(self, data):
        out_data = torch.clone(data)
        mask_values = self.get_mask_values(out_data)
        mask = torch.isin(out_data, mask_values)
        out_data[~mask] = 0
        out_data[mask] = 1
        return out_data, mask_values
    
    def get_fixed_info(self, data, idx):
        lv_centroid_list = []
        global_centroid_list = []
        for b in range(len(data)): 
            current_data = data[b, idx[b]]
            current_data_lv, mask_values = self.get_masked_data(current_data)
            current_data_global = torch.clone(current_data)
            current_data_global[current_data_global > 0] = 1
            # TODO: handle zero slice
            lv_coords = self.get_coords(current_data_lv)
            global_coords = self.get_coords(current_data_global)
            lv_centroid_list.append(lv_coords)
            global_centroid_list.append(global_coords)
        return torch.stack(lv_centroid_list, dim=0).int(), torch.stack(global_centroid_list, dim=0).int(), mask_values
    
    def get_translation(self, data, fixed_centroids, mask_values):
        mask = torch.isin(data, mask_values)
        data[~mask] = 0
        data[mask] = 1
        translation_list = []
        for b in range(len(data)):
            current_fixed = fixed_centroids[b]
            current_video = data[b]
            full_video = torch.full(size=(len(current_video), 2), fill_value=float('nan'), dtype=torch.float32, device=data.device)
            null_mask = torch.all(torch.flatten(current_video, start_dim=1) == 0, dim=-1)
            pos_indices = torch.where(~null_mask)[0]
            current_video = current_video[pos_indices]
            coords = masks_to_boxes(current_video)
            x = coords[:, 0] + ((coords[:, 2] - coords[:, 0]) / 2)
            y = coords[:, 1] + ((coords[:, 3] - coords[:, 1]) / 2)
            x = current_fixed[0] - x
            y = current_fixed[1] - y
            coords = torch.stack([x, y], dim=-1)
            full_video[pos_indices] = coords
            first, last = pos_indices[0], pos_indices[-1]
            full_video[:first] = full_video[first]
            full_video[last + 1:] = full_video[last]
            translation_list.append(full_video)
        return torch.stack(translation_list, dim=0).int()
    
    def translate(self, data, translations):
        batch_list = []
        for b in range(len(data)):
            current_video = data[b]
            video_list = []
            for t in range(len(current_video)):
                current_translation = translations[b, t].tolist()
                translated = affine(current_video[t], scale=1, angle=0, shear=0, translate=current_translation)
                video_list.append(translated)
            current_video = torch.stack(video_list, dim=0)
            batch_list.append(current_video)
        return torch.stack(batch_list, dim=0)
    
    def get_mean_centroid(self, data):
        T, H, W = data.shape
        data[data > 0] = 1
        centroid_list = []
        for t in range(len(data)):
            current_data_time = data[t]
            if torch.count_nonzero(current_data_time) == 0:
                centroid = torch.tensor([H / 2, W / 2], device=data.device).view(1, 2)
            else:
                coords = masks_to_boxes(current_data_time.unsqueeze(0))
                x = coords[:, 0] + ((coords[:, 2] - coords[:, 0]) / 2)
                y = coords[:, 1] + ((coords[:, 3] - coords[:, 1]) / 2)
                centroid = torch.stack([x, y], dim=-1)
            centroid_list.append(centroid)
        centroid_list = torch.cat(centroid_list, dim=0)
        mean_centroid = centroid_list.mean(0)
        return mean_centroid.int()
    
    def adjust_cropping_window(self, centroid):
        half_crop_size = self.crop_size // 2
        x_low = max(0, centroid[0] - half_crop_size)
        x_high = min(self.image_size, centroid[0] + half_crop_size)
        y_low = max(0, centroid[1] - half_crop_size)
        y_high = min(self.image_size, centroid[1] + half_crop_size)

        if x_low == 0:
            x_high = self.crop_size
        if x_high == self.image_size:
            x_low = self.image_size - self.crop_size
        if y_low == 0:
            y_high = self.crop_size
        if y_high == self.image_size:
            y_low = self.image_size - self.crop_size

        pad_left = x_low
        pad_right = self.image_size - x_high
        pad_top = y_low
        pad_bottom = self.image_size - y_high

        out = {'crop_indices': [x_low, x_high, y_low, y_high], 'padding_need': torch.tensor([pad_left, pad_right, pad_top, pad_bottom])}

        return out
    
    def crop_data(self, volume, centroid):
        payload = self.adjust_cropping_window(centroid)
        coords = payload['crop_indices']
        volume = volume[:, :, coords[2]:coords[3], coords[0]:coords[1]]
        return volume, payload['padding_need']
    
    def discretize(self, data):
        out_list = []
        for i in range(len(data)):
            current_data = data[i][None] # B(1), 1, H, W
            if torch.count_nonzero(current_data) == 0:
                softmaxed = torch.zeros_like(current_data)
            else:
                #current_data = NormalizeIntensity()(current_data)
                softmaxed = self.cropping_network(current_data)['pred']
                softmaxed = torch.softmax(softmaxed, dim=1)
            out = torch.argmax(softmaxed, dim=1).squeeze(0) # H, W

            #matplotlib.use('QtAgg')
            #fig, ax = plt.subplots(1, 2)
            #ax[0].imshow(current_data.cpu()[0, 0], cmap='gray')
            #ax[1].imshow(out.cpu(), cmap='gray')
            #plt.show()
            #plt.waitforbuttonpress()
            #plt.close(fig)

            out_list.append(out)
        out_list = torch.stack(out_list, dim=0) # T, H, W
        return out_list
    
    def uncrop(self, output, padding_need, translation_dists):
        output_volume = torch.stack(output['labeled_data'], dim=1)
        assert len(output_volume) == len(padding_need) == len(translation_dists)
        out_list = []
        for b in range(len(output_volume)):
            padded = torch.nn.functional.pad(output_volume[b], pad=padding_need[b])
            video_list = []
            for t in range(len(padded)):
                current_translation = (-translation_dists[b, t]).tolist()
                translated = affine(padded[t], scale=1, angle=0, shear=0, translate=current_translation)
                video_list.append(translated)
            video_list = torch.stack(video_list, dim=0)
            out_list.append(video_list)
        out_volume = torch.stack(out_list, dim=0)
        return out_volume
    
    def uncrop_no_registration(self, output, padding_need):
        assert len(output) == len(padding_need)
        out_list = []
        for b in range(len(output)):
            current_padding_need = tuple(padding_need[b].tolist())
            padded = torch.nn.functional.pad(output[b], pad=current_padding_need)
            out_list.append(padded)
        out_volume = torch.stack(out_list, dim=0)
        return out_volume

    def preprocess(self, data_list, idx):
        #matplotlib.use('QtAgg')
        temp_volume = self.discretize(data_list)

        #fig, ax = plt.subplots(1, 2)
        #print(data_dict['registered_idx'])
        #print(torch.unique(temp_volume[0, 0]))
        #print(torch.unique(temp_volume[0, 1]))
        #ax[0].imshow(temp_volume[0, 0].cpu(), cmap='gray')
        #ax[1].imshow(temp_volume[0, 1].cpu(), cmap='gray')
        #plt.show()

        lv_centroid, global_centroid, mask_values = self.get_fixed_info(temp_volume, idx=idx)
        translation_dists = self.get_translation(temp_volume, lv_centroid, mask_values)
        data_volume = torch.stack(data_list, dim=1)
        translated = self.translate(data_volume, translation_dists)
        cropped_volume, padding_need = self.crop_data(translated, global_centroid)

        #fig, ax = plt.subplots(self.batch_size, 6)
        #for i in range(self.batch_size):
        #    ax[0].imshow(data_volume[i, 0, 0].cpu(), cmap='gray')
        #    ax[1].imshow(data_volume[i, 1, 0].cpu(), cmap='gray')
        #    ax[2].imshow(translated[i, 0, 0].cpu(), cmap='gray')
        #    ax[3].imshow(translated[i, 1, 0].cpu(), cmap='gray')
        #    ax[4].imshow(cropped_volume[i, 0, 0].cpu(), cmap='gray')
        #    ax[5].imshow(cropped_volume[i, 1, 0].cpu(), cmap='gray')
        ##ax[2].imshow(data_volume[0, 0, 0].cpu(), cmap='gray')
        ##ax[0].scatter(mean_centroids[0, 0].cpu(), mean_centroids[0, 1].cpu(), color="red") # plotting single point
        ##ax[1].scatter(mean_centroids[0, 0].cpu(), mean_centroids[0, 1].cpu(), color="red") # plotting single point
        #plt.show()

        assert cropped_volume.shape[-1] == self.crop_size, print(cropped_volume.shape[-1])
        network_input = {'labeled_data': [cropped_volume[:, i] for i in range(cropped_volume.shape[1])]}
        return network_input, padding_need, translation_dists
    
    def crop_and_pad(self, data, mean_centroid):
        '''data: T, 1, H, W'''
        cropped_volume, padding_need = self.crop_data(data, mean_centroid) # T, 1, 128, 128

        #assert torch.all(torch.isfinite(cropped_volume))
        assert cropped_volume.shape[-1] == self.crop_size, print(cropped_volume.shape[-1])

        return cropped_volume, padding_need

    def preprocess_no_registration(self, data):
        '''data: T, 1, H, W'''

        temp_volume = self.discretize(data) # T, H, W

        one_hot_volume = torch.nn.functional.one_hot(temp_volume.long(), num_classes=4).permute(0, 3, 1, 2).contiguous().float().cpu().numpy() # T, 4, H, W
        for i in range(1, 4):
            for j in range(len(one_hot_volume)):
                current = one_hot_volume[j, i]

                current_labeled = measure.label(current)

                info_list = []
                for idx, region in enumerate(measure.regionprops(current_labeled)):
                    info_list.append((idx + 1, region.area))
                info_list = sorted(info_list, key=lambda x: x[1])

                for region in info_list[:-1]:
                    current[current_labeled == region[0]] = 0
                
                one_hot_volume[j, i] = current

        one_hot_volume = torch.from_numpy(one_hot_volume)
        temp_volume = torch.argmax(one_hot_volume, dim=1)

        mean_centroid = self.get_mean_centroid(torch.clone(temp_volume)) # 2
        return mean_centroid, temp_volume
    
    
    def get_strain_mask(self, data, path):
        logging.info(path)
        out_block = []
        with torch.no_grad():
            for t in range(len(data)):
                label = data[t][None, None]
                label = torch.nn.functional.one_hot(label[:, 0].long(), num_classes=4).permute(0, 3, 1, 2).contiguous().float()[:, 1:3].cpu().numpy()
                
                myo = label[0, 1]             
                rv = label[0, 0]

                #fig, ax = plt.subplots(1, 1)
                #ax.imshow(myo, cmap='gray')
                #plt.show()
                #plt.waitforbuttonpress()
                #plt.close(fig)

                myo_labeled = measure.label(myo)
                if len(np.unique(myo_labeled)) > 2:
                    logging.info('bugggggggggggggggggggg11111111111111')
                    #fig, ax = plt.subplots(1, 1)
                    #ax.imshow(myo, cmap='gray')
                    #plt.show()
                    #plt.waitforbuttonpress()
                    #plt.close(fig)

                    info_list = []
                    for idx, region in enumerate(measure.regionprops(myo_labeled)):
                        info_list.append((idx + 1, region.area))
                    info_list = sorted(info_list, key=lambda x: x[1])

                    logging.info(info_list)

                    for region in info_list[:-1]:
                        myo[myo_labeled == region[0]] = 0
                    
                    #fig, ax = plt.subplots(1, 1)
                    #ax.imshow(myo, cmap='gray')
                    #plt.show()
                    #plt.waitforbuttonpress()
                    #plt.close(fig)

                rv_labeled = measure.label(rv)
                if len(np.unique(rv_labeled)) > 2:
                    logging.info('bugggggggggggggggggggg2222222222222222')

                    #fig, ax = plt.subplots(1, 2)
                    #ax[0].imshow(data[t], cmap='gray')
                    #ax[1].imshow(rv_labeled, cmap='gray')
                    #plt.show()
                    #while not plt.waitforbuttonpress(): pass
                    #plt.close(fig)

                    info_list = []
                    for idx, region in enumerate(measure.regionprops(rv_labeled)):
                        info_list.append((idx + 1, region.area))
                    info_list = sorted(info_list, key=lambda x: x[1])

                    logging.info(info_list)

                    for region in info_list[:-1]:
                        rv[rv_labeled == region[0]] = 0

                    #fig, ax = plt.subplots(1, 1)
                    #ax.imshow(rv, cmap='gray')
                    #plt.show()
                    #while not plt.waitforbuttonpress(): pass
                    #plt.close(fig)

                dilated_image = morphology.dilation(myo, morphology.square(3))
                eroded_image = morphology.erosion(myo, morphology.square(3))
                morphological_gradient_myo = (dilated_image - eroded_image).astype(int)

                dilated_image = morphology.dilation(rv, morphology.square(3))
                eroded_image = morphology.erosion(rv, morphology.square(3))
                morphological_gradient_rv = (dilated_image - eroded_image).astype(int)

                mask = np.logical_or(morphological_gradient_myo, morphological_gradient_rv)

                #fig, ax = plt.subplots(1, 1)
                #ax.imshow(mask, cmap='gray')
                #plt.show()
                #plt.waitforbuttonpress()
                #plt.close(fig)

                not_one_hot = distance_transform_edt(1 - mask)
                not_one_hot = (not_one_hot - not_one_hot.min()) / (not_one_hot.max() - not_one_hot.min())
                assert not_one_hot.max() == 1.0
                assert not_one_hot.min() == 0.0
                not_one_hot = 1 - not_one_hot

                one_hot = np.zeros(shape=(3,) + morphological_gradient_myo.shape)
                one_hot[0] = morphological_gradient_rv

                morphological_gradient_myo_labeled = measure.label(morphological_gradient_myo, connectivity=1)
                
                #if len(np.unique(morphological_gradient_myo_labeled)) < 3:
                #    fig, ax = plt.subplots(1, 2)
                #    ax[0].imshow(myo, cmap='gray')
                #    ax[1].imshow(morphological_gradient_myo_labeled, cmap='gray')
                #    plt.show()
                #    plt.waitforbuttonpress()
                #    plt.close(fig)

                if len(np.unique(morphological_gradient_myo_labeled)) < 3:
                    
                    logging.info('******************************************')

                    #fig, ax = plt.subplots(1, 1)
                    #ax.imshow(morphological_gradient_myo_labeled, cmap='gray')
                    #plt.show()
                    #plt.waitforbuttonpress()
                    #plt.close(fig)

                    size = 3
                    dilated_image = morphology.closing(morphological_gradient_myo_labeled, morphology.square(size))
                    dilated_labeled = measure.label(dilated_image, connectivity=1, background=-1)

                    #fig, ax = plt.subplots(1, 1)
                    #ax.imshow(dilated_image, cmap='gray')
                    #plt.show()
                    #plt.waitforbuttonpress()
                    #plt.close(fig)

                    flag = False
                    logging.info(size)
                    while len(measure.regionprops(dilated_labeled)) != 3:
                        if size > 100:
                            flag = True
                            break
                        size += 2
                        logging.info(size)
                        dilated_image = morphology.closing(morphological_gradient_myo_labeled, morphology.square(size))
                        dilated_labeled = measure.label(dilated_image, connectivity=1, background=-1)

                        #fig, ax = plt.subplots(1, 1)
                        #ax.imshow(dilated_image, cmap='gray')
                        #plt.show()
                        #plt.waitforbuttonpress()
                        #plt.close(fig)

                    if flag:
                        logging.info('99999999999999999999999999999999')
                        one_hot[1] = myo == 1
                        one_hot[2] = myo == 1
                    else:

                        i1 = morphology.dilation(dilated_image, morphology.square(3))
                        i2 = morphology.erosion(dilated_image, morphology.square(3))
                        i3 = (i1 - i2).astype(int)

                        morphological_gradient_myo_labeled = measure.label(i3, connectivity=1)

                        #fig, ax = plt.subplots(1, 1)
                        #ax.imshow(morphological_gradient_myo_labeled, cmap='gray')
                        #plt.show()
                        #plt.waitforbuttonpress()
                        #plt.close(fig)

                        if len(np.unique(morphological_gradient_myo_labeled)) < 3:

                            logging.info('+-+-+-++-+-+-+-+-+-+-+-+-+-+--++--++-+--++--+--+')

                            #fig, ax = plt.subplots(1, 2)
                            #ax[0].imshow(dilated_image, cmap='gray')
                            #ax[1].imshow(myo, cmap='gray')
                            #plt.show()
                            #plt.waitforbuttonpress()
                            #plt.close(fig)

                            one_hot[1] = myo == 1
                            one_hot[2] = myo == 1

                            #fig, ax = plt.subplots(1, 3)
                            #ax[0].imshow(one_hot[1], cmap='gray')
                            #ax[1].imshow(one_hot[2], cmap='gray')
                            #plt.show()
                            #plt.waitforbuttonpress()
                            #plt.close(fig)
                        else:
                            one_hot[1] = morphological_gradient_myo_labeled == 1
                            one_hot[2] = morphological_gradient_myo_labeled == 2
                else:
                    one_hot[1] = morphological_gradient_myo_labeled == 1
                    one_hot[2] = morphological_gradient_myo_labeled == 2

                #if np.count_nonzero(one_hot[1]) == 0 or np.count_nonzero(one_hot[2]) == 0:
                #    fig, ax = plt.subplots(1, 2)
                #    ax[0].imshow(one_hot[1], cmap='gray')
                #    ax[1].imshow(one_hot[2], cmap='gray')
                #    plt.show()
                #    plt.waitforbuttonpress()
                #    plt.close(fig)
                    

                for i in range(3):
                    distance = distance_transform_edt(1 - one_hot[i])
                    distance = (distance - distance.min()) / (distance.max() - distance.min())
                    assert distance.max() == 1.0
                    assert distance.min() == 0.0
                    distance = 1 - distance
                    one_hot[i] = distance

                one_hot = np.concatenate([one_hot, not_one_hot[None]], axis=0)

                #fig, ax = plt.subplots(1, 4)
                #ax[0].imshow(one_hot[0], cmap='hot')
                #ax[1].imshow(one_hot[1], cmap='hot')
                #ax[2].imshow(one_hot[2], cmap='hot')
                #ax[3].imshow(not_one_hot, cmap='hot')
                #plt.show()
                #plt.waitforbuttonpress()
                #plt.close(fig)


                out_block.append(one_hot)
        
        out_block = np.stack(out_block, axis=-1)
        return out_block
    

    
    def get_strain_mask_3(self, data, path):
        logging.info(path)
        out_block = []
        with torch.no_grad():
            for t in range(len(data)):
                label = data[t][None, None]
                label = torch.nn.functional.one_hot(label[:, 0].long(), num_classes=4).permute(0, 3, 1, 2).contiguous().float().cpu().numpy()
                
                myo = label[0, 2]             
                rv = label[0, 1]
                lv = label[0, 3]

                #fig, ax = plt.subplots(1, 1)
                #ax.imshow(lv, cmap='gray')
                #plt.show()
                #plt.waitforbuttonpress()
                #plt.close(fig)

                myo_labeled = measure.label(myo)
                if len(np.unique(myo_labeled)) > 2:
                    logging.info('bugggggggggggggggggggg11111111111111')
                    #fig, ax = plt.subplots(1, 1)
                    #ax.imshow(myo, cmap='gray')
                    #plt.show()
                    #plt.waitforbuttonpress()
                    #plt.close(fig)

                    info_list = []
                    for idx, region in enumerate(measure.regionprops(myo_labeled)):
                        info_list.append((idx + 1, region.area))
                    info_list = sorted(info_list, key=lambda x: x[1])

                    logging.info(info_list)

                    for region in info_list[:-1]:
                        myo[myo_labeled == region[0]] = 0
                    
                    #fig, ax = plt.subplots(1, 1)
                    #ax.imshow(myo, cmap='gray')
                    #plt.show()
                    #plt.waitforbuttonpress()
                    #plt.close(fig)

                rv_labeled = measure.label(rv)
                if len(np.unique(rv_labeled)) > 2:
                    logging.info('bugggggggggggggggggggg2222222222222222')

                    #fig, ax = plt.subplots(1, 2)
                    #ax[0].imshow(data[t], cmap='gray')
                    #ax[1].imshow(rv_labeled, cmap='gray')
                    #plt.show()
                    #while not plt.waitforbuttonpress(): pass
                    #plt.close(fig)

                    info_list = []
                    for idx, region in enumerate(measure.regionprops(rv_labeled)):
                        info_list.append((idx + 1, region.area))
                    info_list = sorted(info_list, key=lambda x: x[1])

                    logging.info(info_list)

                    for region in info_list[:-1]:
                        rv[rv_labeled == region[0]] = 0

                    #fig, ax = plt.subplots(1, 1)
                    #ax.imshow(rv, cmap='gray')
                    #plt.show()
                    #while not plt.waitforbuttonpress(): pass
                    #plt.close(fig)

                dilated_image = morphology.dilation(myo, morphology.square(3))
                eroded_image = morphology.erosion(myo, morphology.square(3))
                morphological_gradient_myo = (dilated_image - eroded_image).astype(int)

                dilated_image = morphology.dilation(rv, morphology.square(3))
                eroded_image = morphology.erosion(rv, morphology.square(3))
                morphological_gradient_rv = (dilated_image - eroded_image).astype(int)

                mask = np.logical_or(morphological_gradient_myo, morphological_gradient_rv)

                #fig, ax = plt.subplots(1, 1)
                #ax.imshow(mask, cmap='gray')
                #plt.show()
                #plt.waitforbuttonpress()
                #plt.close(fig)

                not_one_hot = distance_transform_edt(1 - mask)

                #not_one_hot = 4 * np.exp(-not_one_hot) / ((1 + np.exp(-not_one_hot))**2)
                #not_one_hot_2 = 1 / (1 + not_one_hot)

                #fig, ax = plt.subplots(1, 3)
                #ax[0].imshow(not_one_hot_1, cmap='hot', vmin=0.0, vmax=1.0)
                #ax[1].imshow(not_one_hot_2, cmap='hot', vmin=0.0, vmax=1.0)
                #ax[2].imshow(not_one_hot_1**0.01, cmap='hot', vmin=0.0, vmax=1.0)
                #plt.show()
                #while not plt.waitforbuttonpress(): pass
                #plt.close(fig)

                one_hot = np.zeros(shape=(3,) + morphological_gradient_myo.shape)
                one_hot[0] = morphological_gradient_rv
                
                epi = np.logical_or(lv, myo).astype(int)
                dilated_epi = morphology.dilation(epi, morphology.square(3))
                eroded_epi = morphology.erosion(epi, morphology.square(3))
                epi = (dilated_epi - eroded_epi).astype(int)
                one_hot[1] = epi

                dilated_endo = morphology.dilation(lv, morphology.square(3))
                eroded_endo = morphology.erosion(lv, morphology.square(3))
                endo = (dilated_endo - eroded_endo).astype(int)
                one_hot[2] = endo

                for i in range(3):
                    distance = distance_transform_edt(1 - one_hot[i])
                    #distance = 4 * np.exp(-distance) / ((1 + np.exp(-distance))**2)
                    #distance = 1 / (1 + distance)
                    one_hot[i] = distance

                one_hot = np.concatenate([one_hot, not_one_hot[None]], axis=0)

                #fig, ax = plt.subplots(1, 4)
                #ax[0].imshow(one_hot[0], cmap='hot', vmin=0.0, vmax=1.0)
                #ax[1].imshow(one_hot[1], cmap='hot', vmin=0.0, vmax=1.0)
                #ax[2].imshow(one_hot[2], cmap='hot', vmin=0.0, vmax=1.0)
                #ax[3].imshow(not_one_hot, cmap='hot', vmin=0.0, vmax=1.0)
                #plt.show()
                #plt.waitforbuttonpress()
                #plt.close(fig)

                assert np.count_nonzero(one_hot[0].astype(int)) > 0
                assert np.count_nonzero(one_hot[1].astype(int)) > 0
                assert np.count_nonzero(one_hot[2].astype(int)) > 0
                assert np.count_nonzero(one_hot[3].astype(int)) > 0

                out_block.append(one_hot)
        
        out_block = np.stack(out_block, axis=-1)
        return out_block