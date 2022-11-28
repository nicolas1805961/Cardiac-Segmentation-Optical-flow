import torch
from torchvision.ops import masks_to_boxes
from torchvision.transforms.functional import affine

class Processor(object):
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
    
    #def get_mean_centroid(self, data):
    #    mean_centroid_list = []
    #    for b in range(len(data)):
    #        current_data = data[b]
    #        current_flattened_data = torch.flatten(current_data, start_dim=1)
    #        mask = torch.count_nonzero(current_flattened_data, dim=-1) != 0
    #        current_data = current_data[mask]
    #        coords = masks_to_boxes(current_data)
    #        x = coords[:, 0] + ((coords[:, 2] - coords[:, 0]) / 2)
    #        y = coords[:, 1] + ((coords[:, 3] - coords[:, 1]) / 2)
    #        coords = torch.stack([x, y], dim=-1)
    #        mean_centroid = coords.mean(dim=0)
    #        mean_centroid_list.append(mean_centroid)
    #    return torch.stack(mean_centroid_list, dim=0).int()
    
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

        out = {'crop_indices': [x_low, x_high, y_low, y_high], 'padding_need': (pad_left, pad_right, pad_top, pad_bottom)}

        return out
    
    def crop_data(self, volume, centroids):
        data_list = []
        padding_list = []
        for b in range(volume.shape[0]):
            data = volume[b]
            centroid = centroids[b]
            payload = self.adjust_cropping_window(centroid)
            padding_list.append(payload['padding_need'])
            coords = payload['crop_indices']
            data = data[:, :, coords[2]:coords[3], coords[0]:coords[1]]
            data_list.append(data)
        return torch.stack(data_list, dim=0), padding_list
    
    def discretize(self, data_list):
        out_list = []
        for data in data_list:
            out = self.cropping_network(data)['pred']
            out = torch.softmax(out, dim=1)
            out = torch.argmax(out, dim=1)
            out_list.append(out)
        return torch.stack(out_list, dim=1)
    
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