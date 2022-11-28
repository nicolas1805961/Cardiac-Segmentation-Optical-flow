import numpy as np
import cv2 as cv
import os
import torch
from matplotlib.colors import Normalize
from torch.nn.functional import interpolate
import matplotlib as mpl
from matplotlib.colors import ListedColormap


class Visualizer(object):
    def __init__(self, 
                unlabeled, 
                adversarial_loss, 
                affinity,
                middle_unlabeled,
                middle,
                registered_seg,
                writer):
        self.unlabeled = unlabeled
        self.middle_unlabeled = middle_unlabeled
        self.adversarial_loss = adversarial_loss
        self.affinity = affinity
        self.middle = middle
        self.registered_seg = registered_seg
        self.writer = writer
        self.eval_images = self.initialize_image_data()
    
    def reset(self):
        self.eval_images = self.initialize_image_data()

    def initialize_image_data(self):
        log_images_nb = 8
        eval_images = {'rec': None,
                        'df': None,
                        'seg': None}
        
        if self.unlabeled and self.adversarial_loss:
            eval_images['confidence'] = None
        if self.affinity:
            eval_images['affinity'] = None
        if self.middle:
            eval_images['sim'] = None
            eval_images['weights'] = None
            if self.registered_seg:
                eval_images['forward_motion'] = None
                eval_images['backward_motion'] = None
            if self.middle_unlabeled:
                eval_images['sim_unlabeled'] = None
                eval_images['pseudo_label'] = None
            #else:
            #    eval_images['sim'] = None

        for key in eval_images.keys():
            data = []
            scores = []
            if key == 'rec':
                payload = {'input': None,
                            'reconstruction': None}
                score = -1
                data.append(payload)
                scores.append(score)
            elif key == 'sim':
                for i in range(log_images_nb):
                    payload = {'input': None,
                                'middle_input': None,
                                'sim_1': None,
                                'sim_2': None}
                    score = 0
                    data.append(payload)
                    scores.append(score)
            elif key == 'sim_unlabeled':
                for i in range(log_images_nb):
                    payload = {'l1': None,
                                'l2': None,
                                'u1': None,
                                'sim_1': None,
                                'sim_2': None,
                                'sim_3': None,
                                }
                    score = 0
                    data.append(payload)
                    scores.append(score)
            elif key == 'df':
                for i in range(log_images_nb):
                    payload = {'input': None,
                                'gt_df': None,
                                'pred_df': None}
                    score = -1
                    data.append(payload)
                    scores.append(score)
            elif key == 'seg':
                for i in range(log_images_nb):
                    payload = {'input': None,
                                'pred': None,
                                'gt': None}
                    score = 0
                    data.append(payload)
                    scores.append(score)
            elif key == 'confidence':
                for i in range(log_images_nb):
                    payload = {'input': None,
                                'pred': None,
                                'confidence': None}
                    score = 0
                    data.append(payload)
                    scores.append(score)
            elif key == 'affinity':
                for i in range(log_images_nb):
                    payload = {'input': None,
                                'seg_aff': None,
                                'aff': None}
                    score = 0
                    data.append(payload)
                    scores.append(score)
            elif key == 'forward_motion':
                for i in range(log_images_nb):
                    payload = {'moving': None,
                                'registered': None,
                                'fixed': None,
                                'motion': None}
                    score = 0
                    data.append(payload)
                    scores.append(score)
            elif key == 'backward_motion':
                for i in range(log_images_nb):
                    payload = {'moving': None,
                                'registered': None,
                                'fixed': None,
                                'motion': None}
                    score = 0
                    data.append(payload)
                    scores.append(score)
            elif key == 'corr':
                payload = {'corr': None}
                score = 0
                data.append(payload)
                scores.append(score)
            elif key == 'pseudo_label':
                for i in range(log_images_nb):
                    payload = {'input': None,
                                'pred': None}
                    score = 0
                    data.append(payload)
                    scores.append(score)
            elif key == 'weights':
                for i in range(log_images_nb):
                    payload = {'w3': None,
                                'x1': None,
                                'x2': None}
                    score = 0
                    data.append(payload)
                    scores.append(score)
            eval_images[key] = np.stack([np.array(data), np.array(scores)], axis=0)
    
        return eval_images
    
    #def get_custom_colormap(self, colormap):
    #    # extract all colors from the .jet map
    #    cmaplist = [colormap(i) for i in range(colormap.N)]
#
    #    cmaplist[0] = (0, 0, 0, 1.0)
    #    cmaplist[1] = (1.0, 0, 0, 1.0)
    #    cmaplist[2] = (0, 1.0, 0, 1.0)
    #    cmaplist[3] = (0, 0, 1.0, 1.0)
#
    #    # create the new map
    #    cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, colormap.N)
#
    #    # define the bins and normalize
    #    bounds = np.linspace(0, 4, 5)
    #    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
#
    #    return cmap, norm

    def get_custom_colormap(self):
        vals = np.ones((4, 4))
        vals[0, :] = [0.0, 0.0, 0.0, 1.0]
        vals[1, :] = [1.0, 0.0, 0.0, 1.0]
        vals[2, :] = [0.0, 1.0, 0.0, 1.0]
        vals[3, :] = [0.0, 0.0, 1.0, 1.0]
        newcmp = ListedColormap(vals)
        bounds = np.linspace(0, 4, 5)
        norm = mpl.colors.BoundaryNorm(bounds, 4)
        return newcmp, norm
    
    def get_weights_ready(self, w, downscale):
        w = torch.mean(w, dim=1, keepdim=True)
        my_max = torch.max(torch.flatten(w, start_dim=2), dim=-1)[0].unsqueeze(-1).unsqueeze(-1)
        my_min = torch.min(torch.flatten(w, start_dim=2), dim=-1)[0].unsqueeze(-1).unsqueeze(-1)
        w = interpolate(input=w, scale_factor=downscale, mode='bicubic', antialias=True).squeeze()
        w = torch.clamp(w, my_min, my_max)
        return w

    def get_images_ready_for_display(self, image, colormap, vmin=None, vmax=None):
        if colormap is not None:
            if np.count_nonzero(image) > 0:
                image = Normalize(vmin=vmin, vmax=vmax)(image)
                #image = normalize_0_1(image)
            image = colormap(image)[:, :, :-1]
        image = cv.normalize(image, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX).astype(np.uint8)
        if len(image.shape) < 3:
            image = image[:, :, None]
        return image
        
    def get_seg_images_ready_for_display(self, image, colormap, norm):
        image = norm(image)
        image = colormap(image)[:, :, :-1]
        image = cv.normalize(image, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX).astype(np.uint8)
        if len(image.shape) < 3:
            image = image[:, :, None]
        return image

    def log_rec_images(self, epoch):
        input_image = self.eval_images['rec'][0, 0]['input']
        input_image = self.get_images_ready_for_display(input_image, colormap=None)

        reconstruction = self.eval_images['rec'][0, 0]['reconstruction']
        reconstruction = self.get_images_ready_for_display(reconstruction, colormap=None)

        self.writer.add_image(os.path.join('Reconstruction', 'input').replace('\\', '/'), input_image, epoch, dataformats='HWC')
        self.writer.add_image(os.path.join('Reconstruction', 'reconstruction').replace('\\', '/'), reconstruction, epoch, dataformats='HWC')

    def log_sim_images(self, colormap, epoch):
        input_list = [x['input'] for x in self.eval_images['sim'][0]]
        input_list = [self.get_images_ready_for_display(x, colormap=None) for x in input_list]
        input_list = np.stack(input_list, axis=0)

        middle_input_list = [x['middle_input'] for x in self.eval_images['sim'][0]]
        middle_input_list = [self.get_images_ready_for_display(x, colormap=None) for x in middle_input_list]
        middle_input_list = np.stack(middle_input_list, axis=0)

        sim_1_list = [x['sim_1'] for x in self.eval_images['sim'][0]]
        sim_1_list = [self.get_images_ready_for_display(x, colormap) for x in sim_1_list]
        sim_1_list = np.stack(sim_1_list, axis=0)

        sim_2_list = [x['sim_2'] for x in self.eval_images['sim'][0]]
        sim_2_list = [self.get_images_ready_for_display(x, colormap) for x in sim_2_list]
        sim_2_list = np.stack(sim_2_list, axis=0)

        self.writer.add_images(os.path.join('Similarity', 'input').replace('\\', '/'), input_list, epoch, dataformats='NHWC')
        self.writer.add_images(os.path.join('Similarity', 'middle_input').replace('\\', '/'), middle_input_list, epoch, dataformats='NHWC')
        self.writer.add_images(os.path.join('Similarity', 'sim_1').replace('\\', '/'), sim_1_list, epoch, dataformats='NHWC')
        self.writer.add_images(os.path.join('Similarity', 'sim_2').replace('\\', '/'), sim_2_list, epoch, dataformats='NHWC')
    
    def log_sim_images_unlabeled(self, colormap, epoch):
        l1_list = [x['l1'] for x in self.eval_images['sim_unlabeled'][0]]
        l1_list = [self.get_images_ready_for_display(x, colormap=None) for x in l1_list]
        l1_list = np.stack(l1_list, axis=0)

        l2_list = [x['l2'] for x in self.eval_images['sim_unlabeled'][0]]
        l2_list = [self.get_images_ready_for_display(x, colormap=None) for x in l2_list]
        l2_list = np.stack(l2_list, axis=0)

        u1_list = [x['u1'] for x in self.eval_images['sim_unlabeled'][0]]
        u1_list = [self.get_images_ready_for_display(x, colormap=None) for x in u1_list]
        u1_list = np.stack(u1_list, axis=0)

        sim_1_list = [x['sim_1'] for x in self.eval_images['sim_unlabeled'][0]]
        sim_1_list = [self.get_images_ready_for_display(x, colormap) for x in sim_1_list]
        sim_1_list = np.stack(sim_1_list, axis=0)

        sim_2_list = [x['sim_2'] for x in self.eval_images['sim_unlabeled'][0]]
        sim_2_list = [self.get_images_ready_for_display(x, colormap) for x in sim_2_list]
        sim_2_list = np.stack(sim_2_list, axis=0)

        sim_3_list = [x['sim_3'] for x in self.eval_images['sim_unlabeled'][0]]
        sim_3_list = [self.get_images_ready_for_display(x, colormap) for x in sim_3_list]
        sim_3_list = np.stack(sim_3_list, axis=0)

        self.writer.add_images(os.path.join('Similarity', 'l1').replace('\\', '/'), l1_list, epoch, dataformats='NHWC')
        self.writer.add_images(os.path.join('Similarity', 'l2').replace('\\', '/'), l2_list, epoch, dataformats='NHWC')
        self.writer.add_images(os.path.join('Similarity', 'u1').replace('\\', '/'), u1_list, epoch, dataformats='NHWC')
        self.writer.add_images(os.path.join('Similarity', 'sim_1').replace('\\', '/'), sim_1_list, epoch, dataformats='NHWC')
        self.writer.add_images(os.path.join('Similarity', 'sim_2').replace('\\', '/'), sim_2_list, epoch, dataformats='NHWC')
        self.writer.add_images(os.path.join('Similarity', 'sim_3').replace('\\', '/'), sim_3_list, epoch, dataformats='NHWC')

    def log_corr_images(self, colormap, epoch):
        corr = self.eval_images['corr'][0][0]['corr']

        if epoch == 0:
            self.vmin = corr.min()
            self.vmax = corr.max()

        corr = self.get_images_ready_for_display(corr, colormap, vmin=self.vmin, vmax=self.vmax)
        self.writer.add_image(os.path.join('Correlation', 'correlation').replace('\\', '/'), corr, epoch, dataformats='HWC')

    def log_aff_images(self, colormap, epoch):
        input_list = [x['input'] for x in self.eval_images['affinity'][0]]
        input_list = [self.get_images_ready_for_display(x, colormap=None) for x in input_list]
        input_list = np.stack(input_list, axis=0)

        aff_list = [x['aff'] for x in self.eval_images['affinity'][0]]
        aff_list = [self.get_images_ready_for_display(x, colormap) for x in aff_list]
        aff_list = np.stack(aff_list, axis=0)

        seg_aff_list = [x['seg_aff'] for x in self.eval_images['affinity'][0]]
        seg_aff_list = [self.get_images_ready_for_display(x, colormap) for x in seg_aff_list]
        seg_aff_list = np.stack(seg_aff_list, axis=0)

        self.writer.add_images(os.path.join('Affinity', 'input').replace('\\', '/'), input_list, epoch, dataformats='NHWC')
        self.writer.add_images(os.path.join('Affinity', 'affinity').replace('\\', '/'), aff_list, epoch, dataformats='NHWC')
        self.writer.add_images(os.path.join('Affinity', 'segmentation_affinity').replace('\\', '/'), seg_aff_list, epoch, dataformats='NHWC')
    
    def log_weights_images(self, colormap, epoch):
        x1_list = [x['x1'] for x in self.eval_images['weights'][0]]
        x1_list = [self.get_images_ready_for_display(x, colormap=None) for x in x1_list]
        x1_list = np.stack(x1_list, axis=0)
        x1_list = np.tile(x1_list, reps=(1, 1, 1, 3))

        x2_list = [x['x2'] for x in self.eval_images['weights'][0]]
        x2_list = [self.get_images_ready_for_display(x, colormap=None) for x in x2_list]
        x2_list = np.stack(x2_list, axis=0)
        x2_list = np.tile(x2_list, reps=(1, 1, 1, 3))

        weights_list = [x['weights'] for x in self.eval_images['weights'][0]]
        weights_list = [self.get_images_ready_for_display(x, colormap) for x in weights_list]
        weights_list = np.stack(weights_list, axis=0)

        blended_list = []
        for i in range(len(x1_list)):
            blended = cv.addWeighted(x1_list[i], 0.5, weights_list[i], 0.5, 0.0)
            blended_list.append(blended)
        blended_list = np.stack(blended_list, axis=0)
        self.writer.add_images(os.path.join('weights', 'x1').replace('\\', '/'), blended_list, epoch, dataformats='NHWC')

        blended_list = []
        for i in range(len(x2_list)):
            blended = cv.addWeighted(x2_list[i], 0.5, weights_list[i], 0.5, 0.0)
            blended_list.append(blended)
        blended_list = np.stack(blended_list, axis=0)
        self.writer.add_images(os.path.join('weights', 'x2').replace('\\', '/'), blended_list, epoch, dataformats='NHWC')

    def log_heatmap_images(self, colormap, epoch):
        input_list = [x['input'] for x in self.eval_images['heatmap'][0]]
        input_list = [self.get_images_ready_for_display(x, colormap=None) for x in input_list]
        input_list = np.stack(input_list, axis=0)
        input_list = np.tile(input_list, reps=(1, 1, 1, 3))

        heatmap_list = [x['heatmap'] for x in self.eval_images['heatmap'][0]]
        heatmap_list = [self.get_images_ready_for_display(x, colormap) for x in heatmap_list]
        heatmap_list = np.stack(heatmap_list, axis=0)

        gt_heatmap_list = [x['gt_heatmap'] for x in self.eval_images['heatmap'][0]]
        gt_heatmap_list = [self.get_images_ready_for_display(x, colormap) for x in gt_heatmap_list]
        gt_heatmap_list = np.stack(gt_heatmap_list, axis=0)

        blended_list = []
        for i in range(len(input_list)):
            blended = cv.addWeighted(input_list[i], 0.5, heatmap_list[i], 0.5, 0.0)
            blended_list.append(blended)
        blended_list = np.stack(blended_list, axis=0)

        self.writer.add_images(os.path.join('heatmap', 'heatmap').replace('\\', '/'), blended_list, epoch, dataformats='NHWC')
        self.writer.add_images(os.path.join('heatmap', 'gt_heatmap').replace('\\', '/'), gt_heatmap_list, epoch, dataformats='NHWC')

    def log_registered_x_images(self, colormap, colormap_seg, norm, epoch):
        moving_x_list = [x['moving_x'] for x in self.eval_images['registered_x'][0]]
        moving_x_list = [self.get_images_ready_for_display(x, colormap=None) for x in moving_x_list]
        moving_x_list = np.stack(moving_x_list, axis=0)

        registered_x_list = [x['registered_x'] for x in self.eval_images['registered_x'][0]]
        registered_x_list = [self.get_images_ready_for_display(x, colormap=None) for x in registered_x_list]
        registered_x_list = np.stack(registered_x_list, axis=0)

        fixed_x_list = [x['fixed_x'] for x in self.eval_images['registered_x'][0]]
        fixed_x_list = [self.get_images_ready_for_display(x, colormap=None) for x in fixed_x_list]
        fixed_x_list = np.stack(fixed_x_list, axis=0)

        motion_list = [x['motion'] for x in self.eval_images['registered_x'][0]]
        motion_list = [self.get_images_ready_for_display(x, colormap) for x in motion_list]
        motion_list = np.stack(motion_list, axis=0)

        moving_seg_list = [x['moving_seg'] for x in self.eval_images['registered_x'][0]]
        moving_seg_list = [self.get_seg_images_ready_for_display(x, colormap=colormap_seg, norm=norm) for x in moving_seg_list]
        moving_seg_list = np.stack(moving_seg_list, axis=0)

        registered_seg_list = [x['registered_seg'] for x in self.eval_images['registered_x'][0]]
        registered_seg_list = [self.get_seg_images_ready_for_display(x, colormap=colormap_seg, norm=norm) for x in registered_seg_list]
        registered_seg_list = np.stack(registered_seg_list, axis=0)

        self.writer.add_images(os.path.join('Registered_x', 'moving_x').replace('\\', '/'), moving_x_list, epoch, dataformats='NHWC')
        self.writer.add_images(os.path.join('Registered_x', 'registered_x').replace('\\', '/'), registered_x_list, epoch, dataformats='NHWC')
        self.writer.add_images(os.path.join('Registered_x', 'fixed').replace('\\', '/'), fixed_x_list, epoch, dataformats='NHWC')
        self.writer.add_images(os.path.join('Registered_x', 'motion_estimation').replace('\\', '/'), motion_list, epoch, dataformats='NHWC')
        self.writer.add_images(os.path.join('Registered_x', 'moving_seg').replace('\\', '/'), moving_seg_list, epoch, dataformats='NHWC')
        self.writer.add_images(os.path.join('Registered_x', 'registered_seg').replace('\\', '/'), registered_seg_list, epoch, dataformats='NHWC')

    def log_motion_images(self, colormap_seg, colormap, norm, epoch, name):
        moving_list = [x['moving'] for x in self.eval_images[name][0]]
        moving_list = [self.get_seg_images_ready_for_display(x, colormap=colormap_seg, norm=norm) for x in moving_list]
        moving_list = np.stack(moving_list, axis=0)

        registered_list = [x['registered'] for x in self.eval_images[name][0]]
        registered_list = [self.get_seg_images_ready_for_display(x, colormap=colormap_seg, norm=norm) for x in registered_list]
        registered_list = np.stack(registered_list, axis=0)

        fixed_list = [x['fixed'] for x in self.eval_images[name][0]]
        fixed_list = [self.get_seg_images_ready_for_display(x, colormap=colormap_seg, norm=norm) for x in fixed_list]
        fixed_list = np.stack(fixed_list, axis=0)

        motion_list = [x['motion'] for x in self.eval_images[name][0]]
        motion_list = [self.get_images_ready_for_display(x, colormap) for x in motion_list]
        motion_list = np.stack(motion_list, axis=0)

        self.writer.add_images(os.path.join(name, 'moving').replace('\\', '/'), moving_list, epoch, dataformats='NHWC')
        self.writer.add_images(os.path.join(name, 'registered').replace('\\', '/'), registered_list, epoch, dataformats='NHWC')
        self.writer.add_images(os.path.join(name, 'fixed').replace('\\', '/'), fixed_list, epoch, dataformats='NHWC')
        self.writer.add_images(os.path.join(name, 'motion').replace('\\', '/'), motion_list, epoch, dataformats='NHWC')

    def log_df_images(self, colormap, epoch):
        input_list = [x['input'] for x in self.eval_images['df'][0]]
        input_list = [self.get_images_ready_for_display(x, colormap=None) for x in input_list]
        input_list = np.stack(input_list, axis=0)

        gt_df_list = [x['gt_df'] for x in self.eval_images['df'][0]]
        gt_df_list = [self.get_images_ready_for_display(x, colormap) for x in gt_df_list]
        gt_df_list = np.stack(gt_df_list, axis=0)

        pred_df_list = [x['pred_df'] for x in self.eval_images['df'][0]]
        pred_df_list = [self.get_images_ready_for_display(x, colormap) for x in pred_df_list]
        pred_df_list = np.stack(pred_df_list, axis=0)

        self.writer.add_images(os.path.join('Directional_field', 'input').replace('\\', '/'), input_list, epoch, dataformats='NHWC')
        self.writer.add_images(os.path.join('Directional_field', 'ground_truth_directional_field').replace('\\', '/'), gt_df_list, epoch, dataformats='NHWC')
        self.writer.add_images(os.path.join('Directional_field', 'predicted_directional_field').replace('\\', '/'), pred_df_list, epoch, dataformats='NHWC')

    def log_seg_images(self, colormap, norm, epoch):
        input_list = [x['input'] for x in self.eval_images['seg'][0]]
        input_list = [self.get_images_ready_for_display(x, colormap=None) for x in input_list]
        input_list = np.stack(input_list, axis=0)

        gt_list = [x['gt'] for x in self.eval_images['seg'][0]]
        gt_list = [self.get_seg_images_ready_for_display(x, colormap, norm) for x in gt_list]
        gt_list = np.stack(gt_list, axis=0)

        pred_list = [x['pred'] for x in self.eval_images['seg'][0]]
        pred_list = [self.get_seg_images_ready_for_display(x, colormap, norm) for x in pred_list]
        pred_list = np.stack(pred_list, axis=0)

        self.writer.add_images(os.path.join('Segmentation', 'input').replace('\\', '/'), input_list, epoch, dataformats='NHWC')
        self.writer.add_images(os.path.join('Segmentation', 'ground_truth').replace('\\', '/'), gt_list, epoch, dataformats='NHWC')
        self.writer.add_images(os.path.join('Segmentation', 'prediction').replace('\\', '/'), pred_list, epoch, dataformats='NHWC')
    
    def log_pseudo_label_images(self, colormap_seg, norm, epoch):
        input_list = [x['input'] for x in self.eval_images['pseudo_label'][0]]
        input_list = [self.get_images_ready_for_display(x, colormap=None) for x in input_list]
        input_list = np.stack(input_list, axis=0)

        pred_list = [x['pred'] for x in self.eval_images['pseudo_label'][0]]
        pred_list = [self.get_seg_images_ready_for_display(x, colormap_seg, norm) for x in pred_list]
        pred_list = np.stack(pred_list, axis=0)

        self.writer.add_images(os.path.join('Pseudo_label', 'input').replace('\\', '/'), input_list, epoch, dataformats='NHWC')
        self.writer.add_images(os.path.join('Pseudo_label', 'prediction').replace('\\', '/'), pred_list, epoch, dataformats='NHWC')

    def log_middle_seg_images(self, colormap, norm, epoch):
        input_list = [x['input'] for x in self.eval_images['middle_seg'][0]]
        input_list = [self.get_images_ready_for_display(x, colormap=None) for x in input_list]
        input_list = np.stack(input_list, axis=0)

        gt_list = [x['gt'] for x in self.eval_images['middle_seg'][0]]
        gt_list = [self.get_seg_images_ready_for_display(x, colormap, norm) for x in gt_list]
        gt_list = np.stack(gt_list, axis=0)

        pred_list = [x['pred'] for x in self.eval_images['middle_seg'][0]]
        pred_list = [self.get_seg_images_ready_for_display(x, colormap, norm) for x in pred_list]
        pred_list = np.stack(pred_list, axis=0)

        self.writer.add_images(os.path.join('Middle segmentation', 'input').replace('\\', '/'), input_list, epoch, dataformats='NHWC')
        self.writer.add_images(os.path.join('Middle segmentation', 'ground_truth').replace('\\', '/'), gt_list, epoch, dataformats='NHWC')
        self.writer.add_images(os.path.join('Middle segmentation', 'prediction').replace('\\', '/'), pred_list, epoch, dataformats='NHWC')

    def log_inter_seg_images(self, colormap, norm, epoch):
        target_list = [x['target'] for x in self.eval_images['inter'][0]]
        target_list = [self.get_seg_images_ready_for_display(x, colormap, norm) for x in target_list]
        target_list = np.stack(target_list, axis=0)

        middle_target_list = [x['middle_target'] for x in self.eval_images['inter'][0]]
        middle_target_list = [self.get_seg_images_ready_for_display(x, colormap, norm) for x in middle_target_list]
        middle_target_list = np.stack(middle_target_list, axis=0)

        gt_list = [x['gt'] for x in self.eval_images['inter'][0]]
        gt_list = [self.get_seg_images_ready_for_display(x, colormap, norm) for x in gt_list]
        gt_list = np.stack(gt_list, axis=0)

        pred_list = [x['pred'] for x in self.eval_images['inter'][0]]
        pred_list = [self.get_seg_images_ready_for_display(x, colormap, norm) for x in pred_list]
        pred_list = np.stack(pred_list, axis=0)

        self.writer.add_images(os.path.join('Inter segmentation', 'middle_target').replace('\\', '/'), middle_target_list, epoch, dataformats='NHWC')
        self.writer.add_images(os.path.join('Inter segmentation', 'ground_truth').replace('\\', '/'), gt_list, epoch, dataformats='NHWC')
        self.writer.add_images(os.path.join('Inter segmentation', 'target').replace('\\', '/'), target_list, epoch, dataformats='NHWC')
        self.writer.add_images(os.path.join('Inter segmentation', 'prediction').replace('\\', '/'), pred_list, epoch, dataformats='NHWC')

    def log_confidence_images(self, colormap, colormap_seg, norm, epoch):
        input_list = [x['input'] for x in self.eval_images['confidence'][0]]
        input_list = [self.get_images_ready_for_display(x, colormap=None) for x in input_list]
        input_list = np.stack(input_list, axis=0)

        confidence_list = [x['confidence'] for x in self.eval_images['confidence'][0]]
        confidence_list = [self.get_images_ready_for_display(x, colormap) for x in confidence_list]
        confidence_list = np.stack(confidence_list, axis=0)

        pred_list = [x['pred'] for x in self.eval_images['confidence'][0]]
        pred_list = [self.get_seg_images_ready_for_display(x, colormap_seg, norm) for x in pred_list]
        pred_list = np.stack(pred_list, axis=0)

        self.writer.add_images(os.path.join('Confidence', 'input').replace('\\', '/'), input_list, epoch, dataformats='NHWC')
        self.writer.add_images(os.path.join('Confidence', 'confidence_map').replace('\\', '/'), confidence_list, epoch, dataformats='NHWC')
        self.writer.add_images(os.path.join('Confidence', 'prediction').replace('\\', '/'), pred_list, epoch, dataformats='NHWC')

    def set_up_image_seg(self, seg_dice, gt, pred, x):
        seg_dice = seg_dice.cpu().numpy()
        gt = gt.cpu().numpy()
        pred = pred.cpu().numpy()
        x = x.cpu().numpy()

        if self.eval_images['seg'][1, 0] < seg_dice:

            self.eval_images['seg'][0, 0]['gt'] = gt.astype(np.float32)
            self.eval_images['seg'][0, 0]['pred'] = pred.astype(np.float32)
            self.eval_images['seg'][0, 0]['input'] = x.astype(np.float32)
            self.eval_images['seg'][1, 0] = seg_dice

            sorted_indices = self.eval_images['seg'][1, :].argsort()
            self.eval_images['seg'] = self.eval_images['seg'][:, sorted_indices]

    def set_up_image_pseudo_label(self, seg_dice, pred, unlabeled_input):
        seg_dice = seg_dice.cpu().numpy()
        pred = pred.cpu().numpy()
        unlabeled_input = unlabeled_input.cpu().numpy()

        if self.eval_images['pseudo_label'][1, 0] < seg_dice:

            self.eval_images['pseudo_label'][0, 0]['pred'] = pred.astype(np.float32)
            self.eval_images['pseudo_label'][0, 0]['input'] = unlabeled_input.astype(np.float32)
            self.eval_images['pseudo_label'][1, 0] = seg_dice

            sorted_indices = self.eval_images['pseudo_label'][1, :].argsort()
            self.eval_images['pseudo_label'] = self.eval_images['pseudo_label'][:, sorted_indices]

    def set_up_image_inter_seg(self, seg_dice, gt, pred, target, middle_target):
        seg_dice = seg_dice.cpu().numpy()
        gt = gt.cpu().numpy()
        pred = pred.cpu().numpy()
        target = target.cpu().numpy()
        middle_target = middle_target.cpu().numpy()

        if self.eval_images['inter'][1, 0] < seg_dice:

            self.eval_images['inter'][0, 0]['gt'] = gt.astype(np.float32)
            self.eval_images['inter'][0, 0]['pred'] = pred.astype(np.float32)
            self.eval_images['inter'][0, 0]['target'] = target.astype(np.float32)
            self.eval_images['inter'][0, 0]['middle_target'] = middle_target.astype(np.float32)
            self.eval_images['inter'][1, 0] = seg_dice

            sorted_indices = self.eval_images['inter'][1, :].argsort()
            self.eval_images['inter'] = self.eval_images['inter'][:, sorted_indices]

    def set_up_image_middle_seg(self, seg_dice, gt, pred, x):
        seg_dice = seg_dice.cpu().numpy()
        gt = gt.cpu().numpy()
        pred = pred.cpu().numpy()
        x = x.cpu().numpy()

        if self.eval_images['middle_seg'][1, 0] < seg_dice:

            self.eval_images['middle_seg'][0, 0]['gt'] = gt.astype(np.float32)
            self.eval_images['middle_seg'][0, 0]['pred'] = pred.astype(np.float32)
            self.eval_images['middle_seg'][0, 0]['input'] = x.astype(np.float32)
            self.eval_images['middle_seg'][1, 0] = seg_dice

            sorted_indices = self.eval_images['middle_seg'][1, :].argsort()
            self.eval_images['middle_seg'] = self.eval_images['middle_seg'][:, sorted_indices]

    def set_up_image_aff(self, seg_dice,  aff, seg_aff, x):
        seg_dice = seg_dice.cpu().numpy()
        aff = aff.cpu().numpy()
        seg_aff = seg_aff.cpu().numpy()
        x = x.cpu().numpy()

        if self.eval_images['affinity'][1, 0] < seg_dice:

            self.eval_images['affinity'][0, 0]['aff'] = aff.astype(np.float32)
            self.eval_images['affinity'][0, 0]['seg_aff'] = seg_aff.astype(np.float32)
            self.eval_images['affinity'][0, 0]['input'] = x.astype(np.float32)
            self.eval_images['affinity'][1, 0] = seg_dice

            sorted_indices = self.eval_images['affinity'][1, :].argsort()
            self.eval_images['affinity'] = self.eval_images['affinity'][:, sorted_indices]
    
    def set_up_image_weights(self, seg_dice, w3, x1, x2):
        seg_dice = seg_dice.cpu().numpy()
        w3 = w3.cpu().numpy()
        x1 = x1.cpu().numpy()
        x2 = x2.cpu().numpy()

        if self.eval_images['weights'][1, 0] < seg_dice:

            self.eval_images['weights'][0, 0]['weights'] = w3.astype(np.float32)
            self.eval_images['weights'][0, 0]['x1'] = x1.astype(np.float32)
            self.eval_images['weights'][0, 0]['x2'] = x2.astype(np.float32)
            self.eval_images['weights'][1, 0] = seg_dice

            sorted_indices = self.eval_images['weights'][1, :].argsort()
            self.eval_images['weights'] = self.eval_images['weights'][:, sorted_indices]

    def set_up_image_heatmap(self, heatmap_ssim, heatmap, gt_heatmap, x):
        heatmap_ssim = heatmap_ssim.cpu().numpy()
        heatmap = heatmap.cpu().numpy()
        gt_heatmap = gt_heatmap.cpu().numpy()
        x = x.cpu().numpy()

        if self.eval_images['heatmap'][1, 0] < heatmap_ssim:

            self.eval_images['heatmap'][0, 0]['heatmap'] = heatmap.astype(np.float32)
            self.eval_images['heatmap'][0, 0]['gt_heatmap'] = gt_heatmap.astype(np.float32)
            self.eval_images['heatmap'][0, 0]['input'] = x.astype(np.float32)
            self.eval_images['heatmap'][1, 0] = heatmap_ssim

            sorted_indices = self.eval_images['heatmap'][1, :].argsort()
            self.eval_images['heatmap'] = self.eval_images['heatmap'][:, sorted_indices]

    def set_up_image_registered_x(self, ssim, moving_x, registered_x, fixed_x, motion, moving_seg, registered_seg):
        ssim = ssim.cpu().numpy()
        moving_x = moving_x.cpu().numpy()
        registered_x = registered_x.cpu().numpy()
        fixed_x = fixed_x.cpu().numpy()
        motion = motion.cpu().numpy()
        moving_seg = moving_seg.cpu().numpy()
        registered_seg = registered_seg.cpu().numpy()

        if self.eval_images['registered_x'][1, 0] < ssim:

            self.eval_images['registered_x'][0, 0]['moving_x'] = moving_x.astype(np.float32)
            self.eval_images['registered_x'][0, 0]['registered_x'] = registered_x.astype(np.float32)
            self.eval_images['registered_x'][0, 0]['fixed_x'] = fixed_x.astype(np.float32)
            self.eval_images['registered_x'][0, 0]['motion'] = motion.astype(np.float32)
            self.eval_images['registered_x'][0, 0]['moving_seg'] = moving_seg.astype(np.float32)
            self.eval_images['registered_x'][0, 0]['registered_seg'] = registered_seg.astype(np.float32)
            self.eval_images['registered_x'][1, 0] = ssim

            sorted_indices = self.eval_images['registered_x'][1, :].argsort()
            self.eval_images['registered_x'] = self.eval_images['registered_x'][:, sorted_indices]

    def set_up_image_motion(self, seg_dice, moving, registered, fixed, motion, name):
        seg_dice = seg_dice.cpu().numpy()
        moving = moving.cpu().numpy()
        registered = registered.cpu().numpy()
        fixed = fixed.cpu().numpy()
        motion = motion.cpu().numpy()

        if self.eval_images[name][1, 0] < seg_dice:

            self.eval_images[name][0, 0]['moving'] = moving.astype(np.float32)
            self.eval_images[name][0, 0]['registered'] = registered.astype(np.float32)
            self.eval_images[name][0, 0]['fixed'] = fixed.astype(np.float32)
            self.eval_images[name][0, 0]['motion'] = motion.astype(np.float32)
            self.eval_images[name][1, 0] = seg_dice

            sorted_indices = self.eval_images[name][1, :].argsort()
            self.eval_images[name] = self.eval_images[name][:, sorted_indices]
        
    def set_up_image_confidence(self, seg_dice, confidence, pred, x):
        seg_dice = seg_dice.cpu().numpy()
        confidence = confidence.cpu().numpy()
        pred = pred.cpu().numpy()
        x = x.cpu().numpy()

        if self.eval_images['confidence'][1, 0] < seg_dice:

            self.eval_images['confidence'][0, 0]['confidence'] = confidence.astype(np.float32)
            self.eval_images['confidence'][0, 0]['pred'] = pred.astype(np.float32)
            self.eval_images['confidence'][0, 0]['input'] = x.astype(np.float32)
            self.eval_images['confidence'][1, 0] = seg_dice

            sorted_indices = self.eval_images['confidence'][1, :].argsort()
            self.eval_images['confidence'] = self.eval_images['confidence'][:, sorted_indices]

    def set_up_image_df(self, df_ssim, gt_df, pred_df, x):
        df_ssim = df_ssim.cpu().numpy()
        gt_df = gt_df.cpu().numpy()
        pred_df = pred_df.cpu().numpy()
        x = x.cpu().numpy()

        if self.eval_images['df'][1, 0] < df_ssim:

            self.eval_images['df'][0, 0]['gt_df'] = gt_df.astype(np.float32)
            self.eval_images['df'][0, 0]['pred_df'] = pred_df.astype(np.float32)
            self.eval_images['df'][0, 0]['input'] = x.astype(np.float32)
            self.eval_images['df'][1, 0] = df_ssim

            self.eval_images['df'] = self.eval_images['df'][:, self.eval_images['df'][1, :].argsort()]

    def set_up_image_sim(self, seg_dice, sim_1, sim_2, x, middle):
        seg_dice = seg_dice.cpu().numpy()
        sim_2 = sim_2.cpu().numpy()
        sim_1 = sim_1.cpu().numpy()
        middle = middle.cpu().numpy()
        x = x.cpu().numpy()

        if self.eval_images['sim'][1, 0] < seg_dice:

            self.eval_images['sim'][0, 0]['sim_1'] = sim_1.astype(np.float32)
            self.eval_images['sim'][0, 0]['sim_2'] = sim_2.astype(np.float32)
            self.eval_images['sim'][0, 0]['input'] = x.astype(np.float32)
            self.eval_images['sim'][0, 0]['middle_input'] = middle.astype(np.float32)
            self.eval_images['sim'][1, 0] = seg_dice

            self.eval_images['sim'] = self.eval_images['sim'][:, self.eval_images['sim'][1].argsort()]

    def set_up_image_sim_unlabeled(self, seg_dice, sim_1, sim_2, sim_3, l1, l2, u1):
        seg_dice = seg_dice.cpu().numpy()
        sim_1 = sim_1.cpu().numpy()
        sim_2 = sim_2.cpu().numpy()
        sim_3 = sim_3.cpu().numpy()
        l1 = l1.cpu().numpy()
        l2 = l2.cpu().numpy()
        u1 = u1.cpu().numpy()

        if self.eval_images['sim_unlabeled'][1, 0] < seg_dice:

            self.eval_images['sim_unlabeled'][0, 0]['sim_1'] = sim_1.astype(np.float32)
            self.eval_images['sim_unlabeled'][0, 0]['sim_2'] = sim_2.astype(np.float32)
            self.eval_images['sim_unlabeled'][0, 0]['sim_3'] = sim_3.astype(np.float32)
            self.eval_images['sim_unlabeled'][0, 0]['l1'] = l1.astype(np.float32)
            self.eval_images['sim_unlabeled'][0, 0]['l2'] = l2.astype(np.float32)
            self.eval_images['sim_unlabeled'][0, 0]['u1'] = u1.astype(np.float32)
            self.eval_images['sim_unlabeled'][1, 0] = seg_dice

            self.eval_images['sim_unlabeled'] = self.eval_images['sim_unlabeled'][:, self.eval_images['sim_unlabeled'][1].argsort()]

    def set_up_image_corr(self, seg_dice, corr):
        seg_dice = seg_dice.cpu().numpy()
        corr = corr.cpu().numpy()

        if self.eval_images['corr'][1, 0] < seg_dice:

            self.eval_images['corr'][0, 0]['corr'] = corr.astype(np.float32)
            self.eval_images['corr'][1, 0] = seg_dice

    def set_up_image_rec(self, rec_ssim, reconstructed, x):
        rec_ssim = rec_ssim.cpu().numpy()
        reconstructed = reconstructed.cpu().numpy()
        x = x.cpu().numpy()

        if self.eval_images['rec'][1] < rec_ssim:
            self.eval_images['rec'][0, 0]['reconstruction'] = reconstructed.astype(np.float32)
            self.eval_images['rec'][0, 0]['input'] = x.astype(np.float32)
            self.eval_images['rec'][1, 0] = rec_ssim

    def get_similarity_ready(self, sim, image_size, similarity_downscale):
        view_size = int(image_size / similarity_downscale)
        sim = sim[0].view(view_size, view_size)[None, None, :, :]
        min_sim = sim.min()
        max_sim = sim.max()
        sim = interpolate(input=sim, scale_factor=similarity_downscale, mode='bicubic', antialias=True).squeeze()
        sim = torch.clamp(sim, min_sim, max_sim)
        return sim