from cv2 import log
import torch
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import numpy as np
from training_utils import log_reinforcement_metrics, log_metrics, resample_softmax, revert
from sys import getsizeof
from math import erf
import math
from loss import logisticGradientPenalty, DirectionalFieldLoss
import cv2 as cv
from torch.nn.functional import interpolate
from utils import update_average, plot_grad_flow
import psutil
import copy
import global_variables
import torch.nn as nn
from ssim import ssim
from dataset_utils import normalize_0_1
from matplotlib import cm
from loss import AngleLoss
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from training_utils import get_rotation_batched_matrices, resample_logits, resample_logits_scipy, get_translation_batched_matrices, min_max_normalization, transform_image, crop_image, Postprocessing3D, Postprocessing2D, remove_padding, get_metrics, rotate_image, improve_label, rotate90, read_config, build_2d_model_crop
from skimage.transform import resize
import sys
from torchvision.ops import masks_to_boxes
from torchvision.transforms.functional import crop, pad, get_image_size
from typing import List
from torch import Tensor
import numbers
from torchvision.utils import _log_api_usage_once

def my_center_crop(img: Tensor, output_size: List[int], fill_value) -> Tensor:
    """Crops the given image at the center.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.
    If image size is smaller than output size along any edge, image is padded with 0 and then center cropped.

    Args:
        img (PIL Image or Tensor): Image to be cropped.
        output_size (sequence or int): (height, width) of the crop box. If int or sequence with single int,
            it is used for both directions.

    Returns:
        PIL Image or Tensor: Cropped image.
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(my_center_crop)
    if isinstance(output_size, numbers.Number):
        output_size = (int(output_size), int(output_size))
    elif isinstance(output_size, (tuple, list)) and len(output_size) == 1:
        output_size = (output_size[0], output_size[0])

    image_width, image_height = get_image_size(img)
    crop_height, crop_width = output_size

    if crop_width > image_width or crop_height > image_height:
        padding_ltrb = [
            (crop_width - image_width) // 2 if crop_width > image_width else 0,
            (crop_height - image_height) // 2 if crop_height > image_height else 0,
            (crop_width - image_width + 1) // 2 if crop_width > image_width else 0,
            (crop_height - image_height + 1) // 2 if crop_height > image_height else 0,
        ]
        img = pad(img, padding_ltrb, fill=fill_value)  # PIL uses fill value 0
        image_width, image_height = get_image_size(img)
        if crop_width == image_width and crop_height == image_height:
            return img

    crop_top = int(round((image_height - crop_height) / 2.0))
    crop_left = int(round((image_width - crop_width) / 2.0))
    return crop(img, crop_top, crop_left, crop_height, crop_width)


class ReconstructionLoop(object):
    def __init__(self,
                 labeled_train_dataloader,
                 validation_dataloader,
                 validation_random_dataloader,
                 val_dataloader_subset, 
                 overfitting_dataloader,
                 train_random_dataloader,
                 model, 
                 model_optimizer, 
                 console_logger, 
                 file_logger,
                 logging_metrics_iteration_number, 
                 logging_loss_iteration_number, 
                 device,
                 nb_iterations_per_epoch,
                 total_nb_of_iterations,
                 rotation_loss_weight,
                 total_nb_epochs, 
                 deep_supervision_weights, 
                 labeled_loss_object, 
                 plot_gradient_iter_number,
                 model_scheduler, 
                 reconstruction_loss_weight,
                 similarity_downscale,
                 writer, 
                 policy_net,
                 reconstruction,
                 policy_scheduler,
                 val_stride,
                 compute_overfitting,
                 logits,
                 binary,
                 number_of_steps,
                 policy_optimizer,
                 transformer_depth,
                 scaling_loss_weight,
                 reconstruction_rotation_loss_weight,
                 reconstruction_scaling_loss_weight,
                 number_of_intervals,
                 adversarial_loss_weight,
                 seg_discriminator_optimizer,
                 rec_discriminator_optimizer,
                 seg_discriminator_scheduler,
                 rec_discriminator_scheduler,
                 seg_discriminator,
                 rec_discriminator,
                 r1_penalty_iteration,
                 total_number_of_iterations,
                 use_cropped_images,
                 uncertainty_weighting,
                 dynamic_weight_averaging,
                 batch_size,
                 conv_depth,
                 img_size,
                 mode,
                 similarity_weight,
                 directional_field,
                 directional_field_weight,
                 class_loss_weights,
                 learn_transforms,
                 save_path='out'):

                self.mode = mode
                self.plot_gradient_iter_number = plot_gradient_iter_number
                self.binary = binary
                self.use_cropped_images = use_cropped_images
                self.learn_transforms = learn_transforms
                self.uncertainty_weighting = uncertainty_weighting
                self.dynamic_weight_averaging = dynamic_weight_averaging
                self.total_number_of_iterations = total_number_of_iterations
                self.val_stride = val_stride
                self.current_epoch = 0
                self.img_size = img_size
                self.save_path = save_path
                self.total_nb_epochs = total_nb_epochs
                self.writer = writer
                self.model_scheduler = model_scheduler
                self.deep_supervision_weights = deep_supervision_weights
                self.validation_dataloader = validation_dataloader
                self.validation_random_dataloader = validation_random_dataloader
                self.labeled_train_dataloader = labeled_train_dataloader
                self.val_dataloader_subset = val_dataloader_subset
                self.overfitting_dataloader = overfitting_dataloader
                self.train_random_dataloader = train_random_dataloader
                self.model = model
                self.seg_discriminator = seg_discriminator
                self.rec_discriminator = rec_discriminator
                self.model_optimizer = model_optimizer
                self.labeled_loss_object = labeled_loss_object
                self.console_logger = console_logger
                self.file_logger = file_logger
                self.logging_metrics_iteration_number = logging_metrics_iteration_number
                self.logging_loss_iteration_number = logging_loss_iteration_number
                self.device = device
                self.nb_iterations_per_epoch = nb_iterations_per_epoch
                self.total_nb_of_iterations = total_nb_of_iterations
                self.similarity_loss = nn.L1Loss()
                self.similarity_downscale = similarity_downscale
                self.reconstruction_loss_weight = reconstruction_loss_weight
                self.batch_size = batch_size
                self.criterion = nn.BCELoss()
                self.r1_penalty_iteration = r1_penalty_iteration
                self.adversarial_loss_weight = adversarial_loss_weight
                self.seg_discriminator_optimizer = seg_discriminator_optimizer
                self.seg_discriminator_scheduler = seg_discriminator_scheduler
                self.rec_discriminator_optimizer = rec_discriminator_optimizer
                self.rec_discriminator_scheduler = rec_discriminator_scheduler
                self.reconstruction = reconstruction
                self.rotation_loss = AngleLoss()
                self.rotation_loss_weight = rotation_loss_weight
                self.scaling_loss_weight = scaling_loss_weight
                self.policy_net = policy_net
                self.policy_optimizer = policy_optimizer
                self.policy_scheduler = policy_scheduler
                self.number_of_intervals = number_of_intervals
                self.reconstruction_rotation_loss_weight = reconstruction_rotation_loss_weight
                self.reconstruction_scaling_loss_weight = reconstruction_scaling_loss_weight
                self.nb_classes = 4 if not binary else 2
                self.compute_overfitting = compute_overfitting
                self.postprocessing_2d = Postprocessing2D()
                self.postprocessing_3d = Postprocessing3D()
                self.logits = logits
                self.similarity_weight = similarity_weight
                self.directional_field = directional_field
                self.directional_field_weight = directional_field_weight
                self.mse_loss = nn.MSELoss()
                self.directional_field_loss = DirectionalFieldLoss(weights=class_loss_weights, writer=self.writer)
                if dynamic_weight_averaging:

                    self.epoch_average_losses = []
                    #self.epoch_loss_weights = []
                    for i in range(self.total_nb_epochs):
                        self.epoch_average_losses.append({'reconstruction': 0.0, 'similarity': 0.0, 'segmentation': 0.0})
                        #self.epoch_loss_weights.append({'reconstruction': 0.0, 'similarity': 0.0, 'segmentation': 0.0})
                
                self.loss_data = {'segmentation': [1.0, float('nan')]}

                if reconstruction:
                    self.loss_data['reconstruction'] = [self.reconstruction_loss_weight, float('nan')]
                    self.loss_data['similarity'] = [self.similarity_weight, float('nan')]

                if self.learn_transforms:
                    self.loss_data['rotation'] = [self.rotation_loss_weight, float('nan')]
                    self.loss_data['rotation_reconstruction'] = [self.reconstruction_rotation_loss_weight, float('nan')]
                    self.loss_data['scaling'] = [self.scaling_loss_weight, float('nan')]
                    self.loss_data['scaling_reconstruction'] = [self.reconstruction_scaling_loss_weight, float('nan')]
                    self.loss_data['scaling_reconstruction'] = [self.reconstruction_scaling_loss_weight, float('nan')]
                
                if directional_field:
                    self.loss_data['directional_field'] = [self.directional_field_weight, float('nan')]

                self.number_of_steps = number_of_steps
                step_size = 4 / number_of_intervals
                #self.q_value = Q_value(0.9, step_size, 10, 6, step_size, step_size)

                self.num_stages = (len(transformer_depth) + len(conv_depth))
    
    def get_loss_weight(self, current_iteration, end_weight):
        end_iteration = self.total_nb_of_iterations * self.weight_end_percent
        if current_iteration > end_iteration:
            return end_weight
        start_weight = 0
        alpha = (end_weight - start_weight) / end_iteration
        weight = alpha * current_iteration + start_weight
        return weight

    
    def get_images_3d(self, worst_unrotated_label, worst_x, worst_pred, worst_dice_angle=None):
        self.model.eval()
        with torch.no_grad():
            x = worst_x
            unrotated_label = worst_unrotated_label
            pred = worst_pred
            #unrotated_label = torch.argmax(unrotated_label.squeeze(0), dim=0)

            #pred = remove_padding(x, pred)

            if worst_dice_angle is not None:
                worst_dice_x_rotated = rotate_image(x, worst_dice_angle, 'bilinear')
                worst_dice_x_rotated = cv.normalize(worst_dice_x_rotated.squeeze().cpu().numpy(), None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F).astype(np.uint8)[..., np.newaxis]
            else:
                worst_dice_x_rotated = None
            
            embedding = torch.nn.Embedding(4, 3)
            embedding.weight.data = torch.tensor([[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255]], device=pred.device)

            pred = embedding(pred.long())

            unrotated_label = embedding(unrotated_label)
            x = cv.normalize(x.squeeze().cpu().numpy(), None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F).astype(np.uint8)[..., np.newaxis]
            out_dict = {'pred': pred.cpu().numpy().astype(np.uint8), 'y': unrotated_label.cpu().numpy().astype(np.uint8), 'x': x, 'reconstructed': None, 'decoder_sm': None, 'reconstruction_sm': None, 'rotated': worst_dice_x_rotated}
            return out_dict
    
    def handle_reconstruction_images(self, best_x, best_reconstructed, best_decoder_sm, best_reconstruction_sm):
        #best_reconstructed = cv.normalize(best_reconstructed.cpu().numpy(), None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)[:, :, None].astype(np.uint8)
        #best_x = cv.normalize(best_x.cpu().numpy(), None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)[:, :, None].astype(np.uint8)

        best_reconstructed = min_max_normalization(best_reconstructed, 0, 255).unsqueeze(-1).type(torch.uint8)
        best_x = min_max_normalization(best_x, 0, 255).unsqueeze(-1).type(torch.uint8)

        view_size = int(self.img_size / self.similarity_downscale)

        decoder_sm = best_decoder_sm[0].view(view_size, view_size)[None, None, :, :]
        reconstruction_sm = best_reconstruction_sm[0].view(view_size, view_size)[None, None, :, :]
        min_decoder_sm = decoder_sm.min()
        max_decoder_sm = decoder_sm.max()
        decoder_sm = interpolate(input=decoder_sm, scale_factor=self.similarity_downscale, mode='bicubic', antialias=True).squeeze()
        decoder_sm = torch.clamp(decoder_sm, min_decoder_sm, max_decoder_sm)
        min_reconstruction_sm = reconstruction_sm.min()
        max_reconstruction_sm = reconstruction_sm.max()
        reconstruction_sm = interpolate(input=reconstruction_sm, scale_factor=self.similarity_downscale, mode='bicubic', antialias=True).squeeze()
        reconstruction_sm = torch.clamp(reconstruction_sm, min_reconstruction_sm, max_reconstruction_sm)
        decoder_sm = normalize_0_1(decoder_sm)
        reconstruction_sm = normalize_0_1(reconstruction_sm)
        decoder_sm = cm.plasma(decoder_sm.cpu().numpy())[:, :, :-1]
        reconstruction_sm = cm.plasma(reconstruction_sm.cpu().numpy())[:, :, :-1]
        decoder_sm = cv.normalize(decoder_sm, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F).astype(np.uint8)
        reconstruction_sm = cv.normalize(reconstruction_sm, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F).astype(np.uint8)
            
        return best_x, best_reconstructed, decoder_sm, reconstruction_sm
    
    def handle_df_image(self, best_predicted_df, best_gt_df):
        best_predicted_df = normalize_0_1(best_predicted_df)
        best_gt_df = normalize_0_1(best_gt_df)
        best_predicted_df = cm.viridis(best_predicted_df)[:, :, :-1]
        best_gt_df = cm.viridis(best_gt_df)[:, :, :-1]
        best_predicted_df = cv.normalize(best_predicted_df, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX).astype(np.uint8)
        best_gt_df = cv.normalize(best_gt_df, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX).astype(np.uint8)
        return best_predicted_df, best_gt_df

    
    def handle_transformed_image(self, image, pred, angle, scale):
        binary_pred = pred
        binary_pred[binary_pred > 0] = 1
        if torch.count_nonzero(binary_pred) == 0:
            small_image = TF.center_crop(image, 128)
        else:
            boxes = masks_to_boxes(binary_pred.unsqueeze(0)).squeeze()
            centroid = torch.tensor([(boxes[0].item() + boxes[2].item()) / 2, (boxes[1].item() + boxes[3].item()) / 2]) # (x, y)
            small_image, _ = crop_image(centroid, 128, image)
        
        assert small_image.shape == (128, 128)

        z = torch.zeros(size=(1, 1), device=small_image.device)
        min_small_image = small_image.min()
        max_small_image = small_image.max()
        transformed = transform_image(image=small_image.float(), angle=angle, tx=z, ty=z, scale=scale, mode='bicubic').squeeze()
        transformed = torch.clamp(transformed, min_small_image, max_small_image)
        transformed = min_max_normalization(transformed, 0, 255)
        #transformed = cv.normalize(transformed.squeeze().cpu().numpy(), None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F).astype(np.uint8)
        #transformed = torch.from_numpy(transformed)
        return transformed.type(torch.uint8)
    
    def get_images_2d(self, 
                    worst_original_label, 
                    worst_original_x, 
                    worst_original_pred, 
                    best_x_in=None,
                    best_x=None,
                    best_pred=None,
                    best_parameters=None, 
                    best_reconstructed=None, 
                    best_decoder_sm=None, 
                    best_reconstruction_sm=None,
                    best_predicted_df=None,
                    best_gt_df=None):
        self.model.eval()
        with torch.no_grad():
            shape_list = []
            x_list = []
            original_label_list = []
            original_pred_list = []
            transformed_list = []
            embedding = torch.nn.Embedding(4, 3)
            embedding.weight.data = torch.tensor([[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255]], device=worst_original_pred[0].device)
            for i in range(len(worst_original_x)):
                x = worst_original_x[i]
                original_pred = worst_original_pred[i]
                original_label = worst_original_label[i]
                #unrotated_label = torch.argmax(unrotated_label.squeeze(0), dim=0)

                if best_parameters is not None:
                    image = best_x[i]

                    pred = torch.nn.functional.softmax(best_pred[i], dim=1)
                    pred = torch.argmax(pred.squeeze(0), dim=0)
                    pred = self.postprocessing_2d(pred)
                    pred = TF.center_crop(pred, output_size=image.shape)
                    
                    #pred = TF.center_crop(best_pred[i], output_size=image.shape)

                    angle = best_parameters[i, 0].reshape(1, 1)
                    scale = best_parameters[i, 1].reshape(1, 1)
                    transformed = self.handle_transformed_image(image, pred, angle, scale)
                    transformed_list.append(transformed)

                #x = cv.normalize(x.squeeze().cpu().numpy(), None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F).astype(np.uint8)
                x = min_max_normalization(x, 0, 255)
                #x = torch.from_numpy(x)

                assert x.shape == original_label.shape == original_pred.shape

                shape_list.append(torch.tensor(original_pred.shape))

                x_list.append(x)
                original_label_list.append(original_label)
                original_pred_list.append(original_pred)
            
            if self.directional_field:
                best_predicted_df, best_gt_df = self.handle_df_image(best_predicted_df, best_gt_df)
            else:
                best_predicted_df = None
                best_gt_df = None

            if self.reconstruction:
                best_x, reconstructed, decoder_sm, reconstruction_sm = self.handle_reconstruction_images(best_x=best_x_in,
                                                                                                best_reconstructed=best_reconstructed,
                                                                                                best_decoder_sm=best_decoder_sm, 
                                                                                                best_reconstruction_sm=best_reconstruction_sm)
            else:
                best_x = reconstructed = decoder_sm = reconstruction_sm = None
            
            shapes = torch.stack(shape_list, dim=0)
            max_shape = torch.max(shapes, dim=0)[0].tolist()

            for i in range(len(x_list)):
                x = TF.center_crop(x_list[i], output_size=max_shape).unsqueeze(-1)
                original_label = TF.center_crop(original_label_list[i], max_shape)
                original_pred = TF.center_crop(original_pred_list[i], max_shape)

                original_pred = embedding(original_pred.long())
                original_label = embedding(original_label.long())

                x_list[i] = x
                original_label_list[i] = original_label
                original_pred_list[i] = original_pred

            x = torch.stack(x_list, dim=0)
            #original_label = torch.stack(original_label_list, dim=0).cpu().numpy().astype(np.uint8)
            #original_pred = torch.stack(original_pred_list, dim=0).cpu().numpy().astype(np.uint8)
            #transformed = torch.stack(transformed_list, dim=0).unsqueeze(-1).cpu().numpy().astype(np.uint8) if transformed_list else None

            original_label = torch.stack(original_label_list, dim=0)
            original_pred = torch.stack(original_pred_list, dim=0)
            transformed = torch.stack(transformed_list, dim=0).unsqueeze(-1) if transformed_list else None
            
            out_dict = {'pred': original_pred.type(torch.uint8), 'y': original_label.type(torch.uint8), 'x': x.type(torch.uint8), 'best_x': best_x, 'reconstructed': reconstructed, 'decoder_sm': decoder_sm, 'reconstruction_sm': reconstruction_sm, 'transformed': transformed, 'predicted_df': best_predicted_df, 'df': best_gt_df}
            return out_dict
    
    #def validation_loop_acdc_2d(self, val_or_train):
    #    dataset = self.train_dataloader_subset if val_or_train == 'train' else self.val_dataloader_subset
    #    self.model.eval()
    #    metrics ={}
    #    with torch.no_grad():
    #        #losses = torch.tensor([0, 0, 0], dtype=torch.float64)
    #        class_ssim_list = []
    #        class_dice_list = []
    #        original_image_list = []
    #        image_list = []
    #        original_label_list = []
    #        angle_distance_list = []
    #        end_ratio_list = []
    #        parameters_list = []
    #        original_pred_list = []
    #        pred_list = []
    #        x_list = []
    #        reconstructed_list = []
    #        decoder_sm_list = []
    #        reconstruction_sm_list = []
    #        class_hds = [[], [], []]
    #        for data in tqdm(dataset, desc='Validation iteration: ', position=2):
    #            x, original_label, original_image, image, label = data['x'].to(self.device), data['original_label'][0].to(self.device), data['original_image'][0], data['image'][0].to(self.device), data['label'][0]
#
    #            pred_volume_list = []
    #            for j in range(x.shape[2]):
    #                current_x = x[:, :, j]
    #                current_image = image[j]
#
    #                out = self.model(current_x)
#
    #                pred = out['pred'][-1]
    #                predicted_parameters = out['parameters']
#
    #                if self.reconstruction:
    #                    x_list.append(current_x.squeeze())
#
    #                pred = torch.nn.functional.softmax(pred, dim=1)
    #                pred = torch.argmax(pred.squeeze(0), dim=0)
    #                pred = self.postprocessing_2d(pred)
    #                pred_list.append(pred)
#
    #                if predicted_parameters is not None:
    #                    predicted_parameters = predicted_parameters.squeeze()
    #                    image_list.append(current_image)
    #                    pred = TF.center_crop(pred, output_size=current_image.shape)
    #                    parameters_list.append(predicted_parameters)
    #                    parameters = data['parameters'][:, j].squeeze()
#
    #                    if parameters[0] != 0:
    #                        a = math.degrees(predicted_parameters[0]) - math.degrees(parameters[0])
    #                        a = abs((a + 180) % 360 - 180)
    #                        angle_distance_list.append(torch.tensor([a]))
    #                    if parameters[3] != 0:
    #                        start_ratio = torch.sum(label[j] > 0) / 128**2
    #                        end_ratio = start_ratio * predicted_parameters[1]**2
    #                        end_ratio_list.append(torch.tensor([end_ratio]))
    #                else:
    #                    pred = revert(image=pred.float(), parameters=data['parameters'][:, j], interpolation_mode='nearest', padding=data['padding'][0][j])
#
    #                assert pred.shape == current_image.shape, pred.shape
    #                assert pred.dim() == 2
#
    #                pred = F.interpolate(pred.unsqueeze(0).unsqueeze(0), size=original_label[j].shape, mode='nearest-exact', antialias=False).squeeze()
#
    #                #pred = pred.cpu().numpy()
    #                #pred = resize(pred, output_shape=original_label[j].shape, order=3)
    #                pred_volume_list.append(pred)
#
    #                if self.reconstruction:
    #                    reconstructed = out['reconstructed'][-1]
    #                    decoder_sm = out['decoder_sm']
    #                    reconstruction_sm = out['reconstruction_sm']
#
    #                    reconstructed_list.append(reconstructed.squeeze())
    #                    decoder_sm_list.append(decoder_sm.cpu().squeeze())
    #                    reconstruction_sm_list.append(reconstruction_sm.cpu().squeeze())
#
    #                    class_ssim_list.append(ssim(current_x, reconstructed))
#
    #            pred = torch.stack(pred_volume_list, dim=0)
    #            #pred = torch.from_numpy(improve_label(pred).astype(np.uint8))
#
    #            assert pred.shape == original_label.shape == original_image.shape
    #            #fig, ax = plt.subplots(2, len(pred))
    #            #for i in range(len(pred)):
    #            #    ax[0, i].imshow(pred[i].cpu(), cmap='gray')
    #            #    ax[1, i].imshow(y_true[i].cpu(), cmap='gray')
    #            #plt.show()
#
    #            for t in range(len(pred)):
    #                current_pred = pred[t]
    #                current_original_label = original_label[t]
    #                current_original_image = original_image[t]
#
    #                original_image_list.append(current_original_image)
    #                original_pred_list.append(current_pred)
    #                original_label_list.append(current_original_label)
    #                
    #                class_dice, class_hds = get_metrics(current_original_label, current_pred, self.nb_classes, class_hds)
    #                class_dice_list.append(class_dice)
#
    #        if self.reconstruction:
    #            ssims = torch.tensor(class_ssim_list)
    #            mean_ssim = ssims.mean()
    #        else:
    #            mean_ssim = None
#
    #        for j in range(self.nb_classes - 1):
    #            class_hds[j] = [class_hds[j][i] for i in range(len(class_hds[j])) if str(class_hds[j][i]) != 'nan']
#
    #        class_mean_hd = [np.array(class_hds[i]).mean() for i in range(self.nb_classes - 1)]
    #        mean_hd = torch.tensor(class_mean_hd).mean()
#
    #        dices = torch.cat(class_dice_list, axis=0)
    #        class_mean_dice = dices.mean(dim=0)
    #        mean_dices = dices.mean(dim=1)
    #        _, worst_indices = torch.topk(mean_dices, k=8, largest=False, sorted=True)
    #        _, best_indices = torch.topk(mean_dices, k=8, largest=True, sorted=True)
#
    #        worst_original_x = [original_image_list[i] for i in worst_indices]
    #        worst_original_label = [original_label_list[i] for i in worst_indices]
    #        worst_original_pred = [original_pred_list[i] for i in worst_indices]
    #        best_pred = [pred_list[i] for i in best_indices]
#
    #        assert len(worst_original_pred) == len(worst_original_label) == len(worst_original_x)
#
    #        if self.reconstruction:
    #            stacked_reconstructed = torch.stack(reconstructed_list, dim=0)
    #            stacked_decoder_sm = torch.stack(decoder_sm_list, dim=0)
    #            stacked_reconstructed_sm = torch.stack(reconstruction_sm_list, dim=0)
#
    #            best_reconstructed = stacked_reconstructed[best_indices[0]]
    #            best_decoder_sm = stacked_decoder_sm[best_indices[0]]
    #            best_reconstruction_sm = stacked_reconstructed_sm[best_indices[0]]
    #            best_pred = best_pred[0]
    #            best_x_in = [x_list[i] for i in best_indices]
    #            best_x_in = best_x_in[0]
    #        else:
    #            best_reconstructed = None
    #            best_decoder_sm = None
    #            best_reconstruction_sm = None
    #            best_x_in = None
#
    #        if angle_distance_list:
    #            best_x = [image_list[i] for i in best_indices]
    #            angle_distances = torch.cat(angle_distance_list)
    #            end_ratios = torch.cat(end_ratio_list)
    #            mean_angle_distance = angle_distances.mean()
    #            mean_end_ratio = end_ratios.mean()
    #            metrics[val_or_train.title() + ' mean angle distance'] = mean_angle_distance
    #            metrics[val_or_train.title() + ' mean end ratio'] = mean_end_ratio
    #            best_parameters = [parameters_list[i] for i in best_indices]
    #            best_parameters = torch.stack(best_parameters, dim=0)
    #            assert len(worst_original_x) == len(best_x) == len(best_parameters) == len(best_pred)
    #        else:
    #            best_parameters = None
    #            best_x = None
#
    #        images = self.get_images_2d(worst_original_label=worst_original_label, 
    #                                    worst_original_x=worst_original_x, 
    #                                    worst_original_pred=worst_original_pred, 
    #                                    best_x_in=best_x_in,
    #                                    best_x=best_x,
    #                                    best_pred=best_pred,
    #                                    best_parameters=best_parameters, 
    #                                    best_reconstructed=best_reconstructed, 
    #                                    best_decoder_sm=best_decoder_sm, 
    #                                    best_reconstruction_sm=best_reconstruction_sm)
#
    #        global_mean_dice = mean_dices.mean()
#
    #        if self.nb_classes == 4:
    #            metrics[val_or_train.title() + ' mean class dice'] = {'LV': class_mean_dice[2], 'RV': class_mean_dice[0], 'MYO': class_mean_dice[1]}
    #            metrics[val_or_train.title() + ' mean class hd'] = {'LV': class_mean_hd[2], 'RV': class_mean_hd[0], 'MYO': class_mean_hd[1]}
    #        metrics[val_or_train.title() + ' mean dice'] = global_mean_dice
    #        metrics[val_or_train.title() + ' mean hd'] = mean_hd
    #        metrics[val_or_train.title() + ' mean ssim'] = mean_ssim
#
    #        return metrics, images
    
    def validation_loop_acdc_2d(self):
        self.model.eval()
        metrics ={}
        with torch.no_grad():
            #losses = torch.tensor([0, 0, 0], dtype=torch.float64)
            predicted_df_list = []
            gt_df_list = []
            reconstruction_ssim_list = []
            df_ssim_list = []
            class_dice_list = []
            class_hd_list = []
            original_image_list = []
            image_list = []
            original_label_list = []
            angle_distance_list = []
            end_ratio_list = []
            parameters_list = []
            original_pred_list = []
            pred_list = []
            x_list = []
            reconstructed_list = []
            decoder_sm_list = []
            reconstruction_sm_list = []
            class_hds = [[]] * self.nb_classes
            for data in tqdm(self.val_dataloader_subset, desc='Validation iteration: ', position=2):
                x = data['x'].to(self.device)
                original_label = data['original_label'][0].to(self.device)
                original_image = data['original_image'][0]
                image = data['image'][0].to(self.device)
                label = data['label'][0]

                pred_volume_list = []
                for j in range(x.shape[2]):
                    current_x = x[:, :, j]
                    current_image = image[j]

                    out = self.model(current_x)

                    if self.mode == 'swin':
                        pred = out['pred'][-1]
                        predicted_parameters = out['parameters']
                        if self.reconstruction:
                            x_list.append(current_x.squeeze())
                    else:
                        pred = out
                    pred_list.append(pred)

                    if not self.use_cropped_images:
                        image_list.append(current_image)
                        pred = TF.center_crop(pred, output_size=current_image.shape)
                        if self.learn_transforms:
                            predicted_parameters = predicted_parameters.squeeze()
                            parameters_list.append(predicted_parameters)
                            parameters = data['parameters'][:, j].squeeze()

                            if parameters[0] != 0:
                                a = math.degrees(predicted_parameters[0]) - math.degrees(parameters[0])
                                a = abs((a + 180) % 360 - 180)
                                angle_distance_list.append(torch.tensor([a]))
                            if parameters[3] != 0:
                                start_ratio = torch.sum(label[j] > 0) / 128**2
                                end_ratio = start_ratio * predicted_parameters[1]**2
                                end_ratio_list.append(torch.tensor([end_ratio]))
                    else:
                        pred = revert(image=pred.float(), parameters=data['parameters'][:, j].to(self.device), interpolation_mode='bicubic', padding=data['padding'][0][j])

                    assert pred.dim() == 4
                    assert pred.shape[-2:] == current_image.shape, pred.shape

                    pred = resample_logits(pred, original_label[j])


                    pred = torch.nn.functional.softmax(pred, dim=1)
                    #pred = resample_softmax(pred, original_label[j])
                    pred = torch.argmax(pred.squeeze(0), dim=0)
                    pred = self.postprocessing_2d(pred)

                    pred_volume_list.append(pred)

                    if self.directional_field:
                        assert x.shape[2] == data['directional_field'].shape[2]
                        df_gt = data['directional_field']
                        df_pred = out['directional_field'].cpu()
                        predicted_df_list.append(df_pred[0, 0])
                        gt_df_list.append(df_gt[0, 0, j])
                        df_ssim_list.append(ssim(df_gt[:, :, j], df_pred))

                    if self.reconstruction:
                        reconstructed = out['reconstructed'][-1]
                        decoder_sm = out['decoder_sm']
                        reconstruction_sm = out['reconstruction_sm']

                        reconstructed_list.append(reconstructed.squeeze())
                        decoder_sm_list.append(decoder_sm.cpu().squeeze())
                        reconstruction_sm_list.append(reconstruction_sm.cpu().squeeze())

                        reconstruction_ssim_list.append(ssim(current_x, reconstructed))

                pred = torch.stack(pred_volume_list, dim=0)
                #pred = torch.from_numpy(improve_label(pred).astype(np.uint8))

                assert pred.shape == original_label.shape == original_image.shape
                #fig, ax = plt.subplots(2, len(pred))
                #for i in range(len(pred)):
                #    ax[0, i].imshow(pred[i].cpu(), cmap='gray')
                #    ax[1, i].imshow(y_true[i].cpu(), cmap='gray')
                #plt.show()

                for t in range(len(pred)):
                    current_pred = pred[t]
                    current_original_label = original_label[t]
                    current_original_image = original_image[t]

                    original_image_list.append(current_original_image)
                    original_pred_list.append(current_pred)
                    original_label_list.append(current_original_label)
                    
                    class_dice, class_hds = get_metrics(current_original_label, current_pred, self.nb_classes)
                    class_dice_list.append(class_dice)
                    class_hd_list.append(class_hds)

            dices = torch.cat(class_dice_list, axis=0)
            hds = torch.cat(class_hd_list, axis=0)
            class_mean_dice = dices.mean(dim=0)
            class_mean_hd = torch.nanmean(hds, dim=0)
            mean_dices = dices.mean(dim=1)
            mean_hds = torch.nanmean(hds, dim=1)
            _, worst_indices = torch.topk(mean_dices, k=8, largest=False, sorted=True)
            _, best_indices = torch.topk(mean_dices, k=8, largest=True, sorted=True)

            worst_original_x = [original_image_list[i] for i in worst_indices]
            worst_original_label = [original_label_list[i] for i in worst_indices]
            worst_original_pred = [original_pred_list[i] for i in worst_indices]
            best_pred = [pred_list[i] for i in best_indices]

            assert len(worst_original_pred) == len(worst_original_label) == len(worst_original_x)

            if self.directional_field:
                stacked_predicted_df = torch.stack(predicted_df_list, dim=0)
                stacked_gt_df = torch.stack(gt_df_list, dim=0)
                df_ssims = torch.tensor(df_ssim_list)
                best_df_index = torch.argmax(df_ssims)
                best_gt_df = stacked_gt_df[best_df_index]
                best_predicted_df = stacked_predicted_df[best_df_index]
                df_mean_ssim = df_ssims.mean()
            else:
                best_predicted_df = None
                best_gt_df = None
                df_mean_ssim=None

            if self.reconstruction:
                reconstruction_ssims = torch.tensor(reconstruction_ssim_list)
                reconstruction_mean_ssim = reconstruction_ssims.mean()

                stacked_reconstructed = torch.stack(reconstructed_list, dim=0)
                stacked_decoder_sm = torch.stack(decoder_sm_list, dim=0)
                stacked_reconstructed_sm = torch.stack(reconstruction_sm_list, dim=0)

                best_reconstructed = stacked_reconstructed[best_indices[0]]
                best_decoder_sm = stacked_decoder_sm[best_indices[0]]
                best_reconstruction_sm = stacked_reconstructed_sm[best_indices[0]]
                best_pred = best_pred[0]
                best_x_in = [x_list[i] for i in best_indices]
                best_x_in = best_x_in[0]
            else:
                best_reconstructed = None
                best_decoder_sm = None
                best_reconstruction_sm = None
                best_x_in = None
                reconstruction_mean_ssim = None

            if angle_distance_list:
                best_x = [image_list[i] for i in best_indices]
                angle_distances = torch.cat(angle_distance_list)
                end_ratios = torch.cat(end_ratio_list)
                mean_angle_distance = angle_distances.mean()
                mean_end_ratio = end_ratios.mean()
                metrics['Mean angle distance'] = mean_angle_distance
                metrics['Mean end ratio'] = mean_end_ratio
                best_parameters = [parameters_list[i] for i in best_indices]
                best_parameters = torch.stack(best_parameters, dim=0)
                assert len(worst_original_x) == len(best_x) == len(best_parameters) == len(best_pred)
            else:
                best_parameters = None
                best_x = None

            images = self.get_images_2d(worst_original_label=worst_original_label, 
                                        worst_original_x=worst_original_x, 
                                        worst_original_pred=worst_original_pred, 
                                        best_x_in=best_x_in,
                                        best_x=best_x,
                                        best_pred=best_pred,
                                        best_parameters=best_parameters, 
                                        best_reconstructed=best_reconstructed, 
                                        best_decoder_sm=best_decoder_sm, 
                                        best_reconstruction_sm=best_reconstruction_sm,
                                        best_predicted_df=best_predicted_df,
                                        best_gt_df=best_gt_df)

            global_mean_dice = mean_dices.mean()
            global_mean_hd = torch.nanmean(mean_hds)

            if self.nb_classes == 4:
                metrics['Mean class dice'] = {'LV': class_mean_dice[2], 'RV': class_mean_dice[0], 'MYO': class_mean_dice[1]}
                metrics['Mean class hd'] = {'LV': class_mean_hd[2], 'RV': class_mean_hd[0], 'MYO': class_mean_hd[1]}
            metrics['Mean dice'] = global_mean_dice
            metrics['Mean hd'] = global_mean_hd
            metrics['Mean reconstruction ssim'] = reconstruction_mean_ssim
            metrics['Mean df ssim'] = df_mean_ssim

            return metrics, images


    def overfit_loop_acdc_2d(self):
        self.model.eval()
        metrics ={}
        with torch.no_grad():
            #losses = torch.tensor([0, 0, 0], dtype=torch.float64)
            reconstruction_ssim_list = []
            df_ssim_list = []
            class_dice_list = []
            class_hd_list = []
            angle_distance_list = []
            end_ratio_list = []
            class_hds = [[]] * self.nb_classes
            for data in tqdm(self.overfitting_dataloader, desc='Validation iteration: ', position=3):
                x = data['x'].to(self.device)
                original_label = data['original_label'][0].to(self.device)
                original_image = data['original_image'][0]
                image = data['image'][0].to(self.device)
                label = data['label'][0]

                out = self.model(x)

                if self.mode == 'swin':
                    pred = out['pred'][-1]
                    predicted_parameters = out['parameters']
                else:
                    pred = out

                if not self.use_cropped_images:
                    pred = TF.center_crop(pred, output_size=image.shape)
                    if self.learn_transforms:
                        predicted_parameters = predicted_parameters.squeeze()
                        parameters = data['parameters'].squeeze()

                        if parameters[0] != 0:
                            a = math.degrees(predicted_parameters[0]) - math.degrees(parameters[0])
                            a = abs((a + 180) % 360 - 180)
                            angle_distance_list.append(torch.tensor([a]))
                        if parameters[3] != 0:
                            start_ratio = torch.sum(label > 0) / 128**2
                            end_ratio = start_ratio * predicted_parameters[1]**2
                            end_ratio_list.append(torch.tensor([end_ratio]))
                else:
                    pred = revert(image=pred.float(), parameters=data['parameters'].to(self.device), interpolation_mode='bicubic', padding=data['padding'][0])

                assert pred.dim() == 4
                assert pred.shape[-2:] == image.shape, pred.shape

                pred = resample_logits(pred, original_label)
                #pred = resample_logits_scipy(pred, original_label[j])

                #pred = F.interpolate(pred, size=original_label[j].shape, mode='bicubic', antialias=True, align_corners=False)

                #pred = pred.cpu().numpy()
                #pred = resize(pred, output_shape=(1, 4,) + original_label[j].shape, order=3, preserve_range=True)
                #pred = torch.from_numpy(pred).to(self.device)

                pred = torch.nn.functional.softmax(pred, dim=1)
                pred = torch.argmax(pred.squeeze(0), dim=0)
                pred = self.postprocessing_2d(pred)

                if self.directional_field:
                    assert x.shape[2] == data['directional_field'].shape[2]
                    df_gt = data['directional_field']
                    df_pred = out['directional_field'].cpu()
                    df_ssim_list.append(ssim(df_gt, df_pred))

                if self.reconstruction:
                    reconstructed = out['reconstructed'][-1]
                    reconstruction_ssim_list.append(ssim(x, reconstructed))

                assert pred.shape == original_label.shape == original_image.shape
                #fig, ax = plt.subplots(2, len(pred))
                #for i in range(len(pred)):
                #    ax[0, i].imshow(pred[i].cpu(), cmap='gray')
                #    ax[1, i].imshow(y_true[i].cpu(), cmap='gray')
                #plt.show()
                
                class_dice, class_hds = get_metrics(original_label, pred, self.nb_classes)
                class_dice_list.append(class_dice)
                class_hd_list.append(class_hds)

            dices = torch.cat(class_dice_list, axis=0)
            hds = torch.cat(class_hd_list, axis=0)
            class_mean_dice = dices.mean(dim=0)
            class_mean_hd = torch.nanmean(hds, dim=0)
            mean_dices = dices.mean(dim=1)
            mean_hds = torch.nanmean(hds, dim=1)

            if self.directional_field:
                df_ssims = torch.tensor(df_ssim_list)
                df_mean_ssim = df_ssims.mean()
            else:
                df_mean_ssim=None

            if self.reconstruction:
                reconstruction_ssims = torch.tensor(reconstruction_ssim_list)
                reconstruction_mean_ssim = reconstruction_ssims.mean()
            else:
                reconstruction_mean_ssim = None

            if angle_distance_list:
                angle_distances = torch.cat(angle_distance_list)
                end_ratios = torch.cat(end_ratio_list)
                mean_angle_distance = angle_distances.mean()
                mean_end_ratio = end_ratios.mean()
                metrics['Overfit mean angle distance'] = mean_angle_distance
                metrics['Overfit mean end ratio'] = mean_end_ratio

            global_mean_dice = mean_dices.mean()
            global_mean_hd = torch.nanmean(mean_hds)

            if self.nb_classes == 4:
                metrics['Overfit mean class dice'] = {'LV': class_mean_dice[2], 'RV': class_mean_dice[0], 'MYO': class_mean_dice[1]}
                metrics['Overfit mean class hd'] = {'LV': class_mean_hd[2], 'RV': class_mean_hd[0], 'MYO': class_mean_hd[1]}
            metrics['Overfit mean dice'] = global_mean_dice
            metrics['Overfit mean hd'] = global_mean_hd
            metrics['Overfit mean reconstruction ssim'] = reconstruction_mean_ssim
            metrics['Overfit mean df ssim'] = df_mean_ssim

            return metrics
    
       
    def validation_loop_acdc_3d(self, val_or_train):
        dataset = self.train_dataloader_subset if val_or_train == 'train' else self.val_dataloader_subset
        self.model.eval()
        metrics ={}
        with torch.no_grad():
            #losses = torch.tensor([0, 0, 0], dtype=torch.float64)
            class_ssim_sum = 0
            angle_distance = 0
            class_dice_list = []
            angle_distance_list = []
            angle_list = []
            original_image_list = []
            original_label_list = []
            pred_list = []
            class_hd_sum = np.zeros(shape=(self.nb_classes - 1,), dtype=np.float64)
            divider_hd = np.full(shape=(self.nb_classes - 1,), fill_value=1e-10, dtype=np.float64)
            for data in tqdm(dataset, desc='Validation iteration: ', position=2):
                x, original_label_volume, original_image_volume, zoomed_label_volume = data['x'].to(self.device), data['original_label_volume'].squeeze(), data['original_image_volume'].squeeze(), data['zoomed_label_volume'].squeeze()

                out = self.model(x)
                pred = out['pred'][-1]
                out_angle = out['angle']
                x = x.squeeze()
                x, pred = remove_padding(x, pred)

                pred = torch.nn.functional.softmax(pred, dim=1)
                pred = torch.argmax(pred.squeeze(0), dim=0)
                pred = self.postprocessing_3d(pred)

                if out_angle is not None:
                    angle_list.append(out_angle)
                    parameters = data['parameters'].squeeze()
                    a = math.degrees(out_angle) - math.degrees(parameters[0])
                    a = (a + 180) % 360 - 180
                    angle_distance_list.append(torch.tensor([a]))
                    crop_list = []
                    for i in range(len(pred)):
                        crop_list.append(TF.center_crop(pred[i], output_size=zoomed_label_volume.shape[1:]))
                    pred = torch.stack(crop_list, dim=0)
                else:
                    padding = data['padding'].squeeze()
                    assert len(padding) == len(pred)
                    crop_list = []
                    for i in range(len(pred)):
                        crop_list.append(revert(image=pred[i].float(), angle=data['angle'].squeeze(), interpolation_mode='nearest', padding=padding[i]))
                    pred = torch.stack(crop_list, dim=0)

                assert zoomed_label_volume.shape == pred.shape

                pred = pred.cpu().numpy()
                pred = rotate90(to_rotate=pred, to_check_volume=original_image_volume, axes=(1, 2))
                #pred = torch.from_numpy(resize(pred, output_shape=original_label_volume.shape, order=0))
                pred = resize(pred.astype(np.uint8), output_shape=original_label_volume.shape, order=3)
                pred = torch.from_numpy(improve_label(pred))

                #fig, ax = plt.subplots(2, len(pred))
                #for i in range(len(pred)):
                #    ax[0, i].imshow(pred[i].cpu(), cmap='gray')
                #    ax[1, i].imshow(y_true[i].cpu(), cmap='gray')
                #plt.show()

                assert pred.shape == original_label_volume.shape == original_image_volume.shape

                original_image_list.append(original_image_volume)
                pred_list.append(pred)
                original_label_list.append(original_label_volume)

                #fig, ax = plt.subplots(2, len(pred))
                #for i in range(len(pred)):
                #    ax[0, i].imshow(pred[i].cpu(), cmap='gray')
                #    ax[1, i].imshow(y_true[i].cpu(), cmap='gray')
                #plt.show()
                
                class_dice, class_hd = get_metrics(original_label_volume, pred, self.nb_classes)
                #class_dice, class_hd = get_metrics(original_label_volume, pred, self.nb_classes)
                class_dice_list.append(class_dice)
                #class_dice_sum += class_dice
                for i in range(self.nb_classes - 1):
                    if not math.isnan(class_hd[i]):
                        class_hd_sum[i] += class_hd[i]
                        divider_hd[i] += 1

            dices = torch.cat(class_dice_list, axis=0)
            class_mean_dice = dices.mean(dim=0)
            mean_dices = dices.mean(dim=1)
            idx = torch.argmin(mean_dices)

            worst_x = original_image_list[idx]
            worst_unrotated_label = original_label_list[idx]
            worst_pred = pred_list[idx]

            if angle_distance_list:
                angle_distances = torch.cat(angle_distance_list)
                mean_angle_distance = angle_distances.mean()
                metrics[val_or_train.title() + ' angle distance'] = mean_angle_distance
                angles = torch.cat(angle_list)
                worst_dice_angle = angles[idx]
                images = self.get_images_3d(worst_unrotated_label, worst_x, worst_pred, worst_dice_angle)
            else:
                images = self.get_images_3d(worst_unrotated_label, worst_x, worst_pred)

            global_mean_dice = mean_dices.mean()

            #class_dices = class_dice_sum / len(dataset)
            class_hds = class_hd_sum / divider_hd
            if class_ssim_sum != 0:
                class_ssim = class_ssim_sum / len(dataset)
            else:
                class_ssim = None
            if self.nb_classes == 4:
                metrics[val_or_train.title() + ' class dice'] = {'LV': class_mean_dice[2], 'RV': class_mean_dice[0], 'MYO': class_mean_dice[1]}
                metrics[val_or_train.title() + ' class hd'] = {'LV': class_hds[2], 'RV': class_hds[0], 'MYO': class_hds[1]}
            metrics[val_or_train.title() + ' average dice'] = global_mean_dice
            metrics[val_or_train.title() + ' average hd'] = class_hds.mean()
            metrics[val_or_train.title() + ' ssim'] = class_ssim

            return metrics, images
    
    def main_loop_acdc_2d(self):
        for idx, t in enumerate(tqdm(range(self.total_nb_epochs), desc='Epoch: ', position=0)):
            max_memory_allocated = torch.cuda.max_memory_allocated(device=self.device)
            self.console_logger.info(f"Max GPU Memory allocated: {max_memory_allocated / 10e8} Gb")
            self.console_logger.info(f"Max CPU Memory allocated: {psutil.Process(os.getpid()).memory_info().rss / 10e8} Gb")
            self.file_logger.info(f"Max GPU Memory allocated: {max_memory_allocated / 10e8} Gb")
            self.current_epoch = idx
            self.train_loop_acdc_2d()
            if idx % self.val_stride == 0:
                val_metrics, images = self.validation_loop_acdc_2d()
                self.validation_logging(val_metrics, images)

                if self.compute_overfitting:
                    overfit_metrics = self.overfit_loop_acdc_2d()
                    self.validation_logging(overfit_metrics, images=None)

                torch.save(self.model.state_dict(), os.path.join(self.save_path, 'model_weights.pth'))
                self.console_logger.info(f"Saved model to {self.save_path}")
    
    def main_loop_acdc_3d(self):
        for idx, t in enumerate(tqdm(range(self.total_nb_epochs), desc='Epoch: ', position=0)):
            max_memory_allocated = torch.cuda.max_memory_allocated(device=self.device)
            self.console_logger.info(f"Max GPU Memory allocated: {max_memory_allocated / 10e8} Gb")
            self.file_logger.info(f"Max GPU Memory allocated: {max_memory_allocated / 10e8} Gb")
            self.current_epoch = idx
            self.train_loop_acdc_3d()
            if idx % self.val_stride == 0:
                val_metrics, images = self.validation_loop_acdc_3d(val_or_train='val', dimensions=3)
                self.validation_logging(val_metrics, images, val_or_train='val')

                if self.compute_overfitting:
                    train_metrics, images = self.validation_loop_acdc_3d(val_or_train='train', dimensions=3)
                    self.validation_logging(train_metrics, images, val_or_train='train')

                torch.save(self.model.state_dict(), os.path.join(self.save_path, 'model_weights.pth'))
                self.console_logger.info(f"Saved model to {self.save_path}")
    
    def main_loop_acdc_3d_logits(self):
        for idx, t in enumerate(tqdm(range(self.total_nb_epochs), desc='Epoch: ', position=0)):
            max_memory_allocated = torch.cuda.max_memory_allocated(device=self.device)
            self.console_logger.info(f"Max GPU Memory allocated: {max_memory_allocated / 10e8} Gb")
            self.file_logger.info(f"Max GPU Memory allocated: {max_memory_allocated / 10e8} Gb")
            self.current_epoch = idx
            self.train_loop_acdc_3d()
            if idx % self.val_stride == 0:
                val_metrics, images = self.validation_loop_acdc_3d(val_or_train='val', dimensions=3)
                self.validation_logging(val_metrics, images, val_or_train='val')

                if self.compute_overfitting:
                    train_metrics, images = self.validation_loop_acdc_3d(val_or_train='train', dimensions=3)
                    self.validation_logging(train_metrics, images, val_or_train='train')

                torch.save(self.model.state_dict(), os.path.join(self.save_path, 'model_weights.pth'))
                self.console_logger.info(f"Saved model to {self.save_path}")
    
    def main_loop_reinforcement(self):
        for idx, t in enumerate(tqdm(range(self.total_nb_epochs), desc='Epoch: ', position=0)):
            max_memory_allocated = torch.cuda.max_memory_allocated(device=self.device)
            self.console_logger.info(f"Max GPU Memory allocated: {max_memory_allocated / 10e8} Gb")
            self.file_logger.info(f"Max GPU Memory allocated: {max_memory_allocated / 10e8} Gb")
            self.current_epoch = idx
            self.train_loop_reinforcement()
            if idx % self.val_stride == 0:
                correct = self.reinforcement_validation_loop()
                images = self.get_reinforcement_validation_images()
                self.reinforcement_validation_logging(images, correct)
                torch.save(self.policy_net.state_dict(), os.path.join(self.save_path, 'model_weights.pth'))
                self.console_logger.info(f"Saved model to {self.save_path}")
    
    def log_images(self, images):
        if images['x'].ndim > 3:
            self.writer.add_images(os.path.join('Val', 'Image').replace('\\', '/'), images['x'], self.current_epoch, dataformats='NHWC')
            self.writer.add_images(os.path.join('Val', 'Ground truth').replace('\\', '/'), images['y'], self.current_epoch, dataformats='NHWC')
            self.writer.add_images(os.path.join('Val', 'Prediction').replace('\\', '/'), images['pred'], self.current_epoch, dataformats='NHWC')
        else:
            self.writer.add_image(os.path.join('Val', 'Image').replace('\\', '/'), images['x'], self.current_epoch, dataformats='HWC')
            self.writer.add_image(os.path.join('Val', 'Ground truth').replace('\\', '/'), images['y'], self.current_epoch, dataformats='HWC')
            self.writer.add_image(os.path.join('Val', 'Prediction').replace('\\', '/'), images['pred'], self.current_epoch, dataformats='HWC')
        if images['reconstructed'] is not None:
            self.writer.add_image(os.path.join('Val', 'Best_x').replace('\\', '/'), images['best_x'], self.current_epoch, dataformats='HWC')
            self.writer.add_image(os.path.join('Val', 'Reconstructed').replace('\\', '/'), images['reconstructed'], self.current_epoch, dataformats='HWC')
            self.writer.add_image(os.path.join('Val', 'Decoder similarity').replace('\\', '/'), images['decoder_sm'], self.current_epoch, dataformats='HWC')
            self.writer.add_image(os.path.join('Val', 'Reconstruction similarity').replace('\\', '/'), images['reconstruction_sm'], self.current_epoch, dataformats='HWC')
        if images['transformed'] is not None:
            self.writer.add_image(os.path.join('Val', 'Transformed image').replace('\\', '/'), images['transformed'], self.current_epoch, dataformats='NHWC')
        if images['df'] is not None:
            self.writer.add_image(os.path.join('Val', 'Predicted directional field').replace('\\', '/'), images['predicted_df'], self.current_epoch, dataformats='HWC')
            self.writer.add_image(os.path.join('Val', 'Gt directional field').replace('\\', '/'), images['df'], self.current_epoch, dataformats='HWC')

    def validation_logging(self, metrics, images):
        if images is not None:
            self.log_images(images)
        log_metrics(self.console_logger, self.file_logger, self.writer, metrics, self.current_epoch, 'Epoch')
    
    def reinforcement_validation_logging(self, images, correct):
        self.writer.add_image('Epoch/Input image', images['x'], self.current_epoch, dataformats='HWC')
        self.writer.add_image('Epoch/Output image', images['pred'], self.current_epoch, dataformats='HWC')
        self.writer.add_image('Epoch/Ground truth', images['y'], self.current_epoch, dataformats='HWC')
        self.writer.add_image('Epoch/Output ground truth', images['pred_y'], self.current_epoch, dataformats='HWC')
        log_reinforcement_metrics(self.console_logger, self.writer, correct, self.current_epoch, 'Epoch')
        log_reinforcement_metrics(self.file_logger, self.writer, correct, self.current_epoch, 'Epoch')
    
    def get_unlabeled_weight(self, iteration_nb):
        iteration_percent = iteration_nb / self.total_nb_of_iterations
        return self.unlabeled_loss_weight_end * (erf(iteration_percent) / erf(1)) 

    def get_adversarial_loss(self, discriminator, discriminator_optimizer, discriminator_scheduler, real, fake, iter_nb):
        label = torch.full((self.batch_size,), 1, dtype=torch.float, device=self.device)
        discriminator.zero_grad()
        output_real = discriminator(real).reshape(-1)
        loss_real = self.criterion(output_real, label)
        self.writer.add_scalar('Iteration/Discriminator real', output_real.mean(), iter_nb)
        #loss_real.backward() #retain_graph if reuse output of discriminator for r1 penalty

        r1_penalty = 0
        if iter_nb % self.r1_penalty_iteration == 0:
            r1_penalty = logisticGradientPenalty(real, discriminator, weight=5)
    
        loss_real_r1 = loss_real + r1_penalty
        loss_real_r1.backward()

        assert fake.size() == real.size()
        label.fill_(0)
        output_fake = discriminator(fake.detach()).view(-1)
        loss_fake = self.criterion(output_fake, label)
        self.writer.add_scalar('Iteration/Discriminator fake', output_fake.mean(), iter_nb)
        loss_fake.backward()

        discriminator_loss = loss_real + loss_fake + r1_penalty
        discriminator_optimizer.step()
        discriminator_scheduler.step()

        self.model.reconstruction.zero_grad()
        label.fill_(1)
        output = discriminator(fake).view(-1)
        adversarial_loss = self.criterion(output, label)

        return adversarial_loss, discriminator_loss
    
    def train_loop_reinforcement(self):
        nb_iters = len(self.labeled_train_dataloader)

        for batch, data in enumerate(tqdm(self.labeled_train_dataloader, desc='Training iteration: ', position=1), 1):
            iter_nb = (self.current_epoch * nb_iters) + batch
            global_variables.global_iter = iter_nb
            x, q = data['x'], data['q_values']

            self.policy_net.train()
            out = self.policy_net(x)

            correct_batch = torch.sum(torch.argmax(out, dim=1) == torch.argmax(q, dim=1)) / out.size(0)

            loss = self.policy_loss(out, q)
            self.policy_optimizer.zero_grad()
            loss.backward()
            self.policy_optimizer.step()
            self.policy_scheduler.step()

            self.writer.add_scalar('Iteration/Training loss', loss, iter_nb)
            self.writer.add_scalar('Iteration/Correct q_values predicted %', correct_batch, iter_nb)

            if batch % self.logging_metrics_iteration_number == 0:
                correct = self.reinforcement_validation_loop()
                images = self.get_reinforcement_validation_images()
                self.reinforcement_validation_logging(images, correct)
    
    
    def rotate_image_with_center(self, x, angle, parameters):
        r = get_rotation_batched_matrices(angle)
        t = get_translation_batched_matrices(parameters[:, 1], parameters[:, 2])
        t_2 = get_translation_batched_matrices(-parameters[:, 1], -parameters[:, 2])
        theta = torch.bmm(t, torch.bmm(r, t_2))[:, :-1]

        grid = F.affine_grid(theta, x.size())
        x_min = torch.min(torch.flatten(x, start_dim=2), dim=-1)[0].unsqueeze(-1).unsqueeze(-1)
        x_max = torch.max(torch.flatten(x, start_dim=2), dim=-1)[0].unsqueeze(-1).unsqueeze(-1)
        rotated = F.grid_sample(x, grid, mode='bicubic')
        rotated = torch.clamp(rotated, x_min, x_max)

        return rotated
    
    def crop(self, x, image_list, y=None, label_list=None):
        with torch.no_grad():
            out = self.pretrained_model(x)
            pred = out['pred'][-1]
            out_angle = out['angle']

            pred = torch.nn.functional.softmax(pred, dim=1)
            pred = torch.argmax(pred, dim=1, keepdim=True)
            pred = self.postprocessing_2d(pred)

            x_out = []
            y_out = []
            padding_out = []
            for i in range(len(image_list)):
                current_image = image_list[i]
                current_angle = out_angle[i]
                current_pred = TF.center_crop(pred[i], output_size=current_image.shape)
                assert current_image.shape == current_pred.shape[1:]

                binary_pred = current_pred
                binary_pred[binary_pred > 0] = 1
                boxes = masks_to_boxes(binary_pred).squeeze()
                centroid = [(boxes[0].item() + boxes[2].item()) / 2, (boxes[1].item() + boxes[3].item()) / 2] # (x, y)

                x, padding_x = crop_image(centroid=centroid, small_image_size=self.crop_size, image=current_image)
                x = rotate_image(x, current_angle, interpolation_mode='bilinear').squeeze(0)
                assert x.shape == (1, self.crop_size, self.crop_size), x.shape
                x_out.append(x)
                padding_out.append(padding_x)

                if label_list is not None:
                    current_label = label_list[i]
                    assert current_label.shape == current_pred.shape[1:] == current_image.shape
                    y, padding_y = crop_image(centroid=centroid, small_image_size=self.crop_size, image=current_label)
                    assert torch.all(padding_x == padding_y)
                    y = rotate_image(y, current_angle, interpolation_mode='nearest').squeeze()
                    y = torch.nn.functional.one_hot(y.long(), num_classes=4).permute(2, 0, 1).float()
                    assert y.shape == (4, self.crop_size, self.crop_size), y.shape
                    y_out.append(y)
                
            x = torch.stack(x_out, dim=0)
            y = torch.stack(y_out, dim=0)
            padding = torch.stack(padding_out, dim=0)

        return x, y, padding


    def train_loop_acdc_2d(self):
        
        nb_iters = len(self.labeled_train_dataloader)
        
        if self.dynamic_weight_averaging:
            #self.easiness_loss_weight = ((self.end_easiness_loss_weight - self.start_easiness_loss_weight) / self.total_nb_epochs) * self.current_epoch + self.start_easiness_loss_weight
            #self.writer.add_scalar(os.path.join('Epoch', 'Easiness loss weight').replace('\\', '/'), self.easiness_loss_weight, self.current_epoch)
            T = 0.15 # 0.18
            if self.current_epoch > 1:
                average_increase= []
                #average_loss_weights = []
                for key, value in self.epoch_average_losses[self.current_epoch - 1].items():
                    average_loss_value = []
                    #average_loss_weight = []
                    for t in range(self.current_epoch):
                        average_loss_value.append(self.epoch_average_losses[t][key])
                        #average_loss_weight.append(self.epoch_loss_weights[t][key])
                    
                    #EMA here
                    #alpha = 2 / (len(average_loss_value) + 1)
                    average_loss_value = torch.tensor(average_loss_value).mean()
                    #alpha * value + (1 - alpha) * average_loss_value'''
                    average_increase.append(value / average_loss_value)
                    #average_loss_weights.append(torch.tensor(average_loss_weight).mean())
                
                average_increase = torch.tensor(average_increase)
                #average_loss_weights = torch.tensor(average_loss_weights)
                #print(average_increase)
                #print(average_loss_weights)
                average_increase = average_increase / average_increase.sum()
                #average_loss_weights = average_loss_weights / average_loss_weights.sum()
                #print(average_increase)
                #print(average_loss_weights)
                #w_list = average_increase + self.regularization_weight * average_loss_weights
                w_list = average_increase
                #print(w_list)
                #w_list.append(value / average_loss_value) # quantify no change
                #for value_1, value_2 in zip(self.epoch_average_losses[self.current_epoch - 1].values(), self.epoch_average_losses[0].values()):
                #    w_list.append(value_1 / value_2)

                #w_list = torch.tensor(w_list, dtype=float)
                #w_list = w_list / w_list.sum() # normalize so that sum equals 1
                #print(w_list)
                #w_list = self.easiness_loss_weight * w_list + (1 - self.easiness_loss_weight) * (1 - w_list)
            for idx, key in enumerate(self.loss_data.keys()):
                if self.current_epoch > 1:
                    loss_weight = len(self.loss_data) * torch.nn.functional.softmax(w_list / T)[idx]
                    self.loss_data[key][0] = loss_weight
                #self.epoch_loss_weights[self.current_epoch][key] = self.loss_data[key][0]
            #print(self.epoch_loss_weights)


        for batch, data in enumerate(tqdm(self.labeled_train_dataloader, desc='Training iteration: ', position=1), 1):
            self.model.train()
            iter_nb = (self.current_epoch * nb_iters) + batch
            global_variables.global_iter = iter_nb

            x = data['x'].to(self.device)
            y = data['y'].to(self.device)

            out = self.model(x)
            
            #fig, ax = plt.subplots(2, len(x))
            #for i in range(len(x)):
            #    ax[0, i].imshow(x.cpu().numpy()[i, 0], cmap='gray')
            #    ax[1, i].imshow(torch.argmax(y, dim=1).cpu()[i], cmap='gray')
            #plt.show()

            #fig, ax = plt.subplots(2, 12)
            #for i in range(12):
            #    ax[0, i].imshow(x.cpu().numpy()[0, 0, i], cmap='gray')
            #    ax[1, i].imshow(torch.argmax(y, dim=1).cpu()[0, i], cmap='gray')
            #plt.show()

            #print(math.degrees(data['parameters'][:, 0]))
            #print(data['parameters'][:, 1])
            #print(data['parameters'][:, 2])
            #for i in range(12):
            #    fig, ax = plt.subplots(1, 2)
            #    ax[0].imshow(x[0, 0, i].cpu().numpy(), cmap='gray')
            #    ax[1].imshow(torch.argmax(y, dim=1, keepdim=True).cpu().numpy()[0, 0, i], cmap='gray')
            #    figManager = plt.get_current_fig_manager()
            #    figManager.window.showMaximized()
            #    plt.show()

            if self.mode == 'swin':
                pred = out['pred']
                reconstructed = out['reconstructed']
                decoder_sm = out['decoder_sm']
                reconstruction_sm = out['reconstruction_sm']
                predicted_parameters = out['parameters']
            else:
                pred = out

            #pred = [torch.nn.functional.softmax(x, dim=1) for x in pred if x is not None]

            if self.mode == 'swin':

                if self.directional_field:
                    df_pred = out['directional_field']
                    directional_field_loss = self.directional_field_loss(pred=df_pred, y_df=data['directional_field'].to(self.device), y_seg=y, iter_nb=iter_nb)
                    self.loss_data['directional_field'][1] = directional_field_loss

                rotation_loss = 0
                scaling_loss = 0
                if predicted_parameters is not None:
                    temp_shape = predicted_parameters.shape
                    parameters = data['parameters'].to(self.device)

                    mask = parameters[:, 0] != 0
                    masked_predicted_parameters = predicted_parameters[mask]
                    masked_parameters = parameters[mask]
                    masked_x = x[mask]

                    if masked_predicted_parameters.size(0) > 0:

                        rotation_loss = self.rotation_loss(masked_predicted_parameters[:, 0], masked_parameters[:, 0])

                        rotation_reconstruction_loss = 0
                        pred_rotated = self.rotate_image_with_center(masked_x, masked_predicted_parameters[:, 0], masked_parameters)
                        gt_rotated = self.rotate_image_with_center(masked_x, masked_parameters[:, 0], masked_parameters)
                        rotation_reconstruction_loss = self.mse_loss(pred_rotated, gt_rotated)

                        #self.writer.add_scalar('Iteration/Rotation loss', rotation_loss, iter_nb)
                        #self.writer.add_scalar('Iteration/Rotation reconstruction loss', rotation_reconstruction_loss, iter_nb)

                        self.loss_data['rotation'][1] = rotation_loss
                        self.loss_data['rotation_reconstruction'][1] = rotation_reconstruction_loss

                        #rotation_loss = (self.rotation_loss_weight * rotation_loss) + (self.reconstruction_rotation_loss_weight * rotation_reconstruction_loss)
                    
                    assert predicted_parameters.shape == temp_shape
                    
                    mask = parameters[:, 3] != 0
                    masked_predicted_parameters = predicted_parameters[mask]
                    masked_parameters = parameters[mask]
                    masked_x = x[mask]

                    if masked_predicted_parameters.size(0) > 0:

                        scaling_loss = self.scaling_loss(masked_predicted_parameters[:, 1], masked_parameters[:, 3])

                        z = torch.zeros_like(masked_parameters[:, 3])
                        scaling_reconstruction_loss = 0
                        pred_scaled = transform_image(masked_x, z, z, z, scale=masked_predicted_parameters[:, 1], mode='bicubic')
                        gt_scaled = transform_image(masked_x, z, z, z, scale=masked_parameters[:, 3], mode='bicubic')
                        scaling_reconstruction_loss = self.mse_loss(pred_scaled, gt_scaled)

                        #self.writer.add_scalar('Iteration/Scaling loss', scaling_loss, iter_nb)
                        #self.writer.add_scalar('Iteration/Scaling reconstruction loss', scaling_reconstruction_loss, iter_nb)
                        #scaling_loss = (self.scaling_loss_weight * scaling_loss) + (self.reconstruction_scaling_loss_weight * scaling_reconstruction_loss)

                        self.loss_data['scaling'][1] = scaling_loss
                        self.loss_data['scaling_reconstruction'][1] = scaling_reconstruction_loss

                similarity_loss = 0
                reconstruction_loss = 0
                #similarity_weight = 0
                if reconstructed is not None:

                    #a = (self.end_similarity - self.start_similarity) / self.total_nb_of_iterations
                    #b = self.start_similarity
                    #self.current_similarity_weight = a * iter_nb + b
                    #self.loss_data['similarity'][0] = self.current_similarity_weight

                    for layer_pred, weight in zip(reconstructed, self.deep_supervision_weights):
                        computed_loss = self.mse_loss(layer_pred, x)
                        reconstruction_loss += (computed_loss * weight)
                    #self.writer.add_scalar('Iteration/Reconstruction loss', reconstruction_loss, iter_nb)
                    self.loss_data['reconstruction'][1] = reconstruction_loss

                    assert decoder_sm.shape == reconstruction_sm.shape
                    similarity_loss = self.similarity_loss(decoder_sm, reconstruction_sm)
                    #self.writer.add_scalar('Iteration/similarity loss', similarity_loss, iter_nb)
                    self.loss_data['similarity'][1] = similarity_loss
                    if self.dynamic_weight_averaging:
                        self.epoch_average_losses[self.current_epoch]['reconstruction'] += reconstruction_loss.item() / nb_iters
                        self.epoch_average_losses[self.current_epoch]['similarity'] += similarity_loss.item() / nb_iters
                    #similarity_weight = self.get_loss_weight(iter_nb, self.similarity_loss_weight)
                    #self.writer.add_scalar('Iteration/similarity weight', similarity_weight, iter_nb)

                segmentation_loss = 0
                for layer_pred, weight in zip(pred, self.deep_supervision_weights):
                    computed_loss = self.labeled_loss_object(layer_pred, y, iter_nb)
                    segmentation_loss += (computed_loss * weight)
            else:
                segmentation_loss = self.labeled_loss_object(pred, y, iter_nb)
            #self.writer.add_scalar('Iteration/segmentation loss', segmentation_loss, iter_nb)
            self.loss_data['segmentation'][1] = segmentation_loss
            if self.dynamic_weight_averaging:
                self.epoch_average_losses[self.current_epoch]['segmentation'] += segmentation_loss.item() / nb_iters

            seg_adv_loss = 0
            rec_adv_loss = 0
            lrs = {}
            if self.seg_discriminator is not None:
                self.seg_discriminator.train()
                self.rec_discriminator.train()

                seg_adv_loss, seg_dis_loss = self.get_adversarial_loss(self.seg_discriminator, 
                                                                        self.seg_discriminator_optimizer, 
                                                                        self.seg_discriminator_scheduler, 
                                                                        real=torch.cat([x, y], dim=1), 
                                                                        fake=torch.cat([x, pred[-1]], dim=1),
                                                                        iter_nb=iter_nb)
                rec_adv_loss, rec_dis_loss = self.get_adversarial_loss(self.rec_discriminator, 
                                                                        self.rec_discriminator_optimizer, 
                                                                        self.rec_discriminator_scheduler, 
                                                                        real=x, 
                                                                        fake=reconstructed[-1],
                                                                        iter_nb=iter_nb)

                lrs['Discriminators'] = self.seg_discriminator_optimizer.param_groups[0]['lr']
                
                #self.writer.add_scalar('Iteration/Segmentation adversarial loss', seg_adv_loss, iter_nb)
                #self.writer.add_scalar('Iteration/Reconstruction adversarial loss', rec_adv_loss, iter_nb)
                #self.writer.add_scalar('Iteration/Segmentation Discriminator loss', seg_dis_loss, iter_nb)
                #self.writer.add_scalar('Iteration/Reconstruction Discriminator loss', rec_dis_loss, iter_nb)
                self.loss_data['adversarial_segmentation'][1] = seg_adv_loss
                self.loss_data['adversarial_reconstruction'][1] = rec_adv_loss
            
        

            loss = 0
            if self.uncertainty_weighting:
                logsigma = out['logsigma']
                assert len(self.loss_data) == len(logsigma)
                loss = sum(1 / (2 * torch.exp(logsigma[i])) * list(self.loss_data.values())[i][1] + logsigma[i] / 2 for i in range(len(self.loss_data)))
                #loss = 0.5 * (math.exp(-param[0]) * loss_data['reconstruction'][1] + math.exp(-param[1]) * loss_data['similarity'][1] + 2 * math.exp(-param[2]) * loss_data['segmentation'][1] + param[0] + param[1] + param[2])
                self.writer.add_scalars('Iteration/loss params', {'reconstruction param': 1 / (2 * torch.exp(logsigma[0])), 'similarity_param': 1 / (2 * torch.exp(logsigma[1])), 'segmentation param': 1 / (2 * torch.exp(logsigma[2]))}, iter_nb)
            else:
                #{key:value[0] for key, value in self.loss_data.items()}
                self.writer.add_scalars('Iteration/loss weights', {key:value[0] for key, value in self.loss_data.items()}, iter_nb)
                for key, value in self.loss_data.items():
                    self.writer.add_scalar('Iteration/' + key + ' loss', value[1], iter_nb)
                    loss += value[0] * value[1]
                #loss = sum(value[0] * value[1] for value in self.loss_data.values() if not torch.isnan(value[1]))
            

            self.writer.add_scalar('Iteration/Training loss', loss, iter_nb)
            
            # Backpropagation
            self.model_optimizer.zero_grad()
            loss.backward()

            if iter_nb % self.plot_gradient_iter_number == 0:
                plot_grad_flow(self.model.named_parameters(), keywords=['fc1', 'fc2'])
            self.model_optimizer.step()
            self.model_scheduler.step()

            lrs['Model'] = self.model_optimizer.param_groups[0]['lr']

            self.writer.add_scalars('Iteration/Learning rates', lrs, iter_nb)

            if batch % self.logging_loss_iteration_number == 0:
                loss = loss.item()
                self.console_logger.info(f"Training loss: {loss:>7f}")

            if batch % self.logging_metrics_iteration_number == 0:
                max_memory_allocated = torch.cuda.max_memory_allocated(device=self.device)
                self.console_logger.info(f"Max GPU Memory allocated: {max_memory_allocated / 10e8} Gb")
                self.console_logger.info(f"Max CPU Memory allocated: {psutil.Process(os.getpid()).memory_info().rss / 10e8} Gb")
                self.file_logger.info(f"Max GPU Memory allocated: {max_memory_allocated / 10e8} Gb")
                val_metrics, images = self.validation_loop_acdc_2d()
                self.validation_logging(val_metrics, images)

                if self.compute_overfitting:
                    overfit_metrics = self.overfit_loop_acdc_2d()
                    self.validation_logging(overfit_metrics, images=None)

            #self.writer.add_scalars('Iteration/Labeled individual loss weights', self.labeled_loss_object.get_loss_weight(), iter_nb)
            self.labeled_loss_object.update_weight()


    def train_loop_acdc_3d(self):
            
            nb_iters = len(self.labeled_train_dataloader)

            for batch, data in enumerate(tqdm(self.labeled_train_dataloader, desc='Training iteration: ', position=1), 1):
                iter_nb = (self.current_epoch * nb_iters) + batch
                global_variables.global_iter = iter_nb
                x, y = data['x'].to(self.device), data['y'].to(self.device)

                #fig, ax = plt.subplots(2, 9)
                #ax[0, 0].imshow(x.cpu().numpy()[0, 0, 0], cmap='gray')
                #for i in range(1, 9):
                #    ax[0, i].imshow(x.cpu().numpy()[0, 0, i], cmap='plasma')
                #    ax[1, i].imshow(torch.argmax(y, dim=1).cpu()[0], cmap='gray')
                #plt.show()

                #print(math.degrees(data['parameters'][:, 0]))
                #print(data['parameters'][:, 1])
                #print(data['parameters'][:, 2])
                #for i in range(12):
                #    fig, ax = plt.subplots(1, 2)
                #    ax[0].imshow(x[0, 0, i].cpu().numpy(), cmap='gray')
                #    ax[1].imshow(torch.argmax(y, dim=1, keepdim=True).cpu().numpy()[0, 0, i], cmap='gray')
                #    figManager = plt.get_current_fig_manager()
                #    figManager.window.showMaximized()
                #    plt.show()

                self.model.train()
                out = self.model(x)

                pred = out['pred']
                reconstructed = out['reconstructed']
                decoder_sm = out['decoder_sm']
                reconstruction_sm = out['reconstruction_sm']
                out_angle = out['angle']

                pred = [torch.nn.functional.softmax(x.squeeze(1), dim=1) for x in pred if x is not None]

                rotation_loss = 0
                if self.cropping_network:
                    parameters = data['parameters']

                    #fig, ax = plt.subplots(2, 12)
                    #print(x.shape)
                    #for i in range(12):
                    #    print(math.degrees(parameters[0, 0]))
                    #    ax[0, i].imshow(x.cpu().numpy()[0, 0, i], cmap='gray')
                    #    ax[1, i].imshow(torch.argmax(y, dim=1).cpu().numpy()[0, i], cmap='gray')
                    #plt.show()

                    rotation_loss = self.rotation_loss(out_angle, parameters[:, 0])

                    rotation_reconstruction_loss = 0
                    for i in range(x.shape[2]):
                        pred_rotated = self.rotate_image_with_center(x[:, :, i, :, :], out_angle, parameters)
                        gt_rotated = self.rotate_image_with_center(x[:, :, i, :, :], parameters[:, 0], parameters)
                        rotation_reconstruction_loss += self.reconstruction_loss(pred_rotated, gt_rotated)
                    rotation_reconstruction_loss = rotation_reconstruction_loss / x.shape[2]

                    self.writer.add_scalar('Iteration/Rotation loss', rotation_loss, iter_nb)
                    self.writer.add_scalar('Iteration/Rotation reconstruction loss', rotation_reconstruction_loss, iter_nb)
                    rotation_loss = (self.rotation_loss_weight * rotation_loss) + (self.reconstruction_rotation_loss_weight * rotation_reconstruction_loss)
                    self.writer.add_scalar('Iteration/Whole rotation loss', rotation_loss, iter_nb)

                similarity_loss = 0
                reconstruction_loss = 0
                segmentation_loss = 0
                if reconstructed is not None:

                    reconstruction_layer_loss = 0
                    for layer_pred, weight in zip(reconstructed, self.deep_supervision_weights):
                        computed_loss = self.reconstruction_loss(layer_pred, x)
                        reconstruction_layer_loss += (computed_loss * weight)
                    reconstruction_loss = reconstruction_layer_loss / len(reconstructed)
                    self.writer.add_scalar('Iteration/Reconstruction loss', reconstruction_loss, iter_nb)

                    similarity_loss = self.similarity_loss(decoder_sm, reconstruction_sm)
                    self.writer.add_scalar('Iteration/similarity loss', similarity_loss, iter_nb)

                layer_loss = 0
                for layer_pred, weight in zip(pred, self.deep_supervision_weights):
                    computed_loss = self.labeled_loss_object(layer_pred, y, iter_nb)
                    layer_loss += (computed_loss * weight)
                segmentation_loss = layer_loss

                seg_adv_loss = 0
                rec_adv_loss = 0
                lrs = {}
                if self.seg_discriminator is not None:
                    self.seg_discriminator.train()
                    self.rec_discriminator.train()

                    seg_adv_loss, seg_dis_loss = self.get_adversarial_loss(self.seg_discriminator, 
                                                                            self.seg_discriminator_optimizer, 
                                                                            self.seg_discriminator_scheduler, 
                                                                            real=torch.cat([x, y], dim=1), 
                                                                            fake=torch.cat([x, pred[-1]], dim=1),
                                                                            iter_nb=iter_nb)
                    rec_adv_loss, rec_dis_loss = self.get_adversarial_loss(self.rec_discriminator, 
                                                                            self.rec_discriminator_optimizer, 
                                                                            self.rec_discriminator_scheduler, 
                                                                            real=x, 
                                                                            fake=reconstructed[-1],
                                                                            iter_nb=iter_nb)

                    lrs['Discriminators'] = self.seg_discriminator_optimizer.param_groups[0]['lr']
                    
                    self.writer.add_scalar('Iteration/Segmentation adversarial loss', seg_adv_loss, iter_nb)
                    self.writer.add_scalar('Iteration/Reconstruction adversarial loss', rec_adv_loss, iter_nb)
                    self.writer.add_scalar('Iteration/Segmentation Discriminator loss', seg_dis_loss, iter_nb)
                    self.writer.add_scalar('Iteration/Reconstruction Discriminator loss', rec_dis_loss, iter_nb)
                
                loss = segmentation_loss + (self.reconstruction_loss_weight * reconstruction_loss) + (self.similarity_loss_weight * similarity_loss) + (self.adversarial_loss_weight * seg_adv_loss) + (self.adversarial_loss_weight * rec_adv_loss) + rotation_loss

                self.writer.add_scalar('Iteration/segmentation loss', segmentation_loss, iter_nb)
                self.writer.add_scalar('Iteration/Training loss', loss, iter_nb)
                
                # Backpropagation
                self.model_optimizer.zero_grad()
                loss.backward()

                if iter_nb % self.plot_gradient_iter_number == 0:
                    plot_grad_flow(self.model.named_parameters(), step=1)
                self.model_optimizer.step()
                self.model_scheduler.step()

                lrs['Model'] = self.model_optimizer.param_groups[0]['lr']

                self.writer.add_scalars('Iteration/Learning rates', lrs, iter_nb)

                if batch % self.logging_loss_iteration_number == 0:
                    loss = loss.item()
                    self.console_logger.info(f"Training loss: {loss:>7f}")

                if batch % self.logging_metrics_iteration_number == 0:
                    max_memory_allocated = torch.cuda.max_memory_allocated(device=self.device)
                    self.console_logger.info(f"Max GPU Memory allocated: {max_memory_allocated / 10e8} Gb")
                    self.file_logger.info(f"Max GPU Memory allocated: {max_memory_allocated / 10e8} Gb")
                    if self.logits:
                        val_metrics, images = self.validation_loop_acdc_3d(val_or_train='val', dimensions=3)
                    else:
                        val_metrics, images = self.validation_loop_acdc_3d(val_or_train='val', dimensions=3)
                    self.validation_logging(val_metrics, images, val_or_train='val')

                    if self.compute_overfitting:
                        if self.logits:
                            train_metrics, images = self.validation_loop_acdc_3d(val_or_train='train', dimensions=3)
                        else:
                            train_metrics, images = self.validation_loop_acdc_3d(val_or_train='train', dimensions=3)
                        self.validation_logging(train_metrics, images, val_or_train='train')

                #self.writer.add_scalars('Iteration/Labeled individual loss weights', self.labeled_loss_object.get_loss_weight(), iter_nb)
                self.labeled_loss_object.update_weight()