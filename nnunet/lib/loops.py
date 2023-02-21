import torch
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import numpy as np
from test import get_metrics
import sys
import random
from data_augmentation import process_my_augment
from training_utils import get_validation_images_lib, log_gan_metrics, log_metrics, batched_distance_transform, log_ssim
from math import erf, ceil, floor
import math
from torch.nn.functional import adaptive_avg_pool2d
from pytorch_fid.inception import InceptionV3
from pytorch_fid.fid_score import calculate_frechet_distance
from loss import logisticGradientPenalty
import cv2 as cv
from torch.nn.functional import interpolate
import bisect
from torchvision.transforms.functional import resize
from utils import update_average, plot_grad_flow, compute_losses, GetProbabilitiesMatrix
import copy
from spatial_transformer import get_rotation_batched_matrices, get_scaling_batched_matrices, get_translation_batched_matrices
import torch.nn.functional as F
from skimage.measure import regionprops
from torchvision.ops import masks_to_boxes
from torchvision.transforms.functional import center_crop, crop
import global_variables
import torch.nn as nn
from ssim import ssim
from dataset_utils import normalize_0_1
from matplotlib import cm

def reverse_transform(pred, metadata):
    tx = metadata['tx']
    ty = metadata['ty']
    scale = metadata['scale']
    angle = metadata['angle']

    r = get_rotation_batched_matrices(-angle)
    t = get_translation_batched_matrices(-tx, -ty)
    s = get_scaling_batched_matrices(1 / scale)
    theta = torch.bmm(torch.bmm(s, r), t)[:, :-1]

    grid = F.affine_grid(theta, pred.size())
    pred = F.grid_sample(pred, grid, mode='nearest')
    return pred


def transform(pred, metadata, mode):
    #tx = metadata['tx']
    #ty = metadata['ty']
    pred = torch.argmax(pred, dim=1, keepdim=True).float()
    tx = torch.zeros_like(metadata['tx'])
    ty = torch.zeros_like(metadata['ty'])
    scale = metadata['scale']
    angle = metadata['angle']

    r = get_rotation_batched_matrices(angle)
    t = get_translation_batched_matrices(tx, ty)
    s = get_scaling_batched_matrices(scale)
    theta = torch.bmm(torch.bmm(t, r), s)[:, :-1]

    grid = F.affine_grid(theta, pred.size())
    pred = F.grid_sample(pred, grid, mode=mode)
    return pred

def validation_loop_lib(model, val_dataloader):
    model.eval()
    with torch.no_grad():
        class_dice_sum = np.array([0, 0, 0], dtype=np.float64)
        class_hd_sum = np.array([0, 0, 0], dtype=np.float64)
        for data in tqdm(val_dataloader, desc='Validation iteration: ', position=2):
            x_vals, y_vals = data['x'], data['y']
            x_vals = x_vals.permute((1, 0, 2, 3, 4))
            y_vals = y_vals.permute((1, 0, 2, 3, 4))
            pred_val_list = model(x_vals)
            class_dice_sum_frames = np.array([0, 0, 0], dtype=np.float64)
            class_hd_sum_frames = np.array([0, 0, 0], dtype=np.float64)
            divider_frame_hd = np.array([1e-10, 1e-10, 1e-10], dtype=np.float64)
            for ds_x, frame_y in zip(pred_val_list, y_vals):
                class_dice, class_hd = get_metrics(frame_y, ds_x[-1])
                class_dice_sum_frames += class_dice
                for i in range(3):
                    if not math.isnan(class_hd[i]):
                        class_hd_sum_frames[i] += class_hd[i]
                        divider_frame_hd[i] += 1
            class_dice_sum += class_dice_sum_frames / len(pred_val_list)
            class_hd_sum += class_hd_sum_frames / divider_frame_hd
        class_dices = class_dice_sum / len(val_dataloader)
        class_hds = class_hd_sum / len(val_dataloader)
        return class_dices, class_hds

def get_fid_stats(pred):
    # If model output is not scalar, apply global spatial average pooling.
    # This happens if you choose a dimensionality not equal 2048.
    if pred.size(2) != 1 or pred.size(3) != 1:
        pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
    pred = pred.squeeze(3).squeeze(2).cpu().numpy()
    mu = np.mean(pred, axis=0)
    sigma = np.cov(pred, rowvar=False)
    return mu, sigma

class Loops(object):
    def __init__(self,
                 labeled_train_dataloader,
                 validation_dataloader,
                 validation_random_dataloader,
                 val_dataloader_subset, 
                 model, 
                 optimizer1, 
                 optimizer2, 
                 spatial_transformer_optimizer,
                 console_logger, 
                 file_logger, 
                 logging_metrics_iteration_number, 
                 logging_loss_iteration_number, 
                 device,
                 localization_loss_object,
                 unlabeled_loss_weight_end,
                 nb_iterations_per_epoch,
                 total_nb_of_iterations,
                 total_nb_epochs, 
                 similarity_down_scale,
                 deep_supervision_weights, 
                 labeled_loss_object, 
                 unlabeled_loss_object, 
                 spatial_transformer_loss_object,
                 plot_gradient_iter_number,
                 scheduler1, 
                 scheduler2,
                 spatial_transformer_scheduler,
                 spatial_transformer,
                 reconstruction_loss_weight,
                 similarity_loss_weight,
                 writer, 
                 val_stride,
                 img_size,
                 big_img_size,
                 save_path='out'):

                 self.spatial_transformer = spatial_transformer
                 self.plot_gradient_iter_number = plot_gradient_iter_number
                 self.val_stride = val_stride
                 self.current_epoch = 0
                 self.img_size = img_size
                 self.save_path = save_path
                 self.total_nb_epochs = total_nb_epochs
                 self.writer = writer
                 self.scheduler1 = scheduler1
                 self.scheduler2 = scheduler2
                 self.spatial_transformer_scheduler = spatial_transformer_scheduler
                 self.deep_supervision_weights = deep_supervision_weights
                 self.validation_dataloader = validation_dataloader
                 self.validation_random_dataloader = validation_random_dataloader
                 self.labeled_train_dataloader = labeled_train_dataloader
                 self.val_dataloader_subset = val_dataloader_subset
                 self.model = model
                 self.optimizer1 = optimizer1
                 self.optimizer2 = optimizer2
                 self.spatial_transformer_optimizer = spatial_transformer_optimizer
                 self.labeled_loss_object = labeled_loss_object
                 self.unlabeled_loss_object = unlabeled_loss_object
                 self.spatial_transformer_loss_object = spatial_transformer_loss_object
                 self.localization_loss_object = localization_loss_object
                 self.console_logger = console_logger
                 self.file_logger = file_logger
                 self.logging_metrics_iteration_number = logging_metrics_iteration_number
                 self.logging_loss_iteration_number = logging_loss_iteration_number
                 self.device = device
                 self.nb_iterations_per_epoch = nb_iterations_per_epoch
                 self.total_nb_of_iterations = total_nb_of_iterations
                 self.current_unlabeled_loss_weight = 0
                 self.unlabeled_loss_weight_end = unlabeled_loss_weight_end
                 self.big_img_size = big_img_size
                 self.reconstruction_loss = nn.MSELoss()
                 self.similarity_loss = nn.L1Loss()
                 self.reconstruction_loss_weight = reconstruction_loss_weight
                 self.similarity_loss_weight = similarity_loss_weight
                 self.get_probability_matrix = GetProbabilitiesMatrix(similarity_down_scale)
    
    def get_validation_images_acdc(self):
        self.model.eval()
        with torch.no_grad():
            data = next(iter(self.validation_random_dataloader))
            x = data['x']
            y = data['y']
            y = torch.argmax(y.squeeze(0), dim=0)
            embedding = torch.nn.Embedding(4, 3)
            embedding.weight.data = torch.tensor([[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255]], device=self.device)

            out = self.model(x)

            reconstructed = out['reconstructed']
            decoder_sm = out['decoder_sm']
            reconstruction_sm = out['reconstruction_sm']
            if reconstructed is not None:
                reconstructed = reconstructed[-1]
                reconstructed = reconstructed.squeeze()
                reconstructed = cv.normalize(reconstructed.cpu().numpy(), None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F).astype(np.uint8)[..., np.newaxis]
                decoder_sm = decoder_sm.squeeze()[0].view(28, 28)[None, None, :, :]
                reconstruction_sm = reconstruction_sm.squeeze()[0].view(28, 28)[None, None, :, :]
                decoder_sm = interpolate(input=decoder_sm, scale_factor=8, mode='bilinear', antialias=True).squeeze()
                reconstruction_sm = interpolate(input=reconstruction_sm, scale_factor=8, mode='bilinear', antialias=True).squeeze()
                decoder_sm = normalize_0_1(decoder_sm)
                reconstruction_sm = normalize_0_1(reconstruction_sm)
                decoder_sm = cm.plasma(decoder_sm.cpu().numpy())[:, :, :-1]
                reconstruction_sm = cm.plasma(reconstruction_sm.cpu().numpy())[:, :, :-1]
                decoder_sm = cv.normalize(decoder_sm, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F).astype(np.uint8)
                reconstruction_sm = cv.normalize(reconstruction_sm, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F).astype(np.uint8)


            pred = out['pred'][-1]
            pred = torch.argmax(pred.squeeze(0), dim=0)
            pred = embedding(pred)
                
            y = embedding(y)
            x = cv.normalize(x.squeeze().cpu().numpy(), None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F).astype(np.uint8)[..., np.newaxis]
            out_dict = {'pred': pred.cpu().numpy().astype(np.uint8), 'y': y.cpu().numpy().astype(np.uint8), 'x': x, 'reconstructed': reconstructed, 'decoder_sm': decoder_sm, 'reconstruction_sm': reconstruction_sm}
            return out_dict
    
    def get_validation_images_acdc_autoencoder(self):
        self.model.eval()
        with torch.no_grad():
            y = next(iter(self.validation_random_dataloader))
            #embedding = torch.nn.Embedding(4, 3)
            #embedding.weight.data = torch.tensor([[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255]], device=self.device)

            out = self.model(y)

            pred = out[-1]
            #pred = embedding(pred)
                
            #y = torch.argmax(y.squeeze(0), dim=0)
            #y = embedding(y)
            pred = cv.normalize(pred.squeeze().cpu().numpy(), None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F).astype(np.uint8)[..., np.newaxis]
            y = cv.normalize(y.squeeze().cpu().numpy(), None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F).astype(np.uint8)[..., np.newaxis]
            out_dict = {'pred': pred, 'y': y}
            return out_dict
    
    def validation_loop_acdc(self):
        self.model.eval()
        with torch.no_grad():
            #losses = torch.tensor([0, 0, 0], dtype=torch.float64)
            class_ssim_sum = 0
            class_dice_sum = np.array([0, 0, 0], dtype=np.float64)
            class_hd_sum = np.array([0, 0, 0], dtype=np.float64)
            divider_hd = np.array([1e-10, 1e-10, 1e-10], dtype=np.float64)
            for data in tqdm(self.val_dataloader_subset, desc='Validation iteration: ', position=2):
                x, y_true = data['x'], data['y']
                y_true = torch.argmax(y_true.squeeze(0), dim=0)

                out = self.model(x)
                pred = out['pred'][-1]
                reconstructed = out['reconstructed'][-1]

                #out = self.model(x)
                #losses += compute_losses(self.labeled_loss_object, self.reconstruction_loss, self.similarity_loss, x, y_true, out, data['distance_map'])
                #y_true = torch.argmax(y_true.squeeze(0), dim=0)
                #pred = out['pred'][-1]


                pred = torch.argmax(pred.squeeze(0), dim=0)
                
                class_dice, class_hd = get_metrics(y_true, pred)
                class_dice_sum += class_dice
                class_ssim_sum += ssim(x, reconstructed)
                for i in range(3):
                    if not math.isnan(class_hd[i]):
                        class_hd_sum[i] += class_hd[i]
                        divider_hd[i] += 1
            class_dices = class_dice_sum / len(self.val_dataloader_subset)
            class_hds = class_hd_sum / divider_hd
            class_ssim = class_ssim_sum / len(self.val_dataloader_subset)
            return class_dices, class_hds, class_ssim
    
    def validation_loop_acdc_autoencoder(self):
        self.model.eval()
        with torch.no_grad():
            class_ssim_sum = 0
            for label in tqdm(self.val_dataloader_subset, desc='Validation iteration: ', position=2):

                pred = self.model(label)[-1]
                
                class_ssim_sum += ssim(label, pred)
            class_ssim = class_ssim_sum / len(self.val_dataloader_subset)
            return class_ssim
    
    def main_loop_acdc_semi_supervised(self, unlabeled_train_dataloader1, unlabeled_train_dataloader2):
        for idx, t in enumerate(tqdm(range(self.total_nb_epochs), desc='Epoch: ', position=0)):
            max_memory_allocated = torch.cuda.max_memory_allocated(device=self.device)
            self.console_logger.info(f"Max GPU Memory allocated: {max_memory_allocated / 10e8} Gb")
            self.file_logger.info(f"Max GPU Memory allocated: {max_memory_allocated / 10e8} Gb")
            self.current_epoch = idx
            self.train_loop_acdc_semi_supervised_my_augment(unlabeled_train_dataloader1=unlabeled_train_dataloader1, unlabeled_train_dataloader2=unlabeled_train_dataloader2)
            if idx % self.val_stride == 0:
                class_dice, class_hd = self.validation_loop_acdc()
                images = self.get_validation_images_acdc()
                self.validation_logging(images, class_dice, class_hd)
    
    def main_loop_acdc_supervised(self):
        for idx, t in enumerate(tqdm(range(self.total_nb_epochs), desc='Epoch: ', position=0)):
            max_memory_allocated = torch.cuda.max_memory_allocated(device=self.device)
            self.console_logger.info(f"Max GPU Memory allocated: {max_memory_allocated / 10e8} Gb")
            self.file_logger.info(f"Max GPU Memory allocated: {max_memory_allocated / 10e8} Gb")
            self.current_epoch = idx
            self.train_loop_acdc_supervised()
            if idx % self.val_stride == 0:
                class_dice, class_hd, class_ssim = self.validation_loop_acdc()
                images = self.get_validation_images_acdc()
                self.validation_logging(images, class_dice, class_hd, class_ssim)
                torch.save(self.model.state_dict(), os.path.join(self.save_path, 'model_weights.pth'))
                self.console_logger.info(f"Saved model to {self.save_path}")
    
    def main_loop_acdc_autoencoder(self):
        for idx, t in enumerate(tqdm(range(self.total_nb_epochs), desc='Epoch: ', position=0)):
            max_memory_allocated = torch.cuda.max_memory_allocated(device=self.device)
            self.console_logger.info(f"Max GPU Memory allocated: {max_memory_allocated / 10e8} Gb")
            self.file_logger.info(f"Max GPU Memory allocated: {max_memory_allocated / 10e8} Gb")
            self.current_epoch = idx
            self.train_loop_acdc_autoencoder()
            if idx % self.val_stride == 0:
                ssim_metric = self.validation_loop_acdc_autoencoder()
                images = self.get_validation_images_acdc_autoencoder()
                self.validation_logging_autoencoder(images, ssim_metric)
                torch.save(self.model.state_dict(), os.path.join(self.save_path, 'model_weights.pth'))
                self.console_logger.info(f"Saved model to {self.save_path}")
    
    def main_loop_acdc_supervised_spatial_transformer(self):
        for idx, t in enumerate(tqdm(range(self.total_nb_epochs), desc='Epoch: ', position=0)):
            max_memory_allocated = torch.cuda.max_memory_allocated(device=self.device)
            self.console_logger.info(f"Max GPU Memory allocated: {max_memory_allocated / 10e8} Gb")
            self.file_logger.info(f"Max GPU Memory allocated: {max_memory_allocated / 10e8} Gb")
            self.current_epoch = idx
            self.train_loop_acdc_supervised_spatial_transformer()
            if idx % self.val_stride == 0:
                class_dice, class_hd = self.validation_loop_acdc()
                images = self.get_validation_images_acdc()
                self.validation_logging(images, class_dice, class_hd)
    
    def main_loop_lib(self):
        for idx, t in enumerate(tqdm(range(self.total_nb_epochs), desc='Epoch: ', position=0)):
            max_memory_allocated = torch.cuda.max_memory_allocated(device=self.device)
            self.console_logger.info(f"Max GPU Memory allocated: {max_memory_allocated / 10e8} Gb")
            self.file_logger.info(f"Max GPU Memory allocated: {max_memory_allocated / 10e8} Gb")
            self.current_epoch = idx
            self.train_loop_lib()
            if idx % self.val_stride == 0:
                class_dice, class_hd, ssim_metric = validation_loop_lib(self.model, self.validation_dataloader)
                images = get_validation_images_lib(self.model, self.validation_random_dataloader, self.device)
                self.validation_logging(images, class_dice, class_hd, ssim_metric)

    def validation_logging(self, images, class_dice, class_hd, class_ssim):
        self.writer.add_image('Epoch/Image', images['x'], self.current_epoch, dataformats='HWC')
        self.writer.add_image('Epoch/Ground truth', images['y'], self.current_epoch, dataformats='HWC')
        self.writer.add_image('Epoch/Prediction', images['pred'], self.current_epoch, dataformats='HWC')
        if images['reconstructed'] is not None:
            self.writer.add_image('Epoch/Reconstructed', images['reconstructed'], self.current_epoch, dataformats='HWC')
            self.writer.add_image('Epoch/Decoder_sm', images['decoder_sm'], self.current_epoch, dataformats='HWC')
            self.writer.add_image('Epoch/Reconstruction_sm', images['reconstruction_sm'], self.current_epoch, dataformats='HWC')
        log_metrics(self.console_logger, self.writer, class_dice, class_hd, class_ssim, self.current_epoch, 'Epoch')
        log_metrics(self.file_logger, self.writer, class_dice, class_hd, class_ssim, self.current_epoch, 'Epoch')

    def validation_logging_autoencoder(self, images, ssim_metric):
        self.writer.add_image('Epoch/Ground truth', images['y'], self.current_epoch, dataformats='HWC')
        self.writer.add_image('Epoch/Prediction', images['pred'], self.current_epoch, dataformats='HWC')
        log_ssim(self.console_logger, self.writer, ssim_metric, self.current_epoch, 'Epoch')
        log_ssim(self.file_logger, self.writer, ssim_metric, self.current_epoch, 'Epoch')
    
    def get_unlabeled_weight(self, iteration_nb):
        iteration_percent = iteration_nb / self.total_nb_of_iterations
        return self.unlabeled_loss_weight_end * (erf(iteration_percent) / erf(1))

    def train_loop_acdc_semi_supervised_cutmix(self, unlabeled_train_dataloader1, unlabeled_train_dataloader2):
        
        labeled_iterator = iter(self.labeled_train_dataloader)
        unlabeled_iterator1 = iter(unlabeled_train_dataloader1)
        unlabeled_iterator2 = iter(unlabeled_train_dataloader2)
    
        for i in tqdm(range(1, self.nb_iterations_per_epoch + 1), desc='Training iteration: ', position=1):

            self.model.train()
            iter_nb = (self.current_epoch * self.nb_iterations_per_epoch) + i
            unlabeled_minibatch1 = next(unlabeled_iterator1)
            unlabeled_minibatch2 = next(unlabeled_iterator2)
            try:
                labeled_minibatch = next(labeled_iterator)
            except StopIteration:
                labeled_iterator = iter(self.labeled_train_dataloader)
                labeled_minibatch = next(labeled_iterator)

            x_l = labeled_minibatch['x']
            y_l = labeled_minibatch['y']
            distance_map_l = labeled_minibatch['distance_map']
            x_u1 = unlabeled_minibatch1['x']
            x_u2 = unlabeled_minibatch2['x']
            mask = unlabeled_minibatch1['mask']

            unsup_imgs_mixed = x_u1 * (1 - mask) + x_u2 * mask
            with torch.no_grad():
                logits_u1_1 = self.model(x_u1, 1)[-1]
                logits_u2_1 = self.model(x_u2, 1)[-1]
                logits_u1_2 = self.model(x_u1, 2)[-1]
                logits_u2_2 = self.model(x_u2, 2)[-1]
                logits_u1_1 = logits_u1_1.detach()
                logits_u2_1 = logits_u2_1.detach()
                logits_u1_2 = logits_u1_2.detach()
                logits_u2_2 = logits_u2_2.detach()
                #n1 = torch.count_nonzero(torch.max(logits_u1_1, dim=1)[0] > self.thresh)
                #n2 = torch.count_nonzero(torch.max(logits_u2_1, dim=1)[0] > self.thresh)
                #n3 = torch.count_nonzero(torch.max(logits_u1_2, dim=1)[0] > self.thresh)
                #n4 = torch.count_nonzero(torch.max(logits_u2_2, dim=1)[0] > self.thresh)
                #weight_unlabeled = ((n1 + n2 + n3 + n4) / (logits_u1_1[:, 0, :, :].numel() * 4))
            
            unlabeled_loss = 0
            mixed_logits1 = logits_u1_1 * (1 - mask) + logits_u2_1 * mask
            label_1 = torch.argmax(mixed_logits1, dim=1)
            label_1 = torch.nn.functional.one_hot(label_1, num_classes=4).permute(0, 3, 1, 2)
            label_1 = label_1.long()
            distance_map_label_1 = batched_distance_transform(label_1.cpu().numpy()).to(self.device) if self.unlabeled_loss_object.use_dist_maps else None
            mixed_logits2 = logits_u1_2 * (1 - mask) + logits_u2_2 * mask
            label_2 = torch.argmax(mixed_logits2, dim=1)
            label_2 = torch.nn.functional.one_hot(label_2, num_classes=4).permute(0, 3, 1, 2)
            label_2 = label_2.long()
            distance_map_label_2 = batched_distance_transform(label_2.cpu().numpy()).to(self.device) if self.unlabeled_loss_object.use_dist_maps else None
            logits_1_ds = self.model(unsup_imgs_mixed, 1)
            logits_2_ds = self.model(unsup_imgs_mixed, 2)
            for layer_pred1, layer_pred2, weight in zip(logits_1_ds, logits_2_ds, self.deep_supervision_weights):
                unlabeled_computed_loss1 = self.unlabeled_loss_object(layer_pred1, label_2, iter_nb, dist_maps=distance_map_label_2)
                unlabeled_computed_loss2 = self.unlabeled_loss_object(layer_pred2, label_1, iter_nb, dist_maps=distance_map_label_1)
                unlabeled_loss += (unlabeled_computed_loss1 + unlabeled_computed_loss2) * weight
            unlabeled_loss = unlabeled_loss / len(logits_1_ds)

            logits_l_1 = self.model(x_l, 1)
            logits_l_2 = self.model(x_l, 2)

            labeled_loss = 0
            for layer_pred1, layer_pred2, weight in zip(logits_l_1, logits_l_2, self.deep_supervision_weights):
                labeled_computed_loss1 = self.labeled_loss_object(layer_pred1, y_l, iter_nb, dist_maps=distance_map_l)
                labeled_computed_loss2 = self.labeled_loss_object(layer_pred2, y_l, iter_nb, dist_maps=distance_map_l)
                labeled_loss += (labeled_computed_loss1 + labeled_computed_loss2) * weight
            labeled_loss = labeled_loss / len(logits_l_1)

            loss = labeled_loss + (self.current_unlabeled_loss_weight * unlabeled_loss)
            self.optimizer1.zero_grad()
            self.optimizer2.zero_grad()
            loss.backward()
            self.optimizer1.step()
            self.optimizer2.step()

            self.scheduler1.step()
            self.scheduler2.step()
            self.writer.add_scalar('Iteration/Unlabeled loss weight', self.current_unlabeled_loss_weight, iter_nb)
            self.writer.add_scalar('Iteration/Learning rate 1', self.optimizer1.param_groups[0]['lr'], iter_nb)
            
            self.writer.add_scalar('Iteration/Labeled loss', labeled_loss, iter_nb)
            self.writer.add_scalar('Iteration/Unlabeled loss', unlabeled_loss, iter_nb)
            self.writer.add_scalar('Iteration/Training loss', loss, iter_nb)
            if i % self.save_iteration_number == 0:
                torch.save(self.model.state_dict(), os.path.join(self.save_path, 'model_weights.pth'))
                self.console_logger.info(f"Saved model to {self.save_path}")
            
            if i % self.logging_loss_iteration_number == 0:
                loss = loss.item()
                self.console_logger.info(f"Training loss: {loss:>7f}")

            if i % self.logging_metrics_iteration_number == 0:
                max_memory_allocated = torch.cuda.max_memory_allocated(device=self.device)
                self.console_logger.info(f"Max GPU Memory allocated: {max_memory_allocated / 10e8} Gb")
                self.file_logger.info(f"Max GPU Memory allocated: {max_memory_allocated / 10e8} Gb")
                class_dices, class_hds = self.validation_loop_acdc()
                log_metrics(self.console_logger, self.writer, class_dices, class_hds, iter_nb, 'Iteration')
                log_metrics(self.file_logger, self.writer, class_dices, class_hds, iter_nb, 'Iteration')
            self.writer.add_scalars('Iteration/Labeled individual loss weights', self.labeled_loss_object.get_loss_weight(), iter_nb)
            self.writer.add_scalars('Iteration/Unlabeled individual loss weights', self.unlabeled_loss_object.get_loss_weight(), iter_nb)
            self.labeled_loss_object.update_weight()
            self.unlabeled_loss_object.update_weight()
            self.current_unlabeled_loss_weight = self.get_unlabeled_weight(iter_nb)

    def train_loop_acdc_semi_supervised_my_augment(self, unlabeled_train_dataloader1, unlabeled_train_dataloader2):
            
            labeled_iterator = iter(self.labeled_train_dataloader)
            unlabeled_iterator1 = iter(unlabeled_train_dataloader1)
            unlabeled_iterator2 = iter(unlabeled_train_dataloader2)
        
            for i in tqdm(range(1, self.nb_iterations_per_epoch + 1), desc='Training iteration: ', position=1):

                self.model.train()
                iter_nb = (self.current_epoch * self.nb_iterations_per_epoch) + i
                unlabeled_minibatch1 = next(unlabeled_iterator1)
                unlabeled_minibatch2 = next(unlabeled_iterator2)
                try:
                    labeled_minibatch = next(labeled_iterator)
                except StopIteration:
                    labeled_iterator = iter(self.labeled_train_dataloader)
                    labeled_minibatch = next(labeled_iterator)

                x_l = labeled_minibatch['x']
                y_l = labeled_minibatch['y']
                distance_map_l = labeled_minibatch['distance_map']
                x_u1 = unlabeled_minibatch1['x']
                x_u2 = unlabeled_minibatch2['x']

                #unsup_imgs_mixed = x_u1 * (1 - mask) + x_u2 * mask
                with torch.no_grad():
                    logits_u1_1 = self.model(x_u1, 1)[-1]
                    logits_u2_1 = self.model(x_u2, 1)[-1]
                    logits_u1_2 = self.model(x_u1, 2)[-1]
                    logits_u2_2 = self.model(x_u2, 2)[-1]
                    logits_u1_1 = logits_u1_1.detach()
                    logits_u2_1 = logits_u2_1.detach()
                    logits_u1_2 = logits_u1_2.detach()
                    logits_u2_2 = logits_u2_2.detach()
                    #n1 = torch.count_nonzero(torch.max(logits_u1_1, dim=1)[0] > self.thresh)
                    #n2 = torch.count_nonzero(torch.max(logits_u2_1, dim=1)[0] > self.thresh)
                    #n3 = torch.count_nonzero(torch.max(logits_u1_2, dim=1)[0] > self.thresh)
                    #n4 = torch.count_nonzero(torch.max(logits_u2_2, dim=1)[0] > self.thresh)
                    #weight_unlabeled = ((n1 + n2 + n3 + n4) / (logits_u1_1[:, 0, :, :].numel() * 4))
                
                unlabeled_loss = 0
                label_u1_1 = torch.argmax(logits_u1_1, dim=1)
                label_u1_1 = torch.nn.functional.one_hot(label_u1_1, num_classes=4).permute(0, 3, 1, 2)
                label_u1_1 = label_u1_1.long()
                label_u2_1 = torch.argmax(logits_u2_1, dim=1)
                label_u2_1 = torch.nn.functional.one_hot(label_u2_1, num_classes=4).permute(0, 3, 1, 2)
                label_u2_1 = label_u2_1.long()
                new_image_1 = process_my_augment(x_u1, label_u1_1, x_u2, label_u2_1, self.device)
                distance_map_label_1 = batched_distance_transform(label_u1_1.cpu().numpy()).to(self.device) if self.unlabeled_loss_object.use_dist_maps else None
                label_u1_2 = torch.argmax(logits_u1_2, dim=1)
                label_u1_2 = torch.nn.functional.one_hot(label_u1_2, num_classes=4).permute(0, 3, 1, 2)
                label_u1_2 = label_u1_2.long()
                label_u2_2 = torch.argmax(logits_u2_2, dim=1)
                label_u2_2 = torch.nn.functional.one_hot(label_u2_2, num_classes=4).permute(0, 3, 1, 2)
                label_u2_2 = label_u2_2.long()
                #new_image_2 = process_my_augment(x_u1, label_u1_2, x_u2, label_u2_2, self.device)
                distance_map_label_2 = batched_distance_transform(label_u1_2.cpu().numpy()).to(self.device) if self.unlabeled_loss_object.use_dist_maps else None
                logits_1_ds = self.model(new_image_1, 1)
                logits_2_ds = self.model(new_image_1, 2)
                for layer_pred1, layer_pred2, weight in zip(logits_1_ds, logits_2_ds, self.deep_supervision_weights):
                    unlabeled_computed_loss1 = self.unlabeled_loss_object(layer_pred1, label_u1_2, iter_nb, dist_maps=distance_map_label_2)
                    unlabeled_computed_loss2 = self.unlabeled_loss_object(layer_pred2, label_u1_1, iter_nb, dist_maps=distance_map_label_1)
                    unlabeled_loss += (unlabeled_computed_loss1 + unlabeled_computed_loss2) * weight
                unlabeled_loss = unlabeled_loss / len(logits_1_ds)

                logits_l_1 = self.model(x_l, 1)
                logits_l_2 = self.model(x_l, 2)

                labeled_loss = 0
                for layer_pred1, layer_pred2, weight in zip(logits_l_1, logits_l_2, self.deep_supervision_weights):
                    labeled_computed_loss1 = self.labeled_loss_object(layer_pred1, y_l, iter_nb, dist_maps=distance_map_l)
                    labeled_computed_loss2 = self.labeled_loss_object(layer_pred2, y_l, iter_nb, dist_maps=distance_map_l)
                    labeled_loss += (labeled_computed_loss1 + labeled_computed_loss2) * weight
                labeled_loss = labeled_loss / len(logits_l_1)

                loss = labeled_loss + (self.current_unlabeled_loss_weight * unlabeled_loss)
                self.optimizer1.zero_grad()
                self.optimizer2.zero_grad()
                loss.backward()
                self.optimizer1.step()
                self.optimizer2.step()

                self.scheduler1.step()
                self.scheduler2.step()

                self.writer.add_scalar('Iteration/Unlabeled loss weight', self.current_unlabeled_loss_weight, iter_nb)
                self.writer.add_scalar('Iteration/Learning rate 1', self.optimizer1.param_groups[0]['lr'], iter_nb)

                self.writer.add_scalar('Iteration/Training loss', loss, iter_nb)
                self.writer.add_scalar('Iteration/Labeled loss', labeled_loss, iter_nb)
                self.writer.add_scalar('Iteration/Unlabeled loss', unlabeled_loss, iter_nb)
                if i % self.save_iteration_number == 0:
                    torch.save(self.model.state_dict(), os.path.join(self.save_path, 'model_weights.pth'))
                    self.console_logger.info(f"Saved model to {self.save_path}")
                
                if i % self.logging_loss_iteration_number == 0:
                    loss = loss.item()
                    self.console_logger.info(f"Training loss: {loss:>7f}")

                if i % self.logging_metrics_iteration_number == 0:
                    max_memory_allocated = torch.cuda.max_memory_allocated(device=self.device)
                    self.console_logger.info(f"Max GPU Memory allocated: {max_memory_allocated / 10e8} Gb")
                    self.file_logger.info(f"Max GPU Memory allocated: {max_memory_allocated / 10e8} Gb")
                    class_dices, class_hds = self.validation_loop_acdc()
                    log_metrics(self.console_logger, self.writer, class_dices, class_hds, iter_nb, 'Iteration')
                    log_metrics(self.file_logger, self.writer, class_dices, class_hds, iter_nb, 'Iteration')
                self.writer.add_scalars('Iteration/Labeled individual loss weights', self.labeled_loss_object.get_loss_weight(), iter_nb)
                self.writer.add_scalars('Iteration/Unlabeled individual loss weights', self.unlabeled_loss_object.get_loss_weight(), iter_nb)
                self.labeled_loss_object.update_weight()
                self.unlabeled_loss_object.update_weight()
                self.current_unlabeled_loss_weight = self.get_unlabeled_weight(iter_nb)
    
    def train_loop_acdc_autoencoder(self):
        nb_iters = len(self.labeled_train_dataloader)

        for idx, labeled_batch in enumerate(tqdm(self.labeled_train_dataloader, desc='Training iteration: ', position=1), 1):
            iter_nb = (self.current_epoch * nb_iters) + idx
            global_variables.global_iter = iter_nb
            self.model.train()
            out = self.model(labeled_batch)
            loss = 0
            
            layer_loss = 0
            for layer_pred, weight in zip(out, self.deep_supervision_weights):
                computed_loss = self.reconstruction_loss(layer_pred, labeled_batch)
                layer_loss += (computed_loss * weight)
            loss += layer_loss / len(out)

            # Backpropagation
            self.optimizer1.zero_grad()
            loss.backward()
            self.optimizer1.step()
            self.scheduler1.step()
            self.writer.add_scalar('Iteration/Learning rate', self.optimizer1.param_groups[0]['lr'], iter_nb)
            
            self.writer.add_scalar('Iteration/Training loss', loss, iter_nb)
            
            if idx % self.logging_loss_iteration_number == 0:
                loss = loss.item()
                self.console_logger.info(f"Training loss: {loss:>7f}")

            if idx % self.logging_metrics_iteration_number == 0:
                max_memory_allocated = torch.cuda.max_memory_allocated(device=self.device)
                self.console_logger.info(f"Max GPU Memory allocated: {max_memory_allocated / 10e8} Gb")
                self.file_logger.info(f"Max GPU Memory allocated: {max_memory_allocated / 10e8} Gb")
                ssim_metric = self.validation_loop_acdc_autoencoder()
                log_ssim(self.console_logger, self.writer, ssim_metric, self.current_epoch, 'Epoch')
                log_ssim(self.file_logger, self.writer, ssim_metric, self.current_epoch, 'Epoch')

            self.writer.add_scalars('Iteration/Labeled individual loss weights', self.labeled_loss_object.get_loss_weight(), iter_nb)
            self.labeled_loss_object.update_weight()
    
    def train_loop_acdc_supervised(self):
        
        nb_iters = len(self.labeled_train_dataloader)

        for batch, data in enumerate(tqdm(self.labeled_train_dataloader, desc='Training iteration: ', position=1), 1):
            iter_nb = (self.current_epoch * nb_iters) + batch
            global_variables.global_iter = iter_nb
            x, y, dist_map = data['x'], data['y'], data['distance_map']
            self.model.train()
            out = self.model(x)
            loss = 0

            pred = out['pred']
            #vq_loss = out['vq_loss']
            reconstructed = out['reconstructed']
            decoder_sm = out['decoder_sm']
            reconstruction_sm = out['reconstruction_sm']

            if reconstructed is not None:
                layer_loss = 0
                for layer_pred, weight in zip(reconstructed, self.deep_supervision_weights):
                    computed_loss = self.reconstruction_loss(layer_pred, x)
                    layer_loss += (computed_loss * weight)
                reconstruction_loss = layer_loss / len(reconstructed)

                similarity_loss = self.similarity_loss(decoder_sm, reconstruction_sm)

                #probability_matrix = self.get_probability_matrix(pred[-1])
                #probability_loss = self.similarity_loss(probability_matrix, decoder_sm)
                
                self.writer.add_scalar('Iteration/Reconstruction loss', reconstruction_loss, iter_nb)
                self.writer.add_scalar('Iteration/similarity loss', similarity_loss, iter_nb)
                #self.writer.add_scalar('Iteration/probability loss', probability_loss, iter_nb)
                #self.writer.add_scalar('Iteration/VQ loss', self.reconstruction_loss_weight * vq_loss, iter_nb)

                loss += (self.reconstruction_loss_weight * reconstruction_loss) + (self.similarity_loss_weight * similarity_loss)
                #loss += reconstruction_loss + similarity_loss
            
            layer_loss = 0
            for layer_pred, weight in zip(pred, self.deep_supervision_weights):
                computed_loss = self.labeled_loss_object(layer_pred, y, iter_nb, dist_maps=dist_map)
                layer_loss += (computed_loss * weight)
            loss += layer_loss / len(pred)

            # Backpropagation
            self.optimizer1.zero_grad()
            loss.backward()
            if iter_nb % self.plot_gradient_iter_number == 0:
                plot_grad_flow(self.model.named_parameters(), step=1)
            self.optimizer1.step()
            self.scheduler1.step()
            self.writer.add_scalar('Iteration/Learning rate', self.optimizer1.param_groups[0]['lr'], iter_nb)
            
            self.writer.add_scalar('Iteration/Training loss', loss, iter_nb)
            
            if batch % self.logging_loss_iteration_number == 0:
                loss = loss.item()
                self.console_logger.info(f"Training loss: {loss:>7f}")

            if batch % self.logging_metrics_iteration_number == 0:
                max_memory_allocated = torch.cuda.max_memory_allocated(device=self.device)
                self.console_logger.info(f"Max GPU Memory allocated: {max_memory_allocated / 10e8} Gb")
                self.file_logger.info(f"Max GPU Memory allocated: {max_memory_allocated / 10e8} Gb")
                class_dices, class_hds, ssim_metric = self.validation_loop_acdc()
                images = self.get_validation_images_acdc()
                self.validation_logging(images, class_dices, class_hds, ssim_metric)

            self.writer.add_scalars('Iteration/Labeled individual loss weights', self.labeled_loss_object.get_loss_weight(), iter_nb)
            self.labeled_loss_object.update_weight()


    def train_loop_acdc_supervised_spatial_transformer(self):
        
        nb_iters = len(self.labeled_train_dataloader)

        for batch, data in enumerate(tqdm(self.labeled_train_dataloader, desc='Training iteration: ', position=1), 1):
            x, y, dist_map = data['x'], data['y'], data['distance_map']
            iter_nb = (self.current_epoch * nb_iters) + batch

            loss = 0
            self.spatial_transformer.train()
            x_first, x_metadata = self.spatial_transformer(x)

            layer_loss = 0
            for layer_pred, weight in zip(x_first, self.deep_supervision_weights):
                computed_loss = self.spatial_transformer_loss_object(layer_pred, data['y_binary'], iter_nb, dist_maps=dist_map)
                layer_loss += (computed_loss * weight)
            loss += layer_loss / len(x_first)

            x_first = torch.argmax(x_first[-1], dim=1)

            boxes = masks_to_boxes(x_first).int()
            centroids = ((boxes[:, 0] + boxes[:, 2]) // 2, (boxes[:, 1] + boxes[:, 3]) // 2)
            boxes[:, 0] = torch.maximum(torch.tensor([0], device=self.device), centroids[0] - (self.img_size // 2))
            boxes[:, 1] = torch.maximum(torch.tensor([0], device=self.device), centroids[1] - (self.img_size // 2))
            boxes[:, 2] = torch.minimum(torch.tensor([self.big_img_size], device=self.device), centroids[0] + (self.img_size // 2))
            boxes[:, 3] = torch.minimum(torch.tensor([self.big_img_size], device=self.device), centroids[1] + (self.img_size // 2))

            indices = torch.nonzero(boxes[:, 0] == 0)
            boxes[:, 2][indices] = self.img_size

            indices = torch.nonzero(boxes[:, 1] == 0)
            boxes[:, 3][indices] = self.img_size

            indices = torch.nonzero(boxes[:, 2] == self.big_img_size)
            boxes[:, 0][indices] = self.big_img_size - self.img_size

            indices = torch.nonzero(boxes[:, 3] == self.big_img_size)
            boxes[:, 1][indices] = self.big_img_size - self.img_size

            x = torch.stack([x[i, :, boxes[i, 1]: boxes[i, 3], boxes[i, 0]:boxes[i, 2]] for i in range(x.size(0))], dim=0)
            y = torch.stack([y[i, :, boxes[i, 1]: boxes[i, 3], boxes[i, 0]:boxes[i, 2]] for i in range(y.size(0))], dim=0)
            dist_map = torch.stack([dist_map[i, :, boxes[i, 1]: boxes[i, 3], boxes[i, 0]:boxes[i, 2]] for i in range(dist_map.size(0))], dim=0)

            if batch > 230:
                fig, ax = plt.subplots(2, 2)
                ax[0, 0].imshow(x_first.cpu()[0], cmap='gray')
                ax[0, 1].imshow(x.cpu()[0, 0], cmap='gray')
                ax[1, 0].imshow(torch.argmax(y, dim=1).cpu()[0], cmap='gray')
                ax[1, 1].imshow(dist_map.cpu()[0, 1], cmap='jet')
                plt.show()

            x = transform(x, x_metadata, 'bilinear')
            y = transform(y, x_metadata, 'nearest').squeeze().long()
            y = F.one_hot(y, num_classes=4).permute(0, 3, 1, 2).float()
            dist_map = torch.cat([transform(dist_map[:, i].unsqueeze(dim=1), x_metadata, 'bilinear') for i in range(dist_map.size(1))], dim=1)

            loss += self.localization_loss_object(x_metadata, data['metadata'], iter_nb)

            self.model.train()
            pred = self.model(x)
                
            layer_loss = 0
            for layer_pred, weight in zip(pred, self.deep_supervision_weights):
                computed_loss = self.labeled_loss_object(layer_pred, y, iter_nb, dist_maps=dist_map)
                layer_loss += (computed_loss * weight)
            loss += layer_loss / len(pred)

            # Backpropagation
            self.spatial_transformer_optimizer.zero_grad()
            self.optimizer1.zero_grad()
            loss.backward()

            self.optimizer1.step()
            self.spatial_transformer_optimizer.step()

            self.scheduler1.step()
            self.spatial_transformer_scheduler.step()
            self.writer.add_scalars('Iteration/Learning rate', {'model1': self.optimizer1.param_groups[0]['lr'], 'spatial_transformer': self.spatial_transformer_optimizer.param_groups[0]['lr']}, iter_nb)
            
            self.writer.add_scalar('Iteration/Training loss', loss, iter_nb)
            if batch % self.save_iteration_number == 0:
                torch.save(self.model.state_dict(), os.path.join(self.save_path, 'model_weights.pth'))
                self.console_logger.info(f"Saved model to {self.save_path}")
            
            if batch % self.logging_loss_iteration_number == 0:
                loss = loss.item()
                self.console_logger.info(f"Training loss: {loss:>7f}")

            if batch % self.logging_metrics_iteration_number == 0:
                max_memory_allocated = torch.cuda.max_memory_allocated(device=self.device)
                self.console_logger.info(f"Max GPU Memory allocated: {max_memory_allocated / 10e8} Gb")
                self.file_logger.info(f"Max GPU Memory allocated: {max_memory_allocated / 10e8} Gb")
                class_dices, class_hds = self.validation_loop_acdc()
                log_metrics(self.console_logger, self.writer, class_dices, class_hds, iter_nb, 'Iteration')
                log_metrics(self.file_logger, self.writer, class_dices, class_hds, iter_nb, 'Iteration')

            self.writer.add_scalars('Iteration/Labeled individual loss weights', self.labeled_loss_object.get_loss_weight(), iter_nb)
            self.labeled_loss_object.update_weight()

    def train_loop_lib(self):
        nb_iters = len(self.labeled_train_dataloader)
        for batch, data in enumerate(tqdm(self.labeled_train_dataloader, desc='Training iteration: ', position=1), 1):
            x, y = data['x'], data['y']
            iter_nb = (self.current_epoch * nb_iters) + batch
            self.model.train()
            loss = 0
            x = x.permute((1, 0, 2, 3, 4))
            y = y.permute((1, 0, 2, 3, 4))
            preds = self.model(x)

            for pred, y_true, distance_map in zip(preds, y, data['distance_map']):
                layer_loss = 0
                for layer_pred, weight in zip(pred, self.deep_supervision_weights):
                    computed_loss = self.labeled_loss_object(layer_pred, y_true, iter_nb, dist_maps=distance_map)
                    #dice = compute_dice(layer_pred, y_true, smoothing=smoothing, device=device)
                    #loss = 1 - dice
                    layer_loss += (computed_loss * weight)
                loss += layer_loss / len(pred)
            loss = loss / len(y)

            # Backpropagation
            self.optimizer1.zero_grad()
            loss.backward()
            self.optimizer1.step()
            self.scheduler1.step()
            self.writer.add_scalar('Iteration/Learning rate', self.optimizer1.param_groups[0]['lr'], iter_nb)
            
            self.writer.add_scalar('Iteration/Training loss', loss, iter_nb)
            if batch % self.save_iteration_number == 0:
                torch.save(self.model.state_dict(), os.path.join(self.save_path, 'model_weights.pth'))
                self.console_logger.info(f"Saved model to {self.save_path}")
            
            if batch % self.logging_loss_iteration_number == 0:
                loss = loss.item()
                self.console_logger.info(f"Training loss: {loss:>7f}")

            if batch % self.logging_metrics_iteration_number == 0:
                max_memory_allocated = torch.cuda.max_memory_allocated(device=self.device)
                self.console_logger.info(f"Max GPU Memory allocated: {max_memory_allocated / 10e8} Gb")
                self.file_logger.info(f"Max GPU Memory allocated: {max_memory_allocated / 10e8} Gb")
                class_dices, class_hds = validation_loop_lib(model=self.model, val_dataloader=self.val_dataloader_subset)
                log_metrics(self.console_logger, self.writer, class_dices, class_hds, iter_nb, 'Iteration')
                log_metrics(self.file_logger, self.writer, class_dices, class_hds, iter_nb, 'Iteration')

            self.writer.add_scalars('Iteration/Labeled individual loss weights', self.labeled_loss_object.get_loss_weight(), iter_nb)
            self.labeled_loss_object.update_weight()


class GanLoop(object):
    def __init__(self, 
                 batch_size,
                 latent_size,
                 labeled_train_dataloader, 
                 validation_dataloader, 
                 validation_random_dataloader, 
                 val_dataloader_subset, 
                 generator, 
                 discriminator, 
                 generator_optimizer, 
                 discriminator_optimizer, 
                 console_logger, 
                 file_logger, 
                 save_iteration_number,
                 logging_metrics_iteration_number, 
                 logging_loss_iteration_number, 
                 device,
                 nb_iterations_per_epoch,
                 total_nb_of_iterations,
                 total_nb_epochs, 
                 criterion, 
                 generator_scheduler, 
                 discriminator_scheduler,
                 writer, 
                 r1_penalty_iteration,
                 use_ema,
                 ema_decay,
                 val_stride,
                 save_path='out'):

                 self.use_ema = use_ema
                 self.ema_decay = ema_decay
                 if use_ema:
                    # create a shadow copy of the generator
                    self.gen_shadow = copy.deepcopy(generator)
                    # updater function:
                    self.ema_updater = update_average
                    # initialize the gen_shadow weights equal to the weights of gen
                    self.ema_updater(self.gen_shadow, generator, beta=0)

                 self.latent_size = latent_size
                 self.batch_size = batch_size
                 self.num_stages = generator.num_stages
                 stage_growing_length = total_nb_of_iterations / (2 ** (self.num_stages - 1))
                 self.stage_boundaries = [0] + [(2 ** i) * stage_growing_length for i in range(self.num_stages)]
                 print(self.stage_boundaries)
                 self.r1_penalty_iteration = r1_penalty_iteration
                 self.val_stride = val_stride
                 self.current_epoch = 0
                 self.save_path = save_path
                 self.total_nb_epochs = total_nb_epochs
                 self.writer = writer
                 self.generator_scheduler = generator_scheduler
                 self.discriminator_scheduler = discriminator_scheduler
                 self.validation_dataloader = validation_dataloader
                 self.validation_random_dataloader = validation_random_dataloader
                 self.labeled_train_dataloader = labeled_train_dataloader
                 self.val_dataloader_subset = val_dataloader_subset
                 self.generator = generator
                 self.discriminator = discriminator
                 self.generator_optimizer = generator_optimizer
                 self.discriminator_optimizer = discriminator_optimizer
                 self.criterion = criterion
                 self.console_logger = console_logger
                 self.file_logger = file_logger
                 self.save_iteration_number = save_iteration_number
                 self.logging_metrics_iteration_number = logging_metrics_iteration_number
                 self.logging_loss_iteration_number = logging_loss_iteration_number
                 self.device = device
                 self.nb_iterations_per_epoch = nb_iterations_per_epoch
                 self.total_nb_of_iterations = total_nb_of_iterations
                 block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
                 self.inception_model = InceptionV3([block_idx]).to(self.device)
    
    def get_gan_validation_images(self):
        noise  = torch.rand(size=(1, self.latent_size), device=self.device)
        if self.use_ema:
            pred = self.gen_shadow(x=noise, stage_nb=self.stage_nb, alpha=self.alpha, fade_in=self.fade_in)
        else:
            pred = self.generator(x=noise, stage_nb=self.stage_nb, alpha=self.alpha, fade_in=self.fade_in)

        pred = resize(img=pred, size=(224, 224), antialias=True).squeeze(0).squeeze(0)
        #noise = torch.randn(size=(1, 7*7, 768), device=device)
        #pred = generator(noise).squeeze(0).squeeze(0)
        pred = cv.normalize(pred.cpu().numpy(), None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F).astype(np.uint8)[..., np.newaxis]
        return pred
    
    def gan_validation_loop(self):
        self.generator.eval()
        with torch.no_grad():
            fid = 0
            for data in tqdm(self.validation_dataloader, desc='Validation iteration: ', position=2):
                new_size = tuple([int(data.shape[-2] / (2 ** (self.num_stages - self.stage_nb))), int(data.shape[-1] / (2 ** (self.num_stages - self.stage_nb)))])
                data = resize(img=data, size=new_size, antialias=True)
                data = data.repeat(1, 3, 1, 1)
                pred_real = self.inception_model(data)[0]
                mu_real, sigma_real = get_fid_stats(pred_real)

                #noise = torch.randn(size=(1, 768, 7, 7), device=device)
                #fake = generator(noise)
                noise  = torch.rand(size=(1, self.latent_size), device=self.device)
                if self.use_ema:
                    fake = self.gen_shadow(x=noise, stage_nb=self.stage_nb, alpha=self.alpha, fade_in=self.fade_in)
                else:
                    fake = self.generator(x=noise, stage_nb=self.stage_nb, alpha=self.alpha, fade_in=self.fade_in)
                fake = fake.repeat(1, 3, 1, 1)
                pred_fake = self.inception_model(fake)[0]
                mu_fake, sigma_fake = get_fid_stats(pred_fake)

                fid_value = calculate_frechet_distance(mu_fake, sigma_fake, mu_real, sigma_real)
                fid += fid_value
            fid = fid / len(self.validation_dataloader)
            return fid
    
    def validation_logging(self, pred_image, fid):
        self.writer.add_image('Epoch/Prediction', pred_image, self.current_epoch, dataformats='HWC')
        log_gan_metrics(self.console_logger, self.writer, fid, self.current_epoch, 'Epoch')
        log_gan_metrics(self.file_logger, self.writer, fid, self.current_epoch, 'Epoch')
    
    def gan_main_loop(self):
        for idx, t in enumerate(tqdm(range(self.total_nb_epochs), desc='Epoch: ', position=0)):
            max_memory_allocated = torch.cuda.max_memory_allocated(device=self.device)
            self.console_logger.info(f"Max GPU Memory allocated: {max_memory_allocated / 10e8} Gb")
            self.file_logger.info(f"Max GPU Memory allocated: {max_memory_allocated / 10e8} Gb")
            self.current_epoch = idx
            self.train_loop()
            if idx % self.val_stride == 0:
                self.generator.eval()
                with torch.no_grad():
                    fid = self.gan_validation_loop()
                    pred_image = self.get_gan_validation_images()
                    self.validation_logging(pred_image, fid)
    
    def train_loop(self):
        nb_iters = len(self.labeled_train_dataloader)
        for batch, image_batch in enumerate(tqdm(self.labeled_train_dataloader, desc='Training iteration: ', position=1), 1):

            iter_nb = (self.current_epoch * nb_iters) + batch
            #current_phase = ceil(iter_nb / self.growing_phase_length)
            #self.stage_nb = floor((current_phase / 2) + 1)
            self.stage_nb = bisect.bisect_left(self.stage_boundaries, iter_nb)
            if self.stage_nb > 1 and iter_nb < 1.5 * self.stage_boundaries[self.stage_nb - 1]:
                self.fade_in = True
                self.alpha = (iter_nb - self.stage_boundaries[self.stage_nb - 1]) / (1.5 * self.stage_boundaries[self.stage_nb - 1] - self.stage_boundaries[self.stage_nb - 1])
            else:
                self.fade_in = False
                self.alpha = 0
            
            self.writer.add_scalar('Iteration/Growing alpha', self.alpha, iter_nb)
            self.writer.add_scalar('Iteration/Stage number', self.stage_nb, iter_nb)

            new_size = tuple([int(image_batch.shape[-2] / (2 ** (self.num_stages - self.stage_nb))), int(image_batch.shape[-1] / (2 ** (self.num_stages - self.stage_nb)))])
            #image_batch = interpolate(input=image_batch, size=new_size, mode='bilinear')
            image_batch = resize(img=image_batch, size=new_size, antialias=True)
            self.writer.add_scalar('Iteration/image_size', image_batch.size(-1), iter_nb)

            self.discriminator.train()
            self.generator.train()
            if self.use_ema:
                self.gen_shadow.train()

            label = torch.full((self.batch_size,), 1, dtype=torch.float, device=self.device)
            self.discriminator.zero_grad()
            output_real = self.discriminator(x=image_batch, stage_nb=self.stage_nb, alpha=self.alpha, fade_in=self.fade_in).reshape(-1)
            loss_real = self.criterion(output_real, label)
            self.writer.add_scalar('Iteration/Discriminator real', output_real.mean(), iter_nb)
            #loss_real.backward() #retain_graph if reuse output of discriminator for r1 penalty

            r1_penalty = 0
            if iter_nb % self.r1_penalty_iteration == 0:
                r1_penalty = logisticGradientPenalty(image_batch, self.discriminator, weight=5, alpha=self.alpha, stage_nb=self.stage_nb, fade_in=self.fade_in)
        
            loss_real_r1 = loss_real + r1_penalty
            loss_real_r1.backward()

            #noise = torch.randn(size=(batch_size, 768, 7, 7), device=self.device)
            #fake = self.generator(noise)
            noise  = torch.rand(size=(self.batch_size, self.latent_size), device=self.device)
            fake = self.generator(x=noise, stage_nb=self.stage_nb, alpha=self.alpha, fade_in=self.fade_in)
            assert fake.size() == image_batch.size()
            label.fill_(0)
            output_fake = self.discriminator(x=fake.detach(), stage_nb=self.stage_nb, alpha=self.alpha, fade_in=self.fade_in).view(-1)
            loss_fake = self.criterion(output_fake, label)
            self.writer.add_scalar('Iteration/Discriminator fake', output_fake.mean(), iter_nb)
            loss_fake.backward()

            discriminator_loss = loss_real + loss_fake + r1_penalty
            self.discriminator_optimizer.step()

            self.generator.zero_grad()
            label.fill_(1)
            output = self.discriminator(x=fake, stage_nb=self.stage_nb, alpha=self.alpha, fade_in=self.fade_in).view(-1)
            generator_loss = self.criterion(output, label)
            generator_loss.backward()
            self.generator_optimizer.step()

            # if use_ema is true, apply ema to the generator parameters
            if self.use_ema:
                self.ema_updater(self.gen_shadow, self.generator, self.ema_decay)

            #self.discriminator_scheduler.step()
            #self.generator_scheduler.step()
            self.writer.add_scalars('Iteration/Learning rates', {'Discriminator': self.discriminator_optimizer.param_groups[0]['lr'], 'Generator': self.generator_optimizer.param_groups[0]['lr']}, iter_nb)

            self.writer.add_scalars('Iteration/Discriminator individual losses', {'real_data': loss_real, 'fake_data': loss_fake, 'r1_penalty': r1_penalty}, iter_nb)
            self.writer.add_scalar('Iteration/Discriminator training loss', discriminator_loss, iter_nb)
            self.writer.add_scalar('Iteration/Generator training loss', generator_loss, iter_nb)
            if batch % self.save_iteration_number == 0:
                torch.save(self.discriminator.state_dict(), os.path.join(self.save_path, 'discriminator_weights.pth'))
                torch.save(self.generator.state_dict(), os.path.join(self.save_path, 'generator_weights.pth'))
                if self.use_ema:
                    torch.save(self.gen_shadow.state_dict(), os.path.join(self.save_path, 'ema_generator_weights.pth'))
                self.console_logger.info(f"Saved model to {self.save_path}")
            
            if batch % self.logging_loss_iteration_number == 0:
                self.console_logger.info(f"Discriminator loss: {discriminator_loss:>7f}")
                self.console_logger.info(f"Generator loss: {generator_loss:>7f}")
            
            if batch % self.logging_metrics_iteration_number == 0:
                max_memory_allocated = torch.cuda.max_memory_allocated(device=self.device)
                self.console_logger.info(f"Max GPU Memory allocated: {max_memory_allocated / 10e8} Gb")
                self.file_logger.info(f"Max GPU Memory allocated: {max_memory_allocated / 10e8} Gb")
                self.generator.eval()
                with torch.no_grad():
                    fid = self.gan_validation_loop()
                    pred_image = self.get_gan_validation_images()
                    self.validation_logging(pred_image, fid)
