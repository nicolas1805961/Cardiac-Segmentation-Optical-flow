import torch
from acdc_dataset import create_2d_acdc_dataset_crop
from torch.utils.tensorboard import SummaryWriter
import warnings
import os
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import random
import shutil
import time
from pathlib import Path
import data_augmentation
from loss import Loss, LocalizationLoss
import global_variables
import yaml
import reconstruction_loop
import argparse
from training_utils import build_2d_model_crop, build_policy_net, build_discriminator, set_losses, set_augmentations, read_config, build_2d_model, create_loggers, write_model_parameters, count_parameters, get_validation_images_lib, log_metrics


torch.autograd.set_detect_anomaly(True)
warnings.filterwarnings("ignore", category=UserWarning)
#torch.manual_seed(0)
#random.seed(0)
#np.random.seed(0)

global_variables.init_globals()

#for dirpath in Path(".").glob("*-*-*_*H*"):
#    if dirpath.exists() and dirpath.is_dir():
#        shutil.rmtree(dirpath)

model2=None
optimizer2 = None
scheduler2 = None

parser = argparse.ArgumentParser()
parser.add_argument('train_path', help='Path to training input directory')
parser.add_argument('config_file', help='Path to the configuration file')
args = parser.parse_args()

config = read_config(args.config_file)

img_size = 160
train_path = os.path.join(args.train_path, '*')
val_path = os.path.join(args.train_path, '*')
loss_weights = torch.tensor(config['160_loss_weights'] if not config['binary'] else config['160_binary_loss_weights'], device=config['device'])

data_augmentation_utils = set_augmentations(config, data_augmentation, img_size, autoencoder=False)
data_augmentation_utils = None if not data_augmentation_utils else np.array(data_augmentation_utils)
dataloaders = create_2d_acdc_dataset_crop(train_path=train_path, 
                                        val_path=val_path,
                                        batch_size=config['batch_size'],
                                        device=config['device'],
                                        val_subset_size=config['val_subset_size'],
                                        data_augmentation_utils=data_augmentation_utils,
                                        img_size=img_size,
                                        target_ratio=config['target_ratio'],
                                        binary=config['binary'])
nb_iterations_per_epoch = len(dataloaders['labeled_train_dataloader'])
#if config['semi_supervised']:
#    print(len(dataloaders['unlabeled_train_dataloader1'])/len(dataloaders['labeled_train_dataloader']))
#    nb_iterations_per_epoch = max(len(dataloaders['labeled_train_dataloader']), len(dataloaders['unlabeled_train_dataloader1']), len(dataloaders['unlabeled_train_dataloader2']))
#    total_nb_of_iterations = config['epochs'] * nb_iterations_per_epoch
#    model2 = build_model(config)
#    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=config['learning_rate'], weight_decay=0.0001)
#    scheduler2 = CosineAnnealingLR(optimizer2, T_max=total_nb_of_iterations)
#    model = WholeModel(model, model2)

total_nb_of_iterations = config['epochs']*nb_iterations_per_epoch
warmup_iter = int(config['warmup_percent'] * total_nb_of_iterations)
models = {}

model = None
model_optimizer = None
model_scheduler = None
seg_discriminator = None
rec_discriminator = None
seg_discriminator_optimizer = None
rec_discriminator_optimizer = None
seg_discriminator_scheduler = None
rec_discriminator_scheduler = None
if config['reinforcement']:
    policy_net = build_policy_net(config)
    policy_net_input_size = (config['batch_size'], 1, 224, 224)
    policy_net_optimizer = torch.optim.AdamW(policy_net.parameters(), lr=config['policy_net_learning_rate'], weight_decay=0.0001)
    policy_net_scheduler = CosineAnnealingLR(policy_net_optimizer, T_max=total_nb_of_iterations)
    models['policy net'] = (policy_net, policy_net_input_size)
else:
    policy_net = None
    policy_net_optimizer = None
    policy_net_scheduler = None

    model = build_2d_model_crop(config)
    model_optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=0.0001)
    cosine_scheduler = CosineAnnealingLR(model_optimizer, T_max=total_nb_of_iterations - nb_iterations_per_epoch)
    #model_scheduler = CosineAnnealingLR(model_optimizer, T_max=total_nb_of_iterations)
    warmup = LinearLR(optimizer=model_optimizer, start_factor=0.1, end_factor=1, total_iters=nb_iterations_per_epoch)
    model_scheduler = SequentialLR(optimizer=model_optimizer, schedulers=[warmup, cosine_scheduler], milestones=[nb_iterations_per_epoch])
    model_input_size = (config['batch_size'], 1, img_size, img_size)
    models['model'] = (model, model_input_size)

    if config['adversarial']:
        seg_discriminator = build_discriminator(config, discriminator_type='seg')
        seg_discriminator_input_size = (config['batch_size'], 5, 224, 224)
        rec_discriminator = build_discriminator(config, discriminator_type='rec')
        rec_discriminator_input_size = (config['batch_size'], 1, 224, 224)
        seg_discriminator_optimizer = torch.optim.AdamW(seg_discriminator.parameters(), lr=config['discriminator_learning_rate'], betas=(0.0, 0.99), weight_decay=config['discriminator_weight_decay'])
        rec_discriminator_optimizer = torch.optim.AdamW(rec_discriminator.parameters(), lr=config['discriminator_learning_rate'], betas=(0.0, 0.99), weight_decay=config['discriminator_weight_decay'])
        seg_discriminator_scheduler = CosineAnnealingLR(seg_discriminator_optimizer, T_max=total_nb_of_iterations)
        rec_discriminator_scheduler = CosineAnnealingLR(rec_discriminator_optimizer, T_max=total_nb_of_iterations)
        models['segmentation discriminator'] = (seg_discriminator, seg_discriminator_input_size)
        models['reconstruction discriminator'] = (rec_discriminator, rec_discriminator_input_size)

spatial_transformer = None
spatial_transformer_optimizer = None

timestr = time.strftime("%Y-%m-%d_%HH%M")
writer = SummaryWriter(log_dir=timestr)

console_logger, file_logger = create_loggers(timestr)

count_parameters(console_logger, file_logger, config, models)

spatial_transformer_scheduler = None
if config['use_spatial_transformer']:
    spatial_transformer_scheduler = CosineAnnealingLR(spatial_transformer_optimizer, T_max=total_nb_of_iterations)
#scheduler = GradualWarmupScheduler(optimizer, multiplier=1.0, total_epoch=warmup_iter, after_scheduler=scheduler)

if config['deep_supervision']:
    deep_supervision_weights = torch.tensor([1 / 2**x for x in reversed(range(0, 5))])
    deep_supervision_weights = (deep_supervision_weights / deep_supervision_weights.sum()).tolist()
else:
    deep_supervision_weights = [1]

add = (config['lambda_end'] - config['lambda_start']) / total_nb_of_iterations

labeled_losses = set_losses(config, add, loss_weights)

if config['use_spatial_transformer']:
    localization_loss_object = LocalizationLoss(config['localization_weight'], writer)
else:
    localization_loss_object = None
labeled_loss_object = Loss(labeled_losses, writer, 'labeled')
#unlabeled_loss_object = Loss(unlabeled_losses, writer, 'unlabeled')
#spatial_transformer_loss_object = Loss(spatial_transformer_losses, writer, 'spatial transformer')


loop = reconstruction_loop.ReconstructionLoop(labeled_train_dataloader=dataloaders['labeled_train_dataloader'],
                                                mode=config['model'],
                                                validation_dataloader=dataloaders['val_dataloader'],
                                                validation_random_dataloader=dataloaders['val_random_dataloader'],
                                                val_dataloader_subset=dataloaders['val_dataloader_subset'],
                                                train_dataloader_subset=dataloaders['train_dataloader_subset'],
                                                train_random_dataloader=dataloaders['train_random_dataloader'],
                                                compute_overfitting=config['compute_overfitting'],
                                                similarity_downscale=config['similarity_down_scale'],
                                                scaling_loss_weight=config['scaling_loss_weight'],
                                                reconstruction_scaling_loss_weight=config['reconstruction_scaling_loss_weight'],
                                                total_number_of_iterations=total_nb_of_iterations,
                                                binary=config['binary'],
                                                cropping_network=True,
                                                weight_end_percent=config['weight_end_percent'],
                                                uncertainty_weighting=False,
                                                dynamic_weight_averaging=False,
                                                model=model,
                                                logits=False,
                                                number_of_steps=config['number_of_steps'],
                                                number_of_intervals=config['number_of_intervals'],
                                                policy_net=policy_net,
                                                reconstruction_rotation_loss_weight=config['reconstruction_rotation_loss_weight'],
                                                policy_scheduler=policy_net_scheduler,
                                                policy_optimizer=policy_net_optimizer,
                                                rotation_loss_weight=config['rotation_loss_weight'],
                                                r1_penalty_iteration=config['r1_penalty_iteration'],
                                                batch_size=config['batch_size'],
                                                transformer_depth=config['transformer_depth'],
                                                conv_depth=config['conv_depth'],
                                                similarity_loss_weight=config['similarity_loss_weight'],
                                                adversarial_loss_weight=config['adversarial_loss_weight'],
                                                reconstruction_loss_weight=config['reconstruction_loss_weight'],
                                                plot_gradient_iter_number=config['plot_gradient_iter_number'],
                                                img_size=img_size,
                                                model_optimizer=model_optimizer,
                                                console_logger=console_logger,
                                                file_logger=file_logger,
                                                logging_metrics_iteration_number=config['logging_metrics_iteration_number'],
                                                logging_loss_iteration_number=config['logging_loss_iteration_number'],
                                                device=config['device'],
                                                nb_iterations_per_epoch=nb_iterations_per_epoch,
                                                total_nb_of_iterations=total_nb_of_iterations,
                                                total_nb_epochs=config['epochs'], 
                                                deep_supervision_weights=deep_supervision_weights,
                                                labeled_loss_object=labeled_loss_object,
                                                model_scheduler=model_scheduler,
                                                writer=writer,
                                                seg_discriminator_optimizer=seg_discriminator_optimizer,
                                                rec_discriminator_optimizer=rec_discriminator_optimizer,
                                                seg_discriminator_scheduler=seg_discriminator_scheduler,
                                                rec_discriminator_scheduler=rec_discriminator_scheduler,
                                                seg_discriminator=seg_discriminator,
                                                rec_discriminator=rec_discriminator,
                                                reconstruction=config['reconstruction'],
                                                val_stride=config['val_stride'],
                                                save_path=timestr)

try:
    if config['reinforcement']:
        loop.main_loop_reinforcement()
    else:
        loop.main_loop_acdc_2d()
except KeyboardInterrupt:
    global_variables.get_stats_object.write_to_file()

torch.save(model.state_dict(), os.path.join(timestr, 'weights.pth'))
with open(os.path.join(timestr, 'config.yaml'), 'w+') as file:
    yaml.dump(config, file)
writer.close()
print("Done!")


#for idx, t in enumerate(tqdm(range(config['epochs']), desc='Epoch: ', position=0)):
#
#    if config['dataset'] == 'acdc':
#        if config['semi_supervised']:
#            train_loop_acdc_semi_supervised_my_augment(labeled_train_dataloader=dataloaders['labeled_train_dataloader'], 
#                                            unlabeled_train_dataloader1=dataloaders['unlabeled_train_dataloader1'],
#                                            unlabeled_train_dataloader2=dataloaders['unlabeled_train_dataloader2'],
#                                            val_dataloader_subset=dataloaders['val_dataloader_subset'], 
#                                            model=model,
#                                            thresh=config['unlabeled_loss_thresh'],
#                                            bootstrap=config['bootstrap_start'],
#                                            bootstrap_annealing=bootstrap_annealing,
#                                            unlabeled_loss_weight=config['unlabeled_loss_weight_start'],
#                                            unlabeled_loss_weight_annealing=unlabeled_loss_weight_annealing,
#                                            writer=writer,
#                                            optimizer1=optimizer1, 
#                                            optimizer2=optimizer2, 
#                                            console_logger=console_logger, 
#                                            file_logger=file_logger,
#                                            save_iteration_number=1000, 
#                                            logging_metrics_iteration_number=config['logging_metrics_iteration_number'],
#                                            labeled_loss_object=labeled_loss_object,
#                                            unlabeled_loss_object=unlabeled_loss_object,
#                                            logging_loss_iteration_number=config['logging_loss_iteration_number'], 
#                                            device=config['device'], 
#                                            base_lr=config['learning_rate'],
#                                            epoch_nb=idx, 
#                                            nb_iterations_per_epoch=nb_iterations_per_epoch,
#                                            deep_supervision_weights=weights, 
#                                            scheduler1=scheduler1, 
#                                            scheduler2=scheduler2, 
#                                            total_nb_epochs=config['epochs'])
#            if idx % config['val_stride'] == 0:
#                class_dice, class_hd = validation_loop_acdc(model, dataloaders['val_dataloader'])
#                images = get_validation_images_acdc(model, dataloaders['val_random_dataloader'], config['device'])
#        else:
#            train_loop_acdc_supervised(dataloaders['labeled_train_dataloader'], 
#                                       dataloaders['val_dataloader_subset'], 
#                                       model, 
#                                       optimizer1, 
#                                       console_logger, 
#                                       file_logger,
#                                       writer=writer,
#                                       bootstrap=config['bootstrap_start'],
#                                       bootstrap_annealing=bootstrap_annealing,
#                                       save_iteration_number=1000, 
#                                       logging_metrics_iteration_number=config['logging_metrics_iteration_number'],
#                                       labeled_loss_object=labeled_loss_object,
#                                       logging_loss_iteration_number=config['logging_loss_iteration_number'], 
#                                       device=config['device'], 
#                                       base_lr=config['learning_rate'],
#                                       epoch_nb=idx, 
#                                       deep_supervision_weights=weights, 
#                                       scheduler=scheduler1, 
#                                       total_nb_epochs=config['epochs'])
#            if idx % config['val_stride'] == 0:
#                class_dice, class_hd = validation_loop_acdc(model, dataloaders['val_dataloader'])
#                images = get_validation_images_acdc(model, dataloaders['val_random_dataloader'], config['device'])
#    else:
#        train_loop_lib(dataloaders['labeled_train_dataloader'], 
#                       dataloaders['val_dataloader_subset'], 
#                       model, 
#                       optimizer1, 
#                       console_logger, 
#                       file_logger,
#                       writer=writer,
#                       save_iteration_number=1000, 
#                       bootstrap=config['bootstrap_start'],
#                       bootstrap_annealing=bootstrap_annealing,
#                       logging_metrics_iteration_number=config['logging_metrics_iteration_number'],
#                       labeled_loss_object=labeled_loss_object,
#                       logging_loss_iteration_number=config['logging_loss_iteration_number'], 
#                       device=config['device'], 
#                       base_lr=config['learning_rate'],
#                       epoch_nb=idx, 
#                       deep_supervision_weights=weights, 
#                       scheduler=scheduler1, 
#                       total_nb_epochs=config['epochs'])
#        if idx % config['val_stride'] == 0:
#            class_dice, class_hd = validation_loop_lib(model, dataloaders['val_dataloader'])
#            images = get_validation_images_lib(model, dataloaders['val_random_dataloader'], config['device'])
#
#    if idx % config['val_stride'] == 0:
#        writer.add_image('Epoch/Image', images['x'], idx, dataformats='HWC')
#        writer.add_image('Epoch/Ground truth', images['y'], idx, dataformats='HWC')
#        writer.add_image('Epoch/Prediction', images['pred'], idx, dataformats='HWC')
#        log_metrics(console_logger, writer, class_dice, class_hd, idx, 'Epoch')
#        log_metrics(file_logger, writer, class_dice, class_hd, idx, 'Epoch')
#
#torch.save(model.state_dict(), 'out/weights.pth')
#writer.close()
#print("Done!")