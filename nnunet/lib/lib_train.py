from cv2 import mean
import torch
from model import my_model, WholeModel
from tqdm import tqdm
from lib_dataset import create_lib_datasets
from torch.utils.tensorboard import SummaryWriter
import warnings
import os
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
import random
import shutil
import time
from pathlib import Path
import data_augmentation
from loss import Loss, LocalizationLoss
from loops import Loops
import global_variables
from training_utils import build_spatial_transformer, set_losses, set_augmentations, read_config, build_model, create_loggers, write_model_parameters, count_parameters, get_validation_images_lib, log_metrics

warnings.filterwarnings("ignore", category=UserWarning)
#torch.manual_seed(0)
#random.seed(0)
#np.random.seed(0)

global_variables.init_globals()

dirpath = Path('out/')
if dirpath.exists() and dirpath.is_dir():
    shutil.rmtree(dirpath)

config = read_config('lib_config.yaml')

model1 = build_model(config)
optimizer1 = torch.optim.AdamW(model1.parameters(), lr=config['learning_rate'], weight_decay=0.0001)

spatial_transformer = None
spatial_transformer_optimizer = None
if config['use_spatial_transformer']:
    spatial_transformer = build_spatial_transformer(config)
    spatial_transformer_optimizer = torch.optim.AdamW(spatial_transformer.parameters(), lr=config['spatial_transformer_learning_rate'], weight_decay=0.0001)

timestr = time.strftime("%Y-%m-%d_%HH%M")
logdir = os.path.join('out', timestr)
writer = SummaryWriter(log_dir=logdir)

console_logger, file_logger = create_loggers(logdir)

model2=None
optimizer2 = None
scheduler2 = None
#loss_weights = torch.tensor(config['loss_weights_big_image'], device=config['device']) if config['binary'] else torch.tensor(config['loss_weights_image'], device=config['device'])
loss_weights = torch.tensor([0.008820753831094484, 0.2702277098234642, 0.35075953121988374, 0.3701920051255575], device=config['device'])
data_augmentation_utils = set_augmentations(config, data_augmentation)
data_augmentation_utils = None if not data_augmentation_utils else data_augmentation_utils
dataloaders = create_lib_datasets(path=config['path'] + '/*', 
                                    batch_size=config['batch_size'], 
                                    use_spatial_transformer=config['use_spatial_transformer'],
                                    device=config['device'],
                                    binary=config['binary'],
                                    data_augmentation_utils=data_augmentation_utils,
                                    val_subset_size=config['val_subset_size'],
                                    img_size=224) #config['big_image_size'] if config['binary'] else config['image_size'])
nb_iterations_per_epoch = len(dataloaders['labeled_train_dataloader'])
if config['semi_supervised']:
    print(len(dataloaders['unlabeled_train_dataloader1'])/len(dataloaders['labeled_train_dataloader']))
    nb_iterations_per_epoch = max(len(dataloaders['labeled_train_dataloader']), len(dataloaders['unlabeled_train_dataloader1']), len(dataloaders['unlabeled_train_dataloader2']))
    total_nb_of_iterations = config['epochs'] * nb_iterations_per_epoch
    model2 = build_model(config)
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=config['learning_rate'], weight_decay=0.0001)
    scheduler2 = CosineAnnealingLR(optimizer2, T_max=total_nb_of_iterations)
    model = WholeModel(model1, model2)
else:
    model = model1

write_model_parameters(model)
count_parameters(model, console_logger, file_logger, config, 'U-net')

total_nb_of_iterations = config['epochs']*nb_iterations_per_epoch

warmup_iter = int(config['warmup_percent'] * total_nb_of_iterations)
scheduler1 = CosineAnnealingLR(optimizer1, T_max=total_nb_of_iterations)

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

labeled_losses, unlabeled_losses, spatial_transformer_losses = set_losses(config, add, loss_weights)

if config['use_spatial_transformer']:
    localization_loss_object = LocalizationLoss(config['localization_weight'], writer)
else:
    localization_loss_object = None
labeled_loss_object = Loss(labeled_losses, writer, 'labeled')
unlabeled_loss_object = Loss(unlabeled_losses, writer, 'unlabeled')
spatial_transformer_loss_object = Loss(spatial_transformer_losses, writer, 'spatial transformer')

loop = Loops(labeled_train_dataloader=dataloaders['labeled_train_dataloader'],
            validation_dataloader=dataloaders['val_dataloader'],
            validation_random_dataloader=dataloaders['val_random_dataloader'],
            val_dataloader_subset=dataloaders['val_dataloader_subset'],
            model=model,
            img_size=config['image_size'],
            big_img_size=config['big_image_size'],
            optimizer1=optimizer1,
            optimizer2=optimizer2,
            spatial_transformer_optimizer=spatial_transformer_optimizer,
            console_logger=console_logger,
            file_logger=file_logger,
            save_iteration_number=config['save_iteration_number'],
            logging_metrics_iteration_number=config['logging_metrics_iteration_number'],
            logging_loss_iteration_number=config['logging_loss_iteration_number'],
            device=config['device'],
            unlabeled_loss_weight_end=config['unlabeled_loss_weight_end'],
            nb_iterations_per_epoch=nb_iterations_per_epoch,
            total_nb_of_iterations=total_nb_of_iterations,
            total_nb_epochs=config['epochs'], 
            deep_supervision_weights=deep_supervision_weights,
            labeled_loss_object=labeled_loss_object,
            unlabeled_loss_object=unlabeled_loss_object,
            spatial_transformer_loss_object=spatial_transformer_loss_object,
            localization_loss_object=localization_loss_object,
            scheduler1=scheduler1,
            scheduler2=scheduler2,
            spatial_transformer_scheduler=spatial_transformer_scheduler,
            writer=writer,
            spatial_transformer=spatial_transformer,
            val_stride=config['val_stride'],
            save_path='out')

try:
    if config['use_spatial_transformer']:
        loop.main_loop_acdc_supervised_spatial_transformer()
    elif config['semi_supervised']:
        loop.main_loop_acdc_semi_supervised(unlabeled_train_dataloader1=dataloaders['unlabeled_train_dataloader1'],
                                            unlabeled_train_dataloader2=dataloaders['unlabeled_train_dataloader2'])
    else:
        loop.main_loop_acdc_supervised()
except KeyboardInterrupt:
    global_variables.get_stats_object.write_to_file()

torch.save(model.state_dict(), 'out/weights.pth')
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