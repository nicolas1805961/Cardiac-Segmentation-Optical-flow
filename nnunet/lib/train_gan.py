from cv2 import mean
import torch
from model import my_model, WholeModel
from tqdm import tqdm
from datasets import create_lib_datasets, create_acdc_dataset, create_acdc_dataset_alt, create_gan_acdc_dataset
from torch.utils.tensorboard import SummaryWriter
import warnings
import os
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
import random
import shutil
import time
from pathlib import Path
import data_augmentation_lib
import data_augmentation_acdc
import torch.nn as nn
from loss import Loss
from loops import GanLoop
from training_utils import set_losses, set_augmentations, read_config, build_gan, create_loggers, write_model_parameters, count_parameters, get_validation_images_acdc, get_validation_images_lib, log_metrics

warnings.filterwarnings("ignore", category=UserWarning)
#torch.manual_seed(0)
#random.seed(0)
#np.random.seed(0)

dirpath = Path('out/')
if dirpath.exists() and dirpath.is_dir():
    shutil.rmtree(dirpath)

config = read_config('config.yaml')

generator, discriminator = build_gan(config)
generator_optimizer = torch.optim.Adam(generator.parameters(), lr=config['generator_learning_rate'], betas=(0.0, 0.99), weight_decay=config['generator_weight_decay'])
discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=config['discriminator_learning_rate'], betas=(0.0, 0.99), weight_decay=config['discriminator_weight_decay'])

timestr = time.strftime("%Y-%m-%d_%HH%M")
logdir = os.path.join('out', timestr)
writer = SummaryWriter(log_dir=logdir)

console_logger, file_logger = create_loggers(logdir)

optimizer2 = None
scheduler2 = None
if config['dataset'] == 'acdc':
    loss_weights = torch.tensor([0.004661203152372745, 0.34532117286545916, 0.3213283464535728, 0.3286892775285953], device=config['device'])
    data_augmentation_utils = set_augmentations(config, data_augmentation_acdc)
    data_augmentation_utils = None if not data_augmentation_utils else data_augmentation_utils
    dataloaders = create_gan_acdc_dataset(path='ACDC_data/*', 
                                      batch_size=config['batch_size'], 
                                      device=config['device'],
                                      image_or_label=config['image_or_label'],
                                      data_augmentation_utils=data_augmentation_utils)
    nb_iterations_per_epoch = len(dataloaders['labeled_train_dataloader'])
elif config['dataset'] == 'lib':
    data_augmentation_utils = set_augmentations(config, data_augmentation_lib)
    data_augmentation_utils = None if not data_augmentation_utils else data_augmentation_utils
    dataloaders = create_lib_datasets('LIB_data/*', 
                                      nb_frames=config['nb_frames'], 
                                      val_subset_size=config['val_subset_size'], 
                                      batch_size=config['batch_size'], 
                                      method=config['method'], 
                                      device=config['device'], 
                                      data_augmentation_utils=data_augmentation_utils)
    loss_weights = torch.tensor([0.014232762489019338, 0.27990668860712137, 0.34708157120624633, 0.35877897769761297], device=config['device'])
    nb_iterations_per_epoch = len(dataloaders['labeled_train_dataloader'])

#write_model_parameters(model)
count_parameters(generator, console_logger, file_logger, config, 'generator')
count_parameters(discriminator, console_logger, file_logger, config, 'discriminator')

total_nb_of_iterations = config['epochs']*nb_iterations_per_epoch

warmup_iter = int(config['warmup_percent'] * total_nb_of_iterations)
generator_scheduler = CosineAnnealingLR(generator_optimizer, T_max=total_nb_of_iterations)
discriminator_scheduler = CosineAnnealingLR(discriminator_optimizer, T_max=total_nb_of_iterations)
#scheduler = GradualWarmupScheduler(optimizer, multiplier=1.0, total_epoch=warmup_iter, after_scheduler=scheduler)

if config['deep_supervision']:
    deep_supervision_weights = torch.tensor([1 / 2**x for x in reversed(range(0, 5))])
    deep_supervision_weights = (deep_supervision_weights / deep_supervision_weights.sum()).tolist()
else:
    deep_supervision_weights = [1]

add = (config['lambda_end'] - config['lambda_start']) / total_nb_of_iterations

criterion = nn.BCELoss()

loop = GanLoop(use_ema=config['use_ema'],
            ema_decay=config['ema_decay'],
            batch_size=config['batch_size'],
            latent_size=config['latent_size'],
            labeled_train_dataloader=dataloaders['labeled_train_dataloader'],
            validation_dataloader=dataloaders['val_dataloader'],
            validation_random_dataloader=dataloaders['val_random_dataloader'],
            val_dataloader_subset=dataloaders['val_dataloader_subset'],
            generator=generator,
            discriminator=discriminator,
            generator_optimizer=generator_optimizer,
            discriminator_optimizer=discriminator_optimizer,
            console_logger=console_logger,
            file_logger=file_logger,
            save_iteration_number=config['save_iteration_number'],
            logging_metrics_iteration_number=config['logging_metrics_iteration_number'],
            logging_loss_iteration_number=config['logging_loss_iteration_number'],
            device=config['device'],
            nb_iterations_per_epoch=nb_iterations_per_epoch,
            total_nb_of_iterations=total_nb_of_iterations,
            total_nb_epochs=config['epochs'], 
            criterion=criterion,
            generator_scheduler=generator_scheduler,
            discriminator_scheduler=discriminator_scheduler,
            writer=writer,
            r1_penalty_iteration=config['r1_penalty_iteration'],
            val_stride=config['val_stride'],
            save_path='out')

loop.gan_main_loop()

torch.save(discriminator.state_dict(), 'out/weights.pth')
torch.save(generator.state_dict(), 'out/weights.pth')
writer.close()
print("Done!")