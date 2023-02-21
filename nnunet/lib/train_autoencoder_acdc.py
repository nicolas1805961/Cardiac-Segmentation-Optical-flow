import torch
from acdc_dataset import create_acdc_autoencoder_dataset
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
from loss import Loss
from loops import Loops
import global_variables
import yaml
from training_utils import build_autoencoder, set_losses, set_augmentations, read_config, create_loggers, write_model_parameters, count_parameters, get_validation_images_lib, log_metrics

warnings.filterwarnings("ignore", category=UserWarning)
#torch.manual_seed(0)
#random.seed(0)
#np.random.seed(0)

global_variables.init_globals()

dirpath = Path('out/')
if dirpath.exists() and dirpath.is_dir():
    shutil.rmtree(dirpath)

config = read_config('acdc_config.yaml')

model = build_autoencoder(config)
optimizer1 = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=0.0001)

timestr = time.strftime("%Y-%m-%d_%HH%M")
logdir = os.path.join('out', timestr)
writer = SummaryWriter(log_dir=logdir)

console_logger, file_logger = create_loggers(logdir)

loss_weights = torch.tensor(config['loss_weights_big_image'], device=config['device']) if config['binary'] else torch.tensor(config['loss_weights_image'], device=config['device'])
data_augmentation_utils = set_augmentations(config, data_augmentation, autoencoder=True)
data_augmentation_utils = None if not data_augmentation_utils else data_augmentation_utils
dataloaders = create_acdc_autoencoder_dataset(path=config['path'] + '/*', 
                                    batch_size=config['batch_size'], 
                                    device=config['device'],
                                    data_augmentation_utils=data_augmentation_utils,
                                    img_size=224) #config['big_image_size'] if config['binary'] else config['image_size'])
nb_iterations_per_epoch = len(dataloaders['labeled_train_dataloader'])

write_model_parameters(model)
count_parameters(model, console_logger, file_logger, config, 'U-net', autoencoder=True)

total_nb_of_iterations = config['epochs']*nb_iterations_per_epoch

warmup_iter = int(config['warmup_percent'] * total_nb_of_iterations)
scheduler1 = CosineAnnealingLR(optimizer1, T_max=total_nb_of_iterations)

#scheduler = GradualWarmupScheduler(optimizer, multiplier=1.0, total_epoch=warmup_iter, after_scheduler=scheduler)

if config['deep_supervision']:
    deep_supervision_weights = torch.tensor([1 / 2**x for x in reversed(range(0, 5))])
    deep_supervision_weights = (deep_supervision_weights / deep_supervision_weights.sum()).tolist()
else:
    deep_supervision_weights = [1]

add = (config['lambda_end'] - config['lambda_start']) / total_nb_of_iterations

labeled_losses, unlabeled_losses, spatial_transformer_losses = set_losses(config, add, loss_weights)

localization_loss_object = None
labeled_loss_object = Loss(labeled_losses, writer, 'labeled')
unlabeled_loss_object = Loss(unlabeled_losses, writer, 'unlabeled')
spatial_transformer_loss_object = Loss(spatial_transformer_losses, writer, 'spatial transformer')


loop = Loops(labeled_train_dataloader=dataloaders['labeled_train_dataloader'],
            validation_dataloader=dataloaders['val_dataloader'],
            validation_random_dataloader=dataloaders['val_random_dataloader'],
            val_dataloader_subset=dataloaders['val_dataloader_subset'],
            model=model,
            reconstruction_loss_weight=config['reconstruction_loss_weight'],
            plot_gradient_iter_number=config['plot_gradient_iter_number'],
            img_size=config['image_size'],
            big_img_size=config['big_image_size'],
            optimizer1=optimizer1,
            optimizer2=None,
            spatial_transformer_optimizer=None,
            console_logger=console_logger,
            file_logger=file_logger,
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
            scheduler2=None,
            spatial_transformer_scheduler=None,
            writer=writer,
            spatial_transformer=None,
            val_stride=config['val_stride'],
            save_path='out')

try:
    loop.main_loop_acdc_autoencoder()
except KeyboardInterrupt:
    global_variables.get_stats_object.write_to_file()

torch.save(model.state_dict(), 'out/weights.pth')
with open(r'out/config.yaml', 'w+') as file:
    yaml.dump(config, file)
writer.close()
print("Done!")