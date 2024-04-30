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


import argparse
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.run.default_configuration import get_default_configuration
from nnunet.paths import default_plans_identifier
from nnunet.run.load_pretrained_weights import load_pretrained_weights
from nnunet.training.cascade_stuff.predict_next_stage import predict_next_stage
from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer
from nnunet.training.network_training.nnUNetTrainerCascadeFullRes import nnUNetTrainerCascadeFullRes
from nnunet.training.network_training.nnUNetTrainerV2_CascadeFullRes import nnUNetTrainerV2CascadeFullRes
from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name

from nnunet.lib.training_utils import read_config, read_config_video
from pathlib import Path

import warnings

import torch

#torch.autograd.set_detect_anomaly(True)
#warnings.filterwarnings("always", category=UserWarning)
#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("network")
    parser.add_argument("network_trainer")
    parser.add_argument("task", help="can be task name or task id")
    parser.add_argument("fold", help='0, 1, ..., 5 or \'all\'')
    parser.add_argument("-config", help="yaml config file", required=True)
    parser.add_argument("-val", "--validation_only", help="use this if you want to only run the validation",
                        action="store_true")
    parser.add_argument('-w', '--weight_folder', help='folder where to find the model\'s weights',
                        default=None, required=False)
    parser.add_argument("-c", "--continue_training", help="use this if you want to continue a training",
                        action="store_true")
    parser.add_argument("-p", help="plans identifier. Only change this if you created a custom experiment planner",
                        default=default_plans_identifier, required=False)
    parser.add_argument("--use_compressed_data", default=False, action="store_true",
                        help="If you set use_compressed_data, the training cases will not be decompressed. Reading compressed data "
                             "is much more CPU and RAM intensive and should only be used if you know what you are "
                             "doing", required=False)
    parser.add_argument("--deterministic",
                        help="Makes training deterministic, but reduces training speed substantially. I (Fabian) think "
                             "this is not necessary. Deterministic training will make you overfit to some random seed. "
                             "Don't use that.",
                        required=False, default=False, action="store_true")
    parser.add_argument("--npz", required=False, default=False, action="store_true", help="if set then nnUNet will "
                                                                                          "export npz files of "
                                                                                          "predicted segmentations "
                                                                                          "in the validation as well. "
                                                                                          "This is needed to run the "
                                                                                          "ensembling step so unless "
                                                                                          "you are developing nnUNet "
                                                                                          "you should enable this")
    parser.add_argument("--find_lr", required=False, default=False, action="store_true",
                        help="not used here, just for fun")
    parser.add_argument("--valbest", required=False, default=False, action="store_true",
                        help="hands off. This is not intended to be used")
    parser.add_argument("--fp32", required=False, default=False, action="store_true",
                        help="disable mixed precision training and run old school fp32")
    parser.add_argument("--val_folder", required=False, default="validation_raw",
                        help="name of the validation folder. No need to use this for most people")
    parser.add_argument("--disable_saving", required=False, action='store_true',
                        help="If set nnU-Net will not save any parameter files (except a temporary checkpoint that "
                             "will be removed at the end of the training). Useful for development when you are "
                             "only interested in the results and want to save some disk space")
    parser.add_argument("--disable_postprocessing_on_folds", required=False, action='store_true',
                        help="Running postprocessing on each fold only makes sense when developing with nnU-Net and "
                             "closely observing the model performance on specific configurations. You do not need it "
                             "when applying nnU-Net because the postprocessing for this will be determined only once "
                             "all five folds have been trained and nnUNet_find_best_configuration is called. Usually "
                             "running postprocessing on each fold is computationally cheap, but some users have "
                             "reported issues with very large images. If your images are large (>600x600x600 voxels) "
                             "you should consider setting this flag.")
    # parser.add_argument("--interp_order", required=False, default=3, type=int,
    #                     help="order of interpolation for segmentations. Testing purpose only. Hands off")
    # parser.add_argument("--interp_order_z", required=False, default=0, type=int,
    #                     help="order of interpolation along z if z is resampled separately. Testing purpose only. "
    #                          "Hands off")
    # parser.add_argument("--force_separate_z", required=False, default="None", type=str,
    #                     help="force_separate_z resampling. Can be None, True or False. Testing purpose only. Hands off")
    parser.add_argument('--val_disable_overwrite', action='store_false', default=True,
                        help='Validation does not overwrite existing segmentations')
    parser.add_argument('--disable_next_stage_pred', action='store_true', default=False,
                        help='do not predict next stage')
    parser.add_argument('-pretrained_weights', type=str, required=False, default=None,
                        help='path to nnU-Net checkpoint file to be used as pretrained model (use .model '
                             'file, for example model_final_checkpoint.model). Will only be used when actually training. '
                             'Optional. Beta. Use with caution.')

    args = parser.parse_args()

    task = args.task
    fold = args.fold
    network = args.network
    network_trainer = args.network_trainer
    validation_only = args.validation_only
    plans_identifier = args.p
    find_lr = args.find_lr
    weight_folder = args.weight_folder
    disable_postprocessing_on_folds = args.disable_postprocessing_on_folds

    use_compressed_data = args.use_compressed_data
    decompress_data = not use_compressed_data

    deterministic = args.deterministic
    valbest = args.valbest

    fp32 = args.fp32
    run_mixed_precision = not fp32

    val_folder = args.val_folder
    # interp_order = args.interp_order
    # interp_order_z = args.interp_order_z
    # force_separate_z = args.force_separate_z

    if not task.startswith("Task"):
        task_id = int(task)
        task = convert_id_to_task_name(task_id)

    if fold == 'all':
        pass
    else:
        fold = int(fold)

    # if force_separate_z == "None":
    #     force_separate_z = None
    # elif force_separate_z == "False":
    #     force_separate_z = False
    # elif force_separate_z == "True":
    #     force_separate_z = True
    # else:
    #     raise ValueError("force_separate_z must be None, True or False. Given: %s" % force_separate_z)
    if network_trainer in ['nnMTLTrainerV2Video', 
                           'nnMTLTrainerV2Flow', 
                           'nnMTLTrainerV2Flow3', 
                           'nnMTLTrainerV2Flow2', 
                           'nnMTLTrainerV2Flow4', 
                           'nnMTLTrainerV2Flow5', 
                           'nnMTLTrainerV2FlowLabeled', 
                           'nnMTLTrainerV2FlowPrediction',
                           'nnMTLTrainerV2FlowSimple',
                           'nnMTLTrainerV2FlowVideo',
                           'nnMTLTrainerV2StableDiffusion',
                           'nnMTLTrainerV2ControlNet',
                           'nnMTLTrainerV2FlowVariableLength',
                           'nnMTLTrainerV2FlowRecursiveVideo',
                           'nnMTLTrainerV2FlowRecursive',
                           'nnMTLTrainerV2FlowLib',
                           'nnMTLTrainerV2FlowSuccessive',
                           'nnMTLTrainerV2FlowSuccessiveSupervised',
                           'nnMTLTrainerV2FlowSuccessiveEmbedding',
                           'nnMTLTrainerV2FlowSuccessiveOther',
                           'ErrorCorrection',
                           'Final',
                           'FinalFlow',
                           'FinalFlowPred',
                           'SegPrediction',
                           'SegFlowGaussian',
                           'FinalFlowRaft',
                           'StartEnd',
                           'Interpolator',
                           'MemoryAdjacent',
                           'FlowSimple',
                           'nnMTLTrainerV2SegFlow',
                           'TemporalModel',
                           'MTLembedding',
                           'GlobalModel',
                           'nnMTLTrainerV2Raft',
                           'nnMTLTrainerV2Flow3D',
                           'nnMTLTrainerV2Flow3DSupervised',
                           'nnMTLTrainerV2FlowSuccessivePrediction',
                           'nnMTLTrainerV2Flow6']:
        if validation_only:
            config = read_config_video(os.path.join(weight_folder, 'config.yaml'))
        else:
            print(f'My config file is {args.config}')
            config = read_config_video(os.path.join(Path.cwd(), args.config))
    else:
        config = read_config(os.path.join(Path.cwd(), 'adversarial_acdc.yaml'), middle=False, video=False)


    plans_file, output_folder_name, dataset_directory, batch_dice, stage, \
        trainer_class = get_default_configuration(network, task, network_trainer, config, plans_identifier)

    #if network_trainer == 'nnUNetTrainerV2':
    #    plans_file, output_folder_name, dataset_directory, batch_dice, stage, \
    #    trainer_class = get_default_configuration(network, task, network_trainer, plans_identifier)
    #elif network_trainer == 'nnMTLTrainerV2':
    #    #plans_identifier = 'nnMTLPlansv2.1'
    #    plans_file, output_folder_name, dataset_directory, batch_dice, stage, \
    #    trainer_class = get_default_configuration_mtl(network, task, network_trainer, plans_identifier)

    if trainer_class is None:
        raise RuntimeError("Could not find trainer class in nnunet.training.network_training")

    if network == "3d_cascade_fullres":
        assert issubclass(trainer_class, (nnUNetTrainerCascadeFullRes, nnUNetTrainerV2CascadeFullRes)), \
            "If running 3d_cascade_fullres then your " \
            "trainer class must be derived from " \
            "nnUNetTrainerCascadeFullRes"
    else:
        assert issubclass(trainer_class,
                          nnUNetTrainer), "network_trainer was found but is not derived from nnUNetTrainer"
    
    if validation_only:
        output_folder_name = weight_folder

        
    if network_trainer == 'nnMTLTrainerV2':
        if any([x in task for x in ['31', '35', '32', '36']]):
            trainer = trainer_class(plans_file, fold, output_folder=output_folder_name, dataset_directory=dataset_directory,
                                batch_dice=batch_dice, stage=stage, unpack_data=decompress_data,
                                deterministic=deterministic,
                                fp16=run_mixed_precision, binary=False, config=config)
        else:
            trainer = trainer_class(plans_file, fold, output_folder=output_folder_name, dataset_directory=dataset_directory,
                                batch_dice=batch_dice, stage=stage, unpack_data=decompress_data,
                                deterministic=deterministic,
                                fp16=run_mixed_precision, config=config, binary=False)
    else:
        trainer = trainer_class(plans_file, fold, output_folder=output_folder_name, dataset_directory=dataset_directory,
                            batch_dice=batch_dice, stage=stage, unpack_data=decompress_data,
                            deterministic=deterministic,
                            fp16=run_mixed_precision, config=config)

    if args.disable_saving:
        trainer.save_final_checkpoint = False # whether or not to save the final checkpoint
        trainer.save_best_checkpoint = False  # whether or not to save the best checkpoint according to
        # self.best_val_eval_criterion_MA
        trainer.save_intermediate_checkpoints = True  # whether or not to save checkpoint_latest. We need that in case
        # the training chashes
        trainer.save_latest_only = True  # if false it will not store/overwrite _latest but separate files each

    trainer.initialize(not validation_only)

    if find_lr:
        trainer.find_lr()
    else:
        if not validation_only:
            if args.continue_training:
                # -c was set, continue a previous training and ignore pretrained weights
                trainer.load_latest_checkpoint()
            elif (not args.continue_training) and (args.pretrained_weights is not None):
                # we start a new training. If pretrained_weights are set, use them
                load_pretrained_weights(trainer.network, args.pretrained_weights)
            else:
                # new training without pretraine weights, do nothing
                pass

            trainer.run_training()
        else:
            if valbest:
                trainer.load_best_checkpoint(train=False)
            else:
                trainer.load_final_checkpoint(train=False)

        trainer.network.eval()

        if not validation_only:
            output_folder = os.path.join(trainer.log_dir, task, 'fold_' + str(fold))
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
        else:
            output_folder = os.path.join(weight_folder, task, 'fold_' + str(fold))
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

        if network_trainer in ['nnMTLTrainerV2']:

            # predict validation
            trainer.validate(save_softmax=args.npz, validation_folder_name=val_folder,
                            run_postprocessing_on_folds=not disable_postprocessing_on_folds,
                            overwrite=args.val_disable_overwrite, output_folder=output_folder)

        if network == '3d_lowres' and not args.disable_next_stage_pred:
            print("predicting segmentations for the next stage of the cascade")
            predict_next_stage(trainer, join(dataset_directory, trainer.plans['data_identifier'] + "_stage%d" % 1))


if __name__ == "__main__":
    main()
