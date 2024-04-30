#!/bin/bash
#SBATCH --array=1-4   # Number of configuration files
#SBATCH --job-name=gpu_mono          # nom du job
#SBATCH -C v100-16g                  # reserver des GPU 16 Go seulement
##SBATCH --partition=gpu_p2          # de-commente pour la partition gpu_p2
#SBATCH --qos=qos_gpu-t3            # qos_gpu-t4 qos_gpu-dev qos_gpu-t3
#SBATCH --nodes=1                    # on demande un noeud
#SBATCH --ntasks-per-node=1          # avec une tache par noeud (= nombre de GPU ici)
#SBATCH --gres=gpu:1                 # nombre de GPU (1/4 des GPU)
#SBATCH --cpus-per-task=10           # nombre de coeurs CPU par tache (1/4 du noeud 4-GPU)
##SBATCH --cpus-per-task=3           # nombre de coeurs CPU par tache (pour gpu_p2 : 1/8 du noeud 8-GPU)
# /!\ Attention, "multithread" fait reference Ã  l'hyperthreading dans la terminologie Slurm
#SBATCH --hint=nomultithread         # hyperthreading desactive
#SBATCH --time=20:00:00          # 48:00:00 temps maximum d'execution demande (HH:MM:SS) 00:05:00 20:00:00  
#SBATCH --output=gpu_mono%j.out      # nom du fichier de sortie
#SBATCH --error=gpu_mono%j.out       # nom du fichier d'erreur (ici commun avec la sortie)

# nettoyage des modules charges en interactif et herites par defaut
module purge

# chargement des modules
module load pytorch-gpu/py3/1.11.0

# echo des commandes lancees
set -x

export nnUNet_raw_data_base="out/nnUNet_raw_data_base"
export nnUNet_preprocessed="out/nnUNet_preprocessed"
export RESULTS_FOLDER="out/nnUNet_trained_models"

# execution du code
#python run/run_training.py 2d nnMTLTrainerV2 Task026_MMs 0 -p custom_experiment_planner --deterministic
#python run/run_training.py 2d nnMTLTrainerV2 Task027_ACDC 0 -p custom_experiment_planner --deterministic
#python run/run_training.py 2d nnMTLTrainerV2Video Task027_ACDC 0 -p custom_experiment_planner --deterministic
#python run/run_training.py 2d nnMTLTrainerV2Flow Task031_ACDC 0 -p custom_experiment_planner --deterministic
#python run/run_training.py 2d nnMTLTrainerV2Flow4 Task031_ACDC 0 -p custom_experiment_planner --deterministic
#python run/run_training.py 2d nnMTLTrainerV2Flow5 Task031_ACDC 0 -p custom_experiment_planner --deterministic
#python run/run_training.py 2d nnMTLTrainerV2Flow6 Task031_ACDC 0 -p custom_experiment_planner --deterministic
#python run/run_training.py 2d nnMTLTrainerV2FlowLabeled Task031_ACDC 0 -p custom_experiment_planner --deterministic
#python run/run_training.py 2d nnMTLTrainerV2FlowPrediction Task031_ACDC 0 -p custom_experiment_planner --deterministic
#python run/run_training.py 2d nnMTLTrainerV2StableDiffusion Task031_ACDC 0 -p custom_experiment_planner --deterministic
#python run/run_training.py 2d nnMTLTrainerV2ControlNet Task031_ACDC 0 -p custom_experiment_planner --deterministic
#python run/run_training.py 2d nnMTLTrainerV2FlowRecursive Task031_ACDC 0 -p custom_experiment_planner --deterministic
#python run/run_training.py 2d nnMTLTrainerV2FlowRecursiveVideo Task031_ACDC 0 -p custom_experiment_planner --deterministic
#python run/run_training.py 2d nnMTLTrainerV2FlowLib Task032_Lib 0 -p custom_experiment_planner --deterministic
#python run/run_training.py 2d nnMTLTrainerV2FlowLib Task032_Lib 0 -p custom_experiment_planner --deterministic -val -w out/nnUNet_trained_models/nnUNet/2d/Task032_Lib/nnMTLTrainerV2FlowLib__custom_experiment_planner/fold_0/
#python run/run_training.py 2d nnMTLTrainerV2FlowSimple Task031_ACDC 0 -p custom_experiment_planner --deterministic
#python run/run_training.py 2d nnMTLTrainerV2FlowSimple Task032_Lib 0 -p custom_experiment_planner --deterministic -val -w out/nnUNet_trained_models/nnUNet/2d/Task031_ACDC/nnMTLTrainerV2FlowSimple__custom_experiment_planner/fold_0/2023-09-19_19H40/
#python run/run_training.py 2d nnMTLTrainerV2FlowSimple Task032_Lib 0 -p custom_experiment_planner --deterministic
#python run/run_training.py 2d nnMTLTrainerV2FlowRecursiveVideo Task032_Lib 0 -p custom_experiment_planner --deterministic
#python run/run_training.py 2d nnMTLTrainerV2FlowVideo Task032_Lib 0 -p custom_experiment_planner --deterministic
#python run/run_training.py 2d nnMTLTrainerV2FlowSuccessive Task032_Lib 0 -p custom_experiment_planner --deterministic
#python run/run_training.py 2d nnMTLTrainerV2FlowSuccessiveDouble Task032_Lib 0 -p custom_experiment_planner --deterministic
#python run/run_training.py 2d ErrorCorrection Task032_Lib 0 -p custom_experiment_planner --deterministic
#python run/run_training.py 2d nnMTLTrainerV2FlowSuccessiveOther Task032_Lib 0 -p custom_experiment_planner --deterministic
#python run/run_training.py 2d nnMTLTrainerV2Flow3D Task032_Lib 0 -p custom_experiment_planner --deterministic
#python run/run_training.py 2d FlowSimple Task032_Lib 0 -p custom_experiment_planner --deterministic
#python run/run_training.py 2d Interpolator Task032_Lib 0 -p custom_experiment_planner --deterministic
#python run/run_training.py 2d GlobalModel Task032_Lib 0 -p custom_experiment_planner --deterministic
#python run/run_training.py 2d nnMTLTrainerV2FlowSuccessiveEmbedding Task032_Lib 0 -p custom_experiment_planner --deterministic
#python run/run_training.py 2d nnMTLTrainerV2FlowSuccessivePrediction Task032_Lib 0 -p custom_experiment_planner --deterministic
#python run/run_training.py 2d nnMTLTrainerV2FlowSuccessiveSupervised Task032_Lib 0 -p custom_experiment_planner --deterministic
#python run/run_training.py 2d MTLembedding Task032_Lib 0 -p custom_experiment_planner --deterministic
#python run/run_training.py 2d StartEnd Task032_Lib 0 -p custom_experiment_planner --deterministic
#python run/run_training.py 2d Final Task032_Lib 0 -p custom_experiment_planner --deterministic
#python run/run_training.py 2d nnMTLTrainerV2FlowSuccessive Task045_Lib 0 -p custom_experiment_planner --deterministic
#python run/run_training.py 2d FinalFlow Task045_Lib 0 -p custom_experiment_planner --deterministic
#python run/run_training.py 2d nnMTLTrainerV2FlowSuccessive Task032_Lib 0 -p custom_experiment_planner -config config9.yaml --deterministic
#python run/run_training.py 2d FinalFlow Task045_Lib 0 -p custom_experiment_planner -config config${SLURM_ARRAY_TASK_ID}.yaml --deterministic
#python run/run_training.py 2d SegPrediction Task045_Lib 0 -p custom_experiment_planner -config config${SLURM_ARRAY_TASK_ID}.yaml --deterministic
python run/run_training.py 2d SegFlowGaussian Task045_Lib 0 -p custom_experiment_planner -config config${SLURM_ARRAY_TASK_ID}.yaml --deterministic

sleep 1

#python voxelmorph_saver_Lib.py