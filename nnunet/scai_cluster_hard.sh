#!/bin/bash
#SBATCH --job-name=gpu_mono          # nom du job
##SBATCH -C v100-16g                  # reserver des GPU 16 Go seulement
#SBATCH --partition=hard
##SBATCH --qos=qos_gpu-t4            # qos_gpu-t4 qos_gpu-dev qos_gpu-t3
#SBATCH --nodes=1                    # on demande un noeud
#SBATCH --ntasks-per-node=1          # avec une tache par noeud (= nombre de GPU ici)
#SBATCH --gres=gpu:1                 # nombre de GPU (1/4 des GPU)
#SBATCH --cpus-per-task=10           # nombre de coeurs CPU par tache (1/4 du noeud 4-GPU)
##SBATCH --cpus-per-task=3           # nombre de coeurs CPU par tache (pour gpu_p2 : 1/8 du noeud 8-GPU)
# /!\ Attention, "multithread" fait reference Ã  l'hyperthreading dans la terminologie Slurm
#SBATCH --hint=nomultithread         # hyperthreading desactive
#SBATCH --time=100:00:00          # 48:00:00 temps maximum d'execution demande (HH:MM:SS) 00:05:00 20:00:00  
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
#python run/run_training.py 2d nnMTLTrainerV2 Task026_MMs 0 -p custom_experiment_planner
python run/run_training.py 2d nnMTLTrainerV2 Task027_ACDC 0 -p custom_experiment_planner