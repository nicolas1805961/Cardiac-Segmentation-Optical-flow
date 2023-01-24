from lib.training_utils import read_config
from run.default_configuration import get_default_configuration
import warnings
import argparse
from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("-b", '--batch_size', help="Optimal batch size", required=True)
parser.add_argument('-t', '--task_name', help='task name or task ID, required.', required=True)
parser.add_argument('-tr', '--trainer_class_name', help='trainer class', required=True)
parser.add_argument('-o', '--output_folder', help='output_folder', required=True)
parser.add_argument('-w', '--weight_folder', help='weight_folder', required=True)
args = parser.parse_args()
batch_size = int(args.batch_size)
trainer_class_name = args.trainer_class_name
task_name = args.task_name
output_folder = args.output_folder
weight_folder = args.weight_folder

if not task_name.startswith("Task"):
    task_id = int(task_name)
    task_name = convert_id_to_task_name(task_id)

config = read_config('adversarial_acdc.yaml', False, False)

plans_file, output_folder_name, dataset_directory, batch_dice, stage, \
        trainer_class = get_default_configuration('2d', task_name, trainer_class_name, config, 'custom_experiment_planner')

trainer = trainer_class(plans_file, 0, output_folder=weight_folder, dataset_directory=dataset_directory,
                            batch_dice=batch_dice, stage=stage, unpack_data=True, middle=False, video=False,
                            deterministic=True,
                            fp16=True)

trainer.load_final_checkpoint(train=False)

trainer.network.eval()

trainer.get_throughput(optimal_batch_size=batch_size, output_folder=output_folder)