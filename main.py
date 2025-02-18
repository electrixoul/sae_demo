import argparse
import yaml
import wandb
from config import get_device
from experiments import eeg_task, synthetic_task, feature_correlation, sae_3d_visualization, gpt2_task
from utils.data_utils import generate_synthetic_data
import traceback
import torch
import torch.distributed as dist

wandb_on = '0'

def load_config(config_path):
    if config_path:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    else:
        return {}


def init_wandb(config):
    default_config = {
        'project': 'synthetic_validation',
        'config': config
    }
    wandb.init(**default_config)        #project="image"


def run_experiment(config):

    print("wandb_on: ", config['wandb_on'])

    if config['wandb_on'] == '1':
        init_wandb(config)

    device = get_device()
    try:
        if config['experiment'] == 'synthetic':
            print("Running Synthetic Experiment...")
            synthetic_task.run(device, config)
        elif config['experiment'] == 'feature_correlation':
            print("Running Feature Correlation Experiment...")
            feature_correlation.run(device, config)
        elif config['experiment'] == 'sae_3d_visualization':
            print("Running SAE 3D Visualization Experiment...")
            sae_3d_visualization.run(device, config)
        elif config['experiment'] == 'gpt2':
            print("Running GPT-2 Experiment...")
            gpt2_task.run(device, config)
        elif config['experiment'] == 'eeg':
            print("Running EEG Experiment...")
            eeg_task.run(device, config)
        else:
            print(f"Unknown experiment: {config['experiment']}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        traceback.print_exc()
    finally:
        if config['wandb_on'] == '1':
            wandb.finish()


def main():
    parser = argparse.ArgumentParser(description='Run experiments or generate datasets')
    parser.add_argument('--config', type=str, help='Path to the configuration file for experiments or dataset generation')
    parser.add_argument('--wandb', type=str)
    args = parser.parse_args()

    # set global wandb_on to 1 if wandb is on
    global wandb_on
    wandb_on = args.wandb
    
    if args.config:
        config = load_config(args.config)
        # 向 config 中添加 wandb_on 字段和内容
        config['wandb_on'] = wandb_on
        run_experiment(config)
    else:
        print("Error: No configuration file specified. Please provide a configuration file using --config.")


if __name__ == '__main__':

    main()
