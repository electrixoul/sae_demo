import os
import argparse
import yaml
import wandb
from config import get_device
from experiments import eeg_task, synthetic_task, feature_correlation, sae_3d_visualization, gpt2_task
import traceback
import torch
from utils.general_utils import calculate_MMCS, load_specific_run, load_true_features_from_run
from models.sae import SparseAutoencoder
from utils import eeg_utils_comments
from torch.utils.data import DataLoader


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
    wandb.init(**default_config)


def run_experiment(config):
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
        wandb.finish()


def main():
    parser = argparse.ArgumentParser(description='Run experiments or generate datasets')
    parser.add_argument('--config', type=str, help='Path to the configuration file for experiments or dataset generation')
    args = parser.parse_args()

    config = load_config(args.config)

    # init_wandb(config)
    device = get_device()

    ''' load data
    '''
    processed_data_file = "processed_data.pt"

    eeg_dataset = eeg_utils_comments.EEGDataset(processed_data_file)

    dataloader = DataLoader(
        eeg_dataset,
        batch_size=config['hyperparameters']['training_batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    input_dim = eeg_dataset.segment_shape[0]

    config['hyperparameters']['input_size'] = input_dim

    print("input_dim : ", input_dim)
    

    ''' load model
    '''
    torch.set_default_dtype(torch.float32)
    run_id = config['run_id']
    print("run_id: ", run_id)
    specific_run = load_specific_run('synthetic_validation', run_id)

    # Load the full model state dict
    model_artifact = next(art for art in specific_run.logged_artifacts() if art.type == 'model')
    artifact_dir = model_artifact.download()
    model_path = os.path.join(artifact_dir, f"{specific_run.name}_epoch_1.pth")
    print("model_path: ", model_path)
    full_state_dict = torch.load(model_path, map_location=device)

    config['hyperparameters']['input_size'] = input_dim

    # Create a single model with both encoders
    model = SparseAutoencoder(config['hyperparameters'])
    model.load_state_dict(full_state_dict)
    model = model.to(device).to(torch.float32)

    # display model status
    print(model)
    # print the model parameters
    for name, param in model.named_parameters():
        print(name, param.shape)


    
if __name__ == '__main__':
    main()
