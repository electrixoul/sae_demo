import os
import argparse
import yaml
import wandb
from config import get_device
from experiments import eeg_task, synthetic_task, feature_correlation, sae_3d_visualization, gpt2_task
from utils.data_utils import generate_synthetic_data
import traceback
import torch
import torch.distributed as dist
from utils.general_utils import calculate_MMCS, load_specific_run, load_true_features_from_run
from utils.data_utils import generate_synthetic_data
from config import get_device
from models.sae import SparseAutoencoder
from utils import eeg_utils_comments
import pyedflib
from collections import Counter


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
    eeg_data_dir = "eeg_data"
    processed_data_file = "processed_data.pt"
    os.makedirs(eeg_data_dir, exist_ok=True)
    eeg_utils_comments.download_eeg_data(
        config['data']['eeg_data_url'],
        os.environ['EEG_USERNAME'],
        os.environ['EEG_PASSWORD'],
        eeg_data_dir
    )

    edf_files = eeg_utils_comments.find_edf_files(eeg_data_dir)
    sampling_rates = []
    for file_path in edf_files:
        f = pyedflib.EdfReader(file_path)
        # print(file_path)
        n = f.signals_in_file
        for i in range(n):
            fs = f.getSampleFrequency(i)
            sampling_rates.append(fs)
        f._close()

    sampling_rate_counts = Counter(sampling_rates)
    most_common_fs, _ = sampling_rate_counts.most_common(1)[0]
    print(f"Most common sampling rate across all files: {most_common_fs} Hz")

    n_channels, input_size = eeg_utils_comments.preprocess_and_save_data2(
        eeg_data_dir,
        processed_data_file,
        config['hyperparameters']['segment_length_sec'],
        config['hyperparameters']['lowcut'],
        config['hyperparameters']['highcut'],
        config['hyperparameters']['filter_order'],
        most_common_fs
    )

    print("n_channels, input_size : ", n_channels, input_size)


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

    config['hyperparameters']['input_size'] = 8192

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
