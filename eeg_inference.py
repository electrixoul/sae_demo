import os
import argparse
import sys
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
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import numpy as np
from sklearn.decomposition import PCA


def progress_bar(current, total, barLength = 100):
    percent = float(current) * 100 / total
    arrow = '-' * int(percent/100 * barLength - 1) + '>'
    spaces = ' ' * (barLength - len(arrow))

    print('Progress: [%s%s] %d %%' % (arrow, spaces, percent), end='\r')
    sys.stdout.flush()

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
        batch_size=1,              # 定义每一批数据的大小
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
    # model_artifact = next(art for art in specific_run.logged_artifacts() if art.type == 'model')
    # artifact_dir = model_artifact.download()
    # model_path = os.path.join(artifact_dir, f"{specific_run.name}_epoch_1.pth")
    # model_path = "/home/t/workspace/AGI/THBI/sae_demo/artifacts/magic-surf-8_epoch_1:v0/magic-surf-8_epoch_1.pth"
    # model_path = "/home/t/workspace/AGI/THBI/sae_demo/artifacts/worthy-music-2_epoch_1:v0/worthy-music-2_epoch_1.pth"
    # model_path = "/home/t/workspace/AGI/THBI/mutual-feature-regularization/artifacts/lively-durian-4_epoch_10.pth"
    # model_path = "/home/t/workspace/AGI/THBI/mutual-feature-regularization/artifacts/lucky-violet-5_epoch_10.pth"
    model_path = "/home/t/workspace/AGI/THBI/mutual-feature-regularization/artifacts/peach-dream-8_epoch_7.pth"
    print("model_path: ", model_path)
    full_state_dict = torch.load(model_path, map_location=device)

    config['hyperparameters']['input_size'] = input_dim

    # Create a single model with both encoders
    model = SparseAutoencoder(config['hyperparameters'])
    model.load_state_dict(full_state_dict)
    # model = model.to(device).to(torch.float32)

    # display model status
    print(model)
    # print the model parameters
    for name, param in model.named_parameters():
        print(name, param.shape)

    sae_id = 1

    nn_activations = []

    for batch_num, (X_batch,) in enumerate(dataloader):

        progress_bar(batch_num, 3000)

        # print("shape of X_batch: ", X_batch.shape)
        X_batch = X_batch.to(device)
        # 打印 X_batch 中数据的具体类型，例如 torch.float32
        # print("X_batch type: ", X_batch.dtype)
        outputs, activations = model.forward_with_encoded(X_batch)
        # print("shape of outputs: ", len(outputs))
        # print("shape of outputs[0]: ", outputs[sae_id].shape)
        # print("shape of activations: ", activations[sae_id].shape)

        nn_fit = outputs[sae_id].detach().cpu().numpy()[0]
        raw_data = X_batch[0].detach().cpu().numpy()

        # # 计算 nn_fit 和 raw_data 之间的 pearson correlation
        # corr, _ = pearsonr(nn_fit, raw_data)
        # print("pearson correlation: ", corr)

        # # 将 outputs[0] 绘制成 plot
        # plt.plot(outputs[sae_id].detach().cpu().numpy()[0])
        # plt.plot(X_batch[0].detach().cpu().numpy())
        # plt.show()

        nn_activations.append(np.array(activations[sae_id].detach().cpu().numpy()[0]))

        if batch_num >= 3000:
            break

    nn_activations = np.array(nn_activations)
    print("shape of nn_fit_results: ", nn_activations.shape)

    # 对 nn_activations 进行 PCA
    pca = PCA()
    pca.fit(nn_activations)
    # print("explained_variance_ratio_: ", pca.explained_variance_ratio_)
    nn_activations_pca = pca.transform(nn_activations)

    # 绘制 PCA 结果到3D视图中
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(nn_activations_pca[:, 0], nn_activations_pca[:, 1], nn_activations_pca[:, 2])
    ax.scatter(nn_activations_pca[:, 3], nn_activations_pca[:, 4], nn_activations_pca[:, 5])
    plt.show()

    
if __name__ == '__main__':
    main()
