import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from neural_sae import NeuralSparseAutoencoder
from neural_dataset import create_neural_dataloaders
import os
import argparse
from typing import Dict, List, Tuple


def load_trained_model(model_path: str, device: str) -> NeuralSparseAutoencoder:
    """加载训练好的neural SAE模型"""
    print(f"加载模型: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        config = {
            "input_size": 9941,
            "hidden_size": 9941,
            "k_sparse": 497,
            "num_saes": 3,
            "learning_rate": 0.001,
            "ensemble_consistency_weight": 0.01,
            "reinit_threshold": 2.0,
            "warmup_steps": 100,
            "use_amp": False
        }
        print("警告: 未找到模型配置，使用默认配置")
    
    model = NeuralSparseAutoencoder(config)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    print(f"模型加载成功！")
    print(f"  输入维度: {config['input_size']}")
    print(f"  隐藏层维度: {config['hidden_size']}")
    print(f"  SAE数量: {config['num_saes']}")
    
    return model


def extract_data_and_representations(model: NeuralSparseAutoencoder, 
                                    data_loader, 
                                    device: str,
                                    sae_index: int = 0) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """同时提取原始数据和SAE中间层表征"""
    print(f"提取原始数据和SAE #{sae_index}的中间层表征...")
    
    all_representations = []
    all_original_data = []
    all_labels = []
    
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            X_batch = batch[0].to(device)
            
            # 获取SAE的编码表征
            encoder = model.encoders[sae_index]
            encoded = encoder(X_batch)
            
            # 保存数据
            all_representations.append(encoded.cpu().numpy())
            all_original_data.append(X_batch.cpu().numpy())
            
            # 创建批次标签
            batch_labels = [f"batch_{batch_idx}_sample_{i}" for i in range(X_batch.shape[0])]
            all_labels.extend(batch_labels)
    
    # 合并所有数据
    representations = np.vstack(all_representations)
    original_data = np.vstack(all_original_data)
    
    print(f"提取完成！")
    print(f"  原始数据形状: {original_data.shape}")
    print(f"  SAE表征形状: {representations.shape}")
    print(f"  样本数量: {len(all_labels)}")
    
    return original_data, representations, all_labels


def load_all_neural_data(data_dir: str, device: str) -> Tuple[torch.utils.data.DataLoader, Dict]:
    """加载所有神经数据"""
    print("加载所有神经活动数据...")
    
    available_stimuli = []
    
    if os.path.exists(data_dir):
        for file in os.listdir(data_dir):
            if file.startswith('responses_') and file.endswith('.npy'):
                stimulus_name = file.replace('responses_', '').replace('.npy', '')
                available_stimuli.append(stimulus_name)
        
        print(f"找到刺激类型: {available_stimuli}")
    else:
        print(f"错误: 数据目录 {data_dir} 不存在")
        return None, None
    
    try:
        full_loader, _, dataset_metadata = create_neural_dataloaders(
            data_dir=data_dir,
            stimulus_types=available_stimuli,
            batch_size=32,
            train_split=1.0,
            normalize=True,
            num_workers=0,
            pin_memory=False if str(device).startswith('mps') else True
        )
        
        dataset_metadata['stimulus_types'] = available_stimuli
        return full_loader, dataset_metadata
        
    except Exception as e:
        print(f"数据加载失败: {e}")
        return None, None


def perform_dual_tsne_analysis(original_data: np.ndarray, 
                              sae_representations: np.ndarray,
                              labels: List[str],
                              save_prefix: str = "neural_comparison") -> Tuple[np.ndarray, np.ndarray]:
    """对原始数据和SAE表征分别执行t-SNE分析"""
    
    # 设置t-SNE参数
    tsne_params = {
        'n_components': 2,
        'perplexity': min(30, len(original_data) // 4),
        'learning_rate': 200,
        'n_iter': 1000,
        'random_state': 42,
        'metric': 'euclidean'
    }
    
    print(f"对原始数据执行t-SNE分析...")
    print(f"  原始数据形状: {original_data.shape}")
    tsne1 = TSNE(**tsne_params)
    original_tsne = tsne1.fit_transform(original_data)
    print(f"  原始数据t-SNE完成: {original_tsne.shape}")
    
    print(f"对SAE表征执行t-SNE分析...")
    print(f"  SAE表征形状: {sae_representations.shape}")
    tsne2 = TSNE(**tsne_params)
    sae_tsne = tsne2.fit_transform(sae_representations)
    print(f"  SAE表征t-SNE完成: {sae_tsne.shape}")
    
    # 保存结果
    np.savez(f"{save_prefix}_results.npz", 
             original_tsne=original_tsne,
             sae_tsne=sae_tsne,
             original_data=original_data,
             sae_representations=sae_representations,
             labels=labels)
    
    return original_tsne, sae_tsne


def create_stimulus_labels(labels: List[str], 
                          stimulus_types: List[str]) -> List[str]:
    """为每个样本创建刺激类型标签"""
    stimulus_labels = []
    
    samples_per_stimulus = len(labels) // len(stimulus_types)
    
    for i, stimulus in enumerate(stimulus_types):
        start_idx = i * samples_per_stimulus
        end_idx = start_idx + samples_per_stimulus
        
        for j in range(start_idx, min(end_idx, len(labels))):
            stimulus_labels.append(stimulus)
    
    # 处理剩余样本
    while len(stimulus_labels) < len(labels):
        stimulus_labels.append(stimulus_types[-1])
    
    return stimulus_labels


def visualize_tsne_comparison(original_tsne: np.ndarray,
                             sae_tsne: np.ndarray, 
                             stimulus_labels: List[str],
                             save_path: str = "neural_tsne_comparison.png"):
    """可视化原始数据和SAE表征的t-SNE对比"""
    print("生成原始数据vs SAE表征t-SNE对比图...")
    
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    unique_stimuli = list(set(stimulus_labels))
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_stimuli)))
    
    # 第一行：原始神经数据t-SNE
    # 左上：原始数据按刺激类型着色
    ax1 = axes[0, 0]
    for i, stimulus in enumerate(unique_stimuli):
        mask = np.array(stimulus_labels) == stimulus
        ax1.scatter(original_tsne[mask, 0], original_tsne[mask, 1], 
                   c=[colors[i]], label=stimulus, alpha=0.7, s=50)
    
    ax1.set_xlabel('t-SNE Component 1', fontsize=12)
    ax1.set_ylabel('t-SNE Component 2', fontsize=12)
    ax1.set_title('Original Neural Data (by Stimulus Type)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=9, loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # 右上：原始数据密度图
    ax2 = axes[0, 1]
    hist_orig, xedges_orig, yedges_orig = np.histogram2d(original_tsne[:, 0], original_tsne[:, 1], bins=30)
    extent_orig = [xedges_orig[0], xedges_orig[-1], yedges_orig[0], yedges_orig[-1]]
    
    im1 = ax2.imshow(hist_orig.T, extent=extent_orig, origin='lower', cmap='viridis', alpha=0.8)
    ax2.scatter(original_tsne[:, 0], original_tsne[:, 1], 
               c='white', s=20, alpha=0.6, edgecolors='black', linewidth=0.5)
    
    ax2.set_xlabel('t-SNE Component 1', fontsize=12)
    ax2.set_ylabel('t-SNE Component 2', fontsize=12)
    ax2.set_title('Original Neural Data (Density)', fontsize=14, fontweight='bold')
    cbar1 = plt.colorbar(im1, ax=ax2)
    cbar1.set_label('Sample Density', fontsize=10)
    
    # 第二行：SAE表征t-SNE
    # 左下：SAE表征按刺激类型着色
    ax3 = axes[1, 0]
    for i, stimulus in enumerate(unique_stimuli):
        mask = np.array(stimulus_labels) == stimulus
        ax3.scatter(sae_tsne[mask, 0], sae_tsne[mask, 1], 
                   c=[colors[i]], label=stimulus, alpha=0.7, s=50)
    
    ax3.set_xlabel('t-SNE Component 1', fontsize=12)
    ax3.set_ylabel('t-SNE Component 2', fontsize=12)
    ax3.set_title('SAE Representations (by Stimulus Type)', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=9, loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    # 右下：SAE表征密度图
    ax4 = axes[1, 1]
    hist_sae, xedges_sae, yedges_sae = np.histogram2d(sae_tsne[:, 0], sae_tsne[:, 1], bins=30)
    extent_sae = [xedges_sae[0], xedges_sae[-1], yedges_sae[0], yedges_sae[-1]]
    
    im2 = ax4.imshow(hist_sae.T, extent=extent_sae, origin='lower', cmap='viridis', alpha=0.8)
    ax4.scatter(sae_tsne[:, 0], sae_tsne[:, 1], 
               c='white', s=20, alpha=0.6, edgecolors='black', linewidth=0.5)
    
    ax4.set_xlabel('t-SNE Component 1', fontsize=12)
    ax4.set_ylabel('t-SNE Component 2', fontsize=12)
    ax4.set_title('SAE Representations (Density)', fontsize=14, fontweight='bold')
    cbar2 = plt.colorbar(im2, ax=ax4)
    cbar2.set_label('Sample Density', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"t-SNE对比图保存到: {save_path}")
    
    return fig


def print_comparison_stats(original_tsne: np.ndarray, 
                          sae_tsne: np.ndarray,
                          stimulus_labels: List[str]):
    """打印对比统计信息"""
    unique_stimuli = list(set(stimulus_labels))
    
    print(f"\n=== 原始数据 vs SAE表征 t-SNE对比分析 ===")
    print(f"总样本数: {len(original_tsne)}")
    print(f"刺激类型数: {len(unique_stimuli)}")
    print(f"刺激类型: {unique_stimuli}")
    
    # 每种刺激类型的样本数
    for stimulus in unique_stimuli:
        count = sum(1 for label in stimulus_labels if label == stimulus)
        print(f"  {stimulus}: {count} 样本")
    
    print(f"\n原始神经数据t-SNE空间:")
    print(f"  Component 1 范围: [{original_tsne[:, 0].min():.2f}, {original_tsne[:, 0].max():.2f}]")
    print(f"  Component 2 范围: [{original_tsne[:, 1].min():.2f}, {original_tsne[:, 1].max():.2f}]")
    
    print(f"\nSAE表征t-SNE空间:")
    print(f"  Component 1 范围: [{sae_tsne[:, 0].min():.2f}, {sae_tsne[:, 0].max():.2f}]")
    print(f"  Component 2 范围: [{sae_tsne[:, 1].min():.2f}, {sae_tsne[:, 1].max():.2f}]")
    
    # 计算聚类分离度
    def calculate_cluster_separation(tsne_data, labels):
        separations = []
        for i, stim1 in enumerate(unique_stimuli):
            for j, stim2 in enumerate(unique_stimuli):
                if i < j:
                    mask1 = np.array(labels) == stim1
                    mask2 = np.array(labels) == stim2
                    
                    center1 = np.mean(tsne_data[mask1], axis=0)
                    center2 = np.mean(tsne_data[mask2], axis=0)
                    
                    distance = np.linalg.norm(center1 - center2)
                    separations.append(distance)
        
        return np.mean(separations)
    
    orig_separation = calculate_cluster_separation(original_tsne, stimulus_labels)
    sae_separation = calculate_cluster_separation(sae_tsne, stimulus_labels)
    
    print(f"\n聚类分离度分析:")
    print(f"  原始数据平均类间距离: {orig_separation:.2f}")
    print(f"  SAE表征平均类间距离: {sae_separation:.2f}")
    print(f"  分离度变化: {((sae_separation - orig_separation) / orig_separation * 100):+.1f}%")
    
    if sae_separation > orig_separation:
        print("  结论: SAE表征增强了不同刺激类型的分离度")
    else:
        print("  结论: SAE表征保持了原始数据的分离度")


def main():
    parser = argparse.ArgumentParser(description="原始神经数据vs SAE表征t-SNE对比分析")
    parser.add_argument('--model_path', type=str, default='neural_sae_20250606_173219.pth',
                      help='训练好的neural SAE模型路径')
    parser.add_argument('--data_dir', type=str, default='neural_activity_data_20250606_163121',
                      help='神经数据目录')
    parser.add_argument('--sae_index', type=int, default=0,
                      help='要分析的SAE索引 (0-2)')
    
    args = parser.parse_args()
    
    print("===== 原始神经数据 vs SAE表征 t-SNE对比分析 =====")
    print(f"模型路径: {args.model_path}")
    print(f"数据目录: {args.data_dir}")
    print(f"SAE索引: {args.sae_index}")
    
    # 检查文件是否存在
    if not os.path.exists(args.model_path):
        print(f"错误: 模型文件 {args.model_path} 不存在")
        return
    
    if not os.path.exists(args.data_dir):
        print(f"错误: 数据目录 {args.data_dir} 不存在")
        return
    
    # 获取设备
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("使用 MPS (Apple Silicon) 加速")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("使用 CUDA 加速")
    else:
        device = torch.device("cpu")
        print("使用 CPU")
    
    try:
        # 1. 加载训练好的模型
        model = load_trained_model(args.model_path, str(device))
        
        # 2. 加载所有神经数据
        data_loader, dataset_metadata = load_all_neural_data(args.data_dir, str(device))
        if data_loader is None:
            print("数据加载失败，程序退出")
            return
        
        # 3. 同时提取原始数据和SAE表征
        original_data, sae_representations, sample_labels = extract_data_and_representations(
            model, data_loader, str(device), args.sae_index
        )
        
        # 4. 创建刺激类型标签
        stimulus_types = dataset_metadata['stimulus_types']
        stimulus_labels = create_stimulus_labels(sample_labels, stimulus_types)
        
        # 5. 对原始数据和SAE表征分别执行t-SNE分析
        original_tsne, sae_tsne = perform_dual_tsne_analysis(
            original_data, sae_representations, sample_labels,
            f"neural_original_vs_sae{args.sae_index}"
        )
        
        # 6. 生成对比可视化
        comparison_path = f"neural_original_vs_sae{args.sae_index}_comparison.png"
        fig = visualize_tsne_comparison(original_tsne, sae_tsne, stimulus_labels, comparison_path)
        
        # 7. 打印统计分析
        print_comparison_stats(original_tsne, sae_tsne, stimulus_labels)
        
        print(f"\n=== 对比分析完成 ===")
        print(f"原始数据维度: {original_data.shape}")
        print(f"SAE表征维度: {sae_representations.shape}")
        print(f"结果保存: neural_original_vs_sae{args.sae_index}_results.npz")
        print(f"对比图保存: {comparison_path}")
        
    except Exception as e:
        print(f"程序执行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
