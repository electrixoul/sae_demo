import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from PIL import Image
import glob
from models.sae import SparseAutoencoder
from mice_sae_train import MiceDataset, safe_title
import random
from scipy.stats import pearsonr
from collections import defaultdict

# 配置中文字体支持
plt.rcParams['font.sans-serif'] = ['AR PL UKai CN', 'AR PL UMing CN', 'DejaVu Sans', 'WenQuanYi Zen Hei', 'WenQuanYi Micro Hei', 'SimHei', 'Microsoft YaHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

def load_sae_model(model_path, config, device):
    """加载预训练的SAE模型"""
    model = SparseAutoencoder(config)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"模型成功从 {model_path} 加载")
    else:
        print(f"未找到模型文件 {model_path}")
        return None
    
    model = model.to(device)
    return model

def analyze_feature_importance(model, dataloader, device, sae_id=0):
    """
    分析特征重要性，使用多种指标：
    1. 激活频率：特征被激活的频率
    2. 平均激活强度：特征被激活时的平均强度
    3. 重建贡献：特征对重建的贡献度
    """
    print("分析特征重要性...")
    model.eval()
    
    # 隐藏层大小
    hidden_size = model.encoders[0].weight.shape[0]
    
    # 统计每个特征的激活频率和强度
    activation_counts = np.zeros(hidden_size)
    activation_strength_sum = np.zeros(hidden_size)
    activation_strength_when_active = defaultdict(list)
    total_samples = 0
    
    # 统计每个特征的重建贡献（通过针对性地将特征设为0测量重建质量下降）
    feature_importance_by_reconstruction = []
    
    with torch.no_grad():
        # 第一遍：统计激活频率和强度
        for batch_idx, ((X_batch,), _) in enumerate(dataloader):
            X_batch = X_batch.to(device)
            total_samples += len(X_batch)
            
            # 前向传播
            outputs, activations, indices = model.forward_with_encoded(X_batch)
            
            # 获取激活和索引
            batch_activations = activations[sae_id].cpu().numpy()
            batch_indices = indices[sae_id].cpu().numpy()
            
            # 更新统计
            for i, sample_indices in enumerate(batch_indices):
                activation_counts[sample_indices] += 1
                
                # 获取激活值
                for idx, feature_idx in enumerate(sample_indices):
                    act_value = batch_activations[i, feature_idx]
                    activation_strength_sum[feature_idx] += act_value
                    activation_strength_when_active[feature_idx].append(act_value)
            
            if batch_idx % 5 == 0:
                print(f"已处理 {batch_idx * len(X_batch)} 个样本...")

        # 计算每个特征的平均激活强度
        mean_activation_when_active = np.zeros(hidden_size)
        for feature_idx in range(hidden_size):
            if activation_counts[feature_idx] > 0:
                mean_activation_when_active[feature_idx] = np.mean(activation_strength_when_active[feature_idx])
        
        # 计算激活频率
        activation_frequency = activation_counts / max(1, total_samples)
        
        # 第二遍：评估每个特征对重建的贡献（可选，计算量较大）
        evaluate_reconstruction = False
        if evaluate_reconstruction:
            # 使用一个小批量进行评估
            small_batch = next(iter(dataloader))[0][0][:5].to(device)  # 只取5个样本
            _, base_activations, base_indices = model.forward_with_encoded(small_batch)
            base_output = outputs[sae_id][:5]
            
            # 计算基准重建质量
            original = small_batch.cpu().numpy()
            base_reconstructed = base_output.cpu().numpy()
            base_corrs = [pearsonr(original[i], base_reconstructed[i])[0] for i in range(len(original))]
            base_corr = np.mean(base_corrs)
            
            # 评估每个特征的贡献
            for feature_idx in range(hidden_size):
                if activation_frequency[feature_idx] < 0.01:  # 只测试常用特征
                    continue
                    
                # 复制激活并"屏蔽"当前特征
                for i in range(len(small_batch)):
                    idx_in_batch = np.where(base_indices[sae_id][i].cpu().numpy() == feature_idx)[0]
                    if len(idx_in_batch) > 0:  # 如果这个特征被激活了
                        temp_activations = base_activations[sae_id][i].clone()
                        temp_activations[feature_idx] = 0  # 将特征激活设为0
                        
                        # 用修改后的激活进行解码
                        reconstructed = model.decoders[sae_id](temp_activations.unsqueeze(0))
                        
                        # 计算相关系数
                        corr = pearsonr(small_batch[i].cpu().numpy(), reconstructed[0].cpu().numpy())[0]
                        
                        # 记录重建质量下降
                        contribution = base_corrs[i] - corr
                        feature_importance_by_reconstruction.append((feature_idx, contribution))
                        break  # 只测一个样本即可
                        
        # 组合所有指标计算最终重要性分数
        importance_scores = 0.7 * activation_frequency + 0.3 * mean_activation_when_active
        
        # 返回特征重要性信息
        return {
            'activation_frequency': activation_frequency,
            'mean_activation_when_active': mean_activation_when_active,
            'importance_scores': importance_scores,
            'reconstruction_contribution': feature_importance_by_reconstruction
        }

def visualize_feature_importance(importance_info, save_dir, k=100):
    """可视化特征重要性分析结果"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 获取重要性分数
    importance_scores = importance_info['importance_scores']
    activation_frequency = importance_info['activation_frequency']
    mean_activation = importance_info['mean_activation_when_active']
    
    # 特征索引排序（按重要性降序）
    sorted_indices = np.argsort(importance_scores)[::-1]
    
    # 1. 绘制前k个重要特征的重要性分数
    plt.figure(figsize=(12, 6))
    plt.bar(range(k), importance_scores[sorted_indices[:k]], color='skyblue')
    plt.xlabel(safe_title('特征索引（按重要性排序）', 'Feature Index (Sorted by Importance)'))
    plt.ylabel(safe_title('重要性分数', 'Importance Score'))
    plt.title(safe_title(f'前{k}个重要特征的重要性分数', f'Importance Scores of Top {k} Features'))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # 保存图像
    save_path = os.path.join(save_dir, 'feature_importance_scores.png')
    plt.savefig(save_path)
    plt.close()
    print(f"特征重要性分数可视化已保存到 {save_path}")
    
    # 2. 绘制前k个重要特征的激活频率
    plt.figure(figsize=(12, 6))
    plt.bar(range(k), activation_frequency[sorted_indices[:k]], color='lightgreen')
    plt.xlabel(safe_title('特征索引（按重要性排序）', 'Feature Index (Sorted by Importance)'))
    plt.ylabel(safe_title('激活频率', 'Activation Frequency'))
    plt.title(safe_title(f'前{k}个重要特征的激活频率', f'Activation Frequency of Top {k} Features'))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # 保存图像
    save_path = os.path.join(save_dir, 'feature_activation_frequency.png')
    plt.savefig(save_path)
    plt.close()
    print(f"特征激活频率可视化已保存到 {save_path}")
    
    # 3. 绘制前k个重要特征的平均激活强度
    plt.figure(figsize=(12, 6))
    plt.bar(range(k), mean_activation[sorted_indices[:k]], color='salmon')
    plt.xlabel(safe_title('特征索引（按重要性排序）', 'Feature Index (Sorted by Importance)'))
    plt.ylabel(safe_title('平均激活强度', 'Mean Activation Strength'))
    plt.title(safe_title(f'前{k}个重要特征的平均激活强度', f'Mean Activation Strength of Top {k} Features'))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # 保存图像
    save_path = os.path.join(save_dir, 'feature_activation_strength.png')
    plt.savefig(save_path)
    plt.close()
    print(f"特征激活强度可视化已保存到 {save_path}")
    
    return sorted_indices

def visualize_sorted_features(model, sorted_indices, save_dir, n_features=100, image_size=64):
    """可视化按重要性排序的特征"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 从模型获取编码器权重
    with torch.no_grad():
        weights = model.encoders[0].weight.data.cpu().numpy()
    
    # 选择前n_features个最重要的特征
    top_feature_indices = sorted_indices[:n_features]
    top_weights = weights[top_feature_indices]
    
    # 归一化权重以便可视化
    weights_min = top_weights.min()
    weights_max = top_weights.max()
    weights_normalized = (top_weights - weights_min) / (weights_max - weights_min)
    
    # 创建图表
    grid_size = int(np.ceil(np.sqrt(n_features)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    fig.suptitle(safe_title('按重要性排序的Mice SAE特征（前100个）', 'Mice SAE Features Sorted by Importance (Top 100)'), fontsize=16)
    
    # 扁平化axes数组以便索引
    axes = axes.flatten()
    
    # 绘制每个特征
    for i in range(n_features):
        feature = weights_normalized[i].reshape(image_size, image_size)
        axes[i].imshow(feature, cmap='viridis')
        axes[i].set_title(f"#{top_feature_indices[i]}", fontsize=8)
        axes[i].axis('off')
    
    # 处理多余的子图
    for i in range(n_features, grid_size * grid_size):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    
    # 保存图像
    save_path = os.path.join(save_dir, 'mice_sae_features_sorted.png')
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"按重要性排序的特征可视化已保存到 {save_path}")
    
    return top_feature_indices

def main():
    # 设置随机种子确保结果可复现
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # 设置设备 - 优先使用GPU，加速处理
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 特定尺寸参数
    image_size = 64  # 64x64
    input_size = image_size * image_size  # 4096
    hidden_size = 8192  # 隐藏层大小
    
    # 配置参数
    config = {
        "input_size": input_size,      # 4096
        "image_size": image_size,      # 64
        "hidden_size": hidden_size,    # 8192 
        "k_sparse": 100,               # 稀疏参数k
        "num_saes": 5                  # SAE数量
    }
    
    # 创建保存目录
    save_dir = "visualizations/mice/importance"
    os.makedirs(save_dir, exist_ok=True)
    
    # 加载预训练的SAE模型
    model_path = "mice_sae_models/mice_sae_epoch_20.pth"  # 使用最后一个周期的模型
    model = load_sae_model(model_path, config, device)
    
    if model is None:
        print("模型加载失败，退出程序")
        return
    
    # 准备Mice数据集
    data_dir = "data/Mice"
    target_size = (image_size, image_size)
    
    mice_dataset = MiceDataset(
        data_dir=data_dir, 
        target_size=target_size,
        augment=True,
        augment_factor=5
    )
    
    # 处理数据增强（必须调用来初始化图像列表）
    mice_dataset.process_augmentation()
    
    # 创建数据加载器
    batch_size = 16
    mice_dataloader = DataLoader(
        mice_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )
    
    # 分析特征重要性
    importance_info = analyze_feature_importance(model, mice_dataloader, device)
    
    # 可视化特征重要性
    sorted_indices = visualize_feature_importance(importance_info, save_dir)
    
    # 可视化按重要性排序的特征
    visualize_sorted_features(model, sorted_indices, save_dir)
    
    print("\n特征重要性分析完成，结果保存在:", save_dir)

if __name__ == "__main__":
    main()
