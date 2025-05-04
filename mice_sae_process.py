import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from scipy.stats import pearsonr
from PIL import Image
import glob
from models.sae import SparseAutoencoder
import time
import random

class MiceDataset(Dataset):
    def __init__(self, data_dir, target_size=(28, 28), transform=None, augment=True, augment_factor=5):
        """
        加载Mice数据集，预处理并可选地进行数据增强
        
        参数:
            data_dir (str): 数据目录路径
            target_size (tuple): 目标图像大小，默认为MNIST尺寸(28, 28)
            transform (torchvision.transforms): 转换函数
            augment (bool): 是否进行数据增强
            augment_factor (int): 每个原始图像生成的增强图像数量
        """
        self.data_dir = data_dir
        self.target_size = target_size
        self.augment = augment
        self.augment_factor = augment_factor
        
        # 查找所有PNG图像
        self.image_paths = []
        for category in ['1', '2']:
            folder_path = os.path.join(data_dir, category)
            if os.path.exists(folder_path):
                self.image_paths.extend(glob.glob(os.path.join(folder_path, '*.png')))
        
        # 指定默认转换
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(target_size),
                transforms.Grayscale(),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transform
            
        # 为数据增强创建额外的转换
        self.augment_transforms = [
            transforms.Compose([
                transforms.RandomRotation(15),
                transforms.RandomResizedCrop(target_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
                transforms.Grayscale(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
            ]),
            transforms.Compose([
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.Resize(target_size),
                transforms.Grayscale(),
                transforms.ToTensor(),
            ]),
            transforms.Compose([
                transforms.RandomVerticalFlip(p=1.0),
                transforms.Resize(target_size),
                transforms.Grayscale(),
                transforms.ToTensor(),
            ]),
            transforms.Compose([
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.Resize(target_size),
                transforms.Grayscale(),
                transforms.ToTensor(),
            ]),
            transforms.Compose([
                transforms.RandomPerspective(distortion_scale=0.2, p=1.0),
                transforms.Resize(target_size),
                transforms.Grayscale(),
                transforms.ToTensor(),
            ])
        ]
        
        # 处理数据增强
        self.process_augmentation()
        
    def process_augmentation(self):
        """处理数据增强并创建数据集"""
        self.images = []
        self.labels = []  # 存储分类标签 (1 或 2)
        self.paths = []   # 存储原始图像路径
        
        print(f"处理 {len(self.image_paths)} 个原始图像...")
        
        for path in self.image_paths:
            # 确定标签（基于文件夹名称）
            label = int(os.path.basename(os.path.dirname(path)))
            
            try:
                # 加载并转换原始图像
                img = Image.open(path).convert('RGB')
                transformed_img = self.transform(img)
                self.images.append(transformed_img)
                self.labels.append(label)
                self.paths.append(path)
                
                # 数据增强
                if self.augment:
                    # 对每张图像应用多种增强方法
                    for _ in range(self.augment_factor):
                        # 随机选择一种增强方法
                        aug_transform = random.choice(self.augment_transforms)
                        aug_img = aug_transform(img)
                        self.images.append(aug_img)
                        self.labels.append(label)
                        self.paths.append(path + f"_aug_{_}")
            except Exception as e:
                print(f"处理图像 {path} 时出错: {e}")
        
        print(f"数据集创建完成，共 {len(self.images)} 张图像")

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        """返回图像数据和标签"""
        image = self.images[idx]
        
        # 确保图像已经展平为一维向量，与MNIST模型兼容
        flattened_image = image.view(-1)
        
        # 确保向量长度为784
        if flattened_image.shape[0] != 784:
            # 如果不是784，则调整大小并重新展平
            resize_transform = transforms.Resize(self.target_size)
            resized_image = resize_transform(image.view(1, *image.shape))
            flattened_image = resized_image.view(-1)
            
            # 如果仍然不是784，进行截断或填充
            if flattened_image.shape[0] < 784:
                padded = torch.zeros(784)
                padded[:flattened_image.shape[0]] = flattened_image
                flattened_image = padded
            elif flattened_image.shape[0] > 784:
                flattened_image = flattened_image[:784]
        
        return (flattened_image,), self.labels[idx]

def load_sae_model(model_path, config, device):
    """加载预训练的SAE模型"""
    model = SparseAutoencoder(config)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"模型成功从 {model_path} 加载")
    else:
        print(f"未找到模型文件 {model_path}，使用随机初始化的模型")
    
    model = model.to(device)
    return model

def plot_mice_reconstruction(original, reconstructed, indices=None, k_sparse=None, corr=None, label=None, target_size=(28, 28)):
    """
    绘制Mice图像原始图和重构图的对比
    
    参数:
        original: 原始图像张量
        reconstructed: 重构图像张量
        indices: 稀疏编码中激活的索引
        k_sparse: k稀疏值
        corr: Pearson相关系数
        label: 图像标签
        target_size: 目标图像大小
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 原始图像
    axes[0].imshow(original.reshape(*target_size), cmap='gray')
    title = f'原始图像 (类别: {label})' if label is not None else '原始图像'
    axes[0].set_title(title)
    axes[0].axis('off')
    
    # 重构图像
    axes[1].imshow(reconstructed.reshape(*target_size), cmap='gray')
    axes[1].set_title(f'重构图像 (相关系数: {corr:.4f})')
    axes[1].axis('off')
    
    # 稀疏激活可视化
    if indices is not None and k_sparse is not None:
        # 创建一个激活热图
        activation_heatmap = np.zeros(1024)  # SAE隐藏层大小
        activation_heatmap[indices] = 1
        
        # 重塑为2D热图以便可视化
        rows = int(np.sqrt(len(activation_heatmap)))
        activation_2d = activation_heatmap.reshape(rows, -1)
        
        axes[2].imshow(activation_2d, cmap='hot')
        axes[2].set_title(f'稀疏激活 (k={k_sparse})')
        axes[2].axis('off')
    
    plt.tight_layout()
    return fig

def visualize_sae_features(model, device, save_dir, n_features=100):
    """可视化SAE学习到的特征"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 从模型获取编码器权重
    with torch.no_grad():
        weights = model.encoders[0].weight.data.cpu().numpy()
    
    # 选择前n_features个特征来可视化
    weights = weights[:n_features]
    
    # 归一化权重以便可视化
    weights_min = weights.min()
    weights_max = weights.max()
    weights_normalized = (weights - weights_min) / (weights_max - weights_min)
    
    # 创建图表
    grid_size = int(np.ceil(np.sqrt(n_features)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    axes = axes.flatten()
    
    # 绘制每个特征
    for i in range(n_features):
        feature = weights_normalized[i].reshape(28, 28)
        axes[i].imshow(feature, cmap='viridis')
        axes[i].axis('off')
    
    # 处理多余的子图
    for i in range(n_features, grid_size * grid_size):
        axes[i].axis('off')
    
    plt.suptitle('SAE学习到的特征在Mice数据集上的应用', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    
    # 保存图像
    save_path = os.path.join(save_dir, 'mice_sae_features.png')
    plt.savefig(save_path)
    plt.close(fig)
    
    print(f"SAE特征可视化已保存到 {save_path}")
    return save_path

def analyze_sparse_activations(model, dataloader, device, k_sparse=50, save_dir='visualizations'):
    """分析稀疏激活模式"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 收集激活统计信息
    activation_counts = np.zeros(1024)  # 假设隐藏层大小为1024
    total_samples = 0
    
    # 收集每个类别的平均激活
    class_activations = {}
    
    with torch.no_grad():
        for batch_idx, ((X_batch,), labels) in enumerate(dataloader):
            X_batch = X_batch.to(device)
            labels = labels.numpy()
            
            # 前向传播获取激活
            _, activations, indices = model.forward_with_encoded(X_batch)
            
            # 获取第一个SAE的激活
            batch_indices = indices[0].cpu().numpy()
            batch_activations = activations[0].cpu().numpy()
            
            # 统计激活频率
            for sample_indices in batch_indices:
                activation_counts[sample_indices] += 1
            total_samples += X_batch.size(0)
            
            # 按类别收集激活
            for i in range(len(labels)):
                label = int(labels[i])
                if label not in class_activations:
                    class_activations[label] = []
                class_activations[label].append(batch_activations[i])
    
    # 计算每个特征的激活频率
    activation_frequency = activation_counts / total_samples
    
    # 获取最活跃的特征
    top_feature_indices = np.argsort(activation_frequency)[-20:][::-1]
    top_frequencies = activation_frequency[top_feature_indices]
    
    # 绘制特征激活频率条形图
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(top_feature_indices)), top_frequencies, color='skyblue')
    plt.xlabel('特征索引')
    plt.ylabel('激活频率')
    plt.title('Mice数据集上最活跃的SAE特征')
    plt.xticks(range(len(top_feature_indices)), top_feature_indices)
    plt.tight_layout()
    
    # 保存激活频率图
    activation_freq_path = os.path.join(save_dir, 'mice_activation_frequency.png')
    plt.savefig(activation_freq_path)
    plt.close()
    
    # 计算每个类别的平均激活
    class_avg_activations = {}
    for label, acts in class_activations.items():
        class_avg_activations[label] = np.mean(np.array(acts), axis=0)
    
    # 绘制类别间的特征激活对比
    plt.figure(figsize=(14, 7))
    
    # 只显示前100个特征以保持图的可读性
    feature_indices = range(100)
    
    for label, avg_act in class_avg_activations.items():
        plt.plot(feature_indices, avg_act[:100], 
                 label=f'类别 {label}', 
                 linewidth=2)
    
    plt.xlabel('特征索引')
    plt.ylabel('平均激活强度')
    plt.title('不同类别的SAE特征激活模式')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # 保存类别激活对比图
    class_act_path = os.path.join(save_dir, 'mice_class_activations.png')
    plt.savefig(class_act_path)
    plt.close()
    
    print(f"激活分析可视化已保存到 {save_dir}")
    return activation_freq_path, class_act_path

def evaluate_sae_on_mice_dataset(model_path, data_dir='data/Mice', batch_size=16, save_dir='visualizations/mice'):
    """
    在Mice数据集上评估预训练的SAE模型
    
    参数:
        model_path: SAE模型路径
        data_dir: Mice数据集目录
        batch_size: 批量大小
        save_dir: 保存可视化结果的目录
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 设置设备 - 强制使用CPU避免CUDA错误
    device = torch.device("cpu")
    print(f"使用设备: {device} (强制使用CPU避免CUDA错误)")
    
    # 配置模型参数
    config = {
        "input_size": 784,    # 28x28
        "hidden_size": 1024,  # 隐藏层大小
        "k_sparse": 50,       # 稀疏参数k
        "num_saes": 5         # SAE数量
    }
    
    # 加载预训练的SAE模型
    model = load_sae_model(model_path, config, device)
    model.eval()
    
    # 准备Mice数据集
    target_size = (28, 28)  # 与MNIST保持一致
    mice_dataset = MiceDataset(
        data_dir=data_dir, 
        target_size=target_size,
        augment=True,
        augment_factor=5
    )
    
    # 创建数据加载器
    mice_dataloader = DataLoader(
        mice_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )
    
    # 1. 可视化SAE特征
    feature_viz_path = visualize_sae_features(model, device, save_dir)
    
    # 2. 重构示例图像
    reconstruction_results = []
    sae_id = 0  # 使用第一个SAE
    
    with torch.no_grad():
        for batch_idx, ((X_batch,), labels) in enumerate(mice_dataloader):
            if batch_idx >= 2:  # 只处理少量批次用于示例
                break
                
            X_batch = X_batch.to(device)
            
            # 前向传播获取重构和激活
            outputs, activations, indices = model.forward_with_encoded(X_batch)
            
            # 获取第一个SAE的重建、激活和索引
            reconstructed = outputs[sae_id].cpu().numpy()
            original = X_batch.cpu().numpy()
            batch_indices = indices[sae_id].cpu().numpy()
            
            # 保存结果
            for i in range(len(X_batch)):
                # 计算相关系数
                corr, _ = pearsonr(original[i], reconstructed[i])
                
                # 创建可视化
                fig = plot_mice_reconstruction(
                    original[i],
                    reconstructed[i],
                    batch_indices[i],
                    config['k_sparse'],
                    corr,
                    labels[i].item(),
                    target_size
                )
                
                # 保存图像
                save_path = os.path.join(save_dir, f'mice_reconstruction_{batch_idx}_{i}_class_{labels[i].item()}.png')
                plt.savefig(save_path)
                plt.close(fig)
                
                # 添加到结果列表
                reconstruction_results.append({
                    'path': save_path,
                    'correlation': corr,
                    'label': labels[i].item()
                })
                
                print(f"重建示例已保存到 {save_path}, 相关系数: {corr:.4f}")
    
    # 3. 分析稀疏激活模式
    activation_freq_path, class_act_path = analyze_sparse_activations(
        model, mice_dataloader, device, config['k_sparse'], save_dir
    )
    
    # 计算所有图像的平均重建质量
    all_correlations = []
    all_class_correlations = {1: [], 2: []}
    
    with torch.no_grad():
        for batch_idx, ((X_batch,), labels) in enumerate(mice_dataloader):
            X_batch = X_batch.to(device)
            
            # 前向传播获取重构
            outputs, _, _ = model.forward_with_encoded(X_batch)
            
            # 计算相关系数
            reconstructed = outputs[sae_id].cpu().numpy()
            original = X_batch.cpu().numpy()
            
            for i in range(len(X_batch)):
                corr, _ = pearsonr(original[i], reconstructed[i])
                all_correlations.append(corr)
                
                # 按类别记录
                label = int(labels[i].item())
                all_class_correlations[label].append(corr)
    
    # 计算平均值
    mean_corr = np.mean(all_correlations)
    class_mean_corrs = {label: np.mean(corrs) for label, corrs in all_class_correlations.items()}
    
    # 打印结果
    print(f"\n重构完成。整体平均相关系数: {mean_corr:.4f}")
    for label, corr in class_mean_corrs.items():
        print(f"类别 {label} 平均相关系数: {corr:.4f}")
    
    # 返回结果
    return {
        'feature_visualization': feature_viz_path,
        'reconstruction_examples': reconstruction_results,
        'activation_analysis': {
            'frequency': activation_freq_path,
            'class_comparison': class_act_path
        },
        'reconstruction_performance': {
            'overall_mean': mean_corr,
            'class_means': class_mean_corrs
        }
    }

def main():
    # 设置随机种子确保结果可复现
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # MNIST SAE模型路径
    model_path = "mnist_sae_models/mnist_sae_epoch_4.pth"
    
    # Mice数据集目录
    data_dir = "data/Mice"
    
    # 可视化保存目录
    save_dir = "visualizations/mice"
    
    # 评估SAE模型在Mice数据集上的性能
    results = evaluate_sae_on_mice_dataset(
        model_path=model_path,
        data_dir=data_dir,
        batch_size=8,
        save_dir=save_dir
    )
    
    print("\n评估完成，结果保存在:", save_dir)

if __name__ == "__main__":
    main()
