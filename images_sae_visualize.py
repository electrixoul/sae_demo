import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
from images_sae import ImageSparseAutoencoder
from images_dataset import JPGImageDataset
from scipy.stats import pearsonr

def visualize_features(model, dataset, device, output_dir, img_size=64, max_features=100, num_samples=500):
    """
    可视化SAE学习到的特征，并按激活频率排序
    
    参数:
        model: 训练好的SAE模型
        dataset: 数据集，用于计算特征激活频率
        device: 计算设备
        output_dir: 输出目录
        img_size: 图像大小
        max_features: 要可视化的最大特征数量
        num_samples: 用于计算激活频率的样本数量
    """
    os.makedirs(output_dir, exist_ok=True)
    print("计算特征激活频率...")
    
    # 1. 首先计算每个特征的激活频率
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    activation_counts = [np.zeros(encoder.weight.shape[0]) for encoder in model.encoders]
    
    # 处理样本
    with torch.no_grad():
        for idx, sample_idx in enumerate(indices):
            # 获取样本并添加批次维度
            (data,), _ = dataset[sample_idx]
            X = data.unsqueeze(0).to(device)
            
            # 前向传播获取激活
            _, activations, indices_list = model.forward_with_encoded(X)
            
            # 更新激活计数
            for sae_id, (activation, indices) in enumerate(zip(activations, indices_list)):
                for idx in indices[0].cpu().numpy():
                    activation_counts[sae_id][idx] += 1
            
            if idx % 100 == 0:
                print(f"已处理 {idx+1}/{len(indices)} 个样本来计算激活频率")
    
    # 2. 对每个SAE编码器可视化按激活频率排序的特征
    for sae_id, encoder in enumerate(model.encoders):
        # 获取权重
        weights = encoder.weight.data.cpu().numpy()
        
        # 计算通道数
        channels = weights.shape[1] // (img_size * img_size)
        
        # 按激活频率对特征索引进行排序（降序）
        feature_indices = np.argsort(activation_counts[sae_id])[::-1]
        
        # 选择激活频率最高的特征
        top_feature_indices = feature_indices[:max_features]
        num_features = len(top_feature_indices)
        
        # 计算网格尺寸
        grid_size = int(np.ceil(np.sqrt(num_features)))
        
        # 创建画布
        plt.figure(figsize=(20, 20))
        
        # 配置字体以支持中文 - 使用macOS上可用的中文字体
        plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'STHeiti', 'Apple LiGothic', 'SimHei', 'Arial Unicode MS', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False    # 正确显示负号
        plt.rcParams['figure.figsize'] = (20, 20)
        plt.rcParams['figure.autolayout'] = True      # 自动调整布局
        plt.rcParams['figure.titlesize'] = 18
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['axes.titlepad'] = 30            # 增加标题和图像之间的距离
        plt.rcParams['figure.subplot.top'] = 0.85     # 增加顶部边距
        plt.rcParams['figure.subplot.bottom'] = 0.15  # 增加底部边距
        
        # 可视化每个特征
        for i, feature_idx in enumerate(top_feature_indices):
            feature = weights[feature_idx]
            feature = feature.reshape(channels, img_size, img_size)
            
            # 计算激活率（占所有样本的百分比）
            activation_rate = activation_counts[sae_id][feature_idx] / num_samples * 100
            
            # 对于RGB图像，使用平均值进行可视化
            if channels == 3:
                # 计算三个通道的均值
                feature_mean = np.mean(feature, axis=0)
                
                # 归一化用于可视化
                feature_visual = feature_mean
                # 拉伸对比度以提高可见性
                feature_visual = (feature_visual - feature_visual.min()) / (feature_visual.max() - feature_visual.min() + 1e-8)
                
                # 添加到子图
                plt.subplot(grid_size, grid_size, i + 1)
                plt.imshow(feature_visual, cmap='viridis')
                plt.axis('off')
                plt.title(f'特征 {feature_idx}\n激活率: {activation_rate:.1f}%')
            else:
                # 对于灰度图像
                feature_visual = feature.reshape(img_size, img_size)
                feature_visual = (feature_visual - feature_visual.min()) / (feature_visual.max() - feature_visual.min() + 1e-8)
                
                plt.subplot(grid_size, grid_size, i + 1)
                plt.imshow(feature_visual, cmap='gray')
                plt.axis('off')
                plt.title(f'特征 {feature_idx}\n激活率: {activation_rate:.1f}%')
        
        plt.suptitle(f'SAE {sae_id} 特征可视化 (按激活频率排序)', fontsize=20)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'sae{sae_id}_features.png'))
        plt.close()

def visualize_reconstructions(model, dataset, device, output_dir, num_examples=10, img_size=64):
    """
    可视化原始图像和重构图像
    
    参数:
        model: 训练好的SAE模型
        dataset: 图像数据集
        device: 计算设备
        output_dir: 输出目录
        num_examples: 要可视化的样本数量
        img_size: 图像大小
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 选择一些图像进行可视化
    indices = np.linspace(0, len(dataset) - 1, num=num_examples, dtype=int)
    
    # 对每个样本
    for idx, sample_idx in enumerate(indices):
        # 获取样本
        (data,), _ = dataset[sample_idx]
        
        # 将数据转换为张量并添加批次维度
        X = data.unsqueeze(0).to(device)
        
        # 前向传播
        with torch.no_grad():
            outputs, activations, indices = model.forward_with_encoded(X)
        
        # 为每个SAE创建可视化
        for sae_id, (output, activation, index) in enumerate(zip(outputs, activations, indices)):
            # 转换为numpy数组
            original = X.cpu().numpy()[0]
            reconstructed = output.cpu().numpy()[0]
            
            # 计算相关系数
            corr, _ = pearsonr(original, reconstructed)
            
            # 创建可视化
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # 配置字体以支持中文 - 使用macOS上可用的中文字体
            plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'STHeiti', 'Apple LiGothic', 'SimHei', 'Arial Unicode MS', 'sans-serif']
            plt.rcParams['axes.unicode_minus'] = False    # 正确显示负号
            plt.rcParams['figure.autolayout'] = True      # 自动调整布局
            plt.rcParams['axes.titlesize'] = 14
            plt.rcParams['axes.titlepad'] = 20            # 增加标题和图像之间的距离
            
            # 计算通道数
            channels = original.shape[0] // (img_size * img_size)
            
            # 原始图像
            if channels == 3:
                # 重塑为(channels, height, width)然后转换为(height, width, channels)
                orig_display = original.reshape(channels, img_size, img_size).transpose(1, 2, 0)
                recon_display = reconstructed.reshape(channels, img_size, img_size).transpose(1, 2, 0)
                
                # 裁剪到[0,1]范围以避免显示问题
                orig_display = np.clip(orig_display, 0, 1)
                recon_display = np.clip(recon_display, 0, 1)
                
                axes[0].imshow(orig_display)
                axes[1].imshow(recon_display)
            else:
                # 灰度图像
                axes[0].imshow(original.reshape(img_size, img_size), cmap='gray')
                axes[1].imshow(reconstructed.reshape(img_size, img_size), cmap='gray')
            
            axes[0].set_title('原始图像')
            axes[0].axis('off')
            
            axes[1].set_title(f'重构 (相关系数: {corr:.4f})')
            axes[1].axis('off')
            
            # 激活可视化
            if channels == 3:
                # 创建激活热图
                activation_map = np.zeros(original.shape)
                activation_map[index.cpu().numpy()] = 1
                act_display = activation_map.reshape(channels, img_size, img_size).mean(axis=0)
                
                im = axes[2].imshow(act_display, cmap='hot')
                axes[2].set_title(f'稀疏激活 (k={model.k_sparse_values[sae_id]})')
                axes[2].axis('off')
                plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
            else:
                # 灰度图像
                activation_map = np.zeros(original.shape)
                activation_map[index.cpu().numpy()] = 1
                axes[2].imshow(activation_map.reshape(img_size, img_size), cmap='hot')
                axes[2].set_title(f'稀疏激活 (k={model.k_sparse_values[sae_id]})')
                axes[2].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'sample{idx}_sae{sae_id}_reconstruction.png'))
            plt.close(fig)

def visualize_feature_usage(model, dataset, device, output_dir, num_samples=200, img_size=64):
    """
    可视化特征使用情况
    
    参数:
        model: 训练好的SAE模型
        dataset: 图像数据集
        device: 计算设备
        output_dir: 输出目录
        num_samples: 要分析的样本数量
        img_size: 图像大小
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 随机选择样本
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    # 为每个SAE收集激活计数
    activation_counts = [np.zeros(encoder.weight.shape[0]) for encoder in model.encoders]
    activation_values = [[] for _ in model.encoders]
    
    # 处理样本
    for sample_idx in indices:
        # 获取样本
        (data,), _ = dataset[sample_idx]
        
        # 将数据转换为张量并添加批次维度
        X = data.unsqueeze(0).to(device)
        
        # 前向传播
        with torch.no_grad():
            _, activations, indices_list = model.forward_with_encoded(X)
        
        # 更新激活计数
        for sae_id, (activation, indices) in enumerate(zip(activations, indices_list)):
            # 更新激活频率
            for idx in indices[0].cpu().numpy():
                activation_counts[sae_id][idx] += 1
            
            # 收集激活值
            non_zero_values = activation[0, indices[0]].cpu().numpy()
            activation_values[sae_id].extend(non_zero_values)
    
    # 为每个SAE可视化特征使用情况
    for sae_id, (counts, values) in enumerate(zip(activation_counts, activation_values)):
        # 归一化计数
        normalized_counts = counts / num_samples
        
        # 按频率降序排列特征
        sorted_indices = np.argsort(normalized_counts)[::-1]
        sorted_counts = normalized_counts[sorted_indices]
        
        # 前25个最活跃的特征
        top_n = 25
        top_indices = sorted_indices[:top_n]
        top_counts = sorted_counts[:top_n]
        
        # 创建可视化
        plt.figure(figsize=(12, 6))
        
        # 配置字体以支持中文 - 使用macOS上可用的中文字体
        plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'STHeiti', 'Apple LiGothic', 'SimHei', 'Arial Unicode MS', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False    # 正确显示负号
        plt.rcParams['figure.autolayout'] = True      # 自动调整布局
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.titlepad'] = 20            # 增加标题和图像之间的距离
        plt.rcParams['figure.subplot.top'] = 0.85     # 增加顶部边距
        plt.rcParams['figure.subplot.bottom'] = 0.15  # 增加底部边距
        plt.bar(range(top_n), top_counts)
        plt.xlabel('特征索引 (排序后)')
        plt.ylabel('激活频率')
        plt.title(f'SAE {sae_id} Top {top_n} 最活跃特征')
        plt.xticks(range(top_n), [f'{idx}' for idx in top_indices], rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'sae{sae_id}_feature_usage.png'))
        plt.close()
        
        # 可视化激活值分布
        plt.figure(figsize=(12, 6))
        
        # 配置字体以支持中文 - 使用macOS上可用的中文字体
        plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'STHeiti', 'Apple LiGothic', 'SimHei', 'Arial Unicode MS', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False    # 正确显示负号
        plt.rcParams['figure.autolayout'] = True      # 自动调整布局
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.titlepad'] = 20            # 增加标题和图像之间的距离
        plt.rcParams['figure.subplot.top'] = 0.85     # 增加顶部边距
        plt.rcParams['figure.subplot.bottom'] = 0.15  # 增加底部边距
        
        plt.hist(values, bins=50)
        plt.xlabel('激活值')
        plt.ylabel('频率')
        plt.title(f'SAE {sae_id} 激活值分布')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'sae{sae_id}_activation_values.png'))
        plt.close()

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="可视化图像SAE模型")
    parser.add_argument('--model_path', type=str, default='images_sae_models/images_sae_final.pth',
                      help='模型路径')
    parser.add_argument('--output_dir', type=str, default='images_sae_visualizations',
                      help='输出目录')
    parser.add_argument('--device', type=str, choices=['cpu', 'mps', 'cuda'], default='cpu',
                      help='计算设备 (CPU更适合生成可视化)')
    parser.add_argument('--img_size', type=int, default=64,
                      help='图像大小')
    args = parser.parse_args()
    
    # 检查模型是否存在
    if not os.path.exists(args.model_path):
        print(f"错误: 模型 {args.model_path} 不存在。请先训练模型。")
        return
    
    # 设置设备
    if args.device == 'mps' and torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"使用设备: {device} (Apple Silicon GPU)")
    elif args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"使用设备: {device}")
    else:
        device = torch.device("cpu")
        print(f"使用设备: {device}")
    
    # 确定图像尺寸、通道数和输入大小
    img_size = args.img_size
    channels = 3  # RGB图像
    input_size = img_size * img_size * channels
    
    # 配置与训练相同
    config = {
        "input_size": input_size,
        "hidden_size": 3072, # 更新为与训练模型相匹配的大小
        "k_sparse": 128,
        "num_saes": 3,
    }
    
    # 加载模型
    model = ImageSparseAutoencoder(config)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    model.eval()
    print(f"模型加载自: {args.model_path}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"可视化将保存到: {args.output_dir}")
    
    # 加载一小部分数据用于可视化
    print("加载数据集...")
    dataset = JPGImageDataset("output_images_jpg_rename", resize_dim=img_size)
    print(f"数据集大小: {len(dataset)} 个样本")
    
    # 可视化特征
    print("可视化SAE特征...")
    visualize_features(model, dataset, device, args.output_dir, img_size=img_size, max_features=100)
    
    # 可视化重构
    print("可视化图像重构...")
    visualize_reconstructions(model, dataset, device, args.output_dir, num_examples=10, img_size=img_size)
    
    # 可视化特征使用情况
    print("分析特征使用情况...")
    visualize_feature_usage(model, dataset, device, args.output_dir, num_samples=200, img_size=img_size)
    
    print("可视化完成！")

if __name__ == "__main__":
    main()
