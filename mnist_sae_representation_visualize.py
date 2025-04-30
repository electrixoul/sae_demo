import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import seaborn as sns
from models.sae import SparseAutoencoder
import pandas as pd

# 和训练脚本中相同的数据集处理类
class MNISTDataset(Dataset):
    def __init__(self, is_train=True):
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        self.mnist = datasets.MNIST(
            root='./data',
            train=is_train,
            download=True,
            transform=transform
        )
        
    def __len__(self):
        return len(self.mnist)
    
    def __getitem__(self, idx):
        image, label = self.mnist[idx]
        flattened_image = image.view(-1)
        return (flattened_image,), label

def load_model(model_path):
    """加载模型"""
    config = {
        "input_size": 784,
        "hidden_size": 1024,
        "k_sparse": 50,
        "num_saes": 5,
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SparseAutoencoder(config).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    return model, device

def visualize_representations(model, dataloader, device, num_images=5, save_path="representations.png"):
    """可视化MNIST图像的表征"""
    model.eval()
    
    # 获取样本图像及其表征
    images = []
    representations = []
    reconstructions = []
    labels = []
    
    with torch.no_grad():
        for (x_batch,), y_batch in dataloader:
            if len(images) >= num_images:
                break
                
            # 获取模型输出和表征
            x_batch = x_batch.to(device)
            outputs, activations, indices = model.forward_with_encoded(x_batch)  # 使用第一个SAE
            
            for i in range(len(x_batch)):
                if len(images) >= num_images:
                    break
                    
                # 原始图像
                images.append(x_batch[i].cpu().numpy())
                
                # 稀疏表征 (只获取第一个SAE的)
                representations.append(activations[0][i].cpu().numpy())
                
                # 重建图像
                reconstructions.append(outputs[0][i].cpu().numpy())
                
                # 标签
                labels.append(y_batch[i].item())
    
    # 创建可视化图
    fig = plt.figure(figsize=(15, 3 * num_images))
    
    for i in range(num_images):
        # 原始图像
        ax1 = plt.subplot(num_images, 4, i*4 + 1)
        ax1.imshow(images[i].reshape(28, 28), cmap='gray')
        ax1.set_title(f'原始 (标签: {labels[i]})')
        ax1.axis('off')
        
        # 表征的稀疏性可视化 (1D表示)
        ax2 = plt.subplot(num_images, 4, i*4 + 2)
        sparse_repr = representations[i]
        sns.barplot(x=np.arange(len(sparse_repr))[sparse_repr > 0], 
                   y=sparse_repr[sparse_repr > 0], 
                   ax=ax2,
                   color='skyblue')
        ax2.set_title(f'稀疏表征 (非零: {np.sum(sparse_repr > 0)}/{len(sparse_repr)})')
        ax2.set_xlabel('特征索引')
        ax2.set_ylabel('激活值')
        
        # 表征的2D可视化 (热力图)
        ax3 = plt.subplot(num_images, 4, i*4 + 3)
        repr_2d = np.zeros((32, 32))  # 将1024维向量重塑为2D
        repr_2d.flat[:len(representations[i])] = representations[i]
        sns.heatmap(repr_2d, cmap='viridis', ax=ax3, cbar=False)
        ax3.set_title('表征 (2D热力图)')
        ax3.axis('off')
        
        # 重建图像
        ax4 = plt.subplot(num_images, 4, i*4 + 4)
        ax4.imshow(reconstructions[i].reshape(28, 28), cmap='gray')
        ax4.set_title('重建图像')
        ax4.axis('off')
    
    plt.suptitle('MNIST图像表征可视化', fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=150)
    print(f"表征可视化已保存到 {save_path}")
    
    return fig

def visualize_activations_per_feature(model, dataloader, device, num_features=25, num_samples=1000, save_path="feature_usage.png"):
    """可视化不同特征的平均激活情况"""
    model.eval()
    
    # 收集多个样本的激活
    all_activations = []
    
    with torch.no_grad():
        count = 0
        for (x_batch,), _ in dataloader:
            if count >= num_samples:
                break
                
            x_batch = x_batch.to(device)
            _, activations, _ = model.forward_with_encoded(x_batch)
            
            # 收集第一个SAE的激活
            all_activations.append(activations[0].cpu().numpy())
            count += x_batch.size(0)
            
            if count >= num_samples:
                break
    
    # 将所有激活拼接起来
    all_activations = np.vstack(all_activations)[:num_samples]
    
    # 计算每个特征的激活情况
    feature_usage = np.mean(all_activations > 0, axis=0)  # 每个特征激活的频率
    feature_avg_strength = np.zeros_like(feature_usage)
    mask = feature_usage > 0
    feature_avg_strength[mask] = np.mean(all_activations[:, mask], axis=0)  # 每个特征的平均激活强度
    
    # 获取最活跃的特征
    top_features_idx = np.argsort(feature_usage)[-num_features:][::-1]
    
    # 可视化特征激活情况
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # 激活频率
    sns.barplot(x=np.arange(num_features), y=feature_usage[top_features_idx], ax=ax1, color='skyblue')
    ax1.set_title('特征激活频率 (每个特征被激活的概率)')
    ax1.set_xlabel('特征索引')
    ax1.set_ylabel('激活频率')
    ax1.set_xticklabels(top_features_idx)
    
    # 平均激活强度
    sns.barplot(x=np.arange(num_features), y=feature_avg_strength[top_features_idx], ax=ax2, color='salmon')
    ax2.set_title('特征平均激活强度')
    ax2.set_xlabel('特征索引')
    ax2.set_ylabel('平均激活值')
    ax2.set_xticklabels(top_features_idx)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"特征使用统计已保存到 {save_path}")
    
    return fig

def visualize_digit_representations(model, dataloader, device, save_path="digit_representations.png"):
    """可视化不同数字的表征差异"""
    model.eval()
    
    # 为每个数字收集表征
    digit_representations = {i: [] for i in range(10)}
    
    with torch.no_grad():
        for (x_batch,), y_batch in dataloader:
            x_batch = x_batch.to(device)
            _, activations, _ = model.forward_with_encoded(x_batch)
            
            # 获取第一个SAE的激活
            batch_activations = activations[0].cpu().numpy()
            batch_labels = y_batch.numpy()
            
            # 按数字分类
            for i, label in enumerate(batch_labels):
                if len(digit_representations[label]) < 20:  # 每个数字最多使用20个样本
                    digit_representations[label].append(batch_activations[i])
    
    # 计算每个数字的平均表征
    digit_avg_representations = {}
    for digit, reprs in digit_representations.items():
        if reprs:
            digit_avg_representations[digit] = np.mean(np.array(reprs), axis=0)
    
    # 创建可视化
    plt.figure(figsize=(15, 8))
    
    # 绘制前100个特征的平均激活
    feature_view_count = 100
    for digit, avg_repr in digit_avg_representations.items():
        plt.plot(range(feature_view_count), avg_repr[:feature_view_count], label=f'数字 {digit}')
    
    plt.title('不同数字的平均特征激活')
    plt.xlabel('特征索引')
    plt.ylabel('平均激活值')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.savefig(save_path, dpi=150)
    print(f"数字表征差异可视化已保存到 {save_path}")
    
    # 计算特征重要性
    importance_per_digit = {}
    for digit in range(10):
        if digit in digit_avg_representations:
            # 找到对该数字激活最强的前10个特征
            top_features = np.argsort(digit_avg_representations[digit])[-10:][::-1]
            importance_per_digit[digit] = top_features
    
    print("\n不同数字的特征重要性:")
    for digit, features in importance_per_digit.items():
        print(f"数字 {digit} 的关键特征索引: {features}")
    
    return plt.gcf()

def main():
    # 设置参数解析
    parser = argparse.ArgumentParser(description="可视化SAE表征")
    parser.add_argument('--model', type=str, default="mnist_sae_models/mnist_sae_epoch_4.pth",
                      help='模型路径')
    parser.add_argument('--num_images', type=int, default=5,
                      help='要可视化的图像数量')
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs("visualizations", exist_ok=True)
    
    # 加载模型
    model, device = load_model(args.model)
    print(f"已加载模型: {args.model}")
    
    # 创建数据加载器
    test_dataset = MNISTDataset(is_train=False)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.num_images, 
        shuffle=True,
        num_workers=2 if device.type == 'cuda' else 0,
        pin_memory=device.type == 'cuda',
        generator=torch.Generator().manual_seed(42)
    )
    
    # 可视化表征
    visualize_representations(
        model,
        test_loader,
        device,
        num_images=args.num_images,
        save_path="visualizations/image_representations.png"
    )
    
    # 创建新的数据加载器用于收集更多样本
    batch_loader = DataLoader(
        test_dataset, 
        batch_size=64, 
        shuffle=True,
        num_workers=2 if device.type == 'cuda' else 0,
        pin_memory=device.type == 'cuda',
        generator=torch.Generator().manual_seed(43)
    )
    
    # 可视化特征激活统计
    visualize_activations_per_feature(
        model,
        batch_loader,
        device,
        num_features=25,
        num_samples=1000,
        save_path="visualizations/feature_usage_statistics.png"
    )
    
    # 可视化不同数字的表征差异
    visualize_digit_representations(
        model,
        batch_loader,
        device,
        save_path="visualizations/digit_representations.png"
    )

if __name__ == "__main__":
    main()
