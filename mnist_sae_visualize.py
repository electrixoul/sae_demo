import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from scipy.stats import pearsonr
import seaborn as sns
from models.sae import SparseAutoencoder
from PIL import Image
import pandas as pd
import glob

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

def load_model_and_config(model_path):
    """加载模型和配置"""
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
    
    return model, config, device

def visualize_reconstructions(model, dataloader, device, num_images=10, save_path="reconstructions.png"):
    """可视化原始图像和重建图像的对比"""
    model.eval()
    
    # 获取一批数据用于可视化
    all_images = []
    all_labels = []
    all_reconstructions = []
    all_correlations = []
    
    with torch.no_grad():
        for (X_batch,), labels in dataloader:
            if len(all_images) >= num_images:
                break
                
            X_batch = X_batch.to(device)
            outputs, activations, indices = model.forward_with_encoded(X_batch)
            
            # 对于每个SAE，获取重建
            for i, output in enumerate(outputs):
                reconstructed = output.cpu().numpy()
                original = X_batch.cpu().numpy()
                
                for j in range(len(X_batch)):
                    if len(all_images) >= num_images:
                        break
                        
                    # 计算相关系数
                    corr, _ = pearsonr(original[j], reconstructed[j])
                    
                    all_images.append(original[j])
                    all_reconstructions.append(reconstructed[j])
                    all_labels.append(labels[j].item())
                    all_correlations.append(corr)
                
                # 只使用第一个SAE的重建
                break
    
    # 创建可视化
    fig, axes = plt.subplots(num_images, 2, figsize=(10, 2 * num_images + 1))
    fig.suptitle('原始MNIST图像与SAE重建对比', fontsize=16)
    
    for i in range(num_images):
        # 原始图像
        axes[i, 0].imshow(all_images[i].reshape(28, 28), cmap='gray')
        axes[i, 0].set_title(f'原始 (标签: {all_labels[i]})')
        axes[i, 0].axis('off')
        
        # 重建图像
        axes[i, 1].imshow(all_reconstructions[i].reshape(28, 28), cmap='gray')
        axes[i, 1].set_title(f'重建 (相关性: {all_correlations[i]:.4f})')
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"重建图像已保存到 {save_path}")
    
    # 返回平均相关系数
    avg_corr = np.mean(all_correlations)
    print(f"平均相关系数: {avg_corr:.4f}")
    return avg_corr

def extract_training_metrics():
    """从训练日志中提取指标"""
    # 这个函数假设训练日志的内容在终端输出中可用
    # 因为我们没有直接的access to wandb数据
    # 所以我们将通过分析模型保存时间和再次运行评估来近似这些指标
    
    # 获取模型文件
    model_files = sorted(glob.glob("mnist_sae_models/mnist_sae_epoch_*.pth"))
    epochs = [int(f.split('_')[-1].split('.')[0]) for f in model_files]
    
    if not model_files:
        print("未找到模型文件")
        return None
        
    # 运行评估，获取每个epoch模型的重建相关系数
    reconstruction_scores = []
    
    for model_file in model_files:
        model, config, device = load_model_and_config(model_file)
        
        # 创建测试数据加载器
        test_dataset = MNISTDataset(is_train=False)
        test_loader = DataLoader(
            test_dataset, 
            batch_size=64, 
            shuffle=True,
            num_workers=2 if device.type == 'cuda' else 0,
            pin_memory=device.type == 'cuda',
            generator=torch.Generator().manual_seed(42)
        )
        
        # 运行简单版本的评估
        avg_corr = evaluate_model(model, test_loader, device, num_batches=10)
        reconstruction_scores.append(avg_corr)
        
    # 创建一个数据帧来存储指标
    metrics_df = pd.DataFrame({
        'epoch': epochs,
        'correlation': reconstruction_scores,
    })
    
    return metrics_df

def evaluate_model(model, dataloader, device, num_batches=10):
    """评估模型的重建相关系数"""
    model.eval()
    correlations = []
    
    with torch.no_grad():
        for batch_idx, ((X_batch,), _) in enumerate(dataloader):
            if batch_idx >= num_batches:
                break
                
            X_batch = X_batch.to(device)
            outputs, _, _ = model.forward_with_encoded(X_batch)
            
            # 使用第一个SAE的输出
            reconstructed = outputs[0].cpu().numpy()
            original = X_batch.cpu().numpy()
            
            # 计算每个样本的相关系数
            for i in range(len(X_batch)):
                corr, _ = pearsonr(original[i], reconstructed[i])
                correlations.append(corr)
    
    return np.mean(correlations)

def plot_training_metrics(metrics_df, save_path="training_metrics.png"):
    """绘制训练指标"""
    plt.figure(figsize=(10, 5))
    
    plt.plot(metrics_df['epoch'], metrics_df['correlation'], marker='o', linestyle='-', label='重建相关系数')
    
    plt.xlabel('Epoch')
    plt.ylabel('相关系数')
    plt.title('SAE训练过程中的重建质量变化')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"训练指标可视化已保存到 {save_path}")

def create_feature_visualization(model, device, save_path="sae_features.png"):
    """可视化学习到的特征（权重）"""
    model.eval()
    
    # 获取第一个SAE的编码器权重
    with torch.no_grad():
        weights = model.encoders[0].weight.data.cpu().numpy()
    
    # 权重形状应该是 (1024, 784)，每行是一个特征
    num_features = min(100, weights.shape[0])  # 只显示前100个特征
    
    # 创建一个网格来显示特征
    grid_size = int(np.ceil(np.sqrt(num_features)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    
    # 归一化权重以便可视化
    weights_min = weights.min()
    weights_max = weights.max()
    weights_normalized = (weights - weights_min) / (weights_max - weights_min)
    
    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j
            if idx < num_features:
                feature = weights_normalized[idx].reshape(28, 28)
                axes[i, j].imshow(feature, cmap='viridis')
                axes[i, j].axis('off')
            else:
                axes[i, j].axis('off')
    
    plt.suptitle('SAE学习到的特征（前100个）', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"特征可视化已保存到 {save_path}")

def main():
    # 设置参数解析
    parser = argparse.ArgumentParser(description="可视化SAE重建和训练指标")
    parser.add_argument('--model', type=str, default="mnist_sae_models/mnist_sae_epoch_4.pth",
                      help='模型路径')
    parser.add_argument('--num_images', type=int, default=10,
                      help='要可视化的图像数量')
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs("visualizations", exist_ok=True)
    
    # 加载模型
    model, config, device = load_model_and_config(args.model)
    print(f"已加载模型: {args.model}")
    
    # 创建测试数据加载器
    test_dataset = MNISTDataset(is_train=False)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.num_images, 
        shuffle=True,
        num_workers=2 if device.type == 'cuda' else 0,
        pin_memory=device.type == 'cuda',
        generator=torch.Generator().manual_seed(42)
    )
    
    # 可视化重建
    visualize_reconstructions(
        model, 
        test_loader, 
        device, 
        num_images=args.num_images,
        save_path="visualizations/reconstructions.png"
    )
    
    # 可视化特征
    create_feature_visualization(
        model,
        device,
        save_path="visualizations/sae_features.png"
    )
    
    # 提取训练指标
    metrics_df = extract_training_metrics()
    
    if metrics_df is not None:
        # 绘制训练指标
        plot_training_metrics(
            metrics_df,
            save_path="visualizations/training_metrics.png"
        )
        
        # 打印最终结果
        print("\n训练结果:")
        print(metrics_df)
    
    print("\n可视化完成！所有图像都保存在 visualizations/ 目录中")

if __name__ == "__main__":
    main()
