import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from scipy.stats import pearsonr
from models.sae import SparseAutoencoder
from config import get_device
import time

# Import wandb setup function from the template
from wandb_tsinghua_template import setup_wandb

class MNISTDataset(Dataset):
    def __init__(self, is_train=True):
        """
        加载MNIST数据集，并将图像展平成一维向量
        
        参数:
            is_train (bool): 是否加载训练集数据
        """
        # 定义MNIST数据集的标准转换
        transform = transforms.Compose([
            transforms.ToTensor(),  # 转换为张量并归一化到[0,1]
        ])
        
        # 加载MNIST数据集
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
        # 展平图像为一维向量 (1, 28, 28) -> (784,)
        flattened_image = image.view(-1)
        return (flattened_image,), label

def plot_mnist_reconstruction(original, reconstructed, indices=None, k_sparse=None, corr=None):
    """
    绘制MNIST图像原始图和重构图的对比
    
    参数:
        original: 原始图像张量
        reconstructed: 重构图像张量
        indices: 稀疏编码中激活的索引
        k_sparse: k稀疏值
        corr: Pearson相关系数
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # 原始图像
    axes[0].imshow(original.reshape(28, 28), cmap='gray')
    axes[0].set_title('原始图像')
    axes[0].axis('off')
    
    # 重构图像
    axes[1].imshow(reconstructed.reshape(28, 28), cmap='gray')
    axes[1].set_title(f'重构图像 (相关系数: {corr:.4f})')
    axes[1].axis('off')
    
    # 稀疏激活可视化
    if indices is not None and k_sparse is not None:
        # 创建一个表示所有可能神经元的数组
        activation_map = np.zeros(reconstructed.shape)
        # 将激活的神经元设为1
        activation_map[indices] = 1
        
        # 展示激活图
        axes[2].imshow(activation_map.reshape(28, 28), cmap='hot')
        axes[2].set_title(f'稀疏激活 (k={k_sparse})')
        axes[2].axis('off')
    
    plt.tight_layout()
    return fig

def load_sae_model(model_path, input_size, hidden_size, k_sparse, num_saes, device):
    """
    加载预训练的SAE模型
    
    参数:
        model_path: 模型文件路径
        input_size: 输入维度大小
        hidden_size: 隐藏层大小
        k_sparse: 稀疏参数k
        num_saes: SAE数量
        device: 计算设备
    
    返回:
        加载后的SAE模型
    """
    # 定义模型配置
    config = {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "k_sparse": k_sparse,
        "num_saes": num_saes
    }
    
    # 创建模型
    model = SparseAutoencoder(config)
    
    # 加载预训练权重
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"模型成功从 {model_path} 加载")
    else:
        print(f"未找到模型文件 {model_path}，使用随机初始化的模型")
    
    # 将模型移动到指定设备
    model = model.to(device)
    
    return model

def main():
    # 设置随机种子确保结果可复现
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 强制使用CPU，避免CUDA错误
    device = torch.device("cpu")
    print(f"使用设备: {device} (已强制使用CPU以避免CUDA错误)")
    
    # Weights & Biases 配置
    config = {
        "experiment": "mnist_sae",
        "input_size": 784,  # MNIST图像尺寸: 28x28=784
        "hidden_size": 1024,  # 隐藏层大小
        "k_sparse": 50,  # 稀疏参数k
        "num_saes": 5,  # SAE数量
        "learning_rate": 0.001,
        "batch_size": 64,
        "num_examples_to_visualize": 10  # 可视化示例数量
    }
    
    # 初始化 wandb
    run = setup_wandb(
        project_name="mnist-sae-reconstruction", 
        run_name=f"mnist-sae-k{config['k_sparse']}", 
        config=config
    )
    
    # 加载数据集
    print("加载MNIST数据集...")
    mnist_dataset = MNISTDataset(is_train=False)  # 使用测试集
    
    # 创建数据加载器
    raw_dataloader = DataLoader(
        mnist_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,  # 避免多进程的复杂性
        pin_memory=False  # 由于使用CPU，关闭pin_memory
    )
    
    # 创建一个修改过的数据加载器，与SAE模型使用相同的格式
    class DataLoaderWrapper:
        def __init__(self, dataloader):
            self.dataloader = dataloader
            
        def __iter__(self):
            for batch in self.dataloader:
                # 只保留第一个元素(X_batch,)，丢弃标签
                yield batch[0]
                
        def __len__(self):
            return len(self.dataloader)
    
    # 我们保留原始的dataloader，因为在后面代码中我们需要同时访问数据和标签
    dataloader = raw_dataloader
    
    # 加载SAE模型 (模型路径可以根据实际情况修改)
    model_path = "artifacts/epoch_last.pth"  # 这里假设使用本地保存的模型
    model = load_sae_model(
        model_path, 
        config['input_size'], 
        config['hidden_size'], 
        config['k_sparse'], 
        config['num_saes'], 
        device
    )
    
    # 显示模型信息
    print("模型结构:")
    print(model)
    
    # 设置SAE ID (使用哪个SAE)
    sae_id = 0  # 默认使用第一个SAE
    
    # 进行图像重构
    print(f"开始MNIST图像重构 (使用SAE {sae_id})...")
    
    # 记录重构相关系数
    reconstruction_correlations = []
    visualization_examples = []
    total_examples = 0
    
    with torch.no_grad():
        for batch_idx, ((X_batch,), labels) in enumerate(dataloader):
            # 将数据移动到适当设备上
            X_batch = X_batch.to(device)
            
            # 前向传播
            outputs, activations, indices = model.forward_with_encoded(X_batch)
            
            # 获取重构结果和激活信息
            reconstructed = outputs[sae_id].cpu().numpy()
            original = X_batch.cpu().numpy()
            act_indices = [indices[sae_id][i].cpu().numpy() for i in range(len(X_batch))]
            
            # 计算每个样本的Pearson相关系数
            for i in range(len(X_batch)):
                orig = original[i]
                recon = reconstructed[i]
                corr, _ = pearsonr(orig, recon)
                reconstruction_correlations.append(corr)
                
                # 保存一些示例用于可视化
                if len(visualization_examples) < config['num_examples_to_visualize'] and batch_idx % 10 == 0:
                    visualization_examples.append({
                        'original': orig,
                        'reconstructed': recon,
                        'indices': act_indices[i],
                        'label': labels[i].item(),
                        'corr': corr
                    })
            
            total_examples += len(X_batch)
            
            # 每处理100个批次，记录一些指标
            if batch_idx % 100 == 0:
                print(f"已处理 {total_examples} 个样本")
                
                if len(reconstruction_correlations) > 0:
                    mean_corr = np.mean(reconstruction_correlations)
                    print(f"平均重构相关系数: {mean_corr:.4f}")
                    
                    # 记录到wandb
                    run.log({
                        "mean_reconstruction_correlation": mean_corr,
                        "processed_examples": total_examples
                    })
            
            # 只处理一定数量的批次作为演示
            if batch_idx >= 500:  # 约32000个样本
                break
    
    # 记录最终统计信息到wandb
    mean_corr = np.mean(reconstruction_correlations)
    print(f"\n重构完成。平均相关系数: {mean_corr:.4f}")
    run.log({"final_mean_reconstruction_correlation": mean_corr})
    
    # 可视化一些示例，并上传到wandb
    print("生成可视化...")
    for i, example in enumerate(visualization_examples):
        fig = plot_mnist_reconstruction(
            example['original'], 
            example['reconstructed'], 
            example['indices'], 
            config['k_sparse'], 
            example['corr']
        )
        
        # 保存图像到本地
        local_path = f"mnist_reconstruction_{i}_label_{example['label']}.png"
        plt.savefig(local_path)
        plt.close(fig)
        
        # 记录到wandb
        run.log({
            f"reconstruction_example_{i}_label_{example['label']}": run.Image(local_path),
            f"correlation_{i}": example['corr']
        })
        
        # 移除本地文件
        time.sleep(0.1)  # 确保文件已经保存
        os.remove(local_path)
    
    # 完成wandb运行
    run.finish()
    print("任务完成")

if __name__ == "__main__":
    main()
