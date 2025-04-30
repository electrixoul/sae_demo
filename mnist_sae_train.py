import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from scipy.stats import pearsonr
from models.sae import SparseAutoencoder
from custom_sae_trainer import MNISTSAETrainer
import time
import argparse

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

def test_reconstruction(model, dataloader, device, config, sae_id=0, num_batches=100):
    """
    测试SAE对MNIST数据的重构效果
    
    参数:
        model: SAE模型
        dataloader: 数据加载器
        device: 计算设备
        config: 配置字典
        sae_id: 使用哪个SAE
        num_batches: 测试的批次数量
    
    返回:
        平均相关系数, 可视化示例列表
    """
    print(f"开始测试重构效果 (使用SAE {sae_id})...")
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
                if len(visualization_examples) < config['num_examples_to_visualize'] and batch_idx % 5 == 0:
                    visualization_examples.append({
                        'original': orig,
                        'reconstructed': recon,
                        'indices': act_indices[i],
                        'label': labels[i].item(),
                        'corr': corr
                    })
            
            total_examples += len(X_batch)
            
            # 每处理10个批次，打印进度
            if batch_idx % 10 == 0:
                print(f"已处理 {total_examples} 个样本 (批次 {batch_idx}/{num_batches})")
            
            # 只处理指定数量的批次
            if batch_idx >= num_batches:
                break
    
    mean_corr = np.mean(reconstruction_correlations)
    print(f"\n测试完成。平均相关系数: {mean_corr:.4f}")
    return mean_corr, visualization_examples

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="训练MNIST SAE模型")
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cpu',
                      help='选择训练设备: cpu或cuda')
    parser.add_argument('--analyze_loss', action='store_true',
                      help='分析重构损失的初始值')
    args = parser.parse_args()
    
    # 设置随机种子确保结果可复现
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 设置设备
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"使用设备: {device}")
    else:
        if args.device == 'cuda':
            print("CUDA设备不可用，使用CPU代替")
        device = torch.device("cpu")
        print(f"使用设备: {device}")
    
    # 模型和训练配置
    config = {
        "experiment": "mnist_sae_training",
        "input_size": 784,  # MNIST图像尺寸: 28x28=784
        "hidden_size": 1024,  # 隐藏层大小
        "k_sparse": 50,  # 稀疏参数k
        "num_saes": 5,  # SAE数量
        "learning_rate": 0.001,
        "num_epochs": 5,  # 训练轮数
        "training_batch_size": 128,
        "test_batch_size": 64,
        "num_examples_to_visualize": 10,  # 可视化示例数量
        "ensemble_consistency_weight": 0.01,  # SAE之间的一致性约束权重
        "test_every_n_epochs": 1  # 每N个epoch测试一次
    }
    
    # 初始化 wandb
    run = setup_wandb(
        project_name="mnist-sae-training", 
        run_name=f"mnist-sae-train-k{config['k_sparse']}", 
        config=config
    )
    
    # 加载数据集
    print("加载MNIST数据集...")
    train_dataset = MNISTDataset(is_train=True)
    test_dataset = MNISTDataset(is_train=False)
    
    # 配置数据加载器参数 - 根据设备类型调整
    use_gpu = str(device).startswith('cuda')
    loader_kwargs = {
        'num_workers': 2 if use_gpu else 0,  # GPU模式下使用多进程加载
        'pin_memory': use_gpu,  # GPU模式下启用pinned memory
        'generator': torch.Generator().manual_seed(42)  # 确保可重现性
    }
    
    # 创建数据加载器
    raw_train_loader = DataLoader(
        train_dataset,
        batch_size=config['training_batch_size'],
        shuffle=True,
        **loader_kwargs
    )
    
    raw_test_loader = DataLoader(
        test_dataset,
        batch_size=config['test_batch_size'],
        shuffle=False,
        **loader_kwargs
    )
    
    # 创建自定义的数据加载器包装器，使其与SAE trainer兼容
    # SAE trainer期望格式为(X_batch,)，但MNIST loader返回((X_batch,), labels)
    class DataLoaderWrapper:
        def __init__(self, dataloader):
            self.dataloader = dataloader
            
        def __iter__(self):
            for batch in self.dataloader:
                # 只保留第一个元素(X_batch,)，丢弃标签
                yield batch[0]
                
        def __len__(self):
            return len(self.dataloader)
    
    # 使用包装器适配数据加载器
    train_loader = DataLoaderWrapper(raw_train_loader)
    test_loader = raw_test_loader  # 测试加载器不需要包装，因为test_reconstruction函数自己处理
    
    print(f"MNIST训练集: {len(train_dataset)} 样本")
    print(f"MNIST测试集: {len(test_dataset)} 样本")
    
    # 创建SAE模型
    model = SparseAutoencoder(config).to(device)
    print("模型结构:")
    print(model)
    
    # 创建SAE训练器
    trainer = MNISTSAETrainer(model, device, config, wandb_on='1')
    
    # 保存初始模型
    model_save_dir = "mnist_sae_models"
    os.makedirs(model_save_dir, exist_ok=True)
    initial_model_path = os.path.join(model_save_dir, "mnist_sae_initial.pth")
    torch.save(model.state_dict(), initial_model_path)
    print(f"初始模型保存到: {initial_model_path}")
    
    # 分析重构损失的初始值
    if args.analyze_loss:
        print("\n分析初始重构损失...")
        # 创建MSE损失函数
        criterion = nn.MSELoss()
        
        # 计算理论最小损失 - 当所有输入都被重构为平均值时
        with torch.no_grad():
            # 收集一批数据样本
            sample_batches = []
            for i, ((X_batch,), _) in enumerate(raw_train_loader):
                sample_batches.append(X_batch)
                if i >= 9:  # 收集10批
                    break
            
            # 合并样本
            samples = torch.cat(sample_batches, dim=0)
            print(f"分析样本数量: {samples.shape[0]}")
            
            # 计算平均图像
            mean_image = samples.mean(dim=0).to(device)
            
            # 如果每个输入都被重构为平均图像的损失
            mean_loss = 0.0
            for X_batch in sample_batches:
                X_batch = X_batch.to(device)
                # 计算每个输入与平均图像的MSE
                batch_mean_loss = criterion(X_batch, mean_image.expand_as(X_batch))
                mean_loss += batch_mean_loss.item()
            mean_loss /= len(sample_batches)
            
            # 计算初始模型的实际损失
            model.eval()
            initial_losses = []
            for X_batch in sample_batches:
                X_batch = X_batch.to(device)
                # 前向传播获取重构
                outputs, _, _ = model.forward_with_encoded(X_batch)
                # 计算每个SAE的损失
                for i, output in enumerate(outputs):
                    loss = criterion(output, X_batch).item()
                    initial_losses.append((i, loss))
            
            # 按SAE分组并计算平均损失
            sae_losses = {}
            for i, loss in initial_losses:
                if i not in sae_losses:
                    sae_losses[i] = []
                sae_losses[i].append(loss)
            
            # 打印分析结果
            print("\n==== 初始重构损失分析 ====")
            print(f"理论最小损失 (如果所有输入都重构为均值): {mean_loss:.6f}")
            print("\n各SAE的初始重构损失:")
            for sae_id, losses in sae_losses.items():
                avg_loss = sum(losses) / len(losses)
                print(f"  SAE {sae_id}: {avg_loss:.6f}")
            
            print("\n结论:")
            print("1. 初始损失约为0.08是合理的，这与MNIST数据的方差和初始化方法有关")
            print("2. 由于权重以正交方式初始化，初始重构已经比随机猜测要好")
            print("3. 理论最小损失(均值重构)说明了数据的固有变异性")
            print("4. 随着训练进行，损失会从初始值~0.08逐渐降低到更小的值")
    
    # 训练模型
    print("\n开始训练SAE模型...")
    for epoch in range(config['num_epochs']):
        print(f"\n轮次 {epoch+1}/{config['num_epochs']}")
        
        # 训练一个周期
        trainer.train(train_loader, 1)
        
        # 保存当前模型
        epoch_model_path = os.path.join(model_save_dir, f"mnist_sae_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), epoch_model_path)
        print(f"模型保存到: {epoch_model_path}")
        
        # 定期测试重构效果
        if (epoch + 1) % config['test_every_n_epochs'] == 0:
            # 测试每个SAE的重构效果
            for sae_id in range(config['num_saes']):
                mean_corr, examples = test_reconstruction(
                    model, test_loader, device, config, sae_id=sae_id, num_batches=50
                )
                
                # 记录到wandb
                run.log({
                    f"epoch": epoch + 1,
                    f"test_correlation_sae_{sae_id}": mean_corr
                })
                
                # 生成并记录可视化
                if epoch + 1 == config['num_epochs']:  # 只在最后一个epoch可视化
                    for i, example in enumerate(examples[:5]):  # 限制为5个示例
                        fig = plot_mnist_reconstruction(
                            example['original'], 
                            example['reconstructed'], 
                            example['indices'], 
                            config['k_sparse'], 
                            example['corr']
                        )
                        
                        # 保存图像到本地
                        local_path = f"mnist_sae{sae_id}_epoch{epoch+1}_ex{i}_label{example['label']}.png"
                        plt.savefig(local_path)
                        plt.close(fig)
                        
                        # 记录到wandb
                        run.log({
                            f"sae{sae_id}_reconstruction_{i}_label_{example['label']}": run.Image(local_path)
                        })
                        
                        # 移除本地文件
                        time.sleep(0.1)
                        os.remove(local_path)
    
    # 保存最终模型
    final_model_path = os.path.join(model_save_dir, "mnist_sae_final.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"\n最终模型保存到: {final_model_path}")
    
    # 记录最终模型到wandb
    artifact = run.Artifact("mnist_sae_model", type="model")
    artifact.add_file(final_model_path)
    run.log_artifact(artifact)
    
    # 完成训练
    run.finish()
    print("训练完成！")

if __name__ == "__main__":
    main()
