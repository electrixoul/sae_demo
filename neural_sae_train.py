import torch
import os
import argparse
import time
from neural_sae import NeuralSparseAutoencoder
from neural_dataset import create_neural_dataloaders
from neural_sae_trainer import NeuralSAETrainer
import matplotlib.pyplot as plt
import numpy as np


def get_device():
    """获取最佳可用设备"""
    if torch.backends.mps.is_available():
        print("使用 MPS (Apple Silicon) 加速")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("使用 CUDA 加速")
        return torch.device("cuda")
    else:
        print("使用 CPU")
        return torch.device("cpu")


def create_neural_sae_config(input_size: int, device: str) -> dict:
    """为神经数据创建SAE配置 - 1:1对称设计（内存友好版本）"""
    
    # 为了内存效率，使用1:1的对称设计：隐藏层 = 输入层
    # 这样既保持了特征学习能力，又控制了内存消耗
    hidden_size = input_size  # 9941 = 9941
    
    # 稀疏度设计：参考MNIST SAE的比例但调整到1:1架构
    # MNIST: 50/1024 ≈ 4.9%，对于1:1架构我们使用相似比例
    k_sparse = max(50, int(input_size * 0.05))  # 约5%稀疏度
    
    config = {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "k_sparse": k_sparse,
        "num_saes": 3,  # 使用3个SAE进行集成学习
        "learning_rate": 0.001,
        "ensemble_consistency_weight": 0.01,
        "reinit_threshold": 2.0,
        "warmup_steps": 100,
        "use_amp": False if device == "mps" else True,  # MPS不支持AMP
    }
    
    print(f"SAE配置 (1:1对称设计 - 内存友好版):")
    print(f"  输入维度: {config['input_size']:,}")
    print(f"  隐藏层维度: {config['hidden_size']:,} (1:1对称)")
    print(f"  稀疏度K: {config['k_sparse']} ({k_sparse/hidden_size*100:.1f}%)")
    print(f"  SAE数量: {config['num_saes']}")
    print(f"  学习率: {config['learning_rate']}")
    print(f"  参数量估计: {config['input_size'] * config['hidden_size'] * config['num_saes']:,}")
    
    return config


def plot_training_progress(training_history: dict, save_path: str = "neural_sae_training_progress.png"):
    """绘制训练进度"""
    # 设置matplotlib支持中文
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 改为一行三列布局
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 重建损失
    if 'reconstruction_loss' in training_history:
        axes[0].plot(training_history['reconstruction_loss'], 'b-', linewidth=2)
        axes[0].set_title('Reconstruction Loss', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('MSE Loss')
        axes[0].grid(True, alpha=0.3)
        axes[0].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    # 一致性损失
    if 'consistency_loss' in training_history:
        axes[1].plot(training_history['consistency_loss'], 'r-', linewidth=2)
        axes[1].set_title('Consistency Loss', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Consistency Loss')
        axes[1].grid(True, alpha=0.3)
        axes[1].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    # 重建相关性
    if 'reconstruction_correlation' in training_history:
        axes[2].plot(training_history['reconstruction_correlation'], 'g-', linewidth=2, marker='o')
        axes[2].set_title('Reconstruction Correlation', fontsize=12, fontweight='bold')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Pearson R')
        axes[2].set_ylim([0, 1])
        axes[2].grid(True, alpha=0.3)
        
        # 添加最终相关性值标注
        if len(training_history['reconstruction_correlation']) > 0:
            final_corr = training_history['reconstruction_correlation'][-1]
            axes[2].text(0.02, 0.98, f'Final R: {final_corr:.4f}', 
                        transform=axes[2].transAxes, fontsize=10,
                        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
                        verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Training progress saved to: {save_path}")
    
    return save_path


def plot_sae_reconstruction_comparison(model, test_loader, device, dataset_metadata, 
                                     save_path: str = "neural_sae_reconstruction_comparison.png"):
    """绘制SAE的输入输出对比图"""
    model.eval()
    
    with torch.no_grad():
        for batch in test_loader:
            X_batch = batch[0].to(device)
            if X_batch.shape[0] >= 1:
                # 取一个样本进行可视化
                X_sample = X_batch[0:1]  # 保持batch维度
                outputs, activations, indices = model.forward_with_encoded(X_sample)
                
                # 使用第一个SAE的输出
                X_reconstructed = outputs[0][0]  # 去掉batch维度
                X_original = X_sample[0]  # 去掉batch维度
                
                # 反归一化以便分析
                if dataset_metadata.get('normalized', False):
                    data_min = dataset_metadata['data_min']
                    data_max = dataset_metadata['data_max']
                    X_orig_denorm = X_original * (data_max - data_min) + data_min
                    X_recon_denorm = X_reconstructed * (data_max - data_min) + data_min
                else:
                    X_orig_denorm = X_original
                    X_recon_denorm = X_reconstructed
                
                # 转为numpy
                orig_signal = X_orig_denorm.cpu().numpy()
                recon_signal = X_recon_denorm.cpu().numpy()
                
                # 计算Pearson相关系数
                from scipy.stats import pearsonr
                try:
                    corr, _ = pearsonr(orig_signal, recon_signal)
                    corr = corr if not np.isnan(corr) else 0.0
                except:
                    corr = 0.0
                
                # 绘制对比图
                plt.figure(figsize=(12, 6))
                
                # 选择显示前500个神经元（如果总数超过500的话）
                display_neurons = min(500, len(orig_signal))
                neuron_indices = np.arange(display_neurons)
                
                plt.plot(neuron_indices, orig_signal[:display_neurons], 'b-', 
                        linewidth=1.5, alpha=0.8, label='Original Input')
                plt.plot(neuron_indices, recon_signal[:display_neurons], 'r-', 
                        linewidth=1.5, alpha=0.8, label='SAE Reconstruction')
                
                plt.xlabel('Neuron Index', fontsize=12)
                plt.ylabel('Neural Activity', fontsize=12)
                plt.title(f'SAE Reconstruction Comparison (Pearson R = {corr:.4f})', 
                         fontsize=14, fontweight='bold')
                plt.legend(fontsize=11)
                plt.grid(True, alpha=0.3)
                
                # 添加统计信息
                rmse = np.sqrt(np.mean((orig_signal - recon_signal)**2))
                plt.text(0.02, 0.98, f'Pearson R: {corr:.4f}\nRMSE: {rmse:.4f}\nNeurons shown: {display_neurons}/{len(orig_signal)}', 
                        transform=plt.gca().transAxes, fontsize=10,
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                        verticalalignment='top')
                
                plt.tight_layout()
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"SAE reconstruction comparison saved to: {save_path}")
                plt.close()
                
                return corr
    
    return 0.0


def test_model_reconstruction(model, test_loader, device, dataset_metadata, num_samples=5):
    """测试模型重建质量"""
    model.eval()
    
    with torch.no_grad():
        for batch in test_loader:
            X_batch = batch[0].to(device)
            if X_batch.shape[0] >= num_samples:
                # 取前几个样本进行测试
                X_test = X_batch[:num_samples]
                outputs, activations, indices = model.forward_with_encoded(X_test)
                
                # 使用第一个SAE的输出
                X_reconstructed = outputs[0]
                
                # 反归一化以便分析
                if dataset_metadata.get('normalized', False):
                    data_min = dataset_metadata['data_min']
                    data_max = dataset_metadata['data_max']
                    X_test_denorm = X_test * (data_max - data_min) + data_min
                    X_recon_denorm = X_reconstructed * (data_max - data_min) + data_min
                else:
                    X_test_denorm = X_test
                    X_recon_denorm = X_reconstructed
                
                # 计算每个样本的相关性
                correlations = []
                for i in range(num_samples):
                    orig = X_test_denorm[i].cpu().numpy()
                    recon = X_recon_denorm[i].cpu().numpy()
                    
                    from scipy.stats import pearsonr
                    try:
                        corr, _ = pearsonr(orig, recon)
                        correlations.append(corr if not np.isnan(corr) else 0.0)
                    except:
                        correlations.append(0.0)
                
                print(f"\n=== 重建质量测试 ===")
                print(f"样本数量: {num_samples}")
                print(f"神经元数量: {X_test.shape[1]}")
                print(f"原始数据范围: {X_test_denorm.min():.4f} - {X_test_denorm.max():.4f}")
                print(f"重建数据范围: {X_recon_denorm.min():.4f} - {X_recon_denorm.max():.4f}")
                print(f"Pearson相关系数:")
                for i, corr in enumerate(correlations):
                    print(f"  样本 {i+1}: {corr:.4f}")
                print(f"平均相关系数: {np.mean(correlations):.4f}")
                
                # 计算特征使用统计
                activation = activations[0]  # 使用第一个SAE
                active_features = (activation != 0).sum(dim=1)
                total_features = activation.shape[1]
                sparsity = active_features.float() / total_features
                
                print(f"\n特征使用统计:")
                print(f"总特征数: {total_features}")
                print(f"平均激活特征数: {active_features.float().mean():.1f}")
                print(f"稀疏度: {sparsity.mean():.3f} ({sparsity.mean()*100:.1f}%)")
                
                break
    
    return np.mean(correlations)


def main():
    parser = argparse.ArgumentParser(description="神经活动数据SAE训练")
    parser.add_argument('--data_dir', type=str, default='neural_activity_data_20250606_163121',
                      help='神经数据目录')
    parser.add_argument('--stimulus_types', nargs='+', 
                      default=['expanding_circle', 'moving_bar'],
                      help='要使用的刺激类型')
    parser.add_argument('--epochs', type=int, default=10,
                      help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='学习率')
    parser.add_argument('--save_model', action='store_true',
                      help='是否保存训练后的模型')
    
    args = parser.parse_args()
    
    print("===== 神经活动数据SAE训练 =====")
    print(f"数据目录: {args.data_dir}")
    print(f"刺激类型: {args.stimulus_types}")
    print(f"训练轮数: {args.epochs}")
    print(f"批次大小: {args.batch_size}")
    print(f"学习率: {args.learning_rate}")
    
    # 检查数据目录是否存在
    if not os.path.exists(args.data_dir):
        print(f"错误: 数据目录 {args.data_dir} 不存在")
        return
    
    # 获取设备
    device = get_device()
    
    # 创建数据加载器
    print("\n=== 加载数据 ===")
    try:
        train_loader, test_loader, dataset_metadata = create_neural_dataloaders(
            data_dir=args.data_dir,
            stimulus_types=args.stimulus_types,
            batch_size=args.batch_size,
            train_split=0.8,
            normalize=True,
            num_workers=0,  # MPS可能有多进程问题
            pin_memory=False if str(device).startswith('mps') else True
        )
    except Exception as e:
        print(f"数据加载失败: {e}")
        return
    
    print(f"数据加载成功!")
    print(f"训练批次数: {len(train_loader)}")
    print(f"测试批次数: {len(test_loader)}")
    
    # 创建SAE配置
    input_size = dataset_metadata['num_neurons']
    config = create_neural_sae_config(input_size, str(device))
    config['learning_rate'] = args.learning_rate
    
    # 创建模型
    print(f"\n=== 创建模型 ===")
    model = NeuralSparseAutoencoder(config)
    model = model.to(device)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    
    # 创建训练器
    print(f"\n=== 创建训练器 ===")
    trainer = NeuralSAETrainer(
        model=model,
        device=str(device),
        hyperparameters=config,
        wandb_on='0'
    )
    
    # 开始训练
    print(f"\n=== 开始训练 ===")
    start_time = time.time()
    
    try:
        training_history = trainer.train(
            train_loader=train_loader,
            num_epochs=args.epochs,
            test_loader=test_loader,
            eval_every_n_epochs=max(1, args.epochs // 5)  # 每20%的epoch评估一次
        )
        
        end_time = time.time()
        training_time = end_time - start_time
        print(f"\n训练完成! 总用时: {training_time:.1f} 秒")
        
        # 绘制训练进度
        print(f"\n=== 生成训练报告 ===")
        plot_path = plot_training_progress(training_history)
        
        # 最终测试模型性能
        print(f"\n=== 最终性能测试 ===")
        final_correlation = test_model_reconstruction(
            model, test_loader, device, dataset_metadata, num_samples=10
        )
        
        # 生成输入输出对比图
        print(f"\n=== 生成重建对比图 ===")
        comparison_corr = plot_sae_reconstruction_comparison(
            model, test_loader, device, dataset_metadata
        )
        
        # 保存模型
        if args.save_model:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            model_name = f"neural_sae_{timestamp}"
            save_path = model.save_model_local(model_name)
            print(f"\n模型已保存: {save_path}")
        
        # 打印总结
        print(f"\n=== 训练总结 ===")
        print(f"最终重建相关性: {final_correlation:.4f}")
        if training_history.get('reconstruction_correlation'):
            best_correlation = max(training_history['reconstruction_correlation'])
            print(f"训练过程最佳相关性: {best_correlation:.4f}")
        
        if training_history.get('reconstruction_loss'):
            final_loss = training_history['reconstruction_loss'][-1]
            print(f"最终重建损失: {final_loss:.6f}")
        
        print(f"训练图表保存至: {plot_path}")
        
    except KeyboardInterrupt:
        print("\n训练被用户中断")
    except Exception as e:
        print(f"\n训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
