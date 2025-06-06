import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from typing import List, Optional, Dict, Any, Tuple
import itertools
import math
from scipy.stats import pearsonr
import numpy as np


class NeuralSAETrainer:
    """专门用于神经活动数据的SAE训练器"""
    
    def __init__(self, model: nn.Module, device: str, hyperparameters: Dict[str, Any], wandb_on: str = '0'):
        self.model = model
        self.device = device
        self.config = hyperparameters
        
        # 优化器设置
        self.optimizers = [
            torch.optim.Adam(encoder.parameters(), lr=self.config["learning_rate"]) 
            for encoder in self.model.encoders
        ]
        
        # 损失函数
        self.criterion = nn.MSELoss()
        
        # 混合精度训练设置
        self.use_amp = hyperparameters.get("use_amp", True) and not str(device).startswith('mps')
        if self.use_amp:
            self.scalers = [GradScaler() for _ in self.model.encoders]
        
        # 训练参数
        self.ensemble_consistency_weight = hyperparameters.get("ensemble_consistency_weight", 0.01)
        self.reinit_threshold = hyperparameters.get("reinit_threshold", 2.5)
        self.warmup_steps = hyperparameters.get("warmup_steps", 100)
        self.current_step = 0
        self.wandb_on = wandb_on
        
        # 用于存储训练指标
        self.training_history = {
            'reconstruction_loss': [],
            'consistency_loss': [],
            'reconstruction_correlation': [],
            'feature_usage': []
        }
        
        print(f"初始化神经SAE训练器")
        print(f"设备: {device}")
        print(f"使用混合精度训练: {self.use_amp}")
        print(f"SAE数量: {len(self.model.encoders)}")

    def get_warmup_factor(self) -> float:
        """计算warmup因子"""
        if self.current_step >= self.warmup_steps:
            return 1.0
        return 0.5 * (1 + math.cos(math.pi * (self.warmup_steps - self.current_step) / self.warmup_steps))

    def calculate_consensus_loss(self, encoder_weights: List[torch.Tensor]) -> torch.Tensor:
        """计算多个SAE之间的一致性损失"""
        if len(encoder_weights) < 2:
            return torch.tensor(0.0, device=self.device)
            
        pairs = list(itertools.combinations(encoder_weights, 2))
        if not pairs:
            return torch.tensor(0.0, device=self.device)
            
        # 计算权重之间的余弦相似度
        similarities = []
        for w1, w2 in pairs:
            # 归一化权重
            w1_norm = torch.nn.functional.normalize(w1.view(-1), p=2, dim=0)
            w2_norm = torch.nn.functional.normalize(w2.view(-1), p=2, dim=0)
            # 计算余弦相似度
            sim = torch.dot(w1_norm, w2_norm)
            similarities.append(1 - sim)  # 1-相似度作为损失
            
        unweighted_loss = torch.mean(torch.stack(similarities))
        warmup_factor = self.get_warmup_factor()
        return unweighted_loss * self.ensemble_consistency_weight * warmup_factor

    def reinitialize_sae_weights(self, sae_index: int):
        """重新初始化SAE权重"""
        encoder = self.model.encoders[sae_index]
        
        # 重新初始化权重
        if str(self.device).startswith('mps'):
            std = 1.0 / (encoder.weight.size(1) ** 0.5)
            nn.init.normal_(encoder.weight, std=std)
        else:
            nn.init.xavier_uniform_(encoder.weight)
        nn.init.zeros_(encoder.bias)
        
        # 重新创建优化器
        self.optimizers[sae_index] = torch.optim.Adam(
            encoder.parameters(), 
            lr=self.config["learning_rate"]
        )
        
        if self.use_amp:
            self.scalers[sae_index] = GradScaler()
            
        print(f"重新初始化SAE {sae_index} 的权重")

    def check_reinit_condition(self, encoded_activations: List[torch.Tensor]) -> Tuple[List[bool], List[float]]:
        """检查是否需要重新初始化权重"""
        reinit_flags = []
        feature_activity = []
        
        for activations in encoded_activations:
            # 计算激活率
            activation_rates = (activations != 0).float().mean(dim=0)
            target_rate = 0.05  # 目标激活率5%
            
            # 计算与目标激活率的偏差
            rate_deviation = torch.abs(activation_rates - target_rate).mean()
            feature_activity.append(rate_deviation.item())
            
            # 如果偏差过大，标记为需要重初始化
            reinit_flags.append(rate_deviation.item() > self.reinit_threshold)
            
        return reinit_flags, feature_activity

    def calculate_reconstruction_correlation(self, original: torch.Tensor, 
                                           reconstructed: torch.Tensor) -> float:
        """计算重建相关性"""
        # 转换为numpy进行相关性计算
        orig_np = original.detach().cpu().numpy().flatten()
        recon_np = reconstructed.detach().cpu().numpy().flatten()
        
        try:
            correlation, _ = pearsonr(orig_np, recon_np)
            return correlation if not np.isnan(correlation) else 0.0
        except:
            return 0.0

    def evaluate_feature_usage(self, encoded_activations: List[torch.Tensor]) -> Dict[str, float]:
        """评估特征使用情况"""
        feature_stats = {}
        
        for i, activations in enumerate(encoded_activations):
            # 计算激活的特征数量
            active_features = (activations != 0).sum(dim=1).float().mean()
            total_features = activations.shape[1]
            usage_ratio = active_features / total_features
            
            feature_stats[f'sae_{i}_feature_usage'] = usage_ratio.item()
            feature_stats[f'sae_{i}_active_features'] = active_features.item()
            
        return feature_stats

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        epoch_losses = []
        epoch_correlations = []
        epoch_consistency_losses = []
        
        for batch_idx, (X_batch,) in enumerate(train_loader):
            self.current_step += 1
            X_batch = X_batch.to(self.device)
            
            # 前向传播
            if self.use_amp:
                with autocast():
                    outputs, activations, indices = self.model.forward_with_encoded(X_batch)
                    outputs = [output.to(self.device) for output in outputs]
                    
                    # 计算重建损失
                    reconstruction_losses = [self.criterion(output, X_batch) for output in outputs]
                    
                    # 计算一致性损失
                    encoder_weights = [encoder.weight for encoder in self.model.encoders]
                    consistency_loss = self.calculate_consensus_loss(encoder_weights)
            else:
                outputs, activations, indices = self.model.forward_with_encoded(X_batch)
                outputs = [output.to(self.device) for output in outputs]
                
                # 计算重建损失
                reconstruction_losses = [self.criterion(output, X_batch) for output in outputs]
                
                # 计算一致性损失
                encoder_weights = [encoder.weight for encoder in self.model.encoders]
                consistency_loss = self.calculate_consensus_loss(encoder_weights)
            
            # 检查是否需要重新初始化
            reinit_flags, feature_activity = self.check_reinit_condition(activations)
            for i, flag in enumerate(reinit_flags):
                if flag:
                    self.reinitialize_sae_weights(i)
            
            # 如果有重初始化，重新前向传播
            if any(reinit_flags):
                if self.use_amp:
                    with autocast():
                        outputs, activations, indices = self.model.forward_with_encoded(X_batch)
                        outputs = [output.to(self.device) for output in outputs]
                        reconstruction_losses = [self.criterion(output, X_batch) for output in outputs]
                else:
                    outputs, activations, indices = self.model.forward_with_encoded(X_batch)
                    outputs = [output.to(self.device) for output in outputs]
                    reconstruction_losses = [self.criterion(output, X_batch) for output in outputs]
            
            # 计算总损失
            total_losses = [rec_loss + consistency_loss for rec_loss in reconstruction_losses]
            
            # 反向传播和优化 - MPS兼容的方式
            # 先清空所有梯度
            for optimizer in self.optimizers:
                optimizer.zero_grad()
            
            # 计算所有损失的总和进行一次反向传播
            total_combined_loss = sum(total_losses) / len(total_losses)
            
            if self.use_amp:
                # 由于MPS不支持AMP，这部分不会执行
                scaler = self.scalers[0]
                scaler.scale(total_combined_loss).backward()
                for optimizer, scaler in zip(self.optimizers, self.scalers):
                    scaler.step(optimizer)
                    scaler.update()
            else:
                # 一次性反向传播
                total_combined_loss.backward()
                # 分别更新每个优化器
                for optimizer in self.optimizers:
                    optimizer.step()
            
            # 记录指标
            avg_reconstruction_loss = sum(reconstruction_losses) / len(reconstruction_losses)
            epoch_losses.append(avg_reconstruction_loss.item())
            epoch_consistency_losses.append(consistency_loss.item())
            
            # 计算重建相关性（使用第一个SAE）
            correlation = self.calculate_reconstruction_correlation(X_batch, outputs[0])
            epoch_correlations.append(correlation)
            
            # 每100个批次打印进度
            if batch_idx % 100 == 0:
                print(f"批次 {batch_idx}/{len(train_loader)}: "
                      f"重建损失={avg_reconstruction_loss:.6f}, "
                      f"一致性损失={consistency_loss:.6f}, "
                      f"相关性={correlation:.4f}")
        
        # 返回epoch平均指标
        return {
            'reconstruction_loss': np.mean(epoch_losses),
            'consistency_loss': np.mean(epoch_consistency_losses),
            'reconstruction_correlation': np.mean(epoch_correlations)
        }

    def evaluate(self, test_loader: DataLoader, num_batches: Optional[int] = None) -> Dict[str, float]:
        """评估模型性能"""
        self.model.eval()
        eval_losses = []
        eval_correlations = []
        feature_usage_stats = []
        
        with torch.no_grad():
            for batch_idx, (X_batch,) in enumerate(test_loader):
                if num_batches and batch_idx >= num_batches:
                    break
                    
                X_batch = X_batch.to(self.device)
                
                # 前向传播
                outputs, activations, indices = self.model.forward_with_encoded(X_batch)
                outputs = [output.to(self.device) for output in outputs]
                
                # 计算损失（使用第一个SAE）
                loss = self.criterion(outputs[0], X_batch)
                eval_losses.append(loss.item())
                
                # 计算相关性
                correlation = self.calculate_reconstruction_correlation(X_batch, outputs[0])
                eval_correlations.append(correlation)
                
                # 评估特征使用情况
                usage_stats = self.evaluate_feature_usage(activations)
                feature_usage_stats.append(usage_stats)
        
        # 计算平均特征使用统计
        avg_feature_stats = {}
        if feature_usage_stats:
            keys = feature_usage_stats[0].keys()
            for key in keys:
                avg_feature_stats[key] = np.mean([stats[key] for stats in feature_usage_stats])
        
        return {
            'eval_reconstruction_loss': np.mean(eval_losses),
            'eval_reconstruction_correlation': np.mean(eval_correlations),
            **avg_feature_stats
        }

    def train(self, train_loader: DataLoader, num_epochs: int = 1, 
              test_loader: Optional[DataLoader] = None, 
              eval_every_n_epochs: int = 1) -> Dict[str, List[float]]:
        """完整的训练过程"""
        
        print(f"开始训练神经SAE模型，共 {num_epochs} 个epoch")
        
        for epoch in range(num_epochs):
            print(f"\n===== Epoch {epoch + 1}/{num_epochs} =====")
            
            # 训练一个epoch
            train_metrics = self.train_epoch(train_loader)
            
            # 记录训练指标
            for key, value in train_metrics.items():
                if key in self.training_history:
                    self.training_history[key].append(value)
            
            print(f"训练指标: 重建损失={train_metrics['reconstruction_loss']:.6f}, "
                  f"一致性损失={train_metrics['consistency_loss']:.6f}, "
                  f"重建相关性={train_metrics['reconstruction_correlation']:.4f}")
            
            # 定期评估
            if test_loader and (epoch + 1) % eval_every_n_epochs == 0:
                print("评估模型性能...")
                eval_metrics = self.evaluate(test_loader, num_batches=50)
                print(f"评估指标: 重建损失={eval_metrics['eval_reconstruction_loss']:.6f}, "
                      f"重建相关性={eval_metrics['eval_reconstruction_correlation']:.4f}")
                
                # 打印特征使用情况
                for key, value in eval_metrics.items():
                    if 'feature_usage' in key:
                        print(f"{key}: {value:.4f}")
        
        return self.training_history
