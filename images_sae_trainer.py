import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from typing import List, Optional, Dict, Any, Tuple
import itertools
import wandb
import math

class ImageSAETrainer:
    def __init__(self, model: nn.Module, device: str, hyperparameters: Dict[str, Any], true_features: Optional[torch.Tensor] = None, wandb_on: str = '0'):
        self.model = model
        self.device = device
        self.config = hyperparameters
        self.base_model = model
        self.optimizers = [torch.optim.Adam(encoder.parameters(), lr=self.config["learning_rate"]) for encoder in self.base_model.encoders]
        self.criterion = nn.MSELoss()
        self.true_features = true_features.to(device) if true_features is not None else None
        
        # 根据设备自动配置AMP（混合精度训练）和梯度缩放
        self.use_amp = (torch.cuda.is_available() and str(device).startswith('cuda')) or str(device).startswith('mps')
        if self.use_amp:
            self.scalers = [torch.cuda.amp.GradScaler() for _ in self.base_model.encoders]
        else:
            self.scalers = [None for _ in self.base_model.encoders]
            
        self.ensemble_consistency_weight = hyperparameters.get("ensemble_consistency_weight", 0.01)
        # 禁用重初始化功能，设置为一个不可能达到的极高值
        self.reinit_threshold = hyperparameters.get("reinit_threshold", 1e10)
        self.warmup_steps = hyperparameters.get("warmup_steps", 100)
        self.current_step = 0
        self.wandb_on = wandb_on

    def get_warmup_factor(self) -> float:
        if self.current_step >= self.warmup_steps:
            return 1.0
        return 0.5 * (1 + math.cos(math.pi * (self.warmup_steps - self.current_step) / self.warmup_steps))

    def calculate_mmcs(self, A: torch.Tensor, B: torch.Tensor) -> float:
        """计算两个矩阵之间的最大平均余弦相似度 (MMCS)"""
        # 归一化每一行
        A_norm = torch.nn.functional.normalize(A, p=2, dim=1)
        B_norm = torch.nn.functional.normalize(B, p=2, dim=1)
        
        # 计算余弦相似度矩阵
        sim_matrix = torch.mm(A_norm, B_norm.t())
        
        # 获取每一行的最大相似度
        max_sim_A_to_B, _ = torch.max(sim_matrix, dim=1)
        max_sim_B_to_A, _ = torch.max(sim_matrix.t(), dim=1)
        
        # 计算平均
        mean_max_sim = (torch.mean(max_sim_A_to_B) + torch.mean(max_sim_B_to_A)) / 2
        
        return mean_max_sim.item()

    def save_model(self, epoch: int):
        if self.wandb_on == '1':
            run_name = f"{wandb.run.name}_epoch_{epoch}"
            self.base_model.save_model(run_name, alias=f"epoch_{epoch}")

    def calculate_consensus_loss(self, encoder_weights: List[torch.Tensor]) -> torch.Tensor:
        pairs = list(itertools.combinations(encoder_weights, 2))
        if not pairs:
            return torch.tensor(0.0, device=self.device)
            
        # 计算每对编码器权重之间的一致性损失
        consistency_losses = []
        for a, b in pairs:
            # 手动计算相似性，避免依赖外部函数
            mmcs = self.calculate_mmcs(a.clone(), b.clone())
            consistency_losses.append(1 - mmcs)  # 转换为损失

        # 取平均作为总一致性损失
        unweighted_loss = torch.mean(torch.tensor(consistency_losses, device=self.device))
        
        # 应用warmup因子
        warmup_factor = self.get_warmup_factor()
        return unweighted_loss * self.ensemble_consistency_weight * warmup_factor

    def reinitialize_sae_weights(self, sae_index: int):
        encoder = self.base_model.encoders[sae_index]
        dtype, device = encoder.weight.dtype, encoder.weight.device
        
        # MPS友好的初始化方式 - 避免使用orthogonal_，改用更简单的初始化
        if str(device).startswith('mps'):
            # 使用常规初始化方法替代orthogonal_
            std = 1.0 / (encoder.weight.size(1) ** 0.5)
            nn.init.normal_(encoder.weight, std=std)
        else:
            # 其他设备使用正交初始化
            nn.init.orthogonal_(encoder.weight)
            
        nn.init.zeros_(encoder.bias)
        encoder.weight.data = encoder.weight.data.to(dtype=dtype, device=device)
        encoder.bias.data = encoder.bias.data.to(dtype=dtype, device=device)
        self.optimizers[sae_index] = torch.optim.Adam(encoder.parameters(), lr=self.config["learning_rate"])

    def check_reinit_condition(self, encoded_activations: List[torch.Tensor]) -> Tuple[List[bool], List[float]]:
        reinit_flags = []
        feature_activity = []
        for activations in encoded_activations:
            activation_rates = (activations != 0).float().mean(dim=0)
            target_rates = torch.full_like(activation_rates, 0.0625)
            relative_diff = torch.abs(activation_rates - target_rates) / (target_rates + 1e-6)
            sensitive_diff = torch.pow(relative_diff, 3).mean()
            feature_activity.append(sensitive_diff.item())
            reinit_flags.append(sensitive_diff.item() > self.reinit_threshold)
        return reinit_flags, feature_activity

    def train(self, train_loader: DataLoader, num_epochs: int = 1):
        for epoch in range(num_epochs):
            print(f"设备: {self.device}")
            for batch_num, X_batch_tuple in enumerate(train_loader):
                # 获取X_batch
                X_batch = X_batch_tuple[0]
                self.current_step += 1
                X_batch = X_batch.to(self.device)

                # 根据设备类型选择是否使用autocast
                if self.use_amp:
                    # 使用混合精度训练
                    with torch.autocast(device_type=self.device.type if hasattr(self.device, 'type') else 'cpu'):
                        outputs, activations, indices = self.base_model.forward_with_encoded(X_batch)
                        outputs = [output.to(self.device) for output in outputs]
                        encoder_weights = [encoder.weight.t() for encoder in self.base_model.encoders]
                        consensus_loss = self.calculate_consensus_loss(encoder_weights)
                        reconstruction_losses = [self.criterion(output, X_batch) for output in outputs]
                else:
                    # 标准计算
                    outputs, activations, indices = self.base_model.forward_with_encoded(X_batch)
                    outputs = [output.to(self.device) for output in outputs]
                    encoder_weights = [encoder.weight.t() for encoder in self.base_model.encoders]
                    consensus_loss = self.calculate_consensus_loss(encoder_weights)
                    reconstruction_losses = [self.criterion(output, X_batch) for output in outputs]

                reinit_flags, feature_activity = self.check_reinit_condition(activations)
                for i, flag in enumerate(reinit_flags):
                    if flag:
                        print(f"重新初始化 SAE {i} 权重，因为重初始化条件已触发。")
                        self.reinitialize_sae_weights(i)

                if any(reinit_flags):
                    outputs, activations, indices = self.base_model.forward_with_encoded(X_batch)
                    outputs = [output.to(self.device) for output in outputs]
                    reconstruction_losses = [self.criterion(output, X_batch) for output in outputs]

                total_losses = [rec_loss + consensus_loss for rec_loss in reconstruction_losses]

                # 每100个批次打印一次损失以避免输出过多
                if batch_num % 100 == 0:
                    print(f"批次 {batch_num}, 一致性损失: {consensus_loss.item():.6f}, 重构损失: {reconstruction_losses[0].item():.6f}")

                for i, (optimizer, total_loss) in enumerate(zip(self.optimizers, total_losses)):
                    optimizer.zero_grad()
                    
                    if self.use_amp and self.scalers[i] is not None and torch.cuda.is_available():
                        # 使用AMP和梯度缩放 (CUDA模式)
                        self.scalers[i].scale(total_loss).backward(retain_graph=(i < len(self.optimizers) - 1))
                        self.scalers[i].step(optimizer)
                        self.scalers[i].update()
                    else:
                        # 标准反向传播 (CPU或MPS模式)
                        total_loss.backward(retain_graph=(i < len(self.optimizers) - 1))
                        optimizer.step()

                if self.wandb_on == '1' and batch_num % 10 == 0:  # 每10个批次记录一次
                    wandb.log({
                        "Consensus_loss": consensus_loss.item(),
                        **{f"SAE_{i}_reconstruction_loss": rec_loss.item() for i, rec_loss in enumerate(reconstruction_losses)},
                        **{f"Feature_activity_SAE_{i}": scalar for i, scalar in enumerate(feature_activity)}
                    })

            # 每个epoch结束后打印进度
            print(f"完成轮次 {epoch+1}/{num_epochs}")
