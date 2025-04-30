import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from typing import List, Optional, Dict, Any, Tuple
import itertools
import wandb
import math
from utils.general_utils import calculate_MMCS

class MNISTSAETrainer:
    def __init__(self, model: nn.Module, device: str, hyperparameters: Dict[str, Any], true_features: Optional[torch.Tensor] = None, wandb_on: str = '0'):
        self.model = model
        self.device = device
        self.config = hyperparameters
        self.base_model = model
        self.optimizers = [torch.optim.Adam(encoder.parameters(), lr=self.config["learning_rate"]) for encoder in self.base_model.encoders]
        self.criterion = nn.MSELoss()
        self.true_features = true_features.to(device) if true_features is not None else None
        
        # 根据设备自动配置AMP（混合精度训练）和梯度缩放
        self.use_amp = torch.cuda.is_available() and str(device).startswith('cuda')
        if self.use_amp:
            self.scalers = [torch.cuda.amp.GradScaler() for _ in self.base_model.encoders]
        else:
            self.scalers = [None for _ in self.base_model.encoders]
            
        self.ensemble_consistency_weight = hyperparameters.get("ensemble_consistency_weight", 1)
        # 使用非常高的重初始化阈值，基本上禁用重初始化功能
        self.reinit_threshold = hyperparameters.get("reinit_threshold", 100.0)
        self.warmup_steps = hyperparameters.get("warmup_steps", 100)
        self.current_step = 0
        self.wandb_on = wandb_on

    def get_warmup_factor(self) -> float:
        if self.current_step >= self.warmup_steps:
            return 1.0
        return 0.5 * (1 + math.cos(math.pi * (self.warmup_steps - self.current_step) / self.warmup_steps))

    def save_true_features(self):
        if self.wandb_on == '1':
            artifact = wandb.Artifact(f"{wandb.run.name}_true_features", type="true_features")
            with artifact.new_file("true_features.pt", mode="wb") as f:
                torch.save(self.true_features.cpu(), f)
            wandb.log_artifact(artifact)

    def save_model(self, epoch: int):
        if self.wandb_on == '1':
            run_name = f"{wandb.run.name}_epoch_{epoch}"
            self.base_model.save_model(run_name, alias=f"epoch_{epoch}")

    def calculate_consensus_loss(self, encoder_weights: List[torch.Tensor]) -> torch.Tensor:
        pairs = itertools.combinations(encoder_weights, 2)
        mmcs_values = [1 - calculate_MMCS(a.clone(), b.clone(), self.device)[0] for a, b in pairs]
        unweighted_loss = torch.mean(torch.stack(mmcs_values)) if mmcs_values else torch.tensor(0.0, device=self.device)
        warmup_factor = self.get_warmup_factor()
        return unweighted_loss * self.ensemble_consistency_weight * warmup_factor

    def reinitialize_sae_weights(self, sae_index: int):
        encoder = self.base_model.encoders[sae_index]
        dtype, device = encoder.weight.dtype, encoder.weight.device
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
            relative_diff = torch.abs(activation_rates - target_rates) / target_rates
            sensitive_diff = torch.pow(relative_diff, 3).mean()
            feature_activity.append(sensitive_diff.item())
            reinit_flags.append(sensitive_diff.item() > self.reinit_threshold)
        return reinit_flags, feature_activity

    def train(self, train_loader: DataLoader, num_epochs: int = 1):
        for epoch in range(num_epochs):
            print("self.device: ", self.device)
            for batch_num, X_batch_tuple in enumerate(train_loader):
                # 获取X_batch
                X_batch = X_batch_tuple[0]
                self.current_step += 1
                X_batch = X_batch.to(self.device)

                # 根据设备类型选择是否使用autocast
                if self.use_amp:
                    # 使用CUDA的混合精度训练
                    with torch.cuda.amp.autocast():
                        outputs, activations, indices = self.base_model.forward_with_encoded(X_batch)
                        outputs = [output.to(self.device) for output in outputs]
                        encoder_weights = [encoder.weight.t() for encoder in self.base_model.encoders]
                        consensus_loss = self.calculate_consensus_loss(encoder_weights)
                        reconstruction_losses = [self.criterion(output, X_batch) for output in outputs]
                else:
                    # 标准CPU计算
                    outputs, activations, indices = self.base_model.forward_with_encoded(X_batch)
                    outputs = [output.to(self.device) for output in outputs]
                    encoder_weights = [encoder.weight.t() for encoder in self.base_model.encoders]
                    consensus_loss = self.calculate_consensus_loss(encoder_weights)
                    reconstruction_losses = [self.criterion(output, X_batch) for output in outputs]

                reinit_flags, feature_activity = self.check_reinit_condition(activations)
                for i, flag in enumerate(reinit_flags):
                    if flag:
                        print(f"Reinitializing SAE {i} weights due to reinit condition.")
                        self.reinitialize_sae_weights(i)

                if any(reinit_flags):
                    outputs, activations, indices = self.base_model.forward_with_encoded(X_batch)
                    outputs = [output.to(self.device) for output in outputs]
                    reconstruction_losses = [self.criterion(output, X_batch) for output in outputs]

                total_losses = [rec_loss + consensus_loss for rec_loss in reconstruction_losses]

                print("consensus_loss: ", consensus_loss)

                for i, (optimizer, total_loss) in enumerate(zip(self.optimizers, total_losses)):
                    optimizer.zero_grad()
                    
                    if self.use_amp and self.scalers[i] is not None:
                        # 使用AMP和梯度缩放 (CUDA模式)
                        self.scalers[i].scale(total_loss).backward(retain_graph=(i < len(self.optimizers) - 1))
                        self.scalers[i].step(optimizer)
                        self.scalers[i].update()
                    else:
                        # 标准反向传播 (CPU模式)
                        total_loss.backward(retain_graph=(i < len(self.optimizers) - 1))
                        optimizer.step()

                if self.true_features is not None:
                    with torch.no_grad():
                        mmcs = [calculate_MMCS(encoder.weight.t(), self.true_features, self.device)[0] for encoder in self.base_model.encoders]

                if self.wandb_on == '1' and batch_num % 5 == 0:  # 只记录每5个批次
                    wandb.log({
                        "Consensus_loss": consensus_loss,
                        **{f"SAE_{i}_reconstruction_loss": rec_loss.item() for i, rec_loss in enumerate(reconstruction_losses)},
                        **{f"Feature_activity_SAE_{i}": scalar for i, scalar in enumerate(feature_activity)},
                        **(({f"MMCS_SAE_{i}": mmcs_i for i, mmcs_i in enumerate(mmcs)}) if self.true_features is not None else {})
                    })

            # 每个epoch结束后保存模型
            model_path = f"mnist_sae_models/mnist_sae_epoch_{epoch+1}.pth"
            torch.save(self.base_model.state_dict(), model_path)
            print(f"模型保存到: {model_path}")
