import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple
import tempfile
import os


class NeuralSparseAutoencoder(nn.Module):
    """专门用于神经活动数据的稀疏自编码器"""
    
    def __init__(self, hyperparameters: Dict[str, Any]):
        super().__init__()
        self.config: Dict[str, Any] = hyperparameters
        self.encoders: nn.ModuleList = nn.ModuleList([
            nn.Linear(self.config["input_size"], self.config["hidden_size"])
            for i in range(self.config.get("num_saes", 1))
        ])
        self.apply(self._init_weights)
        base_k_sparse = self.config["k_sparse"]
        self.k_sparse_values = [base_k_sparse for i in range(self.config.get("num_saes", 1))]

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            # 针对神经数据的权重初始化
            device_str = str(next(self.parameters()).device) if len(list(self.parameters())) > 0 else "cpu"
            if device_str.startswith('mps'):
                # MPS友好的初始化方式
                std = 1.0 / (m.weight.size(1) ** 0.5)
                nn.init.normal_(m.weight, std=std)
            else:
                # 使用Xavier初始化，适合神经数据的动态范围
                nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward_with_encoded(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        x = x.to(next(self.parameters()).device)
        results = [self._process_layer(encoder, x, i) for i, encoder in enumerate(self.encoders)]
        return [r[0] for r in results], [r[1] for r in results], [r[2] for r in results]

    def _process_layer(self, encoder: nn.Linear, x: torch.Tensor, encoder_idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 编码
        encoded_pre = encoder(x)
        # K稀疏激活
        encoded, indices = self._topk_activation(encoded_pre, encoder_idx)
        # 解码（使用权重绑定）
        normalized_weights: torch.Tensor = F.normalize(encoder.weight, p=2, dim=1)
        decoded: torch.Tensor = F.linear(encoded, normalized_weights.t())
        return decoded, encoded, indices

    def _topk_activation(self, x: torch.Tensor, encoder_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        k: int = self.k_sparse_values[encoder_idx]
        top_values, indices = torch.topk(x, k, dim=1)
        # 创建稀疏激活：只保留前k个最大值
        # 避免in-place操作，创建新的tensor
        threshold = top_values[:, -1:].expand_as(x)
        mask = x >= threshold
        sparse_activation = x.clone() * mask
        return sparse_activation, indices

    def save_model_local(self, run_name: str):
        """本地保存模型"""
        save_path = f"{run_name}.pth"
        torch.save(self.state_dict(), save_path)
        print(f"模型保存到: {save_path}")
        return save_path

    @classmethod
    def load_from_local(cls, model_path: str, hyperparameters: Dict[str, Any], device="cpu"):
        """从本地加载模型"""
        model = cls(hyperparameters)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        return model
