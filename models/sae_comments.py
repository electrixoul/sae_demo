import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple
import wandb
import tempfile
import os

# 定义一个稀疏自编码器类
class SparseAutoencoder(nn.Module):
    def __init__(self, hyperparameters: Dict[str, Any]):
        super().__init__()
        self.config: Dict[str, Any] = hyperparameters
        
        # 创建包含多个编码器层的列表，每个编码器为线性变换
        self.encoders: nn.ModuleList = nn.ModuleList([
            nn.Linear(self.config["input_size"], self.config["hidden_size"] * 1)
            for i in range(self.config.get("num_saes", 1))
        ])
        
        # 初始化编码器权重
        self.apply(self._init_weights)

        # 设置TopK激活函数的稀疏参数k
        base_k_sparse = self.config["k_sparse"]
        self.k_sparse_values = [base_k_sparse * 1 for i in range(self.config.get("num_saes", 1))]

    # 初始化权重为正交，偏置为零
    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight)
            nn.init.zeros_(m.bias)

    # 前向传播，同时返回解码和编码的输出
    def forward_with_encoded(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        x = x.to(next(self.parameters()).device)  # 将输入移动到模型所在的设备上
        
        # 将输入通过每个编码器处理
        results = [self._process_layer(encoder, x, i) for i, encoder in enumerate(self.encoders)]
        
        # 分别返回解码和编码的输出
        return [r[0] for r in results], [r[1] for r in results]

    # 通过单个编码器层处理输入
    def _process_layer(self, encoder: nn.Linear, x: torch.Tensor, encoder_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # 对编码输出应用TopK激活函数
        encoded: torch.Tensor = self._topk_activation(encoder(x), encoder_idx)
        
        # 对编码器权重进行归一化，用于解码
        normalized_weights: torch.Tensor = F.normalize(encoder.weight, p=2, dim=1)
        
        # 解码编码后的表示
        decoded: torch.Tensor = F.linear(encoded, normalized_weights.t())
        
        return decoded, encoded

    # 应用TopK激活函数，仅保留前k个激活
    def _topk_activation(self, x: torch.Tensor, encoder_idx: int) -> torch.Tensor:
        k: int = self.k_sparse_values[encoder_idx]  # 获取该编码器的稀疏参数k
        
        # 计算TopK值及其阈值
        top_values, _ = torch.topk(x, k, dim=1)
        
        # 将低于阈值的激活置为零
        return x * (x >= top_values[:, -1:])

    # 保存模型到文件，集成Weights and Biases（wandb）
    def save_model(self, run_name: str, alias: str = "latest"):
        artifact = wandb.Artifact(run_name, type='model')
        
        # 临时保存模型状态
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pth') as tmp_file:
            torch.save(self.state_dict(), tmp_file.name)
            artifact.add_file(tmp_file.name, f'{run_name}.pth')
        
        # 将模型Artifact记录到wandb
        wandb.log_artifact(artifact, aliases=[alias])
        
        # 删除临时文件
        os.remove(tmp_file.name)

    # 从wandb加载预训练模型
    @classmethod
    def load_from_pretrained(cls, artifact_path: str, hyperparameters, device="cpu"):
        with wandb.init() as run:
            # 下载模型Artifact
            artifact = run.use_artifact(artifact_path, type='model')
            artifact_dir = artifact.download()
            
            # 从下载的Artifact中加载模型状态
            model_path = os.path.join(artifact_dir, f"{artifact_path.split(':')[-1]}.pth")
            model = cls(hyperparameters)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            return model
