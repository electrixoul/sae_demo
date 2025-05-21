import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple
import wandb
import tempfile
import os

class ImageSparseAutoencoder(nn.Module):
    def __init__(self, hyperparameters: Dict[str, Any]):
        super().__init__()
        self.config: Dict[str, Any] = hyperparameters
        self.encoders: nn.ModuleList = nn.ModuleList([
            nn.Linear(self.config["input_size"], self.config["hidden_size"] * (1))
            for i in range(self.config.get("num_saes", 1))
        ])
        self.apply(self._init_weights)
        base_k_sparse = self.config["k_sparse"]
        self.k_sparse_values = [base_k_sparse * (1) for i in range(self.config.get("num_saes", 1))]

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            # 检查设备类型
            device_str = str(next(self.parameters()).device)
            if device_str.startswith('mps'):
                # MPS友好的初始化方式 - 避免使用orthogonal_
                std = 1.0 / (m.weight.size(1) ** 0.5)
                nn.init.normal_(m.weight, std=std)
            else:
                # 其他设备使用正交初始化
                nn.init.orthogonal_(m.weight)
            nn.init.zeros_(m.bias)

    def forward_with_encoded(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        x = x.to(next(self.parameters()).device)
        results = [self._process_layer(encoder, x, i) for i, encoder in enumerate(self.encoders)]
        return [r[0] for r in results], [r[1] for r in results], [r[2] for r in results]

    def _process_layer(self, encoder: nn.Linear, x: torch.Tensor, encoder_idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded, indices = self._topk_activation(encoder(x), encoder_idx)
        normalized_weights: torch.Tensor = F.normalize(encoder.weight, p=2, dim=1)
        decoded: torch.Tensor = F.linear(encoded, normalized_weights.t())
        return decoded, encoded, indices

    def _topk_activation(self, x: torch.Tensor, encoder_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        k: int = self.k_sparse_values[encoder_idx]
        top_values, indices = torch.topk(x, k, dim=1)
        return x * (x >= top_values[:, -1:]), indices

    def save_model(self, run_name: str, alias: str="latest"):
        artifact = wandb.Artifact(run_name, type='model')
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pth') as tmp_file:
            torch.save(self.state_dict(), tmp_file.name)
            artifact.add_file(tmp_file.name, f'{run_name}.pth')
        wandb.log_artifact(artifact, aliases=[alias])
        os.remove(tmp_file.name)

    def save_model_local(self, run_name: str, alias: str="latest"):
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pth') as tmp_file:
            torch.save(self.state_dict(), tmp_file.name)
            print(f"Model saved to {tmp_file.name}")
        os.rename(tmp_file.name, f"{run_name}.pth")

    @classmethod
    def load_from_pretrained(cls, artifact_path: str, hyperparameters, device="cpu"):
        with wandb.init() as run:
            artifact = run.use_artifact(artifact_path, type='model')
            artifact_dir = artifact.download()
            model_path = os.path.join(artifact_dir, f"{artifact_path.split(':')[-1]}.pth")
            model = cls(hyperparameters)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            return model
