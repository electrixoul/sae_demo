import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
from typing import Tuple, List, Dict, Optional


class NeuralActivityDataset(Dataset):
    """神经活动数据集类"""
    
    def __init__(self, data_dir: str, stimulus_types: Optional[List[str]] = None, 
                 normalize: bool = True, use_all_timepoints: bool = True):
        """
        初始化神经活动数据集
        
        Args:
            data_dir: 数据目录路径
            stimulus_types: 要加载的刺激类型列表，None表示加载所有
            normalize: 是否对数据进行归一化
            use_all_timepoints: 是否使用所有时间点（True）还是每个时间点作为单独样本（False）
        """
        self.data_dir = data_dir
        self.normalize = normalize
        self.use_all_timepoints = use_all_timepoints
        
        # 所有可用的刺激类型
        all_stimulus_types = ['basic_sequence', 'moving_bar', 'expanding_circle', 
                             'checkerboard', 'radial_grating']
        
        if stimulus_types is None:
            self.stimulus_types = all_stimulus_types
        else:
            self.stimulus_types = stimulus_types
            
        # 加载数据
        self.data, self.metadata = self._load_data()
        
        print(f"加载了 {len(self.stimulus_types)} 种刺激类型的数据")
        print(f"总数据点: {len(self.data)} 个")
        print(f"神经元数量: {self.metadata['num_neurons']}")
        
    def _load_data(self) -> Tuple[List[torch.Tensor], Dict]:
        """加载神经响应数据"""
        all_responses = []
        total_timepoints = 0
        num_neurons = None
        
        for stimulus_type in self.stimulus_types:
            file_path = os.path.join(self.data_dir, f"responses_{stimulus_type}.npy")
            
            if not os.path.exists(file_path):
                print(f"警告: 文件 {file_path} 不存在，跳过")
                continue
                
            # 加载数据
            responses = np.load(file_path)  # 形状: (时间帧, 神经元数量)
            
            if num_neurons is None:
                num_neurons = responses.shape[1]
            else:
                assert responses.shape[1] == num_neurons, f"神经元数量不一致: {responses.shape[1]} vs {num_neurons}"
            
            print(f"加载 {stimulus_type}: {responses.shape}")
            
            # 转换为torch tensor
            responses_tensor = torch.from_numpy(responses).float()
            
            if self.use_all_timepoints:
                # 将所有时间点连接为一个长序列
                all_responses.append(responses_tensor)
                total_timepoints += responses.shape[0]
            else:
                # 每个时间点作为单独的样本
                for t in range(responses.shape[0]):
                    all_responses.append(responses_tensor[t:t+1])  # 保持2D形状
                    total_timepoints += 1
        
        if self.use_all_timepoints:
            # 连接所有数据
            concatenated_data = torch.cat(all_responses, dim=0)  # (总时间帧, 神经元数量)
            data_list = [concatenated_data[i] for i in range(concatenated_data.shape[0])]
        else:
            data_list = all_responses
            
        # 归一化处理
        if self.normalize and len(data_list) > 0:
            # 计算全局统计量
            all_data = torch.stack(data_list)  # (样本数, 神经元数)
            self.data_mean = all_data.mean()
            self.data_std = all_data.std()
            
            print(f"数据统计: 均值={self.data_mean:.4f}, 标准差={self.data_std:.4f}")
            
            # 归一化到0-1范围，保持神经活动的稀疏性特征
            data_min = all_data.min()
            data_max = all_data.max()
            self.data_min = data_min
            self.data_max = data_max
            
            print(f"数据范围: {data_min:.4f} - {data_max:.4f}")
            
            # 使用min-max归一化到[0, 1]
            data_list = [(item - data_min) / (data_max - data_min + 1e-8) for item in data_list]
            
        metadata = {
            'num_neurons': num_neurons,
            'total_timepoints': total_timepoints,
            'stimulus_types': self.stimulus_types,
            'normalized': self.normalize
        }
        
        if self.normalize:
            metadata.update({
                'data_mean': self.data_mean.item() if hasattr(self, 'data_mean') else None,
                'data_std': self.data_std.item() if hasattr(self, 'data_std') else None,
                'data_min': self.data_min.item() if hasattr(self, 'data_min') else None,
                'data_max': self.data_max.item() if hasattr(self, 'data_max') else None,
            })
            
        return data_list, metadata
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor]:
        """
        获取单个数据点
        
        Returns:
            Tuple containing (neural_response,) to match SAE trainer expected format
        """
        return (self.data[idx],)
    
    def get_sample_for_analysis(self, n_samples: int = 10) -> torch.Tensor:
        """获取用于分析的样本数据"""
        indices = torch.randperm(len(self.data))[:n_samples]
        samples = torch.stack([self.data[i] for i in indices])
        return samples
    
    def denormalize(self, normalized_data: torch.Tensor) -> torch.Tensor:
        """反归一化数据"""
        if not self.normalize:
            return normalized_data
            
        return normalized_data * (self.data_max - self.data_min) + self.data_min


def create_neural_dataloaders(data_dir: str, 
                             stimulus_types: Optional[List[str]] = None,
                             batch_size: int = 64,
                             train_split: float = 0.8,
                             normalize: bool = True,
                             **loader_kwargs) -> Tuple[DataLoader, DataLoader, Dict]:
    """
    创建神经数据的训练和测试数据加载器
    
    Args:
        data_dir: 数据目录
        stimulus_types: 刺激类型列表
        batch_size: 批次大小
        train_split: 训练集比例
        normalize: 是否归一化
        **loader_kwargs: DataLoader的其他参数
        
    Returns:
        (train_loader, test_loader, metadata)
    """
    # 创建完整数据集
    full_dataset = NeuralActivityDataset(
        data_dir=data_dir,
        stimulus_types=stimulus_types,
        normalize=normalize,
        use_all_timepoints=True
    )
    
    # 分割训练和测试集
    total_size = len(full_dataset)
    train_size = int(total_size * train_split)
    test_size = total_size - train_size
    
    train_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        **loader_kwargs
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs
    )
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    
    return train_loader, test_loader, full_dataset.metadata


if __name__ == "__main__":
    # 测试数据加载
    data_dir = "neural_activity_data_20250606_163121"
    
    if os.path.exists(data_dir):
        # 测试单一刺激类型
        dataset = NeuralActivityDataset(
            data_dir=data_dir,
            stimulus_types=['expanding_circle'],
            normalize=True
        )
        
        print(f"数据集大小: {len(dataset)}")
        print(f"样本形状: {dataset[0][0].shape}")
        
        # 测试数据加载器
        train_loader, test_loader, metadata = create_neural_dataloaders(
            data_dir=data_dir,
            stimulus_types=['expanding_circle', 'moving_bar'],
            batch_size=32
        )
        
        # 测试一个批次
        for batch in train_loader:
            neural_batch = batch[0]
            print(f"批次形状: {neural_batch.shape}")
            print(f"批次数据范围: {neural_batch.min():.4f} - {neural_batch.max():.4f}")
            break
            
    else:
        print(f"数据目录 {data_dir} 不存在")
