import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

class JPGImageDataset(Dataset):
    def __init__(self, image_dir, transform=None, resize_dim=64):
        """
        加载JPG图像数据集，并将图像转换成适合SAE处理的格式
        
        参数:
            image_dir (str): 包含JPG图像的目录路径
            transform (callable, optional): 可选的额外转换
            resize_dim (int): 调整图像大小的目标尺寸
        """
        self.image_dir = image_dir
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
        
        # 基础转换：调整大小并转换为张量
        base_transform = transforms.Compose([
            transforms.Resize((resize_dim, resize_dim)),  # 调整为更小的尺寸
            transforms.ToTensor(),  # 转换为张量并归一化到[0,1]
        ])
        
        # 组合自定义转换（如果有的话）
        if transform:
            self.transform = transforms.Compose([base_transform, transform])
        else:
            self.transform = base_transform
        
        # 存储图像尺寸信息
        with Image.open(os.path.join(image_dir, self.image_files[0])) as img:
            self.original_size = img.size
        self.resize_dim = resize_dim
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # 加载图像
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        
        # 应用转换
        if self.transform:
            image = self.transform(image)
        
        # 展平图像为一维向量 (3, resize_dim, resize_dim) -> (3*resize_dim*resize_dim,)
        flattened_image = image.reshape(-1)
        
        # 返回格式为((数据,), 标签)，这里我们没有标签，但保持接口一致
        return (flattened_image,), idx  # 使用索引作为伪标签

def create_image_dataloaders(image_dir, resize_dim=64, train_ratio=0.8, batch_size=64, **loader_kwargs):
    """
    创建训练集和测试集的数据加载器
    
    参数:
        image_dir (str): 图像目录
        resize_dim (int): 调整图像尺寸
        train_ratio (float): 训练集比例
        batch_size (int): 批次大小
        **loader_kwargs: DataLoader的其他参数
    
    返回:
        train_loader, test_loader: 训练和测试数据加载器
    """
    # 创建完整数据集
    dataset = JPGImageDataset(image_dir, resize_dim=resize_dim)
    
    # 计算训练集和测试集大小
    dataset_size = len(dataset)
    train_size = int(train_ratio * dataset_size)
    test_size = dataset_size - train_size
    
    # 随机拆分数据集
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size], 
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
    
    # 创建符合SAE trainer接口的训练数据加载器包装器
    class DataLoaderWrapper:
        def __init__(self, dataloader):
            self.dataloader = dataloader
            
        def __iter__(self):
            for batch in self.dataloader:
                # 只保留第一个元素(X_batch,)，丢弃伪标签
                yield batch[0]
                
        def __len__(self):
            return len(self.dataloader)
    
    # 使用包装器适配训练数据加载器
    wrapped_train_loader = DataLoaderWrapper(train_loader)
    
    return wrapped_train_loader, test_loader, dataset.resize_dim
