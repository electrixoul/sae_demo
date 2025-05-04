import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from scipy.stats import pearsonr
from PIL import Image
import glob
import random
from models.sae import SparseAutoencoder
import time
import matplotlib as mpl
import matplotlib.font_manager as fm

# 配置中文字体支持
plt.rcParams['font.sans-serif'] = ['AR PL UKai CN', 'AR PL UMing CN', 'DejaVu Sans', 'WenQuanYi Zen Hei', 'WenQuanYi Micro Hei', 'SimHei', 'Microsoft YaHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# 确保中文标题能正确显示的函数
def safe_title(zh_title, en_title=None):
    """尝试使用中文标题，如果失败则使用英文标题"""
    try:
        # 测试中文显示
        fig = plt.figure(figsize=(1, 1))
        plt.title(zh_title)
        plt.close(fig)
        return zh_title
    except:
        if en_title:
            return en_title
        else:
            # 如果没有英文替代，将中文转换为拼音或直接返回
            return "Title"

# 打印当前系统可用的中文字体
def print_available_chinese_fonts():
    fonts = [f.name for f in fm.fontManager.ttflist if any(word in f.name.lower() for word in 
             ['han', 'hei', 'kai', 'song', 'ming', 'yuan', 'gothic', 'china', 'chinese', 'cn', 'zh'])]
    print("可用的中文字体:")
    for font in fonts:
        print(f"  - {font}")
    return fonts

class MiceDataset(Dataset):
    def __init__(self, data_dir, target_size=(64, 64), transform=None, augment=True, augment_factor=5):
        """
        加载Mice数据集，预处理并可选地进行数据增强
        
        参数:
            data_dir (str): 数据目录路径
            target_size (tuple): 目标图像大小，默认为64x64像素
            transform (torchvision.transforms): 转换函数
            augment (bool): 是否进行数据增强
            augment_factor (int): 每个原始图像生成的增强图像数量
        """
        self.data_dir = data_dir
        self.target_size = target_size
        self.augment = augment
        self.augment_factor = augment_factor
        self.images = []  # 初始化图像列表
        self.labels = []  # 存储分类标签
        self.paths = []   # 存储原始图像路径
        
        # 查找所有PNG图像
        self.image_paths = []
        for category in ['1', '2']:
            folder_path = os.path.join(data_dir, category)
            if os.path.exists(folder_path):
                self.image_paths.extend(glob.glob(os.path.join(folder_path, '*.png')))
        
        # 指定默认转换
        if transform is None:
            self.transform = transforms.Compose([
                self.center_crop_custom,  # 自定义裁剪函数
                transforms.Resize(target_size),
                transforms.Grayscale(),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transform
            
        # 为数据增强创建额外的转换
        self.augment_transforms = [
            transforms.Compose([
                self.center_crop_custom,  # 先应用自定义裁剪
                transforms.RandomRotation(15),
                transforms.RandomResizedCrop(target_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
                transforms.Grayscale(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
            ]),
            transforms.Compose([
                self.center_crop_custom,  # 先应用自定义裁剪
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.Resize(target_size),
                transforms.Grayscale(),
                transforms.ToTensor(),
            ]),
            transforms.Compose([
                self.center_crop_custom,  # 先应用自定义裁剪
                transforms.RandomVerticalFlip(p=1.0),
                transforms.Resize(target_size),
                transforms.Grayscale(),
                transforms.ToTensor(),
            ]),
            transforms.Compose([
                self.center_crop_custom,  # 先应用自定义裁剪
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.Resize(target_size),
                transforms.Grayscale(),
                transforms.ToTensor(),
            ]),
            transforms.Compose([
                self.center_crop_custom,  # 先应用自定义裁剪
                transforms.RandomPerspective(distortion_scale=0.2, p=1.0),
                transforms.Resize(target_size),
                transforms.Grayscale(),
                transforms.ToTensor(),
            ])
        ]
    
    def center_crop_custom(self, img):
        """
        自定义裁剪函数，处理可能的600x200尺寸图像
        - 如果图像是非方形的，从中心裁剪出最大可能的正方形
        - 特别处理600x200的图像，裁剪中心200x200区域
        """
        width, height = img.size
        
        # 处理特殊情况: 约600x200的图像
        if abs(width/height - 3) < 0.5:  # 宽高比约为3:1
            new_size = min(height, 200)  # 取高度或200中的较小值
            left = (width - new_size) // 2
            top = (height - new_size) // 2
            right = left + new_size
            bottom = top + new_size
            return img.crop((left, top, right, bottom))
        
        # 一般情况: 从中心裁剪最大的正方形
        if width > height:
            left = (width - height) // 2
            top = 0
            right = left + height
            bottom = height
        else:
            left = 0
            top = (height - width) // 2
            right = width
            bottom = top + width
            
        return img.crop((left, top, right, bottom))
        
    def process_augmentation(self):
        """处理数据增强并创建数据集"""
        self.images = []
        self.labels = []  # 存储分类标签 (1 或 2)
        self.paths = []   # 存储原始图像路径
        
        print(f"处理 {len(self.image_paths)} 个原始图像...")
        
        for path in self.image_paths:
            # 确定标签（基于文件夹名称）
            label = int(os.path.basename(os.path.dirname(path)))
            
            try:
                # 加载并转换原始图像
                img = Image.open(path).convert('RGB')
                transformed_img = self.transform(img)
                self.images.append(transformed_img)
                self.labels.append(label)
                self.paths.append(path)
                
                # 数据增强
                if self.augment:
                    # 对每张图像应用多种增强方法
                    for _ in range(self.augment_factor):
                        # 随机选择一种增强方法
                        aug_transform = random.choice(self.augment_transforms)
                        aug_img = aug_transform(img)
                        self.images.append(aug_img)
                        self.labels.append(label)
                        self.paths.append(path + f"_aug_{_}")
            except Exception as e:
                print(f"处理图像 {path} 时出错: {e}")
        
        print(f"数据集创建完成，共 {len(self.images)} 张图像")

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        """返回图像数据和标签"""
        image = self.images[idx]
        
        # 确保图像已经展平为一维向量
        flattened_image = image.view(-1)
        
        return (flattened_image,), self.labels[idx]

class MiceSAETrainer:
    def __init__(self, config, device):
        """
        初始化Mice SAE训练器
        
        参数:
            config (dict): 配置参数
            device (torch.device): 训练设备
        """
        self.config = config
        self.device = device
        self.model = None
        self.optimizers = None
        self.epoch = 0
        self.train_losses = []
        self.recon_losses = []
        self.consistency_losses = []
        self.correlations = []
        
        # 创建保存目录
        os.makedirs(config['model_save_dir'], exist_ok=True)
        os.makedirs(config['visualization_dir'], exist_ok=True)
        
        # 初始化模型
        self._init_model()
        
    def _init_model(self):
        """初始化SAE模型和优化器"""
        # 创建模型
        self.model = SparseAutoencoder(self.config).to(self.device)
        
        # 为每个SAE创建一个优化器
        self.optimizers = []
        for i in range(self.config['num_saes']):
            # 注意：SparseAutoencoder使用绑定权重模式，只需要优化编码器参数
            optimizer = optim.Adam([
                {'params': self.model.encoders[i].parameters()}
            ], lr=self.config['learning_rate'])
            self.optimizers.append(optimizer)
    
    def train(self, train_loader, num_epochs):
        """
        训练SAE模型
        
        参数:
            train_loader (DataLoader): 训练数据加载器
            num_epochs (int): 训练周期数
        """
        print(f"开始训练Mice SAE (设备: {self.device})...")
        
        for epoch in range(num_epochs):
            self.epoch = epoch + 1
            epoch_losses = []
            epoch_recon_losses = []
            epoch_consistency_losses = []
            epoch_correlations = []
            
            # 训练一个周期
            self.model.train()
            for batch_idx, ((X_batch,), _) in enumerate(train_loader):
                X_batch = X_batch.to(self.device)
                
                # 重置优化器梯度
                for optimizer in self.optimizers:
                    optimizer.zero_grad()
                
                # 前向传播
                outputs, activations, indices = self.model.forward_with_encoded(X_batch)
                
                # 对每个SAE计算损失并更新
                for i, optimizer in enumerate(self.optimizers):
                    # 计算重建损失
                    recon_loss = nn.MSELoss()(outputs[i], X_batch)
                    
                    # 计算一致性损失
                    consistency_loss = 0.0
                    if self.config['consistency_lambda'] > 0:
                        for j in range(self.config['num_saes']):
                            if i != j:
                                consistency_loss += nn.MSELoss()(activations[i], activations[j])
                        consistency_loss /= max(1, self.config['num_saes'] - 1)
                    
                    # 计算总损失
                    total_loss = recon_loss + self.config['consistency_lambda'] * consistency_loss
                    
                    # 反向传播和优化
                    total_loss.backward(retain_graph=(i < len(self.optimizers) - 1))
                    optimizer.step()
                    
                    # 记录损失
                    epoch_losses.append(total_loss.item())
                    epoch_recon_losses.append(recon_loss.item())
                    epoch_consistency_losses.append(consistency_loss.item())  # 使用.item()获取标量值，避免梯度计算问题
                    
                    # 计算相关系数
                    with torch.no_grad():
                        for j in range(min(4, len(X_batch))):  # 只对批次中的前几个样本计算
                            orig = X_batch[j].cpu().numpy()
                            recon = outputs[i][j].cpu().numpy()
                            corr, _ = pearsonr(orig, recon)
                            epoch_correlations.append(corr)
                
                # 每10个批次显示进度
                if batch_idx % 10 == 0:
                    print(f"Epoch {self.epoch} [{batch_idx}/{len(train_loader)}] "
                          f"Loss: {np.mean(epoch_losses[-10:]):.4f}, "
                          f"Corr: {np.mean(epoch_correlations[-40:]):.4f}")
            
            # 保存模型和可视化
            if epoch % self.config['save_interval'] == 0 or epoch == num_epochs - 1:
                self.save_model(f"mice_sae_epoch_{self.epoch}.pth")
                self.visualize_reconstructions(train_loader)
                self.visualize_features()
            
            # 更新统计
            self.train_losses.append(np.mean(epoch_losses))
            self.recon_losses.append(np.mean(epoch_recon_losses))
            self.consistency_losses.append(np.mean(epoch_consistency_losses))
            self.correlations.append(np.mean(epoch_correlations))
            
            print(f"Epoch {self.epoch} 完成: Avg Loss={self.train_losses[-1]:.4f}, "
                  f"Avg Corr={self.correlations[-1]:.4f}")
        
        print("训练完成!")
        
        # 绘制训练曲线
        self.plot_training_curves()
        
        return self.model
    
    def save_model(self, filename):
        """保存模型到指定文件"""
        save_path = os.path.join(self.config['model_save_dir'], filename)
        torch.save(self.model.state_dict(), save_path)
        print(f"模型已保存到 {save_path}")
    
    def visualize_reconstructions(self, data_loader, num_samples=5):
        """可视化原始图像和重建图像"""
        self.model.eval()
        
        # 获取一批数据用于可视化
        all_images = []
        all_labels = []
        all_reconstructions = []
        all_correlations = []
        
        with torch.no_grad():
            for (X_batch,), labels in data_loader:
                if len(all_images) >= num_samples:
                    break
                    
                X_batch = X_batch.to(self.device)
                outputs, activations, indices = self.model.forward_with_encoded(X_batch)
                
                # 对于每个SAE，获取重建
                for i, output in enumerate(outputs):
                    reconstructed = output.cpu().numpy()
                    original = X_batch.cpu().numpy()
                    
                    for j in range(len(X_batch)):
                        if len(all_images) >= num_samples:
                            break
                            
                        # 计算相关系数
                        corr, _ = pearsonr(original[j], reconstructed[j])
                        
                        all_images.append(original[j])
                        all_reconstructions.append(reconstructed[j])
                        all_labels.append(labels[j].item())
                        all_correlations.append(corr)
                    
                    # 只使用第一个SAE的重建
                    break
        
        # 创建可视化
        fig, axes = plt.subplots(num_samples, 2, figsize=(10, 2 * num_samples + 1))
        fig.suptitle(safe_title('Mice图像与SAE重建对比', 'Mice Images vs SAE Reconstruction'), fontsize=16)
        
        for i in range(num_samples):
            # 原始图像
            img_shape = (self.config['image_size'], self.config['image_size'])
            axes[i, 0].imshow(all_images[i].reshape(*img_shape), cmap='gray')
            axes[i, 0].set_title(safe_title(f'原始 (类别: {all_labels[i]})', f'Original (Class: {all_labels[i]})'))
            axes[i, 0].axis('off')
            
            # 重建图像
            axes[i, 1].imshow(all_reconstructions[i].reshape(*img_shape), cmap='gray')
            axes[i, 1].set_title(safe_title(f'重建 (相关性: {all_correlations[i]:.4f})', 
                                         f'Reconstructed (Corr: {all_correlations[i]:.4f})'))
            axes[i, 1].axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(self.config['visualization_dir'], f'mice_reconstructions_epoch_{self.epoch}.png')
        plt.savefig(save_path)
        plt.close(fig)
        print(f"重建可视化已保存到 {save_path}")
    
    def visualize_features(self, n_features=100):
        """可视化SAE学习到的特征"""
        self.model.eval()
        
        # 从模型获取编码器权重
        with torch.no_grad():
            weights = self.model.encoders[0].weight.data.cpu().numpy()
        
        # 选择前n_features个特征来可视化
        weights = weights[:n_features]
        
        # 归一化权重以便可视化
        weights_min = weights.min()
        weights_max = weights.max()
        weights_normalized = (weights - weights_min) / (weights_max - weights_min)
        
        # 创建图表
        grid_size = int(np.ceil(np.sqrt(n_features)))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
        axes = axes.flatten()
        
        # 绘制每个特征
        img_shape = (self.config['image_size'], self.config['image_size'])
        for i in range(n_features):
            feature = weights_normalized[i].reshape(*img_shape)
            axes[i].imshow(feature, cmap='viridis')
            axes[i].axis('off')
        
        # 处理多余的子图
        for i in range(n_features, grid_size * grid_size):
            axes[i].axis('off')
        
        plt.suptitle(safe_title('Mice SAE学习到的特征', 'Features Learned by Mice SAE'), fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        
        # 保存图像
        save_path = os.path.join(self.config['visualization_dir'], f'mice_sae_features_epoch_{self.epoch}.png')
        plt.savefig(save_path)
        plt.close(fig)
        print(f"特征可视化已保存到 {save_path}")
    
    def plot_training_curves(self):
        """绘制训练过程中的损失和相关系数曲线"""
        epochs = list(range(1, self.epoch + 1))
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        
        # 绘制损失曲线
        ax1.plot(epochs, self.train_losses, 'b-', label=safe_title('总损失', 'Total Loss'))
        ax1.plot(epochs, self.recon_losses, 'g--', label=safe_title('重建损失', 'Reconstruction Loss'))
        if self.config['consistency_lambda'] > 0:
            scaled_consistency = [c * self.config['consistency_lambda'] for c in self.consistency_losses]
            ax1.plot(epochs, scaled_consistency, 'r-.', label=safe_title('一致性损失', 'Consistency Loss'))
        
        ax1.set_xlabel(safe_title('周期', 'Epoch'))
        ax1.set_ylabel(safe_title('损失', 'Loss'))
        ax1.set_title(safe_title('训练损失曲线', 'Training Loss'))
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # 绘制相关系数曲线
        ax2.plot(epochs, self.correlations, 'm-', label=safe_title('Pearson相关系数', 'Pearson Correlation'))
        ax2.set_xlabel(safe_title('周期', 'Epoch'))
        ax2.set_ylabel(safe_title('相关系数', 'Correlation'))
        ax2.set_title(safe_title('重建质量', 'Reconstruction Quality'))
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        save_path = os.path.join(self.config['visualization_dir'], 'mice_training_curves.png')
        plt.savefig(save_path)
        plt.close(fig)
        print(f"训练曲线已保存到 {save_path}")

def main():
    # 设置随机种子确保结果可复现
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # 打印可用的中文字体
    print_available_chinese_fonts()
    
    # 设置设备 - 优先使用GPU，加速训练
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 特定尺寸参数（更适合Mice图像的尺寸，原始图像为600x600）
    image_size = 64  # 使用64x64而不是28x28
    input_size = image_size * image_size  # 4096
    
    # 计算隐藏层大小，按照MNIST的比例(784->1024, 约1.3倍)
    # 4096 * 1.3 ≈ 5325，向上取整到2的幂次方为8192
    hidden_size = 8192
    
    # 配置参数
    config = {
        "input_size": input_size,    # 64x64=4096
        "image_size": image_size,    # 图像大小，用于重塑向量为图像
        "hidden_size": hidden_size,  # 隐藏层大小(8192)，比输入层大
        "k_sparse": 100,             # 稀疏参数k(保持约1%的稀疏度，与MNIST相似)
        "num_saes": 5,               # SAE数量
        "learning_rate": 0.001,      # 学习率
        "consistency_lambda": 0.1,   # 一致性损失权重
        "save_interval": 1,          # 保存模型间隔（每N个周期）
        "model_save_dir": "mice_sae_models",  # 模型保存目录
        "visualization_dir": "visualizations/mice",  # 可视化保存目录
    }
    
    # 创建保存目录
    os.makedirs(config['model_save_dir'], exist_ok=True)
    os.makedirs(config['visualization_dir'], exist_ok=True)
    
    # 准备Mice数据集
    data_dir = "data/Mice"
    target_size = (image_size, image_size)  # 64x64
    
    mice_dataset = MiceDataset(
        data_dir=data_dir, 
        target_size=target_size,
        augment=True,
        augment_factor=5
    )
    
    # 处理数据增强（必须调用来初始化图像列表）
    mice_dataset.process_augmentation()
    
    # 创建数据加载器
    batch_size = 16  # 小批量大小，适合小数据集
    train_loader = DataLoader(
        mice_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )
    
    # 创建SAE训练器
    trainer = MiceSAETrainer(config, device)
    
    # 训练模型
    num_epochs = 20  # 训练周期数
    trained_model = trainer.train(train_loader, num_epochs)
    
    print(f"\n训练完成，所有模型和可视化结果保存在:\n"
          f"- 模型: {config['model_save_dir']}\n"
          f"- 可视化: {config['visualization_dir']}")

if __name__ == "__main__":
    main()
