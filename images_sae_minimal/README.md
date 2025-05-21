# 图像稀疏自编码器 (SAE) 精简版系统

这是一个最小化的稀疏自编码器系统，专门用于处理图像数据。该系统使用了多个 SAE 模型实现图像的稀疏表示学习，能够提取图像中的关键特征。

## 文件结构

- `images_sae.py` - 定义了 `ImageSparseAutoencoder` 类，实现了稀疏自编码器的核心功能
- `images_sae_trainer.py` - 定义了 `ImageSAETrainer` 类，实现了模型训练逻辑
- `images_dataset.py` - 提供了图像数据集加载和预处理的工具
- `images_sae_train.py` - 主要训练脚本，用于训练和评估模型
- `wandb_tsinghua_template.py` - Weights & Biases 配置工具，用于实验跟踪和可视化

## 系统依赖

- Python 3.8+
- PyTorch 1.13+
- NumPy
- Matplotlib
- SciPy
- Weights & Biases (wandb)
- Pillow (PIL)
- torchvision

## 使用方法

### 1. 准备数据

将 JPG 格式的图像文件放在上一级目录的 `output_images_jpg_rename` 文件夹中。

### 2. 训练模型

基本使用方法：

```bash
python images_sae_train.py
```

高级选项：

```bash
python images_sae_train.py --device cpu --img_size 64 --batch_size 32 --k_sparse 128
```

参数说明：
- `--device` - 选择训练设备：cpu, mps (Mac GPU) 或 cuda (默认: cpu)
- `--img_size` - 调整图像大小 (默认: 64x64)
- `--batch_size` - 批次大小 (默认: 32)
- `--k_sparse` - 稀疏参数k值 (默认: 128)
- `--analyze_loss` - 分析初始重构损失 (可选标志)

### 3. 查看结果

训练过程和结果会通过 Weights & Biases 记录和可视化。您可以在训练完成后，在 Weights & Biases 网站上查看实验结果。

模型会保存在 `models` 目录下：
- `images_sae_initial.pth` - 初始模型
- `images_sae_epoch_X.pth` - 每个 epoch 的模型
- `images_sae_final.pth` - 最终模型

## 模型架构

该系统实现了一个包含多个稀疏自编码器的集成模型：

1. 每个 SAE 包含：
   - 编码器：将输入映射到高维隐空间
   - 稀疏激活：仅保留 k 个最强的激活
   - 解码器：从稀疏激活重构输入

2. 集成策略：
   - 多个 SAE 并行训练
   - 使用一致性约束确保特征学习的多样性
   - 集成结果提供更稳定和更有解释性的特征表示

## 训练过程

训练过程包括：
1. 加载和预处理图像数据
2. 设置模型参数和优化器
3. 迭代训练多个 epoch
4. 定期评估重构质量
5. 记录训练指标和可视化结果
6. 保存训练好的模型

## 注意事项

- 如果原始图像目录 (`../output_images_jpg_rename`) 不存在，请调整 `images_sae_train.py` 第 313 行的路径
- 确保拥有足够的内存来处理指定大小的图像和批次大小
- 较大的 `--img_size` 和 `--batch_size` 会消耗更多内存
- 增加 `--k_sparse` 值会使表示不太稀疏但可能改善重构质量
