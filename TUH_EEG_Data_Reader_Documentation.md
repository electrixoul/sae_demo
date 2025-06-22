# TUH EEG数据读取程序文档

## 程序概述

本文档描述了用于读取和分析TUH EEG数据集的程序实现。程序包含两个主要工具：`eeg_data_viewer.py`和`eeg_data_viewer_simple.py`。

## 数据集信息

### TUH EEG数据集结构
- **数据来源**: Temple University Hospital EEG数据库v2.0.0
- **文件数量**: 517个标注的EEG文件
- **数据格式**: EDF格式信号文件 + REC格式标注文件
- **采样率**: 250 Hz
- **通道数**: 22个标准EEG通道(TCP montage)

### 标注类型
- `spsw` (1): spike and slow wave
- `gped` (2): generalized periodic epileptiform discharge  
- `pled` (3): periodic lateralized epileptiform discharge
- `eyem` (4): eye movement
- `artf` (5): artifact
- `bckg` (6): background

### 通道配置 (TCP montage)
```
通道0-21分别对应:
FP1-F7, F7-T3, T3-T5, T5-O1, FP2-F8, F8-T4, T4-T6, T6-O2,
A1-T3, T3-C3, C3-CZ, CZ-C4, C4-T4, T4-A2, FP1-F3, F3-C3,
C3-P3, P3-O1, FP2-F4, F4-C4, C4-P4, P4-O2
```

## 程序功能

### eeg_data_viewer.py
完整的EEG数据可视化工具，包含以下功能：
- 随机选择数据集中的EEG文件
- 加载EDF格式的信号数据
- 解析REC格式的标注文件
- 生成带标注的多通道EEG信号图

### eeg_data_viewer_simple.py  
命令行版本的数据展示工具，提供：
- 信号统计信息输出
- 标注详细信息汇总
- 信号数据样本展示
- 文本格式的分析报告

## 依赖库
- `pyedflib`: 读取EDF文件
- `numpy`: 数值计算
- `pandas`: 数据处理
- `matplotlib`: 图形绘制(仅完整版)

## 使用方法

### 安装依赖
```bash
conda activate mod
pip install pyedflib
```

### 运行程序
```bash
# 完整版可视化工具
python eeg_data_viewer.py

# 简化版文本工具
python eeg_data_viewer_simple.py
```

## 输出示例

程序运行时会输出：
- 找到的EEG文件总数
- 随机选择的文件路径
- 信号基本信息(通道数、采样点数、时长等)
- 标注统计(各类型标注的数量和分布)
- 通道级别的标注分布
- 具体的标注事件列表

## 技术实现

### 文件查找
遍历`eeg_data/eeg_data_lable`目录下的train和eval子目录，匹配.edf和对应的.rec文件。

### 数据加载
使用pyedflib库读取EDF文件，提取信号数据、通道标签和采样率信息。

### 标注解析
解析REC文件中的逗号分隔格式：`通道,开始时间,结束时间,标签代码`

### 可视化
生成多子图显示不同通道的信号，叠加彩色标注区域表示不同事件类型。

## 数据样本
程序运行结果显示数据集包含不同类型的EEG信号和标注，时长从几分钟到20多分钟不等。标注分布不均匀，背景标注占多数，其他事件类型数量较少。
