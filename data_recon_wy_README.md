# data/data recon-wy 目录文档

此文档提供对`data/data recon-wy`目录的内容、结构和用途的详细分析。

## 目录概述

该目录包含一个神经科学实验数据集，主要研究大脑（特别是视觉皮层）对动物类（animate）和物体类（inanimate）视觉刺激的差异性神经反应。

## 文件结构

```
data/data recon-wy/
├── 20240916_J27_AI_animate_P2sISI6s_420trial_120902.slim    # 实验配置文件
├── animate-obj-grey-big_╞┴─╗│╩╧╓/                          # 刺激图像集1
│   ├── 0000.png
│   ├── 0024.png
│   ├── ...
│   └── 1200.png
├── animate-obj-grey-big_┤╠╝ñ─┌╚▌/                          # 刺激图像集2
│   ├── 0000.png
│   ├── 0024.png
│   ├── ...
│   └── 1200.png
├── resp_neurons.csv                                         # 响应性神经元ID列表
├── RR analysis_basic visual ckeck 1-2-ROI copy.ipynb        # 数据分析Jupyter笔记本
├── trace_with_label.mat                                     # 神经记录数据（含标签）
├── V1_neurons.csv                                           # V1区域神经元ID列表
└── wholebrain_output.mat                                    # 全脑输出数据
```

## 数据文件详解

### MATLAB数据文件

1. **trace_with_label.mat** (~323MB)
   - 包含神经记录轨迹与刺激标签信息
   - 主要数据结构:
     - `whole_trace_ori`: 原始神经活动轨迹（时间序列数据）
     - `start_edge`: 刺激呈现时间点
     - `stim_kinds`: 实验中使用的刺激类型
     - `stim_indexes`: 将刺激呈现映射到神经记录的索引

2. **wholebrain_output.mat** (~404MB)
   - 包含更大规模的神经记录数据，可能是全脑活动的处理结果
   - 使用MATLAB 7.3格式存储

### 刺激图像

两个包含相同内容但名称编码不同的文件夹:
- `animate-obj-grey-big_╞┴─╗│╩╧╓/`
- `animate-obj-grey-big_┤╠╝ñ─┌╚▌/`

每个文件夹包含20个PNG图像（600x200像素），分为两类:
- **动物类刺激** (前10张): 0000.png, 0024.png, 0048.png, 0072.png, 0096.png, 0120.png, 0144.png, 0168.png, 0192.png, 0216.png
- **物体类刺激** (后10张): 0984.png, 1008.png, 1032.png, 1056.png, 1080.png, 1104.png, 1128.png, 1152.png, 1176.png, 1200.png

### 神经元参考文件

1. **resp_neurons.csv**
   - 列出了对视觉刺激有显著反应的神经元ID
   - 包含单列数据: `neuron_id`

2. **V1_neurons.csv**
   - 列出了初级视觉皮层(V1)中的神经元ID
   - 包含单列数据: `neuron_id`

### 实验配置文件

**20240916_J27_AI_animate_P2sISI6s_420trial_120902.slim**
```
[Project]
ProjectName = 20240916_J27_AI_animate_P2sISI6s_420trial_120902
Objective = 20
Exposure = 4.000000
ImageNum = 1750
Nshift = 3
Nx = 468
Ny = 348
Nnum = 15
CenterX = 7132
CenterY = 5337
GroupMode = 1
Interval = 0.000000
Duration = 84.000000
Lambda = 405
```

关键参数说明:
- 项目名称: 20240916_J27_AI_animate_P2sISI6s_420trial_120902
- 总图像数: 1750
- 刺激持续时间: 84秒
- 其他成像和记录参数

### 分析笔记本

**RR analysis_basic visual ckeck 1-2-ROI copy.ipynb**

该Jupyter笔记本包含用于分析神经响应的代码，主要功能包括:

1. 数据加载和预处理
2. 刺激类型定义和分类
3. 响应神经元的筛选
4. 刺激相关神经活动的提取和分析

## 代码分析与关键功能

### 刺激类型定义

```python
# 刺激种类定义
stim_kinds = ['animate-obj\\0000', 'animate-obj\\0024', ..., 'animate-obj\\1200']

# 分类为动物和物体刺激
animate_stim = ['animate-obj\\0000', 'animate-obj\\0024', ..., 'animate-obj\\0216']
object_stim = ['animate-obj\\0984', 'animate-obj\\1008', ..., 'animate-obj\\1200']
```

### 数据加载处理

```python
# 加载MATLAB数据
data = scipy.io.loadmat(os.path.join(res_folder, 'trace_with_label.mat'))

# 提取关键数据结构
whole_trace_ori = data['trace_with_label']['whole_trace_ori'][0, 0]
start_edge = data['trace_with_label']['start_edge'][0, 0].T
start_edge = start_edge.reshape(-1).tolist()

# 处理刺激索引
stim_indexes = data['trace_with_label']['stim_indexes'][0, 0].T
stim_indexes = [stim_indexes[i][0][0].flatten() for i in range(len(stim_kinds))]
stim_indexes = np.array(stim_indexes)  # reshape to (20stim, 20repeat)

# 重组数据结构
trace_with_label = {
    'whole_trace_ori': whole_trace_ori,
    'start_edge': start_edge,
    'stim_indexes': stim_indexes
}
```

### 神经活动提取函数

```python
# 提取与刺激时间锁定的神经活动
def get_stimTrace_original(trace_with_label, stim_kind_i, idx_neuron, edge_before=8, edge_after=32):
    # 提取数据
    whole_trace_ori = trace_with_label['whole_trace_ori']
    start_edge = trace_with_label['start_edge']
    stim_indexes = trace_with_label['stim_indexes']

    # 获取特定刺激类型的索引
    stimIndexes = stim_indexes[stim_kind_i,:]
    stimTrace = []

    # 遍历刺激索引，提取神经活动时间窗
    for j in range(len(stimIndexes)):
        start_edge_time = start_edge[stimIndexes[j]-1]
        start_edge_time = start_edge_time -1  # 从MATLAB索引转为Python索引
        start_time = start_edge_time - edge_before
        end_time = start_edge_time + edge_after - 1

        # 提取特定神经元在刺激前后的活动
        currentTrace = whole_trace_ori[idx_neuron-1, start_time:end_time+1]
        stimTrace.append(currentTrace)

    stimTrace = np.array(stimTrace)
    return stimTrace
```

## 实验详细信息

基于文件名和配置信息分析:

- **实验编号**: 20240916_J27_AI_animate_P2sISI6s_420trial_120902
- **实验类型**: 视觉刺激实验，比较动物类与物体类刺激
- **刺激参数**: 
  - P2s: 2秒刺激呈现时间
  - ISI6s: 6秒刺激间隔
  - 420trial: 总共420次试验
- **刺激类型**:
  - 10种动物类刺激
  - 10种物体类刺激
- **重点分析区域**: 
  - 初级视觉皮层(V1)
  - 可能包含其他响应区域

## 研究目的与应用

该数据集的主要研究目的:

1. **类别选择性**: 研究视觉皮层神经元对动物类vs物体类视觉刺激的类别选择性反应
2. **时间动态**: 分析神经反应的时间特征和动态变化
3. **神经编码**: 探索视觉信息在大脑中的编码方式
4. **区域特异性**: 比较不同脑区（特别是V1）对类别刺激的响应差异

## 项目背景

根据笔记本中的文件路径引用（如`D:\00 researches\mouse\res_0919`），本数据集可能是一个更大规模小鼠视觉神经科学研究项目的一部分，专注于视觉处理和物体识别机制的研究。

---

*注: 本文档基于对目录内容的分析编写，如有更新或更详细的信息，请参考原始数据文件和相关实验记录。*
