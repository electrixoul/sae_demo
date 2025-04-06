# TUH EEG数据集标签结构分析

## 一、《annotation_guidelines_v39.pdf》文件总结

该文档详细描述了 Temple 大学医院 EEG 语料库 (TUH EEG) 的标注指南，重点介绍了标注格式、标签类型和标注约定。

### 主要内容

1. **标签体系**：
   - 文档定义了共27种标签，涵盖癫痫发作、非发作信号、伪影等
   - 主要癫痫发作标签包括：非特异性局灶性发作(FNSZ)、全身性发作(GNSZ)、简单部分性发作(SPSZ)、复杂部分性发作(CPSZ)、失神发作(ABSZ)等
   - 主要伪影标签包括：肌肉伪影(MUSC)、颤抖(SHIV)、咀嚼(CHEW)、眨眼(EYBL)、眼动(EYEM)等
   - **TUH EEG Events Corpus (TUEV)**事件库中特别关注六大主要类别：
     - **尖波/锐波**（SPSW, spike and sharp wave）
     - **广泛性周期痫样放电**（GPED, generalized periodic epileptiform discharges）
     - **局灶性周期痫样放电**（PLED, periodic lateralized epileptiform discharges）
     - **眼动**（EYEM, eye movement artifact）
     - **伪迹**（ARTF, artifact）
     - **背景**（BCKG, background）- 用作"捕获全部"类别，表示明确不属于其他五类的信号

2. **标注流程**：
   - 描述了由专业团队进行多轮审核的标注过程
   - 标注先由一位标注员完成，然后由至少两位其他标注员审核
   - 对于难以判断的情况，采用团队会议讨论达成共识

3. **标注约定**：
   - 癫痫发作必须至少持续10秒，间隔不超过3秒
   - 描述了识别不同类型事件的具体特征
   - 详细说明了如何区分癫痫发作与正常生理现象（如睡眠纺锤波）

## 二、TUH EEG数据集标签文件结构

TUH EEG使用多种文件格式存储标注信息：

### 1. 时间同步事件文件（.tse）

这是一种简单的基于时间的标注格式：

```
version=tse_v1.0.0
0.0000 10.2775 bckg 1.0000
10.2775 35.7775 gnsz 1.0000
35.7775 102.2525 bckg 1.0000
```

**结构特点**：
- 第一行定义版本（如 `version = tse_v1.0.0`）
- 后续每行包含四个字段：起始时间、结束时间、标签、概率
- 简单扁平的结构，适合整体信号分类
- 所有通道共享同一标注
- 时间精度通常保留四位小数（秒），确保足够精确地定位采样点
- 概率字段在人工标注中通常为 1.0000，表示确定的标签
- 对于机器生成的标注，概率字段可以包含实际的后验概率值
- 通常人工标注会将整个记录按时间连续划分，使每一时刻都归属某一类别（包括背景）

### 2. 标签文件（.lbl）

这是一种更复杂的分层标注格式：

```
version=lbl_v1.0.0
montage=0,FP1-F7:EEGFP1-REF--EEGF7-REF
montage=1,F7-T3:EEGF7-REF--EEGT3-REF
...
number_of_levels=1
level[0]=1
symbols[0]={0:'(null)',1:'spsw',2:'gped',...}
label={0,0,0.0000,10.2775,0,[0.0,0.0,0.0,0.0,0.0,0.0,1.0,...]}
label={0,0,10.2775,35.7775,0,[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,...]}
```

**结构特点**：
- 文件开头同样以版本声明开始（如 `version = lbl_v1.0.0`）
- **导联蒙太奇定义块（montage block）**：列出每个导联的编号和名称对应关系
  - 例如：`montage = 0, FP1-F7: EEG FP1-REF -- EEG F7-REF`
  - 这表示导联0对应名称"FP1-F7"，由原始信号"EEG FP1-REF"减去"EEG F7-REF"形成
- **层级信息块**：指定总层级数和各层级的子层级个数
  - 例如：`number_of_levels = 1` 和 `level[0] = 1`
  - 在TUEV事件库中，标注主要用单层级结构，没有进一步的子分类
- **符号表块（symbol block）**：定义标签索引与文本标签的映射列表
  - 例如：`symbols[0] = {0:'(null)',1:'spsw',2:'gped',3:'pled',4:'eyem',5:'artf',6:'bckg',...}`
  - 这表示在层级0中，索引0表示"(null)"空标签，1表示"spsw"，2表示"gped"等
- **标签记录块（label block）**：列出具体的事件标注条目
  - 格式为：`label = {层级, 子层级, 开始时间(秒), 结束时间(秒), 导联编号, [概率向量]}`
  - 例如：`label = {0, 0, 10.2775, 35.7775, 0, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, ...]}`
  - 概率向量是一位热（one-hot）表示，其长度对应符号表中定义的标签总数
  - 在人工标注情况下，向量中只有对应事件类别的索引位置为1.0，其余为0
- 支持特定通道的事件标注，明确指出事件发生在哪个导联上
- 支持层次化标注（可以为同一事件提供不同级别的描述）

### 3. 事件标注文件（.lab）

根据readme.txt文件，这是一种每10微秒提供一个标签的详细标注格式：

```
117100000 117200000 eyem
```

**结构特点**：
- 每行包含三个字段：开始时间（10微秒为单位）、结束时间（10微秒为单位）以及标签
- 使用与.tse类似的标签缩写，但更为详细
- 示例中表示从1171秒到1172秒为眼动（eyem）事件

### 4. 记录文件（.rec）

这是一种基于秒级时间单位的标注格式：

```
13,90.4,91.4,6
```

**结构特点**：
- 每行包含四个字段：通道编号、开始时间（秒）、结束时间（秒）、标签编号
- 标签使用数字编码而非文字缩写：
  - 1: spsw（尖波和慢波）
  - 2: gped（广泛性周期痫样放电）
  - 3: pled（周期性侧向痫样放电）
  - 4: eyem（眼动）
  - 5: artf（伪迹）
  - 6: bckg（背景）
- 示例中表示在通道13上，从90.4秒到91.4秒是背景（bckg）信号

### 5. 双类别变体

每种格式都有双类别(bi-class)变体（.tse_bi 和 .lbl_bi），仅区分"发作"和"非发作"，不具体指明发作类型。这些变体文件简化了标注信息，主要用于二分类任务。

## 三、通道配置和TCP蒙太奇

TUH EEG事件语料库使用标准的ACNS TCP蒙太奇，这是查看癫痫数据的首选方式。蒙太奇定义如下：

```
montage =  0, FP1-F7: EEG FP1-REF --  EEG F7-REF
montage =  1, F7-T3:  EEG F7-REF  --  EEG T3-REF
montage =  2, T3-T5:  EEG T3-REF  --  EEG T5-REF
montage =  3, T5-O1:  EEG T5-REF  --  EEG O1-REF
montage =  4, FP2-F8: EEG FP2-REF --  EEG F8-REF
montage =  5, F8-T4 : EEG F8-REF  --  EEG T4-REF
montage =  6, T4-T6:  EEG T4-REF  --  EEG T6-REF
montage =  7, T6-O2:  EEG T6-REF  --  EEG O2-REF
montage =  8, A1-T3:  EEG A1-REF  --  EEG T3-REF
montage =  9, T3-C3:  EEG T3-REF  --  EEG C3-REF
montage = 10, C3-CZ:  EEG C3-REF  --  EEG CZ-REF
montage = 11, CZ-C4:  EEG CZ-REF  --  EEG C4-REF
montage = 12, C4-T4:  EEG C4-REF  --  EEG T4-REF
montage = 13, T4-A2:  EEG T4-REF  --  EEG A2-REF
montage = 14, FP1-F3: EEG FP1-REF --  EEG F3-REF
montage = 15, F3-C3:  EEG F3-REF  --  EEG C3-REF
montage = 16, C3-P3:  EEG C3-REF  --  EEG P3-REF
montage = 17, P3-O1:  EEG P3-REF  --  EEG O1-REF
montage = 18, FP2-F4: EEG FP2-REF --  EEG F4-REF
montage = 19, F4-C4:  EEG F4-REF  --  EEG C4-REF
montage = 20, C4-P4:  EEG C4-REF  --  EEG P4-REF
montage = 21, P4-O2:  EEG P4-REF  --  EEG O2-REF
```

在.rec和.lab文件中的通道编号直接对应上述TCP蒙太奇定义中的编号。例如，通道1表示F7与T3之间的差异，代表EDF文件中的(F7-REF)-(T3-REF)。

## 四、文件命名与组织结构

### 1. 评估集文件命名

评估集EEG文件的典型路径解析：
```
./edf/eval/032/bckg_032_a_.edf
```

- `edf`：包含EDF文件的目录
- `eval`：表示这是评估集的一部分（与训练集区分）
- `032`：区分每个评估集会话的随机索引
- `bckg_032_a_.edf`：实际的EEG文件
  - `bckg`：表示该文件包含背景标注
  - `032`：对评估索引的引用
  - `a_.edf`：EEG文件被分割成一系列以a_.edf, a_1.edf等为名的文件，表示修剪后的EEG段，原始记录中不感兴趣的部分被删除

### 2. 训练集文件命名

训练集EEG文件的典型路径解析：
```
./edf/train/00002275/00002275_00000001.edf
```

- `edf`：包含EDF文件的目录
- `train`：表示这是训练集的一部分
- `00002275`：索引，可交叉引用到TUH EEG语料库v0.6.1
- `00002275_00000001.edf`：实际的EDF文件
  - `00002275`：对训练索引的引用
  - `00000001`：表示这是与该患者相关的第一个文件

### 3. 数据集统计信息

**评估集**：
- 文件总数：159
- 包含spsw（尖波和慢波）的文件：9
- 包含gped（广泛性周期痫样放电）的文件：28
- 包含pled（周期性侧向痫样放电）的文件：33
- 包含artf（伪迹）的文件：46
- 包含eyem（眼动）的文件：35
- 包含bckg（背景）的文件：89

**训练集**：
- 文件总数：359
- 包含spsw（尖波和慢波）的文件：27
- 包含gped（广泛性周期痫样放电）的文件：51
- 包含pled（周期性侧向痫样放电）的文件：48
- 包含artf（伪迹）的文件：164
- 包含eyem（眼动）的文件：46
- 包含bckg（背景）的文件：211

## 五、标签与原始EDF数据的对齐方式

- TUEV的标注时间基准与原始EEG信号**严格对齐**，采用**相对于记录开始的秒数**来标记事件位置
- 标签中的时间戳（开始和结束时间）可以直接与EDF文件中的时间轴对应
- 标注文件通常保留**四位小数**的时间精度，以确保达到1毫秒或更精细的分辨率
- 这种精度足以唯一定位到EDF中的采样点，无论采样率是250Hz（0.004秒/样本）还是1000Hz（0.001秒/样本）
- 通过已知的采样频率，可将标注时间戳换算为对应的采样点索引，实现与原始信号的**逐样本精确对齐**

## 六、将标签数据载入Python的方法

由于TUH EEG标签数据使用多种格式，下面提供了解析不同格式文件的Python示例：

### 1. TSE文件解析
```python
def read_tse_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # 获取版本信息
    version = lines[0].strip().split('=')[1]
    
    events = []
    # 解析事件行
    for line in lines[1:]:
        if line.strip():
            start, end, label, prob = line.strip().split()
            events.append({
                'start': float(start),
                'end': float(end),
                'label': label,
                'probability': float(prob)
            })
    
    return {'version': version, 'events': events}
```

### 2. LBL文件解析
```python
def read_lbl_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    result = {'montage': {}, 'levels': {}, 'symbols': {}, 'labels': []}
    current_section = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # 解析版本信息
        if line.startswith('version='):
            result['version'] = line.split('=')[1]
        
        # 解析蒙太奇定义
        elif line.startswith('montage='):
            parts = line[len('montage='):].split(',', 1)
            channel_idx = int(parts[0])
            channel_def = parts[1]
            result['montage'][channel_idx] = channel_def
        
        # 解析层级数量
        elif line.startswith('number_of_levels='):
            result['number_of_levels'] = int(line.split('=')[1])
            
        # 解析各层级的子级数量
        elif line.startswith('level['):
            level_idx = int(line[6:line.find(']')])
            level_val = int(line.split('=')[1])
            result['levels'][level_idx] = level_val
            
        # 解析符号定义
        elif line.startswith('symbols['):
            level_idx = int(line[8:line.find(']')])
            symbols_str = line.split('=')[1]
            # 这里需要更复杂的解析来处理符号字典
            # 可以使用eval或ast.literal_eval更安全地将字符串转换为字典
            result['symbols'][level_idx] = symbols_str
            
        # 解析标签行
        elif line.startswith('label='):
            # 提取花括号内容
            label_content = line[line.find('{')+1:line.rfind('}')]
            parts = label_content.split(',', 5)
            
            level = int(parts[0])
            sublevel = int(parts[1])
            start_time = float(parts[2])
            end_time = float(parts[3])
            channel = int(parts[4])
            
            # 提取概率向量
            prob_vector_str = parts[5].strip()[1:-1]  # 移除方括号
            prob_vector = [float(p) for p in prob_vector_str.split(',')]
            
            result['labels'].append({
                'level': level,
                'sublevel': sublevel,
                'start': start_time,
                'end': end_time,
                'channel': channel,
                'probabilities': prob_vector
            })
    
    return result
```

### 3. LAB文件解析
```python
def read_lab_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    events = []
    for line in lines:
        if line.strip():
            parts = line.strip().split()
            if len(parts) == 3:
                start_time_micro = int(parts[0])
                end_time_micro = int(parts[1])
                label = parts[2]
                
                events.append({
                    'start_micro': start_time_micro,
                    'end_micro': end_time_micro,
                    'start_sec': start_time_micro / 100000.0,  # 转换为秒
                    'end_sec': end_time_micro / 100000.0,      # 转换为秒
                    'label': label
                })
    
    return events
```

### 4. REC文件解析
```python
def read_rec_file(file_path):
    # 标签映射
    label_map = {
        '1': 'spsw',  # 尖波和慢波
        '2': 'gped',  # 广泛性周期痫样放电
        '3': 'pled',  # 周期性侧向痫样放电
        '4': 'eyem',  # 眼动
        '5': 'artf',  # 伪迹
        '6': 'bckg'   # 背景
    }
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    events = []
    for line in lines:
        if line.strip():
            parts = line.strip().split(',')
            if len(parts) == 4:
                channel = int(parts[0])
                start_time = float(parts[1])
                end_time = float(parts[2])
                label_code = parts[3]
                
                events.append({
                    'channel': channel,
                    'start': start_time,
                    'end': end_time,
                    'label_code': label_code,
                    'label': label_map.get(label_code, 'unknown')
                })
    
    return events
```

### 5. 与EEG信号同步和实际应用建议

1. **标签与信号对齐**：
   - 使用标签中的时间戳和EDF的采样率将时间点转换为采样点索引
   - 例如，对于250Hz的信号，时间戳10.2775秒对应的采样点索引为`int(10.2775 * 250) = 2569`

2. **使用专业库**：
   - 考虑使用TUH EEG项目提供的官方工具，如NEDC项目的`nedc_ann_tools`或`nedc_pystream`
   - 官方工具通常有更完善的错误处理和格式兼容性

3. **数据流管理**：
   - 设计用于批处理多个文件的数据流管道
   - 从EDF和标签文件自动提取带标签的数据片段，用于训练机器学习模型

4. **标签映射增强**：
   - 创建标签索引到标签名称的映射函数
   - 提供标签的描述性信息，辅助数据分析和结果解释

5. **健壮性处理**：
   - 添加异常处理以应对文件格式变化或损坏
   - 验证时间戳的一致性和连续性

6. **数据集结构理解**：
   - 区分训练集和评估集的组织方式
   - 了解文件命名结构以便于批量处理
   - 考虑数据集中标签分布的不平衡性
