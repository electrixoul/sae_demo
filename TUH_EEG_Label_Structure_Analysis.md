# TUH EEG数据集标签结构分析

## 一、《annotation_guidelines_v39.pdf》文件总结

该文档详细描述了 Temple 大学医院 EEG 语料库 (TUH EEG) 的标注指南，重点介绍了标注格式、标签类型和标注约定。

### 主要内容

1. **标签体系**：
   - 文档定义了共27种标签，涵盖癫痫发作、非发作信号、伪影等
   - 主要癫痫发作标签包括：非特异性局灶性发作(FNSZ)、全身性发作(GNSZ)、简单部分性发作(SPSZ)、复杂部分性发作(CPSZ)、失神发作(ABSZ)等
   - 主要伪影标签包括：肌肉伪影(MUSC)、颤抖(SHIV)、咀嚼(CHEW)、眨眼(EYBL)、眼动(EYEM)等

2. **标注流程**：
   - 描述了由专业团队进行多轮审核的标注过程
   - 标注先由一位标注员完成，然后由至少两位其他标注员审核
   - 对于难以判断的情况，采用团队会议讨论达成共识

3. **标注约定**：
   - 癫痫发作必须至少持续10秒，间隔不超过3秒
   - 描述了识别不同类型事件的具体特征
   - 详细说明了如何区分癫痫发作与正常生理现象（如睡眠纺锤波）

## 二、TUH EEG数据集标签文件结构

TUH EEG使用两种主要文件格式存储标注信息：

### 1. 时间同步事件文件（.tse）

这是一种简单的基于时间的标注格式：

```
version=tse_v1.0.0
0.0000 10.2775 bckg 1.0000
10.2775 35.7775 gnsz 1.0000
35.7775 102.2525 bckg 1.0000
```

**结构特点**：
- 第一行定义版本
- 后续每行包含四个字段：起始时间、结束时间、标签、概率
- 简单扁平的结构，适合整体信号分类
- 所有通道共享同一标注

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
- 包含蒙太奇定义部分，描述电极位置和通道组成
- 定义层级和子级数量，支持分层标注
- 符号块(symbols)定义标签索引映射
- 标签块(label)包含特定格式：层级,子级,开始时间,结束时间,通道索引,[各标签概率向量]
- 支持特定通道的事件标注
- 支持层次化标注（可以为同一事件提供不同级别的描述）

### 3. 双类别变体

每种格式都有双类别(bi-class)变体（.tse_bi 和 .lbl_bi），仅区分"发作"和"非发作"，不具体指明发作类型。

## 三、将标签数据载入Python的方法

基于对文件格式的理解，可以采用以下方法在Python中加载标签数据：

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

### 3. 推荐在实际应用中的改进

1. **使用专业库**：考虑使用TUH EEG项目可能提供的官方工具，如他们的`nedc_pystream`

2. **错误处理**：添加更强健的错误处理和格式验证

3. **映射符号**：在LBL文件中，将数字索引映射到实际标签名称

4. **集成数据流**：设计用于批处理多个文件的数据流管道

5. **与EEG信号同步**：开发函数将标签与相应的EEG信号段关联

由于标签文件格式相对简单且基于文本，Python的标准库足以处理解析。对于更复杂的应用场景，可以考虑使用pandas等库来组织和操作标签数据，特别是当需要与EEG信号数据结合分析时。
