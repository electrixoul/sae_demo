---

# TUH EEG Event Corpus (版本 2.0.0) 标签数据结构及载入思路报告

## 1. 简介

TUH EEG Event Corpus 是 TUH EEG Corpus 的一个子集，专门收录了已知包含以下事件的 EEG 会话：
- **spsw**：Spike and slow wave（尖波/慢波）  
- **gped**：Generalized periodic epileptiform discharge（广泛性周期性癫痫样放电）  
- **pled**：Periodic lateralized epileptiform discharge（局灶性周期性癫痫样放电）  
- **eyem**：Eye movement（眼动，例如眨眼）  
- **artf**：Artifact（伪迹）  
- **bckg**：Background（背景，非上述事件，即非事件段）

此外，最新版本还修正了 EDF 文件头中曾存在的无效问题，从而保证数据在使用时不会因文件格式错误而产生问题。数据集分为训练集和评估集，二者在样本上互不重叠；训练集可用于开发和调试，而评估集仅供测试使用。

## 2. 文件类型

本次发布的标签数据包含以下六种主要文件类型：

- **\*.edf**  
  存储 EEG 原始信号数据，采用欧洲数据格式（EDF）。这些文件经过修正，确保文件头有效。

- **\*.htk**  
  根据“Improved EEG Event Classification Using Differential Energy”方法提取的特征文件。

- **\*.lab**  
  注释文件，采用高分辨率（每 10 微秒记录一次）的标注格式。  
  **格式说明**：  
  每一行记录一个标注，其格式为：  
  ```
  <开始时间> <结束时间> <标签>
  ```
  其中，开始时间和结束时间均以“10 微秒”为单位的整数表示（例如：117100000 117200000 eyem 表示从 117100000 个 10 微秒到 117200000 个 10 微秒的时间段，标记为 eyem）。  
  使用 4 个字母的代码表示各类事件（见下文）。

- **\*.rec**  
  注释文件，使用秒为单位给出标注。  
  **格式说明**：  
  每一行采用逗号分隔的 4 个字段，依次为：  
  ```
  <通道编号>,<开始时间（秒）>,<结束时间（秒）>,<标签代码>
  ```
  标签代码的映射关系为：  
  - 1: spsw  
  - 2: gped  
  - 3: pled  
  - 4: eyem  
  - 5: artf  
  - 6: bckg  

- 另外还有一些辅助文件（如 .htk）和与 EDF 数据对应的其他说明文件，但本报告重点讨论标注文件（.lab 和 .rec）。

## 3. 标签数据结构

### 3.1. Lab 文件

- **时间单位与精度**：  
  Lab 文件中每行记录的开始和结束时间均以“10 微秒”为基本单位。  
  例如，行“117100000 117200000 eyem”表示的时间段可转换为秒：  
  \[
  \text{开始时间} = 117100000 \times 10^{-5} = 1171.0 \text{秒}  
  \]  
  （注意：10 微秒 = 1×10⁻⁵ 秒）

- **标签**：  
  标签为 4 个字母代码，分别对应：  
  - spsw：尖波/慢波  
  - gped：广泛性周期性癫痫样放电  
  - pled：局灶性周期性癫痫样放电  
  - eyem：眼动  
  - artf：伪迹  
  - bckg：背景（非上述事件）

Lab 文件提供了极高的时间分辨率标注，适用于需要精细定位事件边界的应用场景。

### 3.2. Rec 文件

- **格式**：  
  每一行由 4 个逗号分隔的字段组成：  
  1. **通道编号**：表示当前标注所在的通道，通道编号对应于预定义的 TCP 挂接（见下文的通道与蒙太奇说明）。  
  2. **开始时间（秒）**  
  3. **结束时间（秒）**  
  4. **标签代码**：数字形式，映射关系如下：  
     - 1 → spsw  
     - 2 → gped  
     - 3 → pled  
     - 4 → eyem  
     - 5 → artf  
     - 6 → bckg

- **用途**：  
  Rec 文件给出的时间单位为秒，更适合基于秒级标注进行数据切分与事件检测的应用。

### 3.3. 标签说明

- **明确事件**：spsw、gped、pled、eyem、artf 这 5 个标签均代表实际标注的事件。  
- **背景标签（bckg）**：当某个时间段明确不属于上述任何事件时，使用 bckg 标签，作为一种 catch-all 类别。

## 4. 通道与蒙太奇配置

### 4.1. EEG 频道配置

在 TUH EEG Event Corpus 中，尽管原始临床 EEG 可能存在多种通道配置，但本事件子集均包含来自 10/20 系统的标准通道。此外，所有文件均可转换为 **TCP（Temporal Central Parasagittal，即双香蕉）** 挂接，这也是内部处理时的首选挂接。

### 4.2. 挂接（Montage）定义

在本数据集中，.rec 文件中记录的通道编号对应于预先定义的 ACNS TCP 挂接。挂接的定义如下（共 22 个挂接通道，编号 0 到 21）：

```
montage =  0, FP1-F7: EEG FP1-REF -- EEG F7-REF
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

例如，通道编号 1 表示差分信号：(F7-REF) 与 (T3-REF) 之差。这些挂接配置确保所有注释文件（lab 和 rec）中的通道编号可以映射到实际的 EEG 信号导联，从而保证标注与原始信号严格对齐。

## 5. 文件命名与目录结构

数据集中分为两个主要目录：**train**（训练集）和 **eval**（评估集），它们的结构和命名规则有所不同。

### 5.1. 评估集

- **示例文件路径**：  
  `./edf/eval/032/bckg_032_a_.edf`
  
- **组成部分说明**：  
  - `edf`：存放 EDF 数据文件的根目录  
  - `eval`：表示这是评估集  
  - `032`：随机生成的索引用于区分不同评估会话  
  - 文件名 `bckg_032_a_.edf`：  
    - 前缀 `bckg` 表示该文件主要包含背景（bckg）标注  
    - `032` 与 eval 索引对应  
    - 后缀 `a_.edf` 表示原始 EEG 被分割成多个部分（a_.edf, a_1.edf, …），这是因为原始录制中无关部分已被剪除。

### 5.2. 训练集

- **示例文件路径**：  
  `./edf/train/00002275/00002275_00000001.edf`
  
- **组成部分说明**：  
  - `edf`：EDF 文件根目录  
  - `train`：表示训练集  
  - `00002275`：患者索引，此编号用于交叉参考 TUH EEG Corpus 旧版本（v0.6.1）  
  - 文件名 `00002275_00000001.edf`：  
    - 前半部分 `00002275` 为患者索引  
    - 后半部分 `00000001` 表示该患者对应 EEG 文件的序号（第一部分）。

## 6. 数据集统计信息

从 readme.txt 中提供的统计数据来看：

### 评估集（eval）：
- 文件数：159  
  - 包含 spsw 的文件：9  
  - 包含 gped 的文件：28  
  - 包含 pled 的文件：33  
  - 包含 artf 的文件：46  
  - 包含 eyem 的文件：35  
  - 包含 bckg 的文件：89  

### 训练集（train）：
- 文件数：359  
  - 包含 spsw 的文件：27  
  - 包含 gped 的文件：51  
  - 包含 pled 的文件：48  
  - 包含 artf 的文件：164  
  - 包含 eyem 的文件：46  
  - 包含 bckg 的文件：211  

这些统计信息能帮助研究者了解数据集中各类别事件的分布情况，进而设计平衡策略或评估算法性能。

## 7. Python 中的加载与解析思路

由于 lab 和 rec 文件均为文本格式，解析工作相对简单。以下是解析思路概要：

### 7.1. Lab 文件解析

- **读取方法**：  
  使用 Python 内置的文件 I/O（例如 `open()`）按行读取文件内容。
  
- **解析过程**：  
  1. 对每一行，使用 `split()` 方法分隔出三个部分：  
     - 开始时间（字符串形式的整数，单位为 10 微秒）  
     - 结束时间（同上）  
     - 标签（4 字母代码，如 "eyem"）
  2. 将开始和结束时间转换为整数，再乘以 1e-5 转换为秒。
  3. 将标签直接存储为字符串，或根据需要构造一个映射（例如 { "spsw": "spike and slow wave", … }）。
  
- **用途**：  
  可得到每个事件在高时间分辨率下的开始与结束时刻，有助于精细定位事件边界。

### 7.2. Rec 文件解析

- **读取方法**：  
  同样使用标准文件 I/O 读取文件，并利用逗号 `,` 分隔每一行。
  
- **解析过程**：  
  1. 将每一行按逗号拆分成 4 个字段：  
     - 通道编号（整数）  
     - 开始时间（浮点数，单位：秒）  
     - 结束时间（浮点数，单位：秒）  
     - 标签代码（整数）
  2. 定义一个映射字典：  
     ```python
     label_map = {1: "spsw", 2: "gped", 3: "pled", 4: "eyem", 5: "artf", 6: "bckg"}
     ```
  3. 根据标签代码，将其映射到对应的字符串标签。
  4. 解析后的数据可以存储在列表或 DataFrame 中，便于后续与 EDF 信号数据对齐使用。

- **用途**：  
  解析 rec 文件能获得通道级别的事件标注，结合预定义的 TCP 挂接信息，可以准确地将标注与 EEG 信号对应，为事件检测、分类等任务提供标签。

### 7.3. 与 EEG 信号的对齐

- 根据 EDF 文件头信息，获取采样率及信号起始时间。  
- 使用解析得到的时间戳（秒为单位），直接定位 EDF 文件中的对应采样点（例如，通过采样率计算采样索引）。
- 对于 lab 文件，转换后的秒级时间也可与 EDF 信号时间轴对齐。

## 8. 总结

1. **文件格式**：  
   最新数据集包含 EDF、HTK、LAB 和 REC 文件，其中 LAB 文件提供以 10 微秒为单位的高精度标注，REC 文件以秒级给出标注（附带通道编号与数字标签）。

2. **标签数据结构**：  
   LAB 文件的每行记录包括开始与结束时间（以 10 微秒为单位）以及事件标签（4 字母代码）；REC 文件的每行则包含通道编号、开始时间、结束时间及数字标签（1–6 分别代表 spsw、gped、pled、eyem、artf 和 bckg）。

3. **通道与挂接**：  
   虽然 EEG 原始数据可能存在多种通道配置，但本数据集均为标准 10/20 系统，且所有数据可转换为 ACNS TCP 挂接，挂接定义详见 readme 文件中提供的 22 个通道配置。

4. **文件命名与目录结构**：  
   数据集分为 train（训练集）和 eval（评估集），文件命名中嵌入了患者索引、会话编号以及分段信息，便于区分和跨版本对照。

5. **Python 解析思路**：  
   - 对于 LAB 文件，可按行读取并利用字符串分割提取时间戳和标签，再进行单位转换。  
   - 对于 REC 文件，利用逗号分隔解析每行，并根据预定义的数字与标签映射转换标签。  
   - 解析后的标注数据可与 EDF 信号（利用 EDF 文件头获取采样率和起始时间）对齐，便于后续的事件检测或分类任务。

在实际开发过程中，可以根据具体需求（如分类、事件检测或其他任务）进一步完善解析逻辑和数据对齐步骤。相关工具和示例代码（如官方提供的解析脚本）也可作为参考。

---

**参考文献：**

- Harati, A., Golmohammadi, M., Lopez, S., Obeid, I., & Picone, J. (2015). Improved EEG Event Classification Using Differential Energy. Proceedings of the IEEE Signal Processing in Medicine and Biology Symposium.  
  [在线获取](https://www.isip.piconepress.com/publications/conference_proceedings/2015/ieee_spmb/denergy/)

- Obeid, I., & Picone, J. (2016). The Temple University Hospital EEG Data Corpus. Frontiers in Neuroscience, Section Neural Technology, 10, 196.  
  [在线获取](http://doi.org/http://dx.doi.org/10.3389/fnins.2016.00196)

- Lopez, S., Gross, A., Yang, S., Golmohammadi, M., Obeid, I., & Picone, J. (2016). An Analysis of Two Common Reference Points for EEGs. In IEEE Signal Processing in Medicine and Biology Symposium (pp. 1–4).  
  [在线获取](https://www.isip.piconepress.com/publications/conference_proceedings/2016/ieee_spmb/montages/)

如有更多问题或建议，请联系 help@nedcdata.org。

---

这就是针对最新数据集和 readme.txt 内容的修订报告，希望能对你理解和处理 TUH EEG Event Corpus 的标签数据提供全面参考。