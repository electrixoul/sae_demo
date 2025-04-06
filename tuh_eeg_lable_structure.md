明白了。我将分析 TUEV v2.0.1 子集中标签数据的结构和存储规则，判断标签文件的具体格式，并解释如何在 Python 中载入这些标签用于进一步处理（如分类或事件检测），但不提供完整代码实现。
我会尽快整理好研究结果并反馈给你。

# TUH EEG Events Corpus (TUEV v2.0.1) 标签数据格式分析

## 标签文件格式
TUEV v2.0.1的数据集中，事件标签使用了**两种文本文件格式**来存储标注信息：**时间同步事件文件（*.tse）**和**标签文件（*.lbl）** ([annotation_guidelines_v39.pdf](file://file-737Mjvwce42MkQgAqGddMX#:~:text=These%20annotations%20are%20stored%20in,annotation%20as%20a%20flat%20series))。这两类文件均为纯文本格式（即普通的文本文件），而非XML或CSV等标准格式，每种文件有其特定的内容结构，用于描述EEG中的事件标记。其中，`.tse`文件提供**平坦的时间序列标注**（所有导联共用的标签序列），`.lbl`文件提供**基于事件的导联级标注**（可指定发生事件的具体导联，支持层次结构） ([annotation_guidelines_v39.pdf](file://file-737Mjvwce42MkQgAqGddMX#:~:text=These%20annotations%20are%20stored%20in,annotation%20as%20a%20flat%20series))。在TUEV事件库中，所有事件被标注为六大类之一 ([Temple University EEG Corpus - Downloads](https://isip.piconepress.com/projects/nedc/html/tuh_eeg/#:~:text=,BCKG))：**尖波/锐波**（SPSW, spike and sharp wave）、**广泛性周期痫样放电**（GPED, generalized periodic epileptiform discharges）、**局灶性周期痫样放电**（PLED, periodic lateralized epileptiform discharges）、**眼动**（EYEM, eye movement artifact，例如眨眼）、**伪迹**（ARTF, artifact）以及**背景**（BCKG, background） ([Temple University EEG Corpus - Downloads](https://isip.piconepress.com/projects/nedc/html/tuh_eeg/#:~:text=,BCKG))。这些类别以缩写形式出现在标注文件中，作为事件类型标识。

## 标签数据的结构细节

### 时间同步事件文件（*.tse）
`.tse`文件按时间顺序记录整个EEG记录的事件序列，每行对应一个事件片段。文件第一行通常注明版本号，例如`version = tse_v1.0.0`，表示所采用的标注格式版本 ([annotation_guidelines_v39.pdf](file://file-737Mjvwce42MkQgAqGddMX#:~:text=nedc_000%20,0))。之后每一行包含**四个字段**，以空格分隔：**开始时间（秒）**、**结束时间（秒）**、**事件标签**（类别缩写）和**该标签的概率值** ([annotation_guidelines_v39.pdf](file://file-737Mjvwce42MkQgAqGddMX#:~:text=the%20version%20declaration%20use%20a,confidence%20value%2C%20or%20the%20equivalent))。例如，一行内容可能是：

```
0.0000  10.2775  bckg  1.0000
``` 

表示从0.0000秒到10.2775秒这一时间段的事件类别为“bckg”（背景），置信度概率为1.0 ([annotation_guidelines_v39.pdf](file://file-737Mjvwce42MkQgAqGddMX#:~:text=For%20the%20example%20provided%20in,lines%20follow%20the%20same%20format))。对于人工标注的文件，概率字段通常固定为1.0（表示确定的标签） ([annotation_guidelines_v39.pdf](file://file-737Mjvwce42MkQgAqGddMX#:~:text=the%20version%20declaration%20use%20a,confidence%20value%2C%20or%20the%20equivalent))。如果是机器生成的结果，则该字段可用于存储后验概率或置信度。`.tse`文件中的每个时间段标签**适用于所有导联**（不区分具体导联） ([annotation_guidelines_v39.pdf](file://file-737Mjvwce42MkQgAqGddMX#:~:text=Time,is%20shown%20in%20Figure%202)) ([annotation_guidelines_v39.pdf](file://file-737Mjvwce42MkQgAqGddMX#:~:text=the%20version%20declaration%20use%20a,confidence%20value%2C%20or%20the%20equivalent))。通常，人工标注会将整个记录按时间连续划分，使每一时刻都归属某一类别（包括背景） ([annotation_guidelines_v39.pdf](file://file-737Mjvwce42MkQgAqGddMX#:~:text=A%20time,2017))。`.tse`文件不直接包含事件持续时长字段，但可以由结束时间减去开始时间计算得到。

### 事件标签文件（*.lbl）
`.lbl`文件结构比`.tse`更为复杂，用于标注**具体导联上的事件**，并支持标签的层次结构表示 ([annotation_guidelines_v39.pdf](file://file-737Mjvwce42MkQgAqGddMX#:~:text=Label%20files%20use%20event,is%20provided%20in%20Figure%203))。文件开头同样以版本声明开始，如`version = lbl_v1.0.0` ([annotation_guidelines_v39.pdf](file://file-737Mjvwce42MkQgAqGddMX#:~:text=version%20%3D%20lbl_v1))。接下来是**导联蒙太奇定义块（montage block）**，列出每个导联的编号和名称对应关系 ([annotation_guidelines_v39.pdf](file://file-737Mjvwce42MkQgAqGddMX#:~:text=Montage%20Block%3A%20The%20section%20of,to%20form%20this%20output%20channel))。蒙太奇定义明确了当前记录使用的导联配置，例如：`montage = 0, FP1-F7: EEG FP1-REF -- EEG F7-REF`，表示**导联0**对应名称“FP1-F7”，由原始信号中“EEG FP1-REF”减去“EEG F7-REF”形成 ([annotation_guidelines_v39.pdf](file://file-737Mjvwce42MkQgAqGddMX#:~:text=version%20%3D%20lbl_v1,REF))。这个导联映射使我们知道事件发生在原始EEG信号的哪个通道组合上（详见官方提供的导联和通道说明文档 ([annotation_guidelines_v39.pdf](file://file-737Mjvwce42MkQgAqGddMX#:~:text=Montage%20Block%3A%20The%20section%20of,to%20form%20this%20output%20channel))）。

蒙太奇块之后，`.lbl`文件会指定**层级信息**（levels和sublevels）。首先给出总层级数，例如`number_of_levels = 1`，然后为每个层级指定子层级个数，如`level[0] = 1` ([annotation_guidelines_v39.pdf](file://file-737Mjvwce42MkQgAqGddMX#:~:text=number_of_levels%20%3D%201%20level,1))。这些字段允许对事件标签进行分层描述（例如将具体癫痫发作类型作为子类别，归属于更粗的“seiz”癫痫类别）。在TUEV事件库中，标注主要用单层级结构（即只有一个层级），没有进一步的子分类，因此通常`number_of_levels = 1`且每层只有1个子层级。  

接下来是**符号表块（symbol block）**，即标签索引与文本标签的映射列表 ([annotation_guidelines_v39.pdf](file://file-737Mjvwce42MkQgAqGddMX#:~:text=Symbol%20Block%3A%20This%20block%20defines,is%20mapped%20to%20SPSW%2C%20etc))。符号表以字典形式列出可用的标签种类，例如（取TUSZ文档中的例子）：

```
symbols[0] = {0: '(null)', 1: 'spsw', 2: 'gped', 3: 'pled', 4: 'eyem', 5: 'artf', 6: 'bckg', …}
``` 

这表示在层级0中，索引0表示“(null)”空标签（通常不使用），1表示“spsw”，2表示“gped”，…6表示“bckg”等等 ([annotation_guidelines_v39.pdf](file://file-737Mjvwce42MkQgAqGddMX#:~:text=symbols%5B0%5D%20%3D%20,musc%27%2C%2025%3A%20%27elpp%27%2C%2026%3A%20%27elst))。对于TUEV而言，关注的主要是索引1到6对应的六大事件类别，上述符号表即涵盖了这些类别。此外符号表可能列出其它标签（如癫痫发作类别等)但在TUEV事件库中不会实际使用这些超出六类范围的标签。

最后是**标签记录块（label block）**，列出具体的事件标注条目。**每个事件条目**通常格式为：

```
label = {层级, 子层级, 开始时间(秒), 结束时间(秒), 导联编号, [概率向量] }
``` 

例如： 

```
label = {0, 0, 10.2775, 35.7775, 0, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, ...]}
``` 

表示在**层级0**、**子层级0**上，从10.2775秒到35.7775秒，在**导联0**发生了一段事件 ([annotation_guidelines_v39.pdf](file://file-737Mjvwce42MkQgAqGddMX#:~:text=label%20%3D%20,0))。紧随其后的方括号是**标签概率向量**，其长度对应符号表中定义的标签总数。在人工标注情况下，这通常是**一位热（one-hot）**表示——向量中只有对应事件类别的索引位置为1.0，其余为0 ([annotation_guidelines_v39.pdf](file://file-737Mjvwce42MkQgAqGddMX#:~:text=Label%20Block%3A%20The%20last%20section,zero))。例如上例中向量在索引9位置取值1.0，其余为0，那么结合符号表可查出索引9对应标签“spsz”（在此示例中表示一次简单部分癫痫发作) ([annotation_guidelines_v39.pdf](file://file-737Mjvwce42MkQgAqGddMX#:~:text=symbols%5B0%5D%20%3D%20,musc%27%2C%2025%3A%20%27elpp%27%2C%2026%3A%20%27elst))。对于TUEV的数据，由于使用六大类事件标签，人工标注时概率向量中对应该事件类别的位置为1（其余包括背景在内的类别为0）。概率向量也支持机器学习输出的概率分布，因此理论上可以是非0/1的值，但人工标注里主要用0/1二值 ([annotation_guidelines_v39.pdf](file://file-737Mjvwce42MkQgAqGddMX#:~:text=Label%20Block%3A%20The%20last%20section,zero))。

值得注意的是，`.lbl`文件通过“导联编号”字段将事件与特定导联相关联。例如，若某条目导联编号为5，根据前面的蒙太奇定义可确定这是哪个实际通道上的事件。因此，`.lbl`文件的结构清晰地提供了**事件类型、发生时间区间、持续时间（可由起止时间计算）、所属导联**等详细信息 ([annotation_guidelines_v39.pdf](file://file-737Mjvwce42MkQgAqGddMX#:~:text=Label%20Block%3A%20The%20last%20section,zero))。

## 标签与原始EDF数据的对齐方式
TUEV的标注时间基准与原始EEG信号**严格对齐**，采用**相对于记录开始的秒数**来标记事件位置。这意味着标签中的时间戳（开始和结束时间）可以直接与EDF文件中的时间轴对应。例如，0.0000秒对应信号开始，10.2775秒即表示距离开始10.2775秒处 ([annotation_guidelines_v39.pdf](file://file-737Mjvwce42MkQgAqGddMX#:~:text=For%20the%20example%20provided%20in,lines%20follow%20the%20same%20format))。因为EDF信号采样率各异（常见250 Hz～1000 Hz），标注文件通常保留**四位小数**的时间精度，以确保达到1毫秒或更精细的分辨率，足以唯一定位到采样点 ([annotation_guidelines_v39.pdf](file://file-737Mjvwce42MkQgAqGddMX#:~:text=We%20typically%20use%20four%20decimal,by%20four%20decimal%20places%20of))。举例来说，250 Hz采样对应最小时间步长0.004秒，1000 Hz对应0.001秒，因此四位小数（0.0001秒）能够表示这些采样间隔 ([annotation_guidelines_v39.pdf](file://file-737Mjvwce42MkQgAqGddMX#:~:text=We%20typically%20use%20four%20decimal,by%20four%20decimal%20places%20of))。通过已知的采样频率，可将标注时间戳换算为对应的采样点索引，用于精确定位EDF中的数据片段。换言之，标注是基于时间戳的，但本质上可以对应到具体采样点，实现与原始信号的**逐样本精确对齐**。

## 标签文件的命名与组织结构
TUH EEG语料库的文件命名有一定规则，标签文件通常**与对应的EDF数据文件同名，仅扩展名不同**。通常每段EEG记录（EDF文件）会有对应的`.tse`和/或`.lbl`标注文件。一份典型的文件集合可能如下：

```
00010861_s001_t000.edf      # 原始EEG记录
00010861_s001_t000.tse      # 多类别事件的时间序列标注
00010861_s001_t000.lbl      # 多类别事件的导联级标注
```

 ([annotation_guidelines_v39.pdf](file://file-737Mjvwce42MkQgAqGddMX#:~:text=match%20at%20L137%20,tse_bi))上述示例中，`00010861_s001_t000`这一前缀标识了病人和会话，扩展名区分了数据类型。可以看到，`.tse`/`.lbl`文件与EDF数据文件实现**一一对应** ([annotation_guidelines_v39.pdf](file://file-737Mjvwce42MkQgAqGddMX#:~:text=match%20at%20L137%20,tse_bi))。在数据组织上，标注文件通常存放在与原始EDF相同的目录下（如同一患者会话文件夹内），方便根据文件名快速查找匹配。如果下载了官方发布的事件库子集（TUEV），其中应包含经过筛选的EDF文件及其标注文件，目录结构与主库一致。例如官方文档中演示的路径：`.../s003_2003_07_18/00000492_s003_t004.lbl` 表明标注文件放在对应会话日期目录下，与相应的EDF文件一起存储 ([annotation_guidelines_v39.pdf](file://file-737Mjvwce42MkQgAqGddMX#:~:text=s003_2003_07_18%20%3A%20cat%20%2F01_tcp_ar%2F004%2F00000492%2Fs003_2003_07_18%2F00000492_s003_t004))。总之，每个标注文件**命名上直接对应唯一的原始信号文件**，组织上也紧密关联，便于在分析时同时读取信号和标签。

## 在 Python 中读取和解析标签的典型方法
由于`.tse`和`.lbl`均为结构化的纯文本文件，我们可以**使用Python轻松解析**这些标注 ([annotation_guidelines_v39.pdf](file://file-737Mjvwce42MkQgAqGddMX#:~:text=We%20support%20two%20types%20of,scripting%20language%20such%20as%20Python))。官方也提供了一些工具和脚本用于读取和显示标注内容 ([annotation_guidelines_v39.pdf](file://file-737Mjvwce42MkQgAqGddMX#:~:text=of%20events%20with%20start%20and,2018%3B%20McHugh%20%26%20Picone%2C%202016))（例如NEDC项目的Python库中包含`nedc_ann_tools`用于加载标签文件）。在没有现成库的情况下，研究者也常自行编写简短的解析代码来提取标注信息：

- **读取 `.tse` 文件：** 可以按行读取文本，然后对每行（跳过第一行版本声明）使用字符串拆分（如按空格）获取四个字段。前两个字段转换为浮点数即为起止时间（秒），第三个字段是事件类别标签（字符串），第四个字段为概率值（浮点数）。例如，在Python中打开文件后，对每一行执行`line.strip().split()`即可得到包含这四项的列表，然后分别转换/使用即可。由于`.tse`格式简单平坦，相当于四列的表格数据，它也可被看作一种特定的CSV（空格分隔）来处理。

- **读取 `.lbl` 文件：** 解析稍复杂一些，需要分段读取：首先读取文件头的元数据块，然后提取具体事件条目。一般流程是：读取并跳过版本行，然后解析蒙太奇定义（可选地存储导联索引与名称的映射关系）。接下来读取层级和子层级信息，符号表映射等。这些部分的格式为`name = value`或字典形式，可以利用字符串操作提取数字。符号表行例如`symbols[0] = { ... }`可以通过定位花括号`{}`来提取其中的内容，再按照逗号拆分得到索引和标签的对应关系。**事件条目**行以`label = { ... }`开头，可以检索到这一标志后，提取大括号内的内容。例如，可以通过正则表达式或字符串切片将`{}`内的内容拿出，然后按逗号分隔前5个字段（层级、子层级、起始时间、结束时间、导联）和最后的概率向量。概率向量位于方括号`[]`内，也可进一步去掉方括号再按逗号切分得到一组概率值列表。对于人工标注，向量中哪一位为1.0，即表示事件类别的索引，可结合先前读取的符号表字典翻译为实际标签名称 ([annotation_guidelines_v39.pdf](file://file-737Mjvwce42MkQgAqGddMX#:~:text=Label%20Block%3A%20The%20last%20section,zero))。在代码实现上，可以手工解析字符串，或借助Python的`eval`/`ast.literal_eval`将花括号内容当做Python字典/元组解析（需小心格式细节），但总体而言格式较规范，解析并不困难 ([annotation_guidelines_v39.pdf](file://file-737Mjvwce42MkQgAqGddMX#:~:text=We%20support%20two%20types%20of,scripting%20language%20such%20as%20Python))。

需要强调的是，TUH EEG官方提供的工具（如*NEDC EEG Annotation System*）也可以直接读取并操作这些标注格式 ([annotation_guidelines_v39.pdf](file://file-737Mjvwce42MkQgAqGddMX#:~:text=of%20events%20with%20start%20and,2018%3B%20McHugh%20%26%20Picone%2C%202016))。但在研究实践中，如果只是提取标注进行分析或训练模型，使用Python原生方法读取足以胜任。总结来说，**`.tse`和`.lbl`标签文件均为可读的文本格式，非常适合脚本解析** ([annotation_guidelines_v39.pdf](file://file-737Mjvwce42MkQgAqGddMX#:~:text=We%20support%20two%20types%20of,scripting%20language%20such%20as%20Python))。研究人员可以方便地将标注与EDF原始数据结合，用于事件检测、分类等任务的数据准备。例如，读取EDF信号（借助MNE或PyED等库）获得采样率和信号，然后读取对应`.tse`/`.lbl`，根据时间戳定位信号片段并提取出每段的类别标签，从而构建用于机器学习的带标签样本集。通过对TUEV标签数据格式的理解和上述方法的应用，我们即可做好使用该数据集进行事件检测与分类研究的准备。 

**参考文献：**

1. TUH EEG官方项目网站 – *TUH EEG Events Corpus (TUEV v2.0.1)* 简介 ([Temple University EEG Corpus - Downloads](https://isip.piconepress.com/projects/nedc/html/tuh_eeg/#:~:text=,BCKG))  
2. TUH EEG注释指南 (v3.9) – *Annotation File Formats* 部分 ([annotation_guidelines_v39.pdf](file://file-737Mjvwce42MkQgAqGddMX#:~:text=These%20annotations%20are%20stored%20in,annotation%20as%20a%20flat%20series)) ([annotation_guidelines_v39.pdf](file://file-737Mjvwce42MkQgAqGddMX#:~:text=the%20version%20declaration%20use%20a,confidence%20value%2C%20or%20the%20equivalent)) ([annotation_guidelines_v39.pdf](file://file-737Mjvwce42MkQgAqGddMX#:~:text=Label%20Block%3A%20The%20last%20section,zero)) ([annotation_guidelines_v39.pdf](file://file-737Mjvwce42MkQgAqGddMX#:~:text=We%20typically%20use%20four%20decimal,by%20four%20decimal%20places%20of))等  
3. TUH EEG电极和导联说明 (v3.0) – 蒙太奇与导联定义部分 ([annotation_guidelines_v39.pdf](file://file-737Mjvwce42MkQgAqGddMX#:~:text=Montage%20Block%3A%20The%20section%20of,to%20form%20this%20output%20channel)) ([annotation_guidelines_v39.pdf](file://file-737Mjvwce42MkQgAqGddMX#:~:text=version%20%3D%20lbl_v1,REF))  
4. TUH EEG注释文件示例 – 来自官方文档的`.tse`/`.lbl`文件片段 ([annotation_guidelines_v39.pdf](file://file-737Mjvwce42MkQgAqGddMX#:~:text=nedc_000%20,0)) ([annotation_guidelines_v39.pdf](file://file-737Mjvwce42MkQgAqGddMX#:~:text=label%20%3D%20,0))  
5. NEDC工具及文档 – 提供的注释解析工具和使用说明 ([annotation_guidelines_v39.pdf](file://file-737Mjvwce42MkQgAqGddMX#:~:text=of%20events%20with%20start%20and,2018%3B%20McHugh%20%26%20Picone%2C%202016))
