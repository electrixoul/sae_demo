import os
import numpy as np
import pyedflib
from scipy.signal import butter, filtfilt
import torch
from torch.utils.data import Dataset
import subprocess
from collections import Counter

# 使用wget下载EEG数据
# 从指定URL使用用户名和密码下载文件并保存到指定的输出目录
# 参数：
# - url: 数据下载链接
# - username: 用户名
# - password: 密码
# - output_dir: 数据保存的目标路径
def download_eeg_data(url, username, password, output_dir):
    command = f'wget -r -np -nH --cut-dirs=7 --no-clobber --user={username} --password={password} -P {output_dir} {url}'
    subprocess.run(command, shell=True, check=True)

# 查找指定目录下的所有.edf文件
# 递归遍历目录并收集所有以.edf结尾的文件路径
# 参数：
# - root_dir: 搜索的根目录
# 返回：
# - 包含.edf文件路径的列表
def find_edf_files(root_dir):
    edf_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith('.edf'):
                edf_files.append(os.path.join(dirpath, filename))
    return edf_files

# 加载单个.edf文件
# 读取信号、采样率和信号标签，仅保留具有最常见采样率的信号，并确保所有信号长度一致
# 参数：
# - file_path: .edf文件的路径
# 返回：
# - 信号数组、信号标签列表、采样率
def load_edf_file(file_path):
    f = pyedflib.EdfReader(file_path)
    n = f.signals_in_file
    signal_labels = f.getSignalLabels()
    signals = []
    sampling_rates = []
    for i in range(n):
        sig = f.readSignal(i)
        signals.append(sig)
        fs = f.getSampleFrequency(i)
        sampling_rates.append(fs)
    f._close()

    # 找到文件中最常见的采样率
    sampling_rate_counts = Counter(sampling_rates)
    most_common_fs, _ = sampling_rate_counts.most_common(1)[0]

    # 根据最常见采样率过滤信号
    filtered_data = [
        (sig, label)
        for sig, label, fs in zip(signals, signal_labels, sampling_rates)
        if fs == most_common_fs
    ]

    if not filtered_data:
        raise ValueError(f"No signals with the most common sampling rate found in {file_path}.")

    # 提取信号和标签，截断所有信号至最小长度
    signals_filtered, signal_labels_filtered = zip(*filtered_data)
    signals_filtered = list(signals_filtered)
    signal_labels_filtered = list(signal_labels_filtered)
    fs = most_common_fs

    min_length = min(len(sig) for sig in signals_filtered)
    signals_filtered = [sig[:min_length] for sig in signals_filtered]

    signals_array = np.array(signals_filtered)
    return signals_array, signal_labels_filtered, fs

# 对信号应用带通滤波器
# 使用巴特沃斯滤波器保留指定频率范围内的频率分量
# 参数：
# - signals: 信号数组
# - lowcut: 带通滤波器的低截止频率
# - highcut: 带通滤波器的高截止频率
# - fs: 信号的采样率
# - order: 滤波器阶数
# 返回：
# - 滤波后的信号数组
def bandpass_filter(signals, lowcut, highcut, fs, order):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_signals = filtfilt(b, a, signals, axis=1)
    return filtered_signals

# 将连续信号分割成固定长度的片段
# 根据指定的段长（秒）分割信号，必要时添加零填充
# 参数：
# - signals: 信号数组
# - fs: 采样率
# - segment_length_sec: 每段的长度（秒）
# 返回：
# - 包含信号片段的列表
def segment_signal(signals, fs, segment_length_sec):
    segment_length_samples = int(segment_length_sec * fs)
    n_channels, n_samples = signals.shape
    segments = []
    for start in range(0, n_samples - segment_length_samples + 1, segment_length_samples):
        end = start + segment_length_samples
        segment = signals[:, start:end]
        segments.append(segment)
    remainder = n_samples % segment_length_samples
    if remainder != 0:
        start = n_samples - remainder
        segment = signals[:, start:]
        padding = np.zeros((n_channels, segment_length_samples - remainder))
        segment = np.hstack((segment, padding))
        segments.append(segment)
    return segments

# 对信号片段进行标准化
# 减去均值并除以标准差，使信号具有零均值和单位方差
# 参数：
# - segment: 信号片段
# - epsilon: 防止除以零的小常数
# 返回：
# - 标准化后的信号片段
def normalize_segment(segment, epsilon=1e-8):
    means = np.mean(segment, axis=1, keepdims=True)
    stds = np.std(segment, axis=1, keepdims=True)
    stds = np.where(stds < epsilon, epsilon, stds)  # 避免除以零
    normalized_segment = (segment - means) / stds
    return normalized_segment

# 将信号片段展平为一维向量
# 为机器学习模型准备输入数据
# 参数：
# - segments: 信号片段的列表
# 返回：
# - 展平后的信号片段列表
def vectorize_segments(segments):
    return [segment.flatten() for segment in segments]

# 主函数：预处理EEG数据并保存为PyTorch张量
# 处理所有.edf文件：滤波、分段、标准化、向量化，并保存结果
# 参数：
# - root_dir: 包含.edf文件的根目录
# - processed_data_file: 保存处理后数据的路径
# - segment_length_sec: 每段的长度（秒）
# - lowcut: 带通滤波器的低截止频率
# - highcut: 带通滤波器的高截止频率
# - filter_order: 带通滤波器的阶数
# - most_common_fs: 使用的最常见采样率
# 返回：
# - 数据的通道数
def preprocess_and_save_data(root_dir, processed_data_file, segment_length_sec, lowcut, highcut, filter_order, most_common_fs):
    edf_files = find_edf_files(root_dir)  # 查找所有.edf文件
    n_channels = None
    segments_list = []
    for file_idx, file_path in enumerate(edf_files):
        signals, _, fs = load_edf_file(file_path)  # 加载信号及其元数据
        if fs != most_common_fs:  # 跳过采样率不匹配的文件
            print(f"Skipping file {file_path} due to different sampling rate ({fs} Hz).")
            continue

        if n_channels is None:
            n_channels = signals.shape[0]
        elif signals.shape[0] != n_channels:  # 跳过通道数不一致的文件
            print(f"Skipping file {file_path} due to inconsistent number of channels.")
            continue

        filtered_signals = bandpass_filter(signals, lowcut, highcut, fs, filter_order)  # 应用带通滤波
        segments = segment_signal(filtered_signals, fs, segment_length_sec)  # 分段信号
        normalized_segments = [normalize_segment(segment) for segment in segments]  # 标准化每个片段
        vectorized_segments = vectorize_segments(normalized_segments)  # 展平片段

        segments_list.extend(vectorized_segments)

    segments_array = np.array(segments_list, dtype=np.float16)  # 转换为NumPy数组
    print(f"Processed data shape: {segments_array.shape}")

    segments_tensor = torch.from_numpy(segments_array)  # 转换为PyTorch张量
    torch.save(segments_tensor, processed_data_file)  # 保存张量到文件
    return n_channels

# EEG数据集类，用于加载预处理后的EEG数据
# 提供访问存储为PyTorch张量的EEG片段的接口
class EEGDataset(Dataset):
    def __init__(self, processed_data_file):
        self.segments_tensor = torch.load(processed_data_file, map_location='cpu')  # 从文件加载张量
        self.n_segments = self.segments_tensor.shape[0]
        self.segment_shape = self.segments_tensor.shape[1:]

    def __len__(self):
        return self.n_segments

    def __getitem__(self, idx):
        segment = self.segments_tensor[idx]
        return (segment,)
