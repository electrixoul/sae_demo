#!/usr/bin/env python3
"""
TUH EEG数据集标签数据展示程序
功能：随机选择数据集中的一个EEG文件，展示其信号数据和标签信息
"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt
import pyedflib
import pandas as pd
from typing import Dict, List, Tuple, Optional

# 标签代码映射
LABEL_MAP = {
    1: 'spsw',  # spike and slow wave
    2: 'gped',  # generalized periodic epileptiform discharge
    3: 'pled',  # periodic lateralized epileptiform discharge
    4: 'eyem',  # eye movement
    5: 'artf',  # artifact
    6: 'bckg'   # background
}

# 通道映射 (TCP montage)
CHANNEL_MAP = {
    0: 'FP1-F7', 1: 'F7-T3', 2: 'T3-T5', 3: 'T5-O1',
    4: 'FP2-F8', 5: 'F8-T4', 6: 'T4-T6', 7: 'T6-O2',
    8: 'A1-T3', 9: 'T3-C3', 10: 'C3-CZ', 11: 'CZ-C4',
    12: 'C4-T4', 13: 'T4-A2', 14: 'FP1-F3', 15: 'F3-C3',
    16: 'C3-P3', 17: 'P3-O1', 18: 'FP2-F4', 19: 'F4-C4',
    20: 'C4-P4', 21: 'P4-O2'
}

def find_eeg_files(data_dir: str) -> List[Tuple[str, str]]:
    """
    查找数据集中的所有EDF文件
    返回：[(edf_file_path, rec_file_path), ...]
    """
    eeg_files = []
    
    for subset in ['train', 'eval']:
        subset_path = os.path.join(data_dir, subset)
        if not os.path.exists(subset_path):
            continue
            
        for session_dir in os.listdir(subset_path):
            session_path = os.path.join(subset_path, session_dir)
            if not os.path.isdir(session_path):
                continue
                
            # 查找.edf和对应的.rec文件
            for filename in os.listdir(session_path):
                if filename.endswith('.edf'):
                    edf_path = os.path.join(session_path, filename)
                    # 找对应的.rec文件
                    rec_filename = filename.replace('.edf', '.rec')
                    rec_path = os.path.join(session_path, rec_filename)
                    
                    if os.path.exists(rec_path):
                        eeg_files.append((edf_path, rec_path))
    
    return eeg_files

def load_edf_data(edf_path: str) -> Tuple[np.ndarray, List[str], float]:
    """
    加载EDF文件数据
    返回：(信号数据, 通道标签, 采样率)
    """
    try:
        f = pyedflib.EdfReader(edf_path)
        n_channels = f.signals_in_file
        
        # 获取信号标签和采样率
        signal_labels = f.getSignalLabels()
        sampling_rates = [f.getSampleFrequency(i) for i in range(n_channels)]
        
        # 读取所有信号
        signals = []
        for i in range(n_channels):
            signal = f.readSignal(i)
            signals.append(signal)
        
        f._close()
        
        # 转换为numpy数组
        signals = np.array(signals)
        sampling_rate = sampling_rates[0] if sampling_rates else 250  # 默认250Hz
        
        return signals, signal_labels, sampling_rate
        
    except Exception as e:
        print(f"Error loading EDF file {edf_path}: {e}")
        return None, None, None

def load_rec_annotations(rec_path: str) -> pd.DataFrame:
    """
    加载.rec标注文件
    返回：DataFrame with columns [channel, start_time, end_time, label_code, label_name]
    """
    try:
        annotations = []
        with open(rec_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split(',')
                    if len(parts) == 4:
                        channel = int(parts[0])
                        start_time = float(parts[1])
                        end_time = float(parts[2])
                        label_code = int(parts[3])
                        label_name = LABEL_MAP.get(label_code, f'unknown_{label_code}')
                        
                        annotations.append({
                            'channel': channel,
                            'start_time': start_time,
                            'end_time': end_time,
                            'label_code': label_code,
                            'label_name': label_name
                        })
        
        return pd.DataFrame(annotations)
        
    except Exception as e:
        print(f"Error loading REC file {rec_path}: {e}")
        return pd.DataFrame()

def plot_eeg_with_annotations(signals: np.ndarray, signal_labels: List[str], 
                             sampling_rate: float, annotations: pd.DataFrame,
                             duration: float = 10.0, start_time: float = 0.0):
    """
    绘制EEG信号和标注
    """
    # 计算要显示的样本范围
    start_sample = int(start_time * sampling_rate)
    end_sample = int((start_time + duration) * sampling_rate)
    end_sample = min(end_sample, signals.shape[1])
    
    # 提取要显示的信号段
    signal_segment = signals[:, start_sample:end_sample]
    time_axis = np.arange(signal_segment.shape[1]) / sampling_rate + start_time
    
    # 创建图形
    n_channels = min(8, signal_segment.shape[0])  # 最多显示8个通道
    fig, axes = plt.subplots(n_channels, 1, figsize=(15, 2*n_channels), sharex=True)
    if n_channels == 1:
        axes = [axes]
    
    # 绘制信号
    for i in range(n_channels):
        signal = signal_segment[i]
        axes[i].plot(time_axis, signal, 'b-', linewidth=0.5)
        
        # 设置通道标签
        channel_name = CHANNEL_MAP.get(i, f'Ch{i}') if i < 22 else signal_labels[i] if i < len(signal_labels) else f'Ch{i}'
        axes[i].set_ylabel(f'{channel_name}\n(μV)', fontsize=8)
        axes[i].grid(True, alpha=0.3)
        
        # 添加该通道的标注
        channel_annotations = annotations[annotations['channel'] == i]
        for _, ann in channel_annotations.iterrows():
            if ann['start_time'] <= start_time + duration and ann['end_time'] >= start_time:
                # 标注在当前显示范围内
                ann_start = max(ann['start_time'], start_time)
                ann_end = min(ann['end_time'], start_time + duration)
                
                # 添加彩色背景
                color_map = {'spsw': 'red', 'gped': 'orange', 'pled': 'yellow', 
                           'eyem': 'green', 'artf': 'purple', 'bckg': 'lightgray'}
                color = color_map.get(ann['label_name'], 'gray')
                
                axes[i].axvspan(ann_start, ann_end, alpha=0.3, color=color, 
                              label=ann['label_name'] if ann['label_name'] not in [l.get_label() for l in axes[i].collections] else "")
        
        # 添加图例（只在第一个子图）
        if i == 0:
            handles, labels = axes[i].get_legend_handles_labels()
            if handles:
                axes[i].legend(handles, labels, loc='upper right', fontsize=8)
    
    axes[-1].set_xlabel('Time (seconds)')
    plt.suptitle(f'EEG Signals with Annotations\nTime: {start_time:.1f}s - {start_time+duration:.1f}s, Sampling Rate: {sampling_rate}Hz')
    plt.tight_layout()
    plt.show()

def display_annotation_summary(annotations: pd.DataFrame):
    """
    显示标注摘要信息
    """
    if annotations.empty:
        print("No annotations found.")
        return
    
    print("\n=== Annotation Summary ===")
    print(f"Total annotations: {len(annotations)}")
    
    # 按标签类型统计
    label_counts = annotations['label_name'].value_counts()
    print("\nLabel distribution:")
    for label, count in label_counts.items():
        print(f"  {label}: {count}")
    
    # 按通道统计
    channel_counts = annotations['channel'].value_counts().sort_index()
    print(f"\nAnnotations per channel (showing top 10):")
    for channel, count in channel_counts.head(10).items():
        channel_name = CHANNEL_MAP.get(channel, f'Ch{channel}')
        print(f"  Channel {channel} ({channel_name}): {count}")
    
    # 时间范围
    total_duration = annotations['end_time'].max() - annotations['start_time'].min()
    print(f"\nTime range: {annotations['start_time'].min():.1f}s - {annotations['end_time'].max():.1f}s")
    print(f"Total duration with annotations: {total_duration:.1f}s")
    
    # 显示前几个标注
    print("\nFirst 10 annotations:")
    print(annotations[['channel', 'start_time', 'end_time', 'label_name']].head(10).to_string(index=False))

def main():
    """
    主函数：随机选择一个EEG文件并展示
    """
    data_dir = "eeg_data/eeg_data_lable"
    
    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} not found!")
        return
    
    # 查找所有EEG文件
    print("Searching for EEG files...")
    eeg_files = find_eeg_files(data_dir)
    
    if not eeg_files:
        print("No EEG files found!")
        return
    
    print(f"Found {len(eeg_files)} EEG files")
    
    # 随机选择一个文件
    edf_path, rec_path = random.choice(eeg_files)
    print(f"\nRandomly selected file:")
    print(f"EDF: {edf_path}")
    print(f"REC: {rec_path}")
    
    # 加载EDF数据
    print("\nLoading EDF data...")
    signals, signal_labels, sampling_rate = load_edf_data(edf_path)
    
    if signals is None:
        print("Failed to load EDF data!")
        return
    
    print(f"EDF Info:")
    print(f"  Channels: {signals.shape[0]}")
    print(f"  Samples: {signals.shape[1]}")
    print(f"  Sampling rate: {sampling_rate} Hz")
    print(f"  Duration: {signals.shape[1]/sampling_rate:.1f} seconds")
    
    # 加载标注数据
    print("\nLoading annotations...")
    annotations = load_rec_annotations(rec_path)
    
    # 显示标注摘要
    display_annotation_summary(annotations)
    
    # 可视化数据
    print("\nGenerating visualization...")
    
    # 选择一个有标注的时间段进行显示
    if not annotations.empty:
        # 找到第一个非背景标注的时间
        non_bg_annotations = annotations[annotations['label_name'] != 'bckg']
        if not non_bg_annotations.empty:
            start_time = max(0, non_bg_annotations['start_time'].iloc[0] - 2)  # 提前2秒开始
        else:
            start_time = annotations['start_time'].iloc[0]
    else:
        start_time = 0
    
    # 确保不超过信号长度
    max_time = signals.shape[1] / sampling_rate
    start_time = min(start_time, max_time - 10)
    start_time = max(0, start_time)
    
    plot_eeg_with_annotations(signals, signal_labels, sampling_rate, 
                            annotations, duration=10.0, start_time=start_time)

if __name__ == "__main__":
    main()
