#!/usr/bin/env python3
"""
TUH EEG数据集标签数据展示程序 - 简化版
功能：随机选择数据集中的一个EEG文件，展示其信号数据和标签信息（文本形式）
"""

import os
import random
import numpy as np
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

# 标签中文描述
LABEL_DESCRIPTIONS = {
    'spsw': '尖波/慢波',
    'gped': '广泛性周期性癫痫样放电',
    'pled': '局灶性周期性癫痫样放电', 
    'eyem': '眼动',
    'artf': '伪迹',
    'bckg': '背景'
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
    """查找数据集中的所有EDF文件"""
    eeg_files = []
    
    for subset in ['train', 'eval']:
        subset_path = os.path.join(data_dir, subset)
        if not os.path.exists(subset_path):
            continue
            
        for session_dir in os.listdir(subset_path):
            session_path = os.path.join(subset_path, session_dir)
            if not os.path.isdir(session_path):
                continue
                
            for filename in os.listdir(session_path):
                if filename.endswith('.edf'):
                    edf_path = os.path.join(session_path, filename)
                    rec_filename = filename.replace('.edf', '.rec')
                    rec_path = os.path.join(session_path, rec_filename)
                    
                    if os.path.exists(rec_path):
                        eeg_files.append((edf_path, rec_path))
    
    return eeg_files

def load_edf_data(edf_path: str) -> Tuple[np.ndarray, List[str], float]:
    """加载EDF文件数据"""
    try:
        f = pyedflib.EdfReader(edf_path)
        n_channels = f.signals_in_file
        
        signal_labels = f.getSignalLabels()
        sampling_rates = [f.getSampleFrequency(i) for i in range(n_channels)]
        
        # 只读取前几个通道的数据用于展示（避免内存问题）
        signals = []
        max_channels = min(n_channels, 22)  # 最多读取22个通道
        for i in range(max_channels):
            signal = f.readSignal(i)
            signals.append(signal)
        
        f._close()
        
        signals = np.array(signals)
        sampling_rate = sampling_rates[0] if sampling_rates else 250
        
        return signals, signal_labels, sampling_rate
        
    except Exception as e:
        print(f"Error loading EDF file {edf_path}: {e}")
        return None, None, None

def load_rec_annotations(rec_path: str) -> pd.DataFrame:
    """加载.rec标注文件"""
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
                            'label_name': label_name,
                            'duration': end_time - start_time
                        })
        
        return pd.DataFrame(annotations)
        
    except Exception as e:
        print(f"Error loading REC file {rec_path}: {e}")
        return pd.DataFrame()

def display_signal_statistics(signals: np.ndarray, signal_labels: List[str], sampling_rate: float):
    """显示信号统计信息"""
    print("\n" + "="*60)
    print("EEG 信号统计信息")
    print("="*60)
    
    print(f"通道数量: {signals.shape[0]}")
    print(f"采样点数: {signals.shape[1]:,}")
    print(f"采样率: {sampling_rate} Hz")
    print(f"信号时长: {signals.shape[1]/sampling_rate:.2f} 秒")
    print(f"数据大小: {signals.nbytes/1024/1024:.2f} MB")
    
    print(f"\n各通道信号统计:")
    print("-" * 70)
    print(f"{'通道':<6} {'名称':<10} {'均值':<12} {'标准差':<12} {'最小值':<12} {'最大值':<12}")
    print("-" * 70)
    
    for i in range(min(10, signals.shape[0])):  # 只显示前10个通道
        channel_name = CHANNEL_MAP.get(i, f'Ch{i}')
        signal = signals[i]
        mean_val = np.mean(signal)
        std_val = np.std(signal)
        min_val = np.min(signal)
        max_val = np.max(signal)
        
        print(f"{i:<6} {channel_name:<10} {mean_val:<12.2f} {std_val:<12.2f} {min_val:<12.2f} {max_val:<12.2f}")
    
    if signals.shape[0] > 10:
        print(f"... 还有 {signals.shape[0] - 10} 个通道")

def display_annotation_details(annotations: pd.DataFrame):
    """显示详细的标注信息"""
    if annotations.empty:
        print("No annotations found.")
        return
    
    print("\n" + "="*60)
    print("EEG 标注详细信息")
    print("="*60)
    
    print(f"标注总数: {len(annotations)}")
    
    # 按标签类型统计
    print(f"\n标签分布:")
    print("-" * 50)
    label_counts = annotations['label_name'].value_counts()
    for label, count in label_counts.items():
        description = LABEL_DESCRIPTIONS.get(label, label)
        percentage = count / len(annotations) * 100
        print(f"  {label} ({description}): {count} 个 ({percentage:.1f}%)")
    
    # 时间统计
    print(f"\n时间统计:")
    print("-" * 50)
    print(f"时间范围: {annotations['start_time'].min():.2f}s - {annotations['end_time'].max():.2f}s")
    print(f"总标注时长: {annotations['duration'].sum():.2f}s")
    print(f"平均标注时长: {annotations['duration'].mean():.2f}s")
    print(f"最短标注时长: {annotations['duration'].min():.2f}s")
    print(f"最长标注时长: {annotations['duration'].max():.2f}s")
    
    # 按通道统计
    print(f"\n通道标注分布 (前10个):")
    print("-" * 50)
    channel_counts = annotations['channel'].value_counts().sort_index()
    for channel, count in channel_counts.head(10).items():
        channel_name = CHANNEL_MAP.get(channel, f'Ch{channel}')
        print(f"  通道 {channel:2d} ({channel_name:<8}): {count:2d} 个标注")
    
    # 显示具体标注事件
    print(f"\n标注事件详情 (前15个):")
    print("-" * 80)
    print(f"{'通道':<6} {'通道名':<10} {'开始时间':<10} {'结束时间':<10} {'时长':<8} {'标签':<6} {'描述':<12}")
    print("-" * 80)
    
    for _, ann in annotations.head(15).iterrows():
        channel_name = CHANNEL_MAP.get(ann['channel'], f'Ch{ann["channel"]}')
        description = LABEL_DESCRIPTIONS.get(ann['label_name'], ann['label_name'])
        print(f"{ann['channel']:<6} {channel_name:<10} {ann['start_time']:<10.2f} {ann['end_time']:<10.2f} "
              f"{ann['duration']:<8.2f} {ann['label_name']:<6} {description:<12}")
    
    if len(annotations) > 15:
        print(f"... 还有 {len(annotations) - 15} 个标注事件")

def display_signal_sample(signals: np.ndarray, sampling_rate: float, 
                         annotations: pd.DataFrame, start_time: float = 0, duration: float = 5):
    """显示信号数据样本"""
    print(f"\n" + "="*60)
    print(f"信号数据样本 ({start_time:.1f}s - {start_time+duration:.1f}s)")
    print("="*60)
    
    start_sample = int(start_time * sampling_rate)
    end_sample = int((start_time + duration) * sampling_rate)
    end_sample = min(end_sample, signals.shape[1])
    
    # 显示前几个通道的数据样本
    n_channels_show = min(5, signals.shape[0])
    n_samples_show = min(10, end_sample - start_sample)
    
    print(f"显示前 {n_channels_show} 个通道的前 {n_samples_show} 个采样点:")
    print("-" * 60)
    
    for i in range(n_channels_show):
        channel_name = CHANNEL_MAP.get(i, f'Ch{i}')
        signal_segment = signals[i, start_sample:start_sample + n_samples_show]
        
        print(f"通道 {i} ({channel_name}):")
        values_str = ", ".join([f"{val:8.2f}" for val in signal_segment])
        print(f"  [{values_str}]")
    
    # 显示此时间段的标注
    time_annotations = annotations[
        (annotations['start_time'] <= start_time + duration) & 
        (annotations['end_time'] >= start_time)
    ]
    
    if not time_annotations.empty:
        print(f"\n该时间段的标注事件:")
        print("-" * 60)
        for _, ann in time_annotations.iterrows():
            channel_name = CHANNEL_MAP.get(ann['channel'], f'Ch{ann["channel"]}')
            description = LABEL_DESCRIPTIONS.get(ann['label_name'], ann['label_name'])
            print(f"  通道 {ann['channel']} ({channel_name}): {ann['start_time']:.2f}s-{ann['end_time']:.2f}s "
                  f"[{ann['label_name']}-{description}]")
    else:
        print(f"\n该时间段无标注事件")

def main():
    """主函数"""
    data_dir = "eeg_data/eeg_data_lable"
    
    print("TUH EEG 数据集标签数据展示程序")
    print("="*60)
    
    if not os.path.exists(data_dir):
        print(f"错误：数据目录 {data_dir} 未找到！")
        return
    
    # 查找所有EEG文件
    print("正在搜索EEG文件...")
    eeg_files = find_eeg_files(data_dir)
    
    if not eeg_files:
        print("错误：未找到EEG文件！")
        return
    
    print(f"找到 {len(eeg_files)} 个EEG文件")
    
    # 随机选择一个文件
    edf_path, rec_path = random.choice(eeg_files)
    print(f"\n随机选择的文件:")
    print(f"EDF文件: {edf_path}")
    print(f"标注文件: {rec_path}")
    
    # 加载EDF数据
    print(f"\n正在加载EDF数据...")
    signals, signal_labels, sampling_rate = load_edf_data(edf_path)
    
    if signals is None:
        print("错误：加载EDF数据失败！")
        return
    
    # 加载标注数据
    print(f"正在加载标注数据...")
    annotations = load_rec_annotations(rec_path)
    
    # 显示所有信息
    display_signal_statistics(signals, signal_labels, sampling_rate)
    display_annotation_details(annotations)
    
    # 选择有标注的时间段显示信号样本
    if not annotations.empty:
        # 找到第一个非背景标注
        non_bg_annotations = annotations[annotations['label_name'] != 'bckg']
        if not non_bg_annotations.empty:
            start_time = max(0, non_bg_annotations['start_time'].iloc[0])
        else:
            start_time = annotations['start_time'].iloc[0]
    else:
        start_time = 0
    
    display_signal_sample(signals, sampling_rate, annotations, start_time, duration=5)

if __name__ == "__main__":
    main()
