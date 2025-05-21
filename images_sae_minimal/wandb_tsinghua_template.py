"""
Weights & Biases (wandb) Tsinghua 账户模板
"""

import wandb
import os
import time

def setup_wandb(project_name, run_name=None, config=None):
    """
    设置并初始化 wandb 运行
    
    参数:
        project_name (str): 项目名称
        run_name (str, 可选): 运行的名称，如果不提供将自动生成
        config (dict, 可选): 运行的配置参数
    
    返回:
        wandb 运行对象
    """
    # 设置关键的 wandb 环境变量
    os.environ["WANDB_API_KEY"] = "9c973791b10e62adc1089ca11baa273755d50d7f"
    os.environ["WANDB_ENTITY"] = "electrixoul-tsinghua-university"
    
    # 如果没有提供运行名称，创建一个基于时间戳的名称
    if run_name is None:
        run_name = f"run-{time.strftime('%Y%m%d-%H%M%S')}"
    
    # 确保配置是一个字典
    if config is None:
        config = {}
    
    # 添加时间戳到配置中
    config['timestamp'] = time.strftime("%Y-%m-%d_%H-%M-%S")
    
    # 初始化 wandb
    run = wandb.init(
        entity="electrixoul-tsinghua-university",
        project=project_name,
        name=run_name,
        config=config
    )
    
    print(f"wandb 运行已初始化:")
    print(f"  实体: {run.entity}")
    print(f"  项目: {run.project}")
    print(f"  运行名称: {run.name}")
    print(f"  运行 URL: {run.get_url()}")
    
    return run
