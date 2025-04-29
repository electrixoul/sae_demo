"""
Weights & Biases (wandb) Tsinghua 账户模板

这个模板提供了一个简单的框架，用于将实验结果记录到您的
electrixoul-tsinghua-university wandb 账户中。

使用方法:
1. 将此文件复制到您的项目中
2. 根据您的实验需求修改项目名称、运行名称和配置参数
3. 在训练循环中调用 wandb.log() 记录指标
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

# 演示用法 (如果直接运行此文件)
if __name__ == "__main__":
    # 设置项目名称和运行名称
    PROJECT_NAME = "demo-project"
    RUN_NAME = "template-demo"
    
    # 设置配置参数
    config = {
        "model_type": "demo",
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 10
    }
    
    # 初始化 wandb
    run = setup_wandb(PROJECT_NAME, RUN_NAME, config)
    
    # 模拟训练循环
    print("\n开始训练模拟...")
    epochs = 10
    for epoch in range(epochs):
        # 模拟训练过程
        train_loss = 1.0 - 0.05 * epoch - 0.03 * epoch * (epoch / epochs)
        train_acc = 0.5 + 0.05 * epoch
        val_loss = 1.2 - 0.04 * epoch - 0.02 * epoch * (epoch / epochs)
        val_acc = 0.4 + 0.04 * epoch
        
        # 记录指标到 wandb
        wandb.log({
            "epoch": epoch,
            "train/loss": train_loss,
            "train/accuracy": train_acc,
            "val/loss": val_loss,
            "val/accuracy": val_acc
        })
        
        print(f"Epoch {epoch+1}/{epochs}: "
              f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
              f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
        
        # 短暂暂停以便观察
        time.sleep(0.2)
    
    # 记录总结指标
    wandb.run.summary["final_train_loss"] = train_loss
    wandb.run.summary["final_train_acc"] = train_acc
    wandb.run.summary["final_val_loss"] = val_loss
    wandb.run.summary["final_val_acc"] = val_acc
    
    # 完成运行
    print("\n训练完成!")
    print(f"运行数据可在以下位置查看: {run.get_url()}")
    wandb.finish()
