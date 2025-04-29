import wandb
import numpy as np
import time
import os
import subprocess

# 清除任何现有的 wandb 环境变量
for key in list(os.environ.keys()):
    if key.startswith('WANDB_'):
        del os.environ[key]

# 设置关键的 wandb 环境变量
os.environ["WANDB_API_KEY"] = "9c973791b10e62adc1089ca11baa273755d50d7f"
os.environ["WANDB_ENTITY"] = "electrixoul-tsinghua-university"  # 明确指定您的 Tsinghua 账户
os.environ["WANDB_CONSOLE"] = "off"  # 禁用 wandb 控制台以避免输出混淆

print("当前环境变量设置:")
for key, value in os.environ.items():
    if key.startswith('WANDB_'):
        print(f"{key} = {value}")

print("\n初始化 wandb 运行...")

try:
    # 初始化 wandb，明确指定实体
    run = wandb.init(
        entity="electrixoul-tsinghua-university",  # 明确指定 Tsinghua 账户
        project="tsinghua-test",                  # 项目名称
        name="tsinghua-account-test",             # 运行名称
        config={
            "test_type": "tsinghua_account_test",
            "timestamp": time.strftime("%Y-%m-%d_%H-%M-%S")
        }
    )

    print(f"运行已初始化，实体: {run.entity}")
    print(f"运行已初始化，项目: {run.project}")
    print(f"运行 URL: {run.get_url()}")

    # 模拟一个简单的训练过程
    print("\n开始最小测试...")
    for i in range(3):
        # 生成简单指标
        test_accuracy = 0.7 + (i * 0.1)  # 简单递增的精度
        test_loss = 0.3 - (i * 0.09)     # 简单递减的损失
        
        # 记录到 wandb
        wandb.log({
            "test_accuracy": test_accuracy,
            "test_loss": test_loss,
            "step": i
        })
        
        print(f"步骤 {i+1}/3: 精度={test_accuracy:.4f}, 损失={test_loss:.4f}")
        time.sleep(0.5)  # 步骤之间短暂延迟

    # 添加总结指标
    wandb.run.summary["final_accuracy"] = test_accuracy
    wandb.run.summary["final_loss"] = test_loss

    # 结束运行
    print("\n测试完成!")
    print(f"运行数据可在以下位置查看: {run.get_url()}")
    wandb.finish()
    
except Exception as e:
    print(f"错误: {e}")
    
    # 尝试获取更多调试信息
    print("\n---调试信息---")
    print("检查当前登录用户:")
    subprocess.run(["wandb", "whoami"], capture_output=False)
