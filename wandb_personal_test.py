import wandb
import numpy as np
import time
import os

print("Current wandb settings before logout:")
os.system("wandb status")

# First, let's log out
print("\nLogging out from current account...")
os.environ.pop("WANDB_API_KEY", None)  # Clear any API key in environment
os.environ.pop("WANDB_ENTITY", None)   # Clear any entity in environment

# Set the API key programmatically
print("\nLogging in with new API key...")
os.environ["WANDB_API_KEY"] = "9c973791b10e62adc1089ca11baa273755d50d7f"
os.environ["WANDB_ENTITY"] = "electrixoul"  # Force personal account

# Print settings after changes
print("\nCurrent wandb settings after setting API key:")
os.system("wandb status")

# Initialize wandb with explicit settings
print("\nInitializing wandb run...")
run = wandb.init(
    project="direct-personal-test",
    name="api-key-test",
    config={
        "test_type": "direct_api_key_test",
        "timestamp": time.strftime("%Y-%m-%d_%H-%M-%S")
    }
)

print(f"Run is using entity: {run.entity}")
print(f"Run is using project: {run.project}")

# Simulate a very minimal training process (3 steps)
print("\nStarting minimal test...")
for i in range(3):
    # Generate simple metrics
    test_accuracy = 0.6 + (i * 0.15)  # Simple increasing accuracy
    test_loss = 0.4 - (i * 0.12)      # Simple decreasing loss
    
    # Log to wandb
    wandb.log({
        "test_accuracy": test_accuracy,
        "test_loss": test_loss,
        "step": i
    })
    
    print(f"Step {i+1}/3: accuracy={test_accuracy:.4f}, loss={test_loss:.4f}")
    time.sleep(0.5)  # Short delay between steps

# Finish the run
print("\nTest complete!")
wandb.finish()
