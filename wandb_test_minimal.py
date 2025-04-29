import wandb
import numpy as np
import time

# Initialize wandb
wandb.init(
    # Let wandb use the default entity associated with the API key
    project="electrixoul-personal-test",  # Unique project name
    name="personal-account-test",   # Run name
    config={
        "test_type": "connection_verification",
        "timestamp": time.strftime("%Y-%m-%d_%H-%M-%S")
    }
)

# Simulate a very minimal training process (5 steps)
print("Starting minimal test...")
for i in range(5):
    # Generate simple metrics
    test_accuracy = 0.5 + (i * 0.1)  # Simple increasing accuracy
    test_loss = 0.5 - (i * 0.08)     # Simple decreasing loss
    
    # Log to wandb
    wandb.log({
        "test_accuracy": test_accuracy,
        "test_loss": test_loss,
        "step": i
    })
    
    print(f"Step {i+1}/5: accuracy={test_accuracy:.4f}, loss={test_loss:.4f}")
    time.sleep(0.5)  # Short delay between steps

# Finish the run
print("Test complete!")
wandb.finish()
