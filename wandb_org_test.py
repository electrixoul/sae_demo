import wandb
import numpy as np
import time
import os

# Create a descriptive project name that clearly indicates it's for your personal use
PROJECT_NAME = "electrixoul-personal-space"

print("Creating a personal project within the organization...")

# Initialize wandb
run = wandb.init(
    # Use the existing organization but with a unique project name
    project=PROJECT_NAME, 
    name="personal-test-run",
    tags=["personal", "electrixoul"], 
    config={
        "owner": "electrixoul",
        "purpose": "personal_test",
        "timestamp": time.strftime("%Y-%m-%d_%H-%M-%S")
    }
)

print(f"Run initialized with entity: {run.entity}")
print(f"Run initialized with project: {run.project}")
print(f"Run URL: {run.get_url()}")

# Simulate a very minimal training process (3 steps)
print("\nStarting minimal test...")
for i in range(3):
    # Generate simple metrics
    test_accuracy = 0.6 + (i * 0.15)  # Simple increasing accuracy
    test_loss = 0.4 - (i * 0.12)      # Simple decreasing loss
    
    # Log to wandb with ownership metadata
    wandb.log({
        "test_accuracy": test_accuracy,
        "test_loss": test_loss,
        "step": i,
        "owner": "electrixoul"  # Add personal identifier in the metrics
    })
    
    print(f"Step {i+1}/3: accuracy={test_accuracy:.4f}, loss={test_loss:.4f}")
    time.sleep(0.5)  # Short delay between steps

# Add a note to identify this run as belonging to you
wandb.run.notes = "This is a personal test run by electrixoul"

# Add summary metrics
wandb.run.summary["owner"] = "electrixoul"
wandb.run.summary["test_completed"] = True
wandb.run.summary["final_accuracy"] = test_accuracy
wandb.run.summary["final_loss"] = test_loss

# Finish the run
print("\nTest complete!")
print(f"Run data available at: {run.get_url()}")
wandb.finish()
