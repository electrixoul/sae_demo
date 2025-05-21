import os
import torch
import torchvision
from torchvision import datasets, transforms
from PIL import Image

def extract_mnist_images(output_dir='mnist_images'):
    """
    Extract all images from MNIST dataset and save them as image files.
    
    Args:
        output_dir (str): Directory to save images
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create subdirectories for train and test sets
    train_dir = os.path.join(output_dir, 'train')
    test_dir = os.path.join(output_dir, 'test')
    
    for directory in [train_dir, test_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    # Define a transform to convert tensor to PIL Image
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    # Download and load the MNIST dataset
    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    
    print(f"Extracting {len(train_dataset)} images from MNIST training dataset...")
    
    # Extract and save each image from training set
    for idx, (img_tensor, label) in enumerate(train_dataset):
        # Create a directory for each digit (0-9)
        label_dir = os.path.join(train_dir, str(label))
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)
        
        # Convert tensor to PIL Image
        img = transforms.ToPILImage()(img_tensor)
        
        # Save the image
        img_path = os.path.join(label_dir, f"train_{idx}.png")
        img.save(img_path)
        
        # Print progress
        if (idx + 1) % 1000 == 0:
            print(f"Processed {idx + 1}/{len(train_dataset)} training images")
    
    print(f"Extracting {len(test_dataset)} images from MNIST test dataset...")
    
    # Extract and save each image from test set
    for idx, (img_tensor, label) in enumerate(test_dataset):
        # Create a directory for each digit (0-9)
        label_dir = os.path.join(test_dir, str(label))
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)
        
        # Convert tensor to PIL Image
        img = transforms.ToPILImage()(img_tensor)
        
        # Save the image
        img_path = os.path.join(label_dir, f"test_{idx}.png")
        img.save(img_path)
        
        # Print progress
        if (idx + 1) % 1000 == 0:
            print(f"Processed {idx + 1}/{len(test_dataset)} test images")
    
    print(f"All MNIST images have been saved to: {os.path.abspath(output_dir)}")
    print(f"Structure: {output_dir}/[train|test]/[0-9]/")

if __name__ == "__main__":
    extract_mnist_images()
