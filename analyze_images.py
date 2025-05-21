import os
import numpy as np
from PIL import Image

def analyze_image(image_path):
    """
    Analyze an image and return its properties
    """
    try:
        img = Image.open(image_path)
        
        # Basic properties
        print(f"Image path: {image_path}")
        print(f"Format: {img.format}")
        print(f"Size: {img.size}")
        print(f"Mode: {img.mode}")
        
        # Convert to numpy array for further analysis
        img_array = np.array(img)
        
        # Show shape and data type
        print(f"Array shape: {img_array.shape}")
        print(f"Data type: {img_array.dtype}")
        
        # Calculate basic statistics
        print(f"Min value: {np.min(img_array)}")
        print(f"Max value: {np.max(img_array)}")
        print(f"Mean value: {np.mean(img_array)}")
        print(f"Standard deviation: {np.std(img_array)}")
        
        # Return the image size for later use
        return img.size
        
    except Exception as e:
        print(f"Error analyzing image: {e}")
        return None

def main():
    # Define the directory
    image_dir = "output_images_jpg_rename"
    
    # Get all jpg files
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
    
    print(f"Total number of images: {len(image_files)}")
    
    # Analyze first image
    first_image_path = os.path.join(image_dir, image_files[0])
    first_image_size = analyze_image(first_image_path)
    
    # Analyze last image
    last_image_path = os.path.join(image_dir, image_files[-1])
    analyze_image(last_image_path)
    
    # Check if all images have the same dimensions
    if first_image_size:
        same_size = True
        different_sizes = []
        
        for img_file in image_files[1:10]:  # Check first 10 images
            img_path = os.path.join(image_dir, img_file)
            try:
                with Image.open(img_path) as img:
                    if img.size != first_image_size:
                        same_size = False
                        different_sizes.append((img_file, img.size))
            except Exception as e:
                print(f"Error checking size of {img_file}: {e}")
        
        if same_size:
            print("\nAll images appear to have the same dimensions.")
        else:
            print("\nImages have different dimensions:")
            for img_file, size in different_sizes:
                print(f"{img_file}: {size}")
    
if __name__ == "__main__":
    main()
