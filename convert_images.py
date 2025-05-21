import os
from PIL import Image
import glob

def convert_png_to_jpg(src_dir, dest_dir, num_images=1000):
    """
    Convert PNG images to JPG and save them in the destination directory
    
    Args:
        src_dir: Source directory with nested PNG images
        dest_dir: Destination directory for JPG images
        num_images: Maximum number of images to convert
    """
    # Create destination directory if it doesn't exist
    os.makedirs(dest_dir, exist_ok=True)
    
    # Find all PNG files recursively
    png_files = glob.glob(os.path.join(src_dir, "**/*.png"), recursive=True)
    print(f"Found {len(png_files)} PNG files")
    
    # Limit the number of files to convert
    png_files = png_files[:num_images]
    
    # Convert each PNG to JPG
    for i, png_file in enumerate(png_files):
        try:
            img = Image.open(png_file).convert('RGB')
            # Create a sequential filename
            jpg_file = os.path.join(dest_dir, f"image_{i:04d}.jpg")
            img.save(jpg_file, 'JPEG')
            
            if (i + 1) % 100 == 0:
                print(f"Converted {i + 1}/{len(png_files)} images")
        except Exception as e:
            print(f"Error converting {png_file}: {e}")
    
    print(f"Converted {len(png_files)} images to JPG format in {dest_dir}")

if __name__ == "__main__":
    src_dir = "mnist_images_extracted/mnist_images"
    dest_dir = "output_images_jpg_rename"
    convert_png_to_jpg(src_dir, dest_dir, num_images=1000)
