import os
import csv
try:
    import torchvision
    from PIL import Image
except ImportError:
    print("Please install torchvision and pillow: pip install torchvision pillow")
    exit(1)

def main():
    # Define directories
    data_dir = "data"
    images_dir = os.path.join(data_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    # Download CIFAR10 dataset using torchvision
    print("Downloading CIFAR10 dataset...")
    dataset = torchvision.datasets.CIFAR10(root='./cifar10_raw', train=True, download=True)
    
    # Retrieve class names
    classes = dataset.classes
    
    metadata_path = os.path.join(data_dir, "metadata.csv")
    print(f"Saving first 5000 images to {images_dir} and metadata to {metadata_path}...")
    
    # Process the first 5000 images and metadata
    with open(metadata_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['filename', 'label'])
        
        for i in range(5000):
            img, label_idx = dataset[i]
            label_name = classes[label_idx]
            filename = f"cifar10_{i:04d}.png"
            img_path = os.path.join(images_dir, filename)
            
            # Save the image as PNG
            img.save(img_path)
            
            # Write entry to metadata.csv
            writer.writerow([filename, label_name])
            
            if (i + 1) % 1000 == 0:
                print(f"Processed {i + 1}/5000 images.")
    
    print("Data preparation complete. Saved to 'data/' folder.")

if __name__ == "__main__":
    main()
