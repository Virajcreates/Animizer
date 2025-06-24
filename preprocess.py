import os
from PIL import Image
import torch
from torchvision import transforms
from config import data_root, image_size

def preprocess_celeba_images(celeba_dir, celeba_cropped_dir):
    os.makedirs(celeba_cropped_dir, exist_ok=True)
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    
    for img_name in os.listdir(celeba_dir):
        if img_name.endswith('.jpg'):
            img_path = os.path.join(celeba_dir, img_name)
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img)
            # Save as PNG to avoid JPEG artifacts (optional)
            output_path = os.path.join(celeba_cropped_dir, img_name.replace('.jpg', '.png'))
            transforms.ToPILImage()(img_tensor).save(output_path)

if __name__ == "__main__":
    celeba_dir = os.path.join(data_root, "celeba/img_align_celeba")
    celeba_cropped_dir = os.path.join(data_root, "celeba/celeba_cropped")
    preprocess_celeba_images(celeba_dir, celeba_cropped_dir)