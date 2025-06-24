import torch
from PIL import Image
from torchvision import transforms
from config import *
from models import Generator

def generate_anime_image(input_path, output_path, model_path=None):
    # Load and preprocess input image
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    input_img = Image.open(input_path).convert('RGB')
    input_tensor = transform(input_img).unsqueeze(0).to(device)

    # Load generator
    generator = Generator().to(device)
    if model_path and os.path.exists(model_path):
        generator.load_state_dict(torch.load(model_path))
    else:
        print("No valid model checkpoint found. Using untrained generator.")
    generator.eval()

    # Generate anime image
    with torch.no_grad():
        noise = torch.randn(1, nz, 1, 1, device=device)
        fake_img = generator(noise, input_tensor)
        save_image(fake_img, output_path, normalize=True)
        print(f"Generated anime image saved to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate anime image from a human face image.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to input CelebA image")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output image")
    args = parser.parse_args()

    model_path = f"{checkpoint_dir}/generator_epoch_{num_epochs}.pth" if os.path.exists(f"{checkpoint_dir}/generator_epoch_{num_epochs}.pth") else None
    generate_anime_image(args.input_path, args.output_path, model_path)