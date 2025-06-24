import os
import time
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from config import *
from models import Generator, Discriminator
from dataset import CustomImageDataset
import torchvision.datasets as datasets
from torch.cuda.amp import GradScaler, autocast

# Initialize models and optimizers
generator = Generator().to(device)
discriminator = Discriminator().to(device)
optimizerG = torch.optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerD = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
criterion = nn.BCEWithLogitsLoss()
scaler = GradScaler()

# Load datasets
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
celeba_dataset = datasets.ImageFolder(os.path.join(data_root, "celeba_subset_small"), transform=transform)
anime_dataset = CustomImageDataset(os.path.join(data_root, "anime_subset_small"), transform=transform)
celeba_loader = DataLoader(celeba_dataset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True, pin_memory=True)
anime_loader = DataLoader(anime_dataset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True, pin_memory=True)

def train():
    for epoch in range(num_epochs):
        start_epoch = time.time()
        for i, (celeba_imgs, _) in enumerate(celeba_loader):
            start_batch = time.time()
            anime_iter = iter(anime_loader)
            try:
                anime_imgs, _ = next(anime_iter)
            except StopIteration:
                anime_iter = iter(anime_loader)
                anime_imgs, _ = next(anime_iter)
            data_load_time = time.time()
            
            celeba_imgs = celeba_imgs.to(device)
            anime_imgs = anime_imgs.to(device)
            batch_size = celeba_imgs.size(0)
            
            if batch_size < accumulation_steps:
                print(f"Skipping batch {i} with size {batch_size}")
                continue
            
            start_d_train = time.time()
            discriminator.zero_grad(set_to_none=True)
            d_loss_total = 0
            for _ in range(accumulation_steps):
                sub_batch_size = batch_size // accumulation_steps
                if sub_batch_size == 0:
                    break
                sub_celeba_imgs = celeba_imgs[:sub_batch_size]
                sub_anime_imgs = anime_imgs[:sub_batch_size]
                
                with autocast():
                    real_output = discriminator(sub_anime_imgs)
                    real_loss = criterion(real_output, torch.full((sub_batch_size,), real_label, device=device, dtype=torch.float))
                    noise = torch.randn(sub_batch_size, nz, 1, 1, device=device)
                    fake_imgs = generator(noise, sub_celeba_imgs)
                    fake_output = discriminator(fake_imgs.detach())
                    fake_loss = criterion(fake_output, torch.full((sub_batch_size,), fake_label, device=device, dtype=torch.float))
                    d_loss = (real_loss + fake_loss) / accumulation_steps
                scaler.scale(d_loss).backward()
                d_loss_total += d_loss.item()
            
            scaler.step(optimizerD)
            scaler.update()
            end_d_train = time.time()
            
            start_g_train = time.time()
            for _ in range(2):
                generator.zero_grad(set_to_none=True)
                g_loss_total = 0
                for _ in range(accumulation_steps):
                    sub_batch_size = batch_size // accumulation_steps
                    if sub_batch_size == 0:
                        break
                    sub_celeba_imgs = celeba_imgs[:sub_batch_size]
                    
                    with autocast():
                        noise = torch.randn(sub_batch_size, nz, 1, 1, device=device)
                        fake_imgs = generator(noise, sub_celeba_imgs)
                        fake_output = discriminator(fake_imgs)
                        g_loss = criterion(fake_output, torch.full((sub_batch_size,), real_label, device=device, dtype=torch.float)) / accumulation_steps
                    scaler.scale(g_loss).backward()
                    g_loss_total += g_loss.item()
                
                scaler.step(optimizerG)
                scaler.update()
            end_g_train = time.time()
            
            end_batch = time.time()
            if i % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] Batch [{i}/{len(celeba_loader)}] "
                      f"D Loss: {d_loss_total:.4f} G Loss: {g_loss_total:.4f} "
                      f"Batch Time: {end_batch - start_batch:.2f} seconds "
                      f"(Data Load: {(data_load_time - start_batch):.2f}s, "
                      f"D Train: {(end_d_train - start_d_train):.2f}s, "
                      f"G Train: {(end_g_train - start_g_train):.2f}s)")
                
                with torch.no_grad():
                    noise = torch.randn(batch_size, nz, 1, 1, device=device)
                    fake_imgs = generator(noise, celeba_imgs)
                save_image(fake_imgs, f"{output_dir}/epoch_{epoch+1}_batch_{i}.png", normalize=True)
            
            torch.cuda.empty_cache()
        
        end_epoch = time.time()
        print(f"Epoch {epoch+1} took {(end_epoch - start_epoch)/60:.2f} minutes")
        
        if (epoch + 1) % 10 == 0:
            torch.save(generator.state_dict(), f"{checkpoint_dir}/generator_epoch_{epoch+1}.pth")
            torch.save(discriminator.state_dict(), f"{checkpoint_dir}/discriminator_epoch_{epoch+1}.pth")
            print(f"Saved checkpoint for epoch {epoch+1}")

if __name__ == "__main__":
    train()