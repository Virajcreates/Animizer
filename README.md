# Anime Image Generation with DCGAN

This repository contains a Deep Convolutional Generative Adversarial Network (DCGAN) implementation to generate anime-style images from human face images (CelebA dataset). The model is trained on an NVIDIA RTX 3050 with 4GB VRAM, optimized for low-memory environments using mixed precision training and gradient accumulation.

## Features
- Generates anime-style images conditioned on CelebA images.
- Supports dataset subsetting for faster experimentation.
- Includes timing metrics to optimize training performance.
- Compatible with PyTorch and optimized for limited VRAM (4GB).

## Requirements
- **Python 3.8+**
- **PyTorch** (with CUDA support for RTX 3050)
- **torchvision**
- **numpy**
- **pillow**
- **torch-optimizer** (optional, for advanced optimizers)
- **ImageMagick** (for resizing images offline, optional)

Install dependencies:
```bash
pip install torch torchvision numpy pillow
