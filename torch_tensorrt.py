import argparse
import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_tensorrt
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from unet import UNet


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=8, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=1, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=True, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    parser.add_argument('--wandb', action='store_true', default=False, help='Enable wandb')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    model = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    model = model.to(memory_format=torch.channels_last)

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    state_dict = torch.load('/triton_example/checkpoints/checkpoint_epoch25.pth', map_location=device)
    del state_dict['mask_values']
    
    model.load_state_dict(state_dict)
    logging.info(f'Model loaded from {args.load}')
    model: nn.Module = model.eval()
    # print available memory
    logging.info(f'Memory available: {torch.cuda.mem_get_info()}')
    model = model.to(device=device, dtype=torch.half)
    
    inputs = [
        torch_tensorrt.Input(
            min_shape=[1, 3, 480, 640],
            opt_shape=[8, 3, 480, 640],
            max_shape=[8, 3, 480, 640],
            dtype=torch.half,
        )
    ]

    trt_module = torch_tensorrt.compile(model, inputs=inputs, enabled_precisions={torch.half}, debug=True)

    ts_trt_model = torch.jit.trace(trt_module, torch.randn(1, 3, 480, 640, device=device, dtype=torch.half))
    torch.jit.save(ts_trt_model, '/triton_example/model_repository/unet/1/model.pt')
    