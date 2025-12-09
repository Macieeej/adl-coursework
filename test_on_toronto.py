#!/usr/bin/env python3
import time
from multiprocessing import cpu_count
from typing import Union, NamedTuple, Tuple
import torch
import torch.backends.cudnn
import numpy as np
from torch import nn, optim
from torch.nn import functional as F
import torchvision.datasets
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image
import argparse
from pathlib import Path
import sys
import os
from torchvision.transforms import Resize

from mr_cnn import MultiResolutionCNN

#torch.cuda.empty_cache()
current_dir = os.path.dirname(os.path.abspath(__file__))
mit_data_dir = os.path.abspath(os.path.join(current_dir, '..', 'TORONTO_data'))
fix_map_data_dir = os.path.abspath(os.path.join(current_dir, '..', 'ALLFIXATIONMAPS_TORONTO'))
checkpoint_dir = os.path.abspath(os.path.join(current_dir, '..', 'CHECKPOINTS'))
sys.path.append(mit_data_dir)
sys.path.append(fix_map_data_dir)
sys.path.append(checkpoint_dir)
from dataset import MIT, crop_to_region
from metrics import calculate_auc, roc_auc

parser = argparse.ArgumentParser(
    description="'Test the model on the test dataset'",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--checkpoint-name', type=str, required=True, help='Name of the model checkpoint inside the CHECKPOINTS dir')


if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")



def process_predictions(preds, target_shape):
    """
    Process predictions to create a resized grayscale image.

    Args:
        preds (torch.Tensor): 1D tensor of shape [2500].
        target_shape (tuple): Target height and width (H, W).

    Returns:
        torch.Tensor: Resized grayscale image of shape [1, H, W].
    """
    # Step 1: Reshape preds to a 2D grid (assume 50x50 for 2500 elements)
    preds = preds.view(-1)

    grid_size = int(preds.shape[0]**0.5)
    preds_gray = preds.view(grid_size, grid_size) * 255  # Scale to [0, 255]

    # Step 2: Resize to target height and width
    height, width = target_shape
    preds_gray = preds_gray.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    resize = Resize((height, width))
    preds_gray_resized = resize(preds_gray)

    return preds_gray_resized.squeeze(0)  # Remove batch dimension



class Tester:
    def __init__(
        self,
        model: nn.Module,
        test_dataset: MIT,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.device = device
        self.test_dataset = test_dataset

    def test(self):
        preds_dict = {}
        targets_dict = {}
        self.model.eval()

        # No need to track gradients for validation, we're not optimizing.
        with torch.no_grad():
            #for batch, label in self.val_dataset:
            startIndex = 0
            endIndex = 2500

            for i in range(len(self.test_dataset.dataset)):
                batch = []  # To store all 2500 images


                for j in range(startIndex, endIndex):
                    img, label = self.test_dataset.__getitem__(j)  # Unpack the tuple
                    batch.append(img)  # Append the image to the batch list

                file_name = self.test_dataset.dataset[i]['file']
                batch = torch.stack(batch, dim=0).to(self.device)

                logits = self.model(batch)
                preds = torch.sigmoid(logits)

                height, width = self.test_dataset.dataset[i]['X'].shape[1:]  # Shape is [3, H, W]
                # Resize and interpolate
                preds_gray_interpolated = process_predictions(preds, (height, width))

                gt_path_a = f"d{file_name}"
                gt_path = os.path.join(fix_map_data_dir, gt_path_a)

                fixation_map = Image.open(gt_path).convert('L')
                preds_dict[file_name] = preds_gray_interpolated.cpu().numpy()

                fixation_map_np = np.array(fixation_map)  # Convert to NumPy array
                targets_dict[file_name] = fixation_map_np

                if endIndex == 300000:
                    break
                startIndex += 2500
                endIndex += 2500

            mean_auc = calculate_auc(preds_dict, targets_dict)
            print(f"Mean AUC score for test set: {mean_auc}")


def main(args):
    torch.cuda.empty_cache()
    test_dataset = MIT(dataset_path=os.path.join(mit_data_dir, 'toronto_test_data.pth.tar'))


    # Load checkpoint
    checkpoint_directory = os.path.abspath(os.path.join(checkpoint_dir, args.checkpoint_name))
    checkpoint = torch.load(checkpoint_directory, map_location=DEVICE)
    print(f"Resuming model {args.checkpoint_name} that achieved {checkpoint['accuracy']}% accuracy")


    model_args = checkpoint['args']
    model = MultiResolutionCNN(dropout=model_args.dropout)
    model.load_state_dict(checkpoint['model'])
    model.to(DEVICE)
    model.eval()

    print("Model loaded successfully, running tester")
    tester = Tester(
        model, test_dataset, DEVICE,
    )

    tester.test()

if __name__ == "__main__":
    main(parser.parse_args())
