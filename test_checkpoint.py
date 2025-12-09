#!/usr/bin/env python3
import torch
import torch.backends.cudnn
import numpy as np
from torch import nn
from PIL import Image
import argparse
import sys
import os
from torchvision.transforms import Resize

from mr_cnn import MultiResolutionCNN

# torch.cuda.empty_cache()
current_dir = os.path.dirname(os.path.abspath(__file__))
mit_data_dir = os.path.join(current_dir, 'data')
fix_map_data_dir = os.path.join(current_dir, 'data', 'ALLFIXATIONMAPS')
checkpoint_dir = os.path.join(current_dir, 'data', 'CHECKPOINTS')
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
        torch.Tensor: Resized grayscale image of shape [H, W], values in range [0, 255].
    """
    # Step 1: Reshape preds to a 2D grid (assume 50x50 for 2500 elements)
    preds = preds.view(50, 50)  # Reshape to [50, 50]

    # Step 2: Resize to target height and width
    preds = preds.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, 50, 50]
    resize = Resize(target_shape)
    preds_resized = resize(preds)  # Shape: [1, 1, H, W]
    preds_resized = preds_resized.squeeze(0).squeeze(0)  # Shape: [H, W]

    # Step 3: Scale values from [0, 1] to [0, 255]
    preds_resized = preds_resized * 255.0

    return preds_resized  # Return tensor of shape [H, W], values in [0, 255]

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
                preds = torch.sigmoid(logits).squeeze()  # Shape: [2500]

                height, width = self.test_dataset.dataset[i]['X'].shape[1:]  # Shape is [3, H, W]
                # Resize, interpolate, and scale to [0, 255]
                preds_gray_interpolated = process_predictions(preds, (height, width))  # Shape: [H, W], values in [0, 255]

                gt_path_a = f"{file_name[:-5]}_fixMap.jpg"
                gt_path = os.path.join(fix_map_data_dir, gt_path_a)

                fixation_map = Image.open(gt_path).convert('L')
                fixation_map_np = np.array(fixation_map)  # Convert to NumPy array

                # Store the preds_gray_interpolated in preds_dict with values in [0, 255]
                preds_dict[file_name] = preds_gray_interpolated.cpu().numpy()

                # Store the ground truth fixation map
                targets_dict[file_name] = fixation_map_np

                # Save the first 5 preds and preds_gray_interpolated as images
                
                # Save preds as a 50x50 image
                preds_image = preds.view(50, 50).cpu().numpy()
                preds_image = (preds_image * 255).astype(np.uint8)
                preds_pil = Image.fromarray(preds_image, mode='L')
                preds_pil.save(f'preds_{file_name}')

                # Save preds_gray_interpolated as an image
                preds_gray_image = preds_gray_interpolated.cpu().numpy()
                preds_gray_image = preds_gray_image.astype(np.uint8)
                preds_gray_pil = Image.fromarray(preds_gray_image, mode='L')
                preds_gray_pil.save(f'preds_gray_interpolated_{file_name}')

                preds_dict_temp = {}
                targets_dict_temp = {}
                preds_dict_temp[file_name] = preds_dict[file_name]
                targets_dict_temp[file_name] = targets_dict[file_name]

                print(file_name, calculate_auc(preds_dict_temp, targets_dict_temp))

                if endIndex == 250000:
                    break
                startIndex += 2500
                endIndex += 2500

            mean_auc = calculate_auc(preds_dict, targets_dict)
            print(f"Mean AUC score for test set: {mean_auc}")

def main(args):
    torch.cuda.empty_cache()
    test_dataset = MIT(dataset_path=os.path.join(mit_data_dir, 'test_data.pth.tar'))

    # Load checkpoint
    checkpoint_directory = os.path.join(checkpoint_dir, args.checkpoint_name)
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
