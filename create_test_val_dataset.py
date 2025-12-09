import os
import sys
import torch
from torchvision import transforms
from torch import Tensor
from PIL import Image
import argparse
from pathlib import Path

current_dir = os.path.dirname(os.path.abspath(__file__))
nusef_data_dir = os.path.abspath(os.path.join(current_dir, '..', 'NUSEF_data'))
nusef_images_dir = os.path.abspath(os.path.join(current_dir, '..', 'NUSEF_database', 'stimuli'))
toronto_data_dir = os.path.abspath(os.path.join(current_dir, '..', 'TORONTO_data'))
toronto_images_dir = os.path.abspath(os.path.join(current_dir, '..', 'TORONTO_database'))
sys.path.append(nusef_data_dir)
sys.path.append(nusef_images_dir)
sys.path.append(toronto_data_dir)
sys.path.append(toronto_images_dir)

parser = argparse.ArgumentParser(
    description="'Test the model on the test dataset'",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--test-or-val', type=str, required=True, help='prepare from test or val folder')


# Generate evenly spaced coordinates along the height and width
def generate_spatial_coords(image_height: int, image_width: int):
    
    y_coords = torch.linspace(1, image_height, steps=50)
    x_coords = torch.linspace(1, image_width, steps=50)

    y_coords = y_coords.round().long()
    x_coords = x_coords.round().long()

    grid = torch.cartesian_prod(y_coords, x_coords)

    return grid

def create_nusef_dataset(images_dir: str, save_path: str):
    dataset = []
    sizes = [400, 250, 150]
    
    resize_transforms = {size: transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()]) for size in sizes}
    to_tensor = transforms.ToTensor()
    
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.JPG', '.PNG', '.JPEG'))]
    image_files.sort() 

    for i, image_file in enumerate(image_files):
        image_path = os.path.join(images_dir, image_file)
        img = Image.open(image_path).convert('RGB')
        
        orig_h = img.height
        orig_w = img.width

        scaled_images = {size: resize_transforms[size](img) for size in sizes}
        
        coords = generate_spatial_coords(orig_h, orig_w)
        
        dataset.append({
            "X": to_tensor(img),  
            "y": -1,  
            "file": image_file,  
            "X_400": scaled_images[400],  
            "X_250": scaled_images[250],  
            "X_150": scaled_images[150],  
            "spatial_coords": coords  
        })
        
    torch.save(dataset, save_path)


def main(args):
    if args.test_or_val == 'toronto':
        dataset_file_path = os.path.join(toronto_data_dir, "toronto_test_data.pth.tar")
        images_dir = toronto_images_dir
    elif args.test_or_val == 'test':
        dataset_file_path = os.path.join(nusef_data_dir, "nusef_test_data.pth.tar")
        images_dir=os.path.join(nusef_images_dir, args.test_or_val)
    elif args.test_or_val == 'val':
        dataset_file_path = os.path.join(nusef_data_dir, "nusef_val_data.pth.tar")
        images_dir=os.path.join(nusef_images_dir, args.test_or_val)
    else:
        dataset_file_path = os.path.join(nusef_data_dir, "nusef_all_data.pth.tar")
        images_dir=nusef_images_dir

    create_nusef_dataset(images_dir=images_dir, save_path=dataset_file_path)


if __name__ == "__main__":
    main(parser.parse_args())
