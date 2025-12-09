import os
import sys
import torch
from torchvision import transforms
from torch import Tensor
from PIL import Image
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
nusef_data_dir = os.path.abspath(os.path.join(current_dir, '..', 'NUSEF_data'))
nusef_images_dir = os.path.abspath(os.path.join(current_dir, '..', 'NUSEF_database', 'stimuli', 'train'))
fixation_maps_dir = os.path.abspath(os.path.join(current_dir, '..', 'ALLFIXATIONMAPS_NUSEF'))

sys.path.append(nusef_data_dir)
sys.path.append(nusef_images_dir)
sys.path.append(fixation_maps_dir)

def sample_crop_locations(fixation_map, num_fixation, num_nonfixation, crop_size):

    H, W = fixation_map.shape
    half_crop = crop_size // 2

    fixation_pixels = np.argwhere(fixation_map > 0.9)
    nonfixation_pixels = np.argwhere(fixation_map < 0.1)

    fixation_pixels = fixation_pixels[
        (fixation_pixels[:, 0] >= half_crop) & (fixation_pixels[:, 0] < H - half_crop) &
        (fixation_pixels[:, 1] >= half_crop) & (fixation_pixels[:, 1] < W - half_crop)
    ]
    nonfixation_pixels = nonfixation_pixels[
        (nonfixation_pixels[:, 0] >= half_crop) & (nonfixation_pixels[:, 0] < H - half_crop) &
        (nonfixation_pixels[:, 1] >= half_crop) & (nonfixation_pixels[:, 1] < W - half_crop)
    ]

    fixation_coords = fixation_pixels
    if len(fixation_coords) > num_fixation:
        fixation_coords = fixation_coords[np.random.choice(len(fixation_coords), num_fixation, replace=False)]
    else:
        print(f"Not enough fixation pixels, requested {num_fixation}, but found {len(fixation_coords)}")

    nonfixation_coords = nonfixation_pixels
    if len(nonfixation_coords) > num_nonfixation:
        nonfixation_coords = nonfixation_coords[np.random.choice(len(nonfixation_coords), num_nonfixation, replace=False)]
    else:
        print(f"Not enough non-fixation pixels, requested {num_nonfixation}, but found {len(nonfixation_coords)}")

    return fixation_coords.tolist(), nonfixation_coords.tolist()

def extract_crops(img, coords, scales, crop_size):

    crops = []
    x, y = coords 
    orig_w, orig_h = img.size 

    for scale in scales:
        resize_transform = transforms.Resize((scale, scale))
        scaled_img = resize_transform(img)

        scale_w, scale_h = scaled_img.size
        scale_factor_w = scale_w / orig_w
        scale_factor_h = scale_h / orig_h

        x_scaled = int(round(x * scale_factor_w))
        y_scaled = int(round(y * scale_factor_h))

        half_crop = crop_size // 2

        left = x_scaled - half_crop
        upper = y_scaled - half_crop
        right = x_scaled + half_crop
        lower = y_scaled + half_crop

        left = max(0, left)
        upper = max(0, upper)
        right = min(scale_w, right)
        lower = min(scale_h, lower)

        if lower <= upper or right <= left:
            print(f"Invalid crop dimensions for scale {scale} at coordinates ({x}, {y}). Skipping this crop.")
            continue

        crop = scaled_img.crop((left, upper, right, lower))

        if crop.size != (crop_size, crop_size):
            crop = crop.resize((crop_size, crop_size))

        crop_tensor = transforms.ToTensor()(crop)
        crops.append(crop_tensor)
    return crops 

def create_nusef_dataset(images_dir: str, fixation_maps_dir: str, save_path: str):

    dataset = []

    sizes = [400, 250, 150]
    crop_size = 42 
    num_fixation = 10
    num_nonfixation = 20


    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    image_files.sort() 

    for i, image_file in enumerate(image_files):
        image_path = os.path.join(images_dir, image_file)
        img = Image.open(image_path).convert('RGB')


        fixation_map_name = image_file 
        fixation_map_path = os.path.join(fixation_maps_dir, fixation_map_name)
        if not os.path.exists(fixation_map_path):
            fixation_map_name = image_file 
            fixation_map_path = os.path.join(fixation_maps_dir, fixation_map_name)
            if not os.path.exists(fixation_map_path):
                print(f"Fixation map not found for image {image_file}, skipping.")
                continue

        fixation_map = Image.open(fixation_map_path).convert('L')
        fixation_map_np = np.array(fixation_map).astype(np.float32) / 255.0  

  
        fixation_coords, nonfixation_coords = sample_crop_locations(fixation_map_np, num_fixation, num_nonfixation, crop_size)

    
        for coord in fixation_coords:
            y, x = coord  
            crops = extract_crops(img, (x, y), sizes, crop_size)  
            if len(crops) != len(sizes):
                continue  
            X = torch.stack(crops, dim=0)  
            y_label = 1
            dataset.append({
                "X": X,  
                "y": y_label,
                "file": image_file
            })


        for coord in nonfixation_coords:
            y, x = coord
            crops = extract_crops(img, (x, y), sizes, crop_size)
            if len(crops) != len(sizes):
                continue 
            X = torch.stack(crops, dim=0)
            y_label = 0
            dataset.append({
                "X": X,  
                "y": y_label,
                "file": image_file
            })


    torch.save(dataset, save_path)

def main():
    dataset_file_path = os.path.join(nusef_data_dir, "nusef_train_data.pth.tar")

    create_nusef_dataset(images_dir=nusef_images_dir, fixation_maps_dir=fixation_maps_dir, save_path=dataset_file_path)

if __name__ == "__main__":
    main()
