import os
import glob
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

def parse_fix_file(file_path):
    hor_pos_list = []
    ver_pos_list = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
    in_fix_data_section = False
    cols = []
    for i, line in enumerate(lines):
        line = line.strip()
        if line == '[Fix Data 1]':
            in_fix_data_section = True
            # The next line should be "COLS= ..."
            cols_line = lines[i+1].strip()
            if cols_line.startswith('COLS='):
                cols = cols_line[len('COLS='):].strip().split(', ')
            else:
                print(f'Expected COLS line after [Fix Data 1] in {file_path}')
                return [], []
            # Now process data lines
            data_lines = lines[i+2:]
            for data_line in data_lines:
                data_line = data_line.strip()
                if not data_line or data_line.startswith('['):
                    # End of data section
                    break
                if '=' in data_line:
                    idx, rest = data_line.split('=', 1)
                    values = rest.strip().split()
                    if len(values) != len(cols):
                        print(f'Number of values does not match number of columns in line: {data_line}')
                        continue
                    # Create a dict of column name to value
                    data_dict = dict(zip(cols, values))
                    try:
                        hor_pos = float(data_dict['Hor_Pos'])
                        ver_pos = float(data_dict['Ver_Pos'])
                        hor_pos_list.append(hor_pos)
                        ver_pos_list.append(ver_pos)
                    except KeyError as e:
                        print(f'Column missing in data: {e}')
                else:
                    # Line without '='
                    continue
        elif line.startswith('[') and in_fix_data_section:
            # We've reached a new section
            break
    return hor_pos_list, ver_pos_list

# Paths to the image and fixation data
current_dir = os.path.dirname(os.path.abspath(__file__))
image_paths = os.path.abspath(os.path.join(current_dir, '..', 'NUSEF_database', 'stimuli'))
fixation_folders_root = os.path.abspath(os.path.join(current_dir, '..', 'NUSEF_database', 'fix_data'))

# Output directory for the fixation maps
output_dir = '/content/drive/MyDrive/ColabNotebooks/AppliedDeepLearning/ADL_COURSEWORK/ALLFIXATIONMAPS_NUSEF'

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Constants based on the MATLAB code
asl_width = 260   # ASL eye-tracker coordinate width
asl_height = 280  # ASL eye-tracker coordinate height
yoffset = 10      # Vertical offset as per the MATLAB code

# Get list of all image files in the image_paths directory
image_files = glob.glob(os.path.join(image_paths, '*.jpg'))

# Process each image and its corresponding fixation data
for image_path in image_files:
    image_name = os.path.basename(image_path)
    image_base_name = os.path.splitext(image_name)[0]
    fixation_folder = os.path.join(fixation_folders_root, image_base_name)

    # Check if the corresponding fixation folder exists
    if not os.path.isdir(fixation_folder):
        print(f'Fixation data folder not found for image {image_name}')
        continue  # Skip to the next image

    # Load the image
    image = Image.open(image_path)
    img_width, img_height = image.size  # Original image dimensions

    # Initialize the fixation map with zeros matching the original image dimensions
    fixation_map = np.zeros((img_height, img_width), dtype=np.float32)

    # Compute scaling factors to map ASL coordinates to image coordinates
    scale_x = img_width / asl_width
    scale_y = img_height / asl_height

    # Get list of all .fix files in the fixation folder
    fix_files = glob.glob(os.path.join(fixation_folder, '*.fix'))

    # If no fixation files are found, skip this image
    if not fix_files:
        print(f'No fixation files found in {fixation_folder}')
        continue

    # Process each .fix file
    for fix_file in fix_files:
        hor_pos_list, ver_pos_list = parse_fix_file(fix_file)
        for hor_pos, ver_pos in zip(hor_pos_list, ver_pos_list):
            # Compute fixation point in image coordinates
            px = hor_pos * scale_x
            py = ver_pos * scale_y + yoffset
            # Convert to integer pixel coordinates
            x = int(round(px))
            y = int(round(py))
            # Ensure coordinates are within image bounds
            x = max(0, min(img_width - 1, x))
            y = max(0, min(img_height - 1, y))
            fixation_map[y, x] += 1  # Increment the fixation point

    # Apply Gaussian blur to the fixation map
    sigma = 20  # Standard deviation for Gaussian kernel
    blurred_fixation_map = gaussian_filter(fixation_map, sigma=sigma)

    # Normalize the blurred fixation map for visualization
    max_value = blurred_fixation_map.max()
    if max_value > 0:
        normalized_fixation_map = blurred_fixation_map / max_value
    else:
        normalized_fixation_map = blurred_fixation_map

    # Convert the normalized fixation map to an 8-bit grayscale image
    fixation_map_image = (normalized_fixation_map * 255).astype(np.uint8)
    fixation_map_pil = Image.fromarray(fixation_map_image, mode='L')

    # Save the fixation map image to the output directory
    output_filename = f'{image_base_name}.jpg'
    output_path = os.path.join(output_dir, output_filename)
    fixation_map_pil.save(output_path)
    print(f'Fixation map saved to {output_path}')
