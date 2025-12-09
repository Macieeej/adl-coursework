import os

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Join paths without converting to absolute paths
mit_data_dir = os.path.join(current_dir, 'data')
fix_map_data_dir = os.path.join(current_dir, 'data', 'ALLFIXATIONMAPS')
checkpoint_dir = os.path.join(current_dir, 'data', 'CHECKPOINTS')

print(f"MIT Data Directory: {mit_data_dir}")
print(f"Fixation Map Data Directory: {fix_map_data_dir}")
print(f"Checkpoint Directory: {checkpoint_dir}")