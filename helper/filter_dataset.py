"""
This script is used to filter the dataset for specific dance styles.
"""

import os
import shutil
from pathlib import Path

def filter_files(src_path, dest_path, file_list):
    """
    Copies a list of files from a source to a destination directory.
    Handles both files and directories.
    """
    if not src_path.exists():
        print(f"Warning: Source directory {src_path} does not exist. Skipping.")
        return
    dest_path.mkdir(parents=True, exist_ok=True)
    for f in file_list:
        src_file = src_path / f
        if src_file.exists():
            if src_file.is_dir():
                shutil.copytree(src_file, dest_path / f)
            else:
                shutil.copy(src_file, dest_path / f)
        else:
            # Check for extensions
            found = False
            for ext in ['.wav', '.npy', '.pkl']:
                src_file_ext = src_path / (f + ext)
                if src_file_ext.exists():
                    shutil.copy(src_file_ext, dest_path / (f + ext))
                    found = True
                    break
            if not found:
                print(f"Warning: Could not find {f} in {src_path}")


def main():
    # Define the root directories
    base_dir = Path('/data/home/ec24164/Project/EDGE_Trajectory/data')
    dest_base_dir = Path('/data/home/ec24164/Project/EDGE_Trajectory/filtered_dataset')

    # Define the dance style to filter for
    target_styles = ['gJB']

    # Read the original split files
    with open(base_dir / 'splits/crossmodal_train.txt', 'r') as f:
        train_files = [line.strip() for line in f.readlines()]
    with open(base_dir / 'splits/crossmodal_test.txt', 'r') as f:
        test_files = [line.strip() for line in f.readlines()]
    with open(base_dir / 'splits/ignore_list.txt', 'r') as f:
        ignore_files = [line.strip() for line in f.readlines()]

    # Filter for the target style
    filtered_train = [f for f in train_files if any(f.startswith(style) for style in target_styles) and f not in ignore_files]
    filtered_test = [f for f in test_files if any(f.startswith(style) for style in target_styles) and f not in ignore_files]

    print(f"Found {len(filtered_train)} training files for styles {target_styles}")
    print(f"Found {len(filtered_test)} testing files for styles {target_styles}")

    # Create the directory structure
    data_paths = {
        'train/motions': [f + '.pkl' for f in filtered_train],
        'train/wavs': [f + '.wav' for f in filtered_train],
        'test/motions': [f + '.pkl' for f in filtered_test],
        'test/wavs': [f + '.wav' for f in filtered_test],
        'trajectories_full': [f + '.npy' for f in filtered_test],
    }

    # also handle sliced data
    all_files = filtered_train + filtered_test
    
    if (base_dir / 'trajectories_sliced').exists():
        sliced_dirs = os.listdir(base_dir / 'trajectories_sliced')
        filtered_sliced_dirs = [d for d in sliced_dirs if any(d.startswith(f) for f in filtered_test)]
        data_paths['trajectories_sliced'] = filtered_sliced_dirs
    
    if (base_dir / 'jukebox_feats_rectified').exists():
        sliced_juke_dirs = os.listdir(base_dir / 'jukebox_feats_rectified')
        filtered_sliced_juke_dirs = [d for d in sliced_juke_dirs if any(d.startswith(f) for f in all_files)]
        data_paths['jukebox_feats_rectified'] = filtered_sliced_juke_dirs
    
    # Get sliced motion and wav data for both train and test
    for split, filtered_files in [('train', filtered_train), ('test', filtered_test)]:
        motion_sliced_path = base_dir / split / 'motions_sliced'
        wavs_sliced_path = base_dir / split / 'wavs_sliced'
        jukebox_feats_path = base_dir / split / 'jukebox_feats'

        if motion_sliced_path.exists():
            motion_sliced_dirs = os.listdir(motion_sliced_path)
            filtered_motion_sliced = [d for d in motion_sliced_dirs if any(d.startswith(f) for f in filtered_files)]
            data_paths[f'{split}/motions_sliced'] = filtered_motion_sliced

        if wavs_sliced_path.exists():
            wavs_sliced_dirs = os.listdir(wavs_sliced_path)
            filtered_wavs_sliced = [d for d in wavs_sliced_dirs if any(d.startswith(f) for f in filtered_files)]
            data_paths[f'{split}/wavs_sliced'] = filtered_wavs_sliced
            
        if jukebox_feats_path.exists():
            jukebox_feat_files = os.listdir(jukebox_feats_path)
            filtered_jukebox_feats = [d for d in jukebox_feat_files if any(d.startswith(f) for f in filtered_files)]
            data_paths[f'{split}/jukebox_feats'] = filtered_jukebox_feats


    # Now copy all the files
    for sub_dir, file_list in data_paths.items():
        print(f"Processing {sub_dir}...")
        src_path = base_dir / sub_dir
        dest_path = dest_base_dir / sub_dir
        # for jukebox_feats_rectified and trajectories_sliced the files are directories
        filter_files(src_path, dest_path, file_list)

    # Create new split files
    splits_dir = dest_base_dir / 'splits'
    splits_dir.mkdir(exist_ok=True)
    with open(splits_dir / 'crossmodal_train.txt', 'w') as f:
        for item in filtered_train:
            f.write("%s\n" % item)
    with open(splits_dir / 'crossmodal_test.txt', 'w') as f:
        for item in filtered_test:
            f.write("%s\n" % item)

    print("\nFiltered dataset created successfully!")
    print(f"New dataset is in: {dest_base_dir}")

if __name__ == '__main__':
    main()
