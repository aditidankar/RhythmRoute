import os
import glob
import argparse
import numpy as np
from collections import defaultdict
from tqdm import tqdm

# Get the directory of the current script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Navigate up to the project root directory (assuming the script is in a subdirectory of the root)
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))

def get_song_name_from_path(path):
    """Extracts the base song name from a slice path."""
    basename = os.path.basename(path.strip('/'))
    filename, _ = os.path.splitext(basename)
    parts = filename.split('_')
    if 'slice' in parts[-1]:
        return '_'.join(parts[:-1])
    return filename

def main(args):
    full_traj_dir = args.full_traj_dir
    sliced_traj_dir = args.sliced_traj_dir
    feature_dir = args.feature_dir
    slice_length = 150  # frames
    hop_length = 75     # frames

    os.makedirs(sliced_traj_dir, exist_ok=True)

    # Use the feature directory to determine the song names and number of slices
    slice_paths = sorted(glob.glob(os.path.join(feature_dir, "*.npy")))
    if not slice_paths:
        slice_paths = sorted(glob.glob(os.path.join(feature_dir, "*", "*.npy")))

    if not slice_paths:
        print(f"Error: No .npy feature slice files found in {feature_dir} or its subdirectories.")
        return

    # Group slices by their base song name
    songs = defaultdict(list)
    for slice_path in slice_paths:
        song_name = get_song_name_from_path(slice_path)
        songs[song_name].append(slice_path)

    print(f"Found {len(songs)} songs to process based on feature files.")

    for song_name, slice_files in tqdm(songs.items(), desc="Slicing Trajectories"):
        full_traj_path = os.path.join(full_traj_dir, f"{song_name}.npy")

        if not os.path.exists(full_traj_path):
            print(f"Warning: Full trajectory not found for {song_name} at {full_traj_path}. Skipping.")
            continue

        full_trajectory = np.load(full_traj_path)
        num_slices = len(slice_files)

        for i in range(num_slices):
            start_frame = i * hop_length
            end_frame = start_frame + slice_length

            if end_frame > len(full_trajectory):
                print(f"Warning: Not enough frames in full trajectory for {song_name} to create slice {i}. "
                      f"Needed {end_frame}, have {len(full_trajectory)}. Skipping slice.")
                continue

            trajectory_slice = full_trajectory[start_frame:end_frame]

            # The output filename should match the feature slice filename
            # e.g., gBI_sBM_c01_d04_mBI0_ch01_slice5.npy
            slice_basename = os.path.basename(slice_files[i])
            output_path = os.path.join(sliced_traj_dir, slice_basename)
            
            # Create subdirectory if feature files are in subdirectories
            if os.path.dirname(slice_files[i]) != feature_dir:
                slice_subdir_name = os.path.basename(os.path.dirname(slice_files[i]))
                output_subdir = os.path.join(sliced_traj_dir, slice_subdir_name)
                os.makedirs(output_subdir, exist_ok=True)
                output_path = os.path.join(output_subdir, slice_basename)


            np.save(output_path, trajectory_slice)

    print(f"\nTrajectory slicing complete. Slices saved in {sliced_traj_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Slice full-length trajectories into 150-frame segments.")
    
    # Construct default paths relative to the project root
    default_full_traj_dir = os.path.join(PROJECT_ROOT, 'data', 'trajectories_full')
    default_sliced_traj_dir = os.path.join(PROJECT_ROOT, 'data', 'trajectories_sliced')
    default_feature_dir = os.path.join(PROJECT_ROOT, 'data', 'jukebox_feats_rectified')

    parser.add_argument("--full_traj_dir", type=str, default=default_full_traj_dir,
                        help="Directory containing the full-length .npy trajectory files.")
    parser.add_argument("--sliced_traj_dir", type=str, default=default_sliced_traj_dir,
                        help="Directory where the generated trajectory slices will be saved.")
    parser.add_argument("--feature_dir", type=str, default=default_feature_dir,
                        help="Directory containing the cached .npy feature slices, used to determine slicing.")
    
    args = parser.parse_args()
    main(args)
