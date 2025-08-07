import os
import glob
import argparse
import subprocess
import random
from collections import defaultdict

import traj_funcs  # Assuming traj_funcs.py is in the same directory

def get_song_name_from_path(path):
    """Extracts the base song name from a slice path (file or directory).
    e.g., 'data/jukebox_feats_rectified/gWA_sBM_cAll_d26_mWA0_ch02_slice9' -> 'gWA_sBM_cAll_d26_mWA0_ch02'
    e.g., 'data/jukebox_feats_rectified/gWA_sBM_cAll_d26_mWA0_ch02_slice9.npy' -> 'gWA_sBM_cAll_d26_mWA0_ch02'
    """
    basename = os.path.basename(path.strip('/'))
    filename, _ = os.path.splitext(basename)
    parts = filename.split('_')
    if 'slice' in parts[-1]:
        return '_'.join(parts[:-1])
    return filename

def main(args):
    feature_dir = args.feature_dir
    traj_dir = args.traj_dir
    shape = args.shape
    
    os.makedirs(traj_dir, exist_ok=True)
    
    # Get all the individual slice files
    slice_paths = sorted(glob.glob(os.path.join(feature_dir, "*.npy")))
    
    if not slice_paths:
        slice_paths = sorted(glob.glob(os.path.join(feature_dir, "*", "*.npy")))

    if not slice_paths:
        print(f"Error: No .npy feature slice files found in {feature_dir} or its subdirectories.")
        print("Please ensure your feature cache is populated correctly.")
        return

    # Group slices by their base song name
    songs = defaultdict(list)
    for slice_path in slice_paths:
        song_name = get_song_name_from_path(slice_path)
        songs[song_name].append(slice_path)

    print(f"Found {len(slice_paths)} feature slices belonging to {len(songs)} unique songs.")

    available_shapes = ['semicircle', 'line', 'curve', 'curve2', 'circle', 'ellipse']

    for song_name, song_slices in songs.items():
        num_slices = len(song_slices)
        # Assuming 5s slices (150 frames) with a 2.5s hop (75 frames)
        total_frames = 150 + (num_slices - 1) * 75 if num_slices > 0 else 0
        if total_frames == 0:
            print(f"Skipping {song_name} as it has no slices.")
            continue

        output_path = os.path.join(traj_dir, f"{song_name}.npy")
        
        current_shape = shape
        if shape == 'random':
            current_shape = random.choice(available_shapes)
        
        command = [
            "python", "traj_funcs.py",
            "--shape", current_shape,
            "--samples", str(total_frames),
            "--out", output_path
        ]
        
        # Add random parameters for the chosen shape to make them varied
        if current_shape in ['circle', 'semicircle', 'spiral', 'curve', 'curve2', 'ellipse']:
            command.extend(["--radius", f"{random.uniform(0.5, 1.5):.2f}"])
        if current_shape == 'square':
            command.extend(["--side_length", f"{random.uniform(0.8, 2.0):.2f}"])
        if current_shape in ['line', 'spiral', 'curve', 'curve2']:
            command.extend(["--step_length", f"{random.uniform(0.01, 0.05):.2f}"])
        
        print(f"Generating trajectory for song {song_name}... (Shape: {current_shape}, Frames: {total_frames})")
        
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            print(result.stdout.strip())
        except subprocess.CalledProcessError as e:
            print(f"Error generating trajectory for {song_name}:")
            print(e.stderr)
            
    print(f"\nFull trajectory generation complete. {len(songs)} files created in {traj_dir}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate one full-length trajectory for each song based on feature slices.")
    parser.add_argument("--feature_dir", type=str, default="data/jukebox_feats_rectified/", 
                        help="Directory containing the cached .npy feature slices.")
    parser.add_argument("--traj_dir", type=str, default="data/trajectories_full/", 
                        help="Directory where the generated full-length .npy trajectory files will be saved.")
    parser.add_argument("--shape", type=str, default="random", 
                        choices=['semicircle', 'line', 'curve', 'curve2', 'circle', 'ellipse', 'spiral', 'random'],
                        help="The shape of the trajectories to generate. 'random' will pick a random shape for each song.")
    
    args = parser.parse_args()
    main(args)
