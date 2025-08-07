import os
import glob
import argparse
import subprocess
import random
from collections import defaultdict

def get_song_name_from_slice(slice_dir_name):
    """Extracts the base song name from a slice directory name.
    e.g., 'gWA_sBM_cAll_d26_mWA0_ch02_slice9' -> 'gWA_sBM_cAll_d26_mWA0_ch02'
    """
    parts = os.path.basename(slice_dir_name.strip('/')).split('_')
    if 'slice' in parts[-1]:
        return '_'.join(parts[:-1])
    return '_'.join(parts)

def main(args):
    feature_dir = args.feature_dir
    traj_dir = args.traj_dir
    shape = args.shape
    
    # Create trajectory directory if it doesn't exist
    os.makedirs(traj_dir, exist_ok=True)
    
    # Get all the individual slice directories
    slice_dirs = sorted(glob.glob(os.path.join(feature_dir, "*/")))
    
    if not slice_dirs:
        print(f"Error: No slice subdirectories found in {feature_dir}.")
        return

    # Group slices by their base song name
    songs = defaultdict(list)
    for slice_dir in slice_dirs:
        song_name = get_song_name_from_slice(slice_dir)
        songs[song_name].append(slice_dir)

    print(f"Found {len(slice_dirs)} slices belonging to {len(songs)} unique songs.")

    # available_shapes = ['semicircle', 't_shape', 'triangle', 'square', 'line', 'curve', 'curve2', 'circle', 'ellipse', 'spiral']
    available_shapes = ['semicircle', 'line', 'curve', 'curve2', 'circle', 'ellipse']

    for song_name, song_slices in songs.items():
        output_path = os.path.join(traj_dir, f"{song_name}.npy")
        
        # Decide which shape to generate for this song
        current_shape = shape
        if shape == 'random':
            current_shape = random.choice(available_shapes)
        
        command = [
            "python", "traj_funcs.py",
            "--shape", current_shape,
            "--samples", "150",
            "--out", output_path
        ]
        
        # Add random parameters for the chosen shape
        if current_shape in ['circle', 'semicircle', 'spiral', 'curve', 'curve2', 'ellipse']:
            command.extend(["--radius", f"{random.uniform(0.5, 1.5):.2f}"])
        if current_shape in ['square', 'triangle']:
            command.extend(["--side_length", f"{random.uniform(0.8, 2.0):.2f}"])
        if current_shape in ['line', 'spiral', 'curve', 'curve2']:
            command.extend(["--step_length", f"{random.uniform(0.01, 0.05):.2f}"])
        
        print(f"Generating trajectory for song {song_name}... (Shape: {current_shape})")
        
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            print(result.stdout.strip())
        except subprocess.CalledProcessError as e:
            print(f"Error generating trajectory for {song_name}:")
            print(e.stderr)
            
    print(f"\nTrajectory generation complete. {len(songs)} files created.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate one trajectory for each song in the test set.")
    parser.add_argument("--feature_dir", type=str, default="data/jukebox_feats_rectified/", 
                        help="Directory containing the cached feature slices, with one subdirectory per slice.")
    parser.add_argument("--traj_dir", type=str, default="data/trajectories/", 
                        help="Directory where the generated .npy trajectory files will be saved.")
    parser.add_argument("--shape", type=str, default="random", 
                        choices=['semicircle', 'line', 'curve', 'curve2', 'circle', 'ellipse', 'random'],
                        help="The shape of the trajectories to generate. 'random' will pick a random shape for each song.")
    
    args = parser.parse_args()
    main(args)
