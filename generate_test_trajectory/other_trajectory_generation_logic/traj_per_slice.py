"""
This code does not work because it generates full trajectories (circle, ellipse, etc.) for the slices of the songs.
The trajectories should be generated for the entire song duration and then sliced into segments.
"""


import os
import glob
import argparse
import subprocess
import random

def main(args):
    feature_dir = args.feature_dir
    traj_dir = args.traj_dir
    shape = args.shape
    
    # Create trajectory directory if it doesn't exist
    os.makedirs(traj_dir, exist_ok=True)
    
    # Get the list of song subdirectories from the feature cache
    song_dirs = sorted(glob.glob(os.path.join(feature_dir, "*/")))
    
    if not song_dirs:
        print(f"Error: No subdirectories found in {feature_dir}.")
        print("Please ensure your feature cache is populated correctly.")
        return

    print(f"Found {len(song_dirs)} songs to process.")

    # available_shapes = ['semicircle', 't_shape', 'triangle', 'square', 'line', 'curve', 'curve2', 'circle', 'ellipse', 'spiral']
    available_shapes = ['semicircle', 'line', 'circle', 'ellipse']

    for song_dir in song_dirs:
        song_name = os.path.basename(song_dir.strip('/'))
        output_path = os.path.join(traj_dir, f"{song_name}.npy")
        
        current_shape = shape
        if shape == 'random':
            current_shape = random.choice(available_shapes)
        
        command = [
            "python", "../traj_funcs.py",
            "--shape", current_shape,
            "--samples", "150",
            "--out", output_path
        ]
        
        # Add random parameters for the chosen shape to make them varied
        if current_shape in ['circle', 'semicircle', 'spiral', 'curve', 'curve2']:
            command.extend(["--radius", f"{random.uniform(0.5, 1.5):.2f}"])
        if current_shape in ['square', 'triangle']:
            command.extend(["--side_length", f"{random.uniform(0.8, 2.0):.2f}"])
        if current_shape in ['line', 'spiral', 'curve', 'curve2']:
            command.extend(["--step_length", f"{random.uniform(0.01, 0.05):.2f}"])
        
        print(f"Generating trajectory for {song_name}... (Shape: {current_shape})")
        
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            print(result.stdout.strip())
        except subprocess.CalledProcessError as e:
            print(f"Error generating trajectory for {song_name}:")
            print(e.stderr)
            
    print("\nTrajectory generation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate trajectories for all test set songs.")
    parser.add_argument("--feature_dir", type=str, default="../../filtered_dataset/jukebox_feats_rectified/", 
                        help="Directory containing the cached features, with one subdirectory per song.")
    parser.add_argument("--traj_dir", type=str, default="../../filtered_dataset/trajectories_per_slice/", 
                        help="Directory where the generated .npy trajectory files will be saved.")
    parser.add_argument("--shape", type=str, default="random", 
                        choices=['semicircle', 't_shape', 'triangle', 'square', 'line', 'curve', 'curve2', 'circle', 'ellipse', 'spiral', 'random'],
                        help="The shape of the trajectories to generate. 'random' will pick a random shape for each song.")
    
    args = parser.parse_args()
    main(args)
