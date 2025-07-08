# visualize_ground_truth.py
import numpy as np
import os

# Import the rendering function from the EDGE codebase
from vis import skeleton_render

# --- Configuration ---
POSES_FILE = 'gt_poses_example.npy'
TRAJECTORY_FILE = 'gt_trajectory_example.npy'
OUTPUT_DIR = "renders/ground_truth_visualization"

# --- Main Script ---
print("--- Starting Ground Truth Visualization ---")

try:
    # Load the ground truth pose and trajectory data
    print(f"Loading poses from '{POSES_FILE}'...")
    poses = np.load(POSES_FILE)
    print(f"Loading trajectory from '{TRAJECTORY_FILE}'...")
    trajectory = np.load(TRAJECTORY_FILE)
    
    print(f"\nLoaded poses with shape: {poses.shape}")
    print(f"Loaded trajectory with shape: {trajectory.shape}")
    # Expected shapes: (Sequence Length, 24, 3) and (Sequence Length, 3)

except FileNotFoundError:
    print("\nERROR: Ground truth data files not found.")
    print("Please run the training script with --force_reload to generate them first.")
    exit()

# The skeleton_render function requires the root trajectory to be plotted separately.
# We will modify the vis.py script to handle this. For now, let's just render the skeleton.

# We will create a dummy "name" for the output file
output_name = os.path.join(OUTPUT_DIR, "ground_truth_dance")

print(f"\nRendering animation. The output will be saved in the '{OUTPUT_DIR}' directory.")
print("This may take a moment...")

# Call the rendering function with the loaded ground truth poses
skeleton_render(
    poses=poses,
    epoch=0, # Just a label for the filename
    out=OUTPUT_DIR,
    name=output_name,
    sound=False, # We don't have the corresponding audio, so we set this to False
    stitch=False,
    contact=None,
    render=True
)

print(f"\nSuccessfully rendered ground truth animation.")
print(f"Check the '{OUTPUT_DIR}' directory for a .gif file named '{os.path.basename(output_name)}.gif'.")