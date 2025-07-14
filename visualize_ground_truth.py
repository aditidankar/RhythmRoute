# visualize_ground_truth.py
import numpy as np
import os
import torch

# --- Import the necessary tools from the EDGE codebase ---
from vis import skeleton_render, SMPLSkeleton
from dataset.quaternion import ax_from_6v

# --- Configuration ---
POSES_FILE = 'gt_poses_example.npy'
TRAJECTORY_FILE = 'gt_trajectory_example.npy'
OUTPUT_DIR = "renders/ground_truth_visualization"

# --- Main Script ---
print("--- Starting Ground Truth Visualization ---")

try:
    # 1. Load the raw pose feature vectors
    print(f"Loading pose features from '{POSES_FILE}'...")
    pose_feature_vectors = np.load(POSES_FILE) # Shape is (150, 151)
    
    # 2. Load the root trajectory
    print(f"Loading trajectory from '{TRAJECTORY_FILE}'...")
    root_trajectory = np.load(TRAJECTORY_FILE) # Shape is (150, 3)
    
    print(f"\nLoaded pose features with shape: {pose_feature_vectors.shape}")
    print(f"Loaded trajectory with shape: {root_trajectory.shape}")

except FileNotFoundError:
    print("\nERROR: Ground truth data files not found.")
    print("Please run the training script to generate them first.")
    exit()

# --- DECODE THE DATA AND RECONSTRUCT THE SKELETON ---
print("\nReconstructing 3D skeleton from loaded data...")

# A. Prepare the Rotations
# Convert pose features to a PyTorch Tensor
pose_features_tensor = torch.from_numpy(pose_feature_vectors).float()
# Extract just the rotation data (the last 144 components of the vector)
rotations_6d = pose_features_tensor[:, 7:]
# Reshape from a flat (150, 144) to a structured (150, 24, 6)
rotations_6d = rotations_6d.reshape(-1, 24, 6)
# Convert the 6D rotations back to the axis-angle format needed for FK
rotations_ax = ax_from_6v(rotations_6d)

# B. Prepare the Trajectory
# Convert the trajectory to a PyTorch Tensor
root_pos_tensor = torch.from_numpy(root_trajectory).float()

# C. Perform Forward Kinematics (FK)
print("Performing Forward Kinematics...")
smpl = SMPLSkeleton()
# The FK function expects a batch dimension, so we add one with .unsqueeze(0)
final_poses = smpl.forward(rotations_ax.unsqueeze(0), root_pos_tensor.unsqueeze(0))
# Remove the batch dimension for rendering and convert back to NumPy
final_poses = final_poses.squeeze(0).detach().cpu().numpy()

print(f"Successfully reconstructed skeleton. Final shape: {final_poses.shape}")

# --- RENDER THE ANIMATION ---
output_name = os.path.join(OUTPUT_DIR, "ground_truth_dance")
print(f"\nRendering animation to '{OUTPUT_DIR}'...")

# Now, call the rendering function with the correctly formatted poses
skeleton_render(
    poses=final_poses, # Use the (150, 24, 3) final positions
    epoch="gt",
    out=OUTPUT_DIR,
    name=output_name,
    sound=False,
    contact=None, # We don't have contact data, so we leave this as None
    render=True
)

print(f"\nSuccessfully rendered ground truth animation.")
print(f"Check the '{OUTPUT_DIR}' directory for a .gif file.")