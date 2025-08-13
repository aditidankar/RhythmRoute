import argparse
import glob
import os
import pickle

import numpy as np
from tqdm import tqdm


def calc_trajectory_error(dir):
    errors = []
    
    it = glob.glob(os.path.join(dir, "*.pkl"))
    if not it:
        print(f"No .pkl files found in directory: {dir}")
        return

    for pkl_file in tqdm(it):
        with open(pkl_file, "rb") as f:
            data = pickle.load(f)

        if "smpl_trans" not in data or "gt_trajectory" not in data:
            print(f"Skipping {pkl_file}: does not contain required 'smpl_trans' or 'gt_trajectory' keys.")
            continue

        pred_traj = data["smpl_trans"]
        gt_traj = data["gt_trajectory"]
        
        # Ensure trajectories have the same length
        min_len = min(len(pred_traj), len(gt_traj))
        pred_traj = pred_traj[:min_len]
        gt_traj = gt_traj[:min_len]

        # Calculate Euclidean distance for each frame
        frame_errors = np.linalg.norm(pred_traj - gt_traj, axis=-1)
        
        # Calculate the mean error for the sequence
        mean_error = np.mean(frame_errors)
        errors.append(mean_error)

    if not errors:
        print("Could not calculate error for any files.")
        return

    total_mean_error = np.mean(errors)
    print(f"\nMean Trajectory Error over {len(errors)} samples: {total_mean_error:.4f}")


def parse_eval_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--motion_path",
        type=str,
        default="motions/",
        help="Where to load saved motions (.pkl files)",
    )
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_eval_opt()
    calc_trajectory_error(opt.motion_path)
