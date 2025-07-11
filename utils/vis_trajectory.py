import argparse
import glob
import os
import pickle
import random
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict
import sys

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Add parent directory too

from vis import SMPLSkeleton, skeleton_render
from pytorch3d.transforms import (RotateAxisAngle, axis_angle_to_quaternion,
                                  quaternion_multiply,
                                  quaternion_to_axis_angle)

def visualize_trajectory(dir):
    scores = []
    names = []
    genres = defaultdict(list)
    accelerations = []
    up_dir = 2  # z is up
    flat_dirs = [i for i in range(3) if i != up_dir]
    DT = 1 / 30
    smpl = SMPLSkeleton()

    it = glob.glob(os.path.join(dir, "*.pkl"))

    # each genre sample 100
    it_sampled = []
    for pkl in it:
        file_name = pkl.split("/")[-1].split(".")[0]
        genre = file_name.split("_")[0]
        genres[genre].append(pkl)
    
    for genre in genres:
        it_sampled.extend(random.sample(genres[genre], 5))

    for pkl in tqdm(it_sampled):
        info = pickle.load(open(pkl, "rb"))

        if "full_pose" in info.keys():
            joint3d = info["full_pose"]
        else:
            # for GT pkl
            # down-sampling the frames as gt is 60fps and edge is 30fps
            root_pos = torch.Tensor(info["pos"][0::2, :])
            local_q = torch.Tensor(info["q"][0::2, :])
            scale = info["scale"][0]
            
            # to axhe 
            # root_pos = root_pos.reshape((1, root_pos.shape[0], -1, 3))
            root_pos /= scale
            sq, c = local_q.shape
            local_q = local_q.reshape((1, sq, -1, 3))
            root_pos = root_pos.unsqueeze(0)

            # AISTPP dataset comes y-up - rotate to z-up to standardize against the pretrain dataset
            root_q = local_q[:, :, :1, :]  # sequence x 1 x 3
            root_q_quat = axis_angle_to_quaternion(root_q)
            rotation = torch.Tensor(
                [0.7071068, 0.7071068, 0, 0]
            )  # 90 degrees about the x axis
            root_q_quat = quaternion_multiply(rotation, root_q_quat)
            root_q = quaternion_to_axis_angle(root_q_quat)
            local_q[:, :, :1, :] = root_q

            # don't forget to rotate the root position too 😩
            pos_rotation = RotateAxisAngle(90, axis="X", degrees=True)
            root_pos = pos_rotation.transform_points(
                root_pos
            )  # basically (y, z) -> (-z, y), expressed as a rotation for readability
            
            positions = smpl.forward(local_q, root_pos).detach().cpu().numpy()
            joint3d = positions[0]

        skeleton_render(joint3d, out="renders", name=pkl, sound=False, trajectory=True)
        

def parse_eval_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--motion_path",
        type=str,
        default="motions/",
        help="Where to load saved motions",
    )
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_eval_opt()
    opt.motion_path = "/data/scratch/acw750/Data/aist_plusplus_final/motions"
    opt.motion_path = "/data/scratch/acw750/Data/edge_aistpp/train/motions"
    visualize_trajectory(opt.motion_path)
