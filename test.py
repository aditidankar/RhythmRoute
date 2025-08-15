import glob
import os
from functools import cmp_to_key
from pathlib import Path
from tempfile import TemporaryDirectory
import random

import jukemirlib
import numpy as np
import torch
from tqdm import tqdm

from args import parse_test_opt
from data.slice import slice_audio
from EDGE import EDGE
from data.audio_extraction.baseline_features import extract as baseline_extract
from data.audio_extraction.jukebox_features import extract as juke_extract

# sort filenames that look like songname_slice{number}.ext
key_func = lambda x: int(os.path.splitext(x)[0].split("_")[-1].split("slice")[-1])


def stringintcmp_(a, b):
    aa, bb = "".join(a.split("_")[:-1]), "".join(b.split("_")[:-1])
    ka, kb = key_func(a), key_func(b)
    if aa < bb:
        return -1
    if aa > bb:
        return 1
    if ka < kb:
        return -1
    if ka > kb:
        return 1
    return 0


stringintkey = cmp_to_key(stringintcmp_)


def test(opt):
    feature_func = juke_extract if opt.feature_type == "jukebox" else baseline_extract
    sample_length = opt.out_length
    sample_size = int(sample_length / 2.5) - 1

    temp_dir_list = []
    all_cond = []
    all_filenames = []
    if opt.use_cached_features: # Updated - no sampling logic
        print("Using precomputed features")
        # iterate through each slice directory and load all features
        slice_dirs = sorted(glob.glob(os.path.join(opt.feature_cache_dir, "*/")))
        for slice_dir in tqdm(slice_dirs, desc="Processing slices"):
            file_list  = sorted(glob.glob(f"{slice_dir}/*.wav"), key=stringintkey)
            juke_file_list = sorted(glob.glob(f"{slice_dir}/*.npy"), key=stringintkey)
            # Sanity check to ensure the number of audio files matches the number of feature files
            if len(file_list) != len(juke_file_list):
                print(f"Warning: Mismatch in number of audio files and feature files in {slice_dir}")
                continue
            # Load features
            juke_list = [np.load(f) for f in juke_file_list]
            juke_cond = torch.from_numpy(np.array(juke_list))
            num_slices = len(juke_list)
            
            traj_files = []
            song_name = os.path.basename(slice_dir.strip('/'))
            all_found = True
            for f in juke_file_list:
                slice_basename = os.path.basename(f)
                
                # Path for nested structure (e.g., .../trajectories_sliced/song_name/slice.npy)
                nested_path = os.path.join(opt.trajectory_dir, song_name, slice_basename)
                # Path for flat structure (e.g., .../trajectories_per_slice/slice.npy)
                flat_path = os.path.join(opt.trajectory_dir, slice_basename)

                if os.path.exists(nested_path):
                    traj_files.append(nested_path)
                elif os.path.exists(flat_path):
                    traj_files.append(flat_path)
                else:
                    print(f"Warning: Trajectory file not found for slice {slice_basename} in both nested and flat structures.")
                    all_found = False
                    break 
            
            if not all_found:
                print(f"Warning: Not all trajectory files found for slices in {slice_dir}, skipping...")
                continue
            
            # Load all corresponding trajectory slices
            trajectory_list = [torch.from_numpy(np.load(f)).float() for f in traj_files]
            trajectory_cond = torch.stack(trajectory_list, dim=0)

            
            cond_list = {
                "music": juke_cond,
                "trajectory": trajectory_cond,
            }
            
            all_filenames.append(file_list)
            all_cond.append(cond_list)
    else:
        print("Computing features for input music")
        for wav_file in glob.glob(os.path.join(opt.music_dir, "*.wav")):
            songname = os.path.splitext(os.path.basename(wav_file))[0]
            
            # Load corresponding trajectory for the song
            traj_filename = os.path.join(opt.trajectory_dir, songname + ".npy")
            if not os.path.exists(traj_filename):
                print(f"Warning: Trajectory file not found for {wav_file}, skipping...")
                continue
            
            trajectory_data = torch.from_numpy(np.load(traj_filename)).float()

            # create temp folder (or use the cache folder if specified)
            if opt.cache_features:
                save_dir = os.path.join(opt.feature_cache_dir, songname)
                Path(save_dir).mkdir(parents=True, exist_ok=True)
                dirname = save_dir
            else:
                temp_dir = TemporaryDirectory()
                temp_dir_list.append(temp_dir)
                dirname = temp_dir.name

            # slice the audio file
            print(f"Slicing {wav_file}")
            slice_audio(wav_file, 2.5, 5.0, dirname)
            file_list = sorted(glob.glob(f"{dirname}/*.wav"), key=stringintkey)

            # randomly sample a chunk of length at most sample_size
            if len(file_list) < sample_size:
                rand_idx = 0
                num_slices_to_use = len(file_list)
            else:
                rand_idx = random.randint(0, len(file_list) - sample_size)
                num_slices_to_use = sample_size
            
            files_to_use = file_list[rand_idx : rand_idx + num_slices_to_use]
            music_cond_list = []
            
            # generate juke representations
            print(f"Computing features for {len(files_to_use)} slices of {wav_file}")
            for file in tqdm(files_to_use):
                reps, _ = feature_func(file)
                # save reps
                if opt.cache_features:
                    featurename = os.path.splitext(file)[0] + ".npy"
                    np.save(featurename, reps)
                music_cond_list.append(reps)
            
            if not music_cond_list:
                continue

            juke_cond = torch.from_numpy(np.array(music_cond_list))
            
            # Load corresponding trajectory slices for the selected audio slices
            trajectory_cond_list = []
            for file in files_to_use:
                slice_basename = os.path.basename(file).replace('.wav', '.npy')
                song_name = os.path.splitext(os.path.basename(wav_file))[0]

                # Path for nested structure (e.g., .../trajectories_sliced/song_name/slice.npy)
                nested_path = os.path.join(opt.trajectory_dir, song_name, slice_basename)
                # Path for flat structure (e.g., .../trajectories_per_slice/slice.npy)
                flat_path = os.path.join(opt.trajectory_dir, slice_basename)
                
                traj_filename = None
                if os.path.exists(nested_path):
                    traj_filename = nested_path
                elif os.path.exists(flat_path):
                    traj_filename = flat_path

                if traj_filename is None:
                    print(f"Warning: Trajectory slice file not found for {file}, skipping this slice...")
                    continue
                
                trajectory_data = torch.from_numpy(np.load(traj_filename)).float()
                trajectory_cond_list.append(trajectory_data)

            if not trajectory_cond_list:
                continue

            trajectory_cond = torch.stack(trajectory_cond_list, dim=0)

            
            final_cond = {
                "music": juke_cond,
                "trajectory": trajectory_cond,
            }

            all_cond.append(final_cond)
            all_filenames.append(files_to_use)

    model = EDGE(
        opt.feature_type,
        checkpoint_path=opt.checkpoint,
        is_training=False,
        )
    model.eval()

    # directory for optionally saving the dances for eval
    fk_out = None
    if opt.save_motions:
        fk_out = opt.motion_save_dir

    print("Generating dances")
    for i in range(len(all_cond)):
        data_tuple = None, all_cond[i], all_filenames[i]
        model.render_sample(
            data_tuple, "test", opt.render_dir, render_count=-1, fk_out=fk_out, render=not opt.no_render
        )
    print("Done")
    torch.cuda.empty_cache()
    for temp_dir in temp_dir_list:
        temp_dir.cleanup()


if __name__ == "__main__":
    opt = parse_test_opt()
        
    # --- ADD THIS BLOCK TO FIX THE DISTRIBUTED INITIALIZATION ERROR ---
    if torch.cuda.is_available():
        # Set up for a single-process, single-GPU distributed environment.
        # This is a common workaround for models that expect a distributed
        # setup even for single-GPU inference.
        os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
        os.environ.setdefault('MASTER_PORT', '12355') # Use any free port
        torch.distributed.init_process_group(
            backend='nccl',  # 'nccl' is recommended for NVIDIA GPUs
            init_method='env://',
            world_size=1,
            rank=0
        )
    # --- END OF BLOCK ---
    
    test(opt)
