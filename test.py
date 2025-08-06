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
import config

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
            wav_list  = sorted(glob.glob(f"{slice_dir}/*.wav"), key=stringintkey)
            feat_list = sorted(glob.glob(f"{slice_dir}/*.npy"), key=stringintkey)
            # Sanity check to ensure the number of audio files matches the number of feature files
            if len(wav_list) != len(feat_list):
                print(f"Warning: Mismatch in number of audio files and feature files in {slice_dir}")
                continue
            # Load features
            cond_list = [np.load(f) for f in feat_list]
            all_filenames.append(wav_list)
            all_cond.append(torch.from_numpy(np.array(cond_list)))
    else:
        print("Computing features for input music")
        for wav_file in glob.glob(os.path.join(opt.music_dir, "*.wav")):
            # create temp folder (or use the cache folder if specified)
            if opt.cache_features:
                songname = os.path.splitext(os.path.basename(wav_file))[0]
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
            rand_idx = random.randint(0, len(file_list) - sample_size)
            cond_list = []
            # generate juke representations
            print(f"Computing features for {wav_file}")
            for idx, file in enumerate(tqdm(file_list)):
                # if not caching then only calculate for the interested range
                if (not opt.cache_features) and (not (rand_idx <= idx < rand_idx + sample_size)):
                    continue
                # audio = jukemirlib.load_audio(file)
                # reps = jukemirlib.extract(
                #     audio, layers=[66], downsample_target_rate=30
                # )[66]
                reps, _ = feature_func(file)
                # save reps
                if opt.cache_features:
                    featurename = os.path.splitext(file)[0] + ".npy"
                    np.save(featurename, reps)
                # if in the random range, put it into the list of reps we want
                # to actually use for generation
                if rand_idx <= idx < rand_idx + sample_size:
                    cond_list.append(reps)
            cond_list = torch.from_numpy(np.array(cond_list))
            all_cond.append(cond_list)
            all_filenames.append(file_list[rand_idx : rand_idx + sample_size])

    model = EDGE(
        opt.feature_type,
        checkpoint_path=opt.checkpoint,
        is_training=False,
        traj_mean_path=config.TRAJ_MEAN_PATH,
        traj_std_path=config.TRAJ_STD_PATH,
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
