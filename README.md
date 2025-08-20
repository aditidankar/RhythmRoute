## RhythmRoute: Trajectory-Conditioned Dance Generation from Music

### Abstract
RhythmRoute is a method for trajectory-conditioned dance generation from music. It builds upon **EDGE: Editable Dance Generation From Music** (CVPR 2023), a state-of-the-art music-to-dance model. RhythmRoute enhances EDGE by allowing users to specify a 3D trajectory for the dancer to follow, providing fine-grained control over the generated motion's spatial path. This is achieved by conditioning a transformer-based diffusion model on both powerful music features from Jukebox and the desired 3D trajectory data. The model can create realistic, physically-plausible dances that are synchronized with the music while adhering to the user-defined path, enabling new creative possibilities for choreographers and artists.

## Requirements
*   We recommend Linux for performance and compatibility reasons. Windows is not officially supported.
*   64-bit Python 3.8+ (developed on Python 3.8.12)
*   PyTorch 1.13.1
*   1–8 high-end NVIDIA GPUs with at least 16 GB of GPU memory.
*   NVIDIA drivers with CUDA 11.6 support.

The example build this repo was validated on:
*   OS: `Linux 5.14.0-427.22.1.el9_4.x86_64`
*   Python: `3.8.12`
*   PyTorch: `1.13.1` (with `torchvision==0.14.1`, `torchaudio==0.13.1`)
*   GPU: 1 x NVIDIA A100-PCIE-40GB
<!-- *   Note: Detailed CPU, System RAM, and driver versions can be added here.* -->

This repository additionally depends on the following libraries, which may require special installation procedures:
* [jukemirlib](https://github.com/rodrigo-castellon/jukemirlib)
* [pytorch3d](https://github.com/facebookresearch/pytorch3d)
* [accelerate](https://huggingface.co/docs/accelerate/v0.19.0/en/index) (v0.19.0 used in development)
	* Note: after installation, don't forget to run `accelerate config` . We use fp16.

## Getting started
### Installation
Clone the repository and install the required packages:
```.bash
git clone https://github.com/aditidankar/RhythmRoute.git
cd RhythmRoute
pip install -r requirements.txt
```

### Quickstart (Inference)
* Download the saved model checkpoint from [Google Drive](https://drive.google.com/file/d/1BAR712cVEqB8GR37fcEihRV_xOC-fZrZ/view?usp=share_link) or by running `bash download_model.sh`. This will download the original EDGE checkpoint, which can be used as a base if you are training your own model. A fine-tuned RhythmRoute checkpoint will be provided separately.

### Inference on Custom Music
To generate a dance for a custom audio file, you need to provide both the music and a target trajectory. The process is as follows:

**1. Prepare Music & Extract Features**
First, place your custom music files (in `.wav` format) into a directory, e.g., `custom_music/`. It's recommended to use simple filenames without spaces or special characters.

Then, run the `test.py` script to extract and cache the Jukebox music features. This step is necessary to determine the number of frames for which a trajectory is needed.

```.bash
python test.py --music_dir custom_music/ --cache_features --feature_cache_dir cached_features/ --no_render
```
This will process the audio in `custom_music/` and save the feature slices into `cached_features/`.

**2. Generate Trajectory Data**
With the music features extracted, you can now generate corresponding trajectories.

*   **Step 2a: Generate Full Trajectories**
    This script creates a single, continuous trajectory for each song, with a total length matching the full duration of the sliced music. It can generate various shapes like lines or semi-circles.
    ```.bash
    python generate_test_trajectory/generate_full_trajectories.py --feature_dir cached_features/ --traj_dir data/trajectories_full --shape random
    ```
*   **Step 2b: Slice Full Trajectories**
    This script slices the full trajectories into 150-frame segments, corresponding to each music feature slice.
    ```.bash
    python generate_test_trajectory/slice_trajectories.py --full_traj_dir data/trajectories_full --sliced_traj_dir data/trajectories_sliced --feature_dir cached_features/
    ```

**3. Generate the Dance**
Finally, run the generation script using the cached music features and the newly created sliced trajectories.
```.bash
python test.py --checkpoint your_model.pt --use_cached_features --feature_cache_dir cached_features/ --trajectory_dir data/trajectories_sliced/
```
The generated dances will be saved in the `renders/` directory by default.

### (Optional, retraining only) Dataset Download
This project uses the AIST++ dataset. You can download the raw data (motion and audio) from the official website:
**[AIST Dance Video Database](https://aistdancedb.ongaaccel.jp/database_download/)** 

After downloading, you will need to process the data to prepare it for training. This involves creating motion and audio slices and extracting music features:
```.bash
python create_dataset.py --extract-baseline --extract-jukebox
```
This will process the dataset to match the settings used in the paper. The data processing will take a significant amount of time (~24 hrs) and disk space (~50 GB) to precompute all the Jukebox features. The trajectories are generated automatically from the motion data during this process.

### Train your own model
Once the AIST++ dataset is downloaded and processed, run the training script:
```.bash
accelerate launch train.py --batch_size 128  --epochs 2000 --feature_type jukebox
```
to train the model. The training will log progress to `wandb` and intermittently produce sample outputs to visualize learning.

### Evaluate your model
You can evaluate your model's outputs using two metrics: the Physical Foot Contact (PFC) score from the original paper, and a new Trajectory Error metric.

**1. Generate Motions for Evaluation**

First, generate dance motions and save the output `.pkl` files, which contain the necessary joint positions and trajectories. Make sure to provide the ground truth trajectory data for comparison.
```.bash
python test.py --checkpoint your_model.pt --use_cached_features --feature_cache_dir cached_features/ --trajectory_dir data/trajectories_sliced/ --save_motions --motion_save_dir eval/motions
```

**2. Run Evaluation Scripts**

*   **Physical Foot Contact (PFC) Score:** This metric measures the physical plausibility of the generated dance.
    ```.bash
    python eval/eval_pfc.py --motion_path eval/motions
    ```
*   **Mean Trajectory Error:** This metric measures how closely the generated dance follows the input trajectory. It computes the mean Euclidean distance between the predicted root position and the ground truth trajectory.
    ```.bash
    python eval/eval_traj.py --motion_path eval/motions
    ```

## Blender 3D rendering
In order to render generated dances in 3D, we convert them into FBX files to be used in Blender. We provide a sample rig, `SMPL-to-FBX/ybot.fbx`.
After generating dances with the `--save-motions` flag enabled, move the relevant saved `.pkl` files to a folder, e.g. `smpl_samples`
Run
```.bash
python SMPL-to-FBX/Convert.py --input_dir SMPL-to-FBX/smpl_samples/ --output_dir SMPL-to-FBX/fbx_out
```
to convert motions into FBX files, which can be imported into Blender and retargeted onto different rigs, i.e. from [Mixamo](https://www.mixamo.com). A variety of retargeting tools are available, such as the [Rokoko plugin for Blender](https://www.rokoko.com/integrations/blender).

## Development
This project is currently under active development.


## Note:
**This project is built upon EDGE: Editable Dance Generation From Music (CVPR 2023).**<br>
Original Authors: Jonathan Tseng, Rodrigo Castellon, C. Karen Liu<br>
https://arxiv.org/abs/2211.10658

```
@article{tseng2022edge,
  title={EDGE: Editable Dance Generation From Music},
  author={Tseng, Jonathan and Castellon, Rodrigo and Liu, C Karen},
  journal={arXiv preprint arXiv:2211.10658},
  year={2022}
}
```

<!-- ## Citation

## Acknowledgements
We would like to thank [lucidrains](https://github.com/lucidrains) for the [Adan](https://github.com/lucidrains/Adan-pytorch) and [diffusion](https://github.com/lucidrains/denoising-diffusion-pytorch) repos, [softcat477](https://github.com/softcat477) for their [SMPL to FBX](https://github.com/softcat477/SMPL-to-FBX) library, and [BobbyAnguelov](https://github.com/BobbyAnguelov) for their [FBX Converter tool](https://github.com/BobbyAnguelov/FbxFormatConverter). -->
