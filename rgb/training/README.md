# RGB & Fusion Model Training

Train and evaluate feature-based (CLIP, DINOv3, VideoMAE) and multimodal
fusion (feature + skeleton) models.

## Overview

This code trains temporal classifiers on **pre-extracted** visual features
(not raw video). Features are extracted separately using
`rgb/feature_extraction/` and stored as per-video `.npy` files.
Pre-extracted features can be downloaded from [this link](https://drive.google.com/drive/folders/1lB3SWCtetAkRIi-6Arv6d3dh1jecRWaF?usp=sharing).
Pretrained RGB-only and fusion model weights can be downloaded from [this link](https://drive.google.com/drive/folders/1V8waixwypqlMf3WUuLhc2Uozjn5NPSaH?usp=drive_link).

The training pipeline matches `keypoints_train/` (same sliding-window
sampling, splits, and evaluation).

## Model Types

Feature-only (RGB) models use mean pooling or a temporal Transformer over
pre-extracted features. Fusion models combine visual features with skeleton
keypoints using the following strategies:

| Type              | Strategy                                                         | Reference                                                              |
| ----------------- | ---------------------------------------------------------------- | ---------------------------------------------------------------------- |
| `fusion`        | Concat: MeanPool (visual) + CNN1D (keypoints) + linear           | —                                                                     |
| `robust_fusion` | Concat + ModDrop + auxiliary pose head                           | [Neverova et al., TPAMI 2016](https://doi.org/10.1109/TPAMI.2015.2461544) |
| `qmf_fusion`    | Energy-based dynamic weighting with per-sample history           | [Zhang &amp; Wu, ICML 2023](https://arxiv.org/abs/2306.02050)             |
| `ogm_fusion`    | On-the-fly gradient modulation                                   | [Peng et al., CVPR 2022](https://arxiv.org/abs/2203.15332)                |
| `mmcl_fusion`   | Contrastive co-learning (keypoint classifier + visual alignment) | [Liu et al., ACM MM 2024](https://arxiv.org/abs/2407.15706)               |

Tiny variants (`fusion_tiny`, `robust_fusion_tiny`) use a smaller keypoint
encoder and are used for the wheelchair dataset.

## Directory Structure

```
rgb/training/
  train.py                  # Main training script
  models.py                 # All model architectures
  dataset.py                # Dataset loaders (keypoint / feature / fusion)
  transforms.py             # Keypoint augmentations
  qmf_utils.py              # QMF per-sample loss tracking
  infer_pkl.py              # Single-process sliding-window inference
  spawn_multiple_infer_pkl.py  # Parallel inference orchestrator
  configs/
    normal/                 # 18 configs for in-lab dataset
    wheelchair/             # 18 configs for wheelchair dataset
  scripts/                  # Example train + eval pipelines
```

## Quick Start

Commands assume you are in `rgb/training/`.

### 1. Train a model

```bash
# RGB-only: VideoMAE linear head on normal dataset
python train.py --config configs/normal/videomae_linear.py

# Fusion: DINOv3 + CNN1D concat on wheelchair dataset
python train.py --config configs/wheelchair/wheelchair_dinov3_cnn1d_fusion.py
```

Training saves checkpoints and logs to `runs/<config_name>_seq/`.

### 2. Evaluate on test set

After training, run inference on the test split using parallel workers:

```bash
python spawn_multiple_infer_pkl.py \
    --run_dir runs/normal_videomae_linear_seq \
    --pkl_path /home/work/pyskl/Pkl/aic_normal_dataset_with_3d_480p.pkl \
    --out_dict_dir ./eval_results/videomae_linear \
    --total_splits 3 \
    --feature_dir /home/work/rgb/feature_extraction/features/normal_videomae
```

Then compute accuracy and per-class metrics:

```bash
python ../../inference/calculate_accuracy.py \
    --pkl_path ./eval_results/videomae_linear/result.pkl \
    --out_accuracy_results ./eval_results/videomae_linear
```

### 3. Evaluate on non-lab (OOD) test set

Same inference command but with the non-lab pkl and features:

```bash
python spawn_multiple_infer_pkl.py \
    --run_dir runs/normal_videomae_linear_seq \
    --pkl_path /home/work/pyskl/Pkl/aic_normal_test_set_with_split_3d_normalized.pkl \
    --out_dict_dir ./eval_results_nonlab/videomae_linear \
    --total_splits 3 \
    --feature_dir /home/work/rgb/feature_extraction/features/nonlab_test_videomae

python ../../inference/calculate_accuracy.py \
    --pkl_path ./eval_results_nonlab/videomae_linear/result.pkl \
    --out_accuracy_results ./eval_results_nonlab/videomae_linear
```

### 4. Example pipeline scripts

The `scripts/` directory has end-to-end examples that train and evaluate:

```bash
# Normal dataset, RGB-only (VideoMAE) — evaluates in-lab + non-lab
bash scripts/train_eval_normal_rgb_only.sh

# Normal dataset, fusion (VideoMAE + CNN1D) — evaluates in-lab + non-lab
bash scripts/train_eval_normal_fusion.sh

# Wheelchair dataset, RGB-only
bash scripts/train_eval_wheelchair_rgb_only.sh

# Wheelchair dataset, fusion
bash scripts/train_eval_wheelchair_fusion.sh

# Skip training, run eval only:
bash scripts/train_eval_normal_rgb_only.sh --eval-only
```

## Configs

Each config is a plain Python file (no imports) with four dicts:

- `dataset_cfg`: mode, annotation file, feature directory, splits, preprocessing
- `model_cfg`: model type and architecture arguments
- `train_cfg`: epochs, batch size, scheduler, output directory
- `optimizer_cfg`: optimizer type (adam/adamw/sgd), learning rate, weight decay

Config names follow the pattern:
`{backbone}_{head}.py` for feature-only, `{backbone}_{fusion_type}_fusion.py` for fusion.

Wheelchair configs are prefixed with `wheelchair_`.

## Requirements

All dependencies are included in the project Docker images
(`docker/Dockerfile` or `docker/cuda12_setup/Dockerfile`). Key packages:

- PyTorch, numpy, scikit-learn, matplotlib, seaborn, pandas
- `pyskl` (installed as editable on container startup)
- `yapf` (required by mmcv config parser)
