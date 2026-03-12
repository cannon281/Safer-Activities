# ImViA Cross-Dataset Fall Detection (RGB & Fusion)

Evaluate RGB-only and multimodal fusion models trained on SAFER-Activities
on the ImViA fall detection dataset (Charfi et al., 130 videos, 99 falls).

For skeleton-only models (CNN1D, PoseC3D), use `inference/infer_imvia.py`
and `inference/infer_imvia_pyskl.py` instead.

## Pipeline

| Step | Script                        | Description                                                        |
| ---- | ----------------------------- | ------------------------------------------------------------------ |
| 1    | `step1_build_pkl.py`        | Extract ViTPose keypoints from all 130 videos                      |
| 2    | `step2_extract_features.sh` | Extract DINOv3, VideoMAE, CLIP features (on-the-fly bbox cropping) |
| 3    | `step3_run_inference.sh`    | Run 6 trained models (3 RGB-only + 3 concat fusion)                |
| 4    | `step4_results_to_csv.py`   | Convert `result.pkl` to per-video CSVs (binary fall/non\_fall)   |
| 5    | `step5_evaluate.sh`         | Cluster-based fall event F1 evaluation                             |

## Quick Start

```bash
cd rgb/external_eval/imvia

# Step 1: Build pkl (GPU for ViTPose)
python step1_build_pkl.py --imvia_root /path/to/ImViA

# Step 2: Extract features (GPU)
bash step2_extract_features.sh --imvia_root /path/to/ImViA

# Step 3: Run inference
bash step3_run_inference.sh

# Step 4: Convert results to CSV
python step4_results_to_csv.py

# Step 5: Evaluate
bash step5_evaluate.sh
```

## Models Evaluated

| Group  | Model                   | Run Directory                        |
| ------ | ----------------------- | ------------------------------------ |
| RGB    | CLIP MeanPool           | `normal_clip_meanpool_seq`         |
| RGB    | DINOv3 MeanPool         | `normal_dinov3_meanpool_seq`       |
| RGB    | VideoMAE Linear         | `normal_videomae_linear_seq`       |
| Fusion | CLIP + CNN1D Concat     | `normal_clip_cnn1d_fusion_seq`     |
| Fusion | DINOv3 + CNN1D Concat   | `normal_dinov3_cnn1d_fusion_seq`   |
| Fusion | VideoMAE + CNN1D Concat | `normal_videomae_cnn1d_fusion_seq` |

## Prerequisites

- ImViA dataset (download from the [ImViA website](https://imvia.u-bourgogne.fr/))
- ViTPose + YOLOv8 weights in `inference/det_pose_models/`
- Trained model checkpoints in `rgb/training/runs/`
