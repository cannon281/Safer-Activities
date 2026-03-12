# 3D Pose Lifting

Lift 2D ViTPose keypoints (COCO 17-joint) to 3D using [MotionAGFormer](https://github.com/TaatiTeam/MotionAGFormer) (Mehraban et al., WACV 2024). The output pkl files are used by the 3D skeleton action recognition models (DG-STGCN, MSG3D, STGCN++).

## Setup

### 1. Clone MotionAGFormer

```bash
git clone https://github.com/TaatiTeam/MotionAGFormer.git
cd MotionAGFormer
pip install -r requirements.txt
```

### 2. Download checkpoint

Download the **MotionAGFormer-Base** checkpoint trained on Human3.6M (243 frames):

- `motionagformer-b-h36m.pth.tr` from the [MotionAGFormer model zoo](https://github.com/TaatiTeam/MotionAGFormer#model-zoo)

Place it in `MotionAGFormer/checkpoint/`.

### 3. Copy the batch processor

Copy `batch_3d_pose_processor.py` from this directory into the MotionAGFormer repo root:

```bash
cp batch_3d_pose_processor.py /path/to/MotionAGFormer/
```

## Usage

Run from the MotionAGFormer repo root:

```bash
# Non-wheelchair dataset
python batch_3d_pose_processor.py \
    --input_pkl /path/to/Pkl/aic_normal_dataset.pkl \
    --output_pkl /path/to/Pkl/aic_normal_dataset_with_3d.pkl \
    --model_path checkpoint/motionagformer-b-h36m.pth.tr \
    --num_gpus 1

# Wheelchair dataset
python batch_3d_pose_processor.py \
    --input_pkl /path/to/Pkl/aic_wheelchair_dataset.pkl \
    --output_pkl /path/to/Pkl/aic_wheelchair_dataset_with_3d.pkl \
    --model_path checkpoint/motionagformer-b-h36m.pth.tr \
    --num_gpus 1

# Non-lab test set (for OOD evaluation)
python batch_3d_pose_processor.py \
    --input_pkl /path/to/Pkl/aic_normal_test_set_with_split.pkl \
    --output_pkl /path/to/Pkl/aic_normal_test_set_with_split_3d.pkl \
    --model_path checkpoint/motionagformer-b-h36m.pth.tr \
    --num_gpus 1
```


## What it does

1. Loads 2D keypoints from the input pkl (COCO 17-joint format)
2. Converts COCO joint ordering to Human3.6M format
3. Segments each video into non-overlapping 243-frame clips (shorter clips are resampled)
4. Runs MotionAGFormer with test-time augmentation (horizontal flip)
5. Outputs hip-relative 3D coordinates

The output pkl retains all original fields and adds:

- `keypoint_3d`: shape `(num_people, num_frames, 17, 3)` — hip-relative 3D coordinates
- `keypoint_3d_score`: shape `(num_people, num_frames)` — mean 2D confidence per frame

## Output pkl files

Place the generated pkl files in `inference/pyskl/Pkl/` for use with the 3D training configs:

| Input pkl | Output pkl | Used by |
|-----------|-----------|---------|
| `aic_normal_dataset.pkl` | `aic_normal_dataset_with_3d.pkl` | DG-STGCN/MSG3D/STGCN++ 3D non-wheelchair configs |
| `aic_wheelchair_dataset.pkl` | `aic_wheelchair_dataset_with_3d.pkl` | DG-STGCN/MSG3D/STGCN++ 3D wheelchair configs |
| `aic_normal_test_set_with_split.pkl` | `aic_normal_test_set_with_split_3d.pkl` | Non-lab OOD evaluation |
