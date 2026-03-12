# RGB Feature Extraction

Two-stage pipeline to extract per-frame RGB features from the SAFER-Activities dataset:

1. **Bbox clip extraction** — Crop person-centered 640×480 clips from raw 1080p videos using YOLOv8x bounding boxes from the PKL annotation files
2. **Feature extraction** — Extract frozen backbone features from the bbox clips using CLIP, DINOv3, or VideoMAE

## Requirements

### Python packages

All dependencies are included in the Docker images (`docker/Dockerfile` and `docker/cuda12_setup/Dockerfile`), including `transformers` (for DINOv3 and VideoMAE), OpenAI `CLIP`, and `Pillow`.

If running outside Docker, install the additional packages on top of PyTorch:

```bash
pip install transformers Pillow
pip install git+https://github.com/openai/CLIP.git
```

### Model checkpoints

All model weights are downloaded automatically on first run. No manual checkpoint downloads are required.

| Model                     | Source                                       | Auto-download                         | Access requirements                           |
| ------------------------- | -------------------------------------------- | ------------------------------------- | --------------------------------------------- |
| **CLIP ViT-B/16**   | OpenAI                                       | Yes, via `clip.load()`              | None — public checkpoint (~600 MB)           |
| **VideoMAE Base**   | `MCG-NJU/videomae-base-finetuned-kinetics` | Yes, via HuggingFace `transformers` | None — public model (~350 MB)                |
| **DINOv3 ViT-B/16** | `facebook/dinov3-vitb16-pretrain-lvd1689m` | Yes, via HuggingFace `transformers` | **Gated model** — requires steps below |

### DINOv3 access setup (required)

The DINOv3 model is gated on HuggingFace. Before running DINOv3 feature extraction:

1. Create a HuggingFace account at https://huggingface.co/join
2. Visit the model page at https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m
3. Accept the license agreement (click "Agree and access repository")
4. Create an access token at https://huggingface.co/settings/tokens
5. Log in from the command line:
   ```bash
   pip install huggingface_hub
   huggingface-cli login
   # Paste your access token when prompted
   ```

After login, the model weights will be cached locally on first use (~350 MB). Subsequent runs use the cached weights. (To get approved, could take some time: took us about a day)

### System dependencies

- **ffmpeg** — required for Stage 1 (bbox clip extraction). Already included in the Docker image. If running outside Docker: `apt-get install ffmpeg`
- **CUDA GPU** — required for Stage 2 feature extraction. All three models run on GPU.

## Dataset Splits

The paper uses three dataset splits. Run each stage separately for each split:

| Split           | Description              | Video directory         | PKL file                                |
| --------------- | ------------------------ | ----------------------- | --------------------------------------- |
| `normal`      | In-lab training/test set | `normal/Videos/`      | `aic_normal_dataset_with_3d.pkl`      |
| `nonlab_test` | Non-lab (OOD) test set   | `nonlab_test/Videos/` | `aic_nonlab_test_dataset_with_3d.pkl` |
| `wheelchair`  | Wheelchair subset        | `wheelchair/Videos/`  | `aic_wheelchair_dataset_with_3d.pkl`  |

## Stage 1: Bbox Clip Extraction

Extracts person-centered 640×480 crops from raw videos using bounding boxes stored in the PKL file. Also produces an output PKL that preserves the original 1080p keypoints (`keypoint`) and adds 480p bbox-crop coordinates as `keypoint_480p`.

**Parameters:**

- Output resolution: 640×480 (4:3 aspect ratio)
- Bbox padding: 10% on each side
- Chunk size: 300 frames per clip
- Stride: 156 frames (144-frame overlap between clips)
- Encoding: H.264 (libx264, CRF 28, ultrafast preset)

```bash
# In-lab (normal) split
python extract_bbox_clips.py \
    --video-root /path/to/normal/Videos \
    --clips-root /path/to/BBoxClips/normal \
    --pkl-file /path/to/aic_normal_dataset_with_3d.pkl \
    --out-pkl /path/to/aic_normal_dataset_with_3d_480p.pkl \
    --chunk-size 300 --stride 156 --workers 8

# Non-lab (OOD) split
python extract_bbox_clips.py \
    --video-root /path/to/nonlab_test/Videos \
    --clips-root /path/to/BBoxClips/nonlab_test \
    --pkl-file /path/to/aic_nonlab_test_dataset_with_3d.pkl \
    --out-pkl /path/to/aic_nonlab_test_dataset_with_3d_480p.pkl \
    --chunk-size 300 --stride 156 --workers 8

# Wheelchair split
python extract_bbox_clips.py \
    --video-root /path/to/wheelchair/Videos \
    --clips-root /path/to/BBoxClips/wheelchair \
    --pkl-file /path/to/aic_wheelchair_dataset_with_3d.pkl \
    --out-pkl /path/to/aic_wheelchair_dataset_with_3d_480p.pkl \
    --chunk-size 300 --stride 156 --workers 8
```

**Output structure:**

```
BBoxClips/normal/
├── video_01/
│   ├── clip_000000.mp4    # frames 0-299
│   ├── clip_000156.mp4    # frames 156-455
│   ├── clip_000312.mp4    # frames 312-611
│   └── ...
├── video_02/
│   └── ...
└── ...
```

## Stage 2: Feature Extraction

Extract frozen backbone features from the 480p bbox clips. Requires the remapped 480p PKL file from Stage 1.

### CLIP (ViT-B/16) — 512-dim per frame

Extracts the CLS token from `openai/clip-vit-base-patch16` for each frame independently.

```bash
python extract_frame_features.py \
    --pkl_path /path/to/aic_normal_dataset_with_3d_480p.pkl \
    --clips_root /path/to/BBoxClips/normal \
    --model clip \
    --output_dir ./features/normal_clip \
    --stride 156 --workers 2 --batch_size 16
```

### DINOv3 (ViT-B/16) — 768-dim per frame

Extracts the CLS token from `facebook/dinov3-vitb16-pretrain-lvd1689m` for each frame independently.

```bash
python extract_frame_features.py \
    --pkl_path /path/to/aic_normal_dataset_with_3d_480p.pkl \
    --clips_root /path/to/BBoxClips/normal \
    --model dinov3 \
    --output_dir ./features/normal_dinov3 \
    --stride 156 --workers 2 --batch_size 16
```

### VideoMAE (Base) — 768-dim per frame

Extracts mean-pooled patch tokens from `MCG-NJU/videomae-base-finetuned-kinetics` using a 48-frame sliding window subsampled to 16 frames. Runs in FP16 for efficiency.

```bash
python extract_video_features.py \
    --pkl_path /path/to/aic_normal_dataset_with_3d_480p.pkl \
    --clips_root /path/to/BBoxClips/normal \
    --output_dir ./features/normal_videomae \
    --batch_size 16
```

**Note:** The last 47 frames of each video will have zero features (no valid 48-frame window).

## Output Format

Each script produces one `.npy` file per video:

```
features/normal_clip/
├── video_01.mp4.npy    # shape (T, 512), float32
├── video_02.mp4.npy
└── ...

features/normal_dinov3/
├── video_01.mp4.npy    # shape (T, 768), float32
└── ...

features/normal_videomae/
├── video_01.mp4.npy    # shape (T, 768), float32
└── ...
```

Where `T` = total number of frames in the original video. Feature vectors are aligned by frame index with the PKL annotation fields (`labels[i]`, `keypoint[:, i, :, :]`, `bboxes[i]`, etc.).

| Model           | Dimensions | Extraction                   | Notes                                        |
| --------------- | ---------- | ---------------------------- | -------------------------------------------- |
| CLIP ViT-B/16   | 512        | Per-frame CLS token          | `openai/clip-vit-base-patch16`             |
| DINOv3 ViT-B/16 | 768        | Per-frame CLS token          | `facebook/dinov3-vitb16-pretrain-lvd1689m` |
| VideoMAE Base   | 768        | 48-frame window, mean-pooled | `MCG-NJU/videomae-base-finetuned-kinetics` |

### Loading features in Python

```python
import numpy as np

features = np.load('features/normal_clip/video_01.mp4.npy')
print(features.shape)  # (T, 512) for CLIP, (T, 768) for DINOv3/VideoMAE
print(features.dtype)  # float32
```
