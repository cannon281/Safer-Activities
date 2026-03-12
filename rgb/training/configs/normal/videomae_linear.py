# VideoMAE linear probe — frozen VideoMAE-Base features
# Input: pre-extracted VideoMAE feature (768-dim, 1 frame per sample)
# Each feature[t] already summarizes the 48-frame window [t, t+48)
# Head: Linear(768, 15)

optimizer_cfg = dict(
    type='AdamW',
    settings=dict(
        lr=1e-3,
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=5e-4,
    )
)

loss_cfg = dict(
    type='CrossEntropyLoss',
)

log_cfg = dict(
    save_logs=True,
    log_dir='work_dirs/',
)

dataset_type = "normal"
ann_file = "../pyskl/Pkl/aic_normal_dataset_with_3d_480p.pkl"
feature_dir = "/home/work/rgb/feature_extraction/features/normal_videomae"

dataset_cfg = dict(
    dataset_type=dataset_type,
    mode="feature",
    ann_file=ann_file,
    feature_dir=feature_dir,
    preprocess="sequential",
    model_frames=1,
    feat_dim=768,
    splits=dict(
        train="sub_train",
        test="sub_test",
    ),
    seed=111,
)

model_cfg = dict(
    type="meanpool",
    args=dict(
        feat_dim=768,
        num_classes=15,
        num_frames=1,
        kp_num_frames=48,
    )
)

train_cfg = dict(
    epochs=50,
    print_every=5,
    save_ckpt=True,
    save_ckpt_every=5,
    scheduler=dict(type='CosineAnnealingLR', eta_min=1e-6),
    train_settings=dict(
        batch_size=128,
        shuffle=True,
        num_workers=16,
    ),
    val_settings=dict(
        batch_size=128,
        shuffle=False,
        num_workers=16,
        seed=111,
    ),
    test_settings=dict(
        batch_size=128,
        shuffle=False,
        num_workers=16,
        seed=111,
    ),
    output_dir="./runs/normal_videomae_linear_seq",
)
