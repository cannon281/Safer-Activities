optimizer_cfg = dict(
    type='Adam', 
    settings=dict(
        lr=1e-3, 
        betas=(0.9, 0.98), 
        eps=1e-9,
        weight_decay=5e-4  # Increased weight decay
    )
)


loss_cfg = dict(
    type='CrossEntropyLoss',
)

log_cfg = dict(
    save_logs = True,
    log_dir = 'work_dirs/',
)    

data_root = 'data/aicactivity/'
dataset_type="wheelchair" # normal or wheelchair
num_frames = 48
skip_frames = False
skip_stride = 1

dataset_cfg = dict(
    dataset_type=dataset_type, # normal or wheelchair
    dataset_class = "AICActivityDataset",
    test_dataset_class = "AICActivityTestDataset",
    args = dict(
            pickle_file_path=data_root+dataset_type+"/aic_"+dataset_type+"_dataset.pkl",
            clip_length = num_frames,
            label_from = "center", 
            clip_majority_frames = 5,
            skip_window_length = 20,
            skip_frames = skip_frames, 
            skip_stride = skip_stride, 
            apply_transforms = True,
        ),
    
    splits = dict(
        train="view_train",
        test="view_test"
        ),
    
    mappings_json_file = data_root+"mappings.json",
    skip_frames = False,
    seed = 111,
)

model_cfg = dict(
    type='KeypointCNN1DTiny',
    args=dict(
        num_frames = int(num_frames / skip_stride) if skip_frames else num_frames,
        motion_info=True,
    )
)

train_cfg = dict(
    epochs=60,
    print_every=5,
    save_ckpt=True,
    save_ckpt_every=5,

    multi_gpu = True,
    train_settings=dict(
        batch_size=256,
        shuffle=True,
        num_workers = 4,
        transforms = {
            'HorizontalFlip':{'probability':0.5, 'image_width':1920},
            'GaussianNoise':{'mean':0, 'std':0.5, 'pixel_range':2}, 
            'ScaleWithNeckMotion':{}, 
            'To1DInputShape':{}
            }
    ),
    val_settings=dict(
        batch_size=256,
        shuffle=False,
        num_workers = 4,
        seed = 111,
        transforms = {'ScaleWithNeckMotion':{}, 'To1DInputShape':{}}
    ),
    test_settings=dict(
        batch_size=256,
        shuffle=False,
        num_workers = 4,
        seed = 111,
        transforms = {'ScaleWithNeckMotion':{}, 'To1DInputShape':{}}
    ),
)

