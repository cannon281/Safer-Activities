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
dataset_type="normal" # normal or wheelchair
num_frames = 48
skip_frames = False
skip_stride = 1

dataset_cfg = dict(
    dataset_type=dataset_type, # normal or wheelchair
    dataset_class = "AICActivityDataset",
    test_dataset_class = "AICActivityTestDataset",
    args = dict(
            pickle_file_path=data_root+dataset_type+"/aic_"+dataset_type+"_dataset.pkl",
            clip_length = num_frames, # Length of the clip, we use 48 for a temporal stride of 1 and 144 for a temporal stride of 3 
            label_from = "center", # Labels are picked based on the majority frames from the center or end
            clip_majority_frames = 5, # The number of frames from the center or end, whatever is in label_from to pick fetch the label 
            skip_window_length = 20, # This determines the window size from which we pick a sub-window for training and validation
            skip_frames = skip_frames, # If set to true, the frames are skipped based on the skip_stride value
            skip_stride = skip_stride, # Skips frames with a temporal stride if skip_frames is true
            apply_transforms = True, # Enable dataset transformations
        ),
    
    splits = dict(
        train="sub_train",
        test="sub_test"
        ),
    
    mappings_json_file = data_root+"mappings.json",
    skip_frames = False,
    seed = 111,
)

model_cfg = dict(
    type='KeypointCNN1D',
    args=dict(
        num_frames = int(num_frames / skip_stride) if skip_frames else num_frames,
        motion_info = True,
    )
)

train_cfg = dict(
    epochs=50,
    print_every=5,
    save_ckpt=True,
    save_ckpt_every=5,

    multi_gpu = True,
    train_settings=dict(
        batch_size=256,
        shuffle=True,
        num_workers = 4,
        transforms = {
            # 'HorizontalFlip':{'probability':0.5, 'image_width':1920},
            # 'GaussianNoise':{'mean':0, 'std':0.5, 'pixel_range':2}, 
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

