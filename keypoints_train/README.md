# Training and Testing Models

This directory contains the necessary scripts and configuration files to train and test Simple-1D-CNN.

The pickle files for the non-wheelchair and wheelchair datasets can be found through [this link](https://drive.google.com/file/d/1SHKJDlHRIG36eWcjQRplsNQ9jGWrxbMp/view?usp=drive_link). 

## Configuration Files

The configuration files (`.py`) located in the `configs` folder are used to specify the model architecture, training parameters, dataset details, and other relevant settings.

Each configuration file may include sections for optimizer settings (`optimizer_cfg`), loss function configuration (`loss_cfg`), and detailed dataset and model parameters (`dataset_cfg`, `model_cfg`, `train_cfg`). Here's a brief overview of what each section entails:

- `optimizer_cfg`: Defines the optimizer type (e.g., `Adam`) and its parameters like learning rate, betas, epsilon, and weight decay.
- `loss_cfg`: Specifies the loss function to use during training (e.g., `CrossEntropyLoss`).
- `log_cfg`: Configures logging, including whether to save logs and the directory to save them in.
- `dataset_cfg`: Contains details about the dataset, including the path to data, dataset type, preprocessing settings, and split configurations.
- `model_cfg`: Outlines the model to be used, including its type and relevant arguments that define its architecture.
- `train_cfg`: Sets training-specific parameters, such as the number of epochs, batch size, whether to use multiple GPUs, and details about the training, validation, and test settings.


## Training and Testing Models

To train a model, use the `train.py` script along with the necessary command-line arguments.

### Example Usage

With single GPU:
```bash
python train.py --config_path configs/CNN1D_kp.py 
```

With multiple GPUs:
```bash
python train.py --config_path configs/CNN1D_kp.py --visible_gpus 0,1,2,3
```

For testing, use the `test.py` script along with the necessary command-line arguments.

### Example Usage

```bash
python test.py --config_path configs/CNN1D_kp.py --checkpoint_path work_dirs/CNN1D_kp/20240227-013900/_KeypointCNN1D_epoch_40.pt --visible_gpus 0
```

### Saved Checkpoints

Below are the checkpoints for our models. Click on the links to download the checkpoints. All the checkpoints and training logs can be found
[here](https://drive.google.com/drive/folders/1C7rFeIwUzz5oYN21XorRQN79AtcIul2l?usp=drive_link). 

| Checkpoint File                | Google Drive Link                                                                 |
|--------------------------------|------------------------------------------------------------------------------------|
| `CNN1D_kp.py`                  | [Download](https://drive.google.com/file/d/18bfylZZCBU7JkNjzAFkgkLWePHLvJLXs/view?usp=drive_link)|
| `CNN1D_kp_view.py`             | [Download](https://drive.google.com/file/d/1NV4eWuQtbwjfBDZMpOhFnu5Nk25v0xli/view?usp=drive_link)|
| `CNN1D_wc_kp.py`               | [Download](https://drive.google.com/file/d/1SFtWEroCxPXbfi5QnvIrEmNktGmBMqzF/view?usp=drive_link)|
| `CNN1D_wc_kp_view.py`          | [Download](https://drive.google.com/file/d/17Qw3oUaPD6AC6lkkDR8z73TxMiKi0NPB/view?usp=drive_link)|
| `Skip_CNN1D_kp.py`             | [Download](https://drive.google.com/file/d/1dAWjZIUKduDLJLo9qBDeHPxCODod6O-X/view?usp=drive_link)|
| `Skip_CNN1D_kp_view.py`        | [Download](https://drive.google.com/file/d/1wFXsBDcPW5634x0WCsm6TT-xzehnakTO/view?usp=drive_link)|
| `Skip_CNN1D_wc_kp.py`          | [Download](https://drive.google.com/file/d/1swgQ7Rpe2tlVK5nQbH5BDFg5EiaGLSz8/view?usp=drive_link)|
| `Skip_CNN1D_wc_kp_view.py`     | [Download](https://drive.google.com/file/d/1P2s-c-Hug9U3ix2bD8Cx_X42iRDsrLFg/view?usp=drive_link)|



### Data and Dataset Configuration

- `data_root`: Specifies the base directory where the dataset is stored. 
- `dataset_type`: Indicates the type of dataset as "normal" or "wheelchair"
- `num_frames`: Defines the number of frames or the temporal window size in each video clip used for training or testing. 
- `skip_frames`: A boolean value indicating whether to skip frames within each video clip to reduce the temporal resolution.
- `skip_stride`: Determines the interval between frames when `skip_frames` is true. A larger stride means fewer frames are used.
- `mode`: Specifies the mode of operation, either "classification" or "segmentation".

`dataset_cfg`: This dictionary contains all settings related to the dataset, including paths, preprocessing details, and how data is split for training and testing.
- `pickle_file_path`: The full path to the dataset file.
- `clip_length`, `label_from`, `clip_majority_frames`, `skip_window_length`: These parameters control how video clips are extracted and labeled from the full videos.
- `apply_transforms`: Indicates whether data augmentation or preprocessing transforms should be applied to the dataset.
- `splits`: Whether to use the subject or view splits.

### Model Configuration

`model_cfg`: Defines which model architecture to use and its specific configurations.
- `args`: Additional arguments specific to the model, such as `num_frames`, which may be adjusted based on whether frames are skipped, and `motion_info`, a boolean indicating whether motion information should be incorporated into the model inputs.

### Training Configuration

`train_cfg`: Outlines the settings for the training process.
- `epochs`, `print_every`, `save_ckpt`, `save_ckpt_every`: These settings control the overall training loop, including how many epochs to train for, how frequently to print updates and save checkpoints.
- `multi_gpu`: Indicates whether training should be distributed across multiple GPUs.
- `train_settings`, `val_settings`, `test_settings`: These dictionaries specify settings for the DataLoader during training, validation, and testing phases. 


For more details on configuring the training process, see the example configuration file: [CNN1D_kp.py](./configs/CNN1D_kp.py).
