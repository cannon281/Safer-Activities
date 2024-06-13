# Inference and Evaluation Code for SAFER-Activities

This directory contains scripts and configuration files for evaluating the performance of models trained on SAFER-Activities, as well as for running inference. The evaluation includes two additional realistic fall datasets ImViA and Leuven. The data and configuration format is the same as used for training. 

We refer to the dataset from Baldewijns et al. as the Leuven dataset, and the dataset from Charfi et al. as the ImViA dataset.

All pretrained weights for 1DCNN can be found through https://drive.google.com/drive/folders/1FQMkcwpDEKERniblIRIKN-yHYC8Yw_wo.

All pretrained weights for PySKL models can be found through https://drive.google.com/drive/u/0/folders/1kOtWbogvpBdXZdMTXnBxo3bZQPV3pX4I.


<hr>

## Directory Structure

These files are to be used for evaluating models on the SAFER-Activities test set. 
- `infer_pkl.py`: Script for running inference using SAFER-Activities PKL files.
- `infer_pkl_pyskl.py`: Script for running inference using SAFER-Activities PKL files with PySKL.
- `calculate_accuracy.py`: Script to calculate the accuracy of the model's predictions.
- `spawn_multiple_infer_pkl.py`: Script to spawn multiple inference processes simultaneously for faster processing.
- `spawn_multiple_infer_pkl_pyskl.py`: Script to spawn multiple inference processes with PySKL simultaneously for faster processing.

These files are used to run inference on the ImViA dataset
- `infer_imvia.py`: Script for running inference on IMVIA dataset with 1DCNN.
- `infer_imvia_pyskl.py`: Script for running inference on IMVIA dataset using PySKL models.

These files are used to run inference on the Leuven dataset
- `infer_leuven.py`: Script for running inference on Leuven dataset with 1DCNN.
- `infer_leuven_pyskl.py`: Script for running inference on Leuven dataset using PySKL models.
- `Leuven_Removed_Videos.txt`: List of videos removed from Leuven dataset.

These files can be used to run video demo
- `demo_infer_vid.py`: Demo script for running inference on a video.
- `demo_infer_vid_pyskl.py`: Demo script for running inference on a video using PySKL.
- `demo_infer_wheelchair.py`: Demo script for running inference on wheelchair dataset.

- `export_labels_one_video.py`: Script to export predicted and ground-truth results for one video (Requires the ground truth).

- `calculate_fall_accuracy_other_datasets.py`: Script to calculate the fall detection accuracy on Leuven and ImViA datasets

<hr>


## Evaluation on SAFER-Activities

To evaluate the performance of models on the SAFER-Activities test set, you can use the `infer_pkl.py` (for 1DCNN) or `infer_pkl_pyskl.py` (for STGCN++, MSG-3D, and POSEC3D) scripts. These scripts are used to save the predicted and ground-truth results into a dictionary. `calculate_accuracy.py` can be used on the resulting pkl file to get the classification report, as well as the confusion matrix. 


#### Example Usage for 1DCNN Models

```bash
python infer_pkl.py --pkl_path data/aicactivity/normal/aic_normal_dataset.pkl --config_path configs/CNN1D_kp.py --weight_path weights/CNN1D_kp.pt --label_from center --window_size 48 --out_dict_dir ./out_dict/ --out_dict_name result.pkl
```

#### Example Usage for PySKL Models

```bash
python3 infer_pkl_pyskl.py --pkl_path data/aicactivity/normal/aic_normal_dataset.pkl --config_path pyskl/configs/posec3d/safer_activity_xsub/non-wheelchair.py --weight_path pyskl/weights/posec3d/non-wheelchair/non-wheelchair-epoch_44.pth --label_from center --window_size 48 --out_dict_dir ./out_dict/ --out_dict_name result.pkl --device_number 0
```

### Calculating Accuracy
Once the evaluation script `infer_pkl.py` or `infer_pkl_pyskl.py` is finished running, the resulting file saved with the --out_dict_name argument `result.pkl` will be used to calculate the classification report and confusion matrix. Use `calculate_accuracy.py` for this purpose. The scores reported on SAFER-Activities on the paper are based on this script.

#### Example Usage
```bash
python3 calculate_accuracy.py --pkl_path ./out_dict/result_1DCNN.pkl --out_accuracy_results ./out_results/result_1DCNN 
```

<hr>

## Running Inference on Videos
To run inference on videos, you can use the `demo_infer_vid.py` (for 1DCNN) or `demo_infer_wheelchair.py` (for 1DCNN wheelchair) or `demo_infer_vid_pyskl.py` (for STGCN++, MSG-3D, and POSEC3D) scripts. 

#### Example Usage

<br>

1DCNN:
```bash
python demo_infer_vid.py --config_path configs/CNN1D_kp.py --weight_path weights/CNN1D_kp.pt --label_from center --video_path videos/new_fall.mp4
```
<br>

PySKL:
```bash
python demo_infer_vid_pyskl.py --config_path pyskl/configs/posec3d/safer_activity_xsub/non-wheelchair.py --weight_path pyskl/weights/posec3d/non-wheelchair/non-wheelchair-epoch_44.pth --label_from center --label_type normal --video_path videos/new_fall.mp4
```

<hr>


## Evaluation on ImViA Dataset
To evaluate the performance of models on the ImViA dataset, you can use the `infer_imvia.py` (for 1DCNN) or `infer_imvia_pyskl.py` (for STGCN++, MSG-3D, and POSEC3D) scripts. The scripts will save the results in a folder inside the `out_imvia_dir` folder based on the provided value in --save_model_name argument. To calculate the accuracy of fall detection, `calculate_fall_accuracy_other_datasets.py` can be used, providing the path to the saved results. For example, if the results from `infer_imvia.py` or `infer_imvia_pyskl.py` are saved under `out_imvia_dir/cnn1d_imvia`, then the following script should be run:

```bash
python3 calculate_fall_accuracy_other_datasets.py out_imvia_dir/cnn1d_imvia
```

Note that the script excepts the ImViA dataset to be stored under `data/ImViA` folder with the following folders:
`Coffee_room_01`,
`Coffee_room_02`,
`Home_01`,
`Home_02`.

The `Lecture_room` and `Office` folders are removed as they do not contain ground-truth labels.


#### Example Usage for 1DCNN Models
```bash
python3 infer_imvia.py --config_path configs/CNN1D_kp.py --weight_path weights/CNN1D_kp.pt --save_model_name cnn1d_imvia --label_from center
```

#### Example Usage for PySKL Models

```bash
python3 infer_imvia_pyskl.py --config_path pyskl/configs/posec3d/safer_activity_xsub/non-wheelchair.py --weight_path pyskl/weights/posec3d/non-wheelchair/non-wheelchair-epoch_44.pth --save_model_name posec3d_imvia --label_from center
```


<hr>

## Evaluation on Leuven Dataset
To evaluate the performance of models on the Leuven dataset, you can use the `infer_leuven.py` (for 1DCNN) or `infer_leuven_pyskl.py` (for STGCN++, MSG-3D, and POSEC3D) scripts. The process is the same as running evaluation on the ImViA Dataset. The data is expected to be under `data/Leuven` folder. The annotation file `Data_Description.xlsx` should be present, and the `Videos` folder containing all the videos should be present. 

To calculate the accuracy of fall detection, `calculate_fall_accuracy_other_datasets.py` can be used in the same way as the ImViA dataset.

```bash
python3 calculate_fall_accuracy_other_datasets.py out_leuven_dir/cnn1d_leuven
```

#### Example Usage for 1DCNN Models

```bash
python3 infer_leuven.py --config_path configs/CNN1D_kp.py --weight_path weights/CNN1D_kp.pt --save_model_name cnn1d_leuven --label_from center
```

#### Example Usage for PySKL Models

```bash
python3 infer_leuven_pyskl.py --config_path pyskl/configs/posec3d/safer_activity_xsub/non-wheelchair.py --weight_path pyskl/weights/posec3d/non-wheelchair/non-wheelchair-epoch_44.pth --save_model_name posec3d_leuven --label_from center
```

<hr>



