import subprocess
import pickle
import os

def run_script(split_num, split_pos, config_path, pkl_path, weight_path, label_from, window_size, out_dict_dir, out_dict_name):
    command = [
        'python3', 'infer_pkl.py',
        '--anno_split_num', str(split_num),
        '--anno_split_pos', str(split_pos),
        '--config_path', config_path,
        '--pkl_path', pkl_path,
        '--weight_path', weight_path,
        '--label_from', label_from,
        '--window_size', str(window_size),
        '--out_dict_dir', out_dict_dir,
        '--out_dict_name', out_dict_name
    ]
    return subprocess.Popen(command)

if __name__ == '__main__':
    # Configuration for splits
    total_splits = 3  # How many processes you want to run

    # Define paths and settings
    config_path = 'configs/CNN1D_kp.py'
    pkl_path = "../keypoints_train/data/aicactivity/normal/aic_normal_dataset.pkl"
    weight_path = "weights/CNN1D_kp.pt"
    label_from = "center"
    window_size = 48
    out_dict_dir = "./out_dict/cnn1d_center_noskip"

    processes = []
    out_dict_names = []
    for i in range(total_splits):
        out_dict_name = f"result_{i}.pkl"  # Unique output file for each process
        out_dict_names.append(out_dict_name)
        # Start the process
        p = run_script(total_splits - 1, i, config_path, pkl_path, weight_path, label_from, window_size, out_dict_dir, out_dict_name)
        processes.append(p)

    # Wait for all processes to complete
    for p in processes:
        p.wait()
        
    # Combine the results from all the output dictionaries
    combined_results = {}
    for out_dict_name in out_dict_names:
        out_path = os.path.join(out_dict_dir, out_dict_name)
        with open(out_path, 'rb') as f:
            result_dict = pickle.load(f)
            combined_results.update(result_dict)  # Combine dictionaries

    # Save the combined results
    final_output_path = os.path.join(out_dict_dir, 'result.pkl')
    with open(final_output_path, 'wb') as f:
        pickle.dump(combined_results, f)

    # Optionally, you can remove the individual result files if they are no longer needed
    for out_dict_name in out_dict_names:
        os.remove(os.path.join(out_dict_dir, out_dict_name))