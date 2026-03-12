"""
Spawn multiple parallel inference processes for dino_clip_features models.

Same approach as inference/spawn_multiple_infer_pkl.py: splits test annotations
across N processes, runs infer_pkl.py in parallel, then merges results.

Usage:
  python spawn_multiple_infer_pkl.py \
      --run_dir runs/normal_dinov3_cnn1d_fusion_seq \
      --pkl_path ../pyskl/Pkl/aic_normal_dataset_with_3d_480p.pkl \
      --out_dict_dir ./eval_results/dinov3_cnn1d_fusion \
      --total_splits 3
"""

import argparse
import os
import pickle
import subprocess


def run_script(split_num, split_pos, run_dir, pkl_path, feature_dir,
               label_from, window_size, out_dict_dir, out_dict_name, device):
    command = [
        'python3', 'infer_pkl.py',
        '--anno_split_num', str(split_num),
        '--anno_split_pos', str(split_pos),
        '--run_dir', run_dir,
        '--pkl_path', pkl_path,
        '--label_from', label_from,
        '--window_size', str(window_size),
        '--out_dict_dir', out_dict_dir,
        '--out_dict_name', out_dict_name,
        '--device', device,
    ]
    if feature_dir:
        command.extend(['--feature_dir', feature_dir])
    return subprocess.Popen(command)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Spawn multiple dino_clip_features inference processes')
    parser.add_argument('--run_dir', type=str, required=True,
                        help='Path to training run directory')
    parser.add_argument('--pkl_path', type=str, required=True,
                        help='Path to annotation pkl file')
    parser.add_argument('--feature_dir', type=str, default=None,
                        help='Override feature directory from config')
    parser.add_argument('--label_from', type=str, default='center')
    parser.add_argument('--window_size', type=int, default=48)
    parser.add_argument('--out_dict_dir', type=str, required=True,
                        help='Output directory for result.pkl')
    parser.add_argument('--total_splits', type=int, default=3,
                        help='Number of parallel processes')
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    processes = []
    out_dict_names = []

    for i in range(args.total_splits):
        out_dict_name = f"result_{i}.pkl"
        out_dict_names.append(out_dict_name)
        p = run_script(
            args.total_splits - 1, i,
            args.run_dir, args.pkl_path, args.feature_dir,
            args.label_from, args.window_size,
            args.out_dict_dir, out_dict_name, args.device)
        processes.append(p)

    # Wait for all processes to complete
    for p in processes:
        p.wait()

    # Combine results
    combined_results = {}
    for out_dict_name in out_dict_names:
        out_path = os.path.join(args.out_dict_dir, out_dict_name)
        with open(out_path, 'rb') as f:
            result_dict = pickle.load(f)
            combined_results.update(result_dict)

    # Save combined results
    final_output_path = os.path.join(args.out_dict_dir, 'result.pkl')
    with open(final_output_path, 'wb') as f:
        pickle.dump(combined_results, f)

    # Clean up individual split files
    for out_dict_name in out_dict_names:
        os.remove(os.path.join(args.out_dict_dir, out_dict_name))

    print(f"\nCombined results saved to: {final_output_path}")
    print(f"Total videos: {len(combined_results)}")
    print(f"Total predictions: {sum(len(v) for v in combined_results.values())}")
