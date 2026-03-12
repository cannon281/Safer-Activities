"""
Step 4: Convert result.pkl files to per-video CSVs for fall evaluation.

infer_pkl.py produces:
  result.pkl = {video_name: [[pred_label, gt_label, start_frame], ...], ...}

This converts to per-video CSVs with columns: Predicted, GT, Frame, Num_detections
where both Predicted and GT are mapped to binary "fall" / "non_fall", compatible
with inference/calculate_fall_accuracy_other_datasets.py.

Usage:
    python step4_results_to_csv.py --eval_dir eval_results
"""

import argparse
import csv
import os
import pickle
from pathlib import Path


def convert_result_pkl(result_pkl_path, output_dir):
    """Convert a single result.pkl to per-video CSVs."""
    with open(result_pkl_path, 'rb') as f:
        results = pickle.load(f)

    os.makedirs(output_dir, exist_ok=True)

    for video_name, predictions in results.items():
        csv_path = os.path.join(output_dir, f"{video_name}_pred_gt.csv")

        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Predicted', 'GT', 'Frame', 'Num_detections'])

            for pred_label, gt_label, start_frame in predictions:
                # Binary mapping: "fall" stays "fall", everything else -> "non_fall"
                pred_binary = "fall" if pred_label == "fall" else "non_fall"

                # GT: label index 10 was mapped to "fall" by infer_pkl.py,
                # label index 0 was mapped to "no_label"
                gt_binary = "fall" if gt_label == "fall" else "non_fall"

                writer.writerow([pred_binary, gt_binary, start_frame, 1])

        print(f"  {video_name}: {len(predictions)} predictions -> {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert result.pkl files to per-video CSVs')
    parser.add_argument('--eval_dir', type=str, default='eval_results',
                        help='Directory containing imvia_*/result.pkl subdirs')
    args = parser.parse_args()

    eval_dir = args.eval_dir

    # Find all result.pkl files
    result_files = sorted(Path(eval_dir).glob('imvia_*/result.pkl'))

    if not result_files:
        print(f"No result.pkl files found in {eval_dir}/imvia_*/")
        return

    for result_pkl in result_files:
        model_dir = result_pkl.parent
        model_name = model_dir.name
        predictions_dir = model_dir / 'predictions'

        print(f"\n{model_name}:")
        convert_result_pkl(str(result_pkl), str(predictions_dir))

    print(f"\nDone! Converted {len(result_files)} result files.")


if __name__ == '__main__':
    main()
