import os
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed


# This script can be used to simply readjust the labels without modifying the keypoints and detections if the CSV files have been altered

num_threads = 2 # Number of threads to use to make the process faster.
dataset_type = "normal" # normal or wheelchair

root_video_dir=f'data/{dataset_type}/Videos' # Full dataset
out_video_root = 'out_video'
save_out_video = True
resolution = '1080p'
csv_folder = f'data/{dataset_type}/CSVs'
extract_keypoints_path = f'data/{dataset_type}/annotations'
mappings_json_path = "mappings.json"

def run_extraction_command(cmd):
    os.system(cmd)
    
# List to store all the future tasks
futures = []

# Create a ThreadPoolExecutor to run multiple instances in parallel
executor = ThreadPoolExecutor(max_workers=3)

csvs = [os.path.basename(x) for x in glob.glob(os.path.join(csv_folder, '*.csv'))]
for csv in csvs:
    common_prefix = os.path.splitext(csv)[0]
    video_files = glob.glob(os.path.join(root_video_dir, common_prefix + '*.mp4'))

    for video_file in video_files:
        # Extract the video file name without extension
        video_file_name = os.path.splitext(os.path.basename(video_file))[0]
        
        # Change the extension to .pkl
        pickle_file_name = f"{video_file_name}.pkl"
        pickle_file_path = os.path.join(extract_keypoints_path, pickle_file_name)

        # Construct the command
        cmd = (
            f'python3 tools/adjust_labels_per_frame.py '
            f'--csv_file {os.path.join(csv_folder, csv)} '
            f'--video_file {video_file} '  # Removed the redundant os.path.join(video_file, video_file)
            f'--extract_keypoints_dir {extract_keypoints_path} '
            f'--resolution {resolution} '
            f'--pickle_file {pickle_file_path} '
            f'--mappings_json_path {mappings_json_path} '
            f'--extract_mode {dataset_type} ' # Normal or Wheelchair
        )
        print(cmd)

        # Submit the command to the executor
        future = executor.submit(run_extraction_command, cmd)
        futures.append(future)


# Wait for all the futures to complete
for future in as_completed(futures):
    print(future.result())

# Shutdown the executor
executor.shutdown()

