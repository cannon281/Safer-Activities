import pickle

# Configuration
dataset_type = "normal"  # normal or wheelchair
compiled_annotations_path = f'data/{dataset_type}/compiled_Annotations/aic_dataset_{dataset_type}_annotations.pkl'
new_pickle_file_path = f'data/{dataset_type}/compiled_Annotations/aic_{dataset_type}_dataset.pkl'  # Final dataset pickle name

# Load the compiled annotations
with open(compiled_annotations_path, 'rb') as file:
    compiled_annotations = pickle.load(file)

# Subject and view splits configuration
test_names = ["p032", "p009", "p016", "p018", "p022", "p027", "p013", "p021"]
test_views = ["d02", "d05"]

# Initialize splits
train_videos, test_videos = [], []
view_train_videos, view_test_videos = [], []

# Process annotations for subject and view splits
for annotation in compiled_annotations:
    frame_dir = annotation.get('frame_dir')
    video_file_name = frame_dir.split('/')[-1]

    # Subject Split based on test_names pattern
    if any(test_name in video_file_name for test_name in test_names):
        test_videos.append(video_file_name)
    else:
        train_videos.append(video_file_name)

    # View Split based on presence of test_views
    if any(test_view in frame_dir for test_view in test_views):
        view_test_videos.append(video_file_name)
    else:
        view_train_videos.append(video_file_name)


# Prepare the data to be saved with updated splits
data_to_save = {
    "split": {
        "sub_train": train_videos,
        "sub_test": test_videos,
        "view_train": view_train_videos,
        "view_test": view_test_videos
    },
    "annotations": compiled_annotations
}

# Save the updated dataset
with open(new_pickle_file_path, 'wb') as file:
    pickle.dump(data_to_save, file)

print(f"Data saved to {new_pickle_file_path}")
