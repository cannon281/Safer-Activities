import os
import pickle

dataset_type = "normal" # normal or wheelchair

annotations_folder_path = f'data/{dataset_type}/annotations'
compiled_folder_path = f'data/{dataset_type}/compiled_Annotations'
compiled_annotations = []

# Loop through each file in the Annotations folder
for filename in os.listdir(annotations_folder_path):
    if filename.endswith('.pkl'):
        file_path = os.path.join(annotations_folder_path, filename)
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            # Assuming each file contains a list and you need the first element
            if data and isinstance(data, list):
                compiled_annotations.append(data[0])

# Save the compiled data to a new pickle file
compiled_file_path = os.path.join(compiled_folder_path, f'aic_dataset_{dataset_type}_annotations.pkl')
os.makedirs(compiled_folder_path, exist_ok=True)

with open(compiled_file_path, 'wb') as outfile:
    pickle.dump(compiled_annotations, outfile)

print(f"Compiled data saved to {compiled_file_path}")