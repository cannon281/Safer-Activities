import pickle

# Path to the pickle file
pickle_file_path = 'data/normal/annotations/apr_25_2023_fall_p005_d01.pkl'

# Function to load data from a pickle file
def load_data_from_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

# Load the data
data = load_data_from_pickle(pickle_file_path)

print(len(data))
print(data[-1].keys())
print(data[-1]["keypoint"].shape)
print(data[-1]["keypoint_score"].shape)
print(data[-1]["labels"].shape)
print(data[-1]["bbox"].shape)

print(data[-1]["keypoint"][0][127:128])
print(data[-1]["labels"][127:130])
print(data[-1]["labels"][120:170])


