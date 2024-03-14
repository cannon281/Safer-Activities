import pickle

compiled_annotations_path = 'data/normal/compiled_Annotations/aic_normal_dataset.pkl'  # Replace with your path

# Load the compiled annotations
with open(compiled_annotations_path, 'rb') as file:
    data = pickle.load(file)

compiled_annotations = data["annotations"]
# Check the number of elements
print(f"Number of elements in the compiled list: {len(compiled_annotations)}")

# Check that each element is a dictionary and collect keys
all_keys = set()
consistent_keys = True

for item in compiled_annotations:
    if not isinstance(item, dict):
        consistent_keys = False
        print("Error: Not all items are dictionaries.")
        break
    all_keys.update(item.keys())

# Verify keys consistency
if consistent_keys:
    for item in compiled_annotations:
        if item.keys() != all_keys:
            consistent_keys = False
            print("Error: Inconsistent keys found in dictionaries.")
            break

if consistent_keys:
    print("All items are dictionaries with consistent keys.")
    print(f"Sample keys: {list(all_keys)}")  # Display the keys as a sample
else:
    print("The dictionaries in the list have inconsistent keys.")


