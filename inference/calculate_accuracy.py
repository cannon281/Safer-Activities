import pickle
import numpy as np
import pandas as pd
import os
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description='Calculate Accuracy and Save Results')
parser.add_argument('--pkl_path', type=str, required=True)
parser.add_argument('--out_accuracy_results', type=str, default="default")
parser.add_argument('--tolerance', type=int, default=5)

args = parser.parse_args()

pkl_path = args.pkl_path
out_accuracy_path = args.out_accuracy_results

if out_accuracy_path == "default":
    out_accuracy_path = os.path.dirname(pkl_path)
out_accuracy_full_dir = os.path.join(out_accuracy_path, "report")
os.makedirs(out_accuracy_full_dir, exist_ok=True)

# Load your results dictionary from the pickle file
with open(pkl_path, 'rb') as file:
    results = pickle.load(file)

def validate_predictions_with_tolerance(results, tolerance=args.tolerance):
    y_true = []
    y_pred = []
    
    # Iterate through each video and its corresponding predictions
    for video, predictions in results.items():
        for i, current_prediction in enumerate(predictions):
            pred_label, true_label, frame_number = current_prediction
            
            if true_label == 'no_label':
                continue  # Skip predictions where no ground truth label is provided
            
            # Check for matching labels within the tolerance window
            # Look backwards and forwards within the tolerance range
            valid = False
            for offset in range(-tolerance, tolerance + 1):
                index = i + offset
                if index >= 0 and index < len(predictions):
                    _, nearby_true_label, _ = predictions[index]
                    if pred_label == nearby_true_label:
                        valid = True
                        break
            
            if valid:
                y_pred.append(pred_label)
                y_true.append(pred_label)
            else:
                # Optionally, handle cases where no match was found within the tolerance window
                y_pred.append(pred_label)
                y_true.append(true_label)  # This could be adjusted based on how you handle unmatched cases

    return y_true, y_pred

y_true, y_pred = validate_predictions_with_tolerance(results, tolerance=5)

# Convert lists to numpy arrays for easier manipulation and metric calculations
y_true = np.array(y_true)
y_pred = np.array(y_pred)



from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Calculate metrics and handle classes without true samples
def calculate_metrics(y_true, y_pred):
    unique_labels = np.unique(np.concatenate([y_true, y_pred]))
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=1)  # Avoid division by zero by setting to 1
    recall = recall_score(y_true, y_pred, average='macro', zero_division=1)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=1)
    conf_matrix = confusion_matrix(y_true, y_pred, labels=unique_labels)
    class_report = classification_report(y_true, y_pred, zero_division=1, labels=unique_labels)

    return accuracy, precision, recall, f1, conf_matrix, class_report, unique_labels

# Assuming y_true and y_pred are already defined
accuracy, precision, recall, f1, conf_matrix, class_report, unique_labels = calculate_metrics(y_true, y_pred)

# Save the confusion matrix
conf_matrix_df = pd.DataFrame(conf_matrix, index=unique_labels, columns=unique_labels)
conf_matrix_df.to_csv(f'{out_accuracy_full_dir}/confusion_matrix.csv')

# Convert and save the classification report
report_df = pd.DataFrame(classification_report(y_true, y_pred, output_dict=True, zero_division=1)).transpose()
report_df.to_csv(f'{out_accuracy_full_dir}/classification_report.csv')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Classification Report:\n", class_report)
