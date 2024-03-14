from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def get_precision_recall(cm):
    precisions = []
    recalls = []
    for i in range(cm.shape[0]):
        precision = cm[i, i] / (np.sum(cm[:, i]) + 1e-10)
        recall = cm[i, i] / (np.sum(cm[i, :]) + 1e-10)
        precisions.append(precision)
        recalls.append(recall)
    return precisions, recalls


def get_confusion_matrix(y_true, y_pred, classes, cmap=plt.cm.Blues, dataset="Test"):
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Display the matrix
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax, pad=0.1)
    
    print(classes)
    
    # Set the labels, title, and ticks
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title="Confusion Matrix",
           ylabel="Prediction",
           xlabel="Ground Truth")

    plt.xticks(rotation=45)
    
    # Display the values in the matrix
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")

    # Calculate precision, recall, and accuracy
    precision = np.diag(cm) / np.sum(cm, axis=0)
    recall = np.diag(cm) / np.sum(cm, axis=1)
    accuracy = np.trace(cm) / np.sum(cm)

    # Display precision and recall in the matrix
    # Precision
    ax.text(cm.shape[1] + 0.2 , -1, f"Precision", ha="center", va="center", color="black", fontsize=10)
    for i, p in enumerate(precision):
        ax.text(cm.shape[1] + 0.2 , i, f"{p*100:.1f}%", ha="center", va="center", color="black", fontsize=10, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2'))
    
    # Recall moved further below
    ax.text(-1 , cm.shape[0] + 1.5, f"Recall", ha="center", va="center", color="black", fontsize=10)
    for i, r in enumerate(recall):
        ax.text(i, cm.shape[0] + 1.5, f"{r*100:.1f}%", ha="center", va="center", color="black", fontsize=10, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2'))

    ax.text(cm.shape[1] + 0.5, cm.shape[0] + 1.5, f"Acc: {accuracy*100:.1f}%", ha="center", va="center", color="white", fontsize=12, bbox=dict(facecolor='black', edgecolor='black', boxstyle='round,pad=0.2'))

    fig.tight_layout()
    return fig, cm



import pandas as pd  # Import pandas for CSV file handling

def save_confusion_matrix_and_classification_report(true_labels, predicted_labels, labels, cm_save_path, report_save_path, cm_csv_path):
    # Compute confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)

    # Normalize the confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Plotting
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Normalized Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.savefig(cm_save_path, bbox_inches='tight')

    # Save raw confusion matrix to CSV file
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(cm_csv_path)

    # Calculate and save classification metrics
    report = classification_report(true_labels, predicted_labels, digits=3, target_names=labels, output_dict=True)

    # Calculate mean class accuracy
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    mean_class_accuracy = np.nanmean(class_accuracies)  # Use nanmean to ignore NaN values in case of division by zero

    # Append mean class accuracy to the report dictionary
    report['mean_class_accuracy'] = mean_class_accuracy

    # Convert the report dictionary to a string, including mean class accuracy
    report_str = ''
    for label, metrics in report.items():
        if isinstance(metrics, dict):  # For each class
            metrics_str = ' '.join(f'{k}: {v:.3f}' for k, v in metrics.items())
            report_str += f'{label} {metrics_str}\n'
        else:  # For summary metrics (e.g., accuracy) and mean class accuracy
            report_str += f'{label}: {metrics:.3f}\n'

    with open(report_save_path, 'w') as f:
        f.write(report_str)
        
    return report

# Note: Remember to provide the cm_csv_path argument when calling the function, specifying the path to save the CSV file.




def get_confusion_matrix_precision(y_true, y_pred, classes, cmap=plt.cm.viridis, dataset="Test"):
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize the confusion matrix by row (i.e., by the number of samples in each class)
    cm_normalized = cm.astype('float') / cm.sum(axis=0)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Use the normalized confusion matrix for display
    im = ax.imshow(cm_normalized, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm_normalized.shape[1]),
           yticks=np.arange(cm_normalized.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title= "Precision Confusion Matrix for "+dataset+" Dataset",
           ylabel="True Label",
           xlabel="Predicted Label")

    plt.xticks(rotation=45)
    
    # Print the values in the matrix
    thresh = cm_normalized.max() / 2.
    for i in range(cm_normalized.shape[0]):
        for j in range(cm_normalized.shape[1]):
            ax.text(j, i, format(cm_normalized[i, j], '.2f'),
                    ha="center", va="center",
                    color="white" if cm_normalized[i, j] <= thresh else "black")
    
    fig.tight_layout()
    return fig, cm
    

def get_confusion_matrix_recall(y_true, y_pred, classes, cmap=plt.cm.viridis, dataset="Test"):
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize the confusion matrix by column (i.e., by the number of predictions in each class)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[np.newaxis, :]
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Use the normalized confusion matrix for display
    im = ax.imshow(cm_normalized, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm_normalized.shape[1]),
           yticks=np.arange(cm_normalized.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title="Recall Confusion Matrix for "+dataset+" Dataset",
           ylabel="True Label",
           xlabel="Predicted Label")

    plt.xticks(rotation=45)
    
    # Print the values in the matrix
    thresh = cm_normalized.max() / 2.
    for i in range(cm_normalized.shape[0]):
        for j in range(cm_normalized.shape[1]):
            ax.text(j, i, format(cm_normalized[i, j], '.2f'),
                    ha="center", va="center",
                    color="white" if cm_normalized[i, j] <= thresh else "black")
    
    fig.tight_layout()
    return fig, cm


def get_confusion_matrices(y_true, y_pred, classes, dataset="Test"):
    precision_confusion_matrix, _ = get_confusion_matrix_precision(y_true, y_pred, classes)
    recall_confusion_matrix, _ = get_confusion_matrix_recall(y_true, y_pred, classes)
    return precision_confusion_matrix, recall_confusion_matrix