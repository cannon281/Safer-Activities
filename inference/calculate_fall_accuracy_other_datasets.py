import csv
import os
import argparse


def read_file(file_path):
    """
    Open the CSV file specified by file_path and return two lists of boolean values:
    - gt_list: where each element is True if the 'GT' column indicates a fall event, otherwise False.
    - pred_list: where each element is True if the 'Predicted' column or any 'Action' column indicates a fall event, otherwise False.
    
    Args:
    file_path (str): The path to the CSV file to be read.
    
    Returns:
    tuple: A tuple containing two lists (gt_list, pred_list) with boolean values.
    """    
    gt_list = []
    pred_list = []
    
    with open(file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Check the GT column for 'fall'
            gt_list.append(row['GT'] == 'fall')
            
            # Check the 'Predicted' column and any 'Action' columns for 'fall'
            has_fall = row['Predicted'] == 'fall'
            for key in row:
                if 'Action' in key and row[key] == 'fall':
                    has_fall = True
            pred_list.append(has_fall)

    return gt_list, pred_list

def find_fall_clusters(arr):
    """
    This function identifies clusters of True values in a list, ensuring that each cluster
    starts and ends with a True value and contains at least 85% True values. The function
    returns a list of tuples where each tuple contains the start and end index of a cluster.

    Parameters:
    arr (list): A list of boolean values where True represents 'fall' and False represents 'non_fall'.

    Returns:
    list: A list of tuples, each containing the start and end index of a cluster.
    """

    clusters = []
    n = len(arr)
    max_cluster_size = 300 # add maximum fall duration for optimization of calculation below
    start_index = 0
    while start_index < n:
        max_len = 0
        best_end_index = -1
        true_count = 0
        total_count = 0

        for end_index in range(start_index, min(start_index + max_cluster_size, n)):
            total_count += 1
            if arr[end_index]:
                true_count += 1

            if arr[start_index] and arr[end_index] and (true_count / total_count) >= 0.8:
                if total_count > max_len:
                    max_len = total_count
                    best_end_index = end_index

        if best_end_index != -1:
            clusters.append((start_index, best_end_index))
            start_index = best_end_index + 1
        else:
            start_index += 1

    return clusters


def match_clusters_with_max_overlap(file_path, gt_clusters, pred_clusters, window_size):
    """
    Match predicted clusters to ground truth clusters based on maximum overlap within a window size.
    Calculate true positives (TP), false positives (FP), and false negatives (FN).

    Parameters:
    gt_clusters (list of tuples): List of tuples representing the start and end indices of ground truth clusters.
    pred_clusters (list of tuples): List of tuples representing the start and end indices of prediction clusters.
    window_size (int): Tolerance window size.

    Returns:
    tuple: Counts of true positives (TP), false positives (FP), and false negatives (FN).
    """    
    tp, fp, fn = 0, 0, 0
    matched_pred = set()

    def calculate_overlap(gt_start, gt_end, pred_start, pred_end):
        """
        Calculate the overlap between an extended ground truth cluster and a prediction cluster.

        Parameters:
        gt_start (int): Start index of the ground truth cluster.
        gt_end (int): End index of the ground truth cluster.
        pred_start (int): Start index of the prediction cluster.
        pred_end (int): End index of the prediction cluster.

        Returns:
        int: Length of the overlap between the clusters.
        """        
        overlap_start = max(gt_start - window_size, pred_start)
        overlap_end = min(gt_end + window_size, pred_end)
        return max(0, overlap_end - overlap_start + 1)

    for gt_start, gt_end in gt_clusters:
        max_overlap = 0
        best_match = None

        for pred_start, pred_end in pred_clusters:
            if (pred_start, pred_end) in matched_pred:
                continue  # Skip already matched predictions

            overlap = calculate_overlap(gt_start, gt_end, pred_start, pred_end)
            if overlap>0:
                matched_pred.add((pred_start, pred_end))
                
            if overlap > max_overlap:
                max_overlap = overlap
                best_match = (pred_start, pred_end)

        if best_match:
            tp += 1
            matched_pred.add(best_match)
        else:
            fn += 1
            print("FN in GT: ", file_path, " at ", gt_start+2) # add 2 for csv heading and python indexing

    for pred_start, pred_end in pred_clusters:
        if (pred_start, pred_end) not in matched_pred:
            fp += 1
            print("FP in Pred: ", file_path, " at ", pred_start+2) # add 2 for csv heading and python indexing

    return tp, fp, fn


def calculate_f1_score(tp, fp, fn):
    """
    Calculate the F1 score given true positives (TP), false positives (FP), and false negatives (FN).

    Parameters:
    tp (int): True positives.
    fp (int): False positives.
    fn (int): False negatives.

    Returns:
    float: The F1 score.
    """
    if tp == 0:
        return 0.0  # Prevent division by zero if there are no true positives

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0, precision, recall

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process CSV files in a directory to calculate F1 scores.")
    parser.add_argument('directory', type=str, help='Path to the directory containing CSV files')
    args = parser.parse_args()

    total_tp, total_fp, total_fn = 0, 0, 0
    for root, _, files in os.walk(args.directory):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                #print("processing", file)
                gt_list, pred_list = read_file(file_path)
                gt_clusters = find_fall_clusters(gt_list)
                pred_clusters = find_fall_clusters(pred_list)
                #print("GT clusters", gt_clusters)
                #print("pred clusters", pred_clusters)
                tp, fp, fn = match_clusters_with_max_overlap(file, gt_clusters, pred_clusters, 15)
                total_tp = total_tp + tp
                total_fp = total_fp + fp
                total_fn = total_fn + fn
                #f1_score = calculate_f1_score(tp, fp, fn)
                #print(f"True Positives: {tp}, False Positives: {fp}, False Negatives: {fn}")
                #print(f"F1 Score: {f1_score:.2f}")

    print(f"True Positives: {total_tp}, False Positives: {total_fp}, False Negatives: {total_fn}")
    final_f1_score, precision, recall = calculate_f1_score(total_tp, total_fp, total_fn)
    print(f"F1 Score: {final_f1_score:.3f}, Precision: {precision}, Recall:{recall}")
