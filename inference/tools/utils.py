import pandas as pd
import json
import numpy as np
import cv2

def load_mappings_and_initialize_variables(file_path, mode):
    """
    Load mappings from JSON file and initialize variables.
    
    Args:
    - file_path: str, path to the JSON file containing mappings.
    - mode: str, either "normal" or "wheelchair" to specify which mappings to load.
    
    Returns:
    - Initializes global variables based on the selected mode.
    """
    # Load JSON content
    with open(file_path, 'r') as file:
        mappings = json.load(file)[mode]

    # Extract mappings and labels
    action_mappings = mappings['mappings']
    main_labels = list(mappings['labels'].values())
    all_labels = list(mappings['full_labels'].values())

    # Initialize label maps and counts
    ylabelmap = {label: int(idx) for idx, label in mappings['labels'].items()}
    label_count = {label: 0 for label in main_labels}
    actionmap = {label: ylabelmap[action_mappings[label]] if label in action_mappings else ylabelmap[label] for label in all_labels}
        
    all_ylabelmap = {label: int(all_labels.index(label)) for label in all_labels}
    all_label_count = {label: 0 for label in all_labels}
    
    return main_labels, all_labels, label_count, actionmap, all_ylabelmap, all_label_count



def get_sec(time_str):
    """Get seconds from time."""
    h, m, s = time_str.split(':') if len(time_str.split(':'))==3 else time_str.split(':')[1:]
    return int(h) * 3600 + int(m) * 60 + float(s)

def seconds_to_time_str(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02}:{int(minutes):02}:{seconds:06.3f}"

def get_action_dict_from_csv(csv_file):
    """Convert Spreadsheet to Proper Dictionary"""
    action_label_map = {}
    df = pd.read_csv(csv_file)
    for row in df.iterrows():
        rowclass = row[1]["Action"]
        rowStart = row[1]["Start Time"]
        rowEnd = row[1]["End Time"]
        
        # Check if any of the relevant fields are empty or NaN
        if pd.isna(rowclass) or pd.isna(rowStart) or pd.isna(rowEnd):
            continue  # Skip this row
        
        if rowclass in action_label_map:
            action_label_map[rowclass].append([get_sec(rowStart), get_sec(rowEnd)])
        else:
            action_label_map[rowclass] = [[get_sec(rowStart), get_sec(rowEnd)]]
    return action_label_map


def get_action_from_frame_timestamp(action_label_map, timestamp):
    """Get action for current frame"""
    action = None
    for key in action_label_map:
        for i in range(len(action_label_map[key])):
            start = action_label_map[key][i][0]
            end = action_label_map[key][i][1]
            if timestamp>=start and timestamp<=end:
                action = key 
    return action


def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]


def resize_frame(frame, mode):
    original_height, original_width = frame.shape[:2]

    if mode == "1080p":
        new_width, new_height = 1920, 1080
    else:
        # Handle other modes or default sizes
        new_width, new_height = 1920, 1080  # Defaulting to 1080p for this example

    if original_width < new_width and original_height < new_height:
        # If original frame is smaller, use linear interpolation
        interpolation = cv2.INTER_LINEAR
    else:
        # If original frame is larger or equal, use area interpolation
        interpolation = cv2.INTER_AREA

    resized_frame = cv2.resize(frame, dsize=(new_width, new_height), interpolation=interpolation)
    return resized_frame