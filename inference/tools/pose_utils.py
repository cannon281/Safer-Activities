import numpy as np
import cv2
import os

# This function first looks at the confidence threshold of the pose results and rejects those with average keypoint confidence below the threshold
# It also looks at the bounding box area, if it is too small, then it removes such detections
# Then, it looks at the bbox that is closest to the reference point (between the bottom-center and center of the frame) 
def pick_best_pose_result(drawn_frame, pose_results, frame_width, frame_height, confidence_threshold=0.4, bbox_area_threshold=2500):
    pose_results = [
        pr for pr in pose_results 
        if pr['keypoints'][:, 2].mean() > confidence_threshold 
        and pr['bbox'][3] * pr['bbox'][2] > bbox_area_threshold
    ]

    if len(pose_results)==0:
        return [], drawn_frame
    
    # Define the reference point (between the center and the bottom of the frame)
    reference_y = int((frame_height / 2) + (frame_height / 4))  # 3/4 down from the top
    reference_point = (int(frame_width / 2), reference_y)

    # Function to calculate the distance to the reference point
    def distance_to_reference_point(bbox_center):
        return ((bbox_center[0] - reference_point[0]) ** 2 + (bbox_center[1] - reference_point[1]) ** 2) ** 0.5

    closest_pose_result = None
    min_distance = float('inf')
    highest_confidence_score = 0
    highest_confidence_pose = None

    for pose_result in pose_results:
        bbox = pose_result['bbox']
        bbox_area = bbox[3]*bbox[2]
        confidence_score = pose_result['keypoints'][:, 2].mean()
        bbox_score = pose_result['bbox_score']
        
        # Assuming bbox_center and reference_point are calculated correctly
        # bbox_center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
        
        bbox_center = (int(bbox[0] + bbox[2] / 2), int(bbox[1] + bbox[3] / 2))
        reference_point = (int(reference_point[0]), int(reference_point[1]))

        # Draw the line on the image
        cv2.line(drawn_frame, bbox_center, reference_point, (255, 0, 0), 2)

        # Optionally, you can draw the reference point and bbox_center to debug
        cv2.circle(drawn_frame, bbox_center, 5, (0, 255, 0), -1)
        cv2.circle(drawn_frame, reference_point, 5, (0, 0, 255), -1)

        # Calculate the distance to the reference point
        distance = distance_to_reference_point(bbox_center)
        # Display the distance value as a text in drawn_frame
        cv2.putText(drawn_frame, f"Distance: {distance:.2f}, Confidence: {confidence_score}", bbox_center, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)
        bbox_center_offset = (bbox_center[0], bbox_center[1]+100)
        cv2.putText(drawn_frame, f"BBox:{bbox_score}, BBox Area:{bbox[3]*bbox[2]}", bbox_center_offset, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 1)

        # Update the pose with the highest confidence score
        if confidence_score > highest_confidence_score:
            highest_confidence_score = confidence_score
            highest_confidence_pose = pose_result

        # If the confidence score is above the threshold, check the distance to the reference point
        if distance < min_distance:
            min_distance = distance
            closest_pose_result = pose_result

    # If no pose result is above the threshold, return the one with the highest confidence score
    if closest_pose_result is None:
        return [highest_confidence_pose], drawn_frame

    # Otherwise, return the closest pose result or an empty list if none are found
    return [closest_pose_result], drawn_frame


# For wheelchair:
# This function first filters out detections below or above the threshold
# Then it ensures that there are at least two people present on the scene, otherwise, it returns an empty list
# If true, then it tries to return the person in the wheelchair by picking the one with the center of the face keypoints lower in the frame than the other
# The logic is weird, but we found that it works in most scenarios that we tested
def pick_best_pose_result_if_two_person(drawn_frame, pose_results, frame_width, frame_height, confidence_threshold=0.4, bbox_area_threshold=3000):
    # Filter pose results based on confidence and bounding box area
    pose_results_filtered = [
        pr for pr in pose_results 
        if pr['keypoints'][:, 2].mean() > confidence_threshold 
        and pr['bbox'][3] * pr['bbox'][2] > bbox_area_threshold
    ]
    
    # Initialize variables to track the lowest face center
    lowest_face_center = 0  # Initialize with max possible value
    pose_result_to_return = None
    face_centers = []  # To store face centers for drawing lines

    
    if len(pose_results_filtered) == 2:
        for pose_result in pose_results_filtered:
            keypoints = pose_result['keypoints']
            # Indices for eyes, ears, and nose keypoints; adjust these indices as per your model's keypoints
            face_keypoints_indices = [0, 1, 2, 3, 4, 5]  # Example indices
            face_keypoints = keypoints[face_keypoints_indices, :]
            
            # Calculate the average position (x, y) of the face keypoints
            face_center = np.mean(face_keypoints[:, :2], axis=0)
            face_centers.append(tuple(face_center.astype(int)))  # Convert to integer for drawing
            
            # Compare y-coordinate to find the lowest face center (lowest is > because opencv coordinates start from top-left)
            if face_center[1] > lowest_face_center:
                lowest_face_center = face_center[1]
                pose_result_to_return = pose_result
                
        # Draw the line between the two face centers
        if len(face_centers) == 2:
            cv2.line(drawn_frame, face_centers[0], face_centers[1], (255, 0, 0), 2)
        return [pose_result_to_return], drawn_frame
        
    else:
        return [], drawn_frame



def process_pose_results(drawn_frame, pose_results, action, frame_width, frame_height, label, print_coord, print_font_size, out_action_root):
    # Initialize variables
    previous_track_id = None
    keypoints, keypoint_scores, bboxes = np.zeros((1, 34)), np.zeros((1, 17)), np.zeros((1,4))
    len_pose_results = len(pose_results)

    multi_person_actions = ['p_sit_stand', 'p_sit_propel', 'p_sit_walk']
    # Determine action and adjust pose_results accordingly
    if action in multi_person_actions:
        pose_results, drawn_frame = pick_best_pose_result_if_two_person(drawn_frame, pose_results, frame_width=frame_width, frame_height=frame_height)
    elif len(pose_results) > 1:
        pose_results, drawn_frame = pick_best_pose_result(drawn_frame, pose_results, frame_width=frame_width, frame_height=frame_height)

    # Process pose_results based on their count
    if pose_results and len(pose_results) == 1:
        # Extract keypoints, scores, and bounding boxes for single detection
        keypoints = np.array([pr['keypoints'][:, :2].flatten() for pr in pose_results])
        keypoint_scores = np.array([pr['keypoints'][:, 2] for pr in pose_results])
        bboxes = np.array([pose_results[0]['bbox']])
        previous_track_id = pose_results[0]['track_id']
        action_text = 'Valid frame.'
        color = (0, 255, 0)
    else:
        # Handle cases with no valid pose_results or multiple detections without a clear best result
        action_text = 'Invalid frame. No bounding box..' if not pose_results else 'Multiple detections. Data not added.'
        color = (0, 0, 255)

    # Annotate frame
    text = f'Action: {action}, {action_text} Track ID: {previous_track_id if previous_track_id else "N/A"}'
    cv2.putText(drawn_frame, text, print_coord, cv2.FONT_HERSHEY_SIMPLEX, print_font_size, color, 2, cv2.LINE_AA)
    
    # Optional: Save frames with specific conditions
    if action not in multi_person_actions and len_pose_results > 1:
        os.makedirs(out_action_root, exist_ok=True)
        multi_bbox_save_path = os.path.join(out_action_root, f'multi_people_selected_track_id_{previous_track_id}.jpg')
        cv2.imwrite(multi_bbox_save_path, drawn_frame)


    # Return modified frame and data for external processing
    return drawn_frame, keypoints, keypoint_scores, label, bboxes, previous_track_id
    
