import cv2
import numpy as np
from collections import deque
from tools.utils import get_action_from_frame_timestamp, resize_frame, nan_helper
import os
from scipy.signal import butter, filtfilt
from utils.classification_model import cnn1d_infer
from utils.dataset_utils import get_action_dict_from_spreadsheet, get_action_from_48_frames

import sys
pyskl_path = "/home/work/inference/pyskl"
if pyskl_path not in sys.path:
    sys.path.append(pyskl_path)
import mmcv
from pyskl.apis import inference_recognizer, init_recognizer
from utils.dataset_utils import get_majority_labels

MAIN_LABELS =  ["stand", "stand_activity", "walk", "sit", "sit_activity", "sitting_down", "getting_up",
           "bend", "unstable", "fall", "lie_down", "lying_down", "reach", "run", "jump"]

WHEELCHAIR_LABELS = [
    'sit', 'propel',
    'pick_place',
    'sit_activity', 
    'bend', 'getting_up',
    'exercise',
    'sitting_down', 
    'prepare_transfer', 'transfer', 'fall', 
    'lie_down', 'lying_down', 
    'get_propelled',
    'stand'
    ]

def process_one_video(video_path, pose_model, model, transforms, device, show_video, mode=None, save_out_video=True, 
                      out_video_root="./out_video/", fall_threshold = 0.5, label_from="center", viz_kpt=False, label_type="normal", infer_pyskl=False):
    lstm_framecount = 48
    lstm_posedict = {}
    os.makedirs(out_video_root, exist_ok=True)

    framecount = 0
    filtered_framcount=0
    buffercount=10

    number_joints=17
    pose_results_buffer=deque([], maxlen=buffercount)
    pose_results_buffer_list=[]
    pose_results_buffer_list_update=0

    frame_buffer=deque([], maxlen=buffercount)
    id_joints={}

    fs = 17  # Sampling frequency
    fc = 3  # Cut-off frequency of the filter
    wn = fc / (fs / 2)  # Normalize the frequency
    butter_b, butter_a = butter(2, wn)

    vid = cv2.VideoCapture(video_path)
    if save_out_video:
        fps = vid.get(cv2.CAP_PROP_FPS)
        size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        if mode == "1080p":
            size = (1920, 1080)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        videoWriter = cv2.VideoWriter(os.path.join(out_video_root, f'vis_{os.path.basename(video_path)}'), fourcc, fps, size)
        videoWriterKpt = cv2.VideoWriter(os.path.join(out_video_root, f'vis_kpt_{os.path.basename(video_path)}'), fourcc, fps, size)

    framecount = 0
    pose_results_buffer = deque([], maxlen=10)
    video_buffer = [None for i in range(24)] if label_from == "center" else []
    kpt_buffer = [None for i in range(24)] if label_from == "center" and viz_kpt else []
    
    show_fall_popup = False
    fall_popup_counter = 0
    fall_popup_counter_threshold = 90
    
   
    while True:
        ret, frame = vid.read()
        if ret:
            frame = resize_frame(frame, mode) if mode is not None else frame
            if viz_kpt:
                pts, tids, bboxes, drawn_frame, orig_frame, scores, kpt_image = pose_model.inference(frame, framecount, return_kpt_image=True)
            else:
                pts, tids, bboxes, drawn_frame, orig_frame, scores = pose_model.inference(frame, framecount)

            pose_results = []
            # Reshape data in the same format as pose_results from mmpose
            for tid, bbox, pt in zip(tids, bboxes, pts):
                # swap pt 0 and 1 columns
                pt[:,[0, 1]] = pt[:,[1, 0]]
                pose_dict = {
                    "bbox": bbox,   
                    "keypoints": pt,
                    "track_id": tid,
                }
                pose_results.append(pose_dict)

            # ----------------------------------------------------------------

            if framecount>=buffercount:
                if pose_results_buffer_list_update==0:
                    pose_results_buffer_list=list(pose_results_buffer)

                    tracked_id = {}

                    for frame in pose_results_buffer_list:
                        for person in frame:
                            if person['track_id'] in tracked_id:
                                tracked_id[person['track_id']] = tracked_id[person['track_id']]+ 1
                            else:
                                tracked_id[person['track_id']] = 0

                    id_joints={}
                    for id in tracked_id.keys():
                        if tracked_id[id] < 2 or id==-1:   # if trackedid==-1 remove track, if tracked instances less than 2, remove track
                            continue

                        id_joints[id] = [ [0]*buffercount for n in range(number_joints*3) ]

                    frameidx=0
                    for frame in pose_results_buffer_list:
                        for person in frame:

                            person_id=person['track_id']
                            if person_id not in id_joints:
                                continue

                            idx=0
                            for kp in person['keypoints']:
                                id_joints[person_id][idx][frameidx]=kp[0]
                                id_joints[person_id][idx + 1][frameidx]=kp[1]
                                id_joints[person_id][idx + 2][frameidx]=kp[2]
                                idx=idx+3

                        frameidx = frameidx + 1

                    filter_joint = []
                    for track_id in id_joints.keys():
                        jidx = 0
                        for joint in id_joints[track_id]:
                            if not any(joint):
                                filter_joint=joint
                            else:
                                interp_joint = np.array(joint)

                                # interpolate
                                interp_joint[interp_joint == 0] = np.nan
                                nans, x = nan_helper(interp_joint)
                                interp_joint[nans] = np.interp(x(nans), x(~nans), interp_joint[~nans])
                                filter_joint = filtfilt(butter_b, butter_a, interp_joint)

                            id_joints[track_id][jidx]=filter_joint
                            jidx = jidx + 1

                    pose_results_buffer_list_update=pose_results_buffer_list_update+1

                frameidx=pose_results_buffer_list_update-1

                current_joint=[]

                for track_id in id_joints.keys():
                    format_joint=[]

                    for j in range(0,number_joints*3,3):
                        format_joint.append([id_joints[track_id][j][frameidx],id_joints[track_id][j+1][frameidx],id_joints[track_id][j+2][frameidx]])

                    if any(format_joint):
                        lstm_joints=[]
                        jcount=0
                        for j in format_joint:
                            if(j[0]==0 and j[1]==0):
                                continue    
                            lstm_joints.extend([j[0],j[1],j[2]])
                            jcount=jcount+1

                        # TODO: Need to delete old track_id data
                        if track_id in lstm_posedict:
                            lstm_posedict[track_id].append(lstm_joints)
                        else:
                            lstm_posedict[track_id] = deque([], maxlen=lstm_framecount)  #TODO: lstm_posedict does not clean up old track_id pose data, !! mem leak !!
                            lstm_posedict[track_id].append(lstm_joints)


                        if len(lstm_posedict[track_id])==lstm_framecount:
                            if(infer_pyskl):
                                a_input = np.array([list(lstm_posedict[track_id])], dtype=np.float32)
                                reshaped_array = a_input.reshape(1, 48, 17, 3)
                                window_keypoints = reshaped_array[:, :, :, :2]
                                window_scores = reshaped_array[:, :, :, 2]
                                min_x, max_x = 0, 1919
                                min_y, max_y = 0, 1079
                                min_score, max_score = 0, 1
                                window_keypoints[..., 0] = np.clip(window_keypoints[..., 0], min_x, max_x)
                                window_keypoints[..., 1] = np.clip(window_keypoints[..., 1], min_y, max_y)
                                window_scores = np.nan_to_num(window_scores)
                                window_scores = np.clip(window_scores, min_score, max_score)

                                fake_anno = dict(
                                    frame_dir='',
                                    label=-1,
                                    img_shape=(1080, 1920),
                                    original_shape=(1080, 1920),
                                    start_index=0,
                                    modality='Pose',
                                    total_frames=window_keypoints.shape[1],
                                    test_mode = True,
                                    usable_indices = np.array([0]),
                                    usable_label = np.array([1]),
                                    keypoint=window_keypoints,
                                    keypoint_score=window_scores)
                                
                                model_results = inference_recognizer(model, fake_anno)
                                confidence = model_results[0][1]
                                if (label_type== "normal"):
                                    out = MAIN_LABELS[model_results[0][0]]
                                    action2 = MAIN_LABELS[model_results[1][0]]
                                else:
                                    out = WHEELCHAIR_LABELS[model_results[0][0]]
                                    action2 = WHEELCHAIR_LABELS[model_results[1][0]]
                                #predicted_action = labels_model[model_results[0][0]]
                            else:                            
                                a_input = np.array([list(lstm_posedict[track_id])], dtype=np.float32)
                                
                                # This is where to put the model and transformations and fetch the action back
                                out, confidence, action2, _, _, _ = cnn1d_infer(model, transforms, device, a_input, label_type=label_type,
                                                                                return_secondary=True, return_confidence=True)
                            
                            if out == "fall":
                                print("Fall detected with confidence", confidence)
                                if confidence < fall_threshold:
                                    out = action2
                                    print(f"Switched to action {out}")
                                else:
                                    show_fall_popup = True
                                    fall_popup_counter = 0
                                
                            _bbox = None
                            for person in pose_results_buffer_list[frameidx]:
                                if person['track_id'] == track_id:
                                    _bbox = person['bbox']
                                    break
                                
                            # Process the resulting action predicted by the model
                            if _bbox is not None:
                                if confidence is not None:
                                    label = f"{out} ({float(confidence):.2f})"
                                else:
                                    label = out
                                    
                                if out == "fall":
                                    color = (0, 0, 255)
                                else:
                                    color = (0, 255, 0)
                                
                                if label_from == "center":
                                    cv2.putText(video_buffer[25], label, (int(_bbox[0]+40), int(_bbox[1]-10)), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
                                    if viz_kpt:
                                        cv2.putText(kpt_buffer[25], label, (int(_bbox[0]+40), int(_bbox[1]-10)), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
                                else:
                                    cv2.putText(drawn_frame, label, (int(_bbox[0]+40), int(_bbox[1]-10)), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
                                    
                                    if viz_kpt:
                                        cv2.putText(kpt_image, label, (int(_bbox[0]+40), int(_bbox[1]-10)), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
                            
                            current_joint.extend(format_joint)
                        else:
                            current_joint.extend(format_joint)
                    else:
                        current_joint.extend(format_joint)

                filtered_framcount=filtered_framcount+1

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                pose_results_buffer_list_update=pose_results_buffer_list_update+1

                if pose_results_buffer_list_update==buffercount+1:
                    pose_results_buffer_list_update=0

                pose_results_buffer.popleft()
                prev_pos_results = pose_results.copy()
                pose_results_buffer.append(prev_pos_results)
                frame_buffer.append(drawn_frame)

            else:
                prev_pos_results = pose_results.copy()
                pose_results_buffer.append(prev_pos_results)
                frame_buffer.append(drawn_frame)

    # ----------------------------------------------------------------

            if show_fall_popup:
                if fall_popup_counter < fall_popup_counter_threshold:
                    if label_from == "center":
                        cv2.putText(video_buffer[25], "FALL DETECTED", (int(50), int(90)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 2, cv2.LINE_AA)
                    else:
                        cv2.putText(drawn_frame, "FALL DETECTED", (int(50), int(90)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 2, cv2.LINE_AA)
                    
                    fall_popup_counter += 1
                else:
                    show_fall_popup = False
                    fall_popup_counter = 0

    # ----------------------------------------------------------------

            framecount += 1
            
            if label_from == "center":
                video_buffer.append(drawn_frame)
                if len(video_buffer) > 48:
                    video_buffer.pop(0)

                if viz_kpt:
                    kpt_buffer.append(kpt_image)
                    if len(kpt_buffer)>48:
                        kpt_buffer.pop(0)

                if framecount > 24:
                    if show_video:
                        cv2.imshow("Video", video_buffer[24])
                        if viz_kpt:
                            cv2.imshow("kpt_video", kpt_buffer[24])
                    if save_out_video:
                        videoWriter.write(video_buffer[24])
                        if viz_kpt:
                            videoWriterKpt.write(kpt_buffer[24])
            else:
                if show_video:
                    cv2.imshow("Video", drawn_frame)
                    if viz_kpt:
                        cv2.imshow("kpt_video", kpt_image)
                if save_out_video:
                    videoWriter.write(drawn_frame)
                    if viz_kpt:
                        videoWriterKpt.write(kpt_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    vid.release()
    cv2.destroyAllWindows()
    if save_out_video:
        videoWriter.release()


# ----------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------


def infer_export_one_video(video_path, spreadsheet_path, pose_model, model, transforms, device, mapping_labels, actionmap,
                      mode=None, show_video=False, save_out_video=True, 
                      out_video_root="./out_video/", fall_threshold = 0.5, label_from = "center", label_type="normal", infer_pyskl=False):
    
    action_label_map = get_action_dict_from_spreadsheet(spreadsheet_path)
    predictions_list = []
    max_actions_one_frame = 0
    
    lstm_framecount = 48
    lstm_posedict = {}

    framecount = 0
    filtered_framcount=0
    buffercount=10

    number_joints=17
    pose_results_buffer=deque([], maxlen=buffercount)
    pose_results_buffer_list=[]
    pose_results_buffer_list_update=0

    frame_buffer=deque([], maxlen=buffercount)
    id_joints={}

    fs = 17  # Sampling frequency
    fc = 3  # Cut-off frequency of the filter
    wn = fc / (fs / 2)  # Normalize the frequency
    butter_b, butter_a = butter(2, wn)

    vid = cv2.VideoCapture(video_path)
    fps = vid.get(cv2.CAP_PROP_FPS)
    if save_out_video:
        size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        if mode == "1080p":
            size = (1920, 1080)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        videoWriter = cv2.VideoWriter(os.path.join(out_video_root, f'vis_{os.path.basename(video_path)}'), fourcc, fps, size)

    framecount = 0
    pose_results_buffer = deque([], maxlen=10)
    video_buffer = [None for i in range(24)] if label_from == "center" else []
    
    show_fall_popup = False
    fall_popup_counter = 0
    fall_popup_counter_threshold = 90

    while True:
        ret, frame = vid.read()
        if ret:
            frame = resize_frame(frame, mode) if mode is not None else frame
            pts, tids, bboxes, drawn_frame, orig_frame, scores = pose_model.inference(frame, framecount)

            pose_results = []
            # Reshape data in the same format as pose_results from mmpose
            for tid, bbox, pt in zip(tids, bboxes, pts):
                # swap pt 0 and 1 columns
                pt[:,[0, 1]] = pt[:,[1, 0]]
                pose_dict = {
                    "bbox": bbox,   
                    "keypoints": pt,
                    "track_id": tid,
                }
                pose_results.append(pose_dict)
    
            # ----------------------------------------------------------------
            action_this_frame = []            # For exporting current action

            if framecount>=buffercount:
                if pose_results_buffer_list_update==0:
                    pose_results_buffer_list=list(pose_results_buffer)

                    tracked_id = {}

                    for frame in pose_results_buffer_list:
                        for person in frame:
                            if person['track_id'] in tracked_id:
                                tracked_id[person['track_id']] = tracked_id[person['track_id']]+ 1
                            else:
                                tracked_id[person['track_id']] = 0

                    id_joints={}
                    for id in tracked_id.keys():
                        if tracked_id[id] < 2 or id==-1:   # if trackedid==-1 remove track, if tracked instances less than 2, remove track
                            continue

                        id_joints[id] = [ [0]*buffercount for n in range(number_joints*3) ]

                    frameidx=0
                    for frame in pose_results_buffer_list:
                        for person in frame:

                            person_id=person['track_id']
                            if person_id not in id_joints:
                                continue

                            idx=0
                            for kp in person['keypoints']:
                                id_joints[person_id][idx][frameidx]=kp[0]
                                id_joints[person_id][idx + 1][frameidx]=kp[1]
                                id_joints[person_id][idx + 2][frameidx]=kp[2]
                                idx=idx+3

                        frameidx = frameidx + 1

                    filter_joint = []
                    for track_id in id_joints.keys():
                        jidx = 0
                        for joint in id_joints[track_id]:
                            if not any(joint):
                                filter_joint=joint
                            else:
                                interp_joint = np.array(joint)

                                # interpolate
                                interp_joint[interp_joint == 0] = np.nan
                                nans, x = nan_helper(interp_joint)
                                interp_joint[nans] = np.interp(x(nans), x(~nans), interp_joint[~nans])
                                filter_joint = filtfilt(butter_b, butter_a, interp_joint)

                            id_joints[track_id][jidx]=filter_joint
                            jidx = jidx + 1

                    pose_results_buffer_list_update=pose_results_buffer_list_update+1

                frameidx=pose_results_buffer_list_update-1

                current_joint=[]

                for track_id in id_joints.keys():
                    format_joint=[]

                    for j in range(0,number_joints*3,3):
                        format_joint.append([id_joints[track_id][j][frameidx],id_joints[track_id][j+1][frameidx],id_joints[track_id][j+2][frameidx]])

                    if any(format_joint):
                        lstm_joints=[]
                        jcount=0
                        for j in format_joint:
                            if(j[0]==0 and j[1]==0):
                                continue    
                            lstm_joints.extend([j[0],j[1],j[2]])
                            jcount=jcount+1

                        # TODO: Need to delete old track_id data
                        if track_id in lstm_posedict:
                            lstm_posedict[track_id].append(lstm_joints)
                        else:
                            lstm_posedict[track_id] = deque([], maxlen=lstm_framecount)  #TODO: lstm_posedict does not clean up old track_id pose data, !! mem leak !!
                            lstm_posedict[track_id].append(lstm_joints)


                        if len(lstm_posedict[track_id])==lstm_framecount:
                            if(infer_pyskl):
                                a_input = np.array([list(lstm_posedict[track_id])], dtype=np.float32)
                                reshaped_array = a_input.reshape(1, 48, 17, 3)
                                window_keypoints = reshaped_array[:, :, :, :2]
                                window_scores = reshaped_array[:, :, :, 2]
                                min_x, max_x = 0, 1919
                                min_y, max_y = 0, 1079
                                min_score, max_score = 0, 1
                                window_keypoints[..., 0] = np.clip(window_keypoints[..., 0], min_x, max_x)
                                window_keypoints[..., 1] = np.clip(window_keypoints[..., 1], min_y, max_y)
                                window_scores = np.nan_to_num(window_scores)
                                window_scores = np.clip(window_scores, min_score, max_score)

                                fake_anno = dict(
                                    frame_dir='',
                                    label=-1,
                                    img_shape=(1080, 1920),
                                    original_shape=(1080, 1920),
                                    start_index=0,
                                    modality='Pose',
                                    total_frames=window_keypoints.shape[1],
                                    test_mode = True,
                                    usable_indices = np.array([0]),
                                    usable_label = np.array([1]),
                                    keypoint=window_keypoints,
                                    keypoint_score=window_scores)
                                
                                model_results = inference_recognizer(model, fake_anno)
                                confidence = model_results[0][1]
                                if (label_type== "normal"):
                                    out = MAIN_LABELS[model_results[0][0]]
                                    action2 = MAIN_LABELS[model_results[1][0]]
                                else:
                                    out = WHEELCHAIR_LABELS[model_results[0][0]]
                                    action2 = WHEELCHAIR_LABELS[model_results[1][0]]
                                #predicted_action = labels_model[model_results[0][0]]
                            else:
                                a_input = np.array([list(lstm_posedict[track_id])], dtype=np.float32)
                                
                                # This is where to put the model and transformations and fetch the action back
                                out, confidence, action2, _, _, _ = cnn1d_infer(model, transforms, device, a_input, return_secondary=True, return_confidence=True)
                            
                            if out == "fall":
                                if confidence < fall_threshold:
                                    out = action2
                                else:
                                    show_fall_popup = True
                                    fall_popup_counter = 0
                                
                            _bbox = None
                            for person in pose_results_buffer_list[frameidx]:
                                if person['track_id'] == track_id:
                                    _bbox = person['bbox']
                                    break
                                
                            # Process the resulting action predicted by the model
                            if _bbox is not None:
                                if confidence is not None:
                                    label = f"{out} ({float(confidence):.2f})"
                                else:
                                    label = out
                                    
                                if out == "fall":
                                    color = (0, 0, 255)
                                else:
                                    color = (0, 255, 0)
                                
                                if label_from == "center":
                                    cv2.putText(video_buffer[25], label, (int(_bbox[0]+40), int(_bbox[1]-10)), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
                                else:
                                    cv2.putText(drawn_frame, label, (int(_bbox[0]+40), int(_bbox[1]-10)), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
                            
                            action_this_frame.append(out)
                            
                            current_joint.extend(format_joint)
                        else:
                            current_joint.extend(format_joint)
                    else:
                        current_joint.extend(format_joint)

                filtered_framcount=filtered_framcount+1

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                pose_results_buffer_list_update=pose_results_buffer_list_update+1

                if pose_results_buffer_list_update==buffercount+1:
                    pose_results_buffer_list_update=0

                pose_results_buffer.popleft()
                prev_pos_results = pose_results.copy()
                pose_results_buffer.append(prev_pos_results)
                frame_buffer.append(drawn_frame)

            else:
                prev_pos_results = pose_results.copy()
                pose_results_buffer.append(prev_pos_results)
                frame_buffer.append(drawn_frame)

    # ----------------------------------------------------------------

            if show_fall_popup:
                if fall_popup_counter < fall_popup_counter_threshold:
                    if label_from == "center":
                        cv2.putText(video_buffer[25], "FALL DETECTED", (int(50), int(90)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 2, cv2.LINE_AA)
                    else:
                        cv2.putText(drawn_frame, "FALL DETECTED", (int(50), int(90)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 2, cv2.LINE_AA)
                    
                    fall_popup_counter += 1
                else:
                    show_fall_popup = False
                    fall_popup_counter = 0


            
    # ----------------------------------------------------------------
            
            framecount += 1

            # Get action from timestamp
            current_timestamp = framecount/fps
            
            # Getting action from the timestamp
            action = get_action_from_frame_timestamp(action_label_map, current_timestamp)
            # Also ignore if 13 previous frames are None actions
            timestamp_in_x_frames = (framecount-13)/fps
            if get_action_from_frame_timestamp(action_label_map, timestamp_in_x_frames) == "None":
                continue
            
            if framecount>48:
                center_label, end_label = get_action_from_48_frames(action_label_map, framecount-1, fps, num_frames=5)
                action_from_48 = center_label if label_from == "center" else end_label
                

                if action_from_48 != "None":
                    label_this_frame = mapping_labels[actionmap[action_from_48]]
                    
                    if label_from == "center":
                        cv2.putText(video_buffer[25], label_this_frame, (int(100), int(200)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 2, cv2.LINE_AA)
                    else:
                        cv2.putText(drawn_frame, label_this_frame, (int(100), int(200)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 2, cv2.LINE_AA)
                            
                    if len(action_this_frame)==0:
                        action_this_frame.append("No Prediction")
                    center_label = mapping_labels[actionmap[center_label]] if center_label != "None" else "None"
                    end_label = mapping_labels[actionmap[end_label]] if end_label != "None" else "None"
                    # print("Center_Label:", center_label, "End_Label:", end_label)
                    print("Predicted action:", action_this_frame, "GT Action:", label_this_frame, 
                          "Timestamp:", current_timestamp, "num_detections:", len(action_this_frame))
                    if label_this_frame != "None":
                        record = [action_this_frame[0], label_this_frame, current_timestamp, len(action_this_frame)]
                        record.extend(action_this_frame[1:])  # Append additional actions if available
                        predictions_list.append(record)
            
            max_actions_one_frame = len(action_this_frame) if len(action_this_frame) > max_actions_one_frame else max_actions_one_frame
            
    # ----------------------------------------------------------------

            if label_from == "center":
                video_buffer.append(drawn_frame)
                if len(video_buffer) > 48:
                    video_buffer.pop(0)
                if framecount > 24:
                    if show_video:
                        cv2.imshow("Video", video_buffer[24])
                    if save_out_video:
                        videoWriter.write(video_buffer[24])
            else:
                if show_video:
                    cv2.imshow("Video", drawn_frame)
                if save_out_video:
                    videoWriter.write(drawn_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    # ----------------------------------------------------------------

        else:
            break

    vid.release()
    cv2.destroyAllWindows()
    if save_out_video:
        videoWriter.release()

    return predictions_list, max_actions_one_frame



# ----------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------

def infer_imvia_dataset(video_path, txt_path, pose_model, model, transforms, device, mapping_labels, actionmap,
                      mode=None, show_video=False, save_out_video=True, 
                      out_video_root="./out_video/", fall_threshold = 0.5, label_from = "center", infer_pyskl=False):
    
    os.makedirs(out_video_root, exist_ok=True)
    
    # Reading the file and extracting the first two lines
    try:
        with open(txt_path, 'r') as file:
            lines = file.readlines()
            start = int(lines[0].strip())
            end = int(lines[1].strip())
    except:
        return None, 0
    
    predictions_list = []
    max_actions_one_frame = 0
    
    lstm_framecount = 48
    lstm_posedict = {}

    framecount = 0
    filtered_framcount=0
    buffercount=10

    number_joints=17
    pose_results_buffer=deque([], maxlen=buffercount)
    pose_results_buffer_list=[]
    pose_results_buffer_list_update=0

    frame_buffer=deque([], maxlen=buffercount)
    id_joints={}

    fs = 17  # Sampling frequency
    fc = 3  # Cut-off frequency of the filter
    wn = fc / (fs / 2)  # Normalize the frequency
    butter_b, butter_a = butter(2, wn)

    vid = cv2.VideoCapture(video_path)
    fps = vid.get(cv2.CAP_PROP_FPS)
    if save_out_video:
        size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        if mode == "1080p":
            size = (1920, 1080)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        videoWriter = cv2.VideoWriter(os.path.join(out_video_root, f'vis_{os.path.basename(video_path)}'), fourcc, fps, size)

    framecount = 0
    pose_results_buffer = deque([], maxlen=10)
    video_buffer = [None for i in range(24)] if label_from == "center" else []
    
    show_fall_popup = False
    fall_popup_counter = 0
    fall_popup_counter_threshold = 90

    while True:
        ret, frame = vid.read()
        if ret:
            frame = resize_frame(frame, mode) if mode is not None else frame
            pts, tids, bboxes, drawn_frame, orig_frame, scores = pose_model.inference(frame, framecount)

            pose_results = []
            # Reshape data in the same format as pose_results from mmpose
            for tid, bbox, pt in zip(tids, bboxes, pts):
                # swap pt 0 and 1 columns
                pt[:,[0, 1]] = pt[:,[1, 0]]
                pose_dict = {
                    "bbox": bbox,   
                    "keypoints": pt,
                    "track_id": tid,
                }
                pose_results.append(pose_dict)
    
            # ----------------------------------------------------------------
            action_this_frame = []            # For exporting current action

            if framecount>=buffercount:
                if pose_results_buffer_list_update==0:
                    pose_results_buffer_list=list(pose_results_buffer)

                    tracked_id = {}

                    for frame in pose_results_buffer_list:
                        for person in frame:
                            if person['track_id'] in tracked_id:
                                tracked_id[person['track_id']] = tracked_id[person['track_id']]+ 1
                            else:
                                tracked_id[person['track_id']] = 0

                    id_joints={}
                    for id in tracked_id.keys():
                        if tracked_id[id] < 2 or id==-1:   # if trackedid==-1 remove track, if tracked instances less than 2, remove track
                            continue

                        id_joints[id] = [ [0]*buffercount for n in range(number_joints*3) ]

                    frameidx=0
                    for frame in pose_results_buffer_list:
                        for person in frame:

                            person_id=person['track_id']
                            if person_id not in id_joints:
                                continue

                            idx=0
                            for kp in person['keypoints']:
                                id_joints[person_id][idx][frameidx]=kp[0]
                                id_joints[person_id][idx + 1][frameidx]=kp[1]
                                id_joints[person_id][idx + 2][frameidx]=kp[2]
                                idx=idx+3

                        frameidx = frameidx + 1

                    filter_joint = []
                    for track_id in id_joints.keys():
                        jidx = 0
                        for joint in id_joints[track_id]:
                            if not any(joint):
                                filter_joint=joint
                            else:
                                interp_joint = np.array(joint)

                                # interpolate
                                interp_joint[interp_joint == 0] = np.nan
                                nans, x = nan_helper(interp_joint)
                                interp_joint[nans] = np.interp(x(nans), x(~nans), interp_joint[~nans])
                                filter_joint = filtfilt(butter_b, butter_a, interp_joint)

                            id_joints[track_id][jidx]=filter_joint
                            jidx = jidx + 1

                    pose_results_buffer_list_update=pose_results_buffer_list_update+1

                frameidx=pose_results_buffer_list_update-1

                current_joint=[]

                for track_id in id_joints.keys():
                    format_joint=[]

                    for j in range(0,number_joints*3,3):
                        format_joint.append([id_joints[track_id][j][frameidx],id_joints[track_id][j+1][frameidx],id_joints[track_id][j+2][frameidx]])

                    if any(format_joint):
                        lstm_joints=[]
                        jcount=0
                        for j in format_joint:
                            if(j[0]==0 and j[1]==0):
                                continue    
                            lstm_joints.extend([j[0],j[1],j[2]])
                            jcount=jcount+1

                        # TODO: Need to delete old track_id data
                        if track_id in lstm_posedict:
                            lstm_posedict[track_id].append(lstm_joints)
                        else:
                            lstm_posedict[track_id] = deque([], maxlen=lstm_framecount)  #TODO: lstm_posedict does not clean up old track_id pose data, !! mem leak !!
                            lstm_posedict[track_id].append(lstm_joints)


                        if len(lstm_posedict[track_id])==lstm_framecount:
                            if(infer_pyskl):
                                a_input = np.array([list(lstm_posedict[track_id])], dtype=np.float32)
                                reshaped_array = a_input.reshape(1, 48, 17, 3)
                                window_keypoints = reshaped_array[:, :, :, :2]
                                window_scores = reshaped_array[:, :, :, 2]
                                min_x, max_x = 0, 1919
                                min_y, max_y = 0, 1079
                                min_score, max_score = 0, 1
                                window_keypoints[..., 0] = np.clip(window_keypoints[..., 0], min_x, max_x)
                                window_keypoints[..., 1] = np.clip(window_keypoints[..., 1], min_y, max_y)
                                window_scores = np.nan_to_num(window_scores)
                                window_scores = np.clip(window_scores, min_score, max_score)

                                fake_anno = dict(
                                    frame_dir='',
                                    label=-1,
                                    img_shape=(1080, 1920),
                                    original_shape=(1080, 1920),
                                    start_index=0,
                                    modality='Pose',
                                    total_frames=window_keypoints.shape[1],
                                    test_mode = True,
                                    usable_indices = np.array([0]),
                                    usable_label = np.array([1]),
                                    keypoint=window_keypoints,
                                    keypoint_score=window_scores)
                                
                                model_results = inference_recognizer(model, fake_anno)
                                confidence = model_results[0][1]
                                
                                out = MAIN_LABELS[model_results[0][0]]
                                action2 = MAIN_LABELS[model_results[1][0]]
                                #predicted_action = labels_model[model_results[0][0]]
                            else:  
                                a_input = np.array([list(lstm_posedict[track_id])], dtype=np.float32)
                                
                                # This is where to put the model and transformations and fetch the action back
                                out, confidence, action2, _, _, _ = cnn1d_infer(model, transforms, device, a_input, return_secondary=True, return_confidence=True)
                            
                            if out == "fall":
                                if confidence < fall_threshold:
                                    out = action2
                                else:
                                    show_fall_popup = True
                                    fall_popup_counter = 0
                                
                            _bbox = None
                            for person in pose_results_buffer_list[frameidx]:
                                if person['track_id'] == track_id:
                                    _bbox = person['bbox']
                                    break
                                
                            # Process the resulting action predicted by the model
                            if _bbox is not None:
                                if confidence is not None:
                                    label = f"{out} ({float(confidence):.2f})"
                                else:
                                    label = out
                                    
                                if out == "fall":
                                    color = (0, 0, 255)
                                else:
                                    color = (0, 255, 0)
                                
                                if label_from == "center":
                                    cv2.putText(video_buffer[25], label, (int(_bbox[0]+40), int(_bbox[1]-10)), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
                                else:
                                    cv2.putText(drawn_frame, label, (int(_bbox[0]+40), int(_bbox[1]-10)), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
                            
                            action_this_frame.append(out)
                            
                            current_joint.extend(format_joint)
                        else:
                            current_joint.extend(format_joint)
                    else:
                        current_joint.extend(format_joint)

                filtered_framcount=filtered_framcount+1

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                pose_results_buffer_list_update=pose_results_buffer_list_update+1

                if pose_results_buffer_list_update==buffercount+1:
                    pose_results_buffer_list_update=0

                pose_results_buffer.popleft()
                prev_pos_results = pose_results.copy()
                pose_results_buffer.append(prev_pos_results)
                frame_buffer.append(drawn_frame)

            else:
                prev_pos_results = pose_results.copy()
                pose_results_buffer.append(prev_pos_results)
                frame_buffer.append(drawn_frame)

    # ----------------------------------------------------------------

            if show_fall_popup:
                if fall_popup_counter < fall_popup_counter_threshold:
                    if label_from == "center":
                        cv2.putText(video_buffer[25], "FALL DETECTED", (int(50), int(90)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 2, cv2.LINE_AA)
                    else:
                        cv2.putText(drawn_frame, "FALL DETECTED", (int(50), int(90)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 2, cv2.LINE_AA)
                    
                    fall_popup_counter += 1
                else:
                    show_fall_popup = False
                    fall_popup_counter = 0


            
    # ----------------------------------------------------------------
            
            framecount += 1

            if framecount>48:
                
                # Get action from timestamp
                offset = 24 if label_from == "center" else 0
                current_timestamp = (framecount-offset)/fps if label_from == "center" and framecount>48 else framecount/fps
                label_this_frame = "fall" if framecount-offset>start and framecount-offset<=end else "non_fall"

                if label_from == "center":
                    cv2.putText(video_buffer[25], label_this_frame, (int(100), int(150)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2, cv2.LINE_AA)
                else:
                    cv2.putText(drawn_frame, label_this_frame, (int(100), int(150)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2, cv2.LINE_AA)

                
                print("Predicted action:", action_this_frame, "GT Action:", label_this_frame, 
                        "Frame:", current_timestamp, "num_detections:", len(action_this_frame))
                if len(action_this_frame)>0:
                    action = action_this_frame[0] if action_this_frame[0] == "fall" else "non_fall"
                else:
                    action = "No Prediction"
                
                record = [action, label_this_frame, current_timestamp, len(action_this_frame)]
                record.extend([act if act=="fall" else "non_fall" for act in action_this_frame[1:]])  # Append additional actions if available
                predictions_list.append(record)
            
            max_actions_one_frame = len(action_this_frame) if len(action_this_frame) > max_actions_one_frame else max_actions_one_frame
            
    # ----------------------------------------------------------------

            if label_from == "center":
                video_buffer.append(drawn_frame)
                if len(video_buffer) > 48:
                    video_buffer.pop(0)
                if framecount > 24:
                    if show_video:
                        cv2.imshow("Video", video_buffer[24])
                    if save_out_video:
                        videoWriter.write(video_buffer[24])
            else:
                if show_video:
                    cv2.imshow("Video", drawn_frame)
                if save_out_video:
                    videoWriter.write(drawn_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    # ----------------------------------------------------------------

        else:
            break

    vid.release()
    cv2.destroyAllWindows()
    if save_out_video:
        videoWriter.release()

    return predictions_list, max_actions_one_frame




# ----------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------

from datetime import time

def timestamp_to_frame(timestamp, fps):
    if isinstance(timestamp, time):
        # Directly access hours, minutes, and seconds from datetime.time object
        hours, minutes, seconds = timestamp.hour, timestamp.minute, timestamp.second
    else:
        # Fallback if the timestamp is a string
        hours, minutes, seconds = map(int, timestamp.split(':'))

    # Calculate total seconds
    total_seconds = hours * 3600 + minutes * 60 + seconds
    # Return the frame number
    return int(total_seconds * fps)

# ----------------------------------------------------------------------------------------------------------------------------------------------


def infer_leuven_dataset(video_path, start_end, pose_model, model, transforms, device, mapping_labels, actionmap,
                      mode=None, show_video=False, save_out_video=True, 
                      out_video_root="./out_video/", fall_threshold = 0.5, label_from = "center", infer_pyskl=False):
    
    os.makedirs(out_video_root, exist_ok=True)
    
    start, end = start_end
        
    predictions_list = []
    max_actions_one_frame = 0
    
    lstm_framecount = 48
    lstm_posedict = {}

    framecount = 0
    filtered_framcount=0
    buffercount=10

    number_joints=17
    pose_results_buffer=deque([], maxlen=buffercount)
    pose_results_buffer_list=[]
    pose_results_buffer_list_update=0

    frame_buffer=deque([], maxlen=buffercount)
    id_joints={}

    fs = 17  # Sampling frequency
    fc = 3  # Cut-off frequency of the filter
    wn = fc / (fs / 2)  # Normalize the frequency
    butter_b, butter_a = butter(2, wn)

    vid = cv2.VideoCapture(video_path)
    fps = vid.get(cv2.CAP_PROP_FPS)
    
    start = timestamp_to_frame(start, fps) + 10 
    end = timestamp_to_frame(end, fps)
    
    if save_out_video:
        size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        if mode == "1080p":
            size = (1920, 1080)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        videoWriter = cv2.VideoWriter(os.path.join(out_video_root, f'vis_{os.path.basename(video_path)}'), fourcc, fps, size)

    framecount = 0
    pose_results_buffer = deque([], maxlen=10)
    video_buffer = [None for i in range(24)] if label_from == "center" else []
    
    show_fall_popup = False
    fall_popup_counter = 0
    fall_popup_counter_threshold = 90

    while True:
        ret, frame = vid.read()
        if ret:
            frame = resize_frame(frame, mode) if mode is not None else frame
            pts, tids, bboxes, drawn_frame, orig_frame, scores = pose_model.inference(frame, framecount)

            pose_results = []
            # Reshape data in the same format as pose_results from mmpose
            for tid, bbox, pt in zip(tids, bboxes, pts):
                # swap pt 0 and 1 columns
                pt[:,[0, 1]] = pt[:,[1, 0]]
                pose_dict = {
                    "bbox": bbox,   
                    "keypoints": pt,
                    "track_id": tid,
                }
                pose_results.append(pose_dict)
    
            # ----------------------------------------------------------------
            action_this_frame = []            # For exporting current action

            if framecount>=buffercount:
                if pose_results_buffer_list_update==0:
                    pose_results_buffer_list=list(pose_results_buffer)

                    tracked_id = {}

                    for frame in pose_results_buffer_list:
                        for person in frame:
                            if person['track_id'] in tracked_id:
                                tracked_id[person['track_id']] = tracked_id[person['track_id']]+ 1
                            else:
                                tracked_id[person['track_id']] = 0

                    id_joints={}
                    for id in tracked_id.keys():
                        if tracked_id[id] < 2 or id==-1:   # if trackedid==-1 remove track, if tracked instances less than 2, remove track
                            continue

                        id_joints[id] = [ [0]*buffercount for n in range(number_joints*3) ]

                    frameidx=0
                    for frame in pose_results_buffer_list:
                        for person in frame:

                            person_id=person['track_id']
                            if person_id not in id_joints:
                                continue

                            idx=0
                            for kp in person['keypoints']:
                                id_joints[person_id][idx][frameidx]=kp[0]
                                id_joints[person_id][idx + 1][frameidx]=kp[1]
                                id_joints[person_id][idx + 2][frameidx]=kp[2]
                                idx=idx+3

                        frameidx = frameidx + 1

                    filter_joint = []
                    for track_id in id_joints.keys():
                        jidx = 0
                        for joint in id_joints[track_id]:
                            if not any(joint):
                                filter_joint=joint
                            else:
                                interp_joint = np.array(joint)

                                # interpolate
                                interp_joint[interp_joint == 0] = np.nan
                                nans, x = nan_helper(interp_joint)
                                interp_joint[nans] = np.interp(x(nans), x(~nans), interp_joint[~nans])
                                filter_joint = filtfilt(butter_b, butter_a, interp_joint)

                            id_joints[track_id][jidx]=filter_joint
                            jidx = jidx + 1

                    pose_results_buffer_list_update=pose_results_buffer_list_update+1

                frameidx=pose_results_buffer_list_update-1

                current_joint=[]

                for track_id in id_joints.keys():
                    format_joint=[]

                    for j in range(0,number_joints*3,3):
                        format_joint.append([id_joints[track_id][j][frameidx],id_joints[track_id][j+1][frameidx],id_joints[track_id][j+2][frameidx]])

                    if any(format_joint):
                        lstm_joints=[]
                        jcount=0
                        for j in format_joint:
                            if(j[0]==0 and j[1]==0):
                                continue    
                            lstm_joints.extend([j[0],j[1],j[2]])
                            jcount=jcount+1

                        # TODO: Need to delete old track_id data
                        if track_id in lstm_posedict:
                            lstm_posedict[track_id].append(lstm_joints)
                        else:
                            lstm_posedict[track_id] = deque([], maxlen=lstm_framecount)  #TODO: lstm_posedict does not clean up old track_id pose data, !! mem leak !!
                            lstm_posedict[track_id].append(lstm_joints)


                        if len(lstm_posedict[track_id])==lstm_framecount:
                            if(infer_pyskl):
                                a_input = np.array([list(lstm_posedict[track_id])], dtype=np.float32)
                                reshaped_array = a_input.reshape(1, 48, 17, 3)
                                window_keypoints = reshaped_array[:, :, :, :2]
                                window_scores = reshaped_array[:, :, :, 2]
                                min_x, max_x = 0, 1919
                                min_y, max_y = 0, 1079
                                min_score, max_score = 0, 1
                                window_keypoints[..., 0] = np.clip(window_keypoints[..., 0], min_x, max_x)
                                window_keypoints[..., 1] = np.clip(window_keypoints[..., 1], min_y, max_y)
                                window_scores = np.nan_to_num(window_scores)
                                window_scores = np.clip(window_scores, min_score, max_score)

                                fake_anno = dict(
                                    frame_dir='',
                                    label=-1,
                                    img_shape=(1080, 1920),
                                    original_shape=(1080, 1920),
                                    start_index=0,
                                    modality='Pose',
                                    total_frames=window_keypoints.shape[1],
                                    test_mode = True,
                                    usable_indices = np.array([0]),
                                    usable_label = np.array([1]),
                                    keypoint=window_keypoints,
                                    keypoint_score=window_scores)
                                
                                model_results = inference_recognizer(model, fake_anno)
                                confidence = model_results[0][1]

                                out = MAIN_LABELS[model_results[0][0]]
                                action2 = MAIN_LABELS[model_results[1][0]]
                                #predicted_action = labels_model[model_results[0][0]]
                            else:
                                a_input = np.array([list(lstm_posedict[track_id])], dtype=np.float32)
                                
                                # This is where to put the model and transformations and fetch the action back
                                out, confidence, action2, _, _, _ = cnn1d_infer(model, transforms, device, a_input, return_secondary=True, return_confidence=True)
                            
                            if out == "fall":
                                if confidence < fall_threshold:
                                    out = action2
                                else:
                                    show_fall_popup = True
                                    fall_popup_counter = 0
                                
                            _bbox = None
                            for person in pose_results_buffer_list[frameidx]:
                                if person['track_id'] == track_id:
                                    _bbox = person['bbox']
                                    break
                                
                            # Process the resulting action predicted by the model
                            if _bbox is not None:
                                if confidence is not None:
                                    label = f"{out} ({float(confidence):.2f})"
                                else:
                                    label = out
                                    
                                if out == "fall":
                                    color = (0, 0, 255)
                                else:
                                    color = (0, 255, 0)
                                
                                if label_from == "center":
                                    cv2.putText(video_buffer[25], label, (int(_bbox[0]+40), int(_bbox[1]-10)), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
                                else:
                                    cv2.putText(drawn_frame, label, (int(_bbox[0]+40), int(_bbox[1]-10)), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
                            
                            action_this_frame.append(out)
                            
                            current_joint.extend(format_joint)
                        else:
                            current_joint.extend(format_joint)
                    else:
                        current_joint.extend(format_joint)

                filtered_framcount=filtered_framcount+1

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                pose_results_buffer_list_update=pose_results_buffer_list_update+1

                if pose_results_buffer_list_update==buffercount+1:
                    pose_results_buffer_list_update=0

                pose_results_buffer.popleft()
                prev_pos_results = pose_results.copy()
                pose_results_buffer.append(prev_pos_results)
                frame_buffer.append(drawn_frame)

            else:
                prev_pos_results = pose_results.copy()
                pose_results_buffer.append(prev_pos_results)
                frame_buffer.append(drawn_frame)

    # ----------------------------------------------------------------

            if show_fall_popup:
                if fall_popup_counter < fall_popup_counter_threshold:
                    if label_from == "center":
                        cv2.putText(video_buffer[25], "FALL DETECTED", (int(50), int(90)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 2, cv2.LINE_AA)
                    else:
                        cv2.putText(drawn_frame, "FALL DETECTED", (int(50), int(90)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 2, cv2.LINE_AA)
                    
                    fall_popup_counter += 1
                else:
                    show_fall_popup = False
                    fall_popup_counter = 0


            
    # ----------------------------------------------------------------
            
            framecount += 1

            # Get action from timestamp
            current_timestamp = framecount/fps
            offset = 24 if label_from == "center" else 0
            label_this_frame = "fall" if framecount-offset>=start and framecount-offset<=end else "non_fall"

            
            if framecount>48:
                
                if label_from == "center":
                    cv2.putText(video_buffer[25], label_this_frame, (int(100), int(150)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2, cv2.LINE_AA)
                else:
                    cv2.putText(drawn_frame, label_this_frame, (int(100), int(150)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2, cv2.LINE_AA)

                
                print("Predicted action:", action_this_frame, "GT Action:", label_this_frame, 
                        "Timestamp:", current_timestamp, "num_detections:", len(action_this_frame))
                if len(action_this_frame)>0:
                    action = action_this_frame[0] if action_this_frame[0] == "fall" else "non_fall"
                else:
                    action = "No Prediction"
                
                record = [action, label_this_frame, current_timestamp, len(action_this_frame)]
                record.extend([act if act=="fall" else "non_fall" for act in action_this_frame[1:]])  # Append additional actions if available
                predictions_list.append(record)
            
            max_actions_one_frame = len(action_this_frame) if len(action_this_frame) > max_actions_one_frame else max_actions_one_frame
            
    # ----------------------------------------------------------------

            if label_from == "center":
                video_buffer.append(drawn_frame)
                if len(video_buffer) > 48:
                    video_buffer.pop(0)
                if framecount > 24:
                    if show_video:
                        cv2.imshow("Video", video_buffer[24])
                    if save_out_video:
                        videoWriter.write(video_buffer[24])
            else:
                if show_video:
                    cv2.imshow("Video", drawn_frame)
                if save_out_video:
                    videoWriter.write(drawn_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    # ----------------------------------------------------------------

        else:
            break

    vid.release()
    cv2.destroyAllWindows()
    if save_out_video:
        videoWriter.release()

    return predictions_list, max_actions_one_frame