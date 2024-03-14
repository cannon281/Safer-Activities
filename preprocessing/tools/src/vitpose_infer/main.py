from .model_builder import build_model
import torch
from .tracker import byte_tracker
from deepsort.tracker import DeepSortTracker
from deep_sort_realtime.deepsort_tracker import DeepSort

from .pose_utils.visualizer import plot_tracking

# from torch2trt import torch2trt, TRTModule
from .pose_utils.pose_viz import draw_points, draw_skeleton, draw_points_and_skeleton, joints_dict, check_video_rotation
from .pose_utils.pose_utils import pose_points_yolo5, pose_points_yolo5_with_heatmaps
from .pose_utils.pose_utils_deepsort import pose_points_yolo5_deepsort
from .pose_utils.timerr import Timer
from .pose_utils.general_utils import polys_from_pose,make_parser
# ssl._create_default_https_context = ssl._create_unverified_context ##Bypass certificate has expired error for yolov5
import logging
from .pose_utils.logger_helper import CustomFormatter
import numpy as np
from ultralytics import YOLO

import math


logger = logging.getLogger("Tracker !")
logger.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.WARNING)

ch.setFormatter(CustomFormatter())

logger.addHandler(ch)
logger.propagate=False


class VitInference:
    # tracker = byte_tracker.BYTETracker(make_parser().parse_known_args()[0],frame_rate=30)
    def __init__(self,pose_path, tensorrt=False, tracking_method="bytetrack", det_type="yolov8"):
        super(VitInference,self).__init__()
        self.tensorrt = tensorrt
        self.tracking_method = tracking_method
        self.det_type = det_type
        
        # If using BYTETracker, default
        if self.tracking_method == "bytetrack":
            self.tracker = byte_tracker.BYTETracker(make_parser().parse_known_args()[0],frame_rate=30)
        else:
            self.tracker = DeepSort()
        
        self.pose_path = pose_path
        
        if self.det_type == "yolov8":
            self.model = YOLO("yolov8x")
        elif self.det_type == "yolov5":
            self.model = torch.hub.load('ultralytics/yolov5:v6.2', 'yolov5n', pretrained=True)
        elif self.det_type == "yolov7":
            self.model = torch.hub.load('WongKinYiu/yolov7', 'custom', 'yolov7-e6.pt',
                                    force_reload=True, source='local')   
        if self.tensorrt:
            pose_split = self.pose_path.split('.')[-1]
            assert pose_split =='engine'
            from .pose_utils.ViTPose_trt import TRTModule_ViTPose
            self.pose = TRTModule_ViTPose(path=self.pose_path,device='cuda:0')
        else:
            self.pose = build_model('ViTPose_base_coco_256x192',self.pose_path)
            
        self.pose.cuda().eval()

        frame_id = 0
        self.timer = Timer()
        timer_track = Timer()
        timer_det = Timer()

        # frame = cv2.resize(frame,(640,360))

    def inference(self,img,frame_id=0, return_heatmaps=False):
        frame_orig = img.copy()
        self.timer.tic()
        
        if return_heatmaps:
            pts,online_tlwhs,online_ids,online_scores, heatmaps = pose_points_yolo5_with_heatmaps(self.model, img, self.pose, self.tracker,self.tensorrt)
        else:
            if self.tracking_method == "bytetrack":
                pts,online_tlwhs,online_ids,online_scores = pose_points_yolo5(self.model, img, self.pose, self.tracker,self.tensorrt, det_type=self.det_type)
            else:
                pts,online_tlwhs,online_ids,online_scores = pose_points_yolo5_deepsort(self.model, img, self.pose, self.tracker,self.tensorrt,det_type=self.det_type)

        # Remove low confidence detections using online scores from bounding box detetion
        # print("----------------------------------")
        # print(len(pts))
        
        bbox_threshold = 0.3 # 0.3 seems to be working best
        if len(online_scores)!=len([i for i in online_scores if i>bbox_threshold]):
            to_delete = [i for i in range(len(online_scores)) if online_scores[i]<bbox_threshold]
            del_list=sorted(to_delete, key=int, reverse=True)
            for d in del_list:
                del online_tlwhs[d]
                del online_ids[d]
                del online_scores[d]
                pts = np.delete(pts, d, axis=0)
                if return_heatmaps:
                    heatmaps = np.delete(heatmaps, d, axis=0)
                    
        # print(len(pts))

        # Remove low confidence detections using confidence threshold from pose estimation
        keypoints_threshold = 0.1 # 0.3 seems to be working best
        minimum_num_keypoints = 6 # This means that there can be upto minimum_num_keypoints kps that are below the keypoints_threshold
        max_value_threshold = 10000  # Set this to the maximum expected value in your data.

        if pts is not None:
            to_delete = []
            for i, pt in enumerate(pts):
                 # Excluding indices for eyes and ears (1, 2, 3, 4)
                keypoints_to_consider = np.delete(pt, [1, 2, 3, 4], axis=0)
                
                # Checking for NaN and weird values
                valid_mask = ~np.isnan(keypoints_to_consider[:, 2]) & (keypoints_to_consider[:, 2] <= max_value_threshold)
                keypoints_to_consider = keypoints_to_consider[valid_mask]

                # Set coordinates of keypoints below threshold to 0
                low_confidence_mask = keypoints_to_consider[:, 2] < keypoints_threshold
                keypoints_to_consider[low_confidence_mask, 0:2] = 0  # Setting x and y coordinates to 0

                # Count keypoints below the threshold
                below_threshold = np.sum(low_confidence_mask)

                # Check if the count is above the minimum number
                if below_threshold > minimum_num_keypoints:
                    to_delete.append(i)
            
            # Delete the marked keypoints from pts and other related lists.
            for d in sorted(to_delete, reverse=True):
                pts = np.delete(pts, d, axis=0)
                if 'return_heatmaps' in locals() and return_heatmaps:  # Check if heatmaps need to be deleted.
                    heatmaps = np.delete(heatmaps, d, axis=0)
                del online_tlwhs[d]
                del online_ids[d]
                del online_scores[d]
        
        # print(len(pts))
        # print("----------------------------------")


        self.timer.toc()
        if len(online_ids)>0:
            # timer_track.tic()
            # self.timer.tic()
            online_im = frame_orig.copy()
            online_im = plot_tracking(
                frame_orig, online_tlwhs, online_ids, frame_id=frame_id, fps=1/self.timer.average_time
            )
            # self.timer.toc()
            if pts is not None:
                for i, (pt, pid) in enumerate(zip(pts, online_ids)):
                    
                    online_im=draw_points_and_skeleton(online_im, pt, joints_dict()['coco']['skeleton'], person_index=pid,
                                                            points_color_palette='gist_rainbow', skeleton_color_palette='jet',points_palette_samples=10,confidence_threshold=0.3)

        else:
            online_im = frame_orig
        
        if return_heatmaps:
            return pts,online_ids,online_tlwhs,online_im,frame_orig,heatmaps
        return pts,online_ids,online_tlwhs,online_im,frame_orig,online_scores



