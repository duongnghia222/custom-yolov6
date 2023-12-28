import pathlib
import torch
import os
import time
import cv2
from tools.realsense_camera import *
from tools.finger_count import FingersCount
from tools.custom_inferer import Inferer
# from yolov6.utils.events import load_yaml



def run(fc, yolo):
    rs_camera = RealsenseCamera()
    print("Starting RealSense camera detection. Press 'q' to quit.")

    mode_selected = False
    object_selected = False
    finding_mode = False
    last_finger_count = None
    consistent_count_start = None
    object_to_find = None

    while True:
        ret, color_frame, depth_frame = rs_camera.get_frame_stream()
        if not ret:
            print("Error: Could not read frame.")
            break

        finger_counts = fc.infer(color_frame)
        print(finger_counts)

        if not mode_selected:
            # Check for mode selection
            if finger_counts == [0, 1]:
                if last_finger_count == [0, 1]:
                    if time.time() - consistent_count_start >= 3:
                        mode_selected = True
                        finding_mode = True
                        print("Finding mode activated.")
                else:
                    last_finger_count = [0, 1]
                    consistent_count_start = time.time()
            else:
                last_finger_count = finger_counts
        elif finding_mode and not object_selected:
            # Check for object selection
            if last_finger_count == finger_counts:
                if time.time() - consistent_count_start >= 3:
                    object_selected = True
                    object_to_find = finger_counts
                    print(f"Object to find selected: {object_to_find}")
            else:
                last_finger_count = finger_counts
                consistent_count_start = time.time()
        elif object_selected:
            # Object detection logic
            if finger_counts == [0, 0]:
                print("Object found by user, stopping detection.")
                break
            else:
                # Call YOLO object detection here
                detections = yolo.object_finder(color_frame, class_num=finger_counts_mapping_obj(object_to_find))
                # Process and display detections

        cv2.imshow('RealSense Camera Detection', color_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    rs_camera.release()
    cv2.destroyAllWindows()

def finger_counts_mapping_obj(object_to_find):
    if "bottle":
        return 0


def create_inferer(weights='yolov6s_mbla.pt',
        yaml='data/coco.yaml',
        img_size=[640, 640],
        conf_threshold=0.4,
        iou_threshold=0.45,
        max_det=1000,
        device='0',
        save_txt=False,
        not_save_img=True,
        save_dir=None,
        view_img=True,
        classes=None,
        agnostic_nms=False,
        project='runs/inference',
        name='exp',
        hide_labels=False,
        hide_conf=False,
        half=False):
    infer = Inferer(weights, device, yaml, img_size, half, conf_threshold, iou_threshold, agnostic_nms, max_det)
    return infer


if __name__ == "__main__":
    # PATH_YOLOv6 = pathlib.Path(__file__).parent
    # DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # CLASS_NAMES = load_yaml(str(PATH_YOLOv6 / "data/coco.yaml"))['names']
    # Load the YOLOv6 model (choose the appropriate function based on the model size you want to use)\
    screen_width, screen_height = [720, 1280]
    fc = FingersCount(screen_width, screen_height)
    yolo = create_inferer()
    run(fc, yolo)





