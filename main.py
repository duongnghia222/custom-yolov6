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

    mode = 'disabled'
    last_gesture = None
    gesture_start = None

    while True:
        ret, color_frame, depth_frame = rs_camera.get_frame_stream()
        if not ret:
            print("Error: Could not read frame.")
            break

        finger_counts = fc.infer(color_frame)

        # Only change gestures if the current mode is disabled or a mode exit gesture is detected
        if mode == 'disabled' or finger_counts in [[0, 0], [0, 5]]:
            if finger_counts != last_gesture:
                last_gesture = finger_counts
                gesture_start = time.time()

            # Check if the gesture is held for 2 seconds
            if time.time() - gesture_start >= 2:
                if finger_counts == [0, 0]:
                    mode = 'disabled'
                    print("All modes disabled.")
                elif finger_counts == [0, 1]:
                    mode = 'finding'
                    print("Finding mode activated.")
                elif finger_counts == [0, 2]:
                    mode = 'detecting'
                    print("Detecting mode activated.")
                elif finger_counts == [0, 5]:
                    print("Program stopping...")
                    break

        # Implement the functionalities for each mode
        if mode == 'finding':
            # Implement finding functionality
            print("Finding mode")
        elif mode == 'detecting':
            # Implement detecting functionality
            print("Detecting mode")

        cv2.imshow('RealSense Camera Detection', color_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    rs_camera.release()
    cv2.destroyAllWindows()

def finger_counts_mapping_obj(object_code):
    if object_code == [1, 0]:
        return "bottle"


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





