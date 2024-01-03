import pathlib
import torch
import os
import time
import cv2
from tools.realsense_camera import *
from tools.finger_count import FingersCount
from tools.tracker import Tracker
from tools.custom_inferer import Inferer
from yolov6.utils.events import load_yaml


def run(fc, yolo, coco_yaml, custom_dataset_yaml):
    rs_camera = RealsenseCamera()
    print("Starting RealSense camera detection. Press 'q' to quit.")

    mode = 'finding' # for debug, change to disabled after that
    last_gesture = None
    gesture_start = None
    object_to_find = {"name": "bottle", "conf_threshold": 0.4} # for debug, change to None after that

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
                    object_to_find = None
                    print("All modes disabled.")
                elif finger_counts == [0, 1]:
                    mode = 'finding'
                    object_to_find = None
                    print("Finding mode activated.")
                elif finger_counts == [0, 2]:
                    mode = 'detecting'
                    object_to_find = None
                    print("Detecting mode activated.")
                elif finger_counts == [0, 5]:
                    print("Program stopping...")
                    break

        # Implement the functionalities for each mode
        if mode == 'finding':
            # Implement finding functionality
            print("Finding mode")
            if finger_counts != last_gesture:
                last_gesture = finger_counts
                gesture_start = time.time()
            elif time.time() - gesture_start >= 2 and not object_to_find:
                object_to_find = finger_counts_mapping_obj(finger_counts)["name"]
            if object_to_find:
                object_index = coco_yaml.index(object_to_find["name"])
                print(f"Looking for: {object_to_find['name']} with index", object_index)
                conf_threshold = object_to_find["conf_threshold"]
                detection = yolo.object_finder(color_frame, object_index, predict_threshold=conf_threshold)
                print(detection)
                if detection is not None and len(detection):
                    if len(detection) > 1:
                        detection = detection[0]
                    detection_flat = detection.flatten()
                    print(detection_flat)
                    *xyxy, conf, cls = detection_flat
                    xmin, ymin, xmax, ymax = xyxy

                    center_x = (xmin + xmax) / 2
                    center_y = (ymin + ymax) / 2
                    depth_point = depth_frame[int(center_y), int(center_x)]
                    print("Depth Point:", depth_point)
                    print(depth_frame.shape[:2])
                    print(color_frame.shape[:2])
                    # print(xyxy)
                    # yolo.plot_box_and_label(color_frame, max(round(sum(color_frame.shape) / 2 * 0.003), 2), xyxy,\
                    #                         depth_frame, label='', color=(128, 128, 128), txt_color=(255, 255, 255),\
                    #                         font=cv2.FONT_HERSHEY_COMPLEX)


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
        return {"name": "bottle", "conf_threshold": 0.4}
    if object_code == [1, 1]:
        return {"name": "cup", "conf_threshold": 0.4}


def create_inferer(weights='yolov6s_mbla.pt',
        yaml='data/coco.yaml',
        img_size=[640,640],
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
    PATH_YOLOv6 = pathlib.Path(__file__).parent
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CLASS_NAMES = load_yaml(str(PATH_YOLOv6 / "data/coco.yaml"))['names']
    # Load the YOLOv6 model (choose the appropriate function based on the model size you want to use)\
    screen_width, screen_height = [720, 1280]
    # fc = FingersCount(screen_width, screen_height)
    # yolo = create_inferer()
    # run(fc, yolo, coco_yaml=CLASS_NAMES, custom_dataset_yaml=None)

    video_capture = cv2.VideoCapture(0)

    # Read the first frame
    ok, frame = video_capture.read()

    # Define a bounding box (x, y, width, height)
    bbox = (100, 100, 50, 50)

    # Create a Tracker instance
    tracker = Tracker(type="KCF", frame=frame, bbox=bbox)

    while True:
        # Read a new frame from the webcam
        ok, frame = video_capture.read()

        # Track the object in the new frame
        tracker.track(frame, bbox)

        # Display the resulting frame
        cv2.imshow('Object Tracking', frame)

        # Exit if the user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close the window
    video_capture.release()
    cv2.destroyAllWindows()





