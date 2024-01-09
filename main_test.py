import pathlib
import torch
import os
import time
import cv2
import threading
from tools.finger_count import FingersCount
from tools.tracker import Tracker
from tools.instruction import navigate_to_object
from tools.test_pyttx3 import speak
from tools.voice_navigator import TextToSpeech
voice = TextToSpeech()
from tools.realsense_camera import *
from tools.custom_segmentation import segment_object
from tools.custom_inferer import Inferer
from yolov6.utils.events import load_yaml


def run(fc, voice, coco_yaml, custom_dataset_yaml):
    webcam = cv2.VideoCapture(0)
    print("Starting RealSense camera detection. Press 'q' to quit.")
    model = create_inferer()
    mode = 'disabled' # for debug, change to disabled after that
    last_gesture = None
    gesture_start = None
    detection = None
    last_finder_call_time = None
    object_to_find = {"name": "cup", "conf_threshold": 0.5} # for debug, change to None after that
    # object_to_find = None
    depth_frame = 0
    while True:
        ret, color_frame = webcam.read()
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
                    model = None
                    print("All modes disabled.")
                elif finger_counts == [0, 1]:
                    mode = 'finding'
                    object_to_find = None
                    print("loaded")
                    model = create_inferer()
                    print("Finding mode activated.")
                elif finger_counts == [0, 2]:
                    mode = 'detecting'
                    object_to_find = None
                    model = create_inferer(weights='dangerous_obj.pt', yaml='data/dangerous_obj.yaml')
                    print("Detecting mode activated.")
                elif finger_counts == [0, 5]:
                    print("Program stopping...")
                    break

        # Implement the functionalities for each mode
        if mode == 'finding':
            # Implement finding functionality
            if finger_counts != last_gesture:
                last_gesture = finger_counts
                gesture_start = time.time()
            elif time.time() - gesture_start >= 2 and not object_to_find:
                object_to_find = finger_counts_mapping_obj(finger_counts)["name"]
            if object_to_find:
                if last_finder_call_time is None:
                    last_finder_call_time = time.time()
                object_index = coco_yaml.index(object_to_find["name"])
                # print(f"Looking for: {objec   t_to_find['name']} with index", object_index)
                conf_threshold = object_to_find["conf_threshold"]
                if detection is None or (time.time() - last_finder_call_time >= 1):
                    last_finder_call_time = time.time()
                    detection = model.object_finder(color_frame, object_index, predict_threshold=conf_threshold)
                    if detection is not None:
                        if len(detection) > 1:
                            detection = detection[0]
                        detection = detection.flatten()

                if detection is not None and len(detection):
                    *xyxy, conf, cls = detection
                    #[285, 194, 394, 298]
                    xmin, ymin, xmax, ymax = map(int, xyxy)  # Convert each element to an integer
                    object_mask, depth = segment_object(depth_frame, [xmin, ymin, xmax, ymax])
                    # cv2.imshow("Object Mask", object_mask)
                    # color_roi = color_frame[ymin:ymax, xmin:xmax]
                    # _, binary_mask = cv2.threshold(object_mask, 127, 255, cv2.THRESH_BINARY)
                    #
                    # isolated_object = cv2.bitwise_and(color_roi, color_roi, mask=binary_mask)
                    # color_image_with_object = color_frame.copy()
                    # color_image_with_object[ymin:ymax, xmin:xmax] = isolated_object
                    # cv2.imshow("Color Image with Object", color_image_with_object)
                    #
                    #
                    # yolo.plot_box_and_label(color_frame, max(round(sum(color_frame.shape) / 2 * 0.003), 2), xyxy,\
                    #                         depth, label='Distance', color=(128, 128, 128), txt_color=(255, 255, 255),\
                    #                         font=cv2.FONT_HERSHEY_COMPLEX)
                    print("distance", depth)

                    instruction = navigate_to_object([xmin, ymin, xmax, ymax], depth, color_frame)
                    voice.speak(instruction)

        elif mode == 'detecting':
            # Implement detecting functionality
            dangerous_obj = model.dangerous_object_detection(color_frame, conf_threshold=0.5)
            if dangerous_obj is not None:
                if len(dangerous_obj) > 1:
                    dangerous_obj = dangerous_obj[0]
                dangerous_obj = dangerous_obj.flatten()
            if dangerous_obj is not None and len(dangerous_obj):
                *xyxy, conf, cls = dangerous_obj
                if isinstance(cls, torch.Tensor):
                    if cls.nelement() == 1:
                        cls = int(cls.item())


                print(DANGEROUS_CLASS_NAMES[cls])
                # [285, 194, 394, 298]
                xmin, ymin, xmax, ymax = map(int, xyxy)  # Convert each element to an integer
                object_mask, depth = segment_object(depth_frame, [xmin, ymin, xmax, ymax])

                model.plot_box_and_label(color_frame, max(round(sum(color_frame.shape) / 2 * 0.003), 2), xyxy,\
                                        depth, label='Distance', color=(128, 128, 128), txt_color=(255, 255, 255),\
                                        font=cv2.FONT_HERSHEY_COMPLEX)
                instruction = navigate_to_object([xmin, ymin, xmax, ymax], depth, color_frame)
                if instruction == "move forward":
                    instruction = "front"
                elif instruction == "turn left":
                    instruction = "right"
                elif instruction == "turn right":
                    instruction = "left"
                elif instruction == "stop":
                    instruction = "very front"
                guide = DANGEROUS_CLASS_NAMES[cls] + "on the" + instruction + str(depth) + "centimeters away"
                voice.speak(guide)

        cv2.imshow('RealSense Camera Detection', color_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    webcam.release()
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
        project='runs/infereqnce',
        name='exp',
        hide_labels=False,
        hide_conf=False,
        half=False):
    infer = Inferer(weights, device, yaml, img_size, half, conf_threshold, iou_threshold, agnostic_nms, max_det)
    return infer


if __name__ == "__main__":
    speak("Please wait for system to start")
    PATH_YOLOv6 = pathlib.Path(__file__).parent
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CLASS_NAMES = load_yaml(str(PATH_YOLOv6 / "data/coco.yaml"))['names']
    DANGEROUS_CLASS_NAMES = load_yaml(str(PATH_YOLOv6 / "data/dangerous_obj.yaml"))['names']

    # Load the YOLOv6 model (choose the appropriate function based on the model size you want to use)\
    screen_width, screen_height = [720, 1280]
    fc = FingersCount(screen_width, screen_height)
    # yolo = create_inferer()
    # custom_model = create_inferer(weights='dangerous_obj.pt', yaml='data/dangerous_obj.yaml')
    run(fc, voice, coco_yaml=CLASS_NAMES, custom_dataset_yaml=DANGEROUS_CLASS_NAMES)







