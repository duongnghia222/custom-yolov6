import pathlib
import torch
import os
from tools.realsense_camera import *
from tools.finger_count import *
from yolov6.utils.events import load_yaml
from yolov6.core.inferer import Inferer


def run(model):
    # Initialize RealSense camera
    rs_camera = RealsenseCamera()

    print("Starting RealSense camera detection. Press 'q' to quit.")

    while True:
        # Get frame from RealSense camera
        ret, color_frame, depth_frame = rs_camera.get_frame_stream()

        if not ret:
            print("Error: Could not read frame.")
            break

        # Process the color frame for detection
        predictions = model.predict(color_frame)

        # Visualize the predictions on the color frame
        model.show_predict(color_frame, predictions)

        # Display the color frame with detections
        cv2.imshow('RealSense Camera Detection', color_frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the RealSense camera
    rs_camera.release()
    cv2.destroyAllWindows()


def create_inferer(weights='yolov6s_mbla.pt',
        yaml='data/coco.yaml',
        img_size=[640, 640],
        conf_thres=0.4,
        iou_thres=0.45,
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
    inferer = Inferer(source, webcam, webcam_addr, use_depth_cam, weights, device, yaml, img_size, half)


if __name__ == "__main__":
    PATH_YOLOv6 = pathlib.Path(__file__).parent
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CLASS_NAMES = load_yaml(str(PATH_YOLOv6 / "data/coco.yaml"))['names']
    # Load the YOLOv6 model (choose the appropriate function based on the model size you want to use)





