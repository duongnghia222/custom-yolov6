from hubconf import *
from tools.realsense_camera import *
from tools.finger_count import *


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


if __name__ == "__main__":
    # Load the YOLOv6 model (choose the appropriate function based on the model size you want to use)
    model = create_model_v4(model_name="yolov6s_mbla", class_names=CLASS_NAMES, device=DEVICE, img_size=640,
                            conf_thres=0.25, iou_thres=0.45, max_det=1000)

    # Perform real-time detection on RealSense camera feed
    run(model)


