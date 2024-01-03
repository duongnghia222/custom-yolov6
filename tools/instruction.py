import pyttsx3
import cv2


def navigate_to_object(bbox, depth, color_frame):
    """
    Navigates the blind user towards the object using audio instructions based on bounding box and depth information.

    Parameters:
    bbox (list): A list containing the bounding box coordinates [xmin, ymin, xmax, ymax].
    depth (float): The depth of the object from the camera.

    Returns:
    str: Navigation instruction ('turn left', 'turn right', 'move forward', or 'stop').
    """
    xmin, ymin, xmax, ymax = bbox
    box_center_x = int((xmin + xmax) / 2)
    frame_center_x = int(color_frame.shape[1] / 2)


    # Adjust threshold based on depth
    base_threshold = 50
    scaling_factor = 1000
    dynamic_threshold = base_threshold + (scaling_factor / max(depth, 1))

    # Draw dynamic threshold lines on the color frame
    left_threshold = int(frame_center_x - dynamic_threshold)
    right_threshold = int(frame_center_x + dynamic_threshold)
    cv2.line(color_frame, (left_threshold, 0), (left_threshold, color_frame.shape[0]), (0, 255, 0), 2)  # Left line
    cv2.line(color_frame, (right_threshold, 0), (right_threshold, color_frame.shape[0]), (0, 255, 0), 2)  # Right line

    # Determine the direction to move
    if box_center_x < frame_center_x - dynamic_threshold:
        direction = "turn left"
    elif box_center_x > frame_center_x + dynamic_threshold:
        direction = "turn right"
    else:
        direction = "move forward"

    # Incorporate depth information for distance
    minimum_safe_distance = 50  # Define this based on your requirements
    if depth < minimum_safe_distance:
        instruction = "stop"
    else:
        instruction = f"{direction}"


    return instruction
