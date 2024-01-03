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
    middle_x = color_frame.shape[1] // 2
    depth = depth//10
    scale = 0.1
    middle_diff = int(depth * scale)

    if middle_diff > 160:
        middle_diff = 160
    if middle_diff < 70:
        middle_diff = 70
    left_bound = middle_x - middle_diff
    right_bound = middle_x + middle_diff
    cv2.line(color_frame, (left_bound, 0), (left_bound, color_frame.shape[0]), (0, 255, 0), 2)  # Left line
    cv2.line(color_frame, (right_bound, 0), (right_bound, color_frame.shape[0]), (0, 255, 0), 2)  # Right line

    # Determine the direction to move
    if box_center_x < frame_center_x - middle_diff:
        direction = "turn left"
    elif box_center_x > frame_center_x + middle_diff:
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
