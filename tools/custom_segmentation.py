import cv2
import numpy as np


def segment_object(depth_frame, bbox, depth_threshold=1000):
    """
    Segment the object within the bounding box from the depth image and calculate the minimum depth.

    Args:
    depth_frame (numpy.ndarray): The depth image.
    bbox (tuple): The bounding box in xyxy format (xmin, ymin, xmax, ymax).
    depth_threshold (int): The threshold value for depth segmentation.

    Returns:
    numpy.ndarray: Mask of the segmented object.
    int: The minimum depth of the segmented object in the same unit as the depth_frame.
    """
    xmin, ymin, xmax, ymax = bbox
    roi = depth_frame[ymin:ymax, xmin:xmax]

    # Apply binary thresholding
    _, mask = cv2.threshold(roi, depth_threshold, 65535, cv2.THRESH_BINARY_INV)

    kernel = np.ones((3, 3), np.uint8)
    cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)

    # Compute minimum depth of the object
    object_depth_values = np.where(cleaned_mask == 255, roi, 0)
    nonzero_values = object_depth_values[object_depth_values > 0]

    if nonzero_values.size > 0:
        min_depth = np.min(nonzero_values)
    else:
        min_depth = 0  # or set it to a default value

    return cleaned_mask, min_depth
