import cv2
import numpy as np


def segment_object(depth_frame, bbox):
    """
    Segment the object within the bounding box from the depth image.

    Args:
    depth_frame (numpy.ndarray): The depth image.
    bbox (tuple): The bounding box in xyxy format (xmin, ymin, xmax, ymax).

    Returns:
    numpy.ndarray: Mask of the segmented object.
    """
    # Extract ROI from the depth image
    xmin, ymin, xmax, ymax = bbox
    roi = depth_frame[ymin:ymax, xmin:xmax]

    # Convert ROI to 8-bit if necessary
    roi_8bit = cv2.normalize(roi, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

    # Apply Otsu's thresholding
    _, mask = cv2.threshold(roi_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Optional: Apply morphological operations to clean up the mask
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask

