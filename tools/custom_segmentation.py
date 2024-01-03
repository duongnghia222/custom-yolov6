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

    kernel = np.ones((3, 3), np.uint8)
    cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)

    # Compute average depth of the object
    object_depth_values = np.where(cleaned_mask == 255, roi, 0)
    nonzero_count = np.count_nonzero(object_depth_values)

    if nonzero_count > 0:
        average_depth = int(np.sum(object_depth_values) / nonzero_count)
    else:
        average_depth = None  # or set it to a default value

    return cleaned_mask, average_depth

