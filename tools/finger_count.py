import cv2
import mediapipe as mp
import time
# from tools.realsense_camera import *
import numpy as np



class FingersCount:
    def __init__(self, screen_width, screen_height, cap):
        pass

    @staticmethod
    def is_thumb_up(lmlist):
        # Use the vector from the wrist to the base of the index finger as a reference
        vector34 = (lmlist[3][1] - lmlist[4][1], lmlist[3][2] - lmlist[4][2])
        vector32 = (lmlist[3][1] - lmlist[2][1], lmlist[3][2] - lmlist[2][2])

        # Calculate angle between vectors
        angle = abs(calculate_angle(vector34, vector32))

        # Determine if thumb is up (customize the threshold as needed)
        # is_thumb = (angle > 0 and lmlist[4][1] < lmlist[5][1]) or (angle < 0 and lmlist[4][1] > lmlist[5][1])
        print(angle)
        return angle < 40 and lmlist[4][2] < lmlist[2][2]

    @staticmethod
    def calculate_angle(v1, v2):
        # Calculate the dot product of v1 and v2
        dot_product = np.dot(v1, v2)

        # Compute the norms (magnitudes) of the vectors
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)

        # Calculate the cosine of the angle (ensure it's within [-1, 1] to avoid numerical issues)
        cos_angle = np.clip(dot_product / (norm_v1 * norm_v2), -1, 1)

        # Calculate the angle in radians and then convert to degrees
        angle_radians = np.arccos(np.abs(cos_angle))  # Use abs to ensure a positive angle
        angle_degrees = np.degrees(angle_radians)

        return angle_degrees


SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720

def main():
    # rs = RealsenseCamera()
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, SCREEN_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, SCREEN_HEIGHT)
    if not cap.isOpened():
        print("Error: Camera not accessible")
        return

    medhands = mp.solutions.hands
    # Initialize with max_num_hands set to 2 for detecting two hands
    hands = medhands.Hands(max_num_hands=2, min_detection_confidence=0.8)
    draw = mp.solutions.drawing_utils

    ptime = 0

    while True:
        success, img = cap.read()
        if not success:
            break

        img, finger_counts = process_frame(img, medhands, hands, draw)

        # FPS calculation and display
        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime
        cv2.putText(img, f'FPS: {int(fps)}', (10, 40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        # Display finger counts for each hand
        for i, count in enumerate(finger_counts):
            cv2.putText(img, f'Hand {i+1}: {count}', (10, 70 + 30*i), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        cv2.imshow("Hand Gestures", img)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def process_frame(img, medhands, hands, draw):
    # img = cv2.flip(img, 1)
    imgrgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = hands.process(imgrgb)

    finger_counts = []

    if res.multi_hand_landmarks:
        for handlms in res.multi_hand_landmarks:
            h, w, c = img.shape
            finger_count = update_finger_list(handlms, h, w)
            finger_counts.append(finger_count)

            draw.draw_landmarks(img, handlms, medhands.HAND_CONNECTIONS,
                                draw.DrawingSpec(color=(0, 255, 204), thickness=2, circle_radius=2),
                                draw.DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=3))
    return img, finger_counts


def update_finger_list(handlms, h, w):
    lmlist = []
    fingerlist = []
    tipids = [4, 8, 12, 16, 20]  # 4 -> thumb tip, 8,12,16,20 -> index,middle,ring,pinky tips

    # Get all landmarks of a hand
    for id, lm in enumerate(handlms.landmark):
        cx, cy = int(lm.x * w), int(lm.y * h)
        lmlist.append([id, cx, cy])

    # Once we have all 21 landmarks, process them
    if len(lmlist) == 21:
        # Improved thumb detection
        fingerlist.append(is_thumb_up(lmlist))
        # print(lmlist[0][1])

        # Other fingers
        for id in range(1, 5):
            if lmlist[0][1] < SCREEN_WIDTH//2:  # this is left hand
                fingerlist.append(int(lmlist[tipids[id]][1] > lmlist[tipids[id] - 2][1]))
            else:
                fingerlist.append(int(lmlist[tipids[id]][1] < lmlist[tipids[id] - 2][1]))

    return fingerlist.count(1)


def is_thumb_up(lmlist):
    # Use the vector from the wrist to the base of the index finger as a reference
    vector34 = (lmlist[3][1] - lmlist[4][1], lmlist[3][2] - lmlist[4][2])
    vector32 = (lmlist[3][1] - lmlist[2][1], lmlist[3][2] - lmlist[2][2])

    # Calculate angle between vectors
    angle = abs(calculate_angle(vector34, vector32))

    # Determine if thumb is up (customize the threshold as needed)
    # is_thumb = (angle > 0 and lmlist[4][1] < lmlist[5][1]) or (angle < 0 and lmlist[4][1] > lmlist[5][1])
    print(angle)
    return angle < 40 and lmlist[4][2] < lmlist[2][2]


def calculate_angle(v1, v2):
    # Calculate the dot product of v1 and v2
    dot_product = np.dot(v1, v2)

    # Compute the norms (magnitudes) of the vectors
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    # Calculate the cosine of the angle (ensure it's within [-1, 1] to avoid numerical issues)
    cos_angle = np.clip(dot_product / (norm_v1 * norm_v2), -1, 1)

    # Calculate the angle in radians and then convert to degrees
    angle_radians = np.arccos(np.abs(cos_angle))  # Use abs to ensure a positive angle
    angle_degrees = np.degrees(angle_radians)

    return angle_degrees

if __name__ == "__main__":
    main()
