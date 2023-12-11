import cv2
import mediapipe as mp
import time
import math
import numpy as np

def main():
    cap = cv2.VideoCapture(0)
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
    tipids = [4, 8, 12, 16, 20]

    for id, lm in enumerate(handlms.landmark):
        cx, cy = int(lm.x * w), int(lm.y * h)
        lmlist.append([id, cx, cy])

    if len(lmlist) == 21:
        # Improved thumb detection
        if is_thumb_up(lmlist):
            print("thumb up")
        fingerlist.append(is_thumb_up(lmlist))

        # Other fingers
        for id in range(1, 5):
            fingerlist.append(int(lmlist[tipids[id]][2] < lmlist[tipids[id] - 2][2]))

    return fingerlist.count(1)


def is_thumb_up(lmlist):
    # Use the vector from the wrist to the base of the index finger as a reference
    reference_vector = (lmlist[5][1] - lmlist[0][1], lmlist[5][2] - lmlist[0][2])
    thumb_vector = (lmlist[4][1] - lmlist[2][1], lmlist[4][2] - lmlist[2][2])

    # Calculate angle between vectors
    angle = calculate_angle(reference_vector, thumb_vector)

    # Determine if thumb is up (customize the threshold as needed)
    return angle > 60  # Example threshold, adjust based on your testing


def calculate_angle(v1, v2):
    # Returns the angle in degrees between vectors 'v1' and 'v2':
    cosang = np.dot(v1, v2)
    sinang = np.linalg.norm(np.cross(v1, v2))
    return np.arctan2(sinang, cosang) * (180 / np.pi)

if __name__ == "__main__":
    main()
