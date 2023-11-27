import cv2
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
while True:
    ok, img = cap.read()
    hands, img = detector.findHands(img)
    cv2.imshow("Res", img)
    cv2.waitKey(1)