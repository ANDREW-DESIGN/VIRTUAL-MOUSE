from cvlearn import FingerCounter as fc
import cvlearn.HandTrackingModule as handTracker
import cv2

cap = cv2.VideoCapture(0)

detector = handTracker.handDetector(maxHands=1)

counter = fc.FingerCounter()
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 180)

    frame = detector.findHands(frame)
    lmList, bbox = detector.findPosition(frame)

    if lmList:
        frame1 = counter.drawCountedFingers(frame, lmList, bbox)

    cv2.imshow("res", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cv2.destroyAllWindows()
