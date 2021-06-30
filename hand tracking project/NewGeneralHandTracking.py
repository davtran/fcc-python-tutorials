import cv2
import time
import HandTrackingModule as htm

previousTime = 0
currentTime = 0
cap = cv2.VideoCapture(0)
detector = htm.HandDetector()

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    landmark_list = detector.findPosition(img)
    if len(landmark_list) !=0:
        print(landmark_list[4])

    currentTime = time.time()
    fps = 1/(currentTime - previousTime)
    previousTime = currentTime

    cv2.putText(img, str(int(fps)), (10, 70),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 3, (255, 0, 255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
