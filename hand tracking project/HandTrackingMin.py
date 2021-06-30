import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mp_drawing = mp.solutions.drawing_utils
previousTime = 0
currentTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for id, landmark in enumerate(hand_landmarks.landmark):
                height, width, channel = img.shape
                cx, cy = int(landmark.x * width), int(landmark.y * height)
                print(id, cx, cy)
                if id == 4:
                    cv2.circle(img, (cx, cy), 1, (255, 0, 255), cv2.FILLED)

        mp_drawing.draw_landmarks(img, hand_landmarks, mpHands.HAND_CONNECTIONS)
    
    currentTime = time.time()
    fps = 1/(currentTime - previousTime)
    previousTime = currentTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_COMPLEX_SMALL, 3, (255, 0, 255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)