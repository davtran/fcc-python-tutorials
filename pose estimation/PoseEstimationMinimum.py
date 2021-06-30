import cv2
import mediapipe as mp
import time

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture('pose videos/mace swing.mp4')
previousTime = 0

while True:
    success, img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    print(results.pose_landmarks)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, landmark in enumerate(results.pose_landmarks.landmark):
            height, width, channel = img.shape
            print(id, landmark)
            cx, cy = int(landmark.x * width), int(landmark.y * height)
            cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)

    currentTime = time.time()
    fps = 1/(currentTime - previousTime)
    previousTime = currentTime
    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)