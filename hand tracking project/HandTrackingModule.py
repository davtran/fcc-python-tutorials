import cv2
import mediapipe as mp
import time


class HandDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            self.mode, self.maxHands, self.detectionCon, self.trackCon)
        self.mp_drawing = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_drawing.draw_landmarks(img, hand_landmarks, self.mpHands.HAND_CONNECTIONS)
        return img
                
    
    def findPosition(self, img, handNumber=0, draw=True):
        landmark_list = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNumber]

            for id, landmark in enumerate(myHand.landmark):
                        height, width, channel = img.shape
                        cx, cy = int(landmark.x * width), int(landmark.y * height)
                        landmark_list.append([id, cx, cy])
                        if draw:
                            cv2.circle(img, (cx, cy), 1,
                                    (255, 0, 255), cv2.FILLED)
        return landmark_list


def main():
    previousTime = 0
    currentTime = 0
    cap = cv2.VideoCapture(0)
    detector = HandDetector()

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

if __name__ == "__main__":
    main()
