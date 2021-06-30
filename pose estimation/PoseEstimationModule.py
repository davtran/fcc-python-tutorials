import cv2
import mediapipe as mp
import time

class PoseDetector():
    
    def __init__(self, mode = False, upperBody = False, smooth = True, detectionCon = 0.5, trackCon = 0.5):
        self.mode = mode
        self.upperBody = upperBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.upperBody, self.smooth, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
    
    def findPose(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        
        return img

    def findPosition(self, img, draw=True):
        landmark_list = []
        if self.results.pose_landmarks:
            for id, landmark in enumerate(self.results.pose_landmarks.landmark):
                height, width, channel = img.shape
                cx, cy = int(landmark.x * width), int(landmark.y * height)
                landmark_list.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)
        return landmark_list



def main():
    cap = cv2.VideoCapture('pose videos/mace swing.mp4')
    previousTime = 0

    detector = PoseDetector()
    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        landmark_list = detector.findPosition(img)
        print(landmark_list)

        currentTime = time.time()
        fps = 1/(currentTime - previousTime)
        previousTime = currentTime
        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()