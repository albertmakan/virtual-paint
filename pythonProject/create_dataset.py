import cv2
import mediapipe as mp

from utils import extract_hand


def main():
    capture = cv2.VideoCapture(0)
    capture.set(3, 1280)
    capture.set(4, 720)
    n = 0

    with mp.solutions.hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
        while capture.isOpened():
            success, image = capture.read()
            if not success:
                continue
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    hand_img = extract_hand(hand_landmarks, image)
                    # TODO save hand_img
                    n += 1

            cv2.imshow('Creating dataset', cv2.flip(image, 1))
            if cv2.waitKey(5) & 0xFF == 27:
                break
    capture.release()
    print(n)
